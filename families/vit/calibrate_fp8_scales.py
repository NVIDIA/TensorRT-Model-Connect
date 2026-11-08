# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a ViT model config.

Not used when building an engine. Calibrate each supported checkpoint and commit the
updated ``fp8_activation_scales.json``::

    python -m families.vit.calibrate_fp8_scales fetch --out images
    python -m families.vit.calibrate_fp8_scales calibrate \
        --model-dir <original ViT checkpoint> --images images

``fetch`` streams ImageNet validation images (with their class index) from the ungated
``clip-benchmark/wds_imagenet1k`` dataset and writes them as jpg files with a ``.cls`` label
each. The dataset is sorted by class, so ``fetch`` keeps every ``stride``-th image of every
shard and shuffles them with a fixed seed: the first ``--calib-images`` files calibrate and
the rest are left for evaluation, with every class in both.
``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` (max
calibration) over the first ``--calib-images`` images, and stores
``input_quantizer.amax / 448`` for the six projections of every encoder layer under the
checkpoint key (model config plus a weight fingerprint, see ``quantization.scale_key``).
A projection whose scale is more than ``OUTLIER_FACTOR`` times the median of its kind is
stored as ``null`` and stays in the work precision. ``fetch`` needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import statistics
import tarfile
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_PROJECTIONS,
    scale_key,
)

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
_VALIDATION_IMAGES = 50000
OUTLIER_FACTOR = 20.0
# Module paths of the transformers ViT, by release: 5.x renamed the projections. Both map to
# the checkpoint names the engine builder reads (which are the 4.x names).
_NEW_NAMES = {
    "attention.q_proj": "attention.attention.query",
    "attention.k_proj": "attention.attention.key",
    "attention.v_proj": "attention.attention.value",
    "attention.o_proj": "attention.output.dense",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
}
_MODULE_NAME = re.compile(
    r"^(?:vit\.)?(?:layers\.(\d+)\.(attention\.[qkvo]_proj|mlp\.fc[12])"
    r"|encoder\.layer\.(\d+)\.(attention\.attention\.(?:query|key|value)"
    r"|attention\.output\.dense|intermediate\.dense|output\.dense))$"
)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name (without prefix) of an HF module path, or None if not a projection."""
    match = _MODULE_NAME.match(module_name)
    if match is None:
        return None
    if match.group(1) is not None:
        return f"encoder.layer.{match.group(1)}.{_NEW_NAMES[match.group(2)]}"
    return f"encoder.layer.{match.group(3)}.{match.group(4)}"


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace the scale of every outlier projection by None.

    An outlier is a scale above ``OUTLIER_FACTOR`` times the median scale of its kind (the
    same projection in the other layers).
    """
    result: dict[str, float | None] = dict(scales)
    for stem in LAYER_PROJECTIONS:
        kind = {name: v for name, v in scales.items() if name.endswith(f".{stem}")}
        if not kind:
            continue
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_FACTOR * median:
                result[name] = None
    return result


def fetch_images(out: Path, count: int) -> None:
    """Write about ``count`` validation images spread over every class, and their labels."""
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    stride = max(1, _VALIDATION_IMAGES // count)
    shards = sorted(
        (path for path in fs.ls(f"{_DATASET}/test", detail=False) if path.endswith(".tar")),
        key=lambda path: int(Path(path).stem),
    )
    kept: list[tuple[bytes, int]] = []
    for shard in shards:
        samples: dict[str, dict[str, bytes]] = {}
        with fs.open(shard, "rb") as handle:
            with tarfile.open(fileobj=handle, mode="r|") as archive:
                for member in archive:
                    stem, suffix = Path(member.name).stem, Path(member.name).suffix
                    if suffix not in (".jpg", ".cls"):
                        continue
                    samples.setdefault(stem, {})[suffix] = archive.extractfile(member).read()
        for position, stem in enumerate(sorted(samples)):
            if position % stride == 0 and {".jpg", ".cls"} <= set(samples[stem]):
                kept.append((samples[stem][".jpg"], int(samples[stem][".cls"])))
    random.Random(0).shuffle(kept)
    for index, (jpg, label) in enumerate(kept[:count]):
        (out / f"img_{index:04d}.jpg").write_bytes(jpg)
        (out / f"img_{index:04d}.cls").write_text(str(label))
    print(f"wrote {min(count, len(kept))} images to {out}")


def _fp8_config(mtq) -> dict:
    """``FP8_DEFAULT_CFG`` with the encoder projections switched on.

    The default config may leave modules disabled by name; later entries override earlier
    ones, so the enable entries go last. Newer ModelOpt versions keep the entries in a list,
    older ones in a dict keyed by quantizer pattern.
    """
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for kind in ("input_quantizer", "weight_quantizer"):
        pattern = f"*encoder.layer.*.{kind}"
        if isinstance(config["quant_cfg"], list):
            config["quant_cfg"].append({"quantizer_name": pattern, "enable": True})
        else:
            config["quant_cfg"][pattern] = {"enable": True}
    return config


def load_images(images: Path, start: int, stop: int):
    """RGB PIL images ``start..stop`` of the fetched set and their class indices."""
    from PIL import Image

    pictures, labels = [], []
    for index in range(start, stop):
        path = images / f"img_{index:04d}.jpg"
        if not path.is_file():
            break
        pictures.append(Image.open(path).convert("RGB"))
        labels.append(int((images / f"img_{index:04d}.cls").read_text()))
    return pictures, labels


def calibrate(model_dir: Path, images: Path, calib_images: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the checkpoint key."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoImageProcessor, ViTForImageClassification

    from . import config as config_module

    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    cfg = config_module.resolve(model_dir, weights)
    key = scale_key(cfg, weights)

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = ViTForImageClassification.from_pretrained(str(model_dir), dtype=torch.float32)
    model = model.cuda().eval()
    pictures, _ = load_images(images, 0, calib_images)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")

    def forward_loop(calibration_model) -> None:
        """Run every calibration image through the classifier once."""
        with torch.no_grad():
            for start in range(0, len(pictures), 16):
                pixels = processor(images=pictures[start : start + 16], return_tensors="pt")
                calibration_model(pixel_values=pixels["pixel_values"].cuda())

    model = mtq.quantize(model, _fp8_config(mtq), forward_loop)

    scales: dict[str, float] = {}
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            skipped.append(name)
            continue
        scales[f"{cfg['prefix']}{internal}"] = float(amax) / FP8_E4M3_MAX
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = {
        f"{cfg['prefix']}encoder.layer.{layer}.{stem}"
        for layer in range(cfg["num_layers"])
        for stem in LAYER_PROJECTIONS
    }
    if set(scales) != expected:
        raise RuntimeError(
            f"calibration missed {len(expected - set(scales))} projections, e.g. {skipped[:3]}"
        )

    table_scales = mark_outliers(scales)
    outliers = sorted(name for name, value in table_scales.items() if value is None)
    if outliers:
        print(f"outlier projections stored as null (kept in the work precision): {outliers}")
    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(table_scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=2000)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--calib-images", type=int, default=500)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out, args.count)
    else:
        calibrate(args.model_dir, args.images, args.calib_images)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a CLIP model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.clip.calibrate_fp8_scales fetch --out images
    python -m families.clip.calibrate_fp8_scales calibrate \
        --model-dir <original CLIP checkpoint> --images images

``fetch`` streams the ImageNet validation set from the ungated
``clip-benchmark/wds_imagenet1k`` dataset (7 shards, sorted by class, about 7 GB) and
keeps a spread over all 1000 classes: the images whose index modulo 100 is 7 as the
calibration set (500 images) and the images whose index modulo 50 is 33 as the held-out
evaluation set (1000 images, one per class). It writes them as jpg files with a ``.cls``
label each, plus ``classnames.txt``.
``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` while the
vision tower embeds the calibration images and the text tower embeds ``a photo of
<class>`` prompts padded to the model window, and stores ``input_quantizer.amax / 448`` for
every encoder projection under the model-config key (see ``quantization.scale_key``). A
projection whose scale is above 20x the median of its kind in its tower is an outlier and
is stored as ``null``. ``fetch`` needs network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import tarfile
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_PROJECTIONS,
    OUTLIER_RATIO,
    TOWERS,
    scale_key,
)

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
CALIBRATION_MODULO, CALIBRATION_REMAINDER = 100, 7
EVALUATION_MODULO, EVALUATION_REMAINDER = 50, 33
_PROJECTION_NAME = re.compile(
    r"^(?:vision_model|text_model)\.encoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.fc[12])$"
)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of an HF module path, or None when it is not a quantized projection."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def split_of(index: int) -> str | None:
    """``calib`` or ``eval`` for the image at ``index`` of the 50000 validation images."""
    if index % CALIBRATION_MODULO == CALIBRATION_REMAINDER:
        return "calib"
    if index % EVALUATION_MODULO == EVALUATION_REMAINDER:
        return "eval"
    return None


def fetch_images(out: Path) -> None:
    """Write the calibration and evaluation images with their labels, and the class names."""
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    (out / "classnames.txt").write_bytes(fs.read_bytes(f"{_DATASET}/classnames.txt"))
    counts = {"calib": 0, "eval": 0}
    index, pending = -1, {}

    def flush() -> None:
        split = split_of(index)
        if split is None or ".jpg" not in pending or ".cls" not in pending:
            return
        stem = f"{split}_{index:05d}"
        (out / f"{stem}.jpg").write_bytes(pending[".jpg"])
        (out / f"{stem}.cls").write_bytes(pending[".cls"])
        counts[split] += 1

    shards = sorted(
        (Path(entry).name for entry in fs.glob(f"{_DATASET}/test/*.tar")),
        key=lambda name: int(Path(name).stem),
    )
    for shard in shards:
        with fs.open(f"{_DATASET}/test/{shard}", "rb") as handle:
            with tarfile.open(fileobj=handle, mode="r|") as archive:
                key = None
                for member in archive:
                    path = Path(member.name)
                    if path.suffix not in (".jpg", ".cls"):
                        continue
                    if path.stem != key:
                        flush()
                        key, pending, index = path.stem, {}, index + 1
                    pending[path.suffix] = archive.extractfile(member).read()
        print(f"{shard}: {index + 1} images seen, kept {counts}")
    flush()
    print(f"wrote {counts} to {out}")


def _fp8_config(mtq) -> dict:
    """``FP8_DEFAULT_CFG`` with the encoder projections of both towers switched on.

    ModelOpt's default config leaves modules named ``*vision_model*`` unquantized, which
    would drop the whole vision tower. Later entries override earlier ones, so the enable
    entries go last. Newer ModelOpt versions keep the entries in a list, older ones in a
    dict keyed by quantizer pattern.
    """
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for tower, _ in TOWERS:
        for kind in ("input_quantizer", "weight_quantizer"):
            pattern = f"*{tower}.encoder.layers.*.{kind}"
            if isinstance(config["quant_cfg"], list):
                config["quant_cfg"].append({"quantizer_name": pattern, "enable": True})
            else:
                config["quant_cfg"][pattern] = {"enable": True}
    return config


def load_images(images: Path, split: str, limit: int | None = None):
    """RGB PIL images of one split of the fetched set and their class indices."""
    from PIL import Image

    pictures, labels = [], []
    for path in sorted(images.glob(f"{split}_*.jpg"))[:limit]:
        pictures.append(Image.open(path).convert("RGB"))
        labels.append(int(path.with_suffix(".cls").read_text()))
    return pictures, labels


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace scales above ``OUTLIER_RATIO`` x the median of their kind (tower and stem) by None."""
    result: dict[str, float | None] = dict(scales)
    for tower, _ in TOWERS:
        for stem in LAYER_PROJECTIONS:
            kind = {
                name: value
                for name, value in scales.items()
                if name.startswith(f"{tower}.") and name.endswith(f".{stem}")
            }
            if not kind:
                continue
            median = statistics.median(kind.values())
            for name, value in kind.items():
                if value > OUTLIER_RATIO * median:
                    result[name] = None
    return result


def calibrate(model_dir: Path, images: Path, calib_images: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModel, AutoProcessor

    from . import config as config_module

    cfg = config_module.resolve(model_dir)
    key = scale_key(cfg)
    window = cfg["text"]["max_position_embeddings"]

    processor = AutoProcessor.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    pictures, _ = load_images(images, "calib", calib_images)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")
    names = (images / "classnames.txt").read_text().splitlines()
    prompts = [f"a photo of {name}" for name in names]

    def forward_loop(calibration_model) -> None:
        """Embed every calibration image and every class prompt once."""
        with torch.no_grad():
            for start in range(0, len(pictures), 16):
                pixels = processor(images=pictures[start : start + 16], return_tensors="pt")
                calibration_model.get_image_features(pixel_values=pixels["pixel_values"].cuda())
            for start in range(0, len(prompts), 100):
                tokens = processor(
                    text=prompts[start : start + 100],
                    padding="max_length",
                    max_length=window,
                    truncation=True,
                    return_tensors="pt",
                )
                calibration_model.get_text_features(input_ids=tokens["input_ids"].cuda())

    model = mtq.quantize(model, _fp8_config(mtq), forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[internal] = float(amax) / FP8_E4M3_MAX
    # Every projection of both towers must be calibrated: a partial table would build a
    # half-quantized engine without any error.
    expected = {
        f"{tower}.encoder.layers.{layer}.{stem}"
        for tower, tower_key in TOWERS
        for layer in range(cfg[tower_key]["num_layers"])
        for stem in LAYER_PROJECTIONS
    }
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} projections")
    table_scales = mark_outliers(scales)
    outliers = sorted(name for name, value in table_scales.items() if value is None)
    print(f"outlier projections stored as null ({len(outliers)}): {outliers}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(table_scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table_scales)} scales for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--calib-images", type=int, default=500)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out)
    else:
        calibrate(args.model_dir, args.images, args.calib_images)


if __name__ == "__main__":
    main()

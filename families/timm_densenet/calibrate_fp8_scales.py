# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a timm DenseNet model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.timm_densenet.calibrate_fp8_scales fetch --out images
    python -m families.timm_densenet.calibrate_fp8_scales calibrate \
        --model-dir <original timm DenseNet checkpoint> --images images

``fetch`` streams ImageNet validation images (with their class index) from the
ungated ``clip-benchmark/wds_imagenet1k`` dataset and writes them as jpg files with a
``.cls`` label each, at a fixed stride over the class-sorted set. ``calibrate`` loads the
model in PyTorch (timm), runs ModelOpt ``FP8_DEFAULT_CFG`` (max calibration) over
``--calib-images`` images with the checkpoint's own preprocessing, and stores
``input_quantizer.amax / 448`` of every quantized convolution under the model-config key
(see ``quantization.scale_key``). The other images (``split_indices``) are left for
evaluation. ``fetch`` needs network access; ``calibrate`` needs a GPU and ``timm``.

A convolution whose input scale is more than 20 times the median of the same kind has a
massive activation and is not quantized: its entry is stored as ``null``, so the build
leaves it in fp16 on purpose, while a convolution that is missing from the file is an error.
The quantized convolutions are listed by ``quantization.conv_names``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tarfile
from pathlib import Path

from .quantization import (
    ACTIVATION_SCALES_FILENAME,
    FP8_E4M3_MAX,
    LAYER_CONVS,
    TRANSITION_CONV,
    conv_names,
    scale_key,
)

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
_CONV_NAME = re.compile(
    r"^features\.(?:denseblock\d+\.denselayer\d+\.(?:"
    + "|".join(re.escape(conv) for conv in LAYER_CONVS)
    + r")|transition\d+\."
    + re.escape(TRANSITION_CONV)
    + r")$"
)
# Block, layer and transition indices; the conv1 / conv2 suffix names the kind.
_KIND = re.compile(r"(?<=denseblock)\d+|(?<=denselayer)\d+|(?<=transition)\d+")
_OUTLIER_RATIO = 20.0
_VALIDATION_IMAGES = 50_000
_SHARDS = 7


def conv_name(module_name: str) -> str | None:
    """Checkpoint name of a timm module path, or None when it is not a quantized convolution."""
    return module_name if _CONV_NAME.match(module_name) else None


def split_outlier_scales(scales: dict[str, float]) -> tuple[dict[str, float | None], list[str]]:
    """Replace the scale of a convolution far above the median of its kind by ``None``.

    The kind is the module name without the block and layer index (dense-layer ``conv1``,
    dense-layer ``conv2``, transition ``conv``). Returns the table (``None`` marks an
    unquantized convolution) and the sorted names of the convolutions left unquantized.
    """
    by_kind: dict[str, list[float]] = {}
    for name, value in scales.items():
        by_kind.setdefault(_KIND.sub("N", name), []).append(value)
    median = {kind: statistics.median(values) for kind, values in by_kind.items()}
    table: dict[str, float | None] = {}
    excluded = []
    for name, value in scales.items():
        if value > _OUTLIER_RATIO * median[_KIND.sub("N", name)]:
            table[name] = None
            excluded.append(name)
        else:
            table[name] = value
    return table, sorted(excluded)


def fetch_images(out: Path, count: int) -> None:
    """Write ``count`` validation images spread evenly over the 50000, and their class indices.

    The dataset is sorted by class, so the images are taken at a fixed stride (2 per class
    for 2000 images); the first images alone would cover only a few classes.
    """
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    stride = _VALIDATION_IMAGES // count
    fs = HfFileSystem()
    position = written = 0
    for shard in range(_SHARDS):
        pending: dict[str, dict[str, bytes]] = {}
        with fs.open(f"{_DATASET}/test/{shard}.tar", "rb") as handle:
            with tarfile.open(fileobj=handle, mode="r|") as archive:
                for member in archive:
                    suffix = Path(member.name).suffix
                    if suffix not in (".jpg", ".cls"):
                        continue
                    stem = Path(member.name).stem
                    pending.setdefault(stem, {})[suffix] = archive.extractfile(member).read()
                    if len(pending[stem]) < 2:
                        continue
                    files = pending.pop(stem)
                    if position % stride == 0 and written < count:
                        for ext, data in files.items():
                            (out / f"img_{written:05d}{ext}").write_bytes(data)
                        written += 1
                    position += 1
    print(f"wrote {written} images to {out}")


def split_indices(total: int, calib_images: int) -> tuple[list[int], list[int]]:
    """Calibration and held-out image indices, both spread over the classes."""
    step = max(total // calib_images, 2)
    calibration = list(range(0, total, step))[:calib_images]
    chosen = set(calibration)
    return calibration, [index for index in range(total) if index not in chosen]


def image_files(images: Path) -> list[Path]:
    """The fetched images in fetch order."""
    return sorted(Path(images).glob("img_*.jpg"))


def load_images(images: Path, indices: list[int]):
    """RGB PIL images of the given indices of the fetched set and their class indices."""
    from PIL import Image

    files = image_files(images)
    pictures, labels = [], []
    for index in indices:
        if index >= len(files):
            break
        pictures.append(Image.open(files[index]).convert("RGB"))
        labels.append(int(files[index].with_suffix(".cls").read_text()))
    return pictures, labels


def load_timm_model(model_dir: Path):
    """The checkpoint as a timm model in float32 and its evaluation transform."""
    import timm
    import torch

    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    model = timm.create_model(
        raw["architecture"],
        pretrained=True,
        pretrained_cfg_overlay={"file": str(Path(model_dir) / "model.safetensors")},
    )
    model = model.to(torch.float32).eval()
    data_config = timm.data.resolve_data_config({}, model=model)
    return model, timm.data.create_transform(**data_config)


def calibrate(model_dir: Path, images: Path, calib_images: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch

    from .config import ModelConfig
    from .model import _TimmDensenetModel

    config = ModelConfig.from_dir(model_dir)
    # The loader the build uses, so the key (config and weight sums) matches at build time.
    weights = _TimmDensenetModel().load_weights(
        str(model_dir), config, precision="fp16", quantization="fp8"
    )
    cfg = config.raw["_timm_densenet_config"]
    block_layers = cfg["block_layers"]
    key = scale_key(block_layers, (cfg["image_size_h"], cfg["image_size_w"]), weights)

    calibration, _ = split_indices(len(image_files(images)), calib_images)
    pictures, _ = load_images(images, calibration)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")
    model, transform = load_timm_model(model_dir)
    model = model.cuda()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration image through the model once."""
        with torch.no_grad():
            for start in range(0, len(pictures), 16):
                batch = torch.stack([transform(p) for p in pictures[start : start + 16]])
                calibration_model(batch.cuda())

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    scales: dict[str, float] = {}
    for name, module in model.named_modules():
        if conv_name(name) is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is not None and float(amax) > 0:
            scales[name] = float(amax) / FP8_E4M3_MAX
    # Every convolution must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = set(conv_names(block_layers))
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} convolutions")
    table, excluded = split_outlier_scales(scales)
    if excluded:
        print(f"massive activations, left in fp16 (null): {excluded}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stored[key] = dict(sorted(table.items()))
    path.write_text(json.dumps(dict(sorted(stored.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(table)} scales ({len(excluded)} null) for {key!r} to {path}")


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

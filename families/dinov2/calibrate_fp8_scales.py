# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a DINOv2 model config.

Not used when building an engine. Calibrate each supported model config and
commit the updated ``fp8_activation_scales.json``::

    python -m families.dinov2.calibrate_fp8_scales fetch --out images
    python -m families.dinov2.calibrate_fp8_scales calibrate \
        --model-dir <original DINOv2 checkpoint> --images images

``fetch`` streams the first ImageNet validation images from the ungated
``clip-benchmark/wds_imagenet1k`` dataset and writes them as jpg files.
``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` (max
calibration) over the first ``--calib-images`` images, preprocessed the way the family's
runtime does (bicubic resize, centre crop), and stores ``input_quantizer.amax / 448`` of
every encoder projection under the model-config key (see ``quantization.scale_key``).
A projection whose scale is more than ``OUTLIER_RATIO`` times the median of its kind is
stored as ``null`` and stays float32. The images after the first ``--calib-images`` are
left for evaluation. ``fetch`` needs network access; ``calibrate`` needs a GPU.
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
    LAYER_PROJECTIONS,
    OUTLIER_RATIO,
    projections,
    scale_key,
)

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
_PROJECTION_NAME = re.compile(
    r"^encoder\.layer\.\d+\."
    r"(?:attention\.attention\.(?:query|key|value)|attention\.output\.dense|mlp\.fc[12])$"
)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of an HF module path, or None when it is not a quantized projection."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def fetch_images(out: Path, count: int) -> None:
    """Write the first ``count`` validation images to ``out``."""
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    written = 0
    with fs.open(f"{_DATASET}/test/0.tar", "rb") as handle:
        with tarfile.open(fileobj=handle, mode="r|") as archive:
            for member in archive:
                suffix = Path(member.name).suffix
                if suffix not in (".jpg", ".cls"):
                    continue
                index = int(Path(member.name).stem.lstrip("s"))
                if index >= count:
                    break
                (out / f"img_{index:04d}{suffix}").write_bytes(archive.extractfile(member).read())
                written += suffix == ".jpg"
    print(f"wrote {written} images to {out}")


def load_images(images: Path, start: int, stop: int):
    """RGB PIL images ``start..stop`` of the fetched set."""
    from PIL import Image

    pictures = []
    for index in range(start, stop):
        path = images / f"img_{index:04d}.jpg"
        if not path.is_file():
            break
        pictures.append(Image.open(path).convert("RGB"))
    return pictures


def mark_outliers(scales: dict[str, float]) -> dict[str, float | None]:
    """Replace the scale of every outlier projection by None.

    An outlier is a scale more than ``OUTLIER_RATIO`` times the median of its kind (the same
    projection in every layer): one 8-bit scale over such an input loses the other values.
    """
    marked: dict[str, float | None] = dict(scales)
    for stem in LAYER_PROJECTIONS:
        kind = {name: v for name, v in scales.items() if name.endswith(f".{stem}")}
        median = statistics.median(kind.values())
        for name, value in kind.items():
            if value > OUTLIER_RATIO * median:
                marked[name] = None
    return marked


def calibrate(model_dir: Path, images: Path, calib_images: int) -> None:
    """Run ModelOpt FP8 calibration and store the activation scales under the config key."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoImageProcessor, Dinov2Model

    from . import config as config_module

    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    cfg = config_module.resolve(model_dir, weights)
    key = scale_key(cfg)

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = Dinov2Model.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    pictures = load_images(images, 0, calib_images)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")

    def forward_loop(calibration_model) -> None:
        """Run every calibration image through the model once."""
        with torch.no_grad():
            for start in range(0, len(pictures), 16):
                pixels = processor(images=pictures[start : start + 16], return_tensors="pt")
                calibration_model(pixel_values=pixels["pixel_values"].cuda())

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

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
        scales[internal] = float(amax) / FP8_E4M3_MAX
    if skipped:
        print(f"no activation amax: {len(skipped)} projections, e.g. {skipped[:3]}")
    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    expected = set(projections(cfg))
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} projections")

    marked = mark_outliers(scales)
    outliers = sorted(name for name, value in marked.items() if value is None)
    print(f"outliers kept float32 (> {OUTLIER_RATIO}x median of their kind): {outliers}")

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(marked.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(marked)} entries for {key!r} to {path}")


def main() -> None:
    """Parse the command line and run the fetch or calibrate command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--count", type=int, default=400)
    run = commands.add_parser("calibrate")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--images", type=Path, required=True)
    run.add_argument("--calib-images", type=int, default=200)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out, args.count)
    else:
        calibrate(args.model_dir, args.images, args.calib_images)


if __name__ == "__main__":
    main()

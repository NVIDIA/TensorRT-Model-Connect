# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a DINOv3 model shape.

Not used when building an engine. Calibrate each supported model shape and commit the
updated ``fp8_activation_scales.json``::

    python -m families.dinov3.calibrate_fp8_scales fetch --out images
    python -m families.dinov3.calibrate_fp8_scales calibrate \
        --model-dir <original DINOv3 checkpoint> --images images

``fetch`` streams the first ImageNet validation images from the ungated
``clip-benchmark/wds_imagenet1k`` dataset and writes them as jpg files. ``calibrate``
loads the model in PyTorch (float32), runs the first ``--calib-images`` images through the
Transformers image processor and the model, and records ``max(abs(input)) / 448`` of every
quantized projection with forward pre-hooks. That is the statistic ModelOpt's
``FP8_DEFAULT_CFG`` max calibrator stores as ``input_quantizer.amax``; hooks are used
because they do not depend on how ModelOpt wraps the attention modules. The images after
the first ``--calib-images`` are left for evaluation. ``fetch`` needs network access;
``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

from .quantization import ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, scale_key

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
_VIT_MODULE = re.compile(
    r"^(?:model\.)?layer\.(\d+)\.(attention\.q_proj|attention\.o_proj|mlp\.up_proj|mlp\.down_proj|mlp\.gate_proj)$"
)
_CONVNEXT_MODULE = re.compile(r"^(?:model\.)?stages\.(\d+)\.layers\.(\d+)\.pointwise_conv([12])$")


def projection_name(module_name: str) -> str | None:
    """Builder's logical projection name of an HF module path, or None when not quantized.

    The q, k and v projections share one input and run as one fused GEMM, so the input
    of ``attention.q_proj`` is recorded as ``attention.qkv``.
    """
    match = _VIT_MODULE.match(module_name)
    if match:
        layer, stem = match.groups()
        stem = "attention.qkv" if stem == "attention.q_proj" else stem
        return f"layer.{layer}.{stem}"
    match = _CONVNEXT_MODULE.match(module_name)
    if match:
        stage, block, index = match.groups()
        return f"stage.{stage}.block.{block}.pointwise{index}"
    return None


def fetch_images(out: Path, count: int) -> None:
    """Write the first ``count`` validation images and their labels."""
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


def calibrate(
    model_dir: Path, images: Path, calib_images: int, image_size: int | None = None
) -> None:
    """Record the projection input ranges and store the scales under the shape key.

    ``image_size`` overrides the resolution of the checkpoint config (the key and the
    processor); the model itself accepts any multiple of the patch size.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    from .convnext_builder import resolve_convnext_config
    from .model import resolve_vit_config

    raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    model_type = raw["model_type"]
    if image_size is not None:
        raw["image_size"] = image_size
    cfg = resolve_vit_config(raw) if model_type == "dinov3_vit" else resolve_convnext_config(raw)
    key = scale_key(model_type, cfg)

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    if image_size is not None:
        processor.size = {"height": image_size, "width": image_size}
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    pictures, _ = load_images(images, 0, calib_images)
    if not pictures:
        raise RuntimeError(f"no images in {images}; run the fetch command first")

    amax: dict[str, float] = {}
    handles = []
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None:
            continue

        def record(_module, args, internal=internal):
            value = float(args[0].detach().abs().max())
            amax[internal] = max(amax.get(internal, 0.0), value)

        handles.append(module.register_forward_pre_hook(record))

    with torch.no_grad():
        for start in range(0, len(pictures), 16):
            batch = processor(images=pictures[start : start + 16], return_tensors="pt")
            model(pixel_values=batch["pixel_values"].cuda())
    for handle in handles:
        handle.remove()

    # Every projection must be calibrated: a partial table would build a half-quantized
    # engine without any error.
    if model_type == "dinov3_vit":
        expected = {
            f"layer.{layer}.{stem}"
            for layer in range(cfg["num_hidden_layers"])
            for stem in (
                "attention.qkv",
                "attention.o_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                *(("mlp.gate_proj",) if cfg["use_gated_mlp"] else ()),
            )
        }
    else:
        expected = {
            f"stage.{stage}.block.{block}.pointwise{index}"
            for stage, depth in enumerate(cfg["depths"])
            for block in range(depth)
            for index in (1, 2)
        }
    if set(amax) != expected or not all(value > 0 for value in amax.values()):
        raise RuntimeError(f"calibration missed {len(expected - set(amax))} projections")
    scales = {name: value / FP8_E4M3_MAX for name, value in amax.items()}

    path = Path(__file__).parent / ACTIVATION_SCALES_FILENAME
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table[key] = dict(sorted(scales.items()))
    path.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(scales)} scales for {key!r} to {path}")


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
    run.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out, args.count)
    else:
        calibrate(args.model_dir, args.images, args.calib_images, args.image_size)


if __name__ == "__main__":
    main()

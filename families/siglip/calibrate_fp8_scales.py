# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate FP8 activation scales for a SigLIP model shape.

Not used when building an engine. Calibrate each supported model shape and
commit the updated ``fp8_activation_scales.json``::

    python -m families.siglip.calibrate_fp8_scales fetch --out images
    python -m families.siglip.calibrate_fp8_scales calibrate \
        --model-dir <original SigLIP checkpoint> --images images

``fetch`` streams the first ImageNet validation images (with their class names) from the
ungated ``clip-benchmark/wds_imagenet1k`` dataset and writes them as jpg files with a
``.cls`` label each, plus ``classnames.txt``.
``calibrate`` loads the model in PyTorch, runs ModelOpt ``FP8_DEFAULT_CFG`` while the
vision tower embeds the first ``--calib-images`` images and the text tower embeds
``a photo of <class>`` prompts, and stores ``input_quantizer.amax / 448`` for every
encoder projection under the model-shape key (see ``quantization.scale_key``). The
images after the first ``--calib-images`` are left for evaluation. ``fetch`` needs
network access; ``calibrate`` needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import tarfile
from pathlib import Path

from .quantization import _LAYER_PROJECTIONS, ACTIVATION_SCALES_FILENAME, FP8_E4M3_MAX, scale_key

_DATASET = "datasets/clip-benchmark/wds_imagenet1k"
_PROJECTION_NAME = re.compile(
    r"^(?:vision_model|text_model)\.encoder\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.fc[12])$"
)


def projection_name(module_name: str) -> str | None:
    """Checkpoint name of an HF module path, or None when it is not a quantized projection."""
    return module_name if _PROJECTION_NAME.match(module_name) else None


def fetch_images(out: Path, count: int) -> None:
    """Write the first ``count`` validation images and their labels, and the class names."""
    from huggingface_hub import HfFileSystem

    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    (out / "classnames.txt").write_bytes(fs.read_bytes(f"{_DATASET}/classnames.txt"))
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


def _fp8_config(mtq) -> dict:
    """``FP8_DEFAULT_CFG`` with the encoder projections of both towers switched on.

    ModelOpt's default config leaves modules named ``*vision_model*`` unquantized, which
    would drop the whole vision tower. Later entries override earlier ones, so the enable
    entries go last. Newer ModelOpt versions keep the entries in a list, older ones in a
    dict keyed by quantizer pattern.
    """
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for tower in ("vision_model", "text_model"):
        for kind in ("input_quantizer", "weight_quantizer"):
            pattern = f"*{tower}.encoder.layers.*.{kind}"
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
    """Run ModelOpt FP8 calibration and store the activation scales under the shape key."""
    import modelopt.torch.quantization as mtq
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoModel, AutoProcessor

    from . import config as config_module

    weights = load_file(str(Path(model_dir) / "model.safetensors"))
    cfg = config_module.resolve(model_dir, weights)
    key = scale_key(cfg)
    window = cfg["text"]["max_text_len"]

    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=False)
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    pictures, _ = load_images(images, 0, calib_images)
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
    skipped: list[str] = []
    for name, module in model.named_modules():
        internal = projection_name(name)
        if internal is None or not hasattr(module, "input_quantizer"):
            continue
        amax = module.input_quantizer.amax
        if amax is None or not float(amax) > 0:
            # ModelOpt left this projection unquantized; it gets no entry and stays unquantized.
            skipped.append(name)
            continue
        scales[internal] = float(amax) / FP8_E4M3_MAX
    if skipped:
        print(
            f"no activation amax, left unquantized: {len(skipped)} projections, e.g. {skipped[:3]}"
        )
    # Every projection of both towers must be calibrated: a partial table would build a
    # half-quantized engine without any error.
    expected = {
        f"{tower}.encoder.layers.{layer}.{stem}"
        for tower, count in (
            ("vision_model", cfg["vision"]["num_layers"]),
            ("text_model", cfg["text"]["num_layers"]),
        )
        for layer in range(count)
        for stem in _LAYER_PROJECTIONS
    }
    if set(scales) != expected:
        raise RuntimeError(f"calibration missed {len(expected - set(scales))} projections")

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
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_images(args.out, args.count)
    else:
        calibrate(args.model_dir, args.images, args.calib_images)


if __name__ == "__main__":
    main()

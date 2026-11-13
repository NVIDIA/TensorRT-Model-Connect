# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the Dynamic-HD goldens from the checkpoint's own Phi4MMImageProcessor.

usage: python generate.py PROCESSING_PHI4MM_PY

The images are synthetic and written next to this script; the goldens record,
per image, the crop grid, the image token count, the valid patch rows/cols of
every vision batch entry (from image_attention_mask) and sampled pixel values.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
# Grids: 1x1 upscale, exact 1x1, 2x1 with column padding, 1x2 with row
# padding, 3x2, the aspect-ratio search beyond 36 crops (wide and square).
SIZES = [(50, 30), (448, 448), (900, 400), (300, 700), (1200, 800), (4000, 500), (3000, 3000)]
SAMPLES_PER_ENTRY = 48


def synthetic(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([(x * 7 + y * 3) % 256, (x * 2 + y * 11) % 256, ((x ^ y) * 5) % 256], -1)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def main() -> None:
    spec = importlib.util.spec_from_file_location("processing_phi4mm", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.Phi4MMImageProcessor(dynamic_hd=36)
    rng = np.random.default_rng(1410)
    cases = []
    for width, height in SIZES:
        name = f"synthetic-{width}x{height}.png"
        image = synthetic(width, height)
        image.save(HERE / name)
        with Image.open(HERE / name) as loaded:
            out = processor.preprocess([loaded.convert("RGB")], return_tensors="pt")
        pixels = out["input_image_embeds"][0].numpy()  # [entries, 3, 448, 448]
        masks = out["image_attention_mask"][0].numpy()  # [entries, 32, 32]
        image_height, image_width = (int(v) for v in out["image_sizes"][0])
        entries = 1 + (image_height // 448) * (image_width // 448)
        samples = []
        for entry in range(entries):
            for _ in range(SAMPLES_PER_ENTRY):
                c, yy, xx = int(rng.integers(3)), int(rng.integers(448)), int(rng.integers(448))
                samples.append([entry, c, yy, xx, float(pixels[entry, c, yy, xx])])
        cases.append(
            {
                "image": name,
                "grid": [image_width // 448, image_height // 448],
                "num_image_tokens": int(out["num_img_tokens"][0]),
                "crop_valid": [[int(m[:, 0].sum()), int(m[0].sum())] for m in masks[:entries]],
                "pixel_samples": samples,
            }
        )
    (HERE / "hf-dynamic-hd.json").write_text(json.dumps({"cases": cases}, indent=1) + "\n")


if __name__ == "__main__":
    main()

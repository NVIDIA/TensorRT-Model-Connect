# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the independent eager PyTorch reference for native E2E comparisons."""

from pathlib import Path
import sys
import json

import numpy as np
from PIL import Image
import torch

from families.yolov5.tests.reference.model import detections, load_model


def _pixels(source):
    scale = min(640 / source.height, 640 / source.width)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    pad_x, pad_y = (640 - resized.width) // 2, (640 - resized.height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None].copy() / 255.0
    return torch.from_numpy(pixels), scale, pad_x, pad_y


def _check_snapshot(raw, expected):
    values = raw.detach().numpy()
    assert list(values.shape) == expected["shape"]
    indices = np.linspace(0, values.size - 1, len(expected["values"]), dtype=np.int64)
    np.testing.assert_allclose(
        values.reshape(-1)[indices], expected["values"], atol=1e-4, rtol=1e-5
    )
    kept = detections(raw).detach().numpy()
    np.testing.assert_allclose(
        kept, np.asarray(expected["detections"]).reshape(-1, 6), atol=1e-4, rtol=1e-5
    )


def _verify_reference(reference, model_dir, image, original):
    name = "yolov5n"
    snapshot = json.loads((Path(__file__).with_name("snapshots") / f"{name}.json").read_text())
    _check_snapshot(original, snapshot["cases"]["original"])
    mirrored = Image.open(image).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    pixels, *_ = _pixels(mirrored)
    _check_snapshot(reference(pixels), snapshot["cases"]["mirrored"])


def run(model_dir: Path, image: Path, output: Path) -> None:
    reference = load_model(model_dir)
    pixels, scale, pad_x, pad_y = _pixels(Image.open(image).convert("RGB"))
    with torch.inference_mode():
        raw = reference(pixels)
        _verify_reference(reference, model_dir, image, raw)
        values = detections(raw)
    np.savez(output, kept=values.numpy(), scale=scale, pad_x=pad_x, pad_y=pad_y)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))

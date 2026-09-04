# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-only helpers kept outside Model Connect core and families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def preprocess_locateanything(image_path: str) -> dict[str, np.ndarray]:
    from PIL import Image

    size = 448
    patch = 14
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray((0.5, 0.5, 0.5), dtype=np.float32)) / np.asarray(
        (0.5, 0.5, 0.5), dtype=np.float32
    )
    chw = pixels.transpose(2, 0, 1)
    grid = size // patch
    patches = (
        chw.reshape(3, grid, patch, grid, patch)
        .transpose(1, 3, 0, 2, 4)
        .reshape(grid * grid, 3, patch, patch)
    )
    return {
        "pixel_values": patches.astype(np.float32),
        "image_grid_hws": np.asarray([[grid, grid]], dtype=np.int32),
    }


def configure_official_model_args(
    model: Any, *, max_disparity: int, valid_iters: int
) -> None:
    model.args.max_disp = max_disparity
    model.args.valid_iters = valid_iters
    model.args.normalize = True


def fast_foundation_stereo_model_dir(configured: str) -> Path:
    root = Path(configured).expanduser().resolve()
    required = (
        root / "core/foundation_stereo.py",
        root / "core/submodule.py",
        root / "weights/23-36-37/model_best_bp2_serialize.pth",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("Fast Foundation Stereo reference model directory is incomplete")
    return root

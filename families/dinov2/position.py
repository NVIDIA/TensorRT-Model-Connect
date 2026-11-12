# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resize DINOv2's position table to the grid the engine actually runs at.

The checkpoint's table has one class row and one row per patch of the grid it
was trained on - 37x37 for the base model, from its 518-pixel training size.
The published preprocessor crops to 224, which is a 16x16 grid, so the table
never matches the input and the reference interpolates it on every forward
pass.

The engine's input size is fixed at build time, so the interpolation is done
once here and baked into the plan as a constant. That is exact rather than
approximate: this uses the same ``torch.nn.functional.interpolate`` call the
reference uses, and the result was measured against
``Dinov2Embeddings.interpolate_pos_encoding`` at **maxabs 0.0**.

Each knob matters, measured against the correct table on the base checkpoint:

======================================  =======
wrong choice                            maxabs
======================================  =======
truncate the table instead of resizing   0.2372
``mode="bilinear"``                      0.0161
``align_corners=True``                   0.0384
======================================  =======
"""

from __future__ import annotations

import numpy as np


def interpolate_position_table(table: np.ndarray, trained_grid: int, grid: int) -> np.ndarray:
    """``[1 + trained_grid^2, width]`` to ``[1 + grid^2, width]``.

    The class row is carried across untouched; only the patch rows are
    resized, as a ``[1, width, trained_grid, trained_grid]`` image.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "dinov2 requires torch in the build environment to resize the position "
            "table the same way the reference does") from error

    table = np.asarray(table)
    if table.ndim == 3:
        table = table[0]
    expected = trained_grid * trained_grid + 1
    if table.shape[0] != expected:
        raise ValueError(
            f"dinov2 position table has {table.shape[0]} rows, not {expected}")
    if grid == trained_grid:
        return table.astype(np.float32)

    width = table.shape[1]
    patches = torch.from_numpy(np.ascontiguousarray(table[1:])).float()
    patches = patches.reshape(1, trained_grid, trained_grid, width).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        patches, size=(grid, grid), mode="bicubic", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(grid * grid, width).numpy()
    return np.concatenate([table[:1], resized], axis=0).astype(np.float32)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time antialiased position interpolation for the register backbone."""

import numpy as np


def interpolate_position_table(table, trained_grid, grid):
    import torch
    from torch.nn.functional import interpolate

    table = np.asarray(table, dtype=np.float32).reshape(-1, table.shape[-1])
    if table.shape[0] != trained_grid * trained_grid + 1:
        raise ValueError("position table must contain class and trained patch rows")
    if trained_grid == grid:
        return table.copy()
    patches = torch.from_numpy(table[1:].copy()).reshape(1, trained_grid, trained_grid, -1)
    patches = interpolate(patches.permute(0, 3, 1, 2), size=(grid, grid), mode="bicubic",
                          align_corners=False, antialias=True)
    patches = patches.permute(0, 2, 3, 1).reshape(grid * grid, -1).numpy()
    return np.ascontiguousarray(np.concatenate((table[:1], patches)))

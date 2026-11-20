# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# The formula evaluation order follows Transformers 5.20.0; see NOTICE.

"""CPU tables for fixed-profile learned positions and two-axis rotary attention."""

import torch


def position_tables(embedding, height, width, source_side, head_dim, theta):
    if (
        min(height, width, source_side, head_dim) <= 0
        or embedding.ndim != 2
        or embedding.shape[0] != source_side * source_side
        or head_dim % 4
    ):
        raise ValueError("PaddleOCR requires a square position table and complete axial pairs")
    cells = torch.arange(height * width, device="cpu")
    vertical, horizontal = cells // width, cells % width

    def neighbors(coordinates, extent):
        location = coordinates.to(torch.float32) * (source_side - 1) / max(extent - 1, 1)
        integer = location.floor()
        fraction = location - integer
        offsets = torch.tensor([0, 1], device="cpu")
        indices = (integer.to(torch.int64)[:, None] + offsets).clamp(0, source_side - 1)
        factors = (1 - (fraction[:, None] - offsets).abs()).clamp(min=0)
        return indices, factors

    rows, row_weights = neighbors(vertical, height)
    columns, column_weights = neighbors(horizontal, width)
    addresses = (rows[:, :, None] * source_side + columns[:, None, :]).reshape(-1, 4)
    coefficients = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(-1, 4)
    positions = (torch.from_numpy(embedding)[addresses] * coefficients[:, :, None]).sum(1)

    axis_width = head_dim // 2
    exponents = torch.arange(0, axis_width, 2, dtype=torch.float32, device="cpu") / axis_width
    frequencies = 1.0 / theta**exponents
    coordinates = torch.stack([vertical, horizontal], dim=1).to(torch.float32)
    angles = coordinates[..., None] * frequencies

    def interleave_axes(values):
        joined = torch.cat([values[:, 0], values[:, 1]], dim=-1)
        return torch.cat([joined, joined], dim=-1).numpy()

    return positions.numpy(), interleave_axes(angles.cos()), interleave_axes(angles.sin())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side window geometry of a SwinV2 block, evaluated once at build time.

Three things in a SwinV2 attention block depend only on the geometry and the
block's own weights, never on the image, so they are computed here in float64
and handed to the engine as one additive bias per block:

* the **continuous position bias**. SwinV2 does not learn a bias table; a small
  MLP (``Linear(2, 512)``, ReLU, ``Linear(512, heads)`` without bias) maps the
  log-spaced relative offset of every pair of positions in a window to one
  bias per head, and the reference squashes it to ``16 * sigmoid(bias)``.
  Leaving out the ``16 * sigmoid`` still builds and still classifies;
* the **shifted-window mask**. After the cyclic shift, positions that the roll
  placed in one window but that are not neighbours are pushed down by -100.
  The transformers reference adds that mask **twice** (once in the reshaped
  sum and once more on the next line), so a masked pair sits at -200, not at
  the -100 of the original SwinV2 code. This family reproduces the reference;
* the **logit scale**: one learned temperature per head, clamped at
  ``log(1 / 0.01)`` before the exponential, that multiplies the cosine
  similarity of query and key.

The coordinate table follows the reference exactly: offsets are normalised by
``pretrained_window - 1`` when the checkpoint names a pretrained window and by
``window - 1`` otherwise, scaled to [-8, 8], then mapped through
``sign(x) * log2(|x| + 1) / log2(8)``.
"""

from __future__ import annotations

import math

import numpy as np

# The reference pushes non-neighbour pairs of a shifted window this far down,
# and applies the mask twice (see the module docstring).
MASK_FILL = -100.0
MASK_APPLICATIONS = 2
# The reference squashes the continuous position bias to (0, 16).
BIAS_RANGE = 16.0
# The reference clamps each head's log temperature at log(1 / 0.01).
MAX_LOG_SCALE = math.log(1.0 / 0.01)


def relative_coords_table(window: int, pretrained_window: int) -> np.ndarray:
    """The ``[2w - 1, 2w - 1, 2]`` log-spaced offset table the bias MLP reads."""
    offsets = np.arange(-(window - 1), window, dtype=np.float64)
    table = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1)
    if pretrained_window > 0:
        table = table / (pretrained_window - 1)
    elif window > 1:
        table = table / (window - 1)
    table = table * 8.0
    return np.sign(table) * np.log2(np.abs(table) + 1.0) / math.log2(8.0)


def relative_position_index(window: int) -> np.ndarray:
    """Row ``i * A + j`` names the table entry for query position ``i`` and key ``j``."""
    rows, cols = np.meshgrid(np.arange(window), np.arange(window), indexing="ij")
    coords = np.stack([rows.reshape(-1), cols.reshape(-1)])
    relative = coords[:, :, None] - coords[:, None, :]
    relative = relative.transpose(1, 2, 0) + (window - 1)
    return relative[:, :, 0] * (2 * window - 1) + relative[:, :, 1]


def position_bias(
    first_weight: np.ndarray,
    first_bias: np.ndarray,
    second_weight: np.ndarray,
    *,
    window: int,
    pretrained_window: int,
) -> np.ndarray:
    """``16 * sigmoid(MLP(table))`` gathered to ``[heads, A, A]`` for one block."""
    table = relative_coords_table(window, pretrained_window).reshape(-1, 2)
    hidden = table @ np.asarray(first_weight, dtype=np.float64).T
    hidden = np.maximum(hidden + np.asarray(first_bias, dtype=np.float64), 0.0)
    per_offset = hidden @ np.asarray(second_weight, dtype=np.float64).T
    heads = int(per_offset.shape[1])
    area = window * window
    gathered = per_offset[relative_position_index(window).reshape(-1)].reshape(area, area, heads)
    return BIAS_RANGE / (1.0 + np.exp(-np.transpose(gathered, (2, 0, 1))))


def shift_mask(height: int, width: int, window: int, shift: int) -> np.ndarray:
    """The ``[windows, A, A]`` mask of a shifted block, already at its applied depth."""
    region = np.zeros((height, width), dtype=np.float64)
    count = 0
    for rows in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for cols in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            region[rows, cols] = count
            count += 1
    windows = region.reshape(height // window, window, width // window, window)
    windows = windows.transpose(0, 2, 1, 3).reshape(-1, window * window)
    different = windows[:, None, :] != windows[:, :, None]
    return np.where(different, MASK_FILL * MASK_APPLICATIONS, 0.0)


def logit_scale(raw: np.ndarray) -> np.ndarray:
    """Each head's cosine temperature: ``exp(min(raw, log(100)))``."""
    return np.exp(np.minimum(np.asarray(raw, dtype=np.float64).reshape(-1), MAX_LOG_SCALE))


def attention_bias(
    first_weight: np.ndarray,
    first_bias: np.ndarray,
    second_weight: np.ndarray,
    *,
    height: int,
    width: int,
    window: int,
    shift: int,
    pretrained_window: int,
) -> np.ndarray:
    """The additive bias of one block: ``[1 or windows, heads, A, A]``, float32."""
    bias = position_bias(
        first_weight, first_bias, second_weight, window=window, pretrained_window=pretrained_window
    )[None]
    if shift:
        bias = bias + shift_mask(height, width, window, shift)[:, None]
    return np.ascontiguousarray(bias, dtype=np.float32)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time tables for SAM's decomposed relative-position attention.

HF SAM adds a query-dependent bias to every attention logit of a
``size x size`` grid (``get_decomposed_rel_pos``)::

    logit[(i, j), (y, x)] = scale * q[i, j] . k[y, x]
                            + q[i, j] . R_h[i, y] + q[i, j] . R_w[j, x]

Materializing that bias costs a ``[batch, heads, size^2, size^2]`` tensor per
layer. The same logits come out of one Q.K^T product over augmented rows: the
query row carries ``q . R_h[i, :] / scale`` and ``q . R_w[j, :] / scale``, the
key row a one-hot of its row ``y`` and column ``x``, and the product is then
scaled as before. These helpers build the two constant tables; the graph
wiring lives in ``model.py``.
"""

from __future__ import annotations

import numpy as np

# Each appended block is padded to this many columns (zeros on both sides), so
# the augmented head dimension stays aligned for the fused attention kernels.
_BLOCK_ALIGNMENT = 8


def rel_pos_block_width(size: int) -> int:
    """Columns of one appended block (row or column terms) for a grid side."""
    if size <= 0:
        raise ValueError(f"rel-pos grid side must be positive, got {size}")
    return -(-size // _BLOCK_ALIGNMENT) * _BLOCK_ALIGNMENT


def rel_pos_query_table(rel_pos: np.ndarray, logit_scale: float) -> np.ndarray:
    """Per-query-row projection table, ``[size, head_dim, block_width]``.

    ``rel_pos`` is ``get_rel_pos``'s ``[size, size, head_dim]`` table: entry
    ``[i, y]`` holds the embedding for query row/column ``i`` against key
    row/column ``y``. Multiplying query rows that share ``i`` by slice ``[i]``
    gives their terms divided by ``logit_scale``, padded with zero columns.
    """
    if rel_pos.ndim != 3 or rel_pos.shape[0] != rel_pos.shape[1]:
        raise ValueError(f"expected a [size, size, head_dim] table, got {rel_pos.shape}")
    if not logit_scale > 0.0:
        raise ValueError(f"logit scale must be positive, got {logit_scale}")
    size, _, head_dim = rel_pos.shape
    table = np.zeros((size, head_dim, rel_pos_block_width(size)), dtype=np.float32)
    table[:, :, :size] = rel_pos.transpose(0, 2, 1).astype(np.float32) / np.float32(logit_scale)
    return table


def rel_pos_key_table(size: int) -> np.ndarray:
    """One-hot key position rows, ``[size * size, 2 * block_width]``.

    Key ``t = y * size + x`` gets a one in column ``y`` of the row block and in
    column ``x`` of the column block; padding columns stay zero.
    """
    width = rel_pos_block_width(size)
    table = np.zeros((size * size, 2 * width), dtype=np.float32)
    key = np.arange(size * size)
    table[key, key // size] = 1.0
    table[key, width + key % size] = 1.0
    return table

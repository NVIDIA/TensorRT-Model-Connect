# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-sized positions: sine embeddings and deformable-attention reference points.

``OneFormerSinePositionEmbedding(normalize=True)`` with an all-false mask (one
image, no padding) reduces to 1-based row and column indices divided by the
extent plus 1e-6 and scaled by 2*pi; ``dim_t = 10000 ** (2 * floor(i / 2) / F)``
pairs equal frequencies, so ``sin`` and ``cos`` of the same angle interleave.
The y half comes before the x half. The pixel decoder's reference points are
the half-pixel centres divided by the extent (valid ratios are exactly 1).
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

E = trt.ElementWiseOperation


def sine_dim_t(num_pos_feats: int) -> np.ndarray:
    """The reference's float32 ``dim_t``, computed with torch exactly as the model does."""
    import torch

    dim_t = torch.arange(num_pos_feats, dtype=torch.int64).to(torch.float32)
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    return dim_t.numpy().astype(np.float32)


def _normalised(g, embed, extent):
    """``embed / (extent + 1e-6) * 2*pi`` in float32, extent an int32 (1,) tensor."""
    denominator = g.add(g.reshape(g.as_float(extent), (1, 1)), g.f32([1e-6], (1, 1)))
    ratio = g.ew(embed, denominator, E.DIV)
    return g.ew(ratio, g.f32([2.0 * math.pi], (1, 1)), E.PROD)


def _interleaved(g, angles_base, dim_t_even):
    """sin/cos pairs of ``angles_base / dim_t``: [sin a0, cos a0, sin a1, cos a1, ...]."""
    count = int(dim_t_even.shape[0])
    angles = g.ew(g.reshape(angles_base, (0, 0, 1)), g.f32(dim_t_even, (1, 1, count)), E.DIV)
    sine = g.reshape(g.unary(angles, trt.UnaryOperation.SIN), (0, 0, count, 1))
    cosine = g.reshape(g.unary(angles, trt.UnaryOperation.COS), (0, 0, count, 1))
    return g.reshape(g.concat([sine, cosine], 3), (0, 0, 2 * count))


def sine_tokens(g, height, width, dim_t: np.ndarray):
    """Sine positions for an (height, width) map as tokens (1, height*width, 2*F), F = len(dim_t)."""
    if not np.array_equal(dim_t[0::2], dim_t[1::2]):
        raise ValueError("oneformer sine frequencies must come in equal pairs")
    dim_t_even = dim_t[0::2]
    y_embed = _normalised(g, g.grid([height, width], 1.0, [1.0, 0.0]), height)
    x_embed = _normalised(g, g.grid([height, width], 1.0, [0.0, 1.0]), width)
    position = g.concat(
        [_interleaved(g, y_embed, dim_t_even), _interleaved(g, x_embed, dim_t_even)], 2
    )
    return g.reshape(position, (1, -1, 2 * int(dim_t.shape[0])))


def reference_points(g, height, width):
    """Deformable-attention reference points (1, height*width, 2) as normalised (x, y) centres."""
    ref_y = g.ew(
        g.grid([height, width], 0.5, [1.0, 0.0]), g.reshape(g.as_float(height), (1, 1)), E.DIV
    )
    ref_x = g.ew(
        g.grid([height, width], 0.5, [0.0, 1.0]), g.reshape(g.as_float(width), (1, 1)), E.DIV
    )
    pairs = g.concat([g.reshape(ref_x, (0, 0, 1)), g.reshape(ref_y, (0, 0, 1))], 2)
    return g.reshape(pairs, (1, -1, 2))

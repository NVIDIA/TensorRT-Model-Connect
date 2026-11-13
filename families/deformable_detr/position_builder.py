# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sine position embeddings plus the per-level embedding.

The single-image path preserves aspect ratio without padding, so the mask is
all ones. Runtime feature-map geometry determines the positions. The NumPy
implementation is retained as an independent oracle for the dynamic graph.

The formula is read from ``DeformableDetrSinePositionEmbedding.forward``:

* ``num_position_features`` is 128 - half the model width, with the y half
  concatenated **before** the x half;
* normalisation carries a ``- 0.5`` offset that plain DETR does not have:
  ``(embed - 0.5) / (last + 1e-6) * 2*pi``;
* ``dim_t = 10000 ** (2 * floor(i / 2) / 128)``, and sin takes the even
  indices while cos takes the odd ones, interleaved - not the two halves split
  down the middle, which is the other common convention and gives a
  same-shaped, wrong answer.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

_FEATURES = 128
_TEMPERATURE = 10000.0
_SCALE = 2.0 * np.pi
_EPS = 1e-6


def sine_embedding(height: int, width: int, dtype=np.float32) -> np.ndarray:
    """Return ``(1, height * width, 256)`` sine position embeddings."""
    # An all-ones mask makes the cumulative sums the 1-based row and column
    # indices, which is what the reference reduces to when nothing is padded.
    y_embed = np.tile(np.arange(1, height + 1, dtype=np.float64)[:, None], (1, width))
    x_embed = np.tile(np.arange(1, width + 1, dtype=np.float64)[None, :], (height, 1))
    y_embed = (y_embed - 0.5) / (height + _EPS) * _SCALE
    x_embed = (x_embed - 0.5) / (width + _EPS) * _SCALE

    index = np.arange(_FEATURES, dtype=np.float64)
    dim_t = _TEMPERATURE ** (2.0 * np.floor(index / 2.0) / _FEATURES)

    def interleave(embed):
        raw = embed[:, :, None] / dim_t
        return np.stack((np.sin(raw[:, :, 0::2]), np.cos(raw[:, :, 1::2])), axis=3).reshape(
            height, width, _FEATURES
        )

    # y first, then x - the reference concatenates in that order.
    pos = np.concatenate((interleave(y_embed), interleave(x_embed)), axis=2)
    return pos.reshape(1, height * width, 2 * _FEATURES).astype(dtype)


def build_level_embeddings(shapes, level_embed, dtype=np.float32) -> np.ndarray:
    """Position embeddings for every level, each offset by its level embedding.

    ``shapes`` is the measured pyramid as ``(height, width)`` pairs; the result
    is the flattened ``(1, sum(h * w), 256)`` tensor the encoder consumes.
    """
    if len(shapes) != len(level_embed):
        raise ValueError("deformable_detr level count does not match level_embed")
    pieces = []
    for level, (height, width) in enumerate(shapes):
        embedding = sine_embedding(height, width, dtype)
        pieces.append(embedding + np.asarray(level_embed[level], dtype=dtype).reshape(1, 1, -1))
    return np.concatenate(pieces, axis=1)


def build_dynamic_level_embeddings(network, shapes, level_embed, dtype=np.float32):
    """Compute the reference's unpadded positions at the input's actual aspect ratio."""
    frequencies = _TEMPERATURE ** (np.arange(_FEATURES // 2, dtype=np.float32) / (_FEATURES // 2))
    frequency = g.add_constant(network, (1, 1, _FEATURES // 2), frequencies)
    pieces = []
    for level, spatial_shape in enumerate(shapes):
        axes = []
        for axis in (0, 1):
            grid = g.coordinate_grid(network, spatial_shape, axis)
            extent = g.shape_indices(network, spatial_shape, [axis])
            extent = network.add_cast(extent, trt.float32).get_output(0)
            extent = network.add_elementwise(
                extent, g.add_constant(network, (1,), [_EPS]),
                trt.ElementWiseOperation.SUM).get_output(0)
            denominator = g.reshape(network, extent, (1, 1))
            grid = network.add_elementwise(grid, denominator, trt.ElementWiseOperation.DIV).get_output(0)
            grid = network.add_elementwise(
                grid, g.add_constant(network, (1, 1), [_SCALE]),
                trt.ElementWiseOperation.PROD).get_output(0)
            angles = network.add_elementwise(
                g.reshape(network, grid, (1, -1, 1)), frequency,
                trt.ElementWiseOperation.DIV).get_output(0)
            sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
            cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
            pairs = g.concat(network, [g.reshape(network, sine, (1, -1, _FEATURES // 2, 1)),
                                       g.reshape(network, cosine, (1, -1, _FEATURES // 2, 1))], axis=3)
            axes.append(g.reshape(network, pairs, (1, -1, _FEATURES)))
        position = g.concat(network, axes, axis=2)
        position = network.add_elementwise(
            position, g.add_constant(network, (1, 1, 2 * _FEATURES), level_embed[level]),
            trt.ElementWiseOperation.SUM).get_output(0)
        if dtype == np.float16:
            position = network.add_cast(position, trt.float16).get_output(0)
        pieces.append(position)
    return g.concat(network, pieces, axis=1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-scale deformable attention, built from TensorRT's own grid sample.

No CUDA plugin is needed. ``IGridSampleLayer`` reproduces torch's
``grid_sample`` to about 2.4e-07, and the pairing that matters is
``SampleMode.FILL`` for ``padding_mode="zeros"`` - ``CLAMP`` is ``"border"``
and silently changes every sample that falls outside a feature map.

Two rules here are stated in the reference and invisible in the shapes:

1. **The query is posed, the value is not.** ``sampling_offsets`` and
   ``attention_weights`` read ``hidden_states + position_embeddings``, while
   ``value_proj`` reads the raw hidden states. Feeding the unposed tensor to
   all three is the mistake that cost RT-DETR a max difference of 8.883
   against 4.8e-07.
2. **This checkpoint takes the centre-only branch.** With two-dimensional
   reference points the update is
   ``ref_xy + offsets / (W, H)`` per level. The four-dimensional box branch
   (``ref_xy + offsets / n_points * ref_wh * 0.5``) belongs to the
   box-refinement variants; cross-applying the two scores about 0.75.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g


def _shuffle(network, x, *, first=None, reshape=None, second=None):
    layer = network.add_shuffle(x)
    if first is not None:
        layer.first_transpose = first
    if reshape is not None:
        layer.reshape_dims = reshape
    if second is not None:
        layer.second_transpose = second
    return layer.get_output(0)


def _slice(network, x, start, size):
    stride = (1,) * len(start)
    return network.add_slice(x, start, size, stride).get_output(0)


def build_deformable_attention(network, hidden, posed, reference, weights, prefix,
                               shapes, heads, points, dtype=np.float32, value_source=None):
    """One multi-scale deformable attention block.

    ``hidden`` is ``(1, queries, width)``; ``posed`` is the same tensor with the
    position embeddings already added. ``reference`` is the
    ``(1, queries, levels, 2)`` tensor of normalised centres.

    ``value_source`` is what the values are sampled from. The encoder attends to
    itself and leaves it unset; the decoder passes the encoder output, whose
    token count is the pyramid's, not the query count.
    """
    queries = int(hidden.shape[1])
    width = int(hidden.shape[2])
    if value_source is None:
        value_source = hidden
    levels = len(shapes)
    head_dim = width // heads

    # value comes from the UNPOSED source tensor.
    value = g.add_linear(network, value_source, weights[f"{prefix}.value_proj.weight"],
                         weights[f"{prefix}.value_proj.bias"], dtype)

    # offsets and weights come from the POSED hidden states.
    offsets = g.add_linear(network, posed, weights[f"{prefix}.sampling_offsets.weight"],
                           weights[f"{prefix}.sampling_offsets.bias"], dtype)
    offsets = _shuffle(network, offsets, reshape=(queries, heads, levels, points, 2))

    scores = g.add_linear(network, posed, weights[f"{prefix}.attention_weights.weight"],
                          weights[f"{prefix}.attention_weights.bias"], dtype)
    scores = _shuffle(network, scores, reshape=(queries, heads, levels * points))
    # Softmax spans every level and point together, not each level separately.
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    scores = softmax.get_output(0)

    # offsets / (W, H) per level, then the grid convention: 2 * location - 1.
    normaliser = np.empty((1, 1, levels, 1, 2), dtype=dtype)
    for level, (height, level_width) in enumerate(shapes):
        normaliser[0, 0, level, 0, 0] = level_width
        normaliser[0, 0, level, 0, 1] = height
    divisor = g.add_constant(network, normaliser.shape, normaliser, dtype)
    scaled = network.add_elementwise(offsets, divisor, trt.ElementWiseOperation.DIV).get_output(0)

    # reference is (1, queries, levels, 2); broadcast it over heads and points.
    ref = _shuffle(network, reference, reshape=(queries, 1, levels, 1, 2))
    location = network.add_elementwise(ref, scaled, trt.ElementWiseOperation.SUM).get_output(0)
    two = g.add_constant(network, (1, 1, 1, 1, 1), np.array([2.0], dtype=dtype), dtype)
    one = g.add_constant(network, (1, 1, 1, 1, 1), np.array([1.0], dtype=dtype), dtype)
    grid = network.add_elementwise(location, two, trt.ElementWiseOperation.PROD).get_output(0)
    grid = network.add_elementwise(grid, one, trt.ElementWiseOperation.SUB).get_output(0)

    # value split per level and laid out as (heads, head_dim, H, W).
    sampled = []
    start = 0
    for level, (height, level_width) in enumerate(shapes):
        count = height * level_width
        piece = _slice(network, value, (0, start, 0), (1, count, width))
        piece = _shuffle(network, piece, reshape=(count, heads, head_dim),
                         second=(1, 2, 0))
        piece = _shuffle(network, piece, reshape=(heads, head_dim, height, level_width))

        # grid for this level: (queries, heads, points, 2) -> (heads, queries, points, 2)
        level_grid = _slice(network, grid, (0, 0, level, 0, 0),
                            (queries, heads, 1, points, 2))
        level_grid = _shuffle(network, level_grid, reshape=(queries, heads, points, 2),
                              second=(1, 0, 2, 3))

        layer = network.add_grid_sample(piece, level_grid)
        layer.interpolation_mode = trt.InterpolationMode.LINEAR
        # FILL is padding_mode="zeros". CLAMP would be "border".
        layer.sample_mode = trt.SampleMode.FILL
        layer.align_corners = False
        sampled.append(layer.get_output(0))
        start += count

    # (heads, head_dim, queries, points) per level -> (heads, head_dim, queries, levels*points)
    joined = g.concat(network, sampled, axis=3)

    # scores (queries, heads, levels*points) -> (heads, 1, queries, levels*points)
    weighted = _shuffle(network, scores, first=(1, 0, 2),
                        reshape=(heads, 1, queries, levels * points))
    product = network.add_elementwise(
        joined, weighted, trt.ElementWiseOperation.PROD).get_output(0)
    reduced = network.add_reduce(
        product, trt.ReduceOperation.SUM, 1 << 3, keep_dims=False).get_output(0)

    # (heads, head_dim, queries) -> (1, queries, heads * head_dim)
    out = _shuffle(network, reduced, reshape=(1, heads * head_dim, queries),
                   second=(0, 2, 1))
    return g.add_linear(network, out, weights[f"{prefix}.output_proj.weight"],
                        weights[f"{prefix}.output_proj.bias"], dtype)

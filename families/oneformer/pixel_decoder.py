# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneFormer pixel decoder: multi-scale deformable-attention encoder plus one FPN level.

Order and conventions read from ``OneFormerPixelDecoder.forward``:

* levels run coarse to fine: stage 4 (stride 32), stage 3, stage 2, each
  projected by a 1x1 convolution and GroupNorm(32);
* sources and positions are flattened level by level; the encoder's queries are
  ``hidden + position`` (level embedding included) for the sampling offsets and
  attention weights, while values come from the unposed hidden states;
* sampling locations are ``reference + offset / (W, H)`` per level, sampled with
  ``grid_sample(bilinear, zeros, align_corners=False)`` at ``2 * location - 1``,
  and one softmax spans all levels and points of a head;
* encoder layers are post-norm: residual + attention, LayerNorm, residual +
  ReLU MLP, LayerNorm;
* the FPN adds the stride-8 output, bilinearly resized to the stride-4 map, to
  the lateral 1x1 convolution of stage 1, then applies a 3x3 convolution,
  GroupNorm and ReLU; the mask projection is a 1x1 convolution of that map.
"""

from __future__ import annotations

import tensorrt as trt

from .config import DEFORMABLE_POINTS, NUM_FEATURE_LEVELS, NUM_GROUPS
from .position import reference_points, sine_tokens

E = trt.ElementWiseOperation
PREFIX = "model.pixel_level_module.decoder"


def _tokens(g, feature):
    """(1, C, h, w) -> (1, h*w, C)."""
    return g.reshape(feature, (1, int(feature.shape[1]), -1), second=(0, 2, 1))


def _deformable_attention(g, hidden, query, reference, shapes, starts, weights, prefix, heads):
    """One OneFormerPixelDecoderEncoderMultiscaleDeformableAttention block on (1, L, C) tokens."""
    channels = int(hidden.shape[2])
    head_dim = channels // heads
    levels, points = len(shapes), DEFORMABLE_POINTS
    value = g.linear(
        hidden, weights.get(f"{prefix}.value_proj.weight"), weights.get(f"{prefix}.value_proj.bias")
    )
    offsets = g.linear(
        query,
        weights.get(f"{prefix}.sampling_offsets.weight"),
        weights.get(f"{prefix}.sampling_offsets.bias"),
    )
    offsets = g.reshape(offsets, (-1, heads, levels, points, 2))
    scores = g.linear(
        query,
        weights.get(f"{prefix}.attention_weights.weight"),
        weights.get(f"{prefix}.attention_weights.bias"),
    )
    scores = g.softmax(g.reshape(scores, (-1, heads, levels * points)), 2)

    normaliser = g.concat([g.vec([width, height]) for height, width in shapes], 0)
    normaliser = g.reshape(g.as_float(normaliser), (1, 1, levels, 1, 2))
    location = g.add(g.reshape(reference, (-1, 1, 1, 1, 2)), g.ew(offsets, normaliser, E.DIV))
    grid = g.ew(g.ew(location, g.scalar(location, 2.0), E.PROD), g.scalar(location, 1.0), E.SUB)

    sampled = []
    for level, ((height, width), start) in enumerate(zip(shapes, starts)):
        count = g.iop(height, width, E.PROD)
        piece = g.slice(value, [0, start, 0], [1, count, channels])
        piece = g.reshape(piece, [heads, head_dim, height, width], first=(0, 2, 1))
        level_grid = g.slice(grid, [0, 0, level, 0, 0], [g.dim(grid, 0), heads, 1, points, 2])
        level_grid = g.reshape(level_grid, (-1, heads, points, 2), second=(1, 0, 2, 3))
        sampled.append(g.grid_sample(piece, level_grid))
    joined = g.concat(sampled, 3)
    weighted = g.reshape(scores, (heads, 1, -1, levels * points), first=(1, 0, 2))
    reduced = g.network.add_reduce(
        g.ew(joined, weighted, E.PROD), trt.ReduceOperation.SUM, 1 << 3, False
    ).get_output(0)
    out = g.reshape(reduced, (1, channels, -1), second=(0, 2, 1))
    return g.linear(
        out, weights.get(f"{prefix}.output_proj.weight"), weights.get(f"{prefix}.output_proj.bias")
    )


def _encoder_layer(g, hidden, position, reference, shapes, starts, weights, prefix, heads, eps):
    query = g.add(hidden, position)
    attended = _deformable_attention(
        g, hidden, query, reference, shapes, starts, weights, f"{prefix}.self_attn", heads
    )
    hidden = g.layer_norm(
        g.add(hidden, attended),
        weights.get(f"{prefix}.self_attn_layer_norm.weight"),
        weights.get(f"{prefix}.self_attn_layer_norm.bias"),
        eps,
    )
    mlp = g.relu(
        g.linear(hidden, weights.get(f"{prefix}.fc1.weight"), weights.get(f"{prefix}.fc1.bias"))
    )
    mlp = g.linear(mlp, weights.get(f"{prefix}.fc2.weight"), weights.get(f"{prefix}.fc2.bias"))
    return g.layer_norm(
        g.add(hidden, mlp),
        weights.get(f"{prefix}.final_layer_norm.weight"),
        weights.get(f"{prefix}.final_layer_norm.bias"),
        eps,
    )


def build_pixel_decoder(g, features, weights, config, dim_t):
    """Return ([stride-32, stride-16, stride-8] maps, stride-4 mask features), all NCHW."""
    eps = config.layer_norm_eps
    level_embed = weights.get(f"{PREFIX}.level_embed")
    sources, positions, references, shapes = [], [], [], []
    for level, feature in enumerate(features[::-1][:NUM_FEATURE_LEVELS]):
        projected = g.conv(
            feature,
            weights.get(f"{PREFIX}.input_projections.{level}.0.weight"),
            weights.get(f"{PREFIX}.input_projections.{level}.0.bias"),
        )
        projected = g.group_norm(
            projected,
            weights.get(f"{PREFIX}.input_projections.{level}.1.weight"),
            weights.get(f"{PREFIX}.input_projections.{level}.1.bias"),
            NUM_GROUPS,
            1e-5,
        )
        height, width = g.dim(projected, 2), g.dim(projected, 3)
        shapes.append((height, width))
        sources.append(_tokens(g, projected))
        position = sine_tokens(g, height, width, dim_t)
        positions.append(g.add(position, g.constant(level_embed[level], shape=(1, 1, -1))))
        references.append(reference_points(g, height, width))
    hidden = g.concat(sources, 1)
    position = g.concat(positions, 1)
    reference = g.concat(references, 1)
    starts, start = [], 0
    for height, width in shapes:
        starts.append(start)
        count = g.iop(height, width, E.PROD)
        start = count if isinstance(start, int) and start == 0 else g.iop(start, count, E.SUM)

    heads = config.num_attention_heads
    for layer in range(config.encoder_layers):
        hidden = _encoder_layer(
            g,
            hidden,
            position,
            reference,
            shapes,
            starts,
            weights,
            f"{PREFIX}.encoder.layers.{layer}",
            heads,
            eps,
        )

    channels = config.conv_dim
    outputs = []
    for (height, width), begin in zip(shapes, starts):
        count = g.iop(height, width, E.PROD)
        piece = g.slice(hidden, [0, begin, 0], [1, count, channels])
        outputs.append(g.reshape(piece, [1, channels, height, width], first=(0, 2, 1)))

    lateral = g.conv(features[0], weights.get(f"{PREFIX}.adapter_1.0.weight"))
    lateral = g.group_norm(
        lateral,
        weights.get(f"{PREFIX}.adapter_1.1.weight"),
        weights.get(f"{PREFIX}.adapter_1.1.bias"),
        NUM_GROUPS,
        1e-5,
    )
    fpn = g.add(lateral, g.resize_bilinear(outputs[-1], g.shape(lateral)))
    fpn = g.conv(fpn, weights.get(f"{PREFIX}.layer_1.0.weight"), padding=1)
    fpn = g.relu(
        g.group_norm(
            fpn,
            weights.get(f"{PREFIX}.layer_1.1.weight"),
            weights.get(f"{PREFIX}.layer_1.1.bias"),
            NUM_GROUPS,
            1e-5,
        )
    )
    mask_features = g.conv(
        fpn,
        weights.get(f"{PREFIX}.mask_projection.weight"),
        weights.get(f"{PREFIX}.mask_projection.bias"),
    )
    return outputs, mask_features

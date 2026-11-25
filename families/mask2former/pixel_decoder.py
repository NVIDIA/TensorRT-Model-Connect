# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Mask2Former pixel decoder: a multi-scale deformable-attention encoder over the three coarsest backbone levels,
one FPN level on the finest one, and the mask-feature projection (`Mask2FormerPixelDecoder.forward`).

Multi-scale deformable attention is written with TensorRT's grid sample. `SampleMode.FILL` is torch's
`padding_mode="zeros"`; the query and the sampling offsets read the hidden states plus the position embeddings, while
the values read the plain hidden states. Nothing is padded, so every level's valid ratio is one and the reference
points are the normalized cell centres of each token's own level, repeated for every level.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g
from .config import ENCODER_POINTS, FEATURE_LEVELS, GROUP_NORM_GROUPS, NORM_EPS, Mask2FormerConfig

_PREFIX = "model.pixel_level_module.decoder"
_TEMPERATURE = 10000.0
_SINE_EPS = 1e-6


def sine_position_embedding(height: int, width: int, features: int) -> np.ndarray:
    """`Mask2FormerSinePositionEmbedding(normalize=True)` without a mask: (height * width, 2 * features).

    Rows are the flattened (y, x) positions; the y half comes first, and each half interleaves sin of the even and
    cos of the odd channels.
    """

    scale = 2.0 * np.pi
    y_embed = np.arange(1, height + 1, dtype=np.float64) / (height + _SINE_EPS) * scale
    x_embed = np.arange(1, width + 1, dtype=np.float64) / (width + _SINE_EPS) * scale
    dim_t = _TEMPERATURE ** (2.0 * np.floor(np.arange(features, dtype=np.float64) / 2.0) / features)

    def encode(values: np.ndarray) -> np.ndarray:
        raw = values[:, None] / dim_t
        return np.stack((np.sin(raw[:, 0::2]), np.cos(raw[:, 1::2])), axis=2).reshape(
            len(values), features
        )

    pos_y = np.repeat(encode(y_embed)[:, None, :], width, axis=1)
    pos_x = np.repeat(encode(x_embed)[None, :, :], height, axis=0)
    return (
        np.concatenate((pos_y, pos_x), axis=2)
        .reshape(height * width, 2 * features)
        .astype(np.float32)
    )


def reference_points(shapes) -> np.ndarray:
    """Normalized cell centres in (x, y) order for every token of every level: (tokens, 2), float32 as torch."""

    pieces = []
    for height, width in shapes:
        ref_y = (np.arange(height, dtype=np.float32) + np.float32(0.5)) / np.float32(height)
        ref_x = (np.arange(width, dtype=np.float32) + np.float32(0.5)) / np.float32(width)
        grid_y, grid_x = np.meshgrid(ref_y, ref_x, indexing="ij")
        pieces.append(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1))
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _deformable_attention(network, hidden, posed, reference, weights, prefix, shapes, heads):
    """One multi-scale deformable self-attention over the flattened levels: hidden is (1, tokens, width)."""

    _, tokens, width = g.shape(hidden)
    levels = len(shapes)
    head_dim = width // heads
    points = ENCODER_POINTS

    value = g.linear(
        network,
        hidden,
        weights[f"{prefix}.value_proj.weight"],
        weights[f"{prefix}.value_proj.bias"],
    )
    offsets = g.linear(
        network,
        posed,
        weights[f"{prefix}.sampling_offsets.weight"],
        weights[f"{prefix}.sampling_offsets.bias"],
    )
    offsets = g.reshape(network, offsets, (tokens, heads, levels, points, 2))
    scores = g.linear(
        network,
        posed,
        weights[f"{prefix}.attention_weights.weight"],
        weights[f"{prefix}.attention_weights.bias"],
    )
    # One softmax across every level and point of a head.
    scores = g.softmax(network, g.reshape(network, scores, (tokens, heads, levels * points)), 2)

    # location = reference + offset / (W, H) of the sampled level; grid = 2 * location - 1.
    normalizer = np.asarray([[width_l, height_l] for height_l, width_l in shapes], dtype=np.float32)
    scaled = g.div(network, offsets, g.constant(network, normalizer.reshape(1, 1, levels, 1, 2)))
    location = g.add(network, g.constant(network, reference.reshape(tokens, 1, 1, 1, 2)), scaled)
    grid = g.sub(
        network,
        g.mul(network, location, g.scalar_like(network, 2.0, 5)),
        g.scalar_like(network, 1.0, 5),
    )

    sampled = []
    start = 0
    for level, (height_l, width_l) in enumerate(shapes):
        count = height_l * width_l
        piece = g.slice_static(network, value, (0, start, 0), (1, count, width))
        # (count, heads, head_dim) -> (heads, head_dim, H, W)
        piece = g.shuffle(network, piece, reshape=(count, heads, head_dim), transpose=(1, 2, 0))
        piece = g.reshape(network, piece, (heads, head_dim, height_l, width_l))
        level_grid = g.slice_static(
            network, grid, (0, 0, level, 0, 0), (tokens, heads, 1, points, 2)
        )
        level_grid = g.shuffle(
            network, level_grid, reshape=(tokens, heads, points, 2), transpose=(1, 0, 2, 3)
        )
        layer = network.add_grid_sample(piece, level_grid)
        if layer is None:
            raise RuntimeError("TensorRT rejected the Mask2Former grid sample")
        layer.interpolation_mode = trt.InterpolationMode.LINEAR
        layer.sample_mode = trt.SampleMode.FILL
        layer.align_corners = False
        sampled.append(layer.get_output(0))  # (heads, head_dim, tokens, points)
        start += count

    joined = g.concat(
        network, sampled, 3
    )  # (heads, head_dim, tokens, levels * points), level-major
    weights_by_head = g.shuffle(
        network, scores, first_transpose=(1, 0, 2), reshape=(heads, 1, tokens, levels * points)
    )
    reduced = network.add_reduce(
        g.mul(network, joined, weights_by_head), trt.ReduceOperation.SUM, 1 << 3, False
    )
    if reduced is None:
        raise RuntimeError("TensorRT rejected the Mask2Former sampling reduction")
    output = g.shuffle(
        network, reduced.get_output(0), reshape=(1, width, tokens), transpose=(0, 2, 1)
    )
    return g.linear(
        network,
        output,
        weights[f"{prefix}.output_proj.weight"],
        weights[f"{prefix}.output_proj.bias"],
    )


def _encoder_layer(network, hidden, position, reference, weights, prefix, shapes, heads):
    posed = g.add(network, hidden, position)
    attended = _deformable_attention(
        network, hidden, posed, reference, weights, f"{prefix}.self_attn", shapes, heads
    )
    hidden = g.layer_norm(
        network,
        g.add(network, hidden, attended),
        weights[f"{prefix}.self_attn_layer_norm.weight"],
        weights[f"{prefix}.self_attn_layer_norm.bias"],
        NORM_EPS,
    )
    inner = g.relu(
        network,
        g.linear(network, hidden, weights[f"{prefix}.fc1.weight"], weights[f"{prefix}.fc1.bias"]),
    )
    inner = g.linear(network, inner, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"])
    return g.layer_norm(
        network,
        g.add(network, hidden, inner),
        weights[f"{prefix}.final_layer_norm.weight"],
        weights[f"{prefix}.final_layer_norm.bias"],
        NORM_EPS,
    )


def build_pixel_decoder(network, features, weights, config: Mask2FormerConfig):
    """Return the three multi-scale feature maps (coarsest first) and the mask features, all (1, C, H, W)."""

    width = config.feature_size
    levels = features[::-1][:FEATURE_LEVELS]
    shapes = [tuple(g.shape(level)[2:]) for level in levels]
    projected = []
    for index, level in enumerate(levels):
        base = f"{_PREFIX}.input_projections.{index}"
        projected.append(
            g.group_norm(
                network,
                g.conv2d(network, level, weights[f"{base}.0.weight"], weights[f"{base}.0.bias"]),
                weights[f"{base}.1.weight"],
                weights[f"{base}.1.bias"],
                GROUP_NORM_GROUPS,
                NORM_EPS,
            )
        )
    hidden = g.concat(network, [g.tokens_from_nchw(network, level) for level in projected], 1)

    level_embed = np.asarray(weights[f"{_PREFIX}.level_embed"], dtype=np.float32)
    if level_embed.shape != (FEATURE_LEVELS, width):
        raise ValueError(f"mask2former pixel-decoder level_embed has shape {level_embed.shape}")
    position = np.concatenate(
        [
            sine_position_embedding(height, width_l, width // 2) + level_embed[index][None, :]
            for index, (height, width_l) in enumerate(shapes)
        ],
        axis=0,
    )
    position = g.constant(network, position[None])
    reference = reference_points(shapes)
    for layer in range(config.encoder_layers):
        hidden = _encoder_layer(
            network,
            hidden,
            position,
            reference,
            weights,
            f"{_PREFIX}.encoder.layers.{layer}",
            shapes,
            config.num_attention_heads,
        )

    outputs = []
    start = 0
    for height, width_l in shapes:
        count = height * width_l
        piece = g.slice_static(network, hidden, (0, start, 0), (1, count, width))
        outputs.append(g.nchw_from_tokens(network, piece, height, width_l))
        start += count

    finest = features[0]
    lateral = g.group_norm(
        network,
        g.conv2d(network, finest, weights[f"{_PREFIX}.adapter_1.0.weight"]),
        weights[f"{_PREFIX}.adapter_1.1.weight"],
        weights[f"{_PREFIX}.adapter_1.1.bias"],
        GROUP_NORM_GROUPS,
        NORM_EPS,
    )
    height, width_l = g.shape(finest)[2:]
    fpn = g.add(network, lateral, g.resize_bilinear(network, outputs[-1], height, width_l))
    fpn = g.relu(
        network,
        g.group_norm(
            network,
            g.conv2d(network, fpn, weights[f"{_PREFIX}.layer_1.0.weight"], padding=1),
            weights[f"{_PREFIX}.layer_1.1.weight"],
            weights[f"{_PREFIX}.layer_1.1.bias"],
            GROUP_NORM_GROUPS,
            NORM_EPS,
        ),
    )
    mask_features = g.conv2d(
        network,
        fpn,
        weights[f"{_PREFIX}.mask_projection.weight"],
        weights[f"{_PREFIX}.mask_projection.bias"],
    )
    return outputs, mask_features

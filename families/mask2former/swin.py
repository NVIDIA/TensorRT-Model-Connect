# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Swin backbone of Mask2Former, as `SwinBackbone` computes it.

Two properties of the backbone's call are easy to miss:

* it partitions windows unconditionally (`always_partition=True`), so a stage whose resolution equals the window
  size keeps the configured window and, in every odd block, the half-window cyclic shift and its region mask;
* each returned feature map is the stage output *before* patch merging, normalized by that stage's own LayerNorm.
"""

from __future__ import annotations

import math

import numpy as np

from . import graph as g
from .config import Mask2FormerConfig

_PREFIX = "model.pixel_level_module.encoder"
# The region mask Swin adds to attention scores across shifted-window seams.
_SEAM_MASK = -100.0


def relative_position_index(window: int) -> np.ndarray:
    """`SwinSelfAttention.create_relative_position_index` for a square window."""

    coords = np.stack(np.meshgrid(np.arange(window), np.arange(window), indexing="ij"))
    flat = coords.reshape(2, -1)
    relative = (flat[:, :, None] - flat[:, None, :]).transpose(1, 2, 0) + (window - 1)
    return relative[:, :, 0] * (2 * window - 1) + relative[:, :, 1]


def shifted_window_mask(height: int, width: int, window: int, shift: int) -> np.ndarray:
    """`SwinLayer.get_attn_mask`: (windows, window*window, window*window) of 0 and -100."""

    image = np.zeros((height, width), dtype=np.float32)
    count = 0
    for rows in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for columns in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            image[rows, columns] = count
            count += 1
    windows = (
        image.reshape(height // window, window, width // window, window)
        .transpose(0, 2, 1, 3)
        .reshape(-1, window * window)
    )
    difference = windows[:, None, :] - windows[:, :, None]
    return np.where(difference != 0, np.float32(_SEAM_MASK), np.float32(0.0)).astype(np.float32)


def _relative_bias(weights, prefix: str, window: int, heads: int) -> np.ndarray:
    table = np.asarray(weights[f"{prefix}.relative_position_bias_table"], dtype=np.float32)
    index = np.asarray(weights[f"{prefix}.relative_position_index"])
    expected = relative_position_index(window)
    if index.shape != expected.shape or not np.array_equal(index, expected):
        raise ValueError(
            f"mask2former {prefix} relative_position_index does not match a {window}x{window} window"
        )
    if table.shape != ((2 * window - 1) ** 2, heads):
        raise ValueError(
            f"mask2former {prefix} relative_position_bias_table has shape {table.shape}"
        )
    tokens = window * window
    return table[index.reshape(-1)].reshape(tokens, tokens, heads).transpose(2, 0, 1)[None]


def _window_attention(network, hidden, weights, prefix, *, heads, window, mask):
    """Attention inside each window: hidden is (windows, window*window, C)."""

    windows, tokens, channels = g.shape(hidden)
    head_dim = channels // heads
    query = g.split_heads(
        network,
        g.linear(
            network, hidden, weights[f"{prefix}.query.weight"], weights[f"{prefix}.query.bias"]
        ),
        heads,
    )
    key = g.split_heads(
        network,
        g.linear(network, hidden, weights[f"{prefix}.key.weight"], weights[f"{prefix}.key.bias"]),
        heads,
    )
    value = g.split_heads(
        network,
        g.linear(
            network, hidden, weights[f"{prefix}.value.weight"], weights[f"{prefix}.value.bias"]
        ),
        heads,
    )
    scores = g.matmul(network, query, key, transpose_right=True)
    scores = g.div(network, scores, g.scalar_like(network, math.sqrt(head_dim), 4))
    scores = g.add(
        network, scores, g.constant(network, _relative_bias(weights, prefix, window, heads))
    )
    if mask is not None:
        scores = g.add(network, scores, g.constant(network, mask[:, None]))
    context = g.matmul(network, g.softmax(network, scores, 3), value)
    return g.merge_heads(network, context)


def _block(
    network, hidden, weights, prefix, config: Mask2FormerConfig, *, resolution, heads, shift
):
    height, width = resolution
    _, _, channels = g.shape(hidden)
    window = config.window_size
    eps = config.backbone_eps
    shortcut = hidden
    hidden = g.layer_norm(
        network,
        hidden,
        weights[f"{prefix}.layernorm_before.weight"],
        weights[f"{prefix}.layernorm_before.bias"],
        eps,
    )
    hidden = g.reshape(network, hidden, (1, height, width, channels))
    if shift:
        hidden = g.roll(network, hidden, (-shift, -shift), (1, 2))
    rows, columns = height // window, width // window
    windows = g.shuffle(
        network,
        hidden,
        reshape=(rows, window, columns, window, channels),
        transpose=(0, 2, 1, 3, 4),
    )
    windows = g.reshape(network, windows, (rows * columns, window * window, channels))
    mask = shifted_window_mask(height, width, window, shift) if shift else None
    attended = _window_attention(
        network, windows, weights, f"{prefix}.attention.self", heads=heads, window=window, mask=mask
    )
    attended = g.linear(
        network,
        attended,
        weights[f"{prefix}.attention.output.dense.weight"],
        weights[f"{prefix}.attention.output.dense.bias"],
    )
    attended = g.shuffle(
        network,
        attended,
        reshape=(rows, columns, window, window, channels),
        transpose=(0, 2, 1, 3, 4),
    )
    attended = g.reshape(network, attended, (1, height, width, channels))
    if shift:
        attended = g.roll(network, attended, (shift, shift), (1, 2))
    hidden = g.add(network, shortcut, g.reshape(network, attended, (1, height * width, channels)))

    inner = g.layer_norm(
        network,
        hidden,
        weights[f"{prefix}.layernorm_after.weight"],
        weights[f"{prefix}.layernorm_after.bias"],
        eps,
    )
    inner = g.gelu(
        network,
        g.linear(
            network,
            inner,
            weights[f"{prefix}.intermediate.dense.weight"],
            weights[f"{prefix}.intermediate.dense.bias"],
        ),
    )
    inner = g.linear(
        network,
        inner,
        weights[f"{prefix}.output.dense.weight"],
        weights[f"{prefix}.output.dense.bias"],
    )
    return g.add(network, hidden, inner)


def _patch_merging(network, hidden, weights, prefix, *, resolution):
    height, width = resolution
    _, _, channels = g.shape(hidden)
    grid = g.reshape(network, hidden, (1, height, width, channels))
    half = (1, height // 2, width // 2, channels)
    # x[0::2, 0::2], x[1::2, 0::2], x[0::2, 1::2], x[1::2, 1::2] along (rows, columns), in this order.
    pieces = [
        g.slice_static(network, grid, (0, row, column, 0), half, (1, 2, 2, 1))
        for row, column in ((0, 0), (1, 0), (0, 1), (1, 1))
    ]
    merged = g.reshape(
        network, g.concat(network, pieces, 3), (1, (height // 2) * (width // 2), 4 * channels)
    )
    merged = g.layer_norm(
        network, merged, weights[f"{prefix}.norm.weight"], weights[f"{prefix}.norm.bias"], 1e-5
    )
    return g.linear(network, merged, weights[f"{prefix}.reduction.weight"])


def build_backbone(network, pixels, weights, config: Mask2FormerConfig):
    """Return the four normalized stage feature maps, each (1, C, H, W)."""

    hidden = g.conv2d(
        network,
        pixels,
        weights[f"{_PREFIX}.embeddings.patch_embeddings.projection.weight"],
        weights[f"{_PREFIX}.embeddings.patch_embeddings.projection.bias"],
        stride=config.patch_size,
    )
    hidden = g.tokens_from_nchw(network, hidden)
    hidden = g.layer_norm(
        network,
        hidden,
        weights[f"{_PREFIX}.embeddings.norm.weight"],
        weights[f"{_PREFIX}.embeddings.norm.bias"],
        1e-5,
    )

    features = []
    for stage, (depth, heads) in enumerate(zip(config.depths, config.num_heads)):
        resolution = config.stage_resolutions[stage]
        for block in range(depth):
            hidden = _block(
                network,
                hidden,
                weights,
                f"{_PREFIX}.encoder.layers.{stage}.blocks.{block}",
                config,
                resolution=resolution,
                heads=heads,
                shift=0 if block % 2 == 0 else config.window_size // 2,
            )
        name = f"stage{stage + 1}"
        normed = g.layer_norm(
            network,
            hidden,
            weights[f"{_PREFIX}.hidden_states_norms.{name}.weight"],
            weights[f"{_PREFIX}.hidden_states_norms.{name}.bias"],
            1e-5,
        )
        features.append(g.nchw_from_tokens(network, normed, *resolution))
        if stage < len(config.depths) - 1:
            hidden = _patch_merging(
                network,
                hidden,
                weights,
                f"{_PREFIX}.encoder.layers.{stage}.downsample",
                resolution=resolution,
            )
    return features

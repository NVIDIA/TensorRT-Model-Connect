# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision tower of the GR00T N1.7 backbone, for fixed-size camera images.

Every camera has the same patch grid, so the learned position embeddings and the
2D rotary tables are constants computed here exactly as the reference computes
them per call, and each image attends only to itself (a batch of images).
Patches arrive in the processor's order, where four consecutive patches form one
2x2 merge block; that order carries through to the merged image tokens.
"""

from __future__ import annotations

import numpy as np

from . import graph
from .checkpoint import require

WIDTH = 1024
DEPTH = 24
HEADS = 16
HEAD_DIM = 64
MLP = 4096
PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
PATCH_VALUES = 3 * TEMPORAL_PATCH * PATCH * PATCH
OUT = 2048
DEEPSTACK_LAYERS = (5, 11, 17)
POSITION_GRID = 48
LN_EPS = 1e-6
ROPE_THETA = 10000.0

_ROOT = "backbone.model.model.visual."


def _linspace(stop: float, steps: int) -> np.ndarray:
    """torch.linspace(0, stop, steps) in float32, with torch's symmetric two-sided formula."""
    start, end = np.float32(0.0), np.float32(stop)
    step = (end - start) / np.float32(steps - 1)
    index = np.arange(steps)
    halfway = steps // 2
    values = np.where(index < halfway, start + step * index.astype(np.float32),
                      end - step * (steps - index - 1).astype(np.float32))
    return values.astype(np.float32)


def _block_order(table: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Row-major ``[grid_h * grid_w, C]`` -> merge-block order (2x2 blocks, row-major inside)."""
    channels = table.shape[-1]
    blocks = table.reshape(grid_h // MERGE, MERGE, grid_w // MERGE, MERGE, channels).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(blocks.reshape(grid_h * grid_w, channels))


def position_embeddings(pos_embed: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Bilinearly interpolated learned positions ``[grid_h * grid_w, WIDTH]`` in merge-block order.

    Mirrors ``Qwen3VLVisionModel.fast_pos_embed_interpolate`` step for step in float32.
    """
    side = POSITION_GRID
    h_idx, w_idx = _linspace(side - 1, grid_h), _linspace(side - 1, grid_w)
    h_floor, w_floor = h_idx.astype(np.int32), w_idx.astype(np.int32)
    h_ceil, w_ceil = np.minimum(h_floor + 1, side - 1), np.minimum(w_floor + 1, side - 1)
    dh, dw = (h_idx - h_floor.astype(np.float32)).astype(np.float32), (w_idx - w_floor.astype(np.float32)).astype(np.float32)
    one = np.float32(1.0)
    corners = (
        (h_floor[:, None] * side + w_floor[None, :], (one - dh)[:, None] * (one - dw)[None, :]),
        (h_floor[:, None] * side + w_ceil[None, :], (one - dh)[:, None] * dw[None, :]),
        (h_ceil[:, None] * side + w_floor[None, :], dh[:, None] * (one - dw)[None, :]),
        (h_ceil[:, None] * side + w_ceil[None, :], dh[:, None] * dw[None, :]),
    )
    total = None
    for index, weight in corners:
        term = pos_embed[index.reshape(-1)] * weight.reshape(-1, 1).astype(np.float32)
        total = term if total is None else (total + term).astype(np.float32)
    return _block_order(total.astype(np.float32), grid_h, grid_w)


def rotary_tables(grid_h: int, grid_w: int) -> tuple[np.ndarray, np.ndarray]:
    """``cos``/``sin`` ``[grid_h * grid_w, HEAD_DIM]`` of the 2D (row, column) rotary embedding."""
    dim = HEAD_DIM // 2
    inverse = (np.float32(1.0) / (np.float32(ROPE_THETA) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))))
    table = np.outer(np.arange(max(grid_h, grid_w), dtype=np.float32), inverse.astype(np.float32)).astype(np.float32)
    rows, columns = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    coords = _block_order(np.stack([rows, columns], axis=-1).reshape(-1, 2), grid_h, grid_w)
    angles = np.concatenate([table[coords[:, 0]], table[coords[:, 1]]], axis=1)
    angles = np.concatenate([angles, angles], axis=1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _merger(network, h, weights, prefix, *, postshuffle, dtype):
    """Patch merger: four neighbouring patches -> one ``OUT``-wide token (exact-erf gelu MLP)."""
    if postshuffle:
        x = graph.shuffle(network, h, reshape=(-1, WIDTH * MERGE * MERGE))
        x = graph.layer_norm(network, x, LN_EPS, weights[prefix + "norm.weight"], weights[prefix + "norm.bias"])
    else:
        x = graph.layer_norm(network, h, LN_EPS, weights[prefix + "norm.weight"], weights[prefix + "norm.bias"])
        x = graph.shuffle(network, x, reshape=(-1, WIDTH * MERGE * MERGE))
    x = graph.gelu_erf(network, graph.linear(
        network, x, weights[prefix + "linear_fc1.weight"], weights[prefix + "linear_fc1.bias"], dtype=dtype))
    return graph.linear(network, x, weights[prefix + "linear_fc2.weight"], weights[prefix + "linear_fc2.bias"], dtype=dtype)


def build_vision(network, pixel_values, weights, *, images, grid_h, grid_w, dtype):
    """Image tokens ``[images * tokens, OUT]`` and the DeepStack features, all in ``dtype``.

    ``pixel_values[images * grid_h * grid_w, PATCH_VALUES]`` are the processor's normalised
    patches; ``tokens = grid_h * grid_w / 4`` per image.
    """
    patches = grid_h * grid_w
    kernel = require(weights, _ROOT + "patch_embed.proj.weight", (WIDTH, 3, TEMPORAL_PATCH, PATCH, PATCH))
    bias = require(weights, _ROOT + "patch_embed.proj.bias", (WIDTH,))
    x = graph.shuffle(network, pixel_values, reshape=(images, patches, PATCH_VALUES))
    # The Conv3d whose kernel equals its stride is a linear map of each flattened patch.
    h = graph.f32(network, graph.linear(network, x, kernel.reshape(WIDTH, PATCH_VALUES), bias, dtype=dtype))
    table = require(weights, _ROOT + "pos_embed.weight", (POSITION_GRID * POSITION_GRID, WIDTH))
    h = graph.add(network, h, graph.const(network, position_embeddings(table, grid_h, grid_w)[None]))

    cos, sin = rotary_tables(grid_h, grid_w)
    cos = graph.const(network, cos.reshape(1, patches, 1, HEAD_DIM))
    sin = graph.const(network, sin.reshape(1, patches, 1, HEAD_DIM))

    deepstack = []
    for index in range(DEPTH):
        p = f"{_ROOT}blocks.{index}."
        norm = graph.layer_norm(network, h, LN_EPS, weights[p + "norm1.weight"], weights[p + "norm1.bias"])
        qkv = graph.linear(network, norm, weights[p + "attn.qkv.weight"], weights[p + "attn.qkv.bias"], dtype=dtype)
        parts = [graph.slice_static(network, qkv, (0, 0, i * WIDTH), (images, patches, WIDTH)) for i in range(3)]
        q, k = (
            graph.shuffle(network, graph.rotate(
                network, graph.shuffle(network, t, reshape=(images, patches, HEADS, HEAD_DIM)), cos, sin, HEAD_DIM),
                reshape=(images, patches, WIDTH))
            for t in parts[:2]
        )
        context = graph.batched_attention(network, q, k, parts[2], heads=HEADS, head_dim=HEAD_DIM, dtype=dtype)
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, context, weights[p + "attn.proj.weight"], weights[p + "attn.proj.bias"], dtype=dtype)))
        norm = graph.layer_norm(network, h, LN_EPS, weights[p + "norm2.weight"], weights[p + "norm2.bias"])
        hidden = graph.gelu_tanh(network, graph.linear(
            network, norm, weights[p + "mlp.linear_fc1.weight"], weights[p + "mlp.linear_fc1.bias"], dtype=dtype))
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, hidden, weights[p + "mlp.linear_fc2.weight"], weights[p + "mlp.linear_fc2.bias"], dtype=dtype)))
        if index in DEEPSTACK_LAYERS:
            slot = DEEPSTACK_LAYERS.index(index)
            deepstack.append(_merger(
                network, h, weights, f"{_ROOT}deepstack_merger_list.{slot}.", postshuffle=True, dtype=dtype))

    tokens = _merger(network, h, weights, _ROOT + "merger.", postshuffle=False, dtype=dtype)
    return tokens, deepstack


def image_tokens(grid_h: int, grid_w: int) -> int:
    """Image tokens per camera: one per 2x2 merge block of patches."""
    return grid_h * grid_w // (MERGE * MERGE)

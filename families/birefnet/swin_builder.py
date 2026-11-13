# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Swin-T backbone BiRefNet uses.

This is the original Swin implementation, not timm's, and three details were
read off the reference rather than carried over:

* every stage resolution at a 1024 input (256, 128, 64, 32) is **indivisible by
  the window of 7**, so each block zero-pads its bottom and right edges up to a
  whole number of windows and crops back afterwards.
* the shifted-window mask is built from the **padded** size, with -100 between
  regions that must not attend to each other. It is geometry, not a weight, and
  the checkpoint does not carry it.
* patch merging folds the 2x2 neighbourhood in the original order
  ``[0::2 0::2, 1::2 0::2, 0::2 1::2, 1::2 1::2]``, normalises the 4C
  concatenation, and only then reduces. timm interleaves differently, and the
  wrong order still builds.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-5


def shifted_window_mask(height, width, window, shift):
    """The additive attention mask for one shifted-window stage."""
    regions = np.zeros((height, width), dtype=np.int32)
    counter = 0
    bounds = ((0, height - window), (height - window, height - shift), (height - shift, height))
    spans = ((0, width - window), (width - window, width - shift), (width - shift, width))
    for top, bottom in bounds:
        for left, right in spans:
            regions[top:bottom, left:right] = counter
            counter += 1
    windows_h, windows_w = height // window, width // window
    tiles = regions.reshape(windows_h, window, windows_w, window)
    tiles = tiles.transpose(0, 2, 1, 3).reshape(windows_h * windows_w, window * window)
    mask = tiles[:, None, :] - tiles[:, :, None]
    return np.where(mask != 0, -100.0, 0.0).astype(np.float32)


def relative_position_bias(table, index, heads, area):
    """Gather the learned bias into a per-window [heads, area, area] block."""
    flat = np.asarray(index, dtype=np.int64).reshape(-1)
    gathered = np.asarray(table, dtype=np.float32)[flat]
    return gathered.reshape(area, area, heads).transpose(2, 0, 1)


def _window_attention(network, x, weights, prefix, heads, window, mask, dtype):
    """Attention inside each window, with the relative bias and optional mask."""
    shape = tuple(int(v) for v in x.shape)
    height, width, channels = shape[1], shape[2], shape[3]
    windows_h, windows_w = height // window, width // window
    count = windows_h * windows_w
    area = window * window
    head_dim = channels // heads

    # [1, H, W, C] -> [count, area, C]
    partitioned = g.reshape_permute(
        network, x, (windows_h, window, windows_w, window, channels),
        (0, 2, 1, 3, 4), (count, area, channels))

    qkv = g.add_linear(network, partitioned, weights[f"{prefix}.attn.qkv.weight"],
                       weights[f"{prefix}.attn.qkv.bias"], dtype=dtype)
    split = network.add_shuffle(qkv)
    split.reshape_dims = (count, area, 3, heads, head_dim)
    split.second_transpose = (2, 0, 3, 1, 4)        # [3, count, heads, area, head_dim]
    parts = split.get_output(0)

    def take(index):
        return network.add_slice(parts, (index, 0, 0, 0, 0),
                                 (1, count, heads, area, head_dim),
                                 (1, 1, 1, 1, 1)).get_output(0)

    def drop(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (count, heads, area, head_dim)
        return layer.get_output(0)

    query, key, value = drop(take(0)), drop(take(1)), drop(take(2))
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, g.add_constant(network, (1, 1, 1, 1),
                               np.full((1,), 1.0 / math.sqrt(head_dim)), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)

    bias = relative_position_bias(
        weights[f"{prefix}.attn.relative_position_bias_table"],
        weights[f"{prefix}.attn.relative_position_index"], heads, area)
    scores = network.add_elementwise(
        scores, g.add_constant(network, (1, heads, area, area), bias, dtype=dtype),
        trt.ElementWiseOperation.SUM).get_output(0)

    if mask is not None:
        # One mask per window, shared across heads.
        scores = network.add_elementwise(
            scores, g.add_constant(network, (count, 1, area, area), mask, dtype=dtype),
            trt.ElementWiseOperation.SUM).get_output(0)

    weighted = network.add_softmax(scores)
    weighted.axes = 1 << 3
    context = network.add_matrix_multiply(
        weighted.get_output(0), trt.MatrixOperation.NONE, value,
        trt.MatrixOperation.NONE).get_output(0)
    merged = network.add_shuffle(context)
    merged.first_transpose = (0, 2, 1, 3)
    merged.reshape_dims = (count, area, channels)
    projected = g.add_linear(network, merged.get_output(0),
                             weights[f"{prefix}.attn.proj.weight"],
                             weights[f"{prefix}.attn.proj.bias"], dtype=dtype)
    # [count, area, C] -> [1, H, W, C]
    return g.reshape_permute(
        network, projected, (windows_h, windows_w, window, window, channels),
        (0, 2, 1, 3, 4), (1, height, width, channels))


def _block(network, x, weights, prefix, heads, window, shift, dtype):
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[1], shape[2]
    padded_h = int(math.ceil(height / window)) * window
    padded_w = int(math.ceil(width / window)) * window

    residual = x
    hidden = g.add_layer_norm(network, x, weights[f"{prefix}.norm1.weight"],
                              weights[f"{prefix}.norm1.bias"], _LAYER_NORM_EPS, dtype=dtype)
    hidden = g.pad_hwc(network, hidden, padded_h, padded_w, dtype=dtype)
    mask = None
    if shift:
        hidden = g.roll_hwc(network, hidden, -shift, -shift)
        mask = shifted_window_mask(padded_h, padded_w, window, shift)
    hidden = _window_attention(network, hidden, weights, prefix, heads, window, mask, dtype)
    if shift:
        hidden = g.roll_hwc(network, hidden, shift, shift)
    hidden = g.crop_hwc(network, hidden, height, width)
    x = g.add_sum(network, residual, hidden)

    residual = x
    hidden = g.add_layer_norm(network, x, weights[f"{prefix}.norm2.weight"],
                              weights[f"{prefix}.norm2.bias"], _LAYER_NORM_EPS, dtype=dtype)
    hidden = g.add_gelu(network, g.add_linear(
        network, hidden, weights[f"{prefix}.mlp.fc1.weight"],
        weights[f"{prefix}.mlp.fc1.bias"], dtype=dtype))
    hidden = g.add_linear(network, hidden, weights[f"{prefix}.mlp.fc2.weight"],
                          weights[f"{prefix}.mlp.fc2.bias"], dtype=dtype)
    return g.add_sum(network, residual, hidden)


def _patch_merging(network, x, weights, prefix, dtype):
    """Fold 2x2 into channels in the original Swin order, norm, then reduce."""
    shape = tuple(int(v) for v in x.shape)
    height, width, channels = shape[1], shape[2], shape[3]
    # Odd sides are padded before folding, exactly as the reference does.
    padded_h = height + (height % 2)
    padded_w = width + (width % 2)
    x = g.pad_hwc(network, x, padded_h, padded_w, dtype=dtype)

    def quadrant(row, column):
        return network.add_slice(x, (0, row, column, 0),
                                 (1, padded_h // 2, padded_w // 2, channels),
                                 (1, 2, 2, 1)).get_output(0)

    folded = g.concat(network, [quadrant(0, 0), quadrant(1, 0),
                                quadrant(0, 1), quadrant(1, 1)], axis=3)
    folded = g.add_layer_norm(network, folded, weights[f"{prefix}.norm.weight"],
                              weights[f"{prefix}.norm.bias"], _LAYER_NORM_EPS, dtype=dtype)
    return g.add_linear(network, folded, weights[f"{prefix}.reduction.weight"], None, dtype=dtype)


def build_swin(network, pixels, weights, cfg, prefix="bb", dtype=np.float32, debug=None):
    """Return the four stage feature maps, each NCHW and normalised."""
    patch = cfg["patch_size"]
    window = cfg["window_size"]
    depths = cfg["depths"]
    heads = cfg["num_heads"]

    embedded = g.add_conv2d(network, pixels, weights[f"{prefix}.patch_embed.proj.weight"],
                            weights[f"{prefix}.patch_embed.proj.bias"],
                            stride=(patch, patch), dtype=dtype)
    # NCHW -> NHWC for the token-wise stages.
    to_tokens = network.add_shuffle(embedded)
    to_tokens.first_transpose = (0, 2, 3, 1)
    x = g.add_layer_norm(network, to_tokens.get_output(0),
                         weights[f"{prefix}.patch_embed.norm.weight"],
                         weights[f"{prefix}.patch_embed.norm.bias"],
                         _LAYER_NORM_EPS, dtype=dtype)

    if debug is not None:
        debug["patch_embed"] = embedded
    outputs = []
    for stage, depth in enumerate(depths):
        for block in range(depth):
            # Odd blocks shift by half a window; even blocks do not.
            shift = window // 2 if block % 2 else 0
            x = _block(network, x, weights, f"{prefix}.layers.{stage}.blocks.{block}",
                       heads[stage], window, shift, dtype)
            if debug is not None and stage == 0:
                debug[f"L0B{block}"] = x
        normed = g.add_layer_norm(network, x, weights[f"{prefix}.norm{stage}.weight"],
                                  weights[f"{prefix}.norm{stage}.bias"],
                                  _LAYER_NORM_EPS, dtype=dtype)
        back = network.add_shuffle(normed)
        back.first_transpose = (0, 3, 1, 2)
        outputs.append(back.get_output(0))
        if f"{prefix}.layers.{stage}.downsample.reduction.weight" in weights:
            x = _patch_merging(network, x, weights,
                               f"{prefix}.layers.{stage}.downsample", dtype)
    return outputs


def build_dual_scale(network, pixels, weights, cfg, dtype=np.float32):
    """Run the backbone at full and half resolution and concatenate the levels.

    ``mul_scl_ipt: cat`` in the checkpoint's own config means the backbone is
    evaluated twice. The half-resolution levels are resized back up and placed
    *after* the full-resolution ones on the channel axis, which is where the
    doubled ``lateral_channels_in_collection`` comes from. Both interpolations
    use ``align_corners=True``.
    """
    shape = tuple(int(v) for v in pixels.shape)
    height, width = shape[2], shape[3]

    full = build_swin(network, pixels, weights, cfg, dtype=dtype)
    small = g.add_resize_bilinear(network, pixels, (height // 2, width // 2),
                                  align_corners=True)
    half = build_swin(network, small, weights, cfg, dtype=dtype)

    merged = []
    for level, (a, b) in enumerate(zip(full, half)):
        target = tuple(int(v) for v in a.shape)[2:]
        upsampled = g.add_resize_bilinear(network, b, target, align_corners=True)
        merged.append(g.concat(network, [a, upsampled], axis=1))
    return merged

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swin-T vision backbone for Grounding DINO.

The window algorithm is the same one ``families/birefnet`` builds, but the
checkpoint is a ``transformers`` ``SwinModel``, not the original release, and
the tensor names do not line up:

    attn.qkv (fused)               ->  attention.self.{query,key,value}
    attn.proj                      ->  attention.output.dense
    norm1 / norm2                  ->  layernorm_before / layernorm_after
    mlp.fc1 / mlp.fc2              ->  intermediate.dense / output.dense
    norm{stage}                    ->  hidden_states_norms.stage{stage + 1}

Reading the shapes settles two orderings that are otherwise guesswork:

* ``downsample.norm`` is ``[4C]``, so patch merging normalises the folded 4C
  concatenation and only then reduces it to 2C. A ``[2C]`` norm would mean the
  other order.
* the relative position **index** ships in the checkpoint as a ``[49, 49]``
  tensor, so it is gathered rather than recomputed.

Only stages 2, 3 and 4 leave the backbone - ``out_indices`` is ``[2, 3, 4]`` -
at strides 8, 16 and 32. Stage 1 stays inside.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-5
# The checkpoint stores the backbone under the detection model's own prefix.
SWIN_PREFIX = "model.backbone.conv_encoder.model"


def shifted_window_mask(height, width, window, shift):
    """The additive attention mask for one shifted-window stage.

    Geometry, not a weight: the checkpoint does not carry it. Built from the
    PADDED size, with -100 between regions that must not see each other.
    """
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

    def project(name):
        out = g.add_linear(network, partitioned,
                           weights[f"{prefix}.attention.self.{name}.weight"],
                           weights[f"{prefix}.attention.self.{name}.bias"], dtype=dtype)
        shuffle = network.add_shuffle(out)
        shuffle.reshape_dims = (count, area, heads, head_dim)
        shuffle.second_transpose = (0, 2, 1, 3)
        return shuffle.get_output(0)

    # Separate projections, not one fused qkv: this checkpoint stores three.
    query, key, value = project("query"), project("key"), project("value")

    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, g.add_constant(network, (1, 1, 1, 1),
                               np.full((1,), 1.0 / math.sqrt(head_dim)), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)

    bias = relative_position_bias(
        weights[f"{prefix}.attention.self.relative_position_bias_table"],
        weights[f"{prefix}.attention.self.relative_position_index"], heads, area)
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
                             weights[f"{prefix}.attention.output.dense.weight"],
                             weights[f"{prefix}.attention.output.dense.bias"], dtype=dtype)
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
    hidden = g.add_layer_norm(network, x, weights[f"{prefix}.layernorm_before.weight"],
                              weights[f"{prefix}.layernorm_before.bias"],
                              _LAYER_NORM_EPS, dtype=dtype)
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
    hidden = g.add_layer_norm(network, x, weights[f"{prefix}.layernorm_after.weight"],
                              weights[f"{prefix}.layernorm_after.bias"],
                              _LAYER_NORM_EPS, dtype=dtype)
    hidden = g.add_gelu(network, g.add_linear(
        network, hidden, weights[f"{prefix}.intermediate.dense.weight"],
        weights[f"{prefix}.intermediate.dense.bias"], dtype=dtype))
    hidden = g.add_linear(network, hidden, weights[f"{prefix}.output.dense.weight"],
                          weights[f"{prefix}.output.dense.bias"], dtype=dtype)
    return g.add_sum(network, residual, hidden)


def _patch_merging(network, x, weights, prefix, dtype):
    """Fold 2x2 into channels, normalise the 4C concatenation, then reduce."""
    shape = tuple(int(v) for v in x.shape)
    height, width, channels = shape[1], shape[2], shape[3]
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


def build_swin(network, pixels, weights, cfg, prefix=SWIN_PREFIX, dtype=np.float32, debug=None):
    """Return the stages named by ``out_indices``, each NCHW and normalised."""
    patch = cfg["patch_size"]
    window = cfg["window_size"]
    depths = cfg["depths"]
    heads = cfg["num_heads"]
    wanted = set(cfg["out_indices"])

    embedded = g.add_conv2d(
        network, pixels,
        weights[f"{prefix}.embeddings.patch_embeddings.projection.weight"],
        weights[f"{prefix}.embeddings.patch_embeddings.projection.bias"],
        stride=(patch, patch), dtype=dtype)
    # NCHW -> NHWC for the token-wise stages.
    to_tokens = network.add_shuffle(embedded)
    to_tokens.first_transpose = (0, 2, 3, 1)
    x = g.add_layer_norm(network, to_tokens.get_output(0),
                         weights[f"{prefix}.embeddings.norm.weight"],
                         weights[f"{prefix}.embeddings.norm.bias"],
                         _LAYER_NORM_EPS, dtype=dtype)

    outputs = []
    for stage, depth in enumerate(depths):
        for block in range(depth):
            # Odd blocks shift by half a window; even blocks do not.
            shift = window // 2 if block % 2 else 0
            x = _block(network, x, weights,
                       f"{prefix}.encoder.layers.{stage}.blocks.{block}",
                       heads[stage], window, shift, dtype)
            if debug is not None and stage == 0:
                debug[f"L0B{block}"] = x
        # out_indices is one-based, so stage 0 is "stage1" and is not emitted.
        name = f"stage{stage + 1}"
        if stage + 1 in wanted:
            normed = g.add_layer_norm(
                network, x,
                weights[f"{prefix}.hidden_states_norms.{name}.weight"],
                weights[f"{prefix}.hidden_states_norms.{name}.bias"],
                _LAYER_NORM_EPS, dtype=dtype)
            back = network.add_shuffle(normed)
            back.first_transpose = (0, 3, 1, 2)
            outputs.append(back.get_output(0))
            if debug is not None:
                debug[name] = back.get_output(0)
        if f"{prefix}.encoder.layers.{stage}.downsample.reduction.weight" in weights:
            x = _patch_merging(network, x, weights,
                               f"{prefix}.encoder.layers.{stage}.downsample", dtype)
    return outputs

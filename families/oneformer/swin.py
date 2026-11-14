# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Swin backbone of ``OneFormerForUniversalSegmentation`` (Transformers ``SwinBackbone``).

The backbone runs with ``always_partition=True``: every block pads its input to
a multiple of the window, odd blocks roll by half a window and mask attention
between the rolled regions, and the window size never shrinks to the feature
map. All four stages are emitted before patch merging, each through its own
LayerNorm, as NCHW feature maps. Sizes are runtime values:

* patch embedding pads the image bottom/right to a multiple of the patch size;
* a block pads bottom/right with zeros to a window multiple, rolls with a
  wrap-around slice, and crops back after the reverse roll;
* patch merging pads to even sizes and interleaves the four 2x2 phases in the
  order (0,0), (1,0), (0,1), (1,1) before its LayerNorm and reduction.

The shifted-window mask is rebuilt per stage from the padded size: region ids
0/1/2 along each axis split at ``size - window`` and ``size - shift``, and
pairs of tokens in different regions get -100, as in ``SwinLayer.get_attn_mask``.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

E = trt.ElementWiseOperation
PREFIX = "model.pixel_level_module.encoder"
# The embedding, patch-merging and stage-output LayerNorms are built without an
# explicit eps (torch's default); only the per-block norms read layer_norm_eps.
TORCH_LAYER_NORM_EPS = 1e-5


def relative_position_bias(
    table: np.ndarray, index: np.ndarray, heads: int, area: int
) -> np.ndarray:
    """``table[index]`` laid out as the (1, heads, area, area) additive score bias."""
    gathered = np.asarray(table, dtype=np.float32)[np.asarray(index, dtype=np.int64).reshape(-1)]
    return gathered.reshape(area, area, heads).transpose(2, 0, 1)[None].copy()


def _region_ids(g, size, window: int, shift: int):
    """0, 1 or 2 per position along one padded axis: [0, size-window), [.., size-shift), [.., size)."""
    positions = g.grid([size], 0.0, [1.0])
    size_f = g.as_float(size)
    ids = None
    for offset in (window, shift):
        threshold = g.ew(size_f, g.f32([float(offset)]), E.SUB)
        below = g.ew(positions, threshold, E.LESS)
        at_or_above = g.cast(g.unary(below, trt.UnaryOperation.NOT), trt.float32)
        ids = at_or_above if ids is None else g.add(ids, at_or_above)
    return ids


def shifted_window_mask(g, padded_h, padded_w, window: int, shift: int):
    """(windows, 1, area, area) additive mask: -100 between tokens of different regions."""
    rows = g.reshape(_region_ids(g, padded_h, window, shift), (-1, 1))
    cols = g.reshape(_region_ids(g, padded_w, window, shift), (1, -1))
    ids = g.add(g.ew(rows, g.f32([3.0], (1, 1)), E.PROD), cols)
    windows_h = g.iop(padded_h, window, E.FLOOR_DIV)
    windows_w = g.iop(padded_w, window, E.FLOOR_DIV)
    area = window * window
    tiles = g.reshape(ids, [windows_h, window, windows_w, window], second=(0, 2, 1, 3))
    tiles = g.reshape(tiles, (-1, area))
    differs = g.unary(
        g.ew(g.reshape(tiles, (0, 1, area)), g.reshape(tiles, (0, area, 1)), E.EQUAL),
        trt.UnaryOperation.NOT,
    )
    mask = g.ew(g.cast(differs, trt.float32), g.f32([-100.0], (1, 1, 1)), E.PROD)
    return g.reshape(mask, (-1, 1, area, area))


def _window_attention(g, windows, weights, prefix, heads: int, window: int, mask):
    """Attention inside each window: (windows, area, C) -> (windows, area, C)."""
    area = window * window
    channels = int(windows.shape[2])
    head_dim = channels // heads

    def project(name):
        out = g.linear(
            windows,
            weights.get(f"{prefix}.attention.self.{name}.weight"),
            weights.get(f"{prefix}.attention.self.{name}.bias"),
        )
        return g.reshape(out, (-1, area, heads, head_dim), second=(0, 2, 1, 3))

    query, key, value = project("query"), project("key"), project("value")
    scores = g.matmul(query, key, transpose_right=True)
    scores = g.ew(scores, g.scalar(scores, math.sqrt(head_dim)), E.DIV)
    bias = relative_position_bias(
        weights.get(f"{prefix}.attention.self.relative_position_bias_table"),
        weights.get(f"{prefix}.attention.self.relative_position_index", dtype=None),
        heads,
        area,
    )
    scores = g.add(scores, g.constant(bias))
    if mask is not None:
        scores = g.add(scores, mask)
    context = g.matmul(g.softmax(scores, 3), value)
    context = g.reshape(context, (-1, area, channels), first=(0, 2, 1, 3))
    return g.linear(
        context,
        weights.get(f"{prefix}.attention.output.dense.weight"),
        weights.get(f"{prefix}.attention.output.dense.bias"),
    )


def _block(g, x, weights, prefix, heads, window, shift, eps, geometry):
    """One SwinLayer on NHWC (1, h, w, C)."""
    height, width, padded_h, padded_w, mask = geometry
    channels = int(x.shape[3])
    hidden = g.layer_norm(
        x,
        weights.get(f"{prefix}.layernorm_before.weight"),
        weights.get(f"{prefix}.layernorm_before.bias"),
        eps,
    )
    hidden = g.slice(
        hidden, [0, 0, 0, 0], [1, padded_h, padded_w, channels], mode=trt.SampleMode.FILL
    )
    if shift:
        hidden = g.slice(
            hidden,
            [0, shift, shift, 0],
            [1, padded_h, padded_w, channels],
            mode=trt.SampleMode.WRAP,
        )
    windows_h = g.iop(padded_h, window, E.FLOOR_DIV)
    windows_w = g.iop(padded_w, window, E.FLOOR_DIV)
    windows = g.reshape(
        hidden, [1, windows_h, window, windows_w, window, channels], second=(0, 1, 3, 2, 4, 5)
    )
    windows = g.reshape(windows, (-1, window * window, channels))
    attended = _window_attention(
        g, windows, weights, prefix, heads, window, mask if shift else None
    )
    merged = g.reshape(
        attended, [1, windows_h, windows_w, window, window, channels], second=(0, 1, 3, 2, 4, 5)
    )
    merged = g.reshape(merged, [1, padded_h, padded_w, channels])
    if shift:
        # Reverse roll and crop in one wrap-around slice.
        start_h = g.iop(padded_h, shift, E.SUB)
        start_w = g.iop(padded_w, shift, E.SUB)
        merged = g.slice(
            merged, [0, start_h, start_w, 0], [1, height, width, channels], mode=trt.SampleMode.WRAP
        )
    else:
        merged = g.slice(merged, [0, 0, 0, 0], [1, height, width, channels])
    x = g.add(x, merged)
    hidden = g.layer_norm(
        x,
        weights.get(f"{prefix}.layernorm_after.weight"),
        weights.get(f"{prefix}.layernorm_after.bias"),
        eps,
    )
    hidden = g.linear(
        hidden,
        weights.get(f"{prefix}.intermediate.dense.weight"),
        weights.get(f"{prefix}.intermediate.dense.bias"),
    )
    hidden = g.activation(hidden, trt.ActivationType.GELU_ERF)
    hidden = g.linear(
        hidden,
        weights.get(f"{prefix}.output.dense.weight"),
        weights.get(f"{prefix}.output.dense.bias"),
    )
    return g.add(x, hidden)


def _patch_merging(g, x, weights, prefix):
    """Pad to even, fold each 2x2 neighbourhood into channels, LayerNorm(4C), reduce to 2C."""
    channels = int(x.shape[3])
    padded_h = g.ceil_to(g.dim(x, 1), 2)
    padded_w = g.ceil_to(g.dim(x, 2), 2)
    x = g.slice(x, [0, 0, 0, 0], [1, padded_h, padded_w, channels], mode=trt.SampleMode.FILL)
    half_h = g.iop(padded_h, 2, E.FLOOR_DIV)
    half_w = g.iop(padded_w, 2, E.FLOOR_DIV)
    phases = [
        g.slice(x, [0, row, column, 0], [1, half_h, half_w, channels], [1, 2, 2, 1])
        for row, column in ((0, 0), (1, 0), (0, 1), (1, 1))
    ]
    folded = g.layer_norm(
        g.concat(phases, 3),
        weights.get(f"{prefix}.norm.weight"),
        weights.get(f"{prefix}.norm.bias"),
        TORCH_LAYER_NORM_EPS,
    )
    return g.linear(folded, weights.get(f"{prefix}.reduction.weight"))


def build_swin(g, pixels, weights, swin):
    """Return the four stage feature maps (NCHW, after the per-stage LayerNorms)."""
    eps = swin.layer_norm_eps
    patch, window = swin.patch_size, swin.window_size
    shift = window // 2
    padded = g.slice(
        pixels,
        [0, 0, 0, 0],
        [1, 3, g.ceil_to(g.dim(pixels, 2), patch), g.ceil_to(g.dim(pixels, 3), patch)],
        mode=trt.SampleMode.FILL,
    )
    embedded = g.conv(
        padded,
        weights.get(f"{PREFIX}.embeddings.patch_embeddings.projection.weight"),
        weights.get(f"{PREFIX}.embeddings.patch_embeddings.projection.bias"),
        stride=patch,
    )
    x = g.layer_norm(
        g.transpose(embedded, (0, 2, 3, 1)),
        weights.get(f"{PREFIX}.embeddings.norm.weight"),
        weights.get(f"{PREFIX}.embeddings.norm.bias"),
        TORCH_LAYER_NORM_EPS,
    )
    features = []
    for stage, depth in enumerate(swin.depths):
        height, width = g.dim(x, 1), g.dim(x, 2)
        padded_h, padded_w = g.ceil_to(height, window), g.ceil_to(width, window)
        mask = shifted_window_mask(g, padded_h, padded_w, window, shift) if depth > 1 else None
        geometry = (height, width, padded_h, padded_w, mask)
        for block in range(depth):
            x = _block(
                g,
                x,
                weights,
                f"{PREFIX}.encoder.layers.{stage}.blocks.{block}",
                swin.num_heads[stage],
                window,
                shift if block % 2 else 0,
                eps,
                geometry,
            )
        normed = g.layer_norm(
            x,
            weights.get(f"{PREFIX}.hidden_states_norms.stage{stage + 1}.weight"),
            weights.get(f"{PREFIX}.hidden_states_norms.stage{stage + 1}.bias"),
            TORCH_LAYER_NORM_EPS,
        )
        features.append(g.transpose(normed, (0, 3, 1, 2)))
        if stage + 1 < len(swin.depths):
            x = _patch_merging(g, x, weights, f"{PREFIX}.encoder.layers.{stage}.downsample")
    return features

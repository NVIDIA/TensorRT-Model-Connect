# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Nemotron 3 Diarization encoder step.

Every helper takes and returns rows ``[frames, width]`` whose frame count is
dynamic: one encoder step holds the speaker cache, the FIFO queue and one chunk
with its look-ahead, and that total changes from step to step. The arithmetic
stays in float32 throughout; the checkpoint is float32 and the speaker cache
selects frames by comparing scores, which rewards exact parity over speed.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return network.add_constant(data.shape, data).get_output(0)


def add_elementwise(network, left, right, operation):
    return network.add_elementwise(left, right, operation).get_output(0)


def add_sum(network, left, right):
    return add_elementwise(network, left, right, trt.ElementWiseOperation.SUM)


def add_product(network, left, right):
    return add_elementwise(network, left, right, trt.ElementWiseOperation.PROD)


def reshape(network, tensor, dims, transpose=None):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = tuple(dims)
    if transpose is not None:
        layer.second_transpose = tuple(transpose)
    return layer.get_output(0)


def transpose(network, tensor, order):
    layer = network.add_shuffle(tensor)
    layer.first_transpose = tuple(order)
    return layer.get_output(0)


def add_linear(network, rows, weight, bias=None):
    """``rows @ weight.T + bias``, torch.nn.Linear's layout, on ``[frames, in]``."""
    out = network.add_matrix_multiply(
        rows, trt.MatrixOperation.NONE, add_constant(network, weight),
        trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    return add_sum(network, out, add_constant(network, np.asarray(bias).reshape(1, -1)))


def add_layer_norm(network, rows, gamma, beta, eps):
    """Biased-variance layer norm over the last axis, as torch.nn.LayerNorm."""
    mean = network.add_reduce(rows, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    centred = add_elementwise(network, rows, mean, trt.ElementWiseOperation.SUB)
    square = add_product(network, centred, centred)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    shifted = add_sum(network, variance, add_constant(network, np.full((1, 1), eps)))
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = add_elementwise(network, centred, deviation, trt.ElementWiseOperation.DIV)
    scaled = add_product(network, normed, add_constant(network, np.asarray(gamma).reshape(1, -1)))
    return add_sum(network, scaled, add_constant(network, np.asarray(beta).reshape(1, -1)))


def add_activation(network, tensor, kind):
    return network.add_activation(tensor, kind).get_output(0)


def leading_rows(network, table, rows_like):
    """The first ``frames`` rows of a constant table, ``frames`` taken from ``rows_like``."""
    width = int(table.shape[1])
    shape = network.add_shape(rows_like).get_output(0)
    frames = network.add_gather(
        shape, add_constant(network, np.array([0], dtype=np.int32), np.int32), 0).get_output(0)
    size = network.add_concatenation(
        [frames, add_constant(network, np.array([width], dtype=np.int64), np.int64)])
    size.axis = 0
    layer = network.add_slice(table, (0, 0), (1, width), (1, 1))
    layer.set_input(2, size.get_output(0))
    return layer.get_output(0)


def rotary_tables(network, rows_like, cos_table, sin_table):
    """Per-position RoPE factors ``[1, frames, head_dim]`` for this step's frames."""
    cos = leading_rows(network, add_constant(network, cos_table), rows_like)
    sin = leading_rows(network, add_constant(network, sin_table), rows_like)
    head_dim = int(cos_table.shape[1])
    return reshape(network, cos, (1, -1, head_dim)), reshape(network, sin, (1, -1, head_dim))


def apply_rotary(network, heads, cos, sin, head_dim):
    """``x * cos + rotate_half(x) * sin`` with ``rotate_half(x) = [-x2, x1]``.

    rotate_half is a gather plus a sign flip, so it adds no rounding of its own.
    """
    half = head_dim // 2
    order = np.concatenate([np.arange(half, head_dim), np.arange(0, half)]).astype(np.int32)
    rotated = network.add_gather(heads, add_constant(network, order, np.int32), 2).get_output(0)
    signs = np.concatenate([-np.ones(half), np.ones(half)]).reshape(1, 1, head_dim)
    rotated = add_product(network, rotated, add_constant(network, signs))
    return add_sum(network, add_product(network, heads, cos), add_product(network, rotated, sin))


def add_self_attention(network, rows, weights, prefix, cfg, rope, key_bias):
    """Bidirectional multi-head attention over the step's frames.

    ``key_bias`` ``[1, 1, frames]`` holds zero for a valid frame and a large
    negative number for a padding frame, which the runtime supplies: padding is
    excluded as a key, but is still a query, because the head's convolution
    reads its hidden state.
    """
    heads, head_dim = cfg["num_heads"], cfg["head_dim"]

    def split(name):
        projected = add_linear(network, rows, weights[f"{prefix}.{name}.weight"])
        return reshape(network, projected, (-1, heads, head_dim), transpose=(1, 0, 2))

    cos, sin = rope
    query = apply_rotary(network, split("q_proj"), cos, sin, head_dim)
    key = apply_rotary(network, split("k_proj"), cos, sin, head_dim)
    value = split("v_proj")
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = add_product(network, scores, add_constant(network, np.full((1, 1, 1),
                                                                       head_dim**-0.5)))
    scores = add_sum(network, scores, key_bias)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value,
        trt.MatrixOperation.NONE).get_output(0)
    merged = reshape(network, transpose(network, context, (1, 0, 2)), (-1, heads * head_dim))
    return add_linear(network, merged, weights[f"{prefix}.o_proj.weight"],
                      weights[f"{prefix}.o_proj.bias"])


def add_subpixel_upsampler(network, rows, weight, bias, factor):
    """``Conv1d(width, width * factor, 3, padding=1)`` over frames, then one row per sub-frame.

    Output channel block ``s`` of frame ``t`` becomes row ``t * factor + s``, the
    reference's reshape of ``[frames, width * factor]`` into ``[frames * factor, width]``.
    """
    width = int(weight.shape[1])
    image = reshape(network, transpose(network, rows, (1, 0)), (1, width, 1, -1))
    kernel = np.ascontiguousarray(np.asarray(weight, dtype=np.float32)[:, :, None, :])
    conv = network.add_convolution_nd(image, int(weight.shape[0]), (1, 3), trt.Weights(kernel),
                                      trt.Weights(np.ascontiguousarray(bias, dtype=np.float32)))
    conv.padding_nd = (0, 1)
    channels = reshape(network, conv.get_output(0), (int(weight.shape[0]), -1))
    return reshape(network, transpose(network, channels, (1, 0)), (-1, width))

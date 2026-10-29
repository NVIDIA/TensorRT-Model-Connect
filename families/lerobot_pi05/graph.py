# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT graph helpers for the Pi 0.5 graphs.

Every tensor is a row tensor ``[seq, width]`` unless a helper says otherwise
(the vision tower carries a leading batch of cameras). Statistics that the
reference takes in float32 - RMS/LayerNorm moments, softmax, rotary angles -
are taken in float32 here too, whatever the working dtype is.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt


OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # LeRobot/openpi additive mask for blocked keys.


def const(network, values, dtype=trt.float32):
    """A constant tensor. Stored as float32 and cast, so the cast folds at build time."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    tensor = network.add_constant(array.shape, array).get_output(0)
    return tensor if dtype == trt.float32 else network.add_cast(tensor, dtype).get_output(0)


def const_int32(network, values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int32))
    return network.add_constant(array.shape, array).get_output(0)


def cast(network, tensor, dtype):
    return tensor if tensor.dtype == dtype else network.add_cast(tensor, dtype).get_output(0)


def _binary(network, left, right, operation):
    return network.add_elementwise(left, right, operation).get_output(0)


def add(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.SUM)


def sub(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.SUB)


def mul(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.PROD)


def div(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.DIV)


def scalar(network, value, rank, dtype=trt.float32):
    """A broadcastable scalar constant with ``rank`` dimensions."""
    return const(network, np.full((1,) * rank, value, dtype=np.float32), dtype)


def shuffle(network, tensor, *, first=None, reshape=None, second=None):
    layer = network.add_shuffle(tensor)
    if first is not None:
        layer.first_transpose = tuple(first)
    if reshape is not None:
        layer.reshape_dims = tuple(reshape)
    if second is not None:
        layer.second_transpose = tuple(second)
    return layer.get_output(0)


def slice_static(network, tensor, start, shape):
    return network.add_slice(tensor, tuple(start), tuple(shape), (1,) * len(shape)).get_output(0)


def concat(network, tensors, axis):
    layer = network.add_concatenation(list(tensors))
    layer.axis = axis
    return layer.get_output(0)


def linear(network, x, weight, bias=None):
    """``y = x @ W^T + b`` with torch.nn.Linear's ``[out, in]`` weight layout."""
    rank = len(x.shape)
    weight = np.asarray(weight, dtype=np.float32)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    kernel = const(network, weight.reshape(shape), x.dtype)
    out = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, kernel, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    if bias is None:
        return out
    bias = np.asarray(bias, dtype=np.float32).reshape((1,) * (rank - 1) + (-1,))
    return add(network, out, const(network, bias, x.dtype))


def _moments_input(network, x):
    return cast(network, x, trt.float32)


def rms_normalise(network, x, eps):
    """``x / sqrt(mean(x^2) + eps)`` in float32, over the last axis."""
    source = _moments_input(network, x)
    rank = len(x.shape)
    square = mul(network, source, source)
    mean = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
    shifted = add(network, mean, scalar(network, eps, rank))
    root = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    return div(network, source, root)


def gemma_rms_norm(network, x, weight, eps):
    """Gemma RMSNorm, ``normed * (1 + weight)``, returned in ``x``'s dtype."""
    rank = len(x.shape)
    scale = const(network, 1.0 + np.asarray(weight, dtype=np.float32).reshape((1,) * (rank - 1) + (-1,)))
    return cast(network, mul(network, rms_normalise(network, x, eps), scale), x.dtype)


def ada_rms_norm(network, x, scale, shift, eps):
    """Adaptive RMSNorm: ``normed * (1 + scale) + shift``; ``scale``/``shift`` are ``[1, width]``."""
    one = scalar(network, 1.0, len(x.shape))
    modulated = mul(network, rms_normalise(network, x, eps), add(network, cast(network, scale, trt.float32), one))
    return cast(network, add(network, modulated, cast(network, shift, trt.float32)), x.dtype)


def layer_norm(network, x, gamma, beta, eps):
    source = _moments_input(network, x)
    rank = len(x.shape)
    axes = 1 << (rank - 1)
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, axes, True).get_output(0)
    centred = sub(network, source, mean)
    variance = network.add_reduce(mul(network, centred, centred), trt.ReduceOperation.AVG, axes, True).get_output(0)
    root = network.add_unary(add(network, variance, scalar(network, eps, rank)), trt.UnaryOperation.SQRT).get_output(0)
    shape = (1,) * (rank - 1) + (-1,)
    scaled = mul(network, div(network, centred, root), const(network, np.asarray(gamma).reshape(shape)))
    return cast(network, add(network, scaled, const(network, np.asarray(beta).reshape(shape))), x.dtype)


def gelu_tanh(network, x):
    """gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).

    Evaluated in float32 and rounded once, as torch's fused kernel does; spelling
    it out in bfloat16 would round after every elementwise step.
    """
    source = cast(network, x, trt.float32)
    rank = len(x.shape)
    cube = mul(network, mul(network, source, source), source)
    inner = add(network, source, mul(network, cube, scalar(network, 0.044715, rank)))
    inner = mul(network, inner, scalar(network, math.sqrt(2.0 / math.pi), rank))
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    gate = add(network, tanh, scalar(network, 1.0, rank))
    return cast(network, mul(network, mul(network, source, gate), scalar(network, 0.5, rank)), x.dtype)


def silu(network, x):
    gate = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return mul(network, x, gate)


def rope_tables(network, positions, head_dim, theta):
    """Rotary ``cos``/``sin`` tables ``[seq, head_dim]`` in float32 from int32 ``positions[seq]``."""
    seq = int(positions.shape[0])
    inverse = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = network.add_matrix_multiply(
        shuffle(network, cast(network, positions, trt.float32), reshape=(seq, 1)),
        trt.MatrixOperation.NONE,
        const(network, inverse.reshape(1, -1)),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    full = concat(network, [angles, angles], 1)
    cos = network.add_unary(full, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(full, trt.UnaryOperation.SIN).get_output(0)
    return cos, sin


def apply_rope(network, x, cos, sin, heads, head_dim):
    """Rotate ``x[seq, heads * head_dim]`` with the half-split convention (HF ``rotate_half``)."""
    seq = int(x.shape[0])
    half = head_dim // 2
    rows = shuffle(network, cast(network, x, trt.float32), reshape=(seq, heads, head_dim))
    low = slice_static(network, rows, (0, 0, 0), (seq, heads, half))
    high = slice_static(network, rows, (0, 0, half), (seq, heads, half))
    negated = network.add_unary(high, trt.UnaryOperation.NEG).get_output(0)
    rotated = concat(network, [negated, low], 2)
    cos3 = shuffle(network, cos, reshape=(seq, 1, head_dim))
    sin3 = shuffle(network, sin, reshape=(seq, 1, head_dim))
    out = add(network, mul(network, rows, cos3), mul(network, rotated, sin3))
    return cast(network, shuffle(network, out, reshape=(seq, heads * head_dim)), x.dtype)


def _softmax_fp32(network, scores, axis):
    layer = network.add_softmax(cast(network, scores, trt.float32))
    layer.axes = 1 << axis
    return layer.get_output(0)


def attention_mqa(network, query, key, value, bias, *, heads, head_dim):
    """Multi-query attention: ``heads`` query heads share one key/value head.

    ``query[S, heads*D]``, ``key/value[T, D]``, ``bias[S, T]`` additive (or None).
    Folding the heads into the rows turns it into two plain matmuls.
    """
    seq = int(query.shape[0])
    keys = int(key.shape[0])
    rows = shuffle(network, query, reshape=(seq, heads, head_dim), second=(1, 0, 2))
    rows = shuffle(network, rows, reshape=(heads * seq, head_dim))
    scores = network.add_matrix_multiply(
        rows, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = mul(network, scores, scalar(network, head_dim ** -0.5, 2, scores.dtype))
    scores = shuffle(network, scores, reshape=(heads, seq, keys))
    if bias is not None:
        scores = add(network, scores, cast(network, shuffle(network, bias, reshape=(1, seq, keys)), scores.dtype))
    probs = cast(network, _softmax_fp32(network, scores, 2), value.dtype)
    probs = shuffle(network, probs, reshape=(heads * seq, keys))
    context = network.add_matrix_multiply(
        probs, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)
    context = shuffle(network, context, reshape=(heads, seq, head_dim), second=(1, 0, 2))
    return shuffle(network, context, reshape=(seq, heads * head_dim))


def attention_mha(network, query, key, value, *, heads, head_dim):
    """Bidirectional multi-head attention over ``[batch, seq, heads*D]`` without a mask."""
    batch, seq = int(query.shape[0]), int(query.shape[1])

    def to_heads(tensor):
        return shuffle(network, tensor, reshape=(batch, seq, heads, head_dim), second=(0, 2, 1, 3))

    scores = network.add_matrix_multiply(
        to_heads(query), trt.MatrixOperation.NONE, to_heads(key), trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = mul(network, scores, scalar(network, head_dim ** -0.5, 4, scores.dtype))
    probs = cast(network, _softmax_fp32(network, scores, 3), value.dtype)
    context = network.add_matrix_multiply(
        probs, trt.MatrixOperation.NONE, to_heads(value), trt.MatrixOperation.NONE
    ).get_output(0)
    return shuffle(network, context, first=(0, 2, 1, 3), reshape=(batch, seq, heads * head_dim))

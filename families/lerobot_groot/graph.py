# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed TensorRT graph helpers for the GR00T N1.7 graphs.

Token tensors are rows ``[seq, width]`` whose sequence length may be dynamic, so
reshapes copy (``0``) or infer (``-1``) that axis instead of reading it.

Precision follows the reference's bfloat16 autocast: matrix products run in the
``compute`` dtype, while the residual stream, norm statistics, softmax and rotary
angles stay in float32. A float32 build simply has ``compute`` float32 too.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt


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


def f32(network, tensor):
    return cast(network, tensor, trt.float32)


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
    """Transpose/reshape; in ``reshape`` a 0 copies the input axis at that index and -1 is inferred."""
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


def _kernel(network, weight, rank, dtype):
    weight = np.asarray(weight, dtype=np.float32)
    return const(network, weight.reshape((1,) * (rank - 2) + tuple(weight.shape)), dtype)


def _bias(network, out, bias, rank, dtype):
    if bias is None:
        return out
    return add(network, out, const(network, np.asarray(bias, dtype=np.float32).reshape((1,) * (rank - 1) + (-1,)), dtype))


def linear(network, x, weight, bias=None, *, dtype):
    """``x @ W^T + b`` in ``dtype`` for torch.nn.Linear's ``[out, in]`` weight; returns ``dtype``."""
    x = cast(network, x, dtype)
    rank = len(x.shape)
    out = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, _kernel(network, weight, rank, dtype), trt.MatrixOperation.TRANSPOSE).get_output(0)
    return _bias(network, out, bias, rank, dtype)


def linear_in_out(network, x, weight, bias=None, *, dtype):
    """``x @ W + b`` for the ``[in, out]`` weights of GR00T's category-specific layers."""
    x = cast(network, x, dtype)
    rank = len(x.shape)
    out = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, _kernel(network, weight, rank, dtype), trt.MatrixOperation.NONE).get_output(0)
    return _bias(network, out, bias, rank, dtype)


def _reduce_mean(network, x):
    return network.add_reduce(x, trt.ReduceOperation.AVG, 1 << (len(x.shape) - 1), True).get_output(0)


def layer_norm(network, x, eps, gamma=None, beta=None):
    """LayerNorm over the last axis in float32; returns float32."""
    source = f32(network, x)
    rank = len(x.shape)
    centred = sub(network, source, _reduce_mean(network, source))
    variance = _reduce_mean(network, mul(network, centred, centred))
    root = network.add_unary(add(network, variance, scalar(network, eps, rank)), trt.UnaryOperation.SQRT).get_output(0)
    out = div(network, centred, root)
    shape = (1,) * (rank - 1) + (-1,)
    if gamma is not None:
        out = mul(network, out, const(network, np.asarray(gamma).reshape(shape)))
    if beta is not None:
        out = add(network, out, const(network, np.asarray(beta).reshape(shape)))
    return out


def rms_norm(network, x, weight, eps, *, dtype):
    """Qwen RMSNorm: normalised in float32, rounded to ``dtype``, then scaled by the float32 weight."""
    source = f32(network, x)
    rank = len(x.shape)
    mean = _reduce_mean(network, mul(network, source, source))
    root = network.add_unary(add(network, mean, scalar(network, eps, rank)), trt.UnaryOperation.SQRT).get_output(0)
    normed = f32(network, cast(network, div(network, source, root), dtype))
    return mul(network, normed, const(network, np.asarray(weight).reshape((1,) * (rank - 1) + (-1,))))


def gelu_tanh(network, x):
    """gelu with the tanh approximation, evaluated in float32 and rounded back to ``x``'s dtype."""
    source = f32(network, x)
    rank = len(x.shape)
    cube = mul(network, mul(network, source, source), source)
    inner = mul(network, add(network, source, mul(network, cube, scalar(network, 0.044715, rank))),
                scalar(network, math.sqrt(2.0 / math.pi), rank))
    gate = add(network, network.add_activation(inner, trt.ActivationType.TANH).get_output(0), scalar(network, 1.0, rank))
    return cast(network, mul(network, mul(network, source, gate), scalar(network, 0.5, rank)), x.dtype)


def gelu_erf(network, x):
    """Exact gelu, ``0.5 x (1 + erf(x / sqrt 2))``, in float32 and rounded back to ``x``'s dtype."""
    source = f32(network, x)
    rank = len(x.shape)
    erf = network.add_unary(mul(network, source, scalar(network, 1.0 / math.sqrt(2.0), rank)),
                            trt.UnaryOperation.ERF).get_output(0)
    gate = add(network, erf, scalar(network, 1.0, rank))
    return cast(network, mul(network, mul(network, source, gate), scalar(network, 0.5, rank)), x.dtype)


def silu(network, x):
    return mul(network, x, network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))


def relu(network, x):
    return network.add_activation(x, trt.ActivationType.RELU).get_output(0)


def rotate_half_matrix(head_dim):
    """``R`` with ``x @ R == rotate_half(x) == cat(-x2, x1)``; exact, as each column has one +-1."""
    half = head_dim // 2
    matrix = np.zeros((head_dim, head_dim), dtype=np.float32)
    for column in range(half):
        matrix[column + half, column] = -1.0
        matrix[column, column + half] = 1.0
    return matrix


def rotate(network, x, cos, sin, head_dim):
    """``x * cos + rotate_half(x) * sin`` in float32 for ``x[..., head_dim]``; returns float32.

    ``cos``/``sin`` broadcast against ``x``. The half swap is a signed permutation matrix so
    the sequence axis may stay dynamic.
    """
    source = f32(network, x)
    rank = len(x.shape)
    rotated = network.add_matrix_multiply(
        source, trt.MatrixOperation.NONE, _kernel(network, rotate_half_matrix(head_dim), rank, trt.float32),
        trt.MatrixOperation.NONE).get_output(0)
    return add(network, mul(network, source, cos), mul(network, rotated, sin))


def softmax_fp32(network, scores, axis):
    layer = network.add_softmax(f32(network, scores))
    layer.axes = 1 << axis
    return layer.get_output(0)


def attention(network, query, key, value, *, heads, kv_heads, head_dim, dtype, bias=None):
    """Scaled dot-product attention over row tensors, in ``dtype`` with a float32 softmax.

    ``query[S, heads*D]``, ``key``/``value[T, kv_heads*D]`` (grouped query heads share a key
    head, as torch's repeat_kv does), ``bias`` additive, rank 4, broadcastable to
    ``[kv_heads, heads // kv_heads, S, T]``. Either length may be dynamic. Returns ``[S, heads*D]``.
    """
    group = heads // kv_heads
    q = shuffle(network, cast(network, query, dtype), reshape=(0, heads, head_dim), second=(1, 0, 2))
    q = shuffle(network, q, reshape=(kv_heads, group, -1, head_dim))
    k = shuffle(network, cast(network, key, dtype), reshape=(0, kv_heads, head_dim), second=(1, 0, 2))
    k = shuffle(network, k, reshape=(kv_heads, 1, -1, head_dim))
    v = shuffle(network, cast(network, value, dtype), reshape=(0, kv_heads, head_dim), second=(1, 0, 2))
    v = shuffle(network, v, reshape=(kv_heads, 1, -1, head_dim))
    scores = network.add_matrix_multiply(q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = mul(network, f32(network, scores), scalar(network, head_dim ** -0.5, 4))
    if bias is not None:
        scores = add(network, scores, f32(network, bias))
    probs = cast(network, softmax_fp32(network, scores, 3), dtype)
    context = network.add_matrix_multiply(probs, trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE).get_output(0)
    context = shuffle(network, context, reshape=(heads, -1, head_dim), second=(1, 0, 2))
    return shuffle(network, context, reshape=(0, heads * head_dim))


def batched_attention(network, query, key, value, *, heads, head_dim, dtype):
    """Bidirectional multi-head attention over static ``[batch, seq, heads*D]`` tensors."""
    batch, seq = int(query.shape[0]), int(query.shape[1])

    def to_heads(tensor):
        return shuffle(network, cast(network, tensor, dtype), reshape=(batch, seq, heads, head_dim), second=(0, 2, 1, 3))

    scores = network.add_matrix_multiply(
        to_heads(query), trt.MatrixOperation.NONE, to_heads(key), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = mul(network, f32(network, scores), scalar(network, head_dim ** -0.5, 4))
    probs = cast(network, softmax_fp32(network, scores, 3), dtype)
    context = network.add_matrix_multiply(probs, trt.MatrixOperation.NONE, to_heads(value), trt.MatrixOperation.NONE).get_output(0)
    return shuffle(network, context, first=(0, 2, 1, 3), reshape=(batch, seq, heads * head_dim))


def mark_output(network, tensor, name, dtype=None):
    out = network.add_identity(cast(network, tensor, dtype) if dtype is not None else tensor).get_output(0)
    out.name = name
    network.mark_output(out)
    return out

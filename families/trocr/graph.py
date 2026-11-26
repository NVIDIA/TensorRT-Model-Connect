# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for TrOCR.

Both halves use the exact error-function GELU (``gelu`` in the reference) and
torch-style layer norms whose statistics are taken in float32. Rows are kept as
``[seq, width]`` matrices and attention heads as ``[1, heads, seq, head_dim]``.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_prod(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)


def add_scalar(network, rank, value, dtype):
    """A broadcastable ``[1, ..., 1]`` constant of the tensor's own dtype."""
    return add_constant(network, (1,) * rank, np.full((1,), value), dtype=dtype)


def add_cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    """torch.nn.LayerNorm over the last axis with float32 statistics."""
    original = inp.dtype
    source = add_cast(network, inp, trt.float32)
    rank = len(inp.shape)
    axis = 1 << (rank - 1)
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, axis, True).get_output(0)
    centred = network.add_elementwise(source, mean, trt.ElementWiseOperation.SUB).get_output(0)
    variance = network.add_reduce(add_prod(network, centred, centred), trt.ReduceOperation.AVG,
                                  axis, True).get_output(0)
    shifted = add_sum(network, variance, add_scalar(network, rank, eps, np.float32))
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation,
                                     trt.ElementWiseOperation.DIV).get_output(0)
    width = int(inp.shape[-1])
    shape = (1,) * (rank - 1) + (width,)
    scaled = add_prod(network, normed, add_constant(network, shape, gamma))
    affine = add_sum(network, scaled, add_constant(network, shape, beta))
    return add_cast(network, affine, original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32):
    """``x @ W^T + b`` with torch.nn.Linear's ``[out, in]`` weight layout."""
    rank = len(inp.shape)
    weight = np.asarray(weight)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    constant = add_constant(network, shape, weight, dtype=dtype)
    out = network.add_matrix_multiply(inp, trt.MatrixOperation.NONE, constant,
                                      trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias_shape = (1,) * (rank - 1) + (int(weight.shape[0]),)
    return add_sum(network, out, add_constant(network, bias_shape, bias, dtype=dtype))


def add_gelu(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + erf(x / sqrt(2)))``, the reference's ``gelu``.

    Spelled out rather than delegated to TensorRT's GELU activation, which may
    pick the tanh approximation.
    """
    rank = len(inp.shape)
    scaled = add_prod(network, inp, add_scalar(network, rank, 1.0 / np.sqrt(2.0), dtype))
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    gate = add_sum(network, add_scalar(network, rank, 1.0, dtype), erf)
    return add_prod(network, add_prod(network, add_scalar(network, rank, 0.5, dtype), inp), gate)


def to_heads(network, rows, num_heads, head_dim):
    """``[seq, heads * head_dim]`` -> ``[1, heads, seq, head_dim]``."""
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (1, int(rows.shape[0]), num_heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def from_heads(network, heads):
    """``[1, heads, seq, head_dim]`` -> ``[seq, heads * head_dim]``."""
    layer = network.add_shuffle(heads)
    layer.first_transpose = (0, 2, 1, 3)
    layer.reshape_dims = (int(heads.shape[2]), int(heads.shape[1]) * int(heads.shape[3]))
    return layer.get_output(0)


def add_softmax_last(network, scores):
    layer = network.add_softmax(scores)
    layer.axes = 1 << (len(scores.shape) - 1)
    return layer.get_output(0)


def add_attention(network, query, key, value):
    """Unmasked attention of pre-scaled ``query`` heads over ``key``/``value`` heads."""
    scores = network.add_matrix_multiply(query, trt.MatrixOperation.NONE, key,
                                         trt.MatrixOperation.TRANSPOSE).get_output(0)
    weights = add_softmax_last(network, scores)
    return network.add_matrix_multiply(weights, trt.MatrixOperation.NONE, value,
                                       trt.MatrixOperation.NONE).get_output(0)

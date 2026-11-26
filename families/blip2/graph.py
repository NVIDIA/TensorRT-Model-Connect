# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations for the BLIP-2 vision, Q-Former and OPT decoder."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def _as_float32(network, tensor):
    """Cast to fp32 for a reduction's statistics, and remember the way back."""
    if tensor.dtype == trt.float32:
        return tensor, None
    return network.add_cast(tensor, trt.float32).get_output(0), tensor.dtype


def _restore_dtype(network, tensor, original):
    if original is None:
        return tensor
    return network.add_cast(tensor, original).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    source, original = _as_float32(network, inp)
    axis = len(inp.shape) - 1
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(source, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    width = int(inp.shape[-1])
    epsilon = add_constant(network, tuple([1] * len(inp.shape)), np.full((1,), eps))
    shifted = network.add_elementwise(variance, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation, trt.ElementWiseOperation.DIV).get_output(0)
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = network.add_elementwise(
        normed, add_constant(network, shape, gamma), trt.ElementWiseOperation.PROD
    ).get_output(0)
    affine = network.add_elementwise(
        scaled, add_constant(network, shape, beta), trt.ElementWiseOperation.SUM
    ).get_output(0)
    return _restore_dtype(network, affine, original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32, *, matmul=None):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout.

    ``matmul`` (``(network, input) -> output``) replaces the matrix product, which is how
    a quantized projection takes its place; the bias is added the same way either way.
    """
    rank = len(inp.shape)
    if matmul is not None:
        out = matmul(network, inp)
    else:
        weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
        shape = (1,) * (rank - 2) + tuple(weight.shape)
        constant = add_constant(network, shape, weight.reshape(shape), dtype=dtype)
        out = network.add_matrix_multiply(
            inp, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
    if bias is None:
        return out
    bias_shape = (1,) * (rank - 1) + (-1,)
    bias = np.asarray(bias, dtype=dtype).reshape(bias_shape)
    return network.add_elementwise(
        out, add_constant(network, bias.shape, bias, dtype=dtype), trt.ElementWiseOperation.SUM
    ).get_output(0)

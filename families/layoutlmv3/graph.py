# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations for the LayoutLMv3 joint document encoder."""

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
    from .native_kernels import add_norm

    source, original = _as_float32(network, inp)
    width = int(inp.shape[-1])
    scale = add_constant(network, (width,), gamma)
    bias = add_constant(network, (width,), beta)
    return _restore_dtype(network, add_norm(network, source, scale, bias, eps), original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32, *, matmul=None):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout.

    ``matmul`` (``(network, input) -> output``) replaces the matrix product, which is how
    a quantized projection takes its place; the bias is added the same way either way.
    """
    rank = len(inp.shape)
    if matmul is None and bias is not None and np.dtype(dtype) == np.dtype(np.float32):
        from .native_kernels import add_fp32_kernel

        shape = tuple(inp.shape)
        rows = int(np.prod(shape[:-1]))
        flatten = network.add_shuffle(inp)
        flatten.reshape_dims = (rows, shape[-1])
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        result = add_fp32_kernel(
            network,
            "linear",
            [
                flatten.get_output(0),
                add_constant(network, weight.shape, weight),
                add_constant(network, bias.shape, bias),
            ],
            (rows, weight.shape[0]),
        )
        restore = network.add_shuffle(result)
        restore.reshape_dims = shape[:-1] + (weight.shape[0],)
        return restore.get_output(0)
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations for the Wav2Vec2Bert codec decoder."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    # Keep normalization as one mathematical operator so TensorRT owns its
    # statistics accumulation and affine lowering, including fp32 stability.
    rank = len(inp.shape)
    shape = (1,) * (rank - 1) + (int(inp.shape[-1]),)
    layer = network.add_normalization(
        inp,
        add_constant(network, shape, gamma),
        add_constant(network, shape, beta),
        1 << (rank - 1),
    )
    layer.epsilon = eps
    return layer.get_output(0)


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

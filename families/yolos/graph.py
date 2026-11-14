# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the YOLOS ViT."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def _retype(network, tensor, target):
    if tensor.dtype == target:
        return tensor
    cast = network.add_cast(tensor, target)
    return cast.get_output(0)


def add_constant(network, shape, values, dtype=np.float32):
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(tuple(shape), weights).get_output(0)


def add_linear(network, inp, weight, bias, dtype=np.float32):
    """Linear over the last axis. ``weight`` is [out_features, in_features]."""
    weight = np.asarray(weight)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    rank = len(tuple(inp.shape))
    rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
    rhs = add_constant(network, rhs_shape, weight.T.reshape(rhs_shape), dtype=dtype)
    product = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, _retype(network, rhs, inp.dtype), trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is None:
        return product
    bias_shape = (1,) * (rank - 1) + (out_features,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    summed = network.add_elementwise(
        product, _retype(network, bias_t, product.dtype), trt.ElementWiseOperation.SUM
    )
    return summed.get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps, dtype=np.float32):
    """Normalize over the last axis, then scale and shift."""
    rank = len(tuple(inp.shape))
    axes = 1 << (rank - 1)
    width = int(np.asarray(gamma).size)
    shape = (1,) * (rank - 1) + (width,)
    scale = _retype(
        network, add_constant(network, shape, np.asarray(gamma).reshape(shape), dtype=dtype),
        inp.dtype,
    )
    shift = _retype(
        network, add_constant(network, shape, np.asarray(beta).reshape(shape), dtype=dtype),
        inp.dtype,
    )
    layer = network.add_normalization(inp, scale, shift, axes)
    layer.epsilon = float(eps)
    return layer.get_output(0)


def add_gelu(network, inp):
    """The exact erf GELU that transformers' "gelu" activation uses."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_relu(network, inp):
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_sigmoid(network, inp):
    return network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(
        left, _retype(network, right, left.dtype), trt.ElementWiseOperation.SUM
    ).get_output(0)


def split_heads(network, x, tokens, num_heads, head_dim):
    """[1, tokens, hidden] -> [1, heads, tokens, head_dim]."""
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (1, tokens, num_heads, head_dim)
    shuffle.second_transpose = (0, 2, 1, 3)
    return shuffle.get_output(0)


def merge_heads(network, x, tokens, hidden):
    """[1, heads, tokens, head_dim] -> [1, tokens, hidden]."""
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = (0, 2, 1, 3)
    shuffle.reshape_dims = (1, tokens, hidden)
    return shuffle.get_output(0)


def add_attention(network, query, key, value, scale):
    """Softmax(QK^T * scale) V over the token axis of a 4-D [1,H,T,D] layout."""
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scale_t = _retype(
        network, add_constant(network, (1, 1, 1, 1), np.array([scale], dtype=np.float32)),
        scores.dtype,
    )
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD
    ).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    return network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)

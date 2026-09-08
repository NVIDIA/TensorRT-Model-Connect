# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm GhostNet."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def convolution(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype: np.dtype,
):
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=(int(weight.shape[2]), int(weight.shape[3])),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a GhostNet convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a GhostNet ReLU")
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a GhostNet residual add")
    return layer.get_output(0)


def concatenate(network, tensors):
    layer = network.add_concatenation(tensors)
    if layer is None:
        raise RuntimeError("TensorRT rejected a GhostNet concatenation")
    layer.axis = 1
    return layer.get_output(0)


def hard_sigmoid(network, tensor, *, dtype: np.dtype):
    """clamp(x / 6 + 0.5, 0, 1), the timm and PyTorch definition.

    GhostNet gates with this rather than a plain sigmoid. The two agree near
    zero, so substituting one for the other still produces a plausible ranking.
    """
    scaled = network.add_scale(
        tensor,
        trt.ScaleMode.UNIFORM,
        shift=trt.Weights(np.array([0.5], dtype=dtype)),
        scale=trt.Weights(np.array([1.0 / 6.0], dtype=dtype)),
    )
    if scaled is None:
        raise RuntimeError("TensorRT rejected the GhostNet hard sigmoid scale")
    clipped = network.add_activation(scaled.get_output(0), trt.ActivationType.CLIP)
    if clipped is None:
        raise RuntimeError("TensorRT rejected the GhostNet hard sigmoid clip")
    clipped.alpha = 0.0
    clipped.beta = 1.0
    return clipped.get_output(0)


def squeeze_excite(
    network,
    tensor,
    reduce_weight: np.ndarray,
    reduce_bias: np.ndarray,
    expand_weight: np.ndarray,
    expand_bias: np.ndarray,
    *,
    dtype: np.dtype,
):
    """Per-channel gate: mean, two 1x1 convolutions, hard sigmoid, multiply."""
    pooled = network.add_reduce(tensor, trt.ReduceOperation.AVG, (1 << 2) | (1 << 3), True)
    if pooled is None:
        raise RuntimeError("TensorRT rejected the GhostNet squeeze pooling")
    gate = convolution(network, pooled.get_output(0), reduce_weight, reduce_bias, dtype=dtype)
    gate = relu(network, gate)
    gate = convolution(network, gate, expand_weight, expand_bias, dtype=dtype)
    gate = hard_sigmoid(network, gate, dtype=dtype)
    scaled = network.add_elementwise(tensor, gate, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError("TensorRT rejected the GhostNet excitation product")
    return scaled.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected GhostNet global average pooling")
    layer.stride_nd = (1, 1)
    return layer.get_output(0)


def classifier(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dtype: np.dtype,
):
    flattened = network.add_shuffle(tensor)
    if flattened is None:
        raise RuntimeError("TensorRT rejected the GhostNet classifier reshape")
    flattened.reshape_dims = (1, int(weight.shape[1]))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    matrix_layer = network.add_constant(matrix.shape, trt.Weights(matrix))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the GhostNet classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the GhostNet classifier matmul")
    values = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(values.shape, trt.Weights(values))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the GhostNet classifier bias")
    output = network.add_elementwise(
        product.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the GhostNet classifier output")
    return output.get_output(0)

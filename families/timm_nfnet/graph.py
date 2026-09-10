# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm NFNet."""

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
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    dtype: np.dtype,
):
    """A convolution whose padding is given as (top, left, bottom, right).

    NFNet uses TensorFlow's SAME padding, which is one pixel wider on the
    bottom and right when the stride does not divide the input evenly, so the
    padding cannot be expressed as a single symmetric pair.
    """
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=(int(weight.shape[2]), int(weight.shape[3])),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected an NFNet convolution")
    layer.stride_nd = (stride, stride)
    layer.pre_padding = (padding[0], padding[1])
    layer.post_padding = (padding[2], padding[3])
    layer.num_groups = groups
    return layer.get_output(0)


def gamma_activation(network, tensor, *, gamma: float, dtype: np.dtype):
    """GELU scaled by the constant that keeps its output variance at one.

    NFNet has no normalisation layers, so each activation carries the gain that
    a norm would otherwise supply.
    """
    layer = network.add_activation(tensor, trt.ActivationType.GELU_ERF)
    if layer is None:
        raise RuntimeError("TensorRT rejected an NFNet GELU")
    return scale_by(network, layer.get_output(0), gamma, dtype=dtype)


def scale_by(network, tensor, value: float, *, dtype: np.dtype):
    shape = (1,) * len(tuple(tensor.shape))
    constant = network.add_constant(
        shape, trt.Weights(np.array([value], dtype=dtype).reshape(shape))
    )
    if constant is None:
        raise RuntimeError("TensorRT rejected an NFNet scalar")
    values = constant.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected an NFNet scalar cast")
        values = cast.get_output(0)
    layer = network.add_elementwise(tensor, values, trt.ElementWiseOperation.PROD)
    if layer is None:
        raise RuntimeError("TensorRT rejected an NFNet scalar product")
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected an NFNet residual add")
    return layer.get_output(0)


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
    """Per-channel gate: mean, two 1x1 convolutions, sigmoid, multiply."""
    pooled = network.add_reduce(tensor, trt.ReduceOperation.AVG, (1 << 2) | (1 << 3), True)
    if pooled is None:
        raise RuntimeError("TensorRT rejected the NFNet squeeze pooling")
    gate = convolution(network, pooled.get_output(0), reduce_weight, reduce_bias, dtype=dtype)
    relu = network.add_activation(gate, trt.ActivationType.RELU)
    if relu is None:
        raise RuntimeError("TensorRT rejected the NFNet excitation ReLU")
    gate = convolution(network, relu.get_output(0), expand_weight, expand_bias, dtype=dtype)
    activation = network.add_activation(gate, trt.ActivationType.SIGMOID)
    if activation is None:
        raise RuntimeError("TensorRT rejected the NFNet excitation sigmoid")
    scaled = network.add_elementwise(
        tensor, activation.get_output(0), trt.ElementWiseOperation.PROD
    )
    if scaled is None:
        raise RuntimeError("TensorRT rejected the NFNet excitation product")
    return scaled.get_output(0)


def average_pool(network, tensor, *, kernel: int, stride: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected an NFNet average pooling")
    layer.stride_nd = (stride, stride)
    return layer.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected NFNet global average pooling")
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
        raise RuntimeError("TensorRT rejected the NFNet classifier reshape")
    flattened.reshape_dims = (1, int(weight.shape[1]))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    matrix_layer = network.add_constant(matrix.shape, trt.Weights(matrix))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the NFNet classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the NFNet classifier matmul")
    values = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(values.shape, trt.Weights(values))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the NFNet classifier bias")
    output = network.add_elementwise(
        product.get_output(0), bias_layer.get_output(0), trt.ElementWiseOperation.SUM
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the NFNet classifier output")
    return output.get_output(0)

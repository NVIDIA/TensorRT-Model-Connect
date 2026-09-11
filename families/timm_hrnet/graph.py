# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm HRNet."""

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
        raise RuntimeError("TensorRT rejected a HRNet convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a HRNet ReLU")
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a HRNet residual add")
    return layer.get_output(0)


def max_pool(network, tensor, *, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected the HRNet stem pooling")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def nearest_upsample(network, tensor, factor: int):
    """Nearest-neighbour upsample on the two spatial axes."""
    layer = network.add_resize(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected an HRNet upsample")
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.scales = [1.0, 1.0, float(factor), float(factor)]
    return layer.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected HRNet global average pooling")
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
        raise RuntimeError("TensorRT rejected the HRNet classifier reshape")
    flattened.reshape_dims = (1, int(weight.shape[1]))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    matrix_layer = network.add_constant(matrix.shape, trt.Weights(matrix))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the HRNet classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the HRNet classifier matmul")
    values = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(values.shape, trt.Weights(values))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the HRNet classifier bias")
    output = network.add_elementwise(
        product.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the HRNet classifier output")
    return output.get_output(0)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm Res2Net."""

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
        raise RuntimeError("TensorRT rejected a Res2Net convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a Res2Net ReLU")
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a Res2Net residual add")
    return layer.get_output(0)


def max_pool(network, tensor, *, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected the Res2Net stem pooling")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected Res2Net global average pooling")
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
        raise RuntimeError("TensorRT rejected the Res2Net classifier reshape")
    flattened.reshape_dims = (1, int(weight.shape[1]))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    matrix_layer = network.add_constant(matrix.shape, trt.Weights(matrix))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the Res2Net classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the Res2Net classifier matmul")
    values = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(values.shape, trt.Weights(values))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the Res2Net classifier bias")
    output = network.add_elementwise(
        product.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the Res2Net classifier output")
    return output.get_output(0)


def average_pool(
    network, tensor, *, kernel: int, stride: int, padding: int, count_include_pad: bool
):
    """Average pooling.

    TensorRT excludes padded cells from the divisor by default and PyTorch
    includes them, so the caller states which convention the checkpoint was
    trained with rather than inheriting either default.
    """
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected a Res2Net average pooling")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.average_count_excludes_padding = not count_include_pad
    return layer.get_output(0)


def channel_slice(network, tensor, start: int, count: int):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[1], sizes[1] = start, count
    layer = network.add_slice(tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected a Res2Net channel slice")
    return layer.get_output(0)


def concatenate(network, tensors):
    layer = network.add_concatenation(list(tensors))
    if layer is None:
        raise RuntimeError("TensorRT rejected a Res2Net concatenation")
    layer.axis = 1
    return layer.get_output(0)

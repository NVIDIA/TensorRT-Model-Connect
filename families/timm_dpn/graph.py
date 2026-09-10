# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm DPN."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def convolution(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype: np.dtype,
):
    # DPN convolutions carry no bias of their own; a zero bias keeps one code
    # path here instead of a null-weight special case.
    offsets = np.zeros((int(weight.shape[0]),), dtype=dtype) if bias is None else bias
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=(int(weight.shape[2]), int(weight.shape[3])),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(offsets, dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def batch_norm(
    network,
    tensor,
    scale: np.ndarray,
    shift: np.ndarray,
    *,
    dtype: np.dtype,
):
    """Apply a per-channel affine transform.

    DPN puts its norm before the convolution with a ReLU between the two, so
    unlike the other timm families the norm cannot be folded into a
    convolution and has to stand on its own.
    """
    layer = network.add_scale_nd(
        tensor,
        trt.ScaleMode.CHANNEL,
        shift=trt.Weights(np.ascontiguousarray(shift, dtype=dtype)),
        scale=trt.Weights(np.ascontiguousarray(scale, dtype=dtype)),
        power=trt.Weights(np.ascontiguousarray(np.ones_like(scale), dtype=dtype)),
        channel_axis=1,
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN batch norm")
    return layer.get_output(0)


def relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN ReLU")
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN residual add")
    return layer.get_output(0)


def concatenate(network, tensors):
    layer = network.add_concatenation(list(tensors))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN concatenation")
    layer.axis = 1
    return layer.get_output(0)


def channel_slice(network, tensor, start: int, length: int, shape: tuple[int, int, int, int]):
    """Take `length` channels starting at `start`.

    The spatial extent is fixed at build time, so the slice is static.
    """
    layer = network.add_slice(
        tensor,
        start=(0, start, 0, 0),
        shape=(shape[0], length, shape[2], shape[3]),
        stride=(1, 1, 1, 1),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN channel slice")
    return layer.get_output(0)


def max_pool(network, tensor, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DPN max pooling")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected DPN global average pooling")
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
    """DPN classifies with a 1x1 convolution, so the weight arrives 4-D."""
    matrix = weight.reshape(int(weight.shape[0]), int(weight.shape[1]))
    flattened = network.add_shuffle(tensor)
    if flattened is None:
        raise RuntimeError("TensorRT rejected the DPN classifier reshape")
    flattened.reshape_dims = (1, int(matrix.shape[1]))
    values = np.ascontiguousarray(matrix.T, dtype=dtype)
    matrix_layer = network.add_constant(values.shape, trt.Weights(values))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the DPN classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the DPN classifier matmul")
    offsets = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(offsets.shape, trt.Weights(offsets))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the DPN classifier bias")
    output = network.add_elementwise(
        product.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the DPN classifier output")
    return output.get_output(0)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by YOLOv10."""

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
        raise RuntimeError("TensorRT rejected a YOLOv10 convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def silu(network, tensor):
    """x * sigmoid(x), the activation every YOLOv10 convolution uses."""
    gate = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if gate is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 SiLU sigmoid")
    product = network.add_elementwise(tensor, gate.get_output(0), trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 SiLU product")
    return product.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 add")
    return layer.get_output(0)


def concatenate(network, tensors, *, axis: int = 1):
    layer = network.add_concatenation(tensors)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 concatenation")
    layer.axis = axis
    return layer.get_output(0)


def slice_axis(network, tensor, *, axis: int, start: int, count: int):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[axis], sizes[axis] = start, count
    layer = network.add_slice(
        tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 slice")
    return layer.get_output(0)


def max_pool(network, tensor, *, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 max pool")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def nearest_upsample(network, tensor, factor: int):
    layer = network.add_resize(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 upsample")
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.scales = [1.0, 1.0, float(factor), float(factor)]
    return layer.get_output(0)


def reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 reshape")
    layer.reshape_dims = trt.Dims(shape)
    return layer.get_output(0)


def permute(network, tensor, permutation: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 permutation")
    layer.second_transpose = trt.Permutation(permutation)
    return layer.get_output(0)


def matmul(network, left, right, *, transpose_left=False, transpose_right=False):
    layer = network.add_matrix_multiply(
        left,
        trt.MatrixOperation.TRANSPOSE if transpose_left else trt.MatrixOperation.NONE,
        right,
        trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE,
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 matmul")
    return layer.get_output(0)


def softmax(network, tensor, axis: int):
    layer = network.add_softmax(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 softmax")
    layer.axes = 1 << axis
    return layer.get_output(0)


def sigmoid(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 sigmoid")
    return layer.get_output(0)


def scale(network, tensor, factor: float, *, dtype: np.dtype):
    shape = (1,) * len(tuple(tensor.shape))
    layer = network.add_constant(
        shape, trt.Weights(np.array([factor], dtype=dtype).reshape(shape))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 scale constant")
    values = layer.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a YOLOv10 scale cast")
        values = cast.get_output(0)
    product = network.add_elementwise(tensor, values, trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 scale product")
    return product.get_output(0)


def constant(network, values: np.ndarray, *, dtype: np.dtype, like=None):
    layer = network.add_constant(
        values.shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 constant")
    output = layer.get_output(0)
    if like is None or output.dtype == like.dtype:
        return output
    cast = network.add_cast(output, like.dtype)
    if cast is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 constant cast")
    return cast.get_output(0)


def subtract(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUB)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 subtraction")
    return layer.get_output(0)


def reduce_max(network, tensor, axis: int, *, keep_dims: bool):
    layer = network.add_reduce(tensor, trt.ReduceOperation.MAX, 1 << axis, keep_dims)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 max reduction")
    return layer.get_output(0)


def top_k(network, tensor, *, k: int, axis: int):
    """Largest `k` values along one axis, with their indices."""
    layer = network.add_topk(tensor, trt.TopKOperation.MAX, k, 1 << axis)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 top-k")
    return layer.get_output(0), layer.get_output(1)


def gather(network, tensor, indices, *, axis: int):
    layer = network.add_gather(tensor, indices, axis)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 gather")
    return layer.get_output(0)


def multiply(network, left, right):
    """Element-wise product; TensorRT broadcasts size-one axes."""
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.PROD)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 product")
    return layer.get_output(0)


def floor_divide(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.FLOOR_DIV)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOv10 floor division")
    return layer.get_output(0)

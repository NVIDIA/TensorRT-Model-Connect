# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by YOLOX."""

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
        raise RuntimeError("TensorRT rejected a YOLOX convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def silu(network, tensor):
    """SiLU with FP32 internal arithmetic, as in PyTorch's half kernel.

    Rounding sigmoid and its product separately in FP16 differs from the
    upstream single activation and accumulates across the CSP blocks.
    """
    dtype = tensor.dtype
    if dtype == trt.float16:
        cast = network.add_cast(tensor, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX SiLU input cast")
        tensor = cast.get_output(0)
    gate = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if gate is None:
        raise RuntimeError("TensorRT rejected a YOLOX SiLU sigmoid")
    product = network.add_elementwise(tensor, gate.get_output(0), trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected a YOLOX SiLU product")
    output = product.get_output(0)
    if dtype == trt.float16:
        cast = network.add_cast(output, trt.float16)
        if cast is None:
            raise RuntimeError("TensorRT rejected the YOLOX SiLU output cast")
        output = cast.get_output(0)
    return output


def leaky_relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.LEAKY_RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX leaky ReLU")
    layer.alpha = 0.1
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX add")
    return layer.get_output(0)


def concatenate(network, tensors, *, axis: int = 1):
    layer = network.add_concatenation(tensors)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX concatenation")
    layer.axis = axis
    return layer.get_output(0)


def slice_axis(network, tensor, *, axis: int, start: int, count: int):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[axis], sizes[axis] = start, count
    layer = network.add_slice(tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX slice")
    return layer.get_output(0)


def max_pool(network, tensor, *, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX max pool")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def nearest_upsample(network, tensor, factor: int):
    layer = network.add_resize(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX upsample")
    layer.resize_mode = trt.InterpolationMode.NEAREST
    layer.scales = [1.0, 1.0, float(factor), float(factor)]
    return layer.get_output(0)


def reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX reshape")
    layer.reshape_dims = trt.Dims(shape)
    return layer.get_output(0)


def permute(network, tensor, permutation: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX permutation")
    layer.second_transpose = trt.Permutation(permutation)
    return layer.get_output(0)


def sigmoid(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX sigmoid")
    return layer.get_output(0)


def scale(network, tensor, factor: float, *, dtype: np.dtype):
    shape = (1,) * len(tuple(tensor.shape))
    layer = network.add_constant(shape, trt.Weights(np.array([factor], dtype=dtype).reshape(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX scale constant")
    values = layer.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a YOLOX scale cast")
        values = cast.get_output(0)
    product = network.add_elementwise(tensor, values, trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected a YOLOX scale product")
    return product.get_output(0)


def constant(network, values: np.ndarray, *, dtype: np.dtype, like=None):
    layer = network.add_constant(
        values.shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX constant")
    output = layer.get_output(0)
    if like is None or output.dtype == like.dtype:
        return output
    cast = network.add_cast(output, like.dtype)
    if cast is None:
        raise RuntimeError("TensorRT rejected a YOLOX constant cast")
    return cast.get_output(0)


def subtract(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUB)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX subtraction")
    return layer.get_output(0)


def top_k(network, tensor, *, k: int, axis: int):
    """Largest `k` values along one axis, with their indices."""
    layer = network.add_topk(tensor, trt.TopKOperation.MAX, k, 1 << axis)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX top-k")
    return layer.get_output(0), layer.get_output(1)


def multiply(network, left, right):
    """Element-wise product; TensorRT broadcasts size-one axes."""
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.PROD)
    if layer is None:
        raise RuntimeError("TensorRT rejected a YOLOX product")
    return layer.get_output(0)

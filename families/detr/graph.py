# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by DETR."""

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
        raise RuntimeError("TensorRT rejected a DETR convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def relu(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.RELU)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR ReLU")
    return layer.get_output(0)


def sigmoid(network, tensor):
    layer = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR sigmoid")
    return layer.get_output(0)


def max_pool(network, tensor, kernel: int, stride: int, padding: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.MAX, (kernel, kernel))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR max pooling")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR add")
    return layer.get_output(0)


def subtract(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUB)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR subtract")
    return layer.get_output(0)


def multiply(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.PROD)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR multiply")
    return layer.get_output(0)


def reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR reshape")
    layer.reshape_dims = trt.Dims(shape)
    return layer.get_output(0)


def permute(network, tensor, first, permutation, second):
    """One shuffle: reshape, transpose, then an optional second reshape."""
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR shuffle")
    if first is not None:
        layer.reshape_dims = trt.Dims(first)
    layer.second_transpose = trt.Permutation(permutation)
    output = layer.get_output(0)
    return reshape(network, output, second) if second is not None else output


def constant(network, values: np.ndarray, *, dtype: np.dtype, like):
    layer = network.add_constant(
        values.shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR constant")
    output = layer.get_output(0)
    if like is None or output.dtype == like.dtype:
        return output
    cast = network.add_cast(output, like.dtype)
    if cast is None:
        raise RuntimeError("TensorRT rejected a DETR constant cast")
    return cast.get_output(0)


def linear(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
):
    """A Linear layer: `tensor @ weight.T`, plus an optional bias."""
    rank = len(tuple(tensor.shape))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    shape = (1,) * (rank - 2) + matrix.shape
    values = constant(network, matrix.reshape(shape), dtype=dtype, like=tensor)
    product = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    )
    if product is None:
        raise RuntimeError("TensorRT rejected a DETR matmul")
    output = product.get_output(0)
    if bias is None:
        return output
    bias_shape = (1,) * (rank - 1) + (int(bias.shape[0]),)
    offsets = constant(network, bias.reshape(bias_shape), dtype=dtype, like=output)
    return add(network, output, offsets)


def layer_norm(
    network,
    tensor,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    epsilon: float,
    dtype: np.dtype,
):
    """LayerNorm over the last axis."""
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (int(gamma.shape[0]),)
    scale = constant(network, gamma.reshape(shape), dtype=dtype, like=tensor)
    shift = constant(network, beta.reshape(shape), dtype=dtype, like=tensor)
    layer = network.add_normalization_v2(tensor, scale, shift, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR layer norm")
    layer.epsilon = epsilon
    if hasattr(layer, "compute_precision"):
        layer.compute_precision = trt.float32
    return layer.get_output(0)


def attention(network, query, key, value, *, dtype: np.dtype):
    """Scaled dot-product attention over [batch, heads, tokens, head_dim]."""
    head_dim = int(query.shape[-1])
    factor = np.array([1.0 / np.sqrt(head_dim)], dtype=dtype).reshape((1, 1, 1, 1))
    scaled = multiply(network, query, constant(network, factor, dtype=dtype, like=query))
    core = network.add_attention(scaled, key, value, trt.AttentionNormalizationOp.SOFTMAX, False)
    if core is None:
        raise RuntimeError("TensorRT rejected a DETR attention")
    core.decomposable = True
    return core.get_output(0)


def softmax(network, tensor, axis: int):
    layer = network.add_softmax(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR softmax")
    layer.axes = 1 << axis
    return layer.get_output(0)


def slice_axis(network, tensor, start: int, count: int, axis: int):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[axis], sizes[axis] = start, count
    layer = network.add_slice(tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR slice")
    return layer.get_output(0)


def reduce_max(network, tensor, axis: int, *, keep_dims: bool):
    layer = network.add_reduce(tensor, trt.ReduceOperation.MAX, 1 << axis, keep_dims)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR max reduction")
    return layer.get_output(0)


def top_k(network, tensor, *, k: int, axis: int):
    layer = network.add_topk(tensor, trt.TopKOperation.MAX, k, 1 << axis)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR top-k")
    return layer.get_output(0), layer.get_output(1)


def gather(network, tensor, indices, *, axis: int):
    layer = network.add_gather(tensor, indices, axis)
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR gather")
    return layer.get_output(0)


def concatenate(network, tensors, *, axis: int):
    layer = network.add_concatenation(list(tensors))
    if layer is None:
        raise RuntimeError("TensorRT rejected a DETR concatenation")
    layer.axis = axis
    return layer.get_output(0)

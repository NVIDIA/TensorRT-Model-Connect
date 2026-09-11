# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm MobileViT."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT reshape")
    layer.reshape_dims = trt.Dims(shape)
    return layer.get_output(0)


def permute(network, tensor, first, permutation, second):
    """One shuffle: reshape, transpose, then an optional second reshape."""
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT shuffle")
    if first is not None:
        layer.reshape_dims = trt.Dims(first)
    layer.second_transpose = trt.Permutation(permutation)
    output = layer.get_output(0)
    return reshape(network, output, second) if second is not None else output


def matmul_constant(
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
    layer = network.add_constant(shape, trt.Weights(matrix.reshape(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT linear weight")
    values = layer.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a MobileViT linear cast")
        values = cast.get_output(0)
    product = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    )
    if product is None:
        raise RuntimeError("TensorRT rejected a MobileViT matmul")
    output = product.get_output(0)
    if bias is None:
        return output
    bias_shape = (1,) * (rank - 1) + (int(bias.shape[0]),)
    bias_layer = network.add_constant(
        bias_shape, trt.Weights(np.ascontiguousarray(bias.reshape(bias_shape), dtype=dtype))
    )
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT linear bias")
    values = bias_layer.get_output(0)
    if values.dtype != output.dtype:
        cast = network.add_cast(values, output.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a MobileViT bias cast")
        values = cast.get_output(0)
    summed = network.add_elementwise(output, values, trt.ElementWiseOperation.SUM)
    if summed is None:
        raise RuntimeError("TensorRT rejected a MobileViT bias add")
    return summed.get_output(0)


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
    scale = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(gamma.reshape(shape), dtype=dtype))
    )
    shift = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(beta.reshape(shape), dtype=dtype))
    )
    if scale is None or shift is None:
        raise RuntimeError("TensorRT rejected the MobileViT norm parameters")
    scale_tensor, shift_tensor = scale.get_output(0), shift.get_output(0)
    if scale_tensor.dtype != tensor.dtype:
        scale_cast = network.add_cast(scale_tensor, tensor.dtype)
        shift_cast = network.add_cast(shift_tensor, tensor.dtype)
        if scale_cast is None or shift_cast is None:
            raise RuntimeError("TensorRT rejected the MobileViT norm cast")
        scale_tensor, shift_tensor = scale_cast.get_output(0), shift_cast.get_output(0)
    layer = network.add_normalization_v2(tensor, scale_tensor, shift_tensor, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT layer norm")
    layer.epsilon = epsilon
    if hasattr(layer, "compute_precision"):
        layer.compute_precision = trt.float32
    return layer.get_output(0)


def attention(network, query, key, value, mask, *, dtype: np.dtype):
    """Scaled dot-product attention with an additive bias."""
    head_dim = int(query.shape[-1])
    factor = float(1.0 / np.sqrt(head_dim))
    shape = (1, 1, 1, 1)
    layer = network.add_constant(shape, trt.Weights(np.array([factor], dtype=dtype).reshape(shape)))
    if layer is None:
        raise RuntimeError("TensorRT rejected the MobileViT attention scale")
    values = layer.get_output(0)
    if values.dtype != query.dtype:
        cast = network.add_cast(values, query.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the MobileViT attention scale cast")
        values = cast.get_output(0)
    scaled = network.add_elementwise(query, values, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError("TensorRT rejected the MobileViT query scale")
    core = network.add_attention(
        scaled.get_output(0), key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )
    if core is None:
        raise RuntimeError("TensorRT rejected the MobileViT attention")
    core.decomposable = True
    if mask is not None:
        core.mask = mask
    return core.get_output(0)


def constant(network, values: np.ndarray, *, dtype: np.dtype, like):
    layer = network.add_constant(
        values.shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT constant")
    output = layer.get_output(0)
    if output.dtype == like.dtype:
        return output
    cast = network.add_cast(output, like.dtype)
    if cast is None:
        raise RuntimeError("TensorRT rejected a MobileViT constant cast")
    return cast.get_output(0)


def concatenate_tokens(network, tensors):
    layer = network.add_concatenation(tensors)
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT token concatenation")
    layer.axis = 1
    return layer.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT add")
    return layer.get_output(0)


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
    """A convolution. A bias of None means the checkpoint carries none."""
    offsets = np.zeros((int(weight.shape[0]),), dtype=dtype) if bias is None else bias
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=(int(weight.shape[2]), int(weight.shape[3])),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(offsets, dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def silu(network, tensor):
    """x * sigmoid(x), the activation every MobileViT convolution uses."""
    gate = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if gate is None:
        raise RuntimeError("TensorRT rejected a MobileViT SiLU sigmoid")
    product = network.add_elementwise(tensor, gate.get_output(0), trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected a MobileViT SiLU product")
    return product.get_output(0)


def concatenate_channels(network, tensors):
    layer = network.add_concatenation(list(tensors))
    if layer is None:
        raise RuntimeError("TensorRT rejected a MobileViT channel concatenation")
    layer.axis = 1
    return layer.get_output(0)


def global_average_pool(network, tensor, height: int, width: int):
    layer = network.add_pooling_nd(tensor, trt.PoolingType.AVERAGE, (height, width))
    if layer is None:
        raise RuntimeError("TensorRT rejected MobileViT global average pooling")
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
    flattened = reshape(network, tensor, (1, int(weight.shape[1])))
    return matmul_constant(network, flattened, weight, bias, dtype=dtype)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm Swin Transformer."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def patch_convolution(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    patch: int,
    dtype: np.dtype,
):
    """Non-overlapping patch convolution: the stride equals the kernel size."""
    layer = network.add_convolution_nd(
        tensor,
        num_output_maps=int(weight.shape[0]),
        kernel_shape=(patch, patch),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype)),
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected the Swin patch convolution")
    layer.stride_nd = (patch, patch)
    return layer.get_output(0)


def reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin reshape")
    layer.reshape_dims = trt.Dims(shape)
    return layer.get_output(0)


def permute(network, tensor, first, permutation, second):
    """One shuffle: reshape, transpose, then an optional second reshape."""
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin shuffle")
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
        raise RuntimeError("TensorRT rejected a Swin linear weight")
    values = layer.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a Swin linear cast")
        values = cast.get_output(0)
    product = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    )
    if product is None:
        raise RuntimeError("TensorRT rejected a Swin matmul")
    output = product.get_output(0)
    if bias is None:
        return output
    bias_shape = (1,) * (rank - 1) + (int(bias.shape[0]),)
    bias_layer = network.add_constant(
        bias_shape, trt.Weights(np.ascontiguousarray(bias.reshape(bias_shape), dtype=dtype))
    )
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected a Swin linear bias")
    values = bias_layer.get_output(0)
    if values.dtype != output.dtype:
        cast = network.add_cast(values, output.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a Swin bias cast")
        values = cast.get_output(0)
    summed = network.add_elementwise(output, values, trt.ElementWiseOperation.SUM)
    if summed is None:
        raise RuntimeError("TensorRT rejected a Swin bias add")
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
        raise RuntimeError("TensorRT rejected the Swin norm parameters")
    scale_tensor, shift_tensor = scale.get_output(0), shift.get_output(0)
    if scale_tensor.dtype != tensor.dtype:
        scale_cast = network.add_cast(scale_tensor, tensor.dtype)
        shift_cast = network.add_cast(shift_tensor, tensor.dtype)
        if scale_cast is None or shift_cast is None:
            raise RuntimeError("TensorRT rejected the Swin norm cast")
        scale_tensor, shift_tensor = scale_cast.get_output(0), shift_cast.get_output(0)
    layer = network.add_normalization_v2(tensor, scale_tensor, shift_tensor, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin layer norm")
    layer.epsilon = epsilon
    if hasattr(layer, "compute_precision"):
        layer.compute_precision = trt.float32
    return layer.get_output(0)


def gelu(network, tensor, *, dtype: np.dtype):
    """GELU, exact erf form: 0.5 * x * (1 + erf(x / sqrt(2)))."""

    def constant(value: float):
        shape = (1,) * len(tuple(tensor.shape))
        layer = network.add_constant(
            shape, trt.Weights(np.array([value], dtype=dtype).reshape(shape))
        )
        if layer is None:
            raise RuntimeError("TensorRT rejected a Swin GELU constant")
        output = layer.get_output(0)
        if output.dtype == tensor.dtype:
            return output
        cast = network.add_cast(output, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a Swin GELU cast")
        return cast.get_output(0)

    scaled = network.add_elementwise(
        tensor, constant(1.0 / np.sqrt(2.0)), trt.ElementWiseOperation.PROD
    )
    if scaled is None:
        raise RuntimeError("TensorRT rejected the Swin GELU scale")
    error = network.add_unary(scaled.get_output(0), trt.UnaryOperation.ERF)
    if error is None:
        raise RuntimeError("TensorRT rejected the Swin GELU erf")
    shifted = network.add_elementwise(
        error.get_output(0), constant(1.0), trt.ElementWiseOperation.SUM
    )
    half = network.add_elementwise(tensor, constant(0.5), trt.ElementWiseOperation.PROD)
    if shifted is None or half is None:
        raise RuntimeError("TensorRT rejected the Swin GELU combination")
    output = network.add_elementwise(
        half.get_output(0), shifted.get_output(0), trt.ElementWiseOperation.PROD
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the Swin GELU product")
    return output.get_output(0)


def attention(network, query, key, value, mask, *, dtype: np.dtype):
    """Scaled dot-product attention with an additive bias."""
    head_dim = int(query.shape[-1])
    factor = float(1.0 / np.sqrt(head_dim))
    shape = (1, 1, 1, 1)
    layer = network.add_constant(
        shape, trt.Weights(np.array([factor], dtype=dtype).reshape(shape))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected the Swin attention scale")
    values = layer.get_output(0)
    if values.dtype != query.dtype:
        cast = network.add_cast(values, query.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Swin attention scale cast")
        values = cast.get_output(0)
    scaled = network.add_elementwise(query, values, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError("TensorRT rejected the Swin query scale")
    core = network.add_attention(
        scaled.get_output(0), key, value, trt.AttentionNormalizationOp.SOFTMAX, False
    )
    if core is None:
        raise RuntimeError("TensorRT rejected the Swin attention")
    core.decomposable = True
    core.mask = mask
    return core.get_output(0)


def roll(network, tensor, shifts: tuple[int, int], axes: tuple[int, int]):
    """Cyclic shift along two axes, built from slices and concatenations.

    TensorRT has no roll operator, so each axis is split at the shift point and
    the two pieces are concatenated in the opposite order.

    Follows torch.roll: a positive shift moves elements towards higher indices,
    so the split point is measured from the end. Getting this sign backwards
    still produces a well-formed tensor of the right shape, just the wrong one.
    """
    shape = [int(value) for value in tensor.shape]
    for shift, axis in zip(shifts, axes):
        extent = shape[axis]
        offset = (-shift) % extent
        if offset == 0:
            continue
        pieces = []
        for start, size in ((offset, extent - offset), (0, offset)):
            starts, sizes = [0] * len(shape), list(shape)
            starts[axis], sizes[axis] = start, size
            layer = network.add_slice(
                tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape))
            )
            if layer is None:
                raise RuntimeError("TensorRT rejected a Swin roll slice")
            pieces.append(layer.get_output(0))
        concat = network.add_concatenation(pieces)
        if concat is None:
            raise RuntimeError("TensorRT rejected a Swin roll concatenation")
        concat.axis = axis
        tensor = concat.get_output(0)
    return tensor


def slice_tokens(network, tensor, start: int, count: int, axis: int = 1):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[axis], sizes[axis] = start, count
    layer = network.add_slice(
        tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin slice")
    return layer.get_output(0)


def constant(network, values: np.ndarray, *, dtype: np.dtype, like):
    layer = network.add_constant(
        values.shape, trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin constant")
    output = layer.get_output(0)
    if output.dtype == like.dtype:
        return output
    cast = network.add_cast(output, like.dtype)
    if cast is None:
        raise RuntimeError("TensorRT rejected a Swin constant cast")
    return cast.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a Swin add")
    return layer.get_output(0)


def mean_spatial_nhwc(network, tensor):
    """Mean over the two spatial axes of an NHWC tensor, dropping them."""
    layer = network.add_reduce(tensor, trt.ReduceOperation.AVG, (1 << 1) | (1 << 2), False)
    if layer is None:
        raise RuntimeError("TensorRT rejected the Swin spatial mean")
    return layer.get_output(0)

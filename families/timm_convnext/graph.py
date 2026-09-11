# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by timm ConvNeXt."""

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
        raise RuntimeError("TensorRT rejected a ConvNeXt convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    layer.num_groups = groups
    return layer.get_output(0)


def layer_norm_channels(
    network,
    tensor,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    epsilon: float,
    dtype: np.dtype,
):
    """LayerNorm over the channel axis of an NCHW tensor.

    ConvNeXt normalises across channels while keeping the spatial layout, which
    is the transpose of the usual last-axis LayerNorm. Normalising the wrong
    axis still type-checks and still yields a correctly shaped tensor.
    """
    shape = (1, int(gamma.shape[0]), 1, 1)
    scale = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(gamma.reshape(shape), dtype=dtype))
    )
    shift = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(beta.reshape(shape), dtype=dtype))
    )
    if scale is None or shift is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt norm parameters")
    scale_tensor, shift_tensor = scale.get_output(0), shift.get_output(0)
    if scale_tensor.dtype != tensor.dtype:
        scale_cast = network.add_cast(scale_tensor, tensor.dtype)
        shift_cast = network.add_cast(shift_tensor, tensor.dtype)
        if scale_cast is None or shift_cast is None:
            raise RuntimeError("TensorRT rejected the ConvNeXt norm cast")
        scale_tensor, shift_tensor = scale_cast.get_output(0), shift_cast.get_output(0)
    layer = network.add_normalization_v2(tensor, scale_tensor, shift_tensor, 1 << 1)
    if layer is None:
        raise RuntimeError("TensorRT rejected a ConvNeXt layer norm")
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
            raise RuntimeError("TensorRT rejected a ConvNeXt GELU constant")
        output = layer.get_output(0)
        if output.dtype == tensor.dtype:
            return output
        cast = network.add_cast(output, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected a ConvNeXt GELU cast")
        return cast.get_output(0)

    scaled = network.add_elementwise(
        tensor, constant(1.0 / np.sqrt(2.0)), trt.ElementWiseOperation.PROD
    )
    if scaled is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt GELU scale")
    error = network.add_unary(scaled.get_output(0), trt.UnaryOperation.ERF)
    if error is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt GELU erf")
    shifted = network.add_elementwise(
        error.get_output(0), constant(1.0), trt.ElementWiseOperation.SUM
    )
    half = network.add_elementwise(tensor, constant(0.5), trt.ElementWiseOperation.PROD)
    if shifted is None or half is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt GELU combination")
    output = network.add_elementwise(
        half.get_output(0), shifted.get_output(0), trt.ElementWiseOperation.PROD
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt GELU product")
    return output.get_output(0)


def channel_scale(network, tensor, scale: np.ndarray, *, dtype: np.dtype):
    """Multiply each channel by its own learned constant."""
    shape = (1, int(scale.shape[0]), 1, 1)
    layer = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(scale.reshape(shape), dtype=dtype))
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt layer scale")
    values = layer.get_output(0)
    if values.dtype != tensor.dtype:
        cast = network.add_cast(values, tensor.dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the ConvNeXt layer scale cast")
        values = cast.get_output(0)
    product = network.add_elementwise(tensor, values, trt.ElementWiseOperation.PROD)
    if product is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt layer scale product")
    return product.get_output(0)


def add(network, left, right):
    layer = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM)
    if layer is None:
        raise RuntimeError("TensorRT rejected a ConvNeXt residual add")
    return layer.get_output(0)


def mean_spatial(network, tensor):
    """Mean over the two spatial axes, keeping them as size one."""
    layer = network.add_reduce(tensor, trt.ReduceOperation.AVG, (1 << 2) | (1 << 3), True)
    if layer is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt spatial mean")
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
        raise RuntimeError("TensorRT rejected the ConvNeXt classifier reshape")
    flattened.reshape_dims = (1, int(weight.shape[1]))
    matrix = np.ascontiguousarray(weight.T, dtype=dtype)
    matrix_layer = network.add_constant(matrix.shape, trt.Weights(matrix))
    if matrix_layer is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt classifier weights")
    product = network.add_matrix_multiply(
        flattened.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_layer.get_output(0),
        trt.MatrixOperation.NONE,
    )
    if product is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt classifier matmul")
    values = np.ascontiguousarray(bias.reshape(1, -1), dtype=dtype)
    bias_layer = network.add_constant(values.shape, trt.Weights(values))
    if bias_layer is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt classifier bias")
    output = network.add_elementwise(
        product.get_output(0),
        bias_layer.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    if output is None:
        raise RuntimeError("TensorRT rejected the ConvNeXt classifier output")
    return output.get_output(0)

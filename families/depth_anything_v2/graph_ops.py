# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned strongly-typed TensorRT graph helpers for Depth Anything V2."""

from __future__ import annotations

import numpy as np

import tensorrt as trt


def new_network(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    return builder, network, config


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _elementwise(network, lhs, rhs, operation):
    return network.add_elementwise(lhs, rhs, operation).get_output(0)


def constant(network, values: np.ndarray, shape: tuple[int, ...], dtype: np.dtype):
    array = np.ascontiguousarray(values, dtype=dtype).reshape(shape)
    return network.add_constant(shape, trt.Weights(array)).get_output(0)


def shuffle(network, tensor, **attributes):
    layer = network.add_shuffle(tensor)
    for name, value in attributes.items():
        setattr(layer, name, value)
    return layer.get_output(0)


def slice_tensor(network, tensor, start, shape):
    return network.add_slice(tensor, start, shape, (1,) * len(shape)).get_output(0)


def linear(network, tensor, weight: np.ndarray, dtype: np.dtype):
    """Apply a logical [in_features, out_features] matrix."""
    in_features, out_features = weight.shape
    rank = len(tuple(tensor.shape))
    shape = (1,) * max(0, rank - 2) + (in_features, out_features)
    rhs = constant(network, weight, shape, dtype)
    rhs = cast(network, rhs, tensor.dtype)
    return network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)


def add_bias(network, tensor, bias: np.ndarray | None, dtype: np.dtype):
    if bias is None:
        return tensor
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (int(bias.shape[0]),)
    bias_tensor = cast(network, constant(network, bias, shape, dtype), tensor.dtype)
    return _elementwise(network, tensor, bias_tensor, trt.ElementWiseOperation.SUM)


def linear_with_bias(network, tensor, weights, prefix: str, dtype: np.dtype):
    tensor = linear(network, tensor, weights[f"{prefix}.weight"], dtype)
    return add_bias(network, tensor, weights.get(f"{prefix}.bias"), dtype)


def layer_norm(
    network,
    tensor,
    hidden_size: int,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
    dtype: np.dtype,
):
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (hidden_size,)
    gamma = cast(network, constant(network, weight, shape, dtype), tensor.dtype)
    beta = cast(network, constant(network, bias, shape, dtype), tensor.dtype)
    norm = network.add_normalization_v2(tensor, gamma, beta, 1 << (rank - 1))
    norm.epsilon = eps
    return norm.get_output(0)


def gelu(network, tensor, dtype: np.dtype):
    """Exact PyTorch GELU used by the DINOv2 backbone."""
    rank = len(tuple(tensor.shape))
    scalar_shape = (1,) * rank

    def scalar(value: float):
        return cast(network, constant(network, np.asarray(value), scalar_shape, dtype), tensor.dtype)

    scaled = _elementwise(network, tensor, scalar(1.0 / np.sqrt(2.0)), trt.ElementWiseOperation.PROD)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    one_plus = _elementwise(network, erf, scalar(1.0), trt.ElementWiseOperation.SUM)
    half_x = _elementwise(network, tensor, scalar(0.5), trt.ElementWiseOperation.PROD)
    return _elementwise(network, half_x, one_plus, trt.ElementWiseOperation.PROD)


def relu(network, tensor):
    return network.add_activation(tensor, trt.ActivationType.RELU).get_output(0)


def multiply_last_dim(network, tensor, scale: np.ndarray, dtype: np.dtype):
    rank = len(tuple(tensor.shape))
    shape = (1,) * (rank - 1) + (int(scale.shape[0]),)
    scale_tensor = cast(network, constant(network, scale, shape, dtype), tensor.dtype)
    return _elementwise(network, tensor, scale_tensor, trt.ElementWiseOperation.PROD)


def add_scaled_residual(network, residual, tensor, scale: np.ndarray, dtype: np.dtype):
    tensor = multiply_last_dim(network, tensor, scale, dtype)
    return _elementwise(network, residual, tensor, trt.ElementWiseOperation.SUM)


def attention(network, q, k, v, head_dim: int, dtype: np.dtype):
    scalar = constant(network, np.asarray(1.0 / np.sqrt(float(head_dim))), (1, 1, 1, 1), dtype)
    scalar = cast(network, scalar, q.dtype)
    q = _elementwise(network, q, scalar, trt.ElementWiseOperation.PROD)
    scores = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    probs_layer = network.add_softmax(scores)
    probs_layer.axes = 1 << 3
    return network.add_matrix_multiply(
        probs_layer.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
    ).get_output(0)


def conv2d(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int = 1,
    padding: int = 0,
    dtype: np.dtype,
):
    """A standard [out, in, kh, kw] convolution, bias optional."""
    out_channels = int(weight.shape[0])
    kernel = (int(weight.shape[2]), int(weight.shape[3]))
    bias_weights = trt.Weights(np.ascontiguousarray(bias, dtype=dtype)) if bias is not None else trt.Weights()
    layer = network.add_convolution_nd(
        tensor,
        out_channels,
        kernel,
        trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias_weights,
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a Depth Anything V2 convolution")
    layer.stride_nd = (stride, stride)
    layer.padding_nd = (padding, padding)
    return layer.get_output(0)


def conv_transpose2d(
    network,
    tensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: int,
    dtype: np.dtype,
):
    """A [in, out, kh, kw] transposed convolution with kernel_size == stride, no padding.

    PyTorch's `ConvTranspose2d` stores weights as `[in_channels, out_channels,
    kh, kw]` - the input/output axes are swapped relative to `Conv2d`.
    """
    out_channels = int(weight.shape[1])
    kernel = (int(weight.shape[2]), int(weight.shape[3]))
    bias_weights = trt.Weights(np.ascontiguousarray(bias, dtype=dtype)) if bias is not None else trt.Weights()
    layer = network.add_deconvolution_nd(
        tensor,
        out_channels,
        kernel,
        trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias_weights,
    )
    if layer is None:
        raise RuntimeError("TensorRT rejected a Depth Anything V2 deconvolution")
    layer.stride_nd = (stride, stride)
    return layer.get_output(0)


def resize_bilinear(network, tensor, output_shape: tuple[int, int, int, int], *, align_corners: bool):
    """Resize the trailing two (H, W) dims of an NCHW tensor."""
    layer = network.add_resize(tensor)
    layer.shape = output_shape
    layer.resize_mode = trt.ResizeMode.LINEAR
    layer.coordinate_transformation = (
        trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        if align_corners
        else trt.ResizeCoordinateTransformation.HALF_PIXEL
    )
    return layer.get_output(0)

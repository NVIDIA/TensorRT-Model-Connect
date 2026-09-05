# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph builders for timm ConvNeXt classifiers.

The ConvNeXt stack is convolutional, so it needs a different op set from the
transformer families: convolution, folded batch norm, ReLU, pooling, and a
final fully-connected head.
"""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt()


def add_conv2d(
    network,
    inp,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    dtype: np.dtype = np.float32,
):
    """2D convolution wrapper.

    Input: [N, C_in, H, W]
    Weight: [C_out, C_in/groups, kH, kW]
    Output: [N, C_out, H', W']

    timm ResNets always fold the convolution bias into the following batch
    norm, so `bias` is None throughout this family; the parameter is kept so
    the signature matches the other families that own this helper.
    """
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = stride
    conv.padding_nd = padding
    conv.num_groups = groups
    return conv.get_output(0)


def add_batch_norm(
    network,
    x,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
    *,
    dtype=np.float32,
):
    """Fold inference-time batch norm into a single per-channel scale+shift.

    y = (x - mean) / sqrt(var + eps) * gamma + beta
      = x * scale + shift
    Folding avoids emitting a normalization layer whose statistics are
    constant at inference time.
    """
    scale = (gamma / np.sqrt(running_var + eps)).astype(np.float32)
    shift = (beta - running_mean * scale).astype(np.float32)
    layer = network.add_scale(
        x,
        trt.ScaleMode.CHANNEL,
        shift=trt.Weights(np.ascontiguousarray(shift, dtype=dtype)),
        scale=trt.Weights(np.ascontiguousarray(scale, dtype=dtype)),
    )
    return layer.get_output(0)


def add_relu(network, x):
    return network.add_activation(x, trt.ActivationType.RELU).get_output(0)


def add_max_pool2d(network, x, kernel: int, stride: int, padding: int):
    pool = network.add_pooling_nd(x, trt.PoolingType.MAX, (kernel, kernel))
    pool.stride_nd = (stride, stride)
    pool.padding_nd = (padding, padding)
    return pool.get_output(0)


def add_global_avg_pool(network, x, spatial: tuple[int, int]):
    pool = network.add_pooling_nd(x, trt.PoolingType.AVERAGE, spatial)
    pool.stride_nd = (1, 1)
    return pool.get_output(0)


def add_sum(network, a, b):
    return network.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)


def add_fc(
    network,
    x,
    in_features: int,
    out_features: int,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    dtype=np.float32,
):
    """Final classifier: flatten the pooled feature map, then y = x @ W^T + b."""
    flat = network.add_shuffle(x)
    flat.reshape_dims = (1, in_features)
    flat_out = flat.get_output(0)

    # timm stores fc.weight as (out, in); TensorRT wants the (in, out) operand.
    w = np.ascontiguousarray(weight.T, dtype=dtype)
    w_const = network.add_constant((in_features, out_features), trt.Weights(w)).get_output(0)
    mm = network.add_matrix_multiply(
        flat_out,
        trt.MatrixOperation.NONE,
        w_const,
        trt.MatrixOperation.NONE,
    ).get_output(0)

    b = np.ascontiguousarray(bias.reshape(1, out_features), dtype=dtype)
    b_const = network.add_constant((1, out_features), trt.Weights(b)).get_output(0)
    return network.add_elementwise(mm, b_const, trt.ElementWiseOperation.SUM).get_output(0)


def add_layer_norm_channels(network, x, channels: int, gamma, beta, eps: float,
                            dtype: np.dtype = np.float32):
    """LayerNorm over the channel axis of an NCHW tensor.

    ConvNeXt normalises across channels while keeping the spatial layout, which
    is the transpose of the usual last-axis LayerNorm. Normalizing the wrong
    axis here still type-checks and still produces a tensor of the right shape.
    """
    param_shape = (1, channels, 1, 1)
    gamma_t = network.add_constant(
        param_shape, trt.Weights(np.ascontiguousarray(
            np.asarray(gamma).reshape(param_shape), dtype=dtype))).get_output(0)
    beta_t = network.add_constant(
        param_shape, trt.Weights(np.ascontiguousarray(
            np.asarray(beta).reshape(param_shape), dtype=dtype))).get_output(0)
    if gamma_t.dtype != x.dtype:
        gamma_t = network.add_cast(gamma_t, x.dtype).get_output(0)
        beta_t = network.add_cast(beta_t, x.dtype).get_output(0)
    norm = network.add_normalization_v2(x, gamma_t, beta_t, 1 << 1)
    norm.epsilon = eps
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def add_gelu_erf(network, x, dtype: np.dtype = np.float32):
    """GELU (exact, erf-based): 0.5 * x * (1 + erf(x / sqrt(2)))."""
    target_dtype = x.dtype
    shape = (1,) * max(1, len(tuple(x.shape)))

    def _const(value):
        c = network.add_constant(
            shape, trt.Weights(np.array([value], dtype=dtype).reshape(shape))).get_output(0)
        return c if c.dtype == target_dtype else network.add_cast(c, target_dtype).get_output(0)

    scaled = network.add_elementwise(
        x, _const(1.0 / np.sqrt(2.0)), trt.ElementWiseOperation.PROD).get_output(0)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    shifted = network.add_elementwise(
        erf, _const(1.0), trt.ElementWiseOperation.SUM).get_output(0)
    half = network.add_elementwise(
        x, _const(0.5), trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(
        half, shifted, trt.ElementWiseOperation.PROD).get_output(0)


def add_channel_scale(network, x, channels: int, scale, dtype: np.dtype = np.float32):
    """Multiply each channel by its own constant."""
    shape = (1, channels, 1, 1)
    scale_t = network.add_constant(
        shape, trt.Weights(np.ascontiguousarray(
            np.asarray(scale).reshape(shape), dtype=dtype))).get_output(0)
    if scale_t.dtype != x.dtype:
        scale_t = network.add_cast(scale_t, x.dtype).get_output(0)
    return network.add_elementwise(
        x, scale_t, trt.ElementWiseOperation.PROD).get_output(0)


def add_mean_spatial(network, x):
    """Mean over the two spatial axes of an NCHW tensor, keeping them as 1."""
    return network.add_reduce(
        x, trt.ReduceOperation.AVG, (1 << 2) | (1 << 3), True).get_output(0)

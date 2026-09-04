# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph builders for timm MobileNetV2 classifiers.

MobileNetV2 is an inverted-residual stack with no squeeze-excite gate. Its one
addition to the plain convolutional op set is ReLU6, the clipped activation the
architecture uses throughout.
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


def add_relu6(network, x):
    """ReLU6: clamp(x, 0, 6), the activation MobileNetV2 uses throughout."""
    layer = network.add_activation(x, trt.ActivationType.CLIP)
    layer.alpha = 0.0
    layer.beta = 6.0
    return layer.get_output(0)

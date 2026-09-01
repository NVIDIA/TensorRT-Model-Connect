# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph builders for timm ResNet classifiers.

The ResNet stack is convolutional, so it needs a different op set from the
transformer families: convolution, folded batch norm, ReLU, pooling, and a
final fully-connected head.
"""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt()


def add_conv2d(
    network,
    x,
    out_channels: int,
    kernel: tuple[int, int],
    weight: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype=np.float32,
):
    """Bias-free convolution; timm ResNets always fold bias into the BN."""
    conv = network.add_convolution_nd(
        x,
        num_output_maps=out_channels,
        kernel_shape=kernel,
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        # A default-constructed Weights is TensorRT's "no bias": a zero-count
        # array with a non-null pointer fails its parameter check.
        bias=trt.Weights(),
    )
    conv.stride_nd = (stride, stride)
    conv.padding_nd = (padding, padding)
    if groups != 1:
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

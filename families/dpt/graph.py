# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the DPT-Hybrid BiT backbone, ViT encoder and DPT neck."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import as_engine_weights


def retype(network, tensor, target):
    if tensor.dtype == target:
        return tensor
    return network.add_cast(tensor, target).get_output(0)


def add_constant(network, shape, values, dtype=np.float32):
    weights = trt.Weights(as_engine_weights(np.asarray(values).reshape(shape), dtype))
    return network.add_constant(tuple(shape), weights).get_output(0)


def add_conv2d(network, inp, weight, bias, *, stride=(1, 1), pre_padding=(0, 0),
               post_padding=(0, 0), dtype=np.float32):
    """2-D convolution with explicit (possibly asymmetric) zero padding.

    ``weight`` is [out, in, kh, kw]; ``bias`` may be None.
    """
    weight = np.asarray(weight)
    out_channels, _, kernel_h, kernel_w = (int(v) for v in weight.shape)
    conv = network.add_convolution_nd(
        inp, out_channels, (kernel_h, kernel_w),
        trt.Weights(as_engine_weights(weight, dtype)),
        trt.Weights(as_engine_weights(bias, dtype)) if bias is not None
        else trt.Weights(),
    )
    conv.stride_nd = tuple(stride)
    conv.pre_padding = tuple(pre_padding)
    conv.post_padding = tuple(post_padding)
    return conv.get_output(0)


def add_zero_pad(network, inp, pre_padding, post_padding):
    """Pad the spatial axes of [N, C, H, W] with zeros (``F.pad(..., value=0)``)."""
    shape = tuple(int(v) for v in inp.shape)
    if not any(pre_padding) and not any(post_padding):
        return inp
    padded = network.add_slice(
        inp,
        (0, 0, -int(pre_padding[0]), -int(pre_padding[1])),
        (shape[0], shape[1],
         shape[2] + int(pre_padding[0]) + int(post_padding[0]),
         shape[3] + int(pre_padding[1]) + int(post_padding[1])),
        (1, 1, 1, 1),
    )
    padded.mode = trt.SampleMode.FILL
    return padded.get_output(0)


def add_max_pool(network, inp, kernel, stride):
    pool = network.add_pooling_nd(inp, trt.PoolingType.MAX, tuple(kernel))
    pool.stride_nd = tuple(stride)
    return pool.get_output(0)


def add_group_norm(network, inp, gamma, beta, groups: int, eps: float, dtype=np.float32):
    """GroupNorm over [N, C, H, W]: normalize within channel groups, then scale and shift."""
    shape = tuple(int(v) for v in inp.shape)
    channels = shape[1]
    if channels % groups:
        raise ValueError(f"group norm needs {groups} to divide {channels} channels")
    grouped = network.add_shuffle(inp)
    grouped.reshape_dims = (shape[0], groups, channels // groups, shape[2], shape[3])
    # Statistics over the channels of one group and the whole map.
    axes = (1 << 2) | (1 << 3) | (1 << 4)
    unit_shape = (1, groups, 1, 1, 1)
    ones = add_constant(network, unit_shape, np.ones(unit_shape), dtype=dtype)
    zeros = add_constant(network, unit_shape, np.zeros(unit_shape), dtype=dtype)
    layer = network.add_normalization(
        grouped.get_output(0), retype(network, ones, inp.dtype),
        retype(network, zeros, inp.dtype), axes)
    layer.epsilon = float(eps)
    restored = network.add_shuffle(layer.get_output(0))
    restored.reshape_dims = shape
    normalized = restored.get_output(0)

    # The affine term is per channel, not per group.
    affine_shape = (1, channels, 1, 1)
    scale = retype(network, add_constant(network, affine_shape, gamma, dtype=dtype), inp.dtype)
    shift = retype(network, add_constant(network, affine_shape, beta, dtype=dtype), inp.dtype)
    scaled = network.add_elementwise(
        normalized, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(scaled, shift, trt.ElementWiseOperation.SUM).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps, dtype=np.float32):
    """Normalize over the last axis, then scale and shift."""
    rank = len(tuple(inp.shape))
    width = int(np.asarray(gamma).size)
    shape = (1,) * (rank - 1) + (width,)
    scale = retype(network, add_constant(network, shape, gamma, dtype=dtype), inp.dtype)
    shift = retype(network, add_constant(network, shape, beta, dtype=dtype), inp.dtype)
    layer = network.add_normalization(inp, scale, shift, 1 << (rank - 1))
    layer.epsilon = float(eps)
    return layer.get_output(0)


def add_linear(network, inp, weight, bias, dtype=np.float32):
    """Linear over the last axis. ``weight`` is [out_features, in_features]."""
    weight = np.asarray(weight)
    rank = len(tuple(inp.shape))
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
    rhs = retype(network, add_constant(network, rhs_shape, weight.T, dtype=dtype), inp.dtype)
    product = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE).get_output(0)
    if bias is None:
        return product
    bias_shape = (1,) * (rank - 1) + (out_features,)
    bias_t = retype(network, add_constant(network, bias_shape, bias, dtype=dtype), product.dtype)
    return network.add_elementwise(product, bias_t, trt.ElementWiseOperation.SUM).get_output(0)


def add_relu(network, inp):
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_gelu(network, inp):
    """The exact erf GELU that transformers' "gelu" activation uses."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(
        left, retype(network, right, left.dtype), trt.ElementWiseOperation.SUM).get_output(0)


def split_heads(network, x, tokens, num_heads, head_dim):
    """[1, tokens, hidden] -> [1, heads, tokens, head_dim]."""
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (1, tokens, num_heads, head_dim)
    shuffle.second_transpose = (0, 2, 1, 3)
    return shuffle.get_output(0)


def merge_heads(network, x, tokens, hidden):
    """[1, heads, tokens, head_dim] -> [1, tokens, hidden]."""
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = (0, 2, 1, 3)
    shuffle.reshape_dims = (1, tokens, hidden)
    return shuffle.get_output(0)


def add_attention(network, query, key, value, scale):
    """softmax(Q K^T * scale) V over the token axis of a [1, H, T, D] layout."""
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale_t = retype(
        network, add_constant(network, (1, 1, 1, 1), np.array([scale], dtype=np.float32)),
        scores.dtype)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    return network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)


def map_to_tokens(network, x):
    """[1, C, H, W] -> [1, H * W, C], row-major over the map (``flatten(2).transpose(1, 2)``)."""
    shape = tuple(int(v) for v in x.shape)
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (shape[0], shape[1], shape[2] * shape[3])
    shuffle.second_transpose = (0, 2, 1)
    return shuffle.get_output(0)


def tokens_to_map(network, x, height, width):
    """[1, H * W, C] -> [1, C, H, W]."""
    channels = int(tuple(x.shape)[-1])
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (1, height, width, channels)
    shuffle.second_transpose = (0, 3, 1, 2)
    return shuffle.get_output(0)


def add_resize_to(network, inp, target, align_corners: bool):
    """Bilinear resize of [N, C, H, W] onto (height, width).

    ``align_corners`` selects PyTorch's two conventions, which TensorRT spells as
    different coordinate transformations.
    """
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = (
        trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        if align_corners
        else trt.ResizeCoordinateTransformation.HALF_PIXEL
    )
    shape = tuple(int(v) for v in inp.shape)
    resize.shape = (shape[0], shape[1], int(target[0]), int(target[1]))
    return resize.get_output(0)

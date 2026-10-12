# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Stable Diffusion UNet."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def _retype(network, tensor, target):
    if tensor.dtype == target:
        return tensor
    cast = network.add_cast(tensor, target)
    return cast.get_output(0)


def add_constant(network, shape, values, dtype=np.float32):
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(tuple(shape), weights).get_output(0)


def add_linear(network, inp, weight, bias, dtype=np.float32):
    """Linear over the last axis. ``weight`` is [out_features, in_features]."""
    weight = np.asarray(weight)
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    rank = len(tuple(inp.shape))
    rhs_shape = (1,) * (rank - 2) + (in_features, out_features)
    rhs = add_constant(network, rhs_shape, weight.T.reshape(rhs_shape), dtype=dtype)
    product = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, _retype(network, rhs, inp.dtype), trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is None:
        return product
    bias_shape = (1,) * (rank - 1) + (out_features,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    summed = network.add_elementwise(
        product, _retype(network, bias_t, product.dtype), trt.ElementWiseOperation.SUM
    )
    return summed.get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps, dtype=np.float32):
    """Normalize over the last axis, then scale and shift."""
    rank = len(tuple(inp.shape))
    axes = 1 << (rank - 1)
    width = int(np.asarray(gamma).size)
    shape = (1,) * (rank - 1) + (width,)
    scale = _retype(
        network, add_constant(network, shape, np.asarray(gamma).reshape(shape), dtype=dtype),
        inp.dtype,
    )
    shift = _retype(
        network, add_constant(network, shape, np.asarray(beta).reshape(shape), dtype=dtype),
        inp.dtype,
    )
    layer = network.add_normalization(inp, scale, shift, axes)
    layer.epsilon = float(eps)
    return layer.get_output(0)


def add_gelu(network, inp):
    """The exact erf GELU that transformers' "gelu" activation uses."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_relu(network, inp):
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_sigmoid(network, inp):
    return network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(
        left, _retype(network, right, left.dtype), trt.ElementWiseOperation.SUM
    ).get_output(0)


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
    """Softmax(QK^T * scale) V over the token axis of a 4-D [1,H,T,D] layout."""
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scale_t = _retype(
        network, add_constant(network, (1, 1, 1, 1), np.array([scale], dtype=np.float32)),
        scores.dtype,
    )
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD
    ).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    return network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)


def add_conv2d(network, inp, weight, bias, stride=(1, 1), padding=(0, 0), dtype=np.float32):
    """2-D convolution. ``weight`` is [out, in, kh, kw]."""
    weight = np.asarray(weight)
    out_channels, _, kernel_h, kernel_w = (int(v) for v in weight.shape)
    conv = network.add_convolution_nd(
        inp, out_channels, (kernel_h, kernel_w),
        trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        trt.Weights(np.ascontiguousarray(bias, dtype=dtype)) if bias is not None
        else trt.Weights(),
    )
    conv.stride_nd = stride
    conv.padding_nd = padding
    return conv.get_output(0)


def add_deconv2d(network, inp, weight, bias, stride, dtype=np.float32):
    """Transposed convolution, used by the DPT reassemble stage to upsample."""
    weight = np.asarray(weight)
    in_channels, out_channels, kernel_h, kernel_w = (int(v) for v in weight.shape)
    deconv = network.add_deconvolution_nd(
        inp, out_channels, (kernel_h, kernel_w),
        trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        trt.Weights(np.ascontiguousarray(bias, dtype=dtype)) if bias is not None
        else trt.Weights(),
    )
    deconv.stride_nd = stride
    return deconv.get_output(0)


def add_resize_bilinear(network, inp, scale_h, scale_w):
    """Bilinear upsample over the spatial axes of a [N, C, H, W] tensor."""
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    resize.scales = [1.0, 1.0, float(scale_h), float(scale_w)]
    return resize.get_output(0)


def tokens_to_feature_map(network, x, height, width, channels):
    """[1, tokens, C] with the class token already dropped -> [1, C, H, W]."""
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (1, height, width, channels)
    shuffle.second_transpose = (0, 3, 1, 2)
    return shuffle.get_output(0)


def add_resize_to(network, inp, target, align_corners: bool):
    """Bilinear resize of a [N, C, H, W] tensor onto an explicit (height, width).

    DPT mixes the two conventions: the fusion and head upsamples use
    align_corners=True while the residual interpolation uses False, and they map
    to different TensorRT coordinate transformations.
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


def add_group_norm(network, inp, gamma, beta, groups: int, eps: float, dtype=np.float32):
    """GroupNorm over [N, C, H, W]: normalize within channel groups, then scale and shift."""
    shape = tuple(int(v) for v in inp.shape)
    channels = shape[1]
    if channels % groups:
        raise ValueError(f"group norm needs {groups} to divide {channels} channels")
    grouped = network.add_shuffle(inp)
    grouped.reshape_dims = (shape[0], groups, channels // groups, shape[2], shape[3])
    # Normalize over the channel-in-group, height and width axes.
    axes = (1 << 2) | (1 << 3) | (1 << 4)
    ones = add_constant(network, (1, groups, 1, 1, 1), np.ones((1, groups, 1, 1, 1)), dtype=dtype)
    zeros = add_constant(network, (1, groups, 1, 1, 1), np.zeros((1, groups, 1, 1, 1)), dtype=dtype)
    layer = network.add_normalization(
        grouped.get_output(0), _retype(network, ones, inp.dtype),
        _retype(network, zeros, inp.dtype), axes)
    layer.epsilon = float(eps)
    restored = network.add_shuffle(layer.get_output(0))
    restored.reshape_dims = shape
    scaled = restored.get_output(0)

    weight_shape = (1, channels, 1, 1)
    scale = _retype(network, add_constant(
        network, weight_shape, np.asarray(gamma).reshape(weight_shape), dtype=dtype), scaled.dtype)
    shift = _retype(network, add_constant(
        network, weight_shape, np.asarray(beta).reshape(weight_shape), dtype=dtype), scaled.dtype)
    scaled = network.add_elementwise(
        scaled, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(
        scaled, shift, trt.ElementWiseOperation.SUM).get_output(0)


def add_silu(network, inp):
    """x * sigmoid(x)."""
    gate = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(inp, gate, trt.ElementWiseOperation.PROD).get_output(0)


def add_geglu(network, inp, weight, bias, dtype=np.float32):
    """The feed-forward gate used by SD: project to 2x width, split value and gate."""
    projected = add_linear(network, inp, weight, bias, dtype=dtype)
    width = int(np.asarray(weight).shape[0]) // 2
    rank = len(tuple(projected.shape))
    start = (0,) * rank
    size = tuple(int(v) for v in projected.shape[:-1]) + (width,)
    stride = (1,) * rank
    value = network.add_slice(projected, start, size, stride).get_output(0)
    gate_start = (0,) * (rank - 1) + (width,)
    gate = network.add_slice(projected, gate_start, size, stride).get_output(0)
    return network.add_elementwise(
        value, add_gelu(network, gate), trt.ElementWiseOperation.PROD).get_output(0)


def concat(network, tensors, axis: int):
    layer = network.add_concatenation(list(tensors))
    layer.axis = axis
    return layer.get_output(0)


def spatial_to_tokens(network, x):
    """[1, C, H, W] -> [1, H*W, C] for the transformer blocks."""
    shape = tuple(int(v) for v in x.shape)
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = (0, 2, 3, 1)
    shuffle.reshape_dims = (shape[0], shape[2] * shape[3], shape[1])
    return shuffle.get_output(0)


def tokens_to_spatial(network, x, height: int, width: int):
    """[1, H*W, C] -> [1, C, H, W]."""
    shape = tuple(int(v) for v in x.shape)
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (shape[0], height, width, shape[2])
    shuffle.second_transpose = (0, 3, 1, 2)
    return shuffle.get_output(0)


def add_resize_nearest(network, inp, target):
    """Nearest-neighbour resize, which is what diffusers' Upsample2D uses."""
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    shape = tuple(int(v) for v in inp.shape)
    resize.shape = (shape[0], shape[1], int(target[0]), int(target[1]))
    return resize.get_output(0)


def add_quick_gelu(network, inp):
    """CLIP's activation: x * sigmoid(1.702 * x). Not the erf GELU."""
    scale = _retype(
        network, add_constant(network, (1,) * len(tuple(inp.shape)),
                              np.array([1.702], dtype=np.float32)), inp.dtype)
    scaled = network.add_elementwise(inp, scale, trt.ElementWiseOperation.PROD).get_output(0)
    gate = network.add_activation(scaled, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(inp, gate, trt.ElementWiseOperation.PROD).get_output(0)


def add_attention_causal(network, query, key, value, scale, tokens: int, dtype=np.float32):
    """Softmax(QK^T * scale + causal_mask) V over a 4-D [1, H, T, D] layout."""
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scale_t = _retype(
        network, add_constant(network, (1, 1, 1, 1), np.array([scale], dtype=np.float32)),
        scores.dtype)
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    # CLIP's text encoder is causal: a token may not attend to later tokens.
    mask = np.triu(np.full((tokens, tokens), -1e4, dtype=np.float32), k=1).reshape(1, 1, tokens, tokens)
    mask_t = _retype(network, add_constant(network, (1, 1, tokens, tokens), mask, dtype=dtype),
                     scores.dtype)
    scores = network.add_elementwise(
        scores, mask_t, trt.ElementWiseOperation.SUM).get_output(0)

    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    return network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
    ).get_output(0)


def add_gather(network, table, indices, axis: int = 0):
    return network.add_gather(table, indices, axis).get_output(0)

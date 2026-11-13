# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for Deformable DETR."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def shape_of(network, tensor, indices=None):
    shape = network.add_shape(tensor).get_output(0)
    shape = network.add_cast(shape, trt.int32).get_output(0)
    return shape if indices is None else shape_indices(network, shape, indices)


def shape_indices(network, shape, indices):
    index = add_constant(network, (len(indices),), indices, np.int32)
    return network.add_gather(shape, index, 0).get_output(0)


def shape_vector(network, dimensions):
    pieces = [add_constant(network, (1,), [value], np.int32)
              if isinstance(value, (int, np.integer)) else value for value in dimensions]
    return pieces[0] if len(pieces) == 1 else concat(network, pieces, axis=0)


def reshape(network, tensor, dimensions):
    layer = network.add_shuffle(tensor)
    if all(isinstance(value, (int, np.integer)) for value in dimensions):
        layer.reshape_dims = tuple(int(value) for value in dimensions)
    else:
        layer.set_input(1, shape_vector(network, dimensions))
    return layer.get_output(0)


def coordinate_grid(network, spatial_shape, axis, *, normalise=False):
    """Half-pixel centres for an actual, unpadded feature map."""
    layer = network.add_fill((1, 1), trt.FillOperation.LINSPACE)
    layer.set_input(0, spatial_shape)
    layer.set_input(1, add_constant(network, (), [0.5], np.float32))
    layer.set_input(2, add_constant(network, (2,), [axis == 0, axis == 1], np.float32))
    grid = layer.get_output(0)
    if normalise:
        extent = shape_indices(network, spatial_shape, [axis])
        extent = network.add_cast(extent, trt.float32).get_output(0)
        extent = reshape(network, extent, (1, 1))
        grid = network.add_elementwise(grid, extent, trt.ElementWiseOperation.DIV).get_output(0)
    return grid


def fold_batch_norm(weight, gamma, beta, mean, variance, eps):
    """Fold a BatchNorm into the convolution that feeds it.

    The checkpoint stores running statistics rather than folded weights, and
    every convolution in this backbone is bias-free, so the folded bias is the
    only bias the graph ever sees.
    """
    scale = np.asarray(gamma, dtype=np.float32) / np.sqrt(
        np.asarray(variance, dtype=np.float32) + eps)
    folded = np.asarray(weight, dtype=np.float32) * scale.reshape(-1, 1, 1, 1)
    bias = np.asarray(beta, dtype=np.float32) - np.asarray(mean, dtype=np.float32) * scale
    return np.ascontiguousarray(folded), np.ascontiguousarray(bias)


def add_conv2d(network, inp, weight, bias=None, stride=(1, 1), padding=(0, 0), dtype=np.float32):
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    out_channels, _, kh, kw = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (int(kh), int(kw)), trt.Weights(weight),
        trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=dtype)))
        if bias is not None else trt.Weights())
    layer.stride_nd = tuple(int(v) for v in stride)
    layer.padding_nd = tuple(int(v) for v in padding)
    return layer.get_output(0)


def add_relu(network, inp):
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_max_pool(network, inp, window, stride, padding):
    layer = network.add_pooling_nd(inp, trt.PoolingType.MAX, tuple(int(v) for v in window))
    layer.stride_nd = tuple(int(v) for v in stride)
    layer.padding_nd = tuple(int(v) for v in padding)
    return layer.get_output(0)


def add_avg_pool(network, inp, window, stride, padding=(0, 0)):
    layer = network.add_pooling_nd(inp, trt.PoolingType.AVERAGE, tuple(int(v) for v in window))
    layer.stride_nd = tuple(int(v) for v in stride)
    layer.padding_nd = tuple(int(v) for v in padding)
    layer.average_count_excludes_padding = True
    return layer.get_output(0)


def add_silu(network, inp):
    sig = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(inp, sig, trt.ElementWiseOperation.PROD).get_output(0)


def add_gelu(network, inp):
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def concat(network, tensors, axis):
    layer = network.add_concatenation(tensors)
    layer.axis = axis
    return layer.get_output(0)


def add_resize_nearest(network, inp, size):
    layer = network.add_resize(inp)
    shape = tuple(int(v) for v in inp.shape)
    layer.shape = (shape[0], shape[1], int(size[0]), int(size[1]))
    layer.resize_mode = trt.InterpolationMode.NEAREST
    return layer.get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps, dtype=np.float32):
    axis = len(inp.shape) - 1
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(inp, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    width = int(inp.shape[-1])
    epsilon = add_constant(network, tuple([1] * len(inp.shape)), np.full((1,), eps), dtype=dtype)
    shifted = network.add_elementwise(variance, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(
        centred, deviation, trt.ElementWiseOperation.DIV).get_output(0)
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = network.add_elementwise(
        normed, add_constant(network, shape, gamma, dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(
        scaled, add_constant(network, shape, beta, dtype=dtype),
        trt.ElementWiseOperation.SUM).get_output(0)


def add_linear(network, inp, weight, bias=None, dtype=np.float32):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout."""
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    # TensorRT's matrix multiply will not broadcast across differing ranks, so
    # the weight is reshaped to the input's rank with leading singleton axes.
    rank = len(inp.shape)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    constant = add_constant(network, shape, weight.reshape(shape), dtype=dtype)
    out = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias_shape = (1,) * (rank - 1) + (-1,)
    bias = np.asarray(bias, dtype=dtype).reshape(bias_shape)
    return network.add_elementwise(
        out, add_constant(network, bias.shape, bias, dtype=dtype),
        trt.ElementWiseOperation.SUM).get_output(0)


def spatial_to_tokens(network, x):
    """[1, C, H, W] -> [1, H*W, C]"""
    shape = tuple(int(v) for v in x.shape)
    flat = network.add_shuffle(x)
    flat.reshape_dims = (shape[0], shape[1], -1)
    flat.second_transpose = (0, 2, 1)
    return flat.get_output(0)


def tokens_to_spatial(network, x, height, width):
    """[1, H*W, C] -> [1, C, H, W]"""
    shape = tuple(int(v) for v in x.shape)
    back = network.add_shuffle(x)
    back.first_transpose = (0, 2, 1)
    back.reshape_dims = (shape[0], shape[2], int(height), int(width))
    return back.get_output(0)


def add_group_norm(network, inp, gamma, beta, groups, eps, dtype=np.float32):
    """GroupNorm written as an explicit reduction.

    TensorRT ships INormalizationLayer, but it rejected every axis/scale
    combination tried here with mutually contradictory messages, so the
    statistics are reduced by hand. The shapes are static per feature level,
    which keeps this straightforward.
    """
    shape = tuple(inp.shape)
    channels = shape[1]
    if channels % groups:
        raise ValueError("group_norm channels must divide into the group count")

    grouped = network.add_shuffle(inp)
    grouped.reshape_dims = (shape[0], groups, -1)
    x = grouped.get_output(0)

    # Reduce over everything inside a group: axis 2 of (N, groups, rest).
    axis = 1 << 2
    mean = network.add_reduce(x, trt.ReduceOperation.AVG, axis, keep_dims=True).get_output(0)
    centred = network.add_elementwise(x, mean, trt.ElementWiseOperation.SUB).get_output(0)
    squared = network.add_elementwise(
        centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(
        squared, trt.ReduceOperation.AVG, axis, keep_dims=True).get_output(0)
    epsilon = add_constant(network, (1, 1, 1), np.array([eps], dtype=dtype), dtype)
    variance = network.add_elementwise(
        variance, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(
        centred, deviation, trt.ElementWiseOperation.DIV).get_output(0)

    restored = network.add_shuffle(normed)
    restored.set_input(1, shape_of(network, inp))
    y = restored.get_output(0)

    # The affine term is per channel, not per group.
    scale = add_constant(network, (1, channels, 1, 1),
                         np.asarray(gamma, dtype=dtype).reshape(1, channels, 1, 1), dtype)
    shift = add_constant(network, (1, channels, 1, 1),
                         np.asarray(beta, dtype=dtype).reshape(1, channels, 1, 1), dtype)
    y = network.add_elementwise(y, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(y, shift, trt.ElementWiseOperation.SUM).get_output(0)

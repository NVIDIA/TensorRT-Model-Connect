# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for BiRefNet."""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


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
    flat.reshape_dims = (shape[0], shape[1], shape[2] * shape[3])
    flat.second_transpose = (0, 2, 1)
    return flat.get_output(0)


def tokens_to_spatial(network, x, height, width):
    """[1, H*W, C] -> [1, C, H, W]"""
    shape = tuple(int(v) for v in x.shape)
    back = network.add_shuffle(x)
    back.first_transpose = (0, 2, 1)
    back.reshape_dims = (shape[0], shape[2], int(height), int(width))
    return back.get_output(0)


def pad_hwc(network, x, height, width, dtype=np.float32):
    """Zero-pad an NHWC map's bottom and right edges to (height, width).

    Swin pads the feature map up to a whole number of windows and crops back
    afterwards, so every stage resolution here (256, 128, 64, 32 at a window of
    7) needs it.
    """
    shape = tuple(int(v) for v in x.shape)
    if shape[1] == height and shape[2] == width:
        return x
    layer = network.add_slice(x, (0, 0, 0, 0), (shape[0], height, width, shape[3]),
                              (1, 1, 1, 1))
    layer.mode = trt.SampleMode.FILL
    layer.set_input(4, add_constant(network, (1,), np.zeros((1,)), dtype=dtype))
    return layer.get_output(0)


def crop_hwc(network, x, height, width):
    """Undo pad_hwc."""
    shape = tuple(int(v) for v in x.shape)
    if shape[1] == height and shape[2] == width:
        return x
    return network.add_slice(x, (0, 0, 0, 0), (shape[0], height, width, shape[3]),
                             (1, 1, 1, 1)).get_output(0)


def roll_hwc(network, x, shift_h, shift_w):
    """Cyclic shift over the two spatial axes of an NHWC map."""
    if shift_h == 0 and shift_w == 0:
        return x
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[1], shape[2]
    # torch.roll(x, shifts=s) gives out[i] = x[i - s], so the split lands at
    # (-s) % n. Using s % n rolls the other way, which still builds and only
    # shows up once a block actually shifts.
    sh = (-shift_h) % height
    sw = (-shift_w) % width

    def split(tensor, axis, size):
        full = tuple(int(v) for v in tensor.shape)
        first_shape = list(full)
        first_shape[axis] = size
        start = [0, 0, 0, 0]
        first = network.add_slice(tensor, tuple(start), tuple(first_shape),
                                  (1, 1, 1, 1)).get_output(0)
        start[axis] = size
        rest_shape = list(full)
        rest_shape[axis] = full[axis] - size
        rest = network.add_slice(tensor, tuple(start), tuple(rest_shape),
                                 (1, 1, 1, 1)).get_output(0)
        return first, rest

    if sh:
        top, bottom = split(x, 1, sh)
        x = concat(network, [bottom, top], axis=1)
    if sw:
        left, right = split(x, 2, sw)
        x = concat(network, [right, left], axis=2)
    return x


def reshape_permute(network, x, shape, permutation, final_shape):
    """Reshape, transpose, reshape - the window partition and merge pattern."""
    first = network.add_shuffle(x)
    first.reshape_dims = tuple(int(v) for v in shape)
    second = network.add_shuffle(first.get_output(0))
    second.first_transpose = tuple(int(v) for v in permutation)
    second.reshape_dims = tuple(int(v) for v in final_shape)
    return second.get_output(0)


def add_resize_bilinear(network, x, size, align_corners=False):
    """Bilinear resize of an NCHW map.

    BiRefNet's dual-scale path uses ``align_corners=True`` on both the
    downscale and the upscale, which is not TensorRT's default coordinate
    transform. Leaving the default in place shifts every sampled position by
    half a pixel and the error is small enough to look like noise.
    """
    shape = tuple(int(v) for v in x.shape)
    layer = network.add_resize(x)
    layer.shape = (shape[0], shape[1], int(size[0]), int(size[1]))
    layer.resize_mode = trt.InterpolationMode.LINEAR
    layer.coordinate_transformation = (
        trt.ResizeCoordinateTransformation.ALIGN_CORNERS if align_corners
        else trt.ResizeCoordinateTransformation.HALF_PIXEL)
    return layer.get_output(0)

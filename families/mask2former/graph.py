# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph vocabulary owned by the Mask2Former family.

The network is strongly typed and every activation is float32; shapes are static, so every helper works on
concrete dimensions.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def _check(layer, what: str):
    if layer is None:
        raise RuntimeError(f"TensorRT rejected the Mask2Former {what}")
    return layer


def shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def constant(network, values, dtype=np.float32):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    layer = _check(network.add_constant(array.shape, trt.Weights(array)), "constant")
    return layer.get_output(0)


def scalar_like(network, value: float, rank: int):
    return constant(network, np.full((1,) * rank, value, dtype=np.float32))


def elementwise(network, left, right, operation):
    return _check(
        network.add_elementwise(left, right, operation), "elementwise operation"
    ).get_output(0)


def add(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.SUM)


def sub(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.SUB)


def mul(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.PROD)


def div(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.DIV)


def activation(network, tensor, kind):
    return _check(network.add_activation(tensor, kind), "activation").get_output(0)


def relu(network, tensor):
    return activation(network, tensor, trt.ActivationType.RELU)


def sigmoid(network, tensor):
    return activation(network, tensor, trt.ActivationType.SIGMOID)


def gelu(network, tensor):
    """Exact GELU, as torch writes it: 0.5 * x * (1 + erf(x / sqrt(2)))."""

    rank = len(shape(tensor))
    scaled = mul(network, tensor, scalar_like(network, float(1.0 / np.sqrt(2.0)), rank))
    error = _check(network.add_unary(scaled, trt.UnaryOperation.ERF), "GELU erf").get_output(0)
    shifted = add(network, error, scalar_like(network, 1.0, rank))
    half = mul(network, tensor, scalar_like(network, 0.5, rank))
    return mul(network, half, shifted)


def shuffle(network, tensor, *, reshape=None, transpose=None, first_transpose=None):
    layer = _check(network.add_shuffle(tensor), "shuffle")
    if first_transpose is not None:
        layer.first_transpose = trt.Permutation(first_transpose)
    if reshape is not None:
        layer.reshape_dims = trt.Dims(tuple(int(value) for value in reshape))
    if transpose is not None:
        layer.second_transpose = trt.Permutation(transpose)
    return layer.get_output(0)


def reshape(network, tensor, dims):
    return shuffle(network, tensor, reshape=dims)


def transpose(network, tensor, permutation):
    return shuffle(network, tensor, first_transpose=permutation)


def slice_static(network, tensor, start, size, stride=None):
    stride = stride or (1,) * len(start)
    layer = _check(
        network.add_slice(
            tensor, trt.Dims(tuple(start)), trt.Dims(tuple(size)), trt.Dims(tuple(stride))
        ),
        "slice",
    )
    return layer.get_output(0)


def concat(network, tensors, axis: int):
    layer = _check(network.add_concatenation(list(tensors)), "concatenation")
    layer.axis = axis
    return layer.get_output(0)


def softmax(network, tensor, axis: int):
    layer = _check(network.add_softmax(tensor), "softmax")
    layer.axes = 1 << axis
    return layer.get_output(0)


def matmul(network, left, right, *, transpose_left=False, transpose_right=False):
    layer = network.add_matrix_multiply(
        left,
        trt.MatrixOperation.TRANSPOSE if transpose_left else trt.MatrixOperation.NONE,
        right,
        trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE,
    )
    return _check(layer, "matrix multiply").get_output(0)


def linear(network, tensor, weight, bias=None):
    """torch.nn.Linear on the last axis: tensor @ weight.T + bias."""

    rank = len(shape(tensor))
    matrix = np.ascontiguousarray(np.asarray(weight, dtype=np.float32).T)
    if shape(tensor)[-1] != matrix.shape[0]:
        raise ValueError(
            f"linear input width {shape(tensor)[-1]} does not match weight {matrix.shape}"
        )
    output = matmul(
        network, tensor, constant(network, matrix.reshape((1,) * (rank - 2) + matrix.shape))
    )
    if bias is None:
        return output
    return add(
        network,
        output,
        constant(network, np.asarray(bias, dtype=np.float32).reshape((1,) * (rank - 1) + (-1,))),
    )


def layer_norm(network, tensor, gamma, beta, eps: float):
    """torch.nn.LayerNorm over the last axis."""

    rank = len(shape(tensor))
    dims = (1,) * (rank - 1) + (shape(tensor)[-1],)
    scale = constant(network, np.asarray(gamma, dtype=np.float32).reshape(dims))
    bias = constant(network, np.asarray(beta, dtype=np.float32).reshape(dims))
    layer = _check(network.add_normalization_v2(tensor, scale, bias, 1 << (rank - 1)), "layer norm")
    layer.epsilon = float(eps)
    return layer.get_output(0)


def group_norm(network, tensor, gamma, beta, groups: int, eps: float):
    """torch.nn.GroupNorm on an NCHW tensor: statistics per group, affine per channel."""

    batch, channels, height, width = shape(tensor)
    grouped = reshape(network, tensor, (batch, groups, (channels // groups) * height * width))
    axis = 1 << 2
    mean = _check(
        network.add_reduce(grouped, trt.ReduceOperation.AVG, axis, True), "group mean"
    ).get_output(0)
    centred = sub(network, grouped, mean)
    variance = _check(
        network.add_reduce(mul(network, centred, centred), trt.ReduceOperation.AVG, axis, True),
        "group variance",
    ).get_output(0)
    deviation = _check(
        network.add_unary(
            add(network, variance, scalar_like(network, eps, 3)), trt.UnaryOperation.SQRT
        ),
        "group deviation",
    ).get_output(0)
    normed = reshape(network, div(network, centred, deviation), (batch, channels, height, width))
    scale = constant(network, np.asarray(gamma, dtype=np.float32).reshape(1, channels, 1, 1))
    bias = constant(network, np.asarray(beta, dtype=np.float32).reshape(1, channels, 1, 1))
    return add(network, mul(network, normed, scale), bias)


def conv2d(network, tensor, weight, bias=None, *, stride=1, padding=0):
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = _check(
        network.add_convolution_nd(
            tensor,
            int(out_channels),
            (int(kernel_h), int(kernel_w)),
            trt.Weights(weight),
            trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=np.float32)))
            if bias is not None
            else trt.Weights(),
        ),
        "convolution",
    )
    layer.stride_nd = (int(stride), int(stride))
    layer.padding_nd = (int(padding), int(padding))
    return layer.get_output(0)


def resize_bilinear(network, tensor, height: int, width: int):
    """torch.nn.functional.interpolate(mode="bilinear", align_corners=False), without antialiasing."""

    dims = shape(tensor)
    layer = _check(network.add_resize(tensor), "resize")
    layer.shape = dims[:-2] + (int(height), int(width))
    layer.resize_mode = trt.InterpolationMode.LINEAR
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    layer.exclude_outside = False
    return layer.get_output(0)


def tokens_from_nchw(network, tensor):
    """(1, C, H, W) -> (1, H*W, C), row-major over H then W."""

    batch, channels, height, width = shape(tensor)
    return shuffle(network, tensor, reshape=(batch, channels, height * width), transpose=(0, 2, 1))


def nchw_from_tokens(network, tensor, height: int, width: int):
    """(1, H*W, C) -> (1, C, H, W)."""

    batch, _, channels = shape(tensor)
    return shuffle(
        network, tensor, first_transpose=(0, 2, 1), reshape=(batch, channels, height, width)
    )


def split_heads(network, tensor, heads: int):
    """(B, N, heads*D) -> (B, heads, N, D)."""

    batch, tokens, width = shape(tensor)
    return shuffle(
        network, tensor, reshape=(batch, tokens, heads, width // heads), transpose=(0, 2, 1, 3)
    )


def merge_heads(network, tensor):
    """(B, heads, N, D) -> (B, N, heads*D)."""

    batch, heads, tokens, width = shape(tensor)
    return shuffle(
        network, tensor, first_transpose=(0, 2, 1, 3), reshape=(batch, tokens, heads * width)
    )


def roll(network, tensor, shifts, axes):
    """torch.roll: a positive shift moves elements towards higher indices."""

    for shift, axis in zip(shifts, axes):
        dims = list(shape(tensor))
        extent = dims[axis]
        offset = (-shift) % extent
        if offset == 0:
            continue
        pieces = []
        for start, size in ((offset, extent - offset), (0, offset)):
            begin, sizes = [0] * len(dims), list(dims)
            begin[axis], sizes[axis] = start, size
            pieces.append(slice_static(network, tensor, begin, sizes))
        tensor = concat(network, pieces, axis)
    return tensor

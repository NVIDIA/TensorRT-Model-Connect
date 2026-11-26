# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT layers of the ViTPose graph, in float32.

The network is strongly typed and every tensor is float32, so TensorRT may not narrow any layer; the builder also
turns TF32 off. The backbone works on rows ``[tokens, width]``; the head works on ``[1, channels, height, width]``.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def constant(network, values) -> trt.ITensor:
    data = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return network.add_constant(data.shape, data).get_output(0)


def _elementwise(network, left, right, operation) -> trt.ITensor:
    return network.add_elementwise(left, right, operation).get_output(0)


def add(network, left, right) -> trt.ITensor:
    return _elementwise(network, left, right, trt.ElementWiseOperation.SUM)


def multiply(network, left, right) -> trt.ITensor:
    return _elementwise(network, left, right, trt.ElementWiseOperation.PROD)


def linear(network, rows, weight, bias) -> trt.ITensor:
    """``rows @ weight.T + bias`` for rows ``[tokens, in]`` and a torch.nn.Linear weight ``[out, in]``."""

    product = network.add_matrix_multiply(
        rows, trt.MatrixOperation.NONE, constant(network, weight), trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    return add(network, product, constant(network, np.asarray(bias).reshape(1, -1)))


def layer_norm(network, rows, weight, bias, eps: float) -> trt.ITensor:
    """torch.nn.LayerNorm over the last axis: the mean, then the mean of the centred squares (two passes)."""

    axis = 1 << (len(rows.shape) - 1)
    mean = network.add_reduce(rows, trt.ReduceOperation.AVG, axis, True).get_output(0)
    centred = _elementwise(network, rows, mean, trt.ElementWiseOperation.SUB)
    variance = network.add_reduce(
        multiply(network, centred, centred), trt.ReduceOperation.AVG, axis, True
    )
    shifted = add(network, variance.get_output(0), constant(network, np.full((1, 1), eps)))
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normalized = _elementwise(network, centred, deviation, trt.ElementWiseOperation.DIV)
    scaled = multiply(network, normalized, constant(network, np.asarray(weight).reshape(1, -1)))
    return add(network, scaled, constant(network, np.asarray(bias).reshape(1, -1)))


def gelu(network, rows) -> trt.ITensor:
    """``hidden_act: gelu``: the exact ``0.5 * x * (1 + erf(x / sqrt(2)))``, not the tanh approximation."""

    scaled = multiply(network, rows, constant(network, np.full((1, 1), 1.0 / np.sqrt(2.0))))
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    gate = add(network, erf, constant(network, np.ones((1, 1))))
    return multiply(network, multiply(network, rows, constant(network, np.full((1, 1), 0.5))), gate)


def attention(network, query, key, value, *, heads: int, head_dim: int) -> trt.ITensor:
    """Bidirectional multi-head attention over rows ``[tokens, width]``.

    The query is scaled by ``1 / sqrt(head_dim)`` before the product, so the scores never hold the unscaled dot
    products; the softmax runs over the keys in float32.
    """

    tokens = int(query.shape[0])

    def to_heads(rows):
        layer = network.add_shuffle(rows)
        layer.reshape_dims = (tokens, heads, head_dim)
        layer.second_transpose = (1, 0, 2)
        return layer.get_output(0)

    scale = constant(network, np.full((1, 1, 1), 1.0 / np.sqrt(head_dim)))
    scores = network.add_matrix_multiply(
        multiply(network, to_heads(query), scale),
        trt.MatrixOperation.NONE,
        to_heads(key),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 2
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE, to_heads(value), trt.MatrixOperation.NONE
    ).get_output(0)
    rows = network.add_shuffle(context)
    rows.first_transpose = (1, 0, 2)
    rows.reshape_dims = (tokens, heads * head_dim)
    return rows.get_output(0)


def conv2d(
    network, image, weight, bias, *, stride: tuple[int, int], padding: tuple[int, int]
) -> trt.ITensor:
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    bias = np.ascontiguousarray(np.asarray(bias, dtype=np.float32))
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = network.add_convolution_nd(
        image,
        int(out_channels),
        (int(kernel_h), int(kernel_w)),
        trt.Weights(weight),
        trt.Weights(bias),
    )
    layer.stride_nd = stride
    layer.padding_nd = padding
    return layer.get_output(0)


def upsample_bilinear(network, image, factor: int) -> trt.ITensor:
    """torch.nn.Upsample(scale_factor, mode="bilinear", align_corners=False): half-pixel centres, edge clamped."""

    layer = network.add_resize(image)
    layer.resize_mode = trt.InterpolationMode.LINEAR
    layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    batch, channels, height, width = (int(value) for value in image.shape)
    layer.shape = (batch, channels, height * factor, width * factor)
    return layer.get_output(0)

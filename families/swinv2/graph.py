# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by the Swin Transformer V2 family.

The network is strongly typed and runs in the checkpoint's float32; ``dtype``
arguments name the NumPy type constants are stored at.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def _checked(layer, what: str):
    if layer is None:
        raise RuntimeError(f"TensorRT rejected the SwinV2 {what}")
    return layer


def constant(network, values, *, dtype):
    """A constant of ``values`` stored at ``dtype`` (a NumPy dtype)."""
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return _checked(network.add_constant(data.shape, trt.Weights(data)), "constant").get_output(0)


def scalar(network, value: float, rank: int, *, dtype):
    return constant(network, np.full((1,) * rank, value), dtype=dtype)


def elementwise(network, left, right, operation):
    return _checked(network.add_elementwise(left, right, operation), "elementwise").get_output(0)


def add(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.SUM)


def multiply(network, left, right):
    return elementwise(network, left, right, trt.ElementWiseOperation.PROD)


def reshape(network, tensor, shape):
    layer = _checked(network.add_shuffle(tensor), "reshape")
    layer.reshape_dims = trt.Dims(tuple(int(value) for value in shape))
    return layer.get_output(0)


def permute(network, tensor, first, permutation, second):
    """One shuffle: an optional reshape, a transpose, then an optional second reshape."""
    layer = _checked(network.add_shuffle(tensor), "shuffle")
    if first is not None:
        layer.reshape_dims = trt.Dims(tuple(int(value) for value in first))
    layer.second_transpose = trt.Permutation(permutation)
    output = layer.get_output(0)
    return reshape(network, output, second) if second is not None else output


def slice_axis(network, tensor, axis: int, start: int, size: int):
    shape = [int(value) for value in tensor.shape]
    starts, sizes = [0] * len(shape), list(shape)
    starts[axis], sizes[axis] = start, size
    layer = network.add_slice(
        tensor, trt.Dims(starts), trt.Dims(sizes), trt.Dims([1] * len(shape))
    )
    return _checked(layer, "slice").get_output(0)


def roll(network, tensor, shifts: tuple[int, int], axes: tuple[int, int]):
    """``torch.roll`` along two axes, built from slices and concatenations.

    A positive shift moves elements towards higher indices, so the split point
    is measured from the end. The opposite sign still builds a tensor of the
    right shape, just the wrong one.
    """
    for shift, axis in zip(shifts, axes):
        extent = int(tensor.shape[axis])
        offset = (-shift) % extent
        if offset == 0:
            continue
        pieces = [
            slice_axis(network, tensor, axis, offset, extent - offset),
            slice_axis(network, tensor, axis, 0, offset),
        ]
        layer = _checked(network.add_concatenation(pieces), "roll")
        layer.axis = axis
        tensor = layer.get_output(0)
    return tensor


def linear(network, tensor, weight, bias, *, dtype):
    """``tensor @ weight.T + bias``, matching ``torch.nn.Linear``'s weight layout."""
    rank = len(tensor.shape)
    matrix = np.asarray(weight, dtype=np.float32).T
    values = constant(network, matrix.reshape((1,) * (rank - 2) + matrix.shape), dtype=dtype)
    product = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    )
    output = _checked(product, "matmul").get_output(0)
    if bias is None:
        return output
    bias = np.asarray(bias, dtype=np.float32).reshape((1,) * (rank - 1) + (-1,))
    return add(network, output, constant(network, bias, dtype=dtype))


def layer_norm(network, tensor, gamma, beta, *, epsilon: float):
    """LayerNorm over the last axis."""
    rank = len(tensor.shape)
    shape = (1,) * (rank - 1) + (int(np.asarray(gamma).shape[0]),)
    scale = constant(network, np.asarray(gamma).reshape(shape), dtype=np.float32)
    shift = constant(network, np.asarray(beta).reshape(shape), dtype=np.float32)
    layer = _checked(
        network.add_normalization_v2(tensor, scale, shift, 1 << (rank - 1)), "layer norm"
    )
    layer.epsilon = float(epsilon)
    return layer.get_output(0)


def l2_normalize(network, tensor, *, epsilon: float):
    """``torch.nn.functional.normalize(x, dim=-1)``: ``x / max(|x|, eps)``."""
    rank = len(tensor.shape)
    square = multiply(network, tensor, tensor)
    total = _checked(
        network.add_reduce(square, trt.ReduceOperation.SUM, 1 << (rank - 1), True), "norm sum"
    ).get_output(0)
    length = _checked(network.add_unary(total, trt.UnaryOperation.SQRT), "norm sqrt").get_output(0)
    length = elementwise(
        network, length, scalar(network, epsilon, rank, dtype=np.float32),
        trt.ElementWiseOperation.MAX,
    )
    return elementwise(network, tensor, length, trt.ElementWiseOperation.DIV)


def gelu(network, tensor, *, dtype):
    """The exact error-function GELU, ``0.5 * x * (1 + erf(x / sqrt(2)))``.

    ``hidden_act: gelu`` names this form in the reference. It is spelled out
    rather than left to TensorRT's GELU activation, which may use the tanh
    approximation.
    """
    rank = len(tensor.shape)
    scaled = multiply(network, tensor, scalar(network, 1.0 / np.sqrt(2.0), rank, dtype=dtype))
    error = _checked(network.add_unary(scaled, trt.UnaryOperation.ERF), "erf").get_output(0)
    gate = add(network, error, scalar(network, 1.0, rank, dtype=dtype))
    half = multiply(network, tensor, scalar(network, 0.5, rank, dtype=dtype))
    return multiply(network, half, gate)


def attention(network, query, key, value, bias):
    """``softmax(query @ key^T + bias) @ value``; any scaling is already in ``query``."""
    layer = _checked(
        network.add_attention(query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False),
        "attention",
    )
    layer.decomposable = True
    layer.mask = bias
    return layer.get_output(0)


def mean_spatial(network, tensor):
    """Mean over the two spatial axes of an NHWC tensor, dropping them."""
    layer = network.add_reduce(tensor, trt.ReduceOperation.AVG, (1 << 1) | (1 << 2), False)
    return _checked(layer, "spatial mean").get_output(0)

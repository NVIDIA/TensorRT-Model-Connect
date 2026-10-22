# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for wav2vec2.

Every helper here keeps the time axis dynamic. The feature encoder's
normalisation is taken over that axis, so a clip cannot be padded to a fixed
window and trimmed afterwards the way a mel-spectrogram family would: the
padding changes the statistics and therefore every frame. The engine takes the
true sample count instead, and each reshape leaves the time extent as ``-1``.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_gelu(network, inp):
    """The exact GELU. wav2vec2 states ``hidden_act: gelu``, which HF maps to
    ``x/2 * (1 + erf(x/sqrt(2)))`` - not the tanh approximation."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_linear(network, inp, weight, bias=None):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout."""
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    rank = len(inp.shape)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    out = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, add_constant(network, shape, weight),
        trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias = np.asarray(bias, dtype=np.float32)
    return add_sum(network, out,
                   add_constant(network, (1,) * (rank - 1) + (bias.shape[0],), bias))


def add_norm(network, inp, gamma, beta, eps, axis, *, param_axis=None):
    """Normalise over ``axis``, then scale and shift along ``param_axis``.

    The two axes are stated separately because they are not always the same
    one. A LayerNorm over the width normalises and scales along the same axis,
    but the feature encoder's ``GroupNorm(num_groups=C, num_channels=C)``
    normalises each channel over time while its affine term rides the channel
    axis. Deriving the second from the first - for instance by assuming the
    affine always rides the last axis - puts the scale on the time axis for a
    ``[1, C, 1, T]`` tensor, where it is both wrong and dynamic.
    """
    rank = len(inp.shape)
    mask = 1 << axis
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, mask, True).get_output(0)
    centred = network.add_elementwise(inp, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred,
                                     trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, mask, True).get_output(0)
    shifted = add_sum(network, variance, add_constant(network, (1,) * rank, [eps]))
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation,
                                     trt.ElementWiseOperation.DIV).get_output(0)
    width = int(np.asarray(gamma).shape[0])
    if param_axis is None:
        param_axis = rank - 1
    shape = tuple(width if index == param_axis else 1 for index in range(rank))
    scaled = network.add_elementwise(normed, add_constant(network, shape, gamma),
                                     trt.ElementWiseOperation.PROD).get_output(0)
    return add_sum(network, scaled, add_constant(network, shape, beta))


def add_conv1d(network, inp, weight, *, stride=1, padding=0, groups=1, bias=None):
    """Conv1d over ``[1, C, 1, T]``; ``weight`` is torch's ``[out, in/groups, k]``."""
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    out_channels, _, kernel = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (1, int(kernel)),
        trt.Weights(np.ascontiguousarray(weight[:, :, None, :])),
        trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=np.float32)))
        if bias is not None else trt.Weights())
    layer.stride_nd = (1, int(stride))
    layer.padding_nd = (0, int(padding))
    layer.num_groups = int(groups)
    return layer.get_output(0)


def weight_normed_kernel(weight_g, weight_v):
    """Undo torch's weight normalisation: ``w = v * g / ||v||``.

    The reference reparameterises the positional convolution with
    ``weight_norm(..., dim=2)``, so the norm runs over the output and input
    axes and leaves one scale per kernel tap. Using ``weight_v`` directly, or
    taking the norm over the other axes, both load without error and both
    destroy the positional signal, so this is measured against the reference
    kernel in the family's tests rather than assumed.
    """
    weight_v = np.asarray(weight_v, dtype=np.float64)
    norm = np.sqrt((weight_v ** 2).sum(axis=(0, 1), keepdims=True))
    return (weight_v * (np.asarray(weight_g, dtype=np.float64) / norm)).astype(np.float32)


def add_attention(network, query, key, value, *, num_heads, head_dim):
    """Unmasked multi-head self-attention over rows ``[time, width]``.

    There is no mask and no padding: the engine is given the clip's true length,
    so every row is real audio.
    """
    def to_heads(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (-1, num_heads, head_dim)
        layer.second_transpose = (1, 0, 2)
        return layer.get_output(0)

    scores = network.add_matrix_multiply(
        to_heads(query), trt.MatrixOperation.NONE,
        to_heads(key), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, add_constant(network, (1, 1, 1), [head_dim ** -0.5]),
        trt.ElementWiseOperation.PROD).get_output(0)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 2
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        to_heads(value), trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (1, 0, 2)
    back.reshape_dims = (-1, num_heads * head_dim)
    return back.get_output(0)

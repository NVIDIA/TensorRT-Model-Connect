# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for WavLM.

Every helper here keeps the time axis dynamic. A clip cannot be padded to a
fixed window and trimmed afterwards: the group-normalised feature encoder
normalises each channel over that axis, and the attention has no padding mask
in either layout, so padding changes every frame. The engine takes the true
sample count instead, and each reshape leaves the time extent as ``-1``.

The helpers follow the type of the tensor they are given: weights, biases and
the position-bias table are stored as fp16 constants next to an fp16 tensor and
as fp32 constants otherwise. The normalisations always compute in fp32 (see
``add_norm``).
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def numpy_dtype(tensor):
    """numpy dtype for constants that meet ``tensor``: fp16 next to fp16, fp32 otherwise."""
    return np.float16 if tensor.dtype == trt.float16 else np.float32


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_prod(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)


def add_gelu(network, inp):
    """The exact GELU. WavLM states ``hidden_act: gelu``, which HF maps to
    ``x/2 * (1 + erf(x/sqrt(2)))`` - not the tanh approximation."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_linear(network, inp, weight, bias=None):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout."""
    dtype = numpy_dtype(inp)
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    rank = len(inp.shape)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    out = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, add_constant(network, shape, weight, dtype),
        trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias = np.asarray(bias, dtype=np.float32)
    return add_sum(network, out,
                   add_constant(network, (1,) * (rank - 1) + (bias.shape[0],), bias, dtype))


def add_norm(network, inp, gamma, beta, eps, axis, *, param_axis=None):
    """Normalise over ``axis``, then scale and shift along ``param_axis``.

    The two axes are stated separately because they are not always the same
    one. A LayerNorm over the width normalises and scales along the same axis,
    but the feature encoder's ``GroupNorm(num_groups=C, num_channels=C)``
    normalises each channel over time while its affine term rides the channel
    axis.

    An fp16 input is cast to fp32 for the statistics and the affine term and
    cast back afterwards. In the group-normalised layout the feature encoder's
    first norm averages over every time step of the clip, and an fp16 mean and
    variance over that many values is not accurate enough.
    """
    source_type = inp.dtype
    if source_type == trt.float16:
        inp = network.add_cast(inp, trt.float32).get_output(0)
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
    out = add_sum(network, scaled, add_constant(network, shape, beta))
    if source_type == trt.float16:
        out = network.add_cast(out, trt.float16).get_output(0)
    return out


def add_conv1d(network, inp, weight, *, stride=1, padding=0, groups=1, bias=None):
    """Conv1d over ``[1, C, 1, T]``; ``weight`` is torch's ``[out, in/groups, k]``."""
    dtype = numpy_dtype(inp)
    weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float32).astype(dtype))
    out_channels, _, kernel = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (1, int(kernel)),
        trt.Weights(np.ascontiguousarray(weight[:, :, None, :])),
        trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=np.float32).astype(dtype)))
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
    taking the norm over the other axes, both load without error, so the
    formula is checked against the reference kernel in the family's tests
    rather than assumed.
    """
    weight_v = np.asarray(weight_v, dtype=np.float64)
    norm = np.sqrt((weight_v ** 2).sum(axis=(0, 1), keepdims=True))
    return (weight_v * (np.asarray(weight_g, dtype=np.float64) / norm)).astype(np.float32)


def _int32(network, shape, values):
    return add_constant(network, shape, values, np.int32)


def add_position_bias(network, rows, table, max_distance):
    """The relative position bias ``[heads, T, T]`` for the ``T`` rows of ``rows``.

    ``table[d + max_distance]`` holds every head's bias for the key-minus-query
    offset ``d``. Every offset at or beyond ``max_distance`` falls into the
    last bucket of its sign, so the offset is clamped there and the table does
    not grow with the clip.
    """
    frames = network.add_slice(network.add_shape(rows).get_output(0),
                               (0,), (1,), (1,)).get_output(0)
    iota = network.add_fill((1,), trt.FillOperation.LINSPACE, trt.int32)
    iota.set_input(0, frames)
    iota.set_input(1, _int32(network, (), [0]))
    iota.set_input(2, _int32(network, (1,), [1]))
    positions = iota.get_output(0)

    query = network.add_shuffle(positions)
    query.reshape_dims = (-1, 1)
    key = network.add_shuffle(positions)
    key.reshape_dims = (1, -1)
    offset = network.add_elementwise(key.get_output(0), query.get_output(0),
                                     trt.ElementWiseOperation.SUB).get_output(0)
    offset = network.add_elementwise(offset, _int32(network, (1, 1), [-max_distance]),
                                     trt.ElementWiseOperation.MAX).get_output(0)
    offset = network.add_elementwise(offset, _int32(network, (1, 1), [max_distance]),
                                     trt.ElementWiseOperation.MIN).get_output(0)
    index = add_sum(network, offset, _int32(network, (1, 1), [max_distance]))

    table = np.asarray(table, dtype=np.float32)
    values = add_constant(network, table.shape, table, numpy_dtype(rows))
    gathered = network.add_gather(values, index, 0).get_output(0)
    heads_first = network.add_shuffle(gathered)
    heads_first.first_transpose = (2, 0, 1)
    return heads_first.get_output(0)


def add_head_gate(network, rows, weight, bias, scale, *, num_heads, head_dim):
    """Each head's gate on the position bias, ``[heads, T, 1]``.

    ``weight`` ``[2, head_dim]`` and ``bias`` ``[2]`` map each head's slice of
    ``rows`` to its two gate logits, ``scale`` ``[heads]`` is the learned
    per-head constant, and ``gate = a * (b * scale - 1) + 2`` with
    ``a, b = sigmoid(logits)``.
    """
    dtype = numpy_dtype(rows)
    heads = network.add_shuffle(rows)
    heads.reshape_dims = (-1, num_heads, head_dim)
    heads.second_transpose = (1, 0, 2)
    logits = add_linear(network, heads.get_output(0), weight, bias)
    gates = network.add_activation(logits, trt.ActivationType.SIGMOID).get_output(0)
    first = network.add_gather(gates, _int32(network, (1,), [0]), 2).get_output(0)
    second = network.add_gather(gates, _int32(network, (1,), [1]), 2).get_output(0)
    scaled = add_prod(network, second, add_constant(network, (num_heads, 1, 1), scale, dtype))
    shifted = network.add_elementwise(scaled, add_constant(network, (1, 1, 1), [1.0], dtype),
                                      trt.ElementWiseOperation.SUB).get_output(0)
    return add_sum(network, add_prod(network, first, shifted),
                   add_constant(network, (1, 1, 1), [2.0], dtype))


def add_attention(network, query, key, value, bias, *, num_heads, head_dim):
    """Multi-head self-attention over rows ``[time, width]`` with an additive
    ``[heads, time, time]`` score bias.

    There is no padding mask: the engine is given the clip's true length, so
    every row is real audio. The bias is added after the ``head_dim ** -0.5``
    scaling, as the reference's ``attn_mask`` is.
    """
    def to_heads(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (-1, num_heads, head_dim)
        layer.second_transpose = (1, 0, 2)
        return layer.get_output(0)

    scores = network.add_matrix_multiply(
        to_heads(query), trt.MatrixOperation.NONE,
        to_heads(key), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = add_prod(network, scores,
                      add_constant(network, (1, 1, 1), [head_dim ** -0.5], numpy_dtype(query)))
    scores = add_sum(network, scores, bias)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 2
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        to_heads(value), trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (1, 0, 2)
    back.reshape_dims = (-1, num_heads * head_dim)
    return back.get_output(0)

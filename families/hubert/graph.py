# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for HuBERT.

Every helper keeps the time axis dynamic. The feature encoder normalises over
that axis (the group-norm layout) or per frame (the layer-norm layout), and a
group norm over a zero-padded clip is a different clip, so the engine takes the
true sample count and each reshape leaves the time extent as ``-1``.

Weights and biases are stored as fp16 constants next to an fp16 tensor and as
fp32 constants otherwise, so an fp16 build changes the compute type of the
convolutions, projections and attention but not the graph. Normalisations
always compute in fp32 (see ``add_norm``).
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


def add_gelu(network, inp):
    """The exact GELU: HuBERT states ``gelu``, which Transformers maps to
    ``x/2 * (1 + erf(x/sqrt(2)))``, not the tanh approximation."""
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_linear(network, inp, weight, bias=None):
    """y = x @ W^T + b, in torch.nn.Linear's weight layout."""
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

    The two axes differ for the feature encoder's
    ``GroupNorm(num_groups=C, num_channels=C)``: it normalises each channel of a
    ``[1, C, 1, T]`` tensor over time while its affine term rides the channel
    axis. A LayerNorm over the width normalises and scales along one axis.

    An fp16 input is cast to fp32 for the statistics and the affine term and
    cast back afterwards: the group norm averages over every time step of the
    clip, and an fp16 mean and variance over that many values is not accurate.
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


def frames_to_rows(network, frames, width):
    """``[1, C, 1, T]`` convolution layout to ``[T, C]`` rows, one per frame."""
    layer = network.add_shuffle(frames)
    layer.first_transpose = (0, 3, 1, 2)
    layer.reshape_dims = (-1, int(width))
    return layer.get_output(0)


def rows_to_frames(network, rows, width):
    """``[T, C]`` rows back to the ``[1, C, 1, T]`` convolution layout."""
    layer = network.add_shuffle(rows)
    layer.first_transpose = (1, 0)
    layer.reshape_dims = (1, int(width), 1, -1)
    return layer.get_output(0)


def trim_like(network, frames, like):
    """Keep the leading time steps of ``frames`` up to the length of ``like``."""
    rank = len(frames.shape)
    layer = network.add_slice(frames, (0,) * rank, (1,) * rank, (1,) * rank)
    layer.set_input(2, network.add_shape(like).get_output(0))
    return layer.get_output(0)


def weight_normed_kernel(weight_g, weight_v):
    """Undo torch's weight normalisation of the positional convolution.

    The reference wraps that convolution in ``weight_norm(..., dim=2)``: the
    kernel is ``v * g / ||v||`` with the norm taken over the output and input
    axes, one scale per kernel tap. Using ``weight_v`` directly, or norming over
    the other axes, loads without error and still yields a readable transcript,
    so the rule is pinned by a unit test with controls.
    """
    weight_v = np.asarray(weight_v, dtype=np.float64)
    norm = np.sqrt((weight_v ** 2).sum(axis=(0, 1), keepdims=True))
    return (weight_v * (np.asarray(weight_g, dtype=np.float64) / norm)).astype(np.float32)


def add_attention(network, query, key, value, *, num_heads, head_dim):
    """Unmasked multi-head self-attention over rows ``[time, width]``.

    The engine is given the clip's true length, so every row is real audio and
    there is no padding to mask.
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
        scores, add_constant(network, (1, 1, 1), [head_dim ** -0.5], numpy_dtype(query)),
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

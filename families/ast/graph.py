# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for the Audio Spectrogram Transformer.

AST is a plain pre-norm ViT over a spectrogram patch grid, so these are the
helpers families/owlv2 and families/clip already use, minus the pieces a CLIP
tower needs and AST does not. The two differences that matter are spelled out
where they occur: AST activates with the exact GELU rather than QuickGELU, and
its layer norms use ``eps`` 1e-12 rather than 1e-5.

Every helper takes and returns rows ``[seq, width]``.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def _as_float32(network, tensor):
    """Cast to fp32 for a normalisation's statistics, and remember the way back.

    A layer norm sums squares across the width. Keeping that reduction in fp16
    is what cost Grounding DINO its parity, so the statistics are always taken
    in fp32 here; they are cheap next to the matmuls around them.
    """
    if tensor.dtype == trt.float32:
        return tensor, None
    return network.add_cast(tensor, trt.float32).get_output(0), tensor.dtype


def _restore_dtype(network, tensor, original):
    if original is None:
        return tensor
    return network.add_cast(tensor, original).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    source, original = _as_float32(network, inp)
    axis = len(inp.shape) - 1
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(source, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    width = int(inp.shape[-1])
    epsilon = add_constant(network, tuple([1] * len(inp.shape)), np.full((1,), eps))
    shifted = network.add_elementwise(variance, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation,
                                     trt.ElementWiseOperation.DIV).get_output(0)
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = network.add_elementwise(normed, add_constant(network, shape, gamma),
                                     trt.ElementWiseOperation.PROD).get_output(0)
    affine = network.add_elementwise(scaled, add_constant(network, shape, beta),
                                     trt.ElementWiseOperation.SUM).get_output(0)
    return _restore_dtype(network, affine, original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout."""
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    rank = len(inp.shape)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    constant = add_constant(network, shape, weight.reshape(shape), dtype=dtype)
    out = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias_shape = (1,) * (rank - 1) + (-1,)
    bias = np.asarray(bias, dtype=dtype).reshape(bias_shape)
    return network.add_elementwise(out, add_constant(network, bias.shape, bias, dtype=dtype),
                                   trt.ElementWiseOperation.SUM).get_output(0)


def add_conv2d(network, inp, weight, bias=None, stride=(1, 1), dtype=np.float32):
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (int(kernel_h), int(kernel_w)), trt.Weights(weight),
        trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=dtype)))
        if bias is not None else trt.Weights())
    layer.stride_nd = tuple(int(v) for v in stride)
    return layer.get_output(0)


def add_gelu(network, inp):
    """The exact GELU, ``x/2 * (1 + erf(x/sqrt(2)))``.

    AST's config states ``hidden_act: gelu``, and HF maps that to the exact
    function, not the tanh approximation and not QuickGELU. The CLIP-derived
    towers in this repo use QuickGELU, so this is the one activation that must
    not be copied across from them.
    """
    return network.add_activation(inp, trt.ActivationType.GELU_ERF).get_output(0)


def add_attention(network, query, key, value, *, num_heads, head_dim):
    """Unmasked multi-head self-attention over rows ``[seq, width]``.

    AST attends over the whole patch grid plus its two leading tokens with no
    mask: there is no causality in a spectrogram and no padding token, because
    the spectrogram itself is padded to a fixed length before the patches are
    cut.
    """
    seq = int(query.shape[0])
    width = num_heads * head_dim

    def to_heads(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, seq, num_heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    scores = network.add_matrix_multiply(
        to_heads(query), trt.MatrixOperation.NONE,
        to_heads(key), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0 / np.sqrt(head_dim)),
                         dtype=np.dtype(trt.nptype(scores.dtype)))
    scores = network.add_elementwise(scores, scale, trt.ElementWiseOperation.PROD).get_output(0)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 3
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        to_heads(value), trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (0, 2, 1, 3)
    back.reshape_dims = (seq, width)
    return back.get_output(0)

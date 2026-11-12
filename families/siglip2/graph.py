# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for SigLIP 2.

Both towers are pre-norm transformers over a row tensor ``[seq, width]``. The text
window is fixed, but the image row count is the patch count of the request's own
NaFlex grid, so every helper here keeps the row axis symbolic: shapes are taken from
the weights, never from the input, and reshapes infer the row count.

Two details are load-bearing:

* the activation is the **tanh approximation** of GELU, not the exact GELU;
* ``add_attention`` lets the query carry a different row count from the keys,
  because the vision tower is pooled by attending one learned query row against
  every patch.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    """LayerNorm over the last axis, spelled out in fp32."""
    axis = len(inp.shape) - 1
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(inp, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    ones = tuple([1] * len(inp.shape))
    shifted = network.add_elementwise(variance, add_constant(network, ones, np.full((1,), eps)),
                                      trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation, trt.ElementWiseOperation.DIV).get_output(0)
    width = int(np.asarray(gamma).shape[-1])
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = network.add_elementwise(normed, add_constant(network, shape, gamma),
                                     trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(scaled, add_constant(network, shape, beta),
                                   trt.ElementWiseOperation.SUM).get_output(0)


def add_linear(network, inp, weight, bias=None, dtype=np.float32):
    """y = x @ W^T + b for rows ``[seq, in]``, matching torch.nn.Linear's layout."""
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    constant = add_constant(network, tuple(weight.shape), weight, dtype=dtype)
    out = network.add_matrix_multiply(inp, trt.MatrixOperation.NONE, constant,
                                      trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias = np.asarray(bias, dtype=dtype).reshape(1, -1)
    return network.add_elementwise(out, add_constant(network, bias.shape, bias, dtype=dtype),
                                   trt.ElementWiseOperation.SUM).get_output(0)


def add_gelu_tanh(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    Both towers state ``gelu_pytorch_tanh``. This tanh approximation and the exact
    GELU agree to about 1e-3, which the relevance logit notices, so the function is
    spelled out rather than left to an activation enum.
    """
    ones = tuple([1] * len(inp.shape))

    def constant(value):
        return add_constant(network, ones, np.full((1,), value), dtype=dtype)

    def prod(left, right):
        return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)

    cube = prod(prod(inp, inp), inp)
    inner = network.add_elementwise(inp, prod(constant(0.044715), cube),
                                    trt.ElementWiseOperation.SUM).get_output(0)
    scaled = prod(constant(np.sqrt(2.0 / np.pi)), inner)
    tanh = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    gate = network.add_elementwise(constant(1.0), tanh, trt.ElementWiseOperation.SUM).get_output(0)
    return prod(prod(constant(0.5), inp), gate)


def _to_heads(network, rows, num_heads, head_dim):
    """``[seq, width]`` to ``[1, heads, seq, head_dim]`` with the row count inferred."""
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (1, -1, num_heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def add_attention(network, query, key, value, *, num_heads, head_dim):
    """Unmasked multi-head attention over rows, the query free to be shorter.

    ``query`` is ``[q_seq, width]`` and ``key``/``value`` are ``[kv_seq, width]``. No
    mask: the text tower is bidirectional over its whole padded window, and the image
    rows are exactly the request's patches (the reference pads them to the patch budget
    and masks the padding out, which leaves the same function).
    """
    scores = network.add_matrix_multiply(
        _to_heads(network, query, num_heads, head_dim), trt.MatrixOperation.NONE,
        _to_heads(network, key, num_heads, head_dim), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0 / np.sqrt(head_dim)),
                         dtype=np.dtype(trt.nptype(scores.dtype)))
    scores = network.add_elementwise(scores, scale, trt.ElementWiseOperation.PROD).get_output(0)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 3
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        _to_heads(network, value, num_heads, head_dim), trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (0, 2, 1, 3)
    back.reshape_dims = (-1, num_heads * head_dim)
    return back.get_output(0)

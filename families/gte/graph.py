# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for GTE-v1.5.

Everything is a row tensor ``[seq, width]``, so every helper takes and
returns that layout. Three things differ from the rotary encoder next door
in ``families/nomic_bert``, and each is spelled out rather than delegated:

* every projection here **has a bias** except ``up_gate_proj``;
* the gated MLP is one matrix, ``up_gate_proj``, split along its output into
  ``up`` then ``gate`` - and it is the **second** half that goes through the
  activation, which is ``gelu``, not silu;
* the gelu is the **exact** erf form, since ``hidden_act`` is ``gelu`` and
  the reference resolves that to ``nn.functional.gelu`` with no tanh
  approximation.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_product(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    axis = len(inp.shape) - 1
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(inp, mean, trt.ElementWiseOperation.SUB).get_output(0)
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
    return network.add_elementwise(scaled, add_constant(network, shape, beta),
                                   trt.ElementWiseOperation.SUM).get_output(0)


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


def add_gelu(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + erf(x / sqrt(2)))`` - the exact form, not the tanh one.

    ``hidden_act`` is ``gelu``, which the reference's activation table maps
    to the exact erf gelu. The tanh approximation agrees with it to about
    1e-3 of relative error, which is small enough to look like noise and
    large enough to move an embedding.
    """
    rank = len(inp.shape)
    ones = tuple([1] * rank)

    def constant(value):
        return add_constant(network, ones, np.full((1,), value), dtype=dtype)

    scaled = add_product(network, inp, constant(1.0 / np.sqrt(2.0)))
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    gate = add_sum(network, constant(1.0), erf)
    return add_product(network, add_product(network, constant(0.5), inp), gate)


def add_gated_mlp(network, hidden, up_gate_weight, down_weight, down_bias, *,
                  intermediate, dtype=np.float32):
    """``down_proj(gelu(gate) * up)`` from one stacked ``up_gate_proj``.

    The reference splits the *projection output* as ``up_states, gate``, so
    the first ``intermediate`` rows of ``up_gate_proj`` are ``up`` and the
    last ``intermediate`` are ``gate``, and **gate is the half that goes
    through the activation**. Swapping the halves builds and runs; it just
    answers wrong.
    """
    weight = np.asarray(up_gate_weight)
    if weight.shape[0] != 2 * intermediate:
        raise ValueError(
            f"gte up_gate_proj has {weight.shape[0]} rows, expected {2 * intermediate}")
    projected = add_linear(network, hidden, weight, dtype=dtype)
    rank = len(projected.shape)
    shape = tuple(int(v) for v in projected.shape)

    def half(start):
        starts = (0,) * (rank - 1) + (start,)
        sizes = shape[:-1] + (intermediate,)
        return network.add_slice(projected, starts, sizes, (1,) * rank).get_output(0)

    up = half(0)
    gate = add_gelu(network, half(intermediate), dtype=dtype)
    return add_linear(network, add_product(network, gate, up), down_weight, down_bias,
                      dtype=dtype)


def _rotate_half(network, tensor, head_dim):
    """``cat(-x[..., d/2:], x[..., :d/2])`` - the split-half convention."""
    half = head_dim // 2
    rank = len(tensor.shape)
    shape = tuple(int(v) for v in tensor.shape)

    def slice_last(start, size):
        starts = (0,) * (rank - 1) + (start,)
        sizes = shape[:-1] + (size,)
        return network.add_slice(tensor, starts, sizes, (1,) * rank).get_output(0)

    first = slice_last(0, half)
    second = slice_last(half, half)
    negated = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    concat = network.add_concatenation([negated, first])
    concat.axis = rank - 1
    return concat.get_output(0)


def add_rope(network, tensor, cos, sin, head_dim):
    """``x * cos + rotate_half(x) * sin`` over ``[1, heads, seq, head_dim]``."""
    rotated = _rotate_half(network, tensor, head_dim)
    return add_sum(network, add_product(network, tensor, cos),
                   add_product(network, rotated, sin))


def add_attention(network, query, key, value, mask, *, num_heads, head_dim, cos, sin):
    """Bidirectional multi-head attention over rows ``[seq, width]``.

    The order is the reference's: rotate, then score, then **divide by
    sqrt(head_dim)**, then add the mask, then softmax. Scaling the query
    before the matmul instead would give the same numbers in exact
    arithmetic and different ones here; scaling after the mask would scale
    the mask too.

    ``mask`` is an additive term already shaped to broadcast over the score
    tensor. It is what keeps padding out of the result: this family runs at
    a fixed sequence length, so a short text is padded, and an unmasked pad
    row would contribute to every other row's output.
    """
    seq = int(query.shape[0])
    width = num_heads * head_dim

    def to_heads(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, seq, num_heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    heads_q, heads_k, heads_v = to_heads(query), to_heads(key), to_heads(value)
    # Rotary touches the query and the key. The value is carried through
    # unrotated, which is what the reference's apply_rotary_pos_emb(q, k) says.
    heads_q = add_rope(network, heads_q, cos, sin, head_dim)
    heads_k = add_rope(network, heads_k, cos, sin, head_dim)

    scores = network.add_matrix_multiply(
        heads_q, trt.MatrixOperation.NONE,
        heads_k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0 / np.sqrt(head_dim)))
    scores = add_product(network, scores, scale)
    scores = add_sum(network, scores, mask)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 3
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        heads_v, trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (0, 2, 1, 3)
    back.reshape_dims = (seq, width)
    return back.get_output(0)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for nomic-embed-text.

Everything is a row tensor ``[seq, width]``, so every helper takes and
returns that layout. Three things here are not what a BERT-shaped checkpoint
usually wants, and each is spelled out rather than delegated:

* every projection is **biasless** - ``qkv_proj_bias``, ``mlp_fc1_bias`` and
  ``mlp_fc2_bias`` are all false, so ``add_linear`` is called without one;
* the MLP is **SwiGLU**, and the gate is ``fc12`` while the value is
  ``fc11`` - the opposite of the Llama naming most readers carry in;
* attention is **rotary**, with the tables built in ``rope.py``.
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


def _as_float32(network, tensor):
    """Cast to fp32 for a reduction's statistics, and remember the way back."""
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


def add_silu(network, inp, dtype=np.float32):
    """``x * sigmoid(x)``.

    ``activation_function: swiglu`` selects ``F.silu`` in the reference. It
    is written out rather than using TensorRT's SiLU activation so the gate
    the MLP multiplies by is visible next to the multiply.
    """
    del dtype
    gate = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
    return add_product(network, inp, gate)


def add_gated_mlp(network, hidden, value_weight, gate_weight, down_weight, dtype=np.float32):
    """``fc2(fc11(x) * silu(fc12(x)))``.

    The reference reads ``y = self.fc11(x)`` and ``gate = self.fc12(x)``,
    then ``y * self.activation(gate)``. So **fc12 is the one that goes
    through the activation**, which is the reverse of Llama's ``gate_proj``
    naming. Swapping them builds and runs; it just answers wrong.
    """
    value = add_linear(network, hidden, value_weight, dtype=dtype)
    gate = add_linear(network, hidden, gate_weight, dtype=dtype)
    return add_linear(network, add_product(network, value, add_silu(network, gate)),
                      down_weight, dtype=dtype)


def _rotate_half(network, tensor, head_dim):
    """``cat(-x[..., d/2:], x[..., :d/2])`` - the non-interleaved convention."""
    half = head_dim // 2
    rank = len(tensor.shape)
    shape = tuple(int(v) for v in tensor.shape)

    def slice_last(start, size):
        starts = (0,) * (rank - 1) + (start,)
        sizes = shape[:-1] + (size,)
        strides = (1,) * rank
        return network.add_slice(tensor, starts, sizes, strides).get_output(0)

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


def add_attention(network, query, key, value, mask, *, num_heads, head_dim,
                  cos=None, sin=None):
    """Bidirectional multi-head attention over rows ``[seq, width]``.

    ``mask`` is an additive term already shaped to broadcast over the score
    tensor, or None. It is what keeps padding out of the result: this family
    runs at a fixed sequence length, so a short text is padded, and an
    unmasked pad row would contribute to every other row's output.
    """
    seq = int(query.shape[0])
    width = num_heads * head_dim

    def to_heads(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, seq, num_heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    heads_q, heads_k, heads_v = to_heads(query), to_heads(key), to_heads(value)
    if cos is not None:
        # Rotary touches the query and the key. The value is carried through
        # unrotated, which is what the reference's stack(q_rot, k_rot, v) says.
        heads_q = add_rope(network, heads_q, cos, sin, head_dim)
        heads_k = add_rope(network, heads_k, cos, sin, head_dim)

    scores = network.add_matrix_multiply(
        heads_q, trt.MatrixOperation.NONE,
        heads_k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0 / np.sqrt(head_dim)),
                         dtype=np.dtype(trt.nptype(scores.dtype)))
    scores = add_product(network, scores, scale)
    if mask is not None:
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

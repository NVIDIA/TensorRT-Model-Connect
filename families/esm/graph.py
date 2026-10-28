# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for ESM-2.

Rows are ``[seq, width]`` until attention, which works in ``[heads, seq, dim]``.
The piece that is not a BERT is ``add_rotary``: there is no position table in
this checkpoint at all, so position enters only as a rotation applied to the
query and the key inside every layer.
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


def add_sub(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUB).get_output(0)


def add_div(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.DIV).get_output(0)


def _as_float32(network, tensor):
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


def add_gelu(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + erf(x / sqrt(2)))``.

    ESM ships its own gelu and says in a comment why: ``F.gelu`` yields subtly
    wrong results against the original repository. This is that exact form,
    not the tanh approximation.
    """
    rank = len(inp.shape)
    ones = tuple([1] * rank)

    def constant(value):
        return add_constant(network, ones, np.full((1,), value), dtype=dtype)

    scaled = add_product(network, inp, constant(1.0 / np.sqrt(2.0)))
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    gate = add_sum(network, constant(1.0), erf)
    return add_product(network, add_product(network, constant(0.5), inp), gate)


def to_heads(network, tensor, num_heads, head_dim):
    """``[seq, heads * dim]`` -> ``[heads, seq, dim]``."""
    seq = int(tensor.shape[0])
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = (seq, num_heads, head_dim)
    layer.second_transpose = (1, 0, 2)
    return layer.get_output(0)


def from_heads(network, tensor, num_heads, head_dim):
    """``[heads, seq, dim]`` -> ``[seq, heads * dim]``."""
    seq = int(tensor.shape[1])
    layer = network.add_shuffle(tensor)
    layer.first_transpose = (1, 0, 2)
    layer.reshape_dims = (seq, num_heads * head_dim)
    return layer.get_output(0)


def rotary_inv_freq(head_dim: int) -> np.ndarray:
    """The RoFormer frequency schedule over half the head width."""
    return 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))


def rotary_tables(seq_len: int, head_dim: int, inv_freq=None, dtype=np.float32):
    """The cos and sin tables ESM rotates with, as ``[1, seq, dim]`` arrays.

    ``inv_freq`` is the frequency schedule over half the head width, and the
    table is that half concatenated with itself - which is what pairs it with
    the ``rotate_half`` convention below. Positions run from 0 over the whole
    engine window, including the padded tail: the padded rows are removed by
    the attention mask, not by refusing to rotate them.

    The schedule is passed in rather than recomputed, because the checkpoint
    ships one and the reference uses the shipped one. See
    ``model._rotary_inv_freq``.
    """
    half = head_dim // 2
    if inv_freq is None:
        inv_freq = rotary_inv_freq(head_dim)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    if inv_freq.shape != (half,):
        raise ValueError(f"esm rotary needs {half} frequencies, got {inv_freq.shape}")
    angles = np.outer(np.arange(seq_len, dtype=np.float64), inv_freq)
    full = np.concatenate((angles, angles), axis=-1)
    assert full.shape == (seq_len, 2 * half)
    return (np.cos(full).astype(dtype).reshape(1, seq_len, head_dim),
            np.sin(full).astype(dtype).reshape(1, seq_len, head_dim))


def _rotate_half(network, tensor, head_dim):
    """``cat(-x[dim/2:], x[:dim/2])`` over the last axis of ``[heads, seq, dim]``."""
    half = head_dim // 2
    heads, seq = int(tensor.shape[0]), int(tensor.shape[1])
    first = network.add_slice(tensor, (0, 0, 0), (heads, seq, half), (1, 1, 1)).get_output(0)
    second = network.add_slice(tensor, (0, 0, half), (heads, seq, half), (1, 1, 1)).get_output(0)
    negated = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    concat = network.add_concatenation([negated, first])
    concat.axis = 2
    return concat.get_output(0)


def add_rotary(network, tensor, cos, sin, head_dim):
    """``x * cos + rotate_half(x) * sin`` - the RoFormer rotation ESM applies."""
    return add_sum(network, add_product(network, tensor, cos),
                   add_product(network, _rotate_half(network, tensor, head_dim), sin))

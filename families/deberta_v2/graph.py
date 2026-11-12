# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for DeBERTa-v2/v3.

Everything is a row tensor ``[seq, width]`` until attention, which works in
``[heads, seq, *]``. The piece that is not a BERT is ``add_disentangled_scores``:
the attention score is the sum of three terms rather than one, and the two
extra terms are looked up through a constant index matrix.
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
    """``0.5 * x * (1 + erf(x / sqrt(2)))`` - the exact form, not the tanh one."""
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


def _gather_last(network, data, indices):
    """``torch.gather(data, dim=-1, index=indices)`` over equal-rank tensors."""
    layer = network.add_gather_v2(data, indices, trt.GatherMode.ELEMENT)
    layer.axis = len(data.shape) - 1
    return layer.get_output(0)


def add_disentangled_scores(network, query, key, position_key, position_query,
                            c2p_index, p2c_index, *, scale):
    """The three-term attention score, before the mask and the softmax.

    ``query`` and ``key`` are ``[heads, seq, dim]``; ``position_key`` and
    ``position_query`` are ``[heads, 2 * span, dim]`` and are constants,
    because they are projections of the relative embedding table and nothing
    in them depends on the input.

    * content-to-content is the ordinary ``Q @ K^T``;
    * content-to-position is ``Q @ position_key^T`` read at the bucket of
      each pair's distance;
    * position-to-content is ``K @ position_query^T`` read at the bucket of
      the *negated* distance, and then transposed, because it is indexed by
      the key's position and scored against the query's.

    All three share one scale, ``sqrt(head_dim * 3)``: the 3 is the number of
    terms. Dividing by ``sqrt(head_dim)`` as an ordinary attention would is a
    silent factor of 1.73 on every score.
    """
    content = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)

    content_to_position = _gather_last(
        network,
        network.add_matrix_multiply(query, trt.MatrixOperation.NONE, position_key,
                                    trt.MatrixOperation.TRANSPOSE).get_output(0),
        c2p_index)

    position_to_content = _gather_last(
        network,
        network.add_matrix_multiply(key, trt.MatrixOperation.NONE, position_query,
                                    trt.MatrixOperation.TRANSPOSE).get_output(0),
        p2c_index)
    swap = network.add_shuffle(position_to_content)
    swap.first_transpose = (0, 2, 1)
    position_to_content = swap.get_output(0)

    total = add_sum(network, add_sum(network, content, content_to_position), position_to_content)
    divisor = add_constant(network, (1, 1, 1), np.full((1,), 1.0 / scale),
                           dtype=np.dtype(trt.nptype(total.dtype)))
    return add_product(network, total, divisor)

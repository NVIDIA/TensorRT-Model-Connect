# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph of ``VideoMAEForPreTraining``.

Rows are ``[1, sequence, width]`` tensors. The encoder sequence is the number
of visible tubes and the decoder sequence the number of all tubes, so both are
only known at run time and every reshape keeps the sequence axis symbolic.

The transformer layers are pre-norm with a residual after attention and after
the MLP; attention projects queries and values with a bias and keys without
one; the activation is the exact error-function GELU (``hidden_act: gelu``).
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def constant(network, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return network.add_constant(data.shape, data).get_output(0)


def _binary(network, left, right, operation):
    return network.add_elementwise(left, right, operation).get_output(0)


def add(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.SUM)


def _multiply(network, left, right):
    return _binary(network, left, right, trt.ElementWiseOperation.PROD)


def layer_norm(network, rows, gamma, beta, eps):
    """LayerNorm over the last axis with the biased variance, as ``torch.nn.LayerNorm``."""
    axis = 1 << (len(rows.shape) - 1)
    ones = (1,) * (len(rows.shape) - 1)
    mean = network.add_reduce(rows, trt.ReduceOperation.AVG, axis, True).get_output(0)
    centred = _binary(network, rows, mean, trt.ElementWiseOperation.SUB)
    variance = network.add_reduce(_multiply(network, centred, centred), trt.ReduceOperation.AVG,
                                  axis, True).get_output(0)
    shifted = add(network, variance, constant(network, np.full(ones + (1,), eps)))
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = _binary(network, centred, deviation, trt.ElementWiseOperation.DIV)
    width = int(np.asarray(gamma).size)
    scaled = _multiply(network, normed,
                       constant(network, np.asarray(gamma).reshape(ones + (width,))))
    return add(network, scaled, constant(network, np.asarray(beta).reshape(ones + (width,))))


def linear(network, rows, weight, bias=None):
    """``rows @ weight.T + bias`` with ``torch.nn.Linear``'s ``[out, in]`` weight layout."""
    weight = np.asarray(weight, dtype=np.float32)
    batch = (1,) * (len(rows.shape) - 2)
    product = network.add_matrix_multiply(
        rows, trt.MatrixOperation.NONE,
        constant(network, weight.reshape(batch + weight.shape)), trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    if bias is None:
        return product
    bias = np.asarray(bias, dtype=np.float32)
    return add(network, product, constant(network, bias.reshape(batch + (1, bias.size))))


def gelu(network, rows):
    """``0.5 * x * (1 + erf(x / sqrt(2)))``, spelled out so the tanh form cannot be chosen."""
    ones = (1,) * len(rows.shape)

    def scalar(value):
        return constant(network, np.full(ones, value))

    erf = network.add_unary(_multiply(network, rows, scalar(1.0 / np.sqrt(2.0))),
                            trt.UnaryOperation.ERF).get_output(0)
    return _multiply(network, _multiply(network, scalar(0.5), rows), add(network, scalar(1.0), erf))


def _heads(network, rows, heads, head_dim):
    """``[1, S, H * D] -> [1, H, S, D]``; 0 copies the symbolic sequence length."""
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (0, 0, heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def attention(network, rows, weights, prefix, heads):
    """Self-attention of ``VideoMAESelfAttention`` plus its output projection."""
    width = int(rows.shape[-1])
    head_dim = width // heads
    attention_prefix = prefix + "attention.attention."
    query = linear(network, rows, weights[attention_prefix + "query.weight"],
                   weights[attention_prefix + "q_bias"])
    key = linear(network, rows, weights[attention_prefix + "key.weight"])
    value = linear(network, rows, weights[attention_prefix + "value.weight"],
                   weights[attention_prefix + "v_bias"])
    scores = network.add_matrix_multiply(
        _heads(network, query, heads, head_dim), trt.MatrixOperation.NONE,
        _heads(network, key, heads, head_dim), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = _multiply(network, scores, constant(network, np.full((1, 1, 1, 1), head_dim**-0.5)))
    probabilities = network.add_softmax(scores)
    probabilities.axes = 1 << 3
    context = network.add_matrix_multiply(
        probabilities.get_output(0), trt.MatrixOperation.NONE,
        _heads(network, value, heads, head_dim), trt.MatrixOperation.NONE).get_output(0)
    merged = network.add_shuffle(context)
    merged.first_transpose = (0, 2, 1, 3)
    merged.reshape_dims = (0, 0, width)
    return linear(network, merged.get_output(0), weights[prefix + "attention.output.dense.weight"],
                  weights[prefix + "attention.output.dense.bias"])


def transformer_layer(network, rows, weights, prefix, heads, eps):
    """One ``VideoMAELayer``: pre-norm attention and MLP, each added back to its input."""
    normed = layer_norm(network, rows, weights[prefix + "layernorm_before.weight"],
                        weights[prefix + "layernorm_before.bias"], eps)
    rows = add(network, attention(network, normed, weights, prefix, heads), rows)
    normed = layer_norm(network, rows, weights[prefix + "layernorm_after.weight"],
                        weights[prefix + "layernorm_after.bias"], eps)
    hidden = gelu(network, linear(network, normed, weights[prefix + "intermediate.dense.weight"],
                                  weights[prefix + "intermediate.dense.bias"]))
    return add(network, linear(network, hidden, weights[prefix + "output.dense.weight"],
                               weights[prefix + "output.dense.bias"]), rows)


def sinusoid_table(positions: int, width: int) -> np.ndarray:
    """The fixed position table of the reference, computed in float64 and stored as float32.

    Reproduces ``get_sinusoid_encoding_table`` of Hugging Face Transformers 5.2.0
    (Apache-2.0; see NOTICE) operation for operation, so the float32 values match
    the reference bit for bit.
    """

    def angles(position):
        return [position / np.power(10000, 2 * (index // 2) / width) for index in range(width)]

    table = np.array([angles(position) for position in range(positions)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return table.astype(np.float32)


def last_rows(network, rows, skipped, kept, width):
    """``rows[:, -len(kept):]`` where the first ``len(skipped)`` rows are dropped."""

    def lengths(*parts):
        layer = network.add_concatenation(list(parts))
        layer.axis = 0
        return layer.get_output(0)

    skipped_count = network.add_shape(skipped).get_output(0)
    kept_count = network.add_shape(kept).get_output(0)
    zero = constant(network, [0], np.int64)
    layer = network.add_slice(rows, (0, 0, 0), (1, 1, width), (1, 1, 1))
    layer.set_input(1, lengths(zero, skipped_count, zero))
    layer.set_input(2, lengths(constant(network, [1], np.int64), kept_count,
                               constant(network, [width], np.int64)))
    return layer.get_output(0)

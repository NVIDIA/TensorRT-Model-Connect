# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-norm blocks, with rotary positions inside the attention.

ESM is a **pre-norm** transformer, which is the one structural difference from
the BERT-shaped encoders next door::

    hidden = hidden + dense(attention(norm(hidden)))
    hidden = hidden + dense(gelu(dense(norm(hidden))))

Both residuals add the value from *before* the normalisation, and neither
branch normalises afterwards. Wiring it post-norm the way ``bert`` and
``deberta_v2`` are wired loads every tensor and runs; the whole encoder is
then subtly wrong and nothing but a numeric comparison says so.

The other difference is the scale. ESM divides the **query** by
``sqrt(head_dim)`` before the rotation rather than dividing the score after
it, and the reference carries a comment saying so. The two are the same
number here, and this builds the order the reference uses.
"""

from __future__ import annotations

import numpy as np

import tensorrt as trt

from . import graph as g


def build_layers(network, hidden, weights, cfg, *, mask, cos, sin, dtype=np.float32):
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    root = cfg["prefix"]
    scale = g.add_constant(network, (1, 1, 1), np.full((1,), head_dim ** -0.5), dtype=dtype)

    for index in range(cfg["num_layers"]):
        prefix = f"{root}encoder.layer.{index}."

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{prefix}attention.LayerNorm.weight"],
                                  weights[f"{prefix}attention.LayerNorm.bias"], cfg["eps"])

        query = g.to_heads(network, g.add_linear(
            network, normed, weights[f"{prefix}attention.self.query.weight"],
            weights[f"{prefix}attention.self.query.bias"], dtype=dtype), num_heads, head_dim)
        key = g.to_heads(network, g.add_linear(
            network, normed, weights[f"{prefix}attention.self.key.weight"],
            weights[f"{prefix}attention.self.key.bias"], dtype=dtype), num_heads, head_dim)
        value = g.to_heads(network, g.add_linear(
            network, normed, weights[f"{prefix}attention.self.value.weight"],
            weights[f"{prefix}attention.self.value.bias"], dtype=dtype), num_heads, head_dim)

        # Scale first, rotate second: the reference's order.
        query = g.add_rotary(network, g.add_product(network, query, scale), cos, sin, head_dim)
        key = g.add_rotary(network, key, cos, sin, head_dim)

        scores = network.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = g.add_sum(network, scores, mask)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 2
        context = network.add_matrix_multiply(
            probabilities.get_output(0), trt.MatrixOperation.NONE,
            value, trt.MatrixOperation.NONE).get_output(0)

        attended = g.add_linear(network, g.from_heads(network, context, num_heads, head_dim),
                                weights[f"{prefix}attention.output.dense.weight"],
                                weights[f"{prefix}attention.output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attended)

        normed = g.add_layer_norm(network, hidden, weights[f"{prefix}LayerNorm.weight"],
                                  weights[f"{prefix}LayerNorm.bias"], cfg["eps"])
        inner = g.add_gelu(network, g.add_linear(
            network, normed, weights[f"{prefix}intermediate.dense.weight"],
            weights[f"{prefix}intermediate.dense.bias"], dtype=dtype), dtype=dtype)
        outer = g.add_linear(network, inner, weights[f"{prefix}output.dense.weight"],
                             weights[f"{prefix}output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, outer)

    # One normalisation closes the stack, which a pre-norm encoder needs and a
    # post-norm one does not have.
    return g.add_layer_norm(network, hidden,
                            weights[f"{root}encoder.emb_layer_norm_after.weight"],
                            weights[f"{root}encoder.emb_layer_norm_after.bias"], cfg["eps"])

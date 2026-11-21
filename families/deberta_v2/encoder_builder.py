# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The post-norm blocks, with disentangled attention in place of the usual one.

Outside the attention this is an ordinary BERT block::

    hidden = norm(dense(attention(hidden)) + hidden)
    hidden = norm(dense(gelu(dense(hidden))) + hidden)

Inside it, the score is three terms instead of one. The two positional terms
are projections of the same relative embedding table through **this layer's
own** query and key matrices - that is what ``share_att_key`` means - so they
are recomputed per layer, but they are still constants: nothing in them
depends on the input.
"""

from __future__ import annotations

import numpy as np

import tensorrt as trt

from . import graph as g


def _position_projections(network, relative, weights, prefix, cfg, dtype):
    """``[heads, 2 * span, dim]`` for the c2p and p2c lookups.

    ``share_att_key`` is checked in ``config.py``; this is the branch it
    guarantees. With it false the checkpoint would carry ``pos_key_proj`` and
    ``pos_query_proj`` instead, which it does not.
    """
    position_key = g.add_linear(network, relative,
                                weights[f"{prefix}attention.self.key_proj.weight"],
                                weights[f"{prefix}attention.self.key_proj.bias"], dtype=dtype)
    position_query = g.add_linear(network, relative,
                                  weights[f"{prefix}attention.self.query_proj.weight"],
                                  weights[f"{prefix}attention.self.query_proj.bias"], dtype=dtype)
    return (g.to_heads(network, position_key, cfg["num_heads"], cfg["head_dim"]),
            g.to_heads(network, position_query, cfg["num_heads"], cfg["head_dim"]))


def build_layers(network, hidden, relative, weights, cfg, *, mask, c2p_index, p2c_index,
                 dtype=np.float32):
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    scale = float(np.sqrt(head_dim * cfg["scale_factor"]))
    root = cfg["prefix"]

    for index in range(cfg["num_layers"]):
        prefix = f"{root}encoder.layer.{index}."
        query = g.to_heads(network, g.add_linear(
            network, hidden, weights[f"{prefix}attention.self.query_proj.weight"],
            weights[f"{prefix}attention.self.query_proj.bias"], dtype=dtype),
            num_heads, head_dim)
        key = g.to_heads(network, g.add_linear(
            network, hidden, weights[f"{prefix}attention.self.key_proj.weight"],
            weights[f"{prefix}attention.self.key_proj.bias"], dtype=dtype),
            num_heads, head_dim)
        value = g.to_heads(network, g.add_linear(
            network, hidden, weights[f"{prefix}attention.self.value_proj.weight"],
            weights[f"{prefix}attention.self.value_proj.bias"], dtype=dtype),
            num_heads, head_dim)

        position_key, position_query = _position_projections(
            network, relative, weights, prefix, cfg, dtype)
        scores = g.add_disentangled_scores(network, query, key, position_key, position_query,
                                           c2p_index, p2c_index, scale=scale)
        scores = g.add_sum(network, scores, mask)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 2
        context = network.add_matrix_multiply(
            probabilities.get_output(0), trt.MatrixOperation.NONE,
            value, trt.MatrixOperation.NONE).get_output(0)

        attended = g.add_linear(network, g.from_heads(network, context, num_heads, head_dim),
                                weights[f"{prefix}attention.output.dense.weight"],
                                weights[f"{prefix}attention.output.dense.bias"], dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, attended, hidden),
                                  weights[f"{prefix}attention.output.LayerNorm.weight"],
                                  weights[f"{prefix}attention.output.LayerNorm.bias"], cfg["eps"])

        inner = g.add_gelu(network, g.add_linear(
            network, hidden, weights[f"{prefix}intermediate.dense.weight"],
            weights[f"{prefix}intermediate.dense.bias"], dtype=dtype), dtype=dtype)
        outer = g.add_linear(network, inner, weights[f"{prefix}output.dense.weight"],
                             weights[f"{prefix}output.dense.bias"], dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, outer, hidden),
                                  weights[f"{prefix}output.LayerNorm.weight"],
                                  weights[f"{prefix}output.LayerNorm.bias"], cfg["eps"])
    return hidden

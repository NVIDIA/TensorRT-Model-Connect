# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-norm blocks: QK LayerNorm, rotary positions, SwiGLU and the residual scale.

ESMC is a pre-norm transformer, so the two residuals add the value from *before*
the normalisation and neither branch normalises afterwards::

    hidden = hidden + attention(norm(hidden)) / s
    hidden = hidden + swiglu(norm(hidden)) / s

Four things inside the block are not in a BERT-shaped encoder, and each one is
a defect that loads every tensor and still runs:

* the query and the key are normalised after their projections, over the whole
  hidden width rather than per head, and before the rotation. Those two norms
  have a weight and no bias;
* every projection in the block is bias-free, which ``config.resolve`` refuses
  a checkpoint over because the graph cannot tell a dropped bias from a
  different model;
* the MLP is SwiGLU - ``down(silu(gate(x)) * up(x))`` - and the gelu on the
  same three tensors is the wrong function;
* each residual branch is divided by ``s = sqrt(num_layers / 36)``. It is 1.0
  for the 36-layer checkpoint and 0.912871 for the 30-layer one, so leaving it
  out is invisible on 600M and wrong on every row of 300M.

The attention scale multiplies the scores rather than the query before the
rotation, which ``families/esm`` does. The two are the same number; this is the
order the reference evaluates, so a numeric mismatch here has one fewer suspect.
"""

from __future__ import annotations

import numpy as np

import tensorrt as trt

from . import graph as g


def build_layers(network, hidden, weights, cfg, *, mask, cos, sin, dtype=np.float32):
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    root = cfg["prefix"]
    scale = g.add_constant(network, (1, 1, 1), np.full((1,), head_dim**-0.5), dtype=dtype)
    residual_scale = g.add_constant(
        network, (1, 1), np.full((1,), cfg["residual_scale"]), dtype=dtype
    )

    for index in range(cfg["num_layers"]):
        prefix = f"{root}layers.{index}."

        normed = g.add_layer_norm(
            network,
            hidden,
            weights[f"{prefix}input_layernorm.weight"],
            weights[f"{prefix}input_layernorm.bias"],
            cfg["eps"],
        )

        # The projections are bias-free, so each is one matrix multiply.
        query = g.add_linear(
            network, normed, weights[f"{prefix}self_attn.q_proj.weight"], dtype=dtype
        )
        key = g.add_linear(
            network, normed, weights[f"{prefix}self_attn.k_proj.weight"], dtype=dtype
        )
        value = g.add_linear(
            network, normed, weights[f"{prefix}self_attn.v_proj.weight"], dtype=dtype
        )

        query = g.add_layer_norm(
            network, query, weights[f"{prefix}self_attn.q_norm.weight"], None, cfg["eps"]
        )
        key = g.add_layer_norm(
            network, key, weights[f"{prefix}self_attn.k_norm.weight"], None, cfg["eps"]
        )

        query = g.to_heads(network, query, num_heads, head_dim)
        key = g.to_heads(network, key, num_heads, head_dim)
        value = g.to_heads(network, value, num_heads, head_dim)

        query = g.add_rotary(network, query, cos, sin, head_dim)
        key = g.add_rotary(network, key, cos, sin, head_dim)

        scores = network.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = g.add_product(network, scores, scale)
        scores = g.add_sum(network, scores, mask)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 2
        context = network.add_matrix_multiply(
            probabilities.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)

        attended = g.add_linear(
            network,
            g.from_heads(network, context, num_heads, head_dim),
            weights[f"{prefix}self_attn.o_proj.weight"],
            dtype=dtype,
        )
        hidden = g.add_sum(network, hidden, g.add_div(network, attended, residual_scale))

        normed = g.add_layer_norm(
            network,
            hidden,
            weights[f"{prefix}post_attention_layernorm.weight"],
            weights[f"{prefix}post_attention_layernorm.bias"],
            cfg["eps"],
        )
        gate = g.add_silu(
            network,
            g.add_linear(network, normed, weights[f"{prefix}mlp.gate_proj.weight"], dtype=dtype),
        )
        up = g.add_linear(network, normed, weights[f"{prefix}mlp.up_proj.weight"], dtype=dtype)
        inner = g.add_product(network, gate, up)
        outer = g.add_linear(network, inner, weights[f"{prefix}mlp.down_proj.weight"], dtype=dtype)
        hidden = g.add_sum(network, hidden, g.add_div(network, outer, residual_scale))

    # One normalisation closes the stack. Its weight is the only tensor:
    # ``esmc.norm`` is a LayerNorm with ``bias=False``.
    return g.add_layer_norm(network, hidden, weights[f"{root}norm.weight"], None, cfg["eps"])

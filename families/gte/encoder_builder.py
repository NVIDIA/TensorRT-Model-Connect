# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The twelve post-norm blocks.

One block, written the way the reference's ``NewLayer.forward`` writes it::

    hidden = attn_ln(o_proj(attn(qkv_proj(hidden))) + hidden)
    hidden = mlp_ln(down_proj(gelu(gate) * up) + hidden)

Both norms come **after** the residual add. Reading the same weights
pre-norm - normalising on the way into each branch instead - produces a
model that runs and converges on plausible vectors, and is wrong.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def split_qkv(packed: np.ndarray, hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``qkv_proj`` into q, k and v.

    The reference splits the projection *output* with
    ``.split(all_head_size, dim=-1)`` and only then reshapes each piece to
    ``(heads, head_dim)``. So the three matrices are contiguous blocks of
    rows in q, k, v order. Splitting on the head axis instead gives three
    matrices of the right shape built from the wrong rows.
    """
    packed = np.asarray(packed)
    if packed.shape[0] != 3 * hidden:
        raise ValueError(f"gte packed QKV has {packed.shape[0]} rows, expected {3 * hidden}")
    return packed[:hidden], packed[hidden:2 * hidden], packed[2 * hidden:]


def build_layers(network, hidden, weights, cfg, *, mask, cos, sin, dtype=np.float32):
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    width = cfg["hidden_size"]
    intermediate = cfg["intermediate_size"]

    for index in range(cfg["num_layers"]):
        prefix = f"encoder.layer.{index}."
        query, key, value = split_qkv(weights[f"{prefix}attention.qkv_proj.weight"], width)
        q_bias, k_bias, v_bias = split_qkv(
            np.asarray(weights[f"{prefix}attention.qkv_proj.bias"]).reshape(3 * width, 1), width)
        attended = g.add_attention(
            network,
            g.add_linear(network, hidden, query, q_bias.reshape(width), dtype=dtype),
            g.add_linear(network, hidden, key, k_bias.reshape(width), dtype=dtype),
            g.add_linear(network, hidden, value, v_bias.reshape(width), dtype=dtype),
            mask, num_heads=num_heads, head_dim=head_dim, cos=cos, sin=sin)
        projected = g.add_linear(network, attended,
                                 weights[f"{prefix}attention.o_proj.weight"],
                                 weights[f"{prefix}attention.o_proj.bias"], dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, projected, hidden),
                                  weights[f"{prefix}attn_ln.weight"],
                                  weights[f"{prefix}attn_ln.bias"], cfg["eps"])

        mlp = g.add_gated_mlp(network, hidden,
                              weights[f"{prefix}mlp.up_gate_proj.weight"],
                              weights[f"{prefix}mlp.down_proj.weight"],
                              weights[f"{prefix}mlp.down_proj.bias"],
                              intermediate=intermediate, dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, mlp, hidden),
                                  weights[f"{prefix}mlp_ln.weight"],
                                  weights[f"{prefix}mlp_ln.bias"], cfg["eps"])
    return hidden

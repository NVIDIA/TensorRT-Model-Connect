# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ViT's transformer stack.

A plain pre-norm block: norm, self-attention, add; norm, MLP, add. What is
easy to get wrong is the naming, because this checkpoint is named the way
BERT is rather than the way CLIP is, and the two halves of the MLP do not
live under one ``mlp`` prefix at all:

* ``layernorm_before`` / ``layernorm_after``, not ``layer_norm1`` / ``2``
* ``attention.attention.{query,key,value}`` and ``attention.output.dense``,
  not ``self_attn.{q,k,v}_proj`` and ``self_attn.out_proj``
* ``intermediate.dense`` and ``output.dense``, not ``mlp.fc1`` and
  ``mlp.fc2``

A builder copied from another family resolves none of these and fails
loudly on the first missing key, which is the good case. The bad case is a
checkpoint that happens to carry both spellings.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def build_layers(network, hidden, weights, cfg, *, dtype=np.float32):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``."""
    width = cfg["hidden_size"]
    prefix = cfg["prefix"]
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}encoder.layer.{index}"

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.layernorm_before.weight"],
                                  weights[f"{layer}.layernorm_before.bias"], cfg["eps"])
        projections = []
        for name in ("query", "key", "value"):
            projections.append(g.add_linear(
                network, normed, weights[f"{layer}.attention.attention.{name}.weight"],
                weights[f"{layer}.attention.attention.{name}.bias"], dtype=dtype))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = g.add_linear(network, attention,
                                 weights[f"{layer}.attention.output.dense.weight"],
                                 weights[f"{layer}.attention.output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.layernorm_after.weight"],
                                  weights[f"{layer}.layernorm_after.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.intermediate.dense.weight"],
                             weights[f"{layer}.intermediate.dense.bias"], dtype=dtype)
        inner = g.add_gelu(network, inner, dtype=dtype)
        inner = g.add_linear(network, inner, weights[f"{layer}.output.dense.weight"],
                             weights[f"{layer}.output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

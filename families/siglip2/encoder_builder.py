# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transformer stack both SigLIP 2 towers share.

Pre-norm layers: LayerNorm, QKV with bias, an output projection with bias, then a
GELU-tanh MLP, each added back to the residual. The towers differ only in width,
depth and row count.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def _linear(network, hidden, weights, name, dtype):
    return g.add_linear(network, hidden, weights[f"{name}.weight"], weights[f"{name}.bias"],
                        dtype=dtype)


def build_layers(network, hidden, weights, prefix, cfg, *, dtype=np.float32):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``."""
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layers.{index}"

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.layer_norm1.weight"],
                                  weights[f"{layer}.layer_norm1.bias"], cfg["eps"])
        query, key, value = (
            _linear(network, normed, weights, f"{layer}.self_attn.{name}", dtype)
            for name in ("q_proj", "k_proj", "v_proj"))
        attention = g.add_attention(network, query, key, value, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = _linear(network, attention, weights, f"{layer}.self_attn.out_proj", dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.layer_norm2.weight"],
                                  weights[f"{layer}.layer_norm2.bias"], cfg["eps"])
        inner = _linear(network, normed, weights, f"{layer}.mlp.fc1", dtype)
        inner = g.add_gelu_tanh(network, inner, dtype=dtype)
        inner = _linear(network, inner, weights, f"{layer}.mlp.fc2", dtype)
        hidden = g.add_sum(network, hidden, inner)
    return hidden

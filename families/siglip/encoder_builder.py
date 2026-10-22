# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transformer stack both SigLIP towers share.

The towers differ in nothing but width and length: pre-norm, QKV with bias, an
output projection with bias, then a GELU-tanh MLP. Neither is masked. The text
tower in particular is **bidirectional**, which is the opposite of CLIP's, and
the reference says so in a comment next to the mask it does not build.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def build_layers(network, hidden, weights, prefix, cfg, *, dtype=np.float32):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``."""
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layers.{index}"

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm1.weight"],
                                  weights[f"{layer}.layer_norm1.bias"], cfg["eps"])
        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            projections.append(g.add_linear(network, normed,
                                            weights[f"{layer}.self_attn.{name}.weight"],
                                            weights[f"{layer}.self_attn.{name}.bias"],
                                            dtype=dtype))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = g.add_linear(network, attention,
                                 weights[f"{layer}.self_attn.out_proj.weight"],
                                 weights[f"{layer}.self_attn.out_proj.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm2.weight"],
                                  weights[f"{layer}.layer_norm2.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.mlp.fc1.weight"],
                             weights[f"{layer}.mlp.fc1.bias"], dtype=dtype)
        inner = g.add_gelu_tanh(network, inner, dtype=dtype)
        inner = g.add_linear(network, inner, weights[f"{layer}.mlp.fc2.weight"],
                             weights[f"{layer}.mlp.fc2.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

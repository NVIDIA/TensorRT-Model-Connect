# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AST's encoder stack: twelve pre-norm ViT layers over the patch rows.

The checkpoint spells the layer the HF ViT way rather than the CLIP way, so
the tensor names differ from families/owlv2 even though the arithmetic is the
same: ``layernorm_before``/``layernorm_after`` rather than
``layer_norm1``/``layer_norm2``, ``attention.attention.query`` rather than
``self_attn.q_proj``, and ``intermediate.dense``/``output.dense`` rather than
``mlp.fc1``/``mlp.fc2``.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def build_layers(network, hidden, weights, prefix, cfg, *, dtype=np.float32):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``."""
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layer.{index}"

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layernorm_before.weight"],
                                  weights[f"{layer}.layernorm_before.bias"], cfg["eps"])
        projections = []
        for name in ("query", "key", "value"):
            projections.append(g.add_linear(network, normed,
                                            weights[f"{layer}.attention.attention.{name}.weight"],
                                            weights[f"{layer}.attention.attention.{name}.bias"],
                                            dtype=dtype))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = g.add_linear(network, attention,
                                 weights[f"{layer}.attention.output.dense.weight"],
                                 weights[f"{layer}.attention.output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layernorm_after.weight"],
                                  weights[f"{layer}.layernorm_after.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.intermediate.dense.weight"],
                             weights[f"{layer}.intermediate.dense.bias"], dtype=dtype)
        inner = g.add_gelu(network, inner)
        inner = g.add_linear(network, inner, weights[f"{layer}.output.dense.weight"],
                             weights[f"{layer}.output.dense.bias"], dtype=dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

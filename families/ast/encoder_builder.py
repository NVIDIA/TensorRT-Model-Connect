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


def build_layers(network, hidden, weights, prefix, cfg, *, dtype=np.float32, quant_ctx=None):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``.

    ``quant_ctx`` routes the projections it has a scale for through FP8 Q/DQ.
    """
    width = cfg["hidden_size"]

    def linear(inp, index, stem, suffix):
        weight = weights[f"{prefix}.encoder.layer.{index}.{suffix}.weight"]
        bias = weights[f"{prefix}.encoder.layer.{index}.{suffix}.bias"]
        if quant_ctx is None:
            return g.add_linear(network, inp, weight, bias, dtype=dtype)
        return quant_ctx.linear(network, inp, f"layer.{index}.{stem}", weight, bias, dtype)

    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layer.{index}"

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layernorm_before.weight"],
                                  weights[f"{layer}.layernorm_before.bias"], cfg["eps"])
        projections = []
        for name in ("query", "key", "value"):
            projections.append(linear(normed, index, name, f"attention.attention.{name}"))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = linear(attention, index, "attn_out", "attention.output.dense")
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layernorm_after.weight"],
                                  weights[f"{layer}.layernorm_after.bias"], cfg["eps"])
        inner = linear(normed, index, "fc1", "intermediate.dense")
        inner = g.add_gelu(network, inner)
        inner = linear(inner, index, "fc2", "output.dense")
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

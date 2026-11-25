# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned pre-norm attention and LayerScale blocks."""

from __future__ import annotations

from . import graph as g


def _linear(network, hidden, weights, name, *, bias_required=True):
    bias = weights[f"{name}.bias"] if bias_required else weights.get(f"{name}.bias")
    return g.add_linear(network, hidden, weights[f"{name}.weight"], bias)


def build_layers(network, hidden, weights, cfg):
    """Run the checkpoint's transformer layers over all class, register, and patch rows."""
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"encoder.layer.{index}"

        normed = g.add_layer_norm(
            network,
            hidden,
            weights[f"{layer}.norm1.weight"],
            weights[f"{layer}.norm1.bias"],
            cfg["eps"],
        )
        projections = []
        for name in ("query", "key", "value"):
            projections.append(
                _linear(
                    network,
                    normed,
                    weights,
                    f"{layer}.attention.attention.{name}",
                    bias_required=cfg["qkv_bias"],
                )
            )
        attention = g.add_attention(
            network, *projections, num_heads=cfg["num_heads"], head_dim=cfg["head_dim"]
        )
        attention = _linear(network, attention, weights, f"{layer}.attention.output.dense")
        attention = g.add_scale(network, attention, weights[f"{layer}.layer_scale1.lambda1"])
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(
            network,
            hidden,
            weights[f"{layer}.norm2.weight"],
            weights[f"{layer}.norm2.bias"],
            cfg["eps"],
        )
        inner = _linear(network, normed, weights, f"{layer}.mlp.fc1")
        inner = g.add_gelu(network, inner)
        inner = _linear(network, inner, weights, f"{layer}.mlp.fc2")
        inner = g.add_scale(network, inner, weights[f"{layer}.layer_scale2.lambda1"])
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv2's transformer stack.

Two things separate this from the pre-norm stack other ViT families in this
repo build, and a copy that misses either still produces features of the
right shape:

* **LayerScale.** Each branch's output is multiplied by a learned per-channel
  vector before it is added back. Measured on the base checkpoint, that
  vector runs from -1.92 to 0.77 with a mean of 0.009, so it is nothing like
  an identity.
* **the tensor names.** This checkpoint is named the way BERT is -
  ``attention.attention.query`` and ``attention.output.dense`` - not the way
  CLIP is, with ``self_attn.q_proj`` and ``self_attn.out_proj``.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def build_layers(network, hidden, weights, cfg, *, dtype=np.float32):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``."""
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"encoder.layer.{index}"

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm1.weight"],
                                  weights[f"{layer}.norm1.bias"], cfg["eps"])
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
        attention = g.add_scale(network, attention,
                                weights[f"{layer}.layer_scale1.lambda1"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm2.weight"],
                                  weights[f"{layer}.norm2.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.mlp.fc1.weight"],
                             weights[f"{layer}.mlp.fc1.bias"], dtype=dtype)
        inner = g.add_gelu(network, inner, dtype=dtype)
        inner = g.add_linear(network, inner, weights[f"{layer}.mlp.fc2.weight"],
                             weights[f"{layer}.mlp.fc2.bias"], dtype=dtype)
        inner = g.add_scale(network, inner,
                            weights[f"{layer}.layer_scale2.lambda1"], dtype=dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

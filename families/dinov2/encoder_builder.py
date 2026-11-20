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


def _linear(network, hidden, weights, name, quant, dtype):
    """One projection ``name`` (checkpoint name without ``.weight``), FP8 when ``quant`` says so."""
    matmul = None if quant is None else quant.matmul(name)
    weight = None if matmul is not None else weights[f"{name}.weight"]
    return g.add_linear(network, hidden, weight, weights[f"{name}.bias"], dtype=dtype,
                        matmul=matmul)


def build_layers(network, hidden, weights, cfg, *, dtype=np.float32, quant=None):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``.

    ``quant`` (a ``Dinov2QuantContext``) routes its projections through FP8 Q/DQ.
    """
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"encoder.layer.{index}"

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm1.weight"],
                                  weights[f"{layer}.norm1.bias"], cfg["eps"])
        projections = []
        for name in ("query", "key", "value"):
            projections.append(_linear(
                network, normed, weights, f"{layer}.attention.attention.{name}", quant, dtype))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = _linear(network, attention, weights, f"{layer}.attention.output.dense",
                            quant, dtype)
        attention = g.add_scale(network, attention,
                                weights[f"{layer}.layer_scale1.lambda1"], dtype=dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm2.weight"],
                                  weights[f"{layer}.norm2.bias"], cfg["eps"])
        inner = _linear(network, normed, weights, f"{layer}.mlp.fc1", quant, dtype)
        inner = g.add_gelu(network, inner, dtype=dtype)
        inner = _linear(network, inner, weights, f"{layer}.mlp.fc2", quant, dtype)
        inner = g.add_scale(network, inner,
                            weights[f"{layer}.layer_scale2.lambda1"], dtype=dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

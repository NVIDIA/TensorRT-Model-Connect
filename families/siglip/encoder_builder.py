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


def _linear(network, hidden, weights, name, quant, dtype):
    """One projection ``name`` (checkpoint name without ``.weight``), FP8 when ``quant`` says so."""
    matmul = None if quant is None else quant.matmul(name)
    weight = None if matmul is not None else weights[f"{name}.weight"]
    return g.add_linear(network, hidden, weight, weights[f"{name}.bias"], dtype=dtype,
                        matmul=matmul)


def build_layers(network, hidden, weights, prefix, cfg, *, dtype=np.float32, quant=None):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``.

    ``quant`` (a ``SiglipQuantContext``) routes its projections through FP8 Q/DQ.
    """
    width = cfg["hidden_size"]
    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layers.{index}"

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm1.weight"],
                                  weights[f"{layer}.layer_norm1.bias"], cfg["eps"])
        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            projections.append(_linear(network, normed, weights,
                                       f"{layer}.self_attn.{name}", quant, dtype))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"])
        attention = _linear(network, attention, weights, f"{layer}.self_attn.out_proj",
                            quant, dtype)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm2.weight"],
                                  weights[f"{layer}.layer_norm2.bias"], cfg["eps"])
        inner = _linear(network, normed, weights, f"{layer}.mlp.fc1", quant, dtype)
        inner = g.add_gelu_tanh(network, inner, dtype=dtype)
        inner = _linear(network, inner, weights, f"{layer}.mlp.fc2", quant, dtype)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden

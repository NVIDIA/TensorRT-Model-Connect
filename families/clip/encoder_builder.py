# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The transformer stack both CLIP towers share.

The towers differ in width, head count and whether the attention is masked,
but not in layer shape: pre-norm, QKV with bias, an output projection with
bias, then a QuickGELU MLP. One builder serves both.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def build_layers(network, hidden, weights, prefix, cfg, *, mask=None, dtype=np.float32,
                 quant_ctx=None):
    """Run ``cfg["num_layers"]`` pre-norm encoder layers over rows ``[seq, width]``.

    ``quant_ctx`` routes the projections it holds scales for through FP8 Q/DQ.
    """
    width = cfg["hidden_size"]

    def linear(name, inp, layer):
        base = f"{layer}.{name}"
        matmul = quant_ctx.matmul(base) if quant_ctx is not None else None
        return g.add_linear(network, inp, weights[f"{base}.weight"], weights[f"{base}.bias"],
                            dtype=dtype, matmul=matmul)

    for index in range(cfg["num_layers"]):
        layer = f"{prefix}.encoder.layers.{index}"

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm1.weight"],
                                  weights[f"{layer}.layer_norm1.bias"], cfg["eps"])
        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            projections.append(linear(f"self_attn.{name}", normed, layer))
        attention = g.add_attention(network, *projections, num_heads=cfg["num_heads"],
                                    head_dim=cfg["head_dim"], mask=mask)
        attention = linear("self_attn.out_proj", attention, layer)
        hidden = g.add_sum(network, hidden, attention)

        normed = g.add_layer_norm(network, hidden,
                                  weights[f"{layer}.layer_norm2.weight"],
                                  weights[f"{layer}.layer_norm2.bias"], cfg["eps"])
        inner = linear("mlp.fc1", normed, layer)
        inner = g.add_quick_gelu(network, inner, dtype=dtype)
        inner = linear("mlp.fc2", inner, layer)
        hidden = g.add_sum(network, hidden, inner)
        assert int(hidden.shape[-1]) == width
    return hidden


def causal_mask(network, seq, dtype=np.float32):
    """Additive ``[1, 1, seq, seq]`` mask, zero on and below the diagonal.

    The text tower is causal. The pooled row is the first end-of-text token, so
    nothing after it can reach the pooled row anyway - measured: passing the
    padding attention mask as well changes the pooled vector by 0.0. The causal
    mask itself is not optional; it changes every row before the pooled one.
    """
    values = np.triu(np.full((seq, seq), -1.0e4, dtype=np.float32), k=1)
    return g.add_constant(network, (1, 1, seq, seq), values, dtype=dtype)

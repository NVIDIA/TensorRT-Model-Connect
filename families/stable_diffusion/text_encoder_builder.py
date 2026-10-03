# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the CLIP text encoder that conditions the Stable Diffusion UNet.

Two details are load-bearing and neither is visible from the weight shapes:

* the activation is ``quick_gelu`` (``x * sigmoid(1.702 x)``), not the erf GELU
  the rest of this family uses;
* the encoder is **causal** — a token may not attend to later tokens. Without the
  mask the encoder still runs and still produces a plausible embedding.
"""

from __future__ import annotations

import math

import numpy as np

from . import graph as g


def build_text_encoder(network, token_ids, weights, cfg, dtype):
    """Token ids in, the final hidden states the UNet cross-attends to out."""
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = hidden // heads
    tokens = cfg["max_position_embeddings"]
    eps = cfg["layer_norm_eps"]

    token_table = g.add_constant(
        network, tuple(np.asarray(weights["text_model.embeddings.token_embedding.weight"]).shape),
        weights["text_model.embeddings.token_embedding.weight"], dtype=dtype)
    embedded = g.add_gather(network, token_table, token_ids, axis=0)

    position = np.asarray(
        weights["text_model.embeddings.position_embedding.weight"]).reshape(1, tokens, hidden)
    h = g.add_sum(network, embedded,
                  g.add_constant(network, (1, tokens, hidden), position, dtype=dtype))

    scale = 1.0 / math.sqrt(head_dim)
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"text_model.encoder.layers.{layer}"
        residual = h
        normed = g.add_layer_norm(network, h, weights[f"{prefix}.layer_norm1.weight"],
                                  weights[f"{prefix}.layer_norm1.bias"], eps, dtype=dtype)
        attn = f"{prefix}.self_attn"
        query = g.split_heads(network, g.add_linear(
            network, normed, weights[f"{attn}.q_proj.weight"],
            weights[f"{attn}.q_proj.bias"], dtype=dtype), tokens, heads, head_dim)
        key = g.split_heads(network, g.add_linear(
            network, normed, weights[f"{attn}.k_proj.weight"],
            weights[f"{attn}.k_proj.bias"], dtype=dtype), tokens, heads, head_dim)
        value = g.split_heads(network, g.add_linear(
            network, normed, weights[f"{attn}.v_proj.weight"],
            weights[f"{attn}.v_proj.bias"], dtype=dtype), tokens, heads, head_dim)
        context = g.merge_heads(
            network, g.add_attention_causal(network, query, key, value, scale, tokens, dtype=dtype),
            tokens, hidden)
        h = g.add_sum(network, g.add_linear(
            network, context, weights[f"{attn}.out_proj.weight"],
            weights[f"{attn}.out_proj.bias"], dtype=dtype), residual)

        residual = h
        normed = g.add_layer_norm(network, h, weights[f"{prefix}.layer_norm2.weight"],
                                  weights[f"{prefix}.layer_norm2.bias"], eps, dtype=dtype)
        inner = g.add_quick_gelu(network, g.add_linear(
            network, normed, weights[f"{prefix}.mlp.fc1.weight"],
            weights[f"{prefix}.mlp.fc1.bias"], dtype=dtype))
        h = g.add_sum(network, g.add_linear(
            network, inner, weights[f"{prefix}.mlp.fc2.weight"],
            weights[f"{prefix}.mlp.fc2.bias"], dtype=dtype), residual)

    return g.add_layer_norm(network, h, weights["text_model.final_layer_norm.weight"],
                            weights["text_model.final_layer_norm.bias"], eps, dtype=dtype)

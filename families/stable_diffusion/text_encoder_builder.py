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
import tensorrt as trt

from . import graph as g


def build_text_encoder(network, token_ids, weights, cfg, dtype, *,
                       penultimate=False, eos_index=None):
    """Token ids in, the hidden states the UNet cross-attends to out.

    SDXL conditions on the **penultimate** hidden state and does *not* apply
    the final layer norm to it: measured standard deviation 4.95 against 1.01
    for the normalised one. Its second encoder is a
    ``CLIPTextModelWithProjection`` whose pooled vector comes from the fully
    normalised last hidden state at the end-of-text position, so that encoder
    runs all its layers even though the tap stops one short.

    Returns ``(hidden_states, pooled)``; ``pooled`` is ``None`` unless
    ``eos_index`` is supplied.
    """
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

    activation = g.add_gelu if cfg.get("hidden_act") == "gelu" else g.add_quick_gelu
    scale = 1.0 / math.sqrt(head_dim)
    tapped = None
    total = cfg["num_hidden_layers"]
    for layer in range(total):
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
        inner = activation(network, g.add_linear(
            network, normed, weights[f"{prefix}.mlp.fc1.weight"],
            weights[f"{prefix}.mlp.fc1.bias"], dtype=dtype))
        h = g.add_sum(network, g.add_linear(
            network, inner, weights[f"{prefix}.mlp.fc2.weight"],
            weights[f"{prefix}.mlp.fc2.bias"], dtype=dtype), residual)

        if penultimate and layer == total - 2:
            # The tap is the raw hidden state, with no final layer norm.
            tapped = h

    normed = g.add_layer_norm(network, h, weights["text_model.final_layer_norm.weight"],
                              weights["text_model.final_layer_norm.bias"], eps, dtype=dtype)

    pooled = None
    if eos_index is not None:
        gathered = network.add_gather_v2(normed, eos_index, trt.GatherMode.DEFAULT)
        gathered.axis = 1
        pooled = g.add_linear(network, gathered.get_output(0),
                              weights["text_projection.weight"], None, dtype=dtype)

    return (tapped if penultimate else normed), pooled

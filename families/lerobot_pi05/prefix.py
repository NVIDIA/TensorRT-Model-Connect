# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefix graph: images and prompt tokens through Gemma-2B, emitting per-layer K/V.

The action expert never reads the prefix hidden states, only the keys and
values each prefix layer computed, so the last layer stops after its K/V
projection: its attention output and MLP feed nothing.
"""

from __future__ import annotations

import math

import tensorrt as trt

from . import graph, vision
from .checkpoint import embedding_key, require

WIDTH = 2048
DEPTH = 18
MLP = 16384
HEADS = 8
HEAD_DIM = 256
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
VOCAB = 257152

_ROOT = "paligemma.model.language_model."


def build_prefix(network, weights, cfg, *, dtype, debug=False, quant=None):
    """Declare the prefix engine's inputs and outputs on ``network``.

    Inputs: ``pixel_values[slots,3,H,W]`` in [0,1], ``token_ids[T]``,
    ``position_ids[P]`` and the additive ``attention_bias[P,P]``.
    Outputs: ``k_cache`` and ``v_cache``, ``[DEPTH, P, HEAD_DIM]``, keys after RoPE.
    ``quant`` (a ``quantization.Pi05QuantContext``) makes the Gemma projections FP8.
    """
    slots = cfg.camera_slots
    prompt = cfg.tokenizer_max_length
    prefix = slots * vision.TOKENS + prompt

    pixels = network.add_input("pixel_values", trt.float32, (slots, 3, cfg.camera_height, cfg.camera_width))
    token_ids = network.add_input("token_ids", trt.int32, (prompt,))
    positions = network.add_input("position_ids", trt.int32, (prefix,))
    bias = network.add_input("attention_bias", trt.float32, (prefix, prefix))

    images = vision.build_vision(
        network, pixels, weights, height=cfg.camera_height, width=cfg.camera_width, dtype=dtype)
    images = graph.shuffle(network, images, reshape=(slots * vision.TOKENS, WIDTH))

    table = require(weights, embedding_key(weights), (VOCAB, WIDTH))
    # The reference's embedding lookup already carries the sqrt(width) scale.
    lookup = network.add_gather(graph.const(network, table, dtype), token_ids, 0).get_output(0)
    language = graph.mul(network, lookup, graph.scalar(network, math.sqrt(WIDTH), 2, dtype))
    h = graph.concat(network, [graph.cast(network, images, dtype), language], 0)

    if debug:
        # Stage-by-stage parity checks read these; production bundles omit them.
        for name, tensor in (("debug_image_tokens", images), ("debug_prefix_embeds", h)):
            copy = graph.cast(network, tensor, trt.float32)
            copy = network.add_identity(copy).get_output(0)
            copy.name = name
            network.mark_output(copy)
    cos, sin = graph.rope_tables(network, positions, HEAD_DIM, ROPE_THETA)

    def project(x, name):
        if quant is None:
            return graph.linear(network, x, weights[name + ".weight"])
        return quant.linear(network, x, name, weights)

    keys, values = [], []
    for index in range(DEPTH):
        p = f"{_ROOT}layers.{index}."
        norm = graph.gemma_rms_norm(network, h, weights[p + "input_layernorm.weight"], RMS_EPS)
        k = graph.apply_rope(network, project(norm, p + "self_attn.k_proj"), cos, sin, 1, HEAD_DIM)
        v = project(norm, p + "self_attn.v_proj")
        keys.append(graph.shuffle(network, k, reshape=(1, prefix, HEAD_DIM)))
        values.append(graph.shuffle(network, v, reshape=(1, prefix, HEAD_DIM)))
        if index == DEPTH - 1:
            break
        q = graph.apply_rope(network, project(norm, p + "self_attn.q_proj"), cos, sin, HEADS, HEAD_DIM)
        context = graph.attention_mqa(network, q, k, v, bias, heads=HEADS, head_dim=HEAD_DIM)
        h = graph.add(network, h, project(context, p + "self_attn.o_proj"))
        norm = graph.gemma_rms_norm(network, h, weights[p + "post_attention_layernorm.weight"], RMS_EPS)
        gate = graph.gelu_tanh(network, project(norm, p + "mlp.gate_proj"))
        hidden = graph.mul(network, gate, project(norm, p + "mlp.up_proj"))
        h = graph.add(network, h, project(hidden, p + "mlp.down_proj"))

    k_cache = graph.concat(network, keys, 0)
    v_cache = graph.concat(network, values, 0)
    k_cache.name, v_cache.name = "k_cache", "v_cache"
    network.mark_output(k_cache)
    network.mark_output(v_cache)
    return prefix

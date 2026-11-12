# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conditioning engine: T5 text encoder, text projection and the two duration conditioners.

One request runs it once per prompt (and once more for a negative prompt). Its two
outputs are exactly what ``StableAudioPipeline`` hands the denoiser for one branch:

* ``encoder_hidden_states`` [1, L + 2, D]: the projected T5 states with padding rows
  zeroed, followed by the start and end duration tokens;
* ``global_hidden_states`` [1, 1, 2 D]: the start and end duration embeddings side by side.

It is always built in FP32: it runs once per request and T5 is prone to FP16 overflow.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .graph_ops import Graph

# T5's additive mask value for padded keys (torch.finfo(torch.float32).min).
_MASKED = float(np.finfo(np.float32).min)


def relative_position_buckets(length: int, num_buckets: int, max_distance: int) -> np.ndarray:
    """T5's bidirectional relative-position bucket of every (query, key) pair, [L, L].

    Mirrors ``T5Attention._relative_position_bucket`` including its float32 logarithm,
    so the bucket boundaries fall where transformers puts them.
    """
    relative = np.arange(length)[None, :] - np.arange(length)[:, None]
    half = num_buckets // 2
    buckets = (relative > 0).astype(np.int64) * half
    distance = np.abs(relative)
    exact = half // 2
    # Distances below `exact` take the exact branch; clamping keeps their unused log finite.
    scaled = (np.log(np.maximum(distance, 1).astype(np.float32) / np.float32(exact))
              / np.float32(math.log(max_distance / exact)) * np.float32(half - exact))
    large = np.minimum(exact + scaled.astype(np.int64), half - 1)
    return buckets + np.where(distance < exact, distance, large)


def _t5_encoder(g: Graph, input_ids, attention_mask, weights: dict, cfg: dict, length: int):
    heads, d_kv = cfg["num_heads"], cfg["d_kv"]
    inner = heads * d_kv
    table = g.constant(weights["shared.weight"], trt.float32)
    hidden = g.network.add_gather(table, input_ids, 0).get_output(0)  # [1, L, D]

    # Relative position bias (shared by every layer, taken from layer 0) plus the padding mask.
    bias_table = np.asarray(
        weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"], dtype=np.float32)
    buckets = relative_position_buckets(length, cfg["num_buckets"], cfg["max_distance"])
    position_bias = bias_table[buckets].transpose(2, 0, 1)[None]  # [1, H, L, L]
    keep = g.reshape(attention_mask, (1, 1, 1, length))
    padding = g.mul(g.sub(g.scalar(1.0, keep), keep), g.scalar(_MASKED, keep))
    score_bias = g.add(g.constant(position_bias, trt.float32), padding)

    for layer in range(cfg["num_layers"]):
        prefix = f"encoder.block.{layer}.layer"
        normed = g.rms_norm(hidden, weights[f"{prefix}.0.layer_norm.weight"], cfg["eps"])

        def heads_of(name):
            projected = g.linear(normed, weights[f"{prefix}.0.SelfAttention.{name}.weight"])
            return g.reshape(projected, (1, length, heads, d_kv), second=(0, 2, 1, 3))

        query, key, value = heads_of("q"), heads_of("k"), heads_of("v")
        # T5 does not scale its attention scores.
        scores = g.add(g.matmul(query, key, transpose_right=True), score_bias)
        context = g.matmul(g.softmax(scores, 3), value)
        context = g.reshape(context, (1, length, inner), first=(0, 2, 1, 3))
        hidden = g.add(hidden, g.linear(context, weights[f"{prefix}.0.SelfAttention.o.weight"]))

        normed = g.rms_norm(hidden, weights[f"{prefix}.1.layer_norm.weight"], cfg["eps"])
        inner_states = g.relu(g.linear(normed, weights[f"{prefix}.1.DenseReluDense.wi.weight"]))
        hidden = g.add(hidden, g.linear(inner_states, weights[f"{prefix}.1.DenseReluDense.wo.weight"]))
    return g.rms_norm(hidden, weights["encoder.final_layer_norm.weight"], cfg["eps"])


def _duration_embedding(g: Graph, seconds, weights: dict, prefix: str, projection: dict):
    """``StableAudioNumberConditioner``: clamp, normalise, Fourier features, linear -> [1, 1, D]."""
    low, high = projection["min_value"], projection["max_value"]
    clamped = g.minimum(g.maximum(seconds, g.scalar(low, seconds)), g.scalar(high, seconds))
    times = g.div(g.sub(clamped, g.scalar(low, clamped)), g.scalar(high - low, clamped))
    times = g.reshape(times, (1, 1))
    frequencies = np.asarray(weights[f"{prefix}.time_positional_embedding.0.weights"], dtype=np.float32)
    # Same operation order as StableAudioPositionalEmbedding: times * weights * 2 * pi.
    angles = g.mul(times, g.constant(frequencies, trt.float32, shape=(1, frequencies.size)))
    angles = g.mul(g.mul(angles, g.scalar(2.0, angles)), g.scalar(math.pi, angles))
    features = g.concat([times, g.unary(angles, trt.UnaryOperation.SIN),
                         g.unary(angles, trt.UnaryOperation.COS)], axis=1)
    embedded = g.linear(features, weights[f"{prefix}.time_positional_embedding.1.weight"],
                        weights[f"{prefix}.time_positional_embedding.1.bias"])
    return g.reshape(embedded, (1, 1, projection["conditioning_dim"]))


def populate(network: trt.INetworkDefinition, text_weights: dict, projection_weights: dict,
             cfg: dict) -> Graph:
    """Add the conditioning network to ``network`` and return the graph holding its constants."""
    g = Graph(network)
    length = cfg["tokenizer"]["max_length"]
    projection = cfg["projection"]
    input_ids = network.add_input("input_ids", trt.int32, (1, length))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, length))
    seconds_start = network.add_input("seconds_start", trt.float32, (1,))
    seconds_end = network.add_input("seconds_end", trt.float32, (1,))

    text = _t5_encoder(g, input_ids, attention_mask, text_weights, cfg["text_encoder"], length)
    if projection["text_projection"]:
        text = g.linear(text, projection_weights["text_projection.weight"],
                        projection_weights["text_projection.bias"])
    # The pipeline zeroes padded rows after the projection.
    text = g.mul(text, g.reshape(attention_mask, (1, length, 1)))

    start = _duration_embedding(g, seconds_start, projection_weights, "start_number_conditioner", projection)
    end = _duration_embedding(g, seconds_end, projection_weights, "end_number_conditioner", projection)
    encoder_states = g.concat([text, start, end], axis=1)
    global_states = g.concat([start, end], axis=2)
    encoder_states.name = "encoder_hidden_states"
    global_states.name = "global_hidden_states"
    network.mark_output(encoder_states)
    network.mark_output(global_states)
    return g

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 encoder graph of the Chroma text encoder.

Engine I/O: ``input_ids`` and ``attention_mask`` ``[1, L]`` int32 in,
``text_embeddings`` ``[1, L, d_model]`` float32 out.

Type flow of the reference (transformers ``T5EncoderModel`` loaded in BF16)
is kept: the feed-forward output projections ``wo`` stay FP32
(``_keep_in_fp32_modules``), so the residual stream is FP32 from the first
feed-forward block on, while every other projection runs in the compute type.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import tensorrt as trt

from .checkpoint import SafetensorsDirectory, TextEncoderConfig
from .graph_ops import Graph


def relative_position_buckets(length: int, num_buckets: int, max_distance: int) -> np.ndarray:
    """transformers ``T5Attention._relative_position_bucket`` (bidirectional), FP32 arithmetic."""
    context = np.arange(length, dtype=np.int64)[:, None]
    memory = np.arange(length, dtype=np.int64)[None, :]
    relative = memory - context
    half = num_buckets // 2
    buckets = (relative > 0).astype(np.int64) * half
    relative = np.abs(relative)
    max_exact = half // 2
    is_small = relative < max_exact
    safe = np.maximum(relative, 1).astype(np.float32)
    scaled = (
        np.log(safe / np.float32(max_exact))
        / np.float32(math.log(max_distance / max_exact))
        * np.float32(half - max_exact)
    )
    large = max_exact + scaled.astype(np.int64)
    large = np.minimum(large, half - 1)
    return buckets + np.where(is_small, relative, large)


def position_bias(weights: SafetensorsDirectory, config: TextEncoderConfig, length: int) -> np.ndarray:
    """``[heads, L, L]`` relative attention bias shared by every layer (stored dtype preserved)."""
    table = weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    if table.shape != (config.num_buckets, config.num_heads):
        raise ValueError(f"relative_attention_bias has shape {table.shape}")
    buckets = relative_position_buckets(length, config.num_buckets, config.max_distance)
    return np.ascontiguousarray(np.transpose(table[buckets], (2, 0, 1)))


def build_text_encoder_network(
    graph: Graph,
    weights: SafetensorsDirectory,
    config: TextEncoderConfig,
    length: int,
) -> None:
    network = graph.network
    compute = graph.dtype
    heads, head_dim, width = config.num_heads, config.d_kv, config.d_model
    input_ids = network.add_input("input_ids", trt.int32, (1, length))
    attention_mask = network.add_input("attention_mask", trt.int32, (1, length))

    embedding = graph.constant(weights["shared.weight"], compute)
    hidden = network.add_gather(embedding, graph.reshape(input_ids, (length,)), 0).get_output(0)
    hidden = graph.reshape(hidden, (1, length, width))

    # Padded keys get the most negative finite value of the compute type, as
    # transformers' bidirectional mask does; every row keeps the EOS key.
    valid = graph.cast(graph.reshape(attention_mask, (1, 1, 1, length)), trt.float32)
    lowest = float(np.finfo(np.float32).min if compute == trt.float32 else ml_dtypes.finfo(ml_dtypes.bfloat16).min)
    blocked = graph.mul(graph.sub(graph.scalar(1.0, valid), valid), graph.scalar(lowest, valid))
    bias = graph.constant(position_bias(weights, config, length)[None], compute)
    attention_bias = graph.add(bias, graph.cast(blocked, compute))

    def norm(tensor: trt.ITensor, name: str) -> trt.ITensor:
        normalized = graph.rms_norm(tensor, None, config.eps)  # FP32 statistics, cast to input type
        normalized = graph.cast(normalized, compute)
        scale = graph.constant(weights[name].reshape(1, 1, width), compute)
        return graph.mul(normalized, scale)

    for index in range(config.num_layers):
        prefix = f"encoder.block.{index}.layer"
        x = norm(hidden, f"{prefix}.0.layer_norm.weight")
        q, k, v = (
            graph.linear(x, weights[f"{prefix}.0.SelfAttention.{name}.weight"]) for name in ("q", "k", "v")
        )
        q, k, v = (graph.transpose(graph.reshape(item, (1, length, heads, head_dim)), (0, 2, 1, 3)) for item in (q, k, v))
        context = graph.attention(q, k, v, attention_bias)  # T5 attention is unscaled.
        context = graph.reshape(graph.transpose(context, (0, 2, 1, 3)), (1, length, heads * head_dim))
        attended = graph.linear(context, weights[f"{prefix}.0.SelfAttention.o.weight"])
        hidden = graph.add(hidden, graph.cast(attended, hidden.dtype))

        x = norm(hidden, f"{prefix}.1.layer_norm.weight")
        gate = graph.activation(
            graph.linear(x, weights[f"{prefix}.1.DenseReluDense.wi_0.weight"]), trt.ActivationType.GELU_TANH
        )
        linear = graph.linear(x, weights[f"{prefix}.1.DenseReluDense.wi_1.weight"])
        product = graph.mul(gate, linear)
        projected = graph.linear(product, weights[f"{prefix}.1.DenseReluDense.wo.weight"], dtype=trt.float32)
        hidden = graph.add(graph.cast(hidden, trt.float32), projected)

    output = norm(hidden, "encoder.final_layer_norm.weight")
    output = graph.cast(output, trt.float32)
    output.name = "text_embeddings"
    network.mark_output(output)

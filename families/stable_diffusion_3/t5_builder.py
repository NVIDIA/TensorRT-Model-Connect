# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 v1.1 encoder (``T5EncoderModel``) as one engine.

The pipeline runs the encoder without an attention mask, so every position,
padding included, attends to every other one. T5 does not scale attention
scores; the relative position bias of the first block is shared by all
blocks and is a build-time constant for the fixed sequence length.

A BF16 build keeps the feed-forward output projection ``wo`` in FP32. This is
the family's choice: transformers does the same only for FP16 loads
(``_keep_in_fp32_modules``), where T5 v1.1 activations overflow FP16.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .graph_ops import FP32, Graph, build_engine
from .weights import take


def relative_position_buckets(length: int, num_buckets: int, max_distance: int) -> np.ndarray:
    """Bidirectional T5 buckets of ``memory - query`` positions, [length, length] int64.

    The float32 arithmetic follows transformers' ``_relative_position_bucket``,
    including the truncation of the logarithmic part toward zero.
    """
    context = np.arange(length, dtype=np.int64)[:, None]
    memory = np.arange(length, dtype=np.int64)[None, :]
    relative = memory - context
    half = num_buckets // 2
    buckets = (relative > 0).astype(np.int64) * half
    distance = np.abs(relative)
    max_exact = half // 2
    # Distances below max_exact take the exact branch; clamping them keeps the
    # logarithm finite without changing the selected values.
    ratio = np.maximum(distance, max_exact).astype(np.float32) / np.float32(max_exact)
    logarithmic = (np.log(ratio) / np.float32(math.log(max_distance / max_exact))
                   * np.float32(half - max_exact)).astype(np.float32)
    large = max_exact + logarithmic.astype(np.int64)
    large = np.minimum(large, half - 1)
    return buckets + np.where(distance < max_exact, distance, large)


def position_bias(weights: dict, cfg: dict, length: int) -> np.ndarray:
    """[1, heads, length, length] additive attention bias, in the stored dtype."""
    table = np.asarray(take(
        weights, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"))
    buckets = relative_position_buckets(length, cfg["num_buckets"], cfg["max_distance"])
    values = table[buckets]  # [length, length, heads]
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def encode(graph: Graph, input_ids, weights: dict, cfg: dict, length: int):
    """Return the final hidden states [1, length, d_model] (FP32)."""
    heads = cfg["heads"]
    head_dim = cfg["head_dim"]
    eps = cfg["eps"]
    # wo stays FP32 in every build (module docstring).
    output_dtype = FP32

    table = graph.constant(take(weights, "shared.weight"))
    hidden = graph.fp32(graph.network.add_gather(table, input_ids, 0).get_output(0))
    bias = graph.constant(position_bias(weights, cfg, length))

    for index in range(cfg["layers"]):
        block = f"encoder.block.{index}.layer."
        normed = graph.rms_norm(hidden, eps, take(weights, block + "0.layer_norm.weight"))
        projections = [
            graph.split_heads(graph.linear(normed, take(weights, f"{block}0.SelfAttention.{name}.weight")),
                              heads, head_dim)
            for name in ("q", "k", "v")
        ]
        context = graph.attention(*projections, scale=1.0, bias=bias)
        context = graph.linear(graph.merge_heads(context, heads, head_dim),
                               take(weights, block + "0.SelfAttention.o.weight"))
        hidden = graph.add(hidden, context)

        normed = graph.rms_norm(hidden, eps, take(weights, block + "1.layer_norm.weight"))
        gate = graph.gelu_tanh(graph.linear(normed, take(weights, block + "1.DenseReluDense.wi_0.weight")))
        linear = graph.linear(normed, take(weights, block + "1.DenseReluDense.wi_1.weight"))
        feed = graph.linear(graph.mul(gate, linear), take(weights, block + "1.DenseReluDense.wo.weight"),
                            dtype=output_dtype)
        hidden = graph.add(hidden, feed)

    return graph.rms_norm(hidden, eps, take(weights, "encoder.final_layer_norm.weight"))


def build_t5_engine(weights: dict, cfg: dict, length: int, *, precision: str, verbose: bool):
    def populate(graph: Graph) -> None:
        input_ids = graph.network.add_input("input_ids", trt.int32, (1, length))
        graph.output(encode(graph, input_ids, weights, cfg, length), "hidden_states")

    return build_engine(populate, precision=precision, verbose=verbose)

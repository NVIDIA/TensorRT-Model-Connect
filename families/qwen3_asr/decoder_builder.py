# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine for the Qwen3 decoder of Qwen3-ASR, with an in-place KV cache.

The projected audio rows replace the ``<|audio_pad|>`` placeholder embeddings, as
``Qwen3ASRModel.forward`` does with ``masked_scatter``; the decoder is an ordinary Qwen3 language
model (RMSNorm, per-head q / k RMSNorm, rotate-half RoPE, grouped-query attention, SwiGLU, LM head
tied to the token embedding).

``IKVCacheUpdateLayer`` writes the step's keys and values into caches the runtime owns; each
``present_*`` output aliases its ``cache_*`` input. Attention reads the first ``R`` cache rows
(``R`` = length of ``key_positions``), masks keys after each query's position, and zeroes rows at or
after the valid length so stale or uninitialised cache memory cannot reach the result.

The plan has two optimisation profiles over the same weights: 0 runs up to ``prefill_tokens``
prompt tokens per launch (prefill; a longer prompt takes several launches, each appending to the
cache), 1 runs one token (decode).

Engine contract:
  ``input_ids``          [1, S] int32
  ``audio_embeds``       [1, S, hidden] FP32       replaces the embedding where ``audio_mask`` is 1
  ``audio_mask``         [1, S] int32
  ``cache_write_index``  [1] int32                 position of the first of the S tokens
  ``key_positions``      [R] int32                 0 .. R-1, R >= cache_write_index + S
  ``cache_k_<i>``, ``cache_v_<i>``  [1, kv_heads, capacity, head_dim] compute type
  ``logits``             [1, vocab] FP32           next-token scores after the last of the S tokens
  ``present_k_<i>``, ``present_v_<i>``             in-place aliases of the caches
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import tensorrt as trt

from . import trt_ops
from .config import DecoderConfig
from .trt_ops import BF16, FP32, Graph
from .weights import take

_OPTIMIZATION_LEVEL = 3
_WORKSPACE_BYTES = 8 << 30
_PREFIX = "model.language_model"
OPT_TOKENS = 512


def rope_tables(capacity: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """``cos`` / ``sin`` of ``position * theta^(-2j / head_dim)``, ``[capacity, head_dim / 2]`` float32.

    Float32 throughout, as ``Qwen3RotaryEmbedding`` computes them before casting to the hidden type.
    """
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    inverse = (np.float32(1.0) / np.power(np.float32(theta), exponent)).astype(np.float32)
    angles = (np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]).astype(np.float32)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rotate_half(graph: Graph, tensor: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    """``x * cos + rotate_half(x) * sin`` on ``[S, heads, head_dim]`` in the compute type."""
    half = int(tensor.shape[-1]) // 2
    first = trt_ops.slice_static(graph, tensor, 2, 0, half)
    second = trt_ops.slice_static(graph, tensor, 2, half, half)
    low = trt_ops.sub(graph, trt_ops.mul(graph, first, cos), trt_ops.mul(graph, second, sin))
    high = trt_ops.add(graph, trt_ops.mul(graph, second, cos), trt_ops.mul(graph, first, sin))
    return trt_ops.concat(graph, [low, high], 2)


class _Step:
    """Tensors every decoder layer shares within one launch."""

    def __init__(self, graph: Graph, config: DecoderConfig, capacity: int) -> None:
        network = graph.network
        self.input_ids = network.add_input("input_ids", trt.int32, (1, -1))
        self.audio_embeds = network.add_input("audio_embeds", FP32, (1, -1, config.hidden))
        self.audio_mask = network.add_input("audio_mask", trt.int32, (1, -1))
        self.write_index = network.add_input("cache_write_index", trt.int32, (1,))
        self.key_positions = network.add_input("key_positions", trt.int32, (-1,))
        self.length = trt_ops.dimension(graph, self.input_ids, 1)
        self.key_rows = trt_ops.dimension(graph, self.key_positions, 0)

        # Query positions: cache_write_index + 0 .. S-1.
        self.positions = trt_ops.add(graph, trt_ops.iota(graph, self.length), self.write_index)
        cos_table, sin_table = rope_tables(capacity, config.head_dim, config.rope_theta)
        half = config.head_dim // 2
        cos = network.add_gather(trt_ops.constant_compute(graph, cos_table), self.positions, 0)
        sin = network.add_gather(trt_ops.constant_compute(graph, sin_table), self.positions, 0)
        self.cos = trt_ops.reshape(graph, cos.get_output(0), (-1, 1, half))
        self.sin = trt_ops.reshape(graph, sin.get_output(0), (-1, 1, half))

        # BOOL [1, 1, 1, S, R]: key row after the query position.
        query = trt_ops.reshape(graph, self.positions, (1, 1, 1, -1, 1))
        keys = trt_ops.reshape(graph, self.key_positions, (1, 1, 1, 1, -1))
        self.blocked = trt_ops.less(graph, query, keys)
        # BOOL [1, 1, R, 1]: key row inside the valid prefix (position < write index + S).
        valid_length = trt_ops.add(
            graph, self.write_index, trt_ops.cast(graph, self.length, trt.int32)
        )
        self.active_rows = trt_ops.reshape(
            graph, trt_ops.less(graph, self.key_positions, valid_length), (1, 1, -1, 1)
        )


def _attention(
    graph: Graph,
    config: DecoderConfig,
    step: _Step,
    query: trt.ITensor,
    present_k: trt.ITensor,
    present_v: trt.ITensor,
) -> trt.ITensor:
    """Grouped-query attention of ``query [S, heads, head_dim]`` over the first R cache rows."""
    kv_heads = config.kv_heads
    groups = config.heads // kv_heads
    head_dim = config.head_dim
    zero = trt_ops.cast(graph, trt_ops.scalar_fp32(graph, 0.0, 4), graph.compute)

    def active(cache: trt.ITensor) -> trt.ITensor:
        rows = trt_ops.slice_dynamic(graph, cache, 2, 0, step.key_rows)
        return trt_ops.select(graph, step.active_rows, rows, zero)

    keys = trt_ops.reshape(graph, active(present_k), (1, kv_heads, 1, -1, head_dim))
    values = trt_ops.reshape(graph, active(present_v), (1, kv_heads, 1, -1, head_dim))
    # Query head h reads key / value head h // groups, as repeat_kv lays them out.
    grouped = trt_ops.reshape(
        graph, trt_ops.transpose(graph, query, (1, 0, 2)), (1, kv_heads, groups, -1, head_dim)
    )
    scores = graph.network.add_matrix_multiply(
        trt_ops.cast(graph, grouped, FP32),
        trt.MatrixOperation.NONE,
        trt_ops.cast(graph, keys, FP32),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = trt_ops.mul(graph, scores, trt_ops.scalar_fp32(graph, head_dim**-0.5, 5))
    scores = trt_ops.select(
        graph, step.blocked, trt_ops.scalar_fp32(graph, trt_ops.MASKED_SCORE, 5), scores
    )
    probabilities = trt_ops.cast(graph, trt_ops.softmax_last(graph, scores), graph.compute)
    context = graph.network.add_matrix_multiply(
        probabilities, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    ).get_output(0)
    context = trt_ops.reshape(graph, context, (config.heads, -1, head_dim))
    context = trt_ops.transpose(graph, context, (1, 0, 2))
    return trt_ops.reshape(graph, context, (-1, config.heads * head_dim))


def _layer(
    graph: Graph,
    config: DecoderConfig,
    weights: dict,
    index: int,
    step: _Step,
    hidden_states: trt.ITensor,
    capacity: int,
) -> trt.ITensor:
    network = graph.network
    prefix = f"{_PREFIX}.layers.{index}"
    hidden = config.hidden
    head_dim = config.head_dim
    width = config.heads * head_dim
    kv_width = config.kv_heads * head_dim

    def weight(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return take(weights, f"{prefix}.{name}", shape)

    normed = trt_ops.rms_norm(graph, hidden_states, weight("input_layernorm.weight", (hidden,)), config.eps)
    query = trt_ops.linear(graph, normed, weight("self_attn.q_proj.weight", (width, hidden)))
    key = trt_ops.linear(graph, normed, weight("self_attn.k_proj.weight", (kv_width, hidden)))
    value = trt_ops.linear(graph, normed, weight("self_attn.v_proj.weight", (kv_width, hidden)))
    query = trt_ops.reshape(graph, query, (-1, config.heads, head_dim))
    key = trt_ops.reshape(graph, key, (-1, config.kv_heads, head_dim))
    value = trt_ops.reshape(graph, value, (-1, config.kv_heads, head_dim))
    query = trt_ops.rms_norm(graph, query, weight("self_attn.q_norm.weight", (head_dim,)), config.eps)
    key = trt_ops.rms_norm(graph, key, weight("self_attn.k_norm.weight", (head_dim,)), config.eps)
    query = _rotate_half(graph, query, step.cos, step.sin)
    key = _rotate_half(graph, key, step.cos, step.sin)

    cache_shape = (1, config.kv_heads, capacity, head_dim)
    presents = []
    for name, rows in (("k", key), ("v", value)):
        cache = network.add_input(f"cache_{name}_{index}", graph.compute, cache_shape)
        update = trt_ops.reshape(
            graph, trt_ops.transpose(graph, rows, (1, 0, 2)), (1, config.kv_heads, -1, head_dim)
        )
        layer = network.add_kv_cache_update(cache, update, step.write_index, trt.KVCacheMode.LINEAR)
        if layer is None:
            raise RuntimeError("TensorRT could not create the Qwen3-ASR KV-cache update")
        present = layer.get_output(0)
        present.name = f"present_{name}_{index}"
        network.mark_output(present)
        presents.append(present)

    context = _attention(graph, config, step, query, presents[0], presents[1])
    hidden_states = trt_ops.add(
        graph, hidden_states, trt_ops.linear(graph, context, weight("self_attn.o_proj.weight", (hidden, width)))
    )
    normed = trt_ops.rms_norm(
        graph, hidden_states, weight("post_attention_layernorm.weight", (hidden,)), config.eps
    )
    inner = config.intermediate
    gate = trt_ops.silu(graph, trt_ops.linear(graph, normed, weight("mlp.gate_proj.weight", (inner, hidden))))
    up = trt_ops.linear(graph, normed, weight("mlp.up_proj.weight", (inner, hidden)))
    feed_forward = trt_ops.linear(
        graph, trt_ops.mul(graph, gate, up), weight("mlp.down_proj.weight", (hidden, inner))
    )
    return trt_ops.add(graph, hidden_states, feed_forward)


def _profile(graph: Graph, config: DecoderConfig, capacity: int, tokens: tuple[int, int, int]) -> None:
    profile = graph.builder.create_optimization_profile()
    low, opt, high = tokens
    profile.set_shape("input_ids", (1, low), (1, opt), (1, high))
    profile.set_shape("audio_embeds", (1, low, config.hidden), (1, opt, config.hidden), (1, high, config.hidden))
    profile.set_shape("audio_mask", (1, low), (1, opt), (1, high))
    rows = min(OPT_TOKENS, capacity)
    profile.set_shape("key_positions", (1,), (rows,), (capacity,))
    graph.config.add_optimization_profile(profile)


def build_decoder_plan(
    config: DecoderConfig,
    weights: dict,
    output: BinaryIO,
    *,
    capacity: int,
    prefill_tokens: int,
    verbose: bool,
    compute: trt.DataType = BF16,
) -> None:
    """Build the decoder with a KV cache of ``capacity`` positions and write the plan.

    ``prefill_tokens`` bounds the tokens of one prefill launch.
    """
    if not 1 <= capacity <= config.max_positions:
        raise ValueError(f"decoder capacity must be in [1, {config.max_positions}]")
    if not 1 <= prefill_tokens <= capacity:
        raise ValueError("prefill tokens per launch must be in [1, capacity]")
    graph = trt_ops.create_graph(
        verbose=verbose,
        workspace_bytes=_WORKSPACE_BYTES,
        optimization_level=_OPTIMIZATION_LEVEL,
        compute=compute,
    )
    network = graph.network
    step = _Step(graph, config, capacity)

    table = trt_ops.constant_compute(
        graph, take(weights, f"{_PREFIX}.embed_tokens.weight", (config.vocab_size, config.hidden))
    )
    token_ids = trt_ops.reshape(graph, step.input_ids, (-1,))
    embeds = network.add_gather(table, token_ids, 0).get_output(0)
    audio = trt_ops.cast(graph, trt_ops.reshape(graph, step.audio_embeds, (-1, config.hidden)), graph.compute)
    is_audio = trt_ops.less(
        graph, trt_ops.constant(graph, np.zeros((1, 1), dtype=np.int32)),
        trt_ops.reshape(graph, step.audio_mask, (-1, 1)),
    )
    hidden_states = trt_ops.select(graph, is_audio, audio, embeds)

    for index in range(config.layers):
        hidden_states = _layer(graph, config, weights, index, step, hidden_states, capacity)

    hidden_states = trt_ops.rms_norm(
        graph, hidden_states, take(weights, f"{_PREFIX}.norm.weight", (config.hidden,)), config.eps
    )
    last_index = trt_ops.sub(
        graph, step.length, trt_ops.constant(graph, np.ones((1,), dtype=np.int64))
    )
    last = trt_ops.slice_dynamic(
        graph, hidden_states, 0, last_index, trt_ops.constant(graph, np.ones((1,), dtype=np.int64))
    )
    logits = network.add_matrix_multiply(
        last, trt.MatrixOperation.NONE, table, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    logits = trt_ops.cast(graph, logits, FP32)
    logits.name = "logits"
    network.mark_output(logits)

    _profile(graph, config, capacity, (1, min(OPT_TOKENS, prefill_tokens), prefill_tokens))
    _profile(graph, config, capacity, (1, 1, 1))
    output.write(memoryview(trt_ops.serialize(graph, "decoder")))

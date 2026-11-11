# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2 decoder of VibeVoice-ASR with a TensorRT native KV cache.

One engine, two optimization profiles over the same weights: profile 0 takes
1..``max_prefill_rows`` prompt rows, profile 1 exactly one decode row.

Inputs
  input_ids          int32 [S]          token ids; speech rows may hold any valid id
  audio_embeds       bf16  [S, hidden]  connector output for speech rows (others ignored)
  audio_mask         int32 [S]          1 selects ``audio_embeds``, 0 the token embedding
  cache_write_index  int32 [1]          cache row of the first input row (its position)
  cache_k_<i>, cache_v_<i>  fp32 [1, kv_heads, capacity, head_dim], user-owned
Outputs
  logits             fp32  [1, vocab]   for the last input row (BF16 lm_head, cast)
  present_k_<i>, present_v_<i>          alias the cache inputs (updated in place)

Weights, activations and residuals stay BF16 like the reference; RMSNorm, SiLU
and the attention scores, softmax and value sum run in FP32 (the reference's
SDPA accumulates in FP32) and round to BF16 where the reference's tensors do.
RoPE is one FP32 rotation rounded once to BF16, where the reference rounds the
BF16 cos / sin and each BF16 product and sum. The KV cache holds the BF16
keys and values in FP32 storage, so attention reads them without a per-step
widening; the weighted value sum is split over ``KEY_BLOCK`` keys.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import DecoderConfig, Weights
from .graph import Graph, new_builder, serialize

_LM = "model.language_model"
_MASKED_SCORE = -1.0e30
# Keys per block of the split value sum; the KV capacity is a whole number of blocks.
KEY_BLOCK = 512


def layer_name(base: str, index: int) -> str:
    return f"{base}_{index}"


def rope_tables(cfg: DecoderConfig, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """FP32 cos / sin ``[capacity, head_dim / 2]`` as Qwen2RotaryEmbedding computes them."""
    exponent = np.arange(0, cfg.head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(
        cfg.head_dim
    )
    inv_freq = (np.float32(1.0) / np.power(np.float32(cfg.rope_theta), exponent)).astype(np.float32)
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class _Decoder:
    def __init__(self, graph: Graph, cfg: DecoderConfig, weights: Weights, capacity: int) -> None:
        self.g = graph
        self.cfg = cfg
        self.w = weights
        self.capacity = capacity

    def norm(self, rows: trt.ITensor, name: str) -> trt.ITensor:
        hidden = self.cfg.hidden_size
        return self.g.rms_norm(
            rows, self.w.bf16(name, (hidden,)), self.cfg.rms_norm_eps, 1, (1, hidden)
        )

    def projection(
        self, rows: trt.ITensor, name: str, out_features: int, in_features: int, bias: bool
    ) -> trt.ITensor:
        return self.g.linear(
            rows,
            self.w.bf16(f"{name}.weight", (out_features, in_features)),
            self.w.bf16(f"{name}.bias", (out_features,)) if bias else None,
        )

    def heads(self, rows: trt.ITensor, num_heads: int) -> trt.ITensor:
        """``[S, H * D]`` -> ``[1, H, S, D]``."""
        return self.g.reshape(rows, (1, -1, num_heads, self.cfg.head_dim), (0, 2, 1, 3))

    def rope(self, heads: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
        """Rotate-half RoPE computed in FP32 and rounded once to BF16."""
        g = self.g
        half = self.cfg.head_dim // 2
        values = g.cast(heads, trt.float32)
        size = g.concat([g.dim(values, 0), g.dim(values, 1), g.dim(values, 2), g.i64([half])])
        first = g.network.add_slice(values, (0, 0, 0, 0), (1, 1, 1, half), (1, 1, 1, 1))
        first.set_input(2, size)
        second = g.network.add_slice(values, (0, 0, 0, half), (1, 1, 1, half), (1, 1, 1, 1))
        second.set_input(2, size)
        x1, x2 = first.get_output(0), second.get_output(0)
        out1 = g.sub(g.mul(x1, cos), g.mul(x2, sin))
        out2 = g.add(g.mul(x2, cos), g.mul(x1, sin))
        return g.cast(g.concat([out1, out2], axis=3), trt.bfloat16)

    def attention(
        self, query: trt.ITensor, keys: trt.ITensor, values: trt.ITensor, mask: trt.ITensor
    ) -> trt.ITensor:
        """GQA over the whole FP32 cache; masked keys get zero weight.

        The query heads that share one KV head become rows of a single product
        with that head's cache, so no operand is broadcast or copied.
        """
        g = self.g
        cfg = self.cfg
        groups = cfg.num_heads // cfg.num_kv_heads
        rows = g.reshape(g.cast(query, trt.float32), (1, cfg.num_kv_heads, -1, cfg.head_dim))
        scores = g.network.add_matrix_multiply(
            rows, trt.MatrixOperation.NONE, keys, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = g.mul(scores, g.scalar(1.0 / float(np.sqrt(cfg.head_dim)), trt.float32, 4))
        scores = g.reshape(scores, (1, cfg.num_kv_heads, groups, -1, self.capacity))
        scores = g.network.add_select(
            mask, scores, g.scalar(_MASKED_SCORE, trt.float32, 5)
        ).get_output(0)
        softmax = g.network.add_softmax(scores)
        softmax.axes = 1 << 4
        # Weighted value sum, split over key blocks: one product per block, then
        # a sum over blocks. A single product over the whole cache has a tiny
        # output and a capacity-long reduction, which leaves the GPU idle.
        blocks = self.capacity // KEY_BLOCK
        weights = g.reshape(softmax.get_output(0), (1, cfg.num_kv_heads, -1, blocks, KEY_BLOCK))
        weights = g.transpose(weights, (0, 1, 3, 2, 4))
        values = g.reshape(values, (1, cfg.num_kv_heads, blocks, KEY_BLOCK, cfg.head_dim))
        partial = g.network.add_matrix_multiply(
            weights, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
        ).get_output(0)
        context = g.network.add_reduce(partial, trt.ReduceOperation.SUM, 1 << 2, False).get_output(
            0
        )
        context = g.reshape(context, (1, cfg.num_heads, -1, cfg.head_dim))
        rows = g.reshape(context, (-1, cfg.hidden_size), (0, 2, 1, 3), transpose_first=True)
        return g.cast(rows, trt.bfloat16)

    def mlp(self, rows: trt.ITensor, prefix: str) -> trt.ITensor:
        g = self.g
        cfg = self.cfg
        gate = self.projection(
            rows, f"{prefix}.gate_proj", cfg.intermediate_size, cfg.hidden_size, False
        )
        up = self.projection(
            rows, f"{prefix}.up_proj", cfg.intermediate_size, cfg.hidden_size, False
        )
        gate32 = g.cast(gate, trt.float32)
        silu = g.cast(g.mul(gate32, g.activation(gate32, trt.ActivationType.SIGMOID)), trt.bfloat16)
        return self.projection(
            g.mul(silu, up), f"{prefix}.down_proj", cfg.hidden_size, cfg.intermediate_size, False
        )


def build_decoder_engine(
    cfg: DecoderConfig,
    weights: Weights,
    *,
    capacity: int,
    max_prefill_rows: int,
    verbose: bool = False,
) -> bytes:
    if capacity % KEY_BLOCK or not KEY_BLOCK <= capacity <= cfg.max_position_embeddings:
        raise ValueError(
            f"vibevoice_asr KV capacity {capacity} must be a multiple of {KEY_BLOCK} in "
            f"[{KEY_BLOCK}, {cfg.max_position_embeddings}]"
        )
    max_prefill_rows = max(1, min(max_prefill_rows, capacity))
    builder, network, config = new_builder(verbose, optimization_level=3, workspace_bytes=16 << 30)
    g = Graph(network)
    d = _Decoder(g, cfg, weights, capacity)
    hidden = cfg.hidden_size
    kv_size = cfg.num_kv_heads * cfg.head_dim

    input_ids = network.add_input("input_ids", trt.int32, (-1,))
    audio_embeds = network.add_input("audio_embeds", trt.bfloat16, (-1, hidden))
    audio_mask = network.add_input("audio_mask", trt.int32, (-1,))
    write_index = network.add_input("cache_write_index", trt.int32, (1,))
    cache_shape = (1, cfg.num_kv_heads, capacity, cfg.head_dim)
    caches = [
        (
            network.add_input(layer_name("cache_k", i), trt.float32, cache_shape),
            network.add_input(layer_name("cache_v", i), trt.float32, cache_shape),
        )
        for i in range(cfg.num_layers)
    ]

    # Token embeddings, with the connector rows substituted at speech positions.
    table = weights.bf16(f"{_LM}.embed_tokens.weight", (cfg.vocab_size, hidden))
    tokens = network.add_gather(g.constant(table), input_ids, 0).get_output(0)
    speech = g.binary(audio_mask, g.scalar(0, trt.int32, 1), trt.ElementWiseOperation.GREATER)
    speech = g.reshape(speech, (-1, 1))
    rows = network.add_select(speech, audio_embeds, tokens).get_output(0)

    # Absolute positions of the input rows; RoPE and the causal mask follow them.
    count = g.dim(input_ids, 0)
    iota = network.add_fill((1,), trt.FillOperation.LINSPACE, trt.int32)
    iota.set_input(0, count)
    iota.set_input(1, g.constant(np.array(0, dtype=np.int32)))
    iota.set_input(2, g.constant(np.array([1], dtype=np.int32)))
    positions = g.add(iota.get_output(0), write_index)
    cos_table, sin_table = rope_tables(cfg, capacity)
    cos = g.reshape(
        network.add_gather(g.constant(cos_table), positions, 0).get_output(0),
        (1, 1, -1, cfg.head_dim // 2),
    )
    sin = g.reshape(
        network.add_gather(g.constant(sin_table), positions, 0).get_output(0),
        (1, 1, -1, cfg.head_dim // 2),
    )
    limits = g.reshape(g.add(positions, g.scalar(1, trt.int32, 1)), (-1, 1))
    keys = g.constant(np.arange(capacity, dtype=np.int32).reshape(1, capacity))
    visible = g.binary(keys, limits, trt.ElementWiseOperation.LESS)
    mask = g.reshape(visible, (1, 1, 1, -1, capacity))

    for index, (cache_k, cache_v) in enumerate(caches):
        prefix = f"{_LM}.layers.{index}"
        normed = d.norm(rows, f"{prefix}.input_layernorm.weight")
        query = d.projection(normed, f"{prefix}.self_attn.q_proj", hidden, hidden, True)
        key = d.projection(normed, f"{prefix}.self_attn.k_proj", kv_size, hidden, True)
        value = d.projection(normed, f"{prefix}.self_attn.v_proj", kv_size, hidden, True)
        query = d.rope(d.heads(query, cfg.num_heads), cos, sin)
        key = d.rope(d.heads(key, cfg.num_kv_heads), cos, sin)
        value = d.heads(value, cfg.num_kv_heads)
        # The cache stores the BF16-rounded K / V the reference caches, widened
        # to FP32 once on write so attention reads it without per-step casts.
        updated_k = network.add_kv_cache_update(
            cache_k, g.cast(key, trt.float32), write_index, trt.KVCacheMode.LINEAR
        ).get_output(0)
        updated_v = network.add_kv_cache_update(
            cache_v, g.cast(value, trt.float32), write_index, trt.KVCacheMode.LINEAR
        ).get_output(0)
        updated_k.name = layer_name("present_k", index)
        updated_v.name = layer_name("present_v", index)
        network.mark_output(updated_k)
        network.mark_output(updated_v)
        attended = d.projection(
            d.attention(query, updated_k, updated_v, mask),
            f"{prefix}.self_attn.o_proj",
            hidden,
            hidden,
            False,
        )
        rows = g.add(rows, attended)
        rows = g.add(
            rows, d.mlp(d.norm(rows, f"{prefix}.post_attention_layernorm.weight"), f"{prefix}.mlp")
        )

    last_start = g.concat([g.sub(count, g.i64([1])), g.i64([0])])
    last = network.add_slice(rows, (0, 0), (1, hidden), (1, 1))
    last.set_input(1, last_start)
    final = d.norm(last.get_output(0), f"{_LM}.norm.weight")
    logits = g.linear(final, weights.bf16("lm_head.weight", (cfg.vocab_size, hidden)), None)
    logits = g.cast(logits, trt.float32)
    logits.name = "logits"
    network.mark_output(logits)

    for rows_min, rows_opt, rows_max in ((1, max_prefill_rows, max_prefill_rows), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        profile.set_shape("input_ids", (rows_min,), (rows_opt,), (rows_max,))
        profile.set_shape(
            "audio_embeds", (rows_min, hidden), (rows_opt, hidden), (rows_max, hidden)
        )
        profile.set_shape("audio_mask", (rows_min,), (rows_opt,), (rows_max,))
        config.add_optimization_profile(profile)
    return serialize(builder, network, config, "decoder")

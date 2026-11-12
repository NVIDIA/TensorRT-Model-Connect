# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text engine: the Qwen3 decoder with in-engine audio injection and a linear KV cache.

Tensor contract (Sq = query rows of one call, A = audio rows, R = attended cache
rows, S = cache capacity):

    input_ids          int32 [Sq]          prompt or generated token ids
    audio_index        int32 [Sq]          row of ``audio_embeds`` for audio
                                           placeholders, -1 for text tokens
    audio_embeds       work  [A, hidden]   audio engine output of the request
    position_ids       int32 [Sq]          absolute positions (= cache rows)
    cache_write_index  int32 [1]           first cache row written by this call
    key_positions      int32 [R]           0..R-1, R >= valid rows after the call
    cache_k_<i>/v_<i>  work  [1, kv_heads, S, head_dim]   user-owned cache
  outputs
    logits             fp32  [1, vocab]    last query row only
    present_k_<i>/v_<i>                    alias the cache inputs (in-place update)

Profile 0 serves prefill chunks (Sq up to the prefill limit), profile 1 serves
one-token decode steps. Embeddings are gathered in the engine and the audio
rows replace the placeholder rows, as ``masked_scatter`` does in the reference.
"""

from __future__ import annotations

import sys

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .config import MossConfig
from .graph import Graph, builder_config, new_network

_TEXT = "model.language_model."
EMBEDDING = f"{_TEXT}embed_tokens.weight"

# Query rows of one prefill call. Physical KV capacity is independent; longer
# prompts are prefilled in consecutive calls.
PREFILL_CHUNK_TOKENS = 256
# Tactic point for the attended cache rows.
KEY_ROWS_OPTIMUM = 1024
# Score assigned to masked keys before the FP32 softmax.
_MASKED_SCORE = -1.0e30


def rope_inverse_frequencies(head_dim: int, theta: float) -> np.ndarray:
    """Qwen3's FP32 inverse frequencies, computed with PyTorch on the CPU as Hugging Face does."""
    import torch

    with torch.no_grad():
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (exponents / head_dim))
    return np.asarray(inv_freq.numpy(), dtype=np.float32).copy()


def _shape_of(g: Graph, tensor: trt.ITensor) -> trt.ITensor:
    return g.net.add_shape(tensor).get_output(0)


def _prepend_ones(g: Graph, tensor: trt.ITensor, count: int) -> trt.ITensor:
    """Reshape a dynamically shaped tensor to ``[1] * count + shape``."""
    target = g.net.add_concatenation([g.i64(np.ones(count, dtype=np.int64)), _shape_of(g, tensor)])
    target.axis = 0
    layer = g.net.add_shuffle(tensor)
    layer.set_input(1, target.get_output(0))
    return layer.get_output(0)


def _rope_tables(g: Graph, position_ids: trt.ITensor, inv_freq: np.ndarray):
    """cos / sin [Sq, 1, head_dim] in the work precision, built in FP32 like Qwen3RotaryEmbedding."""
    positions = g.reshape(g.cast(position_ids, trt.float32), (-1, 1))
    angles = g.mul(positions, g.f32(inv_freq.reshape(1, -1)))
    tables = []
    for op in (trt.UnaryOperation.COS, trt.UnaryOperation.SIN):
        half = g.unary(angles, op)
        full = g.net.add_concatenation([half, half])
        full.axis = 1
        tables.append(g.reshape(g.cast(full.get_output(0), g.work), (-1, 1, inv_freq.size * 2)))
    return tables[0], tables[1]


def _apply_rope(g: Graph, x: trt.ITensor, heads: int, head_dim: int, cos: trt.ITensor,
                sin: trt.ITensor) -> trt.ITensor:
    """``x * cos + rotate_half(x) * sin`` on [Sq, heads, head_dim], rounding per operation."""
    halves = g.reshape(x, (-1, heads, 2, head_dim // 2))
    swapped = g.net.add_gather(halves, g.i32(np.array([1, 0], dtype=np.int32)), 2).get_output(0)
    sign = g.work_scalar(1.0, 4)
    negative = g.work_scalar(-1.0, 4)
    signs = g.net.add_concatenation([negative, sign])
    signs.axis = 2
    rotated = g.reshape(g.mul(swapped, signs.get_output(0)), (-1, heads, head_dim))
    return g.add(g.mul(x, cos), g.mul(rotated, sin))


def _active_rows(g: Graph, cache: trt.ITensor, key_positions: trt.ITensor) -> trt.ITensor:
    """The first ``len(key_positions)`` rows of a [1, H, S, D] cache."""
    shape = _shape_of(g, cache)
    leading = g.net.add_slice(shape, start=(0,), shape=(2,), stride=(1,)).get_output(0)
    trailing = g.net.add_slice(shape, start=(3,), shape=(1,), stride=(1,)).get_output(0)
    rows = _shape_of(g, key_positions)
    size = g.net.add_concatenation([leading, rows, trailing])
    size.axis = 0
    layer = g.net.add_slice(cache, start=(0, 0, 0, 0), shape=(1, 1, 1, 1), stride=(1, 1, 1, 1))
    layer.set_input(2, size.get_output(0))
    return layer.get_output(0)


def _cache_attention(g: Graph, query: trt.ITensor, key: trt.ITensor, value: trt.ITensor,
                     cache_k: trt.ITensor, cache_v: trt.ITensor, write_index: trt.ITensor,
                     key_positions: trt.ITensor, mask: trt.ITensor, config: MossConfig):
    """Append this call's K/V rows, then attend over the active cache prefix.

    Scores and softmax are FP32; the probabilities are rounded to the work
    precision before the value product, as the fused attention kernels of the
    reference do.
    """
    text = config.text
    groups = text.num_heads // text.num_kv_heads

    def as_cache_rows(rows: trt.ITensor) -> trt.ITensor:
        heads_first = g.transpose(rows, (1, 0, 2))
        return g.reshape(heads_first, (1, text.num_kv_heads, -1, text.head_dim))

    update_k = g.net.add_kv_cache_update(
        cache_k, as_cache_rows(key), write_index, trt.KVCacheMode.LINEAR
    )
    update_v = g.net.add_kv_cache_update(
        cache_v, as_cache_rows(value), write_index, trt.KVCacheMode.LINEAR
    )
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT rejected the KV-cache update")
    present_k, present_v = update_k.get_output(0), update_v.get_output(0)

    keys = g.reshape(_active_rows(g, present_k, key_positions),
                     (1, text.num_kv_heads, 1, -1, text.head_dim))
    values = g.reshape(_active_rows(g, present_v, key_positions),
                       (1, text.num_kv_heads, 1, -1, text.head_dim))
    queries = g.reshape(
        g.transpose(query, (1, 0, 2)), (1, text.num_kv_heads, groups, -1, text.head_dim)
    )
    scale = g.f32(np.full((1, 1, 1, 1, 1), text.head_dim ** -0.5, dtype=np.float32))
    scores = g.net.add_matrix_multiply(
        g.mul(g.cast(queries, trt.float32), scale), trt.MatrixOperation.NONE,
        g.cast(keys, trt.float32), trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    blocked = g.f32(np.full((1, 1, 1, 1, 1), _MASKED_SCORE, dtype=np.float32))
    masked = g.net.add_select(mask, scores, blocked).get_output(0)
    softmax = g.net.add_softmax(masked)
    softmax.axes = 1 << 4
    probabilities = g.cast(softmax.get_output(0), g.work)
    context = g.net.add_matrix_multiply(
        probabilities, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    ).get_output(0)
    context = g.reshape(context, (text.num_heads, -1, text.head_dim))
    context = g.reshape(g.transpose(context, (1, 0, 2)), (-1, text.num_heads * text.head_dim))
    return context, present_k, present_v


def _decoder_layer(g: Graph, hidden: trt.ITensor, index: int, config: MossConfig, cos, sin,
                   cache_k, cache_v, write_index, key_positions, mask):
    text = config.text
    prefix = f"{_TEXT}layers.{index}"
    normed = g.rms_norm(hidden, f"{prefix}.input_layernorm.weight", text.rms_norm_eps)
    query = g.reshape(g.linear(normed, f"{prefix}.self_attn.q_proj.weight"),
                      (-1, text.num_heads, text.head_dim))
    key = g.reshape(g.linear(normed, f"{prefix}.self_attn.k_proj.weight"),
                    (-1, text.num_kv_heads, text.head_dim))
    value = g.reshape(g.linear(normed, f"{prefix}.self_attn.v_proj.weight"),
                      (-1, text.num_kv_heads, text.head_dim))
    query = g.rms_norm(query, f"{prefix}.self_attn.q_norm.weight", text.rms_norm_eps)
    key = g.rms_norm(key, f"{prefix}.self_attn.k_norm.weight", text.rms_norm_eps)
    query = _apply_rope(g, query, text.num_heads, text.head_dim, cos, sin)
    key = _apply_rope(g, key, text.num_kv_heads, text.head_dim, cos, sin)
    context, present_k, present_v = _cache_attention(
        g, query, key, value, cache_k, cache_v, write_index, key_positions, mask, config
    )
    hidden = g.add(hidden, g.linear(context, f"{prefix}.self_attn.o_proj.weight"))
    normed = g.rms_norm(hidden, f"{prefix}.post_attention_layernorm.weight", text.rms_norm_eps)
    gate = g.silu(g.linear(normed, f"{prefix}.mlp.gate_proj.weight"))
    up = g.linear(normed, f"{prefix}.mlp.up_proj.weight")
    down = g.linear(g.mul(gate, up), f"{prefix}.mlp.down_proj.weight")
    return g.add(hidden, down), present_k, present_v


def _embed(g: Graph, table: trt.ITensor, input_ids: trt.ITensor, audio_index: trt.ITensor,
           audio_embeds: trt.ITensor) -> trt.ITensor:
    text_rows = g.net.add_gather(table, input_ids, 0).get_output(0)
    is_audio = g.binary(audio_index, g.i32(np.array([-1], dtype=np.int32)),
                        trt.ElementWiseOperation.GREATER)
    rows = g.binary(audio_index, g.i32(np.array([0], dtype=np.int32)),
                    trt.ElementWiseOperation.MAX)
    audio_rows = g.net.add_gather(audio_embeds, rows, 0).get_output(0)
    return g.net.add_select(g.reshape(is_audio, (-1, 1)), audio_rows, text_rows).get_output(0)


def _last_row(g: Graph, hidden: trt.ITensor, width: int) -> trt.ITensor:
    shape = _shape_of(g, hidden)
    start = g.binary(shape, g.i64(np.array([1, width], dtype=np.int64)),
                     trt.ElementWiseOperation.SUB)
    layer = g.net.add_slice(hidden, start=(0, 0), shape=(1, width), stride=(1, 1))
    layer.set_input(1, start)
    return layer.get_output(0)


def build_text_engine(checkpoint: Checkpoint, config: MossConfig, precision: str,
                      capacity: int, *, verbose: bool = False) -> bytes:
    """Serialize the two-profile decoder engine for a cache of ``capacity`` rows."""
    text = config.text
    if capacity < 2:
        raise ValueError("the KV cache must hold at least two rows")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(builder)
    g = Graph(network, checkpoint, precision)

    input_ids = network.add_input("input_ids", trt.int32, (-1,))
    audio_index = network.add_input("audio_index", trt.int32, (-1,))
    audio_embeds = network.add_input("audio_embeds", g.work, (-1, text.hidden_size))
    position_ids = network.add_input("position_ids", trt.int32, (-1,))
    write_index = network.add_input("cache_write_index", trt.int32, (1,))
    key_positions = network.add_input("key_positions", trt.int32, (-1,))
    cache_shape = (1, text.num_kv_heads, capacity, text.head_dim)
    caches = [
        (network.add_input(f"cache_k_{i}", g.work, cache_shape),
         network.add_input(f"cache_v_{i}", g.work, cache_shape))
        for i in range(text.num_layers)
    ]

    trt_config = builder_config(builder)
    prefill = min(PREFILL_CHUNK_TOKENS, capacity)
    key_opt = min(KEY_ROWS_OPTIMUM, capacity)
    # Profile 0: prefill chunks over the whole audio of a request; profile 1:
    # one-token decode steps with a one-row audio placeholder.
    for rows, audio_rows in ((prefill, capacity), (1, 1)):
        profile = builder.create_optimization_profile()
        for name in ("input_ids", "audio_index", "position_ids"):
            profile.set_shape(name, (1,), (rows,), (rows,))
        profile.set_shape("audio_embeds", (1, text.hidden_size), (audio_rows, text.hidden_size),
                          (audio_rows, text.hidden_size))
        profile.set_shape("key_positions", (1,), (key_opt,), (capacity,))
        trt_config.add_optimization_profile(profile)

    table = g.weight(EMBEDDING)
    hidden = _embed(g, table, input_ids, audio_index, audio_embeds)
    cos, sin = _rope_tables(g, position_ids,
                            rope_inverse_frequencies(text.head_dim, text.rope_theta))
    # Causal mask over cache rows: key row k is visible to the query at absolute
    # position p when k <= p. Rows past the valid prefix are never visible.
    hidden_keys = g.binary(g.reshape(key_positions, (1, -1)), g.reshape(position_ids, (-1, 1)),
                           trt.ElementWiseOperation.GREATER)
    mask = _prepend_ones(g, g.unary(hidden_keys, trt.UnaryOperation.NOT), 3)
    presents = []
    for index, (cache_k, cache_v) in enumerate(caches):
        hidden, present_k, present_v = _decoder_layer(
            g, hidden, index, config, cos, sin, cache_k, cache_v, write_index, key_positions, mask
        )
        presents.append((present_k, present_v))

    hidden = g.rms_norm(_last_row(g, hidden, text.hidden_size), f"{_TEXT}norm.weight",
                        text.rms_norm_eps)
    logits = network.add_matrix_multiply(
        hidden, trt.MatrixOperation.NONE, table, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    logits = g.cast(logits, trt.float32)
    logits.name = "logits"
    network.mark_output(logits)
    for index, (present_k, present_v) in enumerate(presents):
        present_k.name = f"present_k_{index}"
        present_v.name = f"present_v_{index}"
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            f"[trtmc build] moss_transcribe_diarize text engine: {text.num_layers} layers, "
            f"cache={capacity}, prefill={prefill}, precision={precision}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("moss_transcribe_diarize text engine build failed")
    return bytes(plan)


def text_weight_names(config: MossConfig) -> list[str]:
    """Every checkpoint tensor the text engine reads."""
    names = [EMBEDDING, f"{_TEXT}norm.weight"]
    for index in range(config.text.num_layers):
        prefix = f"{_TEXT}layers.{index}"
        names += [f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"]
        names += [f"{prefix}.self_attn.{part}.weight"
                  for part in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        names += [f"{prefix}.mlp.{part}.weight" for part in ("gate_proj", "up_proj", "down_proj")]
    return names

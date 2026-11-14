# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder engine: the Granite language model over ``P`` cached and ``S`` new positions.

One network serves both optimisation profiles: prefill (``P = 0``, the whole prompt) and decode
(``S = 1``). Per layer, ``past_key_i``/``past_value_i`` ``[P, kv_width]`` are read and
``present_key_i``/``present_value_i`` ``[S, kv_width]`` (keys after the rotary embedding) are
written; the runtime binds both onto one cache so the new rows land where the next step reads
them. New position ``i`` sees cache rows and new positions ``j <= i``.

Rows whose ``audio_mask`` is non-zero take ``audio_embeds`` instead of the token embedding, then
every row is multiplied by ``embedding_multiplier`` as Granite does for its ``inputs_embeds``.
Only the last position's logits leave the engine, divided by ``logits_scaling`` in the work type
and widened to FP32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .config import GraniteSpeechConfig, Text
from .constants import rope_tables
from .graph import Graph


def _rotate(g: Graph, x: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor, heads: int, dim: int) -> trt.ITensor:
    """``x * cos + rotate_half(x) * sin`` for ``x [S, heads, dim]`` and tables ``[S, 1, dim]``."""
    half = dim // 2
    rows = g.dim(x, 0)
    size = g.concat([rows, g.i64([heads, half])], 0)
    first = g.slice(x, (0, 0, 0), size)
    second = g.slice(x, (0, 0, half), size)
    rotated = g.concat([g.scale(second, -1.0), first], 2)
    return g.add(g.mul(x, cos), g.mul(rotated, sin))


def _attention(g: Graph, ckpt: Checkpoint, text: Text, normed: trt.ITensor, base: str, cos, sin,
               past_key, past_value, mask: trt.ITensor, new: trt.ITensor,
               total: trt.ITensor) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    heads, kv_heads, dim = text.num_heads, text.num_kv_heads, text.head_dim
    group = heads // kv_heads
    q = g.reshape(g.linear(normed, ckpt.tensor(base + "q_proj.weight")), (-1, heads, dim))
    k = g.reshape(g.linear(normed, ckpt.tensor(base + "k_proj.weight")), (-1, kv_heads, dim))
    v = g.linear(normed, ckpt.tensor(base + "v_proj.weight"))  # [S, kv_width]
    q = _rotate(g, q, cos, sin, heads, dim)
    k = g.reshape(_rotate(g, k, cos, sin, kv_heads, dim), (-1, text.kv_width))
    keys = g.transpose(g.reshape(g.concat([past_key, k], 0), (-1, kv_heads, dim)), (1, 0, 2))
    values = g.transpose(g.reshape(g.concat([past_value, v], 0), (-1, kv_heads, dim)), (1, 0, 2))
    # Query head h reads key/value head h // group: group the query heads under their key head.
    grouped = g.reshape(g.transpose(q, (1, 0, 2)), (kv_heads, -1, dim))  # [kv, group * S, dim]
    wide = trt.float32
    scores = g.matmul(g.cast(grouped, wide), g.cast(keys, wide), transpose_right=True)  # [kv, group*S, P+S]
    scores = g.scale(scores, text.attention_multiplier)
    scores = g.reshape_to(scores, g.concat([g.i64([kv_heads, group]), new, total], 0))
    probs = g.cast(g.softmax(g.add(scores, mask), 3), q.dtype)
    probs = g.reshape_to(probs, g.concat([g.i64([kv_heads]), g.mul(new, g.i64([group])), total], 0))
    attended = g.matmul(probs, values)  # [kv, group*S, dim]
    rows = g.reshape(g.transpose(g.reshape(attended, (heads, -1, dim)), (1, 0, 2)), (-1, heads * dim))
    out = g.linear(rows, ckpt.tensor(base + "o_proj.weight"))
    return out, k, v


def _causal_mask(g: Graph, new: trt.ITensor, past: trt.ITensor, total: trt.ITensor, capacity: int) -> trt.ITensor:
    """``[1, 1, S, P + S]`` FP32: 0 where key ``j <= P + i``, else masked."""
    keys = g.reshape(g.arange(capacity, total), (1, -1))
    rows = g.reshape(g.add(g.arange(capacity, new), g.cast(past, trt.int32)), (-1, 1))
    visible = g.binary(keys, rows, trt.ElementWiseOperation.LESS)
    equal = g.binary(keys, rows, trt.ElementWiseOperation.EQUAL)
    visible = g.binary(visible, equal, trt.ElementWiseOperation.OR)
    return g.reshape_to(g.masked_scores(visible), g.concat([g.i64([1, 1]), new, total], 0))


def build_decoder_network(g: Graph, ckpt: Checkpoint, cfg: GraniteSpeechConfig, capacity: int) -> None:
    text = cfg.text
    hidden = text.hidden_size
    net = g.network
    input_ids = net.add_input("input_ids", trt.int32, (-1,))
    audio_mask = net.add_input("audio_mask", trt.int32, (-1,))
    audio_embeds = net.add_input("audio_embeds", g.work, (-1, hidden))
    pasts = []
    for index in range(text.num_layers):
        pasts.append((net.add_input(f"past_key_{index}", g.work, (-1, text.kv_width)),
                      net.add_input(f"past_value_{index}", g.work, (-1, text.kv_width))))
    new = g.dim(input_ids, 0)
    past = g.dim(pasts[0][0], 0)
    total = g.add(new, past)

    table = g.constant(ckpt.tensor("language_model.model.embed_tokens.weight"), g.work)
    tokens = net.add_gather(table, input_ids, 0).get_output(0)  # [S, hidden]
    is_text = g.reshape(g.binary(audio_mask, g.constant(np.zeros(1, np.int32), trt.int32),
                                 trt.ElementWiseOperation.EQUAL), (-1, 1))
    x = g.select(is_text, tokens, audio_embeds)
    x = g.scale(x, text.embedding_multiplier)

    cos_table, sin_table = rope_tables(text, capacity)
    positions = g.add(g.arange(capacity, new), g.cast(past, trt.int32))
    cos = g.reshape(net.add_gather(g.constant(cos_table, g.work), positions, 0).get_output(0), (-1, 1, text.head_dim))
    sin = g.reshape(net.add_gather(g.constant(sin_table, g.work), positions, 0).get_output(0), (-1, 1, text.head_dim))
    mask = _causal_mask(g, new, past, total, capacity)

    for index in range(text.num_layers):
        base = f"language_model.model.layers.{index}."
        normed = g.rms_norm(x, ckpt.tensor(base + "input_layernorm.weight"), text.rms_norm_eps)
        out, key, value = _attention(g, ckpt, text, normed, base + "self_attn.", cos, sin,
                                     pasts[index][0], pasts[index][1], mask, new, total)
        key.name = f"present_key_{index}"
        value.name = f"present_value_{index}"
        net.mark_output(key)
        net.mark_output(value)
        x = g.add(x, g.scale(out, text.residual_multiplier))
        normed = g.rms_norm(x, ckpt.tensor(base + "post_attention_layernorm.weight"), text.rms_norm_eps)
        gate = g.silu(g.linear(normed, ckpt.tensor(base + "mlp.gate_proj.weight")))
        up = g.linear(normed, ckpt.tensor(base + "mlp.up_proj.weight"))
        mlp = g.linear(g.mul(gate, up), ckpt.tensor(base + "mlp.down_proj.weight"))
        x = g.add(x, g.scale(mlp, text.residual_multiplier))

    last = g.slice(x, g.concat([g.sub(new, g.i64([1])), g.i64([0])], 0), (1, hidden))
    last = g.rms_norm(last, ckpt.tensor("language_model.model.norm.weight"), text.rms_norm_eps)
    logits = g.linear(last, ckpt.tensor("language_model.lm_head.weight"))
    logits = g.cast(g.scale(logits, 1.0 / text.logits_scaling), trt.float32)
    logits.name = "logits"
    net.mark_output(logits)

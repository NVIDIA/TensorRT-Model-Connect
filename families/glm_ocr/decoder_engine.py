# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-OCR text decoder as a prefill engine and a one-token decode engine.

Both engines take ``inputs_embeds [S, hidden]`` (float32; the runtime gathers
token rows and places image features) and the multimodal rotary table
``rotary_cos`` / ``rotary_sin [S, head_dim / 2]`` (one angle per rotary pair,
each pair already reading its temporal, height or width position). Each layer
returns this call's key and value rows ``present_key_<i>`` /
``present_value_<i> [S, kv_heads, head_dim]``, which the runtime binds directly
into its sequence-major cache. The prefill engine attends causally within the
prompt; the decode engine reads the cached rows ``past_key_<i>`` /
``past_value_<i> [P, kv_heads, head_dim]`` and attends one query to all of them
plus its own row. ``logits [1, vocab]`` are the float32 scores of the last row.
"""

from __future__ import annotations

import tensorrt as trt

from .checkpoint import Checkpoint, GlmOcrConfig
from .graph import DTYPES, Net, build_engine

_PREFIX = "model.language_model"
# Tactic selection point for prompt rows and cached rows.
_OPT_ROWS = 1024


def _rotate(net: Net, tensor: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    """Interleaved rotary embedding of ``[S, heads, head_dim]``: pair i is (2i, 2i + 1)."""
    head_dim = tensor.shape[-1]
    pairs = net.reshape(tensor, (0, 0, head_dim // 2, 2))
    even = net.gather(pairs, net.indices([0]), 3)
    odd = net.gather(pairs, net.indices([1]), 3)
    out_even = net.sub(net.mul(even, cos), net.mul(odd, sin))
    out_odd = net.add(net.mul(odd, cos), net.mul(even, sin))
    return net.reshape(net.concat([out_even, out_odd], 3), (0, 0, head_dim))


def _layer(net: Net, x: trt.ITensor, index: int, ck: Checkpoint, config: GlmOcrConfig,
           cos: trt.ITensor, sin: trt.ITensor, decode: bool) -> trt.ITensor:
    t = config.text
    network = net.network
    p = f"{_PREFIX}.layers.{index}"
    heads, kv_heads, head_dim = t.num_attention_heads, t.num_key_value_heads, t.head_dim
    eps = t.rms_norm_eps

    h = net.rms_norm(x, ck.get(f"{p}.input_layernorm.weight"), eps)
    query = net.reshape(net.linear(h, ck.get(f"{p}.self_attn.q_proj.weight")), (0, heads, head_dim))
    key = net.reshape(net.linear(h, ck.get(f"{p}.self_attn.k_proj.weight")), (0, kv_heads, head_dim))
    value = net.reshape(net.linear(h, ck.get(f"{p}.self_attn.v_proj.weight")), (0, kv_heads, head_dim))
    query = _rotate(net, query, cos, sin)
    key = _rotate(net, key, cos, sin)

    for name, tensor in ((f"present_key_{index}", key), (f"present_value_{index}", value)):
        tensor.name = name
        network.mark_output(tensor)
    if decode:
        past_key = network.add_input(f"past_key_{index}", net.dtype, (-1, kv_heads, head_dim))
        past_value = network.add_input(f"past_value_{index}", net.dtype, (-1, kv_heads, head_dim))
        key = net.concat([past_key, key], 0)
        value = net.concat([past_value, value], 0)

    def heads_first(tensor: trt.ITensor, count: int) -> trt.ITensor:
        return net.reshape(tensor, (1, -1, count, head_dim), second_transpose=(0, 2, 1, 3))

    groups = heads // kv_heads
    context = net.attention(
        heads_first(query, heads),
        net.repeat_heads(heads_first(key, kv_heads), groups),
        net.repeat_heads(heads_first(value, kv_heads), groups),
        causal=not decode,
        scale=head_dim ** -0.5,
    )
    context = net.transpose_reshape(context, (0, 2, 1, 3), (-1, heads * head_dim))
    attended = net.linear(context, ck.get(f"{p}.self_attn.o_proj.weight"))
    attended = net.rms_norm(attended, ck.get(f"{p}.post_self_attn_layernorm.weight"), eps)
    x = net.add(x, attended)

    h = net.rms_norm(x, ck.get(f"{p}.post_attention_layernorm.weight"), eps)
    gate_up = ck.get(f"{p}.mlp.gate_up_proj.weight")
    gate = net.linear(h, gate_up[: t.intermediate_size])
    up = net.linear(h, gate_up[t.intermediate_size:])
    mlp = net.linear(net.mul(up, net.silu(gate)), ck.get(f"{p}.mlp.down_proj.weight"))
    mlp = net.rms_norm(mlp, ck.get(f"{p}.post_mlp_layernorm.weight"), eps)
    return net.add(x, mlp)


def _populate(net: Net, ck: Checkpoint, config: GlmOcrConfig, decode: bool) -> None:
    t = config.text
    network = net.network
    half = t.head_dim // 2
    rows = 1 if decode else -1
    embeds = network.add_input("inputs_embeds", trt.float32, (rows, t.hidden_size))
    cos = network.add_input("rotary_cos", trt.float32, (rows, half))
    sin = network.add_input("rotary_sin", trt.float32, (rows, half))
    cos4 = net.reshape(net.cast(cos, net.dtype), (0, 1, half, 1))
    sin4 = net.reshape(net.cast(sin, net.dtype), (0, 1, half, 1))

    x = net.cast(embeds, net.dtype)
    for index in range(t.num_hidden_layers):
        x = _layer(net, x, index, ck, config, cos4, sin4, decode)
    last = x if decode else net.last_row(x)
    last = net.rms_norm(last, ck.get(f"{_PREFIX}.norm.weight"), t.rms_norm_eps)
    logits = net.cast(net.linear(last, ck.get("lm_head.weight")), trt.float32)
    logits.name = "logits"
    network.mark_output(logits)


def build_decoder_engines(ck: Checkpoint, config: GlmOcrConfig, precision: str,
                          max_rows: int, *, verbose: bool) -> tuple[bytes, bytes]:
    """The prefill and decode plans for a cache of ``max_rows`` rows."""
    t = config.text
    half = t.head_dim // 2
    dtype = DTYPES[precision]
    opt = min(_OPT_ROWS, max_rows - 1)
    prefill_profile = {
        "inputs_embeds": ((1, t.hidden_size), (opt, t.hidden_size), (max_rows, t.hidden_size)),
        "rotary_cos": ((1, half), (opt, half), (max_rows, half)),
        "rotary_sin": ((1, half), (opt, half), (max_rows, half)),
    }
    cached = (t.num_key_value_heads, t.head_dim)
    decode_profile = {}
    for index in range(t.num_hidden_layers):
        for kind in ("key", "value"):
            decode_profile[f"past_{kind}_{index}"] = (
                (1, *cached), (opt, *cached), (max_rows - 1, *cached))
    prefill = build_engine(lambda net: _populate(net, ck, config, False), dtype,
                           [prefill_profile], verbose=verbose)
    decode = build_engine(lambda net: _populate(net, ck, config, True), dtype,
                          [decode_profile], verbose=verbose)
    return prefill, decode

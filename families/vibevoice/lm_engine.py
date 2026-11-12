# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2 language-model engine with a runtime-owned, sequence-major KV cache.

Each call appends ``T`` new positions to a sequence whose ``P`` earlier
positions arrive as ``past_key_<l>`` / ``past_value_<l>`` ``[1, P, kv_heads,
head_dim]`` (the contiguous prefix of the runtime's cache) and leaves the new
rows in ``present_key_<l>`` / ``present_value_<l>`` ``[1, T, kv_heads,
head_dim]``, which the runtime binds directly behind that prefix. Rotary
tables for the new positions come from the runtime, so one engine serves the
three sequences of the generation loop. The engine also returns the final
hidden row, the logits of the four control tokens, and the log-normalizer of
the full-vocabulary distribution: the reference picks control tokens from
log-probabilities, so the full softmax denominator is part of the decision.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

PREFILL_PROFILE = 0
DECODE_PROFILE = 1


def _iota(net: g.Net, length: trt.ITensor) -> trt.ITensor:
    """int32 ``[0, 1, ..., length-1]`` for a 1-element int64 shape tensor."""
    fill = net.network.add_fill((0,), trt.FillOperation.LINSPACE, trt.int32)
    fill.set_input(0, length)
    fill.set_input(1, net.const(np.array(0, dtype=np.int32), (), trt.int32))
    fill.set_input(2, net.const(np.array([1], dtype=np.int32), (1,), trt.int32))
    return fill.get_output(0)


def _dim(net: g.Net, tensor: trt.ITensor, axis: int) -> trt.ITensor:
    shape = net.network.add_shape(tensor).get_output(0)
    return net.slice(shape, (axis,), (1,))


def _rotate_half_matrix(head_dim: int) -> np.ndarray:
    """``x @ R == cat(-x[half:], x[:half])``; exact in any precision."""
    half = head_dim // 2
    matrix = np.zeros((head_dim, head_dim), dtype=np.float32)
    for column in range(half):
        matrix[column + half, column] = -1.0
        matrix[column, column + half] = 1.0
    return matrix


def _causal_block_mask(net: g.Net, query_length: trt.ITensor, past_length: trt.ITensor):
    """BOOL ``[1, 1, 1, T, P+T]`` that is true where a key lies after the query."""
    total = net.network.add_elementwise(
        past_length, query_length, trt.ElementWiseOperation.SUM
    ).get_output(0)
    keys = net.reshape(_iota(net, total), (1, 1, 1, 1, -1))
    past32 = net.cast(past_length, trt.int32)
    limits = net.network.add_elementwise(
        _iota(net, query_length), past32, trt.ElementWiseOperation.SUM
    ).get_output(0)
    limits = net.reshape(limits, (1, 1, 1, -1, 1))
    return net.network.add_elementwise(keys, limits, trt.ElementWiseOperation.GREATER).get_output(0)


def _attention(net, query, key, value, blocked, *, heads, kv_heads, head_dim):
    """Grouped-query attention with float32 scores, softmax and context."""
    groups = heads // kv_heads
    q = net.reshape(query, (1, kv_heads, groups, -1, head_dim), first_transpose=(0, 2, 1, 3))
    k = net.reshape(key, (1, kv_heads, 1, -1, head_dim), first_transpose=(0, 2, 1, 3))
    v = net.reshape(value, (1, kv_heads, 1, -1, head_dim), first_transpose=(0, 2, 1, 3))
    q, k, v = (net.cast(t, trt.float32) for t in (q, k, v))
    scores = net.network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = net.mul(scores, net.scalar(1.0 / float(np.sqrt(head_dim)), 5, trt.float32))
    scores = net.network.add_select(
        blocked, net.scalar(-1.0e30, 5, trt.float32), scores
    ).get_output(0)
    softmax = net.network.add_softmax(scores)
    softmax.axes = 1 << 4
    context = net.network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
    ).get_output(0)
    context = net.reshape(context, (1, heads, -1, head_dim), second_transpose=(0, 2, 1, 3))
    context = net.reshape(context, (1, -1, heads * head_dim))
    return net.cast(context, net.dtype)


def _rope(net, tensor, cos, sin, rotate):
    rotated = net.network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rotate, trt.MatrixOperation.NONE
    ).get_output(0)
    return net.add(net.mul(tensor, cos), net.mul(rotated, sin))


def build_lm_engine(
    weights,
    cfg,
    *,
    dtype,
    capacity: int,
    prefill_chunk: int,
    control_token_ids: list[int],
    verbose: bool = False,
) -> bytes:
    """Build the two-profile (prefill, decode) Qwen2 decoder engine."""
    builder, network, config = g.builder_and_network(verbose)
    net = g.Net(network, dtype)
    hidden, heads, kv_heads, head_dim = (
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
    )
    prefix = "model.language_model."

    embeds = network.add_input("inputs_embeds", dtype, (1, -1, hidden))
    cos = network.add_input("rope_cos", trt.float32, (-1, head_dim))
    sin = network.add_input("rope_sin", trt.float32, (-1, head_dim))
    pasts = [
        (
            network.add_input(f"past_key_{layer}", dtype, (1, -1, kv_heads, head_dim)),
            network.add_input(f"past_value_{layer}", dtype, (1, -1, kv_heads, head_dim)),
        )
        for layer in range(cfg.num_layers)
    ]

    # The reference casts the float32 rotary tables to the activation type.
    cos4 = net.reshape(net.cast(cos, dtype), (1, -1, 1, head_dim))
    sin4 = net.reshape(net.cast(sin, dtype), (1, -1, 1, head_dim))
    rotate = net.const(_rotate_half_matrix(head_dim), (1, 1, head_dim, head_dim))
    blocked = _causal_block_mask(net, _dim(net, embeds, 1), _dim(net, pasts[0][0], 1))

    state = embeds
    for layer, (past_key, past_value) in enumerate(pasts):
        base = f"{prefix}layers.{layer}."
        residual = state
        x = net.rms_norm(state, weights.f32(base + "input_layernorm.weight"), cfg.rms_norm_eps, -1)
        q = net.linear(
            x,
            weights.f32(base + "self_attn.q_proj.weight"),
            weights.f32(base + "self_attn.q_proj.bias"),
        )
        k = net.linear(
            x,
            weights.f32(base + "self_attn.k_proj.weight"),
            weights.f32(base + "self_attn.k_proj.bias"),
        )
        v = net.linear(
            x,
            weights.f32(base + "self_attn.v_proj.weight"),
            weights.f32(base + "self_attn.v_proj.bias"),
        )
        q = _rope(net, net.reshape(q, (1, -1, heads, head_dim)), cos4, sin4, rotate)
        k = _rope(net, net.reshape(k, (1, -1, kv_heads, head_dim)), cos4, sin4, rotate)
        v = net.reshape(v, (1, -1, kv_heads, head_dim))
        k.name, v.name = f"present_key_{layer}", f"present_value_{layer}"
        network.mark_output(k)
        network.mark_output(v)
        keys = net.concat([past_key, k], 1)
        values = net.concat([past_value, v], 1)
        attended = _attention(
            net, q, keys, values, blocked, heads=heads, kv_heads=kv_heads, head_dim=head_dim
        )
        attended = net.linear(attended, weights.f32(base + "self_attn.o_proj.weight"))
        state = net.add(residual, attended)
        residual = state
        x = net.rms_norm(
            state, weights.f32(base + "post_attention_layernorm.weight"), cfg.rms_norm_eps, -1
        )
        gate = net.silu(net.linear(x, weights.f32(base + "mlp.gate_proj.weight")))
        up = net.linear(x, weights.f32(base + "mlp.up_proj.weight"))
        state = net.add(
            residual, net.linear(net.mul(gate, up), weights.f32(base + "mlp.down_proj.weight"))
        )

    # Only the last position feeds the diffusion head and the token choice.
    last_index = net.network.add_elementwise(
        net.cast(_dim(net, embeds, 1), trt.int32),
        net.const(np.array([1], dtype=np.int32), (1,), trt.int32),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    last = net.network.add_gather(state, last_index, 1).get_output(0)
    last = net.reshape(last, (1, hidden))
    last = net.rms_norm(last, weights.f32(prefix + "norm.weight"), cfg.rms_norm_eps, -1)
    last.name = "hidden"
    network.mark_output(last)

    # Tied LM head: the reference computes logits in the activation type and
    # converts them to float32 before the log-softmax.
    table = weights.f32(prefix + "embed_tokens.weight")
    logits = net.cast(net.linear(last, table), trt.float32)
    peak = net.network.add_reduce(logits, trt.ReduceOperation.MAX, 1 << 1, True).get_output(0)
    shifted = net.unary(net.sub(logits, peak), trt.UnaryOperation.EXP)
    total = net.network.add_reduce(shifted, trt.ReduceOperation.SUM, 1 << 1, True).get_output(0)
    normalizer = net.add(peak, net.unary(total, trt.UnaryOperation.LOG))
    normalizer = net.reshape(normalizer, (1,))
    normalizer.name = "log_normalizer"
    network.mark_output(normalizer)
    ids = net.const(
        np.asarray(control_token_ids, dtype=np.int32), (len(control_token_ids),), trt.int32
    )
    control = net.network.add_gather(logits, ids, 1).get_output(0)
    control.name = "control_logits"
    network.mark_output(control)

    opt_past = min(1024, capacity - 1)
    for profile_index in (PREFILL_PROFILE, DECODE_PROFILE):
        profile = builder.create_optimization_profile()
        rows = (1, prefill_chunk, prefill_chunk) if profile_index == PREFILL_PROFILE else (1, 1, 1)
        profile.set_shape("inputs_embeds", *[(1, r, hidden) for r in rows])
        profile.set_shape("rope_cos", *[(r, head_dim) for r in rows])
        profile.set_shape("rope_sin", *[(r, head_dim) for r in rows])
        past = [
            (1, 0, kv_heads, head_dim),
            (1, opt_past, kv_heads, head_dim),
            (1, capacity - 1, kv_heads, head_dim),
        ]
        for layer in range(cfg.num_layers):
            profile.set_shape(f"past_key_{layer}", *past)
            profile.set_shape(f"past_value_{layer}", *past)
        config.add_optimization_profile(profile)
    return g.serialize(builder, network, config, "language model")


def build_embed_engine(weights, cfg, *, dtype, max_tokens: int, verbose: bool = False) -> bytes:
    """Token-id gather from the embedding table (prompt and control tokens)."""
    builder, network, config = g.builder_and_network(verbose)
    net = g.Net(network, dtype)
    ids = network.add_input("token_ids", trt.int32, (-1,))
    table = net.const(weights.f32("model.language_model.embed_tokens.weight"))
    rows = network.add_gather(table, ids, 0).get_output(0)
    rows.name = "embeds"
    network.mark_output(rows)
    profile = builder.create_optimization_profile()
    profile.set_shape("token_ids", (1,), (max(1, max_tokens // 2),), (max_tokens,))
    config.add_optimization_profile(profile)
    return g.serialize(builder, network, config, "embedding")

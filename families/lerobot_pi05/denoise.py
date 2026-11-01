# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One flow-matching step of the Pi 0.5 action expert.

The engine maps ``x_t`` to ``x_t + dt * v_t``, so the Euler update lives in the
graph and the runtime only loops. It also emits ``v_t`` for parity checks.
The expert attends to the prefix keys/values the prefix engine produced for the
same layer, plus its own action tokens, which all see each other.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import require

WIDTH = 1024
DEPTH = 18
MLP = 4096
HEADS = 8
HEAD_DIM = 256
RMS_EPS = 1e-6
ROPE_THETA = 10000.0

_ROOT = "gemma_expert.model."


def _time_condition(network, weights, cfg, time):
    """Sine-cosine embedding of ``time[1]`` through the two-layer SiLU MLP -> ``[1, WIDTH]``."""
    half = WIDTH // 2
    fraction = np.linspace(0.0, 1.0, half, dtype=np.float64)
    period = cfg.min_period * (cfg.max_period / cfg.min_period) ** fraction
    angles = graph.mul(
        network,
        graph.shuffle(network, time, reshape=(1, 1)),
        graph.const(network, (1.0 / period * 2 * math.pi).reshape(1, half)),
    )
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    x = graph.concat(network, [sin, cos], 1)
    x = graph.silu(network, graph.linear(network, x, weights["time_mlp_in.weight"], weights["time_mlp_in.bias"]))
    return graph.silu(network, graph.linear(network, x, weights["time_mlp_out.weight"], weights["time_mlp_out.bias"]))


def _modulation(network, cond, dense_weight, dense_bias):
    """Split ``dense(cond)`` into scale, shift and gate, each ``[1, WIDTH]``."""
    mod = graph.linear(network, cond, dense_weight, dense_bias)
    return tuple(graph.slice_static(network, mod, (0, i * WIDTH), (1, WIDTH)) for i in range(3))


def build_denoise(network, weights, cfg, *, dtype, prefix_len):
    """Declare the denoise engine's inputs and outputs on ``network``."""
    steps = cfg.chunk_size
    action_dim = cfg.max_action_dim

    x_t = network.add_input("x_t", trt.float32, (steps, action_dim))
    time = network.add_input("time", trt.float32, (1,))
    dt = network.add_input("dt", trt.float32, (1,))
    offset = network.add_input("position_offset", trt.int32, (1,))
    prefix_bias = network.add_input("prefix_bias", trt.float32, (prefix_len,))
    k_cache = network.add_input("k_cache", dtype, (DEPTH, prefix_len, HEAD_DIM))
    v_cache = network.add_input("v_cache", dtype, (DEPTH, prefix_len, HEAD_DIM))

    require(weights, "action_in_proj.weight", (WIDTH, action_dim))
    require(weights, "action_out_proj.weight", (action_dim, WIDTH))

    cond = _time_condition(network, weights, cfg, time)
    h = graph.cast(network, graph.linear(
        network, x_t, weights["action_in_proj.weight"], weights["action_in_proj.bias"]), dtype)

    positions = graph.add(network, graph.const_int32(network, np.arange(steps)), offset)
    cos, sin = graph.rope_tables(network, positions, HEAD_DIM, ROPE_THETA)
    # Padded prefix keys are blocked; the action tokens see every key.
    blocked = graph.add(
        network,
        graph.shuffle(network, prefix_bias, reshape=(1, prefix_len)),
        graph.const(network, np.zeros((steps, 1), dtype=np.float32)),
    )
    bias = graph.concat(network, [blocked, graph.const(network, np.zeros((steps, steps), dtype=np.float32))], 1)

    for index in range(DEPTH):
        p = f"{_ROOT}layers.{index}."
        scale, shift, gate = _modulation(
            network, cond, weights[p + "input_layernorm.dense.weight"], weights[p + "input_layernorm.dense.bias"])
        norm = graph.ada_rms_norm(network, h, scale, shift, RMS_EPS)
        q = graph.apply_rope(network, graph.linear(network, norm, weights[p + "self_attn.q_proj.weight"]), cos, sin, HEADS, HEAD_DIM)
        k = graph.apply_rope(network, graph.linear(network, norm, weights[p + "self_attn.k_proj.weight"]), cos, sin, 1, HEAD_DIM)
        v = graph.linear(network, norm, weights[p + "self_attn.v_proj.weight"])
        prefix_k = graph.shuffle(network, graph.slice_static(
            network, k_cache, (index, 0, 0), (1, prefix_len, HEAD_DIM)), reshape=(prefix_len, HEAD_DIM))
        prefix_v = graph.shuffle(network, graph.slice_static(
            network, v_cache, (index, 0, 0), (1, prefix_len, HEAD_DIM)), reshape=(prefix_len, HEAD_DIM))
        keys = graph.concat(network, [prefix_k, k], 0)
        values = graph.concat(network, [prefix_v, v], 0)
        context = graph.attention_mqa(network, q, keys, values, bias, heads=HEADS, head_dim=HEAD_DIM)
        out = graph.linear(network, context, weights[p + "self_attn.o_proj.weight"])
        h = graph.add(network, h, graph.mul(network, out, graph.cast(network, gate, dtype)))

        scale, shift, gate = _modulation(
            network, cond, weights[p + "post_attention_layernorm.dense.weight"], weights[p + "post_attention_layernorm.dense.bias"])
        norm = graph.ada_rms_norm(network, h, scale, shift, RMS_EPS)
        act = graph.gelu_tanh(network, graph.linear(network, norm, weights[p + "mlp.gate_proj.weight"]))
        hidden = graph.mul(network, act, graph.linear(network, norm, weights[p + "mlp.up_proj.weight"]))
        out = graph.linear(network, hidden, weights[p + "mlp.down_proj.weight"])
        h = graph.add(network, h, graph.mul(network, out, graph.cast(network, gate, dtype)))

    scale, shift, _ = _modulation(network, cond, weights[_ROOT + "norm.dense.weight"], weights[_ROOT + "norm.dense.bias"])
    h = graph.ada_rms_norm(network, h, scale, shift, RMS_EPS)
    velocity = graph.linear(
        network, graph.cast(network, h, trt.float32), weights["action_out_proj.weight"], weights["action_out_proj.bias"])
    step = graph.mul(network, velocity, graph.shuffle(network, dt, reshape=(1, 1)))
    x_next = network.add_identity(graph.add(network, x_t, step)).get_output(0)
    velocity = network.add_identity(velocity).get_output(0)
    x_next.name, velocity.name = "x_next", "velocity"
    network.mark_output(x_next)
    network.mark_output(velocity)

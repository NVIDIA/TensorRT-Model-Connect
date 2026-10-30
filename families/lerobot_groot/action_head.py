# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 flow-matching action head.

``build_condition`` runs once per observation: the vision-language features
through ``vlln`` and the four self-attention blocks, the state token, and the keys
and values every cross-attention block of the DiT reads (they do not change across
denoising steps). ``build_denoise`` is one Euler step: the noisy actions through
the action encoder, the 32-block DiT and the action decoder, ``x + dt * v``.
Category-specific layers keep only the checkpoint's embodiment.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import category, require

VL_WIDTH = 2048
VL_HEADS = 32
VL_HEAD_DIM = 64
VL_LAYERS = 4
WIDTH = 1536
HEADS = 32
HEAD_DIM = 48
LAYERS = 32
DIT_OUT = 1024
DECODER_HIDDEN = 1024
STATE_HIDDEN = 1024
TIME_CHANNELS = 256
MAX_POSITIONS = 1024
LN_EPS = 1e-5
OUT_EPS = 1e-6
# Cross-attention blocks are the even ones; every other one of them reads the text tokens.
CROSS_BLOCKS = tuple(range(0, LAYERS, 2))

_HEAD = "action_head."
_DIT = "action_head.model."


def _category_linear(network, x, weights, prefix, embodiment, shape, dtype):
    """``x @ W[e]`` in ``dtype`` plus the float32 bias, as the reference's bmm then ``+ b``."""
    weight, bias = category(weights, prefix, embodiment, shape)
    out = graph.f32(network, graph.linear_in_out(network, x, weight, dtype=dtype))
    return graph.add(network, out, graph.const(network, bias.reshape(1, -1)))


def _self_attention_block(network, h, weights, p, dtype):
    """A diffusers BasicTransformerBlock with affine LayerNorms and a tanh-gelu MLP."""
    norm = graph.layer_norm(network, h, LN_EPS, weights[p + "norm1.weight"], weights[p + "norm1.bias"])
    q, k, v = (graph.linear(network, norm, weights[f"{p}attn1.to_{n}.weight"], weights[f"{p}attn1.to_{n}.bias"], dtype=dtype)
               for n in "qkv")
    context = graph.attention(network, q, k, v, heads=VL_HEADS, kv_heads=VL_HEADS, head_dim=VL_HEAD_DIM, dtype=dtype)
    h = graph.add(network, h, graph.f32(network, graph.linear(
        network, context, weights[p + "attn1.to_out.0.weight"], weights[p + "attn1.to_out.0.bias"], dtype=dtype)))
    norm = graph.layer_norm(network, h, LN_EPS, weights[p + "norm3.weight"], weights[p + "norm3.bias"])
    hidden = graph.gelu_tanh(network, graph.linear(
        network, norm, weights[p + "ff.net.0.proj.weight"], weights[p + "ff.net.0.proj.bias"], dtype=dtype))
    return graph.add(network, h, graph.f32(network, graph.linear(
        network, hidden, weights[p + "ff.net.2.weight"], weights[p + "ff.net.2.bias"], dtype=dtype)))


def build_condition(network, weights, backbone_features, state, *, embodiment, max_state_dim, dtype):
    """Returns ``(vl_embeds[S, 2048] f32, state_features[1, 1536] f32, cross_k, cross_v)``.

    ``cross_k``/``cross_v`` are ``[len(CROSS_BLOCKS), S, WIDTH]`` in ``dtype``; ``state`` is the
    normalised, zero-padded ``[1, max_state_dim]`` state.
    """
    vl = graph.layer_norm(network, backbone_features, LN_EPS, weights[_HEAD + "vlln.weight"], weights[_HEAD + "vlln.bias"])
    for index in range(VL_LAYERS):
        vl = _self_attention_block(network, vl, weights, f"{_HEAD}vl_self_attention.transformer_blocks.{index}.", dtype)

    hidden = graph.relu(network, _category_linear(
        network, state, weights, _HEAD + "state_encoder.layer1.", embodiment, (max_state_dim, STATE_HIDDEN), dtype))
    state_features = _category_linear(
        network, hidden, weights, _HEAD + "state_encoder.layer2.", embodiment, (STATE_HIDDEN, WIDTH), dtype)

    keys, values = [], []
    for index in CROSS_BLOCKS:
        p = f"{_DIT}transformer_blocks.{index}.attn1."
        require(weights, p + "to_k.weight", (WIDTH, VL_WIDTH))
        for name, out in (("to_k", keys), ("to_v", values)):
            rows = graph.linear(network, vl, weights[p + name + ".weight"], weights[p + name + ".bias"], dtype=dtype)
            out.append(graph.shuffle(network, rows, reshape=(1, -1, WIDTH)))
    return vl, state_features, graph.concat(network, keys, 0), graph.concat(network, values, 0)


def action_time_frequencies() -> np.ndarray:
    """SinusoidalPositionalEncoding(1536): ``exp(-i * (ln 10000 / 768))``, in the reference's float32 order."""
    half = WIDTH // 2
    step = np.log(np.float32(10000.0)).astype(np.float32) / np.float32(half)
    return np.exp(-np.arange(half, dtype=np.float32) * np.float32(step)).astype(np.float32)


def dit_time_frequencies() -> np.ndarray:
    """diffusers ``get_timestep_embedding``: ``exp(-ln(10000) * i / (half - 1))`` (downscale_freq_shift=1)."""
    half = TIME_CHANNELS // 2
    exponent = (np.float32(-math.log(10000.0)) * np.arange(half, dtype=np.float32)) / np.float32(half - 1)
    return np.exp(exponent.astype(np.float32)).astype(np.float32)


def _sinusoid(network, timestep, frequencies, *, cos_first):
    """``[1, 2 * len(frequencies)]`` sine/cosine embedding of the float32 ``timestep[1]``."""
    angles = graph.mul(network, graph.shuffle(network, timestep, reshape=(1, 1)),
                       graph.const(network, frequencies.reshape(1, -1)))
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    return graph.concat(network, [cos, sin] if cos_first else [sin, cos], 1)


def _modulate(network, h, temb, weight, bias, eps, *, scale_first, dtype):
    """``LayerNorm(h) * (1 + scale) + shift`` with ``scale``/``shift`` from ``linear(silu(temb))``.

    As under the reference's autocast, ``1 + scale`` is formed in the compute dtype.
    """
    mod = graph.linear(network, graph.silu(network, temb), weight, bias, dtype=dtype)
    first, second = (graph.slice_static(network, mod, (0, i * WIDTH), (1, WIDTH)) for i in range(2))
    scale, shift = (first, second) if scale_first else (second, first)
    gain = graph.f32(network, graph.add(network, scale, graph.scalar(network, 1.0, 2, dtype)))
    return graph.add(network, graph.mul(network, graph.layer_norm(network, h, eps), gain), graph.f32(network, shift))


def build_denoise(network, weights, *, embodiment, action_horizon, max_action_dim, dtype,
                  x_t, timestep, dt, state_features, cross_k, cross_v, text_bias, image_bias):
    """One Euler step: returns ``(x_next, velocity)``, both float32 ``[action_horizon, max_action_dim]``."""
    def encoder(x, name, shape):
        return _category_linear(network, x, weights, f"{_HEAD}action_encoder.{name}.", embodiment, shape, dtype)

    actions = encoder(x_t, "W1", (max_action_dim, WIDTH))
    time = _sinusoid(network, timestep, action_time_frequencies(), cos_first=False)
    time = graph.add(network, time, graph.const(network, np.zeros((action_horizon, WIDTH), dtype=np.float32)))
    hidden = graph.silu(network, encoder(graph.concat(network, [actions, time], 1), "W2", (2 * WIDTH, WIDTH)))
    actions = encoder(hidden, "W3", (WIDTH, WIDTH))
    positions = require(weights, _HEAD + "position_embedding.weight", (MAX_POSITIONS, WIDTH))
    actions = graph.add(network, actions, graph.const(network, positions[:action_horizon]))
    h = graph.concat(network, [state_features, actions], 0)

    # diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1) + TimestepEmbedding.
    p = f"{_DIT}timestep_encoder.timestep_embedder."
    projection = _sinusoid(network, timestep, dit_time_frequencies(), cos_first=True)
    temb = graph.silu(network, graph.linear(network, projection, weights[p + "linear_1.weight"], weights[p + "linear_1.bias"], dtype=dtype))
    temb = graph.linear(network, temb, weights[p + "linear_2.weight"], weights[p + "linear_2.bias"], dtype=dtype)

    for index in range(LAYERS):
        p = f"{_DIT}transformer_blocks.{index}."
        norm = _modulate(network, h, temb, weights[p + "norm1.linear.weight"], weights[p + "norm1.linear.bias"], LN_EPS,
                         scale_first=True, dtype=dtype)
        q = graph.linear(network, norm, weights[p + "attn1.to_q.weight"], weights[p + "attn1.to_q.bias"], dtype=dtype)
        if index in CROSS_BLOCKS:
            slot = CROSS_BLOCKS.index(index)
            k, v = (graph.shuffle(network, network.add_gather(cache, graph.const_int32(network, [slot]), 0).get_output(0),
                                  reshape=(-1, WIDTH))
                    for cache in (cross_k, cross_v))
            bias = text_bias if index % 4 == 0 else image_bias
        else:
            k, v = (graph.linear(network, norm, weights[f"{p}attn1.to_{n}.weight"], weights[f"{p}attn1.to_{n}.bias"], dtype=dtype)
                    for n in "kv")
            bias = None
        context = graph.attention(network, q, k, v, heads=HEADS, kv_heads=HEADS, head_dim=HEAD_DIM, dtype=dtype, bias=bias)
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, context, weights[p + "attn1.to_out.0.weight"], weights[p + "attn1.to_out.0.bias"], dtype=dtype)))
        hidden = graph.gelu_tanh(network, graph.linear(
            network, graph.layer_norm(network, h, LN_EPS), weights[p + "ff.net.0.proj.weight"],
            weights[p + "ff.net.0.proj.bias"], dtype=dtype))
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, hidden, weights[p + "ff.net.2.weight"], weights[p + "ff.net.2.bias"], dtype=dtype)))

    h = _modulate(network, h, temb, weights[_DIT + "proj_out_1.weight"], weights[_DIT + "proj_out_1.bias"], OUT_EPS,
                  scale_first=False, dtype=dtype)
    out = graph.linear(network, h, weights[_DIT + "proj_out_2.weight"], weights[_DIT + "proj_out_2.bias"], dtype=dtype)
    hidden = graph.relu(network, _category_linear(
        network, out, weights, _HEAD + "action_decoder.layer1.", embodiment, (DIT_OUT, DECODER_HIDDEN), dtype))
    prediction = _category_linear(
        network, hidden, weights, _HEAD + "action_decoder.layer2.", embodiment, (DECODER_HIDDEN, max_action_dim), dtype)
    velocity = graph.slice_static(network, prediction, (1, 0), (action_horizon, max_action_dim))
    x_next = graph.add(network, x_t, graph.mul(network, velocity, graph.shuffle(network, dt, reshape=(1, 1))))
    return x_next, velocity

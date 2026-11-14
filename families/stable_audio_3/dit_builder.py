# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Stable Audio 3 DiT as one TensorRT network with a dynamic latent length.

One call computes the reference ``DiffusionTransformer._forward`` for a single
text-to-audio sample without classifier-free guidance:

* inputs: noisy latents ``x`` [1, C, T], time ``t`` [1], the conditioning
  engine's context [1, 257, 768] and global conditioning [1, 768], and the
  number of valid latents (the rest are padding the sampler masks out);
* the self-attention keys are the memory tokens plus the valid latents, which is
  what the reference's flash-attention varlen path attends to;
* the inpaint mask and masked input are zero for text-to-audio, so each block's
  local embedding is the constant ``to_local_embed(0)``, added to the latent
  positions only (memory tokens are left-padded with zeros).

FP16 networks keep the reference's FP32 islands: RMSNorm (``force_fp32`` and
the qk-norms), the rotary angles and rotation, and the Fourier features of the
time, which the reference first rounds to the model dtype.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph import (MASKED, Graph, apply_rotary, expo_fourier_features, expo_fourier_frequencies,
                    rotary_tables)

LATENTS = "x"
TIME = "t"
CONTEXT = "cross_attn_cond"
GLOBAL = "global_cond"
VALID = "valid_latents"
VELOCITY = "v"

_TOP_LEVEL = [
    "preprocess_conv.weight", "postprocess_conv.weight",
    "to_cond_embed.0.weight", "to_cond_embed.2.weight",
    "to_global_embed.0.weight", "to_global_embed.2.weight",
    "to_timestep_embed.0.weight", "to_timestep_embed.0.bias",
    "to_timestep_embed.2.weight", "to_timestep_embed.2.bias",
    "transformer.global_cond_embedder.0.weight", "transformer.global_cond_embedder.0.bias",
    "transformer.global_cond_embedder.2.weight", "transformer.global_cond_embedder.2.bias",
    "transformer.memory_tokens", "transformer.project_in.weight", "transformer.project_out.weight",
    "transformer.rotary_pos_emb.inv_freq",
]
_BLOCK = [
    "pre_norm.gamma", "cross_attend_norm.gamma", "ff_norm.gamma", "to_scale_shift_gate",
    "self_attn.to_qkv.weight", "self_attn.to_out.weight", "self_attn.q_norm.gamma", "self_attn.k_norm.gamma",
    "cross_attn.to_q.weight", "cross_attn.to_kv.weight", "cross_attn.to_out.weight",
    "cross_attn.q_norm.gamma", "cross_attn.k_norm.gamma",
    "ff.ff.0.proj.weight", "ff.ff.0.proj.bias", "ff.ff.2.weight", "ff.ff.2.bias",
    "to_local_embed.0.weight", "to_local_embed.0.bias", "to_local_embed.2.weight", "to_local_embed.2.bias",
]


def weight_names(dit: dict) -> list[str]:
    names = list(_TOP_LEVEL)
    for layer in range(dit["depth"]):
        names += [f"transformer.layers.{layer}.{name}" for name in _BLOCK]
    return names


def _silu64(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def local_embedding(weights: dict, layer: int) -> np.ndarray:
    """``to_local_embed(zeros)`` of one block: the inpaint inputs are zero for text-to-audio."""
    prefix = f"transformer.layers.{layer}.to_local_embed."
    hidden = _silu64(weights[prefix + "0.bias"].astype(np.float64))
    out = weights[prefix + "2.weight"].astype(np.float64) @ hidden + weights[prefix + "2.bias"].astype(np.float64)
    return out.astype(np.float32)


def _mlp(g: Graph, x, w1, w2, b1=None, b2=None):
    return g.linear(g.silu(g.linear(x, w1, b1)), w2, b2)


def _heads(g: Graph, x, length, heads: int, head_dim: int):
    """[1, N, heads * head_dim] -> [1, heads, N, head_dim]."""
    return g.transpose(g.reshape(x, [1, length, heads, head_dim]), (0, 2, 1, 3))


def _merge(g: Graph, x, length, width: int):
    return g.reshape(g.transpose(x, (0, 2, 1, 3)), [1, length, width])


def _differential(g: Graph, q, q_diff, k, k_diff, v, scale: float, mask=None):
    return g.sub(g.attention(q, k, v, scale, mask), g.attention(q_diff, k_diff, v, scale, mask))


def _self_attention(g: Graph, x, w, dit: dict, rope, length, mask):
    heads, head_dim, embed = dit["heads"], dit["head_dim"], dit["embed"]
    q, k, v, q_diff, k_diff = g.split_last(g.linear(x, w("self_attn.to_qkv.weight")), 5)
    q, k, v, q_diff, k_diff = (_heads(g, item, length, heads, head_dim) for item in (q, k, v, q_diff, k_diff))
    q_gamma, k_gamma, eps = w("self_attn.q_norm.gamma"), w("self_attn.k_norm.gamma"), dit["qk_norm_eps"]
    q, q_diff = (g.rms_norm(item, q_gamma, eps) for item in (q, q_diff))
    k, k_diff = (g.rms_norm(item, k_gamma, eps) for item in (k, k_diff))
    cos, sin = rope
    q, q_diff, k, k_diff = (apply_rotary(g, item, cos, sin, dit["rope_dim"], head_dim, v.dtype)
                            for item in (q, q_diff, k, k_diff))
    out = _differential(g, q, q_diff, k, k_diff, v, head_dim ** -0.5, mask)
    return g.linear(_merge(g, out, length, embed), w("self_attn.to_out.weight"))


def _cross_attention(g: Graph, x, context, w, dit: dict, length, context_length: int):
    heads, head_dim, embed = dit["heads"], dit["head_dim"], dit["embed"]
    q, q_diff = g.split_last(g.linear(x, w("cross_attn.to_q.weight")), 2)
    k, k_diff, v = g.split_last(g.linear(context, w("cross_attn.to_kv.weight")), 3)
    q, q_diff = (_heads(g, item, length, heads, head_dim) for item in (q, q_diff))
    k, k_diff, v = (_heads(g, item, context_length, heads, head_dim) for item in (k, k_diff, v))
    q_gamma, k_gamma, eps = w("cross_attn.q_norm.gamma"), w("cross_attn.k_norm.gamma"), dit["qk_norm_eps"]
    q, q_diff = (g.rms_norm(item, q_gamma, eps) for item in (q, q_diff))
    k, k_diff = (g.rms_norm(item, k_gamma, eps) for item in (k, k_diff))
    out = _differential(g, q, q_diff, k, k_diff, v, head_dim ** -0.5)
    return g.linear(_merge(g, out, length, embed), w("cross_attn.to_out.weight"))


def _block(g: Graph, x, context, global_cond, weights: dict, dit: dict, layer: int, rope, length,
           context_length: int, mask, local):
    prefix = f"transformer.layers.{layer}."

    def w(name):
        return weights[prefix + name]

    embed, eps = dit["embed"], dit["norm_eps"]
    modulation = g.add(global_cond, g.const(w("to_scale_shift_gate"), shape=(1, 1, 6 * embed)))
    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = g.split_last(modulation, 6)

    def gated(branch, gate):
        return g.mul(branch, g.act(g.add_scalar(g.mul_scalar(gate, -1.0), 1.0), trt.ActivationType.SIGMOID))

    residual = x
    h = g.rms_norm(x, w("pre_norm.gamma"), eps)
    h = g.add(g.mul(h, g.add_scalar(scale_self, 1.0)), shift_self)
    h = _self_attention(g, h, w, dit, rope, length, mask)
    x = g.add(gated(h, gate_self), residual)

    h = g.rms_norm(x, w("cross_attend_norm.gamma"), eps)
    x = g.add(x, _cross_attention(g, h, context, w, dit, length, context_length))

    x = g.add(x, local)

    residual = x
    h = g.rms_norm(x, w("ff_norm.gamma"), eps)
    h = g.add(g.mul(h, g.add_scalar(scale_ff, 1.0)), shift_ff)
    value, gate = g.split_last(g.linear(h, w("ff.ff.0.proj.weight"), w("ff.ff.0.proj.bias")), 2)
    h = g.linear(g.mul(value, g.silu(gate)), w("ff.ff.2.weight"), w("ff.ff.2.bias"))
    return g.add(gated(h, gate_ff), residual)


def _key_mask(g: Graph, valid, memory: int, max_tokens: int, tokens):
    """Additive [1, 1, 1, N] key mask that hides the padded latents from every query."""
    positions = g.slice(g.const(np.arange(max_tokens, dtype=np.int32), trt.int32), (0,), (tokens,))
    limit = g.add(valid, g.const(np.array([memory], dtype=np.int32), trt.int32))
    keep = g.n.add_elementwise(positions, limit, trt.ElementWiseOperation.LESS).get_output(0)
    bias = g.n.add_select(keep, g.const(np.zeros(1, dtype=np.float32), trt.float32),
                          g.const(np.full(1, MASKED, dtype=np.float32), trt.float32)).get_output(0)
    return g.reshape(bias, [1, 1, 1, tokens])


def populate(network: trt.INetworkDefinition, weights: dict, cfg: dict, work: trt.DataType,
             max_latents: int) -> Graph:
    dit = cfg["dit"]
    g = Graph(network, work)
    channels, embed, memory = dit["io_channels"], dit["embed"], dit["memory_tokens"]
    context_length = cfg["text"]["max_length"] + 1
    cond = dit["cond_dim"]

    x_in = network.add_input(LATENTS, trt.float32, (1, channels, -1))
    t_in = network.add_input(TIME, trt.float32, (1,))
    context_in = network.add_input(CONTEXT, trt.float32, (1, context_length, cond))
    global_in = network.add_input(GLOBAL, trt.float32, (1, cond))
    valid_in = network.add_input(VALID, trt.int32, (1,))

    length = g.dim(x_in, 2)
    tokens = g.int_add(length, memory)
    max_tokens = max_latents + memory

    # Conditioning embeddings.
    context = _mlp(g, g.cast(context_in, work), weights["to_cond_embed.0.weight"], weights["to_cond_embed.2.weight"])
    global_embed = _mlp(g, g.cast(global_in, work), weights["to_global_embed.0.weight"],
                        weights["to_global_embed.2.weight"])
    t_model = g.cast(g.cast(g.shuffle(t_in, reshape=(1, 1)), work), trt.float32)
    freqs = expo_fourier_frequencies(dit["timestep_features"], dit["fourier_min_freq"], dit["fourier_max_freq"])
    t_features = g.cast(expo_fourier_features(g, t_model, freqs), work)
    t_embed = _mlp(g, t_features, weights["to_timestep_embed.0.weight"], weights["to_timestep_embed.2.weight"],
                   weights["to_timestep_embed.0.bias"], weights["to_timestep_embed.2.bias"])
    global_embed = g.add(global_embed, t_embed)
    global_cond = _mlp(g, global_embed, weights["transformer.global_cond_embedder.0.weight"],
                       weights["transformer.global_cond_embedder.2.weight"],
                       weights["transformer.global_cond_embedder.0.bias"],
                       weights["transformer.global_cond_embedder.2.bias"])
    global_cond = g.reshape(global_cond, [1, 1, 6 * embed])

    # Latents in: 1x1 conv with residual, then tokens.
    x = g.cast(x_in, work)
    x = g.add(g.matmul(g.const(weights["preprocess_conv.weight"][:, :, 0], shape=(1, channels, channels)), x), x)
    x = g.linear(g.transpose(x, (0, 2, 1)), weights["transformer.project_in.weight"])
    x = g.concat([g.const(weights["transformer.memory_tokens"], shape=(1, memory, embed)), x], 1)

    tables = rotary_tables(g, weights["transformer.rotary_pos_emb.inv_freq"], max_tokens)
    rope = tuple(g.reshape(g.slice(table, (0, 0), (tokens, dit["rope_dim"])), [1, 1, tokens, dit["rope_dim"]])
                 for table in tables)
    mask = _key_mask(g, valid_in, memory, max_tokens, tokens)
    # 0 on the memory tokens, 1 on the latents: the local embedding is left-padded.
    on_latents = g.slice(g.const(np.concatenate([np.zeros(memory), np.ones(max_latents)]), shape=(1, max_tokens, 1)),
                         (0, 0, 0), (1, tokens, 1))

    for layer in range(dit["depth"]):
        local = g.mul(on_latents, g.const(local_embedding(weights, layer), shape=(1, 1, embed)))
        x = _block(g, x, context, global_cond, weights, dit, layer, rope, tokens, context_length, mask, local)

    # Tokens out: drop memory tokens, project, back to channels-first, 1x1 conv with residual.
    x = g.slice(x, (0, memory, 0), (1, length, embed))
    x = g.transpose(g.linear(x, weights["transformer.project_out.weight"]), (0, 2, 1))
    x = g.add(g.matmul(g.const(weights["postprocess_conv.weight"][:, :, 0], shape=(1, channels, channels)), x), x)
    out = g.cast(x, trt.float32)
    out.name = VELOCITY
    network.mark_output(out)
    return g

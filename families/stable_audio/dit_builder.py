# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``StableAudioDiTModel`` denoiser as one TensorRT network.

Engine I/O (batch 1 or 2, the two classifier-free guidance branches):

* ``hidden_states`` [B, C, T] float32, the scheduler-scaled latents;
* ``timestep`` [1] float32, the scheduler's preconditioned noise level;
* ``encoder_hidden_states`` [B, S, D] and ``global_hidden_states`` [B, 1, 2 D] from the
  conditioning engine;
* ``sample`` [B, C, T] float32, the model output.

In an FP16 build the projection / attention operands are FP16 values; the projections
multiply FP16-rounded weights as FP32 and accumulate in FP32 (see
``Graph._fp32_operands``), and the residual stream, the layer norms, the rotary embedding,
the timestep embedding and the 1x1 pre / post convolutions stay in FP32 (diffusers itself
rotates queries and keys in FP32).
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .graph_ops import Graph


def rotary_tables(positions: int, head_dim: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """cos / sin over the whole head: the rotary half as in ``get_1d_rotary_pos_embed``
    (``use_real=True, repeat_interleave_real=False``), identity on the rest."""
    exponents = np.arange(0, rotary_dim, 2, dtype=np.float32)[: rotary_dim // 2] / np.float32(rotary_dim)
    frequencies = np.float32(1.0) / (np.float32(10000.0) ** exponents)
    angles = np.outer(np.arange(positions, dtype=np.float32), frequencies).astype(np.float32)
    cos = np.ones((positions, head_dim), dtype=np.float32)
    sin = np.zeros((positions, head_dim), dtype=np.float32)
    cos[:, :rotary_dim] = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin[:, :rotary_dim] = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos, sin


def rotate_half_plan(head_dim: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Gather indices and signs that turn x into ``cat(-x2, x1)`` on the rotary half, 0 elsewhere."""
    half = rotary_dim // 2
    index = np.arange(head_dim, dtype=np.int32)
    sign = np.zeros(head_dim, dtype=np.float32)
    index[:half] = np.arange(half, rotary_dim)
    sign[:half] = -1.0
    index[half:rotary_dim] = np.arange(0, half)
    sign[half:rotary_dim] = 1.0
    return index, sign


def _expand_kv(weight: np.ndarray, kv_heads: int, heads: int, head_dim: int) -> np.ndarray:
    """Repeat each key/value head's rows so the projection emits ``repeat_interleave``-d heads."""
    weight = np.asarray(weight, dtype=np.float32)
    rows = weight.reshape(kv_heads, head_dim, weight.shape[1])
    return np.repeat(rows, heads // kv_heads, axis=0).reshape(heads * head_dim, weight.shape[1])


class _Denoiser:
    def __init__(self, g: Graph, weights: dict, cfg: dict, work: trt.DataType) -> None:
        self.g = g
        self.w = weights
        self.cfg = cfg
        self.work = work
        self.heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.inner = cfg["inner_dim"]
        self.tokens = cfg["sample_size"] + 1
        cos, sin = rotary_tables(self.tokens, self.head_dim, cfg["rotary_dim"])
        shape = (1, 1, self.tokens, self.head_dim)
        self.cos = g.constant(cos, trt.float32, shape=shape)
        self.sin = g.constant(sin, trt.float32, shape=shape)
        index, sign = rotate_half_plan(self.head_dim, cfg["rotary_dim"])
        self.rotate_index = index
        self.rotate_sign = g.constant(sign, trt.float32, shape=(1, 1, 1, self.head_dim))

    def linear(self, tensor, name: str, *, bias: bool = False, weight=None, out=None):
        """A projection on ``work``-dtype operands; ``out`` FP32 for results that feed FP32 sums."""
        g = self.g
        source = g.cast(tensor, self.work)
        return g.linear(source, self.w[f"{name}.weight"] if weight is None else weight,
                        self.w[f"{name}.bias"] if bias else None, out_dtype=out)

    def split_heads(self, tensor, tokens: int):
        return self.g.reshape(tensor, (0, tokens, self.heads, self.head_dim), second=(0, 2, 1, 3))

    def rotate(self, tensor):
        """Rotary embedding in FP32, as ``StableAudioAttnProcessor2_0`` applies it."""
        g = self.g
        source = g.cast(tensor, trt.float32)
        rotated = g.mul(g.gather(source, self.rotate_index, 3), self.rotate_sign)
        return g.cast(g.add(g.mul(source, self.cos), g.mul(rotated, self.sin)), self.work)

    def attend(self, query, key, value, tokens: int):
        context = self.g.attention(query, key, value, 1.0 / math.sqrt(self.head_dim))
        return self.g.reshape(context, (0, tokens, self.inner), first=(0, 2, 1, 3))

    def block(self, hidden, cross, index: int):
        g = self.g
        prefix = f"transformer_blocks.{index}"
        tokens = self.tokens
        eps = 1e-5

        normed = g.layer_norm(hidden, self.w[f"{prefix}.norm1.weight"], self.w[f"{prefix}.norm1.bias"], eps)
        query = self.rotate(self.split_heads(self.linear(normed, f"{prefix}.attn1.to_q"), tokens))
        key = self.rotate(self.split_heads(self.linear(normed, f"{prefix}.attn1.to_k"), tokens))
        value = self.split_heads(self.linear(normed, f"{prefix}.attn1.to_v"), tokens)
        attended = self.linear(self.attend(query, key, value, tokens), f"{prefix}.attn1.to_out.0",
                               out=trt.float32)
        hidden = g.add(hidden, attended)

        normed = g.layer_norm(hidden, self.w[f"{prefix}.norm2.weight"], self.w[f"{prefix}.norm2.bias"], eps)
        query = self.split_heads(self.linear(normed, f"{prefix}.attn2.to_q"), tokens)
        context_tokens = int(cross.shape[1])
        kv_heads = self.cfg["kv_heads"]
        key = self.split_heads(self.linear(cross, "", weight=_expand_kv(
            self.w[f"{prefix}.attn2.to_k.weight"], kv_heads, self.heads, self.head_dim)), context_tokens)
        value = self.split_heads(self.linear(cross, "", weight=_expand_kv(
            self.w[f"{prefix}.attn2.to_v.weight"], kv_heads, self.heads, self.head_dim)), context_tokens)
        attended = self.linear(self.attend(query, key, value, tokens), f"{prefix}.attn2.to_out.0",
                               out=trt.float32)
        hidden = g.add(hidden, attended)

        normed = g.layer_norm(hidden, self.w[f"{prefix}.norm3.weight"], self.w[f"{prefix}.norm3.bias"], eps)
        # SwiGLU: one projection to 2 x inner, value half times SiLU of the gate half.
        projection = np.asarray(self.w[f"{prefix}.ff.net.0.proj.weight"], dtype=np.float32)
        projection_bias = np.asarray(self.w[f"{prefix}.ff.net.0.proj.bias"], dtype=np.float32)
        width = projection.shape[0] // 2
        source = g.cast(normed, self.work)
        value_half = g.linear(source, projection[:width], projection_bias[:width])
        gate_half = g.linear(source, projection[width:], projection_bias[width:])
        mixed = g.mul(value_half, g.silu(gate_half))
        out = g.linear(mixed, self.w[f"{prefix}.ff.net.2.weight"], self.w[f"{prefix}.ff.net.2.bias"],
                       out_dtype=trt.float32)
        return g.add(hidden, out)


def populate(network: trt.INetworkDefinition, weights: dict, cfg: dict, work: trt.DataType) -> Graph:
    """Add the denoiser network; ``work`` is the projection / attention dtype."""
    g = Graph(network, tf32_engine=work == trt.float16)
    dit = cfg["transformer"]
    channels, frames = dit["in_channels"], dit["sample_size"]
    context_tokens = cfg["tokenizer"]["max_length"] + 2
    sample = network.add_input("hidden_states", trt.float32, (-1, channels, frames))
    timestep = network.add_input("timestep", trt.float32, (1,))
    encoder_states = network.add_input(
        "encoder_hidden_states", trt.float32, (-1, context_tokens, dit["cross_attention_input_dim"]))
    global_states = network.add_input(
        "global_hidden_states", trt.float32, (-1, 1, dit["global_states_input_dim"]))
    model = _Denoiser(g, weights, dit, work)

    # GaussianFourierProjection(flip_sin_to_cos=True, log=False): [cos, sin] of t * w * 2 * pi.
    frequencies = np.asarray(weights["time_proj.weight"], dtype=np.float32)
    angles = g.mul(g.reshape(timestep, (1, 1)), g.constant(frequencies, trt.float32, shape=(1, frequencies.size)))
    angles = g.mul(g.mul(angles, g.scalar(2.0, angles)), g.scalar(math.pi, angles))
    time_features = g.concat([g.unary(angles, trt.UnaryOperation.COS), g.unary(angles, trt.UnaryOperation.SIN)], 1)
    time_embedding = g.linear(time_features, weights["timestep_proj.0.weight"], weights["timestep_proj.0.bias"])
    time_embedding = g.linear(g.silu(time_embedding), weights["timestep_proj.2.weight"],
                              weights["timestep_proj.2.bias"])

    global_token = model.linear(g.silu(model.linear(global_states, "global_proj.0")), "global_proj.2",
                                out=trt.float32)
    global_token = g.add(global_token, g.reshape(time_embedding, (1, 1, dit["inner_dim"])))
    cross = model.linear(g.silu(model.linear(encoder_states, "cross_attention_proj.0")), "cross_attention_proj.2")

    # preprocess_conv (1x1, no bias) with its residual, in the [B, T, C] layout.
    frames_first = g.reshape(sample, None, first=(0, 2, 1))
    pre = np.asarray(weights["preprocess_conv.weight"], dtype=np.float32)[:, :, 0]
    frames_first = g.add(frames_first, g.linear(frames_first, pre))
    hidden = model.linear(frames_first, "proj_in", out=trt.float32)
    hidden = g.concat([global_token, hidden], axis=1)

    for index in range(dit["num_layers"]):
        hidden = model.block(hidden, cross, index)

    out = model.linear(hidden, "proj_out", out=trt.float32)
    out = g.gather(out, np.arange(1, frames + 1), 1)  # drop the prepended global token
    post = np.asarray(weights["postprocess_conv.weight"], dtype=np.float32)[:, :, 0]
    out = g.add(out, g.linear(out, post))
    result = g.reshape(out, None, first=(0, 2, 1))
    result.name = "sample"
    network.mark_output(result)
    return g

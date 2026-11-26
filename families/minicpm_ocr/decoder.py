# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-OCR text decoder: the Qwen3.5 hybrid stack as a one-token step engine.

Gated DeltaNet ("linear_attention") layers carry a depthwise causal-conv state and an FP32 recurrent state; gated
full-attention layers read a fixed KV cache and return the new key/value row. A prompt position that holds an image
token takes a vision feature row instead of the token embedding (``use_visual_embedding``). The math follows
Transformers' ``modeling_qwen3_5``: zero-centred RMSNorm, gated RMSNorm, ``l2norm`` with eps 1e-6, the delta rule in
FP32, partial rotary embedding with float32 angles, the sigmoid output gate before ``o_proj`` and a tied LM head.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint, ModelConfig
from .graph import Graph

MASKED_SCORE = -1.0e4


def rope_tables(positions: int, rotary_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin [positions, rotary_dim / 2] with the reference's float32 inverse frequencies and angles."""
    exponents = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    inv_freq = (np.float32(1.0) / np.power(np.float32(theta), exponents)).astype(np.float32)
    angles = np.arange(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = angles.astype(np.float32)
    return (np.cos(angles.astype(np.float64)).astype(np.float32),
            np.sin(angles.astype(np.float64)).astype(np.float32))


class _Builder:
    """Holds the graph, the checkpoint and the text geometry while one engine is built."""

    def __init__(self, g: Graph, checkpoint: Checkpoint, config: ModelConfig, capacity: int) -> None:
        self.g = g
        self.ck = checkpoint
        self.t = config.text
        self.capacity = capacity
        t = self.t
        self.key_dim = t.linear_key_heads * t.linear_key_dim
        self.value_dim = t.linear_value_heads * t.linear_value_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kv_dim = t.num_kv_heads * t.head_dim

    def weight(self, name: str, shape) -> np.ndarray:
        return self.ck.tensor(name, shape)

    def projection(self, name: str, out_width: int, in_width: int) -> np.ndarray:
        return np.ascontiguousarray(self.weight(name, (out_width, in_width)).T)

    def centred_norm(self, x, name: str, width: int):
        # Qwen3_5RMSNorm: norm in FP32, scale by (1 + w), cast back.
        return self.g.rms_norm(x, 1.0 + self.weight(name, (width,)), self.t.rms_norm_eps)

    def mlp(self, x, prefix: str):
        t, g = self.t, self.g
        gate = g.linear(x, self.projection(f"{prefix}.mlp.gate_proj.weight", t.intermediate_size, t.hidden_size))
        up = g.linear(x, self.projection(f"{prefix}.mlp.up_proj.weight", t.intermediate_size, t.hidden_size))
        hidden = g.mul(g.silu(gate), up)
        return g.linear(hidden, self.projection(f"{prefix}.mlp.down_proj.weight", t.hidden_size,
                                                t.intermediate_size))

    def softplus(self, x):
        """F.softplus (beta 1, threshold 20) in FP32 with an accurate log1p."""
        g = self.g
        clipped = g.binary(x, g.scalar(20.0, trt.float32, 2), trt.ElementWiseOperation.MIN)
        e = g.unary(clipped, trt.UnaryOperation.EXP)
        u = g.add(e, g.scalar(1.0, trt.float32, 2))
        d = g.sub(u, g.scalar(1.0, trt.float32, 2))
        # log1p(e) = log(u) * e / (u - 1); exact e when 1 + e rounds to 1.
        safe_d = g.binary(d, g.scalar(1e-30, trt.float32, 2), trt.ElementWiseOperation.MAX)
        log1p = g.div(g.mul(g.unary(u, trt.UnaryOperation.LOG), e), safe_d)
        tiny = g.binary(d, g.scalar(0.0, trt.float32, 2), trt.ElementWiseOperation.EQUAL)
        log1p = g.network.add_select(tiny, e, log1p).get_output(0)
        above = g.binary(x, g.scalar(20.0, trt.float32, 2), trt.ElementWiseOperation.GREATER)
        return g.network.add_select(above, x, log1p).get_output(0)

    def delta_net(self, x, conv_state, ssm_state, prefix: str):
        g, t = self.g, self.t
        p = f"{prefix}.linear_attn"
        heads, key_dim, value_dim = t.linear_value_heads, t.linear_key_dim, t.linear_value_dim
        qkv = g.linear(x, self.projection(f"{p}.in_proj_qkv.weight", self.conv_dim, t.hidden_size))
        z = g.linear(x, self.projection(f"{p}.in_proj_z.weight", self.value_dim, t.hidden_size))
        b = g.linear(x, self.projection(f"{p}.in_proj_b.weight", heads, t.hidden_size))
        a = g.linear(x, self.projection(f"{p}.in_proj_a.weight", heads, t.hidden_size))

        # Depthwise causal conv over [previous kernel - 1 inputs, current input], then SiLU.
        kernel = t.linear_conv_kernel
        window = g.concat([g.slice(g.to_work(conv_state), (0, 1), (self.conv_dim, kernel - 1)),
                           g.reshape(qkv, (self.conv_dim, 1))], axis=1)
        present_conv = g.fp32(window)
        conv_w = g.constant(g.rounded(self.weight(f"{p}.conv1d.weight", (self.conv_dim, 1, kernel))
                                      .reshape(self.conv_dim, kernel)))
        # Products of build-precision values are exact in FP32; accumulate in FP32.
        conv = g.reduce(g.mul(present_conv, conv_w), trt.ReduceOperation.SUM, 1)
        if self.ck.has(f"{p}.conv1d.bias"):
            conv = g.add(conv, g.constant(g.rounded(self.weight(f"{p}.conv1d.bias", (self.conv_dim,)))
                                          .reshape(self.conv_dim, 1)))
        mixed = g.silu(g.to_work(conv))
        mixed = g.reshape(mixed, (1, self.conv_dim))
        q = g.reshape(g.slice(mixed, (0, 0), (1, self.key_dim)), (heads, key_dim))
        k = g.reshape(g.slice(mixed, (0, self.key_dim), (1, self.key_dim)), (heads, key_dim))
        v = g.reshape(g.slice(mixed, (0, 2 * self.key_dim), (1, self.value_dim)), (heads, value_dim))

        def l2norm(tensor):
            tensor = g.fp32(tensor)
            total = g.reduce(g.mul(tensor, tensor), trt.ReduceOperation.SUM, 1)
            inv = g.unary(g.unary(g.add(total, g.scalar(1e-6, trt.float32, 2)), trt.UnaryOperation.SQRT),
                          trt.UnaryOperation.RECIP)
            return g.mul(tensor, inv)

        q = g.mul(l2norm(q), g.scalar(key_dim ** -0.5, trt.float32, 2))
        k = l2norm(k)
        v = g.fp32(v)
        beta = g.reshape(g.fp32(g.activation(b, trt.ActivationType.SIGMOID)), (heads, 1))
        a_log = self.weight(f"{p}.A_log", (heads,))
        decay_rate = g.constant(-np.exp(g.rounded(a_log)).reshape(1, heads))
        dt_bias = g.constant(g.rounded(self.weight(f"{p}.dt_bias", (heads,))).reshape(1, heads))
        log_decay = g.mul(decay_rate, self.softplus(g.add(g.fp32(a), dt_bias)))
        decay = g.reshape(g.unary(log_decay, trt.UnaryOperation.EXP), (heads, 1, 1))

        # Delta rule on S [heads, key, value], all in FP32 with elementwise products and sums.
        state = g.mul(ssm_state, decay)
        k_col = g.reshape(k, (heads, key_dim, 1))
        memory = g.reshape(g.reduce(g.mul(state, k_col), trt.ReduceOperation.SUM, 1), (heads, value_dim))
        delta = g.mul(g.sub(v, memory), beta)
        state = g.add(state, g.mul(k_col, g.reshape(delta, (heads, 1, value_dim))))
        q_col = g.reshape(q, (heads, key_dim, 1))
        core = g.reshape(g.reduce(g.mul(state, q_col), trt.ReduceOperation.SUM, 1), (heads, value_dim))

        # Qwen3_5RMSNormGated: norm in FP32, cast, scale in the build precision, gate with FP32 SiLU.
        core = g.to_work(core)
        norm_w = g.rounded(self.weight(f"{p}.norm.weight", (value_dim,)))
        normed = g.rms_norm(core, np.ones(value_dim, dtype=np.float32), t.rms_norm_eps)
        normed = g.mul(normed, g.constant(norm_w.reshape(1, value_dim), g.np_work))
        gate = g.silu32(g.reshape(z, (heads, value_dim)))
        gated = g.to_work(g.mul(g.fp32(normed), gate))
        out = g.linear(g.reshape(gated, (1, self.value_dim)),
                       self.projection(f"{p}.out_proj.weight", t.hidden_size, self.value_dim))
        return out, present_conv, state

    def attention(self, x, cache_k, cache_v, mask, position, cos, sin, prefix: str):
        g, t = self.g, self.t
        p = f"{prefix}.self_attn"
        heads, kv_heads, head_dim = t.num_heads, t.num_kv_heads, t.head_dim
        q_proj = self.projection(f"{p}.q_proj.weight", 2 * heads * head_dim, t.hidden_size)
        # The q projection interleaves per head: [query (head_dim), gate (head_dim)].
        columns = np.arange(2 * heads * head_dim).reshape(heads, 2 * head_dim)
        query = g.linear(x, np.ascontiguousarray(q_proj[:, columns[:, :head_dim].reshape(-1)]))
        gate = g.linear(x, np.ascontiguousarray(q_proj[:, columns[:, head_dim:].reshape(-1)]))
        key = g.linear(x, self.projection(f"{p}.k_proj.weight", self.kv_dim, t.hidden_size))
        value = g.linear(x, self.projection(f"{p}.v_proj.weight", self.kv_dim, t.hidden_size))

        def norm_heads(tensor, count, name):
            rows = g.reshape(tensor, (1, count, head_dim))
            normed = g.rms_norm(rows, 1.0 + self.weight(f"{p}.{name}.weight", (head_dim,)), t.rms_norm_eps)
            return normed  # [1, count, head_dim]

        def rotate(tensor, count):
            # [1, count, head_dim] -> [1, count, 1, head_dim] for the rotary layer, positions [1, 1].
            rows = g.reshape(tensor, (1, count, 1, head_dim))
            layer = g.network.add_rotary_embedding(rows, cos, sin, False, t.rotary_dim)
            layer.set_input(3, g.reshape(position, (1, 1)))
            return layer.get_output(0)

        q = rotate(norm_heads(query, heads, "q_norm"), heads)  # [1, heads, 1, dim]
        k = rotate(norm_heads(key, kv_heads, "k_norm"), kv_heads)  # [1, kv_heads, 1, dim]
        k_row = g.reshape(k, (1, self.kv_dim))
        v_row = g.reshape(value, (1, self.kv_dim))
        keys = g.concat([cache_k, k_row], axis=0)
        values = g.concat([cache_v, v_row], axis=0)
        window = self.capacity + 1

        def heads_first(rows):
            return g.transpose(g.reshape(rows, (window, kv_heads, head_dim)), (1, 0, 2))

        context = g.grouped_query_attention(g.reshape(q, (heads, head_dim)), heads_first(keys), heads_first(values),
                                            g.reshape(mask, (window,)), scale=head_dim ** -0.5)
        gated = g.mul(context, g.activation(gate, trt.ActivationType.SIGMOID))
        out = g.linear(gated, self.projection(f"{p}.o_proj.weight", t.hidden_size, heads * head_dim))
        return out, k_row, v_row


def build_decoder_engine(checkpoint: Checkpoint, config: ModelConfig, precision: str, capacity: int,
                         verbose: bool) -> bytes:
    t = config.text
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    if precision == "fp32":
        build_config.clear_flag(trt.BuilderFlag.TF32)
    else:
        # FP32 matrix products only ever multiply FP16 values cast to FP32 (exact in TF32); the recurrence uses
        # elementwise products and sums, which TF32 does not touch.
        build_config.set_flag(trt.BuilderFlag.TF32)
    g = Graph(network, trt.float16 if precision == "fp16" else trt.float32)
    b = _Builder(g, checkpoint, config, capacity)

    token = network.add_input("token_id", trt.int32, (1,))
    position = network.add_input("position_id", trt.int32, (1,))
    mask = network.add_input("attention_mask", trt.float32, (1, capacity + 1))
    visual = network.add_input("visual_embedding", g.work, (1, t.hidden_size))
    use_visual = network.add_input("use_visual_embedding", trt.int32, (1, 1))
    linear_layers = [i for i, kind in enumerate(t.layer_types) if kind == "linear_attention"]
    attention_layers = [i for i, kind in enumerate(t.layer_types) if kind == "full_attention"]
    conv_shape = (b.conv_dim, t.linear_conv_kernel)
    ssm_shape = (t.linear_value_heads, t.linear_key_dim, t.linear_value_dim)
    conv_inputs = [network.add_input(f"conv_state_{i}", trt.float32, conv_shape) for i in range(len(linear_layers))]
    ssm_inputs = [network.add_input(f"ssm_state_{i}", trt.float32, ssm_shape) for i in range(len(linear_layers))]
    cache_k = [network.add_input(f"cache_k_{i}", g.work, (capacity, b.kv_dim)) for i in range(len(attention_layers))]
    cache_v = [network.add_input(f"cache_v_{i}", g.work, (capacity, b.kv_dim)) for i in range(len(attention_layers))]

    embedding = checkpoint.tensor("model.language_model.embed_tokens.weight", (t.vocab_size, t.hidden_size))
    table = g.constant(embedding, g.np_work)
    x = network.add_gather(table, token, 0).get_output(0)
    x = network.add_select(g.cast(use_visual, trt.bool), visual, g.reshape(x, (1, t.hidden_size))).get_output(0)

    cos, sin = rope_tables(capacity + 1, t.rotary_dim, t.rope_theta)
    cos_t, sin_t = g.constant(cos, g.np_work), g.constant(sin, g.np_work)
    attention_mask = mask  # FP32 additive scores, applied in FP32
    presents = []
    linear_index = attention_index = 0
    for layer, kind in enumerate(t.layer_types):
        prefix = f"model.language_model.layers.{layer}"
        normed = b.centred_norm(x, f"{prefix}.input_layernorm.weight", t.hidden_size)
        if kind == "linear_attention":
            out, present_conv, present_ssm = b.delta_net(normed, conv_inputs[linear_index], ssm_inputs[linear_index],
                                                         prefix)
            presents += [(f"present_conv_{linear_index}", present_conv, trt.float32),
                         (f"present_ssm_{linear_index}", present_ssm, trt.float32)]
            linear_index += 1
        else:
            out, k_row, v_row = b.attention(normed, cache_k[attention_index], cache_v[attention_index],
                                            attention_mask, position, cos_t, sin_t, prefix)
            presents += [(f"present_k_{attention_index}", k_row, g.work),
                         (f"present_v_{attention_index}", v_row, g.work)]
            attention_index += 1
        x = g.add(x, out)
        x = g.add(x, b.mlp(b.centred_norm(x, f"{prefix}.post_attention_layernorm.weight", t.hidden_size), prefix))

    x = b.centred_norm(x, "model.language_model.norm.weight", t.hidden_size)
    logits = g.linear(x, np.ascontiguousarray(embedding.T))
    logits = g.fp32(logits)
    logits.name = "logits"
    network.mark_output(logits)
    for name, tensor, dtype in presents:
        tensor = g.cast(tensor, dtype)
        tensor.name = name
        network.mark_output(tensor)

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("minicpm_ocr: TensorRT failed to build the decoder engine")
    return bytes(plan)

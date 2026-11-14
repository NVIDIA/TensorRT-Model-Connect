# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-token MiniCPM-V 4.7 language decoder (Qwen3.5-MoE text): Gated DeltaNet, gated attention, experts.

One engine step consumes one prompt or generated token at its KV slot ``position_id``. A prompt token
may carry a visual embedding row instead of its token embedding, and every token carries its own
canvas M-RoPE coordinates as ``position_id + rotary_position_offsets``. DeltaNet conv/recurrent state
and attention K/V rows are explicit engine inputs and outputs owned by the runtime.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .config import TextConfig
from .graph import E, Graph

_MASKED = -1.0e4


def rope_table(capacity: int, config: TextConfig, cosine: bool) -> np.ndarray:
    """[capacity, rotary_dim / 2] cos or sin of ``position * inv_freq`` evaluated in FP32 like the source."""
    rotary = config.rotary_dim
    exponent = np.arange(0, rotary, 2, dtype=np.float32) / np.float32(rotary)
    inv_freq = (np.float32(1.0) / np.power(np.float32(config.rope_theta), exponent)).astype(np.float32)
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv_freq[None, :]
    return (np.cos(angles) if cosine else np.sin(angles)).astype(np.float32)


def mrope_axes(config: TextConfig) -> np.ndarray:
    """Interleaved (T, H, W) owner of each rotary frequency, as Qwen3.5 recomposes them."""
    half = config.rotary_dim // 2
    axes = np.zeros(half, dtype=np.int32)
    for axis in (1, 2):
        axes[axis:config.mrope_section[axis] * 3:3] = axis
    return axes


class _Layer:
    def __init__(self, checkpoint: Checkpoint, config: TextConfig, index: int):
        prefix = f"model.language_model.layers.{index}"
        hidden = config.hidden_size
        get = checkpoint.get
        self.kind = config.layer_types[index]
        self.input_norm = get(f"{prefix}.input_layernorm.weight", (hidden,))
        self.post_norm = get(f"{prefix}.post_attention_layernorm.weight", (hidden,))
        if self.kind == "linear_attention":
            mixer = f"{prefix}.linear_attn"
            heads = config.linear_num_value_heads
            self.in_proj_qkv = get(f"{mixer}.in_proj_qkv.weight", (config.conv_dim, hidden))
            self.in_proj_z = get(f"{mixer}.in_proj_z.weight", (config.value_dim, hidden))
            self.in_proj_a = get(f"{mixer}.in_proj_a.weight", (heads, hidden))
            self.in_proj_b = get(f"{mixer}.in_proj_b.weight", (heads, hidden))
            self.a_log = get(f"{mixer}.A_log", (heads,))
            self.dt_bias = get(f"{mixer}.dt_bias", (heads,))
            self.conv = get(f"{mixer}.conv1d.weight", (config.conv_dim, 1, config.linear_conv_kernel_dim))
            self.norm = get(f"{mixer}.norm.weight", (config.linear_value_head_dim,))
            self.out_proj = get(f"{mixer}.out_proj.weight", (hidden, config.value_dim))
        else:
            mixer = f"{prefix}.self_attn"
            q_width = config.num_attention_heads * config.head_dim
            kv_width = config.num_key_value_heads * config.head_dim
            self.q_proj = get(f"{mixer}.q_proj.weight", (2 * q_width, hidden))
            self.k_proj = get(f"{mixer}.k_proj.weight", (kv_width, hidden))
            self.v_proj = get(f"{mixer}.v_proj.weight", (kv_width, hidden))
            self.o_proj = get(f"{mixer}.o_proj.weight", (hidden, q_width))
            self.q_norm = get(f"{mixer}.q_norm.weight", (config.head_dim,))
            self.k_norm = get(f"{mixer}.k_norm.weight", (config.head_dim,))
        mlp = f"{prefix}.mlp"
        experts, width = config.num_experts, config.moe_intermediate_size
        shared = config.shared_expert_intermediate_size
        self.router = get(f"{mlp}.gate.weight", (experts, hidden))
        self.gate_up = get(f"{mlp}.experts.gate_up_proj", (experts, 2 * width, hidden))
        self.down = get(f"{mlp}.experts.down_proj", (experts, hidden, width))
        self.shared_gate_proj = get(f"{mlp}.shared_expert.gate_proj.weight", (shared, hidden))
        self.shared_up_proj = get(f"{mlp}.shared_expert.up_proj.weight", (shared, hidden))
        self.shared_down_proj = get(f"{mlp}.shared_expert.down_proj.weight", (hidden, shared))
        self.shared_expert_gate = get(f"{mlp}.shared_expert_gate.weight", (1, hidden))


def load_decoder(checkpoint: Checkpoint, config: TextConfig) -> dict:
    hidden, vocab = config.hidden_size, config.vocab_size
    return {
        "embedding": checkpoint.get("model.language_model.embed_tokens.weight", (vocab, hidden)),
        "layers": [_Layer(checkpoint, config, i) for i in range(config.num_hidden_layers)],
        "final_norm": checkpoint.get("model.language_model.norm.weight", (hidden,)),
        "lm_head": checkpoint.get("lm_head.weight", (vocab, hidden)),
    }


def _deltanet(g: Graph, x: trt.ITensor, conv_in: trt.ITensor, state_in: trt.ITensor, layer: _Layer,
              config: TextConfig):
    """Gated DeltaNet decode step on one token row ``x`` [1, hidden]."""
    k_heads, v_heads = config.linear_num_key_heads, config.linear_num_value_heads
    head = config.linear_key_head_dim
    key_dim, value_dim, conv_dim = config.key_dim, config.value_dim, config.conv_dim
    kernel = config.linear_conv_kernel_dim
    work = x.dtype

    qkv = g.linear(x, layer.in_proj_qkv)  # [1, conv_dim]
    z = g.linear(x, layer.in_proj_z)  # [1, value_dim]
    b = g.linear(x, layer.in_proj_b)  # [1, v_heads]
    a = g.linear(x, layer.in_proj_a)  # [1, v_heads]

    # Causal depthwise conv over the stored window plus this token, FP32 accumulation, one rounding.
    window = g.concat([g.slice(g.cast(conv_in, work), (0, 1), (conv_dim, kernel - 1)),
                       g.reshape(qkv, (conv_dim, 1))], axis=1)  # [conv_dim, kernel]
    weight = g.const(np.asarray(layer.conv, dtype=np.float32).reshape(conv_dim, kernel), trt.float32)
    conv = g.reduce(g.mul(g.f32(window), weight), trt.ReduceOperation.SUM, 1)  # [conv_dim, 1]
    mixed = g.silu(g.cast(g.reshape(conv, (1, conv_dim)), work))

    def heads(tensor, start, width, count):
        return g.reshape(g.slice(tensor, (0, start), (1, width)), (count, head))

    query = g.f32(heads(mixed, 0, key_dim, k_heads))
    key = g.f32(heads(mixed, key_dim, key_dim, k_heads))
    value = g.f32(heads(mixed, 2 * key_dim, value_dim, v_heads))
    repeat = v_heads // k_heads
    if repeat > 1:
        ones = g.const(np.ones((1, repeat, 1), np.float32), trt.float32)
        query = g.reshape(g.mul(g.reshape(query, (k_heads, 1, head)), ones), (v_heads, head))
        key = g.reshape(g.mul(g.reshape(key, (k_heads, 1, head)), ones), (v_heads, head))

    def l2norm(tensor):
        total = g.reduce(g.mul(tensor, tensor), trt.ReduceOperation.SUM, 1)
        root = g.unary(g.add(total, g.scalar(1e-6, 2)), trt.UnaryOperation.SQRT)
        return g.mul(tensor, g.unary(root, trt.UnaryOperation.RECIP))

    query = g.binary(l2norm(query), g.scalar(float(np.float32(head) ** np.float32(0.5)), 2), E.DIV)
    key = l2norm(key)

    beta = g.f32(g.cast(g.sigmoid(g.f32(b)), work))  # [1, v_heads]
    a_log = np.asarray(layer.a_log, dtype=np.float32)
    decay_rate = g.const((-np.exp(a_log)).reshape(1, v_heads), trt.float32)
    dt_bias = g.const(np.asarray(layer.dt_bias, dtype=np.float32).reshape(1, v_heads), trt.float32)
    shifted = g.add(g.f32(a), dt_bias)
    softplus = g.unary(g.add(g.unary(shifted, trt.UnaryOperation.EXP), g.scalar(1.0, 2)), trt.UnaryOperation.LOG)
    # torch.nn.functional.softplus returns its input above the default threshold 20.
    above = g.binary(shifted, g.scalar(20.0, 2), E.GREATER)
    softplus = g.network.add_select(above, shifted, softplus).get_output(0)
    decay = g.unary(g.reshape(g.mul(decay_rate, softplus), (v_heads, 1, 1)), trt.UnaryOperation.EXP)

    state = g.mul(state_in, decay)  # [H, K, V]
    key_column = g.reshape(key, (v_heads, head, 1))
    remembered = g.reduce(g.mul(state, key_column), trt.ReduceOperation.SUM, 1, keep=False)  # [H, V]
    delta = g.mul(g.binary(value, remembered, E.SUB), g.reshape(beta, (v_heads, 1)))
    state = g.add(state, g.mul(key_column, g.reshape(delta, (v_heads, 1, head))))
    core = g.reduce(g.mul(state, g.reshape(query, (v_heads, head, 1))), trt.ReduceOperation.SUM, 1, keep=False)

    # Qwen3_5MoeRMSNormGated: normalize in FP32, round, scale by the weight in the work dtype,
    # then multiply by FP32 SiLU(z) and round once more.
    core = g.cast(core, work)
    core32 = g.f32(core)
    normalized = g.cast(g.mul(core32, g.rsqrt_mean_square(core32, 1, config.rms_norm_eps)), work)
    scaled = g.mul(normalized, g.const(np.asarray(layer.norm).reshape(1, head)))
    z32 = g.f32(g.reshape(z, (v_heads, head)))
    gated = g.cast(g.mul(g.f32(scaled), g.mul(z32, g.sigmoid(z32))), work)
    out = g.linear(g.reshape(gated, (1, value_dim)), layer.out_proj)
    present_conv = g.f32(window)
    return out, present_conv, state


def _attention(g: Graph, x: trt.ITensor, cache_k: trt.ITensor, cache_v: trt.ITensor, mask: trt.ITensor,
               cos: trt.ITensor, sin: trt.ITensor, zero: trt.ITensor, layer: _Layer, config: TextConfig):
    """Gated full attention for one token over the cache window plus itself."""
    heads, kv_heads, head = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    work = x.dtype
    query_gate = g.reshape(g.linear(x, layer.q_proj), (heads, 2 * head))
    query = g.slice(query_gate, (0, 0), (heads, head))
    gate = g.reshape(g.slice(query_gate, (0, head), (heads, head)), (1, heads * head))
    key = g.reshape(g.linear(x, layer.k_proj), (kv_heads, head))
    value = g.reshape(g.linear(x, layer.v_proj), (1, kv_heads * head))
    query = g.rms_norm_centered(query, layer.q_norm, config.rms_norm_eps)
    key = g.rms_norm_centered(key, layer.k_norm, config.rms_norm_eps)

    def rotate(tensor, count):
        tensor = g.reshape(tensor, (1, count, 1, head))
        rope = g.network.add_rotary_embedding(tensor, cos, sin, False, config.rotary_dim)
        rope.set_input(3, zero)
        return g.reshape(rope.get_output(0), (1, count * head))

    query = rotate(query, heads)
    key = rotate(key, kv_heads)

    keys = g.concat([cache_k, key], axis=0)  # [capacity + 1, kv_width]
    values = g.concat([cache_v, value], axis=0)
    window = keys.shape[0]
    group = heads // kv_heads
    q5 = g.reshape(query, (1, kv_heads, group, 1, head))
    k5 = g.reshape(keys, (1, window, kv_heads, 1, head), transpose=(0, 2, 3, 1, 4))
    v5 = g.reshape(values, (1, window, kv_heads, 1, head), transpose=(0, 2, 3, 1, 4))
    mask5 = g.reshape(mask, (1, 1, 1, 1, window))
    context = g.attention(q5, k5, v5, config.head_dim ** -0.5, mask5)  # [1, kv, group, 1, D] FP32
    context = g.cast(g.reshape(context, (1, heads * head)), work)
    gated = g.mul(context, g.cast(g.sigmoid(g.f32(gate)), work))
    return g.linear(gated, layer.o_proj), key, value


def _experts(g: Graph, x: trt.ITensor, layer: _Layer, config: TextConfig) -> trt.ITensor:
    """Softmax top-k routed experts plus the sigmoid-gated shared expert, for one token row."""
    work = x.dtype
    top_k, width, hidden = config.num_experts_per_tok, config.moe_intermediate_size, config.hidden_size
    logits = g.linear(x, layer.router)  # [1, experts], work dtype like the source router
    probabilities = g.softmax(g.f32(logits), 1)
    top = g.network.add_topk(probabilities, trt.TopKOperation.MAX, top_k, 1 << 1)
    weights, indices = top.get_output(0), top.get_output(1)
    weights = g.binary(weights, g.reduce(weights, trt.ReduceOperation.SUM, 1), E.DIV)
    # The source visits hit experts in ascending index order; sort the selection the same way.
    order = g.network.add_topk(g.f32(indices), trt.TopKOperation.MIN, top_k, 1 << 1).get_output(0)
    selected = g.cast(order, trt.int32)  # [1, top_k] ascending
    matches = g.binary(g.reshape(selected, (top_k, 1)), g.reshape(indices, (1, top_k)), E.EQUAL)
    zero = g.const(np.zeros((1, 1), np.float32), trt.float32)
    routed_weights = g.network.add_select(matches, g.reshape(weights, (1, top_k)), zero).get_output(0)
    routed_weights = g.cast(g.reduce(routed_weights, trt.ReduceOperation.SUM, 1), work)  # [top_k, 1]

    flat = g.reshape(selected, (top_k,))
    gate_up = g.gather(g.const(layer.gate_up), flat, 0)  # [top_k, 2 * width, hidden]
    down = g.gather(g.const(layer.down), flat, 0)  # [top_k, hidden, width]
    rows = g.reshape(x, (1, 1, hidden))
    projected = g.matmul(rows, gate_up, transpose_b=True)  # [top_k, 1, 2 * width]
    gate = g.slice(projected, (0, 0, 0), (top_k, 1, width))
    up = g.slice(projected, (0, 0, width), (top_k, 1, width))
    activated = g.mul(g.silu(gate), up)
    expert_out = g.matmul(activated, down, transpose_b=True)  # [top_k, 1, hidden]
    expert_out = g.mul(g.reshape(expert_out, (top_k, hidden)), routed_weights)
    routed = None
    for i in range(top_k):
        row = g.slice(expert_out, (i, 0), (1, hidden))
        routed = row if routed is None else g.add(routed, row)

    shared = g.mul(g.silu(g.linear(x, layer.shared_gate_proj)), g.linear(x, layer.shared_up_proj))
    shared = g.linear(shared, layer.shared_down_proj)
    shared_gate = g.cast(g.sigmoid(g.f32(g.linear(x, layer.shared_expert_gate))), work)
    return g.add(routed, g.mul(shared_gate, shared))


def build_decoder_network(network: trt.INetworkDefinition, weights: dict, config: TextConfig,
                          capacity: int, work: trt.DataType) -> Graph:
    """Define the single-token decoder in ``network``; ``capacity`` is the KV/rotary position bound."""
    g = Graph(network, work)
    hidden = config.hidden_size
    kv_width = config.num_key_value_heads * config.head_dim
    window = capacity + 1

    token = network.add_input("token_id", trt.int32, (1,))
    position = network.add_input("position_id", trt.int32, (1,))
    mask = network.add_input("attention_mask", trt.float32, (1, window))
    visual = network.add_input("visual_embedding", work, (1, hidden))
    use_visual = network.add_input("use_visual_embedding", trt.int32, (1, 1))
    offsets = network.add_input("rotary_position_offsets", trt.int32, (3,))
    conv_inputs, state_inputs, k_inputs, v_inputs = [], [], [], []
    for i in range(config.num_linear_layers):
        conv_inputs.append(network.add_input(f"conv_state_{i}", trt.float32,
                                             (config.conv_dim, config.linear_conv_kernel_dim)))
        state_inputs.append(network.add_input(
            f"ssm_state_{i}", trt.float32,
            (config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim)))
    for i in range(config.num_full_layers):
        k_inputs.append(network.add_input(f"cache_k_{i}", work, (capacity, kv_width)))
        v_inputs.append(network.add_input(f"cache_v_{i}", work, (capacity, kv_width)))

    hidden_state = g.gather(g.const(weights["embedding"]), token, 0)  # [1, hidden]
    hidden_state = network.add_select(g.cast(use_visual, trt.bool), visual, hidden_state).get_output(0)

    # Canvas M-RoPE: gather the T/H/W rows and keep each frequency from its owning axis.
    coordinates = g.add(position, offsets)  # [3]
    owner = mrope_axes(config)
    masks = (np.arange(3, dtype=np.int32)[:, None] == owner[None, :]).astype(np.float32)
    select = g.const(masks, trt.float32)
    rows = []
    for cosine in (True, False):
        table = g.const(rope_table(window, config, cosine), trt.float32)
        picked = g.mul(g.gather(table, coordinates, 0), select)  # [3, half]
        rows.append(g.cast(g.reduce(picked, trt.ReduceOperation.SUM, 0), work))  # [1, half]
    cos, sin = rows
    zero = g.const(np.zeros((1, 1), np.int32), trt.int32)
    attention_mask = g.reshape(mask, (1, window))

    outputs = []
    linear_index = full_index = 0
    for layer in weights["layers"]:
        residual = hidden_state
        normed = g.rms_norm_centered(hidden_state, layer.input_norm, config.rms_norm_eps)
        if layer.kind == "linear_attention":
            mixed, conv, state = _deltanet(g, normed, conv_inputs[linear_index], state_inputs[linear_index],
                                           layer, config)
            outputs += [(f"present_conv_{linear_index}", conv), (f"present_ssm_{linear_index}", state)]
            linear_index += 1
        else:
            mixed, key, value = _attention(g, normed, k_inputs[full_index], v_inputs[full_index],
                                           attention_mask, cos, sin, zero, layer, config)
            outputs += [(f"present_k_{full_index}", key), (f"present_v_{full_index}", value)]
            full_index += 1
        hidden_state = g.add(residual, mixed)
        residual = hidden_state
        normed = g.rms_norm_centered(hidden_state, layer.post_norm, config.rms_norm_eps)
        hidden_state = g.add(residual, _experts(g, normed, layer, config))

    hidden_state = g.rms_norm_centered(hidden_state, weights["final_norm"], config.rms_norm_eps)
    logits = g.f32(g.linear(hidden_state, weights["lm_head"]))
    logits.name = "logits"
    network.mark_output(logits)
    for name, tensor in outputs:
        tensor.name = name
        network.mark_output(tensor)
    return g

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT engine for the Qwen3-VL language model that encodes Qwen-Image 2.1 prompts.

Text-to-image prompts carry no image, so only the language model is built: token embedding and
the decoder layers. The transformer was trained on the last decoder layer's output *before* the
final RMSNorm, so the engine stops there and does not apply ``language_model.norm``.

For text-only input the three multimodal rotary position ids of a token are equal, which reduces
the interleaved MRoPE to ordinary rotate-half RoPE on the token index.

Attention is causal and prompts are padded on the right, so a valid token never attends to a
padding slot and the engine needs no attention mask. Rows at padding slots are meaningless; the
denoiser masks them out.

Engine contract:
  ``input_ids``    [prefix_tokens + text_tokens] int32   templated prompt, padded on the right
  ``text_hidden``  [text_tokens, hidden] FP32            states after the dropped system prefix
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorrt as trt

from . import rope, trt_ops
from .trt_ops import BF16, FP32, Graph
from .weights import read_json, read_safetensors, take

_OPTIMIZATION_LEVEL = 3
_WORKSPACE_BYTES = 8 << 30
_PREFIX = "model.language_model"


@dataclass(frozen=True)
class TextEncoderConfig:
    vocab_size: int
    hidden: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    rope_theta: float
    eps: float


def load_text_encoder(text_encoder_dir: Path) -> tuple[TextEncoderConfig, dict[str, np.ndarray]]:
    raw = read_json(text_encoder_dir / "config.json")
    if raw.get("model_type") != "qwen3_vl":
        raise ValueError(f"unsupported text encoder model_type {raw.get('model_type')!r}")
    text = raw["text_config"]
    if text.get("hidden_act") != "silu" or bool(text.get("attention_bias")):
        raise ValueError("unsupported Qwen3-VL text configuration")
    config = TextEncoderConfig(
        vocab_size=int(text["vocab_size"]),
        hidden=int(text["hidden_size"]),
        num_layers=int(text["num_hidden_layers"]),
        num_heads=int(text["num_attention_heads"]),
        num_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        intermediate=int(text["intermediate_size"]),
        rope_theta=float(text["rope_theta"]),
        eps=float(text["rms_norm_eps"]),
    )
    if config.num_heads * config.head_dim != config.hidden:
        raise ValueError("Qwen3-VL attention width must equal hidden_size")
    if config.num_heads % config.num_kv_heads or config.head_dim % 2:
        raise ValueError("unsupported Qwen3-VL attention head layout")
    return config, read_safetensors(text_encoder_dir, prefixes=(f"{_PREFIX}.",))


def _rotate_half(graph: Graph, tensor: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    """``x * cos + rotate_half(x) * sin`` on ``[S, heads, head_dim]`` in the compute type."""
    half = int(tensor.shape[-1]) // 2
    first = trt_ops.slice_axis(graph, tensor, 2, 0, half)
    second = trt_ops.slice_axis(graph, tensor, 2, half, half)
    low = trt_ops.sub(graph, trt_ops.mul(graph, first, cos), trt_ops.mul(graph, second, sin))
    high = trt_ops.add(graph, trt_ops.mul(graph, second, cos), trt_ops.mul(graph, first, sin))
    return trt_ops.concat(graph, [low, high], 2)


def _to_heads(graph: Graph, tensor: trt.ITensor) -> trt.ITensor:
    """``[S, heads, head_dim]`` -> ``[1, heads, S, head_dim]``."""
    sequence, heads, head_dim = trt_ops.shape_of(tensor)
    moved = trt_ops.transpose(graph, tensor, (1, 0, 2))
    return trt_ops.reshape(graph, moved, (1, heads, sequence, head_dim))


def _layer(
    graph: Graph,
    config: TextEncoderConfig,
    weights: dict,
    index: int,
    hidden_states: trt.ITensor,
    cos: trt.ITensor,
    sin: trt.ITensor,
) -> trt.ITensor:
    hidden = config.hidden
    head_dim = config.head_dim
    kv_width = config.num_kv_heads * head_dim
    sequence = int(hidden_states.shape[0])
    prefix = f"{_PREFIX}.layers.{index}"

    def weight(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return take(weights, f"{prefix}.{name}", shape)

    normed = trt_ops.rms_norm(
        graph, hidden_states, weight("input_layernorm.weight", (hidden,)), config.eps
    )
    query = trt_ops.linear(graph, normed, weight("self_attn.q_proj.weight", (hidden, hidden)))
    key = trt_ops.linear(graph, normed, weight("self_attn.k_proj.weight", (kv_width, hidden)))
    value = trt_ops.linear(graph, normed, weight("self_attn.v_proj.weight", (kv_width, hidden)))
    query = trt_ops.reshape(graph, query, (sequence, config.num_heads, head_dim))
    key = trt_ops.reshape(graph, key, (sequence, config.num_kv_heads, head_dim))
    value = trt_ops.reshape(graph, value, (sequence, config.num_kv_heads, head_dim))
    query = trt_ops.rms_norm(graph, query, weight("self_attn.q_norm.weight", (head_dim,)), config.eps)
    key = trt_ops.rms_norm(graph, key, weight("self_attn.k_norm.weight", (head_dim,)), config.eps)
    context = trt_ops.attention(
        graph,
        _to_heads(graph, _rotate_half(graph, query, cos, sin)),
        _to_heads(graph, _rotate_half(graph, key, cos, sin)),
        _to_heads(graph, value),
        causal=True,
    )
    context = trt_ops.reshape(
        graph, trt_ops.transpose(graph, context, (0, 2, 1, 3)), (sequence, hidden)
    )
    hidden_states = trt_ops.add(
        graph,
        hidden_states,
        trt_ops.linear(graph, context, weight("self_attn.o_proj.weight", (hidden, hidden))),
    )

    normed = trt_ops.rms_norm(
        graph, hidden_states, weight("post_attention_layernorm.weight", (hidden,)), config.eps
    )
    inner = config.intermediate
    gate = trt_ops.silu(
        graph, trt_ops.linear(graph, normed, weight("mlp.gate_proj.weight", (inner, hidden)))
    )
    up = trt_ops.linear(graph, normed, weight("mlp.up_proj.weight", (inner, hidden)))
    feed_forward = trt_ops.linear(
        graph, trt_ops.mul(graph, gate, up), weight("mlp.down_proj.weight", (hidden, inner))
    )
    return trt_ops.add(graph, hidden_states, feed_forward)


def build_text_encoder_plan(
    config: TextEncoderConfig,
    weights: dict[str, np.ndarray],
    output: BinaryIO,
    *,
    prefix_tokens: int,
    text_tokens: int,
    verbose: bool,
    compute: trt.DataType = BF16,
) -> None:
    """Build the text encoder for one static token budget and write its plan to ``output``.

    ``compute`` is BF16 for bundles; the parity tests pass FP32 to build the same graph without
    BF16 rounding.
    """
    if prefix_tokens < 0 or text_tokens < 1:
        raise ValueError("text encoder token counts must be positive")
    sequence = prefix_tokens + text_tokens
    graph = trt_ops.create_graph(
        verbose=verbose,
        workspace_bytes=_WORKSPACE_BYTES,
        optimization_level=_OPTIMIZATION_LEVEL,
        compute=compute,
    )
    network = graph.network
    input_ids = network.add_input("input_ids", trt.int32, (sequence,))

    table = trt_ops.constant_compute(
        graph, take(weights, f"{_PREFIX}.embed_tokens.weight", (config.vocab_size, config.hidden))
    )
    hidden_states = network.add_gather(table, input_ids, 0).get_output(0)

    cos_table, sin_table = rope.rotate_half_tables(sequence, config.head_dim, config.rope_theta)
    half = config.head_dim // 2
    cos = trt_ops.constant_compute(graph, cos_table.reshape(sequence, 1, half))
    sin = trt_ops.constant_compute(graph, sin_table.reshape(sequence, 1, half))
    for index in range(config.num_layers):
        hidden_states = _layer(graph, config, weights, index, hidden_states, cos, sin)

    text_hidden = trt_ops.slice_axis(graph, hidden_states, 0, prefix_tokens, text_tokens)
    text_hidden = trt_ops.cast(graph, text_hidden, FP32)
    text_hidden.name = "text_hidden"
    network.mark_output(text_hidden)

    output.write(memoryview(trt_ops.serialize(graph, "text encoder")))

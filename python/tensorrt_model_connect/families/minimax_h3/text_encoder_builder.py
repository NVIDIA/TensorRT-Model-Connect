# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native BF16 Qwen3-VL language-stack encoder for MiniMax-H3 T2VA.

MiniMax-H3 consumes ``hidden_states[50]``. The engine therefore loads and
builds exactly the embedding plus decoder layers 0..49; the LM head, final
norm, remaining language layers, and vision tower never enter the plan.
"""

from __future__ import annotations

import gc
import math
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES


trt = trt_compat.get_trt()

HIDDEN_SIZE = 5120
NUM_LAYERS = 50
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 25600
VOCAB_SIZE = 151936
ROPE_THETA = 5_000_000.0
NORM_EPS = 1.0e-6


def checkpoint_keys() -> tuple[str, ...]:
    names = ["model.language_model.embed_tokens.weight"]
    for index in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{index}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return tuple(names)


def _per_head_norm(network, tensor, weight, rows: int, heads: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (rows, heads, HEAD_DIM)
    # Preserve the already-qualified fixed-length T2VA TensorRT topology.  The
    # exact Qwen publication order is used by the dynamic multimodal builder,
    # but compounds with different fixed-plan tactics and regresses T2VA E2E.
    normalized = op.rms_norm(network, reshape.get_output(0), weight, HEAD_DIM, NORM_EPS)
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (rows, heads * HEAD_DIM)
    return flatten.get_output(0)


def _repeat_kv(network, tensor):
    repeated = []
    repeat = NUM_HEADS // NUM_KV_HEADS
    for index in range(NUM_KV_HEADS):
        head = network.add_slice(
            tensor, (0, index, 0, 0), (1, 1, int(tensor.shape[2]), HEAD_DIM), (1, 1, 1, 1)
        ).get_output(0)
        repeated.extend([head] * repeat)
    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _rope_cache(network, rows: int):
    inverse = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    frequency = np.outer(np.arange(rows, dtype=np.float32), inverse)
    cos = op.constant(network, np.cos(frequency).reshape(1, rows, HEAD_DIM // 2))
    sin = op.constant(network, np.sin(frequency).reshape(1, rows, HEAD_DIM // 2))
    return op.cast(network, cos, trt.bfloat16), op.cast(network, sin, trt.bfloat16)


def _linear(network, hidden, weights, name: str):
    return op.linear(network, hidden, weights[f"{name}.weight"])


def build_text_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    sequence_length: int,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )
    input_ids = network.add_input("input_ids", trt.int32, (sequence_length,))
    table = op.weight_constant(network, weights["model.language_model.embed_tokens.weight"])
    table = op.cast(network, table, trt.bfloat16)
    hidden = network.add_gather(table, input_ids, 0).get_output(0)
    cos, sin = _rope_cache(network, sequence_length)

    # All fixed-path RMSNorm sites retain the qualified T2VA lowering.  The
    # dynamic conditioned language graph uses the exact Qwen early-publication
    # helper because that topology has different recurrent numerical behavior.
    for index in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{index}"
        normalized = op.rms_norm(
            network, hidden, weights[f"{prefix}.input_layernorm.weight"], HIDDEN_SIZE, NORM_EPS
        )
        q = _linear(network, normalized, weights, f"{prefix}.self_attn.q_proj")
        k = _linear(network, normalized, weights, f"{prefix}.self_attn.k_proj")
        v = _linear(network, normalized, weights, f"{prefix}.self_attn.v_proj")
        q = _per_head_norm(
            network, q, weights[f"{prefix}.self_attn.q_norm.weight"], sequence_length, NUM_HEADS
        )
        k = _per_head_norm(
            network, k, weights[f"{prefix}.self_attn.k_norm.weight"], sequence_length, NUM_KV_HEADS
        )
        q = op.partial_rope(
            network,
            q,
            cos,
            sin,
            rows=sequence_length,
            heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=HEAD_DIM,
        )
        k = op.partial_rope(
            network,
            k,
            cos,
            sin,
            rows=sequence_length,
            heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=HEAD_DIM,
        )
        q4 = op.rows_to_heads(network, q, sequence_length, NUM_HEADS, HEAD_DIM)
        k4 = op.rows_to_heads(network, k, sequence_length, NUM_KV_HEADS, HEAD_DIM)
        v4 = op.rows_to_heads(network, v, sequence_length, NUM_KV_HEADS, HEAD_DIM)
        k4 = _repeat_kv(network, k4)
        v4 = _repeat_kv(network, v4)
        scale = op.constant(
            network,
            np.full((1, 1, 1, 1), 1.0 / math.sqrt(HEAD_DIM), dtype=np.float32),
        )
        scale = op.cast(network, scale, q4.dtype)
        q4 = network.add_elementwise(q4, scale, trt.ElementWiseOperation.PROD).get_output(0)
        attention = network.add_attention(q4, k4, v4, trt.AttentionNormalizationOp.SOFTMAX, True)
        if attention is None:
            raise RuntimeError(f"TensorRT failed to add Qwen3-VL attention layer {index}")
        attention.name = f"{prefix}.self_attn.native_attention"
        attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
        attention.get_output(0).name = f"{attention.name}.output"
        attention.decomposable = False
        update = op.heads_to_rows(
            network, attention.get_output(0), sequence_length, NUM_HEADS * HEAD_DIM
        )
        update = _linear(network, update, weights, f"{prefix}.self_attn.o_proj")
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.post_attention_layernorm.weight"],
            HIDDEN_SIZE,
            NORM_EPS,
        )
        gate = _linear(network, normalized, weights, f"{prefix}.mlp.gate_proj")
        up = _linear(network, normalized, weights, f"{prefix}.mlp.up_proj")
        gate = op.silu(network, gate)
        gated = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(0)
        update = _linear(network, gated, weights, f"{prefix}.mlp.down_proj")
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

    output = op.cast(network, hidden, trt.float32)
    output.name = "encoder_hidden_states"
    network.mark_output(output)
    op.validate_native_network(network, expected_attentions=NUM_LAYERS, label="text encoder")
    print(
        f"[minimax-h3] building native Qwen3-VL text stack: layers={NUM_LAYERS}, "
        f"sequence={sequence_length}",
        file=sys.stderr,
    )
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 text encoder")
    del network, config, builder
    gc.collect()
    return bytes(plan)

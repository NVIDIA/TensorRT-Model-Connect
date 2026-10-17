# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native BF16 Qwen3-VL language-stack conditioner for MiniMax-H3.

MiniMax-H3 consumes ``hidden_states[50]``. The engine therefore loads and
builds exactly the embedding plus decoder layers 0..49; the LM head, final
norm, remaining language layers, and vision tower never enter the plan.
Reference graphs accept separate visual features/DeepStack inputs and 3-axis
interleaved MRoPE positions; the vision tower remains a separate owned plan.
"""

from __future__ import annotations

import gc
import math
import sys

import numpy as np

import tensorrt as trt

from . import graph_ops as op
from .config import TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES


HIDDEN_SIZE = 5120
NUM_LAYERS = 50
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 25600
VOCAB_SIZE = 151936
ROPE_THETA = 5_000_000.0
NORM_EPS = 1.0e-6
MROPE_SECTION = (24, 20, 20)


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


def _qwen_rms_norm(network, value, weight, width):
    # Qwen rounds the normalized activation to its source dtype BEFORE the
    # learned affine multiply, unlike H3 DiT's FP32-affine RMSNorm boundary.
    normalized = op.rms_norm(network, value, np.ones(width, np.float32), width, NORM_EPS)
    shape = (1,) * (len(tuple(value.shape)) - 1) + (width,)
    gamma = op.cast(network, op.weight_constant(network, weight.reshape(shape)), value.dtype)
    return network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)


def _per_head_norm(network, tensor, weight, rows: int, heads: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, HEAD_DIM)
    normalized = _qwen_rms_norm(network, reshape.get_output(0), weight, HEAD_DIM)
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (-1, heads * HEAD_DIM)
    return flatten.get_output(0)


def _repeat_kv(network, tensor):
    repeated = []
    repeat = NUM_HEADS // NUM_KV_HEADS
    for index in range(NUM_KV_HEADS):
        head = op.dynamic_slice(network, tensor, (0, index, 0, 0), (1, 1, None, HEAD_DIM))
        repeated.extend([head] * repeat)
    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def mrope_axes():
    half = HEAD_DIM // 2
    if len(MROPE_SECTION) != 3 or sum(MROPE_SECTION) != half:
        raise ValueError("H3 Qwen MRoPE sections must partition half the attention head")
    axes = np.zeros(half, dtype=np.int32)
    for dim, offset in ((1, 1), (2, 2)):
        axes[offset : MROPE_SECTION[dim] * 3 : 3] = dim
    return axes


def _rope_cache(network, position_ids, *, reference=False):
    inverse = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    positions = op.cast(network, position_ids, trt.float32)
    if reference:
        # Select each rotary frequency's T/H/W coordinate using Qwen3's
        # interleaved layout, not H3 DiT's distinct 96-channel MM-RoPE.
        axes = op.constant(network, mrope_axes(), dtype=np.int32)
        selected = network.add_gather(positions, axes, 0).get_output(0)
        position_shape = network.add_shuffle(selected)
        position_shape.first_transpose = trt.Permutation((1, 0))
        position_shape.reshape_dims = (1, -1, HEAD_DIM // 2)
    else:
        position_shape = network.add_shuffle(positions)
        position_shape.reshape_dims = (1, -1, 1)
    inverse = op.constant(network, inverse.reshape(1, 1, HEAD_DIM // 2))
    frequency = network.add_elementwise(
        position_shape.get_output(0), inverse, trt.ElementWiseOperation.PROD
    ).get_output(0)
    cos = network.add_unary(frequency, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(frequency, trt.UnaryOperation.SIN).get_output(0)
    return op.cast(network, cos, trt.bfloat16), op.cast(network, sin, trt.bfloat16)


def _linear(network, hidden, weights, name: str):
    return op.linear(network, hidden, weights[f"{name}.weight"])


def _qwen_rope(network, tensor, cos_half, sin_half, *, rows, heads):
    # Qwen's language RoPE evaluates two BF16 products and then a BF16 sum.
    # Preserve those explicit rounding boundaries: a collapsed higher-precision
    # multiply/add is NOT the same as the released three-operation expression.
    # The H3 DiT's partial RoPE helper remains unchanged.
    value = op.rows_to_heads(network, tensor, rows, heads, HEAD_DIM)
    half = HEAD_DIM // 2
    first = op.dynamic_slice(network, value, (0, 0, 0, 0), (1, heads, None, half))
    second = op.dynamic_slice(network, value, (0, 0, 0, half), (1, heads, None, half))
    negative = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    rotate = network.add_concatenation((negative, first))
    rotate.axis = 3

    def table_half(table):
        shaped = network.add_shuffle(table)
        shaped.reshape_dims = (1, 1, -1, half)
        duplicate = network.add_concatenation((shaped.get_output(0), shaped.get_output(0)))
        duplicate.axis = 3
        return op.cast(network, duplicate.get_output(0), trt.float32)

    left = network.add_elementwise(
        op.cast(network, value, trt.float32), table_half(cos_half), trt.ElementWiseOperation.PROD
    ).get_output(0)
    right = network.add_elementwise(
        op.cast(network, rotate.get_output(0), trt.float32),
        table_half(sin_half),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    left = op.cast(network, op.cast(network, left, trt.bfloat16), trt.float32)
    right = op.cast(network, op.cast(network, right, trt.bfloat16), trt.float32)
    summed = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)
    return op.heads_to_rows(network, op.cast(network, summed, trt.bfloat16), rows, heads * HEAD_DIM)


def conditioner_input_shapes(sequence_length, *, reference=False):
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise ValueError("H3 conditioner sequence length must be a positive integer")
    if not isinstance(reference, bool):
        raise ValueError("H3 conditioner reference switch must be boolean")
    bounds = (1, min(sequence_length, 128), sequence_length)
    shapes = {
        "input_ids": [(rows,) for rows in bounds],
        "position_ids": [(3, rows) if reference else (rows,) for rows in bounds],
    }
    if reference:
        shapes["visual_mask"] = [(rows, 1) for rows in bounds]
        for name in ("visual_features", *(f"deepstack_features_{i}" for i in range(3))):
            shapes[name] = [(rows, HIDDEN_SIZE) for rows in bounds]
    return shapes


def build_text_encoder_engine(
    weights: dict[str, np.ndarray],
    *,
    sequence_length: int,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    reference: bool = False,
) -> bytes:
    input_shapes = conditioner_input_shapes(sequence_length, reference=reference)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    op.configure_builder(config)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )
    input_ids = network.add_input("input_ids", trt.int32, (-1,))
    position_ids = network.add_input("position_ids", trt.int32, (3, -1) if reference else (-1,))
    visual_inputs = {}
    if reference:
        for name in input_shapes:
            if name in ("input_ids", "position_ids"):
                continue
            visual_inputs[name] = network.add_input(
                name,
                trt.int32 if name == "visual_mask" else trt.bfloat16,
                (-1, 1 if name == "visual_mask" else HIDDEN_SIZE),
            )
        # The native TensorMap ABI has int32 but no bool storage type. Keep
        # this conversion inside the family graph rather than changing core.
        one = op.constant(network, np.ones((1, 1), np.int32), dtype=np.int32)
        visual_mask = network.add_elementwise(
            visual_inputs["visual_mask"], one, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
    profile = builder.create_optimization_profile()
    opt_sequence_length = min(sequence_length, 128)
    for name, shapes in input_shapes.items():
        profile.set_shape(
            name,
            min=shapes[0],
            opt=shapes[1],
            max=shapes[2],
        )
    config.add_optimization_profile(profile)
    table = op.weight_constant(network, weights["model.language_model.embed_tokens.weight"])
    table = op.cast(network, table, trt.bfloat16)
    hidden = network.add_gather(table, input_ids, 0).get_output(0)
    if reference:
        hidden = network.add_select(
            visual_mask, visual_inputs["visual_features"], hidden
        ).get_output(0)
    cos, sin = _rope_cache(network, position_ids, reference=reference)

    for index in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{index}"
        normalized = _qwen_rms_norm(
            network, hidden, weights[f"{prefix}.input_layernorm.weight"], HIDDEN_SIZE
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
        q = _qwen_rope(
            network,
            q,
            cos,
            sin,
            rows=sequence_length,
            heads=NUM_HEADS,
        )
        k = _qwen_rope(
            network,
            k,
            cos,
            sin,
            rows=sequence_length,
            heads=NUM_KV_HEADS,
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

        normalized = _qwen_rms_norm(
            network,
            hidden,
            weights[f"{prefix}.post_attention_layernorm.weight"],
            HIDDEN_SIZE,
        )
        gate = _linear(network, normalized, weights, f"{prefix}.mlp.gate_proj")
        up = _linear(network, normalized, weights, f"{prefix}.mlp.up_proj")
        gate = op.silu(network, gate)
        gated = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(0)
        update = _linear(network, gated, weights, f"{prefix}.mlp.down_proj")
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)
        if reference and index < 3:
            # HF injects the three vision branches AFTER language layers 0/1/2.
            # Only image/video pad rows are visual: vision_start/end tokens are
            # not placeholders, even though H3 tags those tokens as video.
            zero = op.cast(
                network, op.constant(network, np.zeros((1, 1), np.float32)), trt.bfloat16
            )
            features = network.add_select(
                visual_mask, visual_inputs[f"deepstack_features_{index}"], zero
            ).get_output(0)
            hidden = network.add_elementwise(
                hidden, features, trt.ElementWiseOperation.SUM
            ).get_output(0)

    output = op.cast(network, hidden, trt.float32)
    output.name = "encoder_hidden_states"
    network.mark_output(output)
    op.validate_native_network(network, expected_attentions=NUM_LAYERS, label="text encoder")
    print(
        f"[minimax-h3] building native Qwen3-VL text stack: layers={NUM_LAYERS}, "
        f"sequence=1..{sequence_length} (opt={opt_sequence_length})",
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

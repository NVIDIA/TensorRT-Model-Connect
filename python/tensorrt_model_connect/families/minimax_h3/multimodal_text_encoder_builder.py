# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared native Qwen3-VL language stack for MiniMax-H3 public workflows.

The plan contains one copy of the Qwen embedding and decoder layers 0..49.
T2VA binds one disabled dummy visual row; FL2VA and Ref2VA bind compact vision
features plus their presentation-row indices.  The engine scatters the main
features before layer zero, injects DeepStack features after layers 0..2, and
emits ``hidden_states[50]`` before the full model's final normalization.
"""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES
from .fl2va_contract import MultimodalTextProfile, QWEN_TEXT_HIDDEN_SIZE, text_encoder_abi


trt = trt_compat.get_trt()

HIDDEN_SIZE = QWEN_TEXT_HIDDEN_SIZE
NUM_LAYERS = 50
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
NORM_EPS = 1.0e-6
# Bit-exact output of Qwen3VLTextRotaryEmbedding's Torch FP32 construction.
# Recomputing ``theta**exponent`` with NumPy differs by one ULP at index 19.
ROPE_INV_FREQ_BITS = (
    0x3F800000,
    0x3F492C28,
    0x3F1E165D,
    0x3EF875A7,
    0x3EC33F3A,
    0x3E996E52,
    0x3E712429,
    0x3E3D7EFC,
    0x3E14E963,
    0x3DEA09DB,
    0x3DB7EA1A,
    0x3D908687,
    0x3D63251B,
    0x3D327F50,
    0x3D0C44BF,
    0x3CDC7456,
    0x3CAD3D5E,
    0x3C88230F,
    0x3C55F605,
    0x3C282311,
    0x3C042088,
    0x3BCFA8A9,
    0x3BA32F3E,
    0x3B803C3D,
    0x3B498AD3,
    0x3B1E60C3,
    0x3AF8EA93,
    0x3AC39B1C,
    0x3A99B686,
    0x3A7195A5,
    0x3A3DD829,
    0x3A152F76,
    0x39EA77FF,
    0x39B840A8,
    0x3990CA8B,
    0x39639000,
    0x3932D34F,
    0x390C86C1,
    0x38DCDC14,
    0x38AD8EE4,
    0x38886320,
    0x38565AB5,
    0x38287230,
    0x38045EB6,
    0x37D00A62,
    0x37A37C09,
    0x37807896,
    0x3749E9AB,
    0x371EAB4B,
    0x36F95FB6,
    0x36C3F72A,
    0x3699FEDC,
    0x36720755,
    0x363E3180,
    0x361575AB,
    0x35EAE655,
    0x35B8975D,
    0x35910EAE,
    0x3563FB18,
    0x35332777,
    0x350CC8E3,
    0x34DD4405,
    0x34ADE091,
    0x3488A34F,
)


def checkpoint_keys() -> tuple[str, ...]:
    """Return the exhaustive shared language-weight partition."""

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


def _network_shape(binding) -> tuple[int, ...]:
    return tuple(
        minimum if minimum == maximum else -1
        for minimum, maximum in zip(binding.min_shape, binding.max_shape)
    )


def _per_head_norm(network, tensor, weight, heads: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, heads, HEAD_DIM)
    normalized = op.rms_norm(network, reshape.get_output(0), weight, HEAD_DIM, NORM_EPS)
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (-1, heads * HEAD_DIM)
    return flatten.get_output(0)


def _repeat_kv(network, tensor):
    repeated = []
    repeat = NUM_HEADS // NUM_KV_HEADS
    for index in range(NUM_KV_HEADS):
        head = op.dynamic_slice(network, tensor, (0, index, 0, 0), (1, 1, None, HEAD_DIM))
        repeated.extend([head] * repeat)
    concatenation = network.add_concatenation(repeated)
    concatenation.axis = 1
    return concatenation.get_output(0)


def _mrope_cache(network, position_ids):
    """Build Qwen3-VL interleaved ``[T,H,W]`` MRoPE in FP32."""

    positions = op.cast(network, position_ids, trt.float32)
    inverse = np.asarray(ROPE_INV_FREQ_BITS, dtype=np.uint32).view(np.float32)
    inverse = op.constant(network, inverse.reshape(1, HEAD_DIM // 2))
    frequencies = []
    for axis in range(3):
        coordinate = op.dynamic_slice(network, positions, (axis, 0), (1, None))
        reshape = network.add_shuffle(coordinate)
        reshape.reshape_dims = (-1, 1)
        frequencies.append(
            network.add_elementwise(
                reshape.get_output(0), inverse, trt.ElementWiseOperation.PROD
            ).get_output(0)
        )

    height_mask = np.zeros((1, HEAD_DIM // 2), dtype=np.float32)
    width_mask = np.zeros_like(height_mask)
    height_mask[:, 1:60:3] = 1.0
    width_mask[:, 2:60:3] = 1.0
    temporal_mask = 1.0 - height_mask - width_mask
    selected = []
    for frequency, mask in zip(frequencies, (temporal_mask, height_mask, width_mask)):
        selected.append(
            network.add_elementwise(
                frequency,
                op.constant(network, mask),
                trt.ElementWiseOperation.PROD,
            ).get_output(0)
        )
    frequency = network.add_elementwise(
        selected[0], selected[1], trt.ElementWiseOperation.SUM
    ).get_output(0)
    frequency = network.add_elementwise(
        frequency, selected[2], trt.ElementWiseOperation.SUM
    ).get_output(0)
    cos = network.add_unary(frequency, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(frequency, trt.UnaryOperation.SIN).get_output(0)
    return op.cast(network, cos, trt.bfloat16), op.cast(network, sin, trt.bfloat16)


def _visual_count_active(network, vision_count):
    zero = op.constant(network, np.zeros((1,), dtype=np.int32), dtype=np.int32)
    active = network.add_elementwise(
        vision_count, zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    reshape = network.add_shuffle(active)
    reshape.reshape_dims = (1, 1)
    return reshape.get_output(0)


def _scatter_visual_rows(network, base_like, row_indices, compact, active):
    """Scatter compact valid features into a dynamic sequence-sized zero base."""

    zero = op.constant(network, np.zeros((1, 1), dtype=np.float32))
    zero = op.cast(network, zero, base_like.dtype)
    base = network.add_elementwise(base_like, zero, trt.ElementWiseOperation.PROD).get_output(0)
    compact = op.cast(network, compact, base_like.dtype)
    compact = network.add_select(active, compact, zero).get_output(0)
    indices = network.add_shuffle(row_indices)
    indices.reshape_dims = (-1, 1)
    scatter = network.add_scatter(
        base,
        indices.get_output(0),
        compact,
        trt.ScatterMode.ND,
    )
    if scatter is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 compact visual-row scatter")
    return scatter.get_output(0)


def _linear(network, hidden, weights, name: str):
    return op.linear(network, hidden, weights[f"{name}.weight"])


@op.cleanup_failed_build
def build_multimodal_text_encoder_engine(
    weights: dict[str, np.ndarray],
    profile: MultimodalTextProfile = MultimodalTextProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the unified T2VA/FL2VA/Ref2VA Qwen language plan."""

    profile.validate()
    expected_keys = set(checkpoint_keys())
    missing = sorted(expected_keys - set(weights))
    unexpected = sorted(set(weights) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 multimodal text checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )

    abi = text_encoder_abi(profile)
    dtype_map = {"float32": trt.float32, "int32": trt.int32}
    inputs = {
        binding.name: network.add_input(
            binding.name, dtype_map[binding.dtype], _network_shape(binding)
        )
        for binding in abi.inputs
    }
    optimization = builder.create_optimization_profile()
    for binding in abi.inputs:
        if binding.min_shape != binding.max_shape:
            optimization.set_shape(
                binding.name, binding.min_shape, binding.opt_shape, binding.max_shape
            )
    config.add_optimization_profile(optimization)

    table = op.weight_constant(network, weights["model.language_model.embed_tokens.weight"])
    table = op.cast(network, table, trt.bfloat16)
    hidden = network.add_gather(table, inputs["input_ids"], 0).get_output(0)

    active = _visual_count_active(network, inputs["vision_count"])
    visual = _scatter_visual_rows(
        network,
        hidden,
        inputs["vision_row_indices"],
        inputs["vision_embeds"],
        active,
    )
    zero_mask = op.constant(network, np.zeros((1, 1), dtype=np.float32))
    visual_mask = network.add_elementwise(
        inputs["vision_mask"], zero_mask, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    hidden = network.add_select(visual_mask, visual, hidden).get_output(0)
    cos, sin = _mrope_cache(network, inputs["mrope_position_ids"])

    for index in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{index}"
        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.input_layernorm.weight"],
            HIDDEN_SIZE,
            NORM_EPS,
        )
        query = _linear(network, normalized, weights, f"{prefix}.self_attn.q_proj")
        key = _linear(network, normalized, weights, f"{prefix}.self_attn.k_proj")
        value = _linear(network, normalized, weights, f"{prefix}.self_attn.v_proj")
        query = _per_head_norm(
            network, query, weights[f"{prefix}.self_attn.q_norm.weight"], NUM_HEADS
        )
        key = _per_head_norm(
            network, key, weights[f"{prefix}.self_attn.k_norm.weight"], NUM_KV_HEADS
        )
        query = op.partial_rope(
            network,
            query,
            cos,
            sin,
            heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=HEAD_DIM,
        )
        key = op.partial_rope(
            network,
            key,
            cos,
            sin,
            heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            rotary_dim=HEAD_DIM,
        )
        query_heads = op.rows_to_heads(network, query, NUM_HEADS, HEAD_DIM)
        key_heads = op.rows_to_heads(network, key, NUM_KV_HEADS, HEAD_DIM)
        value_heads = op.rows_to_heads(network, value, NUM_KV_HEADS, HEAD_DIM)
        key_heads = _repeat_kv(network, key_heads)
        value_heads = _repeat_kv(network, value_heads)
        scale = op.constant(
            network,
            np.full((1, 1, 1, 1), 1.0 / math.sqrt(HEAD_DIM), dtype=np.float32),
        )
        scale = op.cast(network, scale, query_heads.dtype)
        query_heads = network.add_elementwise(
            query_heads, scale, trt.ElementWiseOperation.PROD
        ).get_output(0)
        attention = network.add_attention(
            query_heads,
            key_heads,
            value_heads,
            trt.AttentionNormalizationOp.SOFTMAX,
            True,
        )
        if attention is None:
            raise RuntimeError(f"TensorRT rejected MiniMax-H3 Qwen attention layer {index}")
        attention.name = f"{prefix}.self_attn.native_attention"
        attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
        attention.get_output(0).name = f"{attention.name}.output"
        attention.decomposable = False
        update = op.heads_to_rows(
            network,
            attention.get_output(0),
            NUM_HEADS * HEAD_DIM,
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

        if index < 3:
            deepstack = _scatter_visual_rows(
                network,
                hidden,
                inputs["vision_row_indices"],
                inputs[f"deepstack_{index}"],
                active,
            )
            hidden = network.add_elementwise(
                hidden, deepstack, trt.ElementWiseOperation.SUM
            ).get_output(0)

    output = op.cast(network, hidden, trt.float32)
    output.name = abi.outputs[0].name
    network.mark_output(output)
    op.validate_native_network(
        network, expected_attentions=NUM_LAYERS, label="multimodal text encoder"
    )
    print(
        "[minimax-h3] building shared Qwen3-VL language stack: "
        f"layers={NUM_LAYERS}, sequence={profile.min_sequence_length}.."
        f"{profile.max_sequence_length}, compact_vision={profile.min_vision_rows}.."
        f"{profile.max_vision_rows}",
        file=sys.stderr,
    )
    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 multimodal text encoder")
    del network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)

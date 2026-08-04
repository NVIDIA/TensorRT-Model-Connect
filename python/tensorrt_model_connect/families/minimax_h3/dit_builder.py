# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native single-device TensorRT Omni-DiT for MiniMax-H3."""

from __future__ import annotations

import math
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import MiniMaxH3Config


trt = trt_compat.get_trt()


def _slice_modulation(network, selected, index: int, rows: int, width: int):
    value = network.add_slice(selected, (0, index, 0), (rows, 1, width), (1, 1, 1)).get_output(0)
    reshape = network.add_shuffle(value)
    reshape.reshape_dims = (rows, width)
    return reshape.get_output(0)


def _per_head_norm(network, tensor, weight, profile: MiniMaxH3Config, rows: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (rows, profile.num_heads, profile.head_dim)
    normalized = op.rms_norm(
        network, reshape.get_output(0), weight, profile.head_dim, profile.norm_eps
    )
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (rows, profile.attention_size)
    return flatten.get_output(0)


def _rope_tables(network, position_ids, profile: MiniMaxH3Config, rows: int):
    positions = op.cast(network, position_ids, trt.float32)
    position_shape = network.add_shuffle(positions)
    position_shape.reshape_dims = (rows, 3, 1)
    inverse = 1.0 / (
        10000.0
        ** (
            np.arange(0, 2 * profile.rope_freq_dim, 2, dtype=np.float32)
            / (2 * profile.rope_freq_dim)
        )
    )
    inverse = op.constant(network, inverse.reshape(1, 1, profile.rope_freq_dim))
    frequency = network.add_elementwise(
        position_shape.get_output(0), inverse, trt.ElementWiseOperation.PROD
    ).get_output(0)
    flatten = network.add_shuffle(frequency)
    flatten.reshape_dims = (1, rows, 3 * profile.rope_freq_dim)
    cos = network.add_unary(flatten.get_output(0), trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(flatten.get_output(0), trt.UnaryOperation.SIN).get_output(0)
    return cos, sin


def _native_attention(network, q, k, v, *, rows: int, profile: MiniMaxH3Config):
    q4 = op.rows_to_heads(network, q, rows, profile.num_heads, profile.head_dim)
    k4 = op.rows_to_heads(network, k, rows, profile.num_heads, profile.head_dim)
    v4 = op.rows_to_heads(network, v, rows, profile.num_heads, profile.head_dim)
    scale = op.constant(
        network,
        np.full((1, 1, 1, 1), 1.0 / math.sqrt(profile.head_dim), dtype=np.float32),
    )
    scale = op.cast(network, scale, q4.dtype)
    q4 = network.add_elementwise(q4, scale, trt.ElementWiseOperation.PROD).get_output(0)
    attention = network.add_attention(q4, k4, v4, trt.AttentionNormalizationOp.SOFTMAX, False)
    if attention is None:
        raise RuntimeError("TensorRT failed to add MiniMax-H3 token-refiner attention")
    attention.decomposable = False
    return op.heads_to_rows(network, attention.get_output(0), rows, profile.attention_size)


def _attention_block(
    network,
    hidden,
    weights,
    prefix: str,
    profile: MiniMaxH3Config,
    rows: int,
    *,
    cos=None,
    sin=None,
):
    q, k, v = op.fused_qkv(network, hidden, weights, f"{prefix}.attn")
    q = _per_head_norm(network, q, weights[f"{prefix}.attn.norm_q.weight"], profile, rows)
    k = _per_head_norm(network, k, weights[f"{prefix}.attn.norm_k.weight"], profile, rows)
    if cos is not None:
        rotary_dim = 6 * profile.rope_freq_dim
        q = op.partial_rope(
            network,
            q,
            cos,
            sin,
            rows=rows,
            heads=profile.num_heads,
            head_dim=profile.head_dim,
            rotary_dim=rotary_dim,
        )
        k = op.partial_rope(
            network,
            k,
            cos,
            sin,
            rows=rows,
            heads=profile.num_heads,
            head_dim=profile.head_dim,
            rotary_dim=rotary_dim,
        )
        attended = op.native_attention(
            network,
            q,
            k,
            v,
            rows=rows,
            heads=profile.num_heads,
            head_dim=profile.head_dim,
        )
    else:
        attended = _native_attention(network, q, k, v, rows=rows, profile=profile)
    return op.linear(network, attended, weights[f"{prefix}.attn.to_out.0.weight"])


def _refine_text(network, text, weights, profile: MiniMaxH3Config):
    hidden = op.linear(
        network, text, weights["context_embedder.weight"], weights["context_embedder.bias"]
    )
    rows = profile.text_rows
    for index in range(profile.num_refiner_layers):
        prefix = f"token_refiner.refiner_blocks.{index}"
        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.norm1.weight"],
            profile.hidden_size,
            profile.norm_eps,
        )
        update = _attention_block(network, normalized, weights, prefix, profile, rows)
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)
        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.norm2.weight"],
            profile.hidden_size,
            profile.norm_eps,
        )
        update = op.swiglu(
            network,
            normalized,
            weights[f"{prefix}.ff.net.0.proj.weight"],
            weights[f"{prefix}.ff.net.2.weight"],
            profile.ffn_dim,
        )
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)
    return op.rms_norm(
        network,
        hidden,
        weights["token_refiner.final_norm.weight"],
        profile.hidden_size,
        profile.norm_eps,
    )


def build_dit_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
) -> bytes:
    """Build the full-sequence single-device H3 TensorRT plan."""

    profile.validate()
    rows = profile.sequence_length
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 96 << 30)
    # Hugging Face keeps TF32 disabled for the FP32 input/output projections.
    # Match that contract while retaining native TensorRT GEMMs.
    config.clear_flag(trt.BuilderFlag.TF32)

    video = network.add_input(
        "video_hidden_states", trt.float32, (profile.video_rows, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (profile.audio_rows, profile.audio_in_channels)
    )
    text = network.add_input(
        "encoder_hidden_states", trt.float32, (profile.text_rows, profile.text_dim)
    )
    positions = network.add_input("position_ids", trt.float32, (rows, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (rows,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (rows,))
    block_modulations = [
        network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(profile.num_layers)
    ]
    final_modulation = network.add_input(
        "final_modulation", trt.bfloat16, (profile.max_timestep_count, 2, profile.hidden_size)
    )
    # The public single-device FL2VA profile is packed as text | audio | video.
    # Projection, text refinement, packing, and full-sequence attention all
    # remain native TensorRT operations on one device.
    text_hidden = _refine_text(network, text, weights, profile)
    audio_hidden = op.linear(
        network, audio, weights["audio_proj_in.weight"], weights["audio_proj_in.bias"], bf16=False
    )
    audio_hidden = op.cast(network, audio_hidden, trt.bfloat16)
    video_hidden = op.linear(
        network, video, weights["proj_in.weight"], weights["proj_in.bias"], bf16=False
    )
    video_hidden = op.cast(network, video_hidden, trt.bfloat16)
    packed = network.add_concatenation((text_hidden, audio_hidden, video_hidden))
    packed.axis = 0
    hidden = packed.get_output(0)

    cos, sin = _rope_tables(network, positions, profile, rows)
    # Pristine Diffusers packs exactly 38,247 rows on one device, so its
    # attention mask is None. Preserve that contract without synthetic padding.
    for index in range(profile.num_layers):
        prefix = f"transformer_blocks.{index}"
        selected = op.gather_rows(network, block_modulations[index], adaln_indices)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            _slice_modulation(network, selected, part, rows, profile.hidden_size)
            for part in range(6)
        )
        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.norm1.weight"],
            profile.hidden_size,
            profile.norm_eps,
        )
        normalized = op.modulate(network, normalized, shift_msa, scale_msa)
        update = _attention_block(
            network,
            normalized,
            weights,
            prefix,
            profile,
            rows,
            cos=cos,
            sin=sin,
        )
        hidden = op.gated_residual(network, hidden, update, gate_msa)

        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.norm2.weight"],
            profile.hidden_size,
            profile.norm_eps,
        )
        normalized = op.modulate(network, normalized, shift_mlp, scale_mlp)
        update = op.swiglu(
            network,
            normalized,
            weights[f"{prefix}.ff.net.0.proj.weight"],
            weights[f"{prefix}.ff.net.2.weight"],
            profile.ffn_dim,
        )
        hidden = op.gated_residual(network, hidden, update, gate_mlp)

    selected = op.gather_rows(network, final_modulation, timestep_indices)
    final_shift = _slice_modulation(network, selected, 0, rows, profile.hidden_size)
    final_scale = _slice_modulation(network, selected, 1, rows, profile.hidden_size)
    hidden = op.rms_norm(
        network, hidden, weights["norm_out.norm.weight"], profile.hidden_size, profile.norm_eps
    )
    hidden = op.modulate(network, hidden, final_shift, final_scale)
    hidden = op.cast(network, hidden, trt.float32)
    video_all = op.linear(
        network, hidden, weights["proj_out.weight"], weights["proj_out.bias"], bf16=False
    )
    audio_all = op.linear(
        network,
        hidden,
        weights["audio_proj_out.weight"],
        weights["audio_proj_out.bias"],
        bf16=False,
    )
    video_all.name = "video_velocity"
    audio_all.name = "audio_velocity"
    network.mark_output(video_all)
    network.mark_output(audio_all)

    print(
        f"[minimax-h3] building native DiT: layers={profile.num_layers}, "
        f"packed={profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 DiT engine")
    return bytes(plan)

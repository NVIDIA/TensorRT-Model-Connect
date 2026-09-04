# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native single-device TensorRT Omni-DiT for MiniMax-H3."""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import (
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
)


trt = trt_compat.get_trt()


def _refiner_checkpoint_keys(profile: MiniMaxH3Config) -> tuple[str, ...]:
    names: list[str] = []
    for index in range(profile.num_refiner_layers):
        prefix = f"token_refiner.refiner_blocks.{index}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                *(f"{prefix}.attn.to_{name}.weight" for name in ("q", "k", "v")),
                f"{prefix}.attn.norm_q.weight",
                f"{prefix}.attn.norm_k.weight",
                f"{prefix}.attn.to_out.0.weight",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.2.weight",
            ]
        )
    return tuple(names)


def _block_checkpoint_keys(indices, *, include_vsa_gates: bool = False) -> tuple[str, ...]:
    names: list[str] = []
    for index in indices:
        prefix = f"transformer_blocks.{index}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                *(f"{prefix}.attn.to_{name}.weight" for name in ("q", "k", "v")),
                f"{prefix}.attn.norm_q.weight",
                f"{prefix}.attn.norm_k.weight",
                f"{prefix}.attn.to_out.0.weight",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.2.weight",
            ]
        )
        if include_vsa_gates:
            names.append(f"{prefix}.attn.to_gate_compress.weight")
    return tuple(names)


def _block_pre_attention_checkpoint_keys(index: int) -> tuple[str, ...]:
    """Weights owned by the segmented VSA pre-attention plan for one block."""

    prefix = f"transformer_blocks.{index}"
    return (
        f"{prefix}.norm1.weight",
        *(f"{prefix}.attn.to_{name}.weight" for name in ("q", "k", "v")),
        f"{prefix}.attn.norm_q.weight",
        f"{prefix}.attn.norm_k.weight",
        f"{prefix}.attn.to_gate_compress.weight",
    )


def _block_post_attention_checkpoint_keys(index: int) -> tuple[str, ...]:
    """Weights owned by the segmented VSA post-attention plan for one block."""

    prefix = f"transformer_blocks.{index}"
    return (
        f"{prefix}.attn.to_out.0.weight",
        f"{prefix}.norm2.weight",
        f"{prefix}.ff.net.0.proj.weight",
        f"{prefix}.ff.net.2.weight",
    )


def vsa_entry_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> tuple[str, ...]:
    """Weights stored exactly once in the FastH3 segmented entry plan."""

    return (
        "proj_in.weight",
        "proj_in.bias",
        "audio_proj_in.weight",
        "audio_proj_in.bias",
        "context_embedder.weight",
        "context_embedder.bias",
        "token_refiner.final_norm.weight",
        *_refiner_checkpoint_keys(profile),
        *_block_pre_attention_checkpoint_keys(0),
    )


def vsa_transition_checkpoint_keys(
    index: int,
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> tuple[str, ...]:
    """Weights for post block ``index`` plus pre block ``index + 1``."""

    if not 0 <= index < profile.num_layers - 1:
        raise ValueError(
            f"MiniMax-H3 VSA transition index must be in [0, {profile.num_layers - 2}]"
        )
    return (
        *_block_post_attention_checkpoint_keys(index),
        *_block_pre_attention_checkpoint_keys(index + 1),
    )


def vsa_finish_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> tuple[str, ...]:
    """Weights for the final post-attention block and output projections."""

    return (
        *_block_post_attention_checkpoint_keys(profile.num_layers - 1),
        *finish_checkpoint_keys(profile),
    )


def vsa_segment_checkpoint_partitions(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> dict[str, tuple[str, ...]]:
    """Return the auditable, non-overlapping 51-plan FastH3 weight partition."""

    return {
        "denoiser_entry": vsa_entry_checkpoint_keys(profile),
        **{
            f"denoiser_transition_{index:02d}": vsa_transition_checkpoint_keys(
                index, profile
            )
            for index in range(profile.num_layers - 1)
        },
        "denoiser_finish": vsa_finish_checkpoint_keys(profile),
    }


def head_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
    *,
    include_vsa_gates: bool = False,
) -> tuple[str, ...]:
    """Weights used by packing, token refinement, and transformer block zero."""

    return (
        "proj_in.weight",
        "proj_in.bias",
        "audio_proj_in.weight",
        "audio_proj_in.bias",
        "context_embedder.weight",
        "context_embedder.bias",
        "token_refiner.final_norm.weight",
        *_refiner_checkpoint_keys(profile),
        *_block_checkpoint_keys(range(1), include_vsa_gates=include_vsa_gates),
    )


def tail_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
    *,
    include_vsa_gates: bool = False,
) -> tuple[str, ...]:
    """Weights used by transformer blocks one through the final block."""

    return _block_checkpoint_keys(
        range(1, profile.num_layers), include_vsa_gates=include_vsa_gates
    )


def finish_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> tuple[str, ...]:
    """Weights used by the final norm and modality-specific projections."""

    return (
        "norm_out.norm.weight",
        "proj_out.weight",
        "proj_out.bias",
        "audio_proj_out.weight",
        "audio_proj_out.bias",
    )


def checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
    *,
    include_vsa_gates: bool = False,
) -> tuple[str, ...]:
    """Checkpoint tensors used by all DiT plans, excluding AdaLN."""

    return (
        *head_checkpoint_keys(profile, include_vsa_gates=include_vsa_gates),
        *tail_checkpoint_keys(profile, include_vsa_gates=include_vsa_gates),
        *finish_checkpoint_keys(profile),
    )


def _slice_modulation(network, selected, index: int, rows: int, width: int):
    value = op.dynamic_slice(network, selected, (0, index, 0), (None, 1, width))
    reshape = network.add_shuffle(value)
    reshape.reshape_dims = (-1, width)
    return reshape.get_output(0)


def _per_head_norm(network, tensor, weight, profile: MiniMaxH3Config, rows: int):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, profile.num_heads, profile.head_dim)
    normalized = op.rms_norm(
        network, reshape.get_output(0), weight, profile.head_dim, profile.norm_eps
    )
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (rows, profile.attention_size)
    return flatten.get_output(0)


def _rope_tables(network, position_ids, profile: MiniMaxH3Config, rows: int):
    positions = op.cast(network, position_ids, trt.float32)
    position_shape = network.add_shuffle(positions)
    position_shape.reshape_dims = (-1, 3, 1)
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
    flatten.reshape_dims = (1, -1, 3 * profile.rope_freq_dim)
    cos = network.add_unary(flatten.get_output(0), trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(flatten.get_output(0), trt.UnaryOperation.SIN).get_output(0)
    return cos, sin


def _native_attention(network, q, k, v, *, rows: int, profile: MiniMaxH3Config, name: str):
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
    attention.name = name
    attention.metadata = f"trtmc.native_op=IAttention;source={name}"
    attention.get_output(0).name = f"{name}.output"
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
    consume_weights: bool = False,
):
    q, k, v = op.fused_qkv(
        network,
        hidden,
        weights,
        f"{prefix}.attn",
        consume_weights=consume_weights,
    )
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
            name=f"{prefix}.attn.native_attention",
        )
    else:
        attended = _native_attention(
            network,
            q,
            k,
            v,
            rows=rows,
            profile=profile,
            name=f"{prefix}.attn.native_attention",
        )
    return op.linear(network, attended, weights[f"{prefix}.attn.to_out.0.weight"])


def _refine_text(
    network,
    text,
    weights,
    profile: MiniMaxH3Config,
    *,
    consume_weights: bool = False,
):
    hidden = op.linear(
        network, text, weights["context_embedder.weight"], weights["context_embedder.bias"]
    )
    rows = -1
    for index in range(profile.num_refiner_layers):
        prefix = f"token_refiner.refiner_blocks.{index}"
        normalized = op.rms_norm(
            network,
            hidden,
            weights[f"{prefix}.norm1.weight"],
            profile.hidden_size,
            profile.norm_eps,
        )
        update = _attention_block(
            network,
            normalized,
            weights,
            prefix,
            profile,
            rows,
            consume_weights=consume_weights,
        )
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


def _packed_hidden(
    network,
    video,
    audio,
    text,
    weights,
    profile: MiniMaxH3Config,
    *,
    consume_weights: bool = False,
):
    """Project and pack text | audio | video exactly like the Diffusers model."""

    text_hidden = _refine_text(
        network,
        text,
        weights,
        profile,
        consume_weights=consume_weights,
    )
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
    return packed.get_output(0)


def _transformer_block(
    network,
    hidden,
    block_modulation,
    adaln_indices,
    cos,
    sin,
    weights,
    profile: MiniMaxH3Config,
    index: int,
    *,
    consume_weights: bool = False,
):
    """Add one native H3 transformer block and return its residual stream."""

    rows = -1
    prefix = f"transformer_blocks.{index}"
    selected = op.gather_rows(network, block_modulation, adaln_indices)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _slice_modulation(network, selected, part, rows, profile.hidden_size) for part in range(6)
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
        consume_weights=consume_weights,
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
    return op.gated_residual(network, hidden, update, gate_mlp)


def _rows_to_head_major(network, tensor, profile: MiniMaxH3Config):
    """Convert packed ``[S, H*D]`` rows to the runtime CUDA ABI ``[H,S,D]``."""

    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, profile.num_heads, profile.head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    return reshape.get_output(0)


def _head_major_to_rows(network, tensor, profile: MiniMaxH3Config):
    """Convert the native CUDA VSA result ``[H,S,D]`` back to packed rows."""

    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([1, 0, 2])
    reshape.reshape_dims = (-1, profile.attention_size)
    return reshape.get_output(0)


def _prepare_vsa_block(
    network,
    hidden,
    block_modulation,
    adaln_indices,
    cos,
    sin,
    weights,
    profile: MiniMaxH3Config,
    index: int,
    *,
    consume_weights: bool = False,
):
    """Build one block through Q/K/V/raw-gate, stopping at native CUDA VSA."""

    rows = -1
    prefix = f"transformer_blocks.{index}"
    selected = op.gather_rows(network, block_modulation, adaln_indices)
    shift_msa = _slice_modulation(network, selected, 0, rows, profile.hidden_size)
    scale_msa = _slice_modulation(network, selected, 1, rows, profile.hidden_size)
    normalized = op.rms_norm(
        network,
        hidden,
        weights[f"{prefix}.norm1.weight"],
        profile.hidden_size,
        profile.norm_eps,
    )
    normalized = op.modulate(network, normalized, shift_msa, scale_msa)

    gate_key = f"{prefix}.attn.to_gate_compress.weight"
    if gate_key not in weights:
        raise ValueError(f"MiniMax-H3 VSA gate is missing: {gate_key}")
    gate = op.linear(network, normalized, weights[gate_key])
    q, k, v = op.fused_qkv(
        network,
        normalized,
        weights,
        f"{prefix}.attn",
        consume_weights=consume_weights,
    )
    q = _per_head_norm(network, q, weights[f"{prefix}.attn.norm_q.weight"], profile, rows)
    k = _per_head_norm(network, k, weights[f"{prefix}.attn.norm_k.weight"], profile, rows)
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
    if consume_weights:
        weights.pop(gate_key)
    return tuple(
        _rows_to_head_major(network, value, profile) for value in (q, k, v, gate)
    )


def _complete_vsa_block(
    network,
    hidden,
    attended_head_major,
    block_modulation,
    adaln_indices,
    weights,
    profile: MiniMaxH3Config,
    index: int,
):
    """Resume one transformer block after the external native CUDA VSA call."""

    rows = -1
    prefix = f"transformer_blocks.{index}"
    selected = op.gather_rows(network, block_modulation, adaln_indices)
    gate_msa = _slice_modulation(network, selected, 2, rows, profile.hidden_size)
    shift_mlp = _slice_modulation(network, selected, 3, rows, profile.hidden_size)
    scale_mlp = _slice_modulation(network, selected, 4, rows, profile.hidden_size)
    gate_mlp = _slice_modulation(network, selected, 5, rows, profile.hidden_size)

    attended = _head_major_to_rows(network, attended_head_major, profile)
    update = op.linear(network, attended, weights[f"{prefix}.attn.to_out.0.weight"])
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
    return op.gated_residual(network, hidden, update, gate_mlp)


def _mark_vsa_boundary_outputs(network, hidden, q, k, v, gate) -> None:
    """Expose one resumable attention boundary with stable binding names."""

    outputs = (
        (hidden, "next_residual_hidden"),
        (q, "vsa_query"),
        (k, "vsa_key"),
        (v, "vsa_value"),
        (gate, "vsa_gate"),
    )
    for tensor, name in outputs:
        tensor.name = name
        network.mark_output(tensor)


def _final_hidden(network, hidden, timestep_indices, final_modulation, weights, profile):
    rows = -1
    selected = op.gather_rows(network, final_modulation, timestep_indices)
    final_shift = _slice_modulation(network, selected, 0, rows, profile.hidden_size)
    final_scale = _slice_modulation(network, selected, 1, rows, profile.hidden_size)
    hidden = op.rms_norm(
        network, hidden, weights["norm_out.norm.weight"], profile.hidden_size, profile.norm_eps
    )
    hidden = op.modulate(network, hidden, final_shift, final_scale)
    return op.cast(network, hidden, trt.float32)


def _mark_full_velocity_outputs(network, hidden, weights):
    """Preserve the original monolithic plan's full packed-sequence outputs."""

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


def _mark_sliced_velocity_outputs(network, hidden, weights, video_reference, audio_reference):
    """Project only rows consumed by the audio and video scheduler updates."""

    audio_hidden = op.slice_rows_like_from_end(
        network, hidden, audio_reference, trailing_reference=video_reference
    )
    video_hidden = op.slice_rows_like_from_end(network, hidden, video_reference)
    video = op.linear(
        network,
        video_hidden,
        weights["proj_out.weight"],
        weights["proj_out.bias"],
        bf16=False,
    )
    audio = op.linear(
        network,
        audio_hidden,
        weights["audio_proj_out.weight"],
        weights["audio_proj_out.bias"],
        bf16=False,
    )
    video.name = "video_velocity"
    audio.name = "audio_velocity"
    network.mark_output(video)
    network.mark_output(audio)


def _native_builder(
    verbose: bool,
    workspace_bytes: int | None,
    *,
    weight_streaming: bool = False,
):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    # Leave TensorRT-RTX at its device-derived maximum unless the caller
    # explicitly asks for a smaller tactic workspace. Dense H3 dynamic builds
    # require the default ceiling on 128 GiB unified-memory systems.
    if workspace_bytes is not None:
        op.configure_workspace(
            config,
            workspace_bytes,
            default_bytes=DENOISER_DEFAULT_WORKSPACE_BYTES,
        )
    # Hugging Face keeps TF32 disabled for the FP32 input/output projections.
    config.clear_flag(trt.BuilderFlag.TF32)
    return logger, builder, network, config


def _add_dynamic_profile(
    builder,
    config,
    profile: MiniMaxH3Config,
    *,
    video_inputs: tuple[str, ...] = (),
    audio_inputs: tuple[str, ...] = (),
    text_inputs: tuple[str, ...] = (),
    packed_inputs: tuple[str, ...] = (),
    head_major_inputs: tuple[str, ...] = (),
) -> None:
    optimization = builder.create_optimization_profile()
    text_shapes = (
        profile.min_text_rows,
        profile.opt_text_rows,
        profile.text_rows,
    )
    packed_shapes = (
        profile.min_sequence_length,
        profile.opt_sequence_length,
        profile.sequence_length,
    )
    for name in video_inputs:
        optimization.set_shape(
            name,
            min=(profile.min_video_rows, profile.video_patch_dim),
            opt=(profile.opt_video_rows, profile.video_patch_dim),
            max=(profile.video_rows, profile.video_patch_dim),
        )
    for name in audio_inputs:
        optimization.set_shape(
            name,
            min=(profile.min_audio_rows, profile.audio_in_channels),
            opt=(profile.opt_audio_rows, profile.audio_in_channels),
            max=(profile.audio_rows, profile.audio_in_channels),
        )
    for name in text_inputs:
        optimization.set_shape(
            name,
            min=(text_shapes[0], profile.text_dim),
            opt=(text_shapes[1], profile.text_dim),
            max=(text_shapes[2], profile.text_dim),
        )
    for name in packed_inputs:
        width = 3 if name == "position_ids" else profile.hidden_size
        if name in ("adaln_indices", "timestep_indices"):
            shapes = tuple((rows,) for rows in packed_shapes)
        else:
            shapes = tuple((rows, width) for rows in packed_shapes)
        optimization.set_shape(name, min=shapes[0], opt=shapes[1], max=shapes[2])
    for name in head_major_inputs:
        optimization.set_shape(
            name,
            min=(profile.num_heads, packed_shapes[0], profile.head_dim),
            opt=(profile.num_heads, packed_shapes[1], profile.head_dim),
            max=(profile.num_heads, packed_shapes[2], profile.head_dim),
        )
    config.add_optimization_profile(optimization)


def _serialize(
    *,
    logger,
    builder,
    network,
    config,
    weights: dict,
    consume_weights: bool,
    label: str,
    output_path: str | Path | None,
) -> bytes | dict[str, int | str]:
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
        raise RuntimeError(f"TensorRT failed to build MiniMax-H3 {label} engine")
    del network, config, builder, logger
    gc.collect()
    return record if record is not None else bytes(plan)


def _require_segmented_vsa_profile(profile: MiniMaxH3Config) -> None:
    profile.validate()
    if profile.first_block_cache:
        raise ValueError("MiniMax-H3 segmented VSA is incompatible with FirstBlockCache")


@op.cleanup_failed_build
def build_dit_vsa_entry_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build packing/refinement plus block-zero's pre-attention VSA boundary."""

    _require_segmented_vsa_profile(profile)
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    video = network.add_input(
        "video_hidden_states", trt.float32, (-1, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (-1, profile.audio_in_channels)
    )
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, profile.text_dim))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    block_modulation = network.add_input(
        "current_block_modulation",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    _add_dynamic_profile(
        builder,
        config,
        profile,
        video_inputs=("video_hidden_states",),
        audio_inputs=("audio_hidden_states",),
        text_inputs=("encoder_hidden_states",),
        packed_inputs=("position_ids", "adaln_indices"),
    )
    hidden = _packed_hidden(
        network,
        video,
        audio,
        text,
        weights,
        profile,
        consume_weights=consume_weights,
    )
    cos, sin = _rope_tables(network, positions, profile, -1)
    q, k, v, gate = _prepare_vsa_block(
        network,
        hidden,
        block_modulation,
        adaln_indices,
        cos,
        sin,
        weights,
        profile,
        0,
        consume_weights=consume_weights,
    )
    _mark_vsa_boundary_outputs(network, hidden, q, k, v, gate)
    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers,
        label="DiT segmented VSA entry",
    )
    print(
        "[minimax-h3] building segmented VSA entry: "
        f"packed={profile.min_sequence_length}..{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT segmented VSA entry",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_dit_vsa_transition_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    index: int,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Complete block ``index`` then stop at block ``index + 1``'s VSA boundary."""

    _require_segmented_vsa_profile(profile)
    vsa_transition_checkpoint_keys(index, profile)
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    hidden = network.add_input("residual_hidden", trt.bfloat16, (-1, profile.hidden_size))
    attended = network.add_input(
        "vsa_attention_output",
        trt.bfloat16,
        (profile.num_heads, -1, profile.head_dim),
    )
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    current_modulation = network.add_input(
        "current_block_modulation",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    next_modulation = network.add_input(
        "next_block_modulation",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    _add_dynamic_profile(
        builder,
        config,
        profile,
        packed_inputs=("residual_hidden", "position_ids", "adaln_indices"),
        head_major_inputs=("vsa_attention_output",),
    )
    hidden = _complete_vsa_block(
        network,
        hidden,
        attended,
        current_modulation,
        adaln_indices,
        weights,
        profile,
        index,
    )
    cos, sin = _rope_tables(network, positions, profile, -1)
    q, k, v, gate = _prepare_vsa_block(
        network,
        hidden,
        next_modulation,
        adaln_indices,
        cos,
        sin,
        weights,
        profile,
        index + 1,
        consume_weights=consume_weights,
    )
    _mark_vsa_boundary_outputs(network, hidden, q, k, v, gate)
    op.validate_native_network(
        network,
        expected_attentions=0,
        label=f"DiT segmented VSA transition {index:02d}",
    )
    print(
        f"[minimax-h3] building segmented VSA transition {index:02d}: "
        f"block={index}->{index + 1}, packed={profile.min_sequence_length}.."
        f"{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label=f"DiT segmented VSA transition {index:02d}",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_dit_vsa_finish_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Complete block 49, then apply final norm and modality projections."""

    _require_segmented_vsa_profile(profile)
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    hidden = network.add_input("residual_hidden", trt.bfloat16, (-1, profile.hidden_size))
    attended = network.add_input(
        "vsa_attention_output",
        trt.bfloat16,
        (profile.num_heads, -1, profile.head_dim),
    )
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    current_modulation = network.add_input(
        "current_block_modulation",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    final_modulation = network.add_input(
        "final_modulation",
        trt.bfloat16,
        (profile.max_timestep_count, 2, profile.hidden_size),
    )
    video = network.add_input(
        "video_hidden_states", trt.float32, (-1, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (-1, profile.audio_in_channels)
    )
    _add_dynamic_profile(
        builder,
        config,
        profile,
        video_inputs=("video_hidden_states",),
        audio_inputs=("audio_hidden_states",),
        packed_inputs=("residual_hidden", "adaln_indices", "timestep_indices"),
        head_major_inputs=("vsa_attention_output",),
    )
    hidden = _complete_vsa_block(
        network,
        hidden,
        attended,
        current_modulation,
        adaln_indices,
        weights,
        profile,
        profile.num_layers - 1,
    )
    hidden = _final_hidden(
        network, hidden, timestep_indices, final_modulation, weights, profile
    )
    _mark_sliced_velocity_outputs(network, hidden, weights, video, audio)
    op.validate_native_network(
        network,
        expected_attentions=0,
        label="DiT segmented VSA finish",
    )
    print(
        "[minimax-h3] building segmented VSA finish: "
        f"block={profile.num_layers - 1}, packed={profile.min_sequence_length}.."
        f"{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT segmented VSA finish",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_dit_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build the full-sequence single-device H3 TensorRT plan."""

    profile.validate()
    if profile.first_block_cache:
        raise ValueError("MiniMax-H3 first_block_cache profile requires the split DiT builders")
    rows = -1
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )

    video = network.add_input(
        "video_hidden_states", trt.float32, (-1, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (-1, profile.audio_in_channels)
    )
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, profile.text_dim))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    _add_dynamic_profile(
        builder,
        config,
        profile,
        video_inputs=("video_hidden_states",),
        audio_inputs=("audio_hidden_states",),
        text_inputs=("encoder_hidden_states",),
        packed_inputs=("position_ids", "adaln_indices", "timestep_indices"),
    )
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
    hidden = _packed_hidden(
        network,
        video,
        audio,
        text,
        weights,
        profile,
        consume_weights=consume_weights,
    )

    cos, sin = _rope_tables(network, positions, profile, rows)
    # The dynamic packed sequence contains live rows only, like Diffusers, so
    # its attention mask remains None for every supported prompt length.
    for index in range(profile.num_layers):
        hidden = _transformer_block(
            network,
            hidden,
            block_modulations[index],
            adaln_indices,
            cos,
            sin,
            weights,
            profile,
            index,
            consume_weights=consume_weights,
        )

    hidden = _final_hidden(network, hidden, timestep_indices, final_modulation, weights, profile)
    _mark_sliced_velocity_outputs(network, hidden, weights, video, audio)

    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + profile.num_layers,
        label="DiT",
    )

    print(
        f"[minimax-h3] building native DiT: layers={profile.num_layers}, "
        f"packed={profile.min_sequence_length}..{profile.sequence_length} "
        f"(opt={profile.opt_sequence_length}), devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT",
        output_path=output_path,
    )


def _require_first_block_cache_profile(profile: MiniMaxH3Config) -> None:
    profile.validate()
    if not profile.first_block_cache:
        raise ValueError("MiniMax-H3 split DiT plans require profile.first_block_cache=True")


@op.cleanup_failed_build
def build_dit_head_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build packing, text refinement, block zero, and the native cache metric."""

    _require_first_block_cache_profile(profile)
    rows = -1
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    video = network.add_input(
        "video_hidden_states", trt.float32, (-1, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (-1, profile.audio_in_channels)
    )
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, profile.text_dim))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    block_modulation = network.add_input(
        "block_modulation_0",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    previous_head_residual = network.add_input(
        "previous_head_residual", trt.bfloat16, (-1, profile.hidden_size)
    )
    _add_dynamic_profile(
        builder,
        config,
        profile,
        video_inputs=("video_hidden_states",),
        audio_inputs=("audio_hidden_states",),
        text_inputs=("encoder_hidden_states",),
        packed_inputs=("position_ids", "adaln_indices", "previous_head_residual"),
    )

    pre_block_hidden = _packed_hidden(
        network,
        video,
        audio,
        text,
        weights,
        profile,
        consume_weights=consume_weights,
    )
    cos, sin = _rope_tables(network, positions, profile, rows)
    head_hidden = _transformer_block(
        network,
        pre_block_hidden,
        block_modulation,
        adaln_indices,
        cos,
        sin,
        weights,
        profile,
        0,
        consume_weights=consume_weights,
    )
    head_residual = network.add_elementwise(
        head_hidden, pre_block_hidden, trt.ElementWiseOperation.SUB
    ).get_output(0)

    # This is the FirstBlockCache decision used by Sol-Engine/Diffusers:
    # sum(abs(current - previous)) / max(sum(abs(previous)), eps). Keep the
    # large tensors in BF16 and expose only the one-element FP32 result.
    delta = network.add_elementwise(
        head_residual, previous_head_residual, trt.ElementWiseOperation.SUB
    ).get_output(0)
    delta_abs = network.add_unary(delta, trt.UnaryOperation.ABS).get_output(0)
    previous_abs = network.add_unary(previous_head_residual, trt.UnaryOperation.ABS).get_output(0)
    delta_abs = op.cast(network, delta_abs, trt.float32)
    previous_abs = op.cast(network, previous_abs, trt.float32)
    reduce_axes = (1 << 0) | (1 << 1)
    numerator = network.add_reduce(
        delta_abs, trt.ReduceOperation.SUM, reduce_axes, True
    ).get_output(0)
    denominator = network.add_reduce(
        previous_abs, trt.ReduceOperation.SUM, reduce_axes, True
    ).get_output(0)
    epsilon = op.constant(network, np.full((1, 1), 1.0e-8, dtype=np.float32))
    denominator = network.add_elementwise(
        denominator, epsilon, trt.ElementWiseOperation.MAX
    ).get_output(0)
    metric = network.add_elementwise(
        numerator, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    metric_shape = network.add_shuffle(metric)
    metric_shape.reshape_dims = (1,)
    cache_metric = metric_shape.get_output(0)

    head_hidden.name = "head_hidden"
    head_residual.name = "head_residual"
    cache_metric.name = "cache_metric"
    network.mark_output(head_hidden)
    network.mark_output(head_residual)
    network.mark_output(cache_metric)
    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + 1,
        label="DiT FirstBlockCache head",
    )
    print(
        f"[minimax-h3] building native DiT cache head: "
        f"packed={profile.min_sequence_length}..{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT FirstBlockCache head",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_dit_tail_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build blocks one through 49 and expose their reusable total residual."""

    _require_first_block_cache_profile(profile)
    rows = -1
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (-1, profile.hidden_size))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    _add_dynamic_profile(
        builder,
        config,
        profile,
        packed_inputs=("head_hidden", "position_ids", "adaln_indices"),
    )
    block_modulations = {
        index: network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(1, profile.num_layers)
    }
    cos, sin = _rope_tables(network, positions, profile, rows)
    hidden = head_hidden
    for index in range(1, profile.num_layers):
        hidden = _transformer_block(
            network,
            hidden,
            block_modulations[index],
            adaln_indices,
            cos,
            sin,
            weights,
            profile,
            index,
            consume_weights=consume_weights,
        )
    tail_residual = network.add_elementwise(
        hidden, head_hidden, trt.ElementWiseOperation.SUB
    ).get_output(0)
    tail_residual.name = "tail_residual"
    network.mark_output(tail_residual)
    op.validate_native_network(
        network,
        expected_attentions=profile.num_layers - 1,
        label="DiT FirstBlockCache tail",
    )
    print(
        f"[minimax-h3] building native DiT cache tail: blocks=1-{profile.num_layers - 1}, "
        f"packed={profile.min_sequence_length}..{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT FirstBlockCache tail",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_dit_finish_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Apply a selected tail residual, final norm, and consumed-row projections."""

    _require_first_block_cache_profile(profile)
    logger, builder, network, config = _native_builder(
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (-1, profile.hidden_size))
    tail_residual = network.add_input("tail_residual", trt.bfloat16, (-1, profile.hidden_size))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    video = network.add_input(
        "video_hidden_states", trt.float32, (-1, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (-1, profile.audio_in_channels)
    )
    _add_dynamic_profile(
        builder,
        config,
        profile,
        video_inputs=("video_hidden_states",),
        audio_inputs=("audio_hidden_states",),
        packed_inputs=("head_hidden", "tail_residual", "timestep_indices"),
    )
    final_modulation = network.add_input(
        "final_modulation", trt.bfloat16, (profile.max_timestep_count, 2, profile.hidden_size)
    )
    hidden = network.add_elementwise(
        head_hidden, tail_residual, trt.ElementWiseOperation.SUM
    ).get_output(0)
    hidden = _final_hidden(network, hidden, timestep_indices, final_modulation, weights, profile)
    _mark_sliced_velocity_outputs(network, hidden, weights, video, audio)
    op.validate_native_network(
        network,
        expected_attentions=0,
        label="DiT FirstBlockCache finish",
    )
    print(
        f"[minimax-h3] building native DiT cache finish: "
        f"packed={profile.min_sequence_length}..{profile.sequence_length}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="DiT FirstBlockCache finish",
        output_path=output_path,
    )

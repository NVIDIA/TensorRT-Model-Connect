# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native single-device TensorRT Omni-DiT for MiniMax-H3."""

from __future__ import annotations

import gc
import math
import sys

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


def _block_checkpoint_keys(indices) -> tuple[str, ...]:
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
    return tuple(names)


def head_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
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
        *_block_checkpoint_keys(range(1)),
    )


def tail_checkpoint_keys(
    profile: MiniMaxH3Config = SOL_ENGINE_1344X768_124F,
) -> tuple[str, ...]:
    """Weights used by transformer blocks one through the final block."""

    return _block_checkpoint_keys(range(1, profile.num_layers))


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
) -> tuple[str, ...]:
    """Checkpoint tensors used by all DiT plans, excluding AdaLN."""

    return (
        *head_checkpoint_keys(profile),
        *tail_checkpoint_keys(profile),
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


def _refine_text(network, text, weights, profile: MiniMaxH3Config):
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


def _packed_hidden(network, video, audio, text, weights, profile: MiniMaxH3Config):
    """Project and pack text | audio | video exactly like the Diffusers model."""

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


def _mark_sliced_velocity_outputs(network, hidden, weights, profile: MiniMaxH3Config):
    """Project only rows consumed by the audio and video scheduler updates."""

    audio_hidden = op.slice_rows_from_end(
        network,
        hidden,
        offset=profile.audio_rows + profile.video_rows,
        rows=profile.audio_rows,
    )
    video_hidden = op.slice_rows_from_end(
        network, hidden, offset=profile.video_rows, rows=profile.video_rows
    )
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


def _native_builder(verbose: bool, workspace_bytes: int | None):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config)
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
    text_inputs: tuple[str, ...] = (),
    packed_inputs: tuple[str, ...] = (),
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
) -> bytes:
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build MiniMax-H3 {label} engine")
    del network, config, builder, logger
    gc.collect()
    return bytes(plan)


def build_dit_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the full-sequence single-device H3 TensorRT plan."""

    profile.validate()
    if profile.first_block_cache:
        raise ValueError("MiniMax-H3 first_block_cache profile requires the split DiT builders")
    rows = -1
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)

    video = network.add_input(
        "video_hidden_states", trt.float32, (profile.video_rows, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (profile.audio_rows, profile.audio_in_channels)
    )
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, profile.text_dim))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    _add_dynamic_profile(
        builder,
        config,
        profile,
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
    hidden = _packed_hidden(network, video, audio, text, weights, profile)

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
        )

    hidden = _final_hidden(network, hidden, timestep_indices, final_modulation, weights, profile)
    _mark_sliced_velocity_outputs(network, hidden, weights, profile)

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
    )


def _require_first_block_cache_profile(profile: MiniMaxH3Config) -> None:
    profile.validate()
    if not profile.first_block_cache:
        raise ValueError("MiniMax-H3 split DiT plans require profile.first_block_cache=True")


def build_dit_head_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build packing, text refinement, block zero, and the native cache metric."""

    _require_first_block_cache_profile(profile)
    rows = -1
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    video = network.add_input(
        "video_hidden_states", trt.float32, (profile.video_rows, profile.video_patch_dim)
    )
    audio = network.add_input(
        "audio_hidden_states", trt.float32, (profile.audio_rows, profile.audio_in_channels)
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
        text_inputs=("encoder_hidden_states",),
        packed_inputs=("position_ids", "adaln_indices", "previous_head_residual"),
    )

    pre_block_hidden = _packed_hidden(network, video, audio, text, weights, profile)
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
    )


def build_dit_tail_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build blocks one through 49 and expose their reusable total residual."""

    _require_first_block_cache_profile(profile)
    rows = -1
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
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
    )


def build_dit_finish_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Apply a selected tail residual, final norm, and consumed-row projections."""

    _require_first_block_cache_profile(profile)
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (-1, profile.hidden_size))
    tail_residual = network.add_input("tail_residual", trt.bfloat16, (-1, profile.hidden_size))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    _add_dynamic_profile(
        builder,
        config,
        profile,
        packed_inputs=("head_hidden", "tail_residual", "timestep_indices"),
    )
    final_modulation = network.add_input(
        "final_modulation", trt.bfloat16, (profile.max_timestep_count, 2, profile.hidden_size)
    )
    hidden = network.add_elementwise(
        head_hidden, tail_residual, trt.ElementWiseOperation.SUM
    ).get_output(0)
    hidden = _final_hidden(network, hidden, timestep_indices, final_modulation, weights, profile)
    _mark_sliced_velocity_outputs(network, hidden, weights, profile)
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
    )

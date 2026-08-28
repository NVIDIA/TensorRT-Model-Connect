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
    FL2VA_KEYFRAME_COUNTS,
    FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
    MiniMaxH3Config,
    REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
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


def _shape_scalar(network, tensor, axis: int = 0):
    """Return one symbolic dimension as a one-element INT64 shape tensor."""

    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (axis,), (1,), (1,)).get_output(0)


def _shape_constant(network, value: int):
    return op.constant(network, np.asarray([value], dtype=np.int64), dtype=np.int64)


def _shape_vector(network, values):
    parts = [
        value if not isinstance(value, int) else _shape_constant(network, value) for value in values
    ]
    layer = network.add_concatenation(tuple(parts))
    layer.axis = 0
    return layer.get_output(0)


def _shape_subtract(network, left, value: int):
    return network.add_elementwise(
        left,
        _shape_constant(network, value),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)


def _dynamic_slice(network, tensor, starts, sizes):
    """Slice a tensor with symbolic start/size vectors and unit strides."""

    rank = len(tuple(tensor.shape))
    layer = network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
    layer.set_input(1, _shape_vector(network, starts))
    layer.set_input(2, _shape_vector(network, sizes))
    layer.set_input(
        3,
        op.constant(network, np.ones((rank,), dtype=np.int64), dtype=np.int64),
    )
    return layer.get_output(0)


def _scatter_rows(network, data, indices, updates, *, name: str):
    """Index-copy dynamic modality rows into one packed sequence buffer."""

    index_matrix = network.add_shuffle(indices)
    index_matrix.reshape_dims = (-1, 1)
    layer = network.add_scatter(
        data,
        index_matrix.get_output(0),
        updates,
        trt.ScatterMode.ND,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add MiniMax-H3 Ref2VA row scatter {name!r}")
    layer.name = name
    return layer.get_output(0)


def _slice_modulation(network, selected, index: int, rows: int, width: int):
    if rows < 0:
        value = _dynamic_slice(
            network,
            selected,
            (0, index, 0),
            (_shape_scalar(network, selected), 1, width),
        )
    else:
        value = network.add_slice(selected, (0, index, 0), (rows, 1, width), (1, 1, 1)).get_output(
            0
        )
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


def _dynamic_fused_qkv(network, tensor, weights: dict, prefix: str):
    """Fused QKV projection whose live row count remains symbolic."""

    packed_weight = np.concatenate(
        [weights[f"{prefix}.to_{name}.weight"] for name in ("q", "k", "v")], axis=0
    )
    packed = op.linear(network, tensor, packed_weight)
    rows = _shape_scalar(network, packed)
    width = int(packed.shape[1]) // 3
    return tuple(
        _dynamic_slice(network, packed, (0, index * width), (rows, width)) for index in range(3)
    )


def _dynamic_swiglu(network, tensor, weight_in, weight_out, ffn_dim: int):
    """SwiGLU without replacing a symbolic row dimension by padding."""

    projected = op.linear(network, tensor, weight_in)
    rows = _shape_scalar(network, projected)
    value = _dynamic_slice(network, projected, (0, 0), (rows, ffn_dim))
    gate = _dynamic_slice(network, projected, (0, ffn_dim), (rows, ffn_dim))
    activated = op.silu(network, gate)
    hidden = network.add_elementwise(value, activated, trt.ElementWiseOperation.PROD).get_output(0)
    return op.linear(network, hidden, weight_out)


def _dynamic_partial_rope(
    network,
    tensor,
    cos_half,
    sin_half,
    *,
    heads: int,
    head_dim: int,
    rotary_dim: int,
):
    """Apply rotate-half MM-RoPE while retaining the symbolic sequence axis."""

    value = op.rows_to_heads(network, tensor, -1, heads, head_dim)
    rows = _shape_scalar(network, value, 2)
    rotary = _dynamic_slice(
        network,
        value,
        (0, 0, 0, 0),
        (1, heads, rows, rotary_dim),
    )
    passthrough = _dynamic_slice(
        network,
        value,
        (0, 0, 0, rotary_dim),
        (1, heads, rows, head_dim - rotary_dim),
    )
    half = rotary_dim // 2
    first = _dynamic_slice(network, rotary, (0, 0, 0, 0), (1, heads, rows, half))
    second = _dynamic_slice(network, rotary, (0, 0, 0, half), (1, heads, rows, half))
    negative_second = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    rotated_layer = network.add_concatenation((negative_second, first))
    rotated_layer.axis = 3

    def duplicate_table(table):
        table = op.cast(network, table, value.dtype)
        reshape = network.add_shuffle(table)
        reshape.reshape_dims = (1, 1, -1, half)
        duplicate = network.add_concatenation((reshape.get_output(0), reshape.get_output(0)))
        duplicate.axis = 3
        return duplicate.get_output(0)

    cos = duplicate_table(cos_half)
    sin = duplicate_table(sin_half)
    left = network.add_elementwise(rotary, cos, trt.ElementWiseOperation.PROD).get_output(0)
    right = network.add_elementwise(
        rotated_layer.get_output(0), sin, trt.ElementWiseOperation.PROD
    ).get_output(0)
    rotated = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)
    result = network.add_concatenation((rotated, passthrough))
    result.axis = 3
    return op.heads_to_rows(network, result.get_output(0), -1, heads * head_dim)


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
    if rows < 0:
        q, k, v = _dynamic_fused_qkv(network, hidden, weights, f"{prefix}.attn")
    else:
        q, k, v = op.fused_qkv(network, hidden, weights, f"{prefix}.attn")
    q = _per_head_norm(network, q, weights[f"{prefix}.attn.norm_q.weight"], profile, rows)
    k = _per_head_norm(network, k, weights[f"{prefix}.attn.norm_k.weight"], profile, rows)
    if cos is not None:
        rotary_dim = 6 * profile.rope_freq_dim
        if rows < 0:
            q = _dynamic_partial_rope(
                network,
                q,
                cos,
                sin,
                heads=profile.num_heads,
                head_dim=profile.head_dim,
                rotary_dim=rotary_dim,
            )
            k = _dynamic_partial_rope(
                network,
                k,
                cos,
                sin,
                heads=profile.num_heads,
                head_dim=profile.head_dim,
                rotary_dim=rotary_dim,
            )
        else:
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
            attention_dtype=trt.bfloat16 if rows < 0 else trt.float16,
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
    rows: int | None = None,
):
    hidden = op.linear(
        network, text, weights["context_embedder.weight"], weights["context_embedder.bias"]
    )
    rows = profile.text_rows if rows is None else rows
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
        if rows < 0:
            update = _dynamic_swiglu(
                network,
                normalized,
                weights[f"{prefix}.ff.net.0.proj.weight"],
                weights[f"{prefix}.ff.net.2.weight"],
                profile.ffn_dim,
            )
        else:
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


def _packed_fl2va_hidden(network, video, audio, text, weights, profile: MiniMaxH3Config):
    """Project HF-ordered video rows and pack the padless FL2VA attention sequence.

    ``video`` follows the released transformer's ABI: zero to two keyframe
    blocks first, followed by the generated target-video rows.  The packed
    attention document is ``text | conditions | audio | target video``.
    """

    text_hidden = _refine_text(network, text, weights, profile, rows=-1)
    audio_hidden = op.linear(
        network,
        audio,
        weights["audio_proj_in.weight"],
        weights["audio_proj_in.bias"],
        bf16=False,
    )
    audio_hidden = op.cast(network, audio_hidden, trt.bfloat16)
    video_hidden = op.linear(
        network,
        video,
        weights["proj_in.weight"],
        weights["proj_in.bias"],
        bf16=False,
    )
    video_hidden = op.cast(network, video_hidden, trt.bfloat16)

    video_input_rows = _shape_scalar(network, video_hidden)
    condition_rows = _shape_subtract(network, video_input_rows, profile.video_rows)
    condition_hidden = _dynamic_slice(
        network,
        video_hidden,
        (0, 0),
        (condition_rows, profile.hidden_size),
    )
    target_hidden = _dynamic_slice(
        network,
        video_hidden,
        (condition_rows, 0),
        (profile.video_rows, profile.hidden_size),
    )
    packed = network.add_concatenation((text_hidden, condition_hidden, audio_hidden, target_hidden))
    packed.axis = 0
    return packed.get_output(0)


def _packed_ref2va_hidden(
    network,
    video,
    audio,
    text,
    video_indices,
    audio_indices,
    weights,
    profile: MiniMaxH3Config,
):
    """Project modality streams and scatter them into request-order Ref2VA rows.

    The video and audio inputs follow the released transformer's modality ABI:
    every condition row in request order within that modality, followed by the
    fixed target rows. ``video_indices`` and ``audio_indices`` are the official
    packed-sequence positions and preserve cross-modality reference order,
    including a video's soundtrack immediately before that video's pixels.
    """

    text_hidden = _refine_text(network, text, weights, profile, rows=-1)
    video_hidden = op.linear(
        network,
        video,
        weights["proj_in.weight"],
        weights["proj_in.bias"],
        bf16=False,
    )
    video_hidden = op.cast(network, video_hidden, trt.bfloat16)
    audio_hidden = op.linear(
        network,
        audio,
        weights["audio_proj_in.weight"],
        weights["audio_proj_in.bias"],
        bf16=False,
    )
    audio_hidden = op.cast(network, audio_hidden, trt.bfloat16)

    # Concatenation supplies a live, dynamically sized buffer without an
    # attention-visible capacity allocation. Every non-text row is overwritten
    # exactly once by the two official index-copy streams.
    buffer = network.add_concatenation((text_hidden, video_hidden, audio_hidden))
    buffer.axis = 0
    hidden = _scatter_rows(
        network,
        buffer.get_output(0),
        video_indices,
        video_hidden,
        name="ref2va.scatter_video_rows",
    )
    return _scatter_rows(
        network,
        hidden,
        audio_indices,
        audio_hidden,
        name="ref2va.scatter_audio_rows",
    )


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
    rows: int | None = None,
):
    """Add one native H3 transformer block and return its residual stream."""

    rows = profile.sequence_length if rows is None else rows
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
    if rows < 0:
        update = _dynamic_swiglu(
            network,
            normalized,
            weights[f"{prefix}.ff.net.0.proj.weight"],
            weights[f"{prefix}.ff.net.2.weight"],
            profile.ffn_dim,
        )
    else:
        update = op.swiglu(
            network,
            normalized,
            weights[f"{prefix}.ff.net.0.proj.weight"],
            weights[f"{prefix}.ff.net.2.weight"],
            profile.ffn_dim,
        )
    return op.gated_residual(network, hidden, update, gate_mlp)


def _final_hidden(
    network,
    hidden,
    timestep_indices,
    final_modulation,
    weights,
    profile,
    *,
    rows: int | None = None,
):
    rows = profile.sequence_length if rows is None else rows
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

    audio_hidden = network.add_slice(
        hidden,
        (profile.text_rows, 0),
        (profile.audio_rows, profile.hidden_size),
        (1, 1),
    ).get_output(0)
    video_hidden = network.add_slice(
        hidden,
        (profile.text_rows + profile.audio_rows, 0),
        (profile.video_rows, profile.hidden_size),
        (1, 1),
    ).get_output(0)
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


def _mark_fl2va_velocity_outputs(network, hidden, weights, profile: MiniMaxH3Config):
    """Expose only generated rows; keyframe conditions never reach an output head."""

    sequence_rows = _shape_scalar(network, hidden)
    video_start = _shape_subtract(network, sequence_rows, profile.video_rows)
    audio_start = _shape_subtract(network, video_start, profile.audio_rows)
    audio_hidden = _dynamic_slice(
        network,
        hidden,
        (audio_start, 0),
        (profile.audio_rows, profile.hidden_size),
    )
    video_hidden = _dynamic_slice(
        network,
        hidden,
        (video_start, 0),
        (profile.video_rows, profile.hidden_size),
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


def fl2va_optimization_profile_shapes(
    profile: MiniMaxH3Config,
) -> tuple[dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]], ...]:
    """Return the exact no-padding TRT shapes for 0/1/2-keyframe FL2VA.

    Each optimization profile fixes the video-input row count to one released
    keyframe mode.  Only the text axis varies inside a profile; consequently a
    runtime cannot bind arbitrary condition-row counts or pad one mode up to
    another mode's attention length.
    """

    profile.validate()
    shapes = []
    for keyframe_count in FL2VA_KEYFRAME_COUNTS:
        video_rows = profile.fl2va_video_input_rows(keyframe_count)
        minimum_rows = profile.fl2va_sequence_length(profile.min_text_rows, keyframe_count)
        optimum_rows = profile.fl2va_sequence_length(profile.text_rows, keyframe_count)
        maximum_rows = profile.fl2va_sequence_length(profile.max_text_rows, keyframe_count)
        shapes.append(
            {
                "video_hidden_states": (
                    (video_rows, profile.video_patch_dim),
                    (video_rows, profile.video_patch_dim),
                    (video_rows, profile.video_patch_dim),
                ),
                "encoder_hidden_states": (
                    (profile.min_text_rows, profile.text_dim),
                    (profile.text_rows, profile.text_dim),
                    (profile.max_text_rows, profile.text_dim),
                ),
                "position_ids": (
                    (minimum_rows, 3),
                    (optimum_rows, 3),
                    (maximum_rows, 3),
                ),
                "token_tags": (
                    (minimum_rows,),
                    (optimum_rows,),
                    (maximum_rows,),
                ),
                "timestep_indices": (
                    (minimum_rows,),
                    (optimum_rows,),
                    (maximum_rows,),
                ),
            }
        )
    return tuple(shapes)


def _add_fl2va_optimization_profiles(builder, config, profile: MiniMaxH3Config) -> None:
    for keyframe_count, input_shapes in zip(
        FL2VA_KEYFRAME_COUNTS,
        fl2va_optimization_profile_shapes(profile),
    ):
        optimization_profile = builder.create_optimization_profile()
        if optimization_profile is None:
            raise RuntimeError("TensorRT failed to create a MiniMax-H3 FL2VA optimization profile")
        for name, (minimum, optimum, maximum) in input_shapes.items():
            accepted = optimization_profile.set_shape(name, minimum, optimum, maximum)
            if accepted is False:
                raise RuntimeError(
                    "TensorRT rejected a MiniMax-H3 FL2VA optimization-profile shape: "
                    f"keyframes={keyframe_count}, input={name!r}, "
                    f"min={minimum}, opt={optimum}, max={maximum}"
                )
        if config.add_optimization_profile(optimization_profile) < 0:
            raise RuntimeError(
                "TensorRT rejected a MiniMax-H3 FL2VA optimization profile: "
                f"keyframes={keyframe_count}"
            )


def ref2va_optimization_profile_shapes(
    profile: MiniMaxH3Config,
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Return the one bounded, padless profile covering Ref2VA card maxima.

    Reference ordering is a runtime value rather than a shape bucket. The
    video, audio, and text axes therefore vary independently inside one profile,
    while every packed structural input tracks their exact live sum.
    """

    profile.validate()
    video_rows = tuple(
        profile.video_rows + rows
        for rows in (
            profile.ref2va_min_condition_video_rows,
            profile.ref2va_opt_condition_video_rows,
            profile.ref2va_max_condition_video_rows,
        )
    )
    audio_rows = tuple(
        profile.audio_rows + rows
        for rows in (
            profile.ref2va_min_condition_audio_rows,
            profile.ref2va_opt_condition_audio_rows,
            profile.ref2va_max_condition_audio_rows,
        )
    )
    text_rows = (
        profile.ref2va_min_text_rows,
        profile.ref2va_opt_text_rows,
        profile.ref2va_max_text_rows,
    )
    packed_rows = tuple(
        profile.ref2va_sequence_length(text, video - profile.video_rows, audio - profile.audio_rows)
        for text, video, audio in zip(text_rows, video_rows, audio_rows)
    )
    return {
        "video_hidden_states": tuple((rows, profile.video_patch_dim) for rows in video_rows),
        "audio_hidden_states": tuple((rows, profile.audio_in_channels) for rows in audio_rows),
        "encoder_hidden_states": tuple((rows, profile.text_dim) for rows in text_rows),
        "video_indices": tuple((rows,) for rows in video_rows),
        "audio_indices": tuple((rows,) for rows in audio_rows),
        "position_ids": tuple((rows, 3) for rows in packed_rows),
        "token_tags": tuple((rows,) for rows in packed_rows),
        "timestep_indices": tuple((rows,) for rows in packed_rows),
    }


def _add_ref2va_optimization_profile(builder, config, profile: MiniMaxH3Config) -> None:
    optimization_profile = builder.create_optimization_profile()
    if optimization_profile is None:
        raise RuntimeError("TensorRT failed to create a MiniMax-H3 Ref2VA optimization profile")
    for name, (minimum, optimum, maximum) in ref2va_optimization_profile_shapes(profile).items():
        accepted = optimization_profile.set_shape(name, minimum, optimum, maximum)
        if accepted is False:
            raise RuntimeError(
                "TensorRT rejected a MiniMax-H3 Ref2VA optimization-profile shape: "
                f"input={name!r}, min={minimum}, opt={optimum}, max={maximum}"
            )
    if config.add_optimization_profile(optimization_profile) < 0:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 Ref2VA optimization profile")


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
    rows = profile.sequence_length
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)

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
    hidden = _packed_hidden(network, video, audio, text, weights, profile)

    cos, sin = _rope_tables(network, positions, profile, rows)
    # Pristine Diffusers packs exactly 38,247 rows on one device, so its
    # attention mask is None. Preserve that contract without synthetic padding.
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
    _mark_full_velocity_outputs(network, hidden, weights)

    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + profile.num_layers,
        label="DiT",
    )

    print(
        f"[minimax-h3] building native DiT: layers={profile.num_layers}, "
        f"packed={profile.sequence_length}, devices=1",
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


def build_fl2va_dit_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    checkpoint_subfolder: str = FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the padless dynamic-text FL2VA Omni-DiT TensorRT plan.

    ABI summary:

    * ``video_hidden_states`` contains 0/1/2 leading 1,008-row keyframe
      blocks followed by exactly ``profile.video_rows`` generated rows.
    * ``encoder_hidden_states`` has a genuinely dynamic text axis.
    * ``position_ids``, ``token_tags`` and ``timestep_indices`` describe the
      complete live sequence in ``text | conditions | audio | target`` order.
    * the only outputs are target-video and target-audio velocity rows.

    Three optimization profiles fix the keyframe count without padded rows.
    The supplied weights must come from the released ``transformer/`` FL2VA
    partition, never ``transformer_ref/``.
    """

    profile.validate()
    if profile.first_block_cache:
        raise ValueError(
            "MiniMax-H3 dynamic FL2VA currently uses the monolithic DiT plan; "
            "the fixed-row FirstBlockCache plans remain a separate ABI"
        )
    if checkpoint_subfolder != FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER:
        raise ValueError(
            "MiniMax-H3 FL2VA DiT requires weights from checkpoint subfolder "
            f"{FL2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER!r}, got {checkpoint_subfolder!r}"
        )

    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    video = network.add_input(
        "video_hidden_states",
        trt.float32,
        (-1, profile.video_patch_dim),
    )
    audio = network.add_input(
        "audio_hidden_states",
        trt.float32,
        (profile.audio_rows, profile.audio_in_channels),
    )
    text = network.add_input(
        "encoder_hidden_states",
        trt.float32,
        (-1, profile.text_dim),
    )
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    token_tags = network.add_input("token_tags", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    block_modulations = [
        network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(profile.num_layers)
    ]
    final_modulation = network.add_input(
        "final_modulation",
        trt.bfloat16,
        (profile.max_timestep_count, 2, profile.hidden_size),
    )

    hidden = _packed_fl2va_hidden(network, video, audio, text, weights, profile)
    modality_count = op.constant(
        network,
        np.asarray([3], dtype=np.int32),
        dtype=np.int32,
    )
    adaln_indices = network.add_elementwise(
        timestep_indices,
        modality_count,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    adaln_indices = network.add_elementwise(
        adaln_indices,
        token_tags,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    cos, sin = _rope_tables(network, positions, profile, -1)
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
            rows=-1,
        )
    hidden = _final_hidden(
        network,
        hidden,
        timestep_indices,
        final_modulation,
        weights,
        profile,
        rows=-1,
    )
    _mark_fl2va_velocity_outputs(network, hidden, weights, profile)
    _add_fl2va_optimization_profiles(builder, config, profile)

    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + profile.num_layers,
        label="FL2VA DiT",
    )
    print(
        "[minimax-h3] building native FL2VA DiT: "
        f"layers={profile.num_layers}, text={profile.min_text_rows}:"
        f"{profile.text_rows}:{profile.max_text_rows}, keyframes=0/1/2, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="FL2VA DiT",
    )


def build_ref2va_dit_engine(
    weights: dict,
    profile: MiniMaxH3Config,
    *,
    checkpoint_subfolder: str = REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
) -> bytes:
    """Build the request-order, fully dynamic Ref2VA Omni-DiT plan.

    ``video_hidden_states`` and ``audio_hidden_states`` each contain all
    condition rows first and their fixed target rows last. Runtime-provided
    ``video_indices`` / ``audio_indices`` scatter those modality streams into
    ``text | reference blocks | target audio | target video`` order. The
    positions, tags, and timestep indices describe that same live sequence.
    Condition counts are therefore unambiguous from each dynamic input length
    minus its fixed target length; there is no redundant scalar that can drift.

    The single wide profile carries the documented 9-image, 3-video, 3-audio,
    12-total-reference maxima without allocating padding rows. It deliberately
    exposes the full 560,382-row upper bound to TensorRT rather than silently
    compiling a smaller task. That bound contains 314,027,985,924 query/key
    pairs per head (17,585,567,211,744 across 56 heads), so full-scale runtime
    practicality is not inferred from the tiny topology test. A backend that
    cannot compile native attention at the declared bound fails serialization.
    """

    profile.validate()
    if profile.first_block_cache:
        raise ValueError(
            "MiniMax-H3 dynamic Ref2VA currently uses the monolithic DiT plan; "
            "the fixed-row FirstBlockCache plans remain a separate ABI"
        )
    if checkpoint_subfolder != REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER:
        raise ValueError(
            "MiniMax-H3 Ref2VA DiT requires weights from checkpoint subfolder "
            f"{REF2VA_TRANSFORMER_CHECKPOINT_SUBFOLDER!r}, got {checkpoint_subfolder!r}"
        )

    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    video = network.add_input(
        "video_hidden_states",
        trt.float32,
        (-1, profile.video_patch_dim),
    )
    audio = network.add_input(
        "audio_hidden_states",
        trt.float32,
        (-1, profile.audio_in_channels),
    )
    text = network.add_input(
        "encoder_hidden_states",
        trt.float32,
        (-1, profile.text_dim),
    )
    video_indices = network.add_input("video_indices", trt.int32, (-1,))
    audio_indices = network.add_input("audio_indices", trt.int32, (-1,))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    token_tags = network.add_input("token_tags", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    block_modulations = [
        network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(profile.num_layers)
    ]
    final_modulation = network.add_input(
        "final_modulation",
        trt.bfloat16,
        (profile.max_timestep_count, 2, profile.hidden_size),
    )

    hidden = _packed_ref2va_hidden(
        network,
        video,
        audio,
        text,
        video_indices,
        audio_indices,
        weights,
        profile,
    )
    modality_count = op.constant(
        network,
        np.asarray([3], dtype=np.int32),
        dtype=np.int32,
    )
    adaln_indices = network.add_elementwise(
        timestep_indices,
        modality_count,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    adaln_indices = network.add_elementwise(
        adaln_indices,
        token_tags,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    cos, sin = _rope_tables(network, positions, profile, -1)
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
            rows=-1,
        )
    hidden = _final_hidden(
        network,
        hidden,
        timestep_indices,
        final_modulation,
        weights,
        profile,
        rows=-1,
    )
    _mark_fl2va_velocity_outputs(network, hidden, weights, profile)
    _add_ref2va_optimization_profile(builder, config, profile)

    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + profile.num_layers,
        label="Ref2VA DiT",
    )
    maximum_rows = profile.ref2va_sequence_length(
        profile.ref2va_max_text_rows,
        profile.ref2va_max_condition_video_rows,
        profile.ref2va_max_condition_audio_rows,
    )
    print(
        "[minimax-h3] building native Ref2VA DiT: "
        f"layers={profile.num_layers}, max_packed={maximum_rows}, devices=1",
        file=sys.stderr,
    )
    return _serialize(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="Ref2VA DiT",
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
    rows = profile.sequence_length
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
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
    block_modulation = network.add_input(
        "block_modulation_0",
        trt.bfloat16,
        (profile.adaln_table_rows, 6, profile.hidden_size),
    )
    previous_head_residual = network.add_input(
        "previous_head_residual", trt.bfloat16, (rows, profile.hidden_size)
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
        f"[minimax-h3] building native DiT cache head: packed={rows}, devices=1",
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
    rows = profile.sequence_length
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (rows, profile.hidden_size))
    positions = network.add_input("position_ids", trt.float32, (rows, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (rows,))
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
        f"packed={rows}, devices=1",
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
    rows = profile.sequence_length
    logger, builder, network, config = _native_builder(verbose, workspace_bytes)
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (rows, profile.hidden_size))
    tail_residual = network.add_input("tail_residual", trt.bfloat16, (rows, profile.hidden_size))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (rows,))
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
        f"[minimax-h3] building native DiT cache finish: packed={rows}, devices=1",
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

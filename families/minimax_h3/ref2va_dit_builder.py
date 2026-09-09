# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT-RTX builder for the distinct MiniMax-H3 Ref2VA DiT.

Unlike the T2VA/FL2VA layout, Ref2VA reference blocks are interleaved in
request order.  The released transformer scatters text, video and audio rows
through explicit index arrays and gathers the output heads through the same
arrays.  This builder preserves that ABI; concatenating ``text|audio|video``
would silently change every reference after the first mixed-modality block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import trt_compat

from . import dit_builder as dense
from . import graph_ops as op
from .adaln_builder import build_adaln_precompute_engine
from .config import MiniMaxH3Config
from .ref2va_checkpoint import (
    REF2VA_ADALN_KEYS,
    REF2VA_DENOISER_KEYS,
    REF2VA_FINISH_KEYS,
    REF2VA_HEAD_KEYS,
    REF2VA_TAIL_KEYS,
)
from .ref2va_contract import (
    Ref2VADenoiserProfile,
    ref2va_denoiser_profiles,
    ref2va_first_block_cache_abis,
)


trt = trt_compat.get_trt()


def checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_DENOISER_KEYS


def adaln_checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_ADALN_KEYS


def head_checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_HEAD_KEYS


def tail_checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_TAIL_KEYS


def finish_checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_FINISH_KEYS


def native_profile(
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    *,
    first_block_cache: bool = False,
) -> MiniMaxH3Config:
    """Translate the public scatter/gather capacity to the shared H3 graph profile."""

    capacity.validate()
    profile = MiniMaxH3Config(
        min_video_rows=capacity.min_video_rows,
        opt_video_rows=capacity.opt_video_rows,
        video_rows=capacity.max_video_rows,
        min_audio_rows=capacity.min_audio_rows,
        opt_audio_rows=capacity.opt_audio_rows,
        audio_rows=capacity.max_audio_rows,
        min_text_rows=capacity.min_text_rows,
        opt_text_rows=capacity.opt_text_rows,
        text_rows=capacity.max_text_rows,
        padded_sequence_length=capacity.max_packed_rows,
        max_timestep_count=4,
        first_block_cache=first_block_cache,
    )
    profile.validate()
    return profile


def _set_profile_shape(optimization, name: str, shapes: tuple[tuple[int, ...], ...]) -> None:
    # TensorRT-RTX returns None even when set_shape succeeds.  Verify the
    # recorded profile instead of interpreting the binding return as a bool.
    error = f"TensorRT rejected MiniMax-H3 Ref2VA profile binding {name}"
    try:
        result = optimization.set_shape(name, min=shapes[0], opt=shapes[1], max=shapes[2])
        recorded = tuple(
            tuple(int(dimension) for dimension in shape) for shape in optimization.get_shape(name)
        )
    except (RuntimeError, ValueError) as exception:
        raise RuntimeError(error) from exception
    if result is False or recorded != shapes or not optimization:
        raise RuntimeError(error)


def _add_optimization_profile(
    builder,
    config,
    capacity: Ref2VADenoiserProfile,
    *,
    expected_index: int = 0,
    extra_memory_target: float | None = None,
) -> None:
    optimization = builder.create_optimization_profile()
    if extra_memory_target is not None:
        optimization.extra_memory_target = extra_memory_target
    video = tuple(
        (rows, 96)
        for rows in (
            capacity.min_video_rows,
            capacity.opt_video_rows,
            capacity.max_video_rows,
        )
    )
    audio = tuple(
        (rows, 32)
        for rows in (
            capacity.min_audio_rows,
            capacity.opt_audio_rows,
            capacity.max_audio_rows,
        )
    )
    text = tuple(
        (rows, 5120)
        for rows in (
            capacity.min_text_rows,
            capacity.opt_text_rows,
            capacity.max_text_rows,
        )
    )
    packed_rows = (
        capacity.min_packed_rows,
        capacity.opt_packed_rows,
        capacity.max_packed_rows,
    )
    _set_profile_shape(optimization, "video_hidden_states", video)
    _set_profile_shape(optimization, "audio_hidden_states", audio)
    _set_profile_shape(optimization, "encoder_hidden_states", text)
    _set_profile_shape(
        optimization,
        "position_ids",
        tuple((rows, 3) for rows in packed_rows),
    )
    for name, rows in (
        ("video_indices", tuple(shape[0] for shape in video)),
        ("audio_indices", tuple(shape[0] for shape in audio)),
        ("text_indices", tuple(shape[0] for shape in text)),
        ("adaln_indices", packed_rows),
        ("timestep_indices", packed_rows),
    ):
        _set_profile_shape(optimization, name, tuple((value,) for value in rows))
    if config.add_optimization_profile(optimization) != expected_index:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 Ref2VA optimization profile")


def _add_optimization_profiles(
    builder,
    config,
    capacity: Ref2VADenoiserProfile,
) -> None:
    for index, optimization_capacity in enumerate(ref2va_denoiser_profiles(capacity)):
        _add_optimization_profile(
            builder,
            config,
            optimization_capacity,
            expected_index=index,
            extra_memory_target=0.0 if index else None,
        )


def _add_cache_optimization_profiles(
    builder, config, capacity: Ref2VADenoiserProfile, component: str
) -> None:
    for index, optimization_capacity in enumerate(ref2va_denoiser_profiles(capacity)):
        optimization = builder.create_optimization_profile()
        if index:
            optimization.extra_memory_target = 0.0
        for binding in ref2va_first_block_cache_abis(optimization_capacity)[component].inputs:
            if binding.name.startswith("block_modulation_") or binding.name == "final_modulation":
                continue
            _set_profile_shape(
                optimization,
                binding.name,
                (binding.min_shape, binding.opt_shape, binding.max_shape),
            )
        if config.add_optimization_profile(optimization) != index:
            raise RuntimeError("TensorRT rejected the MiniMax-H3 Ref2VA cache profile")


def _require_checkpoint_partition(weights: dict, expected: tuple[str, ...], label: str) -> None:
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"MiniMax-H3 transformer_ref {label} checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )


def _scatter_rows(network, base, row_indices, rows, *, label: str):
    indices = network.add_shuffle(row_indices)
    indices.reshape_dims = (-1, 1)
    layer = network.add_scatter(base, indices.get_output(0), rows, trt.ScatterMode.ND)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Ref2VA {label} row scatter")
    layer.name = f"ref2va.scatter_{label}"
    return layer.get_output(0)


def _packed_hidden(
    network,
    video,
    audio,
    text,
    positions,
    video_indices,
    audio_indices,
    text_indices,
    weights,
    profile: MiniMaxH3Config,
    *,
    consume_weights: bool,
):
    text_hidden = dense._refine_text(  # noqa: SLF001 - one family-owned graph vocabulary
        network,
        text,
        weights,
        profile,
        consume_weights=consume_weights,
    )
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

    # Derive the dynamic sequence axis from position_ids, then broadcast a
    # single zero row to [sequence, hidden] without a host-side shape tensor.
    first_position = op.dynamic_slice(network, positions, (0, 0), (None, 1))
    first_position = op.cast(network, first_position, trt.bfloat16)
    zeros = op.constant(network, np.zeros((1, profile.hidden_size), dtype=np.float32))
    zeros = op.cast(network, zeros, trt.bfloat16)
    base = network.add_elementwise(first_position, zeros, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    packed = _scatter_rows(network, base, text_indices, text_hidden, label="text")
    packed = _scatter_rows(network, packed, video_indices, video_hidden, label="video")
    return _scatter_rows(network, packed, audio_indices, audio_hidden, label="audio")


def _mark_gathered_outputs(network, hidden, weights, video_indices, audio_indices) -> None:
    video_hidden = op.gather_rows(network, hidden, video_indices)
    audio_hidden = op.gather_rows(network, hidden, audio_indices)
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


@op.cleanup_failed_build
def build_ref2va_dit_engine(
    weights: dict,
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build one dense native plan from the real ``transformer_ref`` values."""

    expected = set(checkpoint_keys())
    missing = sorted(expected - set(weights))
    unexpected = sorted(set(weights) - expected)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 transformer_ref denoiser checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    profile = native_profile(capacity)
    logger, builder, network, config = dense._native_builder(  # noqa: SLF001
        verbose,
        workspace_bytes,
        weight_streaming=weight_streaming,
    )

    video = network.add_input("video_hidden_states", trt.float32, (-1, 96))
    audio = network.add_input("audio_hidden_states", trt.float32, (-1, 32))
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, 5120))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    video_indices = network.add_input("video_indices", trt.int32, (-1,))
    audio_indices = network.add_input("audio_indices", trt.int32, (-1,))
    text_indices = network.add_input("text_indices", trt.int32, (-1,))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    _add_optimization_profiles(builder, config, capacity)
    block_modulations = tuple(
        network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(profile.num_layers)
    )
    final_modulation = network.add_input(
        "final_modulation",
        trt.bfloat16,
        (profile.max_timestep_count, 2, profile.hidden_size),
    )

    hidden = _packed_hidden(
        network,
        video,
        audio,
        text,
        positions,
        video_indices,
        audio_indices,
        text_indices,
        weights,
        profile,
        consume_weights=consume_weights,
    )
    cos, sin = dense._rope_tables(network, positions, profile)  # noqa: SLF001
    for index in range(profile.num_layers):
        hidden = dense._transformer_block(  # noqa: SLF001
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
    hidden = dense._final_hidden(  # noqa: SLF001
        network,
        hidden,
        timestep_indices,
        final_modulation,
        weights,
        profile,
    )
    _mark_gathered_outputs(network, hidden, weights, video_indices, audio_indices)
    op.validate_native_network(
        network,
        expected_attentions=profile.num_refiner_layers + profile.num_layers,
        label="Ref2VA transformer_ref DiT",
    )
    return dense._serialize(  # noqa: SLF001
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="Ref2VA transformer_ref DiT",
        output_path=output_path,
    )


def build_ref2va_adaln_precompute_engine(
    weights: dict,
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    **kwargs,
) -> bytes | dict[str, int | str]:
    """Build the separate AdaLN plan from ``transformer_ref`` only."""

    expected = set(adaln_checkpoint_keys())
    missing = sorted(expected - set(weights))
    unexpected = sorted(set(weights) - expected)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 transformer_ref AdaLN checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return build_adaln_precompute_engine(weights, native_profile(capacity), **kwargs)


def _target_relative_change(network, current, previous, indices):
    # Compute each generated modality separately; unchanged conditioning rows
    # must never dilute the decision, nor may video length mask audio changes.
    current = op.cast(network, op.gather_rows(network, current, indices), trt.float32)
    previous = op.cast(network, op.gather_rows(network, previous, indices), trt.float32)
    delta = network.add_elementwise(current, previous, trt.ElementWiseOperation.SUB).get_output(0)
    delta_abs = network.add_unary(delta, trt.UnaryOperation.ABS).get_output(0)
    previous_abs = network.add_unary(previous, trt.UnaryOperation.ABS).get_output(0)
    axes = (1 << 0) | (1 << 1)
    numerator = network.add_reduce(delta_abs, trt.ReduceOperation.SUM, axes, True).get_output(0)
    denominator = network.add_reduce(previous_abs, trt.ReduceOperation.SUM, axes, True).get_output(
        0
    )
    epsilon = op.constant(network, np.full((1, 1), 1.0e-8, dtype=np.float32))
    denominator = network.add_elementwise(
        denominator, epsilon, trt.ElementWiseOperation.MAX
    ).get_output(0)
    return network.add_elementwise(numerator, denominator, trt.ElementWiseOperation.DIV).get_output(
        0
    )


def _cache_metric(network, video_change, audio_change):
    metric = network.add_elementwise(
        video_change, audio_change, trt.ElementWiseOperation.MAX
    ).get_output(0)
    # MAX need not propagate NaN from both operands. Explicitly fail closed if
    # either modality is invalid instead of allowing the other to hide it.
    video_nan = network.add_unary(video_change, trt.UnaryOperation.ISNAN).get_output(0)
    audio_nan = network.add_unary(audio_change, trt.UnaryOperation.ISNAN).get_output(0)
    invalid = network.add_elementwise(video_nan, audio_nan, trt.ElementWiseOperation.OR).get_output(
        0
    )
    infinity = op.constant(network, np.full((1, 1), np.inf, dtype=np.float32))
    metric = network.add_select(invalid, infinity, metric).get_output(0)
    reshape = network.add_shuffle(metric)
    reshape.reshape_dims = (1,)
    return reshape.get_output(0)


@op.cleanup_failed_build
def build_ref2va_dit_head_engine(
    weights: dict,
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build Ref2VA scatter packing, block zero and a target-only cache metric."""

    _require_checkpoint_partition(weights, REF2VA_HEAD_KEYS, "cache head")
    profile = native_profile(capacity, first_block_cache=True)
    logger, builder, network, config = dense._native_builder(  # noqa: SLF001
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    video = network.add_input("video_hidden_states", trt.float32, (-1, 96))
    audio = network.add_input("audio_hidden_states", trt.float32, (-1, 32))
    text = network.add_input("encoder_hidden_states", trt.float32, (-1, 5120))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    video_indices = network.add_input("video_indices", trt.int32, (-1,))
    audio_indices = network.add_input("audio_indices", trt.int32, (-1,))
    text_indices = network.add_input("text_indices", trt.int32, (-1,))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    block_modulation = network.add_input(
        "block_modulation_0", trt.bfloat16, (profile.adaln_table_rows, 6, profile.hidden_size)
    )
    previous = network.add_input("previous_head_residual", trt.bfloat16, (-1, profile.hidden_size))
    cache_video_indices = network.add_input("cache_video_indices", trt.int32, (-1,))
    cache_audio_indices = network.add_input("cache_audio_indices", trt.int32, (-1,))
    _add_cache_optimization_profiles(builder, config, capacity, "ref2va_dit_head")
    packed = _packed_hidden(
        network,
        video,
        audio,
        text,
        positions,
        video_indices,
        audio_indices,
        text_indices,
        weights,
        profile,
        consume_weights=consume_weights,
    )
    cos, sin = dense._rope_tables(network, positions, profile)  # noqa: SLF001
    hidden = dense._transformer_block(  # noqa: SLF001
        network,
        packed,
        block_modulation,
        adaln_indices,
        cos,
        sin,
        weights,
        profile,
        0,
        consume_weights=consume_weights,
    )
    residual = network.add_elementwise(hidden, packed, trt.ElementWiseOperation.SUB).get_output(0)
    video_change = _target_relative_change(network, residual, previous, cache_video_indices)
    audio_change = _target_relative_change(network, residual, previous, cache_audio_indices)
    metric = _cache_metric(network, video_change, audio_change)
    for tensor, name in (
        (hidden, "head_hidden"),
        (residual, "head_residual"),
        (metric, "cache_metric"),
    ):
        tensor.name = name
        network.mark_output(tensor)
    op.validate_native_network(
        network, expected_attentions=profile.num_refiner_layers + 1, label="Ref2VA cache head"
    )
    return dense._serialize(  # noqa: SLF001
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="Ref2VA cache head",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_ref2va_dit_tail_engine(
    weights: dict,
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build Ref2VA blocks one through 49 and their reusable packed residual."""

    _require_checkpoint_partition(weights, REF2VA_TAIL_KEYS, "cache tail")
    profile = native_profile(capacity, first_block_cache=True)
    logger, builder, network, config = dense._native_builder(  # noqa: SLF001
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (-1, profile.hidden_size))
    positions = network.add_input("position_ids", trt.float32, (-1, 3))
    adaln_indices = network.add_input("adaln_indices", trt.int32, (-1,))
    modulations = {
        index: network.add_input(
            f"block_modulation_{index}",
            trt.bfloat16,
            (profile.adaln_table_rows, 6, profile.hidden_size),
        )
        for index in range(1, profile.num_layers)
    }
    _add_cache_optimization_profiles(builder, config, capacity, "ref2va_dit_tail")
    cos, sin = dense._rope_tables(network, positions, profile)  # noqa: SLF001
    hidden = head_hidden
    for index in range(1, profile.num_layers):
        hidden = dense._transformer_block(  # noqa: SLF001
            network,
            hidden,
            modulations[index],
            adaln_indices,
            cos,
            sin,
            weights,
            profile,
            index,
            consume_weights=consume_weights,
        )
    residual = network.add_elementwise(
        hidden, head_hidden, trt.ElementWiseOperation.SUB
    ).get_output(0)
    residual.name = "tail_residual"
    network.mark_output(residual)
    op.validate_native_network(
        network, expected_attentions=profile.num_layers - 1, label="Ref2VA cache tail"
    )
    return dense._serialize(  # noqa: SLF001
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="Ref2VA cache tail",
        output_path=output_path,
    )


@op.cleanup_failed_build
def build_ref2va_dit_finish_engine(
    weights: dict,
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Apply the selected residual and gather both Ref2VA velocity outputs."""

    _require_checkpoint_partition(weights, REF2VA_FINISH_KEYS, "cache finish")
    profile = native_profile(capacity, first_block_cache=True)
    logger, builder, network, config = dense._native_builder(  # noqa: SLF001
        verbose, workspace_bytes, weight_streaming=weight_streaming
    )
    head_hidden = network.add_input("head_hidden", trt.bfloat16, (-1, profile.hidden_size))
    tail_residual = network.add_input("tail_residual", trt.bfloat16, (-1, profile.hidden_size))
    timestep_indices = network.add_input("timestep_indices", trt.int32, (-1,))
    video_indices = network.add_input("video_indices", trt.int32, (-1,))
    audio_indices = network.add_input("audio_indices", trt.int32, (-1,))
    final_modulation = network.add_input(
        "final_modulation", trt.bfloat16, (profile.max_timestep_count, 2, profile.hidden_size)
    )
    _add_cache_optimization_profiles(builder, config, capacity, "ref2va_dit_finish")
    hidden = network.add_elementwise(
        head_hidden, tail_residual, trt.ElementWiseOperation.SUM
    ).get_output(0)
    hidden = dense._final_hidden(  # noqa: SLF001
        network, hidden, timestep_indices, final_modulation, weights, profile
    )
    _mark_gathered_outputs(network, hidden, weights, video_indices, audio_indices)
    op.validate_native_network(network, expected_attentions=0, label="Ref2VA cache finish")
    return dense._serialize(  # noqa: SLF001
        logger=logger,
        builder=builder,
        network=network,
        config=config,
        weights=weights,
        consume_weights=consume_weights,
        label="Ref2VA cache finish",
        output_path=output_path,
    )

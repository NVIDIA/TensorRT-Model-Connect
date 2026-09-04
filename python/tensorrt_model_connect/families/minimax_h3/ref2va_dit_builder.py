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

from tensorrt_model_connect import trt_compat

from . import dit_builder as dense
from . import graph_ops as op
from .adaln_builder import build_adaln_precompute_engine
from .config import MiniMaxH3Config
from .ref2va_checkpoint import REF2VA_ADALN_KEYS, REF2VA_DENOISER_KEYS
from .ref2va_contract import Ref2VADenoiserProfile


trt = trt_compat.get_trt()


def checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_DENOISER_KEYS


def adaln_checkpoint_keys() -> tuple[str, ...]:
    return REF2VA_ADALN_KEYS


def native_profile(
    capacity: Ref2VADenoiserProfile = Ref2VADenoiserProfile(),
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
        first_block_cache=False,
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


def _add_optimization_profile(builder, config, capacity: Ref2VADenoiserProfile) -> None:
    optimization = builder.create_optimization_profile()
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
    if config.add_optimization_profile(optimization) != 0:
        raise RuntimeError("TensorRT rejected the MiniMax-H3 Ref2VA optimization profile")


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
    _add_optimization_profile(builder, config, capacity)
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

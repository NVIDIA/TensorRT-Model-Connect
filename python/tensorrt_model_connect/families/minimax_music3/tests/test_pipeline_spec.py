# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 orchestration constants."""

from __future__ import annotations

import importlib

import pytest

spec = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
)


def test_constants_match_the_reference() -> None:
    assert spec.CHUNK_FRAMES == 200
    assert spec.CHUNK_HOP == 100
    assert spec.DEFAULT_INFERENCE_STEPS == 30
    assert spec.CROP_LEFT_LATENT == 86
    assert spec.CROP_RIGHT_LATENT == 258
    assert spec.LATENT_HOP_LENGTH == 512
    assert spec.SAMPLING_RATE == 44100


def test_crops_remove_exactly_one_overlap() -> None:
    assert spec.CROP_LEFT_LATENT + spec.CROP_RIGHT_LATENT == spec.OVERLAP_LATENT_FRAMES


def test_overlap_is_the_hop_expressed_in_latent_frames() -> None:
    """A 100-frame hop at 25 Hz is 4 s, which is ~344 latent frames."""

    hop_seconds = spec.CHUNK_HOP / spec.FRAME_RATE_HZ
    latent_frames = hop_seconds * spec.latent_frames_per_second()

    assert latent_frames == pytest.approx(spec.OVERLAP_LATENT_FRAMES, abs=1.0)


def test_short_generation_uses_a_single_window() -> None:
    for frames in (1, 50, 199, 200):
        plan = spec.chunk_starts(frames)
        assert plan.starts == (0,), frames
        assert plan.count == 1


def test_long_generation_windows_step_by_the_hop() -> None:
    plan = spec.chunk_starts(1000)

    assert plan.starts == tuple(range(0, 900, 100))
    assert plan.count == 9
    # Every window but the last is fully covered by the next hop.
    assert plan.starts[-1] + spec.CHUNK_FRAMES >= 1000


def test_chunk_starts_rejects_an_empty_generation() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        spec.chunk_starts(0)


def test_sigma_schedule_is_a_linear_ramp() -> None:
    sigmas = spec.sigma_schedule(30)

    assert len(sigmas) == 30
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == pytest.approx(1.0 / 30)
    deltas = {round(b - a, 12) for a, b in zip(sigmas, sigmas[1:])}
    assert len(deltas) == 1  # evenly spaced


def test_sigma_schedule_edge_cases() -> None:
    assert spec.sigma_schedule(1) == (1.0,)
    with pytest.raises(ValueError, match="at least 1"):
        spec.sigma_schedule(0)


def test_latent_rate_and_duration_cap() -> None:
    assert spec.latent_frames_per_second() == pytest.approx(44100 / 512)
    # 9000 frames at 25 Hz is six minutes, not the five the request states.
    assert spec.max_audio_seconds() == pytest.approx(360.0)


def test_transformer_call_count_accounts_for_guidance() -> None:
    # A 60 s generation is 1500 frames -> 14 windows.
    frames = int(60 * spec.FRAME_RATE_HZ)
    windows = spec.chunk_starts(frames).count

    assert spec.transformer_calls(frames) == windows * 30 * 2


def test_single_window_generation_call_count() -> None:
    assert spec.transformer_calls(200, num_inference_steps=30) == 60

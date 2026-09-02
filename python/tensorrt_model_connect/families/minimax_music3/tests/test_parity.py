# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 parity checks.

The recorded values these assert against were produced by running the pinned
reference on an A40; this module was then run over those artifacts and every
check passed. The fixtures here stand in for them so the tests carry no
multi-megabyte data.
"""

from __future__ import annotations

import importlib

import pytest

parity = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.parity"
)


class _Fake:
    """Minimal stand-in for an array: a shape and a standard deviation."""

    def __init__(self, shape: tuple[int, ...], std: float) -> None:
        self.shape = shape
        self._std = std

    def std(self) -> float:
        return self._std


def _reference_chunks() -> list[_Fake]:
    return [
        _Fake(parity.BASELINE_LATENT_SHAPE, std)
        for std in parity.BASELINE_LATENT_STD
    ]


def _reference_hiddens() -> _Fake:
    return _Fake(
        parity.BASELINE_FRAME_HIDDENS_SHAPE, parity.BASELINE_FRAME_HIDDENS_STD
    )


def test_recorded_run_passes_every_stage() -> None:
    results = [
        parity.check_chunk_starts(parity.BASELINE_CHUNK_STARTS),
        parity.check_frame_hiddens(_reference_hiddens()),
        parity.check_latent_chunks(_reference_chunks()),
        parity.check_waveform(
            parity.BASELINE_WAVEFORM_SAMPLES, parity.BASELINE_WAVEFORM_CHANNELS
        ),
    ]

    assert all(result.passed for result in results)
    assert parity.first_failure(results) is None


def test_baseline_shapes_agree_with_the_pipeline_spec() -> None:
    spec = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.pipeline_spec"
    )

    # 20 s at 25 Hz is 500 frames, which plans four windows.
    frames = int(parity.BASELINE_AUDIO_SECONDS * spec.FRAME_RATE_HZ)
    assert spec.chunk_starts(frames).starts == parity.BASELINE_CHUNK_STARTS
    assert parity.BASELINE_FRAME_HIDDENS_SHAPE[1] == frames

    # One window is CHUNK_FRAMES of audio expressed in latent frames.
    window_seconds = spec.CHUNK_FRAMES / spec.FRAME_RATE_HZ
    latent_frames = window_seconds * spec.latent_frames_per_second()
    assert parity.BASELINE_LATENT_SHAPE[2] == int(latent_frames)

    # The waveform is the whole request at the vocoder's rate.
    expected = parity.BASELINE_AUDIO_SECONDS * spec.SAMPLING_RATE
    assert parity.BASELINE_WAVEFORM_SAMPLES == pytest.approx(expected, rel=1e-3)


def test_a_different_window_plan_is_caught() -> None:
    result = parity.check_chunk_starts((0, 200, 400))

    assert not result.passed
    assert result.stage == "chunk_starts"


def test_a_wrong_latent_width_is_caught() -> None:
    chunks = _reference_chunks()
    chunks[2] = _Fake((1, 64, 689), parity.BASELINE_LATENT_STD[2])

    result = parity.check_latent_chunks(chunks)

    assert not result.passed
    assert "window 2" in result.detail


def test_collapsed_latents_are_caught() -> None:
    """A window that denoised to nothing has a near-zero spread."""

    chunks = _reference_chunks()
    chunks[0] = _Fake(parity.BASELINE_LATENT_SHAPE, 0.01)

    result = parity.check_latent_chunks(chunks)

    assert not result.passed
    assert "drifted" in result.detail


def test_a_missing_window_is_caught() -> None:
    result = parity.check_latent_chunks(_reference_chunks()[:3])

    assert not result.passed
    assert "3 windows" in result.detail


def test_mono_output_is_caught() -> None:
    result = parity.check_waveform(parity.BASELINE_WAVEFORM_SAMPLES, 1)

    assert not result.passed
    assert "channels" in result.detail


def test_first_failure_names_the_earliest_stage() -> None:
    results = [
        parity.check_chunk_starts(parity.BASELINE_CHUNK_STARTS),
        parity.check_frame_hiddens(_Fake((1, 500, 4096), 0.94)),
        parity.check_latent_chunks(_reference_chunks()[:1]),
    ]

    failure = parity.first_failure(results)

    assert failure is not None
    assert failure.stage == "frame_hiddens"


def test_tolerances_are_loose_enough_for_bfloat16_but_not_meaningless() -> None:
    within = parity.BASELINE_FRAME_HIDDENS_STD + parity.STATISTIC_TOLERANCE * 0.9
    beyond = parity.BASELINE_FRAME_HIDDENS_STD + parity.STATISTIC_TOLERANCE * 1.5

    assert parity.check_frame_hiddens(
        _Fake(parity.BASELINE_FRAME_HIDDENS_SHAPE, within)
    ).passed
    assert not parity.check_frame_hiddens(
        _Fake(parity.BASELINE_FRAME_HIDDENS_SHAPE, beyond)
    ).passed


def test_cross_run_drift_is_recorded_and_above_quantisation() -> None:
    """The recorded waveform and latents are from two runs, not one.

    Decoding the recorded latents and stitching them reproduces the recorded
    sample count exactly and its samples only to about 1e-03 RMS. The reference
    vocoder produces the identical figure on the identical input, so the drift
    is between the two captures, not in any implementation.
    """

    sixteen_bit_step = 1.0 / 32768

    assert parity.CROSS_RUN_WAVEFORM_RMS > sixteen_bit_step * 10
    # Small enough that it is drift rather than a structural error.
    assert parity.CROSS_RUN_WAVEFORM_RMS < 1e-2


def test_waveform_check_asserts_only_what_the_arithmetic_pins() -> None:
    """Length and channels are exact; samples are not compared here."""

    assert parity.check_waveform(
        parity.BASELINE_WAVEFORM_SAMPLES, parity.BASELINE_WAVEFORM_CHANNELS
    ).passed
    assert not parity.check_waveform(parity.BASELINE_WAVEFORM_SAMPLES - 1, 2).passed

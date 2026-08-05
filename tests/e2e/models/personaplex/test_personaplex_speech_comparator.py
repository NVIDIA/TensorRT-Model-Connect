# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex reference-relative audio liveness contracts."""

from __future__ import annotations

import numpy as np

from tests.e2e.models.personaplex.e2e_plugins.comparators.speech_to_speech import (
    SpeechToSpeechComparator,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile


def _compare(*, trt_rms: float, ref_rms: float):
    tokens = np.zeros((4, 8), dtype=np.int32)
    return SpeechToSpeechComparator().compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "returncode": 0,
                "output_tokens": tokens,
                "rms": trt_rms,
                "wav_exists": True,
            },
        ),
        StageOutput(
            stage_name="full_generation",
            data={"reference_tokens": tokens, "rms": ref_rms},
        ),
        ThresholdProfile(
            task_strategy="speech_to_speech",
            metrics={"speech_min_rms": 0.001},
        ),
        StageSpec(name="full_generation"),
    )


def test_speech_rms_floor_accepts_reference_consistent_silence() -> None:
    result = _compare(trt_rms=0.00024, ref_rms=0.00023)

    assert result.metrics["rms_floor"].passed
    assert "reference is also below the floor" in result.metrics["rms_floor"].note


def test_speech_rms_floor_rejects_missing_reference_speech() -> None:
    result = _compare(trt_rms=0.00024, ref_rms=0.04)

    assert not result.metrics["rms_floor"].passed


def test_speech_comparator_rejects_truncated_output_with_matching_prefix() -> None:
    reference_tokens = np.arange(343 * 8, dtype=np.int32).reshape(343, 8)
    trt_tokens = reference_tokens[:51].copy()

    result = SpeechToSpeechComparator().compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "returncode": 0,
                "output_tokens": trt_tokens,
                "rms": 0.04,
                "wav_exists": True,
            },
        ),
        StageOutput(
            stage_name="full_generation",
            data={"reference_tokens": reference_tokens, "rms": 0.04},
        ),
        ThresholdProfile(
            task_strategy="speech_to_speech",
            metrics={
                "depth_token_match_rate": 0.7,
                "audio_token_match_rate": 0.7,
                "frame_exact_match_rate": 0.5,
                "speech_min_rms": 0.001,
            },
        ),
        StageSpec(name="full_generation"),
    )

    assert result.status == "failed"
    assert not result.metrics["frame_count_match"].passed
    assert result.metrics["depth_token_match_rate"].passed
    assert result.metrics["audio_token_match_rate"].passed
    assert result.metrics["frame_exact_match_rate"].passed


def _compare_pipeline_tokens(
    *,
    trt_input: np.ndarray,
    ref_input: np.ndarray,
    trt_text: np.ndarray,
    ref_text: np.ndarray,
    trt_audio: np.ndarray,
    ref_audio: np.ndarray,
):
    return SpeechToSpeechComparator().compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "returncode": 0,
                "input_codec_tokens": trt_input,
                "output_text_tokens": trt_text,
                "output_tokens": trt_audio,
                "rms": 0.04,
                "wav_exists": True,
            },
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "reference_input_codec_tokens": ref_input,
                "reference_text_tokens": ref_text,
                "reference_tokens": ref_audio,
                "rms": 0.04,
            },
        ),
        ThresholdProfile(
            task_strategy="speech_to_speech",
            metrics={
                "input_mimi_token_match_rate": 1.0,
                "output_text_token_match_rate": 1.0,
                "depth_token_match_rate": 0.7,
                "audio_token_match_rate": 0.7,
                "frame_exact_match_rate": 0.5,
                "speech_min_rms": 0.001,
            },
        ),
        StageSpec(name="full_generation"),
    )


def test_speech_comparator_reports_input_mimi_as_first_divergence() -> None:
    input_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)
    trt_input = input_tokens.copy()
    trt_input[2, 3] += 1
    text_tokens = np.arange(4, dtype=np.int32)
    audio_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)

    result = _compare_pipeline_tokens(
        trt_input=trt_input,
        ref_input=input_tokens,
        trt_text=text_tokens,
        ref_text=text_tokens,
        trt_audio=audio_tokens,
        ref_audio=audio_tokens,
    )

    assert result.status == "failed"
    assert not result.metrics["input_mimi_token_match_rate"].passed
    assert "first divergence=input_mimi frame=2" in result.message
    assert "first mismatch frame=2" in result.metrics["input_mimi_token_match_rate"].note


def test_speech_comparator_reports_temporal_text_as_first_divergence() -> None:
    input_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)
    text_tokens = np.arange(4, dtype=np.int32)
    trt_text = text_tokens.copy()
    trt_text[3] += 1
    audio_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)

    result = _compare_pipeline_tokens(
        trt_input=input_tokens,
        ref_input=input_tokens,
        trt_text=trt_text,
        ref_text=text_tokens,
        trt_audio=audio_tokens,
        ref_audio=audio_tokens,
    )

    assert result.status == "failed"
    assert not result.metrics["output_text_token_match_rate"].passed
    assert "first divergence=temporal_text frame=3" in result.message


def test_speech_comparator_reports_depth_audio_as_first_divergence() -> None:
    input_tokens = np.arange(40, dtype=np.int32).reshape(5, 8)
    text_tokens = np.arange(5, dtype=np.int32)
    audio_tokens = np.arange(40, dtype=np.int32).reshape(5, 8)
    trt_audio = audio_tokens.copy()
    trt_audio[4, 6] += 1

    result = _compare_pipeline_tokens(
        trt_input=input_tokens,
        ref_input=input_tokens,
        trt_text=text_tokens,
        ref_text=text_tokens,
        trt_audio=trt_audio,
        ref_audio=audio_tokens,
    )

    assert result.status == "passed"
    assert "first divergence=depth_audio frame=4" in result.message
    assert "first mismatch frame=4" in result.metrics["frame_exact_match_rate"].note


def test_speech_comparator_reports_no_divergence_for_exact_pipeline() -> None:
    input_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)
    text_tokens = np.arange(4, dtype=np.int32)
    audio_tokens = np.arange(32, dtype=np.int32).reshape(4, 8)

    result = _compare_pipeline_tokens(
        trt_input=input_tokens,
        ref_input=input_tokens,
        trt_text=text_tokens,
        ref_text=text_tokens,
        trt_audio=audio_tokens,
        ref_audio=audio_tokens,
    )

    assert result.status == "passed"
    assert "first divergence=none" in result.message

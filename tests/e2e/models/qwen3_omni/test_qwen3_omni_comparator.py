# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni-owned comparator tests."""

from __future__ import annotations

from tests.e2e.models.qwen3_omni.e2e_plugins.comparators.omni import OmniComparator
from tests.e2e_harness.contracts import (
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _invariant_ref(stage_name: str) -> StageOutput:
    return StageOutput(
        stage_name=stage_name,
        data={"_invariant_only": True},
        metadata={"source": "invariant_only"},
    )


def test_omni_invariant_talker_requires_non_empty_audio(tmp_path) -> None:
    audio = tmp_path / "talker.wav"
    audio.write_bytes(b"RIFFaudio")

    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(audio)},
        ),
        _invariant_ref("talker_decode"),
        ThresholdProfile(task_strategy="omni_multimodal"),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["audio_artifact_bytes"].passed is True


def test_omni_invariant_talker_fails_without_audio(tmp_path) -> None:
    result = OmniComparator().compare(
        StageOutput(
            stage_name="talker_decode",
            metadata={"audio_output_path": str(tmp_path / "missing.wav")},
        ),
        _invariant_ref("talker_decode"),
        ThresholdProfile(task_strategy="omni_multimodal"),
        StageSpec(name="talker_decode"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["audio_artifact_bytes"].passed is False


def test_omni_invariant_text_stage_requires_non_empty_output() -> None:
    result = OmniComparator().compare(
        StageOutput(stage_name="end_to_end", text="hello"),
        _invariant_ref("end_to_end"),
        ThresholdProfile(task_strategy="omni_multimodal"),
        StageSpec(name="end_to_end"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["non_empty_text"].passed is True

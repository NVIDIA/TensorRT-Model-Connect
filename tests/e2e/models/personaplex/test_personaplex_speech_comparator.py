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

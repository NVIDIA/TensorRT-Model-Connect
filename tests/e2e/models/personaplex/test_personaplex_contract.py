# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e.models.personaplex.e2e_plugins.contract import (
    PersonaPlexSpeechToSpeechPlugin,
)


def _thresholds() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="speech_to_speech",
        metrics={
            "contract_token_match": 0.5,
            "contract_min_rms": 0.001,
        },
    )


def _case() -> E2ECase:
    return E2ECase(
        name="personaplex-contract",
        hf_id="nvidia/personaplex-7b-v1",
        family="personaplex",
        runtime_strategy="personaplex_speech_to_speech",
        task_strategy="speech_to_speech",
    )


def test_contract_compares_runner_tokens_with_official_reference_tokens() -> None:
    tokens = np.arange(24, dtype=np.int32).reshape(3, 8)
    result = PersonaPlexSpeechToSpeechPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"wav_path": "/tmp/output.wav", "rms": 0.01, "output_tokens": tokens},
        ),
        StageOutput(
            stage_name="full_generation",
            data={"reference_tokens": tokens.copy()},
        ),
        _case(),
        _thresholds(),
    )

    assert result.passed
    assert result.metrics["token_match"].value == 1.0


def test_contract_preserves_original_half_token_match_threshold() -> None:
    reference = np.arange(24, dtype=np.int32).reshape(3, 8)
    actual = reference.copy()
    actual.reshape(-1)[14:] = -1
    result = PersonaPlexSpeechToSpeechPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={
                "wav_path": "/tmp/output.wav",
                "rms": 0.01,
                "output_tokens": actual,
            },
        ),
        StageOutput(
            stage_name="full_generation",
            data={"reference_tokens": reference},
        ),
        _case(),
        _thresholds(),
    )

    assert result.passed
    assert result.metrics["token_match"].value == 14 / 24
    assert result.metrics["token_match"].threshold == 0.5


def test_contract_rejects_missing_reference_tokens() -> None:
    result = PersonaPlexSpeechToSpeechPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"wav_path": "/tmp/output.wav", "rms": 0.01},
        ),
        StageOutput(stage_name="full_generation", data={}),
        _case(),
        _thresholds(),
    )

    assert not result.passed
    assert not result.metrics["reference_tokens_available"].passed

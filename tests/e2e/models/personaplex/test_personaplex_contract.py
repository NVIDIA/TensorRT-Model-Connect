# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e.models.personaplex.e2e_plugins.contract import (
    PersonaPlexSpeechToSpeechPlugin,
)


_MANIFEST_DIR = Path(__file__).parent / "manifests"
_THRESHOLD_PATH = Path(__file__).parent / "thresholds" / "personaplex-7b.json"


@pytest.mark.parametrize(
    "manifest_name",
    (
        "personaplex-7b.json",
        "personaplex-7b-l0.json",
        "personaplex-7b-l0-tp4.json",
    ),
)
def test_personaplex_builds_require_an_exclusive_gpu(manifest_name: str) -> None:
    manifest = json.loads((_MANIFEST_DIR / manifest_name).read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"


@pytest.mark.parametrize(
    "manifest_name",
    ("personaplex-7b.json", "personaplex-7b-l0.json"),
)
def test_personaplex_uses_official_component_precisions(
    manifest_name: str,
) -> None:
    manifest = json.loads((_MANIFEST_DIR / manifest_name).read_text(encoding="utf-8"))

    assert manifest["precision"] == "bf16"
    assert manifest["fp32_layers"] == [2, 3]


def test_personaplex_full_reference_matches_bundle_precision() -> None:
    manifest = json.loads(
        (_MANIFEST_DIR / "personaplex-7b.json").read_text(encoding="utf-8")
    )

    testcase = manifest["testcases"][0]
    assert manifest["precision"] == "bf16"
    assert testcase["reference_precision"] == manifest["precision"]


def _thresholds() -> ThresholdProfile:
    threshold_data = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))
    return ThresholdProfile(
        task_strategy="speech_to_speech",
        metrics=threshold_data["threshold_overrides"],
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
    assert result.metrics["free_generation_token_match_rate"].value == 1.0
    assert result.metrics["free_generation_token_match_rate"].threshold is None
    assert result.metrics["free_generation_depth_token_match_rate"].passed
    assert result.metrics["free_generation_audio_token_match_rate"].passed
    assert result.metrics["free_generation_frame_exact_match_rate"].threshold is None
    assert result.metrics["rms"].threshold == 0.001


def test_contract_reports_free_generation_divergence_without_failing() -> None:
    reference = np.zeros((25, 8), dtype=np.int32)
    actual = reference.copy()
    actual.reshape(-1)[149:] = 1
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
    metric = result.metrics["free_generation_token_match_rate"]
    assert metric.value == pytest.approx(0.745)
    assert metric.threshold is None
    assert metric.passed
    assert result.metrics["free_generation_depth_token_match_rate"].passed
    assert result.metrics["free_generation_audio_token_match_rate"].passed
    assert result.metrics["free_generation_frame_exact_match_rate"].passed


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


def test_contract_rejects_extra_runtime_frames() -> None:
    reference = np.arange(24, dtype=np.int32).reshape(3, 8)
    actual = np.concatenate([reference, reference[-1:]], axis=0)
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

    assert not result.passed
    assert not result.metrics["frame_count_match"].passed
    assert result.metrics["free_generation_token_match_rate"].passed

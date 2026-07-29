# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict generated-token parity for the qualified Mistral model."""

from types import SimpleNamespace

from tests.e2e_harness.contracts import StageOutput, StageStatus, ThresholdProfile
from tests.e2e.models.mistral.e2e_plugins.contract import MistralTranslationPlugin


def _output(token_ids: list[int], text: str = "same decoded translation") -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={"token_ids": token_ids},
        text=text,
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        metrics={
            "contract_ned_threshold": 0.15,
            "contract_token_agreement_rate": 1.0,
        },
    )


def _verify(trt_tokens: list[int], ref_tokens: list[int]):
    return MistralTranslationPlugin().verify(
        _output(trt_tokens),
        _output(ref_tokens),
        SimpleNamespace(inputs={"prompt": ""}, metadata={}),
        _threshold(),
    )


def test_translation_contract_rejects_token_divergence_with_identical_text():
    result = _verify([1, 2, 3], [1, 2, 4])

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["exact_match"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_translation_contract_accepts_exact_generated_tokens():
    result = _verify([1, 2, 3], [1, 2, 3])

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_exact"].passed


def test_translation_contract_rejects_text_divergence_with_identical_tokens():
    result = MistralTranslationPlugin().verify(
        _output([1, 2, 3], "same decoded translation"),
        _output([1, 2, 3], "same decoded translation!"),
        SimpleNamespace(inputs={"prompt": ""}, metadata={}),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert result.metrics["generated_token_exact"].passed

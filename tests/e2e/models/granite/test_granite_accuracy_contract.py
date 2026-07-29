# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict generated-token parity for the qualified Granite model."""

from types import SimpleNamespace

from tests.e2e_harness.contracts import StageOutput, StageStatus, ThresholdProfile
from tests.e2e.models.granite.e2e_plugins.contract import GraniteCausalContinuationPlugin


def _output(token_ids: list[int], text: str = "same decoded continuation") -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={"cpp_returncode": 0, "token_ids": token_ids},
        text=text,
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        metrics={
            "contract_ned_threshold": 0.25,
            "contract_token_agreement_rate": 1.0,
        },
    )


def _verify(trt_tokens: list[int], ref_tokens: list[int]):
    return GraniteCausalContinuationPlugin().verify(
        _output(trt_tokens),
        _output(ref_tokens),
        SimpleNamespace(inputs={"prompt": ""}, metadata={}),
        _threshold(),
    )


def test_continuation_contract_rejects_token_divergence_with_identical_text():
    result = _verify([1, 2, 3], [1, 2, 4])

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_continuation_contract_accepts_exact_generated_tokens():
    result = _verify([1, 2, 3], [1, 2, 3])

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_exact"].passed


def test_continuation_contract_rejects_text_divergence_with_identical_tokens():
    result = GraniteCausalContinuationPlugin().verify(
        _output([1, 2, 3], "same decoded continuation"),
        _output([1, 2, 3], "same decoded continuation!"),
        SimpleNamespace(inputs={"prompt": ""}, metadata={}),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert result.metrics["generated_token_exact"].passed

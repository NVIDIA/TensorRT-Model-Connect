"""Tests for continuation contract edge cases."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e_harness.plugins.causal_continuation import CausalContinuationPlugin


def _case(reference_family: str = "causal_base_continuation") -> E2ECase:
    return E2ECase(
        name="model",
        hf_id="hf/model",
        family="family",
        runtime_strategy="decoder_kv_cache",
        reference_family=reference_family,
        inputs={"prompt": "The capital of France is"},
    )


def test_continuation_rejects_empty_trt_and_reference_outputs() -> None:
    result = CausalContinuationPlugin().verify(
        StageOutput(stage_name="full_generation", text=""),
        StageOutput(stage_name="full_generation", text=""),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["non_empty_continuation"].passed


def test_seq2seq_weak_reference_prompt_text_is_not_stripped_to_empty() -> None:
    result = CausalContinuationPlugin().verify(
        StageOutput(stage_name="full_generation", text=""),
        StageOutput(stage_name="full_generation", text="The capital of France is"),
        _case(reference_family="seq2seq_base_weak"),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].value == 1.0

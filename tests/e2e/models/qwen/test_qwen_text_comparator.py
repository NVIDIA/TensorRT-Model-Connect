"""Qwen-owned text comparator behavior."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile
from tests.e2e.models.qwen.e2e_plugins.comparators.text import TextComparator


def _thresholds() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        profile_name="qwen-test",
        metrics={
            "logit_cosine_p5": 0.99,
            "logit_rel_l2_p95": 0.05,
            "stable_top1_match_rate": 0.9,
            "unstable_topk_hit_rate": 0.8,
            "token_agreement_rate": 0.8,
            "normalized_text_edit_distance": 0.2,
        },
    )


def _matching_logits() -> np.ndarray:
    return np.array(
        [
            [0.0, 3.0, -1.0],
            [0.0, 4.0, -2.0],
            [0.0, 5.0, -3.0],
        ],
        dtype=np.float32,
    )


def _compare(trt: StageOutput, ref: StageOutput):
    return TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_thresholds(),
        stage=StageSpec(name="full_generation"),
    )


def test_expected_answer_gate_accepts_fp8_surface_text_drift() -> None:
    logits = _matching_logits()
    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "prompt": "What is the capital of France? Answer in one word.",
                "expected_answers": ["Paris"],
            },
            text='The answer is "Paris."\nAnswer:\nParis\n\n',
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text="The capital of France is Paris.\nAnswer:\nParis",
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert not result.metrics["normalized_text_edit_distance"].passed
    assert result.metrics["expected_answer_present"].passed
    assert "expected_answer_present" in result.composite_rule


def test_expected_answer_gate_rejects_missing_answer() -> None:
    logits = _matching_logits()
    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "prompt": "What is the capital of France? Answer in one word.",
                "expected_answers": ["Paris"],
            },
            text="The answer is Lyon.",
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text="The capital of France is Paris.",
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["expected_answer_present"].passed


def test_expected_answer_gate_still_requires_token_or_stable_top1_agreement() -> None:
    trt_logits = np.array(
        [
            [3.0, 0.0, -1.0],
            [4.0, 0.0, -2.0],
            [5.0, 0.0, -3.0],
        ],
        dtype=np.float32,
    )
    ref_logits = _matching_logits()
    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={
                "prompt": "What is the capital of France? Answer in one word.",
                "expected_answers": ["Paris"],
            },
            text='The answer is "Paris."',
            logits=trt_logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text="The capital of France is Paris.",
            logits=ref_logits,
        ),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["expected_answer_present"].passed
    assert not result.metrics["token_agreement_rate"].passed

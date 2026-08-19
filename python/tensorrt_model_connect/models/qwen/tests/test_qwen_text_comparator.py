# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned text comparator behavior."""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile
from tensorrt_model_connect.models.qwen.tests.e2e_plugins.comparators.text import TextComparator


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


def test_warning_preamble_before_prompt_does_not_fail_ned() -> None:
    prompt = "Once upon a time there was a little"
    continuation = (
        "girl named Lucy. She was three years old and she was very excited "
        "to go on an adventure."
    )
    warning = (
        "You are using the default legacy behaviour of the "
        "<class 'transformers.models.example.tokenization_example_fast.ExampleTokenizerFast'>."
    )
    logits = _matching_logits()

    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={"prompt": prompt},
            text=f"{warning}\n{prompt} {continuation}",
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text=continuation,
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["normalized_text_edit_distance"].passed


def test_prefix_only_truncation_with_matching_tokens_does_not_hard_fail_ned() -> None:
    prompt = "The capital of France is"
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
            [0.0, 4.0, -4.0],
        ],
        dtype=np.float32,
    )

    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={"prompt": prompt},
            text="The capital of France is the capital of the country.",
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text=(
                "the capital of the country. "
                "The capital of the country is Paris. "
                "The capital of the country"
            ),
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed


def test_normalized_prompt_echo_stripping_handles_punctuation_spacing_drift() -> None:
    prompt = "The quick brown fox jumps over the lazy dog. Once upon a time"
    logits = _matching_logits()

    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={"prompt": prompt},
            text=(
                "The quick brown fox jumps over the lazy dog.Once upon a time "
                "there was a curious fox in the woods."
            ),
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text="there was a curious fox in the woods.",
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed


def test_nonzero_cpp_returncode_is_reported_as_error() -> None:
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
        ],
        dtype=np.float32,
    )

    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={"cpp_returncode": -6},
            text="",
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text="reference text",
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.ERROR.value
    assert "cpp_returncode=-6" in result.message


def test_reference_prompt_phrase_in_generated_text_is_not_stripped() -> None:
    prompt = "The capital of France is"
    logits = _matching_logits()

    result = _compare(
        StageOutput(
            stage_name="full_generation",
            data={"prompt": prompt},
            text=(
                "The capital of France is Paris.\n"
                "The capital of France is Paris.\n"
                "The capital of France is Paris.\n"
                "The"
            ),
            logits=logits,
        ),
        StageOutput(
            stage_name="full_generation",
            text=(
                " Paris.\n"
                "The capital of France is Paris.\n"
                "The capital of France is Paris.\n"
                "The"
            ),
            logits=logits.copy(),
        ),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed

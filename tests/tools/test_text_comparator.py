"""Unit tests for text_generation comparator normalization behavior.

Trace: ARCH-E2E-001, UD-E2E-TEXT-COMPARATOR
Intent: Validate TextComparator warning preamble normalization and NED metric computation
Preconditions: Synthetic TRT and reference stage outputs with logits and text are available
Postconditions: Warning preambles do not inflate NED scores and identical logits produce passing metrics
"""

from __future__ import annotations

import numpy as np

from tests.e2e_harness.comparators.text import TextComparator
from tests.e2e_harness.contracts import StageOutput, StageSpec, StageStatus, ThresholdProfile


def _default_thresholds() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        profile_name="test",
        metrics={
            "logit_cosine_p5": 0.99,
            "logit_rel_l2_p95": 0.05,
            "stable_top1_match_rate": 0.9,
            "unstable_topk_hit_rate": 0.8,
            "token_agreement_rate": 0.8,
            "normalized_text_edit_distance": 0.2,
        },
    )


def test_warning_preamble_before_prompt_does_not_fail_ned() -> None:
    prompt = "Once upon a time there was a little"
    continuation = (
        "girl named Lucy. She was three years old and she was very excited "
        "to go on an adventure."
    )
    warning = (
        "You are using the default legacy behaviour of the "
        "<class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>."
    )
    trt_text = f"{warning}\n{prompt} {continuation}"
    ref_text = continuation

    # Identical logits: all token/logit metrics should pass.
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert "normalized_text_edit_distance" in result.metrics
    assert result.metrics["normalized_text_edit_distance"].passed


def test_prefix_only_truncation_with_matching_tokens_does_not_hard_fail_ned() -> None:
    """If one side is an early-stopped prefix, NED should not hard-fail."""
    prompt = "The capital of France is"
    trt_text = "The capital of France is the capital of the country."
    ref_text = (
        "the capital of the country. "
        "The capital of the country is Paris. "
        "The capital of the country"
    )

    # Identical logits -> token-level agreement is perfect.
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
            [0.0, 4.0, -4.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed


def test_normalized_prompt_echo_stripping_handles_punctuation_spacing_drift() -> None:
    """Prompt stripping should survive minor decode formatting drift."""
    prompt = "The quick brown fox jumps over the lazy dog. Once upon a time"
    # TRT decoded text carries the full prompt, but tokenizer formatting differs:
    # missing space after period means raw substring prompt match fails.
    trt_text = (
        "The quick brown fox jumps over the lazy dog.Once upon a time "
        "there was a curious fox in the woods."
    )
    ref_text = "there was a curious fox in the woods."

    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed


def test_nonzero_cpp_returncode_is_reported_as_error() -> None:
    """Comparator must fail when C++ full-generation path failed."""
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"cpp_returncode": -6},
        text="",
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text="reference text",
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.ERROR.value
    assert "cpp_returncode=-6" in result.message


def test_reference_prompt_phrase_in_generated_text_is_not_stripped() -> None:
    """Do not strip prompt from HF text when phrase appears later naturally."""
    prompt = "The capital of France is"
    trt_text = (
        "The capital of France is Paris.\n"
        "The capital of France is Paris.\n"
        "The capital of France is Paris.\n"
        "The"
    )
    ref_text = (
        " Paris.\n"
        "The capital of France is Paris.\n"
        "The capital of France is Paris.\n"
        "The"
    )

    # Identical logits: token/logit metrics should pass.
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": prompt},
        text=trt_text,
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text=ref_text,
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["token_agreement_rate"].passed
    assert result.metrics["normalized_text_edit_distance"].passed


def test_logit_parity_mode_does_not_gate_on_decoded_text() -> None:
    """Tiny random model text can be meaningless while logits still define parity."""
    logits = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 2.0, -2.0],
            [0.0, 3.0, -3.0],
        ],
        dtype=np.float32,
    )

    trt = StageOutput(
        stage_name="full_generation",
        data={"prompt": "Question?"},
        text="Question? ErrorMessage ErrorMessage",
        logits=logits,
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={},
        text="unrelated random continuation",
        logits=logits.copy(),
    )

    result = TextComparator().compare(
        trt=trt,
        ref=ref,
        threshold=_default_thresholds(),
        stage=StageSpec(name="full_generation", comparison_mode="logit_parity"),
    )

    assert result.status == StageStatus.PASSED.value
    assert not result.metrics["normalized_text_edit_distance"].passed
    assert "not gated" in result.composite_rule

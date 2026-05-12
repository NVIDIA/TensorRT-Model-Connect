"""Tests for VL QA contract comparison helpers."""

from __future__ import annotations

from tests.e2e_harness.contracts import (
    E2ECase,
    OracleLevel,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e_harness.plugins.vl_qa import VLQAPlugin


def _case(
    reference_family: str = "vl_instruct_qa",
    reference_backend: str = "hf_transformers",
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value,
) -> E2ECase:
    return E2ECase(
        name="qwen3-vl-2b",
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        family="qwen_vl",
        runtime_strategy="vision_language",
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        reference_family=reference_family,
        inputs={"prompt": "What color is the vehicle in this image? Answer in one word."},
    )


def test_vl_qa_ignores_terminal_punctuation_for_short_answers() -> None:
    result = VLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "White"}),
        StageOutput(stage_name="full_generation", data={"text": " White."}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed
    assert "raw_answer_ned" in result.metrics


def test_vl_qa_empty_l1_reference_fails_instead_of_has_output_pass() -> None:
    result = VLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "White"}),
        StageOutput(stage_name="full_generation", data={"text": ""}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["has_output"].passed


def test_vl_qa_invariant_only_reference_is_skipped_not_green() -> None:
    result = VLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "Invoice"}),
        StageOutput(
            stage_name="full_generation",
            data={"_invariant_only": True},
            metadata={"source": "invariant_only"},
        ),
        _case(
            reference_family="ocr_markdown",
            reference_backend="invariant_only",
            oracle_level=OracleLevel.L4_INVARIANTS.value,
        ),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.SKIPPED.value


def test_vl_qa_vision_encode_nonzero_returncode_fails() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="vision_encode",
            data={"passed": False, "metrics": {"vision_pass": True}},
            metadata={"returncode": 1},
        ),
        StageOutput(stage_name="vision_encode", data={}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["vision_encode_ok"].passed


def test_vl_qa_full_generation_nonzero_returncode_fails() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "White"},
            metadata={"returncode": 1},
        ),
        StageOutput(stage_name="full_generation", data={"text": "White"}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "return code 1" in result.message


def test_vl_qa_ocr_required_substrings_are_real_contract() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={
                "generated_text": (
                    "DeepSeek-OCR-2 family plugin\n"
                    "Architecture:\n"
                    "- Vision: SAM ViT-B + Qwen2 encoder"
                )
            },
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "required_substrings": [
                    "DeepSeek-OCR-2 family plugin",
                    "Architecture:",
                    "Vision: SAM ViT-B + Qwen2 encoder",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["required_ocr_substrings"].passed


def test_vl_qa_ocr_required_substrings_fail_when_missing() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "DeepSeek-OCR-2 family plugin"},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "required_substrings": [
                    "DeepSeek-OCR-2 family plugin",
                    "Vision: SAM ViT-B + Qwen2 encoder",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "Vision: SAM ViT-B + Qwen2 encoder" in result.message


def test_vl_qa_accepts_single_word_answer_inside_reference_sentence() -> None:
    result = VLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "White"}),
        StageOutput(
            stage_name="full_generation",
            data={"text": "The vehicle in this image is white."},
        ),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed


def test_vl_qa_preserves_ocr_punctuation() -> None:
    result = VLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "Invoice"}),
        StageOutput(stage_name="full_generation", data={"text": "Invoice:"}),
        _case(reference_family="ocr_markdown"),
        ThresholdProfile(task_strategy="vision_language_generation", metrics={"contract_ned_threshold": 0.05}),
    )

    assert result.status == StageStatus.FAILED.value

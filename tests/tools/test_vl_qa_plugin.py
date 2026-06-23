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
    contract_config = {"use_processor": True, "use_chat_template": True}
    if reference_family == "ocr_markdown":
        contract_config["ocr_mode"] = True
    return E2ECase(
        name="example-vl-2b",
        hf_id="example-org/example-vl",
        family="example_vl",
        runtime_strategy="vision_language",
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        reference_family=reference_family,
        inputs={"prompt": "What color is the vehicle in this image? Answer in one word."},
        metadata={"contract_config": contract_config},
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
    ref_text = (
        "OCR-2 is a VL model with a latent-attention language decoder. "
        "Unlike the larger decoder which uses Multi-head Latent Attention (MLA), "
        "OCR-2 uses standard multi-head attention."
    )
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={
                "generated_text": ref_text
            },
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "text": ref_text,
                "required_substrings": [
                    "OCR-2 is a VL model",
                    "Multi-head Latent Attention (MLA)",
                    "OCR-2 uses standard multi-head attention",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["reference_contract_substrings"].passed
    assert result.metrics["required_ocr_substrings"].passed


def test_vl_qa_ocr_required_substrings_need_visible_reference() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "OCR sample family plugin"},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "required_substrings": [
                    "OCR sample family plugin",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "human-readable reference text" in result.message


def test_vl_qa_ocr_required_substrings_must_be_visible_in_reference() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={
                "generated_text": (
                    "OCR sample family plugin. Vision: region encoder + example adapter."
                )
            },
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "text": "OCR sample family plugin.",
                "required_substrings": [
                    "OCR sample family plugin",
                    "Vision: region encoder + example adapter",
                ],
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "hides required text from the report" in result.message


def test_vl_qa_ocr_required_substrings_fail_when_missing() -> None:
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "OCR sample family plugin"},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "text": (
                    "OCR sample family plugin.\n"
                    "Vision: region encoder + example adapter."
                ),
                "required_substrings": [
                    "OCR sample family plugin",
                    "Vision: region encoder + example adapter",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "Vision: region encoder + example adapter" in result.message


def test_vl_qa_ocr_rejects_missing_contracted_architecture_output() -> None:
    ref_text = (
        "Architecture:\n"
        "- Attention: Standard Q/K/V/O.\n"
        "- Vision: region encoder + example adapter."
    )
    result = VLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={
                "generated_text": (
                    "Architecture:\n\n"
                    "Attention:Standard Q/K/V/O (no biases,no GQA-heads == kv heads)\n"
                    "RoPE:Standard rotary position embeddings\n"
                )
            },
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "text": ref_text,
                "required_substrings": [
                    "Architecture",
                    "Attention: Standard Q/K/V/O",
                    "Vision: region encoder + example adapter",
                ]
            },
        ),
        _case(reference_family="ocr_markdown", reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "Vision: region encoder + example adapter" in result.message


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

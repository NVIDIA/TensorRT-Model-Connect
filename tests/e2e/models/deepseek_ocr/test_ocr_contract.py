# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR-owned OCR contract checks."""

from __future__ import annotations

from tests.e2e.models.deepseek_ocr.e2e_plugins.contract import DeepseekOcrVLQAPlugin
from tests.e2e_harness.contracts import (
    E2ECase,
    OracleLevel,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)


def _case(
    reference_backend: str = "hf_transformers",
    oracle_level: str = OracleLevel.L1_EXTERNAL_REFERENCE.value,
) -> E2ECase:
    return E2ECase(
        name="example-ocr",
        hf_id="example-org/example-ocr",
        family="deepseek_ocr",
        runtime_strategy="deepseek_ocr_vision_language",
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        reference_family="ocr_markdown",
        inputs={"prompt": "What color is the vehicle in this image? Answer in one word."},
        metadata={
            "contract_config": {
                "use_processor": True,
                "use_chat_template": True,
                "ocr_mode": True,
            }
        },
    )


def test_vl_qa_invariant_only_reference_is_skipped_not_green() -> None:
    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "Invoice"}),
        StageOutput(
            stage_name="full_generation",
            data={"_invariant_only": True},
            metadata={"source": "invariant_only"},
        ),
        _case(
            reference_backend="invariant_only",
            oracle_level=OracleLevel.L4_INVARIANTS.value,
        ),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.SKIPPED.value


def test_vl_qa_ocr_required_substrings_are_real_contract() -> None:
    ref_text = (
        "OCR-2 is a VL model with a latent-attention language decoder. "
        "Unlike the larger decoder which uses Multi-head Latent Attention (MLA), "
        "OCR-2 uses standard multi-head attention."
    )
    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": ref_text},
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
                ],
            },
        ),
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["reference_contract_substrings"].passed
    assert result.metrics["required_ocr_substrings"].passed


def test_vl_qa_ocr_required_substrings_need_visible_reference() -> None:
    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "OCR sample family plugin"},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={"required_substrings": ["OCR sample family plugin"]},
        ),
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "human-readable reference text" in result.message


def test_vl_qa_ocr_required_substrings_must_be_visible_in_reference() -> None:
    result = DeepseekOcrVLQAPlugin().verify(
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
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "hides required text from the report" in result.message


def test_vl_qa_ocr_required_substrings_fail_when_missing() -> None:
    result = DeepseekOcrVLQAPlugin().verify(
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
                ],
            },
        ),
        _case(reference_backend="golden_snapshot"),
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
    result = DeepseekOcrVLQAPlugin().verify(
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
                ],
            },
        ),
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "Vision: region encoder + example adapter" in result.message


def test_vl_qa_ocr_accepts_literal_fixture_spacing_and_terminal_norm() -> None:
    """Regression for the L0 output observed on GB300 in premerge CI."""
    required = [
        "Architecture:",
        "Attention: Standard Q/K/V/O",
        "RoPE: Standard rotary position embeddings",
        "Layer 0: Dense SwiGLU MLP",
        "Layers 1-11: MoE",
        "Norm: RMS",
    ]
    reference = (
        "Architecture:\n"
        "Attention: Standard Q/K/V/O (no biases, no GQA -- heads == kv_heads)\n"
        "RoPE: Standard rotary position embeddings\n"
        "Layer 0: Dense SwiGLU MLP (intermediate_size=6848)\n"
        "Layers 1-11: MoE (64 experts, top-6, intermediate=896)\n"
        "Norm: RMSNorm"
    )
    generated = (
        "Architecture:\n"
        "Attention:Standard Q/K/V/O (no biases,no GQA-heads == kv heads)\n"
        "RoPE:Standard rotary position embeddings\n"
        "Layer 0: Dense SwiGLU MLP (intermediate_size=6848)\n"
        "Layers 1-11:MoE 64 experts,top-6,intermediate=896)\n"
        "Norm:RMSNorm"
    )

    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": generated},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={"text": reference, "required_substrings": required},
        ),
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["required_ocr_substrings"].value == len(required)


def test_vl_qa_ocr_still_rejects_truncation_before_terminal_norm() -> None:
    required = ["Architecture:", "Norm: RMS"]
    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            data={"generated_text": "Architecture:\nNorm:"},
            metadata={"returncode": 0},
        ),
        StageOutput(
            stage_name="full_generation",
            data={
                "text": "Architecture:\nNorm: RMSNorm",
                "required_substrings": required,
            },
        ),
        _case(reference_backend="golden_snapshot"),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert "Norm: RMS" in result.message


def test_vl_qa_preserves_ocr_punctuation() -> None:
    result = DeepseekOcrVLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "Invoice"}),
        StageOutput(stage_name="full_generation", data={"text": "Invoice:"}),
        _case(),
        ThresholdProfile(
            task_strategy="vision_language_generation",
            metrics={"contract_ned_threshold": 0.05},
        ),
    )

    assert result.status == StageStatus.FAILED.value

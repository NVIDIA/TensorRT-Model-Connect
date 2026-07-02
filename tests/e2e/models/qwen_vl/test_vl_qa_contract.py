# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned VL QA contract checks."""

from __future__ import annotations

from tests.e2e.models.qwen_vl.e2e_plugins.contract import QwenVlVLQAPlugin
from tests.e2e_harness.contracts import (
    E2ECase,
    OracleLevel,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)


def _case() -> E2ECase:
    return E2ECase(
        name="example-vl-2b",
        hf_id="example-org/example-vl",
        family="qwen_vl",
        runtime_strategy="qwen_vl_vision_language",
        reference_backend="hf_transformers",
        oracle_level=OracleLevel.L1_EXTERNAL_REFERENCE.value,
        reference_family="vl_instruct_qa",
        inputs={"prompt": "What color is the vehicle in this image? Answer in one word."},
        metadata={"contract_config": {"use_processor": True, "use_chat_template": True}},
    )


def test_vl_qa_ignores_terminal_punctuation_for_short_answers() -> None:
    result = QwenVlVLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "White"}),
        StageOutput(stage_name="full_generation", data={"text": " White."}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed
    assert "raw_answer_ned" in result.metrics


def test_vl_qa_empty_l1_reference_fails_instead_of_has_output_pass() -> None:
    result = QwenVlVLQAPlugin().verify(
        StageOutput(stage_name="full_generation", data={"generated_text": "White"}),
        StageOutput(stage_name="full_generation", data={"text": ""}),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["has_output"].passed


def test_vl_qa_vision_encode_nonzero_returncode_fails() -> None:
    result = QwenVlVLQAPlugin().verify(
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
    result = QwenVlVLQAPlugin().verify(
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


def test_vl_qa_accepts_single_word_answer_inside_reference_sentence() -> None:
    result = QwenVlVLQAPlugin().verify(
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chat/instruct contract edge cases."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e.models.qwen.e2e_plugins.contract import QwenPostTrainedChatPlugin


def _case(*, native_kv_runtime_contract: bool = False) -> E2ECase:
    metadata = {
        "contract_config": {
            "use_chat_template": True,
            "enable_thinking": False,
        }
    }
    if native_kv_runtime_contract:
        metadata.update(
            expected_prefill_chunks=2,
            expected_prefill_chunk_limit=64,
            expected_kv_cache_rows=40960,
        )
    return E2ECase(
        name="model",
        hf_id="hf/model",
        family="family",
        runtime_strategy="qwen_decoder_kv_cache",
        reference_family="chat_instruct_template",
        user_contract="chat_response",
        inputs={"prompt": "What is the capital of France? Answer in one word."},
        metadata=metadata,
    )


def test_chat_instruct_rejects_thinking_block_when_disabled() -> None:
    result = QwenPostTrainedChatPlugin().verify(
        StageOutput(stage_name="full_generation", text="<think>\nOkay, the user is"),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["thinking_suppressed"].value == 0.0
    assert "thinking disabled" in result.message


def test_chat_instruct_accepts_golden_answer_without_thinking() -> None:
    result = QwenPostTrainedChatPlugin().verify(
        StageOutput(stage_name="full_generation", text="Paris"),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed


def test_chat_instruct_accepts_matching_native_kv_runtime_markers() -> None:
    output = StageOutput(
        stage_name="full_generation",
        text="Paris",
        metadata={
            "cpp": {
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=40960 "
                        "(bundle max=40960, row=1 B)",
                        '[trtmc.engine_timing] '
                        'label="prefill_engine_plan:prefill" '
                        "execute_ms=1 launches=2",
                    ]
                )
            }
        },
    )
    result = QwenPostTrainedChatPlugin().verify(
        output,
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(native_kv_runtime_contract=True),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["prefill_chunks"].passed
    assert result.metrics["native_kv_capacity"].passed


def test_chat_instruct_rejects_missing_native_kv_runtime_markers() -> None:
    result = QwenPostTrainedChatPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            text="Paris",
            metadata={"cpp": {"stderr": ""}},
        ),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(native_kv_runtime_contract=True),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["prefill_chunks"].passed
    assert not result.metrics["native_kv_capacity"].passed
    assert "runtime markers diverged" in result.message


def test_chat_instruct_requires_exact_prefill_launch_count() -> None:
    result = QwenPostTrainedChatPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            text="Paris",
            metadata={
                "cpp": {
                    "stderr": "\n".join(
                        [
                            "[trtmc] KV cache rows=40960 "
                            "(bundle max=40960, row=1 B)",
                            '[trtmc.engine_timing] '
                            'label="prefill_engine_plan:prefill" '
                            "execute_ms=1 launches=20",
                        ]
                    )
                }
            },
        ),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(native_kv_runtime_contract=True),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["prefill_chunks"].passed
    assert result.metrics["native_kv_capacity"].passed

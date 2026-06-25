"""Tests for chat/instruct contract edge cases."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e.models.qwen.e2e_plugins.contract import QwenPostTrainedChatPlugin


def _case() -> E2ECase:
    return E2ECase(
        name="model",
        hf_id="hf/model",
        family="family",
        runtime_strategy="qwen_decoder_kv_cache",
        reference_family="chat_instruct_template",
        user_contract="chat_response",
        inputs={"prompt": "What is the capital of France? Answer in one word."},
        metadata={"contract_config": {"use_chat_template": True, "enable_thinking": False}},
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

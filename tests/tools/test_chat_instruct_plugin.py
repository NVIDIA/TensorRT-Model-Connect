"""Tests for chat/instruct contract edge cases."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e_harness.plugins.chat_instruct import ChatInstructPlugin


def _case() -> E2ECase:
    return E2ECase(
        name="model",
        hf_id="hf/model",
        family="family",
        runtime_strategy="hybrid_mamba_attention",
        reference_family="chat_instruct_template",
        user_contract="chat_response",
        inputs={"prompt": "What is the capital of France? Answer in one word."},
        metadata={"contract_config": {"use_chat_template": True, "enable_thinking": False}},
    )


def test_chat_instruct_rejects_thinking_block_when_disabled() -> None:
    result = ChatInstructPlugin().verify(
        StageOutput(stage_name="full_generation", text="<think>\nOkay, the user is"),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["thinking_suppressed"].value == 0.0
    assert "thinking disabled" in result.message


def test_chat_instruct_accepts_golden_answer_without_thinking() -> None:
    result = ChatInstructPlugin().verify(
        StageOutput(stage_name="full_generation", text="Paris"),
        StageOutput(stage_name="full_generation", text="Paris"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed


def test_chat_instruct_normalizes_sentencepiece_markers() -> None:
    result = ChatInstructPlugin().verify(
        StageOutput(
            stage_name="full_generation",
            text="2\u2581+\u25812\u2581=\u25814\nThe\u2581answer\u2581is\u2581$\\boxed{4}$.",
        ),
        StageOutput(
            stage_name="full_generation",
            text="2 + 2 = 4\nThe answer is $\\boxed{4}$.",
        ),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed


def test_internlm_config_does_not_request_thinking_suppression() -> None:
    case = _case()
    case.name = "internlm2-1.8b"

    assert ChatInstructPlugin().configure_reference(case) == {
        "use_chat_template": True,
        "enable_thinking": True,
        "reference_generation_mode": "hf_generate",
    }

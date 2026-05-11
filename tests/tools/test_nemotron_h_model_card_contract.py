"""Tests for the Nemotron-H model-card no-thinking contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.plugins.nemotron_h_model_card import plugin


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/nemotron-h-nano-9b.json"
WAIVES_PATH = REPO_ROOT / "tests/e2e/waives.txt"


def _case() -> E2ECase:
    return E2ECase(
        name="nemotron-h-nano-9b",
        hf_id="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        family="nemotron_h",
        runtime_strategy="hybrid_mamba_attention",
        reference_backend="invariant_only",
        reference_family="nemotron_h_no_think_model_card",
        user_contract="chat_response",
        inputs={"prompt": "Write a haiku about GPUs", "max_new_tokens": 32},
    )


def _trt_output(text: str, command: list[str] | None = None) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        text=text,
        data={"cpp_returncode": 0},
        metadata={
            "cpp": {
                "command": command
                or [
                    "./build/trtmc",
                    "run",
                    "nemotron-h-nano-9b.trtfb",
                    "--chat-template",
                    "--no-thinking",
                ]
            }
        },
    )


def test_manifest_uses_model_card_no_thinking_case() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["hf_id"] == "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    assert manifest["reference_backend"] == "invariant_only"
    assert manifest["reference_family"] == "nemotron_h_no_think_model_card"
    assert manifest["user_contract"] == "chat_response"
    assert manifest["prompt"] == "Write a haiku about GPUs"
    assert manifest["max_new_tokens"] == 32


def test_nemotron_h_e2e_is_not_waived() -> None:
    waives = WAIVES_PATH.read_text(encoding="utf-8")

    assert "nemotron-h-nano-9b" not in waives


def test_contract_accepts_model_card_no_thinking_haiku() -> None:
    result = plugin.verify(
        _trt_output("GPUs hum softly\nCUDA cores wake at dawn\nData rivers flow"),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.passed
    assert result.metrics["chat_no_thinking_flags"].passed
    assert result.metrics["mentions_gpu_topic"].passed


def test_contract_rejects_missing_no_thinking_flag() -> None:
    result = plugin.verify(
        _trt_output(
            "GPUs hum softly\nCUDA cores wake at dawn",
            command=["./build/trtmc", "run", "nemotron-h-nano-9b.trtfb"],
        ),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert not result.metrics["chat_no_thinking_flags"].passed


def test_contract_rejects_visible_reasoning_trace() -> None:
    result = plugin.verify(
        _trt_output("<think>\nI should write a haiku.\n</think>\nGPUs hum softly"),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert not result.metrics["no_visible_thinking_trace"].passed

"""Tests for the DeepSeek-OCR-2 model-card contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e_harness.plugins.deepseek_ocr_model_card import plugin


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/deepseek-ocr-l0.json"


def _case() -> E2ECase:
    return E2ECase(
        name="deepseek-ocr-l0",
        hf_id="deepseek-ai/DeepSeek-OCR-2",
        family="deepseek_ocr",
        runtime_strategy="vision_language",
        reference_backend="invariant_only",
        reference_family="deepseek_ocr_model_card",
        user_contract="ocr_text",
        inputs={
            "prompt": "<|grounding|>Convert the document to markdown.",
            "max_new_tokens": 80,
        },
        metadata={
            "ocr_expected_fragments": [
                "DeepSeek-OCR-2 family plugin",
                "Standard MHA",
                "DeepSeek-V2-style language decoder",
            ]
        },
    )


def _full_generation(text: str, command: list[str] | None = None) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        text=text,
        data={"generated_text": text},
        metadata={
            "returncode": 0,
            "command": command
            or [
                "./build/trtmc",
                "run",
                "deepseek-ocr-l0.trtfb",
                "--image",
                "tests/e2e/data/orc_test_img.jpeg",
            ],
        },
    )


def test_manifest_uses_model_card_document_markdown_prompt() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["reference_backend"] == "invariant_only"
    assert manifest["reference_family"] == "deepseek_ocr_model_card"
    assert manifest["user_contract"] == "ocr_text"
    assert manifest["prompt"] == "<|grounding|>Convert the document to markdown."
    assert manifest["test_image"] == "tests/e2e/data/orc_test_img.jpeg"
    assert len(manifest["ocr_expected_fragments"]) >= 3


def test_deepseek_ocr_contract_accepts_expected_ocr_fragments() -> None:
    text = (
        "DeepSeek-OCR-2 family plugin - Standard MHA and MoE with shared experts.\n"
        "DeepSeek-OCR-2 is a VL model with a DeepSeek-V2-style language decoder."
    )
    result = plugin.verify(
        _full_generation(text),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert result.passed
    assert result.metrics["expected_fragment_matches"].passed


def test_deepseek_ocr_contract_rejects_missing_image_flag() -> None:
    result = plugin.verify(
        _full_generation(
            "DeepSeek-OCR-2 family plugin - Standard MHA.",
            command=["./build/trtmc", "run", "deepseek-ocr-l0.trtfb"],
        ),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert not result.passed
    assert not result.metrics["image_flag_forwarded"].passed


def test_deepseek_ocr_contract_rejects_missing_expected_text() -> None:
    result = plugin.verify(
        _full_generation("This image contains unrelated text."),
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert not result.passed
    assert not result.metrics["expected_fragment_matches"].passed


def test_deepseek_ocr_contract_checks_vision_encode_status() -> None:
    result = plugin.verify(
        StageOutput("vision_encode", data={"passed": False}),
        StageOutput("vision_encode"),
        _case(),
        ThresholdProfile(task_strategy="vision_language_generation"),
    )

    assert not result.passed
    assert not result.metrics["vision_encode_ok"].passed

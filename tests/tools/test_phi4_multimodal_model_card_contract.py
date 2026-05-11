"""Tests for Phi-4-multimodal model-card reference setup."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e_harness.references.hf_transformers import _vl_fallback_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/phi4-multimodal.json"


def test_phi4_manifest_uses_model_card_vision_prompt() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["hf_id"] == "microsoft/Phi-4-multimodal-instruct"
    assert manifest["runtime_strategy"] == "vision_language"
    assert manifest["trust_remote_code"] is True
    assert manifest["prompt"] == "What is shown in this image?"
    assert manifest["test_image"] == "tests/e2e/data/test_img.jpeg"


def test_phi4_hf_reference_fallback_uses_processor_image_marker() -> None:
    prompt = "What is shown in this image?"

    rendered = _vl_fallback_prompt(
        "microsoft/Phi-4-multimodal-instruct", prompt)

    assert rendered == (
        "<|user|><|image_1|>What is shown in this image?"
        "<|end|><|assistant|>"
    )

"""Tests for the Gemma 2 instruct E2E model-card contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests import test_e2e
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.plugins.chat_instruct import ChatInstructPlugin


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/gemma-2-2b.json"


def test_gemma_manifest_uses_model_card_chat_contract() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    case = load_manifest(MANIFEST_PATH)

    assert raw["hf_id"] == "google/gemma-2-2b-it"
    assert raw["gated"] is True
    assert case.reference_family == "chat_instruct_template"
    assert case.user_contract == "chat_response"
    assert case.inputs["prompt"] == "Write a hello world program"
    assert case.inputs["max_new_tokens"] == 64
    assert any(
        req.kind == "hf_auth_token_present" and req.gating
        for req in case.preflight
    )


def test_gemma_chat_plugin_applies_template() -> None:
    case = load_manifest(MANIFEST_PATH)
    config = ChatInstructPlugin().configure_reference(case)

    assert config["use_chat_template"] is True
    assert config["enable_thinking"] is False


def test_gemma_is_not_globally_waived() -> None:
    waives = test_e2e._load_waives()

    assert "gemma-2-2b" not in waives

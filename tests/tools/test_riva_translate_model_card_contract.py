"""Riva Translate E2E contract checks tied to the Hugging Face model card."""

from __future__ import annotations

import json
from pathlib import Path

from tests import test_e2e
from tests.e2e_harness.manifest_loader import find_manifest_path, load_manifest
from tests.e2e_harness.plugins.translation import plugin as translation_plugin


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = find_manifest_path(
    "riva-translate-4b",
    REPO_ROOT / "tests" / "e2e" / "models",
)
assert MANIFEST_PATH is not None


MODEL_CARD_EN_FR_PROMPT = (
    "<s>System\n"
    "You are an expert at translating text from English to French.</s>\n"
    "<s>User\n"
    "What is the French translation of the sentence: Hello, how are you?</s>\n"
    "<s>Assistant\n"
)


def test_riva_manifest_uses_model_card_translation_contract() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    case = load_manifest(MANIFEST_PATH)

    assert raw["hf_id"] == "nvidia/Riva-Translate-4B-Instruct-v1.1"
    assert raw["gated"] is True
    assert "skip_logit_parity" not in raw
    assert case.reference_family == "translation_chat_template"
    assert case.user_contract == "translation"
    assert case.inputs["prompt"] == MODEL_CARD_EN_FR_PROMPT
    assert case.inputs["max_new_tokens"] == 20
    assert "skip_logit_parity" not in case.metadata
    assert any(
        req.kind == "hf_auth_token_present" and req.gating
        for req in case.preflight
    )


def test_riva_translation_plugin_uses_preformatted_model_card_prompt() -> None:
    case = load_manifest(MANIFEST_PATH)
    reference_cfg = translation_plugin.configure_reference(case)

    assert reference_cfg["use_chat_template"] is False


def test_riva_is_not_globally_waived() -> None:
    waives = test_e2e._load_waives()

    assert "riva-translate-4b" not in waives

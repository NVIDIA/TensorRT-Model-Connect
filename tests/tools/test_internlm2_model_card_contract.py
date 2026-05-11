"""Tests for the InternLM2 model-card E2E contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests import test_e2e
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.references.hf_transformers import (
    _legacy_dynamic_cache_compat_script,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/internlm2-1.8b.json"
MODEL_CARD_PROMPT = "Who are you?"


def test_internlm2_manifest_uses_model_card_chat_contract() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    case = load_manifest(MANIFEST_PATH)

    assert raw["hf_id"] == "internlm/internlm2-math-plus-1_8b"
    assert raw["reference_family"] == "chat_instruct_template"
    assert raw["user_contract"] == "chat_response"
    assert raw["prompt"] == MODEL_CARD_PROMPT
    assert raw["trust_remote_code"] is True
    assert raw["legacy_dynamic_cache_compat"] is True
    assert case.metadata["legacy_dynamic_cache_compat"] is True


def test_internlm2_legacy_cache_compat_is_opt_in() -> None:
    assert _legacy_dynamic_cache_compat_script(False) == ""

    compat = _legacy_dynamic_cache_compat_script(True)

    assert "DynamicCache" in compat
    assert "from_legacy_cache" in compat
    assert "to_legacy_cache" in compat
    assert "past_key_values is None" in compat


def test_internlm2_is_not_globally_waived() -> None:
    waives = test_e2e._load_waives()

    assert "internlm2-1.8b" not in waives

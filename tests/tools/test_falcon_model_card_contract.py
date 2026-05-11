"""Static contract checks for the Falcon-RW-1B E2E manifest."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/falcon-rw-1b.json"
WAIVES_PATH = REPO_ROOT / "tests/e2e/waives.txt"

MODEL_CARD_PROMPT = (
    "Girafatron is obsessed with giraffes, the most glorious animal on the face "
    "of this Earth. Giraftron believes all other animals are irrelevant when "
    "compared to the glorious majesty of the giraffe.\n"
    "Daniel: Hello, Girafatron!\n"
    "Girafatron:"
)


def test_falcon_manifest_uses_model_card_continuation_prompt() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["hf_id"] == "tiiuae/falcon-rw-1b"
    assert manifest["prompt"] == MODEL_CARD_PROMPT
    assert manifest["reference_family"] == "causal_base_continuation"
    assert manifest["user_contract"] == "continuation_parity"
    assert manifest["trust_remote_code"] is False
    assert manifest["max_new_tokens"] > 0


def test_falcon_e2e_is_not_waived() -> None:
    waives = WAIVES_PATH.read_text(encoding="utf-8")

    assert "falcon-rw-1b" not in waives

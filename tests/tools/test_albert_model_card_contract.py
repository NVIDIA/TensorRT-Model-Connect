"""ALBERT E2E manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_albert_manifest_matches_model_card_feature_example() -> None:
    manifest = json.loads((ROOT / "tests/e2e/models/albert-base.json").read_text())

    assert manifest["hf_id"] == "albert/albert-base-v2"
    assert manifest["family"] == "albert"
    assert manifest["runtime_strategy"] == "encoder_only"
    assert manifest["prompt"] == "Replace me by any text you'd like."
    assert manifest["max_new_tokens"] == 0
    assert manifest["threshold_overrides"]["cls_embedding_cosine"] >= 0.95
    assert "AlbertTokenizer + AlbertModel" in manifest["notes"]


def test_albert_is_not_waived() -> None:
    waives = (ROOT / "tests/e2e/waives.txt").read_text()

    assert "albert-base" not in waives

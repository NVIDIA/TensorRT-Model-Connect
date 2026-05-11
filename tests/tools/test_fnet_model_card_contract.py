"""FNet E2E manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_fnet_manifest_matches_model_card_feature_example() -> None:
    manifest = json.loads((ROOT / "tests/e2e/models/fnet-base.json").read_text())

    assert manifest["hf_id"] == "google/fnet-base"
    assert manifest["family"] == "fnet"
    assert manifest["runtime_strategy"] == "encoder_only"
    assert manifest["max_cache_length"] == 512
    assert manifest["builder_optimization_level"] == 0
    assert manifest["prompt"] == "Replace me by any text you'd like."
    assert manifest["max_new_tokens"] == 0
    assert manifest["threshold_overrides"]["cls_embedding_cosine"] >= 0.95
    assert "FNetTokenizer + FNetModel" in manifest["notes"]
    assert "max_length=512" in manifest["notes"]


def test_fnet_is_not_waived() -> None:
    waives = (ROOT / "tests/e2e/waives.txt").read_text()

    assert "fnet-base" not in waives

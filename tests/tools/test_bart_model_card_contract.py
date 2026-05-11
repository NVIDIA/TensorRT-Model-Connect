"""BART E2E contract checks."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.manifest_loader import load_manifest


def test_bart_base_manifest_matches_model_card_feature_extraction() -> None:
    case = load_manifest(Path("tests/e2e/models/bart-base.json"))

    assert case.runtime_strategy == "seq2seq_encoder_decoder"
    assert case.task_strategy == "encoder_only_nlp"
    assert case.reference_family == "encoder_base_features"
    assert case.user_contract == "representation_parity"
    assert case.inputs["prompt"] == "Hello, my dog is cute"
    assert case.stages[0].name == "full_inference"
    assert case.threshold_overrides["cls_embedding_cosine"] >= 0.99

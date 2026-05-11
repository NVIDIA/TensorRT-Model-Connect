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


def test_bart_decoder_exposes_model_card_last_hidden_state() -> None:
    source = Path(
        "tensorrt_model_connect/tensorrt_model_connect/families/bart.py"
    ).read_text(encoding="utf-8")

    assert 'encoder_mask = network.add_input("encoder_mask"' in source
    assert 'hidden_state.name = "decoder_hidden"' in source
    assert "network.mark_output(hidden_state)" in source
    assert "mask=enc_mask_4d.get_output(0)" in source


def test_seq2seq_encode_reads_decoder_hidden_not_encoder_output() -> None:
    source = Path("src/runtime/plugins/seq2seq_plugin.cpp").read_text(encoding="utf-8")

    encode_body = source.split("EmbeddingResult encode", maxsplit=1)[1].split(
        "const char* model_id", maxsplit=1
    )[0]
    assert "setup_cross_attention();" in encode_body
    assert "run_decoder_feature_step(decoder_start_token_id_, result)" in encode_body
    assert 'outputs.find("decoder_hidden")' in source
    assert "cudaMemcpy(result.data.data(), enc_out" not in encode_body

"""NLLB E2E contract checks tied to the Hugging Face model-card usage."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "tensorrt_model_connect"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from tests.e2e_harness.manifest_loader import load_manifest  # noqa: E402
from tests.e2e_harness.plugins.translation import plugin as translation_plugin  # noqa: E402


NLLB_MANIFEST = ROOT / "tests" / "e2e" / "models" / "nllb-200.json"


def test_nllb_manifest_uses_model_card_translation_contract() -> None:
    case = load_manifest(NLLB_MANIFEST)

    assert case.reference_family == "seq2seq_translation"
    assert case.user_contract == "translation"
    assert "skip_reason" not in case.metadata
    assert case.inputs["prompt"] == "UN Chief says there is no military solution in Syria"

    reference_cfg = translation_plugin.configure_reference(case)
    assert reference_cfg["auto_class"] == "AutoModelForSeq2SeqLM"
    assert reference_cfg["src_lang"] == "eng_Latn"
    assert reference_cfg["tgt_lang"] == "fra_Latn"
    assert reference_cfg["forced_bos_token"] == "fra_Latn"


def test_nllb_bundle_config_carries_model_card_language_tokens(monkeypatch) -> None:
    fake_trt = types.ModuleType("tensorrt")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    from tensorrt_model_connect.config import ModelConfig

    m2m_100 = importlib.import_module("tensorrt_model_connect.families.m2m_100")
    cfg = ModelConfig(
        model_type="m2m_100",
        vocab_size=256206,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        raw={
            "tokenizer_class": "NllbTokenizer",
            "encoder_layers": 12,
            "decoder_layers": 12,
            "decoder_start_token_id": 2,
        },
    )

    bundle_cfg = m2m_100.M2M100Plugin().get_vl_config(cfg)

    assert bundle_cfg["source_lang_token"] == "eng_Latn"
    assert bundle_cfg["target_lang_token"] == "fra_Latn"
    assert bundle_cfg["source_lang_token_id"] == 256047
    assert bundle_cfg["forced_bos_token_id"] == 256057

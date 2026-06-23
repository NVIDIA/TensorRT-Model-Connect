"""Whisper-owned diff_logits handler tests."""

from __future__ import annotations

import importlib


def test_whisper_handler_is_model_owned():
    mod = importlib.import_module("diff_logits")
    handler = mod._find_family_diff_logits_handler("whisper")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/whisper/diff_logits.py")
    assert handler.handles_model_type("Whisper") is True


def test_whisper_handler_prompt_cases_override_tokenization():
    mod = importlib.import_module("diff_logits")
    handler = mod._find_family_diff_logits_handler("whisper")

    cases = mod._prompt_cases_for_handler([("default", "ignored")], handler)

    assert len(cases) == 1
    label, display_text, input_ids = cases[0]
    assert label == "whisper-decode"
    assert "decoder start tokens" in display_text
    assert input_ids == [50258, 50259, 50359, 50363]

"""RWKV-owned diff_logits handler tests."""

from __future__ import annotations

import importlib


def test_rwkv_handler_is_model_owned():
    mod = importlib.import_module("diff_logits")
    handler = mod._find_family_diff_logits_handler("rwkv")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/rwkv/diff_logits.py")
    assert handler.handles_model_type("RWKV") is True

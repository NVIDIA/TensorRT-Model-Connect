"""Qwen-VL-owned coverage for diff_vl handler dispatch."""

from __future__ import annotations

import importlib


def test_qwen_vl_variants_are_owned_by_family_handler() -> None:
    mod = importlib.import_module("diff_vl")
    handler = mod._find_family_diff_vl_handler("qwen2_5_vl")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith("families/qwen_vl/diff_vl.py")
    assert handler.handles_model_type("qwen_vl") is True
    assert handler.handles_model_type("Qwen2_5_VL") is True

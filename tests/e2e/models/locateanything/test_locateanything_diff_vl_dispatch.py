"""LocateAnything-owned diff_vl handler tests."""

from __future__ import annotations

import importlib


def test_locateanything_handler_is_model_owned():
    mod = importlib.import_module("diff_vl")
    handler = mod._find_family_diff_vl_handler("locateanything")

    assert handler is not None
    assert handler.__file__.replace("\\", "/").endswith(
        "families/locateanything/diff_vl.py")
    assert handler.handles_model_type("LocateAnything") is True

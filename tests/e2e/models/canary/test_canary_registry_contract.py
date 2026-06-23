"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import plugin modules")

from tensorrt_model_connect.families import find_plugin


def _plugin(model_type: str):
    plugin = find_plugin(model_type)
    assert plugin is not None
    return plugin

def test_runtime_strategy() -> None:
    plugin = _plugin("canary")
    assert getattr(plugin, "runtime_strategy", None) == "speech_to_text"

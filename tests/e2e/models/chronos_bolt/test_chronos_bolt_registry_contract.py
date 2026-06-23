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
    plugin = _plugin("chronos_bolt")
    assert getattr(plugin, "runtime_strategy", None) == "chronos_bolt_trt"


def test_matches_official_t5_config() -> None:
    from tensorrt_model_connect.config import ModelConfig

    plugin = find_plugin(ModelConfig(
        model_type="t5",
        architectures=["ChronosBoltModelForForecasting"],
        raw={
            "model_type": "t5",
            "architectures": ["ChronosBoltModelForForecasting"],
            "chronos_config": {"context_length": 16},
        },
    ))
    assert plugin is not None
    assert plugin.name == "chronos_bolt"

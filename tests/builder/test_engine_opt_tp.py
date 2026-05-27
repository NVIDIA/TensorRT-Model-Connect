"""Focused tests for OPT tensor-parallel dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        attention_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def test_opt_plugin_routes_parallel_builds(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.opt.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"opt-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = module.OPTPlugin().build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"opt-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["activation"] == "relu"
    assert kwargs["mlp_type"] == "gelu_fc"
    assert kwargs["verbose"] is True

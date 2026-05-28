"""Focused tests for GPT-family tensor-parallel dispatch."""

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
        raw={"rotary_pct": 0.25, "use_parallel_residual": True},
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


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("tensorrt_model_connect.families.gpt2.plugin", "GPT2Plugin"),
        ("tensorrt_model_connect.families.gpt_neo.plugin", "GPTNeoPlugin"),
        ("tensorrt_model_connect.families.gpt_neox.plugin", "GPTNeoXPlugin"),
    ],
)
def test_gpt_family_plugins_route_parallel_builds(
    monkeypatch, module_name: str, class_name: str
) -> None:
    module = importlib.import_module(module_name)
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"gpt-tp-plan"

    monkeypatch.setattr(module, "build_dual_profile_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    result = getattr(module, class_name)().build_engine(
        _config(),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"gpt-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["mlp_type"] == "gelu_fc"
    assert kwargs["verbose"] is True

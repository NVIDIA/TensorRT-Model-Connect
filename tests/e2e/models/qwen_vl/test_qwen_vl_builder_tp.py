"""Focused tests for Qwen-VL tensor-parallel dispatch."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="Qwen-VL builder tests require TensorRT")

try:
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(*, qwen3: bool = False) -> SimpleNamespace:
    raw = {"vision_config": {"deepstack_visual_indexes": [5, 11, 17]}} if qwen3 else {}
    return SimpleNamespace(
        raw=raw,
        hidden_size=16,
        vocab_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        attention_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


@pytest.mark.parametrize(
    ("qwen3", "tp_size", "deepstack_num_levels"),
    [
        (False, 2, 0),
        (True, 4, 3),
    ],
)
def test_qwen_vl_plugin_routes_parallel_builds(
    monkeypatch, qwen3: bool, tp_size: int, deepstack_num_levels: int
) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-tp-plan"

    monkeypatch.setattr(module, "build_qwen_vl_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=tp_size, rank=1)
    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=qwen3),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        23,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"qwen-vl-tp-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 23
    assert kwargs["parallel_config"] == parallel
    assert kwargs["embed_input"] is True
    assert kwargs["deepstack_num_levels"] == deepstack_num_levels
    assert kwargs["verbose"] is True


def test_qwen25_vl_plugin_forwards_precision_to_standard_builder(monkeypatch) -> None:
    module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_vl.plugin")
    calls: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-vl-plan"

    monkeypatch.setattr(module, "build_standard_decoder_engine", fake_build)

    result = module.QwenVLPlugin().build_engine(
        _config(qwen3=False),
        {"_attention_size": 16, "_kv_attention_size": 16, "_mlp_size": 32},
        31,
        precision="bf16",
        verbose=True,
    )

    assert result == b"qwen-vl-plan"
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 31
    assert kwargs["precision"] == "bf16"
    assert kwargs["embed_input"] is True
    assert kwargs["verbose"] is True

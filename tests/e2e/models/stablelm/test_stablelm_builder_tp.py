"""Family-owned decoder tensor-parallel tests for stablelm."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
)


FAMILY = 'stablelm'
PLUGIN_CLASS = 'StableLMPlugin'
MODEL_TYPE = 'stablelm'
TP_SIZE = 4
RAW = {'partial_rotary_factor': 0.75}
EXPECTED_KWARGS = {'mlp_type': 'swiglu',
 'norm_type': 'layernorm',
 'partial_rotary_factor': 0.75,
 'position_type': 'rope'}


def _config(model_type: str, tp_size: int, raw: dict[str, object]) -> ModelConfig:
    kv_heads = 2 if tp_size == 2 else 4
    return ModelConfig(
        model_type=model_type,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw=raw,
    )


def test_stablelm_plugin_routes_tp_build(monkeypatch) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_cache_length"] = max_cache_length
        captured["kwargs"] = kwargs
        return b"tp-plan"

    monkeypatch.setattr(
        plugin_mod,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(
        plugin_mod,
        "build_dual_profile_tp_decoder_engine",
        fake_build,
    )

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=TP_SIZE, rank=1)
    plan = getattr(plugin_mod, PLUGIN_CLASS)().build_engine(
        _config(MODEL_TYPE, TP_SIZE, RAW),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
        verbose=True,
        parallel_config=parallel,
    )

    assert plan == b"tp-plan"
    assert captured["max_cache_length"] == 17
    kwargs = captured["kwargs"]
    assert kwargs["precision"] == "fp16"
    assert kwargs["quant_ctx"] is None
    assert kwargs["verbose"] is True
    assert kwargs["parallel_config"] == parallel
    for key, expected in EXPECTED_KWARGS.items():
        assert kwargs[key] == expected


def test_stablelm_plugin_rejects_quantized_tp(monkeypatch) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")
    monkeypatch.setattr(
        plugin_mod,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        getattr(plugin_mod, PLUGIN_CLASS)().build_engine(
            _config(MODEL_TYPE, TP_SIZE, RAW),
            WeightDict(),
            max_cache_length=17,
            quant_ctx=object(),
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=TP_SIZE, rank=0),
        )

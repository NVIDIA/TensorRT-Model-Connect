# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned decoder tensor-parallel tests for starcoder2."""

from __future__ import annotations

import importlib
import inspect

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


FAMILY = 'starcoder2'
PLUGIN_CLASS = 'StarCoder2Plugin'
MODEL_TYPE = 'starcoder2'
TP_SIZE = 2
RAW = {}
SPECIALIZED_STRATEGY_KWARGS = {
    "activation",
    "mlp_type",
    "norm_type",
    "position_type",
}


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
        rms_norm_eps=1e-5,
        raw=raw,
    )


def test_starcoder2_plugin_routes_tp_build(monkeypatch) -> None:
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
    assert SPECIALIZED_STRATEGY_KWARGS.isdisjoint(kwargs)


def test_starcoder2_builders_do_not_expose_generic_strategy_switches() -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")

    for builder in (
        plugin_mod.build_standard_decoder_engine,
        plugin_mod.build_dual_profile_tp_decoder_engine,
    ):
        assert SPECIALIZED_STRATEGY_KWARGS.isdisjoint(
            inspect.signature(builder).parameters)


def test_starcoder2_plugin_rejects_quantized_tp(monkeypatch) -> None:
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

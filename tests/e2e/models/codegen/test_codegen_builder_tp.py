# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned decoder tensor-parallel tests for codegen."""

from __future__ import annotations

import importlib

import numpy as np
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
    shard_standard_decoder_weights,
)


FAMILY = 'codegen'
PLUGIN_CLASS = 'CodeGenPlugin'
MODEL_TYPE = 'codegen'
TP_SIZE = 4
RAW = {'rotary_dim': 2}
EXPECTED_KWARGS = {'activation': 'gelu_new',
 'fp32_lm_head': True,
 'fp32_qk_attention': True,
 'fp32_rope': True,
 'interleaved_rope': True,
 'mlp_type': 'gelu_fc',
 'norm_type': 'layernorm',
 'parallel_residual': True,
 'partial_rotary_factor': 0.5,
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


def test_codegen_plugin_routes_tp_build(monkeypatch) -> None:
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


def test_codegen_plugin_routes_accuracy_precision_boundaries(monkeypatch) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"single-device-plan"

    monkeypatch.setattr(
        plugin_mod,
        "build_standard_decoder_engine",
        fake_build,
    )

    plan = getattr(plugin_mod, PLUGIN_CLASS)().build_engine(
        _config(MODEL_TYPE, 1, RAW),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
    )

    assert plan == b"single-device-plan"
    kwargs = captured["kwargs"]
    assert kwargs["fp32_rope"] is True
    assert kwargs["fp32_qk_attention"] is True
    assert kwargs["fp32_lm_head"] is True


def test_codegen_plugin_rejects_quantized_tp(monkeypatch) -> None:
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


def test_standard_decoder_tp_shards_gelu_fc_input_bias_only() -> None:
    weights = WeightDict({
        "_attention_size": 16,
        "_kv_attention_size": 16,
        "_mlp_size": 32,
        "layer.0.w_fc1": np.zeros((16, 32), dtype=np.float32),
        "layer.0.fc1_bias": np.arange(32, dtype=np.float32),
        "layer.0.w_fc2": np.zeros((32, 16), dtype=np.float32),
        "layer.0.fc2_bias": np.arange(16, dtype=np.float32),
    })

    shard = shard_standard_decoder_weights(
        _config("codegen", 4, {}),
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    np.testing.assert_array_equal(
        shard["layer.0.fc1_bias"],
        np.arange(16, 24, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        shard["layer.0.fc2_bias"],
        weights["layer.0.fc2_bias"],
    )

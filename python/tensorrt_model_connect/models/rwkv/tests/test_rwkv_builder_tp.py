# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for RWKV tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    rwkv_plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.rwkv.model")
    from tensorrt_model_connect.models.rwkv import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(hidden: int = 8, intermediate: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=hidden,
        vocab_size=32,
        num_hidden_layers=1,
        intermediate_size=intermediate,
        rms_norm_eps=1e-5,
    )


def _weights(hidden: int = 8, intermediate: int = 16) -> dict:
    weights: dict[str, object] = {
        "_intermediate_size": intermediate,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "final_norm_beta": np.zeros((hidden,), dtype=np.float32),
        "w_lm_head": np.zeros((hidden, 32), dtype=np.float32),
    }
    prefix = "layer.0"
    weights[f"{prefix}.attn_norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.attn_norm_beta"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.ffn_norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.ffn_norm_beta"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_decay"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_first"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_mix_key"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_mix_value"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_mix_receptance"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.w_attn_k"] = np.zeros((hidden, hidden), dtype=np.float32)
    weights[f"{prefix}.w_attn_v"] = np.zeros((hidden, hidden), dtype=np.float32)
    weights[f"{prefix}.w_attn_r"] = np.zeros((hidden, hidden), dtype=np.float32)
    weights[f"{prefix}.w_attn_o"] = np.zeros((hidden, hidden), dtype=np.float32)
    weights[f"{prefix}.time_mix_ffn_key"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.time_mix_ffn_receptance"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.w_ffn_k"] = np.zeros((hidden, intermediate), dtype=np.float32)
    weights[f"{prefix}.w_ffn_v"] = np.zeros((intermediate, hidden), dtype=np.float32)
    weights[f"{prefix}.w_ffn_r"] = np.zeros((hidden, hidden), dtype=np.float32)
    return weights


def test_rwkv_tp_slices_hidden_and_intermediate_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    config = _config()
    weights = _weights()
    weights["layer.0.time_decay"] = np.arange(8, dtype=np.float32)
    weights["layer.0.w_attn_k"] = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    weights["layer.0.w_attn_o"] = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    weights["layer.0.w_ffn_k"] = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    weights["layer.0.w_ffn_v"] = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)

    sharded = tp_builder.shard_rwkv_weights(config, weights, parallel=parallel)

    np.testing.assert_array_equal(sharded["layer.0.time_decay"], weights["layer.0.time_decay"][4:6])
    np.testing.assert_array_equal(sharded["layer.0.w_attn_k"], weights["layer.0.w_attn_k"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.0.w_attn_o"], weights["layer.0.w_attn_o"][4:6, :])
    np.testing.assert_array_equal(sharded["layer.0.w_ffn_k"], weights["layer.0.w_ffn_k"][:, 8:12])
    np.testing.assert_array_equal(sharded["layer.0.w_ffn_v"], weights["layer.0.w_ffn_v"][8:12, :])
    assert sharded["_intermediate_size"] == 4


def test_rwkv_tp_validation_rejects_non_divisible_hidden_dim():
    with pytest.raises(ValueError, match="hidden_size divisible"):
        tp_builder._validate_rwkv_tp(
            _config(hidden=10),
            _weights(hidden=10),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_rwkv_tp_validation_requires_concrete_rank():
    with pytest.raises(ValueError, match="concrete rank"):
        tp_builder._validate_rwkv_tp(
            _config(),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1),
        )


def test_rwkv_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"rwkv-tp-plan"

    monkeypatch.setattr(
        rwkv_plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_rwkv_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = rwkv_plugin_module.build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"rwkv-tp-plan"
    assert calls["require"][0] == parallel
    assert "RWKV tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

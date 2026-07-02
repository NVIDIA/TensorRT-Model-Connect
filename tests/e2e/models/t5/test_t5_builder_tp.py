# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for T5 tensor-parallel decoder support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    t5_plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.t5.plugin")
    from tensorrt_model_connect.families.t5 import decoder_tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _weights(num_heads: int = 8, d_kv: int = 64, d_ff: int = 2048) -> dict:
    hidden = num_heads * d_kv
    weights: dict[str, object] = {
        "_dec_layers": 1,
        "_num_heads": num_heads,
        "_d_kv": d_kv,
        "_d_ff": d_ff,
        "_hidden": hidden,
        "_vocab_size": 128,
        "_num_buckets": 32,
        "_max_distance": 128,
        "_layer_norm_eps": 1e-6,
        "dec_self_rel_attn_bias": np.arange(32 * num_heads, dtype=np.float32).reshape(
            32, num_heads),
        "dec_cross_rel_attn_bias": np.arange(32 * num_heads, dtype=np.float32).reshape(
            32, num_heads),
    }
    prefix = "layer.0"
    attention = num_heads * d_kv
    for key in ("w_q", "w_k", "w_v", "cross_w_q", "cross_w_k", "cross_w_v"):
        weights[f"{prefix}.{key}"] = np.zeros((hidden, attention), dtype=np.float32)
    for key in ("w_o", "cross_w_o"):
        weights[f"{prefix}.{key}"] = np.zeros((attention, hidden), dtype=np.float32)
    weights[f"{prefix}.w_fc1"] = np.zeros((hidden, d_ff), dtype=np.float32)
    weights[f"{prefix}.w_fc2"] = np.zeros((d_ff, hidden), dtype=np.float32)
    return weights


def _config() -> SimpleNamespace:
    return SimpleNamespace(raw={}, hidden_size=512, vocab_size=128)


def test_t5_tp_slices_projection_columns_rows_and_bias_heads():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()
    weights["layer.0.w_q"] = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    weights["layer.0.w_o"] = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)

    sharded = decoder_tp_builder.shard_t5_decoder_weights(
        weights, parallel=parallel)

    np.testing.assert_array_equal(sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 256:384])
    np.testing.assert_array_equal(sharded["layer.0.w_o"], weights["layer.0.w_o"][256:384, :])
    np.testing.assert_array_equal(
        sharded["dec_self_rel_attn_bias"], weights["dec_self_rel_attn_bias"][:, 4:6])


def test_t5_tp_validation_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="num_heads divisible"):
        decoder_tp_builder._validate_t5_tp(
            _weights(num_heads=6), ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0))


def test_t5_tp_validation_requires_concrete_rank():
    with pytest.raises(ValueError, match="concrete rank"):
        decoder_tp_builder._validate_t5_tp(
            _weights(), ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1))


def test_t5_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"t5-tp-plan"

    monkeypatch.setattr(
        t5_plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(decoder_tp_builder, "build_t5_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = t5_plugin_module.T5Plugin().build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"t5-tp-plan"
    assert calls["require"][0] == parallel
    assert "T5 tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

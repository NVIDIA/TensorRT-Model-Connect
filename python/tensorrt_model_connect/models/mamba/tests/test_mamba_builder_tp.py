# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Mamba tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    mamba_plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.mamba.model")
    from tensorrt_model_connect.models.mamba import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _weights(d_inner: int = 16, state_size: int = 4, conv_kernel: int = 3) -> dict:
    hidden = 8
    dt_rank = 2
    weights: dict[str, object] = {
        "_d_inner": d_inner,
        "_state_size": state_size,
        "_conv_kernel": conv_kernel,
        "_dt_rank": dt_rank,
        "_num_layers": 1,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_lm_head": np.zeros((hidden, 32), dtype=np.float32),
    }
    prefix = "layer.0"
    weights[f"{prefix}.norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.w_in_x"] = np.zeros((hidden, d_inner), dtype=np.float32)
    weights[f"{prefix}.w_in_z"] = np.zeros((hidden, d_inner), dtype=np.float32)
    weights[f"{prefix}.conv1d_weight"] = np.zeros((d_inner, conv_kernel), dtype=np.float32)
    weights[f"{prefix}.conv1d_bias"] = np.zeros((d_inner,), dtype=np.float32)
    weights[f"{prefix}.w_dt_in"] = np.zeros((d_inner, dt_rank), dtype=np.float32)
    weights[f"{prefix}.w_B"] = np.zeros((d_inner, state_size), dtype=np.float32)
    weights[f"{prefix}.w_C"] = np.zeros((d_inner, state_size), dtype=np.float32)
    weights[f"{prefix}.w_dt_out"] = np.zeros((dt_rank, d_inner), dtype=np.float32)
    weights[f"{prefix}.dt_proj_bias"] = np.zeros((d_inner,), dtype=np.float32)
    weights[f"{prefix}.A"] = np.zeros((d_inner, state_size), dtype=np.float32)
    weights[f"{prefix}.D"] = np.zeros((d_inner,), dtype=np.float32)
    weights[f"{prefix}.w_out"] = np.zeros((d_inner, hidden), dtype=np.float32)
    return weights


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
    )


def test_mamba_tp_slices_inner_dimension_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()
    weights["layer.0.w_in_x"] = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    weights["layer.0.w_dt_in"] = np.arange(16 * 2, dtype=np.float32).reshape(16, 2)
    weights["layer.0.w_dt_out"] = np.arange(2 * 16, dtype=np.float32).reshape(2, 16)
    weights["layer.0.w_out"] = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)

    sharded = tp_builder.shard_mamba_weights(weights, parallel=parallel)

    np.testing.assert_array_equal(sharded["layer.0.w_in_x"], weights["layer.0.w_in_x"][:, 8:12])
    np.testing.assert_array_equal(sharded["layer.0.w_dt_in"], weights["layer.0.w_dt_in"][8:12, :])
    np.testing.assert_array_equal(sharded["layer.0.w_dt_out"], weights["layer.0.w_dt_out"][:, 8:12])
    np.testing.assert_array_equal(sharded["layer.0.w_out"], weights["layer.0.w_out"][8:12, :])
    assert sharded["_d_inner"] == 4


def test_mamba_tp_validation_rejects_non_divisible_inner_dim():
    with pytest.raises(ValueError, match="d_inner divisible"):
        tp_builder._validate_mamba_tp(
            _weights(d_inner=18), ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0))


def test_mamba_tp_validation_requires_concrete_rank():
    with pytest.raises(ValueError, match="concrete rank"):
        tp_builder._validate_mamba_tp(
            _weights(), ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1))


def test_mamba_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"mamba-tp-plan"

    monkeypatch.setattr(
        mamba_plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_mamba_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = mamba_plugin_module.build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"mamba-tp-plan"
    assert calls["require"][0] == parallel
    assert "Mamba tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

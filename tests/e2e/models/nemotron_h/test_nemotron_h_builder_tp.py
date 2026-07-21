# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Nemotron-H tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    nemotron_h_module = importlib.import_module(
        "tensorrt_model_connect.families.nemotron_h.plugin")
    from tensorrt_model_connect.families.nemotron_h.model import parallel as tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_heads: int = 4, num_kv_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=3,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=2,
        intermediate_size=16,
        rms_norm_eps=1e-5,
    )


def _weights() -> dict:
    hidden = 8
    d_inner = 16
    d_state = 2
    d_conv = 3
    n_groups = 4
    mamba_heads = 4
    conv_dim = d_inner + 2 * n_groups * d_state
    proj_dim = d_inner + conv_dim + mamba_heads
    weights: dict[str, object] = {
        "_layer_types": ["mamba2", "mlp", "attention"],
        "_d_inner": d_inner,
        "_d_state": d_state,
        "_d_conv": d_conv,
        "_conv_dim": conv_dim,
        "_mamba_num_heads": mamba_heads,
        "_mamba_head_dim": 4,
        "_n_groups": n_groups,
        "_num_mamba_layers": 1,
        "_num_attention_layers": 1,
        "_attention_size": 8,
        "_mlp_size": 16,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_lm_head": np.zeros((hidden, 32), dtype=np.float32),
    }
    weights["layer.0.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.0.mamba_in_proj"] = np.arange(
        hidden * proj_dim, dtype=np.float32).reshape(hidden, proj_dim)
    weights["layer.0.conv1d_weight"] = np.arange(
        conv_dim * d_conv, dtype=np.float32).reshape(conv_dim, d_conv)
    weights["layer.0.conv1d_bias"] = np.arange(conv_dim, dtype=np.float32)
    weights["layer.0.dt_bias"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.A"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.D"] = np.arange(mamba_heads, dtype=np.float32)
    weights["layer.0.mamba_norm"] = np.arange(d_inner, dtype=np.float32)
    weights["layer.0.mamba_out_proj"] = np.arange(
        d_inner * hidden, dtype=np.float32).reshape(d_inner, hidden)

    weights["layer.1.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.1.w_up"] = np.arange(hidden * 16, dtype=np.float32).reshape(hidden, 16)
    weights["layer.1.w_down"] = np.arange(16 * hidden, dtype=np.float32).reshape(16, hidden)

    weights["layer.2.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights["layer.2.w_q"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_k"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_v"] = np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.2.w_o"] = np.arange(8 * hidden, dtype=np.float32).reshape(8, hidden)
    return weights


def test_nemotron_h_tp_slices_mamba_mlp_and_attention_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()

    sharded = tp_builder.shard_nemotron_h_weights(
        _config(), weights, parallel=parallel)

    expected_in_proj = np.concatenate([
        weights["layer.0.mamba_in_proj"][:, 8:12],
        weights["layer.0.mamba_in_proj"][:, 24:28],
        weights["layer.0.mamba_in_proj"][:, 36:38],
        weights["layer.0.mamba_in_proj"][:, 44:46],
        weights["layer.0.mamba_in_proj"][:, 50:51],
    ], axis=-1)
    expected_conv = np.concatenate([
        weights["layer.0.conv1d_weight"][8:12, :],
        weights["layer.0.conv1d_weight"][20:22, :],
        weights["layer.0.conv1d_weight"][28:30, :],
    ], axis=0)

    np.testing.assert_array_equal(sharded["layer.0.mamba_in_proj"], expected_in_proj)
    np.testing.assert_array_equal(sharded["layer.0.conv1d_weight"], expected_conv)
    np.testing.assert_array_equal(sharded["layer.0.mamba_out_proj"], weights["layer.0.mamba_out_proj"][8:12, :])
    np.testing.assert_array_equal(sharded["layer.1.w_up"], weights["layer.1.w_up"][:, 8:12])
    np.testing.assert_array_equal(sharded["layer.1.w_down"], weights["layer.1.w_down"][8:12, :])
    np.testing.assert_array_equal(sharded["layer.2.w_q"], weights["layer.2.w_q"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.2.w_o"], weights["layer.2.w_o"][4:6, :])
    assert sharded["_d_inner"] == 4
    assert sharded["_conv_dim"] == 8
    assert sharded["_mamba_num_heads"] == 1
    assert sharded["_n_groups"] == 1
    assert sharded["_attention_size"] == 2
    assert sharded["_mlp_size"] == 4


def test_nemotron_h_tp_validation_rejects_non_divisible_kv_heads():
    with pytest.raises(ValueError, match="num_key_value_heads"):
        tp_builder._validate_nemotron_h_tp(
            _config(num_kv_heads=2),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_nemotron_h_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"nemotron-h-tp-plan"

    monkeypatch.setattr(
        nemotron_h_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_nemotron_h_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = nemotron_h_module.NemotronHPlugin().build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"nemotron-h-tp-plan"
    assert calls["require"][0] == parallel
    assert "Nemotron-H tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

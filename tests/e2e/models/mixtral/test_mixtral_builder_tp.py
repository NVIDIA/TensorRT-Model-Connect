# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Mixtral tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    mixtral_module = importlib.import_module(
        "tensorrt_model_connect.families.mixtral.model")
    from tensorrt_model_connect.families.mixtral import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_kv_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,
        head_dim=2,
        attention_size=8,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _weights() -> dict:
    hidden = 8
    inter = 16
    weights: dict[str, object] = {
        "_attention_size": 8,
        "_num_experts": 2,
        "_moe_intermediate_size": inter,
        "_num_experts_per_tok": 2,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
        "layer.0.input_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.post_attn_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.router": np.zeros((hidden, 2), dtype=np.float32),
    }
    for key in ("w_q", "w_k", "w_v"):
        weights[f"layer.0.{key}"] = np.arange(
            hidden * 8, dtype=np.float32).reshape(hidden, 8)
    weights["layer.0.w_o"] = np.arange(8 * hidden, dtype=np.float32).reshape(8, hidden)
    for expert_idx in range(2):
        prefix = f"layer.0.expert.{expert_idx}"
        weights[f"{prefix}.w_gate"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
        weights[f"{prefix}.w_up"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
        weights[f"{prefix}.w_down"] = np.arange(
            inter * hidden, dtype=np.float32).reshape(inter, hidden)
    return weights


def test_mixtral_tp_slices_attention_and_expert_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()

    sharded = tp_builder.shard_mixtral_weights(_config(), weights, parallel=parallel)

    np.testing.assert_array_equal(sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 4:6])
    np.testing.assert_array_equal(sharded["layer.0.w_o"], weights["layer.0.w_o"][4:6, :])
    np.testing.assert_array_equal(
        sharded["layer.0.expert.0.w_gate"],
        weights["layer.0.expert.0.w_gate"][:, 8:12],
    )
    np.testing.assert_array_equal(
        sharded["layer.0.expert.0.w_down"],
        weights["layer.0.expert.0.w_down"][8:12, :],
    )
    assert sharded["_attention_size"] == 2
    assert sharded["_moe_intermediate_size"] == 4


def test_mixtral_tp_validation_rejects_non_divisible_kv_heads():
    with pytest.raises(ValueError, match="num_key_value_heads"):
        tp_builder._validate_mixtral_tp(
            _config(num_kv_heads=2),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_mixtral_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"mixtral-tp-plan"

    monkeypatch.setattr(
        mixtral_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_mixtral_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = mixtral_module.build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"mixtral-tp-plan"
    assert calls["require"][0] == parallel
    assert "Mixtral tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

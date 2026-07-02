# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Qwen-MoE tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    qwen_moe_module = importlib.import_module(
        "tensorrt_model_connect.families.qwen_moe.plugin")
    from tensorrt_model_connect.families.qwen_moe import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def test_direct_plugin_module_import_preserves_package_plugin_api():
    from tensorrt_model_connect.families import qwen_moe

    assert qwen_moe.plugin is qwen_moe_module.plugin


def _config(num_kv_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
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
        "_num_experts_per_tok": 2,
        "_moe_intermediate_size": inter,
        "_shared_expert_intermediate_size": inter,
        "_dense_intermediate_size": inter,
        "_mlp_only_layers": [1],
        "_has_shared_expert": True,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
    }
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.q_norm"] = np.arange(8, dtype=np.float32)
        weights[f"{prefix}.k_norm"] = np.arange(8, dtype=np.float32)
        for key in ("w_q", "w_k", "w_v"):
            weights[f"{prefix}.{key}"] = np.arange(
                hidden * 8, dtype=np.float32).reshape(hidden, 8)
        weights[f"{prefix}.w_o"] = np.arange(
            8 * hidden, dtype=np.float32).reshape(8, hidden)

    weights["layer.0.router"] = np.zeros((hidden, 2), dtype=np.float32)
    weights["layer.0.experts.w_gate"] = np.arange(
        2 * hidden * inter, dtype=np.float32).reshape(2, hidden, inter)
    weights["layer.0.experts.w_up"] = np.arange(
        2 * hidden * inter, dtype=np.float32).reshape(2, hidden, inter)
    weights["layer.0.experts.w_down"] = np.arange(
        2 * inter * hidden, dtype=np.float32).reshape(2, inter, hidden)
    for key in ("w_gate", "w_up"):
        weights[f"layer.0.shared_expert.{key}"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
    weights["layer.0.shared_expert.w_down"] = np.arange(
        inter * hidden, dtype=np.float32).reshape(inter, hidden)
    weights["layer.0.shared_expert_gate"] = np.zeros((1, hidden), dtype=np.float32)

    for key in ("w_gate", "w_up"):
        weights[f"layer.1.{key}"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
    weights["layer.1.w_down"] = np.arange(
        inter * hidden, dtype=np.float32).reshape(inter, hidden)
    return weights


def test_qwen_moe_tp_slices_attention_norms_dense_and_packed_experts():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()

    sharded = tp_builder.shard_qwen_moe_weights(
        _config(), weights, parallel=parallel)

    np.testing.assert_array_equal(
        sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 4:6])
    np.testing.assert_array_equal(
        sharded["layer.0.w_o"], weights["layer.0.w_o"][4:6, :])
    np.testing.assert_array_equal(
        sharded["layer.0.q_norm"], weights["layer.0.q_norm"][4:6])
    np.testing.assert_array_equal(
        sharded["layer.0.experts.w_gate"],
        weights["layer.0.experts.w_gate"][:, :, 8:12],
    )
    np.testing.assert_array_equal(
        sharded["layer.0.experts.w_down"],
        weights["layer.0.experts.w_down"][:, 8:12, :],
    )
    np.testing.assert_array_equal(
        sharded["layer.0.shared_expert.w_gate"],
        weights["layer.0.shared_expert.w_gate"][:, 8:12],
    )
    np.testing.assert_array_equal(
        sharded["layer.1.w_down"], weights["layer.1.w_down"][8:12, :])
    assert sharded["_attention_size"] == 2
    assert sharded["_moe_intermediate_size"] == 4
    assert sharded["_dense_intermediate_size"] == 4
    assert sharded["_shared_expert_intermediate_size"] == 4


def test_qwen_moe_tp_validation_rejects_non_divisible_attention_heads():
    cfg = _config()
    cfg.num_attention_heads = 3
    with pytest.raises(ValueError, match="num_attention_heads"):
        tp_builder._validate_qwen_moe_tp(
            cfg,
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        )


def test_qwen_moe_tp_replicates_non_divisible_kv_heads():
    weights = _weights()
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.w_k"] = np.arange(
            8 * 2, dtype=np.float32).reshape(8, 2)
        weights[f"{prefix}.w_v"] = np.arange(
            8 * 2, dtype=np.float32).reshape(8, 2)
        weights[f"{prefix}.k_norm"] = np.arange(2, dtype=np.float32)

    sharded = tp_builder.shard_qwen_moe_weights(
        _config(num_kv_heads=1),
        weights,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1),
    )

    np.testing.assert_array_equal(sharded["layer.0.w_k"], weights["layer.0.w_k"])
    np.testing.assert_array_equal(
        sharded["layer.0.k_norm"], weights["layer.0.k_norm"])
    assert sharded["_attention_size"] == 4
    assert sharded["_kv_attention_size"] == 2
    assert sharded["_num_key_value_heads"] == 1


def test_qwen_moe_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"qwen-moe-tp-plan"

    monkeypatch.setattr(
        qwen_moe_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_qwen_moe_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = qwen_moe_module.Qwen3MoePlugin().build_engine(
        _config(), _weights(), 17,
        precision="fp16",
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"qwen-moe-tp-plan"
    assert calls["require"][0] == parallel
    assert "Qwen-MoE tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["precision"] == "fp16"
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

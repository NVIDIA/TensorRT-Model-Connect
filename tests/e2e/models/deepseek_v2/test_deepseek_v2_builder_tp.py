# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for DeepSeek-V2 tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    deepseek_module = importlib.import_module(
        "tensorrt_model_connect.families.deepseek_v2.plugin")
    from tensorrt_model_connect.families.deepseek_v2 import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=4,
        attention_size=16,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _weights() -> dict:
    hidden = 8
    q_lora = 8
    kv_lora = 8
    heads = 4
    nope = 2
    rope = 2
    v_dim = 2
    inter = 16
    shared = 32
    weights: dict[str, object] = {
        "_attention_size": heads * (nope + rope),
        "_qk_nope_head_dim": nope,
        "_qk_rope_head_dim": rope,
        "_v_head_dim": v_dim,
        "_kv_lora_rank": kv_lora,
        "_q_lora_rank": q_lora,
        "_n_routed_experts": 2,
        "_n_shared_experts": 2,
        "_num_experts_per_tok": 2,
        "_first_k_dense_replace": 1,
        "_moe_layer_freq": 1,
        "_moe_intermediate_size": inter,
        "_shared_intermediate_size": shared,
        "_norm_topk_prob": False,
        "_routed_scaling_factor": 1.0,
        "_scoring_func": "softmax",
        "_topk_method": "greedy",
        "_n_group": 1,
        "_topk_group": 1,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
    }
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.w_q_a"] = np.arange(
            hidden * q_lora, dtype=np.float32).reshape(hidden, q_lora)
        weights[f"{prefix}.q_a_norm"] = np.ones((q_lora,), dtype=np.float32)
        weights[f"{prefix}.w_q_b"] = np.arange(
            q_lora * heads * (nope + rope), dtype=np.float32
        ).reshape(q_lora, heads * (nope + rope))
        weights[f"{prefix}.w_kv_a"] = np.arange(
            hidden * (kv_lora + rope), dtype=np.float32
        ).reshape(hidden, kv_lora + rope)
        weights[f"{prefix}.kv_a_norm"] = np.ones((kv_lora,), dtype=np.float32)
        weights[f"{prefix}.w_kv_b"] = np.arange(
            kv_lora * heads * (nope + v_dim), dtype=np.float32
        ).reshape(kv_lora, heads * (nope + v_dim))
        weights[f"{prefix}.w_o"] = np.arange(
            heads * v_dim * hidden, dtype=np.float32).reshape(heads * v_dim, hidden)

    for key in ("w_gate", "w_up"):
        weights[f"layer.0.{key}"] = np.arange(
            hidden * inter, dtype=np.float32).reshape(hidden, inter)
    weights["layer.0.w_down"] = np.arange(
        inter * hidden, dtype=np.float32).reshape(inter, hidden)

    weights["layer.1.router"] = np.zeros((hidden, 2), dtype=np.float32)
    for expert_idx in range(2):
        prefix = f"layer.1.expert.{expert_idx}"
        for key in ("w_gate", "w_up"):
            weights[f"{prefix}.{key}"] = np.arange(
                hidden * inter, dtype=np.float32).reshape(hidden, inter)
        weights[f"{prefix}.w_down"] = np.arange(
            inter * hidden, dtype=np.float32).reshape(inter, hidden)
    for key in ("w_gate", "w_up"):
        weights[f"layer.1.shared.{key}"] = np.arange(
            hidden * shared, dtype=np.float32).reshape(hidden, shared)
    weights["layer.1.shared.w_down"] = np.arange(
        shared * hidden, dtype=np.float32).reshape(shared, hidden)
    return weights


def test_deepseek_v2_tp_slices_mla_and_mlp_weights():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()

    sharded = tp_builder.shard_deepseek_v2_weights(
        _config(), weights, parallel=parallel)

    np.testing.assert_array_equal(
        sharded["layer.0.w_q_b"], weights["layer.0.w_q_b"][:, 8:12])
    np.testing.assert_array_equal(
        sharded["layer.0.w_kv_b"], weights["layer.0.w_kv_b"][:, 8:12])
    np.testing.assert_array_equal(
        sharded["layer.0.w_kv_a"], weights["layer.0.w_kv_a"])
    np.testing.assert_array_equal(
        sharded["layer.0.w_o"], weights["layer.0.w_o"][4:6, :])
    np.testing.assert_array_equal(
        sharded["layer.0.w_gate"], weights["layer.0.w_gate"][:, 8:12])
    np.testing.assert_array_equal(
        sharded["layer.1.expert.0.w_down"],
        weights["layer.1.expert.0.w_down"][8:12, :],
    )
    np.testing.assert_array_equal(
        sharded["layer.1.shared.w_gate"],
        weights["layer.1.shared.w_gate"][:, 16:24],
    )
    assert sharded["_attention_size"] == 4
    assert sharded["_moe_intermediate_size"] == 4
    assert sharded["_shared_intermediate_size"] == 8


def test_deepseek_v2_tp_validation_rejects_non_divisible_attention_heads():
    with pytest.raises(ValueError, match="num_attention_heads"):
        tp_builder._validate_deepseek_v2_tp(
            _config(num_heads=3),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        )


def test_deepseek_v2_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"deepseek-v2-tp-plan"

    monkeypatch.setattr(
        deepseek_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_deepseek_v2_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    result = deepseek_module.DeepSeekV2Plugin().build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"deepseek-v2-tp-plan"
    assert calls["require"][0] == parallel
    assert "DeepSeek-V2 tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

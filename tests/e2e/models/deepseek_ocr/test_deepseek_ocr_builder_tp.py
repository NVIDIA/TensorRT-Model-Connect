# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for DeepSeek-OCR tensor-parallel support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    deepseek_ocr_module = importlib.import_module(
        "tensorrt_model_connect.families.deepseek_ocr.plugin")
    from tensorrt_model_connect.checkpoint_mapper import WeightDict
    from tensorrt_model_connect.families.deepseek_ocr import tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config(num_heads: int = 4, num_kv_heads: int | None = None) -> SimpleNamespace:
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    return SimpleNamespace(
        raw={},
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=num_heads,
        num_key_value_heads=kv_heads,
        head_dim=4,
        attention_size=num_heads * 4,
        intermediate_size=16,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _weights() -> WeightDict:
    hidden = 8
    attention = 16
    kv_attention = 16
    inter = 16
    shared = 32
    weights = WeightDict({
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_n_routed_experts": 2,
        "_n_shared_experts": 2,
        "_num_experts_per_tok": 2,
        "_first_k_dense_replace": 1,
        "_moe_intermediate_size": inter,
        "_shared_intermediate_size": shared,
        "_norm_topk_prob": False,
        "_routed_scaling_factor": 1.0,
        "embedding": np.zeros((32, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, 32), dtype=np.float32),
    })
    for layer_idx in range(2):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
        weights[f"{prefix}.w_q"] = np.arange(
            hidden * attention, dtype=np.float32).reshape(hidden, attention)
        weights[f"{prefix}.w_k"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_v"] = np.arange(
            hidden * kv_attention, dtype=np.float32).reshape(hidden, kv_attention)
        weights[f"{prefix}.w_o"] = np.arange(
            attention * hidden, dtype=np.float32).reshape(attention, hidden)

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


def test_deepseek_ocr_tp_slices_attention_and_moe_weights() -> None:
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    weights = _weights()

    sharded = tp_builder.shard_deepseek_ocr_weights(
        _config(), weights, parallel=parallel)

    np.testing.assert_array_equal(
        sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_k"], weights["layer.0.w_k"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_v"], weights["layer.0.w_v"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_o"], weights["layer.0.w_o"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.0.w_gate"], weights["layer.0.w_gate"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.0.w_down"], weights["layer.0.w_down"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.expert.0.w_up"],
        weights["layer.1.expert.0.w_up"][:, 8:16])
    np.testing.assert_array_equal(
        sharded["layer.1.expert.0.w_down"],
        weights["layer.1.expert.0.w_down"][8:16, :])
    np.testing.assert_array_equal(
        sharded["layer.1.shared.w_gate"],
        weights["layer.1.shared.w_gate"][:, 16:32])
    assert sharded["_attention_size"] == 8
    assert sharded["_kv_attention_size"] == 8
    assert sharded["_moe_intermediate_size"] == 8
    assert sharded["_shared_intermediate_size"] == 16


def test_deepseek_ocr_tp_validation_rejects_non_divisible_attention_heads() -> None:
    with pytest.raises(ValueError, match="num_attention_heads"):
        tp_builder._validate_deepseek_ocr_tp(
            _config(num_heads=10),
            _weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_deepseek_ocr_plugin_routes_parallel_builds(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"deepseek-ocr-tp-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "require_tensorrt_11_for_tensor_parallel",
        fake_require,
    )
    monkeypatch.setattr(tp_builder, "build_deepseek_ocr_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1)
    plugin = deepseek_ocr_module.DeepSeekOCRPlugin()
    result = plugin.build_engine(
        _config(), _weights(), 4096,
        verbose=True,
        parallel_config=parallel,
    )

    assert result == b"deepseek-ocr-tp-plan"
    assert calls["require"][0] == parallel
    assert "DeepSeek-OCR tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 4096
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True


def test_deepseek_ocr_parallel_build_rejects_debug_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        deepseek_ocr_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="debug layer outputs"):
        deepseek_ocr_module.DeepSeekOCRPlugin().build_engine(
            _config(),
            _weights(),
            max_cache_length=4096,
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
            debug_layer_outputs=True,
        )


def test_deepseek_ocr_forwards_fp16_to_vision_engine(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build(model_dir, config, *, precision, verbose):
        calls.update(
            model_dir=model_dir,
            config=config,
            precision=precision,
            verbose=verbose,
        )
        return b"vision-plan"

    monkeypatch.setattr(
        deepseek_ocr_module,
        "_build_deepseek_ocr_vision_engine",
        fake_build,
    )
    config = _config()

    plan = deepseek_ocr_module.DeepSeekOCRPlugin().build_vision_engine(
        "/model", config, _weights(), precision="fp16", verbose=True)

    assert plan == b"vision-plan"
    assert calls == {
        "model_dir": "/model",
        "config": config,
        "precision": "fp16",
        "verbose": True,
    }

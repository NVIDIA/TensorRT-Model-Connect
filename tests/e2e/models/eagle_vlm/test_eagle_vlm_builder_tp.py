# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Eagle VLM tensor-parallel text-backbone support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig


class _Config:
    raw = {}
    model_type = "llama_nemotron_vl"
    vocab_size = 32
    hidden_size = 16
    num_hidden_layers = 1
    num_attention_heads = 4
    num_key_value_heads = 4
    head_dim = 4
    attention_size = 16
    intermediate_size = 32


def _weights() -> WeightDict:
    hidden = _Config.hidden_size
    attention = _Config.attention_size
    mlp = _Config.intermediate_size
    return WeightDict({
        "embedding": np.zeros((_Config.vocab_size, hidden), dtype=np.float32),
        "layer.0.input_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.post_attn_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.w_q": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_k": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_v": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_o": np.zeros((attention, hidden), dtype=np.float32),
        "layer.0.w_gate": np.zeros((hidden, mlp), dtype=np.float32),
        "layer.0.w_up": np.zeros((hidden, mlp), dtype=np.float32),
        "layer.0.w_down": np.zeros((mlp, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "score_weight": np.zeros((hidden, 1), dtype=np.float32),
        "score_bias": np.zeros((1,), dtype=np.float32),
        "w_out": np.zeros((hidden, 1), dtype=np.float32),
        "_attention_size": attention,
        "_kv_attention_size": attention,
        "_mlp_size": mlp,
    })


def test_eagle_vlm_resolves_nested_legacy_rope_scaling() -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.model")

    class Config(_Config):
        raw = {
            "llm_config": {
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 32.0,
                },
            },
        }

    assert plugin_module._resolve_rope_scaling(Config()) == {
        "rope_type": "llama3",
        "factor": 32.0,
    }


def test_eagle_vlm_prefers_rope_parameters_over_legacy_alias() -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.model")

    class Config(_Config):
        raw = {
            "llm_config": {
                "rope_parameters": {"rope_type": "llama3", "factor": 8.0},
                "rope_scaling": {"rope_type": "llama3", "factor": 32.0},
            },
        }

    assert plugin_module._resolve_rope_scaling(Config())["factor"] == 8.0


def test_eagle_vlm_fp16_reranker_keeps_bounded_tail_in_fp32(
    monkeypatch,
) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.model")
    graph_blocks = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.graph_blocks")
    trt_compat = importlib.import_module("tensorrt_model_connect.trt_compat")
    trt = trt_compat.get_trt()
    original_apply_norm = graph_blocks.apply_norm
    assert plugin_module._RERANKER_FP32_TAIL_LAYERS == 16
    num_layers = plugin_module._RERANKER_FP32_TAIL_LAYERS + 2

    class FinalNormCaptured(RuntimeError):
        pass

    calls = []

    def capture_tail_norms(
        network,
        inp,
        hidden_size,
        gamma,
        beta,
        eps_tensor,
        norm_type,
        *,
        dtype,
        eps=None,
    ):
        calls.append({
            "input_dtype": inp.dtype,
            "eps_dtype": eps_tensor.dtype,
            "dtype": dtype,
        })
        if len(calls) == 2 * num_layers + 1:
            raise FinalNormCaptured
        return original_apply_norm(
            network,
            inp,
            hidden_size,
            gamma,
            beta,
            eps_tensor,
            norm_type,
            dtype=dtype,
            eps=eps,
        )

    monkeypatch.setattr(graph_blocks, "apply_norm", capture_tail_norms)

    class RerankerConfig(_Config):
        raw = {"is_reranker": True}
        num_hidden_layers = num_layers
        rms_norm_eps = 1e-5
        rope_theta = 10_000.0

    weights = _weights()
    layer_weight_names = (
        "input_norm",
        "post_attn_norm",
        "w_q",
        "w_k",
        "w_v",
        "w_o",
        "w_gate",
        "w_up",
        "w_down",
    )
    for layer_idx in range(1, num_layers):
        for name in layer_weight_names:
            weights[f"layer.{layer_idx}.{name}"] = (
                weights[f"layer.0.{name}"].copy())

    with pytest.raises(FinalNormCaptured):
        plugin_module._build_eagle_engine(
            RerankerConfig(),
            weights,
            max_cache_length=4,
            is_reranker=True,
            precision="fp16",
        )

    fp16_call = {
        "input_dtype": trt.float16,
        "eps_dtype": trt.float16,
        "dtype": np.float16,
    }
    fp32_call = {
        "input_dtype": trt.float32,
        "eps_dtype": trt.float32,
        "dtype": np.float32,
    }
    assert calls == [fp16_call] * 4 + [fp32_call] * 33


def test_eagle_vlm_tp_shards_text_backbone_weights() -> None:
    from tensorrt_model_connect.families.eagle_vlm import tp_builder

    sharded = tp_builder.shard_eagle_vlm_weights(
        _Config(),
        _weights(),
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    assert sharded["layer.0.w_q"].shape == (16, 4)
    assert sharded["layer.0.w_k"].shape == (16, 4)
    assert sharded["layer.0.w_v"].shape == (16, 4)
    assert sharded["layer.0.w_o"].shape == (4, 16)
    assert sharded["layer.0.w_gate"].shape == (16, 8)
    assert sharded["layer.0.w_up"].shape == (16, 8)
    assert sharded["layer.0.w_down"].shape == (8, 16)
    assert sharded["score_weight"].shape == (16, 1)
    assert sharded["_attention_size"] == 4
    assert sharded["_kv_attention_size"] == 4
    assert sharded["_mlp_size"] == 8
    assert sharded["_tensor_parallel_size"] == 4
    assert sharded["_tensor_parallel_rank"] == 2


def test_eagle_vlm_tp_builder_rejects_single_device_mode() -> None:
    from tensorrt_model_connect.families.eagle_vlm import tp_builder

    with pytest.raises(ValueError, match="requires tensor_parallel"):
        tp_builder.build_eagle_vlm_tp_engine(
            _Config(),
            _weights(),
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


def test_eagle_vlm_parallel_build_routes_to_tp_builder(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.model")
    from tensorrt_model_connect.families.eagle_vlm import tp_builder

    calls = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = {
            "config": config,
            "weights": weights,
            "max_cache_length": max_cache_length,
            "kwargs": kwargs,
        }
        return b"eagle-vlm-tp-plan"

    monkeypatch.setattr(
        plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_eagle_vlm_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = plugin_module.build_engine(
        _Config(),
        _weights(),
        max_cache_length=512,
        parallel_config=parallel,
    )

    assert result == b"eagle-vlm-tp-plan"
    assert calls["require"] == (parallel, "Eagle VLM tensor-parallel builds")
    assert calls["build"]["max_cache_length"] == 512
    assert calls["build"]["kwargs"]["parallel_config"] == parallel
    assert calls["build"]["kwargs"]["is_reranker"] is False


def test_eagle_vlm_parallel_build_preserves_reranking_mode(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.model")
    from tensorrt_model_connect.families.eagle_vlm import tp_builder

    calls = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["kwargs"] = kwargs
        return b"eagle-rerank-tp-plan"

    monkeypatch.setattr(
        plugin_module,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(tp_builder, "build_eagle_vlm_tp_engine", fake_build)

    class RerankConfig(_Config):
        raw = {"is_reranker": True}

    result = plugin_module.build_engine(
        RerankConfig(),
        _weights(),
        max_cache_length=256,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
    )

    assert result == b"eagle-rerank-tp-plan"
    assert calls["kwargs"]["is_reranker"] is True

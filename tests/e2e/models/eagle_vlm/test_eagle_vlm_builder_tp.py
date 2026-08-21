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
        "tensorrt_model_connect.families.eagle_vlm.plugin")

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
        "tensorrt_model_connect.families.eagle_vlm.plugin")

    class Config(_Config):
        raw = {
            "llm_config": {
                "rope_parameters": {"rope_type": "llama3", "factor": 8.0},
                "rope_scaling": {"rope_type": "llama3", "factor": 32.0},
            },
        }

    assert plugin_module._resolve_rope_scaling(Config())["factor"] == 8.0


def test_eagle_vlm_fp16_reranker_keeps_residual_and_norms_in_fp32(
    monkeypatch,
) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.plugin")
    graph_blocks = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.graph_blocks")
    graph_ops = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.graph_ops")
    trt_compat = importlib.import_module("tensorrt_model_connect.trt_compat")
    trt = trt_compat.get_trt()
    original_apply_norm = graph_blocks.apply_norm
    original_add_matmul = graph_ops.add_matmul_rhs_constant
    original_add_attention = graph_ops.add_attention_from_rows
    original_add_mlp = graph_blocks.add_swiglu_mlp
    num_layers = 6

    class FinalNormCaptured(RuntimeError):
        pass

    norm_calls = []
    matmul_dtypes = []
    matmul_input_dtypes = []
    matmul_fp32_compute = []
    attention_fp32 = []
    mlp_dtypes = []
    mlp_fp32_down_projection = []

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
        norm_calls.append({
            "input_dtype": inp.dtype,
            "eps_dtype": eps_tensor.dtype,
            "dtype": dtype,
        })
        if len(norm_calls) == 2 * num_layers + 1:
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

    def capture_matmul(*args, **kwargs):
        matmul_dtypes.append(kwargs["dtype"])
        matmul_input_dtypes.append(args[1].dtype)
        matmul_fp32_compute.append(kwargs.get("fp32_compute", False))
        return original_add_matmul(*args, **kwargs)

    def capture_attention(*args, **kwargs):
        attention_fp32.append(kwargs["fp32_accumulation"])
        return original_add_attention(*args, **kwargs)

    def capture_mlp(*args, **kwargs):
        mlp_dtypes.append(kwargs["dtype"])
        mlp_fp32_down_projection.append(kwargs["fp32_down_projection"])
        return original_add_mlp(*args, **kwargs)

    monkeypatch.setattr(graph_blocks, "apply_norm", capture_tail_norms)
    monkeypatch.setattr(graph_ops, "add_matmul_rhs_constant", capture_matmul)
    monkeypatch.setattr(graph_ops, "add_attention_from_rows", capture_attention)
    monkeypatch.setattr(graph_blocks, "add_swiglu_mlp", capture_mlp)

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

    fp32_call = {
        "input_dtype": trt.float32,
        "eps_dtype": trt.float32,
        "dtype": np.float32,
    }
    assert norm_calls == [fp32_call] * (2 * num_layers + 1)
    expected_fp32_down = [False] * (num_layers - 4) + [True] * 4
    expected_matmul_dtypes = []
    expected_matmul_input_dtypes = []
    for fp32_down in expected_fp32_down:
        expected_matmul_dtypes.extend(
            [np.float16] * 6 + [np.float32 if fp32_down else np.float16])
        expected_matmul_input_dtypes.extend(
            [trt.float16] * 6 + [trt.float32 if fp32_down else trt.float16])
    assert matmul_dtypes == expected_matmul_dtypes
    assert matmul_input_dtypes == expected_matmul_input_dtypes
    assert matmul_fp32_compute == ([True] * 3 + [False] * 4) * num_layers
    assert attention_fp32 == [True] * num_layers
    assert mlp_dtypes == [np.float16] * num_layers
    assert mlp_fp32_down_projection == expected_fp32_down


def test_eagle_vlm_reranker_executes_actual_sequence_length() -> None:
    import tensorrt as trt

    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.eagle_vlm.plugin")

    class RerankerConfig(_Config):
        raw = {"is_reranker": True}
        rms_norm_eps = 1e-5
        rope_theta = 10_000.0

    plan = plugin_module._build_eagle_engine(
        RerankerConfig(),
        _weights(),
        max_cache_length=32,
        is_reranker=True,
        precision="fp16",
    )
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)

    assert engine is not None
    assert tuple(engine.get_tensor_shape("input_ids")) == (-1, -1)
    assert tuple(engine.get_tensor_shape("attention_mask")) == (-1, -1)
    assert tuple(
        tuple(shape) for shape in engine.get_tensor_profile_shape("input_ids", 0)
    ) == ((1, 1), (2, 32), (2, 32))
    assert tuple(
        tuple(shape) for shape in engine.get_tensor_profile_shape("attention_mask", 0)
    ) == ((1, 1), (2, 32), (2, 32))
    io_names = {engine.get_tensor_name(index) for index in range(engine.num_io_tensors)}
    assert "input_embed" not in io_names
    assert "use_input_embed" not in io_names


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
        "tensorrt_model_connect.families.eagle_vlm.plugin")
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
    result = plugin_module.plugin.build_engine(
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
        "tensorrt_model_connect.families.eagle_vlm.plugin")
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

    result = plugin_module.plugin.build_engine(
        RerankConfig(),
        _weights(),
        max_cache_length=256,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
    )

    assert result == b"eagle-rerank-tp-plan"
    assert calls["kwargs"]["is_reranker"] is True

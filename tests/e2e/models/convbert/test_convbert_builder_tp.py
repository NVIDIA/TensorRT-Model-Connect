# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel tests for ConvBERT encoder support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.convbert import model as ConvBertModel
from tensorrt_model_connect.parallel_config import ParallelConfig

convbert_plugin = importlib.import_module(
    "tensorrt_model_connect.families.convbert.model")

_LAYERS = 1
_HIDDEN = 16
_HEADS = 4
_EFFECTIVE_HEADS = 2
_HEAD_SIZE = 4
_ALL_HEAD = _EFFECTIVE_HEADS * _HEAD_SIZE
_MLP = 32
_VOCAB = 24
_KERNEL = 3


def _convbert_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.families.convbert.tp_builder",
        reason="TensorRT is required for ConvBERT TP builder tests",
    )


def _make_config(
    *,
    hidden: int = _HIDDEN,
    heads: int = _HEADS,
    layers: int = _LAYERS,
    mlp: int = _MLP,
) -> ModelConfig:
    return ModelConfig(
        model_type="convbert",
        vocab_size=_VOCAB,
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        _head_dim=hidden // heads,
        raw={"head_ratio": 2, "conv_kernel_size": _KERNEL},
    )


def _matrix(rows: int, cols: int, offset: int = 0) -> np.ndarray:
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) + offset


def _make_encoder_weights(
    *,
    hidden: int = _HIDDEN,
    layers: int = _LAYERS,
    mlp: int = _MLP,
    vocab: int = _VOCAB,
) -> WeightDict:
    weights = WeightDict({
        "embedding": _matrix(vocab, hidden),
        "position_embedding": _matrix(32, hidden, 1000),
        "token_type_embedding": np.zeros((2, hidden), dtype=np.float32),
        "embed_norm": np.ones(hidden, dtype=np.float32),
        "embed_norm_beta": np.zeros(hidden, dtype=np.float32),
        "_convbert_new_num_heads": np.array([_EFFECTIVE_HEADS], dtype=np.int32),
        "_convbert_head_size": np.array([_HEAD_SIZE], dtype=np.int32),
        "_convbert_all_head_size": np.array([_ALL_HEAD], dtype=np.int32),
        "_convbert_conv_kernel_size": np.array([_KERNEL], dtype=np.int32),
    })
    for layer_idx in range(layers):
        prefix = f"layer.{layer_idx}"
        weights[f"{prefix}.w_q"] = _matrix(hidden, _ALL_HEAD, 10)
        weights[f"{prefix}.w_k"] = _matrix(hidden, _ALL_HEAD, 20)
        weights[f"{prefix}.w_v"] = _matrix(hidden, _ALL_HEAD, 30)
        weights[f"{prefix}.q_bias"] = np.arange(_ALL_HEAD, dtype=np.float32)
        weights[f"{prefix}.k_bias"] = np.arange(_ALL_HEAD, dtype=np.float32) + 10
        weights[f"{prefix}.v_bias"] = np.arange(_ALL_HEAD, dtype=np.float32) + 20
        weights[f"{prefix}.sep_conv_dw"] = _matrix(hidden, _KERNEL, 40)
        weights[f"{prefix}.sep_conv_pw"] = _matrix(_ALL_HEAD, hidden, 50)
        weights[f"{prefix}.sep_conv_bias"] = np.arange(_ALL_HEAD, dtype=np.float32)
        weights[f"{prefix}.conv_kernel_w"] = _matrix(_ALL_HEAD, _EFFECTIVE_HEADS * _KERNEL, 60)
        weights[f"{prefix}.conv_kernel_bias"] = np.arange(_EFFECTIVE_HEADS * _KERNEL, dtype=np.float32)
        weights[f"{prefix}.conv_out_w"] = _matrix(hidden, _ALL_HEAD, 70)
        weights[f"{prefix}.conv_out_bias"] = np.arange(_ALL_HEAD, dtype=np.float32)
        weights[f"{prefix}.w_o"] = _matrix(2 * _ALL_HEAD, hidden, 80)
        weights[f"{prefix}.o_bias"] = np.arange(hidden, dtype=np.float32) + 30
        weights[f"{prefix}.post_attn_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.post_attn_norm_beta"] = np.zeros(hidden, dtype=np.float32)
        weights[f"{prefix}.w_fc1"] = _matrix(hidden, mlp, 90)
        weights[f"{prefix}.fc1_bias"] = np.arange(mlp, dtype=np.float32)
        weights[f"{prefix}.w_fc2"] = _matrix(mlp, hidden, 100)
        weights[f"{prefix}.fc2_bias"] = np.arange(hidden, dtype=np.float32) + 40
        weights[f"{prefix}.output_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.output_norm_beta"] = np.zeros(hidden, dtype=np.float32)
    return weights


def test_convbert_tp_builder_rejects_single_device_mode():
    tp_builder = _convbert_tp_builder_module()

    with pytest.raises(ValueError, match="requires tensor_parallel mode"):
        tp_builder.build_tp_convbert_encoder_engine(
            _make_config(),
            _make_encoder_weights(),
            max_seq_length=8,
            parallel_config=ParallelConfig(),
        )


def test_convbert_tp_validation_rejects_tp4_for_effective_heads():
    tp_builder = _convbert_tp_builder_module()

    with pytest.raises(ValueError, match="effective attention heads divisible"):
        tp_builder._validate_convbert_tp(
            _make_config(),
            _make_encoder_weights(),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_convbert_tp_shards_rank_local_encoder_weights():
    tp_builder = _convbert_tp_builder_module()
    config = _make_config()
    weights = _make_encoder_weights()
    shard = tp_builder.shard_convbert_weights(
        config,
        weights,
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1),
    )

    all_start = _ALL_HEAD // 2
    all_end = _ALL_HEAD
    mlp_start = _MLP // 2
    mlp_end = _MLP

    assert isinstance(shard, WeightDict)
    assert shard["_tensor_parallel_size"] == 2
    assert shard["_tensor_parallel_rank"] == 1
    assert int(shard["_convbert_new_num_heads"][0]) == _EFFECTIVE_HEADS // 2
    assert int(shard["_convbert_full_new_num_heads"][0]) == _EFFECTIVE_HEADS
    assert shard["embedding"] is weights["embedding"]

    np.testing.assert_array_equal(
        shard["layer.0.w_q"],
        weights["layer.0.w_q"][:, all_start:all_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.sep_conv_pw"],
        weights["layer.0.sep_conv_pw"][all_start:all_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.conv_kernel_w"],
        weights["layer.0.conv_kernel_w"][all_start:all_end, :],
    )
    np.testing.assert_array_equal(
        shard["layer.0.conv_out_w"],
        weights["layer.0.conv_out_w"][:, all_start:all_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_o"],
        np.concatenate([
            weights["layer.0.w_o"][all_start:all_end, :],
            weights["layer.0.w_o"][_ALL_HEAD + all_start:2 * _ALL_HEAD, :],
        ], axis=0),
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_fc1"],
        weights["layer.0.w_fc1"][:, mlp_start:mlp_end],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_fc2"],
        weights["layer.0.w_fc2"][mlp_start:mlp_end, :],
    )


def test_convbert_plugin_routes_tp_build(monkeypatch):
    tp_builder = _convbert_tp_builder_module()
    captured = {}

    def fake_build(config, weights, max_seq_length, **kwargs):
        captured["config"] = config
        captured["weights"] = weights
        captured["max_seq_length"] = max_seq_length
        captured["kwargs"] = kwargs
        return b"convbert-tp-plan"

    monkeypatch.setattr(
        convbert_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )
    monkeypatch.setattr(tp_builder, "build_tp_convbert_encoder_engine", fake_build)

    plan = ConvBertModel.build_engine(
        _make_config(),
        _make_encoder_weights(),
        max_cache_length=8,
        parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1),
    )

    assert plan == b"convbert-tp-plan"
    assert captured["config"].model_type == "convbert"
    assert captured["max_seq_length"] == 8
    assert captured["kwargs"]["parallel_config"].tp_size == 2
    assert captured["kwargs"]["parallel_config"].rank == 1


def test_convbert_plugin_rejects_quantized_tp(monkeypatch):
    monkeypatch.setattr(
        convbert_plugin,
        "require_tensorrt_11_for_tensor_parallel",
        lambda parallel, *, feature: None,
    )

    with pytest.raises(ValueError, match="do not support quantization"):
        ConvBertModel.build_engine(
            _make_config(),
            _make_encoder_weights(),
            max_cache_length=8,
            quant_ctx=object(),
            parallel_config=ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        )

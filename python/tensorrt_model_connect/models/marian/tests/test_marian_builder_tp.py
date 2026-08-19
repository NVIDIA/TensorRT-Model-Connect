# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Marian tensor-parallel decoder support."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    marian_plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.marian.model")
    from tensorrt_model_connect.models.marian import decoder_tp_builder
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _weights(dec_heads: int = 8, dec_ffn: int = 2048) -> dict:
    hidden = 512
    weights: dict[str, object] = {
        "_dec_layers": 1,
        "_dec_heads": dec_heads,
        "_dec_ffn": dec_ffn,
        "_max_position_embeddings": 128,
        "dec_embedding": np.zeros((128, hidden), dtype=np.float32),
        "dec_pos_embedding": np.zeros((128, hidden), dtype=np.float32),
        "w_out": np.zeros((hidden, 128), dtype=np.float32),
        "final_logits_bias": np.zeros((128,), dtype=np.float32),
    }
    prefix = "layer.0"
    for key in ("w_q", "w_k", "w_v", "cross_w_q", "cross_w_k", "cross_w_v"):
        weights[f"{prefix}.{key}"] = np.zeros((hidden, hidden), dtype=np.float32)
    for key in ("q_bias", "k_bias", "v_bias", "cross_b_q", "cross_b_k", "cross_b_v"):
        weights[f"{prefix}.{key}"] = np.zeros((hidden,), dtype=np.float32)
    for key in ("w_o", "cross_w_o"):
        weights[f"{prefix}.{key}"] = np.zeros((hidden, hidden), dtype=np.float32)
    for key in ("o_bias", "cross_b_o"):
        weights[f"{prefix}.{key}"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.input_norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.input_norm_beta"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.cross_attn_norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.cross_attn_norm_beta"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.w_fc1"] = np.zeros((hidden, dec_ffn), dtype=np.float32)
    weights[f"{prefix}.fc1_bias"] = np.zeros((dec_ffn,), dtype=np.float32)
    weights[f"{prefix}.w_fc2"] = np.zeros((dec_ffn, hidden), dtype=np.float32)
    weights[f"{prefix}.fc2_bias"] = np.zeros((hidden,), dtype=np.float32)
    weights[f"{prefix}.post_attn_norm"] = np.ones((hidden,), dtype=np.float32)
    weights[f"{prefix}.post_attn_norm_beta"] = np.zeros((hidden,), dtype=np.float32)
    return weights


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        raw={},
        hidden_size=512,
        vocab_size=128,
        rms_norm_eps=1e-5,
    )


def test_marian_tp_slices_projection_columns_rows_and_biases():
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    weights = _weights()
    weights["layer.0.w_q"] = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    weights["layer.0.q_bias"] = np.arange(512, dtype=np.float32)
    weights["layer.0.w_o"] = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    weights["layer.0.o_bias"] = np.arange(512, dtype=np.float32)

    sharded = decoder_tp_builder.shard_marian_decoder_weights(
        weights, parallel=parallel)

    np.testing.assert_array_equal(sharded["layer.0.w_q"], weights["layer.0.w_q"][:, 256:384])
    np.testing.assert_array_equal(sharded["layer.0.q_bias"], weights["layer.0.q_bias"][256:384])
    np.testing.assert_array_equal(sharded["layer.0.w_o"], weights["layer.0.w_o"][256:384, :])
    np.testing.assert_array_equal(sharded["layer.0.o_bias"], weights["layer.0.o_bias"])


def test_marian_tp_validation_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="decoder_attention_heads divisible"):
        decoder_tp_builder._validate_marian_tp(
            _config(),
            _weights(dec_heads=6),
            ParallelConfig(mode="tensor_parallel", tp_size=4, rank=0),
        )


def test_marian_tp_validation_requires_concrete_rank():
    with pytest.raises(ValueError, match="concrete rank"):
        decoder_tp_builder._validate_marian_tp(
            _config(), _weights(), ParallelConfig(mode="tensor_parallel", tp_size=4, rank=-1))


def test_marian_plugin_routes_parallel_builds(monkeypatch):
    calls: dict[str, object] = {}

    def fake_require(parallel, *, feature):
        calls["require"] = (parallel, feature)

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = (config, weights, max_cache_length, kwargs)
        return b"marian-tp-plan"

    monkeypatch.setattr(
        marian_plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(decoder_tp_builder, "build_marian_tp_decoder_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = marian_plugin_module.build_engine(
        _config(), _weights(), 17,
        verbose=True,
        debug_layer_outputs=True,
        parallel_config=parallel,
    )

    assert result == b"marian-tp-plan"
    assert calls["require"][0] == parallel
    assert "Marian tensor-parallel" in calls["require"][1]
    _, _, max_cache_length, kwargs = calls["build"]
    assert max_cache_length == 17
    assert kwargs["parallel_config"] == parallel
    assert kwargs["verbose"] is True
    assert kwargs["debug_layer_outputs"] is True

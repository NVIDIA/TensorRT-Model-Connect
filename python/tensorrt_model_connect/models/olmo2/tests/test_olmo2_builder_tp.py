# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for OLMo2 tensor-parallel support."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.parallel_config import ParallelConfig


class _Config:
    model_type = "olmo2"
    vocab_size = 32
    hidden_size = 16
    num_hidden_layers = 1
    num_attention_heads = 4
    num_key_value_heads = 4
    head_dim = 4
    attention_size = 16
    intermediate_size = 32
    rope_theta = 10000.0
    rms_norm_eps = 1e-6

    def __init__(self) -> None:
        self.raw = {}


def _weights() -> WeightDict:
    hidden = _Config.hidden_size
    attention = _Config.attention_size
    mlp = _Config.intermediate_size
    return WeightDict({
        "embedding": np.zeros((_Config.vocab_size, hidden), dtype=np.float32),
        "layer.0.post_attn_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.post_ff_norm": np.ones((hidden,), dtype=np.float32),
        "layer.0.w_q": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_k": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_v": np.zeros((hidden, attention), dtype=np.float32),
        "layer.0.w_o": np.zeros((attention, hidden), dtype=np.float32),
        "layer.0.q_norm": np.ones((attention,), dtype=np.float32),
        "layer.0.k_norm": np.ones((attention,), dtype=np.float32),
        "layer.0.w_gate": np.zeros((hidden, mlp), dtype=np.float32),
        "layer.0.w_up": np.zeros((hidden, mlp), dtype=np.float32),
        "layer.0.w_down": np.zeros((mlp, hidden), dtype=np.float32),
        "final_norm": np.ones((hidden,), dtype=np.float32),
        "w_out": np.zeros((hidden, _Config.vocab_size), dtype=np.float32),
        "_attention_size": attention,
        "_kv_attention_size": attention,
        "_mlp_size": mlp,
    })


def test_olmo2_tp_shards_post_norm_decoder_weights() -> None:
    from tensorrt_model_connect.models.olmo2 import tp_builder

    sharded = tp_builder.shard_olmo2_weights(
        _Config(),
        _weights(),
        parallel=ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2),
    )

    assert sharded["layer.0.w_q"].shape == (16, 4)
    assert sharded["layer.0.w_k"].shape == (16, 4)
    assert sharded["layer.0.w_v"].shape == (16, 4)
    assert sharded["layer.0.w_o"].shape == (4, 16)
    assert sharded["layer.0.q_norm"].shape == (4,)
    assert sharded["layer.0.k_norm"].shape == (4,)
    assert sharded["layer.0.w_gate"].shape == (16, 8)
    assert sharded["layer.0.w_up"].shape == (16, 8)
    assert sharded["layer.0.w_down"].shape == (8, 16)
    assert sharded["final_norm"].shape == (16,)
    assert sharded["w_out"].shape == (16, 32)
    assert sharded["_attention_size"] == 4
    assert sharded["_kv_attention_size"] == 4
    assert sharded["_mlp_size"] == 8
    assert sharded["_tensor_parallel_size"] == 4
    assert sharded["_tensor_parallel_rank"] == 2


def test_olmo2_tp_builder_rejects_single_device_mode() -> None:
    from tensorrt_model_connect.models.olmo2 import tp_builder

    with pytest.raises(ValueError, match="requires tensor_parallel"):
        tp_builder.build_olmo2_tp_engine(
            _Config(),
            _weights(),
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


def test_olmo2_plugin_routes_parallel_builds(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.olmo2.model")
    from tensorrt_model_connect.models.olmo2 import tp_builder

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
        return b"olmo2-tp-plan"

    monkeypatch.setattr(
        plugin_module, "require_tensorrt_11_for_tensor_parallel", fake_require)
    monkeypatch.setattr(tp_builder, "build_olmo2_tp_engine", fake_build)

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=1)
    result = plugin_module.build_engine(
        _Config(),
        _weights(),
        max_cache_length=256,
        parallel_config=parallel,
    )

    assert result == b"olmo2-tp-plan"
    assert calls["require"] == (parallel, "OLMo2 tensor-parallel builds")
    assert calls["build"]["max_cache_length"] == 256
    assert calls["build"]["kwargs"]["parallel_config"] == parallel


def test_olmo2_plugin_routes_split_prefill_builds(monkeypatch) -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.models.olmo2.model")
    prefill_builder = importlib.import_module(
        "tensorrt_model_connect.models.olmo2.prefill_builder")

    calls = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        calls["build"] = {
            "config": config,
            "weights": weights,
            "max_cache_length": max_cache_length,
            "kwargs": kwargs,
        }
        return b"olmo2-prefill-plan"

    monkeypatch.setattr(
        prefill_builder, "build_olmo2_prefill_engine", fake_build)

    config = _Config()
    config.raw["_decoder_engine_role"] = "prefill"
    result = plugin_module.build_engine(
        config,
        _weights(),
        max_cache_length=352,
        precision="fp32",
    )

    assert plugin_module.supports_split_decoder_roles is True
    assert result == b"olmo2-prefill-plan"
    assert calls["build"]["max_cache_length"] == 352
    assert calls["build"]["kwargs"] == {
        "precision": "fp32",
        "verbose": False,
    }

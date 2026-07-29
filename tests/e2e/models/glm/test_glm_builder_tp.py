# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned tensor-parallel rejection tests for native-KV GLM."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import (
    ParallelConfig,
)


FAMILY = 'glm'
PLUGIN_CLASS = 'GlmPlugin'
MODEL_TYPE = 'glm'
TP_SIZE = 2
RAW = {
    "attention_bias": True,
    "partial_rotary_factor": 0.5,
    "_decoder_engine_layout": "split",
    "_decoder_engine_role": "decode",
}


def _config(model_type: str, tp_size: int, raw: dict[str, object]) -> ModelConfig:
    kv_heads = 2 if tp_size == 2 else 4
    return ModelConfig(
        model_type=model_type,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        architectures=["GlmForCausalLM"],
        _head_dim=128,
        raw=raw,
    )


def test_glm_plugin_rejects_tp_without_fallback(monkeypatch) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")
    called = False

    def fake_build(*args, **kwargs):
        nonlocal called
        called = True
        return b"unexpected"

    monkeypatch.setattr(
        plugin_mod,
        "build_native_decoder_engine",
        fake_build,
    )

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=TP_SIZE, rank=1)
    with pytest.raises(ValueError, match="tensor parallel"):
        getattr(plugin_mod, PLUGIN_CLASS)().build_engine(
            _config(MODEL_TYPE, TP_SIZE, RAW),
            WeightDict(),
            max_cache_length=64,
            precision="bf16",
            verbose=True,
            parallel_config=parallel,
        )
    assert not called


def test_glm_plugin_rejects_quantized_tp(monkeypatch) -> None:
    plugin_mod = importlib.import_module(
        f"tensorrt_model_connect.families.{FAMILY}.plugin")

    with pytest.raises(ValueError, match="tensor parallel|quantized"):
        getattr(plugin_mod, PLUGIN_CLASS)().build_engine(
            _config(MODEL_TYPE, TP_SIZE, RAW),
            WeightDict(),
            max_cache_length=64,
            precision="bf16",
            quant_ctx=object(),
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=TP_SIZE, rank=0),
        )

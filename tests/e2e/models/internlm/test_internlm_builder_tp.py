# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternLM2 native KV rejects the removed tensor-parallel build route."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from tensorrt_model_connect.checkpoint_mapper import WeightDict
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.parallel_config import ParallelConfig


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="internlm2",
        architectures=["InternLM2ForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=1_000_000.0,
        hidden_act="silu",
        raw={
            "bias": False,
            "_decoder_engine_layout": "split",
            "_decoder_engine_role": "decode",
        },
    )


def test_internlm_native_kv_fails_closed_for_tensor_parallel() -> None:
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.internlm.plugin")

    with pytest.raises(ValueError, match="tensor parallel"):
        plugin_module.plugin.build_engine(
            _config(),
            WeightDict(),
            max_cache_length=128,
            precision="bf16",
            parallel_config=ParallelConfig(
                mode="tensor_parallel", tp_size=4, rank=0),
        )

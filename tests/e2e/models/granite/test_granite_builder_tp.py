# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granite native KV fails closed for unsupported tensor parallel builds."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from tensorrt_model_connect.families.granite.config import ModelConfig
    from tensorrt_model_connect.parallel_config import ParallelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="granite",
        architectures=["GraniteForCausalLM"],
        raw={"_decoder_engine_layout": "split"},
        hidden_size=64,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=16,
        hidden_act="silu",
    )


def test_granite_plugin_rejects_parallel_instead_of_falling_back() -> None:
    module = importlib.import_module("tensorrt_model_connect.families.granite.plugin")

    parallel = ParallelConfig(mode="tensor_parallel", tp_size=4, rank=2)
    with pytest.raises(NotImplementedError, match="tensor parallel"):
        module.GranitePlugin().build_engine(
            _config(),
            {},
            16,
            precision="fp16",
            parallel_config=parallel,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy-preserving precision routing for CodeGen."""

from __future__ import annotations

import importlib

import pytest


pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.codegen.checkpoint_mapper import WeightDict  # noqa: E402
from families.codegen.config import ModelConfig  # noqa: E402


def _config() -> ModelConfig:
    return ModelConfig(
        model_type="codegen",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        raw={"rotary_dim": 2},
    )


def test_codegen_model_routes_accuracy_precision_boundaries(monkeypatch) -> None:
    model_module = importlib.import_module("families.codegen.model")
    captured: dict[str, object] = {}

    def fake_build(config, weights, max_cache_length, **kwargs):
        captured["kwargs"] = kwargs
        return b"single-device-plan"

    monkeypatch.setattr(
        model_module,
        "build_standard_decoder_engine",
        fake_build,
    )

    plan = model_module._CodeGenModel().build_engine(
        _config(),
        WeightDict(),
        max_cache_length=17,
        precision="fp16",
    )

    assert plan == b"single-device-plan"
    kwargs = captured["kwargs"]
    assert kwargs["fp32_rope"] is True
    assert kwargs["fp32_qk_attention"] is True
    assert kwargs["fp32_lm_head"] is True

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_model_connect.families.qwen.builder_policy import (
    configure_qwen_builder,
)


def _quant_context(format_name: str):
    return SimpleNamespace(
        profile=SimpleNamespace(
            format=SimpleNamespace(name=format_name),
            exclude_patterns=[],
        ))


def test_qwen_fp8_uses_accuracy_stable_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)
    quant_context = _quant_context("fp8")

    configure_qwen_builder(
        config, quant_context, "11.2.0.113", num_hidden_layers=28)

    assert config.builder_optimization_level == 0
    assert quant_context.profile.exclude_patterns == []


def test_qwen_fp8_preserves_trt_11_0_builder_level_and_fp16_tail():
    config = SimpleNamespace(builder_optimization_level=5)
    quant_context = _quant_context("fp8")

    configure_qwen_builder(
        config, quant_context, "11.0.0.114", num_hidden_layers=28)

    assert config.builder_optimization_level == 5
    assert quant_context.profile.exclude_patterns == [
        "layer.20.w_up",
        "layer.21.w_up",
        "layer.22.w_up",
        "layer.23.w_up",
        "layer.24.w_up",
        "layer.25.w_up",
        "layer.26.w_up",
        "layer.27.w_up",
    ]


def test_qwen_non_fp8_preserves_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(
        config, _quant_context("int8_sq"), "11.2.0.113", num_hidden_layers=28)

    assert config.builder_optimization_level == 5


def test_qwen_unquantized_preserves_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(config, None, "11.2.0.113", num_hidden_layers=28)

    assert config.builder_optimization_level == 5


def test_qwen_unknown_trt_version_preserves_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(
        config, _quant_context("fp8"), "unknown", num_hidden_layers=28)

    assert config.builder_optimization_level == 5


def test_qwen_builder_level_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "3")
    config = SimpleNamespace(builder_optimization_level=3)

    configure_qwen_builder(
        config, _quant_context("fp8"), "11.2.0.113", num_hidden_layers=28)

    assert config.builder_optimization_level == 3

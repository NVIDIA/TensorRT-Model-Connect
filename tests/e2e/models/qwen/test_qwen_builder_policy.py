# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from tensorrt_model_connect.families.qwen.builder_policy import (
    configure_qwen_builder,
)


def _quant_context(format_name: str):
    return SimpleNamespace(
        profile=SimpleNamespace(format=SimpleNamespace(name=format_name)))


def test_qwen_fp8_uses_accuracy_stable_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(config, _quant_context("fp8"))

    assert config.builder_optimization_level == 0


def test_qwen_non_fp8_preserves_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(config, _quant_context("int8_sq"))

    assert config.builder_optimization_level == 5


def test_qwen_unquantized_preserves_builder_level():
    config = SimpleNamespace(builder_optimization_level=5)

    configure_qwen_builder(config, None)

    assert config.builder_optimization_level == 5


def test_qwen_builder_level_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("TRTMC_BUILDER_OPTIMIZATION_LEVEL", "3")
    config = SimpleNamespace(builder_optimization_level=3)

    configure_qwen_builder(config, _quant_context("fp8"))

    assert config.builder_optimization_level == 3

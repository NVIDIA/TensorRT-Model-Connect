# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned registry disambiguation contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="Qwen-VL registry tests require TensorRT")

from tensorrt_model_connect.families import find_plugin


def test_qwen_vl_matches_vl_plugin() -> None:
    plugin = find_plugin("qwen2_vl")
    assert plugin is not None
    assert plugin.name == "qwen_vl"


def test_qwen_vl_runtime_contract() -> None:
    plugin = find_plugin("qwen2_vl")
    assert plugin is not None
    assert getattr(plugin, "runtime_strategy", None) == "qwen_vl_vision_language"
    assert "decoder_kv" in getattr(plugin, "runtime_capabilities", set())
    assert getattr(plugin, "embed_input", False) is True
    assert callable(getattr(plugin, "build_vision_engine", None))
    assert callable(getattr(plugin, "get_vl_config", None))


def test_plain_qwen_does_not_match_vl() -> None:
    plugin = find_plugin("qwen2_vl")
    assert plugin is not None
    assert plugin.name == "qwen_vl"
    assert not plugin.matches("qwen3")

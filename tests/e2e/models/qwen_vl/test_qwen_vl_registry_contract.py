# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL-owned registry disambiguation contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="Qwen-VL registry tests require TensorRT")

import tensorrt_model_connect.families.qwen_vl.model as model


def test_qwen_vl_matches_vl_plugin() -> None:
    assert model.matches("qwen2_vl")
    assert model.name == "qwen_vl"


def test_qwen_vl_runtime_contract() -> None:
    assert model.matches("qwen2_vl")
    assert getattr(model, "runtime_strategy", None) == "qwen_vl_vision_language"
    assert "decoder_kv" in getattr(model, "runtime_capabilities", set())
    assert getattr(model, "embed_input", False) is True
    assert callable(getattr(model, "build_vision_engine", None))
    assert callable(getattr(model, "get_vl_config", None))


def test_plain_qwen_does_not_match_vl() -> None:
    assert model.matches("qwen2_vl")
    assert model.name == "qwen_vl"
    assert not model.matches("qwen3")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import plugin modules")

from tensorrt_model_connect.families import find_plugin


def _plugin(model_type: str):
    plugin = find_plugin(model_type)
    assert plugin is not None
    return plugin

def test_qwen_owned_runtime_strategy() -> None:
    plugin = _plugin("qwen")
    assert getattr(plugin, "runtime_strategy", None) == "qwen_decoder_kv_cache"


def test_no_embed_input() -> None:
    plugin = _plugin("qwen")
    assert not getattr(plugin, "embed_input", False)


def test_qwen_does_not_match_qwen_vl() -> None:
    plugin = _plugin("qwen")
    assert not plugin.matches("qwen2_vl")


def test_qwen3_alias_dispatches_to_qwen() -> None:
    plugin = _plugin("qwen3")
    assert plugin.name == "qwen"

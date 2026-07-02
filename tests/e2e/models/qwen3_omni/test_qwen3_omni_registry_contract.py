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

def test_vision_language_runtime_contract() -> None:
    plugin = _plugin("qwen3_omni")
    assert getattr(plugin, "runtime_strategy", None) == "qwen3_omni_multimodal"
    assert getattr(plugin, "embed_input", False) is True
    assert callable(getattr(plugin, "build_vision_engine", None))
    assert callable(getattr(plugin, "get_vl_config", None))

    assert callable(getattr(plugin, "build_extra_engines", None))

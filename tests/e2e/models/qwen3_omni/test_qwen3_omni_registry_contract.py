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

def test_text_only_runtime_contract() -> None:
    plugin = _plugin("qwen3_omni")
    assert getattr(plugin, "runtime_strategy", None) == "qwen3_omni_multimodal"
    assert getattr(plugin, "embed_input", True) is False
    assert getattr(plugin, "build_vision_engine", None) is None
    assert getattr(plugin, "get_vl_config", None) is None
    assert getattr(plugin, "build_extra_engines", None) is None

    # Keep the dormant implementation private for future native multimodal work.
    assert callable(getattr(plugin, "_build_vision_engine", None))
    assert callable(getattr(plugin, "_get_vl_config", None))
    assert callable(getattr(plugin, "_build_extra_engines", None))

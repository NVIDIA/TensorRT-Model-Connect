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

def test_runtime_strategy() -> None:
    plugin = _plugin("personaplex")
    assert getattr(plugin, "runtime_strategy", None) == "personaplex_speech_to_speech"


def test_bundle_overrides() -> None:
    from tensorrt_model_connect.config import ModelConfig

    plugin = _plugin("personaplex")
    overrides = plugin.get_bundle_config_overrides(ModelConfig(model_type="personaplex"))
    assert overrides["eos_token_id"] == 2
    assert overrides["speech_depth_temperature"] == pytest.approx(0.0)
    assert overrides["speech_depth_top_k"] == 0
    assert "speech_system_prompt" not in overrides
    assert overrides["speech_text_prompt_ids"] == []

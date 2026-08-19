# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import model modules")

from tensorrt_model_connect.models import find_model


def _model(model_type: str):
    model = find_model(model_type)
    assert model is not None
    return model


def test_runtime_strategy() -> None:
    model = _model("personaplex")
    assert getattr(model, "runtime_strategy", None) == "personaplex_speech_to_speech"


def test_bundle_overrides() -> None:
    from tensorrt_model_connect.config import ModelConfig

    model = _model("personaplex")
    overrides = model.get_bundle_config_overrides(ModelConfig(model_type="personaplex"))
    assert overrides["eos_token_id"] == 2
    assert overrides["speech_depth_temperature"] == pytest.approx(0.0)
    assert overrides["speech_depth_top_k"] == 0
    assert overrides["speech_system_prompt"] == ""
    assert overrides["speech_text_prompt_ids"] == []

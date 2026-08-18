# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import model modules")

import tensorrt_model_connect.families.locateanything.model as model


def _model(model_type: str):
    assert model.matches(model_type)
    return model

def test_vision_language_runtime_contract() -> None:
    model_module = _model("locateanything")
    assert getattr(model_module, "runtime_strategy", None) == "locateanything_vision_language"
    assert getattr(model_module, "embed_input", False) is True
    assert callable(getattr(model_module, "build_vision_engine", None))
    assert callable(getattr(model_module, "get_vl_config", None))

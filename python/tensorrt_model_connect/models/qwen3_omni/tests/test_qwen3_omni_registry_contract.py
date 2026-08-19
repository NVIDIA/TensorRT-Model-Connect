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


def test_vision_language_runtime_contract() -> None:
    model = _model("qwen3_omni")
    assert getattr(model, "runtime_strategy", None) == "qwen3_omni_multimodal"
    assert getattr(model, "embed_input", False) is True
    assert callable(getattr(model, "build_vision_engine", None))
    assert callable(getattr(model, "get_vl_config", None))

    assert callable(getattr(model, "build_extra_engines", None))

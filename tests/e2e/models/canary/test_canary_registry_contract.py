# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned registry contract tests."""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt", reason="registry contract tests import model modules")

from tensorrt_model_connect.families import find_model


def _model(model_type: str):
    model = find_model(model_type)
    assert model is not None
    return model


def test_runtime_strategy() -> None:
    model = _model("canary")
    assert getattr(model, "runtime_strategy", None) == "canary_speech_to_text"

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
    model = _model("chronos_bolt")
    assert getattr(model, "runtime_strategy", None) == "chronos_bolt_trt"


def test_matches_official_t5_config() -> None:
    from tensorrt_model_connect.config import ModelConfig

    model = find_model(ModelConfig(
        model_type="t5",
        architectures=["ChronosBoltModelForForecasting"],
        raw={
            "model_type": "t5",
            "architectures": ["ChronosBoltModelForForecasting"],
            "chronos_config": {"context_length": 16},
        },
    ))
    assert model is not None
    assert model.name == "chronos_bolt"

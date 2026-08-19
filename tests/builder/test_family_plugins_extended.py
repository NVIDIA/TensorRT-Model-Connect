# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic guard for extended family plugin weight test ownership.

Concrete extended family plugin load_weights assertions live under
``python/tensorrt_model_connect/models/<family>/``. This shared file intentionally keeps no
model-specific checkpoint keys or plugin assertions.
"""

from __future__ import annotations

from pathlib import Path


def test_extended_family_plugin_weight_tests_are_model_owned() -> None:
    models_dir = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tensorrt_model_connect"
        / "models"
    )
    owned_tests = sorted(models_dir.glob("*/tests/test_*family_plugin_weights.py"))

    assert owned_tests, "expected family-owned plugin weight tests"
    assert all("/python/tensorrt_model_connect/models/" in test.as_posix() for test in owned_tests)

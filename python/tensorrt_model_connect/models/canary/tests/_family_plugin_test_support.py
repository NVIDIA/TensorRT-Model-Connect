# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned support for Canary plugin weight tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "tensorrt",
    reason="family plugin weight tests import TensorRT-backed plugin modules",
)

try:
    from tensorrt_model_connect.checkpoint_mapper import WeightDict  # noqa: F401
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
    from tensorrt_model_connect.parallel_config import ParallelConfig  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

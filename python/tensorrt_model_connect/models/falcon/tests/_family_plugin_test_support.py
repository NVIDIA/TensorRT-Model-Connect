# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned support for plugin weight tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "tensorrt",
    reason="family plugin weight tests import TensorRT-backed plugin modules",
)

try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

RNG = np.random.RandomState(42)


def _rand(*shape: int) -> np.ndarray:
    return RNG.randn(*shape).astype(np.float32)


def _write_config(model_dir: Path, config: dict) -> None:
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_safetensors(
    model_dir: Path,
    tensors: dict[str, np.ndarray],
    filename: str = "model.safetensors",
) -> None:
    save_file(tensors, str(model_dir / filename))

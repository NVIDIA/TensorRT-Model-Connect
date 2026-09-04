# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the exact Inception-v3 safetensors checkpoint."""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
from safetensors import safe_open


WeightDict = dict[str, np.ndarray]


def _target_np_dtype(precision: str) -> np.dtype:
    if precision == "fp16":
        return np.float16
    if precision == "fp32":
        return np.float32
    raise ValueError(f"Unsupported timm_inception precision: {precision}")


def _open_safetensors(model_dir: Path):
    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing timm Inception checkpoint: {path}")
    return safe_open(str(path), framework="numpy")


def _load_tensor(reader, name: str) -> np.ndarray:
    if name not in reader.keys():
        raise KeyError(f"Tensor not found: {name}")
    return np.asarray(reader.get_tensor(name), dtype=np.float32)

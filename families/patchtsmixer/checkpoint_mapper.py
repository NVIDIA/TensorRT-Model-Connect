# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict PatchTSMixer safetensors reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


def target_numpy_dtype(precision: str) -> np.dtype:
    if precision == "fp16":
        return np.dtype(np.float16)
    if precision == "fp32":
        return np.dtype(np.float32)
    raise ValueError(f"PatchTSMixer supports only fp16 or fp32, got {precision!r}")


def open_safetensors(model_dir: Path) -> dict[str, Any]:
    """Open the one checkpoint format this family supports."""

    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"PatchTSMixer checkpoint is missing {path.name}: {model_dir}")
    reader = safe_open(str(path), framework="numpy")
    return {name: reader for name in reader.keys()}


def load_tensor(readers: dict[str, Any], name: str) -> np.ndarray:
    reader = readers.get(name)
    if reader is None:
        raise KeyError(f"PatchTSMixer tensor is missing: {name}")
    tensor = reader.get_tensor(name)
    dtype_text = str(tensor.dtype)
    if tensor.dtype == np.uint16 or dtype_text == "bfloat16":
        bits = tensor.view(np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32)
    return np.asarray(tensor, dtype=np.float32)

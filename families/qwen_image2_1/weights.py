# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint readers for the Qwen-Image 2.1 diffusers snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Qwen-Image 2.1 checkpoint is missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_safetensors(directory: Path, prefixes: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Read every tensor whose name starts with one of ``prefixes`` in its stored dtype.

    BF16 tensors stay BF16, so the engines carry the checkpoint's values without a round trip
    through another precision.
    """
    # Registers bfloat16 with NumPy; safetensors needs it to return BF16 arrays.
    import ml_dtypes  # noqa: F401
    from safetensors import safe_open

    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {directory}")
    weights: dict[str, np.ndarray] = {}
    for path in files:
        with safe_open(str(path), framework="numpy") as handle:
            for key in handle.keys():
                if prefixes and not key.startswith(prefixes):
                    continue
                if key in weights:
                    raise ValueError(f"duplicate tensor {key!r} in {directory}")
                weights[key] = handle.get_tensor(key)
    if not weights:
        raise ValueError(f"no tensors named {prefixes} in {directory}")
    return weights


def take(weights: dict[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return one tensor after checking that the checkpoint stores the expected shape."""
    if name not in weights:
        raise KeyError(f"Qwen-Image 2.1 checkpoint is missing tensor {name!r}")
    array = weights[name]
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"tensor {name!r} has shape {tuple(array.shape)}, expected {tuple(shape)}")
    return array

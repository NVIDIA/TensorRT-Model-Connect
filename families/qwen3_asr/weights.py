# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint tensor reader for Qwen3-ASR."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_safetensors(directory: Path) -> dict[str, np.ndarray]:
    """Read every tensor in its stored dtype; BF16 stays BF16 (``ml_dtypes.bfloat16``)."""
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
                if key in weights:
                    raise ValueError(f"duplicate tensor {key!r} in {directory}")
                weights[key] = handle.get_tensor(key)
    return weights


def take(weights: dict[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return one tensor after checking that the checkpoint stores the expected shape."""
    if name not in weights:
        raise KeyError(f"Qwen3-ASR checkpoint is missing tensor {name!r}")
    array = weights[name]
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"tensor {name!r} has shape {tuple(array.shape)}, expected {tuple(shape)}")
    return array

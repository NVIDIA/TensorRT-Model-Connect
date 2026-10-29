# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the family-owned LeRobot Pi 0.5 graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors import safe_open


_PREFIX = "paligemma_with_expert."
# The language embedding table is tied to the PaliGemma lm_head, and the action
# expert has no embedding of its own: its lm_head is a training leftover.
EMBEDDING_KEY = "paligemma.lm_head.weight"
_UNUSED = ("gemma_expert.lm_head.weight",)


def load_checkpoint(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Load inference weights as contiguous float32, keyed without the shared prefix."""
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot Pi 0.5 checkpoint not found: {path}")
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as reader:
        for name in reader.keys():
            key = name[len(_PREFIX):] if name.startswith(_PREFIX) else name
            if key in _UNUSED:
                continue
            weights[key] = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
    if EMBEDDING_KEY not in weights:
        raise KeyError(f"LeRobot Pi 0.5 checkpoint is missing {EMBEDDING_KEY}")
    return weights


def require(weights: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return one tensor after checking it exists with the expected shape."""
    if key not in weights:
        raise KeyError(f"LeRobot Pi 0.5 checkpoint is missing tensor {key}")
    value = weights[key]
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"LeRobot Pi 0.5 tensor {key} has shape {tuple(value.shape)}, expected {tuple(shape)}"
        )
    return value

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the family-owned GR00T N1.7 graphs."""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with NumPy; checkpoints trained in bf16 store it)
import numpy as np
from safetensors import safe_open


# LeRobot saves the policy's state dict, so every tensor sits under ``_groot_model.``.
_PREFIX = "_groot_model."
# The Qwen3-VL embedding table is tied to the unused LM head, so a checkpoint stores it under
# either name. LeRobot looks tokens up in ``embed_tokens``, so it wins when both exist.
EMBEDDING_KEYS = ("backbone.model.model.language_model.embed_tokens.weight", "backbone.model.lm_head.weight")
# The action head reads the last kept decoder layer before the final norm.
_UNUSED = ("backbone.model.model.language_model.norm.weight",)


def canonical_name(name: str) -> str:
    return name[len(_PREFIX):] if name.startswith(_PREFIX) else name


def embedding_key(weights: dict[str, np.ndarray]) -> str:
    """The key of the language embedding table in ``weights``."""
    for key in EMBEDDING_KEYS:
        if key in weights:
            return key
    raise KeyError(f"LeRobot GR00T checkpoint has none of {', '.join(EMBEDDING_KEYS)}")


def load_checkpoint(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Load inference weights as contiguous float32 under their canonical names."""
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot GR00T checkpoint not found: {path}")
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as reader:
        for name in reader.keys():
            key = canonical_name(name)
            if key in _UNUSED:
                continue
            if key in weights:
                raise ValueError(f"LeRobot GR00T checkpoint names {key} twice")
            weights[key] = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
    embedding_key(weights)
    return weights


def require(weights: dict[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    """Return one tensor after checking it exists with the expected shape."""
    if key not in weights:
        raise KeyError(f"LeRobot GR00T checkpoint is missing tensor {key}")
    value = weights[key]
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"LeRobot GR00T tensor {key} has shape {tuple(value.shape)}, expected {tuple(shape)}")
    return value


def category(weights: dict[str, np.ndarray], prefix: str, embodiment: int, shape: tuple[int, int]):
    """One embodiment's ``[in, out]`` weight and bias from a category-specific linear layer."""
    stacked = weights.get(prefix + "W")
    if stacked is None or stacked.ndim != 3 or tuple(stacked.shape[1:]) != tuple(shape):
        found = None if stacked is None else tuple(stacked.shape)
        raise ValueError(f"LeRobot GR00T tensor {prefix}W has shape {found}, expected (embodiments, {shape[0]}, {shape[1]})")
    if not 0 <= embodiment < stacked.shape[0]:
        raise ValueError(f"LeRobot GR00T embodiment id {embodiment} is outside {prefix}W's {stacked.shape[0]} slots")
    bias = require(weights, prefix + "b", (stacked.shape[0], shape[1]))
    return stacked[embodiment], bias[embodiment]

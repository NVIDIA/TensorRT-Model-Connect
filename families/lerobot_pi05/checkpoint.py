# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading for the family-owned LeRobot Pi 0.5 graphs."""

from __future__ import annotations

from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with NumPy; checkpoints trained in bf16 store it)
import numpy as np
from safetensors import safe_open


# A converted checkpoint (``lerobot/pi05_libero_base``) names tensors ``paligemma_with_expert.*``;
# a ``lerobot-train`` output saves the policy's state dict, so the same names sit under ``model.``.
_OPTIONAL_PREFIXES = ("model.", "paligemma_with_expert.")
# The language embedding table is stored as ``embed_tokens`` or, when tied, only as the PaliGemma
# lm_head. LeRobot looks tokens up in ``embed_tokens``, so it wins when both exist. The action
# expert has no embedding of its own: its lm_head is a training leftover.
EMBEDDING_KEYS = ("paligemma.model.language_model.embed_tokens.weight", "paligemma.lm_head.weight")
_UNUSED = ("gemma_expert.lm_head.weight",)


def canonical_name(name: str) -> str:
    """Drop the optional ``model.`` and ``paligemma_with_expert.`` prefixes, in that order."""
    for prefix in _OPTIONAL_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def embedding_key(weights: dict[str, np.ndarray]) -> str:
    """The key of the language embedding table in ``weights``."""
    for key in EMBEDDING_KEYS:
        if key in weights:
            return key
    raise KeyError(f"LeRobot Pi 0.5 checkpoint has none of {', '.join(EMBEDDING_KEYS)}")


def load_checkpoint(model_dir: str | Path) -> dict[str, np.ndarray]:
    """Load inference weights as contiguous float32 under their canonical names."""
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot Pi 0.5 checkpoint not found: {path}")
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as reader:
        for name in reader.keys():
            key = canonical_name(name)
            if key in _UNUSED:
                continue
            if key in weights:
                raise ValueError(f"LeRobot Pi 0.5 checkpoint names {key} twice")
            weights[key] = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
    embedding_key(weights)
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

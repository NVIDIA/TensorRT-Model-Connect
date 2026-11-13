# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint readers and tensor normalization for Parakeet TDT."""

from __future__ import annotations

import numpy as np


class WeightDict(dict):
    """Normalized family-owned build tensors."""


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _transpose(value) -> np.ndarray:
    array = _array(value)
    if array.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got {array.shape}")
    return np.ascontiguousarray(array.T)


def _transpose_2d(value, name: str, precision: str = "fp32") -> np.ndarray:
    """Family graph-builder adapter with the repository's mapper signature."""
    del name, precision
    return _transpose(value)

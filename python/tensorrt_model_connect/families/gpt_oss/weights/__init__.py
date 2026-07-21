# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS's internal TensorRT weight representation."""

from __future__ import annotations

import numpy as np


class WeightDict(dict):
    """GPT-OSS logical weights and builder metadata."""


def _transpose_2d(array: np.ndarray, name: str) -> np.ndarray:
    """Convert a Hugging Face projection to TensorRT's input-major layout."""
    if array.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {name}")
    return np.ascontiguousarray(array.T, dtype=np.float32)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy twin of the family's random streams (``runtime/random.cpp``).

Value ``i`` of a stream is SplitMix64 output ``i`` of a generator keyed by the seed and the stream, so the reference
model can be run with exactly the draws the native runtime makes.
"""

from __future__ import annotations

import math

import numpy as np

TRUNK_INIT = 1
LM_DROPOUT = 2
COORDINATES = 3
ROTATION = 4
TRANSLATION = 5
CHURN = 6

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_TRUNCATION_CANDIDATES = 8


def _mix(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint64, copy=True)
    with np.errstate(over="ignore"):
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _key(seed: int, stream: int) -> np.uint64:
    seeded = _mix(np.asarray([seed], dtype=np.uint64))
    with np.errstate(over="ignore"):
        return _mix(seeded + np.uint64(stream))[0]


def bits(seed: int, stream: int, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return _mix(_key(seed, stream) + (indices + np.uint64(1)) * _GAMMA)


def normal(seed: int, stream: int, indices: np.ndarray) -> np.ndarray:
    """Box-Muller (cosine branch) from outputs 2i and 2i + 1, in double precision, rounded to float32."""

    indices = np.asarray(indices, dtype=np.uint64)
    first = bits(seed, stream, indices * np.uint64(2))
    second = bits(seed, stream, indices * np.uint64(2) + np.uint64(1))
    u1 = ((first >> np.uint64(11)) + np.uint64(1)).astype(np.float64) / 9007199254740992.0
    u2 = (second >> np.uint64(11)).astype(np.float64) / 9007199254740992.0
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(6.283185307179586 * u2)).astype(np.float32)


def truncated_normal(seed: int, stream: int, indices: np.ndarray, std: float) -> np.ndarray:
    """The first of eight candidates within three standard deviations, times ``std`` (float32)."""

    indices = np.asarray(indices, dtype=np.uint64)
    candidates = normal(
        seed, stream, indices[:, None] * np.uint64(_TRUNCATION_CANDIDATES) + np.arange(_TRUNCATION_CANDIDATES, dtype=np.uint64)
    )
    inside = np.abs(candidates.astype(np.float64)) <= 3.0
    first = np.where(inside.any(axis=1), inside.argmax(axis=1), _TRUNCATION_CANDIDATES - 1)
    chosen = candidates[np.arange(len(indices)), first]
    chosen = np.clip(chosen.astype(np.float64), -3.0, 3.0).astype(np.float32)
    return chosen * np.float32(std)


def keep_threshold(dropout: float) -> int:
    """``llround((1 - dropout) * 2^24)``: halves round away from zero, as in the runtime."""

    return int(math.floor((1.0 - dropout) * 16777216.0 + 0.5))


def keep(seed: int, stream: int, indices: np.ndarray, dropout: float) -> np.ndarray:
    return (bits(seed, stream, indices) >> np.uint64(40)) < np.uint64(keep_threshold(dropout))

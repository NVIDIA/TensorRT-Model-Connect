# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python mirror of the family's counter-based Gaussian noise (``runtime/noise.h``).

The E2E test draws the native runtime's initial latents and per-step noise with it and
replays them into the reference pipeline. Integer arithmetic wraps modulo 2^64.
"""

from __future__ import annotations

import numpy as np

_GOLDEN = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1


def mix64(values: np.ndarray) -> np.ndarray:
    """The SplitMix64 finalizer, element-wise on uint64."""
    z = np.asarray(values, dtype=np.uint64).copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def gaussian(seed: int, stream: int, count: int) -> np.ndarray:
    """``count`` float32 standard normal values of one (seed, stream)."""
    if seed < 0 or stream < 0 or count < 0:
        raise ValueError("seed, stream and count must be non-negative")
    key = mix64(np.array([(seed + _GOLDEN * (stream + 1)) & _MASK], dtype=np.uint64))[0]
    pairs = (count + 1) // 2
    counters = np.arange(1, 2 * pairs + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        words = mix64(key + np.uint64(_GOLDEN) * counters)
    u1 = ((words[0::2] >> np.uint64(11)) + np.uint64(1)).astype(np.float64) * 2.0**-53
    u2 = (words[1::2] >> np.uint64(11)).astype(np.float64) * 2.0**-53
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 6.283185307179586 * u2
    values = np.empty(2 * pairs, dtype=np.float32)
    values[0::2] = (radius * np.cos(angle)).astype(np.float32)
    values[1::2] = (radius * np.sin(angle)).astype(np.float32)
    return values[:count]


def initial_latents(seed: int, channels: int, frames: int) -> np.ndarray:
    """Stream 0: the unit-variance latents before the scheduler's initial scale, [1, C, T]."""
    return gaussian(seed, 0, channels * frames).reshape(1, channels, frames)


def step_noise(seed: int, step: int, channels: int, frames: int) -> np.ndarray:
    """Stream ``step + 1``: the noise the scheduler injects at denoising step ``step``, [1, C, T]."""
    return gaussian(seed, step + 1, channels * frames).reshape(1, channels, frames)

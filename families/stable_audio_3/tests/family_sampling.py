# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy twin of ``runtime/sampling.h``: render sizing, schedule and the family noise.

The reference harness feeds this noise to the official pipeline so that the
native and the official runs start from, and re-noise with, the same values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


# Longest accepted duration (runtime/sampling.h kMaxDurationSeconds); the render stops at max_latents.
MAX_DURATION_SECONDS = 1.0e9


@dataclass(frozen=True)
class RenderPlan:
    latents: int
    effective_latents: int
    valid_latents: int
    output_samples: int


def plan_render(seconds: float, *, sample_rate: int = 44100, downsampling_ratio: int = 4096,
                latent_align: int = 2, max_latents: int = 1292, headroom_seconds: float = 6.0,
                headroom_latents: int = 64) -> RenderPlan:
    if not 0 < seconds <= MAX_DURATION_SECONDS:
        raise ValueError("duration must be in (0, 1e9] seconds")
    samples = int((seconds + headroom_seconds) * sample_rate)
    samples = -(-samples // downsampling_ratio) * downsampling_ratio
    align = downsampling_ratio * latent_align
    samples = min(-(-samples // align) * align, max_latents * downsampling_ratio)
    audio = int(seconds * sample_rate)
    effective = math.ceil(audio / downsampling_ratio)
    latents = samples // downsampling_ratio
    return RenderPlan(latents, effective, min(effective + headroom_latents, latents), min(audio, samples))


def linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    if steps == 1:
        return np.array([start], dtype=np.float32)
    start32, end32 = np.float32(start), np.float32(end)
    step = np.float32((end32 - start32) / np.float32(steps - 1))
    index = np.arange(steps)
    return np.where(index < steps // 2, start32 + step * index.astype(np.float32),
                    end32 - step * (steps - index - 1).astype(np.float32)).astype(np.float32)


def schedule(steps: int, effective_latents: int, *, anchor_length: float = 2000.0, anchor_logsnr: float = -6.2,
             rate: float = 0.0, logsnr_end: float = 2.0) -> np.ndarray:
    t = linspace_f32(1.0, 0.0, steps + 1)
    seq_len = np.float32(max(effective_latents, 1))
    log2_ratio = np.log2(np.float32(seq_len / np.float32(anchor_length)))
    start = np.float32(np.float32(anchor_logsnr) - np.float32(rate) * log2_ratio)
    span = np.float32(np.float32(logsnr_end) - start)
    logsnr = (np.float32(logsnr_end) - t * span).astype(np.float32)
    shifted = (1.0 / (1.0 + np.exp(logsnr.astype(np.float64)))).astype(np.float32)
    shifted = np.where(t <= 0, np.float32(0.0), np.where(t >= 1, np.float32(1.0), shifted))
    shifted[0] = 1.0
    return shifted.astype(np.float32)


def round_to_half(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(np.float16).astype(np.float32)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    z = (values + np.uint64(0x9E3779B97F4A7C15)) & _MASK
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK
    return z ^ (z >> np.uint64(31))


def gaussian_noise(seed: int, draw: int, count: int) -> np.ndarray:
    """Draw ``draw`` of the stream for ``seed``: SplitMix64 counters, Box-Muller in float64."""
    with np.errstate(over="ignore"):
        key = _splitmix64(_splitmix64(np.array([seed], dtype=np.uint64)) + np.uint64(draw))[0]
        pairs = (count + 1) // 2
        counters = key + np.arange(2 * pairs, dtype=np.uint64)
        bits = _splitmix64(counters)
    u1 = ((bits[0::2] >> np.uint64(11)) + np.uint64(1)).astype(np.float64) * 2.0**-53
    u2 = (bits[1::2] >> np.uint64(11)).astype(np.float64) * 2.0**-53
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * math.pi * u2
    values = np.empty(2 * pairs, dtype=np.float64)
    values[0::2] = radius * np.cos(angle)
    values[1::2] = radius * np.sin(angle)
    return values[:count].astype(np.float32)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The historical ten-sample performance stability protocol."""

import math
import statistics
from typing import Sequence


def measurement_stability(samples: Sequence[float]) -> dict:
    if len(samples) != 10 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in samples
    ):
        raise ValueError("stability requires ten finite positive latency samples")
    median = statistics.median(samples)
    first = statistics.median(samples[:5])
    last = statistics.median(samples[5:])
    drift = abs(last - first) / first
    close = sum(abs(sample - median) / median <= 0.05 for sample in samples)
    return {
        "stable": drift <= 0.05 and close >= 8,
        "first_half_median_ms": first,
        "last_half_median_ms": last,
        "median_drift": drift,
        "samples_within_five_percent": close,
    }

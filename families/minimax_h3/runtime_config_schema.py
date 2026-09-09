# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the MiniMax-H3 options accepted by the build entrypoint."""

from __future__ import annotations

import math


_VALIDATORS = {
    "first_block_cache": lambda value: isinstance(value, bool),
    "ref2va_first_block_cache": lambda value: isinstance(value, bool),
    "ref2va_first_block_cache_threshold": lambda value: (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    ),
    "first_block_cache_threshold": lambda value: (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    ),
    "transformer_ref": lambda value: isinstance(value, str),
    "quantized_transformer": lambda value: isinstance(value, str),
    "super_resolution_model": lambda value: isinstance(value, str),
    "super_resolution_weak_model": lambda value: isinstance(value, str),
    "retain_engines": lambda value: isinstance(value, bool),
    "retained_tail_weight_budget_gib": lambda value: (
        isinstance(value, int) and not isinstance(value, bool) and 0 < value <= ((2**63 - 1) >> 30)
    ),
}
_RUNTIME_ONLY = {"retain_engines", "retained_tail_weight_budget_gib"}


def normalize_build_options(values: dict[str, object]) -> dict[str, object]:
    """Validate family options without injecting runtime defaults."""

    unknown = sorted(set(values) - set(_VALIDATORS))
    if unknown:
        raise ValueError(f"unknown MiniMax-H3 option(s): {', '.join(unknown)}")
    for name, value in values.items():
        if not _VALIDATORS[name](value):
            raise ValueError(f"invalid MiniMax-H3 option {name}={value!r}")
    disallowed = sorted(set(values) & _RUNTIME_ONLY)
    if disallowed:
        raise ValueError(f"MiniMax-H3 option(s) are runtime-only: {', '.join(disallowed)}")
    return dict(values)

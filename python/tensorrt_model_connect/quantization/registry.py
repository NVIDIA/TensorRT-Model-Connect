# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quantization format registry.

Auto-registers built-in formats on import. New formats are registered by
calling register_format() — no edits to existing code needed.
"""

from __future__ import annotations

from .formats import (
    FP8Format,
    INT4AWQFormat,
    INT8SmoothQuantFormat,
    NVFP4Format,
    QuantFormat,
    W4A8Format,
)

_FORMATS: dict[str, QuantFormat] = {}


def register_format(fmt: QuantFormat) -> None:
    """Register a quantization format by name."""
    _FORMATS[fmt.name] = fmt


def get_format(name: str) -> QuantFormat:
    """Look up a registered format by name."""
    if name not in _FORMATS:
        available = ", ".join(sorted(_FORMATS.keys()))
        raise ValueError(
            f"Unknown quantization format: {name!r}. Available: {available}")
    return _FORMATS[name]


def list_formats() -> list[str]:
    """Return sorted list of registered format names."""
    return sorted(_FORMATS.keys())


# Auto-register built-in formats
register_format(FP8Format())
register_format(INT8SmoothQuantFormat())
register_format(INT4AWQFormat())
register_format(NVFP4Format())
register_format(W4A8Format())

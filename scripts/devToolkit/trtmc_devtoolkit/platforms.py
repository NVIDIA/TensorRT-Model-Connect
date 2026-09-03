# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize platform facts used by execution-context providers."""

from __future__ import annotations

import platform

from .models import DevToolkitError


_ARCHITECTURE_ALIASES = {
    "amd64": "x86_64",
    "arm64": "aarch64",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
}


def normalize_architecture(value: str | None = None) -> str:
    raw = (value or platform.machine()).strip().lower()
    try:
        return _ARCHITECTURE_ALIASES[raw]
    except KeyError as error:
        raise DevToolkitError(f"Unsupported host architecture: {raw or '<empty>'}") from error

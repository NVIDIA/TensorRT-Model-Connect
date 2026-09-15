# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small family-local boundary around the TensorRT Python module."""

from __future__ import annotations

import importlib
import importlib.metadata
from types import ModuleType


_module: ModuleType | None = None


def get_trt() -> ModuleType:
    """Load TensorRT lazily so dependency-light commands remain importable."""
    global _module
    if _module is None:
        _module = importlib.import_module("tensorrt")
    return _module


def module_version() -> str:
    """Return the installed TensorRT distribution version when available."""
    try:
        return importlib.metadata.version("tensorrt")
    except importlib.metadata.PackageNotFoundError:
        # A target-installed bindings wheel may not include the umbrella
        # distribution metadata. Record the loaded runtime's version instead.
        version = getattr(get_trt(), "__version__", "")
        if not version:
            raise RuntimeError("Cannot determine the loaded TensorRT version")
        return str(version)

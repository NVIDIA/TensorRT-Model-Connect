# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed build-time loading for the OpenPI model DSO."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


_OPENPI_PLUGIN_LIBRARY = "libtrtmc_model_openpi.so"
_openpi_plugin_handle = None


def _plugin_library_path() -> Path:
    configured_library = os.environ.get("TRTMC_OPENPI_TRT_PLUGIN")
    if configured_library:
        return Path(configured_library).expanduser().absolute()

    configured_plugin_dir = os.environ.get("TRTMC_MODEL_PLUGIN_DIR")
    if configured_plugin_dir:
        plugin_dir = Path(configured_plugin_dir).expanduser().absolute()
        candidates = (
            plugin_dir / "openpi" / _OPENPI_PLUGIN_LIBRARY,
            plugin_dir / _OPENPI_PLUGIN_LIBRARY,
        )
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    installed_library = Path(__file__).resolve().parents[2] / "bin" / _OPENPI_PLUGIN_LIBRARY
    if installed_library.is_file():
        return installed_library

    return (
        Path(__file__).resolve().parents[4] / "build" / "models" / "openpi" / _OPENPI_PLUGIN_LIBRARY
    )


def require_openpi_plugin_creator(name: str, *, trt):
    """Return one registered creator after loading the selected OpenPI DSO."""

    global _openpi_plugin_handle

    registry = trt.get_plugin_registry()
    creator = registry.get_creator(name, "1", "")
    if creator is not None:
        return creator

    plugin_path = _plugin_library_path()
    if plugin_path.is_symlink() or not plugin_path.is_file():
        raise RuntimeError(
            f"OpenPI TensorRT engine construction requires the regular model DSO at {plugin_path}"
        )
    try:
        _openpi_plugin_handle = ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise RuntimeError(
            f"failed to load the required OpenPI model DSO {plugin_path}: {error}"
        ) from error

    creator = registry.get_creator(name, "1", "")
    if creator is None:
        raise RuntimeError(f"OpenPI model DSO {plugin_path} does not register creator {name!r} v1")
    return creator

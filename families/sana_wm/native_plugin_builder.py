# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the family-owned SANA-WM TensorRT plugin."""

from __future__ import annotations

import ctypes
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import Any


_PLUGIN_HANDLE: Any | None = None
_PLUGIN_PATH: Path | None = None


def _configure_command(
    source_dir: Path,
    build_dir: Path,
    torch_prefix: str,
    tensorrt_library: Path | None,
) -> list[str]:
    command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={torch_prefix}",
    ]
    if tensorrt_library is not None:
        command.append(f"-DSANA_WM_TRT_LIBRARY={tensorrt_library}")
    return command


def _installed_tensorrt_library() -> Path | None:
    """Return the TensorRT library shipped with the active Python installation."""

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None:
        return None
    for location in spec.submodule_search_locations or ():
        library = Path(location) / "libnvinfer.so.11"
        if library.is_file():
            return library
    return None


def ensure_native_plugin(*, verbose: bool = False) -> Path:
    """Build a fresh plugin library and return it."""
    import torch

    source_dir = Path(__file__).with_name("native_plugins")
    build_dir = Path(tempfile.mkdtemp(prefix="sana-wm-plugin-"))
    output = build_dir / "libtrtmc_sana_wm_native_plugin.so"
    configure = _configure_command(
        source_dir,
        build_dir,
        torch.utils.cmake_prefix_path,
        _installed_tensorrt_library(),
    )
    build = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "trtmc_sana_wm_native_plugin",
        "-j2",
    ]
    kwargs = (
        {} if verbose else {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True}
    )
    try:
        subprocess.run(configure, check=True, **kwargs)
        subprocess.run(build, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"SANA-WM native plugin build failed\n{error.stdout or ''}") from error
    if not output.is_file():
        raise RuntimeError(f"SANA-WM native plugin build did not produce {output}")
    return output


def load_native_plugin(*, verbose: bool = False) -> Path:
    """Load the family-local build plugin for TensorRT graph construction."""

    global _PLUGIN_HANDLE, _PLUGIN_PATH
    if _PLUGIN_PATH is not None:
        return _PLUGIN_PATH
    path = ensure_native_plugin(verbose=verbose).resolve()
    _PLUGIN_HANDLE = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _PLUGIN_PATH = path
    return path

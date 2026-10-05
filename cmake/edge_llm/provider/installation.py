# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic Edge installation discovery and child environment mechanics."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def descriptor(path: Path) -> dict:
    """Validate an explicit installation, independently of its package origin."""
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "schema_version",
        "version",
        "python",
        "python_paths",
        "library_paths",
        "native_module",
        "plugin",
    }
    if not isinstance(data, dict) or data.keys() - allowed:
        raise ValueError("Unknown Edge provider descriptor fields")
    if data.get("schema_version") != 1 or not isinstance(data.get("version"), str):
        raise ValueError("Expected provider schema 1 and an explicit version")
    for key in ("python", "native_module", "plugin"):
        if key not in data and key != "python":
            continue
        value = Path(data.get(key, ""))
        if not value.is_absolute() or not value.is_file():
            raise ValueError(f"Provider {key} must name an existing absolute file")
    if not os.access(data["python"], os.X_OK):
        raise ValueError("Provider Python must be executable")
    for key in ("python_paths", "library_paths"):
        paths = data.setdefault(key, [])
        if not isinstance(paths, list) or any(
            not isinstance(p, str) or not Path(p).is_absolute() or not Path(p).is_dir()
            for p in paths
        ):
            raise ValueError(f"Provider {key} must contain existing absolute directories")
    return data


def environment(data: dict) -> dict[str, str]:
    """Child-only native library search; never mutate the Model Connect process."""
    env = os.environ.copy()
    paths = data["library_paths"]
    if paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*paths, *filter(None, [env.get("LD_LIBRARY_PATH", "")])]
        )
    # A stale wheel selector or source plugin must not contaminate this provider.
    env.pop("EDGELLM_PLUGIN_PATH", None)
    if data.get("plugin"):
        env["EDGELLM_PLUGIN_PATH"] = data["plugin"]
    return env


def command(worker: Path, path: Path, operation: str, *arguments: str) -> tuple[list[str], dict]:
    """Return an argv vector and child environment, never a shell command."""
    data = descriptor(path)
    return [
        data["python"],
        "-I",
        str(worker.resolve()),
        "--provider",
        str(path.resolve()),
        operation,
        *arguments,
    ], environment(data)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_native(data: dict):
    """Use a source-built extension or the wheel's public payload selector."""
    sys.path[:0] = data["python_paths"]
    from tensorrt_edgellm._version import __version__

    if __version__ != data["version"]:
        raise ValueError(
            f"Provider version mismatch: expected {data['version']}, got {__version__}"
        )
    if data.get("plugin"):
        os.environ["EDGELLM_PLUGIN_PATH"] = data["plugin"]
    if data.get("native_module"):
        spec = importlib.util.spec_from_file_location("_edgellm_runtime", data["native_module"])
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load source-built Edge Python bindings")
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
    else:
        from tensorrt_edgellm import runtime

        native = runtime.load()
    plugin = Path(os.environ.get("EDGELLM_PLUGIN_PATH", ""))
    if not plugin.is_file():
        raise ValueError("Provider must identify the actual Edge plugin")
    # Hashes reject accidental replacement even if a local build reuses a version label.
    import tensorrt
    from cuda.bindings import runtime as cuda
    import platform

    def checked(result):
        error, value = result
        if error != cuda.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot identify provider CUDA device: {error}")
        return value

    device = checked(cuda.cudaGetDevice())
    gpu = checked(cuda.cudaGetDeviceProperties(device))
    cuda_version = checked(cuda.cudaRuntimeGetVersion())
    identity = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "sm": gpu.major * 10 + gpu.minor,
        "cuda_runtime_version": cuda_version,
        "abi": 1,
        "version": __version__,
        "runtime_sha256": sha256(Path(native.__file__)),
        "plugin_sha256": sha256(plugin),
        "tensorrt_version": tensorrt.__version__,
    }
    return native, identity, plugin

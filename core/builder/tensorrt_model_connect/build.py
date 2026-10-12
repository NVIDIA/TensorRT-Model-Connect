# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a build to exactly one model family."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import re
import shlex
import sys
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .bundle_writer import BundleWriter
from .graph_transform import GraphTransform, graph_transform


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class BuildRequest:
    """Inputs shared by the build core and one family-owned builder."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str = "trt"
    max_sequence_length: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    video_num_frames: int | None = None
    max_batch_size: int = 1
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    quantization: str | None = None
    fp32_layers: tuple[int, ...] = ()
    dynamic_kv_cache: bool = False
    verbose: bool = False
    graph_transform: GraphTransform | None = None

    def __post_init__(self) -> None:
        if not self.precision:
            raise ValueError("precision must be non-empty")
        _validate_id("family", self.family)
        _validate_id("task", self.task)
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        for field in ("image_height", "image_width", "video_num_frames"):
            value = getattr(self, field)
            if value is not None and value < 1:
                raise ValueError(f"{field} must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive")
        if self.quantization is not None and not self.quantization:
            raise ValueError("quantization must be non-empty when provided")
        if any(layer < 0 for layer in self.fp32_layers):
            raise ValueError("fp32_layers must contain non-negative indices")
        if not isinstance(self.dynamic_kv_cache, bool):
            raise ValueError("dynamic_kv_cache must be a bool")
        if self.graph_transform is not None and not callable(self.graph_transform):
            raise ValueError("graph_transform must be callable when provided")


def subprocess_environment(
    overrides: dict[str, str], *, prepend_paths: dict[str, str] | None = None
) -> dict[str, str]:
    """Copy the parent environment for one child without mutating process state.

    Callers own explicit tool settings; this helper only merges values and
    prepends search paths using the executing platform's path separator.
    """
    environment = os.environ.copy()
    environment.update(overrides)
    for name, value in (prepend_paths or {}).items():
        previous = environment.get(name)
        environment[name] = value + (os.pathsep + previous if previous else "")
    return environment


def cmake_prefixes() -> list[Path]:
    """Return explicit standard CMake prefixes followed by the Python prefix."""
    prefixes = [
        Path(value) for value in os.environ.get("CMAKE_PREFIX_PATH", "").split(os.pathsep) if value
    ]
    return [*prefixes, Path(sys.prefix)]


def _cuda_toolkit_version() -> str:
    """Identify the selected native compiler, not cuda-python's build toolkit."""
    compiler = os.environ.get("CUDACXX")
    try:
        command = shlex.split(compiler) if compiler else []
    except ValueError as error:
        raise RuntimeError(f"Invalid CUDACXX command: {error}") from error
    if not compiler:
        root = next(
            (os.environ[key] for key in ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH")
             if os.environ.get(key)), None
        )
        compiler = str(Path(root) / "bin" / "nvcc") if root else shutil.which("nvcc")
        command = [compiler] if compiler else []
    if not command:
        raise RuntimeError("CUDA toolkit not found; set CUDAToolkit_ROOT or CUDACXX")
    try:
        result = subprocess.run(
            [*command, "--version"], check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot query CUDA toolkit from {compiler}: {error}") from error
    version = re.search(r"release\s+(\d+\.\d+)", result.stdout)
    if version is None:
        raise RuntimeError(f"Cannot identify CUDA toolkit from {compiler} --version")
    return version.group(1)


def detect_local_platform() -> dict:
    """Return executing GPU and native SDK identity without selecting a model.

    Returns:
        OS/release, CPU architecture, GPU SM, CUDA and TensorRT versions.

    Raises:
        ImportError: Native SDK Python bindings are unavailable.
        RuntimeError: CUDA cannot identify the executing device.
    """
    import tensorrt as trt
    from cuda.bindings import runtime

    def checked(result):
        if int(result[0]) != 0:
            raise RuntimeError(f"CUDA device discovery failed: {result[0]}")
        return result[1]

    device = checked(runtime.cudaGetDevice())
    gpu = checked(runtime.cudaGetDeviceProperties(device))
    cuda_version = _cuda_toolkit_version()
    try:
        release = platform.freedesktop_os_release() if sys.platform == "linux" else {}
    except OSError:
        release = {}
    return {
        "os": sys.platform,
        "os_version": release.get("VERSION_ID", platform.release()),
        "arch": platform.machine(),
        "sm": gpu.major * 10 + gpu.minor,
        "cuda_version": cuda_version,
        "tensorrt_version": trt.__version__,
    }


def _validate_id(field: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a lowercase identifier containing only "
            "letters, digits, and underscores"
        )
    return value


def content_cache_key(domain: str, *payloads: bytes) -> str:
    """Return a domain-separated key for model-agnostic content caches."""

    if not domain or any(not isinstance(payload, bytes) for payload in payloads):
        raise ValueError("cache-key domain and byte payloads must be valid")
    value = hashlib.sha256(domain.encode("utf-8") + b"\0")
    for payload in payloads:
        value.update(len(payload).to_bytes(8, "little"))
        value.update(payload)
    return value.hexdigest()


def _resolve_family(request: BuildRequest) -> str:
    """Return the explicit family dispatch key."""

    return _validate_id("family", request.family)


def _load_family(family: str) -> ModuleType:
    """Import only ``families.<family>.model`` for an exact family ID."""

    family = _validate_id("family", family)
    family_package = f"families.{family}"
    module_name = f"{family_package}.model"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name in {family_package, module_name}:
            raise ModuleNotFoundError(
                f"family {family!r} does not provide {module_name}"
            ) from error
        raise


def select_backend(backend: str) -> None:
    """Bind the explicit build backend before importing a family builder."""

    if backend not in {"trt", "trt_rtx"}:
        raise ValueError("backend must be 'trt' or 'trt_rtx'")
    loaded = sys.modules.get("tensorrt")
    if backend == "trt":
        if sys.modules.get("tensorrt_rtx") is not None:
            raise RuntimeError("TensorRT-RTX is already loaded in this process")
        return

    rtx = importlib.import_module("tensorrt_rtx")
    if loaded is not None and loaded is not rtx:
        raise RuntimeError("TensorRT is already loaded in this process")
    sys.modules["tensorrt"] = rtx


_select_backend = select_backend  # Compatibility for existing Python callers.


def build(request: BuildRequest) -> None:
    """Run one family builder and publish its bundle on success."""

    family = _resolve_family(request)
    _select_backend(request.backend)
    family_module = _load_family(family)
    writer = BundleWriter(request.output_path)
    try:
        with graph_transform(request.graph_transform):
            family_module.build(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic deployment-target facts used to select implementation capsules."""

from __future__ import annotations

import platform
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .manifest import TargetScalar


class TargetResolutionError(ValueError):
    """The requested deployment target cannot be normalized safely."""


def _normalize_os(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"gnu/linux": "linux", "windows": "windows", "darwin": "macos"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"linux", "windows", "macos"}:
        raise TargetResolutionError(f"Unsupported target operating system: {value!r}")
    return normalized


def _normalize_architecture(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"x86_64", "aarch64"}:
        raise TargetResolutionError(f"Unsupported target architecture: {value!r}")
    return normalized


def _cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise TargetResolutionError(
                "Unable to inspect the current GPU because CUDA Python is unavailable"
            ) from exc
    return cudart


def _probe_current_target_with_device() -> tuple[Mapping[str, TargetScalar], int]:
    """Probe target facts plus the process-local active CUDA ordinal.

    The ordinal is transient launch context. It is intentionally excluded from
    normalized target facts so it cannot affect manifest matching or enter a
    portable bundle.
    """

    runtime = _cuda_runtime()
    success = getattr(getattr(runtime, "cudaError_t", None), "cudaSuccess", 0)
    try:
        status, device = runtime.cudaGetDevice()
        if status not in (success, 0):
            raise TargetResolutionError(f"cudaGetDevice failed with status {status}")
        device = int(device)
        status, properties = runtime.cudaGetDeviceProperties(device)
        if status not in (success, 0):
            raise TargetResolutionError(
                f"cudaGetDeviceProperties({device}) failed with status {status}"
            )
        status, device_count = runtime.cudaGetDeviceCount()
        if status not in (success, 0):
            raise TargetResolutionError(f"cudaGetDeviceCount failed with status {status}")
        device_count = int(device_count)
    except TargetResolutionError:
        raise
    except Exception as exc:
        raise TargetResolutionError(f"Unable to query the active CUDA device: {exc}") from exc

    if device < 0 or device >= device_count:
        raise TargetResolutionError(
            f"active CUDA device ordinal {device} is outside visible device count {device_count}"
        )

    name = getattr(properties, "name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace").rstrip("\x00")
    memory = getattr(properties, "totalGlobalMem", 0)
    major = getattr(properties, "major", 0)
    minor = getattr(properties, "minor", 0)
    memory_mib = int(memory) // (1024 * 1024)
    if not str(name).strip() or memory_mib <= 0 or device_count <= 0:
        raise TargetResolutionError("active CUDA device returned incomplete target facts")
    if type(major) is not int or type(minor) is not int or major <= 0 or minor < 0:
        raise TargetResolutionError("active CUDA device returned an invalid compute capability")
    kind = "jetson" if Path("/etc/nv_tegra_release").exists() else "discrete"
    facts: dict[str, TargetScalar] = {
        "target_id": f"current-{kind}-sm{major}{minor}",
        "os": _normalize_os(platform.system()),
        "architecture": _normalize_architecture(platform.machine()),
        "platform_kind": kind,
        "gpu_architecture": f"sm{major}{minor}",
        "gpu_memory_mib": memory_mib,
        "gpu_count": device_count,
        "gpu_name": str(name).strip(),
    }
    return MappingProxyType(facts), device

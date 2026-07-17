# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic deployment-target facts used to select implementation capsules."""

from __future__ import annotations

import json
import os
import platform
import stat
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .manifest import TargetScalar


TARGET_SCHEMA_VERSION = 1
_MAX_TARGET_BYTES = 1024 * 1024


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


def _normalize_platform_kind(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {"dgpu": "discrete", "integrated": "soc"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"discrete", "soc", "jetson"}:
        raise TargetResolutionError(f"Unsupported target platform kind: {value!r}")
    return normalized


def _normalize_gpu_architecture(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace(".", "")
        if normalized.startswith("sm"):
            normalized = normalized[2:]
        if not normalized.isdigit():
            raise TargetResolutionError(
                "target gpu.compute_capability must be an SM integer or MAJOR.MINOR"
            )
        sm = int(normalized)
    elif type(value) is int:
        sm = value
    else:
        raise TargetResolutionError(
            "target gpu.compute_capability must be an SM integer or MAJOR.MINOR"
        )
    if sm < 10 or sm > 999:
        raise TargetResolutionError("target GPU architecture is outside the supported range")
    return f"sm{sm}"


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise TargetResolutionError(f"{field} must be a positive integer")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetResolutionError(f"{field} must be a non-empty string")
    return value.strip()


def normalize_target_descriptor(root: Mapping[str, Any]) -> Mapping[str, TargetScalar]:
    """Convert the public nested target document into stable flat facts."""

    if not isinstance(root, Mapping):
        raise TargetResolutionError("Target descriptor root must be an object")
    allowed_root = {"schema_version", "target_id", "gpu", "system"}
    unknown = sorted(set(root) - allowed_root)
    missing = sorted({"schema_version", "gpu", "system"} - set(root))
    if unknown:
        raise TargetResolutionError(f"Unknown target descriptor fields: {unknown}")
    if missing:
        raise TargetResolutionError(f"Missing target descriptor fields: {missing}")
    if root.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise TargetResolutionError(
            f"Unsupported target schema_version: {root.get('schema_version')!r}"
        )
    gpu = root.get("gpu")
    system = root.get("system")
    if not isinstance(gpu, Mapping) or not isinstance(system, Mapping):
        raise TargetResolutionError("target gpu and system fields must be objects")

    allowed_gpu = {"name", "compute_capability", "memory_mib", "count"}
    allowed_system = {"os", "arch", "architecture", "kind", "platform_kind"}
    unknown_gpu = sorted(set(gpu) - allowed_gpu)
    unknown_system = sorted(set(system) - allowed_system)
    if unknown_gpu or unknown_system:
        raise TargetResolutionError(
            f"Unknown target facts: gpu={unknown_gpu}, system={unknown_system}"
        )
    required_gpu = {"name", "compute_capability", "memory_mib", "count"}
    missing_gpu = sorted(required_gpu - set(gpu))
    if missing_gpu:
        raise TargetResolutionError(f"Missing target GPU facts: {missing_gpu}")
    if "os" not in system:
        raise TargetResolutionError("Missing target system fact: os")
    arch_value = system.get("architecture", system.get("arch"))
    kind_value = system.get("platform_kind", system.get("kind"))
    if arch_value is None or kind_value is None:
        raise TargetResolutionError(
            "Target system requires architecture (or arch) and platform_kind (or kind)"
        )
    if "architecture" in system and "arch" in system:
        raise TargetResolutionError("Target system must not specify both arch and architecture")
    if "platform_kind" in system and "kind" in system:
        raise TargetResolutionError("Target system must not specify both kind and platform_kind")

    target_id = root.get("target_id")
    facts: dict[str, TargetScalar] = {
        "os": _normalize_os(_nonempty_string(system["os"], "target system.os")),
        "architecture": _normalize_architecture(
            _nonempty_string(arch_value, "target system.architecture")
        ),
        "platform_kind": _normalize_platform_kind(
            _nonempty_string(kind_value, "target system.platform_kind")
        ),
        "gpu_architecture": _normalize_gpu_architecture(gpu["compute_capability"]),
        "gpu_memory_mib": _positive_int(gpu["memory_mib"], "target gpu.memory_mib"),
        "gpu_count": _positive_int(gpu["count"], "target gpu.count"),
        "gpu_name": _nonempty_string(gpu["name"], "target gpu.name"),
    }
    if target_id is not None:
        facts["target_id"] = _nonempty_string(target_id, "target_id")
    return MappingProxyType(facts)


def _read_target(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TargetResolutionError(f"Unable to open target descriptor {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise TargetResolutionError(f"Target descriptor is not a regular file: {path}")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_TARGET_BYTES:
            raise TargetResolutionError(
                f"Target descriptor size must be between 1 and {_MAX_TARGET_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            raw = source.read(_MAX_TARGET_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetResolutionError(f"Invalid target descriptor JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TargetResolutionError("Target descriptor root must be an object")
    return value


def _cuda_runtime() -> Any:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart
        except ImportError as exc:
            raise TargetResolutionError(
                "Unable to resolve target=current because CUDA Python is unavailable; "
                "provide a target descriptor JSON file"
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
    kind = "jetson" if Path("/etc/nv_tegra_release").exists() else "discrete"
    root = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "target_id": f"current-{kind}-sm{major}{minor}",
        "gpu": {
            "name": name,
            "compute_capability": major * 10 + minor,
            "memory_mib": memory // (1024 * 1024),
            "count": device_count,
        },
        "system": {
            "os": platform.system(),
            "architecture": platform.machine(),
            "platform_kind": kind,
        },
    }
    return normalize_target_descriptor(root), device


def probe_current_target() -> Mapping[str, TargetScalar]:
    """Probe the active CUDA device without importing a performance runtime."""

    return _probe_current_target_with_device()[0]


def resolve_target(
    request: str | Path | None,
    *,
    current_probe: Callable[[], Mapping[str, TargetScalar]] | None = None,
) -> Mapping[str, TargetScalar]:
    """Resolve ``current`` or a public target JSON file into exact facts."""

    raw = str(request or "current").strip()
    if not raw or raw.lower() == "current":
        return MappingProxyType(dict((current_probe or probe_current_target)()))
    return normalize_target_descriptor(_read_target(Path(raw).expanduser()))

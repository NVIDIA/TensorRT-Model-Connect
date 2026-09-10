# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Toolchain observation helpers shared by built-in providers."""

from __future__ import annotations

import hashlib
import platform
import re
from pathlib import Path

from .models import DevToolkitError, ToolchainObservation
from .platforms import normalize_architecture
from .runner import Runner, command_output


CUDA_RELEASE = re.compile(r"release\s+([0-9]+\.[0-9]+)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensorrt_header_version_text(text: str, source: str | Path) -> str:
    definitions = dict(
        re.findall(r"^#define\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\b", text, re.MULTILINE)
    )
    parts: list[str] = []
    for name in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        value = definitions.get(f"NV_TENSORRT_{name}", "")
        value = definitions.get(value, value)
        if not value.isdigit():
            raise DevToolkitError(f"Could not resolve TensorRT {name.lower()} from {source}")
        parts.append(value)
    return ".".join(parts)


def tensorrt_header_version(path: Path) -> str:
    return tensorrt_header_version_text(path.read_text(encoding="utf-8"), path)


def observe_local_toolchain(
    runner: Runner,
    *,
    repository: Path,
    python: str | Path,
    nvcc: str | Path,
    tensorrt_include_dir: Path,
    tensorrt_library: Path,
    environment: dict[str, str],
    cuda_root: Path | None = None,
) -> ToolchainObservation:
    python_version = command_output(
        runner,
        [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        cwd=repository,
        env=environment,
    )
    nvcc_output = command_output(
        runner,
        [nvcc, "--version"],
        cwd=repository,
        env=environment,
    )
    match = CUDA_RELEASE.search(nvcc_output)
    if match is None:
        raise DevToolkitError(f"Could not resolve CUDA version from nvcc: {nvcc_output}")
    tensorrt_python = command_output(
        runner,
        [python, "-c", "import tensorrt; print(tensorrt.__version__)"],
        cwd=repository,
        env=environment,
    )
    tensorrt_native = command_output(
        runner,
        [
            python,
            "-c",
            (
                "import ctypes; "
                f"lib=ctypes.CDLL({str(tensorrt_library)!r}); "
                "names=('Major','Minor','Patch','Build'); "
                "fs=[getattr(lib, f'getInferLib{name}Version') for name in names]; "
                "[setattr(f, 'restype', ctypes.c_int32) for f in fs]; "
                "print('.'.join(str(f()) for f in fs))"
            ),
        ],
        cwd=repository,
        env=environment,
    )
    return ToolchainObservation(
        python_version=python_version,
        cuda_version=match.group(1),
        tensorrt_python_version=tensorrt_python,
        tensorrt_native_version=tensorrt_native,
        tensorrt_header_version=tensorrt_header_version(tensorrt_include_dir / "NvInferVersion.h"),
        tensorrt_include_dir=str(tensorrt_include_dir),
        tensorrt_library=str(tensorrt_library),
        cuda_root=str(cuda_root) if cuda_root is not None else None,
        architecture=normalize_architecture(platform.machine()),
        evidence={
            "nvcc": _sha256(Path(nvcc)),
            "tensorrt-header": _sha256(tensorrt_include_dir / "NvInferVersion.h"),
            "tensorrt-library": _sha256(tensorrt_library),
        },
    )

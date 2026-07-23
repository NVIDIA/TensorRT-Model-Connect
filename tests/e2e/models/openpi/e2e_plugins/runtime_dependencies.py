# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed dependency proof for the native OpenPI runtime."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.e2e.models.openpi import qualification


class OpenPIRuntimeDependencyError(RuntimeError):
    """Raised when OpenPI's runtime dependency contract is not satisfied."""


_ARTIFACTS = (
    ("runner", "trtmc-openpi"),
    ("core", "libtrtmc_core.so"),
    ("tensorrt_backend", "libtrtmc_backend_trt.so"),
    ("openpi_model", "libtrtmc_model_openpi.so"),
)
_NEEDED = re.compile(r"\(NEEDED\).*Shared library: \[([^\]]+)\]")
_FORBIDDEN = re.compile(
    r"(?:onnx|onnxruntime|python|torch|c10|tvm|jax|tensorflow|opencv|"
    r"protobuf|sentencepiece)",
    re.IGNORECASE,
)
_ALLOWED = tuple(
    re.compile(pattern)
    for pattern in (
        r"linux-vdso\.so\.1",
        r"ld-linux(?:-[A-Za-z0-9_-]+)?\.so(?:\.[A-Za-z0-9_+.-]+)*",
        r"lib(?:c|m|dl|pthread|rt|stdc\+\+|gcc_s|atomic|util)\.so"
        r"(?:\.[A-Za-z0-9_+.-]+)*",
        r"lib(?:cuda|cudart|cublas|cublasLt|nvrtc|nvrtc-builtins|nvJitLink)\.so"
        r"(?:\.[A-Za-z0-9_+.-]+)*",
        r"libnvinfer(?:_[A-Za-z0-9_]+)?\.so(?:\.[A-Za-z0-9_+.-]+)*",
        r"libtrtmc_(?:core|backend_trt(?:_[0-9_]+)?|model_openpi)\.so"
        r"(?:\.[A-Za-z0-9_+.-]+)*",
    )
)


def _run_tool(command: Sequence[str], *, ld_library_path: str) -> str:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ld_library_path
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise OpenPIRuntimeDependencyError(f"{command[0]} failed for {command[-1]}: {detail}")
    return completed.stdout


def _readelf_dependencies(output: str, *, artifact: str) -> tuple[str, ...]:
    dependencies = tuple(
        match.group(1)
        for line in output.splitlines()
        if (match := _NEEDED.search(line)) is not None
    )
    if not dependencies:
        raise OpenPIRuntimeDependencyError(f"{artifact}: readelf reported no direct dependencies")
    return tuple(sorted(set(dependencies)))


def _ldd_dependencies(output: str, *, artifact: str) -> tuple[str, ...]:
    if "not found" in output.lower():
        raise OpenPIRuntimeDependencyError(f"{artifact}: ldd reported an unresolved dependency")
    dependencies: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        token = line.split("=>", 1)[0].strip() if "=>" in line else line.split()[0]
        name = Path(token).name
        if name:
            dependencies.add(name)
    if not dependencies:
        raise OpenPIRuntimeDependencyError(f"{artifact}: ldd reported no dependencies")
    return tuple(sorted(dependencies))


def _validate(dependencies: Sequence[str], *, artifact: str, source: str) -> None:
    forbidden = sorted(name for name in dependencies if _FORBIDDEN.search(name))
    if forbidden:
        raise OpenPIRuntimeDependencyError(
            f"{artifact}: forbidden {source} dependencies: {forbidden}"
        )
    unknown = sorted(
        name for name in dependencies if not any(pattern.fullmatch(name) for pattern in _ALLOWED)
    )
    if unknown:
        raise OpenPIRuntimeDependencyError(f"{artifact}: unknown {source} dependencies: {unknown}")


def audit_openpi_runtime_dependencies(
    *,
    runner: str | Path,
    core: str | Path,
    tensorrt_backend: str | Path,
    openpi_model: str | Path,
    ld_library_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Prove that the exact OpenPI runtime closure is TensorRT/C++ only."""

    if isinstance(ld_library_path, (str, os.PathLike)):
        library_path = os.fspath(ld_library_path)
    else:
        library_path = os.pathsep.join(os.fspath(path) for path in ld_library_path)
    if not library_path:
        raise OpenPIRuntimeDependencyError("LD_LIBRARY_PATH must be explicit and non-empty")

    paths = (Path(runner), Path(core), Path(tensorrt_backend), Path(openpi_model))
    evidence: list[dict[str, Any]] = []
    for (role, expected_name), path in zip(_ARTIFACTS, paths, strict=True):
        if path.name != expected_name:
            raise OpenPIRuntimeDependencyError(
                f"{role}: expected artifact {expected_name!r}, got {path.name!r}"
            )
        if not path.is_file():
            raise OpenPIRuntimeDependencyError(f"{role}: artifact is not a file: {path}")

        direct = _readelf_dependencies(
            _run_tool(("readelf", "-d", str(path)), ld_library_path=library_path),
            artifact=expected_name,
        )
        closure = _ldd_dependencies(
            _run_tool(("ldd", str(path)), ld_library_path=library_path),
            artifact=expected_name,
        )
        _validate(direct, artifact=expected_name, source="direct")
        _validate(closure, artifact=expected_name, source="transitive")
        if role == "openpi_model":
            if not any(
                re.fullmatch(r"libnvinfer(?:_[A-Za-z0-9_]+)?\.so(?:\..+)*", name) for name in direct
            ):
                raise OpenPIRuntimeDependencyError(
                    "libtrtmc_model_openpi.so must directly depend on libnvinfer"
                )
            if not any(re.fullmatch(r"libcublas\.so(?:\..+)*", name) for name in direct):
                raise OpenPIRuntimeDependencyError(
                    "libtrtmc_model_openpi.so must directly depend on libcublas"
                )
        evidence.append(
            {
                "role": role,
                "name": expected_name,
                "sha256": qualification.sha256_file(path),
                "direct_dependencies": list(direct),
                "dependencies": list(closure),
            }
        )

    return {
        "schema_version": 1,
        "runtime_contract": "native_cpp_tensorrt_only",
        "artifacts": evidence,
    }

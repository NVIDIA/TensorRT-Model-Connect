# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OpenPI's model-owned runtime dependency proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .e2e_plugins import runtime_dependencies as audit


_NAMES = (
    "trtmc-openpi",
    "libtrtmc_core.so",
    "libtrtmc_backend_trt.so",
    "libtrtmc_model_openpi.so",
)
_DIRECT = {
    "trtmc-openpi": ("libtrtmc_core.so", "libstdc++.so.6", "libc.so.6"),
    "libtrtmc_core.so": (
        "libcudart.so.13",
        "libcublas.so.13",
        "libdl.so.2",
        "libstdc++.so.6",
    ),
    "libtrtmc_backend_trt.so": (
        "libtrtmc_core.so",
        "libnvinfer.so.11",
        "libcudart.so.13",
    ),
    "libtrtmc_model_openpi.so": (
        "libtrtmc_core.so",
        "libnvinfer.so.11",
        "libcublas.so.13",
        "libcudart.so.13",
    ),
}


def _artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = [tmp_path / name for name in _NAMES]
    for index, path in enumerate(paths):
        path.write_bytes(f"artifact-{index}".encode())
    return dict(zip(("runner", "core", "tensorrt_backend", "openpi_model"), paths, strict=True))


def _fake_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    *,
    injected_dependency: str | None = None,
    model_direct: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append({"command": command, **kwargs})
        name = Path(command[-1]).name
        dependencies = (
            model_direct
            if name == "libtrtmc_model_openpi.so" and model_direct is not None
            else _DIRECT[name]
        )
        if command[0] == "readelf":
            output = "\n".join(
                f" 0x0000000000000001 (NEEDED) Shared library: [{dependency}]"
                for dependency in dependencies
            )
        else:
            closure = list(dependencies) + ["linux-vdso.so.1", "libgcc_s.so.1"]
            if injected_dependency is not None and name == "libtrtmc_model_openpi.so":
                closure.append(injected_dependency)
            output = "\n".join(
                f"{dependency} => /runtime/{dependency} (0x0000000000010000)"
                for dependency in closure
            )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(audit.subprocess, "run", run)
    return calls


def test_runtime_dependency_proof_returns_small_serializable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    calls = _fake_subprocess(monkeypatch)
    library_dirs = (tmp_path, tmp_path / "models" / "openpi")

    result = audit.audit_openpi_runtime_dependencies(**artifacts, ld_library_path=library_dirs)

    json.dumps(result)
    assert result["runtime_contract"] == "native_cpp_tensorrt_only"
    assert [entry["name"] for entry in result["artifacts"]] == list(_NAMES)
    for entry, path in zip(result["artifacts"], artifacts.values(), strict=True):
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["direct_dependencies"]
        assert entry["dependencies"]
    assert len(calls) == 8
    expected_library_path = ":".join(str(path) for path in library_dirs)
    assert all(call["env"]["LD_LIBRARY_PATH"] == expected_library_path for call in calls)


@pytest.mark.parametrize(
    "dependency",
    (
        "libnvonnxparser.so.11",
        "libonnxruntime.so.1",
        "libpython3.12.so.1.0",
        "libtorch.so",
        "libtvm_ffi.so",
        "libjax_runtime.so",
        "libtensorflow.so.2",
        "libopencv_core.so.4",
        "libprotobuf.so.32",
        "libsentencepiece.so.0",
    ),
)
def test_runtime_dependency_proof_rejects_forbidden_frameworks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependency: str
) -> None:
    artifacts = _artifacts(tmp_path)
    _fake_subprocess(monkeypatch, injected_dependency=dependency)

    with pytest.raises(audit.OpenPIRuntimeDependencyError, match="forbidden transitive"):
        audit.audit_openpi_runtime_dependencies(**artifacts, ld_library_path=tmp_path)


def test_runtime_dependency_proof_rejects_unknown_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts(tmp_path)
    _fake_subprocess(monkeypatch, injected_dependency="libmystery.so.1")

    with pytest.raises(audit.OpenPIRuntimeDependencyError, match="unknown transitive"):
        audit.audit_openpi_runtime_dependencies(**artifacts, ld_library_path=tmp_path)


@pytest.mark.parametrize(
    ("model_direct", "message"),
    (
        (
            ("libtrtmc_core.so", "libcublas.so.13", "libcudart.so.13"),
            "directly depend on libnvinfer",
        ),
        (
            ("libtrtmc_core.so", "libnvinfer.so.11", "libcudart.so.13"),
            "directly depend on libcublas",
        ),
    ),
)
def test_runtime_dependency_proof_requires_direct_model_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_direct: tuple[str, ...],
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    _fake_subprocess(monkeypatch, model_direct=model_direct)

    with pytest.raises(audit.OpenPIRuntimeDependencyError, match=message):
        audit.audit_openpi_runtime_dependencies(**artifacts, ld_library_path=tmp_path)

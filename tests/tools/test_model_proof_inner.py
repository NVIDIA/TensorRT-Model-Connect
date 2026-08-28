# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused build-output contracts for the inner model-proof pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ci.context import CiContext
from tools.ci.model_proof import ModelProofRequest
from tools.ci.model_proof_inner import ModelProofInnerPipeline
from tools.ci.process import CiError


def _readelf_output(*dependencies: str) -> str:
    entries = [
        f" 0x0000000000000001 (NEEDED)             Shared library: [{dependency}]"
        for dependency in dependencies
    ]
    return "\n".join(
        [
            "Dynamic section at offset 0x100 contains entries:",
            *entries,
            " 0x000000000000000e (SONAME)             Library soname: [libpython-not-needed.so]",
        ]
    )


def _pipeline_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dynamic_by_name: dict[str, str] | None = None,
) -> tuple[ModelProofInnerPipeline, str, dict[str, Path], dict[str, object]]:
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    runtime_library = "libtrtmc_model_fixture.so"
    binaries = {
        "model": work / "build/models/fixture" / runtime_library,
        "core": work / "build/libtrtmc_core.so",
        "backend": work / "build/libtrtmc_backend_trt.so",
        "cli": work / "build/trtmc",
    }
    for path in binaries.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFfixture")

    context = CiContext(source, {})
    outputs = dynamic_by_name or {}
    calls: list[Path] = []

    def output(command: list[object], **_kwargs: object) -> str:
        assert command[:2] == ["readelf", "-d"]
        elf = Path(command[2])
        calls.append(elf)
        return outputs.get(elf.name, _readelf_output("libstdc++.so.6", "libc.so.6"))

    monkeypatch.setattr(context, "output", output)
    facts: dict[str, object] = {}
    pipeline = ModelProofInnerPipeline(
        context,
        ModelProofRequest("fixture", revision="a" * 40),
    )
    pipeline.source = source
    pipeline.work = work
    pipeline.artifacts = artifacts
    pipeline.status = SimpleNamespace(
        step=lambda *_args: None,
        fact=lambda key, value: facts.__setitem__(key, value),
    )
    facts["readelf_calls"] = calls
    return pipeline, runtime_library, binaries, facts


def test_dso_isolation_accepts_python_free_scratch_runtime_elfs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, runtime_library, binaries, facts = _pipeline_fixture(tmp_path, monkeypatch)

    staged, _, runtime_source = pipeline._validate_dso("fixture", runtime_library)

    assert staged.read_bytes() == binaries["model"].read_bytes()
    assert runtime_source == "scratch-build"
    assert facts["readelf_calls"] == list(binaries.values())
    assert facts["scratch_native_elf_dependency_audit"] == "direct-dt-needed"
    assert facts["scratch_native_elf_dependency_scan_count"] == 4
    assert facts["scratch_python_runtime_dt_needed_count"] == 0
    assert {path.name for path in pipeline.artifacts.glob("*.dynamic.txt")} == {
        "model-dso.dynamic.txt",
        "core-dso.dynamic.txt",
        "trt-backend-dso.dynamic.txt",
        "trtmc.dynamic.txt",
    }


@pytest.mark.parametrize(
    ("binary_key", "dependency"),
    [
        ("model", "libpython3.12.so.1.0"),
        ("core", "libpython.so"),
        ("backend", "libpython3.12d.so"),
        ("cli", "/opt/runtime/lib/libpython3.11.so.1.0"),
    ],
)
def test_dso_isolation_rejects_python_runtime_dt_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binary_key: str,
    dependency: str,
) -> None:
    pipeline, runtime_library, binaries, _ = _pipeline_fixture(tmp_path, monkeypatch)
    pipeline.context.output = lambda command, **_kwargs: (
        _readelf_output(dependency)
        if Path(command[2]) == binaries[binary_key]
        else _readelf_output("libstdc++.so.6")
    )

    with pytest.raises(
        CiError,
        match=r"links a forbidden Python runtime via DT_NEEDED",
    ):
        pipeline._validate_dso("fixture", runtime_library)


def test_dso_isolation_rejects_unparseable_dt_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, runtime_library, binaries, _ = _pipeline_fixture(tmp_path, monkeypatch)
    pipeline.context.output = lambda command, **_kwargs: (
        " 0x0000000000000001 (NEEDED) libpython3.12.so.1.0"
        if Path(command[2]) == binaries["model"]
        else _readelf_output("libstdc++.so.6")
    )

    with pytest.raises(CiError, match=r"could not parse DT_NEEDED entry"):
        pipeline._validate_dso("fixture", runtime_library)

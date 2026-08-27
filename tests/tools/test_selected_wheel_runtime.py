# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the explicit nightly selected-wheel runtime."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.ci.context import CiContext
from tools.ci import model_proof_inner as model_proof_inner_module
from tools.ci.model_proof import ModelProofRequest, ModelProofRunner
from tools.ci.model_proof_inner import ModelProofInnerPipeline
from tools.ci.process import CiError
from tools.ci.quality import UnitTestRunner
from tools.ci.selected_wheel import SelectedWheelRuntime


TENSORRT_VERSION = "11.2.1.2"
PACKAGE_VERSION = "0.1.0+trt112"
PYTHON_TAG = "py312"


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        '[tool.tensorrt-model-connect.package]\nbase-version = "0.1.0"\n',
        encoding="utf-8",
    )
    return repository


def _wheel(directory: Path, suffix: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / (
        f"tensorrt_model_connect-{PACKAGE_VERSION}-{PYTHON_TAG}-none-"
        f"manylinux_2_39_aarch64{suffix}.whl"
    )
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: tensorrt-model-connect\n"
        f"Version: {PACKAGE_VERSION}\n"
        f"Requires-Dist: tensorrt == {TENSORRT_VERSION}\n\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tensorrt_model_connect-0.1.0.dist-info/METADATA", metadata)
    return wheel


def _environment(directory: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin",
        "TRTMC_SELECTED_WHEEL_DIR": str(directory),
        "TRTMC_SELECTED_WHEEL_PYTHON_TAG": PYTHON_TAG,
        "TRTMC_SELECTED_WHEEL_TENSORRT_VERSION": TENSORRT_VERSION,
    }


def test_selected_wheel_installs_one_wheel_without_dependencies_and_records_safe_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    wheel = _wheel(tmp_path / "wheels")
    context = CiContext(repository, _environment(wheel.parent))
    commands: list[list[object]] = []
    work = tmp_path / "work"
    provenance = tmp_path / "artifacts/selected-wheel.json"
    base_python = tmp_path / "venv/bin/python"
    base_python.parent.mkdir(parents=True)
    base_python.symlink_to(sys.executable)

    def run(command: list[object], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "pip" in command:
            target = work / "site-packages"
            package = target / "tensorrt_model_connect"
            server_package = target / "trtmc_server"
            (package / "bin").mkdir(parents=True)
            (package / "__init__.py").touch()
            (package / "bin/trtmc").write_bytes(b"\x7fELFfixture")
            server_package.mkdir()
            (server_package / "__init__.py").touch()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def output(command: list[object], **_kwargs: object) -> str:
        assert command[-2] == "-c"
        package = work / "site-packages/tensorrt_model_connect/__init__.py"
        server_package = work / "site-packages/trtmc_server/__init__.py"
        return json.dumps(
            {
                "python": str(base_python.resolve()),
                "python_tag": PYTHON_TAG,
                "package_file": str(package.resolve()),
                "server_package_file": str(server_package.resolve()),
                "legacy_server_present": False,
                "package_version": PACKAGE_VERSION,
                "tensorrt_distribution_version": TENSORRT_VERSION,
                "tensorrt_runtime_version": TENSORRT_VERSION,
                "trtmc": str((work / "site-packages/tensorrt_model_connect/bin/trtmc").resolve()),
            }
        )

    monkeypatch.setattr(context, "run", run)
    monkeypatch.setattr(context, "output", output)

    runtime = SelectedWheelRuntime.prepare(
        context,
        work,
        provenance,
        base_python=base_python,
    )

    assert runtime is not None and runtime.wheel == wheel.resolve()
    assert not [command for command in commands if command[1:3] == ["-m", "venv"]]
    install = next(command for command in commands if "pip" in command)
    assert install[0] == base_python
    assert "--no-deps" in install and install[-1] == wheel.resolve()
    assert install[install.index("--target") + 1] == runtime.site_packages
    assert runtime.python == base_python
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert payload["wheel"] == wheel.name
    assert payload["tensorrt_version"] == TENSORRT_VERSION
    assert payload["package_version"] == PACKAGE_VERSION
    assert not any(str(tmp_path) in str(value) for value in payload.values())


def test_unit_python_tests_use_selected_wheel_target_without_source_python(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    target = tmp_path / "work/site-packages"
    runtime = SelectedWheelRuntime(
        wheel=tmp_path / "wheel.whl",
        site_packages=target,
        python=Path("/opt/venv/bin/python"),
        trtmc=target / "tensorrt_model_connect/bin/trtmc",
        python_tag=PYTHON_TAG,
        tensorrt_version=TENSORRT_VERSION,
        package_version=PACKAGE_VERSION,
        provenance=tmp_path / "selected-wheel.json",
    )

    class RecordingContext:
        env = {
            "GITHUB_WORKSPACE": str(repository),
            "TRTMC_PREMERGE_UNIT_SCOPE": "builder",
            "TRTMC_CI_SCRATCH_DIR": str(tmp_path / "scratch"),
            "TRTMC_UNIT_BUILD_JOBS": "1",
            "TRTMC_UNIT_TEST_JOBS": "1",
            "PATH": "/usr/bin",
        }

        def __init__(self) -> None:
            self.repository = repository
            self.calls: list[tuple[list[object], dict[str, object]]] = []

        def positive_integer(self, value: str, _name: str) -> int:
            return int(value)

        def run(self, command: list[object], **kwargs: object) -> subprocess.CompletedProcess[str]:
            self.calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        SelectedWheelRuntime,
        "prepare",
        lambda *_args, **_kwargs: runtime,
    )
    context = RecordingContext()

    UnitTestRunner(context).premerge()

    command, options = next(
        call for call in context.calls if call[0][:3] == [str(runtime.python), "-m", "pytest"]
    )
    assert command[0] == str(runtime.python)
    updates = options["updates"]
    assert isinstance(updates, dict)
    assert updates["PYTHONPATH"] == f"{runtime.site_packages}:{repository}"
    assert str(repository / "python") not in updates["PYTHONPATH"]
    assert str(repository / "server/python") not in updates["PYTHONPATH"]
    assert "--import-mode=importlib" in command
    assert updates["TRTMC_BINARY"] == str(runtime.trtmc)
    assert updates["TRTMC_TEST_INSTALLED_WHEEL"] == "1"
    assert updates["PATH"].startswith(f"{runtime.trtmc.parent}:")
    assert "VIRTUAL_ENV" not in updates
    assert not [call for call in context.calls if call[0][0] in {"cmake", "ctest"}]


def test_model_proof_mounts_selected_wheels_read_only_and_forwards_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    selected = tmp_path / "selected"
    selected.mkdir()
    context = CiContext(repository, _environment(selected))
    runner = ModelProofRunner(context, ModelProofRequest("fixture"))
    runner.lease = SimpleNamespace(
        gpu_id=0,
        slot_ids=[0],
        slots_per_gpu=1,
        resource_class="shared",
        min_free_gpu_memory_mib=0,
        lock_namespace="fixture",
    )
    runner.artifacts_dir = tmp_path / "artifacts"
    runner.artifacts_dir.mkdir()
    runner.revision = "a" * 40
    captured: list[list[object]] = []
    monkeypatch.setattr(
        context,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        runner,
        "_run_logged",
        lambda command, _path: captured.append(command) or 0,
    )

    runner._run_proof_container(
        tmp_path / "projection",
        tmp_path / "work",
        tmp_path / "hf",
        "fixture-image",
        SimpleNamespace(reference_cache=None),
        None,
    )

    command = captured[0]
    assert f"type=bind,src={selected.resolve()},dst=/selected-wheel,readonly" in command
    assert "TRTMC_SELECTED_WHEEL_DIR=/selected-wheel" in command
    assert f"TRTMC_SELECTED_WHEEL_PYTHON_TAG={PYTHON_TAG}" in command
    assert f"TRTMC_SELECTED_WHEEL_TENSORRT_VERSION={TENSORRT_VERSION}" in command


def test_model_proof_python_and_e2e_route_only_through_selected_wheel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    artifacts = tmp_path / "artifacts"
    (artifacts / "e2e").mkdir(parents=True)
    target = work / "selected-wheel-runtime/site-packages"
    runtime = SelectedWheelRuntime(
        wheel=tmp_path / "wheel.whl",
        site_packages=target,
        python=Path("/opt/venv/bin/python"),
        trtmc=target / "tensorrt_model_connect/bin/trtmc",
        python_tag=PYTHON_TAG,
        tensorrt_version=TENSORRT_VERSION,
        package_version=PACKAGE_VERSION,
        provenance=artifacts / "selected-wheel.json",
    )
    runtime.provenance.write_text(
        json.dumps(
            {
                "wheel": runtime.wheel.name,
                "package_version": PACKAGE_VERSION,
                "tensorrt_version": TENSORRT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    context = CiContext(source, {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/runtime/lib"})
    pipeline = ModelProofInnerPipeline(
        context,
        ModelProofRequest("fixture", revision="a" * 40),
    )
    pipeline.source = source
    pipeline.work = work
    pipeline.artifacts = artifacts
    pipeline.selected_wheel = runtime
    pipeline.status = SimpleNamespace(step=lambda *_args: None, fact=lambda *_args: None)
    pipeline.selection = SimpleNamespace(e2e_models=["fixture"], e2e_cases=["case"])
    logged: list[tuple[list[object], dict[str, str]]] = []

    def run_logged(
        command: list[object],
        _path: Path,
        *,
        append: bool = False,
        updates: dict[str, str] | None = None,
    ) -> None:
        del append
        logged.append((command, updates or {}))

    def run(command: list[object], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "verify-builds" in command:
            (artifacts / "engine-build-verification.json").write_text(
                json.dumps({"passed": True, "records": [], "builds_per_model": 1}),
                encoding="utf-8",
            )
        if "verify-results" in command:
            (artifacts / "e2e-verification.json").write_text(
                json.dumps({"results": [{"proof_kind": "reference"}]}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(pipeline, "_run_logged", run_logged)
    monkeypatch.setattr(context, "run", run)
    validation: dict[str, object] = {}

    class RecordingValidationRunner:
        def __init__(self, _context, _suite, _model, **kwargs: object) -> None:
            validation.update(kwargs)

        def run(self) -> bool:
            return True

    monkeypatch.setattr(
        model_proof_inner_module,
        "ValidationRunner",
        RecordingValidationRunner,
    )

    pipeline._run_python_tests({"python_tests": ["tests/fixture.py"]})
    pipeline._run_e2e({"e2e_test": "tests/e2e.py"})
    pipeline._run_validation("fixture")

    for command, environment in logged:
        assert command[0] == str(runtime.python)
        assert environment["PYTHONPATH"] == f"{runtime.site_packages}:{source}"
        assert str(source / "python") not in environment["PYTHONPATH"]
        assert environment["TRTMC_BINARY"] == str(runtime.trtmc)
        assert environment["TRTMC_TEST_INSTALLED_WHEEL"] == "1"
        assert "--import-mode=importlib" in command
    e2e_command, e2e_environment = logged[-1]
    assert e2e_command[e2e_command.index("--trtmc-binary") + 1] == str(runtime.trtmc)
    assert e2e_command[e2e_command.index("--hf-python") + 1] == str(runtime.python)
    assert e2e_environment["LD_LIBRARY_PATH"] == "/runtime/lib"
    assert str(work / "build") not in e2e_environment["LD_LIBRARY_PATH"]
    assert validation == {
        "python": str(runtime.python),
        "trtmc": str(runtime.trtmc),
        "pythonpath": f"{runtime.site_packages}:{source}",
        "installed_wheel": True,
    }
    proof = pipeline._selected_wheel_proof()
    assert proof["selected_wheel_evidence"] == "selected-wheel.json"
    assert proof["selected_wheel"] == runtime.wheel.name
    assert proof["selected_wheel_package_version"] == PACKAGE_VERSION
    assert not any(str(tmp_path) in str(value) for value in proof.values())


def test_model_proof_stages_wheel_model_dso_but_keeps_source_cpp_tests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    runtime_library = "libtrtmc_model_fixture.so"
    scratch = work / "build/models/fixture" / runtime_library
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"\x7fELFsource")
    target = work / "selected-wheel-runtime/site-packages"
    packaged = target / "tensorrt_model_connect/bin" / runtime_library
    packaged.parent.mkdir(parents=True)
    packaged.write_bytes(b"\x7fELFwheel")
    packaged_core = packaged.parent / "libtrtmc_core.so"
    packaged_core.write_bytes(b"\x7fELFwheel-core")
    runtime = SelectedWheelRuntime(
        wheel=tmp_path / "wheel.whl",
        site_packages=target,
        python=Path("/opt/venv/bin/python"),
        trtmc=target / "tensorrt_model_connect/bin/trtmc",
        python_tag=PYTHON_TAG,
        tensorrt_version=TENSORRT_VERSION,
        package_version=PACKAGE_VERSION,
        provenance=artifacts / "selected-wheel.json",
    )
    context = CiContext(source, {})
    monkeypatch.setattr(context, "output", lambda *_args, **_kwargs: "")
    facts: dict[str, object] = {}
    pipeline = ModelProofInnerPipeline(
        context,
        ModelProofRequest("fixture", revision="a" * 40),
    )
    pipeline.source = source
    pipeline.work = work
    pipeline.artifacts = artifacts
    pipeline.selected_wheel = runtime
    pipeline.status = SimpleNamespace(
        step=lambda *_args: None,
        fact=lambda key, value: facts.__setitem__(key, value),
    )

    staged, scratch_digest, runtime_source = pipeline._validate_dso(
        "fixture", runtime_library
    )

    assert staged.read_bytes() == packaged.read_bytes()
    assert staged.read_bytes() != scratch.read_bytes()
    assert scratch_digest == hashlib.sha256(scratch.read_bytes()).hexdigest()
    assert runtime_source == "selected-wheel"
    assert facts["runtime_library_source"] == "selected-wheel"
    assert facts["runtime_library_sha256"] == hashlib.sha256(
        packaged.read_bytes()
    ).hexdigest()
    staged_core = staged.parent / packaged_core.name
    core_digest = hashlib.sha256(packaged_core.read_bytes()).hexdigest()
    assert staged_core.is_file() and not staged_core.is_symlink()
    assert staged_core.read_bytes() == packaged_core.read_bytes()
    assert staged_core.resolve().is_relative_to(staged.parent.resolve())
    assert facts["runtime_core_library"] == packaged_core.name
    assert facts["runtime_core_library_source"] == "selected-wheel"
    assert facts["runtime_core_library_sha256"] == core_digest
    assert facts["staged_runtime_core_library_sha256"] == core_digest

    outside_core = tmp_path / "outside-libtrtmc_core.so"
    outside_core.write_bytes(b"\x7fELFoutside")
    packaged_core.unlink()
    packaged_core.symlink_to(outside_core)
    with pytest.raises(CiError, match="selected wheel core DSO is missing or unsafe"):
        pipeline._validate_dso("fixture", runtime_library)

    calls: list[tuple[list[object], dict[str, str]]] = []
    monkeypatch.setattr(
        pipeline,
        "_run_logged",
        lambda command, _path, **kwargs: calls.append((command, kwargs["updates"])),
    )
    pipeline._run_cpp_tests(["test_fixture"])
    assert calls[0][1]["TRTMC_MODEL_PLUGIN_DIR"] == str(work / "build/models")

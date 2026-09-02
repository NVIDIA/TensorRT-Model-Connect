# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for the repository-local TRTMC devToolkit."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from tools.ci.package import (
    WheelArchiveValidator,
    WheelPackageManager,
    _validate_manylinux_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLKIT_ROOT = REPO_ROOT / "scripts" / "devToolkit"
sys.path.insert(0, str(DEVTOOLKIT_ROOT))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DockerTarget,
    LocalTarget,
    PrepareRequest,
    validation_handoff,
)
from trtmc_devtoolkit.cohorts import CohortRegistry, normalize_architecture  # noqa: E402
from trtmc_devtoolkit.doctor import EnvironmentDoctor  # noqa: E402
from trtmc_devtoolkit.models import (  # noqa: E402
    DevToolkitError,
    EnvironmentHandle,
    ToolchainObservation,
)
from trtmc_devtoolkit.planner import image_fingerprint  # noqa: E402
from trtmc_devtoolkit.receipt import write_failure, write_success  # noqa: E402
from trtmc_devtoolkit.targets import (  # noqa: E402
    LocalEnvironment,
    _development_runtime_environment,
    _write_local_activation,
)
from trtmc_devtoolkit.toolchain import ManagedLocalProvisioner  # noqa: E402


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []

    def run(
        self,
        command,
        *,
        cwd: Path,
        env=None,
        check: bool = True,
        capture_output: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, timeout
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        self.environments.append(dict(env) if env is not None else None)
        output = ""
        returncode = 0
        if (
            arguments[:3]
            in (
                ["docker", "image", "inspect"],
                ["docker", "container", "inspect"],
            )
            and "--format" not in arguments
        ):
            returncode = 1
        elif arguments[0] == "nvidia-smi":
            output = "NVIDIA GB300, GPU-uuid, 595.58.03, 10.0, 191000\n"
        elif arguments[:2] == ["docker", "version"]:
            output = "28.0.0\n"
        elif arguments[:2] == ["docker", "exec"]:
            if "import ctypes, sys, tensorrt" in " ".join(arguments):
                output = "3.12 11.1.0.106 11.1.0.106\n"
            elif "import ctypes, re, sys, tensorrt" in " ".join(arguments):
                output = "3.12 11.1.0.106 11.1.0.106 11.1.0.106\n"
            elif arguments[-2:] == ["nvcc", "--version"]:
                output = "Cuda compilation tools, release 13.3, V13.3.0\n"
            elif "--query-gpu=compute_cap" in arguments:
                output = "10.0\n"
            elif arguments[-3:-1] == ["sh", "-c"]:
                output = "100"
        elif "-c" in arguments and "importlib.metadata.requires" in arguments[-1]:
            output = json.dumps(
                [
                    "sentencepiece>=0.1.99",
                    "apache-tvm-ffi==0.1.12",
                ]
            )
        result = subprocess.CompletedProcess(arguments, returncode, output, "")
        if check and returncode:
            raise DevToolkitError(f"fake command failed: {arguments}")
        return result


class LocalProbeRunner(RecordingRunner):
    def __init__(self, *, python_version: str = "3.12", native_version: str = "11.1.0.106"):
        super().__init__()
        self.python_version = python_version
        self.native_version = native_version

    def run(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        result = super().run(command, **kwargs)
        arguments = [str(item) for item in command]
        output = result.stdout
        if arguments[0] == "nvcc":
            output = "Cuda compilation tools, release 13.3, V13.3.0\n"
        elif "-c" in arguments:
            script = arguments[-1]
            if "sys.version_info" in script:
                output = f"{self.python_version}\n"
            elif "import tensorrt" in script:
                output = "11.1.0.106\n"
            elif "getInferLib" in script:
                output = f"{self.native_version}\n"
        return subprocess.CompletedProcess(arguments, result.returncode, output, result.stderr)


class TimeoutRunner(RecordingRunner):
    def run(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        raise subprocess.TimeoutExpired([str(item) for item in command], 30)


class DockerHeaderMismatchRunner(RecordingRunner):
    def run(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        result = super().run(command, **kwargs)
        arguments = [str(item) for item in command]
        if arguments[:2] == ["docker", "exec"] and "NvInferVersion.h" in " ".join(arguments):
            return subprocess.CompletedProcess(
                arguments,
                result.returncode,
                "3.12 11.1.0.106 11.1.0.106 11.0.0.1\n",
                result.stderr,
            )
        return result


class DockerCudaMismatchRunner(RecordingRunner):
    def run(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        result = super().run(command, **kwargs)
        arguments = [str(item) for item in command]
        if arguments[:2] == ["docker", "exec"] and arguments[-2:] == ["nvcc", "--version"]:
            return subprocess.CompletedProcess(
                arguments,
                result.returncode,
                "Cuda compilation tools, release 12.8, V12.8.0\n",
                result.stderr,
            )
        return result


def _write_trt_header(path: Path, version: str = "11.1.0.106") -> None:
    major, minor, patch, build = version.split(".")
    path.write_text(
        "\n".join(
            (
                f"#define NV_TENSORRT_MAJOR {major}",
                f"#define NV_TENSORRT_MINOR {minor}",
                f"#define NV_TENSORRT_PATCH {patch}",
                f"#define NV_TENSORRT_BUILD {build}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _minimal_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    cohort_dir = repository / "configs" / "environment-cohorts"
    cohort_dir.mkdir(parents=True)
    shutil.copy(
        REPO_ROOT / "configs" / "environment-cohorts" / "trt111-cu133.json",
        cohort_dir / "trt111-cu133.json",
    )
    shutil.copy(REPO_ROOT / "Dockerfile.dev.aarch64", repository / "Dockerfile.dev.aarch64")
    requirements = repository / "requirements"
    requirements.mkdir()
    (requirements / "community-ci.txt").write_text("pytest==8.4.2\n", encoding="utf-8")
    return repository


def test_resolves_exact_supported_cohort() -> None:
    registry = CohortRegistry(REPO_ROOT / "configs" / "environment-cohorts")

    cohort = registry.resolve(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        python_version="3.12",
        target="docker",
        allow_experimental=False,
    )

    assert cohort.id == "trt111-cu133"
    assert cohort.architectures["aarch64"].docker_context == "requirements"


def test_checked_in_cohorts_match_schema_and_package_default() -> None:
    root = REPO_ROOT / "configs" / "environment-cohorts"
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    cohorts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
        if path.name != "schema.json"
    ]
    for cohort in cohorts:
        jsonschema.validate(cohort, schema)
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        package = tomllib.load(stream)["tool"]["tensorrt-model-connect"]["package"]

    docker_supported = [
        cohort
        for cohort in cohorts
        if cohort["status"] == "supported" and "docker" in cohort["targets"]
    ]
    assert len(docker_supported) == 1
    assert docker_supported[0]["tensorrt"]["version"] == package["default-tensorrt-version"]
    for architecture in ("x86_64", "aarch64"):
        dockerfile = REPO_ROOT / docker_supported[0]["architectures"][architecture]["dockerfile"]
        assert f"tensorrt.__version__ == '{package['default-tensorrt-version']}'" in (
            dockerfile.read_text(encoding="utf-8")
        )


def test_resolves_trt112_managed_local_cohort() -> None:
    registry = CohortRegistry(REPO_ROOT / "configs" / "environment-cohorts")

    cohort = registry.resolve(
        tensorrt="11.2.1.2",
        cuda="13.3",
        architecture="x86_64",
        python_version="3.12",
        target="local",
        allow_experimental=False,
    )

    assert cohort.id == "trt112-cu133"
    assert cohort.targets == ("local",)
    assert cohort.architectures["x86_64"].tensorrt_headers.sha256 == (
        "419b21ac4cdb18b4ddf65b72e5e816fccab9db789730aa96cf228da982104e29"
    )


def test_rejects_trt112_docker_target_before_provisioning() -> None:
    toolkit = DevToolkit.from_checkout(REPO_ROOT)

    with pytest.raises(DevToolkitError, match="does not support the docker target"):
        toolkit.plan(
            PrepareRequest(
                tensorrt="11.2.1.2",
                cuda="13.3",
                architecture="x86_64",
                target=DockerTarget(),
            )
        )


def test_rejects_nearest_or_partial_version_match() -> None:
    registry = CohortRegistry(REPO_ROOT / "configs" / "environment-cohorts")

    with pytest.raises(DevToolkitError, match="No exact environment cohort"):
        registry.resolve(
            tensorrt="11.1",
            cuda="13.3",
            architecture="aarch64",
            python_version="3.12",
            target="local",
            allow_experimental=False,
        )


def test_rejects_python_version_not_present_in_docker_image(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
    )

    with pytest.raises(DevToolkitError, match="Docker image uses Python 3.12"):
        toolkit.plan(
            PrepareRequest(
                tensorrt="11.1.0.106",
                cuda="13.3",
                python_version="3.10",
                architecture="aarch64",
                target=DockerTarget(),
            )
        )


def test_local_doctor_checks_exact_python_and_native_tensorrt(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    _write_trt_header(include_dir / "NvInferVersion.h")
    architecture = replace(
        cohort.architectures["aarch64"],
        tensorrt_include_dir=str(include_dir),
        tensorrt_library_dir=str(tmp_path / "lib"),
    )
    cohort = replace(cohort, architectures={"aarch64": architecture})
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(python="python3.12", dependency_mode="system"),
    )

    probes, sm = EnvironmentDoctor(repository, LocalProbeRunner()).inspect(
        request, cohort, "aarch64"
    )

    assert sm == "100"
    assert {probe.name: probe.status for probe in probes}["python"] == "pass"
    assert {probe.name: probe.status for probe in probes}["tensorrt-native"] == "pass"


@pytest.mark.parametrize(
    ("runner", "failure"),
    (
        (LocalProbeRunner(python_version="3.11"), "python: requested 3.12; found 3.11"),
        (
            LocalProbeRunner(native_version="11.0.0.1"),
            "tensorrt-native: requested 11.1.0.106; found 11.0.0.1",
        ),
    ),
)
def test_local_doctor_rejects_version_mismatch(
    tmp_path: Path, runner: LocalProbeRunner, failure: str
) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    _write_trt_header(include_dir / "NvInferVersion.h")
    architecture = replace(
        cohort.architectures["aarch64"],
        tensorrt_include_dir=str(include_dir),
        tensorrt_library_dir=str(tmp_path / "lib"),
    )
    cohort = replace(cohort, architectures={"aarch64": architecture})
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(python="python3.12", dependency_mode="system"),
    )

    with pytest.raises(DevToolkitError, match=re.escape(failure)):
        EnvironmentDoctor(repository, runner).inspect(request, cohort, "aarch64")


def test_local_doctor_rejects_tensorrt_header_version_mismatch(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    include_dir = tmp_path / "include"
    include_dir.mkdir()
    _write_trt_header(include_dir / "NvInferVersion.h", "11.0.0.1")
    architecture = replace(
        cohort.architectures["aarch64"],
        tensorrt_include_dir=str(include_dir),
        tensorrt_library_dir=str(tmp_path / "lib"),
    )
    cohort = replace(cohort, architectures={"aarch64": architecture})
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(python="python3.12", dependency_mode="system"),
    )

    with pytest.raises(
        DevToolkitError,
        match="tensorrt-headers: requested 11.1.0.106; found 11.0.0.1",
    ):
        EnvironmentDoctor(repository, LocalProbeRunner()).inspect(request, cohort, "aarch64")


def test_system_local_build_uses_the_tensorrt_path_validated_by_doctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _minimal_repository(tmp_path)
    include_dir = tmp_path / "custom-trt-include"
    include_dir.mkdir()
    _write_trt_header(include_dir / "NvInferVersion.h")
    monkeypatch.setenv("TRTMC_TRT_INCLUDE_DIR", str(include_dir))
    runner = LocalProbeRunner()
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
        runner=runner,
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.1.0.106",
            cuda="13.3",
            architecture="aarch64",
            target=LocalTarget(python="python3.12", dependency_mode="system"),
        )
    )

    result = toolkit.apply(plan)

    configure = next(command for command in runner.commands if command[:2] == ["cmake", "-S"])
    assert f"-DTRTMC_TRT_INCLUDE_DIR={include_dir}" in configure
    assert result.environment.environment["TRTMC_TRT_INCLUDE_DIR"] == str(include_dir)


def test_managed_local_doctor_checks_bootstrap_prerequisites_not_system_toolchain(
    tmp_path: Path,
) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    runner = LocalProbeRunner()
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(python="python3.12", dependency_mode="managed"),
    )

    probes, sm = EnvironmentDoctor(repository, runner).inspect(request, cohort, "aarch64")

    names = {probe.name for probe in probes}
    assert sm == "100"
    assert {"cxx-compiler", "git", "dpkg-deb", "python"} <= names
    assert "cuda-toolkit" not in names
    assert "tensorrt-python" not in names
    assert not any(command[0] == "nvcc" for command in runner.commands)


def test_environment_doctor_classifies_probe_timeout(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(python="python3.12", dependency_mode="managed"),
    )

    with pytest.raises(DevToolkitError, match="gpu: Command.*timed out after 30 seconds"):
        EnvironmentDoctor(repository, TimeoutRunner()).inspect(request, cohort, "aarch64")


def test_docker_prepare_rejects_tensorrt_header_version_mismatch(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
        runner=DockerHeaderMismatchRunner(),
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.1.0.106",
            cuda="13.3",
            architecture="aarch64",
            target=DockerTarget(),
        )
    )

    with pytest.raises(DevToolkitError, match="Container TensorRT .*header mismatch"):
        toolkit.apply(plan)


def test_docker_prepare_rejects_cuda_version_mismatch(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
        runner=DockerCudaMismatchRunner(),
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.1.0.106",
            cuda="13.3",
            architecture="aarch64",
            target=DockerTarget(),
        )
    )

    with pytest.raises(DevToolkitError, match="Container CUDA mismatch"):
        toolkit.apply(plan)


def test_managed_toolchain_header_and_linker_contract(tmp_path: Path) -> None:
    header = tmp_path / "NvInferVersion.h"
    header.write_text(
        "\n".join(
            (
                "#define TRT_MAJOR_ENTERPRISE 11",
                "#define TRT_MINOR_ENTERPRISE 1",
                "#define TRT_PATCH_ENTERPRISE 0",
                "#define TRT_BUILD_ENTERPRISE 106",
                "#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE",
                "#define NV_TENSORRT_MINOR TRT_MINOR_ENTERPRISE",
                "#define NV_TENSORRT_PATCH TRT_PATCH_ENTERPRISE",
                "#define NV_TENSORRT_BUILD TRT_BUILD_ENTERPRISE",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libnvinfer.so.11").touch()

    assert ManagedLocalProvisioner._header_version(header) == "11.1.0.106"
    linker = ManagedLocalProvisioner._ensure_linker_name(library_dir, "libnvinfer.so", "11")

    assert linker.is_symlink()
    assert linker.resolve() == library_dir / "libnvinfer.so.11"


def test_local_activation_restores_full_runtime_environment(tmp_path: Path) -> None:
    venv = tmp_path / "venv with spaces"
    activate = venv / "bin" / "activate"
    activate.parent.mkdir(parents=True)
    activate.write_text("export FROM_VENV=ready\n", encoding="utf-8")
    script = tmp_path / "activate.sh"
    environment = {
        "CUDA_HOME": "/managed cuda",
        "TRTMC_MODEL_PLUGIN_DIR": "/models/it's-ready",
    }

    _write_local_activation(script, venv, environment)
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; printf "%s\\n" "$FROM_VENV" "$CUDA_HOME" "$TRTMC_MODEL_PLUGIN_DIR"',
            "bash",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["ready", "/managed cuda", "/models/it's-ready"]
    assert f". {shlex.quote(str(activate))}" in script.read_text(encoding="utf-8")


def test_development_runtime_environment_places_native_cli_on_path(tmp_path: Path) -> None:
    build = tmp_path / "build-sm103"

    environment = _development_runtime_environment({"PATH": "/venv/bin"}, build)

    assert environment["PATH"] == f"{build}:/venv/bin"
    assert environment["TRTMC_MODEL_PLUGIN_DIR"] == str(build / "models")


def test_managed_local_editable_install_preserves_cohort_tensorrt_version() -> None:
    runner = RecordingRunner()
    toolkit = DevToolkit.from_checkout(
        REPO_ROOT,
        source_revision_override="a" * 40,
        runner=runner,
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.2.1.2",
            cuda="13.3",
            architecture="x86_64",
            target=LocalTarget(python="python3.12"),
        )
    )

    LocalEnvironment(REPO_ROOT, runner)._build_install(
        plan,
        Path("/managed/venv/bin/python"),
        {"PATH": "/managed/venv/bin"},
        "100",
    )

    editable_index = next(
        index
        for index, command in enumerate(runner.commands)
        if command[:4] == ["/managed/venv/bin/python", "-m", "pip", "install"]
    )
    assert runner.environments[editable_index] is not None
    assert runner.environments[editable_index]["TRTMC_PACKAGE_TENSORRT_VERSION"] == "11.2.1.2"


def test_system_local_installs_base_dependencies_without_replacing_tensorrt() -> None:
    runner = RecordingRunner()
    toolkit = DevToolkit.from_checkout(
        REPO_ROOT,
        source_revision_override="a" * 40,
        runner=runner,
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.2.1.2",
            cuda="13.3",
            architecture="aarch64",
            target=LocalTarget(python="python3.12", dependency_mode="system"),
        )
    )

    LocalEnvironment(REPO_ROOT, runner)._build_install(
        plan,
        Path("/system/venv/bin/python"),
        {"PATH": "/system/venv/bin"},
        "110",
    )

    installs = [
        command
        for command in runner.commands
        if command[:4] == ["/system/venv/bin/python", "-m", "pip", "install"]
    ]
    assert "--no-deps" in installs[0]
    assert installs[1][-2:] == ["sentencepiece>=0.1.99", "apache-tvm-ffi==0.1.12"]
    assert not any(argument.lower().startswith("tensorrt") for argument in installs[1][4:])


def test_terminal_receipts_are_mutually_exclusive(tmp_path: Path) -> None:
    toolkit = DevToolkit.from_checkout(
        REPO_ROOT,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.2.1.2",
            cuda="13.3",
            architecture="aarch64",
            target=LocalTarget(),
        )
    )
    environment = EnvironmentHandle(
        kind="local",
        fingerprint=plan.run_id,
        trtmc="/run/trtmc",
        python="/run/python",
        activate_command=". /run/activate.sh",
        observation=ToolchainObservation(
            python_version="3.12",
            cuda_version="13.3",
            tensorrt_python_version="11.2.1.2",
            tensorrt_native_version="11.2.1.2",
            tensorrt_header_version="11.2.1.2",
            tensorrt_include_dir="/usr/include/aarch64-linux-gnu",
            tensorrt_library="/usr/lib/aarch64-linux-gnu/libnvinfer.so",
        ),
    )

    write_failure(plan, RuntimeError("first attempt"))
    write_success(plan, environment, wheel=None, bundle=None)

    assert (plan.state_dir / "receipt.json").is_file()
    assert not (plan.state_dir / "failure-summary.json").exists()

    write_failure(plan, RuntimeError("retry"))

    assert not (plan.state_dir / "receipt.json").exists()
    assert (plan.state_dir / "failure-summary.json").is_file()


def test_success_receipt_uses_observed_toolchain_identity(tmp_path: Path) -> None:
    toolkit = DevToolkit.from_checkout(
        REPO_ROOT,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
    )
    plan = toolkit.plan(
        PrepareRequest(
            tensorrt="11.2.1.2",
            cuda="13.3",
            architecture="aarch64",
            target=LocalTarget(),
        )
    )
    observation = ToolchainObservation(
        python_version="3.12",
        cuda_version="13.3",
        tensorrt_python_version="11.2.1.2",
        tensorrt_native_version="11.2.1.2",
        tensorrt_header_version="11.2.1.2",
        tensorrt_include_dir="/resolved/include",
        tensorrt_library="/resolved/lib/libnvinfer.so",
    )
    environment = EnvironmentHandle(
        kind="local",
        fingerprint=plan.run_id,
        trtmc="/run/trtmc",
        python="/run/python",
        activate_command=". /run/activate.sh",
        observation=observation,
    )

    receipt = json.loads(
        write_success(plan, environment, wheel=None, bundle=None).read_text(encoding="utf-8")
    )

    assert receipt["tensorrt"] == observation.tensorrt_native_version
    assert receipt["cuda"] == observation.cuda_version
    assert receipt["attestation"]["tensorrt_header_version"] == "11.2.1.2"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("amd64", "x86_64"), ("arm64", "aarch64"), ("aarch64", "aarch64")),
)
def test_normalizes_architecture_aliases(raw: str, expected: str) -> None:
    assert normalize_architecture(raw) == expected


def test_image_fingerprint_tracks_dockerfile_and_context(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    cohort = CohortRegistry(repository / "configs" / "environment-cohorts").load_all()[0]
    contract = cohort.architectures["aarch64"]

    initial = image_fingerprint(repository, cohort.source, contract)
    (repository / "requirements" / "community-ci.txt").write_text(
        "pytest==8.4.3\n", encoding="utf-8"
    )

    assert image_fingerprint(repository, cohort.source, contract) != initial


def test_plan_is_read_only_and_apply_prepares_owned_container(
    tmp_path: Path,
) -> None:
    repository = _minimal_repository(tmp_path)
    runner = RecordingRunner()
    state_root = tmp_path / "runs"
    toolkit = DevToolkit.from_checkout(
        repository,
        state_root=state_root,
        source_revision_override="a" * 40,
        runner=runner,
    )
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=DockerTarget(gpu="0", container_name="trtmc-dev-gb300-test"),
    )

    plan = toolkit.plan(request)

    assert not plan.state_dir.exists()
    assert plan.state_dir.parent == state_root
    assert [step.id for step in plan.steps] == [
        "doctor",
        "provision",
        "build-install",
        "verify-install",
        "receipt",
    ]

    result = toolkit.apply(plan)

    assert result.receipt.is_file()
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "ready"
    assert receipt["environment"]["container_name"] == "trtmc-dev-gb300-test"
    docker_build = next(
        command for command in runner.commands if command[:2] == ["docker", "build"]
    )
    assert docker_build[-1] == str(repository / "requirements")
    docker_run = next(command for command in runner.commands if command[:2] == ["docker", "run"])
    assert f"org.nvidia.trtmc.devtoolkit-environment={plan.environment_id}" in docker_run
    assert ["--gpus", "device=0"] == docker_run[
        docker_run.index("--gpus") : docker_run.index("--gpus") + 2
    ]

    handoff = validation_handoff(
        result,
        model="qwen3-0.6b",
        workload="qwen.generate",
        bundle=plan.state_dir / "qwen.bundle",
        output=plan.state_dir / "validation",
    )
    assert handoff.command[:3] == ("docker", "exec", "--env")
    assert "tools/trtmc_validate.py" in handoff.command
    assert "/trtmc-devtoolkit-run/qwen.bundle" in handoff.command


def test_legacy_environment_state_survives_source_revisions(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)
    request = PrepareRequest(
        tensorrt="11.1.0.106",
        cuda="13.3",
        architecture="aarch64",
        target=LocalTarget(),
    )
    first = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="a" * 40,
    ).plan(request)
    second = DevToolkit.from_checkout(
        repository,
        state_root=tmp_path / "runs",
        source_revision_override="b" * 40,
    ).plan(request)

    assert first.environment_id == second.environment_id
    assert first.state_dir == second.state_dir
    assert first.run_id != second.run_id


def test_rejects_invalid_source_revision_override(tmp_path: Path) -> None:
    repository = _minimal_repository(tmp_path)

    with pytest.raises(DevToolkitError, match="source_revision_override"):
        DevToolkit.from_checkout(repository, source_revision_override="working-tree")


def test_x86_64_wheel_platform_is_accepted_and_selected(tmp_path: Path) -> None:
    wheel = tmp_path / "dist" / "tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_x86_64.whl"
    wheel.parent.mkdir()
    wheel.touch()
    context = SimpleNamespace(
        repository=tmp_path,
        env={"TRTMC_PACKAGE_WHEEL_ARCH": "manylinux_2_39_x86_64"},
    )

    validator = WheelArchiveValidator(context, "manylinux_2_39_x86_64")

    assert validator.architecture == "x86_64"
    assert WheelPackageManager(context).select_wheel("py312") == wheel


def test_x86_64_auditwheel_result_is_validated_for_its_architecture(tmp_path: Path) -> None:
    wheel = tmp_path / "tensorrt_model_connect-0.1.0-py312-none-manylinux_2_39_x86_64.whl"

    _validate_manylinux_audit(
        wheel,
        'The wheel is consistent with the following platform tag: "manylinux_2_39_x86_64".',
        platform="manylinux_2_39_x86_64",
        architecture="x86_64",
        max_glibc_minor=39,
    )

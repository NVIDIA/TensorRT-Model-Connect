# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public capability API tests for the repository-local TRTMC DevToolkit."""

from __future__ import annotations

import hashlib
import json
import sys
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLKIT_ROOT = REPO_ROOT / "scripts" / "devToolkit"
sys.path.insert(0, str(DEVTOOLKIT_ROOT))

from trtmc_devtoolkit import (  # noqa: E402
    ArtifactPin,
    ArtifactUnavailable,
    AttestationFailed,
    BuildSpec,
    CommandSpec,
    ContextLock,
    ContextHandle,
    CudaPolicy,
    DevToolkit,
    EnvironmentRequest,
    ExecutionContext,
    ExecutionTarget,
    IncompatibleCombination,
    ProviderDescriptor,
    ProviderRegistry,
    ProvisionPolicy,
    ToolchainObservation,
    ToolchainCandidate,
    ToolchainSource,
    repository_path,
)
from trtmc_devtoolkit.models import DevToolkitError  # noqa: E402


def test_extension_protocols_are_public() -> None:
    assert ExecutionContext.__name__ == "ExecutionContext"
    assert ToolchainSource.__name__ == "ToolchainSource"


class BuiltinProbeRunner:
    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, env, check, capture_output, timeout
        arguments = [str(item) for item in command]
        if arguments[-1] == "--version":
            output = "Cuda compilation tools, release 12.8, V12.8.0\n"
        elif "sys.version_info" in arguments[-1]:
            output = "3.12\n"
        elif "tensorrt.__version__" in arguments[-1]:
            output = "11.2.0.113\n"
        elif "getInferLib" in arguments[-1]:
            output = "11.2.0.113\n"
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, 0, output, "")


class DockerAdoptionRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []

    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, check, capture_output, timeout
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        self.environments.append(dict(env) if env is not None else None)
        if arguments[:2] == ["docker", "inspect"]:
            output = json.dumps(
                {
                    "Id": "container-123",
                    "Image": "sha256:image-456",
                    "State": {"Running": True},
                    "Config": {"Image": "campaign:latest"},
                }
            )
        elif arguments[:2] == ["docker", "exec"]:
            output = json.dumps(
                {
                    "python": "3.12",
                    "python_executable": "/usr/bin/python3",
                    "cuda": "12.8",
                    "cuda_root": "/usr/local/cuda-12.8",
                    "tensorrt_python": "11.0.2.2",
                    "tensorrt_native": "11.0.2.2",
                    "tensorrt_headers": "11.0.2.2",
                    "tensorrt_include_dir": "/usr/include/aarch64-linux-gnu",
                    "tensorrt_library": "/usr/lib/aarch64-linux-gnu/libnvinfer.so.11",
                    "cuda_complete": True,
                    "architecture": "aarch64",
                }
            )
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(arguments, 0, output, "")


class ManagedProvisionRunner:
    def __init__(self, root: Path) -> None:
        self.cuda = root / "managed-cuda"
        self.trt = root / "managed-trt"
        for directory in (self.cuda / "bin", self.cuda / "include", self.cuda / "lib"):
            directory.mkdir(parents=True)
        for relative in (
            "bin/nvcc",
            "include/cuda.h",
            "lib/libcudart.so",
            "lib/libcublas.so",
            "lib/libcurand.so",
        ):
            (self.cuda / relative).touch()
        self.trt.mkdir()
        (self.trt / "libnvinfer.so").touch()

    def run(self, command, **kwargs):
        del kwargs
        arguments = [str(item) for item in command]
        output = ""
        if arguments[:2] == ["dpkg-deb", "--extract"]:
            include = Path(arguments[3]) / "usr" / "include" / "x86_64-linux-gnu"
            include.mkdir(parents=True)
            (include / "NvInferVersion.h").write_text(
                "\n".join(
                    (
                        "#define NV_TENSORRT_MAJOR 11",
                        "#define NV_TENSORRT_MINOR 2",
                        "#define NV_TENSORRT_PATCH 0",
                        "#define NV_TENSORRT_BUILD 113",
                    )
                ),
                encoding="utf-8",
            )
        elif "m.distribution" in arguments[-1]:
            output = json.dumps(
                {
                    "cuda_root": str(self.cuda),
                    "trt_library": str(self.trt / "libnvinfer.so"),
                }
            )
        elif arguments[-1] == "--version":
            output = "Cuda compilation tools, release 13.3, V13.3.0\n"
        elif "sys.version_info" in arguments[-1]:
            output = "3.12\n"
        elif "tensorrt.__version__" in arguments[-1]:
            output = "11.2.0.113\n"
        elif "getInferLib" in arguments[-1]:
            output = "11.2.0.113\n"
        return subprocess.CompletedProcess(arguments, 0, output, "")


class CommandRecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command, **kwargs):
        del kwargs
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        output = "trtmc 0.1\n"
        if arguments[0] == "sha256sum":
            output = f"{'c' * 64}  {arguments[1]}\n"
        return subprocess.CompletedProcess(arguments, 0, output, "")


class NonzeroCommandRunner:
    def __init__(self) -> None:
        self.checks: list[bool] = []

    def run(
        self,
        command,
        *,
        cwd,
        env=None,
        check=True,
        capture_output=False,
        timeout=None,
    ):
        del cwd, env, capture_output, timeout
        arguments = [str(item) for item in command]
        self.checks.append(check)
        return subprocess.CompletedProcess(arguments, 7, "", "expected failure")


class SecretEchoFailureRunner:
    def run(self, command, *, env=None, **kwargs):
        del command, kwargs
        raise RuntimeError(f"runner leaked environment: {env}")


class StaticLocalContext:
    descriptor = ProviderDescriptor("test-local", "tests==1", 1)

    def resolve(self, request, *, repository, runner):
        del repository, runner
        return ContextLock(
            provider=self.descriptor,
            operating_system="linux",
            architecture=request.architecture or "aarch64",
            identity={"kind": "local"},
            locator={"gpu": "0"},
        )

    def provision(self, lock, *, repository, state_dir, policy, runner):
        del repository, runner
        assert policy is ProvisionPolicy.ADOPT_OR_CREATE
        state_dir.mkdir(parents=True, exist_ok=True)
        return ContextHandle(
            provider=self.descriptor,
            identity=lock.context.identity,
            locator={"root": str(state_dir)},
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )

    def execute(
        self,
        context,
        command,
        *,
        repository,
        state_dir,
        runner,
        check,
        capture_output,
    ):
        del context, state_dir
        arguments = [
            str(repository / argument.path)
            if hasattr(argument, "scope") and argument.scope.value == "repository"
            else str(argument)
            for argument in command.arguments
        ]
        cwd = repository / command.cwd.path
        return runner.run(
            arguments,
            cwd=cwd,
            env=command.environment,
            check=check,
            capture_output=capture_output,
        )


class ExistingToolchainSource:
    descriptor = ProviderDescriptor("test-system", "tests==1", 1)

    def __init__(self) -> None:
        self.requested_versions: list[str] = []

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        self.requested_versions.append(request.tensorrt)
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="system",
                tensorrt=request.tensorrt,
                cuda="12.8",
                python=request.python,
                identity={"prefix": "/opt/nvidia"},
            ),
        )

    def provision(self, lock, context, *, execution, repository, state_dir, runner):
        del lock, execution, repository, state_dir, runner
        return context

    def observe(self, lock, context, *, execution, repository, runner):
        del context, execution, repository, runner
        return ToolchainObservation(
            python_version=lock.toolchain.python,
            cuda_version=lock.toolchain.cuda,
            tensorrt_python_version=lock.toolchain.tensorrt,
            tensorrt_native_version=lock.toolchain.tensorrt,
            tensorrt_header_version=lock.toolchain.tensorrt,
            tensorrt_include_dir="/opt/nvidia/include",
            tensorrt_library="/opt/nvidia/lib/libnvinfer.so",
        )


class ReplacementToolchainSource(ExistingToolchainSource):
    descriptor = ProviderDescriptor("test-system", "tests==2", 1)


class EmptySystemToolchainSource:
    descriptor = ProviderDescriptor("empty-system", "tests==1", 1)

    def resolve(self, request, context, *, repository, runner):
        del request, context, repository, runner
        return ()


class ManagedCudaToolchainSource:
    descriptor = ProviderDescriptor("test-managed", "tests==1", 1)

    def __init__(self) -> None:
        self.provisioned = False

    def resolve(self, request, context, *, repository, runner):
        del context, repository, runner
        return (
            ToolchainCandidate(
                provider=self.descriptor,
                origin="managed",
                tensorrt=request.tensorrt,
                cuda="13.3",
                python=request.python,
                identity={"distribution": "test"},
                artifacts=(
                    ArtifactPin(
                        name="cuda-toolkit",
                        uri="https://example.invalid/cuda-13.3.tar.xz",
                        sha256="a" * 64,
                        verification="pinned-digest",
                    ),
                ),
            ),
        )

    def provision(self, lock, context, *, execution, repository, state_dir, runner):
        del lock, execution, repository, state_dir, runner
        self.provisioned = True
        return context

    def observe(self, lock, context, *, execution, repository, runner):
        del context, execution, repository, runner
        return ToolchainObservation(
            python_version=lock.toolchain.python,
            cuda_version=lock.toolchain.cuda,
            tensorrt_python_version=lock.toolchain.tensorrt,
            tensorrt_native_version=lock.toolchain.tensorrt,
            tensorrt_header_version=lock.toolchain.tensorrt,
            tensorrt_include_dir="/managed/include",
            tensorrt_library="/managed/lib/libnvinfer.so",
        )


def test_arbitrary_tensorrt_reaches_provider_without_a_cohort(tmp_path: Path) -> None:
    source = ExistingToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(source)
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            cuda=CudaPolicy.system_first(fallback="13.3"),
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    assert source.requested_versions == ["11.2.0.113"]
    assert lock.tensorrt == "11.2.0.113"
    assert lock.cuda == "12.8"
    assert lock.decision == "system"
    assert len(lock.lock_id) == 64
    assert not state_root.exists()


def test_system_first_falls_back_to_managed_cuda_13_3(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    registry.register_toolchain(ManagedCudaToolchainSource())
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.0.114",
            target=ExecutionTarget("test-local"),
        )
    )

    assert lock.cuda == "13.3"
    assert lock.cuda_origin == "managed-default"
    assert lock.toolchain.artifacts[0].verification == "pinned-digest"


def test_adopt_only_never_materializes_a_managed_toolchain(tmp_path: Path) -> None:
    source = ManagedCudaToolchainSource()
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(source)
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.0.114",
            target=ExecutionTarget("test-local"),
        )
    )

    with pytest.raises(AttestationFailed, match="adopt-only"):
        toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert source.provisioned is False


@pytest.mark.parametrize("version", ["latest", "11.2", ">=11.1", "11.2.*"])
def test_environment_request_requires_an_exact_four_part_tensorrt_version(
    version: str,
) -> None:
    with pytest.raises(DevToolkitError, match="exact four-part"):
        EnvironmentRequest(tensorrt=version, target=ExecutionTarget.local())


def test_cuda_policy_rejects_an_unknown_runtime_kind() -> None:
    with pytest.raises(DevToolkitError, match="CUDA policy kind"):
        CudaPolicy("unknown")  # type: ignore[arg-type]


def test_environment_lock_rejects_a_tampered_identity(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=registry.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    with pytest.raises(DevToolkitError, match="lock ID"):
        replace(lock, lock_id="../outside")


def test_builtin_local_provider_discovers_complete_system_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = tmp_path / "cuda"
    (cuda / "bin").mkdir(parents=True)
    (cuda / "include").mkdir()
    (cuda / "lib64").mkdir()
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib64/libcudart.so",
        "lib64/libcublas.so",
        "lib64/libcurand.so",
    ):
        (cuda / relative).touch()
    trt_include = tmp_path / "trt" / "include"
    trt_library_dir = tmp_path / "trt" / "lib"
    trt_include.mkdir(parents=True)
    trt_library_dir.mkdir(parents=True)
    (trt_library_dir / "libnvinfer.so").touch()
    major, minor, patch, build = "11.2.0.113".split(".")
    (trt_include / "NvInferVersion.h").write_text(
        "\n".join(
            (
                f"#define NV_TENSORRT_MAJOR {major}",
                f"#define NV_TENSORRT_MINOR {minor}",
                f"#define NV_TENSORRT_PATCH {patch}",
                f"#define NV_TENSORRT_BUILD {build}",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("TRTMC_TRT_INCLUDE_DIR", str(trt_include))
    monkeypatch.setenv("TRTMC_TRT_LIBRARY_DIR", str(trt_library_dir))
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(python="python3.12"),
        )
    )

    assert lock.toolchain.provider.name == "system"
    assert lock.toolchain.identity["cuda_root"] == str(cuda)
    assert lock.toolchain.identity["tensorrt_include_dir"] == str(trt_include)

    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert environment.observation.cuda_version == "12.8"
    assert environment.observation.tensorrt_header_version == "11.2.0.113"


def test_resolution_distinguishes_unavailable_from_incompatible(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(EmptySystemToolchainSource())
    unavailable = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    with pytest.raises(ArtifactUnavailable):
        unavailable.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                target=ExecutionTarget("test-local"),
            )
        )

    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    incompatible = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())

    with pytest.raises(IncompatibleCombination):
        incompatible.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                cuda=CudaPolicy.exact("13.3"),
                target=ExecutionTarget("test-local"),
            )
        )


def test_provision_attests_and_writes_a_v2_receipt(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    environment = toolkit.provision(lock)

    assert environment.environment_id == lock.lock_id
    assert environment.observation.tensorrt_header_version == "11.2.0.113"
    receipt = json.loads(environment.receipt.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["lock_id"] == lock.lock_id
    assert receipt["cuda_origin"] == "system"
    assert receipt["observed"]["tensorrt_native_version"] == "11.2.0.113"


def test_provision_rejects_a_different_toolchain_provider_implementation(
    tmp_path: Path,
) -> None:
    original = ProviderRegistry()
    original.register_context(StaticLocalContext())
    original.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=original.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    replacement = ProviderRegistry()
    replacement.register_context(StaticLocalContext())
    replacement.register_toolchain(ReplacementToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=replacement.freeze(),
    )

    with pytest.raises(AttestationFailed, match="toolchain provider"):
        toolkit.provision(lock)


def test_provision_records_failure_when_locked_provider_is_not_registered(
    tmp_path: Path,
) -> None:
    original = ProviderRegistry()
    original.register_context(StaticLocalContext())
    original.register_toolchain(ExistingToolchainSource())
    lock = DevToolkit.from_checkout(tmp_path, providers=original.freeze()).resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    missing_toolchain = ProviderRegistry()
    missing_toolchain.register_context(StaticLocalContext())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=missing_toolchain.freeze(),
    )

    with pytest.raises(DevToolkitError, match="Unknown toolchain source provider"):
        toolkit.provision(lock)

    state_dir = state_root / "environments" / lock.lock_id
    failure = json.loads((state_dir / "provision-failure.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "DevToolkitError"
    assert not (state_dir / "provision-receipt.json").exists()


def test_builtin_docker_provider_adopts_a_probed_campaign_container(
    tmp_path: Path,
) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    assert lock.toolchain.provider.name == "container-image"
    assert lock.context.identity["image_id"] == "sha256:image-456"
    assert environment.context.locator["container"] == "jedha-campaign"
    assert environment.observation.tensorrt_native_version == "11.0.2.2"
    assert not any(command[:2] == ["docker", "run"] for command in runner.commands)


def test_docker_command_forwards_environment_without_values_in_argv(tmp_path: Path) -> None:
    runner = DockerAdoptionRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.0.2.2",
            target=ExecutionTarget.docker(container="jedha-campaign"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock, policy=ProvisionPolicy.ADOPT_ONLY)

    toolkit.run(
        environment,
        CommandSpec(("trtmc", "version"), environment={"TOKEN": "super-secret"}),
    )

    assert "super-secret" not in " ".join(runner.commands[-1])
    assert any(
        pair == ("--env", "TOKEN")
        for pair in zip(runner.commands[-1], runner.commands[-1][1:], strict=False)
    )
    assert runner.environments[-1] == {"TOKEN": "super-secret"}


def test_generic_command_is_routed_by_the_execution_context(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = CommandRecordingRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    result = toolkit.run(
        environment,
        CommandSpec(
            ("trtmc", "inspect", repository_path("example.bundle")),
            cwd=repository_path("."),
        ),
        capture_output=True,
    )

    assert runner.commands[-1] == ["trtmc", "inspect", str(tmp_path / "example.bundle")]
    assert result.stdout == "trtmc 0.1\n"
    receipt = json.loads(result.receipt.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["environment_id"] == environment.environment_id
    assert receipt["occurrence_id"] != receipt["invocation_digest"]


def test_run_trtmc_forwards_check_policy(tmp_path: Path) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = NonzeroCommandRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    result = toolkit.run_trtmc(environment, ("version",), check=False)

    assert result.returncode == 7
    assert runner.checks == [False]


def test_command_failure_receipt_does_not_serialize_environment_values(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    state_root = tmp_path / "state"
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=state_root,
        providers=registry.freeze(),
        runner=SecretEchoFailureRunner(),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    with pytest.raises(RuntimeError, match="super-secret"):
        toolkit.run(
            environment,
            CommandSpec(("false",), environment={"TOKEN": "super-secret"}),
        )

    receipt = next((environment.state_dir / "commands").glob("*.json"))
    assert "super-secret" not in receipt.read_text(encoding="utf-8")


def test_native_build_identity_includes_source_sm_options_and_outputs(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    runner = CommandRecordingRunner()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )
    environment = toolkit.provision(lock)

    build = toolkit.build(
        environment,
        BuildSpec(
            targets=("trtmc",),
            cmake_defines={"TRTMC_BUILD_TESTS": False},
            cuda_architectures=("100",),
            source_identity="b" * 40,
        ),
    )

    assert len(build.build_request_id) == 64
    assert len(build.build_id) == 64
    assert build.build_id != build.build_request_id
    assert build.artifacts[0].sha256 == "c" * 64
    configure = next(command for command in runner.commands if command[:2] == ["cmake", "-S"])
    assert "-DCMAKE_CUDA_ARCHITECTURES=100-real" in configure
    assert "-DTRTMC_BUILD_TESTS=OFF" in configure
    editable = next(
        command for command in runner.commands if command[1:4] == ["-m", "pip", "install"]
    )
    assert "--no-deps" in editable
    receipt = json.loads(build.receipt.read_text(encoding="utf-8"))
    assert receipt["environment_id"] == environment.environment_id
    assert receipt["source"]["revision"] == "b" * 40
    assert receipt["artifacts"][0]["sha256"] == "c" * 64


def test_cohort_is_optional_qualification_provenance_not_an_allowlist(
    tmp_path: Path,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "qualified.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "qualified-trt",
                "status": "supported",
                "targets": ["local"],
                "tensorrt": {"version": "11.2.0.113"},
                "cuda": {"version": "12.8"},
                "python_versions": ["3.12"],
                "architectures": {"aarch64": {}},
            }
        ),
        encoding="utf-8",
    )
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualification_roots=(presets,),
    )

    qualified = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
            preset="qualified-trt",
            require_qualification=True,
        )
    )
    unrestricted = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget("test-local"),
            architecture="aarch64",
        )
    )

    assert qualified.qualifications[0].name == "qualified-trt"
    assert len(qualified.qualifications[0].digest) == 64
    assert qualified.lock_id == unrestricted.lock_id


def test_required_qualification_fails_without_restricting_default_resolution(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(tmp_path, providers=registry.freeze())
    request = EnvironmentRequest(
        tensorrt="11.2.0.113",
        target=ExecutionTarget("test-local"),
        architecture="aarch64",
    )

    assert toolkit.resolve(request).qualifications == ()
    with pytest.raises(IncompatibleCombination, match="qualification"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt=request.tensorrt,
                target=request.target,
                architecture=request.architecture,
                require_qualification=True,
            )
        )


def test_invalid_optional_qualification_metadata_does_not_gate_resolution(
    tmp_path: Path,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "broken.json").write_text("{not-json", encoding="utf-8")
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualification_roots=(presets,),
    )
    request = EnvironmentRequest(
        tensorrt="11.2.0.113",
        target=ExecutionTarget("test-local"),
        architecture="aarch64",
    )

    assert toolkit.resolve(request).qualifications == ()
    with pytest.raises(DevToolkitError, match="Invalid qualification"):
        toolkit.resolve(replace(request, require_qualification=True))


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("python_versions", "3.12"),
        ("architectures", "aarch64"),
        ("targets", "local"),
    ),
)
def test_qualification_rejects_string_in_place_of_collection(
    tmp_path: Path,
    field: str,
    invalid: str,
) -> None:
    presets = tmp_path / "presets"
    presets.mkdir()
    payload = {
        "schema_version": 1,
        "id": "malformed",
        "status": "supported",
        "targets": ["local"],
        "tensorrt": {"version": "11.2.0.113"},
        "cuda": {"version": "12.8"},
        "python_versions": ["3.12"],
        "architectures": {"aarch64": {}},
    }
    payload[field] = invalid
    (presets / "malformed.json").write_text(json.dumps(payload), encoding="utf-8")
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        providers=registry.freeze(),
        qualification_roots=(presets,),
    )

    with pytest.raises(DevToolkitError, match="Invalid qualification"):
        toolkit.resolve(
            EnvironmentRequest(
                tensorrt="11.2.0.113",
                target=ExecutionTarget("test-local"),
                architecture="aarch64",
                require_qualification=True,
            )
        )


def test_builtin_managed_source_accepts_arbitrary_trt_with_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    artifacts = (
        ArtifactPin(
            name="tensorrt-headers",
            uri="https://example.invalid/libnvinfer-headers.deb",
            sha256="d" * 64,
            verification="pinned-digest",
        ),
        ArtifactPin(
            name="tensorrt-wheel",
            uri="https://example.invalid/tensorrt-11.2.0.113.whl?token=super-secret",
            sha256="e" * 64,
            verification="pinned-digest",
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
        )
    )

    assert lock.toolchain.provider.name == "managed-artifacts"
    assert lock.toolchain.cuda == "13.3"
    assert lock.cuda_origin == "managed-default"
    assert lock.toolchain.artifacts == artifacts
    assert "super-secret" not in json.dumps(lock.as_dict())


def test_builtin_managed_source_materializes_and_attests_pinned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    headers = tmp_path / "libnvinfer-headers.deb"
    wheel = tmp_path / "tensorrt-11.2.0.113-py3-none-any.whl"
    headers.write_bytes(b"pinned headers")
    wheel.write_bytes(b"pinned wheel")

    def pin(name: str, path: Path) -> ArtifactPin:
        return ArtifactPin(
            name,
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "pinned-digest",
        )

    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=ManagedProvisionRunner(tmp_path),
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=(pin("tensorrt-headers", headers), pin("tensorrt-wheel", wheel)),
        )
    )

    environment = toolkit.provision(lock)

    assert environment.observation.cuda_version == "13.3"
    assert environment.observation.tensorrt_native_version == "11.2.0.113"
    assert environment.context.environment["CUDA_HOME"] == str(tmp_path / "managed-cuda")
    receipt = json.loads(environment.receipt.read_text(encoding="utf-8"))
    assert receipt["observed"]["tensorrt_header_version"] == "11.2.0.113"


def test_builtin_managed_source_rejects_an_incomplete_cuda_toolkit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    headers = tmp_path / "headers.deb"
    wheel = tmp_path / "tensorrt.whl"
    headers.write_bytes(b"headers")
    wheel.write_bytes(b"wheel")
    artifacts = tuple(
        ArtifactPin(
            name,
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "pinned-digest",
        )
        for name, path in (("tensorrt-headers", headers), ("tensorrt-wheel", wheel))
    )
    runner = ManagedProvisionRunner(tmp_path)
    (runner.cuda / "lib" / "libcurand.so").unlink()
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        runner=runner,
    )
    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
        )
    )

    with pytest.raises(DevToolkitError, match="incomplete CUDA"):
        toolkit.provision(lock)


def test_managed_tensorrt_uses_complete_system_cuda_before_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = tmp_path / "cuda"
    for directory in (cuda / "bin", cuda / "include", cuda / "lib64"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib64/libcudart.so",
        "lib64/libcublas.so",
        "lib64/libcurand.so",
    ):
        (cuda / relative).touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("TRTMC_TRT_INCLUDE_DIR", str(tmp_path / "missing-trt"))
    artifacts = (
        ArtifactPin(
            "tensorrt-headers",
            "https://example.invalid/headers.deb",
            "1" * 64,
            "pinned-digest",
        ),
        ArtifactPin(
            "tensorrt-wheel",
            "https://example.invalid/tensorrt.whl",
            "2" * 64,
            "pinned-digest",
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(),
            artifacts=artifacts,
        )
    )

    assert lock.toolchain.origin == "managed"
    assert lock.toolchain.cuda_source == "system"
    assert lock.cuda == "12.8"
    assert lock.cuda_origin == "system"


def test_environment_identity_depends_on_resolution_not_request_spelling(
    tmp_path: Path,
) -> None:
    registry = ProviderRegistry()
    registry.register_context(StaticLocalContext())
    registry.register_toolchain(ExistingToolchainSource())
    toolkit_a = DevToolkit.from_checkout(
        tmp_path,
        source_revision_override="a" * 40,
        providers=registry.freeze(),
    )
    toolkit_b = DevToolkit.from_checkout(
        tmp_path,
        source_revision_override="b" * 40,
        providers=registry.freeze(),
    )
    common = {
        "tensorrt": "11.2.0.113",
        "target": ExecutionTarget("test-local"),
        "architecture": "aarch64",
    }

    default_lock = toolkit_a.resolve(EnvironmentRequest(**common))
    explicit_lock = toolkit_b.resolve(EnvironmentRequest(**common, cuda=CudaPolicy.exact("12.8")))

    assert default_lock.lock_id == explicit_lock.lock_id
    assert default_lock.cuda_origin == "system"
    assert explicit_lock.cuda_origin == "explicit"


def test_builtin_prefix_provider_uses_user_owned_toolchain_root(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "toolchain"
    for directory in (prefix / "bin", prefix / "include", prefix / "lib"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib/libcudart.so",
        "lib/libcublas.so",
        "lib/libcurand.so",
        "lib/libnvinfer.so",
    ):
        (prefix / relative).touch()
    major, minor, patch, build = "11.2.0.113".split(".")
    (prefix / "include" / "NvInferVersion.h").write_text(
        "\n".join(
            (
                f"#define NV_TENSORRT_MAJOR {major}",
                f"#define NV_TENSORRT_MINOR {minor}",
                f"#define NV_TENSORRT_PATCH {patch}",
                f"#define NV_TENSORRT_BUILD {build}",
            )
        ),
        encoding="utf-8",
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(prefix=str(prefix)),
        )
    )

    assert lock.toolchain.provider.name == "prefix"
    assert lock.toolchain.origin == "prefix"
    assert lock.toolchain.identity["cuda_root"] == str(prefix)


def test_managed_tensorrt_follows_cuda_from_explicit_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "cuda"
    for directory in (prefix / "bin", prefix / "include", prefix / "lib"):
        directory.mkdir(parents=True)
    for relative in (
        "bin/nvcc",
        "include/cuda.h",
        "lib/libcudart.so",
        "lib/libcublas.so",
        "lib/libcurand.so",
    ):
        (prefix / relative).touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "missing-cuda"))
    artifacts = (
        ArtifactPin(
            "tensorrt-headers",
            "https://example.invalid/headers.deb",
            "3" * 64,
            "pinned-digest",
        ),
        ArtifactPin(
            "tensorrt-wheel",
            "https://example.invalid/tensorrt.whl",
            "4" * 64,
            "pinned-digest",
        ),
    )
    toolkit = DevToolkit.from_checkout(tmp_path, runner=BuiltinProbeRunner())

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            target=ExecutionTarget.local(prefix=str(prefix)),
            artifacts=artifacts,
        )
    )

    assert lock.toolchain.provider.name == "managed-artifacts"
    assert lock.toolchain.cuda_source == "prefix"
    assert lock.toolchain.cuda == "12.8"
    assert lock.cuda_origin == "system"

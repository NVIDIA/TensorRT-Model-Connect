# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for independently composable execution-target lifecycle APIs."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLKIT_ROOT = REPO_ROOT / "scripts" / "devToolkit"
sys.path.insert(0, str(DEVTOOLKIT_ROOT))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DevToolkitError,
    DockerGpuRequest,
    DockerImageBuild,
    DockerImageRef,
    DockerMount,
    DockerTarget,
    DockerTargetPolicy,
    EnvironmentRequest,
    ExecutionTarget,
    PullPolicy,
    TargetPlan,
    ToolchainRuntime,
    builtin_provider_registry,
)
from trtmc_devtoolkit.spi import (  # noqa: E402
    ProviderDescriptor,
    ProviderRegistry,
    TargetHandle,
    TargetProvider,
    ToolchainCandidate,
)


def _environment_dict(values: list[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in values)


class DockerLifecycleRunner:
    def __init__(
        self,
        *,
        image_exists: bool = False,
        container: dict | None = None,
        inspect_failure: bool = False,
    ) -> None:
        self.commands: list[list[str]] = []
        self.environments: list[dict[str, str] | None] = []
        self.image_exists = image_exists
        self.container = container
        self.image_id = "sha256:image-123"
        self.image_reference = "registry.example/cuda:13.3"
        self.daemon_id = "daemon-123"
        self.image_labels: dict[str, str] = {}
        self.image_environment = ["TOKEN=super-secret-value"]
        self.image_volumes: dict[str, dict[str, object]] = {}
        self.inspect_failure = inspect_failure

    def _image(self) -> dict[str, object]:
        return {
            "Id": self.image_id,
            "RepoDigests": ["registry.example/cuda@sha256:" + "a" * 64],
            "Config": {
                "Env": self.image_environment,
                "Labels": self.image_labels,
                "Volumes": self.image_volumes,
            },
        }

    def _default_container(self, arguments: list[str]) -> dict[str, object]:
        name = arguments[arguments.index("--name") + 1]
        labels: dict[str, str] = {}
        mounts: list[dict[str, object]] = []
        device_requests: list[dict[str, object]] = []
        index = 0
        while index < len(arguments):
            if arguments[index] == "--label":
                key, value = arguments[index + 1].split("=", 1)
                labels[key] = value
                index += 2
                continue
            if arguments[index] == "--mount":
                values = dict(
                    item.split("=", 1) for item in arguments[index + 1].split(",") if "=" in item
                )
                mounts.append(
                    {
                        "Type": "bind",
                        "Source": values["source"],
                        "Destination": values["target"],
                        "RW": "readonly" not in arguments[index + 1],
                    }
                )
                index += 2
                continue
            if arguments[index] == "--gpus":
                value = arguments[index + 1]
                device_requests.append(
                    {
                        "Driver": "nvidia",
                        "DeviceIDs": value.removeprefix("device=").split(","),
                        "Count": 0,
                        "Capabilities": [["gpu"]],
                    }
                )
                index += 2
                continue
            index += 1
        workdir = arguments[arguments.index("--workdir") + 1]
        image_index = arguments.index(self.image_id)
        environment = _environment_dict(self.image_environment)
        if "--env-file" in arguments:
            environment_file = Path(arguments[arguments.index("--env-file") + 1])
            environment.update(
                _environment_dict(environment_file.read_text(encoding="utf-8").splitlines())
            )
        mounted_targets = {str(mount["Destination"]) for mount in mounts}
        for target in self.image_volumes.keys() - mounted_targets:
            mounts.append(
                {
                    "Type": "volume",
                    "Source": "fixture-volume-" + target.strip("/").replace("/", "-"),
                    "Destination": target,
                    "RW": True,
                }
            )
        return {
            "Id": "container-123",
            "Image": self.image_id,
            "Name": "/" + name,
            "State": {"Running": False, "Status": "created"},
            "Config": {
                "Image": self.image_id,
                "Cmd": arguments[image_index + 1 :],
                "WorkingDir": workdir,
                "Labels": labels,
                "Env": [f"{name}={value}" for name, value in environment.items()],
            },
            "HostConfig": {
                "DeviceRequests": device_requests,
                "ShmSize": 64 * 1024**3 if "--shm-size" in arguments else None,
            },
            "Mounts": mounts,
        }

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
        del cwd, capture_output, timeout
        arguments = [str(item) for item in command]
        self.commands.append(arguments)
        self.environments.append(dict(env) if env is not None else None)
        output = ""
        error_output = ""
        returncode = 0
        if arguments == ["docker", "context", "show"]:
            output = "test-context\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "version"]:
            output = "28.0.0\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "info"]:
            output = self.daemon_id + "\n"
        elif arguments[:6] == [
            "docker",
            "--context",
            "test-context",
            "inspect",
            "--type",
            "container",
        ]:
            if self.inspect_failure:
                returncode = 1
                error_output = "permission denied"
            elif self.container is None:
                returncode = 1
                error_output = "No such container"
            else:
                output = json.dumps(self.container)
        elif arguments[:5] == [
            "docker",
            "--context",
            "test-context",
            "image",
            "inspect",
        ]:
            if not self.image_exists:
                returncode = 1
                error_output = "No such image"
            else:
                output = json.dumps(self._image())
        elif arguments[:4] == ["docker", "--context", "test-context", "pull"]:
            self.image_exists = True
        elif arguments[:4] == ["docker", "--context", "test-context", "build"]:
            self.image_exists = True
            label = arguments[arguments.index("--label") + 1]
            key, value = label.split("=", 1)
            self.image_labels[key] = value
        elif arguments[:4] == ["docker", "--context", "test-context", "create"]:
            self.container = self._default_container(arguments)
            output = "container-123\n"
        elif arguments[:4] == ["docker", "--context", "test-context", "start"]:
            assert self.container is not None
            self.container["State"] = {"Running": True, "Status": "running"}
        elif arguments[:4] == ["docker", "--context", "test-context", "exec"]:
            if arguments[-2:] == ["cat", "/etc/os-release"]:
                output = 'ID="ubuntu"\nVERSION_ID="24.04"\n'
            else:
                raise AssertionError(arguments)
        else:
            raise AssertionError(arguments)
        result = subprocess.CompletedProcess(arguments, returncode, output, error_output)
        if check and returncode:
            raise DevToolkitError(f"simulated Docker failure: {arguments}")
        return result


def target(tmp_path: Path, *, image=None, environment=None) -> DockerTarget:
    workspace = tmp_path / "workspace"
    runs = tmp_path / "runs"
    workspace.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    return DockerTarget(
        name="trtmc-dev-gb300-smoke",
        image=image or DockerImageRef("registry.example/cuda:13.3"),
        gpus=DockerGpuRequest.devices("1"),
        mounts=(
            DockerMount(workspace, PurePosixPath("/workspace/trtmc")),
            DockerMount(runs, PurePosixPath("/runs")),
        ),
        workspace=PurePosixPath("/workspace/trtmc"),
        state=PurePosixPath("/state/devtoolkit"),
        command=("sleep", "infinity"),
        environment=environment or {},
    )


def test_target_provider_protocol_is_exported() -> None:
    assert TargetProvider.__name__ == "TargetProvider"


def test_custom_target_provider_composes_without_docker(tmp_path: Path) -> None:
    class CustomTargetProvider:
        descriptor = ProviderDescriptor("custom-target", "custom-target==1", 1)

        def __init__(self) -> None:
            self.attested = False
            self.policy = None

        def resolve(self, request, *, repository, runner):
            del repository, runner
            return TargetPlan(self.descriptor, "a" * 64, {"name": request.name}, request)

        def provision(self, plan, *, policy, repository, state_dir, runner):
            del repository, state_dir, runner
            self.policy = policy
            return TargetHandle(
                provider=self.descriptor,
                plan_id=plan.plan_id,
                target_id="b" * 64,
                action="created",
                policy="custom-policy",
                identity={"resource": "custom-id"},
                observation={"ready": True},
                execution_target=ExecutionTarget.local(),
                request=plan.request,
            )

        def attest(self, target, *, repository, runner):
            del target, repository, runner
            self.attested = True

    provider = CustomTargetProvider()
    registry = ProviderRegistry()
    registry.register_target(provider)
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
    )

    custom_policy = object()
    provisioned = toolkit.targets.ensure(
        SimpleNamespace(provider="custom-target", name="custom-resource"),
        policy=custom_policy,
    )

    receipt = json.loads(provisioned.receipt.read_text(encoding="utf-8"))
    assert provider.attested
    assert provider.policy is custom_policy
    assert provisioned.execution_target.provider == "local"
    assert receipt["policy"] == "custom-policy"
    assert receipt["attested"] is True


def test_target_resolution_is_read_only_when_resources_are_missing(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    plan = toolkit.targets.resolve(target(tmp_path))

    assert plan.provider.name == "docker"
    assert len(plan.plan_id) == 64
    assert not any(
        command[3] in {"pull", "build", "create", "start"}
        for command in runner.commands
        if len(command) > 3
    )


def test_inspect_failure_does_not_trigger_mutation(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner(inspect_failure=True)
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    with pytest.raises(DevToolkitError, match="Could not inspect Docker container"):
        toolkit.targets.ensure(target(tmp_path))

    assert not any(
        command[3] in {"pull", "build", "create", "start"}
        for command in runner.commands
        if len(command) > 3
    )


def test_target_plan_intent_is_deeply_immutable(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    plan = toolkit.targets.resolve(target(tmp_path))

    container = plan.intent["container"]
    with pytest.raises(TypeError):
        container["working_dir"] = "/changed"  # type: ignore[index]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_target_plan_rejects_non_finite_identity_numbers(value: float) -> None:
    with pytest.raises(DevToolkitError, match="must be finite"):
        TargetPlan(
            ProviderDescriptor("custom-target", "custom-target==1", 1),
            "a" * 64,
            {"value": value},
            object(),
        )


def test_ensure_pulls_creates_starts_and_returns_execution_target(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    provisioned = toolkit.targets.ensure(target(tmp_path), policy=DockerTargetPolicy.ENSURE)

    operations = [command[3] for command in runner.commands if len(command) > 3]
    assert "pull" in operations
    assert "create" in operations
    assert "start" in operations
    assert provisioned.action == "created"
    assert provisioned.execution_target.provider == "docker"
    assert provisioned.execution_target.options["container"] == "container-123"
    assert provisioned.execution_target.options["workspace"] == "/workspace/trtmc"
    assert provisioned.receipt.is_file()


def test_provisioned_docker_target_feeds_environment_resolution(tmp_path: Path) -> None:
    class PinnedToolchain:
        descriptor = ProviderDescriptor("pinned-test-toolchain", "pinned-test-toolchain==1", 1)

        def resolve(self, request, context, *, repository, runner):
            del context, repository, runner
            return (
                ToolchainCandidate(
                    provider=self.descriptor,
                    origin="image",
                    cuda_source="image",
                    tensorrt=request.tensorrt,
                    cuda="13.3",
                    python=request.python,
                    identity={"fixture": "pinned"},
                    runtime=ToolchainRuntime(
                        python_executable="python3",
                        cuda_root="/usr/local/cuda",
                        nvcc="/usr/local/cuda/bin/nvcc",
                        tensorrt_include_dir="/usr/include",
                        tensorrt_library="/usr/lib/libnvinfer.so",
                    ),
                ),
            )

    runner = DockerLifecycleRunner()
    registry = builtin_provider_registry()
    registry.register_toolchain(PinnedToolchain())
    toolkit = DevToolkit.from_checkout(
        tmp_path,
        state_root=tmp_path / "state",
        providers=registry.freeze(),
        runner=runner,
    )
    provisioned = toolkit.targets.ensure(target(tmp_path))

    lock = toolkit.resolve(
        EnvironmentRequest(
            tensorrt="11.2.0.113",
            architecture="x86_64",
            target=provisioned.execution_target,
            toolchain="pinned-test-toolchain",
        )
    )

    assert lock.context.identity["container_id"] == provisioned.identity["container_id"]
    assert lock.toolchain.tensorrt == "11.2.0.113"


def test_repeated_ensure_is_idempotent(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)

    first = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)
    second = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)

    assert first.target_id == second.target_id
    assert second.action == "adopted"
    assert sum(command[3] == "create" for command in runner.commands if len(command) > 3) == 1


def test_mount_order_changes_do_not_fail_target_attestation(tmp_path: Path) -> None:
    class ReorderingMountRunner(DockerLifecycleRunner):
        def run(self, command, **kwargs):
            arguments = [str(item) for item in command]
            if (
                arguments[:6]
                == [
                    "docker",
                    "--context",
                    "test-context",
                    "inspect",
                    "--type",
                    "container",
                ]
                and self.container is not None
            ):
                self.container["Mounts"].reverse()
            return super().run(command, **kwargs)

    runner = ReorderingMountRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)

    first = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)
    second = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)

    assert first.target_id == second.target_id
    assert second.action == "adopted"


def test_undeclared_environment_is_rejected(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)
    toolkit.targets.ensure(request)
    assert runner.container is not None
    runner.container["Config"]["Env"].append("UNDECLARED=value")  # type: ignore[index]

    with pytest.raises(DevToolkitError, match="environment:UNDECLARED"):
        toolkit.targets.ensure(request)


def test_null_inspected_environment_reports_drift(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    provisioned = toolkit.targets.ensure(target(tmp_path))
    assert runner.container is not None
    runner.container["Config"]["Env"] = None  # type: ignore[index]

    with pytest.raises(DevToolkitError, match="configuration changed"):
        toolkit.targets.attest(provisioned)


def test_requested_environment_may_override_image_default(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    runner.image_environment.append("FEATURE=image-default")
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path, environment={"FEATURE": "requested"})

    first = toolkit.targets.ensure(request)
    second = toolkit.targets.ensure(request)

    assert first.target_id == second.target_id


def test_undeclared_mount_is_rejected(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)
    toolkit.targets.ensure(request)
    assert runner.container is not None
    runner.container["Mounts"].append(  # type: ignore[index]
        {
            "Type": "bind",
            "Source": "/undeclared",
            "Destination": "/undeclared",
            "RW": True,
        }
    )

    with pytest.raises(DevToolkitError, match="mount:/undeclared"):
        toolkit.targets.ensure(request)


@pytest.mark.parametrize("volume_target", ["/image-cache", "/runs"])
def test_image_declared_volume_is_allowed_or_overridden(
    tmp_path: Path,
    volume_target: str,
) -> None:
    runner = DockerLifecycleRunner()
    runner.image_volumes[volume_target] = {}
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)

    first = toolkit.targets.ensure(request)
    second = toolkit.targets.ensure(request)

    assert first.target_id == second.target_id


def test_adopt_never_pulls_even_when_image_policy_is_always(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(
        tmp_path,
        image=DockerImageRef(
            "registry.example/cuda:13.3",
            pull=PullPolicy.ALWAYS,
        ),
    )
    toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)
    runner.commands.clear()

    provisioned = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ADOPT)

    assert provisioned.action == "adopted"
    assert not any(command[3] == "pull" for command in runner.commands if len(command) > 3)


def test_create_rejects_foreign_name_collision_before_image_mutation(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    toolkit.targets.ensure(target(tmp_path))
    runner.commands.clear()
    different_plan = target(
        tmp_path,
        image=DockerImageRef(
            "registry.example/cuda:13.3",
            pull=PullPolicy.ALWAYS,
        ),
    )

    with pytest.raises(DevToolkitError, match="not owned by this plan"):
        toolkit.targets.ensure(different_plan, policy=DockerTargetPolicy.CREATE)

    assert not any(command[3] == "pull" for command in runner.commands if len(command) > 3)


def test_stopped_matching_container_is_started(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)
    first = toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)
    assert runner.container is not None
    runner.container["State"] = {"Running": False, "Status": "exited"}

    second = toolkit.targets.ensure(request, policy=DockerTargetPolicy.START)

    assert first.target_id == second.target_id
    assert second.action == "started"


def test_mismatched_existing_container_fails_without_replacement(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path)
    toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)
    assert runner.container is not None
    runner.container["Config"]["WorkingDir"] = "/unexpected"  # type: ignore[index]

    with pytest.raises(DevToolkitError, match="configuration does not match"):
        toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)

    assert not any("rm" in command for command in runner.commands)


def test_dockerfile_image_is_built_before_container_creation(tmp_path: Path) -> None:
    context = tmp_path / "image"
    context.mkdir()
    (context / "Dockerfile.dev").write_text("FROM scratch\n", encoding="utf-8")
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(
        tmp_path,
        image=DockerImageBuild(context, dockerfile=Path("Dockerfile.dev")),
    )

    toolkit.targets.ensure(request, policy=DockerTargetPolicy.CREATE)

    operations = [command[3] for command in runner.commands if len(command) > 3]
    assert operations.index("build") < operations.index("create")


def test_repeated_ensure_reuses_a_plan_owned_dockerfile_image(tmp_path: Path) -> None:
    context = tmp_path / "image"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(tmp_path, image=DockerImageBuild(context))

    toolkit.targets.ensure(request)
    toolkit.targets.ensure(request)

    assert sum(command[3] == "build" for command in runner.commands if len(command) > 3) == 1


def test_environment_values_are_not_written_to_receipt_or_command_log(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    toolkit.targets.ensure(
        target(tmp_path, environment={"TOKEN": "super-secret-value"}),
        policy=DockerTargetPolicy.ENSURE,
    )

    evidence = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "state" / "targets").rglob("*.json")
    )
    assert "super-secret-value" not in evidence
    assert all(
        "super-secret-value" not in argument for command in runner.commands for argument in command
    )


def test_build_argument_values_are_not_written_to_evidence_or_argv(tmp_path: Path) -> None:
    context = tmp_path / "image"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)

    toolkit.targets.ensure(
        target(
            tmp_path,
            image=DockerImageBuild(
                context,
                build_args={"REGISTRY_TOKEN": "build-argument-secret"},
            ),
        )
    )

    evidence = "".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "state" / "targets").rglob("*.json")
    )
    assert "build-argument-secret" not in evidence
    assert all(
        "build-argument-secret" not in argument
        for command in runner.commands
        for argument in command
    )
    assert any(
        environment is not None and environment.get("REGISTRY_TOKEN") == "build-argument-secret"
        for environment in runner.environments
    )


def test_remote_docker_bind_source_need_not_exist_on_the_client(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = DockerTarget(
        name="remote-target",
        image=DockerImageRef("registry.example/cuda:13.3"),
        docker_context="test-context",
        mounts=(
            DockerMount(
                Path("/path/owned/by/remote/docker/daemon"),
                PurePosixPath("/workspace/trtmc"),
            ),
        ),
        workspace=PurePosixPath("/workspace/trtmc"),
    )

    plan = toolkit.targets.resolve(request)

    assert plan.intent["docker_context"] == "test-context"


@pytest.mark.parametrize("reference", ["", " image", "image\nname", "--all-tags"])
def test_image_reference_rejects_unsafe_values(reference: str) -> None:
    with pytest.raises(DevToolkitError, match="image reference"):
        DockerImageRef(reference)


def test_pull_never_requires_a_local_image(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(tmp_path, state_root=tmp_path / "state", runner=runner)
    request = target(
        tmp_path,
        image=DockerImageRef(
            "registry.example/cuda:13.3",
            pull=PullPolicy.NEVER,
        ),
    )

    with pytest.raises(DevToolkitError, match="not available locally"):
        toolkit.targets.ensure(request, policy=DockerTargetPolicy.ENSURE)

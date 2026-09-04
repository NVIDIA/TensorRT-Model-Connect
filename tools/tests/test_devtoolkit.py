# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the checkout-only TRTMC DevToolkit."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "apps/devtoolkit"))

from trtmc_devtoolkit import (  # noqa: E402
    DevToolkit,
    DevToolkitError,
    DockerMount,
    DockerTargetPolicy,
    PreparedEnvironment,
)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str] | None]] = []

    @property
    def commands(self) -> list[list[str]]:
        return [call[0] for call in self.calls]

    def run(
        self,
        command,
        *,
        cwd: Path,
        env=None,
        check=True,
        capture_output=False,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output
        arguments = [str(argument) for argument in command]
        self.calls.append((arguments, cwd, None if env is None else dict(env)))
        return subprocess.CompletedProcess(arguments, 0, "", "")


def _environment(values: list[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in values)


class DockerLifecycleRunner(RecordingRunner):
    """Small stateful Docker boundary; no host Docker daemon is used."""

    def __init__(self, *, inspect_failure: bool = False) -> None:
        super().__init__()
        self.inspect_failure = inspect_failure
        self.image_exists = False
        self.image_id = "sha256:image-123"
        self.image_environment = ["PATH=/usr/bin", "BASE=image-default"]
        self.image_volumes: dict[str, dict[str, object]] = {}
        self.container: dict[str, object] | None = None
        self.environment_files: list[tuple[Path, int, str]] = []

    def _image(self) -> dict[str, object]:
        return {
            "Id": self.image_id,
            "Config": {
                "Env": list(self.image_environment),
                "Volumes": dict(self.image_volumes),
            },
        }

    def _created_container(self, arguments: list[str]) -> dict[str, object]:
        labels: dict[str, str] = {}
        mounts: list[dict[str, object]] = []
        device_requests: list[dict[str, object]] = []
        environment = _environment(self.image_environment)
        image_index = arguments.index(self.image_id)
        index = 2
        while index < image_index:
            option = arguments[index]
            value = arguments[index + 1]
            if option == "--label":
                name, label_value = value.split("=", 1)
                labels[name] = label_value
            elif option == "--mount":
                components = value.split(",")
                fields = dict(
                    component.split("=", 1) for component in components if "=" in component
                )
                mounts.append(
                    {
                        "Type": "bind",
                        "Source": fields["source"],
                        "Destination": fields["target"],
                        "RW": "readonly" not in components,
                    }
                )
            elif option == "--gpus":
                if value == "all":
                    device_requests.append({"Driver": "nvidia", "Count": -1, "DeviceIDs": None})
                else:
                    device_requests.append(
                        {
                            "Driver": "nvidia",
                            "Count": 0,
                            "DeviceIDs": value.removeprefix("device=").split(","),
                        }
                    )
            elif option == "--env-file":
                path = Path(value)
                content = path.read_text(encoding="utf-8")
                self.environment_files.append((path, stat.S_IMODE(path.stat().st_mode), content))
                environment.update(_environment(content.splitlines()))
            index += 2
        ipc = arguments[arguments.index("--ipc") + 1] if "--ipc" in arguments[:image_index] else ""
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
            "State": {"Running": False},
            "Config": {
                "Image": self.image_id,
                "Cmd": arguments[image_index + 1 :],
                "WorkingDir": arguments[arguments.index("--workdir") + 1],
                "Labels": labels,
                "Env": [f"{name}={value}" for name, value in environment.items()],
            },
            "HostConfig": {
                "DeviceRequests": device_requests,
                "IpcMode": ipc,
            },
            "Mounts": mounts,
        }

    def run(
        self,
        command,
        *,
        cwd: Path,
        env=None,
        check=True,
        capture_output=False,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output
        arguments = [str(argument) for argument in command]
        self.calls.append((arguments, cwd, None if env is None else dict(env)))
        stdout = ""
        stderr = ""
        returncode = 0
        if arguments == [sys.executable, "-m", "tools.ci", "image", "ensure"]:
            self.image_exists = True
        elif arguments[:3] == ["docker", "container", "inspect"]:
            identifier = arguments[3]
            if self.inspect_failure:
                returncode = 1
                stderr = "permission denied"
            elif self.container is None or identifier not in {
                "trtmc-dev",
                "trtmc-test",
                str(self.container.get("Id")),
            }:
                returncode = 1
                stderr = f"No such container: {identifier}"
            else:
                stdout = json.dumps([self.container])
        elif arguments[:3] == ["docker", "image", "inspect"]:
            if not self.image_exists:
                returncode = 1
                stderr = f"No such image: {arguments[3]}"
            else:
                stdout = json.dumps([self._image()])
        elif arguments[:2] == ["docker", "create"]:
            if self.container is not None:
                raise AssertionError("test fixture refuses container replacement")
            self.container = self._created_container(arguments)
            stdout = "container-123\n"
        elif arguments[:2] == ["docker", "start"]:
            assert self.container is not None
            self.container["State"] = {"Running": True}
        elif arguments[:2] == ["docker", "exec"]:
            pass
        else:
            raise AssertionError(arguments)
        result = subprocess.CompletedProcess(arguments, returncode, stdout, stderr)
        if check and returncode:
            raise subprocess.CalledProcessError(
                returncode,
                arguments,
                output=stdout,
                stderr=stderr,
            )
        return result


def _checkout(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text(
        "FROM example/base@sha256:" + "a" * 64 + "\n", encoding="utf-8"
    )
    (tmp_path / "tools/ci").mkdir(parents=True)
    (tmp_path / "tools/ci/__main__.py").write_text("", encoding="utf-8")
    alpha = tmp_path / "families/alpha"
    alpha.mkdir(parents=True)
    (alpha / "model.py").write_text("def build(request, writer): pass\n", encoding="utf-8")
    (alpha / "requirements.txt").write_text("alpha-package\n", encoding="utf-8")
    beta = tmp_path / "families/beta"
    beta.mkdir()
    (beta / "model.py").write_text("def build(request, writer): pass\n", encoding="utf-8")
    return tmp_path


def _mutation_commands(runner: RecordingRunner) -> list[list[str]]:
    return [
        command
        for command in runner.commands
        if command == [sys.executable, "-m", "tools.ci", "image", "ensure"]
        or command[:2] in (["docker", "create"], ["docker", "start"])
    ]


def test_docker_uses_source_ci_and_only_the_selected_family_dependency(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    runner = DockerLifecycleRunner()

    result = DevToolkit.from_checkout(checkout, runner=runner).prepare_docker(
        family="alpha",
        gpu="0",
        image="trtmc:test",
        container="trtmc-test",
    )

    image_call = runner.calls[
        runner.commands.index([sys.executable, "-m", "tools.ci", "image", "ensure"])
    ]
    assert image_call[2] is not None
    assert image_call[2]["TRTMC_CI_WORKSPACE"] == str(checkout)
    assert image_call[2]["TRTMC_CI_IMAGE"] == "trtmc:test"
    assert runner.commands[-1] == [
        "docker",
        "exec",
        "-w",
        str(checkout),
        "container-123",
        "python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        "families/alpha/requirements.txt",
    ]
    assert result == PreparedEnvironment(
        kind="docker",
        repository=checkout,
        python="python3",
        family="alpha",
        container="trtmc-test",
        container_id="container-123",
        image_id="sha256:image-123",
    )
    assert result.command("bash") == (
        "docker",
        "exec",
        "-w",
        str(checkout),
        "container-123",
        "bash",
    )


def test_family_without_dependencies_does_not_add_an_install_step(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()

    DevToolkit.from_checkout(_checkout(tmp_path), runner=runner).prepare_docker(family="beta")

    assert not any(command[:2] == ["docker", "exec"] for command in runner.commands)


def test_repeated_ensure_is_idempotent(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)

    first = toolkit.prepare_docker(gpu="0")
    second = toolkit.prepare_docker(gpu="0")

    assert first.container_id == second.container_id == "container-123"
    assert (
        sum(
            command == [sys.executable, "-m", "tools.ci", "image", "ensure"]
            for command in runner.commands
        )
        == 2
    )
    assert sum(command[:2] == ["docker", "create"] for command in runner.commands) == 1
    assert sum(command[:2] == ["docker", "start"] for command in runner.commands) == 1


@pytest.mark.parametrize(
    "policy",
    (DockerTargetPolicy.ENSURE, DockerTargetPolicy.CREATE),
)
def test_foreign_name_collision_fails_before_any_mutation(
    tmp_path: Path,
    policy: DockerTargetPolicy,
) -> None:
    runner = DockerLifecycleRunner()
    runner.image_exists = True
    runner.container = {
        "Id": "foreign-container",
        "Image": runner.image_id,
        "State": {"Running": True},
        "Config": {"Labels": {}},
    }
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)

    with pytest.raises(DevToolkitError, match="not owned by this checkout"):
        toolkit.prepare_docker(policy=policy)

    assert _mutation_commands(runner) == []
    assert not any("rm" in command for command in runner.commands)


def test_inspection_failure_fails_before_any_mutation(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner(inspect_failure=True)
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)

    with pytest.raises(DevToolkitError, match="Could not inspect Docker container"):
        toolkit.prepare_docker()

    assert _mutation_commands(runner) == []


def test_start_starts_matching_stopped_container_without_rebuilding(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="0")
    assert runner.container is not None
    runner.container["State"] = {"Running": False}
    runner.calls.clear()

    result = toolkit.prepare_docker(gpu="0", policy=DockerTargetPolicy.START)

    assert result.container_id == "container-123"
    assert _mutation_commands(runner) == [["docker", "start", "container-123"]]


def test_adopt_rejects_a_stopped_container_without_mutation(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="0")
    assert runner.container is not None
    runner.container["State"] = {"Running": False}
    runner.calls.clear()

    with pytest.raises(DevToolkitError, match="is not running"):
        toolkit.prepare_docker(gpu="0", policy=DockerTargetPolicy.ADOPT)

    assert _mutation_commands(runner) == []


def test_adopt_does_not_install_family_dependencies(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(family="alpha", gpu="0")
    runner.calls.clear()

    toolkit.prepare_docker(
        family="alpha",
        gpu="0",
        policy=DockerTargetPolicy.ADOPT,
    )

    assert _mutation_commands(runner) == []
    assert not any(command[:2] == ["docker", "exec"] for command in runner.commands)


def test_malformed_running_state_fails_without_starting(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="0")
    assert runner.container is not None
    runner.container["State"] = {}
    runner.calls.clear()

    with pytest.raises(DevToolkitError, match="invalid container state"):
        toolkit.prepare_docker(gpu="0", policy=DockerTargetPolicy.START)

    assert _mutation_commands(runner) == []


def test_created_container_is_attested_and_started_by_immutable_id(tmp_path: Path) -> None:
    class NameSwapRunner(DockerLifecycleRunner):
        def __init__(self) -> None:
            super().__init__()
            self.created: dict[str, object] | None = None

        def run(self, command, *, cwd, env=None, check=True, capture_output=False):
            arguments = [str(argument) for argument in command]
            if arguments[:2] == ["docker", "create"]:
                result = super().run(
                    arguments,
                    cwd=cwd,
                    env=env,
                    check=check,
                    capture_output=capture_output,
                )
                self.created = self.container
                self.container = {
                    "Id": "foreign-container",
                    "Image": self.image_id,
                    "State": {"Running": False},
                    "Config": {"Labels": {}},
                }
                return result
            if (
                arguments[:3] == ["docker", "container", "inspect"]
                and arguments[3] == "container-123"
                and self.created is not None
            ):
                by_name = self.container
                self.container = self.created
                try:
                    return super().run(
                        arguments,
                        cwd=cwd,
                        env=env,
                        check=check,
                        capture_output=capture_output,
                    )
                finally:
                    self.container = by_name
            if arguments == ["docker", "start", "container-123"]:
                assert self.created is not None
                self.calls.append((arguments, cwd, None if env is None else dict(env)))
                self.created["State"] = {"Running": True}
                return subprocess.CompletedProcess(arguments, 0, "", "")
            return super().run(
                arguments,
                cwd=cwd,
                env=env,
                check=check,
                capture_output=capture_output,
            )

    runner = NameSwapRunner()
    result = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner).prepare_docker(gpu="none")

    assert result.container_id == "container-123"
    assert runner.created is not None
    assert runner.created["State"] == {"Running": True}
    assert runner.container is not None
    assert runner.container["State"] == {"Running": False}


@pytest.mark.parametrize(
    ("drift", "evidence"),
    (
        ("command", "command"),
        ("environment", "environment:UNDECLARED"),
        ("mount", "mount:/foreign"),
        ("gpus", "gpus"),
    ),
)
def test_configuration_drift_fails_without_replacement(
    tmp_path: Path,
    drift: str,
    evidence: str,
) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="0")
    assert runner.container is not None
    if drift == "command":
        runner.container["Config"]["Cmd"] = ["tail", "-f", "/dev/null"]  # type: ignore[index]
    elif drift == "environment":
        runner.container["Config"]["Env"].append("UNDECLARED=value")  # type: ignore[index]
    elif drift == "mount":
        runner.container["Mounts"].append(  # type: ignore[union-attr]
            {
                "Type": "bind",
                "Source": "/foreign",
                "Destination": "/foreign",
                "RW": True,
            }
        )
    else:
        runner.container["HostConfig"]["DeviceRequests"] = []  # type: ignore[index]
    runner.calls.clear()

    with pytest.raises(DevToolkitError, match=evidence):
        toolkit.prepare_docker(gpu="0", policy=DockerTargetPolicy.ENSURE)

    assert _mutation_commands(runner) == []
    assert not any("rm" in command for command in runner.commands)


def test_mount_order_is_not_configuration_drift(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    mount = DockerMount(cache, PurePosixPath("/cache"), read_only=True)
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(checkout, runner=runner)
    toolkit.prepare_docker(gpu="none", mounts=[mount])
    assert runner.container is not None
    runner.container["Mounts"].reverse()  # type: ignore[union-attr]
    runner.calls.clear()

    result = toolkit.prepare_docker(
        gpu="none",
        mounts=[mount],
        policy=DockerTargetPolicy.ADOPT,
    )

    assert result.container_id == "container-123"
    assert _mutation_commands(runner) == []


@pytest.mark.parametrize("volume_target", ("/image-cache", "checkout"))
def test_image_declared_volume_is_allowed_or_overridden(
    tmp_path: Path,
    volume_target: str,
) -> None:
    checkout = _checkout(tmp_path)
    runner = DockerLifecycleRunner()
    target = str(checkout) if volume_target == "checkout" else volume_target
    runner.image_volumes[target] = {}
    toolkit = DevToolkit.from_checkout(checkout, runner=runner)

    first = toolkit.prepare_docker(gpu="none")
    second = toolkit.prepare_docker(gpu="none")

    assert first.container_id == second.container_id == "container-123"


def test_requested_environment_overrides_an_image_default(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)

    first = toolkit.prepare_docker(
        gpu="none",
        environment={"BASE": "requested"},
    )
    second = toolkit.prepare_docker(
        gpu="none",
        environment={"BASE": "requested"},
    )

    assert first.container_id == second.container_id == "container-123"


def test_missing_inspected_environment_is_configuration_drift(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="none")
    assert runner.container is not None
    runner.container["Config"]["Env"] = None  # type: ignore[index]
    runner.calls.clear()

    with pytest.raises(DevToolkitError, match="environment:BASE"):
        toolkit.prepare_docker(gpu="none", policy=DockerTargetPolicy.START)

    assert _mutation_commands(runner) == []


def test_requested_secrets_use_a_private_ephemeral_file_and_are_not_disclosed(
    tmp_path: Path,
) -> None:
    secret = "devtoolkit-secret-value-that-must-not-leak"
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(gpu="none", environment={"API_TOKEN": secret})

    assert len(runner.environment_files) == 1
    path, mode, content = runner.environment_files[0]
    assert mode == 0o600
    assert content == f"API_TOKEN={secret}\n"
    assert not path.exists()
    assert all(secret not in argument for command in runner.commands for argument in command)
    assert all(
        secret not in value
        for _, _, environment in runner.calls
        if environment is not None
        for value in environment.values()
    )

    assert runner.container is not None
    runner.container["Config"]["WorkingDir"] = "/foreign"  # type: ignore[index]
    with pytest.raises(DevToolkitError) as error:
        toolkit.prepare_docker(
            gpu="none",
            environment={"API_TOKEN": secret},
            policy=DockerTargetPolicy.START,
        )
    assert secret not in str(error.value)


def test_image_reference_is_part_of_checkout_ownership(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)
    toolkit.prepare_docker(image="trtmc:first", gpu="none")
    runner.calls.clear()

    with pytest.raises(DevToolkitError, match="not owned by this checkout"):
        toolkit.prepare_docker(
            image="trtmc:second",
            gpu="none",
            policy=DockerTargetPolicy.START,
        )

    assert _mutation_commands(runner) == []


def test_local_uses_only_the_explicit_existing_interpreter(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path)
    runner = RecordingRunner()

    result = DevToolkit.from_checkout(checkout, runner=runner).prepare_local(
        python=sys.executable,
        family="alpha",
    )

    assert runner.calls == [
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(checkout / "families/alpha/requirements.txt"),
            ],
            checkout,
            None,
        )
    ]
    assert result.kind == "local"
    assert result.python == sys.executable
    assert result.command("python", "-V") == ("python", "-V")


@pytest.mark.parametrize("family", ("missing", "../alpha", "Alpha"))
def test_unknown_or_invalid_family_fails_closed(tmp_path: Path, family: str) -> None:
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=RecordingRunner())
    with pytest.raises(DevToolkitError):
        toolkit.prepare_local(python=sys.executable, family=family)


def test_invalid_docker_inputs_fail_before_commands(tmp_path: Path) -> None:
    runner = DockerLifecycleRunner()
    toolkit = DevToolkit.from_checkout(_checkout(tmp_path), runner=runner)

    with pytest.raises(DevToolkitError):
        toolkit.prepare_docker(container="../foreign")
    with pytest.raises(DevToolkitError):
        toolkit.prepare_docker(gpu="0,0")
    with pytest.raises(DevToolkitError):
        toolkit.prepare_docker(environment={"TOKEN": "line-one\nline-two"})
    with pytest.raises(DevToolkitError):
        toolkit.prepare_docker(policy="ensure")  # type: ignore[arg-type]

    assert runner.calls == []


def test_devtoolkit_has_no_legacy_registry_or_cohort_modules() -> None:
    root = REPO / "apps/devtoolkit"
    legacy = REPO / "scripts/devToolkit"
    assert not legacy.exists() or not [path for path in legacy.rglob("*") if path.is_file()]
    assert not (root / "cohorts").exists()
    assert not [
        name
        for name in (
            "builtin_providers.py",
            "builtin_registry.py",
            "cohorts.py",
            "planner.py",
            "providers.py",
            "receipt.py",
            "spi.py",
            "target_contracts.py",
            "target_service.py",
            "toolchain.py",
        )
        if (root / "trtmc_devtoolkit" / name).exists()
    ]

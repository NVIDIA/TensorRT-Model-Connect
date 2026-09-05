# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed Docker lifecycle for one checkout-owned development target."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator

from .models import DevToolkitError
from .runner import CommandRunner


_CONTAINER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DEVICE = re.compile(r"[A-Za-z0-9._-]+\Z")
_ENVIRONMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MANAGED_LABEL = "org.nvidia.trtmc.devtoolkit.managed"
_REPOSITORY_LABEL = "org.nvidia.trtmc.devtoolkit.repository"
_IMAGE_LABEL = "org.nvidia.trtmc.devtoolkit.image"
_DEFAULT_COMMAND = ("sleep", "infinity")


class DockerTargetPolicy(str, Enum):
    """Allowed mutations for an explicitly named Docker target."""

    ADOPT = "adopt"
    START = "start"
    ENSURE = "ensure"


@dataclass(frozen=True)
class DockerMount:
    """One explicit bind mount added beside the checkout mount."""

    source: Path
    target: PurePosixPath
    read_only: bool = False

    def __post_init__(self) -> None:
        source = Path(self.source)
        target = PurePosixPath(self.target)
        if not source.is_absolute() or ".." in source.parts:
            raise DevToolkitError("Docker mount source must be an absolute safe path")
        if not target.is_absolute() or target == PurePosixPath("/") or ".." in target.parts:
            raise DevToolkitError("Docker mount target must be an absolute non-root safe path")
        if any(character in f"{source}{target}" for character in ",\r\n\0"):
            raise DevToolkitError("Docker mount paths cannot contain commas or line breaks")
        if not isinstance(self.read_only, bool):
            raise DevToolkitError("Docker mount read_only must be boolean")
        object.__setattr__(self, "source", source.resolve())
        object.__setattr__(self, "target", target)


@dataclass(frozen=True)
class DockerTargetRequest:
    """The complete concrete configuration of one checkout container."""

    repository: Path
    name: str
    image: str
    gpu: str = "all"
    environment: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[DockerMount, ...] = ()
    command: tuple[str, ...] = _DEFAULT_COMMAND
    ipc: str | None = None

    def __post_init__(self) -> None:
        repository = Path(self.repository).resolve()
        if not _CONTAINER.fullmatch(self.name):
            raise DevToolkitError(f"invalid container name: {self.name!r}")
        if (
            not isinstance(self.image, str)
            or not self.image
            or self.image != self.image.strip()
            or self.image.startswith("-")
            or any(character.isspace() or character == "\0" for character in self.image)
        ):
            raise DevToolkitError(f"invalid Docker image: {self.image!r}")
        _gpu_devices(self.gpu)
        values = dict(self.environment)
        for name, value in values.items():
            if _ENVIRONMENT.fullmatch(name) is None:
                raise DevToolkitError(f"invalid Docker environment name: {name!r}")
            if not isinstance(value, str) or any(character in value for character in "\r\n\0"):
                raise DevToolkitError(f"Docker environment value for {name!r} must be one line")
        mounts = tuple(self.mounts)
        if any(not isinstance(mount, DockerMount) for mount in mounts):
            raise DevToolkitError("Docker mounts must contain DockerMount values")
        targets = [mount.target for mount in mounts]
        if PurePosixPath(str(repository)) in targets or len(targets) != len(set(targets)):
            raise DevToolkitError("Docker mount targets must be unique and exclude the checkout")
        command = tuple(self.command)
        if not command or any(
            not isinstance(argument, str) or not argument or "\0" in argument
            for argument in command
        ):
            raise DevToolkitError("Docker target command must contain non-empty arguments")
        if self.ipc is not None and self.ipc not in {"host", "private"}:
            raise DevToolkitError("Docker IPC mode must be host or private")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "environment", MappingProxyType(values))
        object.__setattr__(self, "mounts", mounts)
        object.__setattr__(self, "command", command)


@dataclass(frozen=True)
class DockerTargetState:
    name: str
    container_id: str
    image_id: str


def _gpu_devices(value: str) -> tuple[str, ...] | None:
    if value == "all":
        return None
    if value == "none":
        return ()
    if not isinstance(value, str) or not value:
        raise DevToolkitError(f"invalid GPU selection: {value!r}")
    devices = tuple(value.split(","))
    if len(devices) != len(set(devices)) or any(
        _DEVICE.fullmatch(device) is None for device in devices
    ):
        raise DevToolkitError(f"invalid GPU selection: {value!r}")
    return devices


def _environment_values(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise DevToolkitError("Docker inspect returned invalid environment data")
    values: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, str) or "=" not in item:
            raise DevToolkitError("Docker inspect returned invalid environment data")
        name, _, value = item.partition("=")
        values[name] = value
    return values


@contextmanager
def _environment_file(values: Mapping[str, str]) -> Iterator[Path | None]:
    if not values:
        yield None
        return
    descriptor, raw_path = tempfile.mkstemp(prefix="trtmc-devtoolkit-", suffix=".env")
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for name in sorted(values):
                output.write(f"{name}={values[name]}\n")
        yield path
    finally:
        path.unlink(missing_ok=True)


class DockerLifecycle:
    """Inspect and prepare one container without deleting or replacing anything."""

    def __init__(self, repository: Path, runner: CommandRunner) -> None:
        self.repository = repository.resolve()
        self.runner = runner

    def _inspect(self, kind: str, identifier: str) -> dict[str, Any] | None:
        result = self.runner.run(
            ["docker", kind, "inspect", identifier],
            cwd=self.repository,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            missing = f"No such {kind}" in (result.stderr or "")
            if missing:
                return None
            raise DevToolkitError(f"Could not inspect Docker {kind} {identifier}")
        try:
            values = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise DevToolkitError(f"Docker returned invalid {kind} inspection data") from error
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise DevToolkitError(f"Docker returned invalid {kind} inspection data")
        return values[0]

    def _container(self, identifier: str) -> dict[str, Any] | None:
        return self._inspect("container", identifier)

    def _image(self, identifier: str) -> dict[str, Any] | None:
        return self._inspect("image", identifier)

    @staticmethod
    def _container_id(container: Mapping[str, Any], description: str) -> str:
        value = container.get("Id")
        if not isinstance(value, str) or not value:
            raise DevToolkitError(f"Docker container {description} has no immutable ID")
        return value

    @staticmethod
    def _image_id(image: Mapping[str, Any], description: str) -> str:
        value = image.get("Id")
        if not isinstance(value, str) or not value:
            raise DevToolkitError(f"Docker image {description} has no immutable ID")
        return value

    @staticmethod
    def _running(container: Mapping[str, Any]) -> bool:
        state = container.get("State")
        running = state.get("Running") if isinstance(state, Mapping) else None
        if not isinstance(running, bool):
            raise DevToolkitError("Docker inspect returned invalid container state")
        return running

    @staticmethod
    def _require_owned(request: DockerTargetRequest, container: Mapping[str, Any]) -> None:
        config = container.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if not isinstance(labels, Mapping) or (
            labels.get(_MANAGED_LABEL) != "true"
            or labels.get(_REPOSITORY_LABEL) != str(request.repository)
            or labels.get(_IMAGE_LABEL) != request.image
        ):
            raise DevToolkitError(
                f"Docker container {request.name} already exists and is not owned by this checkout"
            )

    @staticmethod
    def _expected_mounts(request: DockerTargetRequest) -> tuple[DockerMount, ...]:
        checkout = DockerMount(request.repository, PurePosixPath(str(request.repository)))
        return (checkout, *request.mounts)

    @staticmethod
    def _gpu_matches(request: DockerTargetRequest, container: Mapping[str, Any]) -> bool:
        host = container.get("HostConfig")
        raw = host.get("DeviceRequests") if isinstance(host, Mapping) else None
        requests = raw if isinstance(raw, list) else []
        nvidia = [
            item
            for item in requests
            if isinstance(item, Mapping) and item.get("Driver") in {None, "", "nvidia"}
        ]
        devices = _gpu_devices(request.gpu)
        if devices is None:
            return len(nvidia) == 1 and nvidia[0].get("Count") == -1
        if not devices:
            return not nvidia
        return len(nvidia) == 1 and tuple(nvidia[0].get("DeviceIDs") or ()) == devices

    def _mismatches(
        self,
        request: DockerTargetRequest,
        container: Mapping[str, Any],
        image: Mapping[str, Any],
    ) -> list[str]:
        mismatches: list[str] = []
        image_id = self._image_id(image, request.image)
        if container.get("Image") != image_id:
            mismatches.append("image")
        config = container.get("Config")
        config = config if isinstance(config, Mapping) else {}
        if config.get("WorkingDir") != str(request.repository):
            mismatches.append("working_dir")
        if tuple(config.get("Cmd") or ()) != request.command:
            mismatches.append("command")

        image_config = image.get("Config")
        image_config = image_config if isinstance(image_config, Mapping) else {}
        expected_environment = _environment_values(image_config.get("Env"))
        expected_environment.update(request.environment)
        actual_environment = _environment_values(config.get("Env"))
        for name in sorted(expected_environment.keys() | actual_environment.keys()):
            if expected_environment.get(name) != actual_environment.get(name):
                mismatches.append(f"environment:{name}")

        actual_mounts = container.get("Mounts")
        actual_mounts = actual_mounts if isinstance(actual_mounts, list) else []
        by_target: dict[str, list[Mapping[str, Any]]] = {}
        for item in actual_mounts:
            destination = item.get("Destination") if isinstance(item, Mapping) else None
            if isinstance(destination, str):
                by_target.setdefault(destination, []).append(item)
        expected_mounts = {str(mount.target): mount for mount in self._expected_mounts(request)}
        for target, mount in expected_mounts.items():
            actual = by_target.get(target, [])
            if (
                len(actual) != 1
                or actual[0].get("Type") != "bind"
                or actual[0].get("Source") != str(mount.source)
                or actual[0].get("RW") is not (not mount.read_only)
            ):
                mismatches.append(f"mount:{target}")
        raw_volumes = image_config.get("Volumes")
        image_volumes = (
            {str(target) for target in raw_volumes} if isinstance(raw_volumes, Mapping) else set()
        )
        for target in image_volumes - expected_mounts.keys():
            actual = by_target.get(target, [])
            if len(actual) != 1 or actual[0].get("Type") != "volume":
                mismatches.append(f"mount:{target}")
        allowed_targets = expected_mounts.keys() | image_volumes
        for target in by_target.keys() - allowed_targets:
            mismatches.append(f"mount:{target}")
        if not self._gpu_matches(request, container):
            mismatches.append("gpus")
        host = container.get("HostConfig")
        host = host if isinstance(host, Mapping) else {}
        if request.ipc is not None and host.get("IpcMode") != request.ipc:
            mismatches.append("ipc")
        return sorted(set(mismatches))

    def _create(self, request: DockerTargetRequest, image_id: str) -> str:
        command = [
            "docker",
            "create",
            "--name",
            request.name,
            "--label",
            f"{_MANAGED_LABEL}=true",
            "--label",
            f"{_REPOSITORY_LABEL}={request.repository}",
            "--label",
            f"{_IMAGE_LABEL}={request.image}",
            "--workdir",
            str(request.repository),
        ]
        devices = _gpu_devices(request.gpu)
        if devices is None:
            command.extend(["--gpus", "all"])
        elif devices:
            command.extend(["--gpus", "device=" + ",".join(devices)])
        for mount in self._expected_mounts(request):
            value = f"type=bind,source={mount.source},target={mount.target}"
            if mount.read_only:
                value += ",readonly"
            command.extend(["--mount", value])
        if request.ipc is not None:
            command.extend(["--ipc", request.ipc])
        with _environment_file(request.environment) as environment_file:
            if environment_file is not None:
                command.extend(["--env-file", str(environment_file)])
            command.append(image_id)
            command.extend(request.command)
            result = self.runner.run(
                command,
                cwd=self.repository,
                capture_output=True,
            )
        container_id = result.stdout.strip()
        if not container_id or any(character.isspace() for character in container_id):
            raise DevToolkitError("Docker create did not return one immutable container ID")
        return container_id

    def prepare(
        self,
        request: DockerTargetRequest,
        *,
        policy: DockerTargetPolicy,
        build_image: Callable[[], None],
    ) -> DockerTargetState:
        if not isinstance(policy, DockerTargetPolicy):
            raise DevToolkitError("Docker policy must be a DockerTargetPolicy")
        current = self._container(request.name)
        current_image: dict[str, Any] | None = None
        if current is not None:
            self._require_owned(request, current)
            raw_image = current.get("Image")
            current_image = self._image(str(raw_image)) if isinstance(raw_image, str) else None
            if current_image is None:
                raise DevToolkitError(f"Could not inspect Docker container {request.name} image")
            mismatches = self._mismatches(request, current, current_image)
            if mismatches:
                raise DevToolkitError(
                    f"Docker container {request.name} configuration does not match target: "
                    + ", ".join(mismatches)
                )
        if current is None and policy in {DockerTargetPolicy.ADOPT, DockerTargetPolicy.START}:
            raise DevToolkitError(f"Docker container {request.name} does not exist")

        if policy is DockerTargetPolicy.ENSURE:
            build_image()
            image = self._image(request.image)
        else:
            image = current_image
        if image is None:
            raise DevToolkitError(f"Docker image {request.image} is unavailable")
        image_id = self._image_id(image, request.image)
        container_id: str

        if current is not None:
            container_id = self._container_id(current, request.name)
            mismatches = self._mismatches(request, current, image)
            if mismatches:
                raise DevToolkitError(
                    f"Docker container {request.name} configuration does not match target: "
                    + ", ".join(mismatches)
                )
            if not self._running(current):
                if policy is DockerTargetPolicy.ADOPT:
                    raise DevToolkitError(f"Docker container {request.name} is not running")
                self.runner.run(
                    ["docker", "start", container_id],
                    cwd=self.repository,
                )
        else:
            created_id = self._create(request, image_id)
            current = self._container(created_id)
            if current is None:
                raise DevToolkitError(f"Docker container {request.name} was not created")
            self._require_owned(request, current)
            mismatches = self._mismatches(request, current, image)
            if mismatches:
                raise DevToolkitError(
                    f"Docker container {request.name} configuration does not match target: "
                    + ", ".join(mismatches)
                )
            container_id = self._container_id(current, request.name)
            self.runner.run(
                ["docker", "start", container_id],
                cwd=self.repository,
            )

        final = self._container(container_id)
        if final is None or not self._running(final):
            raise DevToolkitError(f"Docker container {request.name} is not running")
        self._require_owned(request, final)
        mismatches = self._mismatches(request, final, image)
        if mismatches:
            raise DevToolkitError(
                f"Docker container {request.name} configuration does not match target: "
                + ", ".join(mismatches)
            )
        return DockerTargetState(
            name=request.name,
            container_id=self._container_id(final, request.name),
            image_id=image_id,
        )

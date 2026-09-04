# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable execution-target lifecycle capabilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from .docker_support import docker_environment_file
from .models import DevToolkitError
from .providers import FrozenProviderRegistry
from .receipt import exclusive_lock, write_json
from .resolution import ExecutionTarget, ProviderDescriptor
from .runner import Runner, command_output


_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_DOCKER_SIZE = re.compile(r"([0-9]+)(b|k|kb|m|mb|g|gb)?", re.IGNORECASE)


class PullPolicy(Enum):
    NEVER = "never"
    IF_MISSING = "if-missing"
    ALWAYS = "always"


class TargetPolicy(Enum):
    ADOPT = "adopt"
    START = "start"
    ENSURE = "ensure"
    CREATE = "create"


@dataclass(frozen=True)
class DockerImageRef:
    reference: str
    pull: PullPolicy = PullPolicy.IF_MISSING
    expected_digest: str | None = None
    platform: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reference, str)
            or not self.reference
            or self.reference != self.reference.strip()
            or self.reference.startswith("-")
            or any(character.isspace() or character == "\0" for character in self.reference)
        ):
            raise DevToolkitError("Docker image reference must be non-empty and whitespace-free")
        if not isinstance(self.pull, PullPolicy):
            raise DevToolkitError("Docker image pull must be a PullPolicy")
        if self.expected_digest is not None and (
            not isinstance(self.expected_digest, str)
            or _DIGEST.fullmatch(self.expected_digest) is None
        ):
            raise DevToolkitError("Docker image expected_digest must be sha256:<64 lowercase hex>")
        if self.platform is not None and (
            not isinstance(self.platform, str)
            or not self.platform
            or any(character.isspace() or character == "\0" for character in self.platform)
        ):
            raise DevToolkitError("Docker image platform must be non-empty and whitespace-free")


@dataclass(frozen=True)
class DockerImageBuild:
    context: Path
    dockerfile: Path = Path("Dockerfile")
    tag: str | None = None
    target: str | None = None
    platform: str | None = None
    build_args: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", Path(self.context))
        object.__setattr__(self, "dockerfile", Path(self.dockerfile))
        if self.dockerfile.is_absolute() or ".." in self.dockerfile.parts:
            raise DevToolkitError("Dockerfile must be a safe path relative to its build context")
        if self.tag is not None and (
            not isinstance(self.tag, str)
            or not self.tag
            or self.tag != self.tag.strip()
            or self.tag.startswith("-")
            or any(c.isspace() or c == "\0" for c in self.tag)
        ):
            raise DevToolkitError("Docker build tag must be non-empty and whitespace-free")
        for name, value in (("target", self.target), ("platform", self.platform)):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or any(character.isspace() or character == "\0" for character in value)
            ):
                raise DevToolkitError(f"Docker build {name} must be non-empty and whitespace-free")
        values = dict(self.build_args)
        for name, value in values.items():
            if _ENVIRONMENT_NAME.fullmatch(name) is None:
                raise DevToolkitError(f"Invalid Docker build argument name: {name!r}")
            if not isinstance(value, str) or any(character in value for character in "\r\n\0"):
                raise DevToolkitError(f"Docker build argument {name!r} must be one text line")
        object.__setattr__(self, "build_args", MappingProxyType(values))


DockerImage = DockerImageRef | DockerImageBuild


@dataclass(frozen=True)
class DockerMount:
    source: Path
    target: PurePosixPath
    read_only: bool = False

    def __post_init__(self) -> None:
        source = Path(self.source)
        target = PurePosixPath(self.target)
        if not source.is_absolute():
            raise DevToolkitError("Docker bind-mount source must be absolute")
        if not target.is_absolute() or target == PurePosixPath("/"):
            raise DevToolkitError("Docker mount target must be an absolute non-root path")
        if ".." in source.parts or ".." in target.parts:
            raise DevToolkitError("Docker bind-mount paths cannot contain parent traversal")
        if any(character in str(source) + str(target) for character in ",\r\n\0"):
            raise DevToolkitError("Docker bind-mount paths cannot contain commas or line breaks")
        if not isinstance(self.read_only, bool):
            raise DevToolkitError("Docker bind-mount read_only must be boolean")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)


@dataclass(frozen=True)
class DockerGpuRequest:
    device_ids: tuple[str, ...] = ()
    all_devices: bool = False

    @classmethod
    def none(cls) -> DockerGpuRequest:
        return cls()

    @classmethod
    def all(cls) -> DockerGpuRequest:
        return cls(all_devices=True)

    @classmethod
    def devices(cls, *device_ids: str) -> DockerGpuRequest:
        return cls(tuple(str(value) for value in device_ids))

    def __post_init__(self) -> None:
        if not isinstance(self.all_devices, bool):
            raise DevToolkitError("Docker GPU all_devices must be boolean")
        if isinstance(self.device_ids, (str, bytes)):
            raise DevToolkitError("Docker GPU device_ids must be a sequence of IDs")
        values = tuple(self.device_ids)
        if self.all_devices and values:
            raise DevToolkitError("Docker GPU request cannot combine all with device IDs")
        if any(
            not isinstance(value, str)
            or not value
            or "," in value
            or any(c.isspace() or c == "\0" for c in value)
            for value in values
        ):
            raise DevToolkitError("Docker GPU device IDs must be non-empty and whitespace-free")
        if len(values) != len(set(values)):
            raise DevToolkitError("Docker GPU device IDs must be unique")
        object.__setattr__(self, "device_ids", values)


@dataclass(frozen=True)
class DockerTarget:
    name: str
    image: DockerImage | None = None
    docker_context: str | None = None
    python: str = "python3"
    workspace: PurePosixPath = PurePosixPath("/workspace/tensorrt-model-connect")
    state: PurePosixPath = PurePosixPath("/tmp/trtmc-devtoolkit")
    working_dir: PurePosixPath | None = None
    command: tuple[str, ...] | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    mounts: tuple[DockerMount, ...] = ()
    gpus: DockerGpuRequest = field(default_factory=DockerGpuRequest.none)
    ipc: str | None = None
    shm_size: str | None = None
    provider: str = field(default="docker", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _CONTAINER_NAME.fullmatch(self.name) is None:
            raise DevToolkitError(f"Invalid Docker container name: {self.name!r}")
        if self.image is not None and not isinstance(
            self.image, (DockerImageRef, DockerImageBuild)
        ):
            raise DevToolkitError(
                "Docker target image must be a DockerImageRef or DockerImageBuild"
            )
        for name, value in (("python", self.python), ("docker_context", self.docker_context)):
            if value is not None and (
                not isinstance(value, str) or not value or any(c in value for c in "\r\n\0")
            ):
                raise DevToolkitError(f"Docker target {name} must be one non-empty text line")
        workspace = PurePosixPath(self.workspace)
        state = PurePosixPath(self.state)
        working_dir = PurePosixPath(self.working_dir) if self.working_dir is not None else workspace
        for name, path in (
            ("workspace", workspace),
            ("state", state),
            ("working_dir", working_dir),
        ):
            if not path.is_absolute():
                raise DevToolkitError(f"Docker target {name} must be absolute")
        command = tuple(self.command) if self.command is not None else None
        if command is not None and (
            not command
            or any(not isinstance(value, str) or not value or "\0" in value for value in command)
        ):
            raise DevToolkitError("Docker command must contain non-empty arguments")
        environment = dict(self.environment)
        for name, value in environment.items():
            if _ENVIRONMENT_NAME.fullmatch(name) is None:
                raise DevToolkitError(f"Invalid Docker environment name: {name!r}")
            if not isinstance(value, str) or any(character in value for character in "\r\n\0"):
                raise DevToolkitError(f"Docker environment value for {name!r} must be one line")
        mounts = tuple(self.mounts)
        if any(not isinstance(mount, DockerMount) for mount in mounts):
            raise DevToolkitError("Docker target mounts must contain DockerMount values")
        if not isinstance(self.gpus, DockerGpuRequest):
            raise DevToolkitError("Docker target gpus must be a DockerGpuRequest")
        targets = [mount.target for mount in mounts]
        if len(targets) != len(set(targets)):
            raise DevToolkitError("Docker mount targets must be unique")
        if self.ipc is not None and self.ipc not in {"private", "host"}:
            raise DevToolkitError("Docker ipc must be private or host")
        if self.shm_size is not None:
            _docker_size_bytes(self.shm_size)
            object.__setattr__(self, "shm_size", self.shm_size.strip())
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "working_dir", working_dir)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(self, "mounts", mounts)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    if isinstance(value, (Path, PurePosixPath)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (Path, PurePosixPath, Enum)):
        return _freeze(_plain(value))
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) for name in value):
            raise DevToolkitError("Target provider state names must be strings")
        return MappingProxyType({name: _freeze(value[name]) for name in sorted(value)})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    raise DevToolkitError(
        f"Target provider state must be JSON-compatible, got {type(value).__name__}"
    )


def _digest(namespace: bytes, value: object) -> str:
    encoded = json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(namespace + b"\0" + encoded).hexdigest()


@dataclass(frozen=True)
class TargetPlan:
    provider: ProviderDescriptor
    plan_id: str
    intent: Mapping[str, object]
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None:
            raise DevToolkitError("Target plans require a lowercase SHA-256 plan ID")
        frozen = _freeze(self.intent)
        if not isinstance(frozen, Mapping):
            raise DevToolkitError("Target plan intent must be a mapping")
        object.__setattr__(self, "intent", frozen)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "provider": {
                "name": self.provider.name,
                "implementation": self.provider.implementation,
                "lock_schema": self.provider.lock_schema,
            },
            "intent": _plain(self.intent),
        }


@dataclass(frozen=True)
class TargetHandle:
    provider: ProviderDescriptor
    plan_id: str
    target_id: str
    action: str
    identity: Mapping[str, object]
    observation: Mapping[str, object]
    execution_target: ExecutionTarget
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.target_id) is None
        ):
            raise DevToolkitError("Target handles require lowercase SHA-256 identities")
        if not self.action:
            raise DevToolkitError("Target handles require a non-empty action")
        identity = _freeze(self.identity)
        observation = _freeze(self.observation)
        if not isinstance(identity, Mapping) or not isinstance(observation, Mapping):
            raise DevToolkitError("Target handle identity and observation must be mappings")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "observation", observation)


@dataclass(frozen=True)
class ProvisionedTarget:
    provider: ProviderDescriptor
    plan_id: str
    target_id: str
    action: str
    identity: Mapping[str, object]
    observation: Mapping[str, object]
    execution_target: ExecutionTarget
    receipt: Path = field(compare=False)
    request: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        identity = _freeze(self.identity)
        observation = _freeze(self.observation)
        if not isinstance(identity, Mapping) or not isinstance(observation, Mapping):
            raise DevToolkitError("Provisioned target evidence must be mappings")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "observation", observation)


class TargetService:
    def __init__(
        self,
        repository: Path,
        state_root: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
    ) -> None:
        self.repository = repository
        self.state_root = state_root
        self.providers = providers
        self.runner = runner

    def resolve(self, request: object) -> TargetPlan:
        provider_name = getattr(request, "provider", None)
        if not isinstance(provider_name, str) or not provider_name:
            raise DevToolkitError("Target request must declare a provider")
        provider = self.providers.target(provider_name)
        plan = provider.resolve(
            request,
            repository=self.repository,
            runner=self.runner,
        )
        if not isinstance(plan, TargetPlan) or plan.provider != provider.descriptor:
            raise DevToolkitError("Target provider returned an incompatible target plan")
        return plan

    def provision(
        self,
        plan: TargetPlan,
        *,
        policy: TargetPolicy = TargetPolicy.ENSURE,
    ) -> ProvisionedTarget:
        if not isinstance(plan, TargetPlan):
            raise DevToolkitError("Target provisioning requires a TargetPlan")
        if not isinstance(policy, TargetPolicy):
            raise DevToolkitError("Target provisioning policy must be a TargetPolicy")
        state_dir = self.state_root / "targets" / plan.plan_id
        state_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(state_dir / ".provision.lock"):
            write_json(state_dir / "target-plan.json", plan.as_dict())
            (state_dir / "provision-receipt.json").unlink(missing_ok=True)
            (state_dir / "provision-failure.json").unlink(missing_ok=True)
            try:
                provider = self.providers.target(plan.provider.name)
                if provider.descriptor != plan.provider:
                    raise DevToolkitError("Registered target provider does not match target plan")
                handle = provider.provision(
                    plan,
                    policy=policy,
                    repository=self.repository,
                    state_dir=state_dir,
                    runner=self.runner,
                )
                if handle.provider != plan.provider or handle.plan_id != plan.plan_id:
                    raise DevToolkitError("Target provider returned an incompatible target handle")
                provider.attest(
                    handle,
                    repository=self.repository,
                    runner=self.runner,
                )
                receipt = write_json(
                    state_dir / "provision-receipt.json",
                    {
                        "schema_version": 1,
                        "status": "ready",
                        "plan_id": plan.plan_id,
                        "target_id": handle.target_id,
                        "action": handle.action,
                        "policy": policy.value,
                        "attested": True,
                        "provider": plan.as_dict()["provider"],
                        "identity": _plain(handle.identity),
                        "observation": _plain(handle.observation),
                    },
                )
                return ProvisionedTarget(
                    provider=handle.provider,
                    plan_id=handle.plan_id,
                    target_id=handle.target_id,
                    action=handle.action,
                    identity=handle.identity,
                    observation=handle.observation,
                    execution_target=handle.execution_target,
                    receipt=receipt,
                    request=handle.request,
                )
            except Exception as error:
                write_json(
                    state_dir / "provision-failure.json",
                    {
                        "schema_version": 1,
                        "status": "failed",
                        "plan_id": plan.plan_id,
                        "policy": policy.value,
                        "error_type": type(error).__name__,
                    },
                )
                raise

    def ensure(
        self,
        request: object,
        *,
        policy: TargetPolicy = TargetPolicy.ENSURE,
    ) -> ProvisionedTarget:
        return self.provision(self.resolve(request), policy=policy)

    def attest(self, target: ProvisionedTarget) -> None:
        provider = self.providers.target(target.provider.name)
        if provider.descriptor != target.provider:
            raise DevToolkitError("Registered target provider does not match provisioned target")
        provider.attest(
            TargetHandle(
                provider=target.provider,
                plan_id=target.plan_id,
                target_id=target.target_id,
                action=target.action,
                identity=target.identity,
                observation=target.observation,
                execution_target=target.execution_target,
                request=target.request,
            ),
            repository=self.repository,
            runner=self.runner,
        )


def _docker_command(context: str, *arguments: str) -> list[str]:
    return ["docker", "--context", context, *arguments]


def _docker_version(runner: Runner, repository: Path, context: str) -> None:
    output = command_output(
        runner,
        _docker_command(context, "version", "--format", "{{.Client.Version}}"),
        cwd=repository,
        timeout=30,
    )
    match = re.match(r"([0-9]+)\.([0-9]+)", output)
    if match is None or tuple(int(value) for value in match.groups()) < (20, 10):
        raise DevToolkitError(f"Docker CLI 20.10 or newer is required; found {output}")


def _docker_daemon(runner: Runner, repository: Path, context: str) -> str:
    value = command_output(
        runner,
        _docker_command(context, "info", "--format", "{{.ID}}"),
        cwd=repository,
        timeout=30,
    )
    if not value:
        raise DevToolkitError(f"Docker context {context!r} has no daemon identity")
    return value


def _inspect(
    runner: Runner,
    repository: Path,
    command: Sequence[str],
    description: str,
) -> dict[str, object] | None:
    result = runner.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if any(marker in detail.lower() for marker in ("no such", "not found")):
            return None
        raise DevToolkitError(
            f"Could not inspect {description}: {detail or f'Docker exited {result.returncode}'}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise DevToolkitError(f"Could not inspect {description}: {error}") from error
    if not isinstance(payload, dict):
        raise DevToolkitError(f"Docker inspect returned invalid data for {description}")
    return payload


def _inspect_container(
    runner: Runner, repository: Path, context: str, name: str
) -> dict[str, object] | None:
    return _inspect(
        runner,
        repository,
        _docker_command(context, "inspect", "--type", "container", "--format", "{{json .}}", name),
        f"Docker container {name}",
    )


def _inspect_image(
    runner: Runner, repository: Path, context: str, reference: str
) -> dict[str, object] | None:
    return _inspect(
        runner,
        repository,
        _docker_command(context, "image", "inspect", "--format", "{{json .}}", reference),
        f"Docker image {reference}",
    )


def _context_digest(context: Path) -> str:
    digest = hashlib.sha256(b"trtmc-devtoolkit-docker-context-v1\0")
    for path in sorted(context.rglob("*")):
        relative = path.relative_to(context)
        if ".git" in relative.parts:
            continue
        digest.update(str(relative).encode() + b"\0")
        digest.update(f"{path.lstat().st_mode & 0o7777:o}".encode() + b"\0")
        if path.is_symlink():
            digest.update(b"link\0" + os.readlink(path).encode() + b"\0")
        elif path.is_file():
            digest.update(b"file\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"directory\0")
    return digest.hexdigest()


def _environment_identity(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: hashlib.sha256(b"trtmc-devtoolkit-target-env-v1\0" + value.encode()).hexdigest()
        for name, value in sorted(environment.items())
    }


def _docker_size_bytes(value: str) -> int:
    if not isinstance(value, str):
        raise DevToolkitError("Docker shm_size must be a Docker size string")
    match = _DOCKER_SIZE.fullmatch(value.strip())
    if match is None:
        raise DevToolkitError("Docker shm_size must use bytes, k, m, or g units")
    amount = int(match.group(1))
    if amount < 1:
        raise DevToolkitError("Docker shm_size must be greater than zero")
    unit = (match.group(2) or "b").lower()
    multiplier = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }[unit]
    return amount * multiplier


def _image_id(image: Mapping[str, object], description: str) -> str:
    value = image.get("Id")
    if not isinstance(value, str) or not value:
        raise DevToolkitError(f"{description} has no image ID")
    return value


def _container_id(container: Mapping[str, object], description: str) -> str:
    value = container.get("Id")
    if not isinstance(value, str) or not value:
        raise DevToolkitError(f"{description} has no container ID")
    return value


def _running(container: Mapping[str, object]) -> bool:
    state = container.get("State")
    return isinstance(state, Mapping) and state.get("Running") is True


def _actual_config_identity(container: Mapping[str, object]) -> dict[str, object]:
    config = container.get("Config") if isinstance(container.get("Config"), Mapping) else {}
    host = container.get("HostConfig") if isinstance(container.get("HostConfig"), Mapping) else {}
    mounts = container.get("Mounts") if isinstance(container.get("Mounts"), list) else []
    environment: dict[str, str] = {}
    for item in config.get("Env", []) if isinstance(config, Mapping) else []:
        if isinstance(item, str) and "=" in item:
            name, value = item.split("=", 1)
            environment[name] = hashlib.sha256(
                b"trtmc-devtoolkit-target-env-v1\0" + value.encode()
            ).hexdigest()
    return {
        "image_id": container.get("Image"),
        "command": config.get("Cmd") if isinstance(config, Mapping) else None,
        "working_dir": config.get("WorkingDir") if isinstance(config, Mapping) else None,
        "environment": environment,
        "mounts": mounts,
        "device_requests": host.get("DeviceRequests") if isinstance(host, Mapping) else None,
        "ipc": host.get("IpcMode") if isinstance(host, Mapping) else None,
        "shm_size": host.get("ShmSize") if isinstance(host, Mapping) else None,
    }


def _mount_identity(mount: DockerMount) -> dict[str, object]:
    return {
        "source": str(mount.source),
        "target": str(mount.target),
        "read_only": mount.read_only,
    }


class DockerTargetProvider:
    descriptor = ProviderDescriptor("docker", "trtmc-devtoolkit-docker-target==1", 1)

    def resolve(
        self,
        request: object,
        *,
        repository: Path,
        runner: Runner,
    ) -> TargetPlan:
        if not isinstance(request, DockerTarget):
            raise DevToolkitError("Docker target provider requires DockerTarget")
        context = request.docker_context
        if context is None:
            context = command_output(runner, ["docker", "context", "show"], cwd=repository)
        if not context:
            raise DevToolkitError("Docker target requires a non-empty context")
        _docker_version(runner, repository, context)
        daemon_id = _docker_daemon(runner, repository, context)
        mounts = []
        for mount in request.mounts:
            mounts.append(_mount_identity(mount))
        image_intent: dict[str, object] | None
        if isinstance(request.image, DockerImageRef):
            image_intent = {
                "kind": "ref",
                "reference": request.image.reference,
                "pull": request.image.pull.value,
                "expected_digest": request.image.expected_digest,
                "platform": request.image.platform,
            }
        elif isinstance(request.image, DockerImageBuild):
            build_context = request.image.context.resolve()
            dockerfile = (build_context / request.image.dockerfile).resolve()
            if not build_context.is_dir() or not dockerfile.is_file():
                raise DevToolkitError("Docker build context and Dockerfile must exist")
            try:
                dockerfile.relative_to(build_context)
            except ValueError as error:
                raise DevToolkitError("Dockerfile escapes its build context") from error
            image_intent = {
                "kind": "build",
                "context": str(build_context),
                "context_digest": _context_digest(build_context),
                "dockerfile": str(request.image.dockerfile),
                "tag": request.image.tag,
                "target": request.image.target,
                "platform": request.image.platform,
                "build_args": _environment_identity(request.image.build_args),
            }
        elif request.image is None:
            image_intent = None
        else:
            raise DevToolkitError("Unsupported Docker image source")
        intent: dict[str, object] = {
            "daemon_id": daemon_id,
            "docker_context": context,
            "container": {
                "name": request.name,
                "image": image_intent,
                "python": request.python,
                "workspace": str(request.workspace),
                "state": str(request.state),
                "working_dir": str(request.working_dir),
                "command": request.command,
                "environment": _environment_identity(request.environment),
                "mounts": sorted(mounts, key=lambda item: str(item["target"])),
                "gpus": {
                    "all": request.gpus.all_devices,
                    "device_ids": request.gpus.device_ids,
                },
                "ipc": request.ipc,
                "shm_size": request.shm_size,
            },
        }
        plan_id = _digest(
            b"trtmc-devtoolkit-target-plan-v1",
            {
                "provider": {
                    "name": self.descriptor.name,
                    "implementation": self.descriptor.implementation,
                    "lock_schema": self.descriptor.lock_schema,
                },
                "intent": intent,
            },
        )
        return TargetPlan(self.descriptor, plan_id, MappingProxyType(intent), request)

    def _materialize_image(
        self,
        plan: TargetPlan,
        request: DockerTarget,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
        context: str,
        allow_mutation: bool,
        current: Mapping[str, object] | None,
    ) -> tuple[str | None, str]:
        if request.image is None:
            return None, "adopted"
        if isinstance(request.image, DockerImageRef):
            inspected = _inspect_image(runner, repository, context, request.image.reference)
            should_pull = allow_mutation and (
                request.image.pull is PullPolicy.ALWAYS
                or (inspected is None and request.image.pull is PullPolicy.IF_MISSING)
            )
            if inspected is None and request.image.pull is PullPolicy.NEVER:
                inspected = self._existing_container_image(
                    plan,
                    request,
                    current,
                    repository=repository,
                    runner=runner,
                    context=context,
                )
            if should_pull:
                command = _docker_command(context, "pull")
                if request.image.platform:
                    command.extend(["--platform", request.image.platform])
                command.append(request.image.reference)
                runner.run(command, cwd=repository)
                inspected = _inspect_image(runner, repository, context, request.image.reference)
            elif inspected is None:
                inspected = self._existing_container_image(
                    plan,
                    request,
                    current,
                    repository=repository,
                    runner=runner,
                    context=context,
                )
            if inspected is None:
                if request.image.pull is PullPolicy.NEVER:
                    raise DevToolkitError(
                        f"Docker image {request.image.reference} is not available locally"
                    )
                verb = "pull" if not allow_mutation else "resolve"
                raise DevToolkitError(
                    f"Docker image {request.image.reference} is unavailable; cannot {verb} it"
                )
            image_id = _image_id(inspected, f"Docker image {request.image.reference}")
            if request.image.expected_digest is not None:
                repo_digests = inspected.get("RepoDigests")
                values = repo_digests if isinstance(repo_digests, list) else []
                if image_id != request.image.expected_digest and not any(
                    isinstance(value, str)
                    and value.rsplit("@", 1)[-1] == request.image.expected_digest
                    for value in values
                ):
                    raise DevToolkitError(
                        f"Docker image {request.image.reference} digest does not match expectation"
                    )
            return image_id, "pulled" if should_pull else "adopted"
        assert isinstance(request.image, DockerImageBuild)
        image_intent = plan.intent["container"]["image"]  # type: ignore[index]
        assert isinstance(image_intent, Mapping)
        current_digest = _context_digest(request.image.context.resolve())
        if current_digest != image_intent["context_digest"]:
            raise DevToolkitError("Docker build context changed after target resolution")
        tag = request.image.tag or f"trtmc-devtoolkit:{plan.plan_id[:12]}"
        inspected = _inspect_image(runner, repository, context, tag)
        if self._image_belongs_to_plan(inspected, plan.plan_id):
            assert inspected is not None
            return _image_id(inspected, f"Docker image {tag}"), "adopted"
        if not allow_mutation:
            inspected = self._existing_container_image(
                plan,
                request,
                current,
                repository=repository,
                runner=runner,
                context=context,
            )
            if inspected is None:
                raise DevToolkitError(
                    f"Docker image {tag} is not a previously built image for this target plan"
                )
            return _image_id(inspected, f"Docker image {tag}"), "adopted"
        command = _docker_command(
            context,
            "build",
            "--file",
            str((request.image.context / request.image.dockerfile).resolve()),
            "--tag",
            tag,
            "--label",
            f"org.nvidia.trtmc.devtoolkit.target-plan={plan.plan_id}",
        )
        if request.image.target:
            command.extend(["--target", request.image.target])
        if request.image.platform:
            command.extend(["--platform", request.image.platform])
        for name in sorted(request.image.build_args):
            command.extend(["--build-arg", name])
        command.append(str(request.image.context.resolve()))
        runner.run(command, cwd=repository, env=request.image.build_args)
        inspected = _inspect_image(runner, repository, context, tag)
        if inspected is None:
            raise DevToolkitError(f"Docker build did not produce image {tag}")
        return _image_id(inspected, f"Docker image {tag}"), "built"

    @staticmethod
    def _image_belongs_to_plan(
        image: Mapping[str, object] | None,
        plan_id: str,
    ) -> bool:
        if image is None:
            return False
        config = image.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        return (
            isinstance(labels, Mapping)
            and labels.get("org.nvidia.trtmc.devtoolkit.target-plan") == plan_id
        )

    def _existing_container_image(
        self,
        plan: TargetPlan,
        request: DockerTarget,
        current: Mapping[str, object] | None,
        *,
        repository: Path,
        runner: Runner,
        context: str,
    ) -> dict[str, object] | None:
        if current is None:
            return None
        config = current.get("Config")
        config = config if isinstance(config, Mapping) else {}
        labels = config.get("Labels")
        labels = labels if isinstance(labels, Mapping) else {}
        declared_image = config.get("Image")
        owned = labels.get("org.nvidia.trtmc.devtoolkit.target-plan") == plan.plan_id
        declared_ref = (
            isinstance(request.image, DockerImageRef) and declared_image == request.image.reference
        )
        if not owned and not declared_ref:
            return None
        image_id = current.get("Image")
        if not isinstance(image_id, str) or not image_id:
            return None
        return _inspect_image(runner, repository, context, image_id)

    @staticmethod
    def _mismatches(
        request: DockerTarget,
        container: Mapping[str, object],
        image_id: str | None,
    ) -> list[str]:
        mismatches: list[str] = []
        if image_id is not None and container.get("Image") != image_id:
            mismatches.append("image")
        config = container.get("Config")
        config = config if isinstance(config, Mapping) else {}
        if config.get("WorkingDir") != str(request.working_dir):
            mismatches.append("working_dir")
        if request.command is not None and tuple(config.get("Cmd") or ()) != request.command:
            mismatches.append("command")
        actual_environment: dict[str, str] = {}
        for value in config.get("Env", []):
            if isinstance(value, str) and "=" in value:
                name, item = value.split("=", 1)
                actual_environment[name] = item
        for name, value in request.environment.items():
            if actual_environment.get(name) != value:
                mismatches.append(f"environment:{name}")
        actual_mounts = container.get("Mounts")
        actual_mounts = actual_mounts if isinstance(actual_mounts, list) else []
        for mount in request.mounts:
            expected_source = str(mount.source)
            matches = [
                item
                for item in actual_mounts
                if isinstance(item, Mapping) and item.get("Destination") == str(mount.target)
            ]
            if (
                len(matches) != 1
                or matches[0].get("Source") != expected_source
                or (matches[0].get("RW") is not (not mount.read_only))
            ):
                mismatches.append(f"mount:{mount.target}")
        host = container.get("HostConfig")
        host = host if isinstance(host, Mapping) else {}
        device_requests = host.get("DeviceRequests")
        device_requests = device_requests if isinstance(device_requests, list) else []
        nvidia = [
            item
            for item in device_requests
            if isinstance(item, Mapping) and item.get("Driver") in {None, "", "nvidia"}
        ]
        if request.gpus.all_devices:
            if not any(item.get("Count") == -1 for item in nvidia):
                mismatches.append("gpus")
        elif request.gpus.device_ids:
            if not any(
                tuple(item.get("DeviceIDs") or ()) == request.gpus.device_ids for item in nvidia
            ):
                mismatches.append("gpus")
        elif nvidia:
            mismatches.append("gpus")
        if request.ipc is not None and host.get("IpcMode") != request.ipc:
            mismatches.append("ipc")
        if request.shm_size is not None and host.get("ShmSize") != _docker_size_bytes(
            request.shm_size
        ):
            mismatches.append("shm_size")
        return mismatches

    def _create(
        self,
        plan: TargetPlan,
        request: DockerTarget,
        image_id: str,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
        context: str,
    ) -> None:
        command = _docker_command(
            context,
            "create",
            "--name",
            request.name,
            "--label",
            "org.nvidia.trtmc.devtoolkit.managed=true",
            "--label",
            f"org.nvidia.trtmc.devtoolkit.target-plan={plan.plan_id}",
            "--workdir",
            str(request.working_dir),
        )
        if request.gpus.all_devices:
            command.extend(["--gpus", "all"])
        elif request.gpus.device_ids:
            command.extend(["--gpus", "device=" + ",".join(request.gpus.device_ids)])
        for mount in request.mounts:
            value = f"type=bind,source={mount.source},target={mount.target}" + (
                ",readonly" if mount.read_only else ""
            )
            command.extend(["--mount", value])
        if request.ipc is not None:
            command.extend(["--ipc", request.ipc])
        if request.shm_size is not None:
            command.extend(["--shm-size", request.shm_size])
        with docker_environment_file(state_dir, request.environment) as environment_file:
            if environment_file is not None:
                command.extend(["--env-file", str(environment_file)])
            command.append(image_id)
            command.extend(request.command or ("sleep", "infinity"))
            runner.run(command, cwd=repository)

    def provision(
        self,
        plan: TargetPlan,
        *,
        policy: TargetPolicy,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> TargetHandle:
        if not isinstance(plan.request, DockerTarget):
            raise DevToolkitError("Docker target plan lost its typed request")
        request = plan.request
        container_intent = plan.intent.get("container")
        context = plan.intent.get("docker_context")
        daemon_id = plan.intent.get("daemon_id")
        if not isinstance(container_intent, Mapping) or not isinstance(context, str):
            raise DevToolkitError("Docker target plan is malformed")
        if _docker_daemon(runner, repository, context) != daemon_id:
            raise DevToolkitError("Docker daemon changed after target resolution")
        current = _inspect_container(runner, repository, context, request.name)
        if current is None and policy in {TargetPolicy.ADOPT, TargetPolicy.START}:
            raise DevToolkitError(f"Docker container {request.name} does not exist")
        if current is None and request.image is None:
            raise DevToolkitError("Creating a Docker container requires an image")
        if current is not None:
            if policy is TargetPolicy.CREATE and not self._container_belongs_to_plan(
                current, plan.plan_id
            ):
                raise DevToolkitError(
                    f"Docker container {request.name} already exists and is not owned by this plan"
                )
            mismatches = self._mismatches(request, current, None)
            if mismatches:
                raise DevToolkitError(
                    f"Docker container {request.name} configuration does not match target: "
                    + ", ".join(mismatches)
                )
        image_id, image_action = self._materialize_image(
            plan,
            request,
            repository=repository,
            state_dir=state_dir,
            runner=runner,
            context=context,
            allow_mutation=policy in {TargetPolicy.ENSURE, TargetPolicy.CREATE},
            current=current,
        )
        action = "adopted"
        if current is not None:
            mismatches = self._mismatches(request, current, image_id)
            if mismatches:
                raise DevToolkitError(
                    f"Docker container {request.name} configuration does not match target: "
                    + ", ".join(mismatches)
                )
            if not _running(current):
                if policy is TargetPolicy.ADOPT:
                    raise DevToolkitError(f"Docker container {request.name} is not running")
                runner.run(
                    _docker_command(context, "start", _container_id(current, request.name)),
                    cwd=repository,
                )
                action = "started"
        else:
            assert image_id is not None
            try:
                self._create(
                    plan,
                    request,
                    image_id,
                    repository=repository,
                    state_dir=state_dir,
                    runner=runner,
                    context=context,
                )
            except DevToolkitError:
                raced = _inspect_container(runner, repository, context, request.name)
                wrong_owner = policy is TargetPolicy.CREATE and not self._container_belongs_to_plan(
                    raced, plan.plan_id
                )
                if raced is None or wrong_owner or self._mismatches(request, raced, image_id):
                    raise
            current = _inspect_container(runner, repository, context, request.name)
            if current is None:
                raise DevToolkitError(f"Docker container {request.name} was not created")
            runner.run(
                _docker_command(context, "start", _container_id(current, request.name)),
                cwd=repository,
            )
            action = "created"
        current = _inspect_container(runner, repository, context, request.name)
        if current is None or not _running(current):
            raise DevToolkitError(f"Docker container {request.name} is not running")
        mismatches = self._mismatches(request, current, image_id)
        if mismatches:
            raise DevToolkitError(
                f"Docker container {request.name} configuration does not match target: "
                + ", ".join(mismatches)
            )
        container_id = _container_id(current, request.name)
        actual_image_id = _image_id({"Id": current.get("Image")}, request.name)
        config_identity = _actual_config_identity(current)
        config_digest = _digest(b"trtmc-devtoolkit-container-config-v1", config_identity)
        identity = {
            "daemon_id": daemon_id,
            "container_id": container_id,
            "image_id": actual_image_id,
            "config_digest": config_digest,
        }
        target_id = _digest(b"trtmc-devtoolkit-target-v1", identity)
        observation = {
            "running": True,
            "image_action": image_action,
            "container_action": action,
            "container_name": request.name,
        }
        return TargetHandle(
            provider=self.descriptor,
            plan_id=plan.plan_id,
            target_id=target_id,
            action=action,
            identity=MappingProxyType(identity),
            observation=MappingProxyType(observation),
            execution_target=ExecutionTarget.docker(
                python=request.python,
                docker_context=context,
                container=container_id,
                workspace=str(request.workspace),
                state=str(request.state),
            ),
            request=request,
        )

    @staticmethod
    def _container_belongs_to_plan(
        container: Mapping[str, object] | None,
        plan_id: str,
    ) -> bool:
        if container is None:
            return False
        config = container.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        return (
            isinstance(labels, Mapping)
            and labels.get("org.nvidia.trtmc.devtoolkit.target-plan") == plan_id
        )

    def attest(
        self,
        target: TargetHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> None:
        request = target.request
        if not isinstance(request, DockerTarget):
            raise DevToolkitError("Docker target attestation lost its typed request")
        context = str(target.execution_target.options["docker_context"])
        if _docker_daemon(runner, repository, context) != target.identity["daemon_id"]:
            raise DevToolkitError("Docker daemon changed after target provisioning")
        current = _inspect_container(
            runner, repository, context, str(target.identity["container_id"])
        )
        if current is None or not _running(current):
            raise DevToolkitError("Docker target is no longer running")
        if current.get("Image") != target.identity["image_id"]:
            raise DevToolkitError("Docker target image changed after provisioning")
        config_digest = _digest(
            b"trtmc-devtoolkit-container-config-v1", _actual_config_identity(current)
        )
        if config_digest != target.identity["config_digest"]:
            raise DevToolkitError("Docker target configuration changed after provisioning")

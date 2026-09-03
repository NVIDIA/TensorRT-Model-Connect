# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opaque command execution routed through an environment's execution context."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence
from uuid import uuid4

from .models import DevToolkitError
from .providers import FrozenProviderRegistry
from .provisioning import ProvisionedEnvironment, attest_environment
from .receipt import write_json
from .runner import Runner


class PathScope(Enum):
    REPOSITORY = "repository"
    STATE = "state"
    TARGET = "target"


@dataclass(frozen=True)
class EnvironmentPath:
    scope: PathScope
    path: PurePosixPath

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if self.scope is not PathScope.TARGET and (path.is_absolute() or ".." in path.parts):
            raise DevToolkitError(
                f"{self.scope.value} paths must be relative and cannot contain '..'"
            )
        if self.scope is PathScope.TARGET and not path.is_absolute():
            raise DevToolkitError("target paths must be absolute")
        object.__setattr__(self, "path", path)


def repository_path(path: str | Path = ".") -> EnvironmentPath:
    return EnvironmentPath(PathScope.REPOSITORY, PurePosixPath(path))


def state_path(path: str | Path = ".") -> EnvironmentPath:
    return EnvironmentPath(PathScope.STATE, PurePosixPath(path))


def target_path(path: str | Path) -> EnvironmentPath:
    return EnvironmentPath(PathScope.TARGET, PurePosixPath(path))


CommandArgument = str | EnvironmentPath


@dataclass(frozen=True)
class ArtifactInput:
    name: str
    path: EnvironmentPath
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise DevToolkitError("Artifact inputs require a name and lowercase SHA-256")


@dataclass(frozen=True)
class CommandSpec:
    arguments: tuple[CommandArgument, ...]
    cwd: EnvironmentPath = field(default_factory=repository_path)
    environment: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    artifacts: tuple[ArtifactInput, ...] = ()

    def __init__(
        self,
        arguments: Sequence[CommandArgument],
        *,
        cwd: EnvironmentPath | None = None,
        environment: Mapping[str, str] | None = None,
        provenance: Mapping[str, str] | None = None,
        artifacts: Sequence[ArtifactInput] = (),
    ) -> None:
        if not arguments:
            raise DevToolkitError("A command requires at least one argument")
        object.__setattr__(self, "arguments", tuple(arguments))
        object.__setattr__(self, "cwd", cwd or repository_path())
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(dict(environment or {})),
        )
        resolved_provenance = dict(provenance or {})
        if any(not name or not value for name, value in resolved_provenance.items()):
            raise DevToolkitError("Command provenance names and values must be non-empty")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(resolved_provenance),
        )
        object.__setattr__(self, "artifacts", tuple(artifacts))


@dataclass(frozen=True)
class CommandResult:
    occurrence_id: str
    invocation_digest: str
    returncode: int
    stdout: str
    stderr: str
    receipt: Path


def _path_payload(path: EnvironmentPath) -> dict[str, str]:
    return {"scope": path.scope.value, "path": str(path.path)}


def _invocation_digest(environment: ProvisionedEnvironment, command: CommandSpec) -> str:
    environment_digest = hashlib.sha256(
        json.dumps(dict(command.environment), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 3,
        "environment_id": environment.environment_id,
        "arguments": [
            _path_payload(argument) if isinstance(argument, EnvironmentPath) else argument
            for argument in command.arguments
        ],
        "cwd": _path_payload(command.cwd),
        "environment_names": sorted(command.environment),
        "environment_digest": environment_digest,
        "provenance": dict(command.provenance),
        "artifacts": [
            {
                "name": artifact.name,
                "path": _path_payload(artifact.path),
                "sha256": artifact.sha256,
            }
            for artifact in command.artifacts
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"trtmc-devtoolkit-command-v3\0" + encoded).hexdigest()


class CommandExecutor:
    def __init__(
        self,
        repository: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
    ) -> None:
        self.repository = repository
        self.providers = providers
        self.runner = runner

    def run(
        self,
        environment: ProvisionedEnvironment,
        command: CommandSpec,
        *,
        check: bool,
        capture_output: bool,
    ) -> CommandResult:
        invocation = _invocation_digest(environment, command)
        occurrence = uuid4().hex
        receipt = environment.state_dir / "commands" / f"{occurrence}.json"
        try:
            context_provider = self.providers.context(environment.context.provider.name)
            attest_environment(
                environment,
                repository=self.repository,
                providers=self.providers,
                runner=self.runner,
            )
            for artifact in command.artifacts:
                digest_result = context_provider.execute(
                    environment.context,
                    CommandSpec(("sha256sum", artifact.path)),
                    repository=self.repository,
                    state_dir=environment.state_dir,
                    runner=self.runner,
                    check=True,
                    capture_output=True,
                )
                output = digest_result.stdout.strip()
                observed = output.split(None, 1)[0] if output else ""
                if observed != artifact.sha256:
                    raise DevToolkitError(
                        f"Artifact {artifact.name!r} changed after build: "
                        f"expected {artifact.sha256}, observed {observed or 'no digest'}"
                    )
            completed = context_provider.execute(
                environment.context,
                command,
                repository=self.repository,
                state_dir=environment.state_dir,
                runner=self.runner,
                check=check,
                capture_output=capture_output,
            )
        except Exception as error:
            write_json(
                receipt,
                {
                    "schema_version": 3,
                    "status": "failed",
                    "environment_id": environment.environment_id,
                    "occurrence_id": occurrence,
                    "invocation_digest": invocation,
                    "provenance": dict(command.provenance),
                    "artifacts": [artifact.name for artifact in command.artifacts],
                    "error_type": type(error).__name__,
                },
            )
            raise
        write_json(
            receipt,
            {
                "schema_version": 3,
                "status": "completed" if completed.returncode == 0 else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "environment_id": environment.environment_id,
                "occurrence_id": occurrence,
                "invocation_digest": invocation,
                "provenance": dict(command.provenance),
                "artifacts": [artifact.name for artifact in command.artifacts],
                "returncode": completed.returncode,
            },
        )
        return CommandResult(
            occurrence_id=occurrence,
            invocation_digest=invocation,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            receipt=receipt,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic, evidence-producing source builds driven by caller-selected recipes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from .commands import CommandSpec, EnvironmentPath, repository_path, state_path
from .models import DevToolkitError, ToolchainRuntime
from .providers import FrozenProviderRegistry
from .provisioning import ProvisionedEnvironment, attest_environment
from .receipt import exclusive_lock, write_json
from .runner import Runner


@dataclass(frozen=True)
class SourceSnapshot:
    revision: str
    content_digest: str
    dirty: bool


@dataclass(frozen=True)
class BuildArtifact:
    name: str
    path: EnvironmentPath
    sha256: str


@dataclass(frozen=True)
class BuildResult:
    build_request_id: str
    build_id: str
    environment_id: str
    recipe: str
    source: SourceSnapshot
    build_dir: EnvironmentPath
    artifacts: tuple[BuildArtifact, ...]
    receipt: Path

    def artifact(self, name: str) -> BuildArtifact:
        matches = [artifact for artifact in self.artifacts if artifact.name == name]
        if len(matches) != 1:
            raise DevToolkitError(f"Build has no unique artifact named {name!r}")
        return matches[0]


@dataclass(frozen=True)
class BuildContext:
    """Stable facts and a read-only probe available to a build recipe."""

    runtime: ToolchainRuntime
    architecture: str
    _probe: Callable[[CommandSpec], str] = field(repr=False, compare=False)

    def probe(self, command: CommandSpec) -> str:
        return self._probe(command)


@dataclass(frozen=True)
class BuildPlan:
    commands: tuple[CommandSpec, ...]
    outputs: Mapping[str, EnvironmentPath]

    def __post_init__(self) -> None:
        if not self.commands:
            raise DevToolkitError("A build plan requires at least one command")
        outputs = dict(self.outputs)
        if not outputs or any(not name for name in outputs):
            raise DevToolkitError("A build plan requires named outputs")
        object.__setattr__(self, "commands", tuple(self.commands))
        object.__setattr__(self, "outputs", MappingProxyType(outputs))


class BuildRecipe(Protocol):
    """Extension seam for user-defined source build flows."""

    descriptor: str

    def inputs(self, context: BuildContext) -> Mapping[str, object]: ...

    def plan(
        self,
        context: BuildContext,
        inputs: Mapping[str, object],
        build_dir: EnvironmentPath,
    ) -> BuildPlan: ...


def _digest(payload: object, domain: bytes) -> str:
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise DevToolkitError(f"Build identity must be JSON-compatible: {error}") from error
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


class Builder:
    def __init__(
        self,
        repository: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
    ) -> None:
        self.repository = repository
        self.providers = providers
        self.runner = runner

    def _execute(
        self,
        environment: ProvisionedEnvironment,
        command: CommandSpec,
        *,
        check: bool = True,
        capture_output: bool = False,
    ):
        context = self.providers.context(environment.context.provider.name)
        return context.execute(
            environment.context,
            command,
            repository=self.repository,
            state_dir=environment.state_dir,
            runner=self.runner,
            check=check,
            capture_output=capture_output,
        )

    def _completed_result(
        self,
        environment: ProvisionedEnvironment,
        recipe: BuildRecipe,
        source: SourceSnapshot,
        inputs: Mapping[str, object],
        request_id: str,
        build_dir: EnvironmentPath,
        receipt: Path,
        plan: BuildPlan,
    ) -> BuildResult | None:
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected_identity = {
            "schema_version": 3,
            "environment_id": environment.environment_id,
            "source": asdict(source),
            "recipe": recipe.descriptor,
            "inputs": inputs,
            "build_request_id": request_id,
            "status": "completed",
        }
        try:
            normalized_identity = json.loads(
                json.dumps(expected_identity, sort_keys=True, separators=(",", ":"))
            )
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or any(
            payload.get(name) != value for name, value in normalized_identity.items()
        ):
            return None
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(plan.outputs):
            return None
        artifacts_by_name: dict[str, object] = {}
        for raw_artifact in raw_artifacts:
            if not isinstance(raw_artifact, dict):
                return None
            name = raw_artifact.get("name")
            if not isinstance(name, str) or name in artifacts_by_name:
                return None
            artifacts_by_name[name] = raw_artifact
        artifacts: list[BuildArtifact] = []
        for name, path in sorted(plan.outputs.items()):
            raw_artifact = artifacts_by_name.get(name)
            if not isinstance(raw_artifact, dict):
                return None
            recorded_sha256 = raw_artifact.get("sha256")
            if (
                raw_artifact.get("path") != str(path.path)
                or not isinstance(recorded_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", recorded_sha256) is None
            ):
                return None
            hash_result = self._execute(
                environment,
                CommandSpec(("sha256sum", path)),
                check=False,
                capture_output=True,
            )
            output = hash_result.stdout.strip()
            observed_sha256 = output.split(None, 1)[0] if output else ""
            if hash_result.returncode != 0 or observed_sha256 != recorded_sha256:
                return None
            artifacts.append(BuildArtifact(name, path, recorded_sha256))
        artifact_payload = [
            {"name": item.name, "path": str(item.path.path), "sha256": item.sha256}
            for item in artifacts
        ]
        build_id = _digest(
            {"build_request_id": request_id, "artifacts": artifact_payload},
            b"trtmc-devtoolkit-build-result-v3",
        )
        if payload.get("build_id") != build_id:
            return None
        return BuildResult(
            build_request_id=request_id,
            build_id=build_id,
            environment_id=environment.environment_id,
            recipe=recipe.descriptor,
            source=source,
            build_dir=build_dir,
            artifacts=tuple(artifacts),
            receipt=receipt,
        )

    def _source_snapshot(self, environment: ProvisionedEnvironment) -> SourceSnapshot:
        if environment.context.supports_target_operations:
            safe_directory = environment.context.map_path(repository_path())
        else:
            # Third-party contexts written against the original SPI may not expose
            # target path mapping yet. Their commands historically ran locally.
            safe_directory = str(self.repository.resolve())

        def git(*arguments: str) -> CommandSpec:
            # Bind the exception to this invocation only. Do not mutate either the
            # user's or the target environment's global Git configuration.
            return CommandSpec(("git", "-c", f"safe.directory={safe_directory}", *arguments))

        revision = self._execute(
            environment,
            git("rev-parse", "HEAD"),
            capture_output=True,
        ).stdout.strip()
        diff = self._execute(
            environment,
            git("diff", "--binary", "HEAD"),
            capture_output=True,
        ).stdout
        untracked_output = self._execute(
            environment,
            git("ls-files", "--others", "--exclude-standard"),
            capture_output=True,
        ).stdout
        untracked: list[tuple[str, str]] = []
        for path in sorted(line for line in untracked_output.splitlines() if line):
            file_hash = self._execute(
                environment,
                git("hash-object", "--", path),
                capture_output=True,
            ).stdout.strip()
            untracked.append((path, file_hash))
        content = _digest(
            {"revision": revision, "diff": diff, "untracked": untracked},
            b"trtmc-devtoolkit-source-snapshot-v3",
        )
        return SourceSnapshot(revision, content, bool(diff or untracked))

    def build(
        self,
        environment: ProvisionedEnvironment,
        recipe: BuildRecipe,
    ) -> BuildResult:
        preflight_occurrence = uuid4().hex
        preflight_stage = "attestation"
        try:
            attest_environment(
                environment,
                repository=self.repository,
                providers=self.providers,
                runner=self.runner,
            )
            preflight_stage = "source-snapshot"
            source = self._source_snapshot(environment)
            context = BuildContext(
                runtime=environment.toolchain.runtime,
                architecture=environment.lock.context.architecture,
                _probe=lambda command: (
                    self._execute(
                        environment,
                        command,
                        capture_output=True,
                    ).stdout
                ),
            )
            preflight_stage = "recipe-inputs"
            inputs = dict(recipe.inputs(context))
            if not recipe.descriptor:
                raise DevToolkitError("Build recipe descriptor must be non-empty")
        except Exception as error:
            write_json(
                environment.state_dir / "builds" / "preflight" / f"{preflight_occurrence}.json",
                {
                    "schema_version": 3,
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "environment_id": environment.environment_id,
                    "occurrence_id": preflight_occurrence,
                    "stage": preflight_stage,
                    "error_type": type(error).__name__,
                },
            )
            raise
        input_payload = {
            "schema_version": 3,
            "environment_id": environment.environment_id,
            "source": asdict(source),
            "recipe": recipe.descriptor,
            "inputs": inputs,
        }
        request_id = _digest(input_payload, b"trtmc-devtoolkit-build-request-v3")
        build_dir = state_path(f"builds/{request_id}/build")
        receipt = environment.state_dir / "builds" / request_id / "receipt.json"
        with exclusive_lock(receipt.parent / ".build.lock"):
            try:
                plan = recipe.plan(context, MappingProxyType(inputs), build_dir)
                completed = self._completed_result(
                    environment,
                    recipe,
                    source,
                    inputs,
                    request_id,
                    build_dir,
                    receipt,
                    plan,
                )
                if completed is not None:
                    return completed
                for command in plan.commands:
                    self._execute(environment, command)
                artifacts: list[BuildArtifact] = []
                for name, path in sorted(plan.outputs.items()):
                    output = self._execute(
                        environment,
                        CommandSpec(("sha256sum", path)),
                        capture_output=True,
                    ).stdout.strip()
                    sha256 = output.split(None, 1)[0] if output else ""
                    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
                        raise DevToolkitError(f"Could not hash build output {name}: {output}")
                    artifacts.append(BuildArtifact(name, path, sha256))
                artifact_payload = [
                    {"name": item.name, "path": str(item.path.path), "sha256": item.sha256}
                    for item in artifacts
                ]
                build_id = _digest(
                    {"build_request_id": request_id, "artifacts": artifact_payload},
                    b"trtmc-devtoolkit-build-result-v3",
                )
                write_json(
                    receipt,
                    {
                        **input_payload,
                        "status": "completed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "build_request_id": request_id,
                        "build_id": build_id,
                        "artifacts": artifact_payload,
                    },
                )
                return BuildResult(
                    build_request_id=request_id,
                    build_id=build_id,
                    environment_id=environment.environment_id,
                    recipe=recipe.descriptor,
                    source=source,
                    build_dir=build_dir,
                    artifacts=tuple(artifacts),
                    receipt=receipt,
                )
            except Exception as error:
                write_json(
                    receipt,
                    {
                        **input_payload,
                        "status": "failed",
                        "build_request_id": request_id,
                        "error_type": type(error).__name__,
                    },
                )
                raise

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTMC native source builds keyed independently from environment identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from .commands import CommandSpec, EnvironmentPath, repository_path, state_path
from .models import DevToolkitError
from .providers import FrozenProviderRegistry
from .provisioning import ProvisionedEnvironment
from .receipt import write_json
from .runner import Runner


_SOURCE_IDENTITY = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


@dataclass(frozen=True)
class BuildSpec:
    targets: tuple[str, ...] = ("trtmc", "trtmc_backend_trt")
    cmake_defines: Mapping[str, str | int | bool] = field(default_factory=dict)
    cuda_architectures: tuple[str, ...] | None = None
    build_type: str = "Release"
    generator: str = "Ninja"
    jobs: int | None = None
    outputs: Mapping[str, str] = field(default_factory=lambda: {"trtmc": "trtmc"})
    source_identity: str | None = None
    install_python_editable: bool = True

    def __post_init__(self) -> None:
        if not self.targets or any(not target for target in self.targets):
            raise DevToolkitError("A native build requires at least one non-empty target")
        if self.cuda_architectures is not None and not self.cuda_architectures:
            raise DevToolkitError("cuda_architectures cannot be empty")
        if self.jobs is not None and self.jobs < 1:
            raise DevToolkitError("Build jobs must be positive")
        if (
            self.source_identity is not None
            and _SOURCE_IDENTITY.fullmatch(self.source_identity) is None
        ):
            raise DevToolkitError("source_identity must be 40 or 64 lowercase hex characters")
        for name, relative in self.outputs.items():
            path = PurePosixPath(relative)
            if not name or path.is_absolute() or ".." in path.parts:
                raise DevToolkitError("Build output paths must be named, safe relative paths")
        object.__setattr__(self, "targets", tuple(self.targets))
        if self.cuda_architectures is not None:
            object.__setattr__(
                self,
                "cuda_architectures",
                tuple(self.cuda_architectures),
            )
        object.__setattr__(self, "cmake_defines", MappingProxyType(dict(self.cmake_defines)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))


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
    source: SourceSnapshot
    cuda_architectures: tuple[str, ...]
    build_dir: EnvironmentPath
    artifacts: tuple[BuildArtifact, ...]
    receipt: Path


def _digest(payload: object, domain: bytes) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(domain + b"\0" + encoded).hexdigest()


def _define_value(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


class NativeBuilder:
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
        capture_output: bool = False,
    ):
        context = self.providers.context(environment.context.provider.name)
        return context.execute(
            environment.context,
            command,
            repository=self.repository,
            state_dir=environment.state_dir,
            runner=self.runner,
            check=True,
            capture_output=capture_output,
        )

    def _source_snapshot(
        self,
        environment: ProvisionedEnvironment,
        override: str | None,
    ) -> SourceSnapshot:
        if override is not None:
            content = _digest(
                {"source_identity": override},
                b"trtmc-devtoolkit-source-override-v2",
            )
            return SourceSnapshot(override, content, False)
        revision = self._execute(
            environment,
            CommandSpec(("git", "rev-parse", "HEAD")),
            capture_output=True,
        ).stdout.strip()
        diff = self._execute(
            environment,
            CommandSpec(("git", "diff", "--binary", "HEAD")),
            capture_output=True,
        ).stdout
        untracked_output = self._execute(
            environment,
            CommandSpec(("git", "ls-files", "--others", "--exclude-standard")),
            capture_output=True,
        ).stdout
        untracked: list[tuple[str, str]] = []
        for path in sorted(line for line in untracked_output.splitlines() if line):
            file_hash = self._execute(
                environment,
                CommandSpec(("git", "hash-object", "--", path)),
                capture_output=True,
            ).stdout.strip()
            untracked.append((path, file_hash))
        content = _digest(
            {"revision": revision, "diff": diff, "untracked": untracked},
            b"trtmc-devtoolkit-source-snapshot-v2",
        )
        return SourceSnapshot(revision, content, bool(diff or untracked))

    def _architectures(
        self,
        environment: ProvisionedEnvironment,
        requested: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if requested is not None:
            return requested
        output = self._execute(
            environment,
            CommandSpec(
                (
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader,nounits",
                )
            ),
            capture_output=True,
        ).stdout
        architectures = tuple(
            line.strip().replace(".", "") for line in output.splitlines() if line.strip()
        )
        if not architectures:
            raise DevToolkitError("Could not resolve a CUDA architecture for the build")
        return architectures

    def build(
        self,
        environment: ProvisionedEnvironment,
        spec: BuildSpec,
    ) -> BuildResult:
        source = self._source_snapshot(environment, spec.source_identity)
        architectures = self._architectures(environment, spec.cuda_architectures)
        input_payload = {
            "schema_version": 2,
            "environment_id": environment.environment_id,
            "source": asdict(source),
            "cuda_architectures": architectures,
            "targets": spec.targets,
            "cmake_defines": dict(spec.cmake_defines),
            "build_type": spec.build_type,
            "generator": spec.generator,
            "outputs": dict(spec.outputs),
            "install_python_editable": spec.install_python_editable,
        }
        request_id = _digest(input_payload, b"trtmc-devtoolkit-build-request-v2")
        build_dir = state_path(f"builds/{request_id}/build")
        receipt = environment.state_dir / "builds" / request_id / "receipt.json"
        try:
            if spec.install_python_editable:
                python = str(environment.context.locator.get("python", "python3"))
                self._execute(
                    environment,
                    CommandSpec(
                        (
                            python,
                            "-m",
                            "pip",
                            "install",
                            "--no-deps",
                            "-e",
                            repository_path("."),
                            "-C",
                            "py-only=true",
                        )
                    ),
                )
            defines: dict[str, str | int | bool] = {
                "TRTMC_BUILD_BACKEND_TRT": True,
                "TRTMC_BUILD_BACKEND_RTX": False,
                **dict(spec.cmake_defines),
                "CMAKE_CUDA_ARCHITECTURES": ";".join(
                    item if not item.isdigit() else f"{item}-real" for item in architectures
                ),
                "TRTMC_TRT_INCLUDE_DIR": environment.observation.tensorrt_include_dir,
                "TRTMC_TRT_LIBRARY": environment.observation.tensorrt_library,
            }
            configure = [
                "cmake",
                "-S",
                repository_path("."),
                "-B",
                build_dir,
                "-G",
                spec.generator,
                f"-DCMAKE_BUILD_TYPE={spec.build_type}",
            ]
            configure.extend(
                f"-D{name}={_define_value(value)}" for name, value in sorted(defines.items())
            )
            self._execute(environment, CommandSpec(configure))
            build_command: list[str | EnvironmentPath] = [
                "cmake",
                "--build",
                build_dir,
                "--parallel",
            ]
            if spec.jobs is not None:
                build_command.append(str(spec.jobs))
            build_command.extend(("--target", *spec.targets))
            self._execute(environment, CommandSpec(build_command))
            artifacts: list[BuildArtifact] = []
            for name, relative in sorted(spec.outputs.items()):
                path = state_path(f"builds/{request_id}/build/{relative}")
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
                b"trtmc-devtoolkit-build-result-v2",
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
                source=source,
                cuda_architectures=architectures,
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

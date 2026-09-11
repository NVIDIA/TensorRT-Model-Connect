# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose checkout preparation with evidence-producing environment capabilities."""

from __future__ import annotations

import platform
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from .building import BuildRecipe, BuildResult, Builder
from .commands import (
    ArtifactInput,
    CommandArgument,
    CommandExecutor,
    CommandResult,
    CommandSpec,
)
from .docker_target import (
    DockerLifecycle,
    DockerMount,
    DockerTargetPolicy,
    DockerTargetRequest,
)
from .models import DevToolkitError, PreparedEnvironment
from .providers import FrozenProviderRegistry, ProviderRegistry
from .provisioning import EnvironmentProvisioner, ProvisionedEnvironment, ProvisionPolicy
from .qualifications import QualificationRegistry, QualificationSource
from .resolution import EnvironmentLock, EnvironmentRequest, EnvironmentResolver
from .runner import CommandRunner, Runner


_FAMILY = re.compile(r"[a-z][a-z0-9_]*\Z")


class DevToolkit:
    """Prepare a checkout or compose its resolved environment capabilities."""

    def __init__(
        self,
        repository: Path,
        runner: Runner | None = None,
        *,
        state_root: Path | None = None,
        providers: FrozenProviderRegistry | None = None,
        qualifications: Sequence[QualificationSource] = (),
    ):
        self.repository = repository.resolve()
        self.runner = runner or CommandRunner()
        self._capability_state_root = (state_root or self.repository / ".devtoolkit").resolve()
        self._providers = providers or ProviderRegistry.with_builtins().freeze()
        self._qualifications = QualificationRegistry(tuple(qualifications))
        if not (self.repository / "pyproject.toml").is_file():
            raise DevToolkitError(f"not a TensorRT-Model-Connect checkout: {self.repository}")
        if not (self.repository / "families").is_dir():
            raise DevToolkitError(f"checkout has no families directory: {self.repository}")

    @classmethod
    def from_checkout(
        cls,
        repository: Path | None = None,
        *,
        state_root: Path | None = None,
        runner: Runner | None = None,
        providers: FrozenProviderRegistry | None = None,
        qualifications: Sequence[QualificationSource] = (),
    ) -> "DevToolkit":
        return cls(
            repository or Path.cwd(),
            runner=runner,
            state_root=state_root,
            providers=providers,
            qualifications=qualifications,
        )

    def resolve(self, request: EnvironmentRequest) -> EnvironmentLock:
        """Resolve environment intent without mutating the target or state root."""
        return EnvironmentResolver(
            self.repository,
            self._providers,
            self.runner,
            self._qualifications,
        ).resolve(request)

    def provision(
        self,
        lock: EnvironmentLock,
        *,
        policy: ProvisionPolicy = ProvisionPolicy.ADOPT_OR_CREATE,
    ) -> ProvisionedEnvironment:
        """Idempotently satisfy a lock, attest it, and write a receipt."""
        return EnvironmentProvisioner(
            self.repository,
            self._capability_state_root,
            self._providers,
            self.runner,
        ).provision(lock, policy=policy)

    def run(
        self,
        environment: ProvisionedEnvironment,
        command: CommandSpec,
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> CommandResult:
        """Run one opaque command through the selected execution context."""
        return CommandExecutor(self.repository, self._providers, self.runner).run(
            environment,
            command,
            check=check,
            capture_output=capture_output,
        )

    def build(
        self,
        environment: ProvisionedEnvironment,
        recipe: BuildRecipe,
    ) -> BuildResult:
        """Execute a caller-selected source build recipe inside an environment."""
        return Builder(self.repository, self._providers, self.runner).build(
            environment,
            recipe,
        )

    def run_trtmc(
        self,
        environment: ProvisionedEnvironment,
        arguments: Sequence[CommandArgument],
        *,
        build: BuildResult | None = None,
        artifact: str = "trtmc",
        check: bool = True,
        capture_output: bool = False,
    ) -> CommandResult:
        """Run arbitrary TRTMC CLI arguments without interpreting family semantics."""
        executable: CommandArgument = "trtmc"
        provenance: dict[str, str] = {}
        artifacts: tuple[ArtifactInput, ...] = ()
        if build is not None:
            if build.environment_id != environment.environment_id:
                raise DevToolkitError("Build result belongs to a different environment")
            selected = build.artifact(artifact)
            executable = selected.path
            provenance = {
                "build_id": build.build_id,
                f"artifact:{selected.name}": selected.sha256,
            }
            artifacts = (ArtifactInput(selected.name, selected.path, selected.sha256),)
        return self.run(
            environment,
            CommandSpec(
                (executable, *arguments),
                provenance=provenance,
                artifacts=artifacts,
            ),
            check=check,
            capture_output=capture_output,
        )

    def prepare_docker(
        self,
        *,
        family: str | None = None,
        gpu: str = "all",
        image: str = "trtmc-dev:current",
        container: str = "trtmc-dev",
        policy: DockerTargetPolicy = DockerTargetPolicy.ENSURE,
        environment: Mapping[str, str] | None = None,
        mounts: Sequence[DockerMount] = (),
        command: Sequence[str] = ("sleep", "infinity"),
        ipc: str | None = None,
    ) -> PreparedEnvironment:
        """Prepare one exact checkout container without replacing collisions."""
        machine = platform.machine()
        dockerfile = {
            "x86_64": "Dockerfile.dev.x86",
            "aarch64": "Dockerfile.dev.aarch64",
        }.get(machine)
        if dockerfile is None:
            raise DevToolkitError(f"unsupported Docker development host architecture: {machine}")
        if not (self.repository / dockerfile).is_file():
            raise DevToolkitError(f"checkout does not provide {dockerfile}")
        requirements = self._family_requirements(family)
        target = DockerTargetRequest(
            repository=self.repository,
            name=container,
            image=image,
            gpu=gpu,
            environment=dict(environment or {}),
            mounts=tuple(mounts),
            command=tuple(command),
            ipc=ipc,
        )
        state = DockerLifecycle(self.repository, self.runner).prepare(
            target,
            policy=policy,
            build_image=lambda: self._build_docker_image(image, dockerfile),
        )
        if requirements is not None and policy is not DockerTargetPolicy.ADOPT:
            self.runner.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(self.repository),
                    state.container_id,
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    str(requirements.relative_to(self.repository)),
                ],
                cwd=self.repository,
            )
        return PreparedEnvironment(
            kind="docker",
            repository=self.repository,
            python="python3",
            family=family,
            container=container,
            container_id=state.container_id,
            image_id=state.image_id,
        )

    def _build_docker_image(self, image: str, dockerfile: str) -> None:
        self.runner.run(
            ["docker", "build", "--file", dockerfile, "--tag", image, "requirements"],
            cwd=self.repository,
        )

    def prepare_local(
        self,
        *,
        python: str,
        family: str | None = None,
    ) -> PreparedEnvironment:
        """Use one explicit existing system or virtual-environment interpreter."""
        requirements = self._family_requirements(family)
        executable = shutil.which(python)
        if executable is None:
            raise DevToolkitError(f"Python executable was not found: {python}")
        if requirements is not None:
            self.runner.run(
                [
                    executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    str(requirements),
                ],
                cwd=self.repository,
            )
        return PreparedEnvironment(
            kind="local",
            repository=self.repository,
            python=executable,
            family=family,
        )

    def _family_requirements(self, family: str | None) -> Path | None:
        if family is None:
            return None
        if _FAMILY.fullmatch(family) is None:
            raise DevToolkitError(f"invalid family name: {family!r}")
        family_root = self.repository / "families" / family
        if not (family_root / "model.py").is_file():
            raise DevToolkitError(f"unknown family: {family}")
        requirements = family_root / "requirements.txt"
        return requirements if requirements.is_file() else None

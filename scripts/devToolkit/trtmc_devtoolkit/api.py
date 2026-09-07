# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public capability API for TRTMC development environments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .builtin_registry import builtin_provider_registry
from .building import BuildRecipe, BuildResult, Builder
from .commands import ArtifactInput, CommandArgument, CommandExecutor, CommandResult, CommandSpec
from .models import DevToolkitError
from .providers import FrozenProviderRegistry
from .qualifications import QualificationRegistry, QualificationSource
from .provisioning import (
    EnvironmentProvisioner,
    ProvisionedEnvironment,
    ProvisionPolicy,
)
from .resolution import EnvironmentLock, EnvironmentRequest, EnvironmentResolver
from .runner import CommandRunner, Runner
from .target_service import TargetService


class DevToolkit:
    """Compose resolved environments, source builds, and arbitrary commands."""

    def __init__(
        self,
        repository: Path,
        *,
        state_root: Path | None = None,
        runner: Runner | None = None,
        providers: FrozenProviderRegistry | None = None,
        qualifications: Sequence[QualificationSource] = (),
    ):
        self.repository = repository.resolve()
        self._capability_state_root = (state_root or self.repository / ".devtoolkit").resolve()
        self._runner = runner
        self._providers = providers or builtin_provider_registry().freeze()
        self._qualifications = QualificationRegistry(tuple(qualifications))

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
            state_root=state_root,
            runner=runner,
            providers=providers,
            qualifications=qualifications,
        )

    def resolve(self, request: EnvironmentRequest) -> EnvironmentLock:
        """Resolve environment intent without mutating the target or state root."""
        runner = self._runner or CommandRunner()
        return EnvironmentResolver(
            self.repository,
            self._providers,
            runner,
            self._qualifications,
        ).resolve(request)

    @property
    def targets(self) -> TargetService:
        """Compose execution-target lifecycle operations independently of toolchains."""
        runner = self._runner or CommandRunner()
        return TargetService(
            self.repository,
            self._capability_state_root,
            self._providers,
            runner,
        )

    def provision(
        self,
        lock: EnvironmentLock,
        *,
        policy: ProvisionPolicy = ProvisionPolicy.ADOPT_OR_CREATE,
    ) -> ProvisionedEnvironment:
        """Idempotently satisfy a lock, attest it, and write a receipt."""
        runner = self._runner or CommandRunner()
        return EnvironmentProvisioner(
            self.repository,
            self._capability_state_root,
            self._providers,
            runner,
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
        runner = self._runner or CommandRunner()
        return CommandExecutor(self.repository, self._providers, runner).run(
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
        runner = self._runner or CommandRunner()
        return Builder(self.repository, self._providers, runner).build(
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
        """Run arbitrary TRTMC CLI arguments without interpreting model semantics."""
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

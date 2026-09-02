# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for planning and applying TRTMC environment preparation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .building import BuildResult, BuildSpec, NativeBuilder
from .commands import CommandArgument, CommandExecutor, CommandResult, CommandSpec
from .doctor import EnvironmentDoctor
from .models import PrepareRequest, PrepareResult, PreparationPlan
from .planner import Planner
from .providers import FrozenProviderRegistry, ProviderRegistry
from .qualifications import QualificationRegistry
from .provisioning import (
    EnvironmentProvisioner,
    ProvisionedEnvironment,
    ProvisionPolicy,
)
from .receipt import write_doctor, write_failure, write_plan, write_success
from .resolution import EnvironmentLock, EnvironmentRequest, EnvironmentResolver
from .runner import CommandRunner, Runner
from .targets import DockerEnvironment, LocalEnvironment


class DevToolkit:
    """Compose resolved environments, source builds, and arbitrary commands."""

    def __init__(
        self,
        repository: Path,
        *,
        state_root: Path | None = None,
        source_revision_override: str | None = None,
        runner: Runner | None = None,
        providers: FrozenProviderRegistry | None = None,
        qualification_roots: Sequence[Path] = (),
    ):
        self.repository = repository.resolve()
        self._capability_state_root = (state_root or self.repository / ".devtoolkit").resolve()
        self._runner = runner
        self._providers = providers or ProviderRegistry.with_builtins().freeze()
        self._qualifications = QualificationRegistry(
            (
                self.repository / "configs" / "environment-cohorts",
                *(Path(root).resolve() for root in qualification_roots),
            )
        )
        self._planner = Planner(
            self.repository,
            state_root,
            source_revision_override,
        )

    @classmethod
    def from_checkout(
        cls,
        repository: Path | None = None,
        *,
        state_root: Path | None = None,
        source_revision_override: str | None = None,
        runner: Runner | None = None,
        providers: FrozenProviderRegistry | None = None,
        qualification_roots: Sequence[Path] = (),
    ) -> "DevToolkit":
        return cls(
            repository or Path.cwd(),
            state_root=state_root,
            source_revision_override=source_revision_override,
            runner=runner,
            providers=providers,
            qualification_roots=qualification_roots,
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
        spec: BuildSpec | None = None,
    ) -> BuildResult:
        """Build model-agnostic TRTMC native targets inside an environment."""
        runner = self._runner or CommandRunner()
        return NativeBuilder(self.repository, self._providers, runner).build(
            environment,
            spec or BuildSpec(),
        )

    def run_trtmc(
        self,
        environment: ProvisionedEnvironment,
        arguments: Sequence[CommandArgument],
        *,
        executable: str = "trtmc",
        capture_output: bool = False,
    ) -> CommandResult:
        """Run arbitrary TRTMC CLI arguments without interpreting model semantics."""
        return self.run(
            environment,
            CommandSpec((executable, *arguments)),
            capture_output=capture_output,
        )

    def plan(self, request: PrepareRequest) -> PreparationPlan:
        """Resolve an immutable plan without changing Docker, venvs, or build state."""
        return self._planner.create(request)

    def apply(self, plan: PreparationPlan) -> PrepareResult:
        """Apply one plan and leave a reusable environment plus a receipt."""
        plan.state_dir.mkdir(parents=True, exist_ok=True)
        write_plan(plan)
        runner = self._runner or CommandRunner(plan.state_dir / "commands.log")
        try:
            probes, sm = EnvironmentDoctor(self.repository, runner).inspect(
                plan.request,
                plan.cohort,
                plan.architecture,
            )
            write_doctor(plan, probes, sm)
            if plan.request.target.kind == "docker":
                environment, wheel, bundle = DockerEnvironment(self.repository, runner).prepare(
                    plan, sm=sm
                )
            else:
                environment, wheel, bundle = LocalEnvironment(self.repository, runner).prepare(
                    plan, sm=sm
                )
            receipt = write_success(
                plan,
                environment,
                wheel=wheel,
                bundle=bundle,
            )
            return PrepareResult(
                plan=plan,
                environment=environment,
                receipt=receipt,
                wheel=wheel,
                bundle=bundle,
            )
        except BaseException as error:
            write_failure(plan, error)
            raise

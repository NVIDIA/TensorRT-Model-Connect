# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public API for planning and applying TRTMC environment preparation."""

from __future__ import annotations

from pathlib import Path

from .doctor import EnvironmentDoctor
from .models import PrepareRequest, PrepareResult, PreparationPlan
from .planner import Planner
from .receipt import write_doctor, write_failure, write_plan, write_success
from .runner import CommandRunner, Runner
from .targets import DockerEnvironment, LocalEnvironment


class DevToolkit:
    """Prepare a checkout-local, reproducible TRTMC environment."""

    def __init__(
        self,
        repository: Path,
        *,
        state_root: Path | None = None,
        source_revision_override: str | None = None,
        runner: Runner | None = None,
    ):
        self.repository = repository.resolve()
        self._runner = runner
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
    ) -> "DevToolkit":
        return cls(
            repository or Path.cwd(),
            state_root=state_root,
            source_revision_override=source_revision_override,
            runner=runner,
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
                environment, wheel, bundle = DockerEnvironment(
                    self.repository, runner
                ).prepare(plan, sm=sm)
            else:
                environment, wheel, bundle = LocalEnvironment(
                    self.repository, runner
                ).prepare(plan, sm=sm)
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

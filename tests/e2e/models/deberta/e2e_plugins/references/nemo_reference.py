"""Generic NeMo reference backend registration.

Model-specific NeMo reference implementations live under
``tests/e2e/models/<family>/e2e_plugins/references``.
"""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class NemoReference:
    """Reference backend placeholder for model-owned NeMo implementations."""

    @property
    def backend_name(self) -> str:
        return "nemo"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        raise ValueError(
            "Shared NeMo reference backend does not implement model-specific "
            f"task_strategy={case.task_strategy!r} "
            f"runtime_strategy={case.runtime_strategy!r}"
        )


plugin = NemoReference()

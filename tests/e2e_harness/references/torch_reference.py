"""Generic torch reference backend.

Model-specific torch references live under
``tests/e2e/models/<family>/e2e_plugins/references``. The shared backend only
loads precomputed golden snapshots.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


PROJECT_DIR = Path(__file__).resolve().parents[3]
E2E_DIR = PROJECT_DIR / "tests" / "e2e"


class TorchReference:
    """Reference backend for generic golden snapshot artifacts."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        return self._run_golden_snapshot(case, stage, ctx)

    def _run_golden_snapshot(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        golden_dir = case.inputs.get("golden_dir", "")
        if not golden_dir:
            golden_dir = os.path.join(ctx.engine_dir, f"{case.name}_golden")

        if not os.path.isabs(golden_dir):
            golden_dir = str(E2E_DIR / golden_dir)

        golden_file = os.path.join(golden_dir, f"{stage.name}.npy")
        if not os.path.exists(golden_file):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Golden snapshot not found: {golden_file}"},
            )

        try:
            import numpy as np

            golden_data = np.load(golden_file, allow_pickle=True)
            return StageOutput(
                stage_name=stage.name,
                data={"golden_data": golden_data, "source_path": golden_file},
                metadata={"backend": "torch_reference"},
            )
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load golden snapshot: {exc}"},
            )


plugin = TorchReference()

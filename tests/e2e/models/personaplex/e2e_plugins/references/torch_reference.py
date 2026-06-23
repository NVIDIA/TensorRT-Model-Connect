"""PersonaPlex-owned torch reference backend."""

from __future__ import annotations

import os
from pathlib import Path

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


PROJECT_DIR = Path(__file__).resolve().parents[6]
E2E_DIR = PROJECT_DIR / "tests" / "e2e"


class TorchReference:
    """Load PersonaPlex reference token snapshots."""

    @property
    def backend_name(self) -> str:
        return "torch_reference"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        if case.task_strategy != "speech_to_speech":
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unsupported PersonaPlex task_strategy: {case.task_strategy}"},
            )
        return self._run_reference_tokens(case, stage)

    def _run_reference_tokens(self, case: E2ECase, stage: StageSpec) -> StageOutput:
        ref_tokens_path = case.inputs.get(
            "speech_reference_tokens",
            case.metadata.get("speech_reference_tokens", ""),
        )
        if not ref_tokens_path:
            return StageOutput(
                stage_name=stage.name,
                data={"error": "No speech_reference_tokens path in manifest"},
            )

        if not os.path.isabs(ref_tokens_path):
            ref_tokens_path = str(E2E_DIR / ref_tokens_path)

        if not os.path.exists(ref_tokens_path):
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Reference tokens file not found: {ref_tokens_path}"},
            )

        try:
            import numpy as np

            ref_tokens = np.load(ref_tokens_path)
        except Exception as exc:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Failed to load reference tokens: {exc}"},
            )

        return StageOutput(
            stage_name=stage.name,
            data={
                "reference_tokens": ref_tokens,
                "num_frames": ref_tokens.shape[0] if ref_tokens.ndim >= 1 else 0,
                "token_shape": list(ref_tokens.shape),
                "source_path": ref_tokens_path,
            },
            metadata={"backend": "torch_reference"},
        )


plugin = TorchReference()

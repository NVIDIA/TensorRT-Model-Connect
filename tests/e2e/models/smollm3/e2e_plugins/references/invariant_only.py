# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-op reference output for SmolLM3 runtime-invariant contracts."""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class InvariantOnlyReference:
    """Let a contract validate TRT output without an external oracle."""

    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        del case, ctx
        return StageOutput(
            stage_name=stage.name,
            data={"_invariant_only": True},
            timing_s=0.0,
            metadata={"source": "invariant_only"},
        )


plugin = InvariantOnlyReference()

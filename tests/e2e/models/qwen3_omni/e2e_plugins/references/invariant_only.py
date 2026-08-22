# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariant-only reference for the native Qwen3-Omni Thinker path."""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class InvariantOnlyReference:
    """Return a marker that asks the comparator to validate TRT output only."""

    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        del case, ctx
        return StageOutput(
            stage_name=stage.name,
            data={"_invariant_only": True},
            metadata={
                "source": "invariant_only",
                "note": "Comparator checks native Thinker output invariants",
            },
        )


plugin = InvariantOnlyReference()

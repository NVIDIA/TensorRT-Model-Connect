# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariant-only reference backend for Cosmos3 native qualification."""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class InvariantOnlyReference:
    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        return StageOutput(
            stage_name=stage.name,
            data={"_invariant_only": True},
            timing_s=0.0,
            metadata={"source": "invariant_only"},
        )


plugin = InvariantOnlyReference()

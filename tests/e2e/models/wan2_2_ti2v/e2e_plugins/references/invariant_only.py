# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariant-only reference for source-bound qualification outside ordinary CI."""

from __future__ import annotations

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class Wan22InvariantReference:
    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del case, ctx
        return StageOutput(
            stage_name=stage.name,
            data={"_invariant_only": True},
        )


plugin = Wan22InvariantReference()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariant-only reference for the public OpenFold3 premerge smoke."""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


class OpenFold3InvariantReference:
    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del case, ctx
        return StageOutput(stage_name=stage.name, data={"_invariant_only": True})


reference = OpenFold3InvariantReference()

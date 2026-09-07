# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invariant-only reference backend — no external reference needed.

Returns a dummy StageOutput. The comparator only checks invariants
(nan, range, shape) on the TRT output itself. Used for models where
no reference implementation is available.
"""

from __future__ import annotations

import logging

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class InvariantOnlyReference:
    """No-op reference backend for invariant-only testing."""

    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        logger.info(
            "Invariant-only reference for %s/%s — returning dummy output",
            case.name, stage.name,
        )
        return StageOutput(
            stage_name=stage.name,
            data={"_invariant_only": True},
            text=None,
            timing_s=0.0,
            metadata={
                "source": "invariant_only",
                "note": "No external reference; comparator checks invariants only",
            },
        )


plugin = InvariantOnlyReference()

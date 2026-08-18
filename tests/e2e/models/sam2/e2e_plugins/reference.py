# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the externally provisioned SAM2 golden evidence."""

from __future__ import annotations

from pathlib import Path

from tests.e2e.models.sam2.e2e_plugins.runner import (
    _FRAME_PIXELS,
    _HEIGHT,
    _MASK_TO_GRAYSCALE,
    _WIDTH,
    _load_golden,
    _png,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


class Sam2LocalGoldenReference:
    @property
    def backend_name(self) -> str:
        return "sam2_local_golden"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        fixture_dir = Path(ctx.engine_dir) / str(case.inputs["fixture_dir"])
        _, masks, manifest_hash = _load_golden(fixture_dir)
        if not ctx.artifacts_dir:
            raise RuntimeError("SAM2 reference report artifacts are unavailable")
        output_dir = Path(ctx.artifacts_dir) / case.name
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "reference_segmentation_mask.png"
        mask_path.write_bytes(
            _png(_WIDTH, _HEIGHT, 1, masks[:_FRAME_PIXELS].translate(_MASK_TO_GRAYSCALE))
        )
        return StageOutput(
            stage_name=stage.name,
            data={"golden_manifest_sha256": manifest_hash, "viz_path": str(mask_path)},
        )


class Sam2InvariantReference:
    @property
    def backend_name(self) -> str:
        return "invariant_only"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del case, ctx
        return StageOutput(stage_name=stage.name, data={"_invariant_only": True})


reference = (Sam2LocalGoldenReference(), Sam2InvariantReference())

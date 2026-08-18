# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the exact five-frame PyTorch reference retained in the source archive."""

from __future__ import annotations

from .._schema import (
    FRAME_COUNT,
    load_npz_arrays,
    resolve_project_path,
    structured_summary,
    validate_dimensions,
)
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec


class Sam2HoiArchiveReference:
    """Load, validate, and expose the package's trusted PyTorch NPZ output."""

    @property
    def backend_name(self) -> str:
        return "sam2_hoi_archive_reference"

    def run_stage(
        self,
        case: E2ECase,
        stage: StageSpec,
        ctx: RunContext,
    ) -> StageOutput:
        del ctx
        if stage.name != "full_tracking":
            raise ValueError(f"Unsupported SAM2 HOI reference stage: {stage.name}")
        expected_source_commit = case.metadata.get("source_commit")
        model_root = resolve_project_path(case.hf_id)
        source_commit_path = model_root / "SOURCE_COMMIT"
        try:
            source_commit = source_commit_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(
                f"Could not read SAM2 HOI source provenance: {source_commit_path}"
            ) from error
        if source_commit != expected_source_commit:
            raise RuntimeError(
                "SAM2 HOI reference source commit mismatch: "
                f"expected {expected_source_commit}, got {source_commit}"
            )

        value = case.inputs.get("reference_npz")
        if not isinstance(value, str) or not value:
            raise ValueError("SAM2 HOI E2E input reference_npz is required")
        reference_path = resolve_project_path(value)
        arrays = load_npz_arrays(reference_path)
        validate_dimensions(
            arrays,
            height=int(case.inputs.get("expected_height", 1280)),
            width=int(case.inputs.get("expected_width", 1088)),
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 1,
                "frame_count": FRAME_COUNT,
                "frames": structured_summary(arrays),
                "output_path": str(reference_path),
                "output_npz": str(reference_path),
            },
            metadata={
                "source_commit": source_commit,
                "reference_kind": "packaged_pytorch_bf16_snapshot",
            },
        )


plugin = Sam2HoiArchiveReference()

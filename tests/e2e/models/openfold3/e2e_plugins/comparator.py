# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate native OpenFold3 structure and confidence invariants."""

from __future__ import annotations

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


class OpenFold3InvariantComparator:
    @property
    def task_strategy(self) -> str:
        return "structure_prediction"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        del threshold
        if ref.data.get("_invariant_only") is not True:
            raise RuntimeError("OpenFold3 premerge requires its invariant-only oracle")
        data = trt.data.get("runtime_invariants", {})
        tokens = 76
        atoms = 601
        checks = {
            "schema_version": trt.data.get("schema_version") == 1,
            "mixed_fp16_default": data.get("precision") == "fp16-mixed",
            "repeat_exact": data.get("repeat_exact") is True,
            "bundle_unchanged": data.get("bundle_unchanged") is True,
            "token_count": data.get("token_count") == tokens,
            "atom_count": data.get("atom_count") == atoms,
            "atom_rows": data.get("atom_rows") == atoms,
            "plddt_extent": data.get("plddt_count") == atoms,
            "pae_extent": data.get("pae_count") == tokens * tokens,
            "pde_extent": data.get("pde_count") == tokens * tokens,
            "finite_confidence": data.get("finite_confidence") is True,
            "valid_confidence_ranges": data.get("valid_confidence_ranges") is True,
            "finite_coordinates": data.get("finite_coordinates") is True,
            "nondegenerate_coordinates": float(data.get("coordinate_extent", 0.0)) > 1.0,
            "mmcif_auth_atom_id": data.get("mmcif_auth_atom_id") is True,
            "mmcif_label_entity_id": data.get("mmcif_label_entity_id") is True,
            "sample_rank": data.get("sample_rank") == 0,
            "ranking_not_applicable": data.get("ranking_not_applicable") is True,
        }
        metrics = {
            name: MetricResult(
                value=1.0 if passed else 0.0,
                threshold=1.0,
                operator="==",
                passed=passed,
            )
            for name, passed in checks.items()
        }
        passed = all(checks.values())
        failed = sorted(name for name, value in checks.items() if not value)
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all native OpenFold3 structure invariants must pass",
            message=(
                "OpenFold3 mixed-FP16 native smoke: PASS"
                if passed
                else "OpenFold3 mixed-FP16 native smoke: FAIL (" + ", ".join(failed) + ")"
            ),
        )


comparator = OpenFold3InvariantComparator()

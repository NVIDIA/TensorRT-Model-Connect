# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed native-output checks for the Boltz-2 E2E smoke."""

from __future__ import annotations

import math

from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    ThresholdProfile,
)


class Boltz2StructureComparator:
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
        expected_atoms = int(threshold.metrics.get("atom_count", 899))
        expected_tokens = int(threshold.metrics.get("token_count", 117))
        confidence_atol = float(threshold.metrics.get("confidence_atol", 1.0e-6))
        checks = {
            "mmcif_header_valid": bool(trt.data.get("mmcif_header_valid")),
            "coordinates_finite": bool(trt.data.get("coordinates_finite")),
            "b_factors_finite": bool(trt.data.get("b_factors_finite")),
            "atom_count": int(trt.data.get("atom_count", -1)) == expected_atoms,
            "token_count": int(trt.data.get("token_count", -1)) == expected_tokens,
            "reference_atom_count": int(ref.data.get("atom_count", -1)) == expected_atoms,
            "reference_token_count": int(ref.data.get("token_count", -1)) == expected_tokens,
            "structure_sha256": trt.data.get("structure_sha256")
            == ref.data.get("structure_sha256"),
            "metadata_sha256": trt.data.get("metadata_sha256")
            == ref.data.get("metadata_sha256"),
        }
        for name in ("confidence_score", "complex_plddt", "ptm"):
            value = trt.data.get(name)
            reference = ref.data.get(name)
            checks[name] = (
                isinstance(value, (int, float))
                and isinstance(reference, (int, float))
                and math.isfinite(value)
                and math.isfinite(reference)
                and 0 <= value <= 1
                and abs(value - reference) <= confidence_atol
            )
        metrics = {
            name: MetricResult(value=1.0 if passed else 0.0, threshold=1.0,
                               operator="==", passed=passed)
            for name, passed in checks.items()
        }
        passed = all(checks.values())
        return CompareResult(
            stage_name=stage.name,
            status="passed" if passed else "failed",
            metrics=metrics,
            composite_rule=(
                f"valid mmCIF AND exact qualified output snapshot AND finite "
                f"{expected_atoms}-atom coordinates AND {expected_tokens} confidence rows"
            ),
            message="Boltz-2 native structure snapshot passed" if passed
            else "Boltz-2 native structure snapshot failed",
        )


comparator = Boltz2StructureComparator()

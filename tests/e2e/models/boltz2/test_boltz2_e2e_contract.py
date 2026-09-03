# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static and fail-closed checks for the Boltz-2 E2E contract."""

from pathlib import Path

from tests.e2e.models.boltz2.e2e_plugins.comparator import Boltz2StructureComparator
from tests.e2e.models.boltz2.e2e_plugins.reference import _SNAPSHOTS
from tests.e2e.models.boltz2.e2e_plugins.runner import _inspect_mmcif
from tests.e2e_harness.contracts import CompareResult, StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_model_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
)


_MODEL_DIR = Path(__file__).resolve().parent


def _compare(candidate: dict) -> CompareResult:
    snapshot = _SNAPSHOTS["boltz2-protein-monomer"]
    return Boltz2StructureComparator().compare(
        StageOutput(stage_name="end_to_end", data=candidate),
        StageOutput(stage_name="end_to_end", data=snapshot),
        ThresholdProfile(
            task_strategy="structure_prediction",
            metrics={"atom_count": 899, "token_count": 117},
        ),
        StageSpec(name="end_to_end"),
    )


def test_manifest_declares_native_structure_contract() -> None:
    model = load_model_manifest(_MODEL_DIR / "manifests/boltz2-protein-monomer.json")
    case = model.testcases[0]
    assert case.family == "boltz2"
    assert case.runtime_strategy == "boltz2_structure_prediction"
    assert case.task_strategy == "structure_prediction"
    assert case.reference_backend == "boltz2_snapshot"
    assert case.oracle_level == "L3_snapshot_regression"


def test_primary_manifest_reuses_one_bundle_for_shape_compatible_requests() -> None:
    model = load_model_manifest(_MODEL_DIR / "manifests/boltz2-protein-monomer.json")
    assert len(model.testcases) == 2
    case = model.testcases[1]
    assert case.inputs["request"].endswith("protein_monomer_variant.yaml")
    assert case.threshold_overrides["token_count"] == 117
    assert case.threshold_overrides["atom_count"] == 899


def test_model_owned_plugins_are_discoverable() -> None:
    activate_model_plugins(_MODEL_DIR)
    assert get_runner("structure_prediction").strategy_name == "structure_prediction"
    assert get_comparator("structure_prediction").task_strategy == "structure_prediction"
    assert get_reference("boltz2_snapshot").backend_name == "boltz2_snapshot"


def test_native_mmcif_parser_requires_complete_atom_rows(tmp_path: Path) -> None:
    structure = tmp_path / "prediction.cif"
    structure.write_text(
        "data_boltz2\n#\nloop_\n"
        "ATOM 1 C CA ALA A 1 1.0 2.0 3.0 1.00 75.0 1\n",
        encoding="utf-8",
    )
    result = _inspect_mmcif(structure)
    assert result == {
        "mmcif_header_valid": True,
        "atom_count": 1,
        "coordinates_finite": True,
        "b_factors_finite": True,
    }


def test_comparator_fails_closed_when_confidence_is_missing() -> None:
    result = _compare(
        {
            "mmcif_header_valid": True,
            "coordinates_finite": True,
            "b_factors_finite": True,
            "atom_count": 899,
            "token_count": 117,
        }
    )
    assert result.status == "failed"
    assert not result.metrics["confidence_score"].passed


def test_comparator_rejects_output_that_differs_from_snapshot() -> None:
    snapshot = _SNAPSHOTS["boltz2-protein-monomer"]
    result = _compare(
        {
            **snapshot,
            "structure_sha256": "0" * 64,
            "mmcif_header_valid": True,
            "coordinates_finite": True,
            "b_factors_finite": True,
        }
    )

    assert result.status == "failed"
    assert not result.metrics["structure_sha256"].passed

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape/representation checks do not replace family numerical parity tests."""

from copy import deepcopy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

import tools.perf_matrix as perf


OPERATIONS = {
    "geometry": ("image_to_metric_geometry", "metric-geometry-shape"),
    "predict_structure": ("molecular_document_to_structure", "molecular-structure-shape"),
    "refine_pose": ("pose_hypotheses_crops_to_refined_poses", "pose-refinement-shape"),
}


def _entry(operation):
    return perf.ResolvedEntry(
        spec={"id": "structured-fixture", "operation": operation, "baseline": {}},
        model=SimpleNamespace(task=OPERATIONS[operation][0]),
        case=SimpleNamespace(
            selected_task=None,
            request={},
            measurement=SimpleNamespace(asset_loading_included=False),
        ),
        manifest={},
        reference_precision="fp32",
        baseline_timing={
            "timing_scope": "task-pipeline-call-wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        },
    )


def _geometry(root: Path):
    root.mkdir()
    # Two HWC pixels: a valid metric point and an invalid +Inf sentinel.
    (root / "points.f32").write_bytes(struct.pack("<6f", 1, 2, 3, *([float("inf")] * 3)))
    (root / "depth.f32").write_bytes(struct.pack("<2f", 3, float("inf")))
    (root / "mask.u8").write_bytes(bytes((1, 0)))
    return {
        "geometry_images": 1,
        "geometry_pixels": 2,
        "height": 1,
        "width": 2,
        "point_shape": [1, 2, 3],
        "valid_pixels": 1,
        "normalized_intrinsics": [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        "units": "meters",
        "camera_axes": ["right", "down", "forward"],
        "intrinsics_coordinates": "normalized_uv",
        "points_artifact": str(root / "points.f32"),
        "depth_artifact": str(root / "depth.f32"),
        "valid_mask_artifact": str(root / "mask.u8"),
    }


def _structure(root: Path):
    root.mkdir()
    structure = b"data_fixture\n\x00retained bytes\n"
    metadata = b"opaque\x00metadata"
    (root / "structure.cif").write_bytes(structure)
    (root / "metadata.json").write_bytes(metadata)
    return {
        "structures": 1,
        "structure_bytes": len(structure),
        "metadata_bytes": len(metadata),
        "document_bytes": 10,
        "format": "mmcif",
        "input_encoding": "b2rq",
        "source_path": "",
        "confidence": {
            "confidence_score": 0.1,
            "ptm": 0.2,
            "iptm": 0.3,
            "ligand_iptm": 0.4,
            "protein_iptm": 0.5,
            "complex_plddt": 66.0,
            "complex_iplddt": 77.0,
            "plddt": [88.0, 0.0, 99.0],
        },
        "structure_artifact": str(root / "structure.cif"),
        "metadata_artifact": str(root / "metadata.json"),
    }


def _poses(_root: Path):
    values = [float(row == column) for row in range(4) for column in range(4)] * 2
    return {
        "refined_hypotheses": 2,
        "shape": [2, 4, 4],
        "refined_poses": values,
        "scores": [0.25, 0.5],
        "best_index": 1,
        "all_poses_rigid": True,
        "refinement_ms": 0.0,
        "scoring_ms": 0.0,
        "crop_queries": [
            {"stage": "refinement", "iteration": 0, "shape": [2, 4, 4], "poses": list(values)},
            {"stage": "scoring", "iteration": 1, "shape": [2, 4, 4], "poses": list(values)},
        ],
    }


FIXTURES = {"geometry": _geometry, "predict_structure": _structure, "refine_pose": _poses}


def _compare(operation, left, right):
    entry = _entry(operation)
    candidate = {
        "output_summary": left,
        "timing_scope": "public_task_call_wall",
        "asset_loading_included": False,
        "metrics": {"latency_ms": {"p50": 1.0}},
    }
    baseline = {
        "output_summary": right,
        "measurement_policy": entry.baseline_timing,
        "precision": "fp32",
        "metrics": {"latency_ms": {"p50": 1.0}},
    }
    return perf.compare(entry, candidate, baseline)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_complete_typed_artifacts_reach_performance_comparison_without_numeric_parity(
    tmp_path, operation
):
    left = FIXTURES[operation](tmp_path / "candidate")
    right = FIXTURES[operation](tmp_path / "reference")
    assert perf._contract_name(_entry(operation)) == OPERATIONS[operation][1]
    status, evidence = _compare(operation, left, right)
    assert status == "yellow"
    assert evidence["output_contract"] == {
        "contract": OPERATIONS[operation][1],
        "numerical_parity_checked": False,
    }


@pytest.mark.parametrize(
    "operation,field",
    [
        ("geometry", name)
        for name in (
            "geometry_images",
            "geometry_pixels",
            "height",
            "width",
            "point_shape",
            "valid_pixels",
            "normalized_intrinsics",
            "units",
            "camera_axes",
            "intrinsics_coordinates",
            "points_artifact",
            "depth_artifact",
            "valid_mask_artifact",
        )
    ]
    + [
        ("predict_structure", name)
        for name in (
            "structures",
            "structure_bytes",
            "metadata_bytes",
            "document_bytes",
            "format",
            "input_encoding",
            "source_path",
            "confidence",
            "structure_artifact",
            "metadata_artifact",
        )
    ]
    + [
        ("refine_pose", name)
        for name in (
            "refined_hypotheses",
            "shape",
            "refined_poses",
            "scores",
            "best_index",
            "all_poses_rigid",
            "refinement_ms",
            "scoring_ms",
            "crop_queries",
        )
    ],
)
def test_summary_counts_cannot_replace_any_complete_typed_output_field(tmp_path, operation, field):
    left = FIXTURES[operation](tmp_path / "candidate")
    right = deepcopy(left)
    del right[field]
    status, evidence = _compare(operation, left, right)
    assert status == "contract-mismatch"
    assert evidence["reason"].startswith("reference ")
    assert evidence["output_contract"]["numerical_parity_checked"] is False


@pytest.mark.parametrize("side", ("candidate", "reference"))
@pytest.mark.parametrize(
    "operation,field",
    [
        ("geometry", "points_artifact"),
        ("geometry", "depth_artifact"),
        ("geometry", "valid_mask_artifact"),
        ("predict_structure", "structure_artifact"),
        ("predict_structure", "metadata_artifact"),
    ],
)
@pytest.mark.parametrize("malformation", ("missing", "truncated", "longer"))
def test_each_artifact_must_exist_and_match_its_own_declared_length(
    tmp_path, side, operation, field, malformation
):
    left = FIXTURES[operation](tmp_path / "candidate")
    right = FIXTURES[operation](tmp_path / "reference")
    summary = left if side == "candidate" else right
    artifact = Path(summary[field])
    if malformation == "missing":
        summary[field] = str(artifact.with_name("not-written"))
    elif malformation == "truncated":
        artifact.write_bytes(artifact.read_bytes()[:-1])
    else:
        artifact.write_bytes(artifact.read_bytes() + b"\x00")
    status, evidence = _compare(operation, left, right)
    assert status == "contract-mismatch"
    assert evidence["reason"].startswith(side + " ")
    assert field in evidence["reason"]


@pytest.mark.parametrize(
    "field,value",
    (
        ("height", 0),
        ("width", True),
        ("geometry_images", 2),
        ("geometry_pixels", 3),
        ("point_shape", [2, 1, 3]),
        ("point_shape", [True, 2, 3]),
        ("valid_pixels", 2),
        ("valid_pixels", True),
        ("normalized_intrinsics", [[1, 0, 0], [0, 1, 0]]),
        ("normalized_intrinsics", [[1, 0, 0], [0, float("inf"), 0], [0, 0, 1]]),
        ("units", "centimeters"),
        ("camera_axes", ["right", "up", "backward"]),
        ("intrinsics_coordinates", "pixels"),
    ),
)
def test_geometry_grid_calibration_and_representation_are_checked(tmp_path, field, value):
    left = _geometry(tmp_path / "candidate")
    right = deepcopy(left)
    right[field] = value
    assert _compare("geometry", left, right)[0] == "contract-mismatch"


def test_geometry_shape_does_not_filter_invalid_infinity_or_add_depth_accuracy_rules(tmp_path):
    left = _geometry(tmp_path / "candidate")
    right = _geometry(tmp_path / "reference")
    originals = {
        name: Path(left[name]).read_bytes()
        for name in (
            "points_artifact",
            "depth_artifact",
            "valid_mask_artifact",
        )
    }
    # Different numeric values, valid-pixel count and estimated intrinsics are
    # not compared. Nor must every family's invalid representation be +Inf.
    Path(right["points_artifact"]).write_bytes(struct.pack("<6f", 10, 20, -3, 0, 0, 0))
    Path(right["depth_artifact"]).write_bytes(struct.pack("<2f", -3, 0))
    Path(right["valid_mask_artifact"]).write_bytes(bytes((1, 1)))
    right["valid_pixels"] = 2
    right["normalized_intrinsics"][0][0] = 2.0
    assert _compare("geometry", left, right)[0] == "yellow"
    assert all(Path(left[name]).read_bytes() == data for name, data in originals.items())
    assert struct.unpack("<2f", originals["depth_artifact"])[1] == float("inf")


def test_optional_reference_calibration_artifact_is_not_silently_ignored(tmp_path):
    left = _geometry(tmp_path / "candidate")
    right = _geometry(tmp_path / "reference")
    path = tmp_path / "reference/intrinsics.json"
    right["intrinsics_artifact"] = str(path)
    calibration = {
        "height": 1,
        "width": 2,
        "normalized": True,
        "intrinsics": right["normalized_intrinsics"],
    }
    path.write_text(json.dumps(calibration))
    assert _compare("geometry", left, right)[0] == "yellow"
    calibration["intrinsics"] = deepcopy(calibration["intrinsics"])
    calibration["intrinsics"][0][0] = True  # Python equality alone would equate True and 1.0.
    path.write_text(json.dumps(calibration))
    assert _compare("geometry", left, right)[0] == "contract-mismatch"
    calibration["intrinsics"] = right["normalized_intrinsics"]
    calibration["normalized"] = False
    path.write_text(json.dumps(calibration))
    assert _compare("geometry", left, right)[0] == "contract-mismatch"
    path.write_bytes(b"not json")
    assert _compare("geometry", left, right)[0] == "contract-mismatch"


def test_geometry_compares_real_grid_axes_not_only_pixel_count(tmp_path):
    left = _geometry(tmp_path / "candidate")
    right = _geometry(tmp_path / "reference")
    right.update(height=2, width=1, point_shape=[2, 1, 3])
    assert left["geometry_pixels"] == right["geometry_pixels"]
    assert _compare("geometry", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("structures", 2),
        ("structure_bytes", 0),
        ("metadata_bytes", True),
        ("document_bytes", 0),
        ("format", "json"),
        ("format", []),
        ("input_encoding", ""),
        ("source_path", None),
        ("confidence", []),
    ),
)
def test_structure_rejects_malformed_representation(tmp_path, field, value):
    left = _structure(tmp_path / "candidate")
    right = deepcopy(left)
    right[field] = value
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field",
    (
        "confidence_score",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "plddt",
    ),
)
def test_structure_cannot_discard_any_confidence_component(tmp_path, field):
    left = _structure(tmp_path / "candidate")
    right = deepcopy(left)
    del right["confidence"][field]
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("ptm", True),
        ("complex_plddt", float("nan")),
        ("plddt", [1, float("inf")]),
    ),
)
def test_structure_confidence_obeys_existing_finite_numeric_contract(tmp_path, field, value):
    left = _structure(tmp_path / "candidate")
    right = deepcopy(left)
    right["confidence"][field] = value
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("format", "pdb"),
        ("input_encoding", "yaml"),
        ("source_path", "different/request.yaml"),
        ("document_bytes", 11),
        ("confidence", None),
    ),
)
def test_structure_compares_format_provenance_and_confidence_presence(tmp_path, field, value):
    left = _structure(tmp_path / "candidate")
    right = deepcopy(left)
    right[field] = value
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"


def test_structure_preserves_opaque_bytes_without_inventing_parsing_or_numeric_parity(tmp_path):
    left = _structure(tmp_path / "candidate")
    right = _structure(tmp_path / "reference")
    Path(right["structure_artifact"]).write_bytes(b"a different\x00opaque serialization\xff")
    right["structure_bytes"] = Path(right["structure_artifact"]).stat().st_size
    Path(right["metadata_artifact"]).write_bytes(b"")
    right["metadata_bytes"] = 0
    right["confidence"]["ptm"] = 9.0  # No shared score normalization or range rule.
    right["confidence"]["plddt"] = [1.0, 2.0, 3.0]
    assert _compare("predict_structure", left, right)[0] == "yellow"
    right["confidence"]["plddt"].pop()
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"
    left["confidence"] = right["confidence"] = None
    assert _compare("predict_structure", left, right)[0] == "yellow"
    right["confidence"] = {**_structure(tmp_path / "third")["confidence"], "plddt": []}
    assert _compare("predict_structure", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("refined_hypotheses", True),
        ("refined_hypotheses", 0),
        ("shape", [2, 16]),
        ("shape", [2, 4.0, 4]),
        ("refined_poses", [0.0] * 31),
        ("refined_poses", [float("nan")] * 32),
        ("scores", []),
        ("scores", [0.1]),
        ("scores", [0.1, True]),
        ("best_index", -1),
        ("best_index", 2),
        ("best_index", True),
        ("all_poses_rigid", 1),
        ("refinement_ms", None),
        ("scoring_ms", float("inf")),
        ("crop_queries", None),
    ),
)
def test_pose_checks_complete_matrices_scores_selection_and_flags(tmp_path, field, value):
    left = _poses(tmp_path)
    right = deepcopy(left)
    right[field] = value
    assert _compare("refine_pose", left, right)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    "field,value",
    (
        ("stage", "rendering"),
        ("stage", []),
        ("iteration", -1),
        ("iteration", 0.0),
        ("shape", [2, 16]),
        ("shape", [False, 4, 4]),
        ("poses", [0.0] * 31),
        ("poses", [float("inf")] * 32),
    ),
)
def test_pose_callback_trace_is_complete_and_typed(tmp_path, field, value):
    left = _poses(tmp_path)
    right = deepcopy(left)
    right["crop_queries"][0][field] = value
    assert _compare("refine_pose", left, right)[0] == "contract-mismatch"


def test_pose_compares_callback_order_stage_iteration_and_shape_not_invented_scheduler(tmp_path):
    left = _poses(tmp_path)
    for change in ("order", "stage", "iteration", "shape", "missing_query", "rigid"):
        right = deepcopy(left)
        if change == "order":
            right["crop_queries"].reverse()
        elif change == "stage":
            right["crop_queries"][0]["stage"] = "scoring"
        elif change == "iteration":
            right["crop_queries"][0]["iteration"] = 10
        elif change == "shape":
            right["crop_queries"][0]["shape"] = [1, 4, 4]
            right["crop_queries"][0]["poses"] = right["crop_queries"][0]["poses"][:16]
        elif change == "missing_query":
            right["crop_queries"].pop()
        else:
            right["all_poses_rigid"] = False
        assert _compare("refine_pose", left, right)[0] == "contract-mismatch", change
    # The public callback contract does not require zero-based contiguous
    # iterations, a fixed final scoring step or identical input/output counts.
    left["crop_queries"] = [
        {"stage": "scoring", "iteration": 7, "shape": [1, 4, 4], "poses": [0.0] * 16},
        {"stage": "refinement", "iteration": 20, "shape": [2, 4, 4], "poses": [0.0] * 32},
    ]
    assert _compare("refine_pose", left, deepcopy(left))[0] == "yellow"


def test_pose_shape_is_not_matrix_score_argmax_or_timing_parity(tmp_path):
    left = _poses(tmp_path)
    right = deepcopy(left)
    right["refined_poses"][3] = 2.0
    right["scores"] = [0.9, 0.1]
    right["best_index"] = 0
    right["refinement_ms"], right["scoring_ms"] = 7.0, 8.0
    right["crop_queries"][0]["poses"][3] = 3.0
    assert _compare("refine_pose", left, right)[0] == "yellow"
    left["all_poses_rigid"] = right["all_poses_rigid"] = False
    assert _compare("refine_pose", left, right)[0] == "yellow"


def test_single_pose_may_be_unscored_but_absence_is_not_a_zero_score(tmp_path):
    left = _poses(tmp_path)
    left.update(
        refined_hypotheses=1,
        shape=[1, 4, 4],
        refined_poses=left["refined_poses"][:16],
        scores=[],
        best_index=0,
        crop_queries=[],
    )
    right = deepcopy(left)
    assert _compare("refine_pose", left, right)[0] == "yellow"
    right["scores"] = [0.0]
    assert _compare("refine_pose", left, right)[0] == "contract-mismatch"

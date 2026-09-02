# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np

from tests.e2e.models.foundationpose.e2e_plugins import comparator, runner
from tests.e2e.models.foundationpose.e2e_plugins import report
from tests.e2e_harness.contracts import StageSpec

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _generator_module():
    path = _REPO_ROOT / "scripts" / "generate_e2e_report.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("foundationpose_report_generator_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def _pose(x: float, y: float, z: float) -> list[float]:
    return [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1]


def _result(artifact_dir: Path) -> dict:
    native = artifact_dir / "native"
    native.mkdir(parents=True)
    _write_f32(native / "candidate_poses.f32", _pose(0, 0, 0.5) + _pose(0.1, 0, 0.5))
    _write_f32(native / "trt_refined_poses.f32", _pose(0.01, 0, 0.5) + _pose(0.12, 0, 0.5))
    _write_f32(native / "trt_scores.f32", [0.2, 0.8])
    return {
        "_artifact_dir": str(artifact_dir),
        "case_name": "foundationpose-report-test",
        "status": "pass",
        "oracle_level": "L1_external_reference",
        "case_config": {
            "family": "foundationpose",
            "task_strategy": "pose_hypothesis_refinement",
            "reference_backend": "foundationpose_onnxruntime",
            "inputs": {"num_hypotheses": 2},
        },
        "stages": {
            "synthetic_crop_pose_refinement": {
                "metrics": {
                    "pose_max_abs_error": {"value": 0.000058, "passed": True},
                    "score_max_abs_error": {"value": 0.00765, "passed": True},
                    "tracking_throughput_hz": {"value": 313.5, "passed": True},
                    "tracking_latency_p95_ms": {"value": 3.22, "passed": True},
                }
            }
        },
        "stage_outputs": {
            "trt_synthetic_crop_pose_refinement": {"data": {"best_index": 1}},
            "ref_synthetic_crop_pose_refinement": {"data": {"best_index": 1}},
        },
    }


def test_report_renders_pose_ranking_and_performance_evidence(tmp_path: Path) -> None:
    result = _result(tmp_path / "case")

    rendered = report.render(result, project_dir=_REPO_ROOT)

    assert "FoundationPose refinement and ranking" in rendered
    assert "Pose max |error|" in rendered
    assert "5.8e-05" in rendered
    assert "313.5" in rendered
    assert "1 ★" in rendered
    assert "best-hypothesis agreement: <strong>yes</strong>" in rendered
    assert "0.010000" in rendered


def test_shared_report_recognizes_strategy_and_loads_owned_renderer(tmp_path: Path) -> None:
    result = _result(tmp_path / "case")
    generator = _generator_module()

    rendered = generator.render_model_section(result, _REPO_ROOT)

    assert generator.classify_modality(result) == "numeric"
    assert generator.validate_evidence([result], _REPO_ROOT) == []
    assert rendered.index("FoundationPose refinement and ranking") < rendered.index(
        "Numerical evidence"
    )


def test_strict_evidence_rejects_missing_pose_artifacts_and_reference_index(
    tmp_path: Path,
) -> None:
    generator = _generator_module()
    missing_artifact = _result(tmp_path / "missing-artifact")
    (Path(missing_artifact["_artifact_dir"]) / "native" / "candidate_poses.f32").unlink()

    issues = generator.validate_evidence([missing_artifact], _REPO_ROOT)

    assert any("candidate_poses.f32" in issue for issue in issues)

    missing_reference = _result(tmp_path / "missing-reference")
    missing_reference["stage_outputs"]["ref_synthetic_crop_pose_refinement"]["data"].pop(
        "best_index"
    )

    issues = generator.validate_evidence([missing_reference], _REPO_ROOT)

    assert any("reference best_index" in issue for issue in issues)


def test_report_fails_closed_for_missing_and_escaping_artifacts(tmp_path: Path) -> None:
    missing = report.render({"_artifact_dir": str(tmp_path / "missing")}, project_dir=_REPO_ROOT)
    assert "FoundationPose evidence unavailable" in missing

    result = _result(tmp_path / "case")
    native = Path(result["_artifact_dir"]) / "native"
    outside = tmp_path / "outside.f32"
    outside.write_bytes((native / "trt_scores.f32").read_bytes())
    (native / "trt_scores.f32").unlink()
    (native / "trt_scores.f32").symlink_to(outside)

    escaped = report.render(result, project_dir=_REPO_ROOT)

    assert "escapes the artifact directory" in escaped
    assert "FoundationPose refinement and ranking" not in escaped


def test_report_escapes_status_and_rejects_nonfinite_values(tmp_path: Path) -> None:
    result = _result(tmp_path / "case")
    result["status"] = '<script>alert("status")</script>'
    rendered = report.render(result, project_dir=_REPO_ROOT)
    assert '<script>alert("status")</script>' not in rendered
    assert "&lt;SCRIPT&gt;" in rendered

    native = Path(result["_artifact_dir"]) / "native"
    _write_f32(native / "trt_scores.f32", [0.2, float("nan")])
    rejected = report.render(result, project_dir=_REPO_ROOT)
    assert "contains non-finite values" in rejected


def test_ranking_agreement_requires_the_complete_order() -> None:
    reference_scores = np.asarray([0.9, 0.6, 0.2], dtype=np.float32)

    assert comparator._ranking_agreement(reference_scores, reference_scores) == 1.0
    assert (
        comparator._ranking_agreement(
            np.asarray([0.9, 0.2, 0.6], dtype=np.float32), reference_scores
        )
        == 0.0
    )


def test_native_runner_forwards_manifest_pose_parameters(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def run(command, **_kwargs):
        captured.extend(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        _write_f32(output_dir / "trt_refined_poses.f32", _pose(0, 0, 0) * 2)
        _write_f32(output_dir / "trt_scores.f32", [0.1, 0.2])
        return SimpleNamespace(
            returncode=0,
            stdout='{"num_hypotheses": 2, "best_index": 1}',
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    case = SimpleNamespace(
        name="foundationpose-runner-test",
        bundle="fixture.bundle",
        inputs={
            "num_hypotheses": 2,
            "refinement_iterations": 4,
            "mesh_diameter": 0.25,
        },
    )
    context = SimpleNamespace(
        artifacts_dir=str(tmp_path),
        binary_path=str(tmp_path / "build" / "trtmc"),
        engine_dir=str(tmp_path / "engines"),
        model_plugin_dir="",
        ld_library_path="",
    )

    output = runner.FoundationPoseRunner().run_stage(
        case, StageSpec(name="synthetic_crop_pose_refinement"), context
    )

    assert captured[captured.index("--num-hypotheses") + 1] == "2"
    assert captured[captured.index("--refinement-iterations") + 1] == "4"
    assert captured[captured.index("--mesh-diameter") + 1] == "0.25"
    assert output.data["refined_poses"].shape == (2, 4, 4)

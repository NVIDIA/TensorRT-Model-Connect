# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct native and ONNX Runtime qualification for FoundationPose refinement."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from .official_reference import run as run_reference


FAMILY = "foundationpose"
TASK = "pose_hypothesis_refinement"
TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = TEST_ROOT / "manifests"
THRESHOLD_ROOT = TEST_ROOT / "thresholds"


def _case_index() -> dict[str, tuple[dict, dict]]:
    result = {}
    for path in sorted(MANIFEST_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["family"] == FAMILY
        assert manifest["task"] == TASK
        for case in manifest["testcases"]:
            name = str(case["name"])
            assert name not in result
            result[name] = (manifest, case)
    return result


CASES = _case_index()


def _selected_cases(config) -> tuple[list[str], bool]:
    model_filters = set()
    for raw in config.getoption("--e2e-model") or []:
        model_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        model_filters.update(
            line.strip()
            for line in Path(models_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    testcase_filters = set()
    for raw in config.getoption("--e2e-testcase") or []:
        testcase_filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_filters and not testcase_filters:
        return sorted(CASES), False
    selected = []
    for name, (manifest, _) in CASES.items():
        model_match = (
            not model_filters
            or FAMILY in model_filters
            or name in model_filters
            or manifest["name"] in model_filters
        )
        if model_match and (not testcase_filters or name in testcase_filters):
            selected.append(name)
    return sorted(selected), True


def pytest_generate_tests(metafunc) -> None:
    if "case_name" not in metafunc.fixturenames:
        return
    names, enabled = _selected_cases(metafunc.config)
    parameters = names
    if not enabled:
        parameters = [
            pytest.param(
                name,
                marks=pytest.mark.skip(
                    reason="direct E2E requires one of the three explicit E2E selectors"
                ),
            )
            for name in names
        ]
    metafunc.parametrize("case_name", parameters, ids=names)


def _required_path(value: str | None, label: str) -> Path:
    assert value, f"selected {FAMILY} E2E requires {label}"
    path = Path(value)
    assert path.exists(), f"selected {FAMILY} E2E {label} does not exist: {path}"
    return path


def _model_dir(manifest: dict) -> Path:
    model_dir = _required_path(
        os.environ.get("TRTMC_FOUNDATIONPOSE_MODEL_DIR"),
        "TRTMC_FOUNDATIONPOSE_MODEL_DIR",
    )
    missing = [
        dependency["path"]
        for dependency in manifest["external_files"]
        if not (model_dir / dependency["path"]).is_file()
    ]
    assert not missing, f"FoundationPose model directory is missing declared files: {missing}"
    return model_dir


def _runtime() -> tuple[Path, Path]:
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    qualification = native_build / "families/foundationpose/foundationpose_qualification"
    assert qualification.is_file(), (
        f"missing FoundationPose qualification executable: {qualification}"
    )
    assert (runtime_root / "libtrtmc_core.so").is_file()
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_foundationpose.so").is_file()
    import torch

    assert torch.cuda.is_available(), "selected FoundationPose E2E requires CUDA"
    return qualification, runtime_root


def _build(model_dir: Path, bundle: Path, manifest: dict) -> None:
    build(
        BuildRequest(
            model_dir=model_dir,
            output_path=bundle,
            family=FAMILY,
            task=manifest["task"],
            precision=manifest["precision"],
            max_sequence_length=manifest.get("max_sequence_length"),
            max_batch_size=int(manifest.get("max_batch_size", 1)),
            tensor_parallel_size=int(manifest["tensor_parallel_size"]),
        )
    )


def _native(
    qualification: Path,
    runtime_root: Path,
    bundle: Path,
    case: dict,
    output_dir: Path,
) -> dict[str, object]:
    command = [
        str(qualification),
        "--qualify",
        "--bundle",
        str(bundle),
        "--output-dir",
        str(output_dir),
        "--runtime-root",
        str(runtime_root),
        "--benchmark",
        "20",
        "--warmup",
        "3",
        "--num-hypotheses",
        str(case["num_hypotheses"]),
        "--refinement-iterations",
        str(case["refinement_iterations"]),
        "--mesh-diameter",
        str(case["mesh_diameter"]),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (str(runtime_root), str(qualification.parents[2]), env.get("LD_LIBRARY_PATH"))
        if value
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(case.get("runtime_timeout_s", 600)),
    )
    summaries = []
    for line in completed.stdout.splitlines():
        try:
            summaries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert len(summaries) == 1, f"qualification returned no unique summary: {completed.stdout}"
    count = int(case["num_hypotheses"])
    poses = np.fromfile(output_dir / "trt_refined_poses.f32", dtype="<f4").reshape(count, 4, 4)
    scores = np.fromfile(output_dir / "trt_scores.f32", dtype="<f4").reshape(count)
    return {"refined_poses": poses, "scores": scores, "summary": summaries[0]}


def _rigid_fraction(poses: np.ndarray) -> float:
    valid = []
    for pose in poses:
        rotation = pose[:3, :3].astype(np.float64)
        valid.append(
            np.isfinite(pose).all()
            and np.max(np.abs(rotation @ rotation.T - np.eye(3))) <= 2.0e-3
            and abs(np.linalg.det(rotation) - 1.0) <= 6.0e-3
            and np.max(np.abs(pose[3] - np.asarray([0, 0, 0, 1], dtype=np.float32))) <= 2.0e-3
        )
    return float(np.mean(valid))


def _assert_parity(actual: dict, expected: dict, limits: dict) -> None:
    poses = np.asarray(actual["refined_poses"], dtype=np.float32)
    reference_poses = np.asarray(expected["refined_poses"], dtype=np.float32)
    scores = np.asarray(actual["scores"], dtype=np.float32)
    reference_scores = np.asarray(expected["scores"], dtype=np.float32)
    summary = actual["summary"]
    count = int(summary["num_hypotheses"])
    assert 1 <= count <= 252
    assert poses.shape == reference_poses.shape == (count, 4, 4)
    assert scores.shape == reference_scores.shape == (count,)
    assert np.isfinite(poses).all() and np.isfinite(reference_poses).all()
    assert np.isfinite(scores).all() and np.isfinite(reference_scores).all()
    assert float(np.max(np.abs(poses - reference_poses))) <= float(limits["pose_max_abs_error"])
    assert float(np.max(np.abs(scores - reference_scores))) <= float(limits["score_max_abs_error"])
    actual_order = np.argsort(-scores, kind="stable")
    reference_order = np.argsort(-reference_scores, kind="stable")
    ranking = float(np.array_equal(actual_order, reference_order))
    assert ranking >= float(limits["ranking_agreement"])
    assert _rigid_fraction(poses) >= float(limits["rigid_pose_fraction"])
    assert summary["all_poses_rigid"] is True
    assert float(summary["tracking_throughput_hz"]) >= float(limits["tracking_throughput_hz"])
    assert float(summary["tracking_latency_p95_ms"]) <= float(limits["tracking_latency_p95_ms"])
    assert float(summary["tracking_jitter_ms"]) <= float(limits["tracking_jitter_ms"])
    assert float(summary["startup_ms"]) <= float(limits["startup_ms"])
    assert float(summary["gpu_memory_delta_mib"]) <= float(limits["gpu_memory_delta_mib"])


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    model_dir = _model_dir(manifest)
    qualification, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    _build(model_dir, bundle, manifest)
    native_dir = tmp_path / "native"
    actual = _native(qualification, runtime_root, bundle, case, native_dir)
    expected = run_reference(
        model_dir,
        native_dir,
        int(case["num_hypotheses"]),
        int(case["refinement_iterations"]),
        float(case["mesh_diameter"]),
    )
    _assert_parity(actual, expected, _thresholds(case_name))

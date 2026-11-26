# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and upstream-PyTorch-reference E2E for Uni3D."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from .official_reference import INPUT_DIM, export_onnx
from .official_reference import run as run_reference


FAMILY = "uni3d"
TASK = "points_to_embedding"
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
    model_dir = _required_path(os.environ.get("TRTMC_UNI3D_MODEL_DIR"), "TRTMC_UNI3D_MODEL_DIR")
    required = [dependency["path"] for dependency in manifest["external_files"]]
    missing = [name for name in required if not (model_dir / name).is_file()]
    assert not missing, f"Uni3D model directory is missing declared files: {missing}"
    return model_dir


def _runtime() -> tuple[Path, Path]:
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    qualification = native_build / "families/uni3d/uni3d_qualification"
    assert qualification.is_file(), f"missing Uni3D qualification executable: {qualification}"
    assert (runtime_root / "libtrtmc_core.so").is_file()
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_uni3d.so").is_file()
    import torch

    assert torch.cuda.is_available(), "selected Uni3D E2E requires CUDA"
    return qualification, runtime_root


def synthetic_cloud(seed: int, count: int) -> np.ndarray:
    """A colored mug (open cylinder plus a ring handle), normalized like upstream pc_norm."""
    rng = np.random.default_rng(seed)
    body = count * 4 // 5
    angle = rng.uniform(0, 2 * np.pi, body)
    height = rng.uniform(-1.0, 1.0, body)
    cylinder = np.stack([0.6 * np.cos(angle), 0.6 * np.sin(angle), height], axis=1)
    u, v = rng.uniform(0, 2 * np.pi, (2, count - body))
    ring = np.stack([0.6 + (0.35 + 0.08 * np.cos(v)) * np.cos(u), 0.08 * np.sin(v),
                     (0.35 + 0.08 * np.cos(v)) * np.sin(u)], axis=1)
    xyz = np.concatenate([cylinder, ring]) + rng.normal(scale=0.005, size=(count, 3))
    xyz -= xyz.mean(axis=0)
    xyz /= np.linalg.norm(xyz, axis=1).max()
    rgb = np.clip(0.5 + 0.4 * xyz[:, [2, 0, 1]], 0.0, 1.0)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


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


def _native(qualification: Path, runtime_root: Path, bundle: Path, points: Path,
            output_dir: Path, case: dict) -> dict[str, object]:
    command = [
        str(qualification), "--qualify",
        "--bundle", str(bundle),
        "--output-dir", str(output_dir),
        "--runtime-root", str(runtime_root),
        "--points", str(points),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", "")) if value
    )
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env,
        timeout=int(case.get("runtime_timeout_s", 600)),
    )
    record_evidence("commands", {"argv": getattr(completed, "args", None)})
    record_evidence("native", {"stdout": completed.stdout, "stderr": completed.stderr})
    summaries = []
    for line in completed.stdout.splitlines():
        try:
            summaries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert len(summaries) == 1, f"qualification returned no unique summary: {completed.stdout}"
    embedding = np.fromfile(output_dir / "trt_embedding.f32", dtype=np.float32)
    return {"embedding": embedding, "summary": summaries[0]}


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def _assert_parity(actual: dict, expected: np.ndarray, limits: dict) -> None:
    summary = actual["summary"]
    assert summary == {"count": expected.size, "embedding_space": "open_clip/EVA02-E-14-plus",
                       "pooling": "class_token", "normalization": "l2"}
    embedding = np.asarray(actual["embedding"], dtype=np.float64)
    assert embedding.shape == expected.shape
    assert abs(float(np.linalg.norm(embedding)) - 1.0) <= 1e-4
    cosine = float(embedding @ expected / np.linalg.norm(expected))
    record_evidence("compare", {"cosine": cosine})
    assert cosine >= float(limits["min_cosine"]), cosine


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    qualification, runtime_root = _runtime()
    points = synthetic_cloud(int(case["seed"]), int(case["num_points"]))
    assert points.shape == (int(case["num_points"]), INPUT_DIM)
    points_path = tmp_path / "points.f32"
    points.tofile(points_path)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        # Only the upstream checkpoint is published, so the builder's ONNX input is exported here.
        export_onnx(model_dir, tmp_path / "model", manifest["precision"])
        _build(tmp_path / "model", bundle, manifest)
    with evidence_stage("native"):
        actual = _native(qualification, runtime_root, bundle, points_path, tmp_path / "native", case)
    with evidence_stage("reference"):
        expected = run_reference(model_dir, points).astype(np.float64)
    with evidence_stage("compare"):
        _assert_parity(actual, expected, record_evidence("thresholds", _thresholds(case_name)))

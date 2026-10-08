# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct build, native-runtime, and upstream-PyTorch-reference E2E for PointNet."""

from __future__ import annotations

from tools.e2e_evidence import evidence_stage, record_evidence

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from tensorrt_model_connect import BuildRequest, build

from .official_reference import run as run_reference


FAMILY = "pointnet"
TASK = "points_to_semantic_segmentation"
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
        os.environ.get("TRTMC_POINTNET_MODEL_DIR"), "TRTMC_POINTNET_MODEL_DIR"
    )
    required = [dependency["path"] for dependency in manifest["external_files"]]
    required.append("best_model.pth")
    missing = [name for name in required if not (model_dir / name).is_file()]
    assert not missing, f"PointNet model directory is missing declared files: {missing}"
    return model_dir


def _runtime() -> tuple[Path, Path]:
    runtime_root = _required_path(os.environ.get("TRTMC_RUNTIME_ROOT"), "TRTMC_RUNTIME_ROOT")
    native_build = _required_path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR"), "TRTMC_NATIVE_BUILD_DIR"
    )
    qualification = native_build / "families/pointnet/pointnet_qualification"
    assert qualification.is_file(), f"missing PointNet qualification executable: {qualification}"
    assert (runtime_root / "libtrtmc_core.so").is_file()
    assert (runtime_root / "libtrtmc_backend_trt.so").is_file()
    assert (runtime_root / "libtrtmc_model_pointnet.so").is_file()
    import torch

    assert torch.cuda.is_available(), "selected PointNet E2E requires CUDA"
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


def _asset(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = TEST_ROOT / path
    assert path.is_file(), f"selected {FAMILY} E2E asset does not exist: {path}"
    record_evidence("inputs", {"asset": path})
    return path


def _native(
    qualification: Path, runtime_root: Path, bundle: Path, case: dict, output_dir: Path
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
        "--points",
        str(_asset(case["test_points"])),
        "--input-dim",
        "9",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (str(runtime_root), env.get("LD_LIBRARY_PATH", ""))
        if value
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
    count = int(case["num_points"])
    labels = np.fromfile(output_dir / "trt_labels.i32", dtype=np.int32)
    logits = np.fromfile(output_dir / "trt_logits.f32", dtype=np.float32).reshape(count, 13)
    assert labels.shape == (count,)
    return {"labels": labels, "logits": logits, "summary": summaries[0]}


def _assert_parity(actual: dict, expected: dict, limits: dict) -> None:
    actual_labels = np.asarray(actual["labels"])
    expected_labels = np.asarray(expected["labels"])
    assert actual_labels.shape == expected_labels.shape
    agreement = float(np.mean(actual_labels == expected_labels))
    assert agreement >= float(limits["argmax_agreement"])
    actual_logits = np.asarray(actual["logits"], dtype=np.float32)
    expected_logits = np.asarray(expected["logits"], dtype=np.float32)
    assert actual_logits.shape == expected_logits.shape
    error = np.abs(actual_logits - expected_logits)
    assert float(np.mean(error)) <= float(limits["mean_abs_logits_error"])
    assert float(np.max(error)) <= float(limits["max_abs_logits_error"])


def _thresholds(case_name: str) -> dict:
    path = THRESHOLD_ROOT / f"{case_name}.json"
    assert path.is_file(), f"selected {FAMILY} E2E requires exact thresholds: {path}"
    return json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]


def test_official_checkpoint_e2e(case_name: str, tmp_path: Path) -> None:
    manifest, case = CASES[case_name]
    record_evidence("inputs", {"manifest": manifest, "case": case})
    model_dir = _model_dir(manifest)
    qualification, runtime_root = _runtime()
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        _build(model_dir, bundle, manifest)
    native_dir = tmp_path / "native"
    with evidence_stage("native"):
        actual = _native(qualification, runtime_root, bundle, case, native_dir)
    record_evidence("native", actual)
    with evidence_stage("reference"):
        expected = run_reference(model_dir, _asset(case["test_points"]), int(case["num_points"]))
    record_evidence("reference", expected)
    with evidence_stage("compare"):
        _assert_parity(actual, expected, record_evidence("thresholds", _thresholds(case_name)))

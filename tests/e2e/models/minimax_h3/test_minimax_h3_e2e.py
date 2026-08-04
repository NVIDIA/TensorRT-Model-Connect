# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint and static contract checks for MiniMax-H3."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.e2e.models.minimax_h3.e2e_plugins.comparator import comparator
from tensorrt_model_connect.families.minimax_h3.provenance import file_record
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_model_manifest
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
)


_MODEL_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODEL_DIR.parents[3]
_MANIFEST_PATH = _MODEL_DIR / "manifests" / "minimax-h3-768p.json"
_RUNNER_PATH = _MODEL_DIR / "runner.py"
_SPEC = importlib.util.spec_from_file_location(
    f"{_MODEL_DIR.name}_e2e_runner",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_runner)


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        metafunc.parametrize("case_name", _runner.model_case_names(metafunc.config))


def test_minimax_h3_manifest_is_truthful_cp4_contract() -> None:
    model = load_model_manifest(_MANIFEST_PATH)
    assert model.name == "minimax-h3-768p"
    assert model.family == "minimax_h3"
    assert len(model.testcases) == 1

    case = model.testcases[0]
    assert case.runtime_strategy == "diffusion_minimax_h3"
    assert case.task_strategy == "diffusion_media_generation"
    assert case.metadata["ci_tier"] == "multi_device"
    assert case.metadata["build_args"]["parallel"] == {
        "mode": "context_parallel",
        "cp_size": 4,
    }
    assert any(
        requirement.kind == "gpu_count_min" and requirement.args.get("count") == 4
        for requirement in case.preflight
    )
    assert case.inputs["video_num_frames"] == 124
    assert case.inputs["video_height"] == 768
    assert case.inputs["video_width"] == 1344
    assert case.inputs["num_inference_steps"] == 50
    assert case.threshold_overrides["minimum_psnr_db"] == 40.0
    assert case.threshold_overrides["maximum_mean_absolute_error"] == pytest.approx(1.0 / 255.0)


def test_minimax_h3_plugins_cover_native_reference_and_comparison() -> None:
    activate_model_plugins(_MODEL_DIR)
    assert get_runner("diffusion_media_generation") is not None
    assert get_reference("hf_diffusers") is not None
    assert get_comparator("diffusion_media_generation") is not None


def test_minimax_h3_comparator_gates_decoded_pixel_drift(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    reference = np.zeros((2, 2, 2, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0, 0, 0] = 1.0
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)
    revision = "1" * 40
    result = comparator.compare(
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "frames_path": str(candidate_path),
                "source_revision": revision,
            },
        ),
        StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": 0,
                "frames_path": str(reference_path),
                "source_revision": revision,
            },
        ),
        ThresholdProfile(
            task_strategy="diffusion_media_generation",
            metrics={
                "exact_num_frames": 2,
                "exact_video_height": 2,
                "exact_video_width": 2,
                "minimum_psnr_db": 40.0,
                "maximum_mean_absolute_error": 1.0 / 255.0,
            },
        ),
        StageSpec(name="end_to_end"),
    )
    assert result.status == "failed"
    assert not result.metrics["psnr_db"].passed
    assert not result.metrics["mean_absolute_error"].passed


def test_compare_video_cli_binds_threshold_schema_and_run_receipts(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    frames = np.zeros((1, 1, 1, 3), dtype=np.float32)
    np.save(reference_path, frames)
    np.save(candidate_path, frames)
    revision = "1" * 40
    workload = {
        "prompt": "test",
        "seed": 0,
        "height": 1,
        "width": 1,
        "num_frames": 1,
        "num_inference_steps": 1,
    }
    reference_receipt_path = tmp_path / "reference.json"
    candidate_receipt_path = tmp_path / "candidate.json"
    reference_receipt_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "source_revision": revision,
                "checkpoint_inventory_sha256": "a" * 64,
                "workload": workload,
                "frames": file_record(reference_path),
            }
        )
    )
    candidate_receipt = {
        "status": "passed",
        "source_revision": revision,
        "checkpoint_inventory_sha256": "a" * 64,
        "request": workload,
        "frames": file_record(candidate_path),
    }
    candidate_receipt_path.write_text(json.dumps(candidate_receipt))
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "threshold_overrides": {
                    "exact_num_frames": 1,
                    "exact_video_height": 1,
                    "exact_video_width": 1,
                    "minimum_psnr_db": 40.0,
                    "maximum_mean_absolute_error": 1.0 / 255.0,
                }
            }
        )
    )
    output_path = tmp_path / "comparison.json"
    command = [
        sys.executable,
        str(_MODEL_DIR / "compare_video.py"),
        str(reference_path),
        str(candidate_path),
        "--reference-receipt",
        str(reference_receipt_path),
        "--candidate-receipt",
        str(candidate_receipt_path),
        "--thresholds",
        str(thresholds_path),
        "--output",
        str(output_path),
        "--source-revision",
        revision,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_PROJECT_DIR / "python")
    result = subprocess.run(
        command, cwd=_PROJECT_DIR, env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text())["passed"] is True

    candidate_receipt["source_revision"] = "2" * 40
    candidate_receipt_path.write_text(json.dumps(candidate_receipt))
    result = subprocess.run(
        command, cwd=_PROJECT_DIR, env=environment, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "different source revision" in result.stderr


def test_model_e2e(case_name: str, request) -> None:
    _runner.run_model_e2e(case_name, request)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL accuracy and validation-gate contracts."""

from __future__ import annotations

import json
from pathlib import Path

from tools.validation import engine as validation_engine


def test_qwen25_validation_uses_aligned_fp32_precision() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "qwen25vl-3b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert {
        testcase["reference_precision"] for testcase in manifest["testcases"]
    } == {"fp32"}


def test_qwen_vl_rejects_observed_point_eight_sample_acceptance() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "vlm_mmmu_pro_vision_fixed_mcq",
    )
    gates = suite["gates"]
    sample_acceptance = suite["sample_acceptance"]
    assert gates == {"max_accuracy_drop_from_hf": 0.02}
    assert sample_acceptance == {
        "min_pass_rate": 0.95,
        "min_allowed_failures": 0,
    }

    observed_result = {
        "status": "passed",
        "sample_count": 20,
        "valid_count": 20,
        "passed_count": 16,
        "prediction_agreement_rate": 0.8,
    }
    validation_engine._apply_sample_acceptance(
        observed_result,
        sample_acceptance,
    )

    assert observed_result["status"] == "failed"
    assert observed_result["error_type"] == "BenchmarkGateError"
    assert observed_result["gate_failures"] == [
        {
            "gate": "sample_acceptance",
            "metric": "failed_samples",
            "actual": 4,
            "required": 1,
        }
    ]

    boundary_result = {
        "status": "passed",
        "sample_count": 20,
        "valid_count": 20,
        "passed_count": 19,
        "prediction_agreement_rate": 0.95,
    }
    validation_engine._apply_sample_acceptance(
        boundary_result,
        sample_acceptance,
    )
    assert boundary_result["status"] == "passed"
    assert boundary_result.get("gate_failures", []) == []

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


def test_qwen_vl_rejects_observed_point_eight_prediction_agreement() -> None:
    suite = validation_engine.suite_by_id(
        validation_engine.load_suites(),
        "vlm_mmmu_pro_vision_fixed_mcq",
    )
    gates = suite["gates"]
    assert gates["min_prediction_agreement"] == 0.95

    observed_summary = {
        "hf": {"overall_accuracy": 0.0},
        "bundle": {"overall_accuracy": 0.0},
        "prediction_agreement_rate": 0.8,
        "correctness_agreement_rate": 1.0,
    }
    observed_result = validation_engine.prediction_agreement_gate_result(
        observed_summary,
        gates,
    )

    assert observed_result["status"] == "failed"
    assert observed_result["error_type"] == "BenchmarkGateError"
    assert observed_result["gate_failures"] == [
        {
            "gate": "min_prediction_agreement",
            "metric": "prediction_agreement_rate",
            "actual": 0.8,
            "required": 0.95,
        }
    ]

    boundary_summary = {
        **observed_summary,
        "prediction_agreement_rate": 0.95,
    }
    boundary_result = validation_engine.prediction_agreement_gate_result(
        boundary_summary,
        gates,
    )
    assert boundary_result["status"] == "passed"
    assert boundary_result["gate_failures"] == []

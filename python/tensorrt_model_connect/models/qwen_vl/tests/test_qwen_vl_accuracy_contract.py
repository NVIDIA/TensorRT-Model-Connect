# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL accuracy and validation-gate contracts."""

from __future__ import annotations

import json
from pathlib import Path

from tools.validation import catalog as validation_catalog
from tools.validation.gate_policy import evaluate_sample_acceptance


def test_qwen25_validation_uses_aligned_fp32_precision() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "qwen25vl-3b.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["precision"] == "fp32"
    assert {
        testcase["reference_precision"] for testcase in manifest["testcases"]
    } == {"fp32"}


def test_qwen_vl_rejects_point_eight_sample_acceptance() -> None:
    suite = validation_catalog.suite_by_id(
        validation_catalog.load_suites(
            _owners={"qwen_vl"},
            _require_all_suites=False,
        ),
        "vlm_mmmu_pro_vision_fixed_mcq",
    )
    assert suite["gates"] == {"max_accuracy_drop_from_hf": 0.02}
    assert suite["sample_acceptance"] == {
        "min_pass_rate": 0.95,
        "min_allowed_failures": 0,
    }
    sample_count = int(suite["sample_limit"])
    assert sample_count == 5

    observed = evaluate_sample_acceptance(
        policy=suite["sample_acceptance"],
        sample_count=sample_count,
        passed_count=4,
        expected_count=sample_count,
    )
    boundary = evaluate_sample_acceptance(
        policy=suite["sample_acceptance"],
        sample_count=sample_count,
        passed_count=5,
        expected_count=sample_count,
    )

    assert observed["passed_count"] / observed["sample_count"] == 0.8
    assert observed["allowed_failures"] == 0
    assert observed["verdict"] == "fail"
    assert boundary["passed_count"] / boundary["sample_count"] == 1.0
    assert boundary["allowed_failures"] == 0
    assert boundary["verdict"] == "pass"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenPI ownership tests for its HTML trajectory visualization."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _report_module():
    scripts = str(Path(__file__).resolve().parents[4] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("generate_e2e_report")


def test_openpi_actions_render_as_a_tensor_rt_reference_svg() -> None:
    report = _report_module()
    result = {
        "case_name": "pi05-droid",
        "status": "pass",
        "oracle_level": "L1_external_reference",
        "case_config": {
            "family": "openpi",
            "task_strategy": "robot_action_generation",
            "reference_backend": "upstream_replay",
            "inputs": {"horizon": 2},
        },
        "stages": {},
        "stage_outputs": {
            "trt_actions": {"data": {"output_field": [[0.1, -0.2], [0.3, -0.4]]}},
            "ref_actions": {"data": {"output_field": [[0.11, -0.19], [0.31, -0.39]]}},
        },
        "repro_commands": {},
        "timing": {},
    }

    rendered = report.render_model_section(result, project_dir=None)

    assert report.classify_modality(result) == "neural_operator"
    assert "Output Series Comparison" in rendered
    assert 'aria-label="TRT and reference numeric output plot"' in rendered
    assert "TRT / Base" in rendered
    assert "Reference" in rendered
    assert report.validate_evidence([result], project_dir=None) == []

    result["stage_outputs"]["trt_actions"]["data"].pop("output_field")
    issues = report.validate_evidence([result], project_dir=None)
    assert len(issues) == 1
    assert "missing numeric TRT/base field output" in issues[0]

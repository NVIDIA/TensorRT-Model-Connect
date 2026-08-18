# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint for the Fast Foundation Stereo family."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.validation import catalog as validation_catalog

_RUNNER_PATH = Path(__file__).with_name("runner.py")
_SPEC = importlib.util.spec_from_file_location(
    f"{Path(__file__).resolve().parent.name}_e2e_runner",
    _RUNNER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_runner)


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        case_names = _runner.model_case_names(metafunc.config)
        if not case_names:
            pytest.skip("No model manifests selected", allow_module_level=True)
        metafunc.parametrize("case_name", case_names)


def test_model_e2e(case_name: str, request) -> None:
    _runner.run_model_e2e(case_name, request)


def test_validation_contract_uses_model_plugin_parity() -> None:
    suite = next(
        item
        for item in validation_catalog.load_suites()
        if item["id"] == "fast_foundation_stereo_synthetic_parity"
    )
    requests = json.loads(
        (Path(__file__).with_name("validation") / "fast-foundation-stereo.json").read_text(
            encoding="utf-8"
        )
    )["requests"]
    assert (suite["scoring"], suite["gates"], requests) == (
        {"scorer": "model_plugin_parity"},
        {"min_sample_pass_rate": 1.0},
        json.loads(
            '[{"sample_id":"fast-foundation-stereo-shift-12",'
            '"testcase":"fast-foundation-stereo","stage":"full_inference",'
            '"category":"synthetic-rectified-stereo","inputs":{"pixel_shift":12}}]'
        ),
    )

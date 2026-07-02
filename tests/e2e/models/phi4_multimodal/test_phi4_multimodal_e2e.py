# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned E2E entrypoint for the phi4_multimodal family."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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

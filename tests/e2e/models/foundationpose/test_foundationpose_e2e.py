# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("foundationpose_e2e_runner", Path(__file__).with_name("runner.py"))
assert _SPEC is not None and _SPEC.loader is not None
_runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_runner)


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        cases = _runner.model_case_names(metafunc.config)
        if not cases:
            pytest.skip("No FoundationPose manifests selected", allow_module_level=True)
        metafunc.parametrize("case_name", cases)


def test_model_e2e(case_name: str, request) -> None:
    _runner.run_model_e2e(case_name, request)

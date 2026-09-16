# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .catalog import QualificationError, discover, select
from .runtime import context_from_pytest


REPOSITORY = Path(__file__).resolve().parents[3]


def pytest_generate_tests(metafunc) -> None:
    fixture_names = {"accuracy_case", "performance_case"} & set(metafunc.fixturenames)
    if not fixture_names:
        return
    try:
        all_cases = discover(REPOSITORY)
        requested = metafunc.config.getoption("--qualification-model")
        selected = select(all_cases, requested)
    except QualificationError as error:
        raise pytest.UsageError(str(error)) from error
    enabled = os.environ.get("TRTMC_QUALIFICATION") == "1" or bool(requested)
    fixture = fixture_names.pop()
    kind = fixture.removesuffix("_case")
    cases = [case for case in selected if case.kind == kind]
    parameters = []
    for case in cases:
        marks = [
            pytest.mark.qualification,
            pytest.mark.gpu,
            pytest.mark.trt,
            getattr(pytest.mark, kind),
        ]
        if not enabled:
            marks.append(
                pytest.mark.skip(
                    reason=(
                        "qualification requires TRTMC_QUALIFICATION=1 or "
                        "--qualification-model"
                    )
                )
            )
        parameters.append(pytest.param(case, id=case.id, marks=marks))
    metafunc.parametrize(fixture, parameters)


@pytest.fixture(scope="session")
def qualification_context(pytestconfig):
    try:
        return context_from_pytest(pytestconfig, REPOSITORY)
    except QualificationError as error:
        pytest.fail(str(error), pytrace=False)

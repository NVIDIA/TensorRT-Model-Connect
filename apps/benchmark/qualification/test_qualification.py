# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .accuracy import run_accuracy
from .catalog import QualificationError
from .performance import run_performance


def test_accuracy(accuracy_case, qualification_context) -> None:
    try:
        result = run_accuracy(accuracy_case, qualification_context)
    except QualificationError as error:
        raise AssertionError(str(error)) from error
    assert result["status"] == "passed", result


def test_performance(performance_case, qualification_context) -> None:
    try:
        result = run_performance(performance_case, qualification_context)
    except QualificationError as error:
        raise AssertionError(str(error)) from error
    assert result["status"] == "passed", result

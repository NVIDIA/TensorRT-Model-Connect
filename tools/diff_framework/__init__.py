# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified diff test framework — auto-discovers checks and exports public API."""

from .protocol import DiffResult, TestContext, DiffTest
from .registry import register, get_all_tests, get_tests_for_strategy, get_test_by_name
from .runner import (
    detect_runtime_strategy,
    detect_runtime_strategy_from_bundle,
    list_tests,
    run_tests,
)

# Trigger auto-discovery of check modules
from . import checks as _checks  # noqa: F401

__all__ = [
    "DiffResult", "TestContext", "DiffTest",
    "register", "get_all_tests", "get_tests_for_strategy", "get_test_by_name",
    "detect_runtime_strategy", "detect_runtime_strategy_from_bundle",
    "list_tests", "run_tests",
]

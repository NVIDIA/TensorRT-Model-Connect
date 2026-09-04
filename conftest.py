# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source paths and the three selectors shared by family-owned tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys


if os.environ.get("TRTMC_TEST_INSTALLED_WHEEL") != "1":
    repository = Path(__file__).resolve().parent
    for source in (repository / "core/builder", repository / "apps/benchmark"):
        sys.path.insert(0, str(source))


def pytest_addoption(parser):
    parser.addoption(
        "--e2e-model",
        action="append",
        default=[],
        help="Select a family or manifest; repeat or comma-separate values",
    )
    parser.addoption(
        "--e2e-testcase",
        action="append",
        default=[],
        help="Select an exact family-owned testcase",
    )
    parser.addoption(
        "--e2e-models-file",
        default=None,
        help="Select names listed one per line in a file",
    )

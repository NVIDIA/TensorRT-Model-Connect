# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source paths and the three selectors shared by family-owned tests."""

from __future__ import annotations

import os
from pathlib import Path
import sys

pytest_plugins = ("tools.e2e_evidence",)


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
    group = parser.getgroup("qualification")
    group.addoption(
        "--qualification-model",
        action="append",
        default=[],
        help="Select an exact family, model, or qualification case; repeat or comma-separate",
    )
    group.addoption("--qualification-data-root", help="Root containing staged Accuracy datasets")
    group.addoption(
        "--qualification-artifacts",
        default="artifacts/qualification",
        help="Directory for qualification evidence and reports",
    )
    group.addoption("--qualification-env-root", help="Cache for family reference environments")
    group.addoption("--qualification-bundle-cache", help="Managed TRTMC bundle cache")
    group.addoption(
        "--qualification-bundle-root", action="append", default=[], help="Existing bundle root"
    )
    group.addoption("--qualification-runtime-root", help="Installed TRTMC runtime library directory")
    group.addoption("--qualification-trtmc-bench", help="Installed trtmc-bench executable")
    group.addoption("--qualification-trtmc", help="Installed trtmc executable")
    group.addoption("--qualification-worker", help="Installed trtmc benchmark worker")
    group.addoption(
        "--qualification-reference-python",
        action="append",
        default=[],
        metavar="MODEL_OR_FAMILY=PATH",
        help="Use an existing reference environment instead of creating the declared one",
    )
    group.addoption("--qualification-no-build", action="store_true")
    group.addoption("--qualification-verbose", action="store_true")

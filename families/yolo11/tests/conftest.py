# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The upstream oracle runs in a separate test-only Python environment."""

from pathlib import Path
import os

import pytest


@pytest.fixture(scope="session")
def reference_python() -> Path:
    configured = os.environ.get("TRTMC_REFERENCE_PYTHON")
    if configured:
        python = Path(configured).resolve()
        if not python.is_file():
            raise RuntimeError(f"reference Python does not exist: {python}")
        return python
    from trtmc_perf_serving.environments import reference_python as prepare

    tests = Path(__file__).resolve().parent
    return prepare(
        tests / "requirements.txt",
        tests.parents[2] / ".ci" / "reference" / "yolo11",
        prepare=tests / "reference/prepare.py",
    )

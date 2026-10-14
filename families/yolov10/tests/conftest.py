# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The upstream oracle runs in a separate test-only Python environment."""

from pathlib import Path
import os

import pytest


@pytest.fixture(scope="session")
def reference_python(tmp_path_factory) -> Path:
    configured = os.environ.get("TRTMC_REFERENCE_PYTHON")
    if configured:
        python = Path(configured).resolve()
        if not python.is_file():
            raise RuntimeError(f"reference Python does not exist: {python}")
        return python
    from trtmc_perf_serving.environments import reference_python as prepare

    tests = Path(__file__).resolve().parent
    cache = tests.parents[2] / ".ci" / "reference" / "yolov10"
    if not cache.is_dir():
        # Source-only runners can mount the checkout read-only. Their online
        # first-use reference setup belongs in pytest's writable scratch.
        cache = tmp_path_factory.mktemp("yolov10-reference")
    return prepare(
        tests / "requirements.txt",
        cache,
        prepare=tests / "reference/prepare.py",
    )

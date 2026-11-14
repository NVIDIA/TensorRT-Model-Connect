# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build family-owned test executables in CI's configured native build tree."""

import os
from pathlib import Path
import subprocess

import pytest


def _build_probe(target):
    root = Path(
        os.environ.get("TRTMC_NATIVE_BUILD_DIR", os.environ.get("TRTMC_RUNTIME_ROOT", "build"))
    ).resolve()
    assert (root / "CMakeCache.txt").is_file(), (
        "configure TRTMC_NATIVE_BUILD_DIR with TRTMC_BUILD_TESTS=ON before running Laya tests"
    )
    subprocess.run(
        ["cmake", "--build", str(root), "--parallel", "8", "--target", target], check=True
    )
    probe = root / target
    assert probe.is_file(), f"native build did not produce {target}"
    return probe


@pytest.fixture(scope="session")
def laya_engine_probe():
    return _build_probe("laya_engine_probe")


@pytest.fixture(scope="session")
def laya_task_probe():
    return _build_probe("laya_task_probe")


@pytest.fixture(scope="session")
def laya_routing_probe():
    return _build_probe("laya_routing_probe")

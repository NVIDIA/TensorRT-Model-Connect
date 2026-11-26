# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build family test executables independently of the installed runtime tree."""

from functools import lru_cache
import os
from pathlib import Path
import subprocess

import pytest


@lru_cache(maxsize=None)
def native_test_binary(target):
    build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if build:
        root = Path(build).resolve()
        assert (root / "CMakeCache.txt").is_file(), "configure TRTMC_NATIVE_BUILD_DIR first"
        subprocess.run(
            ["cmake", "--build", str(root), "--parallel", "2", "--target", target],
            check=True,
            timeout=300,
        )
    elif runtime:
        root = Path(runtime).resolve()
    else:
        pytest.skip("requires a configured native build or prebuilt family test executables")
    binary = root / target
    assert binary.is_file(), f"build the owning {target} CMake target in {root}"
    return binary

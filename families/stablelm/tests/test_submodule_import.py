# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StableLM family modules must be importable before the family entry point."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


_PACKAGE = "families.stablelm"


@pytest.mark.parametrize(
    "submodule",
    ["default_decoder", "default_dual_profile_decoder", "model"],
)
def test_submodule_imports_first(submodule: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {_PACKAGE}.{submodule}"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
    )
    assert result.returncode == 0, result.stderr

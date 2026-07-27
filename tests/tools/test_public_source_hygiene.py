# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prevent personal paths and private infrastructure in public source."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PATTERNS = (
    r"/(workspace/users|home|localhome)/[^/[:space:]]+/",
    r"https?://[^/[:space:]]*(artifactory|intranet|internal)[^/[:space:]]*/",
    r"(^|[^A-Za-z0-9_])[a-z][a-z0-9-]*-(compute|runner|node)[0-9]+"
    r"([^A-Za-z0-9_]|$)",
)
INTERNAL_ONLY_FILES = (
    REPO_ROOT / "Dockerfile.tensorrt-sdk",
    REPO_ROOT / "scripts/publish_tensorrt_sdk.sh",
)


def test_public_tree_has_no_personal_or_private_runner_fingerprints() -> None:
    pattern = "|".join(FORBIDDEN_PATTERNS)
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-I",
            "-i",
            "-E",
            pattern,
            "--",
            ".",
            ":(exclude)tests/tools/test_public_source_hygiene.py",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout


def test_internal_execution_material_is_not_published() -> None:
    assert not [path for path in INTERNAL_ONLY_FILES if path.exists()]

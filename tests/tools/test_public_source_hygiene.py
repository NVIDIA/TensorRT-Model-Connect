# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prevent private release inputs and host-specific paths in public source."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PATTERNS = (
    r"/(workspace/users|home|localhome)/[^/[:space:]]+/",
    r"gitlab-master[.]nvidia[.]com",
    r"NVIDIA-dev/",
    r"trtmc-[^[:space:]\"']*a100[^[:space:]\"']*-proof",
    r"p2021",
    r"11[.]2[.]0[.]113",
    r"trt11[.]2",
    r"[a-z0-9-]+[.]pages[.]github[.]io",
)
INTERNAL_ONLY_FILES = (
    REPO_ROOT / "Dockerfile.tensorrt-sdk",
    REPO_ROOT / "scripts/publish_tensorrt_sdk.sh",
)


def test_public_tree_has_no_private_or_host_specific_fingerprints() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-I",
            "-i",
            "-E",
            "|".join(FORBIDDEN_PATTERNS),
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

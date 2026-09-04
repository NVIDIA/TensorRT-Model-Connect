# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prevent private release inputs and host-specific paths in public source."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PATTERNS = (
    r"/(workspace/users|home|localhome)/[^/[:space:]]+/",
    r"(gitlab|jenkins|artifactory)[-a-z0-9]*[.]nvidia[.]com",
    r"NVIDIA-[a-z0-9-]+/",
    r"trtmc[-_a-z0-9]*(actions[-_a-z0-9]*runners?|a100[-_a-z0-9]*proof)",
    r"(^|[^a-z0-9])p[0-9]{4}([^a-z0-9]|$)",
    r"[a-z0-9-]+[.]pages[.]github[.]io",
)
INTERNAL_ONLY_FILES = (
    REPO_ROOT / "Dockerfile.tensorrt-sdk",
    REPO_ROOT / "scripts/publish_tensorrt_sdk.sh",
)
LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"


def test_public_tree_has_no_private_or_host_specific_fingerprints() -> None:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-I",
            "-i",
            "-E",
            "|".join(FORBIDDEN_PATTERNS),
            "--",
            ".",
            ":(exclude)tools/tests/test_public_source_hygiene.py",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    matched_paths = sorted(path for path in result.stdout.splitlines() if path)
    assert result.returncode == 1, (
        "public source hygiene rule matched tracked files: " + ", ".join(matched_paths)
    )


def test_internal_execution_material_is_not_published() -> None:
    assert not [path for path in INTERNAL_ONLY_FILES if path.exists()]


def test_published_static_content_describes_only_the_new_architecture() -> None:
    forbidden = (
        "runtime_strategy",
        "MODEL.toml",
        "python/tensorrt_model_connect/families",
        "src/runtime/models",
        "tests/e2e/models",
        "Pipeline Registry",
    )
    violations = []
    for path in (REPO_ROOT / "website/static").rglob("*"):
        if not path.is_file() or path.suffix not in {".svg", ".txt"}:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert violations == []


def test_repo_agent_skills_do_not_route_to_retired_architecture() -> None:
    forbidden = (
        "MODEL.toml",
        "python/tensorrt_model_connect/families",
        "src/runtime/models",
        "tests/e2e/models",
        "runtime_strategy",
        "tools/trtmc_validate.py",
        "scripts/",
    )
    violations = []
    for path in (REPO_ROOT / "plugins").rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".json", ".yaml"}:
            continue
        source = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert violations == []


def test_tracked_lfs_pointers_have_filter_rules() -> None:
    pointers = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "-I",
            "-F",
            LFS_POINTER_HEADER,
            "--",
            ".",
            ":(exclude)tools/tests/test_public_source_hygiene.py",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert pointers.returncode in (0, 1), pointers.stderr

    for path in pointers.stdout.splitlines():
        attribute = subprocess.run(
            ["git", "check-attr", "filter", "--", path],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert attribute.stdout.rstrip().endswith(": filter: lfs"), (
            f"{path} is an LFS pointer without a matching filter rule"
        )

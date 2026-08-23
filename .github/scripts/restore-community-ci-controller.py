#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore the trusted Community CPU controller from an authorized base SHA.

The public workflow checks out an untrusted pull-request merge so it can test
the proposed runtime and package source. Before it installs dependencies or
selects tests, it loads this script from the authorized base commit and uses it
to replace the CI controller, build configuration, and baseline tests with
their versions from that same commit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePosixPath


COMMON_PATHS = (
    ".clang-format",
    "Dockerfile.community-cpu",
    "requirements/community-ci.txt",
    "ruff.toml",
    "tools",
)

FORBIDDEN_PATHS = ("tools.py",)

PROFILE_PATHS = {
    "impact": (),
    "source-quality": (
        "tests/e2e_harness/__init__.py",
        "tests/e2e_harness/threshold_policy.py",
        "tests/tools/test_model_plugin_encapsulation_static.py",
    ),
    "unit": (
        "CMakeLists.txt",
        "cmake",
        "conftest.py",
        "pyproject.toml",
        "tests",
    ),
}


class RestoreError(RuntimeError):
    """Report a fail-closed controller restoration error."""


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RestoreError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _validate_revision(repository: Path, revision: str) -> str:
    resolved = _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}").strip()
    if len(resolved) != 40 or any(character not in "0123456789abcdef" for character in resolved):
        raise RestoreError("authorized base did not resolve to a full commit SHA")
    if revision != resolved:
        raise RestoreError("authorized base must be supplied as its full commit SHA")
    return resolved


def _safe_path(repository: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise RestoreError(f"unsafe controller path: {relative}")
    destination = repository.joinpath(*candidate.parts)
    if not destination.is_relative_to(repository):
        raise RestoreError(f"controller path escapes the repository: {relative}")
    parent = repository
    for part in candidate.parts[:-1]:
        parent /= part
        if parent.is_symlink():
            raise RestoreError(f"controller path crosses a symlink: {relative}")
    return destination


def restore(repository: Path, base: str, profile: str) -> None:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        # A GitHub Actions checkout uses a .git directory. Worktrees may use a
        # .git file, which is also safe for local regression tests.
        if not (repository / ".git").is_file():
            raise RestoreError(f"not a Git checkout: {repository}")
    revision = _validate_revision(repository, base)
    try:
        profile_paths = PROFILE_PATHS[profile]
    except KeyError as error:
        raise RestoreError(f"unknown restoration profile: {profile}") from error

    paths = tuple(dict.fromkeys((*COMMON_PATHS, *profile_paths)))
    destinations = tuple(
        (relative, _safe_path(repository, relative))
        for relative in (*paths, *FORBIDDEN_PATHS)
    )
    for _relative, destination in destinations:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        elif destination.exists() or destination.is_symlink():
            destination.unlink()

    _git(repository, "restore", f"--source={revision}", "--worktree", "--", *paths)
    print(f"Restored trusted Community CPU {profile} controller from {revision}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_PATHS), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        restore(arguments.repository, arguments.base, arguments.profile)
    except RestoreError as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

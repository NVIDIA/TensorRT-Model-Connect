# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from tools import check_dco


def _git(repository: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _initialize(repository: Path) -> str:
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Test Committer")
    _git(repository, "config", "user.email", "committer@example.com")
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "--quiet", "-m", "base")
    return _git(repository, "rev-parse", "HEAD")


def _commit(repository: Path, message: str) -> str:
    tracked = repository / "tracked.txt"
    tracked.write_text(tracked.read_text(encoding="utf-8") + message.splitlines()[0] + "\n")
    _git(repository, "add", "tracked.txt")
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "External Contributor",
        "GIT_AUTHOR_EMAIL": "contributor@example.com",
    }
    _git(repository, "commit", "--quiet", "-m", message, env=environment)
    return _git(repository, "rev-parse", "HEAD")


def test_matching_author_signoff_passes(tmp_path: Path) -> None:
    base = _initialize(tmp_path)
    head = _commit(
        tmp_path,
        "fix: signed contribution\n\nSigned-off-by: External Contributor <contributor@example.com>",
    )

    commits, failures = check_dco.check_range(tmp_path, base, head)

    assert commits == [head]
    assert failures == []


def test_missing_signoff_reports_the_commit_and_expected_author(tmp_path: Path) -> None:
    base = _initialize(tmp_path)
    head = _commit(tmp_path, "fix: unsigned contribution")

    commits, failures = check_dco.check_range(tmp_path, base, head)

    assert commits == [head]
    assert failures == [
        f"{head[:12]}: missing Signed-off-by: "
        "External Contributor <contributor@example.com>"
    ]


def test_different_signoff_does_not_certify_the_author(tmp_path: Path) -> None:
    base = _initialize(tmp_path)
    head = _commit(
        tmp_path,
        "fix: wrong signoff\n\nSigned-off-by: Someone Else <other@example.com>",
    )

    _, failures = check_dco.check_range(tmp_path, base, head)

    assert len(failures) == 1
    assert "does not match author External Contributor <contributor@example.com>" in failures[0]
    assert "Someone Else <other@example.com>" in failures[0]


def test_every_introduced_commit_is_checked(tmp_path: Path) -> None:
    base = _initialize(tmp_path)
    _commit(
        tmp_path,
        "fix: first\n\nSigned-off-by: External Contributor <contributor@example.com>",
    )
    unsigned = _commit(tmp_path, "fix: second")
    head = _commit(
        tmp_path,
        "fix: third\n\nSigned-off-by: External Contributor <contributor@example.com>",
    )

    commits, failures = check_dco.check_range(tmp_path, base, head)

    assert len(commits) == 3
    assert len(failures) == 1
    assert failures[0].startswith(unsigned[:12])

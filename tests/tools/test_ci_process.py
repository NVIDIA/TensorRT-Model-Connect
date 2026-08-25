# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared CI process and GitHub file-command primitives."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci.process import GitHubFiles


@pytest.mark.parametrize(
    ("variable", "operation"),
    [
        ("GITHUB_ENV", "environment"),
        ("GITHUB_OUTPUT", "output"),
    ],
)
def test_github_assignment_uses_delimiter_for_multiline_values(
    tmp_path: Path,
    variable: str,
    operation: str,
) -> None:
    destination = tmp_path / variable.lower()
    github = GitHubFiles({variable: str(destination)})

    getattr(github, operation)("result", "first line\ninjected=value")

    assert destination.read_text(encoding="utf-8") == (
        "result<<TRTMC_EOF\nfirst line\ninjected=value\nTRTMC_EOF\n"
    )


def test_github_assignment_avoids_delimiter_collision(tmp_path: Path) -> None:
    destination = tmp_path / "github-output"
    github = GitHubFiles({"GITHUB_OUTPUT": str(destination)})

    github.output("result", "first line\nTRTMC_EOF\nlast line")

    assert destination.read_text(encoding="utf-8") == (
        "result<<TRTMC_EOF_\nfirst line\nTRTMC_EOF\nlast line\nTRTMC_EOF_\n"
    )

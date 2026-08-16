# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for source-quality changed-file selection."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.ci.quality import SourceQualityChecks


class _RecordingContext:
    def __init__(self, repository: Path):
        self.repository = repository
        self.env = {"CI_BASE_REF": "origin/main"}
        self.commands: list[list[object]] = []

    def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[:2] == ["git", "diff"]:
            changed = (
                "existing.py\ndeleted.py\n" if "*.py" in command else "existing.cpp\ndeleted.h\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=changed, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_lint_changed_files_skips_deleted_paths_without_skipping_existing_files(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "existing.py").touch()
    (tmp_path / "existing.cpp").touch()
    context = _RecordingContext(tmp_path)
    monkeypatch.setattr("tools.ci.quality.shutil.which", lambda _name: "/usr/bin/tool")

    SourceQualityChecks(context).lint_changed_files()

    assert ["ruff", "check", "--config", "ruff.toml", "existing.py"] in context.commands
    assert ["clang-format", "--dry-run", "--Werror", "existing.cpp"] in context.commands
    assert all("deleted.py" not in command for command in context.commands)
    assert all("deleted.h" not in command for command in context.commands)

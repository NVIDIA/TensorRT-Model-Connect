# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic artifact retry selection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.select_latest_attempt_artifact import select_latest_attempt


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "tools" / "select_latest_attempt_artifact.py"


def _artifact(root: Path, name: str, filename: str = "result.json") -> Path:
    artifact = root / name
    artifact.mkdir(parents=True)
    (artifact / filename).write_text("{}\n", encoding="utf-8")
    return artifact


def test_selects_highest_attempt_without_merging_payloads(tmp_path: Path) -> None:
    first = _artifact(tmp_path, "nightly-vlm-42-1")
    third = _artifact(tmp_path, "nightly-vlm-42-3")
    _artifact(tmp_path, "nightly-vlm-42-2")

    attempt, selected, files = select_latest_attempt(
        tmp_path, "nightly-vlm-42-", 3, "result.json"
    )

    assert attempt == 3
    assert selected == third.resolve()
    assert files == [third / "result.json"]
    assert selected != first.resolve()


@pytest.mark.parametrize(
    "name,max_attempt,error",
    (
        ("other-42-1", 1, "unexpected artifact directory"),
        ("nightly-vlm-42-2", 1, "exceeds current attempt"),
    ),
)
def test_rejects_ambiguous_or_future_artifact_directories(
    tmp_path: Path, name: str, max_attempt: int, error: str
) -> None:
    _artifact(tmp_path, name)

    with pytest.raises(ValueError, match=error):
        select_latest_attempt(tmp_path, "nightly-vlm-42-", max_attempt, "result.json")


def test_rejects_an_incomplete_attempt_instead_of_falling_back(tmp_path: Path) -> None:
    _artifact(tmp_path, "nightly-vlm-42-1")
    (tmp_path / "nightly-vlm-42-2").mkdir()

    with pytest.raises(ValueError, match="has no file matching"):
        select_latest_attempt(tmp_path, "nightly-vlm-42-", 2, "result.json")


def test_cli_records_selected_directory_and_file(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "wheel-99-2", "package.whl")
    github_output = tmp_path / "github-output"

    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--parts-dir",
            str(tmp_path),
            "--artifact-prefix",
            "wheel-99-",
            "--max-attempt",
            "2",
            "--required-glob",
            "*.whl",
            "--github-output",
            str(github_output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = github_output.read_text(encoding="utf-8")
    assert "selected_attempt=2\n" in output
    assert f"selected_dir={artifact.resolve()}\n" in output
    assert f"selected_file={(artifact / 'package.whl').resolve()}\n" in output

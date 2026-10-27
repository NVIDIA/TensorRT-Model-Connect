# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools.ci.process import GitHubFiles


def test_github_files_output_uses_delimiter_for_multiline(tmp_path: Path) -> None:
    output_file = tmp_path / "github_output.txt"
    gh = GitHubFiles({"GITHUB_OUTPUT": str(output_file)})
    gh.output("summary", "first line\nsecond line\ninjected=value")
    text = output_file.read_text(encoding="utf-8")
    assert text.startswith("summary<<TRTMC_EOF\n")
    assert "first line\nsecond line\ninjected=value\nTRTMC_EOF\n" in text


def test_github_files_output_keeps_compact_form_for_single_line(tmp_path: Path) -> None:
    output_file = tmp_path / "github_output.txt"
    gh = GitHubFiles({"GITHUB_OUTPUT": str(output_file)})
    gh.output("status", "ok")
    assert output_file.read_text(encoding="utf-8") == "status=ok\n"


def test_github_files_environment_uses_delimiter_for_multiline(tmp_path: Path) -> None:
    env_file = tmp_path / "github_env.txt"
    gh = GitHubFiles({"GITHUB_ENV": str(env_file)})
    gh.environment("NOTES", "a\nb")
    text = env_file.read_text(encoding="utf-8")
    assert text.startswith("NOTES<<TRTMC_EOF\n")
    assert text.endswith("TRTMC_EOF\n")

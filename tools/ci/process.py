# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide subprocess and GitHub file-command primitives to CI classes.

Boundary: uniform mechanics and errors only; commands and policy come from callers.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


class CiError(RuntimeError):
    """A user-facing CI configuration or execution error."""


class CommandRunner:
    """Run external tools with explicit arguments and predictable error handling."""

    def __init__(self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None):
        self.cwd = cwd
        self.env = dict(env or os.environ)

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            list(command),
            cwd=self.cwd,
            env=dict(env or self.env),
            text=True,
            check=False,
            capture_output=capture_output,
        )
        if check and result.returncode != 0:
            rendered = " ".join(command)
            detail = (result.stderr or result.stdout or "").strip()
            raise CiError(f"Command failed ({result.returncode}): {rendered}\n{detail}".rstrip())
        return result


class GitHubFiles:
    """Write GitHub Actions environment, output, and summary file commands."""

    def __init__(self, env: Mapping[str, str] | None = None):
        self.env = dict(env or os.environ)

    def environment(self, name: str, value: str) -> None:
        self._append("GITHUB_ENV", f"{name}={value}\n")

    def output(self, name: str, value: str) -> None:
        self._append("GITHUB_OUTPUT", f"{name}={value}\n", create_parent=True)

    def summary(self, text: str = "") -> None:
        self._append("GITHUB_STEP_SUMMARY", f"{text}\n")

    def _append(self, variable: str, text: str, *, create_parent: bool = False) -> None:
        destination = self.env.get(variable, "")
        if not destination:
            return
        path = Path(destination)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

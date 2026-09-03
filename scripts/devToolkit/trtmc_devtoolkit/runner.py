# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess execution with append-only, secret-free command logging."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .models import DevToolkitError


class Runner(Protocol):
    def run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        capture_output: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class CommandRunner:
    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path

    def run(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        check: bool = True,
        capture_output: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        arguments = [str(item) for item in command]
        self._log(arguments)
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env={**os.environ, **dict(env or {})},
            text=True,
            check=False,
            capture_output=capture_output,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DevToolkitError(
                f"Command failed ({result.returncode}): {shlex.join(arguments)}\n{detail}".rstrip()
            )
        return result

    def _log(self, command: Sequence[str]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(shlex.join(command) + "\n")


def command_output(
    runner: Runner,
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: int | None = None,
) -> str:
    return runner.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=timeout,
    ).stdout.strip()

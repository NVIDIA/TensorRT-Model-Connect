# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide subprocess and GitHub file-command primitives to CI classes.

Boundary: uniform mechanics and errors only; commands and policy come from callers.
"""

from __future__ import annotations

import datetime as dt
import os
import signal
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


class CiError(RuntimeError):
    """A user-facing CI configuration or execution error."""


@dataclass(frozen=True)
class ObservedProcessResult:
    """Completed direct-child execution plus durable process-boundary evidence."""

    completed: subprocess.CompletedProcess[str]
    execution_id: str
    pid: int
    cwd: Path
    started_at_utc: str
    finished_at_utc: str
    duration_ms: int
    timeout_seconds: int | None

    @property
    def stdout(self) -> str:
        return self.completed.stdout or ""

    @property
    def stderr(self) -> str:
        return self.completed.stderr or ""

    @property
    def returncode(self) -> int:
        return self.completed.returncode

    def receipt(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "pid": self.pid,
            "argv": list(self.completed.args),
            "cwd": str(self.cwd),
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "duration_ms": self.duration_ms,
            "timeout_seconds": self.timeout_seconds,
            "returncode": self.returncode,
        }


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
        timeout: int | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            list(command),
            cwd=cwd or self.cwd,
            env=dict(env or self.env),
            text=True,
            check=False,
            capture_output=capture_output,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            rendered = " ".join(command)
            detail = (result.stderr or result.stdout or "").strip()
            raise CiError(f"Command failed ({result.returncode}): {rendered}\n{detail}".rstrip())
        return result

    def run_observed(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
        cwd: Path | None = None,
    ) -> ObservedProcessResult:
        """Run one direct child and retain its PID, timing, cwd, argv, and rc."""
        selected_cwd = cwd if cwd is not None else self.cwd
        resolved_cwd = (selected_cwd or Path.cwd()).resolve()
        selected_env = self.env if env is None else env
        arguments = list(command)
        started_at = dt.datetime.now(dt.UTC)
        started_ns = time.monotonic_ns()
        process = subprocess.Popen(
            arguments,
            cwd=resolved_cwd,
            env=dict(selected_env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                # The direct child may have exited between communicate() timing
                # out and delivery of the group signal. The timeout remains a
                # failed execution even when there is no process left to kill.
                pass
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
            detail = (stderr or stdout or "").strip()
            rendered = " ".join(arguments)
            raise CiError(
                f"Command timed out after {timeout}s: {rendered}\n{detail}".rstrip()
            ) from error
        finished_at = dt.datetime.now(dt.UTC)
        result = subprocess.CompletedProcess(
            arguments,
            process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        observed = ObservedProcessResult(
            completed=result,
            execution_id=str(uuid.uuid4()),
            pid=process.pid,
            cwd=resolved_cwd,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            duration_ms=max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
            timeout_seconds=timeout,
        )
        if check and result.returncode != 0:
            rendered = " ".join(arguments)
            detail = (stderr or stdout or "").strip()
            raise CiError(f"Command failed ({result.returncode}): {rendered}\n{detail}".rstrip())
        return observed


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

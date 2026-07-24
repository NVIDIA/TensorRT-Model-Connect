# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for direct-child CI process observations."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tools.ci.process as process_module
from tools.ci.context import CiContext
from tools.ci.process import CiError, CommandRunner


def test_observed_process_records_the_direct_child_boundary(tmp_path: Path) -> None:
    runner = CommandRunner(cwd=tmp_path, env=os.environ)

    observed = runner.run_observed(
        [
            sys.executable,
            "-c",
            "import json, os, pathlib; "
            "print(json.dumps({'pid': os.getpid(), 'cwd': str(pathlib.Path.cwd())}))",
        ]
    )

    child = json.loads(observed.stdout)
    assert observed.pid == child["pid"]
    assert observed.cwd == tmp_path.resolve()
    assert child["cwd"] == str(tmp_path.resolve())
    assert observed.returncode == 0
    assert observed.duration_ms >= 0
    assert observed.started_at_utc <= observed.finished_at_utc
    assert observed.receipt()["argv"][0] == sys.executable


def test_observed_process_timeout_fails_closed(tmp_path: Path) -> None:
    runner = CommandRunner(cwd=tmp_path, env=os.environ)

    with pytest.raises(CiError, match="timed out after 1s"):
        runner.run_observed(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1,
        )


def test_observed_process_timeout_tolerates_child_exit_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExitedProcess:
        pid = 4242
        returncode = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.communicate_count = 0

        def communicate(self, timeout: int | None = None) -> tuple[str, str]:
            self.communicate_count += 1
            if self.communicate_count == 1:
                raise subprocess.TimeoutExpired(["fake-child"], timeout)
            return "", ""

    monkeypatch.setattr(process_module.subprocess, "Popen", ExitedProcess)

    def exited_before_signal(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(process_module.os, "killpg", exited_before_signal)

    with pytest.raises(CiError, match="timed out after 1s"):
        CommandRunner(cwd=tmp_path, env=os.environ).run_observed(
            ["fake-child"],
            timeout=1,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("1", 1),
        ("30s", 30),
        ("45m", 2700),
        ("2h", 7200),
        ("1d", 86400),
    ],
)
def test_context_parses_observed_process_limits(value: str | None, expected: int | None) -> None:
    assert CiContext._duration_seconds(value) == expected


@pytest.mark.parametrize("value", ["0", "0s", "-1", "1.5m", "soon"])
def test_context_rejects_invalid_observed_process_limits(value: str) -> None:
    with pytest.raises(CiError, match="positive duration"):
        CiContext._duration_seconds(value)

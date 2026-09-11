# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from trtmc_benchmark.qualification_environment import prepare_family_environment


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process lifecycle evidence")


def _wait_stopped(pid):
    for _ in range(100):
        path = Path(f"/proc/{pid}/stat")
        if not path.exists() or path.read_text().split()[2] == "Z":
            return
        time.sleep(0.01)
    pytest.fail(f"owned descendant {pid} is still running")


def _kill(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _child_source(pidfile):
    return (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )


def test_environment_timeout_cleans_installation_children(tmp_path):
    family = tmp_path / "family"
    hook = family / "tests/qualification/prepare_environment.py"
    hook.parent.mkdir(parents=True)
    pidfile = tmp_path / "child.pid"
    hook.write_text(_child_source(pidfile))
    pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            prepare_family_environment(
                family_root=family,
                cases=[],
                environment={},
                common_python=Path(sys.executable),
                directory=tmp_path / "environment",
                timeout=1,
                reuse=False,
            )
        pid = int(pidfile.read_text())
        _wait_stopped(pid)
    finally:
        if pid is None and pidfile.exists():
            pid = int(pidfile.read_text())
        if pid:
            _kill(pid)


@pytest.mark.parametrize("cancel_signal", [signal.SIGTERM, signal.SIGINT])
def test_cancellation_cleans_worker_descendants(tmp_path, cancel_signal):
    pidfile = tmp_path / "child.pid"
    code = (
        "import sys\nfrom trtmc_benchmark.qualification_process import run_process\n"
        f"run_process([sys.executable, '-c', {_child_source(pidfile)!r}], timeout=30)\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) as parent:
        pid = None
        try:
            for _ in range(500):
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                if parent.poll() is not None:
                    pytest.fail("launcher exited before creating its worker")
                time.sleep(0.01)
            else:
                pytest.fail("launcher did not create its worker")
            pid = int(pidfile.read_text())
            parent.send_signal(cancel_signal)
            assert parent.wait(timeout=5) != 0
            _wait_stopped(pid)
        finally:
            if parent.poll() is None:
                parent.kill()
            if pid:
                _kill(pid)

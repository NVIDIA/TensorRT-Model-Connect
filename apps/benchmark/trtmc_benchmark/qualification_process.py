# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the lifetime of a qualification command and its descendants."""

from contextlib import contextmanager
import os
import signal
import subprocess
import threading


@contextmanager
def _cancellation():
    previous = {}
    if threading.current_thread() is threading.main_thread():

        def cancel(signum, _frame):
            raise KeyboardInterrupt(f"qualification cancelled by signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_process(command, *, timeout, check=False, capture_output=False, **kwargs):
    """Wait for the command; never leave its workers running on the allocation.

    Qualification execution requires POSIX process groups. Commands must not
    detach their children into another session. SIGINT/SIGTERM and timeouts
    terminate the whole group, including workers whose parent already exited.
    """
    if os.name != "posix":
        raise OSError("qualification execution requires POSIX process groups")
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with _cancellation():
        with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            finally:
                # Also remove orphaned descendants after an ordinary parent exit.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result

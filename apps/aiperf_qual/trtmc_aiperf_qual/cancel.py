# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation of a run's concurrent work (both sides' Acc at once): set when the run is interrupted, so the
other side's AIPerf runs and server starts stop promptly instead of running to completion."""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")

EVENT = threading.Event()


class Cancelled(BaseException):
    """The run was interrupted: no phase or fallback handler absorbs it."""


def check() -> None:
    if EVENT.is_set():
        raise Cancelled("the run was interrupted")


def wait_for(call: Callable[[], T]) -> T:
    """``call()`` in a helper thread, waited on in one-second steps that observe cancellation: a blocking request
    no longer holds the run once it is cancelled (the helper ends when the server it waits on stops)."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = call()
        except BaseException as error:  # noqa: BLE001 - re-raised in the waiting thread
            outcome["error"] = error

    helper = threading.Thread(target=run, daemon=True)
    helper.start()
    while helper.is_alive():
        helper.join(1)
        check()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def output(command: list[str], timeout_s: float) -> str:
    """A short query's standard output (for example nvidia-smi), waited on in one-second steps that observe
    cancellation; the process is killed and reaped on cancellation or at its deadline (TimeoutExpired), and a
    failing exit raises CalledProcessError."""
    import subprocess
    import time

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    deadline = time.time() + timeout_s
    try:
        while True:
            try:
                stdout, _ = process.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                check()
                if time.time() > deadline:
                    raise
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return stdout

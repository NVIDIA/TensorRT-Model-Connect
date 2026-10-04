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

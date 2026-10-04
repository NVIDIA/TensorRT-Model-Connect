# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation of a run's concurrent work (both sides' Acc at once): set when the run is interrupted, so the
other side's AIPerf runs and server starts stop promptly instead of running to completion."""

from __future__ import annotations

import threading

EVENT = threading.Event()


class Cancelled(BaseException):
    """The run was interrupted: no phase or fallback handler absorbs it."""


def check() -> None:
    if EVENT.is_set():
        raise Cancelled("the run was interrupted")

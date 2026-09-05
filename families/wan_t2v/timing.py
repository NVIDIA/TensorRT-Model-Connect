# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan build timing contexts."""

from contextlib import contextmanager
import time


@contextmanager
def _timed(timing: dict | None, name: str):
    start = time.monotonic()
    yield
    if timing is not None:
        timing.setdefault("phases", {})[name] = time.monotonic() - start


timed_trt_compile = _timed
timed_weight_loading = _timed

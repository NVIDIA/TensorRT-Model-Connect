# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX build timing helpers."""

from contextlib import contextmanager
import time


@contextmanager
def timed_weight_loading(timing: dict | None, name: str):
    start = time.monotonic()
    yield
    if timing is not None:
        timing.setdefault("phases", {})[name] = time.monotonic() - start


timed_trt_compile = timed_weight_loading


def add_trt_compile_timing(timing: dict | None, name: str, elapsed: float) -> None:
    if timing is not None:
        timing.setdefault("phases", {})[name] = elapsed

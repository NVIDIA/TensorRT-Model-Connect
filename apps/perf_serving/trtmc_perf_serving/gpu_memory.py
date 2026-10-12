# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Peak GPU memory of a backend call (``serve --memory-probe``).

NVML samples the device's used memory while a call runs; the device's use before the backend
loaded (other processes, driver reservations) is subtracted. Device-level sampling covers every
backend alike, including the TRTMC worker's separate process, so the value is the server's peak
footprint only while no other server loads or runs on the same GPU.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable, TypeVar

T = TypeVar("T")
MIB = 1024 * 1024


class MemoryProbe:
    def __init__(self, nvml: Any, handle: Any, interval_s: float = 0.02) -> None:
        self._nvml = nvml
        self._handle = handle
        self._interval_s = interval_s
        self.baseline_mb = self._used_mb()

    def _used_mb(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / MIB

    def measure(self, call: Callable[[], T]) -> tuple[T, float]:
        """``call()`` and the peak memory above the baseline (MiB) while it ran."""
        peak = [self._used_mb()]
        done = threading.Event()

        def sample() -> None:
            while not done.wait(self._interval_s):
                peak[0] = max(peak[0], self._used_mb())

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            result = call()
        finally:
            done.set()
            sampler.join()
        peak[0] = max(peak[0], self._used_mb())
        return result, round(max(peak[0] - self.baseline_mb, 0.0), 1)


def _device_handle(nvml: Any) -> Any:
    """The first visible device: an index or a UUID in CUDA_VISIBLE_DEVICES, else device 0."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible.startswith(("GPU-", "MIG-")):
        return nvml.nvmlDeviceGetHandleByUUID(visible)
    return nvml.nvmlDeviceGetHandleByIndex(int(visible) if visible.isdigit() else 0)


def open_probe() -> MemoryProbe:
    """Start NVML and record the baseline; call before the backend loads."""
    import pynvml

    pynvml.nvmlInit()
    return MemoryProbe(pynvml, _device_handle(pynvml))

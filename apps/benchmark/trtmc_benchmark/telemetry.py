# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort, low-frequency GPU telemetry kept outside timed calls."""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import threading
import time
from typing import Any


class GpuTelemetry:
    """Sample nvidia-smi without making telemetry a benchmark prerequisite."""

    _FIELDS = ("index", "name", "gpu_utilization_percent", "memory_used_mib", "power_w")

    def __init__(self, mode: str, interval_ms: int) -> None:
        self.mode = mode
        self.interval_seconds = interval_ms / 1000.0
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> GpuTelemetry:
        self._started_at = time.monotonic()
        if self.mode == "auto" and shutil.which("nvidia-smi"):
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        elif self.mode == "auto":
            self.error = "nvidia-smi was not found"
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _device_selector(self) -> str | None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", maxsplit=1)[0].strip()
        return visible or None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                command = [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ]
                selector = self._device_selector()
                if selector is not None:
                    command.extend(["--id", selector])
                completed = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for row in csv.reader(io.StringIO(completed.stdout)):
                    if len(row) != len(self._FIELDS):
                        continue
                    values = [value.strip() for value in row]
                    self.samples.append(
                        {
                            "elapsed_s": time.monotonic() - self._started_at,
                            "index": int(values[0]),
                            "name": values[1],
                            "gpu_utilization_percent": _float_or_none(values[2]),
                            "memory_used_mib": _float_or_none(values[3]),
                            "power_w": _float_or_none(values[4]),
                        }
                    )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.error = str(exc)
            self._stop.wait(self.interval_seconds)

    def result(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sampler": "nvidia-smi" if self.samples else None,
            "interval_ms": int(self.interval_seconds * 1000),
            "device_selector": self._device_selector(),
            "sampling_scope": "whole_worker_process_not_timed_call",
            "samples": self.samples,
            "summary": _summarize(self.samples),
            "error": self.error,
        }


def _float_or_none(value: str) -> float | None:
    if value in {"", "N/A", "[Not Supported]"}:
        return None
    return float(value)


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(samples)}
    for field in ("gpu_utilization_percent", "memory_used_mib", "power_w"):
        values = [sample[field] for sample in samples if sample[field] is not None]
        if values:
            summary[f"{field}_sampled_max"] = max(values)
            summary[f"{field}_sampled_mean"] = sum(values) / len(values)
            first = values[0]
            summary[f"{field}_sampled_max_delta_from_first"] = max(values) - first
    return summary

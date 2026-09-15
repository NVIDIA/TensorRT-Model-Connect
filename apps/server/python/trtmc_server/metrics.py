# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free Prometheus accumulator for the text MVP."""

from __future__ import annotations

import threading
from collections import Counter


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, int]] = Counter()
        self._active = 0
        self._queue_seconds = 0.0
        self._inference_seconds = 0.0
        self._completed = 0
        self._setup_seconds = 0.0
        self._prefill_seconds = 0.0
        self._decode_seconds = 0.0
        self._succeeded = 0

    def begin(self) -> None:
        with self._lock:
            self._active += 1

    def finish(
        self,
        route: str,
        status: int,
        *,
        queue_seconds: float = 0.0,
        inference_seconds: float = 0.0,
        timings: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            self._requests[(route, status)] += 1
            self._active = max(0, self._active - 1)
            self._queue_seconds += queue_seconds
            self._inference_seconds += inference_seconds
            self._completed += 1
            if status == 200 and timings is not None:
                self._setup_seconds += timings["setup_ms"] / 1000.0
                self._prefill_seconds += timings["prefill_ms"] / 1000.0
                self._decode_seconds += timings["decode_ms"] / 1000.0
                self._succeeded += 1

    def reject(self, route: str, status: int) -> None:
        with self._lock:
            self._requests[(route, status)] += 1

    def render(self, *, ready: bool, busy: int) -> str:
        with self._lock:
            lines = [
                "# TYPE trtmc_server_ready gauge",
                f"trtmc_server_ready {int(ready)}",
                "# TYPE trtmc_server_queue_depth gauge",
                "trtmc_server_queue_depth 0",
                "# TYPE trtmc_server_active_requests gauge",
                f"trtmc_server_active_requests {self._active}",
                "# TYPE trtmc_server_busy_replicas gauge",
                f"trtmc_server_busy_replicas {busy}",
                "# TYPE trtmc_server_requests_total counter",
            ]
            for (route, status), count in sorted(self._requests.items()):
                lines.append(
                    f'trtmc_server_requests_total{{route="{route}",status="{status}"}} {count}'
                )
            lines.extend(
                [
                    "# TYPE trtmc_server_queue_duration_seconds summary",
                    f"trtmc_server_queue_duration_seconds_sum {self._queue_seconds}",
                    f"trtmc_server_queue_duration_seconds_count {self._completed}",
                    "# TYPE trtmc_server_inference_duration_seconds summary",
                    f"trtmc_server_inference_duration_seconds_sum {self._inference_seconds}",
                    f"trtmc_server_inference_duration_seconds_count {self._completed}",
                    "# TYPE trtmc_server_task_setup_duration_seconds summary",
                    f"trtmc_server_task_setup_duration_seconds_sum {self._setup_seconds}",
                    f"trtmc_server_task_setup_duration_seconds_count {self._succeeded}",
                    "# TYPE trtmc_server_task_prefill_duration_seconds summary",
                    f"trtmc_server_task_prefill_duration_seconds_sum {self._prefill_seconds}",
                    f"trtmc_server_task_prefill_duration_seconds_count {self._succeeded}",
                    "# TYPE trtmc_server_task_decode_duration_seconds summary",
                    f"trtmc_server_task_decode_duration_seconds_sum {self._decode_seconds}",
                    f"trtmc_server_task_decode_duration_seconds_count {self._succeeded}",
                ]
            )
        return "\n".join(lines) + "\n"

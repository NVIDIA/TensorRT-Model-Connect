# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One background worker that runs steps in order; the GPU does one thing at a time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import itertools
import logging
import queue
import threading
import time
import traceback
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: int
    project: str
    step: str
    episode: int | None
    params: dict[str, Any]
    status: str = "queued"
    message: str = ""
    error: str = ""
    started: float | None = None
    finished: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "project": self.project, "step": self.step, "episode": self.episode,
                "status": self.status, "message": self.message, "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1) if self.started else None}


class Jobs:
    def __init__(self, execute: Callable[[Job], None]):
        self._execute = execute
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._ids = itertools.count(1)
        self._history: deque[Job] = deque(maxlen=50)
        self._current: Job | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="short-drama-jobs", daemon=True)
        self._thread.start()

    def submit(self, project: str, step: str, episode: int | None, params: dict[str, Any]) -> Job:
        job = Job(next(self._ids), project, step, episode, params)
        with self._lock:
            self._history.append(job)
        self._queue.put(job)
        return job

    def cancel(self) -> None:
        """Cancel everything queued and ask the running step to stop at its next check."""

        with self._lock:
            for job in self._history:
                if job.status in {"queued", "running"}:
                    job.cancelled.set()

    def views(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.view() for job in reversed(self._history)]

    def busy(self) -> bool:
        with self._lock:
            return any(job.status in {"queued", "running"} for job in self._history)

    def close(self) -> None:
        self.cancel()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while (job := self._queue.get()) is not None:
            if job.cancelled.is_set():
                job.status, job.message = "cancelled", "cancelled before it started"
                continue
            job.status, job.started = "running", time.time()
            try:
                self._execute(job)
                job.status = "done"
            except InterruptedError:
                job.status = "cancelled"
            except Exception as error:
                job.status, job.error = "failed", f"{type(error).__name__}: {error}"
                log.error("step %s failed:\n%s", job.step, traceback.format_exc())
                # Later steps of the same episode (or of the whole project, for a project step)
                # depend on this one; other episodes go on.
                with self._lock:
                    for later in self._history:
                        if (later.status == "queued" and later.project == job.project and
                                (job.episode is None or later.episode == job.episode)):
                            later.cancelled.set()
            finally:
                job.finished = time.time()

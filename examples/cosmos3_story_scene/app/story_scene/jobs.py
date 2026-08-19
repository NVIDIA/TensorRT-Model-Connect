# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe job state and the application's one serialized worker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Protocol
import uuid

from .prompts import Submission, compile_prompt
from .runtime import PipelineError


class JobNotFound(KeyError):
    pass


class JobManagerClosed(RuntimeError):
    pass


class JobProcessor(Protocol):
    def __call__(
        self,
        job_dir: Path,
        submission: Submission,
        compiled_prompt: str,
        update_progress: Callable[[int], None],
    ) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    job_id: str
    status: str
    progress: int
    preset: str
    subject: str
    seed: int
    compiled_prompt: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None
    clean_video_url: str | None
    social_video_url: str | None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "preset": self.preset,
            "subject": self.subject,
            "seed": self.seed,
            "compiled_prompt": self.compiled_prompt,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.clean_video_url is not None:
            result["clean_video_url"] = self.clean_video_url
            result["social_video_url"] = self.social_video_url
            result["outputs"] = {
                "horizontal": self.clean_video_url,
                "social": self.social_video_url,
            }
        return result


@dataclass(slots=True)
class _Job:
    job_id: str
    directory: Path
    submission: Submission
    compiled_prompt: str
    created_at: float
    status: str = "queued"
    progress: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    outputs: dict[str, str] | None = None


_STOP = object()


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


class JobManager:
    """Own exactly one worker thread, guaranteeing serialized GPU jobs."""

    def __init__(
        self,
        output_root: Path,
        processor: JobProcessor,
        *,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._output_root = output_root
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._processor = processor
        self._uuid_factory = uuid_factory
        self._clock = clock
        self._jobs: dict[str, _Job] = {}
        self._queue: Queue[str | object] = Queue()
        self._condition = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(
            target=self._work,
            name="cosmos3-serialized-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, submission: Submission) -> JobSnapshot:
        compiled_prompt = compile_prompt(submission)
        with self._condition:
            if self._closed:
                raise JobManagerClosed("The job manager is shutting down")
            for _attempt in range(16):
                job_id = str(self._uuid_factory())
                directory = self._output_root / job_id
                try:
                    directory.mkdir(mode=0o700)
                except FileExistsError:
                    continue
                if job_id not in self._jobs:
                    break
            else:
                raise RuntimeError("Could not allocate a unique job directory")
            job = _Job(
                job_id=job_id,
                directory=directory,
                submission=submission,
                compiled_prompt=compiled_prompt,
                created_at=self._clock(),
            )
            self._jobs[job_id] = job
            snapshot = self._snapshot(job)
            self._queue.put(job_id)
            self._condition.notify_all()
            return snapshot

    def get(self, job_id: str) -> JobSnapshot:
        try:
            normalized = str(uuid.UUID(job_id))
        except (ValueError, AttributeError) as exc:
            raise JobNotFound(job_id) from exc
        if normalized != job_id.lower():
            raise JobNotFound(job_id)
        with self._condition:
            job = self._jobs.get(normalized)
            if job is None:
                raise JobNotFound(job_id)
            return self._snapshot(job)

    def wait(self, job_id: str, timeout: float = 10.0) -> JobSnapshot:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    raise JobNotFound(job_id)
                if job.status in {"succeeded", "failed"}:
                    return self._snapshot(job)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Job {job_id} did not finish")
                self._condition.wait(remaining)

    def close(self, timeout: float = 10.0) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
            self._condition.notify_all()
        self._worker.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def _update_progress(self, job_id: str, progress: int) -> None:
        if not 5 <= progress <= 99:
            raise ValueError("Running progress must be between 5 and 99")
        with self._condition:
            job = self._jobs[job_id]
            if job.status not in {"running", "packaging"}:
                return
            if progress >= 70:
                job.status = "packaging"
            job.progress = max(job.progress, progress)
            self._condition.notify_all()

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, str)
                self._run_job(item)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs[job_id]
            job.status = "running"
            job.progress = 5
            job.started_at = self._clock()
            self._condition.notify_all()
        try:
            outputs = self._processor(
                job.directory,
                job.submission,
                job.compiled_prompt,
                lambda progress: self._update_progress(job_id, progress),
            )
        except Exception as exc:  # Keep the worker alive for later customer jobs.
            if isinstance(exc, PipelineError):
                error = str(exc)
            else:
                error = "Generation failed in the local media pipeline."
            with self._condition:
                job.status = "failed"
                job.progress = 100
                job.error = error
                job.finished_at = self._clock()
                self._condition.notify_all()
            return
        with self._condition:
            job.outputs = outputs
            job.status = "succeeded"
            job.progress = 100
            job.finished_at = self._clock()
            self._condition.notify_all()

    @staticmethod
    def _snapshot(job: _Job) -> JobSnapshot:
        clean_url: str | None = None
        social_url: str | None = None
        if job.outputs is not None:
            clean_url = f"/outputs/{job.job_id}/{job.outputs['horizontal']}"
            social_url = f"/outputs/{job.job_id}/{job.outputs['social']}"
        return JobSnapshot(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            preset=job.submission.preset,
            subject=job.submission.subject,
            seed=job.submission.seed,
            compiled_prompt=job.compiled_prompt,
            created_at=_timestamp(job.created_at) or "",
            started_at=_timestamp(job.started_at),
            finished_at=_timestamp(job.finished_at),
            error=job.error,
            clean_video_url=clean_url,
            social_video_url=social_url,
        )

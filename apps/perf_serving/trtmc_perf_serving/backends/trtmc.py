# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTMC candidate backend: one persistent ``trtmc_benchmark_worker --serve`` process."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import uuid
from pathlib import Path
from typing import IO, Any, Mapping

from .base import BackendError, BackendUnavailable, Invocation


class TrtmcWorkerBackend:
    """Loads the bundle once and forwards one JSONL request per call.

    The worker applies the same dispatch and ``public_task_call_wall`` timing as
    the one-shot benchmark, so ``model_call_ms`` equals a perf-matrix sample.
    Calls must be serialized by the caller (one execution lane per worker).
    """

    def __init__(
        self,
        *,
        worker: Path,
        session: Mapping[str, Any],
        scratch: Path,
        startup_timeout_s: float = 900.0,
        request_timeout_s: float = 900.0,
        full_observations: bool = False,
        isolate_requests: bool = False,
    ) -> None:
        self.operation = str(session["operation"])
        self._worker = worker
        self._startup_timeout_s = startup_timeout_s
        self._request_timeout_s = request_timeout_s
        self._full_observations = full_observations
        # Fresh worker per request: matches one-process-per-sample qualification and separates model
        # parity from state carried across requests in one loaded session.
        self._isolate_requests = isolate_requests
        self._served = 0
        scratch.mkdir(parents=True, exist_ok=True)
        self._session_path = scratch / "worker-session.json"
        self._session_path.write_text(json.dumps(dict(session), indent=2))
        self._stderr = open(scratch / "worker.stderr.log", "ab")
        ready = self._start()
        self._identity = {"backend": "trtmc", "worker": str(worker), "bundle": session["bundle"],
                          "isolate_requests": isolate_requests,
                          **{key: ready[key] for key in ("operation", "task", "selected_task", "timing_scope",
                                                         "asset_loading_included", "load_ms")}}

    def _start(self) -> dict[str, Any]:
        self._process = subprocess.Popen(
            [str(self._worker), "--serve", str(self._session_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=_pump, args=(self._process.stdout, self._lines), daemon=True).start()
        self._available = True
        try:
            ready = self._read(self._startup_timeout_s, "worker startup")
            if ready.get("event") != "ready":
                raise BackendUnavailable(f"worker failed to load: {ready.get('error', ready)}")
        except BackendUnavailable:
            self.close()
            raise
        return ready

    def describe(self) -> Mapping[str, Any]:
        return dict(self._identity)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        if self._isolate_requests and self._served:
            self._stop_process()
            self._start()
        if not self._available:
            raise BackendUnavailable("worker is no longer available")
        self._served += 1
        message_id = uuid.uuid4().hex
        self._write({"id": message_id, "request": dict(request), "artifact_base": str(artifact_base),
                     "full_observation": self._full_observations})
        reply = self._read(self._request_timeout_s, "worker request")
        if reply.get("id") != message_id:
            self._retire()
            raise BackendUnavailable("worker protocol desynchronized")
        if not reply.get("ok"):
            raise BackendError(str(reply.get("error", "worker request failed")))
        observation = dict(reply["observation"])
        model_call_ms = float(observation.pop("runtime_e2e_wall_ms"))
        return Invocation(observation=observation, model_call_ms=model_call_ms)

    def close(self) -> None:
        self._stop_process()
        self._stderr.close()

    def _stop_process(self) -> None:
        if self._process.poll() is None:
            try:
                self._write({"id": "shutdown", "type": "shutdown"})
                self._process.wait(timeout=30)
            except (BackendUnavailable, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()

    def _write(self, message: Mapping[str, Any]) -> None:
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self._retire()
            raise BackendUnavailable(f"worker exited: {error}") from error

    def _read(self, timeout_s: float, label: str) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty as error:
            self._retire()
            raise BackendUnavailable(f"{label} timed out after {timeout_s:.0f}s") from error
        if line is None:
            self._retire()
            raise BackendUnavailable(f"worker exited during {label} (code {self._process.poll()})")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            self._retire()
            raise BackendUnavailable(f"worker emitted a non-protocol line: {line[:200]!r}") from error

    def _retire(self) -> None:
        # A timed-out or desynchronized worker cannot be trusted with another request.
        self._available = False
        if self._process.poll() is None:
            self._process.kill()


def _pump(stream: IO[str], lines: "queue.Queue[str | None]") -> None:
    for line in stream:
        lines.put(line)
    lines.put(None)

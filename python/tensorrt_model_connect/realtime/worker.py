# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Injectable native JSONL worker transport for realtime speech sessions."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
import shutil
from typing import Any, Protocol, runtime_checkable


JsonObject = dict[str, Any]


class WorkerError(RuntimeError):
    """The native worker failed or violated its JSONL contract."""


def find_native_worker(explicit: Path | None = None) -> Path:
    """Find ``trtmc_realtime_worker`` in source and installed layouts."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    configured = os.environ.get("TRTMC_REALTIME_WORKER")
    if configured:
        candidates.append(Path(configured).expanduser())
    repository = Path(__file__).resolve().parents[3]
    candidates.extend(
        repository / build / "trtmc_realtime_worker"
        for build in ("build", "build-make", "build-local")
    )
    candidates.append(Path(__file__).resolve().parents[1] / "bin/trtmc_realtime_worker")
    discovered = shutil.which("trtmc_realtime_worker")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise WorkerError("cannot find executable trtmc_realtime_worker")


@runtime_checkable
class RealtimeWorker(Protocol):
    """One native speech session, represented as request/event JSON objects."""

    async def start(self) -> None: ...

    async def send(self, message: JsonObject) -> None: ...

    async def receive(self) -> JsonObject | None: ...

    async def close(self) -> None: ...


class NativeJsonlWorker:
    """Run one explicitly configured executable without a shell.

    Every stdin and stdout line is one JSON object. Stderr is discarded: the
    public WebSocket surface reports a bounded generic worker error and never
    forwards process diagnostics or filesystem details to a remote client.
    """

    def __init__(
        self,
        executable: str | Path,
        arguments: Sequence[str] = (),
        *,
        max_line_bytes: int = 1 << 20,
        shutdown_timeout_seconds: float = 2.0,
    ) -> None:
        executable_text = str(executable)
        if not executable_text:
            raise ValueError("native worker executable must not be empty")
        if any(not isinstance(argument, str) or not argument for argument in arguments):
            raise ValueError("native worker arguments must be non-empty strings")
        if max_line_bytes <= 0:
            raise ValueError("native worker line bound must be positive")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("native worker shutdown timeout must be positive")
        self._command = (executable_text, *arguments)
        self._max_line_bytes = max_line_bytes
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._process is not None:
            raise WorkerError("native worker is already started")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=self._max_line_bytes + 1,
            )
        except (OSError, ValueError) as exc:
            raise WorkerError("native worker could not be started") from exc

    async def send(self, message: JsonObject) -> None:
        process = self._running_process()
        if process.stdin is None:
            raise WorkerError("native worker input is unavailable")
        try:
            payload = json.dumps(message, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        except (TypeError, ValueError) as exc:
            raise WorkerError("native worker request is not JSON serializable") from exc
        if len(payload) > self._max_line_bytes:
            raise WorkerError("native worker request exceeds its message bound")
        async with self._write_lock:
            try:
                process.stdin.write(payload + b"\n")
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionError) as exc:
                raise WorkerError("native worker closed its input") from exc

    async def receive(self) -> JsonObject | None:
        process = self._running_process(allow_exited=True)
        if process.stdout is None:
            raise WorkerError("native worker output is unavailable")
        try:
            line = await process.stdout.readline()
        except (ValueError, asyncio.LimitOverrunError) as exc:
            raise WorkerError("native worker event exceeds its message bound") from exc
        if not line:
            return_code = await process.wait()
            if return_code != 0:
                raise WorkerError("native worker exited unsuccessfully")
            return None
        if len(line) > self._max_line_bytes + 1 or not line.endswith(b"\n"):
            raise WorkerError("native worker event exceeds its message bound")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerError("native worker returned invalid JSON") from exc
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise WorkerError("native worker event must be a JSON object")
        return value

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionError):
                pass
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_seconds)
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_seconds)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

    def _running_process(self, *, allow_exited: bool = False) -> asyncio.subprocess.Process:
        process = self._process
        if process is None:
            raise WorkerError("native worker is not started")
        if not allow_exited and process.returncode is not None:
            raise WorkerError("native worker is no longer running")
        return process

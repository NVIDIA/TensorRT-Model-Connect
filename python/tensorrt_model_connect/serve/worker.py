# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived JSONL subprocess transport for ``trtmc _serve-worker``."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .errors import (
    WorkerCrashedError,
    WorkerError,
    WorkerProtocolError,
    WorkerRequestTooLargeError,
    WorkerSaturatedError,
    WorkerRemoteError,
    WorkerStartupError,
    WorkerTimeoutError,
)


_Response = dict[str, Any] | WorkerError
_MAX_STDERR_LINE_CHARS = 8192
_MAX_REQUEST_LINE_BYTES = 16 * 1024 * 1024
_WORKER_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CONDA_PREFIX",
        "CUDA_CACHE_DISABLE",
        "CUDA_CACHE_MAXSIZE",
        "CUDA_CACHE_PATH",
        "CUDA_DEVICE_ORDER",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMPI_COMM_WORLD_JOBID",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMP_NUM_THREADS",
        "PATH",
        "PMI_RANK",
        "PMI_SIZE",
        "PMIX_NAMESPACE",
        "RANK",
        "SLURM_PROCID",
        "TMPDIR",
        "TRTMC_MODEL_PLUGIN_DIR",
        "TRTMC_MODEL_PLUGIN_STRICT",
        "TRTMC_NCCL_RENDEZVOUS",
        "TRTMC_NCCL_SKIP_DESTROY",
        "TRTMC_TRT_LIBRARY_DIR",
        "TZ",
        "USER",
        "VIRTUAL_ENV",
        "WORLD_SIZE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    }
)
_WINDOWS_WORKER_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
    }
)


def _worker_environment() -> dict[str, str]:
    """Copy only runtime variables that a native worker is allowed to observe."""

    allowed = _WORKER_ENVIRONMENT_ALLOWLIST
    if os.name == "nt":
        allowed = allowed | _WINDOWS_WORKER_ENVIRONMENT_ALLOWLIST
    return {name: os.environ[name] for name in allowed if name in os.environ}


@dataclass(frozen=True)
class WorkerLoadOptions:
    """Native load-time options forwarded to every ``_serve-worker``."""

    backend_dirs: tuple[str, ...] = ()
    model_plugin_dirs: tuple[str, ...] = ()
    runtime_cache: str | None = None
    kernel_bindings: str | None = None
    config: str | None = None
    set_values: tuple[str, ...] = ()
    cuda_graphs: bool = False

    def argv(self) -> list[str]:
        result: list[str] = []
        for path in self.backend_dirs:
            result.extend(("--backend-dir", path))
        for path in self.model_plugin_dirs:
            result.extend(("--model-plugin-dir", path))
        if self.runtime_cache:
            result.extend(("--runtime-cache", self.runtime_cache))
        if self.kernel_bindings:
            result.extend(("--kernel-bindings", self.kernel_bindings))
        if self.config:
            result.extend(("--config", self.config))
        for value in self.set_values:
            result.extend(("--set", value))
        if self.cuda_graphs:
            result.append("--cuda-graphs")
        return result


class WorkerSession:
    """Exclusive sequence of operations against one stateful worker stream."""

    def __init__(self, worker: "WorkerProcess", release: Callable[[], None]) -> None:
        self._worker = worker
        self._release = release
        self._closed = False
        self._lock = threading.Lock()

    def request(
        self,
        op: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if self._closed:
            raise WorkerCrashedError(f"worker session for {self._worker.name!r} is closed")
        return self._worker._request_locked(op, payload, timeout=timeout)  # noqa: SLF001

    def submit(
        self,
        op: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Future[Any]:
        if self._closed:
            raise WorkerCrashedError(f"worker session for {self._worker.name!r} is closed")
        return self._worker._executor.submit(  # noqa: SLF001
            self.request, op, payload, timeout=timeout
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._release()

    def __enter__(self) -> "WorkerSession":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class WorkerProcess:
    """Own one persistent native model worker.

    Stdout is reserved for one JSON object per line. Stderr is drained on a
    dedicated thread and retained as a bounded diagnostic tail. Requests are
    assigned opaque IDs and serialized because the native worker owns mutable
    pipeline state. ``acquire_session`` holds that serialization lock across a
    realtime start/chunk/finish sequence.
    """

    def __init__(
        self,
        *,
        name: str,
        bundle: str | os.PathLike[str],
        trtmc_binary: str | os.PathLike[str],
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        stderr_lines: int = 100,
        max_request_line_bytes: int = _MAX_REQUEST_LINE_BYTES,
        load_options: WorkerLoadOptions | None = None,
    ) -> None:
        if startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("worker timeouts must be positive")
        if stderr_lines <= 0:
            raise ValueError("stderr_lines must be positive")
        if max_request_line_bytes <= 0:
            raise ValueError("max_request_line_bytes must be positive")
        self.name = name
        self.bundle = Path(bundle)
        self.trtmc_binary = Path(trtmc_binary)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.max_request_line_bytes = int(max_request_line_bytes)
        self.load_options = load_options or WorkerLoadOptions()

        self._state = "new"
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"trtmc-{name}")
        self._response: queue.Queue[_Response] = queue.Queue(maxsize=1)
        self._expected_request_id: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._ready_payload: dict[str, Any] = {}
        self._last_error: WorkerError | None = None
        self._stderr_tail: deque[str] = deque(maxlen=stderr_lines)

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return (
                self._state == "ready"
                and self._process is not None
                and self._process.poll() is None
            )

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            return self._process.pid if self._process is not None else None

    @property
    def ready_payload(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._ready_payload)

    @property
    def stderr_tail(self) -> list[str]:
        with self._state_lock:
            return list(self._stderr_tail)

    def start(self) -> None:
        """Launch the worker and wait for its JSON ready record."""

        with self._state_lock:
            if self._state == "ready":
                return
            if self._state != "new":
                raise WorkerStartupError(
                    f"worker {self.name!r} cannot start from state {self._state!r}"
                )
            self._state = "starting"
            self._ready_event.clear()

        command = [
            str(self.trtmc_binary),
            "_serve-worker",
            str(self.bundle),
            *self.load_options.argv(),
        ]
        worker_environment = _worker_environment()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=(os.name != "nt"),
                env=worker_environment,
            )
        except OSError as exc:
            error = WorkerStartupError(f"failed to launch worker {self.name!r}")
            self._mark_failed(error)
            raise error from exc

        assert process.stdout is not None
        assert process.stderr is not None
        with self._state_lock:
            self._process = process
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            name=f"trtmc-{self.name}-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            name=f"trtmc-{self.name}-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        if not self._ready_event.wait(self.startup_timeout):
            error = WorkerStartupError(
                f"worker {self.name!r} did not become ready within {self.startup_timeout:g}s"
            )
            self._mark_failed(error)
            self._terminate_process()
            self._finalize_io()
            raise error

        with self._state_lock:
            ready = self._state == "ready"
            error = self._last_error or WorkerStartupError(
                f"worker {self.name!r} failed before readiness"
            )
        if not ready:
            self._terminate_process()
            self._finalize_io()
            if isinstance(error, WorkerStartupError):
                raise error
            raise WorkerStartupError(str(error)) from error

    def request(
        self,
        op: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Run one serialized worker operation and return its unwrapped result."""

        deadline = self.request_timeout if timeout is None else float(timeout)
        if deadline <= 0:
            raise ValueError("request timeout must be positive")
        with self.acquire_session(timeout=deadline) as session:
            return session.request(op, payload, timeout=deadline)

    def acquire_session(
        self,
        *,
        timeout: float | None = None,
        on_close: Callable[["WorkerProcess"], None] | None = None,
    ) -> WorkerSession:
        """Acquire exclusive worker ownership for a stateful request sequence."""

        deadline = self.request_timeout if timeout is None else float(timeout)
        if deadline < 0:
            raise ValueError("session timeout must be non-negative")
        if not self._operation_lock.acquire(timeout=deadline):
            raise WorkerTimeoutError(f"worker {self.name!r} remained busy for {deadline:g}s")
        try:
            self._assert_usable()
        except BaseException:
            self._operation_lock.release()
            raise

        def release() -> None:
            self._operation_lock.release()
            if on_close is not None:
                on_close(self)

        return WorkerSession(self, release)

    def _request_locked(
        self,
        op: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout: float | None,
        allow_closing: bool = False,
    ) -> Any:
        deadline = self.request_timeout if timeout is None else float(timeout)
        if not op or not isinstance(op, str):
            raise ValueError("worker operation must be a non-empty string")
        if payload:
            reserved = {"id", "op"}.intersection(payload)
            if reserved:
                field = sorted(reserved)[0]
                raise ValueError(f"worker payload cannot override {field!r}")

        with self._state_lock:
            self._assert_usable_locked(allow_closing=allow_closing)
            process = self._process
            assert process is not None
            request_id = uuid.uuid4().hex
            if self._expected_request_id is not None:
                raise WorkerProtocolError(f"worker {self.name!r} already has an active request")
            self._expected_request_id = request_id

        message: dict[str, Any] = {"id": request_id, "op": op}
        if payload:
            message.update(payload)

        try:
            encoded = json.dumps(message, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            with self._state_lock:
                self._expected_request_id = None
            raise WorkerProtocolError(f"worker request is not JSON serializable: {exc}") from exc
        encoded_bytes = len(encoded.encode("utf-8"))
        if encoded_bytes > self.max_request_line_bytes:
            with self._state_lock:
                self._expected_request_id = None
            raise WorkerRequestTooLargeError(
                f"worker request for {op!r} is {encoded_bytes} bytes; maximum is "
                f"{self.max_request_line_bytes} bytes"
            )

        try:
            assert process.stdin is not None
            process.stdin.write(encoded + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            error = WorkerCrashedError(f"worker {self.name!r} transport failed during {op!r}")
            self._mark_failed(error)
            self._terminate_process()
            self._finalize_io()
            raise error from exc

        try:
            response = self._response.get(timeout=deadline)
        except queue.Empty as exc:
            error = WorkerTimeoutError(
                f"worker {self.name!r} timed out after {deadline:g}s during {op!r}"
            )
            self._mark_failed(error)
            self._terminate_process()
            self._finalize_io()
            raise error from exc
        finally:
            with self._state_lock:
                self._expected_request_id = None

        if isinstance(response, WorkerError):
            self._terminate_process()
            self._finalize_io()
            raise response
        return self._unwrap_response(op, response)

    def close(self, *, grace_period: float = 1.0) -> None:
        """Best-effort protocol shutdown followed by deterministic cleanup."""

        with self._state_lock:
            if self._state == "closed":
                return
            was_ready = self._state == "ready"

        acquired = self._operation_lock.acquire(timeout=max(0.0, grace_period))
        try:
            with self._state_lock:
                if self._state == "closed":
                    return
                self._state = "closing"
            if acquired and was_ready:
                try:
                    self._request_locked(
                        "shutdown",
                        None,
                        timeout=max(0.05, grace_period),
                        allow_closing=True,
                    )
                except WorkerError:
                    pass
        finally:
            if acquired:
                self._operation_lock.release()

        self._terminate_process(grace_period=grace_period)
        self._finalize_io()
        self._executor.shutdown(wait=True, cancel_futures=True)
        error = WorkerCrashedError(f"worker {self.name!r} was closed")
        self._wake_request(error)
        with self._state_lock:
            self._state = "closed"

    @property
    def busy(self) -> bool:
        return self._operation_lock.locked()

    def _assert_usable(self) -> None:
        with self._state_lock:
            self._assert_usable_locked()

    def _assert_usable_locked(self, *, allow_closing: bool = False) -> None:
        allowed = {"ready"}
        if allow_closing:
            allowed.add("closing")
        process = self._process
        if self._state not in allowed or process is None or process.poll() is not None:
            if self._last_error is not None:
                raise WorkerCrashedError(str(self._last_error)) from self._last_error
            raise WorkerCrashedError(f"worker {self.name!r} is unavailable (state={self._state!r})")

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._fail_protocol(f"worker {self.name!r} emitted invalid JSONL: {exc.msg}")
                    return
                if not isinstance(message, dict):
                    self._fail_protocol(f"worker {self.name!r} emitted a non-object JSON record")
                    return

                if message.get("event") == "ready":
                    with self._state_lock:
                        if self._state != "starting":
                            self._fail_protocol(
                                f"worker {self.name!r} emitted an unexpected ready record"
                            )
                            return
                        self._ready_payload = dict(message)
                        self._state = "ready"
                        self._ready_event.set()
                    continue

                request_id = message.get("id")
                if not isinstance(request_id, str) or not request_id:
                    self._fail_protocol(
                        f"worker {self.name!r} response is missing a string request id"
                    )
                    return
                with self._state_lock:
                    expected = self._expected_request_id
                    terminal = self._state in {"failed", "closing", "closed"}
                if request_id != expected:
                    # A timed-out request may race with termination. Ignore only
                    # after failure/close; otherwise the protocol is desynchronized.
                    if terminal:
                        continue
                    self._fail_protocol(
                        f"worker {self.name!r} responded with unknown id {request_id!r}"
                    )
                    return
                ok = message.get("ok")
                if not isinstance(ok, bool):
                    self._fail_protocol(
                        f"worker {self.name!r} response is missing a boolean ok field"
                    )
                    return
                if ok and "result" not in message:
                    self._fail_protocol(f"worker {self.name!r} success response is missing result")
                    return
                if not ok and not isinstance(message.get("error"), Mapping):
                    self._fail_protocol(
                        f"worker {self.name!r} failure response is missing an error object"
                    )
                    return
                try:
                    self._response.put_nowait(message)
                except queue.Full:
                    self._fail_protocol(f"worker {self.name!r} emitted more than one response")
                    return
        except (OSError, ValueError):
            self._mark_failed(WorkerCrashedError(f"worker {self.name!r} stdout failed"))
            return
        finally:
            with self._state_lock:
                process = self._process
                state = self._state
            returncode = process.poll() if process is not None else None
            error = WorkerCrashedError(
                f"worker {self.name!r} exited"
                + (f" with code {returncode}" if returncode is not None else "")
            )
            self._wake_request(error)
            if state not in {"failed", "closing", "closed"}:
                self._mark_failed(error)
            try:
                stream.close()
            except OSError:
                pass
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

    def _read_stderr(self, stream: TextIO) -> None:
        try:
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if line:
                    with self._state_lock:
                        self._stderr_tail.append(line[-_MAX_STDERR_LINE_CHARS:])
        except (OSError, ValueError):
            return
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _unwrap_response(self, op: str, response: dict[str, Any]) -> Any:
        if not response["ok"]:
            details = response["error"]
            message = str(details.get("message") or details.get("code") or details)
            raise WorkerRemoteError(
                f"worker {self.name!r} failed {op!r}: {message}", details=details
            )
        return response["result"]

    def _fail_protocol(self, message: str) -> None:
        self._mark_failed(WorkerProtocolError(message))
        self._terminate_process()

    def _mark_failed(self, error: WorkerError) -> None:
        with self._state_lock:
            if self._state in {"closing", "closed"}:
                self._wake_request(error)
                self._ready_event.set()
                return
            self._state = "failed"
            self._last_error = error
            self._ready_event.set()
        self._wake_request(error)

    def _wake_request(self, error: WorkerError) -> None:
        with self._state_lock:
            pending = self._expected_request_id is not None
        if pending:
            try:
                self._response.put_nowait(error)
            except queue.Full:
                pass

    def _terminate_process(self, *, grace_period: float = 0.5) -> None:
        with self._state_lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=max(0.05, grace_period))
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass

    def _finalize_io(self) -> None:
        with self._state_lock:
            process = self._process
            threads = (self._stdout_thread, self._stderr_thread)
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        current = threading.current_thread()
        for thread in threads:
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=0.5)

    def __enter__(self) -> "WorkerProcess":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


class WorkerGroup:
    """Route one logical model to a fixed set of native worker replicas."""

    def __init__(self, name: str, workers: Iterable[WorkerProcess]) -> None:
        self.name = name
        self._workers = tuple(workers)
        if not self._workers:
            raise ValueError("worker group must contain at least one replica")
        self._idle: queue.Queue[WorkerProcess] = queue.Queue(maxsize=len(self._workers))
        for worker in self._workers:
            if not worker.ready:
                raise ValueError(f"worker {worker.name!r} is not ready")
            self._idle.put_nowait(worker)
        self._closed = False
        self._lock = threading.Lock()

    @property
    def replicas(self) -> int:
        return len(self._workers)

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._closed and any(worker.ready for worker in self._workers)

    def acquire_session(self) -> WorkerSession:
        """Lease one idle replica, or fail immediately when all are busy."""

        with self._lock:
            if self._closed:
                raise WorkerCrashedError(f"worker group {self.name!r} is closed")

        while True:
            try:
                worker = self._idle.get_nowait()
            except queue.Empty as exc:
                if not self.ready:
                    raise WorkerCrashedError(
                        f"worker group {self.name!r} has no healthy replicas"
                    ) from exc
                raise WorkerSaturatedError(
                    f"all {self.replicas} replicas for model {self.name!r} are busy"
                ) from exc

            if not worker.ready:
                continue
            try:
                return worker.acquire_session(timeout=0, on_close=self._release)
            except WorkerCrashedError:
                continue

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for worker in reversed(self._workers):
            worker.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            closed = self._closed
        ready_replicas = sum(worker.ready for worker in self._workers)
        idle_replicas = sum(worker.ready and not worker.busy for worker in self._workers)
        ready = not closed and ready_replicas > 0
        return {
            "state": "closed" if closed else "ready" if ready else "failed",
            "ready": ready,
            "replicas": self.replicas,
            "ready_replicas": ready_replicas,
            "idle_replicas": idle_replicas,
            "busy": idle_replicas < ready_replicas,
        }

    def _release(self, worker: WorkerProcess) -> None:
        with self._lock:
            if self._closed or not worker.ready:
                return
            self._idle.put_nowait(worker)

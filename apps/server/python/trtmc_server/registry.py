# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own text-model worker processes without leaking serving concerns into the runtime."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ModelNotFoundError, WorkerProtocolError
from .worker import WorkerGroup, WorkerLoadOptions, WorkerProcess, WorkerSession


_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    bundle: Path
    replicas: int = 1

    def __post_init__(self) -> None:
        if _MODEL_NAME.fullmatch(self.name) is None:
            raise ValueError(f"invalid model name: {self.name!r}")
        if self.replicas <= 0:
            raise ValueError("model replicas must be positive")
        object.__setattr__(self, "bundle", Path(self.bundle))


class ModelRegistry:
    def __init__(
        self,
        specs: list[ModelSpec],
        *,
        worker_binary: Path,
        load_options: WorkerLoadOptions,
        startup_timeout: float,
        request_timeout: float,
    ) -> None:
        if not specs:
            raise ValueError("at least one text model is required")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("model names must be unique")
        self._specs = {spec.name: spec for spec in specs}
        self._order = [spec.name for spec in specs]
        self._worker_binary = worker_binary
        self._load_options = load_options
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._groups: dict[str, WorkerGroup] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._started_at = 0
        self._accepting = False
        self._lock = threading.RLock()

    @property
    def accepting(self) -> bool:
        with self._lock:
            return self._accepting

    def start(self) -> None:
        started: list[WorkerProcess] = []
        try:
            for name in self._order:
                spec = self._specs[name]
                workers: list[WorkerProcess] = []
                metadata: dict[str, Any] | None = None
                for replica in range(spec.replicas):
                    worker = WorkerProcess(
                        name=f"{name}-{replica + 1}",
                        bundle=spec.bundle,
                        trtmc_binary=self._worker_binary,
                        startup_timeout=self._startup_timeout,
                        request_timeout=self._request_timeout,
                        load_options=self._load_options,
                    )
                    worker.start()
                    started.append(worker)
                    self._validate_ready(worker.ready_payload)
                    if metadata is None:
                        metadata = worker.ready_payload
                    elif metadata != worker.ready_payload:
                        raise WorkerProtocolError(f"model {name!r} replicas disagree on metadata")
                    workers.append(worker)
                self._groups[name] = WorkerGroup(name, workers)
                self._metadata[name] = metadata or {}
            with self._lock:
                self._accepting = True
                self._started_at = int(time.time())
        except BaseException:
            for worker in reversed(started):
                worker.close()
            self._groups.clear()
            self._metadata.clear()
            raise

    def close(self) -> None:
        with self._lock:
            self._accepting = False
        for group in reversed(self._groups.values()):
            group.close()
        self._groups.clear()

    def acquire(self, model: str) -> WorkerSession:
        with self._lock:
            if not self._accepting:
                raise RuntimeError("server is draining")
            group = self._groups.get(model)
            if group is None:
                raise ModelNotFoundError(f"model {model!r} is not registered")
            return group.acquire_session()

    def max_tokens(self, model: str, hard_cap: int) -> int:
        metadata = self._metadata.get(model)
        if metadata is None:
            raise ModelNotFoundError(f"model {model!r} is not registered")
        value = metadata.get("default_max_new_tokens", 128)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            value = 128
        return min(value, hard_cap)

    def models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": name,
                "object": "model",
                "created": self._started_at,
                "owned_by": "tensorrt-model-connect",
            }
            for name in self._order
        ]

    def status(self) -> dict[str, Any]:
        model_status = {name: group.status() for name, group in self._groups.items()}
        ready = self.accepting and len(model_status) == len(self._specs) and all(
            status["ready"] for status in model_status.values()
        )
        return {
            "ready": ready,
            "degraded": self.accepting and any(
                status["degraded"] for status in model_status.values()
            ),
            "models": model_status,
        }

    @staticmethod
    def _validate_ready(metadata: dict[str, Any]) -> None:
        if metadata.get("event") != "ready" or metadata.get("protocol_version") != 1:
            raise WorkerProtocolError("worker uses an unsupported ready protocol")
        capabilities = metadata.get("capabilities")
        if not isinstance(capabilities, list) or "text_generation" not in capabilities:
            raise WorkerProtocolError("worker does not advertise text_generation")

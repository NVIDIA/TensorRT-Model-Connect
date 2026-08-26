# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model registration and native worker lifecycle management."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .errors import ModelCapabilityError, ModelNotFoundError, WorkerProtocolError
from .worker import WorkerGroup, WorkerLoadOptions, WorkerProcess, WorkerSession


ModelKind = Literal["chat", "transcription"]
_WORKER_PROTOCOL_VERSION = 2
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PUBLIC_MODEL_METADATA_FIELDS = (
    "default_max_new_tokens",
    "max_cache_length",
    "streaming_transcription",
)
_PUBLIC_MODEL_STATUS_FIELDS = (
    "ready",
    "replicas",
    "ready_replicas",
    "idle_replicas",
    "busy",
)


@dataclass(frozen=True)
class ModelSpec:
    """One API-visible name backed by a fixed native worker group."""

    name: str
    bundle: Path
    kind: ModelKind

    def __post_init__(self) -> None:
        if _MODEL_NAME.fullmatch(self.name) is None:
            raise ValueError(
                "model name must start with an alphanumeric character and contain only "
                "letters, digits, '.', '_', or '-' (maximum 128 characters)"
            )
        if self.kind not in {"chat", "transcription"}:
            raise ValueError(f"unsupported model kind: {self.kind!r}")
        object.__setattr__(self, "bundle", Path(self.bundle))


WorkerFactory = Callable[[ModelSpec], WorkerProcess]


class ModelRegistry:
    """Own all model workers exposed by one server process."""

    def __init__(
        self,
        specs: Iterable[ModelSpec],
        *,
        trtmc_binary: str | Path,
        default_chat_model: str | None = None,
        default_transcription_model: str | None = None,
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        load_options: WorkerLoadOptions | None = None,
        model_replicas: Mapping[str, int] | None = None,
        required_streaming_transcription: Iterable[str] = (),
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        ordered_specs = list(specs)
        names = [spec.name for spec in ordered_specs]
        if len(names) != len(set(names)):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate model name(s): {', '.join(duplicates)}")

        self._specs = {spec.name: spec for spec in ordered_specs}
        self._order = names
        self._trtmc_binary = Path(trtmc_binary)
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._load_options = load_options or WorkerLoadOptions()
        self._worker_factory = worker_factory
        configured_replicas = dict(model_replicas or {})
        unknown_replicas = sorted(set(configured_replicas) - set(self._specs))
        if unknown_replicas:
            raise ValueError(
                "replicas configured for unknown model(s): " + ", ".join(unknown_replicas)
            )
        for name, replicas in configured_replicas.items():
            if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas <= 0:
                raise ValueError(f"replicas for model {name!r} must be a positive integer")
        self._replicas = {name: configured_replicas.get(name, 1) for name in names}
        self._groups: dict[str, WorkerGroup] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._started = False
        self._started_at: int | None = None
        self._lock = threading.RLock()
        self._required_streaming = set(required_streaming_transcription)

        for name in sorted(self._required_streaming):
            spec = self._specs.get(name)
            if spec is None:
                raise ValueError(
                    f"required streaming transcription model {name!r} is not registered"
                )
            if spec.kind != "transcription":
                raise ValueError(
                    f"required streaming transcription model {name!r} is not transcription"
                )

        self.default_chat_model = self._resolve_default("chat", default_chat_model)
        self.default_transcription_model = self._resolve_default(
            "transcription", default_transcription_model
        )

    @property
    def request_timeout(self) -> float:
        return self._request_timeout

    @property
    def names(self) -> list[str]:
        return list(self._order)

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._started
                and bool(self._groups)
                and len(self._groups) == len(self._specs)
                and all(group.ready for group in self._groups.values())
            )

    @property
    def has_healthy_worker(self) -> bool:
        """Return whether this process can still execute any model request."""

        with self._lock:
            return self._started and any(group.ready for group in self._groups.values())

    def start(self) -> None:
        """Start every configured worker replica."""

        with self._lock:
            if self._started:
                return
            started_workers: list[WorkerProcess] = []
            try:
                for name in self._order:
                    spec = self._specs[name]
                    workers: list[WorkerProcess] = []
                    ready_payload: dict[str, Any] | None = None
                    for replica in range(self._replicas[name]):
                        worker = self._make_worker(spec, replica)
                        worker.start()
                        workers.append(worker)
                        started_workers.append(worker)
                        payload = worker.ready_payload
                        self._validate_metadata(name, payload)
                        if ready_payload is None:
                            ready_payload = payload
                        elif payload != ready_payload:
                            raise WorkerProtocolError(
                                f"worker replicas for model {name!r} reported inconsistent metadata"
                            )

                    assert ready_payload is not None
                    metadata = {
                        key: ready_payload[key]
                        for key in _PUBLIC_MODEL_METADATA_FIELDS
                        if key in ready_payload
                    }
                    metadata["streaming_transcription"] = False
                    if name in self._required_streaming:
                        for worker in workers:
                            probe = worker.request(
                                "probe_transcription_stream",
                                {
                                    "config": {
                                        "sample_rate_hz": 24000,
                                        "channels": 1,
                                        "audio_format": "pcm16le",
                                    }
                                },
                            )
                            if not isinstance(probe, Mapping) or probe.get("supported") is not True:
                                raise WorkerProtocolError(
                                    f"worker {worker.name!r} returned an invalid streaming probe result"
                                )
                        metadata["streaming_transcription"] = True
                    self._groups[name] = WorkerGroup(name, workers)
                    self._metadata[name] = metadata
                self._started = True
                self._started_at = int(time.time())
            except BaseException:
                for worker in reversed(started_workers):
                    worker.close()
                self._groups.clear()
                self._metadata.clear()
                self._started = False
                raise

    def close(self) -> None:
        with self._lock:
            self._started = False
            for group in reversed(self._groups.values()):
                group.close()
            self._groups.clear()

    def resolve_model(
        self,
        kind: ModelKind,
        requested_name: str | None,
    ) -> ModelSpec:
        """Resolve an API-visible model without exposing worker transport state."""

        return self._resolve(kind, requested_name)[0]

    def _resolve(
        self,
        kind: ModelKind,
        requested_name: str | None,
    ) -> tuple[ModelSpec, WorkerGroup]:
        if requested_name is None:
            requested_name = (
                self.default_chat_model if kind == "chat" else self.default_transcription_model
            )
        if requested_name is None:
            raise ModelNotFoundError(f"no default {kind} model is configured")

        spec = self._specs.get(requested_name)
        if spec is None:
            raise ModelNotFoundError(f"model {requested_name!r} is not registered")
        if spec.kind != kind:
            raise ModelCapabilityError(
                f"model {requested_name!r} is registered for {spec.kind}, not {kind}"
            )
        with self._lock:
            group = self._groups.get(requested_name)
        if group is None:
            raise ModelNotFoundError(f"model {requested_name!r} is not ready")
        return spec, group

    def acquire_session(
        self,
        kind: ModelKind,
        requested_name: str | None,
    ) -> tuple[ModelSpec, WorkerSession]:
        """Resolve a model and lease one fixed worker lane atomically."""

        with self._lock:
            spec, group = self._resolve(kind, requested_name)
            return spec, group.acquire_session()

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            created = self._started_at or int(time.time())
            return [
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "trtmc",
                    "capabilities": [self._specs[name].kind],
                    "metadata": {
                        key: self._metadata[name][key]
                        for key in _PUBLIC_MODEL_METADATA_FIELDS
                        if key in self._metadata.get(name, {})
                    },
                }
                for name in self._order
            ]

    def status(self) -> dict[str, Any]:
        with self._lock:
            statuses = {
                name: (
                    self._groups[name].status()
                    if name in self._groups
                    else {
                        "state": "not_started",
                        "ready": False,
                        "pid": None,
                        "pids": [],
                        "returncode": None,
                        "error": None,
                        "replicas": self._replicas[name],
                        "ready_replicas": 0,
                        "idle_replicas": 0,
                        "busy": False,
                    }
                )
                for name in self._order
            }
        return {"ready": self.ready, "models": statuses}

    def public_status(self) -> dict[str, Any]:
        """Return the minimal process-independent readiness contract."""

        status = self.status()
        return {
            "ready": status["ready"],
            "models": {
                name: {
                    key: model_status[key]
                    for key in _PUBLIC_MODEL_STATUS_FIELDS
                    if key in model_status
                }
                for name, model_status in status["models"].items()
            },
        }

    def metadata_for(self, name: str) -> dict[str, Any]:
        if name not in self._specs:
            raise ModelNotFoundError(f"model {name!r} is not registered")
        with self._lock:
            return dict(self._metadata.get(name, {}))

    def _make_worker(self, spec: ModelSpec, replica: int) -> WorkerProcess:
        if self._worker_factory is not None:
            return self._worker_factory(spec)
        worker_name = spec.name if self._replicas[spec.name] == 1 else f"{spec.name}-{replica + 1}"
        return WorkerProcess(
            name=worker_name,
            bundle=spec.bundle,
            trtmc_binary=self._trtmc_binary,
            startup_timeout=self._startup_timeout,
            request_timeout=self._request_timeout,
            load_options=self._load_options,
        )

    @staticmethod
    def _validate_metadata(name: str, metadata: Mapping[str, Any]) -> None:
        if metadata.get("protocol_version") != _WORKER_PROTOCOL_VERSION:
            raise WorkerProtocolError(f"worker {name!r} uses an unsupported protocol version")
        runtime_strategy = metadata.get("runtime_strategy")
        default_max_new_tokens = metadata.get("default_max_new_tokens")
        max_cache_length = metadata.get("max_cache_length")
        if not isinstance(runtime_strategy, str):
            raise WorkerProtocolError(f"worker {name!r} metadata.runtime_strategy must be a string")
        if (
            isinstance(default_max_new_tokens, bool)
            or not isinstance(default_max_new_tokens, int)
            or default_max_new_tokens < 0
        ):
            raise WorkerProtocolError(
                f"worker {name!r} metadata.default_max_new_tokens must be a non-negative integer"
            )
        if (
            isinstance(max_cache_length, bool)
            or not isinstance(max_cache_length, int)
            or max_cache_length < 0
        ):
            raise WorkerProtocolError(
                f"worker {name!r} metadata.max_cache_length must be a non-negative integer"
            )

    def _resolve_default(
        self,
        kind: ModelKind,
        requested_name: str | None,
    ) -> str | None:
        available = [spec.name for spec in self._specs.values() if spec.kind == kind]
        if requested_name is None:
            return available[0] if available else None
        spec = self._specs.get(requested_name)
        if spec is None:
            raise ValueError(f"default {kind} model {requested_name!r} is not registered")
        if spec.kind != kind:
            raise ValueError(f"default {kind} model {requested_name!r} has kind {spec.kind!r}")
        return requested_name

    def __enter__(self) -> "ModelRegistry":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

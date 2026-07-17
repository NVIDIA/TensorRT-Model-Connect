# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed task adapter registration for incremental suite migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class TaskAdapter(Protocol):
    """Task-local prepare, fidelity, and measurement-unit contract."""

    kind: str
    version: str

    def prepare(self, work_dir: Path, *, suite_id: str) -> Any: ...

    def fidelity_metrics(
        self,
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        gates: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def measurement_units(self) -> tuple[str, ...]: ...


class UnknownTaskAdapterError(LookupError):
    """Raised when no native adapter owns a requested dataset kind."""


class TaskAdapterRegistry:
    """Local registry with explicit ownership and no implicit fallbacks."""

    def __init__(self) -> None:
        self._adapters: dict[str, TaskAdapter] = {}

    def register(self, adapter: TaskAdapter) -> None:
        kind = str(getattr(adapter, "kind", "")).strip()
        version = str(getattr(adapter, "version", "")).strip()
        if not kind:
            raise ValueError("Task adapter kind must be non-empty")
        if not version:
            raise ValueError(f"Task adapter {kind!r} version must be non-empty")
        missing_methods = [
            name
            for name in ("prepare", "fidelity_metrics", "measurement_units")
            if not callable(getattr(adapter, name, None))
        ]
        if missing_methods:
            raise TypeError(
                f"Task adapter {kind!r} is missing required methods: " + ", ".join(missing_methods)
            )
        if kind in self._adapters:
            raise ValueError(f"Task adapter {kind!r} is already registered")
        self._adapters[kind] = adapter

    def get(self, kind: str) -> TaskAdapter:
        try:
            return self._adapters[kind]
        except KeyError as exc:
            raise UnknownTaskAdapterError(
                f"No native task adapter is registered for {kind!r}"
            ) from exc

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

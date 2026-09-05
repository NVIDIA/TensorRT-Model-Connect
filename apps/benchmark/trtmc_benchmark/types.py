# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small value objects shared by the benchmark application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


COMMAND_DIAGNOSTIC_SCHEMA = "trtmc.command-diagnostic/v1"
COMMAND_DIAGNOSTIC_PREFIX = "TRTMC_DIAGNOSTIC_JSON="


class BenchmarkError(RuntimeError):
    """The requested benchmark cannot be resolved or executed."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        domain: str | None = None,
        code: str | None = None,
        artifacts: tuple[tuple[str, Path], ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.domain = domain
        self.code = code
        self.artifacts = artifacts

    def command_diagnostic(self) -> dict[str, Any] | None:
        if not self.stage or not self.domain or not self.code:
            return None
        return {
            "schema_version": COMMAND_DIAGNOSTIC_SCHEMA,
            "stage": self.stage,
            "domain": self.domain,
            "code": self.code,
            "artifacts": [
                {"label": label, "path": str(path)}
                for label, path in self.artifacts
                if path.is_file()
            ],
        }


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    hf_id: str
    hf_revision: str
    bundle_name: str
    family: str
    task: str
    precision: str
    manifest_path: Path
    testcases: tuple[Mapping[str, Any], ...]
    build_settings: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "hf_id": self.hf_id,
            "bundle_name": self.bundle_name,
            "family": self.family,
            "task": self.task,
            "precision": self.precision,
            "manifest_path": str(self.manifest_path),
            "build": dict(self.build_settings),
        }
        return value


@dataclass(frozen=True)
class MeasurementSpec:
    warmup: int
    iterations: int
    telemetry: str = "auto"
    telemetry_interval_ms: int = 1000
    timing_scope: str = "public_task_call_wall"
    asset_loading_included: bool = False

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise BenchmarkError("measurement.warmup must be non-negative")
        if self.iterations <= 0:
            raise BenchmarkError("measurement.iterations must be positive")
        if self.telemetry not in {"auto", "off"}:
            raise BenchmarkError("telemetry.gpu must be 'auto' or 'off'")
        if self.telemetry_interval_ms < 100:
            raise BenchmarkError("telemetry.interval_ms must be at least 100")
        if self.timing_scope != "public_task_call_wall":
            raise BenchmarkError("measurement.timing_scope must be 'public_task_call_wall'")
        if not isinstance(self.asset_loading_included, bool):
            raise BenchmarkError("measurement.asset_loading_included must be a boolean")

    def to_json(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "iterations": self.iterations,
            "telemetry": self.telemetry,
            "telemetry_interval_ms": self.telemetry_interval_ms,
            "timing_scope": self.timing_scope,
            "asset_loading_included": self.asset_loading_included,
        }


@dataclass(frozen=True)
class ResolvedCase:
    name: str
    model: ModelDescriptor
    testcase_name: str
    bundle_path: Path
    operation: str
    request: Mapping[str, Any]
    runtime_root: Path | None
    measurement: MeasurementSpec
    sources: Mapping[str, str]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "trtmc.benchmark-case/v2",
            "name": self.name,
            "model": self.model.summary(),
            "testcase": self.testcase_name,
            "bundle_path": str(self.bundle_path),
            "operation": self.operation,
            "request": dict(self.request),
            "runtime_root": str(self.runtime_root) if self.runtime_root else "",
            "measurement": self.measurement.to_json(),
            "sources": dict(self.sources),
        }

    def worker_request(self) -> dict[str, Any]:
        if self.runtime_root is None:
            raise BenchmarkError("runtime_root must be explicit before execution")
        model_root = self.model.manifest_path.parent.parent
        return {
            "schema_version": 2,
            "case_name": self.name,
            "bundle": str(self.bundle_path),
            "runtime_root": str(self.runtime_root),
            "operation": self.operation,
            "request": _absolute_artifact_paths(self.request, model_root),
            "measurement": {
                "warmup": self.measurement.warmup,
                "iterations": self.measurement.iterations,
                "timing_scope": self.measurement.timing_scope,
                "asset_loading_included": self.measurement.asset_loading_included,
            },
        }

    def with_values(
        self,
        *,
        name: str | None = None,
        bundle_path: Path | None = None,
        request: Mapping[str, Any] | None = None,
        runtime_root: Path | None = None,
        measurement: MeasurementSpec | None = None,
        sources: Mapping[str, str] | None = None,
    ) -> "ResolvedCase":
        return replace(
            self,
            name=self.name if name is None else name,
            bundle_path=self.bundle_path if bundle_path is None else bundle_path,
            request=self.request if request is None else request,
            runtime_root=self.runtime_root if runtime_root is None else runtime_root,
            measurement=self.measurement if measurement is None else measurement,
            sources=self.sources if sources is None else sources,
        )


def _absolute_artifact_paths(value: Any, model_root: Path) -> Any:
    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        for name, nested in value.items():
            if isinstance(nested, str) and name.endswith("_path"):
                path = Path(nested).expanduser()
                resolved[name] = str(path if path.is_absolute() else (model_root / path).resolve())
            else:
                resolved[name] = _absolute_artifact_paths(nested, model_root)
        return resolved
    if isinstance(value, list):
        return [_absolute_artifact_paths(item, model_root) for item in value]
    return value

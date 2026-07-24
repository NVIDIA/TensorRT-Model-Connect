# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable values shared by the benchmark CLI, resolver, and worker adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class BenchmarkError(RuntimeError):
    """A benchmark request cannot be resolved or executed safely."""


@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    hf_id: str
    hf_revision: str
    bundle_name: str
    family: str
    task_strategy: str
    runtime_strategy: str
    precision: str
    manifest_path: Path
    testcases: tuple[Mapping[str, Any], ...]
    build_settings: Mapping[str, Any]
    distributed_runtime: Mapping[str, Any]

    def identity(self) -> dict[str, Any]:
        try:
            manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkError(f"cannot hash model manifest {self.manifest_path}: {exc}") from exc
        value = {
            "name": self.name,
            "hf_id": self.hf_id,
            "family": self.family,
            "task_strategy": self.task_strategy,
            "runtime_strategy": self.runtime_strategy,
            "precision": self.precision,
            "manifest": f"{self.family}/manifests/{self.manifest_path.name}",
            "manifest_sha256": manifest_sha256,
            "bundle_name": self.bundle_name,
            "build": dict(self.build_settings),
        }
        if self.hf_revision:
            value["hf_revision"] = self.hf_revision
        if self.distributed_runtime:
            value["distributed_runtime"] = dict(self.distributed_runtime)
        return value

    def summary(self) -> dict[str, Any]:
        value = self.identity()
        value["manifest_path"] = str(self.manifest_path)
        return value


@dataclass(frozen=True)
class MeasurementSpec:
    warmup: int
    iterations: int
    telemetry: str = "auto"
    telemetry_interval_ms: int = 1000
    timing_scope: str = "public_pipeline_call_wall"
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
        if self.timing_scope not in {"public_pipeline_call_wall", "model_call_wall"}:
            raise BenchmarkError(
                "measurement.timing_scope must be 'public_pipeline_call_wall' "
                "or 'model_call_wall'"
            )
        if not isinstance(self.asset_loading_included, bool):
            raise BenchmarkError("measurement.asset_loading_included must be a boolean")
        if self.timing_scope == "model_call_wall" and self.asset_loading_included:
            raise BenchmarkError(
                "measurement.asset_loading_included cannot be true for model_call_wall"
            )

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
    runtime: Mapping[str, Any]
    measurement: MeasurementSpec
    sources: Mapping[str, str]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "trtmc.benchmark-case/v1",
            "name": self.name,
            "model": self.model.identity(),
            "testcase": self.testcase_name,
            "bundle_name": self.model.bundle_name,
            "bundle_path": str(self.bundle_path),
            "operation": self.operation,
            "request": dict(self.request),
            "runtime": dict(self.runtime),
            "measurement": self.measurement.to_json(),
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.identity_payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["model"]["manifest_path"] = str(self.model.manifest_path)
        value["resolved_case_digest"] = self.digest
        value["sources"] = dict(self.sources)
        return value

    def worker_request(self) -> dict[str, Any]:
        model_root = self.model.manifest_path.parent.parent
        request = _absolute_artifact_paths(self.request, model_root)
        runtime = _absolute_artifact_paths(self.runtime, model_root)
        return {
            "schema_version": 1,
            "case_name": self.name,
            "case_digest": self.digest,
            "bundle": str(self.bundle_path),
            "operation": self.operation,
            "request": request,
            "runtime": runtime,
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
        runtime: Mapping[str, Any] | None = None,
        measurement: MeasurementSpec | None = None,
        sources: Mapping[str, str] | None = None,
    ) -> ResolvedCase:
        return replace(
            self,
            name=self.name if name is None else name,
            bundle_path=self.bundle_path if bundle_path is None else bundle_path,
            request=self.request if request is None else request,
            runtime=self.runtime if runtime is None else runtime,
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve family-owned manifests into benchmark cases."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .task_adapters import default_operation, resolve_task_case, supported_tasks
from .types import BenchmarkError, MeasurementSpec, ModelDescriptor, ResolvedCase


_OVERRIDE_NAMESPACES = {"request", "measurement", "telemetry"}


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    operation: str
    family: str
    precision: str
    hf_id: str
    status: str
    reason: str = ""
    model: ModelDescriptor | None = None


def default_manifest_root() -> Path:
    catalog = Path(__file__).resolve().parent / "_catalog"
    if not catalog.is_dir():
        raise BenchmarkError(
            f"packaged benchmark catalog does not exist: {catalog}; use --manifest-root"
        )
    return catalog


class ManifestCatalog:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_manifest_root()).expanduser().resolve()

    def _manifest_paths(self) -> tuple[Path, ...]:
        if not self.root.is_dir():
            raise BenchmarkError(f"manifest root does not exist: {self.root}")
        return tuple(sorted(self.root.glob("*/tests/manifests/*.json")))

    def entries(self) -> tuple[CatalogEntry, ...]:
        entries: list[CatalogEntry] = []
        supported = set(supported_tasks())
        for path in self._manifest_paths():
            try:
                model = self._load(path)
            except BenchmarkError as error:
                entries.append(
                    CatalogEntry(path.stem, "-", path.parents[2].name, "-", "-", "invalid", str(error))
                )
                continue
            if model.task not in supported:
                entries.append(
                    CatalogEntry(
                        model.name,
                        "-",
                        model.family,
                        model.precision,
                        model.hf_id or "-",
                        "unsupported",
                        f"task {model.task!r} has no benchmark implementation",
                        model,
                    )
                )
                continue
            operation = default_operation(model.task)
            tp = int(model.build_settings.get("tensor_parallel_size", 1))
            cp = int(model.build_settings.get("context_parallel_size", 1))
            if tp > 1 or cp > 1:
                entries.append(
                    CatalogEntry(
                        model.name,
                        operation,
                        model.family,
                        model.precision,
                        model.hf_id or "-",
                        "distributed",
                        f"requires tensor_parallel_size={tp}, context_parallel_size={cp}",
                        model,
                    )
                )
            else:
                entries.append(
                    CatalogEntry(
                        model.name,
                        operation,
                        model.family,
                        model.precision,
                        model.hf_id or "-",
                        "ready",
                        model=model,
                    )
                )
        return tuple(sorted(entries, key=lambda entry: entry.name))

    def models(self) -> tuple[ModelDescriptor, ...]:
        return tuple(
            entry.model
            for entry in self.entries()
            if entry.status == "ready" and entry.model is not None
        )

    def resolve(self, selector: str) -> ModelDescriptor:
        direct = Path(selector).expanduser()
        if direct.is_file():
            model = self._load(direct.resolve())
            _require_single_process(model)
            return model
        matches = []
        for path in self._manifest_paths():
            try:
                model = self._load(path)
            except BenchmarkError:
                continue
            if selector in {path.stem, model.name, model.hf_id}:
                matches.append(model)
        if not matches:
            raise BenchmarkError(f"unknown model {selector!r} under {self.root}")
        if len(matches) != 1:
            paths = ", ".join(str(model.manifest_path) for model in matches)
            raise BenchmarkError(f"ambiguous model {selector!r}: {paths}")
        _require_single_process(matches[0])
        return matches[0]

    @staticmethod
    def _load(path: Path) -> ModelDescriptor:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkError(f"cannot read model manifest {path}: {error}") from error
        if not isinstance(raw, dict):
            raise BenchmarkError(f"model manifest must be an object: {path}")
        required = {"name", "bundle", "family", "task", "precision", "testcases"}
        missing = sorted(required - raw.keys())
        if missing:
            raise BenchmarkError(f"model manifest {path} is missing: {', '.join(missing)}")
        testcases = raw["testcases"]
        if not isinstance(testcases, list) or not testcases or not all(
            isinstance(value, Mapping) for value in testcases
        ):
            raise BenchmarkError(f"model manifest must contain testcase objects: {path}")
        settings = {
            key: raw[key]
            for key in (
                "max_sequence_length",
                "image_height",
                "image_width",
                "video_num_frames",
                "max_batch_size",
                "tensor_parallel_size",
                "context_parallel_size",
                "quantization",
                "fp32_layers",
            )
            if key in raw
        }
        settings.setdefault("max_batch_size", 1)
        settings.setdefault("tensor_parallel_size", 1)
        settings.setdefault("context_parallel_size", 1)
        return ModelDescriptor(
            name=_string(raw["name"], "name", path),
            hf_id=_optional_string(raw.get("hf_id", ""), "hf_id", path),
            hf_revision=_optional_string(raw.get("hf_revision", ""), "hf_revision", path),
            bundle_name=_string(raw["bundle"], "bundle", path),
            family=_string(raw["family"], "family", path),
            task=_string(raw["task"], "task", path),
            precision=_string(raw["precision"], "precision", path),
            manifest_path=path.resolve(),
            testcases=tuple(testcases),
            build_settings=settings,
        )


def _string(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{field} must be a non-empty string in {path}")
    return value


def _optional_string(value: object, field: str, path: Path) -> str:
    if not isinstance(value, str):
        raise BenchmarkError(f"{field} must be a string in {path}")
    return value


def _require_single_process(model: ModelDescriptor) -> None:
    tp = int(model.build_settings.get("tensor_parallel_size", 1))
    cp = int(model.build_settings.get("context_parallel_size", 1))
    if tp > 1 or cp > 1:
        raise BenchmarkError(
            f"model {model.name!r} is distributed; the benchmark worker is single-process"
        )


def find_bundle(
    model: ModelDescriptor,
    *,
    explicit: Path | None = None,
    roots: Iterable[Path] = (),
) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise BenchmarkError(f"bundle does not exist: {path}")
        return path
    for root in roots:
        candidate = root.expanduser().resolve() / model.bundle_name
        if candidate.is_file():
            return candidate
        nested = root.expanduser().resolve() / model.name / model.bundle_name
        if nested.is_file():
            return nested
    return None


def resolve_case(
    model: ModelDescriptor,
    bundle_path: Path,
    *,
    case_name: str | None = None,
    operation: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedCase:
    testcase = _select_testcase(model, case_name)
    resolution = resolve_task_case(
        model.task,
        testcase,
        model.manifest_path.parent.parent,
        operation=operation,
    )
    request = dict(resolution.request)
    for field, manifest_field in (
        ("height", "image_height"),
        ("width", "image_width"),
        ("num_frames", "video_num_frames"),
    ):
        if not request.get(field) and manifest_field in model.build_settings:
            request[field] = int(model.build_settings[manifest_field])
    if int(request.get("num_frames", 1)) > 1:
        request["media_type"] = "video"
    measurement = resolution.measurement
    sources = dict(resolution.sources)
    for field, value in (overrides or {}).items():
        namespace, separator, name = field.partition(".")
        if not separator or namespace not in _OVERRIDE_NAMESPACES or not name:
            raise BenchmarkError(
                f"override must be request.*, measurement.*, or telemetry.*: {field!r}"
            )
        if namespace == "request":
            request[name] = value
            sources[name] = "benchmark override"
        elif namespace == "measurement":
            measurement = _measurement_update(measurement, name, value)
        else:
            measurement = _measurement_update(
                measurement,
                "telemetry" if name == "gpu" else "telemetry_interval_ms",
                value,
            )
    return ResolvedCase(
        name=str(testcase["name"]),
        model=model,
        testcase_name=str(testcase["name"]),
        bundle_path=bundle_path.expanduser().resolve(),
        operation=resolution.operation,
        request=request,
        runtime_root=None,
        measurement=measurement,
        sources=sources,
    )


def _select_testcase(model: ModelDescriptor, name: str | None) -> Mapping[str, Any]:
    if name is None:
        return model.testcases[0]
    matches = [case for case in model.testcases if case.get("name") == name]
    if len(matches) != 1:
        raise BenchmarkError(f"model {model.name!r} has no unique testcase {name!r}")
    return matches[0]


def _measurement_update(
    measurement: MeasurementSpec, field: str, value: Any
) -> MeasurementSpec:
    fields = {
        "warmup",
        "iterations",
        "timing_scope",
        "asset_loading_included",
        "telemetry",
        "telemetry_interval_ms",
    }
    if field not in fields:
        raise BenchmarkError(f"unknown measurement field {field!r}")
    if field in {"warmup", "iterations", "telemetry_interval_ms"}:
        value = int(value)
    return replace(measurement, **{field: value})


def apply_overrides(
    case: ResolvedCase, overrides: Mapping[str, Any]
) -> ResolvedCase:
    request = dict(case.request)
    measurement = case.measurement
    sources = dict(case.sources)
    for field, value in overrides.items():
        namespace, separator, name = field.partition(".")
        if not separator:
            raise BenchmarkError(f"invalid override {field!r}")
        if namespace == "request":
            request[name] = value
            sources[name] = "benchmark override"
        elif namespace == "measurement":
            measurement = _measurement_update(measurement, name, value)
        elif namespace == "telemetry":
            target = "telemetry" if name == "gpu" else "telemetry_interval_ms"
            measurement = _measurement_update(measurement, target, value)
        else:
            raise BenchmarkError(f"unsupported override namespace {namespace!r}")
    return case.with_values(request=request, measurement=measurement, sources=sources)


def expand_sweeps(
    case: ResolvedCase, sweeps: Mapping[str, list[Any]]
) -> tuple[ResolvedCase, ...]:
    if not sweeps:
        return (case,)
    fields = tuple(sweeps)
    if any(not values for values in sweeps.values()):
        raise BenchmarkError("sweep axes must be non-empty")
    result = []
    for index, values in enumerate(itertools.product(*(sweeps[field] for field in fields)), start=1):
        overrides = dict(zip(fields, values, strict=True))
        resolved = apply_overrides(case, overrides)
        suffix = ",".join(f"{field}={value}" for field, value in overrides.items())
        result.append(resolved.with_values(name=f"{case.name}[{index}:{suffix}]"))
    return tuple(result)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve family-owned manifests into benchmark cases."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from tensorrt_model_connect import family_cli

from .task_adapters import default_operation, resolve_task_case, supported_tasks
from .types import BenchmarkError, MeasurementSpec, ModelDescriptor, ResolvedCase


_OVERRIDE_NAMESPACES = {"request", "measurement", "telemetry"}
_MANIFEST_METADATA_FIELDS = {
    "name", "bundle", "family", "task", "precision", "testcases", "hf_id", "hf_revision",
    "trust_remote_code", "hf_dependencies", "reference_precision", "external_files",
    "base_reference_trust_remote_code", "build",
    "tensor_parallel_size", "context_parallel_size",
}


def family_build_spec(family: str) -> dict[str, Any] | None:
    """Find the selected family's build declaration without importing its builder."""
    try:
        descriptor = family_cli.load_family_cli(family)
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read CLI declaration for {family}: {error}") from error
    if descriptor is None:
        return None
    return next((command for command in descriptor["commands"] if command["name"] == "build"), None)


def _family_build_settings(
    raw: Mapping[str, Any], spec: Mapping[str, Any], path: Path,
) -> dict[str, Any]:
    arguments = {argument["name"]: argument for argument in spec["arguments"]}
    unknown = sorted(raw.keys() - _MANIFEST_METADATA_FIELDS - arguments.keys())
    if unknown:
        raise BenchmarkError(f"undeclared build fields in {path}: {', '.join(unknown)}")
    explicit = raw.get("build", {})
    if not isinstance(explicit, Mapping):
        raise BenchmarkError(f"build must be an object in {path}")
    unknown = sorted(explicit.keys() - arguments.keys())
    if unknown:
        raise BenchmarkError(f"undeclared build fields in {path}: {', '.join(unknown)}")
    bound = {"model", "output", "task", "precision"}
    reserved = sorted(explicit.keys() & bound)
    if reserved:
        raise BenchmarkError(f"build cannot override manifest fields in {path}: {', '.join(reserved)}")
    duplicate = sorted(explicit.keys() & raw.keys())
    if duplicate:
        raise BenchmarkError(f"duplicate build fields in {path}: {', '.join(duplicate)}")
    settings = {name: raw[name] for name in arguments if name not in bound and name in raw}
    settings.update(explicit)
    values = dict(settings)
    context = {
        "model": raw.get("hf_id") or raw["name"], "output": raw["bundle"],
        "task": raw["task"], "precision": raw["precision"],
    }
    values.update((name, value) for name, value in context.items() if name in arguments)
    try:
        family_cli.serialize_arguments(spec, values)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"invalid family build arguments in {path}: {error}") from error
    return settings


def _parallel_sizes(model: ModelDescriptor) -> tuple[int, int]:
    settings = dict(model.build_settings)
    if model.parallelism is not None:
        for name, value in zip(("tensor_parallel_size", "context_parallel_size"), model.parallelism):
            if value is not None:
                settings.setdefault(name, value)
    spec = family_build_spec(model.family)
    if spec is not None:
        for argument in spec["arguments"]:
            if "default" in argument:
                settings.setdefault(argument["name"], argument["default"])
    try:
        return int(settings.get("tensor_parallel_size", 1)), int(settings.get("context_parallel_size", 1))
    except (TypeError, ValueError, OverflowError) as error:
        raise BenchmarkError(f"model {model.name!r} has invalid benchmark parallelism: {error}") from error


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
        self.root = root.expanduser().resolve() if root is not None else None

    def _root(self) -> Path:
        root = self.root or default_manifest_root()
        if not root.is_dir():
            raise BenchmarkError(f"manifest root does not exist: {root}")
        return root

    def _manifest_paths(self) -> tuple[Path, ...]:
        return tuple(sorted(self._root().glob("*/tests/manifests/*.json")))

    def entries(self) -> tuple[CatalogEntry, ...]:
        entries: list[CatalogEntry] = []
        supported = set(supported_tasks())
        for path in self._manifest_paths():
            try:
                model = self._load(path)
                task = selected_task_for_case(model)
                if task in supported:
                    tp, cp = _parallel_sizes(model)
            except BenchmarkError as error:
                entries.append(
                    CatalogEntry(
                        path.stem, "-", path.parents[2].name, "-", "-", "invalid", str(error)
                    )
                )
                continue
            if task not in supported:
                entries.append(
                    CatalogEntry(
                        model.name,
                        "-",
                        model.family,
                        model.precision,
                        model.hf_id or "-",
                        "unsupported",
                        f"task {task!r} has no benchmark implementation",
                        model,
                    )
                )
                continue
            operation = default_operation(task)
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
            raise BenchmarkError(f"unknown model {selector!r} under {self._root()}")
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
        if (
            not isinstance(testcases, list)
            or not testcases
            or not all(isinstance(value, Mapping) for value in testcases)
        ):
            raise BenchmarkError(f"model manifest must contain testcase objects: {path}")
        parallelism = tuple(raw.get(name) for name in ("tensor_parallel_size", "context_parallel_size"))
        for name in ("tensor_parallel_size", "context_parallel_size"):
            if name in raw and (type(raw[name]) is not int or raw[name] < 1):
                raise BenchmarkError(f"{name} must be a positive integer in {path}")
        spec = family_build_spec(_string(raw["family"], "family", path))
        if spec is not None:
            settings = _family_build_settings(raw, spec, path)
        else:
            if "build" in raw:
                raise BenchmarkError(f"family {raw['family']!r} does not declare build arguments")
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
                    "backend",
                    "dynamic_kv_cache",
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
            parallelism=parallelism if any(size is not None for size in parallelism) else None,
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
    tp, cp = _parallel_sizes(model)
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
    selected_task: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedCase:
    testcase = _select_testcase(model, case_name)
    task = selected_task_for_case(model, case_name, selected_task=selected_task)
    resolution = resolve_task_case(
        task,
        testcase,
        model.manifest_path.parent.parent,
        operation=operation,
    )
    explicit_task = selected_task is not None or "selected_task" in testcase
    if explicit_task and resolution.task != task:
        raise BenchmarkError("operation cannot change an explicitly selected Task")
    request = dict(resolution.request)
    if resolution.task in {
        "image_generation", "image_edit", "image_generation_batch", "world_model_generation",
    }:
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
    resolved = ResolvedCase(
        name=str(testcase["name"]),
        model=model,
        testcase_name=str(testcase["name"]),
        bundle_path=bundle_path.expanduser().resolve(),
        operation=resolution.operation,
        request=request,
        runtime_root=None,
        measurement=measurement,
        sources=sources,
        selected_task=resolution.task if explicit_task or resolution.task != model.task else None,
    )
    return apply_overrides(resolved, overrides or {})


def selected_task_for_case(
    model: ModelDescriptor, case_name: str | None = None, *, selected_task: str | None = None,
) -> str:
    testcase = _select_testcase(model, case_name)
    task = testcase.get("selected_task", model.task) if selected_task is None else selected_task
    if not isinstance(task, str) or not task or task != task.strip():
        raise BenchmarkError("selected_task must be a nonempty Task ID without surrounding whitespace")
    return task


def _select_testcase(model: ModelDescriptor, name: str | None) -> Mapping[str, Any]:
    if name is None:
        return model.testcases[0]
    matches = [case for case in model.testcases if case.get("name") == name]
    if len(matches) != 1:
        raise BenchmarkError(f"model {model.name!r} has no unique testcase {name!r}")
    return matches[0]


def _measurement_update(measurement: MeasurementSpec, field: str, value: Any) -> MeasurementSpec:
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


def apply_overrides(case: ResolvedCase, overrides: Mapping[str, Any]) -> ResolvedCase:
    request = dict(case.request)
    measurement = case.measurement
    sources = dict(case.sources)
    for field, value in overrides.items():
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
        elif namespace == "telemetry":
            targets = {"gpu": "telemetry", "interval_ms": "telemetry_interval_ms"}
            if name not in targets:
                raise BenchmarkError(f"unknown telemetry field {name!r}")
            target = targets[name]
            measurement = _measurement_update(measurement, target, value)
    return case.with_values(request=request, measurement=measurement, sources=sources)


def expand_sweeps(case: ResolvedCase, sweeps: Mapping[str, list[Any]]) -> tuple[ResolvedCase, ...]:
    if not sweeps:
        return (case,)
    fields = tuple(sweeps)
    if any(not values for values in sweeps.values()):
        raise BenchmarkError("sweep axes must be non-empty")
    result = []
    for index, values in enumerate(
        itertools.product(*(sweeps[field] for field in fields)), start=1
    ):
        overrides = dict(zip(fields, values, strict=True))
        resolved = apply_overrides(case, overrides)
        suffix = ",".join(f"{field}={value}" for field, value in overrides.items())
        result.append(resolved.with_values(name=f"{case.name}[{index}:{suffix}]"))
    return tuple(result)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve existing TRTMC model manifests into executable benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from .operations import operation_for_name
from .task_adapters import adapter_for_task_strategy
from .types import BenchmarkError, MeasurementSpec, ModelDescriptor, ResolvedCase

_OVERRIDE_NAMESPACES = {"request", "runtime", "measurement", "telemetry"}


@dataclass(frozen=True)
class CatalogEntry:
    """One manifest exposed by ``trtmc-bench list models``."""

    name: str
    operation: str
    family: str
    precision: str
    hf_id: str
    status: str
    reason: str = ""
    model: ModelDescriptor | None = None


def default_manifest_root() -> Path:
    configured = os.environ.get("TRTMC_BENCH_MANIFEST_ROOT")
    if configured:
        return Path(configured).expanduser()
    module = Path(__file__).resolve()
    candidates = (
        module.parents[3] / "tests/e2e/models",
        module.parent / "_catalog",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise BenchmarkError(f"cannot locate the TRTMC model catalog; searched: {searched}")


class ManifestCatalog:
    """Production-side reader for the repository's existing E2E manifests."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_manifest_root()).expanduser().resolve()

    def models(self) -> tuple[ModelDescriptor, ...]:
        """Return single-process catalog models executable by the benchmark."""
        return tuple(
            entry.model
            for entry in self.entries()
            if entry.status == "ready" and entry.model is not None
        )

    def entries(self) -> tuple[CatalogEntry, ...]:
        """Return every declared model manifest with an explicit support status."""
        if not self.root.is_dir():
            raise BenchmarkError(f"model manifest root does not exist: {self.root}")
        entries: list[CatalogEntry] = []
        for path in self._manifest_paths():
            try:
                model = self._load(path)
            except BenchmarkError as exc:
                entries.append(
                    CatalogEntry(
                        name=path.stem,
                        operation="-",
                        family=path.parent.parent.name,
                        precision="-",
                        hf_id="-",
                        status="invalid",
                        reason=str(exc),
                    )
                )
                continue
            try:
                adapter = adapter_for_task_strategy(model.task_strategy)
            except BenchmarkError as exc:
                entries.append(_catalog_entry(model, "unsupported", str(exc), operation="-"))
                continue
            config = model.distributed_runtime
            if config.get("enabled"):
                launcher = str(config.get("launcher", "mpirun") or "mpirun")
                world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
                entries.append(
                    _catalog_entry(
                        model,
                        "distributed",
                        f"requires {launcher}, world_size={world_size}",
                        operation=adapter.operation,
                    )
                )
                continue
            try:
                adapter.resolve_case(model.testcases[0], model.manifest_path.parent.parent)
            except BenchmarkError as exc:
                entries.append(
                    _catalog_entry(model, "invalid", str(exc), operation=adapter.operation)
                )
                continue
            if all(
                testcase.get("test_category", "e2e") == "regression"
                for testcase in model.testcases
            ):
                entries.append(
                    _catalog_entry(
                        model,
                        "regression",
                        "requires explicit regression selection",
                        operation=adapter.operation,
                    )
                )
                continue
            entries.append(_catalog_entry(model, "ready", operation=adapter.operation))
        return tuple(sorted(entries, key=lambda item: (item.name, item.hf_id)))

    def _manifest_paths(self) -> tuple[Path, ...]:
        """Read the same manifest declarations used by the E2E model catalog."""
        paths: list[Path] = []
        for family_root in sorted(path for path in self.root.iterdir() if path.is_dir()):
            descriptor = family_root / "MODEL.toml"
            if not descriptor.is_file():
                paths.extend(sorted((family_root / "manifests").glob("*.json")))
                continue
            try:
                with descriptor.open("rb") as stream:
                    raw = tomllib.load(stream)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise BenchmarkError(f"cannot read model descriptor {descriptor}: {exc}") from exc
            declared = raw.get("test_manifests")
            if not isinstance(declared, list) or not all(
                isinstance(item, str) and item for item in declared
            ):
                raise BenchmarkError(
                    f"model descriptor {descriptor} must declare test_manifests strings"
                )
            paths.extend((family_root / item).resolve() for item in declared)
        return tuple(paths)

    def resolve(self, selector: str) -> ModelDescriptor:
        direct = Path(selector).expanduser()
        if direct.is_file():
            model = self._load(direct.resolve())
            _require_supported_model(model)
            return model
        if not self.root.is_dir():
            raise BenchmarkError(f"model manifest root does not exist: {self.root}")
        matches: list[ModelDescriptor] = []
        for path in self._manifest_paths():
            try:
                model = self._load(path)
            except BenchmarkError:
                continue
            if selector in {model.name, model.hf_id}:
                matches.append(model)
        if not matches:
            raise BenchmarkError(f"unknown model {selector!r} under {self.root}")
        identities = {(item.name, item.hf_id) for item in matches}
        if len(identities) > 1 or len(matches) > 1:
            paths = ", ".join(str(item.manifest_path) for item in matches)
            raise BenchmarkError(f"ambiguous model {selector!r}: {paths}")
        model = matches[0]
        _require_supported_model(model)
        return model

    @staticmethod
    def _load(path: Path) -> ModelDescriptor:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"cannot read model manifest {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise BenchmarkError(f"model manifest must be a JSON object: {path}")
        required = {
            "name",
            "hf_id",
            "bundle",
            "family",
            "task_strategy",
            "runtime_strategy",
            "testcases",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise BenchmarkError(f"model manifest {path} is missing: {', '.join(missing)}")
        testcases = raw["testcases"]
        if not isinstance(testcases, list) or not testcases:
            raise BenchmarkError(f"model manifest has no testcases: {path}")
        if not all(isinstance(item, dict) for item in testcases):
            raise BenchmarkError(f"model manifest testcases must be objects: {path}")
        test_categories = [
            testcase.get("test_category", "e2e") for testcase in testcases
        ]
        invalid_categories = sorted(
            repr(category)
            for category in test_categories
            if not isinstance(category, str) or category not in {"e2e", "regression"}
        )
        if invalid_categories:
            raise BenchmarkError(
                f"model manifest {path} has unsupported test categories: "
                f"{invalid_categories}"
            )
        distributed_runtime = raw.get("distributed_runtime", {})
        if not isinstance(distributed_runtime, Mapping):
            raise BenchmarkError(f"distributed_runtime in model manifest {path} must be an object")
        hf_revision = raw.get("hf_revision", "")
        if not isinstance(hf_revision, str):
            raise BenchmarkError(f"hf_revision in model manifest {path} must be a string")
        task_strategy = str(raw["task_strategy"])
        model_defaults = _model_defaults(path, task_strategy)
        build_settings = {
            key: model_defaults[key] for key in ("build_cli_args",) if key in model_defaults
        }
        build_settings.update(
            {
                key: raw[key]
                for key in (
                    "build_args",
                    "build_cli_args",
                    "build_env",
                    "build_timeout_s",
                    "fp8_scales",
                    "fp32_layers",
                    "max_batch_size",
                    "max_cache_length",
                    "quantization",
                    "trust_remote_code",
                )
                if key in raw
            }
        )
        if "fp8_scales" in build_settings:
            _resolve_manifest_asset(path, "fp8_scales", build_settings["fp8_scales"])
        return ModelDescriptor(
            name=str(raw["name"]),
            hf_id=str(raw["hf_id"]),
            hf_revision=hf_revision.strip(),
            bundle_name=str(raw["bundle"]),
            family=str(raw["family"]),
            task_strategy=task_strategy,
            runtime_strategy=str(raw["runtime_strategy"]),
            precision=str(raw.get("precision", "fp32")),
            manifest_path=path,
            testcases=tuple(testcases),
            build_settings=build_settings,
            distributed_runtime=dict(distributed_runtime),
        )


def _resolve_manifest_asset(path: Path, field: str, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field} in model manifest {path} must be a non-empty path")
    declared = Path(value).expanduser()
    candidate = declared if declared.is_absolute() else path.parent.parent / declared
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise BenchmarkError(
            f"{field} file declared by model manifest {path} is missing: {resolved}"
        )
    return resolved


def _require_supported_model(model: ModelDescriptor) -> None:
    adapter = adapter_for_task_strategy(model.task_strategy)
    config = model.distributed_runtime
    if config.get("enabled"):
        launcher = str(config.get("launcher", "mpirun") or "mpirun")
        world_size = int(config.get("world_size", config.get("tp_size", 2)) or 2)
        raise BenchmarkError(
            f"model profile {model.name!r} requires distributed execution "
            f"({launcher}, world_size={world_size}), but trtmc-bench currently supports "
            "single-process benchmark workers only"
        )
    adapter.resolve_case(model.testcases[0], model.manifest_path.parent.parent)


def _catalog_entry(
    model: ModelDescriptor,
    status: str,
    reason: str = "",
    *,
    operation: str | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        name=model.name,
        operation=operation or adapter_for_task_strategy(model.task_strategy).operation,
        family=model.family,
        precision=model.precision,
        hf_id=model.hf_id,
        status=status,
        reason=reason,
        model=model,
    )


def _model_defaults(path: Path, task_strategy: str) -> Mapping[str, Any]:
    descriptor = path.parent.parent / "MODEL.toml"
    if not descriptor.is_file():
        return {}
    try:
        with descriptor.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkError(f"cannot read model descriptor {descriptor}: {exc}") from exc
    defaults = raw.get("e2e_defaults", {})
    if not isinstance(defaults, Mapping):
        return {}
    selected = defaults.get(task_strategy, {})
    return selected if isinstance(selected, Mapping) else {}


def find_bundle(
    model: ModelDescriptor,
    *,
    explicit: Path | None = None,
    roots: Iterable[Path] = (),
) -> Path | None:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise BenchmarkError(f"bundle does not exist: {candidate}")
        return candidate
    search_roots = [Path(item).expanduser().resolve() for item in roots]
    configured = os.environ.get("TRTMC_BENCH_BUNDLE_ROOT")
    if configured:
        search_roots.extend(
            Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item
        )
    direct = [root / model.bundle_name for root in search_roots]
    for candidate in direct:
        if candidate.is_file():
            return candidate
    recursive: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            recursive.extend(path.resolve() for path in root.rglob(model.bundle_name))
    unique = sorted(set(recursive))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        paths = ", ".join(str(path) for path in unique)
        raise BenchmarkError(
            f"multiple bundles match {model.bundle_name}; use --bundle or YAML bundle: {paths}"
        )
    return None


def resolve_bundle(
    model: ModelDescriptor,
    *,
    explicit: Path | None = None,
    roots: Iterable[Path] = (),
) -> Path:
    bundle = find_bundle(model, explicit=explicit, roots=roots)
    if bundle is not None:
        return bundle
    search_roots = [Path(item).expanduser().resolve() for item in roots]
    configured = os.environ.get("TRTMC_BENCH_BUNDLE_ROOT")
    if configured:
        search_roots.extend(
            Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item
        )
    roots_text = ", ".join(str(path) for path in search_roots) or "<none>"
    raise BenchmarkError(
        f"cannot find {model.bundle_name} in bundle roots {roots_text}; "
        "set --bundle, --bundle-root, or TRTMC_BENCH_BUNDLE_ROOT"
    )


def _select_testcase(model: ModelDescriptor, name: str | None) -> Mapping[str, Any]:
    if name is None or name == "default":
        return model.testcases[0]
    matches = [item for item in model.testcases if str(item.get("name")) == name]
    if not matches:
        available = ", ".join(str(item.get("name")) for item in model.testcases)
        raise BenchmarkError(f"model {model.name} has no case {name!r}; available: {available}")
    if len(matches) != 1:
        raise BenchmarkError(f"model {model.name} has duplicate testcase {name!r}")
    return matches[0]


def resolve_case(
    model: ModelDescriptor,
    bundle: Path,
    *,
    case_name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedCase:
    adapter = adapter_for_task_strategy(model.task_strategy)
    testcase = _select_testcase(model, case_name)
    testcase_name = str(testcase.get("name", "default"))
    request_resolution = adapter.resolve_case(testcase, model.manifest_path.parent.parent)
    request = dict(request_resolution.request)
    runtime: dict[str, Any] = {"cuda_graphs": False, **request_resolution.runtime}
    measurement = adapter.default_measurement
    sources: dict[str, str] = {
        **{f"request.{key}": source for key, source in request_resolution.request_sources.items()},
        "runtime.cuda_graphs": "benchmark default",
        **{f"runtime.{key}": source for key, source in request_resolution.runtime_sources.items()},
        "measurement.warmup": "task strategy default",
        "measurement.iterations": "task strategy default",
        "telemetry.gpu": "benchmark default",
        "telemetry.interval_ms": "benchmark default",
    }
    resolved = ResolvedCase(
        name=case_name or testcase_name,
        model=model,
        testcase_name=testcase_name,
        bundle_path=bundle,
        operation=adapter.operation,
        request=request,
        runtime=runtime,
        measurement=measurement,
        sources=sources,
    )
    return apply_overrides(resolved, overrides or {})


def apply_overrides(case: ResolvedCase, overrides: Mapping[str, Any]) -> ResolvedCase:
    request = dict(case.request)
    runtime = dict(case.runtime)
    measurement_values = case.measurement.to_json()
    sources = dict(case.sources)
    for dotted, value in overrides.items():
        if not isinstance(dotted, str) or "." not in dotted:
            raise BenchmarkError(f"override must use namespace.field: {dotted!r}")
        namespace, field = dotted.split(".", 1)
        if namespace not in _OVERRIDE_NAMESPACES:
            raise BenchmarkError(f"unsupported override namespace {namespace!r}")
        if namespace == "request":
            request[field] = value
        elif namespace == "runtime":
            runtime[field] = value
        elif namespace == "measurement":
            measurement_values[field] = value
        elif namespace == "telemetry":
            mapped = {"gpu": "telemetry", "interval_ms": "telemetry_interval_ms"}.get(field)
            if mapped is None:
                raise BenchmarkError(f"unsupported telemetry override {field!r}")
            measurement_values[mapped] = value
        sources[dotted] = "user override"
    request = _with_artifact_digests(request, case.model.manifest_path.parent.parent)
    for path_name in (name for name in request if name.endswith("_path")):
        digest_name = f"{path_name.removesuffix('_path')}_sha256"
        if digest_name not in request:
            continue
        path_source = sources.get(f"request.{path_name}", "resolved request")
        sources[f"request.{digest_name}"] = f"derived from {path_source}"
    measurement = MeasurementSpec(
        warmup=int(measurement_values["warmup"]),
        iterations=int(measurement_values["iterations"]),
        telemetry=str(measurement_values["telemetry"]),
        telemetry_interval_ms=int(measurement_values["telemetry_interval_ms"]),
        timing_scope=str(measurement_values["timing_scope"]),
        asset_loading_included=measurement_values["asset_loading_included"],
    )
    batch_size = request.get("batch_size", 1)
    if type(batch_size) is not int or batch_size <= 0:
        raise BenchmarkError("request.batch_size must be a positive integer")
    if batch_size != 1 and not operation_for_name(case.operation).supports_batch:
        raise BenchmarkError(
            f"operation {case.operation!r} supports request.batch_size=1 only in the public "
            "pipeline API; the benchmark never simulates batching with sequential requests"
        )
    return case.with_values(
        request=request,
        runtime=runtime,
        measurement=measurement,
        sources=sources,
    )


def _with_artifact_digests(request: Mapping[str, Any], model_root: Path) -> dict[str, Any]:
    resolved = dict(request)
    for name, value in tuple(resolved.items()):
        if not name.endswith("_path") or not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        source = path if path.is_absolute() else model_root / path
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as exc:
            raise BenchmarkError(f"cannot read request artifact {source}: {exc}") from exc
        resolved[f"{name.removesuffix('_path')}_sha256"] = digest
    return resolved


def expand_sweeps(
    case: ResolvedCase,
    axes: Mapping[str, list[Any]],
    *,
    max_cells: int = 64,
) -> tuple[ResolvedCase, ...]:
    if not axes:
        return (case,)
    names = list(axes)
    values = [axes[name] for name in names]
    if any(not axis for axis in values):
        raise BenchmarkError("sweep axes cannot be empty")
    count = 1
    for axis in values:
        count *= len(axis)
    if count > max_cells:
        raise BenchmarkError(f"sweep expands to {count} cells; maximum is {max_cells}")
    cells: list[ResolvedCase] = []
    for combination in itertools.product(*values):
        overrides = dict(zip(names, combination, strict=True))
        suffix = "-".join(
            f"{_short_field(name)}-{_slug(value)}" for name, value in overrides.items()
        )
        resolved = apply_overrides(case, overrides)
        cells.append(replace(resolved, name=f"{case.name}-{suffix}"))
    return tuple(cells)


def _short_field(name: str) -> str:
    field = name.rsplit(".", 1)[-1]
    aliases = {"batch_size": "b", "num_inference_steps": "s", "iterations": "n"}
    return aliases.get(field, field)


def _slug(value: Any) -> str:
    text = str(value).lower()
    return re.sub(r"[^a-z0-9_.-]+", "-", text).strip("-") or "value"

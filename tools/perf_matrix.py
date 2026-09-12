#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and report the TRTMC release performance matrix."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
from array import array
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
BUILDER_SOURCE = REPOSITORY / "core/builder"
BENCHMARK_SOURCE = REPOSITORY / "apps/benchmark"
MANIFEST_ROOT = REPOSITORY / "families"
for source in (REPOSITORY, BUILDER_SOURCE, BENCHMARK_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from apps.benchmark.performance.baselines.timing_contracts import timing_contract  # noqa: E402
from apps.benchmark.performance.baselines.hf_transformers import flatten_config  # noqa: E402
from trtmc_benchmark.catalog import ManifestCatalog, resolve_case, selected_task_for_case  # noqa: E402
from trtmc_benchmark.task_adapters import default_operation  # noqa: E402
from trtmc_benchmark.types import BenchmarkError  # noqa: E402


SUITE_SCHEMA = "trtmc.perf-suite/v2"
ENVIRONMENT_SCHEMA = "trtmc.perf-environment/v2"
RESULT_SCHEMA = "trtmc.perf-matrix/v2"
REPORT_SCHEMA = "trtmc.perf-report/v2"
PREPARATION_SCHEMA = "trtmc.perf-bundle-preparation/v2"
TERMINAL_COMPARISONS = {"green", "yellow", "red"}
FINISHED_RESULTS = TERMINAL_COMPARISONS | {"contract-mismatch"}
HF_CACHE_ENVIRONMENT_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_ASSETS_CACHE",
    "HF_MODULES_CACHE",
    "HF_XET_CACHE",
)
OUTPUT_CONTRACTS = {
    "audio-shape",
    "classification-top-class",
    "disparity-parity",
    "embedding-shape",
    "exact-text",
    "exact-token-ids",
    "forecast-shape",
    "head-scores-shape",
    "regression-distribution",
    "regression-values",
    "generated-token-count",
    "image-features-shape",
    "localization",
    "media-shape",
    "metric-geometry-shape",
    "molecular-structure-shape",
    "normalized-text",
    "ocr-text",
    "offline-speech-shape",
    "pose-refinement-shape",
    "reranking-order",
    "robot-action-shape",
    "segmentation-shape",
    "transcription-text",
}
REFERENCE_INPUTS = {
    "pytorch-lerobot-act": (("source_root", "lerobot_repo"),),
    "upstream-elf": (("reference_repo", "elf_repo"),),
    "upstream-lance": (("reference_repo", "lance_repo"),),
    "upstream-sana-wm": (
        ("reference_repo", "sana_repo"),
        ("model_dir", "sana_model"),
    ),
    "pytorch-personaplex": (("official_repo", "personaplex_repo"),),
    "upstream-fast-foundation-stereo": (("model_dir", "fast_foundation_stereo_model"),),
}
REFERENCE_FIELDS = {
    "elf_repo",
    "lance_repo",
    "lerobot_repo",
    "sana_repo",
    "sana_model",
    "personaplex_repo",
    "fast_foundation_stereo_model",
}


class PerfMatrixError(RuntimeError):
    pass


@dataclass(frozen=True)
class Environment:
    name: str
    trtmc_bench: Path
    worker: Path
    hf_runner: Path
    task_runner: Path
    results_root: Path
    scratch_root: Path
    bundle_cache: Path
    bundle_roots: tuple[Path, ...]
    runtime_root: Path
    bundle_retention: str
    local_files_only: bool
    timeout_seconds: int
    references: Mapping[str, str]
    storage_root: Path | None = None
    hf_cache_mode: str = "shared"
    hf_cache_retention: str = "retain"


@dataclass(frozen=True)
class ResolvedEntry:
    spec: Mapping[str, Any]
    model: Any
    case: Any
    manifest: Mapping[str, Any]
    reference_precision: str
    baseline_timing: Mapping[str, Any]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("suite", type=Path)
        command.add_argument("--environment", required=True, type=Path)
        command.add_argument("--entry", action="append", default=[])
        command.add_argument("--model", action="append", default=[])
        command.add_argument("--model-selection", type=Path)
        command.add_argument("--verbose", action="store_true")
        if name == "prepare":
            command.add_argument("--output", required=True, type=Path)
        if name == "run":
            command.add_argument("--no-build", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("run_directory", type=Path)
    resume.add_argument("--verbose", action="store_true")
    resume.add_argument("--no-build", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("run_directory", type=Path)
    report.add_argument("--preparation-receipt", type=Path)
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PerfMatrixError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PerfMatrixError(f"{label} must contain an object")
    return value


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for name, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(name), Mapping):
            result[name] = _deep_merge(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


def load_suite(path: Path) -> tuple[str, list[dict[str, Any]], set[str]]:
    owner = (
        path.parent.parent.name
        if path.name == "performance.yaml" and path.parent.name == "tests"
        and path.parent.parent.parent.resolve() == MANIFEST_ROOT.resolve()
        else None
    )
    if owner is not None:
        _family_file(owner, "tests/performance.yaml", "performance suite")
    name, entries, excluded = _load_suite_file(path, owner=owner)
    canonical = REPOSITORY / "apps/benchmark/performance/release.yaml"
    if path.resolve() != canonical.resolve():
        return name, entries, excluded
    ids = {entry["id"] for entry in entries}
    for family_suite in sorted(MANIFEST_ROOT.glob("*/tests/performance.yaml")):
        owner = family_suite.parents[1].name
        _family_file(owner, "tests/performance.yaml", "performance suite")
        _, owned_entries, _ = _load_suite_file(family_suite, owner=owner)
        for entry in owned_entries:
            if entry["id"] in ids:
                raise PerfMatrixError(f"duplicate suite entry {entry['id']!r}")
            ids.add(entry["id"])
            entries.append(entry)
    return name, entries, excluded


def _load_suite_file(
    path: Path, *, owner: str | None = None,
) -> tuple[str, list[dict[str, Any]], set[str]]:
    raw = _read_yaml(path.resolve(), "performance suite")
    if owner is not None and "excluded_profiles" in raw:
        raise PerfMatrixError(f"family suite {owner!r} cannot declare exclusions")
    if raw.get("schema_version") != SUITE_SCHEMA:
        raise PerfMatrixError(f"suite schema_version must be {SUITE_SCHEMA}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise PerfMatrixError("suite name must be non-empty")
    defaults = raw.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise PerfMatrixError("suite defaults must be an object")
    default_measurement = defaults.get("measurement", {})
    default_baseline = defaults.get("baseline", {})
    margin = float(defaults.get("equivalence_margin_percent", 5.0))
    if not isinstance(default_measurement, Mapping) or not isinstance(default_baseline, Mapping):
        raise PerfMatrixError("suite measurement and baseline defaults must be objects")

    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise PerfMatrixError("suite entries must be a non-empty list")
    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for value in entries_raw:
        if not isinstance(value, Mapping):
            raise PerfMatrixError("each suite entry must be an object")
        entry = deepcopy(dict(value))
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise PerfMatrixError("each suite entry requires an id")
        if entry_id in by_id:
            raise PerfMatrixError(f"duplicate suite entry {entry_id!r}")
        entry["measurement"] = _deep_merge(default_measurement, entry.get("measurement", {}))
        entry["baseline"] = _deep_merge(default_baseline, entry.get("baseline", {}))
        entry.setdefault("equivalence_margin_percent", margin)
        _validate_entry(entry)
        by_id[entry_id] = entry
        entries.append(entry)

    additional = raw.get("additional_profiles", [])
    if not isinstance(additional, list):
        raise PerfMatrixError("additional_profiles must be a list")
    for value in additional:
        if not isinstance(value, Mapping):
            raise PerfMatrixError("each additional profile must be an object")
        parent_id = value.get("inherit")
        model = value.get("model")
        if not isinstance(parent_id, str) or parent_id not in by_id:
            raise PerfMatrixError(f"additional profile inherits unknown entry {parent_id!r}")
        if not isinstance(model, str) or not model:
            raise PerfMatrixError("additional profile requires a model")
        update = {key: item for key, item in value.items() if key != "inherit"}
        entry = _deep_merge(by_id[parent_id], update)
        entry["id"] = f"{parent_id}@{model}"
        workload_update = value.get("workload", {})
        if not isinstance(workload_update, Mapping) or "testcase" not in workload_update:
            entry.setdefault("workload", {})["testcase"] = model
        if entry["id"] in by_id:
            raise PerfMatrixError(f"duplicate suite entry {entry['id']!r}")
        _validate_entry(entry)
        by_id[entry["id"]] = entry
        entries.append(entry)

    excluded_raw = raw.get("excluded_profiles", [])
    if not isinstance(excluded_raw, list):
        raise PerfMatrixError("excluded_profiles must be a list")
    excluded: set[str] = set()
    for value in excluded_raw:
        if not isinstance(value, Mapping) or not isinstance(value.get("model"), str):
            raise PerfMatrixError("excluded profile entries require model and reason")
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise PerfMatrixError(f"excluded profile {value['model']} requires a reason")
        excluded.add(value["model"])
    if owner is not None:
        for entry in entries:
            _validate_family_entry(entry, owner)
    return name, entries, excluded


def _family_file(family: str, relative: str, label: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise PerfMatrixError(f"{label} must be a relative path inside its family")
    if (not isinstance(family, str) or not family or family in {".", ".."}
            or "/" in family or "\\" in family):
        raise PerfMatrixError(f"{label} has an invalid family owner")
    root = MANIFEST_ROOT / family
    path = root
    for part in ("", *Path(relative).parts):
        path = path / part
        if path.is_symlink():
            raise PerfMatrixError(f"{label} cannot use symlinks")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise PerfMatrixError(f"{label} is missing or outside its family: {relative}")
    return resolved


def _family_script(entry: Mapping[str, Any]) -> Path:
    script = entry["baseline"].get("script")
    if not isinstance(script, str) or Path(script).suffix != ".py":
        raise PerfMatrixError(f"entry {entry['id']} requires a family-local .py script")
    return _family_file(str(entry["family"]), script, "reference script")


def _validate_family_entry(entry: Mapping[str, Any], owner: str) -> None:
    if entry["family"] != owner:
        raise PerfMatrixError(f"family suite {owner!r} cannot declare another family")
    try:
        model = ManifestCatalog(MANIFEST_ROOT).resolve(str(entry["model"]))
        selected_task_for_case(model, str(entry["workload"]["testcase"]))
    except BenchmarkError as error:
        raise PerfMatrixError(f"entry {entry['id']} has no owned model/testcase: {error}") from error
    manifests = (MANIFEST_ROOT / owner / "tests/manifests").resolve()
    if model.family != owner or not model.manifest_path.is_relative_to(manifests):
        raise PerfMatrixError(f"family suite {owner!r} cannot use another family's manifest")


def _validate_entry(entry: Mapping[str, Any]) -> None:
    for field in ("id", "family", "operation", "model"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise PerfMatrixError(f"entry requires non-empty {field}")
    workload = entry.get("workload")
    baseline = entry.get("baseline")
    measurement = entry.get("measurement")
    if not isinstance(workload, Mapping) or not isinstance(workload.get("testcase"), str):
        raise PerfMatrixError(f"entry {entry['id']} requires workload.testcase")
    if not isinstance(baseline, Mapping) or baseline.get("runner") not in {
        "hf-transformers",
        "task-reference",
    }:
        raise PerfMatrixError(f"entry {entry['id']} has an invalid baseline runner")
    if "script" in baseline:
        if baseline["runner"] != "task-reference" or "adapter" in baseline:
            raise PerfMatrixError(f"entry {entry['id']} script requires task-reference without adapter")
        _family_script(entry)
    elif baseline["runner"] == "task-reference" and not isinstance(baseline.get("adapter"), str):
        raise PerfMatrixError(f"entry {entry['id']} requires baseline.adapter")
    if not isinstance(measurement, Mapping):
        raise PerfMatrixError(f"entry {entry['id']} requires measurement")
    warmup = measurement.get("warmup")
    iterations = measurement.get("iterations")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise PerfMatrixError(f"entry {entry['id']} warmup must be non-negative")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise PerfMatrixError(f"entry {entry['id']} iterations must be positive")


def _expand(value: str, field: str) -> str:
    try:
        expanded = Template(value).substitute(os.environ)
    except KeyError as error:
        raise PerfMatrixError(f"environment {field} requires {error.args[0]}") from error
    if not expanded.strip():
        raise PerfMatrixError(f"environment {field} must be non-empty")
    return expanded


def _path(value: str, field: str) -> Path:
    expanded = Path(_expand(value, field)).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (REPOSITORY / expanded).resolve()


def _path_list(value: Any, field: str) -> tuple[Path, ...]:
    if isinstance(value, str):
        try:
            expanded = Template(value).substitute(os.environ)
        except KeyError as error:
            raise PerfMatrixError(f"environment {field} requires {error.args[0]}") from error
        values = [item for item in expanded.split(os.pathsep) if item]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = [_expand(item, field) for item in value]
    else:
        raise PerfMatrixError(f"environment {field} must be a path list")
    return tuple(_path(item, field) for item in values)


def load_environment(path: Path) -> Environment:
    raw = _read_yaml(path.resolve(), "performance environment")
    if raw.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise PerfMatrixError(f"environment schema_version must be {ENVIRONMENT_SCHEMA}")
    tools = raw.get("tools")
    storage = raw.get("storage")
    execution = raw.get("execution")
    references = raw.get("references")
    if not all(isinstance(value, Mapping) for value in (tools, storage, execution, references)):
        raise PerfMatrixError(
            "environment tools, storage, execution, and references must be objects"
        )
    assert isinstance(tools, Mapping)
    assert isinstance(storage, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(references, Mapping)
    required_tools = {
        "trtmc_bench",
        "trtmc_worker",
        "hf_transformers_runner",
        "task_reference_runner",
    }
    required_storage = {
        "results_root",
        "scratch_root",
        "bundle_cache",
        "bundle_roots",
        "runtime_root",
    }
    if missing := sorted(required_tools - tools.keys()):
        raise PerfMatrixError("environment tools is missing: " + ", ".join(missing))
    if missing := sorted(required_storage - storage.keys()):
        raise PerfMatrixError("environment storage is missing: " + ", ".join(missing))
    if missing := sorted(REFERENCE_FIELDS - references.keys()):
        raise PerfMatrixError("environment references is missing: " + ", ".join(missing))
    if not all(isinstance(references[name], str) for name in REFERENCE_FIELDS):
        raise PerfMatrixError("environment reference inputs must be strings")
    timeout = execution.get("timeout_seconds", 7200)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise PerfMatrixError("execution.timeout_seconds must be positive")
    retention = storage.get("bundle_retention", "retain")
    if retention not in {"retain", "delete_on_pass", "delete_always"}:
        raise PerfMatrixError("bundle_retention must be retain, delete_on_pass, or delete_always")
    storage_root = storage.get("storage_root")
    if storage_root is not None and (not isinstance(storage_root, str) or not storage_root.strip()):
        raise PerfMatrixError("storage_root must be a path or null")
    hf_cache_mode = execution.get("hf_cache_mode", "shared")
    if hf_cache_mode not in {"shared", "per_entry"}:
        raise PerfMatrixError("hf_cache_mode must be shared or per_entry")
    hf_cache_retention = execution.get("hf_cache_retention", "retain")
    if hf_cache_retention not in {"retain", "delete_on_pass", "delete_always"}:
        raise PerfMatrixError("hf_cache_retention must be retain, delete_on_pass, or delete_always")
    if hf_cache_mode == "shared" and hf_cache_retention != "retain":
        raise PerfMatrixError("a shared Hugging Face cache can only be retained")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise PerfMatrixError("environment name must be non-empty")
    return Environment(
        name=name,
        trtmc_bench=_path(str(tools["trtmc_bench"]), "tools.trtmc_bench"),
        worker=_path(str(tools["trtmc_worker"]), "tools.trtmc_worker"),
        hf_runner=_path(str(tools["hf_transformers_runner"]), "tools.hf_transformers_runner"),
        task_runner=_path(str(tools["task_reference_runner"]), "tools.task_reference_runner"),
        results_root=_path(str(storage["results_root"]), "storage.results_root"),
        scratch_root=_path(str(storage["scratch_root"]), "storage.scratch_root"),
        bundle_cache=_path(str(storage["bundle_cache"]), "storage.bundle_cache"),
        bundle_roots=_path_list(storage["bundle_roots"], "storage.bundle_roots"),
        runtime_root=_path(str(storage["runtime_root"]), "storage.runtime_root"),
        bundle_retention=retention,
        local_files_only=bool(execution.get("local_files_only", False)),
        timeout_seconds=timeout,
        references={name: str(references[name]) for name in REFERENCE_FIELDS},
        storage_root=(
            _path(storage_root, "storage.storage_root") if storage_root is not None else None
        ),
        hf_cache_mode=str(hf_cache_mode),
        hf_cache_retention=str(hf_cache_retention),
    )


def _selection_families(path: Path | None) -> set[str]:
    if path is None:
        return set()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PerfMatrixError(f"cannot read model selection {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise PerfMatrixError("model selection must contain an object")
    families = value.get("families", [])
    if not isinstance(families, list) or not all(isinstance(item, str) for item in families):
        raise PerfMatrixError("model selection families must be strings")
    return set(families)


def select_entries(
    entries: Sequence[dict[str, Any]],
    *,
    entry_ids: Sequence[str] = (),
    models: Sequence[str] = (),
    model_selection: Path | None = None,
) -> list[dict[str, Any]]:
    modes = sum(bool(value) for value in (entry_ids, models, model_selection))
    if modes > 1:
        raise PerfMatrixError("entry, model, and model-selection are mutually exclusive")
    if entry_ids:
        requested = set(entry_ids)
        selected = [entry for entry in entries if entry["id"] in requested]
        missing = requested - {entry["id"] for entry in selected}
        if missing:
            raise PerfMatrixError("unknown entries: " + ", ".join(sorted(missing)))
        return selected
    if models:
        requested = set(models)
        selected = [entry for entry in entries if entry["model"] in requested]
        missing = requested - {entry["model"] for entry in selected}
        if missing:
            raise PerfMatrixError("unknown models: " + ", ".join(sorted(missing)))
        return selected
    if model_selection is not None:
        families = _selection_families(model_selection)
        if not families:
            raise PerfMatrixError("model selection matches no release entries")
        selected = [entry for entry in entries if entry["family"] in families]
        if not selected:
            raise PerfMatrixError("model selection matches no release entries")
        return selected
    return list(entries)


def _coverage(entries: Sequence[Mapping[str, Any]], excluded: set[str]) -> None:
    catalog_entries = ManifestCatalog(MANIFEST_ROOT).entries()
    ready = {
        entry.name
        for entry in catalog_entries
        if entry.status == "ready"
        and "-l0" not in f"-{entry.name}-"
        and "-regression-" not in f"-{entry.name}-"
    }
    covered = {str(entry["model"]) for entry in entries}
    missing = sorted(ready - covered - excluded)
    unknown = sorted(excluded - ready)
    repeated = sorted(covered & excluded)
    if missing:
        raise PerfMatrixError("release suite omits ready models: " + ", ".join(missing))
    if unknown:
        raise PerfMatrixError("release suite excludes unknown models: " + ", ".join(unknown))
    if repeated:
        raise PerfMatrixError("models are both configured and excluded: " + ", ".join(repeated))


def resolve_entries(
    entries: Sequence[Mapping[str, Any]], environment: Environment
) -> list[ResolvedEntry]:
    catalog = ManifestCatalog(MANIFEST_ROOT)
    resolved: list[ResolvedEntry] = []
    for spec in entries:
        try:
            model = catalog.resolve(str(spec["model"]))
            if model.family != spec["family"]:
                raise PerfMatrixError(
                    f"entry {spec['id']} family {spec['family']} does not match {model.family}"
                )
            workload = spec["workload"]
            operation = default_operation(selected_task_for_case(model, str(workload["testcase"])))
            if (spec["operation"], operation) in {("generate", "translate"), ("solve", "regress")}:
                # A semantic primary may separate a formerly overloaded operation.
                # Keep the release entry identity and all measurement thresholds.
                spec = {**spec, "operation": operation}
            overrides = {
                f"request.{name}": value for name, value in workload.get("request", {}).items()
            }
            timing = timing_contract(runner=str(spec["baseline"]["runner"]), declared=spec["baseline"])
            baseline_timing = _baseline_timing(timing)
            overrides.update(
                {
                    "measurement.warmup": int(spec["measurement"]["warmup"]),
                    "measurement.iterations": int(spec["measurement"]["iterations"]),
                    "measurement.timing_scope": timing["candidate_timing_scope"],
                    "measurement.asset_loading_included": baseline_timing["asset_loading_included"],
                    "telemetry.gpu": "off",
                }
            )
            bundle = environment.bundle_cache / model.name / model.bundle_name
            case = resolve_case(
                model,
                bundle,
                case_name=str(workload["testcase"]),
                operation=str(spec["operation"]),
                overrides=overrides,
            ).with_values(runtime_root=environment.runtime_root)
            manifest = json.loads(model.manifest_path.read_text(encoding="utf-8"))
            reference_precision = _reference_precision(spec, case.testcase_name, manifest, model)
        except (BenchmarkError, OSError, ValueError, json.JSONDecodeError) as error:
            raise PerfMatrixError(f"cannot resolve {spec['id']}: {error}") from error
        resolved.append(
            ResolvedEntry(spec, model, case, manifest, reference_precision, baseline_timing)
        )
    return resolved


def _reference_precision(
    spec: Mapping[str, Any],
    testcase_name: str,
    manifest: Mapping[str, Any],
    model: Any,
) -> str:
    if spec["baseline"].get("precision"):
        return str(spec["baseline"]["precision"])
    for testcase in manifest.get("testcases", []):
        if isinstance(testcase, Mapping) and testcase.get("name") == testcase_name:
            if testcase.get("reference_precision"):
                return str(testcase["reference_precision"])
    if manifest.get("reference_precision"):
        return str(manifest["reference_precision"])
    return str(model.precision)


def _baseline_timing(declared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timing_scope": declared["timing_scope"],
        "input_preparation_included": declared["input_preparation_included"],
        "asset_loading_included": declared["asset_loading_included"],
    }


def preflight(
    suite_entries: Sequence[Mapping[str, Any]],
    environment: Environment,
    *,
    require_runtime: bool,
) -> list[ResolvedEntry]:
    files = (
        ("trtmc-bench", environment.trtmc_bench),
        ("HF reference runner", environment.hf_runner),
        ("task reference runner", environment.task_runner),
    )
    if require_runtime:
        files += (("TRTMC worker", environment.worker),)
    for label, path in files:
        if not path.is_file():
            raise PerfMatrixError(f"{label} does not exist: {path}")
    executables = [("trtmc-bench", environment.trtmc_bench)]
    if require_runtime:
        executables.append(("TRTMC worker", environment.worker))
    for label, path in executables:
        if not os.access(path, os.X_OK):
            raise PerfMatrixError(f"{label} is not executable: {path}")
    if require_runtime:
        if not environment.runtime_root.is_dir():
            raise PerfMatrixError(f"runtime_root does not exist: {environment.runtime_root}")
        for library in ("libtrtmc_runtime.so", "libtrtmc_backend_trt.so"):
            if not (environment.runtime_root / library).is_file():
                raise PerfMatrixError(f"runtime_root is missing {library}")
    if environment.storage_root is not None:
        if not environment.storage_root.is_dir():
            raise PerfMatrixError(f"storage_root does not exist: {environment.storage_root}")
        for label, path in (
            ("results_root", environment.results_root),
            ("scratch_root", environment.scratch_root),
            ("bundle_cache", environment.bundle_cache),
        ):
            if not path.is_relative_to(environment.storage_root):
                raise PerfMatrixError(
                    f"{label} must stay below storage_root {environment.storage_root}: {path}"
                )
    environment.results_root.mkdir(parents=True, exist_ok=True)
    environment.scratch_root.mkdir(parents=True, exist_ok=True)
    environment.bundle_cache.mkdir(parents=True, exist_ok=True)
    resolved = resolve_entries(suite_entries, environment)
    for entry in resolved:
        _contract_name(entry)
    if require_runtime:
        for entry in resolved:
            family_library = environment.runtime_root / f"libtrtmc_model_{entry.model.family}.so"
            if not family_library.is_file():
                raise PerfMatrixError(f"runtime_root is missing {family_library.name}")
            baseline_command(
                entry,
                environment,
                environment.scratch_root / f"{_entry_slug(str(entry.spec['id']))}.reference.json",
            )
    return resolved


def _candidate_base(entry: ResolvedEntry, environment: Environment) -> list[str]:
    arguments = [
        str(environment.trtmc_bench),
        "run",
        "--model",
        entry.model.name,
        "--case",
        entry.case.testcase_name,
        "--operation",
        str(entry.spec["operation"]),
        "--manifest-root",
        str(MANIFEST_ROOT),
        "--bundle-cache",
        str(environment.bundle_cache),
    ]
    if entry.case.selected_task is not None:
        arguments.extend(("--task", entry.case.selected_task))
    for root in environment.bundle_roots:
        arguments.extend(("--bundle-root", str(root)))
    arguments.extend(
        (
            "--warmup",
            str(entry.case.measurement.warmup),
            "--iterations",
            str(entry.case.measurement.iterations),
            "--telemetry",
            "off",
            "--set",
            "measurement.timing_scope=public_task_call_wall",
            "--set",
            "measurement.asset_loading_included="
            + ("true" if entry.case.measurement.asset_loading_included else "false"),
        )
    )
    return arguments


def candidate_command(
    entry: ResolvedEntry,
    environment: Environment,
    output: Path | None,
    *,
    prepare_only: bool = False,
    no_build: bool = False,
) -> list[str]:
    arguments = _candidate_base(entry, environment)
    if prepare_only:
        arguments.append("--prepare-only")
        return arguments
    arguments.extend(
        (
            "--runtime-root",
            str(environment.runtime_root),
            "--worker",
            str(environment.worker),
        )
    )
    if no_build:
        arguments.append("--no-build")
    if output is None:
        raise AssertionError("candidate output is required")
    arguments.extend(("--output", str(output)))
    return arguments


def _baseline_task(entry: ResolvedEntry) -> str:
    if configured := entry.spec["baseline"].get("task"):
        return str(configured)
    if entry.spec["operation"] in {"encode", "embed"}:
        return "encoder"
    return "causal-lm"


def _adapter_options(entry: ResolvedEntry, environment: Environment) -> dict[str, Any]:
    configured = entry.spec["baseline"].get("adapter_options", {})
    if not isinstance(configured, Mapping):
        raise PerfMatrixError(f"entry {entry.spec['id']} adapter_options must be an object")
    options = dict(configured)
    inputs = REFERENCE_INPUTS.get(str(entry.spec["baseline"].get("adapter", "")), ())
    for option_name, field in inputs:
        path = _path(environment.references[field], f"references.{field}")
        if not path.is_dir():
            raise PerfMatrixError(
                f"entry {entry.spec['id']} reference path is not a directory: {path}"
            )
        _validate_reference_path(entry, field, path)
        options[option_name] = str(path)
    return options


def _validate_reference_path(entry: ResolvedEntry, field: str, path: Path) -> None:
    required = {
        "elf_repo": ("src",),
        "lance_repo": ("inference_lance.py",),
        "lerobot_repo": ("lerobot/common/policies/act/modeling_act.py",),
        "personaplex_repo": ("moshi",),
        "fast_foundation_stereo_model": (
            "core/foundation_stereo.py",
            "core/submodule.py",
            "weights/23-36-37/model_best_bp2_serialize.pth",
        ),
    }.get(field, ())
    missing = [relative for relative in required if not (path / relative).exists()]
    if missing:
        raise PerfMatrixError(
            f"entry {entry.spec['id']} reference path {path} is missing: " + ", ".join(missing)
        )


def _baseline_mode(baseline: Mapping[str, Any]) -> str:
    return str(baseline.get("mode", "reference" if "script" in baseline else "torch-compile"))


def baseline_command(entry: ResolvedEntry, environment: Environment, output: Path) -> list[str]:
    baseline = entry.spec["baseline"]
    runner = str(baseline["runner"])
    request = json.dumps(flatten_config(entry.case.request), ensure_ascii=True, separators=(",", ":"))
    common = [
        "--model",
        str(_adapter_options(entry, environment).get("model_id", entry.model.hf_id)),
        "--request-json",
        request,
        "--precision",
        entry.reference_precision,
        "--mode",
        _baseline_mode(baseline),
        "--warmup",
        str(entry.case.measurement.warmup),
        "--iterations",
        str(entry.case.measurement.iterations),
        "--case-name",
        str(entry.spec["id"]),
        "--output",
        str(output),
    ]
    if runner == "hf-transformers":
        arguments = [
            sys.executable,
            str(environment.hf_runner),
            "--task",
            str(baseline.get("task", _baseline_task(entry))),
            "--max-length",
            str(entry.model.build_settings.get("max_sequence_length", 256)),
            "--padding",
            str(baseline.get("padding", "longest")),
            "--output-token-policy",
            str(baseline.get("output_token_policy", "new-tokens")),
            *common,
        ]
        if baseline.get("model_class"):
            arguments.extend(("--model-class", str(baseline["model_class"])))
        if baseline.get("generation_method"):
            arguments.extend(("--generation-method", str(baseline["generation_method"])))
        if baseline.get("experts_implementation"):
            arguments.extend(("--experts-implementation", str(baseline["experts_implementation"])))
        if baseline.get("mode") == "torch-compile":
            arguments.extend(("--compile-mode", str(baseline.get("compile_mode", "default"))))
            if bool(baseline.get("fullgraph", False)):
                arguments.append("--compile-fullgraph")
            if bool(baseline.get("dynamic", True)):
                arguments.append("--compile-dynamic")
    else:
        executable = (
            [str(_family_script(entry.spec))]
            if "script" in baseline
            else [str(environment.task_runner), "--adapter", str(baseline["adapter"])]
        )
        arguments = [
            sys.executable,
            *executable,
            "--family",
            entry.model.family,
            "--operation",
            str(entry.spec["operation"]),
            "--manifest",
            str(entry.model.manifest_path),
            "--adapter-options-json",
            json.dumps(
                _adapter_options(entry, environment),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            "--timing-contract-json",
            json.dumps(entry.baseline_timing, ensure_ascii=True, separators=(",", ":")),
            "--padding",
            str(baseline.get("padding", "longest")),
            *common,
        ]
        if "script" in baseline or entry.case.selected_task is not None:
            arguments.extend(("--selected-task", _effective_task(entry)))
    revision = entry.model.hf_revision
    if revision:
        arguments.extend(("--revision", revision))
    if bool(entry.manifest.get("trust_remote_code", False)):
        arguments.append("--trust-remote-code")
    if bool(baseline.get("local_files_only", environment.local_files_only)):
        arguments.append("--local-files-only")
    return arguments


def _validate_script_result(
    entry: ResolvedEntry, environment: Environment, result: Mapping[str, Any],
) -> None:
    expected = {
        "schema_version": "trtmc.perf-baseline/v1",
        "status": "completed",
        "model": str(_adapter_options(entry, environment).get("model_id", entry.model.hf_id)),
        "family": entry.model.family,
        "operation": str(entry.spec["operation"]),
        "case_name": str(entry.spec["id"]),
        "selected_task": _effective_task(entry),
        "precision": entry.reference_precision,
        "mode": _baseline_mode(entry.spec["baseline"]),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise PerfMatrixError(f"family reference result has mismatched {field}")
    measurement = result.get("measurement")
    for field in ("warmup", "iterations"):
        if (not isinstance(measurement, Mapping) or type(measurement.get(field)) is not int
                or measurement[field] != getattr(entry.case.measurement, field)):
            raise PerfMatrixError(f"family reference result has mismatched {field}")
    policy = result.get("measurement_policy")
    for field, value in entry.baseline_timing.items():
        if (not isinstance(policy, Mapping) or result.get(field) != value
                or policy.get(field) != value
                or type(result.get(field)) is not type(value)
                or type(policy.get(field)) is not type(value)):
            raise PerfMatrixError(f"family reference result has mismatched {field}")
    samples = result.get("samples_ms")
    if (not isinstance(samples, list) or len(samples) != entry.case.measurement.iterations
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in samples)):
        raise PerfMatrixError("family reference must return one finite positive sample per iteration")
    if _p50(result) != statistics.median(samples):
        raise PerfMatrixError("family reference latency p50 does not match its samples")
    if not isinstance(result.get("output_summary"), Mapping):
        raise PerfMatrixError("family reference result has no output summary")


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    paths = (str(BUILDER_SOURCE), str(BENCHMARK_SOURCE), str(REPOSITORY))
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join((*paths, existing) if existing else paths)
    return environment


def _entry_command_environment(environment: Environment, work: Path) -> dict[str, str]:
    values = _command_environment()
    if environment.hf_cache_mode == "per_entry":
        values["HF_HOME"] = str((work / "hf-cache").resolve())
        for name in HF_CACHE_ENVIRONMENT_NAMES[1:]:
            values.pop(name, None)
    return values


def run_command(
    arguments: Sequence[str],
    *,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    verbose: bool,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(shlex.join(_reported_arguments(arguments)), flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=REPOSITORY,
            env=dict(env) if env is not None else _command_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        code = 124
        stdout = _stream_text(error.stdout)
        stderr = _stream_text(error.stderr) + f"\ncommand timed out after {timeout}s\n"
    except OSError as error:
        code = 127
        stdout = ""
        stderr = str(error)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "argv": _reported_arguments(arguments),
        "cwd": str(REPOSITORY),
        "exit_code": code,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _reported_arguments(arguments: Sequence[str]) -> list[str]:
    result = list(arguments)
    if "--revision" in result:
        index = result.index("--revision") + 1
        if index < len(result):
            result[index] = "<model-revision>"
    return result


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PerfMatrixError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PerfMatrixError(f"{label} must contain an object")
    return value


def _candidate_result(path: Path) -> dict[str, Any]:
    result = _json_file(path / "result.json", "candidate result")
    cells = result.get("cells")
    if (
        result.get("schema_version") != "trtmc.benchmark-run/v2"
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], Mapping)
    ):
        raise PerfMatrixError("candidate result has an invalid schema")
    cell = dict(cells[0])
    if cell.get("status") != "completed":
        raise PerfMatrixError(str(cell.get("error", "candidate failed")))
    return {
        "metrics": cell.get("metrics", {}),
        "samples_ms": cell.get("samples_ms", []),
        "output_summary": cell.get("output_summary", {}),
        "timing_scope": cell.get("timing_scope"),
        "asset_loading_included": cell.get("asset_loading_included"),
        "preparation": result.get("preparation", {}),
    }


def _p50(value: Mapping[str, Any]) -> float:
    metrics = value.get("metrics", {})
    latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
    p50 = latency.get("p50") if isinstance(latency, Mapping) else None
    if isinstance(p50, bool) or not isinstance(p50, (int, float)) or not math.isfinite(float(p50)):
        raise PerfMatrixError("measurement has no finite latency p50")
    return float(p50)


_TIMING_STABILITY_SAMPLE_COUNT = 10
_TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT = 5.0
_TIMING_STABILITY_MEDIAN_BAND_PERCENT = 5.0
_TIMING_STABILITY_MIN_IN_BAND = 8


def _timing_stability(values: Sequence[Any]) -> dict[str, Any]:
    if len(values) != _TIMING_STABILITY_SAMPLE_COUNT:
        return {
            "status": "not_evaluated",
            "sample_count": len(values),
            "reason": "requires_10_samples",
        }
    try:
        samples = [float(value) for value in values]
    except (TypeError, ValueError):
        return {
            "status": "not_evaluated",
            "sample_count": len(values),
            "reason": "invalid_samples",
        }
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        return {
            "status": "not_evaluated",
            "sample_count": len(samples),
            "reason": "invalid_samples",
        }

    middle = len(samples) // 2
    median_ms = float(statistics.median(samples))
    first_half_median_ms = float(statistics.median(samples[:middle]))
    second_half_median_ms = float(statistics.median(samples[middle:]))
    half_change_percent = (
        abs(second_half_median_ms - first_half_median_ms) / first_half_median_ms * 100.0
    )
    samples_within_band = sum(
        abs(sample - median_ms) / median_ms * 100.0 <= _TIMING_STABILITY_MEDIAN_BAND_PERCENT
        for sample in samples
    )
    stable = (
        half_change_percent <= _TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT
        and samples_within_band >= _TIMING_STABILITY_MIN_IN_BAND
    )
    return {
        "status": "stable" if stable else "unstable",
        "sample_count": len(samples),
        "median_ms": median_ms,
        "first_half_median_ms": first_half_median_ms,
        "second_half_median_ms": second_half_median_ms,
        "half_median_change_percent": half_change_percent,
        "samples_within_band": samples_within_band,
    }


def _measurement_stability(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_result = _timing_stability(reference.get("samples_ms", []))
    candidate_result = _timing_stability(candidate.get("samples_ms", []))
    statuses = {reference_result["status"], candidate_result["status"]}
    if statuses == {"stable"}:
        status = "stable"
    elif "unstable" in statuses:
        status = "unstable"
    else:
        status = "not_evaluated"
    return {
        "status": status,
        "policy": {
            "required_samples": _TIMING_STABILITY_SAMPLE_COUNT,
            "max_half_median_change_percent": _TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT,
            "median_band_percent": _TIMING_STABILITY_MEDIAN_BAND_PERCENT,
            "minimum_samples_within_band": _TIMING_STABILITY_MIN_IN_BAND,
        },
        "reference": reference_result,
        "candidate": candidate_result,
    }


def compare(
    entry: ResolvedEntry,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    mismatch = _timing_mismatch(entry, candidate, baseline)
    if mismatch:
        return "contract-mismatch", {"reason": mismatch}
    matched, reason, evidence = _output_contract(entry, candidate, baseline)
    if not matched:
        value: dict[str, Any] = {"reason": reason}
        if evidence:
            value["output_contract"] = evidence
        return "contract-mismatch", value
    candidate_p50 = _p50(candidate)
    reference_p50 = _p50(baseline)
    ratio = reference_p50 / candidate_p50
    margin = float(entry.spec.get("equivalence_margin_percent", 5.0)) / 100.0
    if ratio > 1.0 + margin:
        status = "green"
    elif ratio < 1.0 - margin:
        status = "red"
    else:
        status = "yellow"
    value = {
        "candidate_p50_ms": candidate_p50,
        "reference_p50_ms": reference_p50,
        "reference_over_candidate_p50": ratio,
        "equivalence_margin_percent": margin * 100.0,
    }
    if evidence:
        value["output_contract"] = evidence
    return status, value


def _timing_mismatch(
    entry: ResolvedEntry,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    if candidate.get("timing_scope") != "public_task_call_wall":
        return "candidate timing scope is not public_task_call_wall"
    if candidate.get("asset_loading_included") is not bool(
        entry.case.measurement.asset_loading_included
    ):
        return "candidate asset-loading policy differs from the suite"
    policy = baseline.get("measurement_policy", {})
    if not isinstance(policy, Mapping):
        return "reference timing policy is missing"
    for name, expected in entry.baseline_timing.items():
        if policy.get(name) != expected:
            return f"reference {name} differs from the suite"
    if baseline.get("precision") != entry.reference_precision:
        return "reference precision differs from the suite"
    return ""


def _effective_task(entry: ResolvedEntry) -> str:
    return entry.model.task if entry.case.selected_task is None else entry.case.selected_task


def _contract_name(entry: ResolvedEntry) -> str:
    configured = entry.spec["baseline"].get("output_contract")
    if configured:
        contract = str(configured)
    elif entry.spec["operation"] in {"generate", "translate"}:
        if float(flatten_config(entry.case.request).get("temperature", 0.0)) > 0.0:
            contract = "generated-token-count"
        else:
            contract = "exact-token-ids"
    else:
        contract = {
            "classify": "classification-top-class",
            "embed": "embedding-shape",
            "encode": "embedding-shape",
            "head_scores": "head-scores-shape",
            "geometry": "metric-geometry-shape",
            "predict_structure": "molecular-structure-shape",
            "refine_pose": "pose-refinement-shape",
            "control": "robot-action-shape",
            "segment": "segmentation-shape",
            "solve": "forecast-shape",
            "regress": ("regression-values" if _effective_task(entry) == "series_to_regression_values"
                        else "regression-distribution"),
            "transcribe": "transcription-text",
            "speech_dialogue": (
                "offline-speech-shape" if _effective_task(entry) == "offline_speech_dialogue"
                else ""
            ),
        }.get(str(entry.spec["operation"]), "")
    if contract not in OUTPUT_CONTRACTS:
        raise PerfMatrixError(
            f"entry {entry.spec['id']} has unsupported output contract: {contract or '<none>'}"
        )
    return contract


def _output_contract(
    entry: ResolvedEntry,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    left = candidate.get("output_summary", {})
    right = baseline.get("output_summary", {})
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False, "output summary is missing", None
    contract = _contract_name(entry)
    if contract == "exact-token-ids":
        matched = left.get("token_ids") == right.get("token_ids")
        return matched, "generated token ids differ" if not matched else "", None
    if contract == "generated-token-count":
        left_count = _token_count(left)
        right_count = _token_count(right)
        matched = left_count is not None and left_count == right_count
        return matched, "generated token count differs" if not matched else "", None
    if contract == "exact-text":
        matched = left.get("text") == right.get("text")
        return matched, "generated text differs" if not matched else "", None
    if contract == "normalized-text":
        matched = _normalized_text(left.get("text")) == _normalized_text(right.get("text"))
        return matched, "normalized generated text differs" if not matched else "", None
    if contract == "transcription-text":
        left_text = _normalized_text(left.get("text"))
        right_text = _normalized_text(right.get("text"))
        matched = bool(left_text) and left_text == right_text
        return matched, "normalized transcription text differs" if not matched else "", None
    if contract == "ocr-text":
        required = [str(value) for value in entry.spec["baseline"].get("required_substrings", [])]
        left_text = _normalized_text(left.get("text"))
        right_text = _normalized_text(right.get("text"))
        if any(_normalized_text(value) not in left_text for value in required):
            return False, "candidate OCR text misses required content", None
        if any(_normalized_text(value) not in right_text for value in required):
            return False, "reference OCR text misses required content", None
        distance = _text_distance(left_text, right_text)
        limit = float(entry.spec["baseline"].get("max_normalized_edit_distance", 0.5))
        return (
            distance <= limit,
            "OCR text distance exceeds the contract",
            {
                "normalized_edit_distance": distance,
                "maximum": limit,
            },
        )
    if contract == "localization":
        return _localization_contract(entry, left, right)
    if contract in {"metric-geometry-shape", "molecular-structure-shape", "pose-refinement-shape"}:
        signature = {
            "metric-geometry-shape": _metric_geometry_signature,
            "molecular-structure-shape": _molecular_structure_signature,
            "pose-refinement-shape": _pose_refinement_signature,
        }[contract]
        evidence = {"contract": contract, "numerical_parity_checked": False}
        try:
            first = signature(left)
        except (ValueError, OSError) as error:
            return False, f"candidate {contract}: {error}", evidence
        try:
            second = signature(right)
        except (ValueError, OSError) as error:
            return False, f"reference {contract}: {error}", evidence
        matched = first == second
        return matched, f"{contract} representation differs" if not matched else "", evidence
    if contract == "offline-speech-shape":
        return _offline_speech_contract(entry, left, right)
    if contract == "audio-shape":
        left_shape = (
            left.get("num_samples", left.get("audio_samples")),
            left.get("sample_rate"),
        )
        right_shape = (
            right.get("num_samples", right.get("audio_samples")),
            right.get("sample_rate"),
        )
        matched = None not in left_shape and left_shape == right_shape
        return matched, "audio output shape differs" if not matched else "", None
    if contract == "media-shape":
        left_shape = _media_shape(left)
        right_shape = _media_shape(right)
        matched = None not in left_shape and left_shape == right_shape
        return matched, "media output shape differs" if not matched else "", None
    if contract == "segmentation-shape":
        left_shape = tuple(left.get(name) for name in ("num_masks", "height", "width"))
        right_shape = tuple(right.get(name) for name in ("num_masks", "height", "width"))
        matched = None not in left_shape and left_shape == right_shape
        return matched, "segmentation output shape differs" if not matched else "", None
    if contract == "classification-top-class":
        left_class = left.get("top_class")
        right_class = right.get("top_class")
        matched = (
            isinstance(left_class, int)
            and not isinstance(left_class, bool)
            and left_class == right_class
        )
        return matched, "classification top class differs" if not matched else "", None
    if contract == "image-features-shape":
        left_shape = (
            left.get("last_hidden_state_shape"),
            left.get("pooler_output_shape"),
        )
        right_shape = (
            right.get("last_hidden_state_shape"),
            right.get("pooler_output_shape"),
        )
        matched = left_shape == right_shape and all(value for value in left_shape)
        return matched, "image feature output shape differs" if not matched else "", None
    if contract == "reranking-order":
        left_scores = left.get("scores")
        right_scores = right.get("scores")
        if not isinstance(left_scores, list) or not isinstance(right_scores, list):
            return False, "reranking scores are missing", None
        left_order = sorted(range(len(left_scores)), key=lambda index: (-left_scores[index], index))
        right_order = sorted(
            range(len(right_scores)), key=lambda index: (-right_scores[index], index)
        )
        matched = left_order == right_order
        return matched, "reranking order differs" if not matched else "", None
    if contract == "robot-action-shape":
        names = ("action_steps", "action_dim", "action_values")
        left_shape = tuple(left.get(name) for name in names)
        right_shape = tuple(right.get(name) for name in names)
        matched = (
            None not in left_shape
            and left_shape == right_shape
            and left.get("within_training_bounds") is True
            and right.get("finite") is True
        )
        return matched, "robot action output contract differs" if not matched else "", None
    if contract == "disparity-parity":
        evidence = _disparity(entry, left, right)
        return bool(evidence["passed"]), str(evidence.get("reason", "")), evidence
    if contract == "embedding-shape":
        left_elements = left.get("element_count", left.get("embedding_elements"))
        right_elements = right.get("element_count", right.get("embedding_elements"))
        matched = left_elements == right_elements and left.get("dim") == right.get("dim")
        return matched, "embedding output shape differs" if not matched else "", None
    if contract == "head-scores-shape":
        def signature(value: Mapping[str, Any]) -> tuple[Any, ...] | None:
            shape, values = value.get("shape"), value.get("values")
            if (not isinstance(shape, list) or not shape
                    or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                           for size in shape)
                    or not isinstance(values, list) or len(values) != math.prod(shape)):
                return None
            if any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in values):
                return None
            kind = value.get("score_kind")
            if not isinstance(kind, str) or kind not in {"logit", "probability", "unbounded"}:
                return None
            metadata = (value.get("pooling"), value.get("normalization"))
            if not all(isinstance(item, str) and item for item in metadata):
                return None
            return tuple(shape), kind, *metadata
        first, second = signature(left), signature(right)
        matched = first is not None and first == second
        return matched, "head score shape or representation differs" if not matched else "", None
    if contract == "forecast-shape":
        left_elements = left.get("forecast_elements", left.get("element_count"))
        right_elements = right.get("forecast_elements", right.get("element_count"))
        left_shape = left.get("shape")
        right_shape = right.get("shape")
        task = _effective_task(entry)
        semantic = task in {"series_to_point_forecast", "series_to_quantile_forecast"}
        expected_axes = (["horizon", "channel"] if task == "series_to_point_forecast"
                         else ["quantile", "horizon", "channel"])
        shape_matches = (
            left_shape == right_shape and isinstance(left_shape, list)
            and len(left_shape) == len(expected_axes)
            and left.get("axes") == right.get("axes") == expected_axes
            and isinstance(left.get("horizon_steps"), list)
            and len(left["horizon_steps"]) == left_shape[expected_axes.index("horizon")]
            and left.get("horizon_steps") == right.get("horizon_steps")
            and left.get("quantile_levels") == right.get("quantile_levels")
        ) if semantic else (
            isinstance(left_shape, list) and isinstance(right_shape, list)
            and sorted(left_shape) == sorted(right_shape)
        )
        if semantic and task == "series_to_quantile_forecast":
            levels = left.get("quantile_levels")
            shape_matches = (shape_matches and isinstance(levels, list)
                             and len(levels) == left_shape[0])
        matched = (
            isinstance(left_elements, int)
            and not isinstance(left_elements, bool)
            and left_elements > 0
            and left_elements == right_elements
            and isinstance(left_shape, list)
            and isinstance(right_shape, list)
            and shape_matches
        )
        return matched, "forecast output shape differs" if not matched else "", None
    if contract == "regression-values":
        def signature(value: Mapping[str, Any]) -> tuple[Any, ...] | None:
            targets, values = value.get("target_count"), value.get("values")
            if (isinstance(targets, bool) or not isinstance(targets, int) or targets <= 0
                    or not isinstance(values, list) or len(values) != targets
                    or value.get("axes") != ["target"]
                    or value.get("kind") != "regression_values"):
                return None
            if any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in values):
                return None
            for key in ("target_names", "target_units"):
                metadata = value.get(key, [])
                if (not isinstance(metadata, list) or len(metadata) not in {0, targets}
                        or not all(isinstance(item, str) for item in metadata)):
                    return None
            return targets,
        first, second = signature(left), signature(right)
        matched = first is not None and first == second
        for key in ("target_names", "target_units"):
            if left.get(key) and right.get(key) and left[key] != right[key]:
                matched = False
        return matched, "regression value/target axes differ" if not matched else "", None
    if contract == "regression-distribution":
        def signature(value: Mapping[str, Any]) -> tuple[Any, ...] | None:
            targets = value.get("target_count")
            parameters = value.get("parameters")
            if (value.get("distribution") not in {"normal", "student_t", "negative_binomial"}
                    or isinstance(targets, bool) or not isinstance(targets, int) or targets <= 0
                    or not isinstance(parameters, list) or not parameters
                    or value.get("axes") != ["target"]):
                return None
            names = []
            for parameter in parameters:
                if (not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str)
                        or not parameter["name"]):
                    return None
                values = parameter.get("values")
                if not isinstance(values, list) or len(values) != targets:
                    return None
                if any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) for item in values):
                    return None
                names.append(parameter["name"])
            if len(names) != len(set(names)):
                return None
            return value.get("distribution"), targets, tuple(names)
        left_signature, right_signature = signature(left), signature(right)
        matched = left_signature is not None and left_signature == right_signature
        return matched, "regression distribution/target axes differ" if not matched else "", None
    raise PerfMatrixError(f"output contract is not implemented: {contract}")


def _structured_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _structured_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def _structured_values(value: Any, name: str, count: int | None = None) -> int:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise ValueError(f"{name} has an invalid array length")
    for item in value:
        _structured_number(item, name)
    return len(value)


def _structured_artifact(summary: Mapping[str, Any], name: str, size: int) -> bytes:
    value = summary.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must identify an artifact file")
    path = Path(value)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{name} artifact size does not match its declared shape/byte count")
    payload = path.read_bytes()
    if len(payload) != size:
        raise ValueError(f"{name} artifact changed while being read")
    return payload


def _metric_geometry_signature(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Check complete typed geometry representation, not predicted-value parity."""
    height = _structured_int(summary.get("height"), "height", 1)
    width = _structured_int(summary.get("width"), "width", 1)
    pixels = height * width
    images = _structured_int(summary.get("geometry_images"), "geometry_images", 1)
    count = _structured_int(summary.get("geometry_pixels"), "geometry_pixels", 1)
    shape = summary.get("point_shape")
    if images != 1 or count != pixels or shape != [height, width, 3]:
        raise ValueError("geometry grid/count/point shape differs from [H,W,3]")
    if not isinstance(shape, list) or any(type(axis) is not int for axis in shape):
        raise ValueError("point_shape must have integer axes")
    if (summary.get("units") != "meters"
            or summary.get("camera_axes") != ["right", "down", "forward"]
            or summary.get("intrinsics_coordinates") != "normalized_uv"):
        raise ValueError("geometry units, camera axes or intrinsics coordinates differ from the Task")
    intrinsics = summary.get("normalized_intrinsics")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3:
        raise ValueError("normalized_intrinsics must be a complete [3,3] matrix")
    for row in intrinsics:
        _structured_values(row, "normalized_intrinsics", 3)
    # Raw float32 data is deliberately not filtered or rewritten: the validity
    # mask is authoritative and invalid pixels may retain +Inf (or another
    # family representation). Depth positivity is not a shared shape rule.
    _structured_artifact(summary, "points_artifact", pixels * 3 * 4)
    _structured_artifact(summary, "depth_artifact", pixels * 4)
    mask = _structured_artifact(summary, "valid_mask_artifact", pixels)
    valid = _structured_int(summary.get("valid_pixels"), "valid_pixels")
    if valid != mask.count(1):
        raise ValueError("valid_pixels differs from the complete mask artifact")
    if "intrinsics_artifact" in summary:
        path = summary["intrinsics_artifact"]
        if not isinstance(path, str) or not path:
            raise ValueError("intrinsics_artifact must identify an artifact file")
        calibration = json.loads(Path(path).read_bytes())
        if (not isinstance(calibration, Mapping)
                or type(calibration.get("height")) is not int
                or type(calibration.get("width")) is not int
                or calibration.get("height") != height or calibration.get("width") != width
                or calibration.get("normalized") is not True
                or calibration.get("intrinsics") != intrinsics):
            raise ValueError("intrinsics artifact differs from the output calibration")
        for row in calibration["intrinsics"]:
            _structured_values(row, "intrinsics artifact", 3)
    return height, width, tuple(shape), "meters", ("right", "down", "forward"), "normalized_uv"


def _molecular_structure_signature(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Validate opaque outputs and confidence presence/shape, without PDB/CIF parsing."""
    structures = _structured_int(summary.get("structures"), "structures", 1)
    format_name = summary.get("format")
    if structures != 1 or not isinstance(format_name, str) or format_name not in {"pdb", "mmcif"}:
        raise ValueError("structure count or format is invalid")
    structure_bytes = _structured_int(summary.get("structure_bytes"), "structure_bytes", 1)
    metadata_bytes = _structured_int(summary.get("metadata_bytes"), "metadata_bytes")
    document_bytes = _structured_int(summary.get("document_bytes"), "document_bytes", 1)
    encoding, source_path = summary.get("input_encoding"), summary.get("source_path")
    if not isinstance(encoding, str) or not encoding or not isinstance(source_path, str):
        raise ValueError("structure input encoding/source_path is missing")
    _structured_artifact(summary, "structure_artifact", structure_bytes)
    # metadata_json is verbatim family output, including empty bytes or NULs.
    # Parsing it here would invent a shared schema not present in the C API.
    _structured_artifact(summary, "metadata_artifact", metadata_bytes)
    if "confidence" not in summary:
        raise ValueError("confidence must explicitly be null or a complete confidence object")
    confidence = summary["confidence"]
    confidence_shape = None
    if confidence is not None:
        if not isinstance(confidence, Mapping):
            raise ValueError("confidence must be null or an object")
        for name in ("confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
                     "complex_plddt", "complex_iplddt"):
            _structured_number(confidence.get(name), f"confidence.{name}")
        confidence_shape = _structured_values(confidence.get("plddt"), "confidence.plddt")
    # Textual numbers may differ in width without changing the representation;
    # each artifact length is checked above, not equated across predictions.
    return structures, format_name, document_bytes, encoding, source_path, confidence_shape


def _pose_refinement_signature(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Validate all row-major pose/query arrays; family tests retain numerical oracles."""
    count = _structured_int(summary.get("refined_hypotheses"), "refined_hypotheses", 1)
    shape = summary.get("shape")
    if (not isinstance(shape, list) or shape != [count, 4, 4]
            or any(type(axis) is not int for axis in shape)):
        raise ValueError("refined pose shape must be [N,4,4]")
    _structured_values(summary.get("refined_poses"), "refined_poses", count * 16)
    scores = _structured_values(summary.get("scores"), "scores")
    if scores != count and not (count == 1 and scores == 0):
        raise ValueError("scores must cover every hypothesis, except a single unscored pose")
    best = _structured_int(summary.get("best_index"), "best_index")
    if best >= count:
        raise ValueError("best_index is outside the returned hypotheses")
    rigid = summary.get("all_poses_rigid")
    if not isinstance(rigid, bool):
        raise ValueError("all_poses_rigid must preserve the family boolean flag")
    for name in ("refinement_ms", "scoring_ms"):
        _structured_number(summary.get(name), name)
    queries = summary.get("crop_queries")
    if not isinstance(queries, list):
        raise ValueError("crop_queries must preserve the complete callback trace")
    trace = []
    for query in queries:
        if not isinstance(query, Mapping):
            raise ValueError("crop query must be an object")
        stage = query.get("stage")
        if not isinstance(stage, str) or stage not in {"refinement", "scoring"}:
            raise ValueError("crop query has an unknown stage")
        iteration = _structured_int(query.get("iteration"), "crop query iteration")
        query_shape = query.get("shape")
        if (not isinstance(query_shape, list) or len(query_shape) != 3
                or query_shape[1:] != [4, 4]
                or any(type(axis) is not int for axis in query_shape)):
            raise ValueError("crop query shape must be [N,4,4]")
        query_count = _structured_int(query_shape[0], "crop query count", 1)
        _structured_values(query.get("poses"), "crop query poses", query_count * 16)
        trace.append((stage, iteration, tuple(query_shape)))
    # No argmax, rigid-transform tolerance or fixed iteration/scheduler policy.
    return tuple(shape), scores, rigid, tuple(trace)


def _offline_speech_wav_chunks(payload: bytes) -> dict[bytes, bytes]:
    if (len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE"
            or int.from_bytes(payload[4:8], "little") + 8 != len(payload)):
        raise ValueError("audio_artifact must be a complete little-endian RIFF/WAVE file")
    chunks: dict[bytes, bytes] = {}
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError("audio_artifact has a truncated chunk header")
        kind = payload[offset:offset + 4]
        size = int.from_bytes(payload[offset + 4:offset + 8], "little")
        end = offset + 8 + size
        if end + (size & 1) > len(payload):
            raise ValueError("audio_artifact has a truncated chunk")
        if kind in {b"fmt ", b"data"}:
            if kind in chunks:
                raise ValueError("audio_artifact repeats a format or data chunk")
            chunks[kind] = payload[offset + 8:end]
        offset = end + (size & 1)
    if b"fmt " not in chunks or b"data" not in chunks:
        raise ValueError("audio_artifact requires explicit format and data chunks")
    return chunks


def _offline_speech_wav(path_value: Any) -> tuple[list[float], int, int]:
    # The reference writes soundfile subtype=FLOAT. The existing PCM WAV helper
    # cannot read that format and also downmixes; neither conversion belongs here.
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("audio_artifact must identify the complete Float32 WAV output")
    chunks = _offline_speech_wav_chunks(Path(path_value).read_bytes())
    if len(chunks[b"fmt "]) < 16:
        raise ValueError("audio_artifact has an incomplete WAV format")
    kind, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if kind != 3 or bits != 32 or channels < 1 or rate < 1:
        raise ValueError("audio_artifact must use IEEE Float32 WAV with a valid audio format")
    data = chunks[b"data"]
    if alignment != channels * 4 or byte_rate != rate * alignment or len(data) % alignment:
        raise ValueError("audio_artifact byte layout does not match its complete audio frames")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("audio_artifact contains nonfinite PCM")
    return values.tolist(), rate, channels


def _offline_speech_input(summary: Mapping[str, Any]) -> tuple[int, int, int]:
    count = _structured_int(summary.get("input_samples"), "input_samples", 1)
    rate = _structured_int(summary.get("input_sample_rate"), "input_sample_rate", 1)
    channels = _structured_int(summary.get("input_channels"), "input_channels", 1)
    if count > (1 << 64) - 1 or rate > (1 << 32) - 1 or channels > (1 << 32) - 1:
        raise ValueError("input count or format exceeds its public integer representation")
    if count % channels:
        raise ValueError("input_samples must contain complete interleaved frames")
    if "input_frames" in summary:
        frames = _structured_int(summary["input_frames"], "input_frames", 1)
        if frames != count // channels:
            raise ValueError("input_frames differs from input_samples and channels")
    return count, rate, channels


def _offline_speech_duration(value: Any, samples: int, rate: int, channels: int) -> None:
    _structured_number(value, "audio duration")
    if not math.isclose(value, samples / channels / rate, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("audio duration differs from the complete PCM count and format")


def _offline_speech_event(event: Any, rate: int, channels: int) -> tuple[str, int]:
    kinds = {
        "agent_audio", "agent_text", "user_transcript", "turn_started", "turn_finished",
        "yielded", "reset", "input_finished", "user_speech_started", "user_speech_stopped",
        "input_cleared",
    }
    if not isinstance(event, Mapping):
        raise ValueError("offline speech events must be objects")
    kind = event.get("kind")
    if not isinstance(kind, str) or kind not in kinds:
        raise ValueError("unsupported or failed offline speech event kind")
    epoch = _structured_int(event.get("epoch"), "event epoch")
    _structured_int(event.get("sequence"), "event sequence")
    if not isinstance(event.get("text"), str) or not isinstance(event.get("is_final"), bool):
        raise ValueError("speech event text and final marker must be explicit typed values")
    count = _structured_int(event.get("audio_samples"), "event audio_samples")
    _structured_values(event.get("audio"), "event audio", count)
    if any(abs(value) > 3.4028234663852886e38 for value in event["audio"]):
        raise ValueError("event audio exceeds the finite Float32 representation")
    event_channels = _structured_int(event.get("channels"), "event channels")
    event_rate = event.get("sample_rate")
    if event_rate is not None:
        _structured_int(event_rate, "event sample_rate", 1)
    if kind == "agent_audio":
        if (event_rate, event_channels) != (rate, channels) or count % channels:
            raise ValueError("agent audio chunks must retain the declared output format")
    elif count:
        raise ValueError("non-audio speech event contains unaccounted PCM")
    return kind, epoch


def _offline_speech_candidate(summary: Mapping[str, Any]) -> dict[str, Any]:
    events, output_format = summary.get("events"), summary.get("output_format")
    if not isinstance(events, list) or not isinstance(output_format, Mapping):
        raise ValueError("offline speech requires complete events and an explicit output format")
    rate = _structured_int(output_format.get("sample_rate"), "output sample_rate", 1)
    channels = _structured_int(output_format.get("channels"), "output channels", 1)
    if rate > (1 << 32) - 1 or channels > (1 << 32) - 1:
        raise ValueError("output audio format must fit its public uint32 representation")
    turns: dict[int, dict[str, Any]] = {}
    audio: list[float] = []
    for event in events:
        kind, epoch = _offline_speech_event(event, rate, channels)
        if kind == "agent_audio":
            audio.extend(event["audio"])
        elif kind == "agent_text":
            turn = turns.setdefault(epoch, {"partial": "", "final": None})
            if event["is_final"]:
                turn["final"] = event["text"]
            else:
                turn["partial"] += event["text"]
    # Epoch order is first appearance, not numeric sorting. A final string is a
    # complete replacement, never another delta appended to the partial text.
    text = " ".join(piece for turn in turns.values()
                    if (piece := turn["final"] if turn["final"] is not None else turn["partial"]))
    _offline_speech_duration(summary.get("output_audio_seconds"), len(audio), rate, channels)
    return {"text": text, "audio": audio, "sample_rate": rate, "channels": channels,
            "input": _offline_speech_input(summary)}


def _offline_speech_reference(summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary.get("text"), str):
        raise ValueError("offline speech reference must retain its complete text output")
    count = _structured_int(summary.get("num_samples"), "num_samples")
    declared = _structured_int(summary.get("audio_samples"), "audio_samples")
    rate = _structured_int(summary.get("sample_rate"), "sample_rate", 1)
    channels = _structured_int(summary.get("channels"), "channels", 1)
    audio, artifact_rate, artifact_channels = _offline_speech_wav(summary.get("audio_artifact"))
    if count != declared or count != len(audio) or (rate, channels) != (artifact_rate, artifact_channels):
        raise ValueError("reference counts/format differ from the complete PCM artifact")
    if summary.get("finite") is not True:
        raise ValueError("reference finite marker is missing or contradicts the PCM artifact")
    _offline_speech_duration(summary.get("audio_seconds"), count, rate, channels)
    return {"text": summary["text"], "audio": audio, "sample_rate": rate, "channels": channels,
            "input": _offline_speech_input(summary)}


def _offline_speech_contract(
    entry: ResolvedEntry, left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    evidence: dict[str, Any] = {
        "contract": "offline-speech-shape", "numerical_parity_checked": False,
        "text_parity_checked": False,
    }
    if entry.spec["operation"] != "speech_dialogue" or _effective_task(entry) != "offline_speech_dialogue":
        return False, "offline speech contract cannot qualify live, tool or non-dialogue Tasks", evidence
    for side, summary, parse in (("candidate", left, _offline_speech_candidate),
                                 ("reference", right, _offline_speech_reference)):
        try:
            value = parse(summary)
        except (ValueError, OSError, OverflowError) as error:
            return False, f"{side} offline-speech-shape: {error}", evidence
        evidence[side] = {
            "text": value["text"], "audio_samples": len(value["audio"]),
            "sample_rate": value["sample_rate"], "channels": value["channels"],
            "input_samples": value["input"][0], "input_sample_rate": value["input"][1],
            "input_channels": value["input"][2],
        }
    names = ("audio_samples", "sample_rate", "channels", "input_samples",
             "input_sample_rate", "input_channels")
    matched = all(evidence["candidate"][name] == evidence["reference"][name] for name in names)
    return matched, "offline speech PCM/input shape or format differs" if not matched else "", evidence


def _token_count(value: Mapping[str, Any]) -> int | None:
    tokens = value.get("token_ids")
    if isinstance(tokens, list):
        return len(tokens)
    count = value.get("output_tokens")
    return int(count) if isinstance(count, int) and not isinstance(count, bool) else None


def _media_shape(value: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    media_type = value.get("media_type")
    if media_type == "image":
        count = value.get("batch_size", value.get("generated_images", value.get("media_count")))
    else:
        count = value.get("num_frames", value.get("generated_frames", value.get("media_count")))
    return count, value.get("height"), value.get("width"), value.get("channels")


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _text_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _localizations(text: str) -> tuple[str, list[tuple[float, ...]]] | None:
    if "<ref>" not in text or "</ref>" not in text:
        return None
    groups = re.findall(r"<(?:box|point)>(.*?)</(?:box|point)>", text, flags=re.DOTALL)
    if not groups:
        return None
    values = []
    kind = ""
    for group in groups:
        numbers = tuple(float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", group))
        if len(numbers) == 4:
            current = "box"
        elif len(numbers) == 2:
            current = "point"
        else:
            return None
        if kind and current != kind:
            return None
        kind = current
        values.append(numbers)
    return kind, values


def _localization_contract(
    entry: ResolvedEntry,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any] | None]:
    candidate = _localizations(str(left.get("text", "")))
    reference = _localizations(str(right.get("text", "")))
    if candidate is None or reference is None:
        return False, "localization markup is invalid", None
    if candidate[0] != reference[0] or len(candidate[1]) != len(reference[1]):
        return False, "localization type or count differs", None
    evidence: dict[str, Any] = {"kind": candidate[0], "count": len(candidate[1])}
    if candidate[0] == "box":
        scores = [_box_iou(a, b) for a, b in zip(candidate[1], reference[1], strict=True)]
        minimum = min(scores)
        limit = float(entry.spec["baseline"]["min_localization_box_iou"])
        evidence.update(minimum_iou=minimum, required_iou=limit)
        if minimum < limit:
            return False, "localization box IoU is below the contract", evidence
    else:
        distances = [math.dist(a, b) for a, b in zip(candidate[1], reference[1], strict=True)]
        maximum = max(distances)
        limit = float(entry.spec["baseline"]["max_localization_point_distance"])
        evidence.update(maximum_point_distance=maximum, allowed_point_distance=limit)
        if maximum > limit:
            return False, "localization point distance exceeds the contract", evidence
    distance = _text_distance(
        _normalized_text(left.get("text")), _normalized_text(right.get("text"))
    )
    limit = float(entry.spec["baseline"]["max_normalized_edit_distance"])
    evidence.update(normalized_edit_distance=distance, maximum_text_distance=limit)
    return distance <= limit, "localization text distance exceeds the contract", evidence


def _box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _disparity(
    entry: ResolvedEntry,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_shape = tuple(left.get(name) for name in ("height", "width", "element_count"))
    right_shape = tuple(right.get(name) for name in ("height", "width", "element_count"))
    if left_shape != right_shape or None in left_shape:
        return {"passed": False, "reason": "disparity output shapes differ"}
    left_values = _float_artifact(left)
    right_values = _float_artifact(right)
    if len(left_values) != len(right_values) or not left_values:
        return {"passed": False, "reason": "disparity artifacts have different lengths"}
    finite = all(math.isfinite(value) and value >= 0.0 for value in left_values)
    reference_finite = all(math.isfinite(value) and value >= 0.0 for value in right_values)
    dot = math.fsum(a * b for a, b in zip(left_values, right_values, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    differences = [abs(a - b) for a, b in zip(left_values, right_values, strict=True)]
    mean_error = math.fsum(differences) / len(differences)
    bad_fraction = sum(value > 2.0 for value in differences) / len(differences)
    baseline = entry.spec["baseline"]
    passed = (
        finite
        and reference_finite
        and cosine >= float(baseline["min_disparity_cosine"])
        and mean_error <= float(baseline["max_disparity_mean_abs_error"])
        and bad_fraction <= float(baseline["max_disparity_bad_2px_fraction"])
    )
    return {
        "passed": passed,
        "reason": "" if passed else "disparity parity is outside the contract",
        "cosine": cosine,
        "mean_abs_error": mean_error,
        "bad_2px_fraction": bad_fraction,
    }


def _float_artifact(summary: Mapping[str, Any]) -> list[float]:
    path_value = summary.get("disparity_artifact")
    count = summary.get("element_count")
    if not isinstance(path_value, str) or not isinstance(count, int) or count < 1:
        raise PerfMatrixError("disparity summary is missing its artifact")
    payload = Path(path_value).read_bytes()
    if len(payload) != count * 4:
        raise PerfMatrixError("disparity artifact size does not match element_count")
    values = array("f")
    values.frombytes(payload)
    return [float(value) for value in values]


def _entry_slug(entry_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", entry_id).strip("-")
    return "entry" if value in {"", ".", ".."} else value


def _execute_entry(
    entry: ResolvedEntry,
    environment: Environment,
    run_directory: Path,
    *,
    no_build: bool,
    verbose: bool,
    attempt: int,
) -> dict[str, Any]:
    entry_slug = _entry_slug(str(entry.spec["id"]))
    artifact_root = run_directory / "artifacts" / entry_slug
    while True:
        artifact = artifact_root / f"attempt-{attempt}"
        try:
            artifact.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        break
    entry_work = environment.scratch_root / _entry_slug(run_directory.name) / entry_slug
    work = entry_work / f"attempt-{attempt}"
    command_environment = _entry_command_environment(environment, work)
    commands: dict[str, Any] = {}
    measurement_attempts: list[dict[str, Any]] = []
    measurement_stability: dict[str, Any] | None = None
    candidate: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    status = "white"
    comparison: dict[str, Any] = {}
    cleanup = None
    cache_cleanup = None
    try:
        for measurement_attempt in (1, 2):
            status = "white"
            comparison = {}
            measurement_dir = artifact if measurement_attempt == 1 else artifact / "measurement-2"
            candidate_output = measurement_dir / "candidate"
            reference_output = measurement_dir / "reference.json"
            logs = measurement_dir / "logs"
            suffix = "" if measurement_attempt == 1 else "_measurement_2"

            candidate_arguments = candidate_command(
                entry, environment, candidate_output, no_build=no_build
            )
            candidate_result = run_command(
                candidate_arguments,
                timeout=environment.timeout_seconds,
                stdout_path=logs / "candidate.stdout.log",
                stderr_path=logs / "candidate.stderr.log",
                verbose=verbose,
                env=command_environment,
            )
            commands["candidate" + suffix] = candidate_result
            if candidate_result["exit_code"] != 0:
                raise PerfMatrixError("candidate command failed")
            candidate = _candidate_result(candidate_output)

            reference_arguments = baseline_command(entry, environment, reference_output)
            reference_result = run_command(
                reference_arguments,
                timeout=environment.timeout_seconds,
                stdout_path=logs / "reference.stdout.log",
                stderr_path=logs / "reference.stderr.log",
                verbose=verbose,
                env=command_environment,
            )
            commands["reference" + suffix] = reference_result
            if reference_result["exit_code"] != 0:
                raise PerfMatrixError("reference command failed")
            reference = _json_file(reference_output, "reference result")
            if reference.get("status") != "completed":
                raise PerfMatrixError(str(reference.get("error", "reference failed")))
            if "script" in entry.spec["baseline"]:
                _validate_script_result(entry, environment, reference)

            status, comparison = compare(entry, candidate, reference)
            if status not in TERMINAL_COMPARISONS:
                measurement_stability = None
                break
            stability = _measurement_stability(reference, candidate)
            measurement_attempts.append(
                {
                    "attempt": measurement_attempt,
                    "reference": {
                        "samples_ms": list(reference.get("samples_ms", [])),
                        "stability": stability["reference"],
                    },
                    "candidate": {
                        "samples_ms": list(candidate.get("samples_ms", [])),
                        "stability": stability["candidate"],
                    },
                }
            )
            measurement_stability = {**stability, "attempts": list(measurement_attempts)}
            if stability["status"] == "stable":
                if measurement_attempt == 2:
                    measurement_stability["status"] = "stable_after_retry"
                break
            if stability["status"] == "not_evaluated":
                status = "white"
                comparison = {"reason": "timing stability requires ten valid samples per side"}
                measurement_stability["status"] = "measurement_inconclusive"
                break
            if measurement_attempt == 2:
                status = "white"
                comparison = {"reason": "timing did not settle after one remeasurement"}
                measurement_stability["status"] = "measurement_inconclusive"
    finally:
        passed = status in TERMINAL_COMPARISONS
        if environment.bundle_retention == "delete_always" or (
            environment.bundle_retention == "delete_on_pass" and passed
        ):
            cleanup = _cleanup_managed_bundle(candidate or None, entry, environment)
        cache_cleanup = _cleanup_entry_work(entry_work, environment, passed=passed)
    row = {
        "id": entry.spec["id"],
        "model": entry.model.name,
        "family": entry.model.family,
        "operation": entry.spec["operation"],
        "testcase": entry.case.testcase_name,
        "status": status,
        "attempts": attempt,
        "artifact_dir": str(artifact.relative_to(run_directory)),
        "candidate": candidate,
        "reference": reference,
        "comparison": comparison,
        "bundle_cleanup": cleanup,
        "hf_cache_cleanup": cache_cleanup,
        "commands": commands,
    }
    if measurement_stability is not None:
        row["measurement_stability"] = measurement_stability
    return row


def _cleanup_managed_bundle(
    candidate: Mapping[str, Any] | None, entry: ResolvedEntry, environment: Environment
) -> dict[str, Any] | None:
    if candidate is None:
        records: Sequence[Any] = [
            {"model": entry.model.name, "bundle": str(entry.case.bundle_path)},
        ]
    else:
        preparation = candidate.get("preparation", {})
        records = preparation.get("bundles", []) if isinstance(preparation, Mapping) else []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, Mapping) or record.get("model") != entry.model.name:
            continue
        raw = record.get("bundle")
        if not isinstance(raw, str) or not raw:
            continue
        bundle = Path(raw).expanduser().resolve()
        try:
            relative = bundle.relative_to(environment.bundle_cache)
        except ValueError:
            return {"status": "preserved", "reason": "bundle is outside managed cache"}
        if len(relative.parts) != 2 or relative.parts[0] != entry.model.name:
            raise PerfMatrixError(f"refusing to delete unexpected managed bundle path {bundle}")
        bundle.unlink(missing_ok=True)
        return {"status": "deleted", "bundle": str(bundle)}
    return None


def _cleanup_entry_work(
    work: Path, environment: Environment, *, passed: bool
) -> dict[str, Any] | None:
    if environment.hf_cache_mode != "per_entry":
        return None
    policy = environment.hf_cache_retention
    evidence: dict[str, Any] = {"path": str(work), "policy": policy, "status": "retained"}
    if policy == "retain" or (policy == "delete_on_pass" and not passed):
        return evidence
    try:
        relative = work.resolve().relative_to(environment.scratch_root)
    except ValueError as error:
        raise PerfMatrixError(
            f"refusing to delete HF cache outside scratch_root: {work}"
        ) from error
    if len(relative.parts) < 2:
        raise PerfMatrixError(f"refusing to delete broad HF cache path: {work}")
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        evidence["status"] = "already_absent"
    except OSError as error:
        evidence.update(status="failed", error=str(error))
    else:
        evidence["status"] = "deleted"
    return evidence


def _new_run_directory(root: Path) -> Path:
    base = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    for index in range(1000):
        candidate = root / (base if index == 0 else f"{base}-{index}")
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            continue
        return candidate
    raise PerfMatrixError("cannot allocate a run directory")


def _initial_results(
    suite_path: Path,
    environment_path: Path,
    suite_name: str,
    environment: Environment,
    entries: Sequence[ResolvedEntry],
    *,
    no_build: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "running",
        "started_at": _now(),
        "suite": suite_name,
        "environment": environment.name,
        "suite_path": str(suite_path.resolve()),
        "environment_path": str(environment_path.resolve()),
        "selected_entry_ids": [entry.spec["id"] for entry in entries],
        "no_build": no_build,
        "rows": [],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_rows(
    run_directory: Path,
    results: dict[str, Any],
    entries: Sequence[ResolvedEntry],
    environment: Environment,
    *,
    no_build: bool,
    verbose: bool,
) -> int:
    current = {
        str(row.get("id")): row for row in results.get("rows", []) if isinstance(row, Mapping)
    }
    for entry in entries:
        entry_id = str(entry.spec["id"])
        previous = current.get(entry_id)
        if previous and previous.get("status") in FINISHED_RESULTS:
            continue
        attempt = int(previous.get("attempts", 0)) + 1 if previous else 1
        artifact_root = run_directory / "artifacts" / _entry_slug(entry_id)
        while (artifact_root / f"attempt-{attempt}").exists():
            attempt += 1
        try:
            row = _execute_entry(
                entry,
                environment,
                run_directory,
                no_build=no_build,
                verbose=verbose,
                attempt=attempt,
            )
        except (OSError, PerfMatrixError, subprocess.SubprocessError) as error:
            row = {
                "id": entry_id,
                "model": entry.model.name,
                "family": entry.model.family,
                "operation": entry.spec["operation"],
                "testcase": entry.case.testcase_name,
                "status": "white",
                "attempts": attempt,
                "error": str(error),
            }
        current[entry_id] = row
        results["rows"] = [
            current[str(value.spec["id"])] for value in entries if str(value.spec["id"]) in current
        ]
        _write_json(run_directory / "results.json", results)
        write_report(run_directory, results)

    results["finished_at"] = _now()
    results["status"] = (
        "completed"
        if all(row.get("status") in TERMINAL_COMPARISONS for row in results["rows"])
        else "failed"
    )
    _write_json(run_directory / "results.json", results)
    write_report(run_directory, results)
    return 0 if results["status"] == "completed" else 1


def write_report(
    run_directory: Path,
    results: Mapping[str, Any],
    preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_schema = results.get("schema_version", RESULT_SCHEMA)
    row_key = "cases" if source_schema == "trtmc.perf-matrix/v1" else "rows"
    rows = [dict(row) for row in results.get(row_key, []) if isinstance(row, Mapping)]
    selected_ids = results.get("selected_entry_ids")
    if not isinstance(selected_ids, list) or not all(
        isinstance(value, str) for value in selected_ids
    ):
        raise PerfMatrixError("matrix results selected_entry_ids must be strings")
    completed_ids = {str(row.get("id")) for row in rows}
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("green", "yellow", "red", "contract-mismatch", "white")
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_schema_version": source_schema,
        "generated_at": _now(),
        "status": results.get("status", "unknown"),
        "suite": results.get("suite"),
        "environment": results.get("environment"),
        "summary": {
            "selected": len(selected_ids),
            "pending": sum(entry_id not in completed_ids for entry_id in selected_ids),
            "comparable": counts["green"] + counts["yellow"] + counts["red"],
            **counts,
        },
        "rows": rows,
    }
    if preparation is not None:
        report["preparation"] = dict(preparation)
    _write_json(run_directory / "report.json", report)
    (run_directory / "report.html").write_text(_report_html(report), encoding="utf-8")
    return report


def _measurement_html(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "Not recorded"
    metrics = value.get("metrics", {})
    latency = metrics.get("latency_ms", {}) if isinstance(metrics, Mapping) else {}
    samples = value.get("samples_ms", [])
    measured = (
        sorted(
            float(sample)
            for sample in samples
            if isinstance(sample, (int, float))
            and not isinstance(sample, bool)
            and math.isfinite(sample)
            and sample > 0
        )
        if isinstance(samples, list)
        else []
    )
    parts = []
    for name, percentile in (("p50", 0.5), ("p95", 0.95)):
        number = latency.get(name) if isinstance(latency, Mapping) else None
        if number is None and measured:
            position = (len(measured) - 1) * percentile
            lower, upper = math.floor(position), math.ceil(position)
            number = measured[lower] + (measured[upper] - measured[lower]) * (position - lower)
        if (
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
        ):
            parts.append(f"{name}: {number:,.3f} ms")
    policy = value.get("measurement_policy", {})
    scope = value.get("timing_scope") or (
        policy.get("timing_scope") if isinstance(policy, Mapping) else None
    )
    parts.append("Scope: " + html.escape(str(scope or "not recorded")))
    if isinstance(metrics, Mapping):
        parts.extend(
            f"{html.escape(name)}: {number:,.3f}"
            for name, number in metrics.items()
            if name.endswith("_per_s")
            and isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(number)
        )
    return "<br>".join(parts)


def _report_html(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report.get("rows", []):
        comparison = row.get("comparison", {}) if isinstance(row, Mapping) else {}
        measurement_stability = (
            row.get("measurement_stability", {}) if isinstance(row, Mapping) else {}
        )
        candidate = _measurement_html(
            row.get("candidate")
            or {"metrics": {"latency_ms": {"p50": comparison.get("candidate_p50_ms")}}}
        )
        reference = _measurement_html(
            row.get("reference", row.get("baseline"))
            or {"metrics": {"latency_ms": {"p50": comparison.get("reference_p50_ms")}}}
        )
        reason = comparison.get("reason", row.get("error", ""))
        stability = (
            measurement_stability.get("status", "")
            if isinstance(measurement_stability, Mapping)
            else ""
        )
        evidence = {
            key: row[key]
            for key in (
                "commands",
                "comparison",
                "measurement_stability",
                "resolved_settings",
                "baseline_contract",
            )
            if key in row
        }
        detail = html.escape(json.dumps(evidence, indent=2, sort_keys=True))
        link = ""
        artifact = row.get("artifact_dir")
        if (
            isinstance(artifact, str)
            and artifact
            and not Path(artifact).is_absolute()
            and ".." not in Path(artifact).parts
            and not any(value in artifact for value in (":", "\\"))
        ):
            from urllib.parse import quote

            link = f'<a href="{quote(artifact, safe="/.")}/">Case artifacts</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{html.escape(str(row.get('operation', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(stability))}</td>"
            f"<td>{candidate}</td>"
            f"<td>{reference}</td>"
            f"<td>{html.escape(str(reason))}</td>"
            f"<td><details><summary>Evidence / commands</summary>{link}<pre>{detail}</pre></details></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>TRTMC performance</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th,td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style></head><body>
<h1>TRTMC performance matrix</h1>
<p>Source data: {html.escape(str(report.get("source_schema_version", RESULT_SCHEMA)))}. Original timing scopes and recorded outcomes are preserved.</p>
<p>Status: {html.escape(str(report.get("status", "unknown")))};
selected: {html.escape(str(report.get("summary", {}).get("selected", 0)))};
pending: {html.escape(str(report.get("summary", {}).get("pending", 0)))}</p>
<label>Filter model, case, or status <input id="filter" type="search"></label>
<table><thead><tr><th>Entry</th><th>Model</th><th>Operation</th><th>Status</th>
<th>Stability</th><th>Candidate measurements</th><th>Reference measurements</th><th>Reason</th><th>Evidence</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>Green is faster by more than the margin, yellow is within the margin, red is
slower by more than the margin, and white is not comparable.</p>
<p>Output comparison follows the suite's declared contract; matching dimensions or token counts alone does not establish model correctness. Use family correctness tests for acceptance.</p>
<p><a href="results.json">Original results</a> · <a href="report.json">Report data</a></p>
<script>document.querySelector('#filter').addEventListener('input',function(){{
const query=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>{{
row.hidden=!row.textContent.toLowerCase().includes(query);}});}});</script>
</body></html>
"""


def prepare_entries(
    entries: Sequence[ResolvedEntry],
    environment: Environment,
    output: Path,
    *,
    verbose: bool,
) -> int:
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    log_root = output.resolve().parent / (output.stem + "-logs")
    for entry in entries:
        command = candidate_command(entry, environment, None, prepare_only=True)
        result = run_command(
            command,
            timeout=environment.timeout_seconds,
            stdout_path=log_root / f"{_entry_slug(str(entry.spec['id']))}.stdout.log",
            stderr_path=log_root / f"{_entry_slug(str(entry.spec['id']))}.stderr.log",
            verbose=verbose,
        )
        if result["exit_code"] != 0:
            raise PerfMatrixError(f"bundle preparation failed for {entry.spec['id']}")
        try:
            payload = json.loads(Path(result["stdout_log"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PerfMatrixError(
                f"bundle preparation returned invalid JSON for {entry.spec['id']}"
            ) from error
        records = payload.get("bundles") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            raise PerfMatrixError(f"bundle preparation returned no bundles for {entry.spec['id']}")
        for record in records:
            if not isinstance(record, Mapping):
                continue
            key = (str(record.get("model", "")), str(record.get("bundle", "")))
            bundles[key] = dict(record)
    receipt = {
        "schema_version": PREPARATION_SCHEMA,
        "created_at": _now(),
        "included_in_performance_metrics": False,
        "bundles": list(bundles.values()),
    }
    _write_json(output.resolve(), receipt)
    print(f"Prepared {len(bundles)} bundle(s): {output.resolve()}")
    return 0


def _load_results(run_directory: Path, *, allow_legacy_report: bool = False) -> dict[str, Any]:
    value = _json_file(run_directory.resolve() / "results.json", "matrix results")
    supported = {RESULT_SCHEMA}
    if allow_legacy_report:
        supported.add("trtmc.perf-matrix/v1")
    if value.get("schema_version") not in supported:
        raise PerfMatrixError("run directory has an unsupported results schema")
    return value


def _common(
    arguments: argparse.Namespace,
) -> tuple[Path, Path, str, list[dict[str, Any]], set[str], Environment, list[ResolvedEntry]]:
    suite_path = arguments.suite.resolve()
    environment_path = arguments.environment.resolve()
    suite_name, all_entries, excluded = load_suite(suite_path)
    selected = select_entries(
        all_entries,
        entry_ids=arguments.entry,
        models=arguments.model,
        model_selection=arguments.model_selection,
    )
    environment = load_environment(environment_path)
    _coverage(all_entries, excluded)
    resolved = preflight(
        selected,
        environment,
        require_runtime=arguments.command in {"check", "run"},
    )
    selected_ids = {entry["id"] for entry in selected}
    resolved_selected = [entry for entry in resolved if entry.spec["id"] in selected_ids]
    return (
        suite_path,
        environment_path,
        suite_name,
        selected,
        excluded,
        environment,
        resolved_selected,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command in {"check", "prepare", "run"}:
            (
                suite_path,
                environment_path,
                suite_name,
                _selected,
                _excluded,
                environment,
                resolved,
            ) = _common(arguments)
            if arguments.command == "check":
                print(f"Ready: {len(resolved)} performance entrie(s)")
                return 0
            if arguments.command == "prepare":
                return prepare_entries(
                    resolved, environment, arguments.output, verbose=arguments.verbose
                )
            run_directory = _new_run_directory(environment.results_root)
            results = _initial_results(
                suite_path,
                environment_path,
                suite_name,
                environment,
                resolved,
                no_build=arguments.no_build,
            )
            _write_json(run_directory / "results.json", results)
            print(f"Run directory: {run_directory}")
            return _run_rows(
                run_directory,
                results,
                resolved,
                environment,
                no_build=arguments.no_build,
                verbose=arguments.verbose,
            )
        if arguments.command == "resume":
            run_directory = arguments.run_directory.resolve()
            results = _load_results(run_directory)
            suite_name, all_entries, excluded = load_suite(Path(results["suite_path"]))
            environment = load_environment(Path(results["environment_path"]))
            stored_ids = results.get("selected_entry_ids")
            if (
                not isinstance(stored_ids, list)
                or not stored_ids
                or not all(isinstance(value, str) and value for value in stored_ids)
            ):
                raise PerfMatrixError("matrix results has no selected entry IDs")
            selected_ids = set(stored_ids)
            missing = selected_ids - {str(entry["id"]) for entry in all_entries}
            if missing:
                raise PerfMatrixError(
                    "selected entries are missing from the suite: " + ", ".join(sorted(missing))
                )
            _coverage(all_entries, excluded)
            selected = [entry for entry in all_entries if entry["id"] in selected_ids]
            resolved = preflight(selected, environment, require_runtime=True)
            results["suite"] = suite_name
            results["status"] = "running"
            return _run_rows(
                run_directory,
                results,
                resolved,
                environment,
                no_build=arguments.no_build or bool(results.get("no_build")),
                verbose=arguments.verbose,
            )
        if arguments.command == "report":
            run_directory = arguments.run_directory.resolve()
            results = _load_results(run_directory, allow_legacy_report=True)
            preparation = (
                _json_file(arguments.preparation_receipt.resolve(), "preparation receipt")
                if arguments.preparation_receipt
                else None
            )
            if preparation is not None and preparation.get("schema_version") != PREPARATION_SCHEMA:
                raise PerfMatrixError("preparation receipt has an unsupported schema")
            report = write_report(run_directory, results, preparation)
            print(
                f"{report['status']}: {report['summary']['comparable']}/"
                f"{report['summary']['selected']} comparable"
            )
            return 0
    except (OSError, PerfMatrixError, ValueError, yaml.YAMLError) as error:
        print(f"perf-matrix: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

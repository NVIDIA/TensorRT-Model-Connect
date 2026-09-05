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
from trtmc_benchmark.catalog import ManifestCatalog, resolve_case  # noqa: E402
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
    "generated-token-count",
    "image-features-shape",
    "localization",
    "media-shape",
    "normalized-text",
    "ocr-text",
    "reranking-order",
    "robot-action-shape",
    "segmentation-shape",
    "transcription-text",
}
SEQUENCE_FAMILIES = {"bart", "m2m_100", "marian", "t5"}
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
    raw = _read_yaml(path.resolve(), "performance suite")
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
    return name, entries, excluded


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
    if baseline["runner"] == "task-reference" and not isinstance(baseline.get("adapter"), str):
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
            overrides = {
                f"request.{name}": value for name, value in workload.get("request", {}).items()
            }
            timing = timing_contract(runner=str(spec["baseline"]["runner"]), family=model.family)
            overrides.update(
                {
                    "measurement.warmup": int(spec["measurement"]["warmup"]),
                    "measurement.iterations": int(spec["measurement"]["iterations"]),
                    "measurement.timing_scope": "public_task_call_wall",
                    "measurement.asset_loading_included": bool(timing["asset_loading_included"]),
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
            baseline_timing = _baseline_timing(spec, timing)
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


def _baseline_timing(spec: Mapping[str, Any], declared: Mapping[str, Any]) -> dict[str, Any]:
    baseline = spec["baseline"]
    result = {
        "timing_scope": declared["timing_scope"],
        "input_preparation_included": declared["input_preparation_included"],
        "asset_loading_included": declared["asset_loading_included"],
    }
    for name in tuple(result):
        if name in baseline:
            result[name] = baseline[name]
    return result


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
    if entry.spec["operation"] in {"encode", "embed"}:
        return "encoder"
    return "seq2seq-lm" if entry.model.family in SEQUENCE_FAMILIES else "causal-lm"


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


def baseline_command(entry: ResolvedEntry, environment: Environment, output: Path) -> list[str]:
    baseline = entry.spec["baseline"]
    runner = str(baseline["runner"])
    request = json.dumps(entry.case.request, ensure_ascii=True, separators=(",", ":"))
    common = [
        "--model",
        str(_adapter_options(entry, environment).get("model_id", entry.model.hf_id)),
        "--request-json",
        request,
        "--precision",
        entry.reference_precision,
        "--mode",
        str(baseline.get("mode", "torch-compile")),
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
        arguments = [
            sys.executable,
            str(environment.task_runner),
            "--adapter",
            str(baseline["adapter"]),
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
    revision = entry.model.hf_revision
    if revision:
        arguments.extend(("--revision", revision))
    if bool(entry.manifest.get("trust_remote_code", False)):
        arguments.append("--trust-remote-code")
    if bool(baseline.get("local_files_only", environment.local_files_only)):
        arguments.append("--local-files-only")
    return arguments


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


def _contract_name(entry: ResolvedEntry) -> str:
    configured = entry.spec["baseline"].get("output_contract")
    if configured:
        contract = str(configured)
    elif entry.spec["operation"] == "generate":
        if float(entry.case.request.get("temperature", 0.0)) > 0.0:
            contract = "generated-token-count"
        else:
            contract = "exact-token-ids"
    else:
        contract = {
            "classify": "classification-top-class",
            "embed": "embedding-shape",
            "encode": "embedding-shape",
            "control": "robot-action-shape",
            "segment": "segmentation-shape",
            "solve": "forecast-shape",
            "transcribe": "transcription-text",
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
    if contract == "forecast-shape":
        left_elements = left.get("forecast_elements", left.get("element_count"))
        right_elements = right.get("forecast_elements", right.get("element_count"))
        left_shape = left.get("shape")
        right_shape = right.get("shape")
        matched = (
            isinstance(left_elements, int)
            and not isinstance(left_elements, bool)
            and left_elements > 0
            and left_elements == right_elements
            and isinstance(left_shape, list)
            and isinstance(right_shape, list)
            and sorted(left_shape) == sorted(right_shape)
        )
        return matched, "forecast output shape differs" if not matched else "", None
    raise PerfMatrixError(f"output contract is not implemented: {contract}")


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
    rows = [dict(row) for row in results.get("rows", []) if isinstance(row, Mapping)]
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


def _report_html(report: Mapping[str, Any]) -> str:
    rows = []
    for row in report.get("rows", []):
        comparison = row.get("comparison", {}) if isinstance(row, Mapping) else {}
        measurement_stability = (
            row.get("measurement_stability", {}) if isinstance(row, Mapping) else {}
        )
        candidate = comparison.get("candidate_p50_ms", "")
        reference = comparison.get("reference_p50_ms", "")
        reason = comparison.get("reason", row.get("error", ""))
        stability = (
            measurement_stability.get("status", "")
            if isinstance(measurement_stability, Mapping)
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('id', '')))}</td>"
            f"<td>{html.escape(str(row.get('model', '')))}</td>"
            f"<td>{html.escape(str(row.get('operation', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(stability))}</td>"
            f"<td>{html.escape(str(candidate))}</td>"
            f"<td>{html.escape(str(reference))}</td>"
            f"<td>{html.escape(str(reason))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TRTMC performance</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th,td {{ border: 1px solid #ddd; padding: .5rem; text-align: left; }}
th {{ background: #f3f3f3; }}
</style></head><body>
<h1>TRTMC performance matrix</h1>
<p>Status: {html.escape(str(report.get("status", "unknown")))};
selected: {html.escape(str(report.get("summary", {}).get("selected", 0)))};
pending: {html.escape(str(report.get("summary", {}).get("pending", 0)))}</p>
<table><thead><tr><th>Entry</th><th>Model</th><th>Operation</th><th>Status</th>
<th>Stability</th><th>Candidate p50 ms</th><th>Reference p50 ms</th><th>Reason</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p>Green is faster by more than the margin, yellow is within the margin, red is
slower by more than the margin, and white is not comparable.</p>
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


def _load_results(run_directory: Path) -> dict[str, Any]:
    value = _json_file(run_directory.resolve() / "results.json", "matrix results")
    if value.get("schema_version") != RESULT_SCHEMA:
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
            results = _load_results(run_directory)
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

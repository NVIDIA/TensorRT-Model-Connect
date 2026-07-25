#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and report a TRTMC-versus-reference performance matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = REPOSITORY / "python"
for source_root in (REPOSITORY, PYTHON_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from benchmarks.performance.baselines.timing_contracts import timing_contract  # noqa: E402
from tensorrt_model_connect.benchmark.catalog import ManifestCatalog  # noqa: E402
from tensorrt_model_connect.benchmark.types import BenchmarkError  # noqa: E402
from tensorrt_model_connect.benchmark.worker import (  # noqa: E402
    find_worker,
    worker_metadata,
)
from tensorrt_model_connect.python_profiles import (  # noqa: E402
    default_execution_profiles,
    resolve_profile_python,
)


RESULT_SCHEMA = "trtmc.perf-matrix/v1"
SUITE_SCHEMA = "trtmc.perf-suite/v2"
ENVIRONMENT_SCHEMA = "trtmc.perf-environment/v1"
SEQUENCE_RUNTIME_MARKERS = ("bart_", "marian_", "m2m_100_", "t5_")
TASK_REFERENCE_ADAPTERS = {
    "hf-diffusers",
    "hf-qwen3-omni",
    "hf-transformers-asr",
    "hf-transformers-embedding",
    "hf-transformers-reranking",
    "hf-transformers-tts",
    "hf-transformers-vision",
    "hf-transformers-vlm",
    "nemo-asr",
    "nemo-tts",
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-lance",
    "upstream-sana-wm",
}
REPRODUCTION_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "TRANSFORMERS_CACHE",
    "TRTMC_BENCH_MANIFEST_ROOT",
    "TRTMC_ELF_REFERENCE_REPO",
    "TRTMC_LANCE_REFERENCE_REPO",
    "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY",
    "TRTMC_PYTHON_PROFILE_ROOT",
    "TRTMC_SANA_WM_REFERENCE_REPO",
    "PERSONAPLEX_OFFICIAL_REPO",
)


class PerfMatrixError(RuntimeError):
    """The performance matrix request or evidence is invalid."""


@dataclass(frozen=True)
class RunOptions:
    output: Path
    scratch_root: Path
    trtmc_bench: str
    trtmc_worker: Path | None
    hf_transformers_runner: Path
    task_reference_runner: Path
    bundle_cache: Path | None
    bundle_roots: tuple[Path, ...]
    runtime_dirs: tuple[Path, ...]
    local_files_only: bool
    minimum_free_space_gib: int
    timeout_seconds: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "resolve and validate the selected matrix entries"),
        ("run", "run the selected matrix entries"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("suite", type=Path, help="performance suite YAML")
        command.add_argument(
            "--environment",
            required=True,
            type=Path,
            help="execution environment YAML",
        )
        command.add_argument(
            "--entry",
            action="append",
            default=[],
            help="exact matrix entry id; repeatable",
        )
    resume = commands.add_parser("resume", help="continue an incomplete run")
    resume.add_argument("run_directory", type=Path)
    return parser


def _read_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PerfMatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerfMatrixError(f"{label} must contain a YAML object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = _read_yaml_object(path, "performance suite")
    if value.get("schema_version") != SUITE_SCHEMA:
        raise PerfMatrixError(f"suite schema_version must be {SUITE_SCHEMA!r}")
    return value


_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment_value(value: str, field: str) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        configured = os.environ.get(name)
        if configured is None or not configured.strip():
            missing.append(name)
            return ""
        return configured

    expanded = _ENVIRONMENT_VARIABLE.sub(replace, value)
    if missing:
        raise PerfMatrixError(
            f"environment {field} requires: {', '.join(sorted(set(missing)))}"
        )
    return expanded


def _resolved_path(value: str, field: str) -> Path:
    expanded = _expand_environment_value(value, field)
    path = Path(expanded).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _resolved_executable(value: str, field: str) -> str:
    expanded = _expand_environment_value(value, field)
    if Path(expanded).is_absolute() or os.sep in expanded:
        return str(_resolved_path(expanded, field))
    return expanded


def _resolved_path_list(value: Any, field: str) -> tuple[Path, ...]:
    if isinstance(value, str):
        expanded = _expand_environment_value(value, field)
        configured = [item for item in expanded.split(os.pathsep) if item]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        configured = [
            _expand_environment_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    else:
        raise PerfMatrixError(f"environment {field} must be a path list or path-list string")
    return tuple(_resolved_path(item, field) for item in configured)


def _read_environment(path: Path) -> dict[str, Any]:
    source = path.resolve()
    raw = _read_yaml_object(source, "performance environment")
    if raw.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise PerfMatrixError(
            f"environment schema_version must be {ENVIRONMENT_SCHEMA!r}"
        )
    name = raw.get("name")
    tools = raw.get("tools")
    storage = raw.get("storage")
    execution = raw.get("execution")
    if not isinstance(name, str) or not name.strip():
        raise PerfMatrixError("environment name must be a non-empty string")
    for field, value in (
        ("tools", tools),
        ("storage", storage),
        ("execution", execution),
    ):
        if not isinstance(value, Mapping):
            raise PerfMatrixError(f"environment {field} must be an object")
    assert isinstance(tools, Mapping)
    assert isinstance(storage, Mapping)
    assert isinstance(execution, Mapping)
    required_tools = (
        "trtmc_bench",
        "trtmc_worker",
        "hf_transformers_runner",
        "task_reference_runner",
    )
    missing_tools = [
        field
        for field in required_tools
        if not isinstance(tools.get(field), str) or not str(tools[field]).strip()
    ]
    if missing_tools:
        raise PerfMatrixError(
            f"environment tools is missing: {', '.join(missing_tools)}"
        )
    required_storage = ("results_root", "scratch_root", "bundle_roots", "runtime_dirs")
    missing_storage = [field for field in required_storage if field not in storage]
    if missing_storage:
        raise PerfMatrixError(
            f"environment storage is missing: {', '.join(missing_storage)}"
        )
    bundle_cache = storage.get("bundle_cache")
    if bundle_cache is not None and not isinstance(bundle_cache, str):
        raise PerfMatrixError("environment storage.bundle_cache must be a path or null")
    timeout_seconds = execution.get("timeout_seconds")
    minimum_free_space_gib = storage.get("minimum_free_space_gib", 0)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise PerfMatrixError("environment execution.timeout_seconds must be positive")
    if (
        isinstance(minimum_free_space_gib, bool)
        or not isinstance(minimum_free_space_gib, int)
        or minimum_free_space_gib < 0
    ):
        raise PerfMatrixError(
            "environment storage.minimum_free_space_gib must be non-negative"
        )
    local_files_only = execution.get("local_files_only", False)
    if not isinstance(local_files_only, bool):
        raise PerfMatrixError("environment execution.local_files_only must be boolean")
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "name": name.strip(),
        "source": str(source),
        "sha256": _sha256_file(source),
        "tools": {
            "trtmc_bench": _resolved_executable(
                str(tools["trtmc_bench"]), "tools.trtmc_bench"
            ),
            "trtmc_worker": str(
                _resolved_path(str(tools["trtmc_worker"]), "tools.trtmc_worker")
            ),
            "hf_transformers_runner": str(
                _resolved_path(
                    str(tools["hf_transformers_runner"]),
                    "tools.hf_transformers_runner",
                )
            ),
            "task_reference_runner": str(
                _resolved_path(
                    str(tools["task_reference_runner"]),
                    "tools.task_reference_runner",
                )
            ),
        },
        "storage": {
            "results_root": str(
                _resolved_path(str(storage["results_root"]), "storage.results_root")
            ),
            "scratch_root": str(
                _resolved_path(str(storage["scratch_root"]), "storage.scratch_root")
            ),
            "bundle_cache": (
                str(_resolved_path(bundle_cache, "storage.bundle_cache"))
                if bundle_cache is not None
                else None
            ),
            "bundle_roots": [
                str(path)
                for path in _resolved_path_list(
                    storage["bundle_roots"], "storage.bundle_roots"
                )
            ],
            "runtime_dirs": [
                str(path)
                for path in _resolved_path_list(
                    storage["runtime_dirs"], "storage.runtime_dirs"
                )
            ],
            "minimum_free_space_gib": minimum_free_space_gib,
        },
        "execution": {
            "local_files_only": local_files_only,
            "timeout_seconds": timeout_seconds,
        },
    }


def _run_options(environment: Mapping[str, Any], output: Path) -> RunOptions:
    tools = environment["tools"]
    storage = environment["storage"]
    execution = environment["execution"]
    return RunOptions(
        output=output,
        scratch_root=Path(str(storage["scratch_root"])),
        trtmc_bench=str(tools["trtmc_bench"]),
        trtmc_worker=Path(str(tools["trtmc_worker"])),
        hf_transformers_runner=Path(str(tools["hf_transformers_runner"])),
        task_reference_runner=Path(str(tools["task_reference_runner"])),
        bundle_cache=(
            Path(str(storage["bundle_cache"])) if storage.get("bundle_cache") else None
        ),
        bundle_roots=tuple(Path(str(value)) for value in storage["bundle_roots"]),
        runtime_dirs=tuple(Path(str(value)) for value in storage["runtime_dirs"]),
        local_files_only=bool(execution["local_files_only"]),
        minimum_free_space_gib=int(storage["minimum_free_space_gib"]),
        timeout_seconds=int(execution["timeout_seconds"]),
    )


def _cases(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    defaults = suite.get("defaults", {})
    configured = suite.get("entries")
    additional_profiles = suite.get("additional_profiles", [])
    if not isinstance(defaults, Mapping):
        raise PerfMatrixError("suite defaults must be an object")
    if not isinstance(configured, list) or not configured:
        raise PerfMatrixError("suite entries must be a non-empty list")
    if not isinstance(additional_profiles, list):
        raise PerfMatrixError("suite additional_profiles must be a list")
    cases: list[dict[str, Any]] = []
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerfMatrixError("every suite entry must be an object")
        merged = _merge_case(defaults, raw)
        _validate_case_shape(merged)
        cases.append(merged)
    cases.extend(_additional_profile_cases(cases, additional_profiles))
    _validate_unique_ids(cases)
    return cases


def _additional_profile_cases(
    base_cases: Sequence[Mapping[str, Any]],
    configured: Sequence[Any],
) -> list[dict[str, Any]]:
    templates = {str(case["id"]): case for case in base_cases}
    cases: list[dict[str, Any]] = []
    allowed = {"id", "model", "inherit", "workload", "measurement", "baseline"}
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerfMatrixError("every additional profile must be an object")
        unsupported = sorted(set(raw) - allowed)
        if unsupported:
            raise PerfMatrixError(
                "additional profile has unsupported fields: " + ", ".join(unsupported)
            )
        model = raw.get("model")
        inherited_id = raw.get("inherit")
        if not isinstance(model, str) or not model.strip():
            raise PerfMatrixError("additional profile model must be a non-empty string")
        if not isinstance(inherited_id, str) or inherited_id not in templates:
            raise PerfMatrixError(
                f"additional profile {model} inherits unknown entry {inherited_id!r}"
            )
        configured_id = raw.get("id")
        if configured_id is not None and (
            not isinstance(configured_id, str) or not configured_id.strip()
        ):
            raise PerfMatrixError(f"additional profile {model} id must be a non-empty string")
        overrides = {
            key: value
            for key, value in raw.items()
            if key in {"measurement", "baseline"}
        }
        case = _merge_case(templates[inherited_id], overrides)
        case["id"] = configured_id or f"{inherited_id}@{model}"
        case["model"] = model
        workload = deepcopy(dict(case["workload"]))
        workload["testcase"] = model
        configured_workload = raw.get("workload")
        if configured_workload is not None:
            if not isinstance(configured_workload, Mapping):
                raise PerfMatrixError(
                    f"additional profile {model} workload must be an object"
                )
            workload.update(deepcopy(dict(configured_workload)))
        case["workload"] = workload
        _validate_case_shape(case)
        cases.append(case)
    return cases


def _merge_case(defaults: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in case.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_case_shape(case: Mapping[str, Any]) -> None:
    required = ("id", "family", "operation", "model", "workload", "measurement", "baseline")
    missing = [name for name in required if name not in case]
    if missing:
        raise PerfMatrixError(f"case is missing {', '.join(missing)}: {case}")
    _validate_workload(case)
    _validate_measurement(case)
    _validate_baseline(case)


def _validate_workload(case: Mapping[str, Any]) -> None:
    workload = case["workload"]
    if not isinstance(workload, Mapping):
        raise PerfMatrixError(f"case {case['id']} workload must be an object")
    unsupported = sorted(set(workload) - {"testcase", "request"})
    if unsupported:
        raise PerfMatrixError(
            f"case {case['id']} workload has unsupported fields: {', '.join(unsupported)}"
        )
    testcase = workload.get("testcase")
    if not isinstance(testcase, str) or not testcase.strip():
        raise PerfMatrixError(
            f"case {case['id']} workload.testcase must be explicit; "
            "dataset workloads are not implemented yet"
        )
    request = workload.get("request", {})
    if not isinstance(request, Mapping):
        raise PerfMatrixError(f"case {case['id']} workload.request must be an object")


def _validate_measurement(case: Mapping[str, Any]) -> None:
    measurement = case["measurement"]
    if not isinstance(measurement, Mapping):
        raise PerfMatrixError(f"case {case['id']} measurement must be an object")
    for name in ("warmup", "iterations"):
        value = measurement.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "warmup" else 1)
        ):
            raise PerfMatrixError(f"case {case['id']} measurement.{name} is invalid")


def _validate_baseline(case: Mapping[str, Any]) -> None:
    baseline = case["baseline"]
    if not isinstance(baseline, Mapping) or baseline.get("runner") not in {
        "hf-transformers",
        "task-reference",
    }:
        raise PerfMatrixError(f"case {case['id']} has an unsupported baseline runner")
    if baseline.get("runner") == "task-reference":
        adapter = str(baseline.get("adapter", ""))
        if adapter not in TASK_REFERENCE_ADAPTERS:
            raise PerfMatrixError(
                f"case {case['id']} has an unsupported task-reference adapter: {adapter}"
            )
        if not baseline.get("reference_backend"):
            raise PerfMatrixError(
                f"case {case['id']} task-reference baseline needs a reference_backend"
            )
        if not isinstance(baseline.get("adapter_options", {}), Mapping):
            raise PerfMatrixError(
                f"case {case['id']} task-reference adapter_options must be an object"
            )
        expected_mode = (
            "pytorch-eager"
            if adapter
            in {
                "nemo-asr",
                "nemo-tts",
                "pytorch-personaplex",
                "pytorch-timeseries",
                "upstream-elf",
                "upstream-lance",
                "upstream-sana-wm",
            }
            else "hf-eager"
        )
        if baseline.get("mode") != expected_mode:
            raise PerfMatrixError(
                f"case {case['id']} adapter {adapter} requires mode {expected_mode}"
            )
    token_policy = baseline.get("output_token_policy", "new-tokens")
    if token_policy not in {"new-tokens", "strip-start", "strip-start-and-eos"}:
        raise PerfMatrixError(f"case {case['id']} baseline output token policy is invalid")
    if baseline.get("padding", "longest") not in {"longest", "max-length"}:
        raise PerfMatrixError(f"case {case['id']} baseline padding is invalid")
    if baseline.get("precision") not in {None, "fp16", "fp32", "bf16"}:
        raise PerfMatrixError(f"case {case['id']} baseline precision is invalid")
    if baseline.get("model_class", "task") not in {"task", "auto"}:
        raise PerfMatrixError(f"case {case['id']} baseline model class is invalid")
    if baseline.get("generation_method", "generate") not in {"generate", "ar-generate"}:
        raise PerfMatrixError(f"case {case['id']} baseline generation method is invalid")
    if baseline.get("experts_implementation") not in {
        None,
        "eager",
        "batched_mm",
        "grouped_mm",
    }:
        raise PerfMatrixError(f"case {case['id']} baseline experts implementation is invalid")
    output_contract = baseline.get("output_contract", "exact-token-ids")
    if output_contract not in {
        "audio-shape",
        "exact-token-ids",
        "exact-text",
        "generated-token-count",
        "media-shape",
        "normalized-text",
        "ocr-text",
        "segmentation-shape",
        "token-agreement",
    }:
        raise PerfMatrixError(f"case {case['id']} baseline output contract is invalid")
    if output_contract == "token-agreement":
        for name in ("min_positional_token_agreement", "max_normalized_edit_distance"):
            value = baseline.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise PerfMatrixError(
                    f"case {case['id']} token-agreement contract has invalid {name}"
                )
    if output_contract == "ocr-text":
        required = baseline.get("required_substrings")
        if (
            not isinstance(required, list)
            or not required
            or any(not isinstance(value, str) or not value.strip() for value in required)
        ):
            raise PerfMatrixError(
                f"case {case['id']} OCR output contract needs required_substrings"
            )
        limit = baseline.get("max_normalized_edit_distance")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not 0.0 <= float(limit) <= 1.0
        ):
            raise PerfMatrixError(
                f"case {case['id']} OCR output contract has an invalid edit-distance limit"
            )
    expected_timing = timing_contract(
        runner=str(baseline["runner"]),
        family=str(case["family"]),
    )
    for name in (
        "timing_scope",
        "input_preparation_included",
        "asset_loading_included",
    ):
        if baseline.get(name) != expected_timing[name]:
            raise PerfMatrixError(
                f"case {case['id']} baseline.{name} must be "
                f"{expected_timing[name]!r} for its reference"
            )
    mode = baseline.get("mode", "torch-compile")
    allowed_modes = (
        {"hf-eager", "pytorch-eager"}
        if baseline.get("runner") == "task-reference"
        else {"torch-compile", "hf-eager"}
    )
    if mode not in allowed_modes:
        raise PerfMatrixError(f"case {case['id']} baseline mode is invalid: {mode}")


def _validate_unique_ids(cases: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(case["id"]) for case in cases]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise PerfMatrixError(f"duplicate entry ids: {', '.join(duplicates)}")


def _validate_coverage(cases: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        entry.name: (entry.family, entry.operation)
        for entry in ManifestCatalog().entries()
        if entry.status == "ready"
    }
    actual_models = [str(case["model"]) for case in cases]
    actual = set(actual_models)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    duplicates = sorted(
        model for model, count in Counter(actual_models).items() if count > 1
    )
    mismatched = sorted(
        (
            model,
            f"{case['family']}.{case['operation']}",
            f"{expected[model][0]}.{expected[model][1]}",
        )
        for case in cases
        if (model := str(case["model"])) in expected
        and (str(case["family"]), str(case["operation"])) != expected[model]
    )
    if missing or extra or duplicates or mismatched:
        details = _coverage_details(missing, extra, duplicates, mismatched)
        raise PerfMatrixError(
            "suite profile coverage does not match ready catalog: " + "; ".join(details)
        )


def _coverage_details(
    missing: Sequence[str],
    extra: Sequence[str],
    duplicates: Sequence[str],
    mismatched: Sequence[tuple[str, str, str]],
) -> list[str]:
    details = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if extra:
        details.append("extra=" + ",".join(extra))
    if duplicates:
        details.append("duplicate=" + ",".join(duplicates))
    if mismatched:
        details.append(
            "family-operation="
            + ",".join(f"{model}:{actual}!={expected}" for model, actual, expected in mismatched)
        )
    return details


def _selected_cases(
    cases: Sequence[dict[str, Any]],
    requested: Sequence[str],
) -> list[dict[str, Any]]:
    requested = [value for value in requested if value]
    _validate_requested_cases(cases, requested)
    selected = [case for case in cases if not requested or case["id"] in requested]
    selected.sort(key=lambda case: str(case["id"]))
    return selected


def _validate_requested_cases(cases: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> None:
    known = {str(case["id"]) for case in cases}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PerfMatrixError(f"unknown entry ids: {', '.join(unknown)}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    configured = os.environ.get("TRTMC_PERF_SOURCE_REVISION") or os.environ.get("GITHUB_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"gpu": None, "driver": None}
    try:
        output = subprocess.run(
            [executable, "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        name, uuid, driver = (part.strip() for part in output.split(",", maxsplit=2))
        return {"gpu": name, "gpu_uuid": uuid, "driver": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {"gpu": None, "driver": None}


def _initial_results(
    suite_path: Path,
    suite: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    environment_config: Mapping[str, Any],
) -> dict[str, Any]:
    catalog_counts = Counter(entry.status for entry in ManifestCatalog().entries())
    environment = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        **_gpu_environment(),
    }
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "running",
        "suite": str(suite_path.resolve()),
        "suite_name": str(suite.get("name", suite_path.stem)),
        "suite_sha256": _sha256_file(suite_path),
        "repository_root": str(REPOSITORY),
        "git_commit": _git_commit(),
        "started_at": _now(),
        "environment": environment,
        "environment_config": deepcopy(dict(environment_config)),
        "catalog_coverage": {
            "total_profiles": sum(catalog_counts.values()),
            "ready_profiles": catalog_counts["ready"],
            "distributed_profiles": catalog_counts["distributed"],
            "other_profiles": sum(
                count
                for status, count in catalog_counts.items()
                if status not in {"ready", "distributed"}
            ),
        },
        "selected_entry_ids": [str(case["id"]) for case in selected],
        "cases": [
            {
                "id": case["id"],
                "family": case["family"],
                "operation": case["operation"],
                "model": case["model"],
                "workload_contract": deepcopy(dict(case["workload"])),
                "measurement_contract": deepcopy(dict(case["measurement"])),
                "baseline_contract": dict(case["baseline"]),
                "status": "pending",
            }
            for case in cases
        ],
    }


def _load_resume(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot resume results {path}: {exc}") from exc
    if value.get("schema_version") != RESULT_SCHEMA:
        raise PerfMatrixError(f"cannot resume non-{RESULT_SCHEMA} results")
    if not isinstance(value.get("cases"), list):
        raise PerfMatrixError("cannot resume results without matrix entries")
    value["status"] = "running"
    value.pop("finished_at", None)
    return value


def _result_rows(results: MutableMapping[str, Any]) -> dict[str, MutableMapping[str, Any]]:
    return {str(row["id"]): row for row in results["cases"]}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _yaml_cli_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _candidate_base_argv(case: Mapping[str, Any], options: RunOptions) -> list[str]:
    measurement = case["measurement"]
    argv = [
        options.trtmc_bench,
        "run",
        "--model",
        str(case["model"]),
        "--warmup",
        str(measurement["warmup"]),
        "--iterations",
        str(measurement["iterations"]),
        "--telemetry",
        "off",
        "--set",
        "measurement.timing_scope="
        + _yaml_cli_value(_candidate_timing_scope(case)),
        "--set",
        "measurement.asset_loading_included="
        + _yaml_cli_value(bool(case["baseline"]["asset_loading_included"])),
    ]
    workload = case["workload"]
    argv.extend(["--case", str(workload["testcase"])])
    if options.bundle_cache is not None:
        argv.extend(["--bundle-cache", str(options.bundle_cache.resolve())])
    if options.trtmc_worker is not None:
        argv.extend(["--worker", str(options.trtmc_worker.resolve())])
    for root in options.bundle_roots:
        argv.extend(["--bundle-root", str(root.resolve())])
    for directory in options.runtime_dirs:
        argv.extend(["--runtime-dir", str(directory.resolve())])
    request = workload.get("request", {})
    for name, value in sorted((request or {}).items()):
        argv.extend(["--set", f"request.{name}={_yaml_cli_value(value)}"])
    return argv


def _candidate_timing_scope(case: Mapping[str, Any]) -> str:
    return str(
        timing_contract(
            runner=str(case["baseline"]["runner"]),
            family=str(case["family"]),
        )["candidate_timing_scope"]
    )


def _candidate_build_python_profile(resolved: Mapping[str, Any]) -> tuple[str, str]:
    model = resolved["model"]
    profile = default_execution_profiles(family=str(model["family"]))["build"]
    try:
        python = resolve_profile_python(profile, sys.executable)
    except (OSError, ValueError, RuntimeError) as exc:
        raise PerfMatrixError(
            f"candidate build Python profile {profile!r} is unavailable: {exc}"
        ) from exc
    return profile, python


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    paths = [str(PYTHON_SOURCE)]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _resolve_candidate(
    case: Mapping[str, Any], options: RunOptions
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    argv = [*_candidate_base_argv(case, options), "--dry-run"]
    command = _run_command(argv, _command_environment(), options.timeout_seconds)
    if command["exit_code"] != 0:
        raise PerfMatrixError(
            f"trtmc-bench could not resolve {case['id']}: {command['stderr_tail']}"
        )
    try:
        resolved = json.loads(command["stdout"])
    except json.JSONDecodeError as exc:
        raise PerfMatrixError(f"trtmc-bench dry-run returned invalid JSON: {exc}") from exc
    if not isinstance(resolved, list) or len(resolved) != 1 or not isinstance(resolved[0], dict):
        raise PerfMatrixError(f"case {case['id']} must resolve to exactly one trtmc-bench case")
    value = resolved[0]
    if (value.get("model", {}).get("family"), value.get("operation")) != (
        case["family"],
        case["operation"],
    ):
        raise PerfMatrixError(f"case {case['id']} does not match its resolved family-operation")
    measurement = value.get("measurement", {})
    expected_scope = _candidate_timing_scope(case)
    if not isinstance(measurement, Mapping) or measurement.get("timing_scope") != expected_scope:
        raise PerfMatrixError(
            f"case {case['id']} TRTMC timing scope did not resolve to {expected_scope}"
        )
    expected_asset_loading = bool(case["baseline"]["asset_loading_included"])
    if measurement.get("asset_loading_included") is not expected_asset_loading:
        raise PerfMatrixError(
            f"case {case['id']} TRTMC asset-loading scope did not resolve to "
            f"{expected_asset_loading}"
        )
    request = value.get("request", {})
    if expected_asset_loading and not any(
        name in request for name in ("audio_path", "image_path")
    ):
        raise PerfMatrixError(
            f"case {case['id']} includes asset loading but resolves no timed asset path"
        )
    profile, _ = _candidate_build_python_profile(value)
    value["_candidate_build_python_profile"] = profile
    command.pop("stdout", None)
    return value, argv, command


def _validate_worker_metadata(
    metadata: Mapping[str, Any], expected_revision: str
) -> None:
    if metadata.get("schema_version") != "trtmc.benchmark-worker-metadata/v1":
        raise PerfMatrixError("worker metadata schema is unsupported")
    build = metadata.get("build")
    if not isinstance(build, Mapping):
        raise PerfMatrixError("worker metadata is missing build provenance")
    if build.get("configuration") != "Release":
        raise PerfMatrixError("worker build configuration must be Release")
    revision = str(build.get("source_revision", "")).strip()
    if revision != expected_revision:
        raise PerfMatrixError(
            f"worker source revision {revision or '<missing>'} does not match "
            f"requested source revision {expected_revision}"
        )


def _preflight_worker(options: RunOptions) -> dict[str, Any]:
    expected_revision = _git_commit()
    if not expected_revision:
        raise PerfMatrixError("cannot determine requested source revision for worker preflight")
    try:
        worker = find_worker(options.trtmc_worker)
        metadata = worker_metadata(worker)
    except BenchmarkError as exc:
        raise PerfMatrixError(f"candidate worker preflight failed: {exc}") from exc
    _validate_worker_metadata(metadata, expected_revision)
    return {
        **metadata,
        "path": str(worker),
        "validated_against": expected_revision,
    }


def _preflight_candidates(
    cases: Sequence[Mapping[str, Any]],
    options: RunOptions,
) -> tuple[
    dict[str, tuple[dict[str, Any], list[str], dict[str, str]]],
    dict[str, dict[str, Any]],
]:
    resolved: dict[str, tuple[dict[str, Any], list[str], dict[str, str]]] = {}
    failures: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["id"])
        print(f"[{case_id}] timing preflight", flush=True)
        try:
            resolved[case_id] = _resolve_candidate(case, options)
        except (PerfMatrixError, OSError, ValueError, RuntimeError) as exc:
            argv = [*_candidate_base_argv(case, options), "--dry-run"]
            failures[case_id] = {
                "stage": "candidate-preflight",
                "reason": str(exc),
                "argv": argv,
            }
    return resolved, failures


def _preflight_references(
    cases: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, tuple[dict[str, Any], list[str], dict[str, str]]],
    options: RunOptions,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    entries = []
    failures: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["id"])
        if case_id not in preflight:
            continue
        resolved = preflight[case_id][0]
        output = options.scratch_root / "preflight" / _slug(case_id) / "baseline.json"
        try:
            argv, profile = _baseline_argv(case, resolved, output, options)
            executable = argv[0]
            if os.sep in executable:
                if not Path(executable).is_file():
                    raise PerfMatrixError(
                        f"reference Python does not exist: {executable}"
                    )
            elif shutil.which(executable) is None:
                raise PerfMatrixError(f"reference Python is not on PATH: {executable}")
            entries.append(
                {
                    "id": case_id,
                    "runner": case["baseline"]["runner"],
                    "mode": case["baseline"].get("mode", "torch-compile"),
                    "python_profile": profile,
                    "python": executable,
                }
            )
        except (PerfMatrixError, OSError, ValueError, RuntimeError) as exc:
            failures[case_id] = {
                "stage": "reference-preflight",
                "reason": str(exc),
                "argv": preflight[case_id][1],
            }
    return (
        {
            "checked_at": _now(),
            "entry_count": len(entries),
            "failed_entry_count": len(failures),
            "status": "partial" if failures else "ready",
            "entries": entries,
            "failures": [
                {"id": case_id, **failure}
                for case_id, failure in sorted(failures.items())
            ],
        },
        failures,
    )


def _preflight_selected(
    cases: Sequence[Mapping[str, Any]],
    options: RunOptions,
) -> tuple[
    dict[str, tuple[dict[str, Any], list[str], dict[str, str]]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    preflight, failures = _preflight_candidates(cases, options)
    references, reference_failures = _preflight_references(cases, preflight, options)
    failures.update(reference_failures)
    for case_id in reference_failures:
        preflight.pop(case_id, None)
    return preflight, references, failures


def _preflight_evidence(
    cases: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, tuple[dict[str, Any], list[str], dict[str, str]]],
) -> dict[str, Any]:
    contracts = []
    for case in cases:
        expected = timing_contract(
            runner=str(case["baseline"]["runner"]),
            family=str(case["family"]),
        )
        resolved = preflight[str(case["id"])][0]
        contracts.append(
            {
                "id": case["id"],
                "reference_timing_scope": expected["timing_scope"],
                "candidate_timing_scope": resolved["measurement"]["timing_scope"],
                "candidate_build_python_profile": resolved["_candidate_build_python_profile"],
                "input_preparation_included": expected["input_preparation_included"],
                "asset_loading_included": expected["asset_loading_included"],
                "status": "aligned",
            }
        )
    return {
        "checked_at": _now(),
        "case_count": len(contracts),
        "status": "aligned",
        "contracts": contracts,
    }


def _workload_digest(resolved: Mapping[str, Any]) -> str:
    model = resolved["model"]
    payload = {
        "schema_version": "trtmc.perf-workload/v1",
        "model": {
            "hf_id": model["hf_id"],
            "family": model["family"],
            "task_strategy": model["task_strategy"],
        },
        "operation": resolved["operation"],
        "request": resolved["request"],
        "precision": model["precision"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _manifest_values(resolved: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(resolved["model"]["manifest_path"]))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot read resolved model manifest {path}: {exc}") from exc
    return value


def _baseline_task(resolved: Mapping[str, Any]) -> str:
    strategy = str(resolved["model"]["task_strategy"])
    runtime = str(resolved["model"]["runtime_strategy"])
    if strategy == "encoder_only_nlp":
        return "encoder"
    if strategy != "text_generation_causal":
        raise PerfMatrixError(f"hf-transformers runner does not support task {strategy!r}")
    return "seq2seq-lm" if runtime.startswith(SEQUENCE_RUNTIME_MARKERS) else "causal-lm"


def _reference_python(resolved: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[str, str]:
    explicit = str(baseline.get("python_profile", "") or "").strip()
    model = resolved["model"]
    profile = (
        explicit
        or default_execution_profiles(
            family=str(model["family"]),
            runtime_strategy=str(model["runtime_strategy"]),
            reference_backend=str(baseline.get("reference_backend", "hf_transformers")),
        )["reference"]
    )
    return profile, resolve_profile_python(profile, sys.executable)


def _baseline_argv(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    output: Path,
    options: RunOptions,
) -> tuple[list[str], str]:
    baseline = case["baseline"]
    profile, python = _reference_python(resolved, baseline)
    model = resolved["model"]
    manifest = _manifest_values(resolved)
    mode = str(baseline.get("mode", "torch-compile"))
    if baseline.get("runner") == "task-reference":
        return _task_reference_argv(
            case=case,
            resolved=resolved,
            manifest=manifest,
            output=output,
            options=options,
            profile=profile,
            python=python,
            mode=mode,
        ), profile
    build = model.get("build", {})
    max_length = int(baseline.get("max_length", build.get("max_cache_length", 256)))
    argv = [
        python,
        str(options.hf_transformers_runner.resolve()),
        "--model",
        str(model["hf_id"]),
        "--task",
        str(baseline.get("task") or _baseline_task(resolved)),
        "--request-json",
        json.dumps(resolved["request"], ensure_ascii=True, separators=(",", ":")),
        "--precision",
        str(baseline.get("precision", model["precision"])),
        "--max-length",
        str(max_length),
        "--padding",
        str(baseline.get("padding", "longest")),
        "--mode",
        mode,
        "--warmup",
        str(case["measurement"]["warmup"]),
        "--iterations",
        str(case["measurement"]["iterations"]),
        "--workload-digest",
        _workload_digest(resolved),
        "--output-token-policy",
        str(baseline.get("output_token_policy", "new-tokens")),
        "--output",
        str(output),
    ]
    revision = baseline.get("revision", manifest.get("hf_revision"))
    if revision:
        argv.extend(["--revision", str(revision)])
    if bool(manifest.get("trust_remote_code", False)):
        argv.append("--trust-remote-code")
    if baseline.get("experts_implementation"):
        argv.extend(["--experts-implementation", str(baseline["experts_implementation"])])
    if baseline.get("model_class"):
        argv.extend(["--model-class", str(baseline["model_class"])])
    if baseline.get("generation_method"):
        argv.extend(["--generation-method", str(baseline["generation_method"])])
    if options.local_files_only:
        argv.append("--local-files-only")
    if mode == "torch-compile":
        argv.extend(["--compile-mode", str(baseline.get("compile_mode", "default"))])
        if bool(baseline.get("fullgraph", False)):
            argv.append("--compile-fullgraph")
        if bool(baseline.get("dynamic", True)):
            argv.append("--compile-dynamic")
    return argv, profile


def _task_reference_argv(
    *,
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output: Path,
    options: RunOptions,
    profile: str,
    python: str,
    mode: str,
) -> list[str]:
    del profile
    baseline = case["baseline"]
    model = resolved["model"]
    adapter_options = _resolved_adapter_options(baseline)
    argv = [
        python,
        str(options.task_reference_runner.resolve()),
        "--adapter",
        str(baseline["adapter"]),
        "--family",
        str(model["family"]),
        "--operation",
        str(resolved["operation"]),
        "--model",
        str(model["hf_id"]),
        "--manifest",
        str(model["manifest_path"]),
        "--request-json",
        json.dumps(resolved["request"], ensure_ascii=True, separators=(",", ":")),
        "--runtime-json",
        json.dumps(resolved.get("runtime", {}), ensure_ascii=True, separators=(",", ":")),
        "--adapter-options-json",
        json.dumps(adapter_options, ensure_ascii=True, separators=(",", ":")),
        "--timing-contract-json",
        json.dumps(
            {
                name: baseline[name]
                for name in (
                    "timing_scope",
                    "input_preparation_included",
                    "asset_loading_included",
                )
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "--precision",
        str(baseline.get("precision", model["precision"])),
        "--mode",
        mode,
        "--padding",
        str(baseline.get("padding", "longest")),
        "--warmup",
        str(case["measurement"]["warmup"]),
        "--iterations",
        str(case["measurement"]["iterations"]),
        "--workload-digest",
        _workload_digest(resolved),
        "--output",
        str(output),
    ]
    revision = baseline.get("revision", manifest.get("hf_revision"))
    if revision:
        argv.extend(["--revision", str(revision)])
    if bool(manifest.get("trust_remote_code", False)):
        argv.append("--trust-remote-code")
    if options.local_files_only:
        argv.append("--local-files-only")
    return argv


def _resolved_adapter_options(baseline: Mapping[str, Any]) -> dict[str, Any]:
    configured = baseline.get("adapter_options", {})
    options = dict(configured) if isinstance(configured, Mapping) else {}
    adapter = str(baseline.get("adapter", ""))
    external_checkout = {
        "upstream-elf": ("reference_repo", "TRTMC_ELF_REFERENCE_REPO"),
        "upstream-lance": ("reference_repo", "TRTMC_LANCE_REFERENCE_REPO"),
        "upstream-sana-wm": (
            "reference_repo",
            "TRTMC_SANA_WM_REFERENCE_REPO",
        ),
        "pytorch-personaplex": ("official_repo", "PERSONAPLEX_OFFICIAL_REPO"),
    }.get(adapter)
    if external_checkout is not None:
        option_name, environment_name = external_checkout
        environment_value = os.environ.get(environment_name, "").strip()
        if option_name not in options and environment_value:
            options[option_name] = environment_value
        if option_name not in options:
            raise PerfMatrixError(
                f"baseline adapter {adapter!r} requires adapter_options.{option_name} "
                f"or {environment_name}"
            )
    return options


def _run_command(
    argv: Sequence[str], environment: Mapping[str, str], timeout_seconds: int
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        process = subprocess.run(
            list(argv),
            cwd=REPOSITORY,
            env=dict(environment),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = _decode_timeout_stream(exc.stdout)
        stderr = (
            _decode_timeout_stream(exc.stderr) + f"\ncommand timed out after {timeout_seconds}s"
        )
    except OSError as exc:
        exit_code = 127
        stdout = ""
        stderr = str(exc)
    elapsed = time.monotonic() - started
    return {
        "argv": list(argv),
        "rendered": shlex.join(argv),
        "cwd": str(REPOSITORY),
        "environment": {
            name: environment[name]
            for name in REPRODUCTION_ENVIRONMENT_NAMES
            if environment.get(name)
        },
        "redacted_environment_names": [
            name for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if environment.get(name)
        ],
        "exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "stdout": stdout,
        "stdout_tail": stdout[-16000:],
        "stderr_tail": stderr[-16000:],
    }


def _gpu_memory_usage_mib() -> list[tuple[int, int]]:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if process.returncode != 0:
        return []
    usage = []
    for line in process.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            return []
        try:
            used, total = (int(field) for field in fields)
        except ValueError:
            return []
        if total <= 0 or used < 0:
            return []
        usage.append((used, total))
    return usage


def _wait_for_gpu_memory_headroom(
    *,
    timeout_seconds: float = 120.0,
    minimum_free_fraction: float = 0.45,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    announced = False
    while True:
        usage = _gpu_memory_usage_mib()
        if not usage or all(
            (total - used) / total >= minimum_free_fraction for used, total in usage
        ):
            return
        if time.monotonic() >= deadline:
            rendered = ", ".join(f"{used}/{total} MiB" for used, total in usage)
            raise PerfMatrixError(
                "GPU memory did not recover enough free headroom after a backend "
                f"process exited (required {minimum_free_fraction:.0%} free): {rendered}"
            )
        if not announced:
            rendered = ", ".join(f"{used}/{total} MiB" for used, total in usage)
            print(f"Waiting for GPU memory headroom: {rendered}", flush=True)
            announced = True
        time.sleep(1.0)


def _decode_timeout_stream(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _candidate_result(run_dir: Path, workload_digest: str) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot read trtmc-bench result: {exc}") from exc
    if result.get("schema_version") != "trtmc.benchmark-run/v1":
        raise PerfMatrixError("trtmc-bench returned an unsupported result schema")
    cells = result.get("cells", [])
    if result.get("status") != "completed" or len(cells) != 1:
        error = cells[0].get("error") if len(cells) == 1 else result.get("status")
        raise PerfMatrixError(f"trtmc-bench did not complete one case: {error}")
    cell = cells[0]
    artifact = run_dir / str(cell["artifact_dir"])
    observations = []
    try:
        for line in (artifact / "observations.jsonl").read_text(encoding="utf-8").splitlines():
            observations.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot read trtmc-bench observations: {exc}") from exc
    samples = [float(item["runtime_e2e_wall_ms"]) for item in observations]
    return {
        "schema_version": result["schema_version"],
        "status": "completed",
        "backend": "trtmc-bench",
        "workload_digest": workload_digest,
        "measurement_policy": result.get("measurement_policy", {}),
        "samples_ms": samples,
        "metrics": cell["metrics"],
        "output_summary": cell.get("output_summary", {}),
        "environment": result.get("environment", {}),
    }


def _read_baseline(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot read baseline result: {exc}") from exc
    if value.get("schema_version") != "trtmc.perf-baseline/v1":
        raise PerfMatrixError("baseline returned an unsupported result schema")
    return value


def _median(result: Mapping[str, Any]) -> float:
    values = result.get("samples_ms")
    if not isinstance(values, list) or not values:
        raise PerfMatrixError("backend result has no timing samples")
    return float(statistics.median(float(value) for value in values))


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().casefold()


def _normalized_text_edit_distance(left: Any, right: Any) -> float:
    a = _normalized_text(left)
    b = _normalized_text(right)
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, left_character in enumerate(a, start=1):
        current = [row]
        for column, right_character in enumerate(b, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def _missing_ocr_substrings(text: Any, required: Sequence[str]) -> list[str]:
    normalized = re.sub(r"\s*:\s*", ":", _normalized_text(text))
    return [
        value
        for value in required
        if re.sub(r"\s*:\s*", ":", _normalized_text(value)) not in normalized
    ]


def _output_contract(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    left = candidate.get("output_summary", {})
    right = baseline.get("output_summary", {})
    operation = str(case["operation"])
    contract = _effective_output_contract(case, request)
    if contract == "segmentation-shape":
        left_shape = tuple(left.get(name) for name in ("num_masks", "height", "width"))
        right_shape = tuple(right.get(name) for name in ("num_masks", "height", "width"))
        matched = None not in left_shape and left_shape == right_shape
        return matched, "segmentation output shape differs" if not matched else ""
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
        return matched, "audio output shape differs" if not matched else ""
    if contract == "media-shape":
        names = ("height", "width", "channels")
        media_type = right.get("media_type")
        if media_type == "image":
            left_count = left.get(
                "batch_size",
                left.get("media_count", left.get("num_frames")),
            )
        else:
            left_count = left.get(
                "num_frames",
                left.get("media_count", left.get("batch_size")),
            )
        left_shape = (
            left_count,
            *(left.get(name) for name in names),
        )
        right_shape = (
            right.get("num_frames", right.get("media_count")),
            *(right.get(name) for name in names),
        )
        matched = None not in left_shape and left_shape == right_shape
        return matched, "media output shape differs" if not matched else ""
    if operation == "generate":
        if contract == "generated-token-count":
            left_tokens = left.get("token_ids")
            right_tokens = right.get("token_ids")
            left_count = len(left_tokens) if isinstance(left_tokens, list) else None
            right_count = (
                int(right["output_tokens"])
                if isinstance(right.get("output_tokens"), int)
                and not isinstance(right["output_tokens"], bool)
                else len(right_tokens)
                if isinstance(right_tokens, list)
                else None
            )
            matched = left_count is not None and left_count == right_count
            return matched, "generated token count differs" if not matched else ""
        if contract == "exact-text":
            matched = left.get("text") == right.get("text")
            return matched, "generated text differs" if not matched else ""
        if contract == "normalized-text":
            matched = _normalized_text(left.get("text")) == _normalized_text(
                right.get("text")
            )
            return matched, "normalized generated text differs" if not matched else ""
        if contract == "token-agreement":
            left_tokens = left.get("token_ids")
            right_tokens = right.get("token_ids")
            if not isinstance(left_tokens, list) or not isinstance(right_tokens, list):
                return False, "token-agreement output is missing token ids"
            token_count = max(len(left_tokens), len(right_tokens))
            agreement = (
                sum(a == b for a, b in zip(left_tokens, right_tokens, strict=False)) / token_count
                if token_count
                else 1.0
            )
            minimum = float(case["baseline"]["min_positional_token_agreement"])
            if agreement < minimum:
                return False, "positional token agreement is below the configured contract"
            distance = _normalized_text_edit_distance(left.get("text"), right.get("text"))
            limit = float(case["baseline"]["max_normalized_edit_distance"])
            if distance > limit:
                return False, "normalized text distance exceeds the configured contract"
            return True, ""
        if contract == "ocr-text":
            required = list(case["baseline"]["required_substrings"])
            candidate_missing = _missing_ocr_substrings(left.get("text"), required)
            if candidate_missing:
                return False, "TRTMC OCR text misses required content"
            baseline_missing = _missing_ocr_substrings(right.get("text"), required)
            if baseline_missing:
                return False, "baseline OCR text misses required content"
            distance = _normalized_text_edit_distance(left.get("text"), right.get("text"))
            limit = float(case["baseline"]["max_normalized_edit_distance"])
            if distance > limit:
                return False, "normalized OCR text distance exceeds the configured contract"
            return True, ""
        matched = left.get("token_ids") == right.get("token_ids")
        return matched, "generated token ids differ" if not matched else ""
    if operation == "encode":
        left_elements = left.get("element_count")
        right_elements = right.get("embedding_elements")
        matched = left_elements == right_elements and left.get("dim") == right.get("dim")
        return matched, "encoder output shape differs" if not matched else ""
    return True, ""


def _effective_output_contract(
    case: Mapping[str, Any],
    request: Mapping[str, Any] | None = None,
) -> str:
    configured = case["baseline"].get("output_contract")
    if configured is not None:
        return str(configured)
    if (
        str(case["operation"]) == "generate"
        and request is not None
        and float(request.get("temperature", 0.0)) > 0.0
    ):
        return "generated-token-count"
    return "exact-token-ids"


def _classify(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    mismatch = _baseline_contract_mismatch(case, candidate, baseline)
    if mismatch:
        return "contract-mismatch", {"reason": mismatch}
    outputs_match, reason = _output_contract(
        case,
        candidate,
        baseline,
        request=request,
    )
    if not outputs_match:
        return "contract-mismatch", {"reason": reason}
    candidate_p50 = _median(candidate)
    baseline_p50 = _median(baseline)
    ratio = baseline_p50 / candidate_p50
    margin = float(case.get("equivalence_margin_percent", 5.0)) / 100.0
    status = _comparison_status(ratio, margin)
    return status, {
        "baseline_over_trtmc_p50": ratio,
        "equivalence_margin_percent": margin * 100.0,
    }


def _baseline_contract_mismatch(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    timing_mismatch = _timing_contract_mismatch(case, candidate, baseline)
    if timing_mismatch:
        return timing_mismatch
    expected_mode = str(case["baseline"].get("mode", "torch-compile"))
    if baseline.get("mode") != expected_mode:
        return "baseline mode differs from the suite"
    expected_precision = case["baseline"].get("precision", candidate.get("precision"))
    if baseline.get("precision") != expected_precision:
        return "baseline precision differs from the suite"
    expected_padding = case["baseline"].get("padding", "longest")
    if baseline.get("padding") != expected_padding:
        return "baseline padding differs from the suite"
    expected_experts = case["baseline"].get("experts_implementation")
    if baseline.get("experts_implementation") != expected_experts:
        return "baseline experts implementation differs from the suite"
    expected_adapter = case["baseline"].get("adapter")
    if expected_adapter and baseline.get("adapter") != expected_adapter:
        return "baseline adapter differs from the suite"
    if (
        "model_class" in case["baseline"]
        and baseline.get("model_class") != case["baseline"]["model_class"]
    ):
        return "baseline model class differs from the suite"
    if (
        "generation_method" in case["baseline"]
        and baseline.get("generation_method") != case["baseline"]["generation_method"]
    ):
        return "baseline generation method differs from the suite"
    digest = candidate.get("workload_digest")
    if not digest or baseline.get("workload_digest") != digest:
        return "candidate and baseline workload differ"
    if case["baseline"].get("runner") == "task-reference":
        if baseline.get("resolved_revision") in {None, "", "unresolved"}:
            return "task reference model revision is unresolved"
        if baseline.get("model_load_included") is not False:
            return "task reference timing includes model loading"
        return ""
    if expected_mode != "torch-compile":
        return ""
    evidence = baseline.get("compile_evidence")
    if not isinstance(evidence, Mapping) or not evidence.get("applied"):
        return "torch.compile evidence is missing"
    if not evidence.get("timed_callable_uses_compiled_target"):
        return "compiled target was not used while timing"
    return ""


def _timing_contract_mismatch(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str:
    if "timing_scope" not in case["baseline"]:
        return ""
    expected = timing_contract(
        runner=str(case["baseline"]["runner"]),
        family=str(case["family"]),
    )
    candidate_policy = candidate.get("measurement_policy", {})
    if not isinstance(candidate_policy, Mapping):
        return "TRTMC timing policy is missing"
    candidate_fields = {
        "timing_scope": "candidate_timing_scope",
        "input_preparation_included": "input_preparation_included",
        "asset_loading_included": "asset_loading_included",
    }
    for actual_name, expected_name in candidate_fields.items():
        if candidate_policy.get(actual_name) != expected[expected_name]:
            label = actual_name.replace("_", " ")
            return f"TRTMC {label} differs from the reference contract"
    baseline_policy = baseline.get("measurement_policy", {})
    if not isinstance(baseline_policy, Mapping):
        return "baseline timing policy is missing"
    for name in (
        "timing_scope",
        "input_preparation_included",
        "asset_loading_included",
    ):
        if baseline_policy.get(name) != expected[name]:
            return f"baseline {name.replace('_', ' ')} differs from the suite"
    return ""


def _comparison_status(ratio: float, margin: float) -> str:
    if ratio > 1.0 + margin:
        return "green"
    if ratio < 1.0 - margin:
        return "red"
    return "yellow"


def _case_row(
    case: Mapping[str, Any], resolved: Mapping[str, Any], workload_digest: str
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "model": case["model"],
        "status": "running",
        "resolved_settings": {
            "model": resolved["model"],
            "testcase": resolved["testcase"],
            "request": resolved["request"],
            "runtime": resolved["runtime"],
            "measurement": case["measurement"],
            "candidate_build_python_profile": resolved["_candidate_build_python_profile"],
            "output_contract": _effective_output_contract(
                case,
                resolved["request"],
            ),
            "workload": {
                "source": "testcase",
                "testcase": case["workload"]["testcase"],
                "request": resolved["request"],
            },
            "workload_digest": workload_digest,
        },
        "workload_contract": deepcopy(dict(case["workload"])),
        "measurement_contract": deepcopy(dict(case["measurement"])),
        "baseline_contract": dict(case["baseline"]),
        "commands": {},
        "started_at": _now(),
    }


def _run_supported_case(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    options: RunOptions,
    case_work: Path,
    row: MutableMapping[str, Any],
) -> None:
    environment = _command_environment()
    digest = _workload_digest(resolved)
    candidate_dir = case_work / "trtmc"
    baseline_path = case_work / "baseline.json"
    candidate_argv = [*_candidate_base_argv(case, options), "--output", str(candidate_dir)]
    baseline_argv, profile = _baseline_argv(case, resolved, baseline_path, options)
    row["resolved_settings"]["baseline_python_profile"] = profile
    commands = {
        "trtmc": {"argv": candidate_argv, "rendered": shlex.join(candidate_argv)},
        "baseline": {"argv": baseline_argv, "rendered": shlex.join(baseline_argv)},
    }
    row["commands"] = commands
    print(f"[{case['id']}] TRTMC: {commands['trtmc']['rendered']}", flush=True)
    print(f"[{case['id']}] baseline: {commands['baseline']['rendered']}", flush=True)
    order = ("trtmc", "baseline") if _stable_even(str(case["id"])) else ("baseline", "trtmc")
    for side in order:
        argv = candidate_argv if side == "trtmc" else baseline_argv
        _wait_for_gpu_memory_headroom()
        command = _run_command(argv, environment, options.timeout_seconds)
        command.pop("stdout", None)
        commands[side] = command
        if command["exit_code"] != 0:
            row["status"] = "failed"
            row["reason"] = f"{side} command failed with rc={command['exit_code']}"
            return
    candidate = _candidate_result(candidate_dir, digest)
    candidate["precision"] = str(resolved["model"]["precision"])
    baseline = _read_baseline(baseline_path)
    status, comparison = _classify(
        case,
        candidate,
        baseline,
        request=resolved["request"],
    )
    row["candidate"] = candidate
    row["baseline"] = baseline
    row["comparison"] = comparison
    row["status"] = status


def _stable_even(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:2], 16) % 2 == 0


def _run_one(
    case: Mapping[str, Any],
    options: RunOptions,
    work_root: Path,
    preflight: tuple[dict[str, Any], list[str], dict[str, str]],
) -> dict[str, Any]:
    resolved, _, dry_command = preflight
    digest = _workload_digest(resolved)
    row = _case_row(case, resolved, digest)
    row["commands"]["resolve"] = dry_command
    case_work = work_root / _slug(str(case["id"]))
    if case_work.exists():
        shutil.rmtree(case_work)
    case_work.mkdir(parents=True, exist_ok=True)
    try:
        _run_supported_case(case, resolved, options, case_work, row)
    except (PerfMatrixError, OSError, ValueError, RuntimeError) as exc:
        row["status"] = "failed"
        row["reason"] = str(exc)
    row["finished_at"] = _now()
    return row


def _resolution_failure_row(
    case: Mapping[str, Any], failure: Mapping[str, Any]
) -> dict[str, Any]:
    argv = failure.get("argv", [])
    commands = {}
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        commands["resolve"] = {
            "argv": list(argv),
            "rendered": shlex.join(str(value) for value in argv),
        }
    return {
        "id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "model": case["model"],
        "status": "failed",
        "failure_stage": failure.get("stage", "preflight"),
        "reason": str(failure.get("reason", "preflight failed")),
        "baseline_contract": dict(case["baseline"]),
        "commands": commands,
        "started_at": _now(),
        "finished_at": _now(),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "case"


def _should_skip(row: Mapping[str, Any]) -> bool:
    return row.get("status") in {"green", "yellow", "red", "contract-mismatch"}


def _final_status(rows: Iterable[Mapping[str, Any]]) -> str:
    statuses = {str(row.get("status")) for row in rows}
    invalid = {"failed", "contract-mismatch", "partial", "pending", "running"}
    return "completed-with-errors" if statuses & invalid else "completed"


def _light(status: str) -> str:
    return {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴",
    }.get(status, "⚪")


def _baseline_label(row: Mapping[str, Any]) -> str:
    contract = row.get("baseline_contract", {})
    precision = contract.get("precision") or row.get("resolved_settings", {}).get("model", {}).get(
        "precision", ""
    )
    precision_suffix = f" · {precision}" if precision else ""
    if contract.get("runner") == "task-reference":
        mode = {
            "hf-eager": "HF eager",
            "pytorch-eager": "PyTorch eager",
        }.get(str(contract.get("mode", "")), str(contract.get("mode", "")))
        return f"{mode} · {contract.get('adapter', 'task reference')}{precision_suffix}"
    mode = contract.get("mode", "torch-compile")
    scope = contract.get("compile_scope", "model.forward") if mode == "torch-compile" else ""
    details = []
    if contract.get("precision"):
        details.append(str(contract["precision"]))
    if contract.get("experts_implementation"):
        details.append(f"experts={contract['experts_implementation']}")
    if contract.get("model_class"):
        details.append(f"loader={contract['model_class']}")
    if contract.get("generation_method"):
        details.append(f"generate={contract['generation_method']}")
    suffix = f", {', '.join(details)}" if details else ""
    label = f"torch.compile ({scope}{suffix})" if mode == "torch-compile" else f"HF eager{suffix}"
    return label if contract.get("precision") else label + precision_suffix


def _report_note(row: Mapping[str, Any]) -> str:
    status = row.get("status")
    if status == "failed":
        return "Execution failed; inspect internal results.json."
    comparison = row.get("comparison", {})
    if isinstance(comparison, Mapping) and comparison.get("reason"):
        return str(comparison["reason"])
    return str(row.get("reason", ""))


def _timing_scope(result: Mapping[str, Any]) -> str | None:
    policy = result.get("measurement_policy")
    if isinstance(policy, Mapping) and policy.get("timing_scope"):
        return str(policy["timing_scope"])
    scope = result.get("timing_scope")
    return str(scope) if scope else None


def _timing_policy_value(result: Mapping[str, Any], name: str) -> Any:
    policy = result.get("measurement_policy")
    if isinstance(policy, Mapping) and name in policy:
        return policy[name]
    return result.get(name)


def _timing_scope_details(result: Mapping[str, Any], side: str) -> dict[str, str]:
    scope = _timing_scope(result)
    if not scope:
        return {
            "measured": "No timing result",
            "included": "—",
            "excluded": "—",
        }
    if side == "candidate" and scope == "public_pipeline_call_wall":
        asset_included = _timing_policy_value(result, "asset_loading_included") is True
        return {
            "measured": (
                "asset load and public pipeline call"
                if asset_included
                else "public pipeline call"
            ),
            "included": (
                "asset loading, pipeline-internal preprocessing, model execution, returned output"
                if asset_included
                else "pipeline-internal preprocessing, model execution, returned output"
            ),
            "excluded": (
                "bundle/model load, warmup, telemetry"
                if asset_included
                else "bundle/model load, warmup, asset loading, telemetry"
            ),
        }
    if side == "candidate" and scope == "model_call_wall":
        return {
            "measured": "first TensorRT module call through returned output",
            "included": (
                "module input transfer, model execution, inter-module work, "
                "output materialization"
            ),
            "excluded": (
                "bundle/model load, warmup, pipeline preprocessing, asset loading, telemetry"
            ),
        }
    if scope == "public_operation_call_wall":
        return {
            "measured": "public operation call",
            "included": "tokenization, device transfers, model operation, output materialization",
            "excluded": "model load, compile setup, warmup",
        }
    if scope == "task-model-call-wall":
        return {
            "measured": "task model call",
            "included": "prepared model invocation through returned summary",
            "excluded": "model load, warmup, input preparation, asset loading",
        }
    if scope == "task-pipeline-call-wall":
        asset_included = _timing_policy_value(result, "asset_loading_included") is True
        return {
            "measured": (
                "asset load and task pipeline call"
                if asset_included
                else "task pipeline call"
            ),
            "included": (
                "asset loading, reference input preparation, model operation, returned summary"
                if asset_included
                else "reference input preparation, model operation, returned summary"
            ),
            "excluded": (
                "model load, warmup"
                if asset_included
                else "model load, warmup, asset loading"
            ),
        }
    return {
        "measured": scope,
        "included": "not declared",
        "excluded": "not declared",
    }


def _timing_scope_html(row: Mapping[str, Any]) -> str:
    parts = []
    for key, label in (("candidate", "TRTMC"), ("baseline", "Baseline")):
        result = row.get(key, {})
        if not isinstance(result, Mapping):
            result = {}
        details = _timing_scope_details(result, key)
        parts.append(
            "<div class='scope-side'>"
            f"<div class='scope-title'>{label}</div>"
            f"<div>Measured: {html.escape(details['measured'])}</div>"
            f"<div>Includes: {html.escape(details['included'])}</div>"
            f"<div>Excludes: {html.escape(details['excluded'])}</div>"
            "</div>"
        )
    return "".join(parts)


def _timing_value_html(result: Any) -> str:
    if not isinstance(result, Mapping):
        return "—"
    values = result.get("samples_ms")
    if not isinstance(values, list) or not values:
        return "—"
    try:
        samples = [float(value) for value in values]
    except (TypeError, ValueError):
        return "—"
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        return "—"
    value = statistics.median(samples)
    return f"{value:,.3f}<div class='timing-meta'>n={len(samples)}</div>"


def _raw_commands_html(row: Mapping[str, Any], default_cwd: str) -> str:
    commands = row.get("commands", {})
    if not isinstance(commands, Mapping):
        return "—"
    recorded = []
    for side, label in (("trtmc", "TRTMC"), ("baseline", "Baseline")):
        command = commands.get(side)
        if not isinstance(command, Mapping) or not command.get("rendered"):
            continue
        recorded.append((label, command))
    if not recorded:
        return "—"
    working_directories = {
        str(command.get("cwd") or default_cwd) for _, command in recorded
    }
    parts = ["<details><summary>Show raw commands</summary>"]
    if len(working_directories) == 1:
        cwd = next(iter(working_directories))
        parts.append(
            "<div class='command-label'>Working directory</div>"
            f"<pre><code>{html.escape(cwd)}</code></pre>"
        )
    for label, command in recorded:
        cwd = str(command.get("cwd") or default_cwd)
        command_label = label if len(working_directories) == 1 else f"{label} (cwd: {cwd})"
        parts.append(
            f"<div class='command-label'>{html.escape(command_label)}</div>"
            f"<pre><code>{html.escape(str(command['rendered']))}</code></pre>"
        )
        environment = command.get("environment")
        if isinstance(environment, Mapping) and environment:
            rendered_environment = "\n".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in sorted(environment.items())
            )
            parts.append(
                "<div class='command-label'>Environment</div>"
                f"<pre><code>{html.escape(rendered_environment)}</code></pre>"
            )
    parts.append("</details>")
    return "".join(parts)


def _report_html(results: Mapping[str, Any]) -> str:
    rows = list(results["cases"])
    family_counts = Counter(str(row["family"]) for row in rows)
    family_count = len(family_counts)
    repeated_families = [
        f"{family} ({count} profiles)" for family, count in family_counts.items() if count > 1
    ]
    counts = {
        status: sum(row.get("status") == status for row in rows)
        for status in ("green", "yellow", "red")
    }
    body = []
    default_cwd = str(results.get("repository_root", ""))
    for row in rows:
        status = str(row.get("status", "pending"))
        reason = html.escape(_report_note(row))
        commands = _raw_commands_html(row, default_cwd)
        candidate_timing = _timing_value_html(row.get("candidate"))
        baseline_timing = _timing_value_html(row.get("baseline"))
        timing_scope = _timing_scope_html(row)
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['family']))}</td>"
            f"<td>{html.escape(str(row['operation']))}</td>"
            f"<td><code>{html.escape(str(row['model']))}</code></td>"
            f"<td>{html.escape(_baseline_label(row))}</td>"
            f"<td class='timing-value'>{candidate_timing}</td>"
            f"<td class='timing-value'>{baseline_timing}</td>"
            f"<td>{timing_scope}</td>"
            f"<td class='light'>{_light(status)}</td>"
            f"<td>{reason}</td>"
            f"<td>{commands}</td>"
            "</tr>"
        )
    generated = html.escape(str(results.get("finished_at", results.get("started_at", ""))))
    summary = f"🟢 {counts['green']} &nbsp; 🟡 {counts['yellow']} &nbsp; 🔴 {counts['red']} &nbsp; ⚪ {len(rows) - sum(counts.values())}"
    repeated_note = (
        " " + html.escape("; ".join(repeated_families)) + " contribute multiple rows."
        if repeated_families
        else ""
    )
    preflight = results.get("timing_preflight", {})
    preflight_count = (
        int(preflight.get("case_count", 0)) if isinstance(preflight, Mapping) else 0
    )
    catalog = results.get("catalog_coverage", {})
    ready_profiles = int(catalog.get("ready_profiles", len(rows)))
    distributed_profiles = int(catalog.get("distributed_profiles", 0))
    other_profiles = int(catalog.get("other_profiles", 0))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRTMC performance matrix</title>
<style>
body{{font-family:Arial,sans-serif;margin:28px;color:#1f2937}} h1{{margin-bottom:4px}}
.meta{{color:#4b5563;margin:4px 0 18px}} .table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;min-width:1500px}}
th,td{{border:1px solid #9ca3af;padding:7px;text-align:left;vertical-align:top}}
th{{background:#e5e7eb}} tr:nth-child(even){{background:#f9fafb}} code{{font-size:12px}}
.command-label{{font-weight:600;margin-top:8px}} pre{{margin:4px 0 10px;max-width:720px;white-space:pre-wrap;overflow-wrap:anywhere}}
.light{{font-size:20px;text-align:center}} .timing-value{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.timing-meta{{color:#6b7280;font-size:11px;margin-top:2px}} .scope-side{{font-size:11px;margin-bottom:7px;min-width:270px}}
.scope-title{{font-size:12px;font-weight:700;margin-bottom:2px}}
</style></head><body>
<h1>TRTMC performance matrix</h1>
<p class="meta">Generated {generated}. {family_count} families across {len(rows)} model-profile comparisons.{repeated_note}</p>
<p class="meta">Catalog coverage: {ready_profiles} ready single-process profiles. {distributed_profiles} distributed profiles require a separate multi-process run and are outside this report. {other_profiles} other or unsupported profiles.</p>
<p class="meta">Timing contracts were validated before execution for {preflight_count} comparisons. A row is classified only when its recorded TRTMC and reference policies match that contract.</p>
<p class="meta">Times are the p50 wall time from the recorded timed samples. Measured scope states the exact recorded boundary and the work included and excluded on each side.</p>
<p><strong>{summary}</strong></p>
<div class="table-wrap"><table><thead><tr><th>Family</th><th>Operation</th><th>Model profile</th><th>Baseline</th><th>TRTMC p50 (ms)</th><th>Baseline p50 (ms)</th><th>Measured scope</th><th>Light</th><th>Note</th><th>Commands</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<p class="meta">Green: TRTMC is more than the configured margin faster. Yellow: within the margin. Red: TRTMC is more than the margin slower. White: not run, partial, or invalid comparison. Commands are the original recorded argv and must be run from the displayed working directory with the same model cache and dependencies.</p>
</body></html>"""


def _write_artifacts(output: Path, results: Mapping[str, Any]) -> None:
    _write_json(output / "results.json", results)
    (output / "report.html").write_text(_report_html(results), encoding="utf-8")
    legacy_replay = output / "reproduce.py"
    if legacy_replay.is_file():
        legacy_replay.unlink()


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _environment_preflight(
    environment: Mapping[str, Any], options: RunOptions
) -> dict[str, Any]:
    executable = options.trtmc_bench
    if os.sep in executable:
        executable_path = Path(executable)
        if not executable_path.is_file():
            raise PerfMatrixError(f"trtmc-bench does not exist: {executable_path}")
    elif shutil.which(executable) is None:
        raise PerfMatrixError(f"trtmc-bench is not on PATH: {executable}")
    for label, path in (
        ("TRTMC worker", options.trtmc_worker),
        ("HF Transformers runner", options.hf_transformers_runner),
        ("task reference runner", options.task_reference_runner),
    ):
        if path is None or not path.is_file():
            raise PerfMatrixError(f"{label} does not exist: {path}")
    paths = {
        "results_root": Path(str(environment["storage"]["results_root"])),
        "scratch_root": options.scratch_root,
    }
    if options.bundle_cache is not None:
        paths["bundle_cache"] = options.bundle_cache
    filesystems: dict[tuple[int, int, int], dict[str, Any]] = {}
    for label, path in paths.items():
        probe = _existing_ancestor(path)
        usage = shutil.disk_usage(probe)
        key = (usage.total, usage.used, usage.free)
        record = filesystems.setdefault(
            key,
            {
                "paths": [],
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
        )
        record["paths"].append({"name": label, "path": str(path), "probe": str(probe)})
    minimum_bytes = options.minimum_free_space_gib * 1024**3
    insufficient = [
        value for value in filesystems.values() if value["free_bytes"] < minimum_bytes
    ]
    if insufficient:
        available = min(value["free_bytes"] for value in insufficient) / 1024**3
        raise PerfMatrixError(
            f"environment has only {available:.1f} GiB free; "
            f"requires {options.minimum_free_space_gib} GiB"
        )
    for value in filesystems.values():
        labels = ", ".join(item["name"] for item in value["paths"])
        print(
            f"Storage: {labels}: {value['free_bytes'] / 1024**3:.1f} GiB free",
            flush=True,
        )
    return {"checked_at": _now(), "filesystems": list(filesystems.values())}


def _new_run_directory(
    results_root: Path, suite: Mapping[str, Any]
) -> tuple[str, Path]:
    results_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = (_git_commit() or "unknown")[:8]
    base = f"{_slug(str(suite.get('name', 'performance-matrix')))}-{timestamp}-{revision}"
    for index in range(1, 1000):
        run_id = base if index == 1 else f"{base}-{index}"
        output = results_root / run_id
        try:
            output.mkdir()
        except FileExistsError:
            continue
        return run_id, output
    raise PerfMatrixError(f"cannot allocate a unique run directory under {results_root}")


def _selected_rows(
    results: Mapping[str, Any], selected_ids: set[str]
) -> list[Mapping[str, Any]]:
    return [row for row in results["cases"] if str(row["id"]) in selected_ids]


def _campaign_failed(results: Mapping[str, Any], selected_ids: set[str]) -> bool:
    valid = {"green", "yellow", "red"}
    return any(row.get("status") not in valid for row in _selected_rows(results, selected_ids))


def _execute_campaign(
    *,
    selected: Sequence[Mapping[str, Any]],
    options: RunOptions,
    results: MutableMapping[str, Any],
    preflight: Mapping[str, tuple[dict[str, Any], list[str], dict[str, str]]],
    preflight_failures: Mapping[str, Mapping[str, Any]],
    reference_preflight: Mapping[str, Any],
    worker: Mapping[str, Any],
    storage_preflight: Mapping[str, Any],
) -> int:
    results["candidate_worker_preflight"] = dict(worker)
    results["storage_preflight"] = deepcopy(dict(storage_preflight))
    results["reference_preflight"] = deepcopy(dict(reference_preflight))
    ready = [case for case in selected if str(case["id"]) in preflight]
    timing_preflight = _preflight_evidence(ready, preflight)
    timing_preflight["selected_case_count"] = len(selected)
    timing_preflight["failed_case_count"] = len(preflight_failures)
    if preflight_failures:
        timing_preflight["status"] = "partial"
    results["timing_preflight"] = timing_preflight
    rows = _result_rows(results)
    work_root = options.scratch_root / options.output.name
    work_root.mkdir(parents=True, exist_ok=True)
    for case in selected:
        case_id = str(case["id"])
        failure = preflight_failures.get(case_id)
        if failure is None or _should_skip(rows[case_id]):
            continue
        rows[case_id].clear()
        rows[case_id].update(_resolution_failure_row(case, failure))
    _write_artifacts(options.output, results)
    for case in selected:
        case_id = str(case["id"])
        existing = rows[case_id]
        if _should_skip(existing):
            print(f"[{case['id']}] resume: keeping {existing['status']}", flush=True)
            continue
        if case_id not in preflight:
            print(
                f"[{case_id}] skipped: {existing.get('reason', 'preflight failed')}",
                flush=True,
            )
            continue
        row = _run_one(case, options, work_root, preflight[case_id])
        rows[case_id].clear()
        rows[case_id].update(row)
        _write_artifacts(options.output, results)
    selected_ids = {str(case["id"]) for case in selected}
    results["finished_at"] = _now()
    results["status"] = _final_status(_selected_rows(results, selected_ids))
    _write_artifacts(options.output, results)
    shutil.rmtree(work_root, ignore_errors=True)
    try:
        options.scratch_root.rmdir()
    except OSError:
        pass
    print(f"Results: {options.output / 'results.json'}")
    print(f"Report: {options.output / 'report.html'}")
    return 1 if _campaign_failed(results, selected_ids) else 0


def _load_suite_request(
    arguments: argparse.Namespace,
) -> tuple[
    Path,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    suite_path = arguments.suite.resolve()
    suite = _read_yaml(suite_path)
    cases = _cases(suite)
    _validate_coverage(cases)
    selected = _selected_cases(cases, arguments.entry)
    if not selected:
        raise PerfMatrixError("selection contains no entries")
    environment = _read_environment(arguments.environment)
    return suite_path, suite, cases, selected, environment


def _check(arguments: argparse.Namespace) -> int:
    _, _, _, selected, environment = _load_suite_request(arguments)
    options = _run_options(
        environment, Path(str(environment["storage"]["results_root"]))
    )
    storage = _environment_preflight(environment, options)
    worker = _preflight_worker(options)
    preflight, references, failures = _preflight_selected(selected, options)
    ready = [case for case in selected if str(case["id"]) in preflight]
    evidence = _preflight_evidence(ready, preflight)
    print(f"Environment: {environment['name']} ({environment['source']})")
    print(f"Entries: {len(selected)} selected, {evidence['case_count']} ready")
    print(f"Timing: {evidence['status']}")
    print(f"Worker: {worker['path']}")
    print(f"Reference profiles: {references['entry_count']}")
    print(f"Storage filesystems: {len(storage['filesystems'])}")
    for case_id, failure in sorted(failures.items()):
        print(f"[{case_id}] {failure['stage']}: {failure['reason']}", file=sys.stderr)
    return 1 if failures else 0


def _run_new(arguments: argparse.Namespace) -> int:
    suite_path, suite, cases, selected, environment = _load_suite_request(arguments)
    results_root = Path(str(environment["storage"]["results_root"]))
    preliminary_options = _run_options(environment, results_root)
    storage = _environment_preflight(environment, preliminary_options)
    worker = _preflight_worker(preliminary_options)
    preflight, references, failures = _preflight_selected(selected, preliminary_options)
    run_id, output = _new_run_directory(results_root, suite)
    options = _run_options(environment, output)
    results = _initial_results(suite_path, suite, cases, selected, environment)
    results["run_id"] = run_id
    return _execute_campaign(
        selected=selected,
        options=options,
        results=results,
        preflight=preflight,
        preflight_failures=failures,
        reference_preflight=references,
        worker=worker,
        storage_preflight=storage,
    )


def _resume(arguments: argparse.Namespace) -> int:
    output = arguments.run_directory.resolve()
    result_path = output / "results.json"
    results = _load_resume(result_path)
    suite_path = Path(str(results.get("suite", "")))
    environment_record = results.get("environment_config")
    if not suite_path.is_file():
        raise PerfMatrixError(f"cannot resume because suite is unavailable: {suite_path}")
    if not isinstance(environment_record, Mapping):
        raise PerfMatrixError("cannot resume without an environment configuration")
    environment_path = Path(str(environment_record.get("source", "")))
    if not environment_path.is_file():
        raise PerfMatrixError(
            f"cannot resume because environment is unavailable: {environment_path}"
        )
    if results.get("suite_sha256") != _sha256_file(suite_path):
        raise PerfMatrixError("cannot resume because the suite content changed")
    if environment_record.get("sha256") != _sha256_file(environment_path):
        raise PerfMatrixError("cannot resume because the environment content changed")
    if results.get("git_commit") != _git_commit():
        raise PerfMatrixError("cannot resume because the repository revision changed")
    suite = _read_yaml(suite_path)
    cases = _cases(suite)
    _validate_coverage(cases)
    selected_ids = results.get("selected_entry_ids")
    if not isinstance(selected_ids, list) or not all(
        isinstance(value, str) for value in selected_ids
    ):
        raise PerfMatrixError("cannot resume without selected entry ids")
    selected = _selected_cases(cases, selected_ids)
    environment = _read_environment(environment_path)
    if environment != environment_record:
        raise PerfMatrixError(
            "cannot resume because the resolved environment values changed"
        )
    options = _run_options(environment, output)
    storage = _environment_preflight(environment, options)
    worker = _preflight_worker(options)
    preflight, references, failures = _preflight_selected(selected, options)
    return _execute_campaign(
        selected=selected,
        options=options,
        results=results,
        preflight=preflight,
        preflight_failures=failures,
        reference_preflight=references,
        worker=worker,
        storage_preflight=storage,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        if arguments.command == "check":
            return _check(arguments)
        if arguments.command == "run":
            return _run_new(arguments)
        if arguments.command == "resume":
            return _resume(arguments)
        raise PerfMatrixError(f"unsupported command: {arguments.command}")
    except PerfMatrixError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

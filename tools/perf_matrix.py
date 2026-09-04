#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and report a TRTMC-versus-reference performance matrix."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
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
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = REPOSITORY / "python"
for source_root in (REPOSITORY, PYTHON_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from benchmarks.performance.baselines.timing_contracts import timing_contract  # noqa: E402
from tensorrt_model_connect.benchmark.catalog import ManifestCatalog  # noqa: E402
from tensorrt_model_connect.benchmark.types import (  # noqa: E402
    COMMAND_DIAGNOSTIC_PREFIX,
    COMMAND_DIAGNOSTIC_SCHEMA,
    BenchmarkError,
)
from tensorrt_model_connect.benchmark.worker import (  # noqa: E402
    find_worker,
    worker_metadata,
)
from tensorrt_model_connect.families.locateanything.task_contract import (  # noqa: E402
    matched_box_iou,
    matched_point_distance,
    parse_localizations,
)
from tensorrt_model_connect.python_profiles import (  # noqa: E402
    default_execution_profiles,
    resolve_profile_python,
)
from tools.reporting_html import (  # noqa: E402
    COMMON_REPORT_STYLES,
    ReportFilter,
    render_report_filter_script,
    render_report_filters,
    sorted_filter_values,
    task_type_label,
)
from tools import case_evidence, model_selection  # noqa: E402
from tools.performance import catalog as performance_catalog  # noqa: E402
from tools.execution_ledger import ExecutionLedger, ExecutionLedgerError  # noqa: E402
from tools import qualification_report  # noqa: E402


RESULT_SCHEMA = "trtmc.perf-matrix/v1"
PREPARATION_SCHEMA = "trtmc.perf-bundle-preparation/v1"
ENVIRONMENT_SCHEMA = "trtmc.perf-environment/v1"
EXACT_SOURCE_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
SEQUENCE_RUNTIME_MARKERS = ("bart_", "marian_", "m2m_100_", "t5_")
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
HF_CACHE_ENVIRONMENT_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_ASSETS_CACHE",
    "HF_XET_CACHE",
)
RETENTION_POLICIES = {"retain", "delete_on_pass", "delete_always"}


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
    minimum_gpu_free_fraction: float
    timeout_seconds: int
    storage_root: Path | None = None
    bundle_retention: str = "retain"
    hf_cache_mode: str = "shared"
    hf_cache_retention: str = "retain"
    verbose: bool = False
    require_prebuilt_bundles: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("check", "resolve and validate the selected matrix entries"),
        ("prepare", "materialize selected bundles without measuring them"),
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
        command.add_argument(
            "--model",
            action="append",
            default=[],
            help="canonical model name; selects every matching entry; repeatable",
        )
        command.add_argument(
            "--model-selection",
            type=Path,
            help=(
                "JSON owner/family selection emitted by tools/model_ci.py; "
                "selects matching model profiles"
            ),
        )
        command.add_argument(
            "--verbose",
            action="store_true",
            help="print full TRTMC and reference commands",
        )
        if name == "prepare":
            command.add_argument(
                "--output",
                required=True,
                type=Path,
                help="write bundle preparation evidence to this JSON file",
            )
        if name == "run":
            command.add_argument(
                "--no-build",
                action="store_true",
                help="require bundles prepared before the performance campaign",
            )
    resume = commands.add_parser("resume", help="continue an incomplete run")
    resume.add_argument("run_directory", type=Path)
    resume.add_argument(
        "--verbose",
        action="store_true",
        help="print full TRTMC and reference commands",
    )
    resume.add_argument(
        "--no-build",
        action="store_true",
        help="require bundles prepared before the resumed campaign",
    )
    resume.add_argument(
        "--invalidate-model",
        action="append",
        default=[],
        help=(
            "re-run every selected Performance case for this model; repeatable and "
            "only valid while resuming"
        ),
    )
    report = commands.add_parser(
        "report",
        help="render an existing run with optional test-task preparation evidence",
    )
    report.add_argument("run_directory", type=Path)
    report.add_argument(
        "--preparation-receipt",
        type=Path,
        help="JSON receipt for bundles prepared before the matrix campaign",
    )
    return parser


def _read_yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PerfMatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerfMatrixError(f"{label} must contain a YAML object")
    return value


def _load_performance_suite(path: Path) -> performance_catalog.PerformanceSuite:
    try:
        return performance_catalog.load_suite(path)
    except performance_catalog.PerformanceSuiteError as exc:
        raise PerfMatrixError(str(exc)) from exc


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
        raise PerfMatrixError(f"environment {field} requires: {', '.join(sorted(set(missing)))}")
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
            _expand_environment_value(item, f"{field}[{index}]") for index, item in enumerate(value)
        ]
    else:
        raise PerfMatrixError(f"environment {field} must be a path list or path-list string")
    return tuple(_resolved_path(item, field) for item in configured)


def _read_environment(path: Path) -> dict[str, Any]:
    source = path.resolve()
    raw = _read_yaml_object(source, "performance environment")
    if raw.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise PerfMatrixError(f"environment schema_version must be {ENVIRONMENT_SCHEMA!r}")
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
        raise PerfMatrixError(f"environment tools is missing: {', '.join(missing_tools)}")
    required_storage = ("results_root", "scratch_root", "bundle_roots", "runtime_dirs")
    missing_storage = [field for field in required_storage if field not in storage]
    if missing_storage:
        raise PerfMatrixError(f"environment storage is missing: {', '.join(missing_storage)}")
    bundle_cache = storage.get("bundle_cache")
    if bundle_cache is not None and not isinstance(bundle_cache, str):
        raise PerfMatrixError("environment storage.bundle_cache must be a path or null")
    storage_root = storage.get("storage_root")
    if storage_root is not None and (not isinstance(storage_root, str) or not storage_root.strip()):
        raise PerfMatrixError("environment storage.storage_root must be a path or null")
    bundle_retention = storage.get("bundle_retention", "retain")
    if bundle_retention not in RETENTION_POLICIES:
        raise PerfMatrixError(
            "environment storage.bundle_retention must be retain, delete_on_pass, or delete_always"
        )
    if bundle_retention != "retain" and bundle_cache is None:
        raise PerfMatrixError("non-retained bundles require environment storage.bundle_cache")
    timeout_seconds = execution.get("timeout_seconds")
    minimum_gpu_free_fraction = execution.get("minimum_gpu_free_fraction", 0.45)
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
        raise PerfMatrixError("environment storage.minimum_free_space_gib must be non-negative")
    if (
        isinstance(minimum_gpu_free_fraction, bool)
        or not isinstance(minimum_gpu_free_fraction, (int, float))
        or not 0 <= float(minimum_gpu_free_fraction) <= 1
    ):
        raise PerfMatrixError("environment execution.minimum_gpu_free_fraction must be in [0, 1]")
    local_files_only = execution.get("local_files_only", False)
    if not isinstance(local_files_only, bool):
        raise PerfMatrixError("environment execution.local_files_only must be boolean")
    hf_cache_mode = execution.get("hf_cache_mode", "shared")
    if hf_cache_mode not in {"shared", "per_entry"}:
        raise PerfMatrixError("environment execution.hf_cache_mode must be shared or per_entry")
    hf_cache_retention = execution.get("hf_cache_retention", "retain")
    if hf_cache_retention not in RETENTION_POLICIES:
        raise PerfMatrixError(
            "environment execution.hf_cache_retention must be retain, "
            "delete_on_pass, or delete_always"
        )
    if hf_cache_mode == "shared" and hf_cache_retention != "retain":
        raise PerfMatrixError("a shared Hugging Face cache only supports hf_cache_retention=retain")
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "name": name.strip(),
        "source": str(source),
        "sha256": _sha256_file(source),
        "tools": {
            "trtmc_bench": _resolved_executable(str(tools["trtmc_bench"]), "tools.trtmc_bench"),
            "trtmc_worker": str(_resolved_path(str(tools["trtmc_worker"]), "tools.trtmc_worker")),
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
            "storage_root": (
                str(_resolved_path(storage_root, "storage.storage_root"))
                if storage_root is not None
                else None
            ),
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
                for path in _resolved_path_list(storage["bundle_roots"], "storage.bundle_roots")
            ],
            "runtime_dirs": [
                str(path)
                for path in _resolved_path_list(storage["runtime_dirs"], "storage.runtime_dirs")
            ],
            "minimum_free_space_gib": minimum_free_space_gib,
            "bundle_retention": bundle_retention,
        },
        "execution": {
            "local_files_only": local_files_only,
            "hf_cache_mode": hf_cache_mode,
            "hf_cache_retention": hf_cache_retention,
            "minimum_gpu_free_fraction": float(minimum_gpu_free_fraction),
            "timeout_seconds": timeout_seconds,
        },
    }


def _run_options(
    environment: Mapping[str, Any],
    output: Path,
    *,
    verbose: bool = False,
    require_prebuilt_bundles: bool = False,
) -> RunOptions:
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
        bundle_cache=(Path(str(storage["bundle_cache"])) if storage.get("bundle_cache") else None),
        bundle_roots=tuple(Path(str(value)) for value in storage["bundle_roots"]),
        runtime_dirs=tuple(Path(str(value)) for value in storage["runtime_dirs"]),
        local_files_only=bool(execution["local_files_only"]),
        minimum_free_space_gib=int(storage["minimum_free_space_gib"]),
        minimum_gpu_free_fraction=float(execution["minimum_gpu_free_fraction"]),
        timeout_seconds=int(execution["timeout_seconds"]),
        storage_root=(Path(str(storage["storage_root"])) if storage.get("storage_root") else None),
        bundle_retention=str(storage["bundle_retention"]),
        hf_cache_mode=str(execution["hf_cache_mode"]),
        hf_cache_retention=str(execution["hf_cache_retention"]),
        verbose=verbose,
        require_prebuilt_bundles=require_prebuilt_bundles,
    )


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


def _execution_revision() -> str:
    try:
        return case_evidence.exact_source_revision(
            _git_commit(), label="repository revision"
        )
    except case_evidence.CaseEvidenceError as error:
        raise PerfMatrixError(str(error)) from error


def _gpu_environment() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"gpu": None, "driver": None}
    try:
        output = subprocess.run(
            [
                executable,
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
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
    performance_suite: performance_catalog.PerformanceSuite,
    selected: Sequence[Mapping[str, Any]],
    environment_config: Mapping[str, Any],
) -> dict[str, Any]:
    suite_path = performance_suite.source
    suite = performance_suite.definition
    cases = performance_suite.cases
    catalog_entries = ManifestCatalog().entries()
    catalog_counts = Counter(entry.status for entry in catalog_entries)
    explicit_exclusions = performance_suite.excluded_profiles
    excluded_l0_profiles = sum(
        entry.status == "ready" and performance_catalog.is_l0_profile(entry.name)
        for entry in catalog_entries
    )
    environment = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        **_gpu_environment(),
    }
    started_at = _now()
    revision = _execution_revision()
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "running",
        "suite": str(suite_path.resolve()),
        "suite_name": str(suite.get("name", suite_path.stem)),
        "suite_sha256": _sha256_file(suite_path),
        "repository_root": str(REPOSITORY),
        "git_commit": revision,
        "source_revisions": [revision],
        "execution_attempts": [
            {
                "source_revision": revision,
                "started_at": started_at,
                "finished_at": None,
            }
        ],
        "started_at": started_at,
        "environment": environment,
        "environment_config": deepcopy(dict(environment_config)),
        "catalog_coverage": {
            "total_profiles": sum(catalog_counts.values()),
            "ready_profiles": catalog_counts["ready"],
            "release_profiles": (
                catalog_counts["ready"] - excluded_l0_profiles - len(explicit_exclusions)
            ),
            "explicitly_excluded_profiles": len(explicit_exclusions),
            "explicit_exclusions": [
                {"model": model, "reason": reason} for model, reason in explicit_exclusions.items()
            ],
            "excluded_l0_profiles": excluded_l0_profiles,
            "distributed_profiles": catalog_counts["distributed"],
            "other_profiles": sum(
                count
                for status, count in catalog_counts.items()
                if status not in {"ready", "distributed"}
            ),
        },
        "selected_entry_ids": [str(case["id"]) for case in selected],
        "cases": [_pending_perf_row(case) for case in cases],
    }


def _pending_perf_row(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "family": case["family"],
        "operation": case["operation"],
        "model": case["model"],
        "workload_contract": deepcopy(dict(case["workload"])),
        "measurement_contract": deepcopy(dict(case["measurement"])),
        "baseline_contract": dict(case["baseline"]),
        "status": "pending",
    }


def _open_perf_ledger(
    output: Path,
    selected: Sequence[Mapping[str, Any]],
    results: Mapping[str, Any],
) -> ExecutionLedger:
    fingerprint_input = {
        "suite_sha256": results.get("suite_sha256"),
        "environment_sha256": results.get("environment_config", {}).get("sha256"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        return ExecutionLedger.open(
            output,
            campaign_id=str(results.get("run_id") or output.name),
            task_kind="performance",
            fingerprint=fingerprint,
            cases=[{"id": str(case["id"]), "report": _pending_perf_row(case)} for case in selected],
        )
    except ExecutionLedgerError as error:
        raise PerfMatrixError(str(error)) from error


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


def _record_execution_attempt(
    results: MutableMapping[str, Any], source_revision: str
) -> None:
    previous_revision = str(results.get("git_commit", "") or "").strip().lower()
    revisions = results.setdefault("source_revisions", [])
    if not isinstance(revisions, list):
        raise PerfMatrixError("cannot resume with invalid source revision history")
    if (
        case_evidence.EXACT_SOURCE_REVISION.fullmatch(previous_revision)
        and previous_revision not in revisions
    ):
        revisions.append(previous_revision)
    if source_revision not in revisions:
        revisions.append(source_revision)
    attempts = results.setdefault("execution_attempts", [])
    if not isinstance(attempts, list):
        raise PerfMatrixError("cannot resume with invalid execution attempt history")
    if not attempts and case_evidence.EXACT_SOURCE_REVISION.fullmatch(previous_revision):
        attempts.append(
            {
                "source_revision": previous_revision,
                "started_at": results.get("started_at"),
                "finished_at": results.get("finished_at"),
                "legacy": True,
            }
        )
    attempts.append(
        {
            "source_revision": source_revision,
            "started_at": _now(),
            "finished_at": None,
        }
    )
    results["git_commit"] = source_revision


def _finish_execution_attempt(results: MutableMapping[str, Any]) -> None:
    attempts = results.get("execution_attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], MutableMapping):
        attempts[-1]["finished_at"] = _now()


def _result_rows(
    results: MutableMapping[str, Any],
) -> dict[str, MutableMapping[str, Any]]:
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
        "measurement.timing_scope=" + _yaml_cli_value(_candidate_timing_scope(case)),
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
    if getattr(options, "require_prebuilt_bundles", False):
        argv.append("--no-build")
    for root in options.bundle_roots:
        argv.extend(["--bundle-root", str(root.resolve())])
    for directory in options.runtime_dirs:
        argv.extend(["--runtime-dir", str(directory.resolve())])
    request = workload.get("request", {})
    for name, value in sorted((request or {}).items()):
        argv.extend(["--set", f"request.{name}={_yaml_cli_value(value)}"])
    runtime = workload.get("runtime", {})
    for name, value in sorted((runtime or {}).items()):
        argv.extend(["--set", f"runtime.{name}={_yaml_cli_value(value)}"])
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


def _candidate_build_dependency_preflight(
    resolved: Mapping[str, Any],
    *,
    profile: str,
    python: str,
    timeout_seconds: int,
) -> None:
    model = resolved.get("model", {})
    build = model.get("build", {}) if isinstance(model, Mapping) else {}
    quantization = build.get("quantization", {}) if isinstance(build, Mapping) else {}
    if not isinstance(quantization, Mapping) or (
        str(quantization.get("format", "")).strip().lower(),
        str(quantization.get("scale_source", "")).strip().lower(),
    ) != ("fp8", "modelopt"):
        return

    command = _run_command(
        [python, "-c", "import modelopt.torch.quantization"],
        _command_environment(),
        timeout_seconds,
    )
    if command["exit_code"] != 0:
        detail = str(command.get("stderr_tail", "")).strip()
        suffix = f": {detail}" if detail else ""
        raise PerfMatrixError(
            f"candidate build Python profile {profile!r} is missing "
            f"nvidia-modelopt required for automatic FP8 calibration{suffix}"
        )


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    paths = [str(PYTHON_SOURCE)]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _case_command_environment(options: RunOptions, case_work: Path) -> dict[str, str]:
    environment = _command_environment()
    if getattr(options, "hf_cache_mode", "shared") == "per_entry":
        environment["HF_HOME"] = str((case_work / "hf-cache").resolve())
        for name in HF_CACHE_ENVIRONMENT_NAMES[1:]:
            environment.pop(name, None)
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
    if expected_asset_loading and not any(name in request for name in ("audio_path", "image_path")):
        raise PerfMatrixError(
            f"case {case['id']} includes asset loading but resolves no timed asset path"
        )
    profile, python = _candidate_build_python_profile(value)
    _candidate_build_dependency_preflight(
        value,
        profile=profile,
        python=python,
        timeout_seconds=options.timeout_seconds,
    )
    value["_candidate_build_python_profile"] = profile
    command.pop("stdout", None)
    return value, argv, command


def _validate_worker_metadata(metadata: Mapping[str, Any], expected_revision: str) -> None:
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
                    raise PerfMatrixError(f"reference Python does not exist: {executable}")
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
                {"id": case_id, **failure} for case_id, failure in sorted(failures.items())
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


def _reference_precision(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    explicit = case["baseline"].get("precision")
    if explicit is not None:
        return str(explicit)
    testcase_name = str(resolved.get("testcase") or case.get("workload", {}).get("testcase") or "")
    testcases = manifest.get("testcases", [])
    if isinstance(testcases, list):
        for testcase in testcases:
            if (
                isinstance(testcase, Mapping)
                and str(testcase.get("name") or "") == testcase_name
                and testcase.get("reference_precision") is not None
            ):
                return str(testcase["reference_precision"])
    declared = manifest.get("reference_precision")
    if declared is not None:
        return str(declared)
    return str(resolved["model"]["precision"])


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
    reference_precision = _reference_precision(case, resolved, manifest)
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
        reference_precision,
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
    if baseline.get("local_files_only", options.local_files_only):
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
    reference_precision = _reference_precision(case, resolved, manifest)
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
        reference_precision,
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
    if baseline.get("local_files_only", options.local_files_only):
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
    result = {
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
        "stderr": stderr,
        "stdout_tail": stdout[-16000:],
        "stderr_tail": stderr[-16000:],
    }
    diagnostic = _command_diagnostic(stderr)
    if diagnostic is not None:
        result["diagnostic"] = diagnostic
    return result


def _command_diagnostic(stderr: str) -> dict[str, Any] | None:
    """Read the stable benchmark diagnostic marker without parsing prose."""

    for line in reversed(stderr.splitlines()):
        if not line.startswith(COMMAND_DIAGNOSTIC_PREFIX):
            continue
        try:
            value = json.loads(line[len(COMMAND_DIAGNOSTIC_PREFIX) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict) or value.get("schema_version") != COMMAND_DIAGNOSTIC_SCHEMA:
            return None
        if not all(str(value.get(name, "")).strip() for name in ("stage", "domain", "code")):
            return None
        artifacts = value.get("artifacts", [])
        if not isinstance(artifacts, list):
            return None
        if any(
            not isinstance(item, Mapping)
            or not str(item.get("label", "")).strip()
            or not str(item.get("path", "")).strip()
            for item in artifacts
        ):
            return None
        return deepcopy(value)
    return None


def _materialize_command_logs(
    output: Path,
    case_id: str,
    side: str,
    command: MutableMapping[str, Any],
    *,
    case_attempt: int = 1,
    measurement_attempt: int = 1,
) -> list[dict[str, str]]:
    directory = output / "artifacts" / _slug(case_id) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    links = []
    attempt_suffix = "" if case_attempt == 1 else f".case-attempt-{case_attempt}"
    measurement_suffix = "" if measurement_attempt == 1 else f".measurement-{measurement_attempt}"
    for stream in ("stdout", "stderr"):
        path = directory / (f"{side}{attempt_suffix}{measurement_suffix}.{stream}.log")
        path.write_text(str(command.pop(stream, "")), encoding="utf-8")
        href = path.relative_to(output).as_posix()
        command[f"{stream}_log"] = href
        side_label = "TRTMC" if side == "trtmc" else "Reference"
        measurement_label = (
            "" if measurement_attempt == 1 else f" measurement {measurement_attempt}"
        )
        links.append(
            {
                "label": f"{side_label}{measurement_label} {stream}",
                "href": href,
            }
        )
    diagnostic = command.get("diagnostic")
    if isinstance(diagnostic, MutableMapping):
        materialized = []
        artifacts = diagnostic.get("artifacts", [])
        artifacts = artifacts if isinstance(artifacts, list) else []
        for index, artifact in enumerate(artifacts, start=1):
            if not isinstance(artifact, Mapping):
                continue
            source = Path(str(artifact.get("path", "")))
            label = str(artifact.get("label", "") or f"Diagnostic {index}")
            if not source.is_file():
                continue
            suffix = "".join(source.suffixes) or ".log"
            path = directory / (
                f"{side}{attempt_suffix}{measurement_suffix}.{_slug(label)}{suffix}"
            )
            shutil.copyfile(source, path)
            record = {"label": label, "href": path.relative_to(output).as_posix()}
            materialized.append(record)
            links.append(record)
        diagnostic["artifacts"] = materialized
    return links


def _perf_attempt_evidence(
    output: Path,
    case_id: str,
    case_attempt: int,
    resolve_command: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    directory = output / "artifacts" / _slug(case_id) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if case_attempt == 1 else f".case-attempt-{case_attempt}"
    logs = []
    for side, label in (("trtmc", "TRTMC"), ("baseline", "Reference")):
        for stream in ("stdout", "stderr"):
            path = directory / f"{side}{suffix}.{stream}.log"
            path.touch(exist_ok=True)
            logs.append(
                {
                    "label": f"{label} {stream} case attempt {case_attempt}",
                    "href": path.relative_to(output).as_posix(),
                }
            )
    return {
        "source_revision": _execution_revision(),
        "commands": {"resolve": deepcopy(dict(resolve_command))},
        "logs": logs,
        "environment": {
            name: environment[name]
            for name in REPRODUCTION_ENVIRONMENT_NAMES
            if environment.get(name)
        },
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
    preparation = result.get("preparation", {})
    if not isinstance(preparation, Mapping):
        raise PerfMatrixError("trtmc-bench returned invalid preparation evidence")
    if preparation.get("included_in_performance_metrics", False) is not False:
        raise PerfMatrixError("trtmc-bench included preparation in performance metrics")
    bundles = preparation.get("bundles", [])
    if not isinstance(bundles, list) or not all(isinstance(bundle, Mapping) for bundle in bundles):
        raise PerfMatrixError("trtmc-bench returned invalid bundle preparation evidence")
    if any(bundle.get("included_in_performance_metrics", False) is not False for bundle in bundles):
        raise PerfMatrixError("trtmc-bench included a bundle build in performance metrics")
    return {
        "schema_version": result["schema_version"],
        "status": "completed",
        "backend": "trtmc-bench",
        "workload_digest": workload_digest,
        "measurement_policy": result.get("measurement_policy", {}),
        "preparation": {
            "included_in_performance_metrics": False,
            "bundles": [dict(bundle) for bundle in bundles],
        },
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


_TIMING_STABILITY_SAMPLE_COUNT = 10
_TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT = 5.0
_TIMING_STABILITY_MEDIAN_BAND_PERCENT = 5.0
_TIMING_STABILITY_MIN_IN_BAND = 8


def _timing_stability(values: Sequence[Any]) -> dict[str, Any]:
    """Decide whether one side's ten timed samples are settled."""

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


def _measurement_stability_evidence(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    reference = _timing_stability(baseline.get("samples_ms", []))
    trtmc = _timing_stability(candidate.get("samples_ms", []))
    side_statuses = {reference["status"], trtmc["status"]}
    if side_statuses == {"stable"}:
        status = "stable"
    elif "unstable" in side_statuses:
        status = "retry_recommended" if mode == "shadow" else "unstable"
    else:
        status = "not_evaluated"
    evidence = {
        "mode": mode,
        "status": status,
        "policy": {
            "required_samples": _TIMING_STABILITY_SAMPLE_COUNT,
            "max_half_median_change_percent": (_TIMING_STABILITY_MAX_HALF_CHANGE_PERCENT),
            "median_band_percent": _TIMING_STABILITY_MEDIAN_BAND_PERCENT,
            "minimum_samples_within_band": _TIMING_STABILITY_MIN_IN_BAND,
        },
        "reference": reference,
        "trtmc": trtmc,
    }
    return evidence


def _measurement_stability(row: Mapping[str, Any], result: str | None) -> dict[str, Any] | None:
    stored = row.get("measurement_stability")
    if isinstance(stored, Mapping):
        return deepcopy(dict(stored))
    if result not in qualification_report.RGB_RESULTS:
        return None
    baseline = row.get("baseline", {})
    candidate = row.get("candidate", {})
    baseline = baseline if isinstance(baseline, Mapping) else {}
    candidate = candidate if isinstance(candidate, Mapping) else {}
    return _measurement_stability_evidence(
        baseline,
        candidate,
        mode="shadow",
    )


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
    if contract == "disparity-parity":
        evidence = _disparity_contract_evidence(case, left, right)
        return bool(evidence["passed"]), str(evidence.get("reason", ""))
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
    if contract == "image-features-shape":
        left_shape = (
            left.get("last_hidden_state_shape"),
            left.get("pooler_output_shape"),
        )
        right_shape = (
            right.get("last_hidden_state_shape"),
            right.get("pooler_output_shape"),
        )
        matched = all(isinstance(value, list) and value for value in left_shape + right_shape)
        matched = matched and left_shape == right_shape
        return matched, "image feature output shape differs" if not matched else ""
    if contract == "image-features-parity":
        evidence = _image_feature_contract_evidence(case, left, right)
        return bool(evidence["passed"]), str(evidence.get("reason", ""))
    if contract == "reranking-order":
        left_scores = left.get("scores")
        right_scores = right.get("scores")
        if (
            not isinstance(left_scores, list)
            or not isinstance(right_scores, list)
            or not left_scores
            or len(left_scores) != len(right_scores)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (*left_scores, *right_scores)
            )
        ):
            return False, "reranking scores are missing or invalid"
        left_order = sorted(
            range(len(left_scores)), key=lambda index: (-float(left_scores[index]), index)
        )
        right_order = sorted(
            range(len(right_scores)), key=lambda index: (-float(right_scores[index]), index)
        )
        matched = left_order == right_order
        return matched, "reranking order differs" if not matched else ""
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
            matched = _normalized_text(left.get("text")) == _normalized_text(right.get("text"))
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
                return (
                    False,
                    "positional token agreement is below the configured contract",
                )
            distance = _normalized_text_edit_distance(left.get("text"), right.get("text"))
            limit = float(case["baseline"]["max_normalized_edit_distance"])
            if distance > limit:
                return False, "normalized text distance exceeds the configured contract"
            return True, ""
        if contract == "localization":
            candidate_localizations = parse_localizations(
                str(left.get("text", "")), require_reference=True
            )
            reference_localizations = parse_localizations(
                str(right.get("text", "")), require_reference=True
            )
            if candidate_localizations is None:
                return False, "TRTMC localization markup is invalid"
            if reference_localizations is None:
                return False, "baseline localization markup is invalid"
            if candidate_localizations[0].kind != reference_localizations[0].kind:
                return False, "localization output type differs"
            if len(candidate_localizations) != len(reference_localizations):
                return False, "localization output count differs"
            if reference_localizations[0].kind == "box":
                minimum_iou = matched_box_iou(candidate_localizations, reference_localizations)
                if minimum_iou < float(case["baseline"]["min_localization_box_iou"]):
                    return (
                        False,
                        "localization box IoU is below the configured contract",
                    )
            else:
                maximum_distance = matched_point_distance(
                    candidate_localizations, reference_localizations
                )
                if maximum_distance > float(case["baseline"]["max_localization_point_distance"]):
                    return False, ("localization point distance exceeds the configured contract")
            distance = _normalized_text_edit_distance(left.get("text"), right.get("text"))
            if distance > float(case["baseline"]["max_normalized_edit_distance"]):
                return (
                    False,
                    "normalized localization text distance exceeds the configured contract",
                )
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
                return (
                    False,
                    "normalized OCR text distance exceeds the configured contract",
                )
            return True, ""
        matched = left.get("token_ids") == right.get("token_ids")
        return matched, "generated token ids differ" if not matched else ""
    if operation == "encode":
        left_elements = left.get("element_count")
        right_elements = right.get("embedding_elements")
        matched = left_elements == right_elements and left.get("dim") == right.get("dim")
        return matched, "encoder output shape differs" if not matched else ""
    return True, ""


def _disparity_values(summary: Mapping[str, Any]) -> tuple[array[float], Path]:
    raw_path = summary.get("disparity_artifact")
    count = summary.get("element_count")
    if not isinstance(raw_path, str) or not raw_path:
        raise PerfMatrixError("disparity output is missing its FP32 artifact")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PerfMatrixError("disparity output has an invalid element count")
    path = Path(raw_path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PerfMatrixError(f"cannot read disparity artifact {path}: {exc}") from exc
    if len(payload) != count * 4:
        raise PerfMatrixError(
            f"disparity artifact {path} has {len(payload)} bytes, expected {count * 4}"
        )
    values = array("f")
    values.frombytes(payload)
    return values, path


def _image_feature_values(
    summary: Mapping[str, Any], shape_name: str, values_name: str
) -> tuple[tuple[int, ...], list[float]]:
    shape = summary.get(shape_name)
    values = summary.get(values_name)
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in shape
        )
    ):
        raise PerfMatrixError(f"image feature output has invalid {shape_name}")
    if not isinstance(values, list):
        raise PerfMatrixError(f"image feature output is missing {values_name}")
    expected = math.prod(shape)
    if len(values) != expected or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise PerfMatrixError(f"image feature output has invalid {values_name}")
    return tuple(shape), [float(value) for value in values]


def _image_feature_contract_evidence(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        left_hidden_shape, left_hidden = _image_feature_values(
            candidate, "last_hidden_state_shape", "last_hidden_state"
        )
        right_hidden_shape, right_hidden = _image_feature_values(
            reference, "last_hidden_state_shape", "last_hidden_state"
        )
        left_pooler_shape, left_pooler = _image_feature_values(
            candidate, "pooler_output_shape", "pooler_output"
        )
        right_pooler_shape, right_pooler = _image_feature_values(
            reference, "pooler_output_shape", "pooler_output"
        )
    except PerfMatrixError as error:
        return {"passed": False, "reason": str(error)}
    shapes_match = (
        left_hidden_shape == right_hidden_shape
        and left_pooler_shape == right_pooler_shape
    )
    if not shapes_match:
        return {
            "passed": False,
            "reason": "image feature output shape differs",
            "candidate_shapes": [list(left_hidden_shape), list(left_pooler_shape)],
            "reference_shapes": [list(right_hidden_shape), list(right_pooler_shape)],
        }

    def cosine(left: Sequence[float], right: Sequence[float]) -> float:
        dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(math.fsum(value * value for value in left))
        right_norm = math.sqrt(math.fsum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 1.0 if left == right else 0.0
        return dot / (left_norm * right_norm)

    hidden_cosine = cosine(left_hidden, right_hidden)
    pooler_cosine = cosine(left_pooler, right_pooler)
    reference_norm = math.sqrt(math.fsum(value * value for value in right_hidden))
    difference_norm = math.sqrt(
        math.fsum(
            (left - right) ** 2
            for left, right in zip(left_hidden, right_hidden, strict=True)
        )
    )
    relative_frobenius = difference_norm / max(reference_norm, 1e-12)
    thresholds = {
        "full_hidden_state_cosine_min": float(
            case["baseline"]["min_image_feature_full_cosine"]
        ),
        "pooler_output_cosine_min": float(
            case["baseline"]["min_image_feature_pooler_cosine"]
        ),
        "full_hidden_state_relative_frobenius_max": float(
            case["baseline"]["max_image_feature_relative_frobenius"]
        ),
    }
    metrics = {
        "full_hidden_state_cosine": hidden_cosine,
        "pooler_output_cosine": pooler_cosine,
        "full_hidden_state_relative_frobenius": relative_frobenius,
    }
    passed = (
        hidden_cosine >= thresholds["full_hidden_state_cosine_min"]
        and pooler_cosine >= thresholds["pooler_output_cosine_min"]
        and relative_frobenius
        <= thresholds["full_hidden_state_relative_frobenius_max"]
    )
    return {
        "passed": passed,
        "reason": (
            ""
            if passed
            else "image feature numeric parity is outside the configured contract"
        ),
        "metrics": metrics,
        "thresholds": thresholds,
        "candidate_shapes": [list(left_hidden_shape), list(left_pooler_shape)],
        "reference_shapes": [list(right_hidden_shape), list(right_pooler_shape)],
    }


def _disparity_contract_evidence(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    left_shape = tuple(candidate.get(name) for name in ("height", "width", "element_count"))
    right_shape = tuple(reference.get(name) for name in ("height", "width", "element_count"))
    expected_shape = (700, 700, 700 * 700)
    if left_shape != expected_shape or right_shape != expected_shape:
        return {
            "passed": False,
            "reason": "disparity outputs do not both have the required 700x700 shape",
            "candidate_shape": list(left_shape),
            "reference_shape": list(right_shape),
        }
    left, left_path = _disparity_values(candidate)
    right, right_path = _disparity_values(reference)
    finite_fraction = sum(math.isfinite(value) for value in left) / len(left)
    nonnegative_fraction = sum(math.isfinite(value) and value >= 0.0 for value in left) / len(left)
    reference_finite_fraction = sum(math.isfinite(value) for value in right) / len(right)
    reference_nonnegative_fraction = (
        sum(math.isfinite(value) and value >= 0.0 for value in right) / len(right)
    )
    dot = math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right))
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)]
    mean_abs_error = math.fsum(differences) / len(differences)
    bad_2px_fraction = sum(value > 2.0 for value in differences) / len(differences)
    thresholds = {
        "finite_fraction": 1.0,
        "nonnegative_fraction": 1.0,
        "global_cosine": float(case["baseline"]["min_disparity_cosine"]),
        "mean_abs_error": float(case["baseline"]["max_disparity_mean_abs_error"]),
        "bad_2px_fraction": float(case["baseline"]["max_disparity_bad_2px_fraction"]),
    }
    metrics = {
        "finite_fraction": finite_fraction,
        "nonnegative_fraction": nonnegative_fraction,
        "reference_finite_fraction": reference_finite_fraction,
        "reference_nonnegative_fraction": reference_nonnegative_fraction,
        "global_cosine": cosine,
        "mean_abs_error": mean_abs_error,
        "bad_2px_fraction": bad_2px_fraction,
    }
    passed = (
        finite_fraction == thresholds["finite_fraction"]
        and nonnegative_fraction == thresholds["nonnegative_fraction"]
        and reference_finite_fraction == 1.0
        and reference_nonnegative_fraction == 1.0
        and cosine >= thresholds["global_cosine"]
        and mean_abs_error <= thresholds["mean_abs_error"]
        and bad_2px_fraction <= thresholds["bad_2px_fraction"]
    )
    return {
        "passed": passed,
        "reason": "" if passed else "disparity output parity is outside the configured contract",
        "metrics": metrics,
        "thresholds": thresholds,
        "candidate_artifact": str(left_path),
        "reference_artifact": str(right_path),
        "candidate_sha256": hashlib.sha256(left_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(right_path.read_bytes()).hexdigest(),
    }


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
    reference_precision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    mismatch = _baseline_contract_mismatch(
        case,
        candidate,
        baseline,
        reference_precision=reference_precision,
    )
    if mismatch:
        return "contract-mismatch", {"reason": mismatch}
    outputs_match, reason = _output_contract(
        case,
        candidate,
        baseline,
        request=request,
    )
    output_evidence = None
    output_contract = _effective_output_contract(case, request)
    if output_contract == "disparity-parity":
        output_evidence = _disparity_contract_evidence(
            case,
            candidate.get("output_summary", {}),
            baseline.get("output_summary", {}),
        )
    elif output_contract == "image-features-parity":
        output_evidence = _image_feature_contract_evidence(
            case,
            candidate.get("output_summary", {}),
            baseline.get("output_summary", {}),
        )
    if not outputs_match:
        comparison = {"reason": reason}
        if output_evidence is not None:
            comparison["output_contract"] = output_evidence
        return "contract-mismatch", comparison
    candidate_p50 = _median(candidate)
    baseline_p50 = _median(baseline)
    ratio = baseline_p50 / candidate_p50
    margin = float(case.get("equivalence_margin_percent", 5.0)) / 100.0
    status = _comparison_status(ratio, margin)
    comparison = {
        "baseline_over_trtmc_p50": ratio,
        "equivalence_margin_percent": margin * 100.0,
    }
    if output_evidence is not None:
        comparison["output_contract"] = output_evidence
    return status, comparison


def _baseline_contract_mismatch(
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    reference_precision: str | None = None,
) -> str:
    timing_mismatch = _timing_contract_mismatch(case, candidate, baseline)
    if timing_mismatch:
        return timing_mismatch
    expected_mode = str(case["baseline"].get("mode", "torch-compile"))
    if baseline.get("mode") != expected_mode:
        return "baseline mode differs from the suite"
    expected_precision = (
        reference_precision
        if reference_precision is not None
        else case["baseline"].get("precision", candidate.get("precision"))
    )
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
    row = {
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
    task_type, user_contract, task_strategy = _report_task_metadata(row)
    row.update(
        {
            "task_type": task_type,
            "user_contract": user_contract,
            "task_strategy": task_strategy,
        }
    )
    return row


def _materialize_disparity_artifacts(
    case: Mapping[str, Any],
    candidate: MutableMapping[str, Any],
    baseline: MutableMapping[str, Any],
    output: Path,
    *,
    case_attempt: int,
    measurement_attempt: int,
) -> None:
    configured_baseline = case.get("baseline", {})
    if (
        not isinstance(configured_baseline, Mapping)
        or configured_baseline.get("output_contract") != "disparity-parity"
    ):
        return
    directory = output / "artifacts" / _slug(str(case["id"])) / "disparity"
    directory.mkdir(parents=True, exist_ok=True)
    for side, result in (("trtmc", candidate), ("reference", baseline)):
        summary = result.get("output_summary")
        if not isinstance(summary, MutableMapping):
            raise PerfMatrixError(f"{side} disparity output summary is invalid")
        source = Path(str(summary.get("disparity_artifact", "")))
        if not source.is_file():
            raise PerfMatrixError(f"{side} disparity artifact does not exist: {source}")
        destination = directory / (
            f"{side}.case-attempt-{case_attempt}.measurement-{measurement_attempt}.f32"
        )
        shutil.copyfile(source, destination)
        summary["disparity_artifact"] = str(destination.resolve())
        summary["disparity_artifact_href"] = destination.relative_to(output).as_posix()


def _run_measurement(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    options: RunOptions,
    case_work: Path,
    row: MutableMapping[str, Any],
    *,
    measurement_attempt: int,
    case_attempt: int = 1,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]] | None:
    environment = _case_command_environment(options, case_work)
    digest = _workload_digest(resolved)
    commands = row.setdefault("commands", {})
    order = ("trtmc", "baseline") if _stable_even(str(case["id"])) else ("baseline", "trtmc")
    measurement_work = (
        case_work if measurement_attempt == 1 else case_work / f"measurement-{measurement_attempt}"
    )
    measurement_work.mkdir(parents=True, exist_ok=True)
    candidate_dir = measurement_work / "trtmc"
    baseline_path = measurement_work / "baseline.json"
    candidate_argv = [
        *_candidate_base_argv(case, options),
        "--output",
        str(candidate_dir),
    ]
    baseline_argv, profile = _baseline_argv(case, resolved, baseline_path, options)
    row["resolved_settings"]["baseline_python_profile"] = profile
    reference_precision = baseline_argv[baseline_argv.index("--precision") + 1]
    row["resolved_settings"]["baseline_precision"] = reference_precision
    suffix = "" if measurement_attempt == 1 else f"_measurement_{measurement_attempt}"
    command_keys = {
        "trtmc": f"trtmc{suffix}",
        "baseline": f"baseline{suffix}",
    }
    for side, argv in (("trtmc", candidate_argv), ("baseline", baseline_argv)):
        commands[command_keys[side]] = {
            "argv": argv,
            "rendered": shlex.join(argv),
            "cwd": str(REPOSITORY),
        }
    if getattr(options, "verbose", False):
        label = "" if measurement_attempt == 1 else " remeasurement"
        print(
            f"[{case['id']}] TRTMC{label}: {commands[command_keys['trtmc']]['rendered']}",
            flush=True,
        )
        print(
            f"[{case['id']}] baseline{label}: {commands[command_keys['baseline']]['rendered']}",
            flush=True,
        )
    for side in order:
        argv = candidate_argv if side == "trtmc" else baseline_argv
        stage = "candidate" if side == "trtmc" else "reference"
        if measurement_attempt > 1:
            stage = f"remeasure-{stage}"
        if progress is not None:
            progress(
                stage,
                {
                    "commands": deepcopy(row["commands"]),
                    "logs": deepcopy(row.get("logs", [])),
                },
            )
        _wait_for_gpu_memory_headroom(minimum_free_fraction=options.minimum_gpu_free_fraction)
        command = _run_command(argv, environment, options.timeout_seconds)
        row.setdefault("logs", []).extend(
            _materialize_command_logs(
                getattr(options, "output", case_work),
                str(case["id"]),
                side,
                command,
                case_attempt=case_attempt,
                measurement_attempt=measurement_attempt,
            )
        )
        commands[command_keys[side]] = command
        if command["exit_code"] != 0:
            row["status"] = "failed"
            row["reason"] = (
                f"{side} measurement {measurement_attempt} command failed "
                f"with rc={command['exit_code']}"
            )
            diagnostic = command.get("diagnostic", {})
            diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
            row["failure_stage"] = str(diagnostic.get("stage") or stage)
            row["failure_domain"] = str(diagnostic.get("domain") or "harness/unknown")
            row["failure_code"] = str(diagnostic.get("code") or "execution_failure")
            row.pop("measurement_stability", None)
            return None
    candidate = _candidate_result(candidate_dir, digest)
    candidate["precision"] = str(resolved["model"]["precision"])
    baseline = _read_baseline(baseline_path)
    _materialize_disparity_artifacts(
        case,
        candidate,
        baseline,
        getattr(options, "output", case_work),
        case_attempt=case_attempt,
        measurement_attempt=measurement_attempt,
    )
    status, comparison = _classify(
        case,
        candidate,
        baseline,
        request=resolved["request"],
        reference_precision=reference_precision,
    )
    return candidate, baseline, status, comparison


def _run_supported_case(
    case: Mapping[str, Any],
    resolved: Mapping[str, Any],
    options: RunOptions,
    case_work: Path,
    row: MutableMapping[str, Any],
    *,
    case_attempt: int = 1,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> None:
    measurement_attempts: list[dict[str, Any]] = []
    for measurement_attempt in (1, 2):
        measured = _run_measurement(
            case,
            resolved,
            options,
            case_work,
            row,
            measurement_attempt=measurement_attempt,
            case_attempt=case_attempt,
            progress=progress,
        )
        if measured is None:
            return
        candidate, baseline, status, comparison = measured
        row.update(
            candidate=candidate,
            baseline=baseline,
            comparison=comparison,
            status=status,
        )
        if status not in qualification_report.RGB_RESULTS:
            row.pop("measurement_stability", None)
            return

        stability = _measurement_stability_evidence(
            baseline,
            candidate,
            mode="enforced",
        )
        measurement_attempts.append(
            {
                "attempt": measurement_attempt,
                "result": status,
                "reference": {
                    "samples_ms": deepcopy(baseline.get("samples_ms", [])),
                    "stability": deepcopy(stability["reference"]),
                },
                "trtmc": {
                    "samples_ms": deepcopy(candidate.get("samples_ms", [])),
                    "stability": deepcopy(stability["trtmc"]),
                },
            }
        )
        row["measurement_stability"] = {
            **stability,
            "attempts": deepcopy(measurement_attempts),
        }
        if stability["status"] == "stable":
            if measurement_attempt == 2:
                row["measurement_stability"]["status"] = "stable_after_retry"
            return
        if stability["status"] == "not_evaluated":
            row["status"] = "measurement-inconclusive"
            row["reason"] = "timing stability requires ten valid samples per side"
            row["failure_stage"] = "measure"
            row["failure_domain"] = "harness/unknown"
            row["failure_code"] = "measurement_inconclusive"
            row["measurement_stability"]["status"] = "measurement_inconclusive"
            return
        if measurement_attempt == 1:
            print(
                f"[{case['id']}] timing is not settled; measuring once more",
                flush=True,
            )

    row["status"] = "measurement-inconclusive"
    row["reason"] = "timing did not settle after one remeasurement"
    row["failure_stage"] = "measure"
    row["failure_domain"] = "harness/unknown"
    row["failure_code"] = "measurement_inconclusive"
    row["measurement_stability"]["status"] = "measurement_inconclusive"


def _stable_even(value: str) -> bool:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:2], 16) % 2 == 0


def _run_one(
    case: Mapping[str, Any],
    options: RunOptions,
    work_root: Path,
    preflight: tuple[dict[str, Any], list[str], dict[str, str]],
    *,
    case_attempt: int = 1,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    resolved, _, dry_command = preflight
    digest = _workload_digest(resolved)
    row = _case_row(case, resolved, digest)
    row["commands"]["resolve"] = dry_command
    case_work = work_root / _slug(str(case["id"])) / f"attempt-{case_attempt}"
    if case_work.exists():
        shutil.rmtree(case_work)
    case_work.mkdir(parents=True, exist_ok=True)
    try:
        _run_supported_case(
            case,
            resolved,
            options,
            case_work,
            row,
            case_attempt=case_attempt,
            progress=progress,
        )
    except (PerfMatrixError, OSError, ValueError, RuntimeError) as exc:
        row["status"] = "failed"
        row["reason"] = str(exc)
    row["finished_at"] = _now()
    return row


def _delete_for_retention(policy: str, *, passed: bool) -> bool:
    return policy == "delete_always" or (policy == "delete_on_pass" and passed)


def _cleanup_managed_bundle(
    resolved: Mapping[str, Any],
    options: RunOptions,
    *,
    passed: bool,
) -> dict[str, Any]:
    policy = options.bundle_retention
    evidence: dict[str, Any] = {"policy": policy, "status": "retained"}
    if not _delete_for_retention(policy, passed=passed):
        return evidence
    if options.bundle_cache is None:
        return {**evidence, "status": "failed", "error": "bundle cache is not managed"}
    raw_bundle = resolved.get("bundle_path")
    if not isinstance(raw_bundle, str) or not raw_bundle:
        return {
            **evidence,
            "status": "failed",
            "error": "resolved entry does not record bundle_path",
        }
    cache = options.bundle_cache.expanduser().resolve()
    bundle = Path(raw_bundle).expanduser().resolve()
    evidence["bundle"] = str(bundle)
    try:
        relative = bundle.relative_to(cache)
    except ValueError:
        return {
            **evidence,
            "status": "preserved_external",
            "reason": "bundle is outside the managed cache",
        }
    if len(relative.parts) < 3 or bundle.parent == cache:
        return {
            **evidence,
            "status": "failed",
            "error": "managed bundle lacks the expected model/cache-key/bundle layout",
        }
    cache_entry = bundle.parent
    evidence["cache_entry"] = str(cache_entry)
    evidence["scope"] = "bundle_only"
    try:
        bundle.unlink()
    except FileNotFoundError:
        evidence["status"] = "already_absent"
    except OSError as exc:
        evidence.update({"status": "failed", "error": str(exc)})
    else:
        evidence["status"] = "deleted"
    return evidence


def _cleanup_entry_work(
    case_work: Path,
    options: RunOptions,
    *,
    passed: bool,
) -> dict[str, Any]:
    hf_policy = (
        options.hf_cache_retention if options.hf_cache_mode == "per_entry" else "delete_on_pass"
    )
    evidence: dict[str, Any] = {
        "path": str(case_work),
        "hf_cache_mode": options.hf_cache_mode,
        "hf_cache_retention": (
            options.hf_cache_retention
            if options.hf_cache_mode == "per_entry"
            else "shared_retained"
        ),
        "status": "retained",
    }
    if not _delete_for_retention(hf_policy, passed=passed):
        return evidence
    try:
        shutil.rmtree(case_work)
    except FileNotFoundError:
        evidence["status"] = "already_absent"
    except OSError as exc:
        evidence.update({"status": "failed", "error": str(exc)})
    else:
        evidence["status"] = "deleted"
    return evidence


def _resolution_failure_row(case: Mapping[str, Any], failure: Mapping[str, Any]) -> dict[str, Any]:
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
    materialized = list(rows)
    statuses = {str(row.get("status")) for row in materialized}
    invalid = {"failed", "contract-mismatch", "partial", "pending", "running"}
    if statuses & invalid:
        return "completed-with-errors"
    if any(row.get("warnings") for row in materialized):
        return "completed-with-warnings"
    return "completed"


def _light(status: str) -> str:
    return {
        "green": "🟢",
        "yellow": "🟡",
        "red": "🔴",
    }.get(status, "⚪")


def _perf_precision(row: Mapping[str, Any]) -> dict[str, str]:
    resolved = row.get("resolved_settings", {})
    resolved = resolved if isinstance(resolved, Mapping) else {}
    model = resolved.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    baseline = row.get("baseline", {})
    baseline = baseline if isinstance(baseline, Mapping) else {}
    candidate = row.get("candidate", {})
    candidate = candidate if isinstance(candidate, Mapping) else {}
    contract = row.get("baseline_contract", {})
    contract = contract if isinstance(contract, Mapping) else {}
    reference = (
        resolved.get("baseline_precision") or baseline.get("precision") or contract.get("precision")
    )
    base_precision = candidate.get("precision") or model.get("precision")
    build = model.get("build", {})
    build = build if isinstance(build, Mapping) else {}
    quantization = build.get("quantization", {})
    quantization = quantization if isinstance(quantization, Mapping) else {}
    quantization_format = str(quantization.get("format", "") or "").lower()
    if quantization_format and quantization_format not in {"none", "false"}:
        trtmc = (
            f"{quantization_format} ({str(base_precision).lower()} base)"
            if base_precision
            else quantization_format
        )
    else:
        trtmc = base_precision
    return {
        "reference": str(reference).lower() if reference else "Not recorded",
        "candidate": str(trtmc).lower() if trtmc else "Not recorded",
    }


def _perf_state_and_result(row: Mapping[str, Any]) -> tuple[str, str | None]:
    status = str(row.get("status", "pending"))
    if status in {"pending", "running"}:
        return status, None
    if (
        status in qualification_report.RGB_RESULTS
        and "Not recorded" in _perf_precision(row).values()
    ):
        return "terminal", "white"
    return "terminal", status if status in qualification_report.RGB_RESULTS else "white"


def _sync_perf_results_from_ledger(
    results: MutableMapping[str, Any],
    ledger: ExecutionLedger,
) -> None:
    rows = _result_rows(results)
    for entry in ledger.snapshot():
        case = entry["case"]
        receipt = entry["receipt"]
        case_id = str(case["id"])
        if case_id not in rows:
            raise PerfMatrixError(f"Performance ledger contains unknown case {case_id!r}")
        base = case.get("report")
        if not isinstance(base, Mapping):
            raise PerfMatrixError(f"invalid Performance ledger descriptor for {case_id!r}")
        if receipt["state"] == "terminal":
            row = deepcopy(receipt["payload"])
            state, derived = _perf_state_and_result(row)
            if state != "terminal" or receipt["result"] != derived:
                raise PerfMatrixError(
                    f"Performance ledger result mismatch for {case_id!r}: "
                    f"receipt={receipt['result']}, result={derived}"
                )
        else:
            row = {**deepcopy(dict(base)), "status": receipt["state"]}
            active = receipt["attempts"][-1] if receipt["attempts"] else {}
            evidence = active.get("evidence", {})
            evidence = evidence if isinstance(evidence, Mapping) else {}
            row["commands"] = deepcopy(dict(evidence.get("commands", {})))
            row["logs"] = deepcopy(list(evidence.get("logs", [])))
        row["progress"] = {
            "stage": receipt["stage"],
            "attempt": receipt["active_attempt"] or len(receipt["attempts"]),
        }
        rows[case_id].clear()
        rows[case_id].update(row)


def _perf_issue(row: Mapping[str, Any], result: str | None) -> dict[str, str] | None:
    if result != "white":
        return None
    status = str(row.get("status", "failed"))
    reason = str(
        row.get("reason")
        or (
            row.get("comparison", {}).get("reason")
            if isinstance(row.get("comparison"), Mapping)
            else ""
        )
        or "performance comparison did not complete"
    )
    if status == "contract-mismatch":
        return {
            "priority": "P1",
            "stage": "compare",
            "domain": "model-workload",
            "code": "output_not_comparable",
            "message": reason,
        }
    if status in qualification_report.RGB_RESULTS:
        return {
            "priority": "P1",
            "stage": "preflight",
            "domain": "policy-config",
            "code": "comparison_contract",
            "message": "Reference and TRTMC compute precision were not both recorded",
        }
    return {
        "priority": "P1",
        "stage": str(row.get("failure_stage", "candidate")),
        "domain": str(row.get("failure_domain", "harness/unknown")),
        "code": str(row.get("failure_code", "execution_failure")),
        "message": reason,
    }


def _perf_output_validation(
    row: Mapping[str, Any],
    result: str | None,
) -> dict[str, Any]:
    resolved = row.get("resolved_settings", {})
    resolved = resolved if isinstance(resolved, Mapping) else {}
    contract = str(resolved.get("output_contract", "") or "Not recorded")
    comparison = row.get("comparison", {})
    comparison = comparison if isinstance(comparison, Mapping) else {}
    if str(row.get("status")) in {
        *qualification_report.RGB_RESULTS,
        "measurement-inconclusive",
    }:
        status = "Pass"
        reason = None
    elif str(row.get("status")) == "contract-mismatch":
        status = "Fail"
        reason = str(comparison.get("reason", "") or row.get("reason", ""))
    else:
        status = "Not completed"
        reason = str(row.get("reason", "") or "") or None
    return {"status": status, "contract": contract, "reason": reason}


def _perf_latency(row: Mapping[str, Any], result: str | None) -> dict[str, float | None]:
    if result not in qualification_report.RGB_RESULTS:
        return {"reference_ms": None, "candidate_ms": None}
    try:
        return {
            "reference_ms": _median(row["baseline"]),
            "candidate_ms": _median(row["candidate"]),
        }
    except (KeyError, PerfMatrixError, TypeError, ValueError):
        return {"reference_ms": None, "candidate_ms": None}


def _public_perf_result(row: Mapping[str, Any]) -> dict[str, Any]:
    state, result = _perf_state_and_result(row)
    public = deepcopy(dict(row))
    commands = public.get("commands", {})
    if isinstance(commands, Mapping):
        for command in commands.values():
            if isinstance(command, MutableMapping):
                command.pop("stdout_tail", None)
                command.pop("stderr_tail", None)
    public.update(
        {
            "state": state,
            "result": result,
            "precision": _perf_precision(row),
            "output_validation": _perf_output_validation(row, result),
            "latency": _perf_latency(row, result),
            "measurement_stability": _measurement_stability(row, result),
            "issue": _perf_issue(row, result),
            "debug": {
                "logs": [dict(record) for record in row.get("logs", [])],
            },
        }
    )
    return public


def _public_perf_results(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = {str(value) for value in results.get("selected_entry_ids", [])}
    return [
        _public_perf_result(row)
        for row in results.get("cases", [])
        if str(row.get("id", "")) in selected
    ]


def _materialize_public_perf_report(
    output: Path,
    results: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    public_results = _public_perf_results(results)
    revision_summary = case_evidence.summarize_model_revisions(
        {**row, "task": "performance"} for row in public_results
    )
    source_revisions = revision_summary["source_revisions"]
    report_revision = source_revisions[0] if len(source_revisions) == 1 else None
    environment = results.get("environment", {})
    environment = environment if isinstance(environment, Mapping) else {}
    run = {
        "source_revision": report_revision,
        "source_revisions": source_revisions,
        "hostname": environment.get("hostname"),
        "gpu": environment.get("gpu"),
        "gpu_uuid": environment.get("gpu_uuid"),
        "driver": environment.get("driver"),
        "platform": environment.get("platform"),
        "python": environment.get("python"),
        "python_executable": environment.get("python_executable"),
        "cuda_visible_devices": environment.get("cuda_visible_devices"),
        "nvidia_visible_devices": environment.get("nvidia_visible_devices"),
        "started_at": results.get("started_at"),
        "finished_at": results.get("finished_at"),
    }
    return qualification_report.materialize_report(
        output,
        report_kind="performance",
        title="TRTMC Performance Qualification",
        identity={
            "run_id": output.name,
            "disposition": results.get("status", "running"),
            "source_revision": report_revision,
        },
        run=run,
        results=public_results,
        metadata={
            "campaign": {
                "suite_name": results.get("suite_name"),
                "suite_sha256": results.get("suite_sha256"),
                "status": results.get("status", "running"),
            },
            "model_source_identity": revision_summary,
        },
    )


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
                "asset load and public pipeline call" if asset_included else "public pipeline call"
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
                "module input transfer, model execution, inter-module work, output materialization"
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
                "asset load and task pipeline call" if asset_included else "task pipeline call"
            ),
            "included": (
                "asset loading, reference input preparation, model operation, returned summary"
                if asset_included
                else "reference input preparation, model operation, returned summary"
            ),
            "excluded": (
                "model load, warmup" if asset_included else "model load, warmup, asset loading"
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


def _duration_html(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:,.3f} s"
    if seconds < 3600.0:
        minutes, remaining = divmod(seconds, 60.0)
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, remaining = divmod(seconds, 3600.0)
    minutes, remaining = divmod(remaining, 60.0)
    return f"{int(hours)}h {int(minutes)}m {remaining:.0f}s"


def _wall_time_seconds(record: Mapping[str, Any]) -> float | None:
    started = record.get("started_at")
    finished = record.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        seconds = (
            datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        ).total_seconds()
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds) or seconds < 0.0:
        return None
    return seconds


def _wall_time_html(record: Mapping[str, Any]) -> str:
    seconds = _wall_time_seconds(record)
    if seconds is None:
        return "—"
    return f"{_duration_html(seconds)}<div class='timing-meta'>{seconds:,.3f} s</div>"


def _bundle_preparation_records(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidate = row.get("candidate")
    if not isinstance(candidate, Mapping):
        return []
    preparation = candidate.get("preparation")
    if not isinstance(preparation, Mapping):
        return []
    bundles = preparation.get("bundles")
    if not isinstance(bundles, list):
        return []
    return [bundle for bundle in bundles if isinstance(bundle, Mapping)]


def _test_task_bundle_preparation(
    results: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    preparation = results.get("bundle_preparation")
    if not isinstance(preparation, Mapping):
        return {}
    bundles = preparation.get("bundles")
    if not isinstance(bundles, list):
        return {}
    return {
        (str(bundle.get("model", "")), str(bundle.get("bundle", ""))): bundle
        for bundle in bundles
        if isinstance(bundle, Mapping)
    }


def _effective_bundle_preparation_records(
    results: Mapping[str, Any], row: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    task_records = _test_task_bundle_preparation(results)
    return [
        task_records.get(
            (str(bundle.get("model", "")), str(bundle.get("bundle", ""))),
            bundle,
        )
        for bundle in _bundle_preparation_records(row)
    ]


def _bundle_preparation_html(results: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    bundles = _effective_bundle_preparation_records(results, row)
    if not bundles:
        return "—<div class='timing-meta'>Not recorded</div>"
    labels = {
        "built": "Built",
        "cache_hit": "Cache hit",
        "reused": "Existing bundle",
        "would_build": "Would build",
    }
    parts = []
    for bundle in bundles:
        status = str(bundle.get("status", "unknown"))
        label = labels.get(status, status.replace("_", " ").title())
        build_time = bundle.get("build_time_s")
        if isinstance(build_time, (int, float)) and math.isfinite(build_time):
            parts.append(
                f"{html.escape(label)} · {_duration_html(float(build_time))}"
                f"<div class='timing-meta'>{float(build_time):,.3f} s</div>"
            )
        else:
            parts.append(html.escape(label))
    parts.append("<div class='timing-meta'>Excluded from comparison</div>")
    return "<br>".join(parts)


def _bundle_preparation_summary(results: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    records = [
        record for row in rows for record in _effective_bundle_preparation_records(results, row)
    ]
    built = [record for record in records if record.get("status") == "built"]
    build_seconds = sum(
        float(record["build_time_s"])
        for record in built
        if isinstance(record.get("build_time_s"), (int, float))
        and math.isfinite(float(record["build_time_s"]))
    )
    prebuilt = sum(record.get("status") in {"cache_hit", "reused"} for record in records)
    unrecorded = sum(not _effective_bundle_preparation_records(results, row) for row in rows)
    return (
        f"Bundle preparation: {len(built)} built in this test task "
        f"({_duration_html(build_seconds)} total), {prebuilt} cache hits or existing "
        f"bundles, and {unrecorded} rows not recorded. Preparation is reported for "
        "run observability and is excluded from the infer-time traffic-light comparison."
    )


def _bundle_preparation_filter(results: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    statuses = {
        str(record.get("status", "unknown"))
        for record in _effective_bundle_preparation_records(results, row)
    }
    return " ".join(sorted(statuses)) if statuses else "unrecorded"


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
    working_directories = {str(command.get("cwd") or default_cwd) for _, command in recorded}
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
                f"{name}={shlex.quote(str(value))}" for name, value in sorted(environment.items())
            )
            parts.append(
                "<div class='command-label'>Environment</div>"
                f"<pre><code>{html.escape(rendered_environment)}</code></pre>"
            )
    parts.append("</details>")
    return "".join(parts)


@cache
def _manifest_user_contract(manifest: str, testcase: str) -> str:
    path = REPOSITORY / "tests" / "e2e" / "models" / manifest
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    testcases = payload.get("testcases", [])
    if not isinstance(testcases, list):
        return ""
    selected = next(
        (
            candidate
            for candidate in testcases
            if isinstance(candidate, Mapping) and candidate.get("name") == testcase
        ),
        None,
    )
    if selected is None:
        selected = next(
            (candidate for candidate in testcases if isinstance(candidate, Mapping)),
            {},
        )
    return str(selected.get("user_contract", "") or "")


def _report_task_metadata(row: Mapping[str, Any]) -> tuple[str, str, str]:
    resolved = row.get("resolved_settings", {})
    resolved = resolved if isinstance(resolved, Mapping) else {}
    model = resolved.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    task_strategy = str(row.get("task_strategy", "") or model.get("task_strategy", "") or "")
    user_contract = str(row.get("user_contract", "") or "")
    if not user_contract:
        user_contract = _manifest_user_contract(
            str(model.get("manifest", "") or ""),
            str(resolved.get("testcase", row.get("model", "")) or ""),
        )
    request = resolved.get("request", {})
    task_type = str(row.get("task_type", "") or "") or task_type_label(
        user_contract=user_contract,
        task_strategy=task_strategy,
        operation=row.get("operation", ""),
        request=request,
    )
    return task_type, user_contract, task_strategy


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
        status_filter = status if status in {"green", "yellow", "red"} else "white"
        task_type, user_contract, task_strategy = _report_task_metadata(row)
        search_filter = " ".join(
            (
                str(row.get("family", "")),
                str(row.get("operation", "")),
                str(row.get("model", "")),
                task_type,
                user_contract,
                task_strategy,
                str(row.get("id", "")),
            )
        ).lower()
        preparation_filter = _bundle_preparation_filter(results, row)
        reason = html.escape(_report_note(row))
        commands = _raw_commands_html(row, default_cwd)
        candidate_timing = _timing_value_html(row.get("candidate"))
        baseline_timing = _timing_value_html(row.get("baseline"))
        model_wall_time = _wall_time_html(row)
        bundle_preparation = _bundle_preparation_html(results, row)
        timing_scope = _timing_scope_html(row)
        task_type_detail = (
            f"<div class='detail'>{html.escape(user_contract)}</div>" if user_contract else ""
        )
        body.append(
            f"<tr data-filter-search='{html.escape(search_filter)}' "
            f"data-filter-model-type='{html.escape(str(row['family']))}' "
            f"data-filter-operation='{html.escape(str(row['operation']))}' "
            f"data-filter-task-type='{html.escape(task_type)}' "
            f"data-filter-status='{html.escape(status_filter)}' "
            f"data-filter-preparation='{html.escape(preparation_filter)}'>"
            f"<td><code>{html.escape(str(row['family']))}</code></td>"
            f"<td><code>{html.escape(str(row['operation']))}</code></td>"
            f"<td><code>{html.escape(str(row['model']))}</code></td>"
            f"<td><strong>{html.escape(task_type or '—')}</strong>{task_type_detail}</td>"
            f"<td class='timing-value'>{model_wall_time}</td>"
            f"<td>{html.escape(_baseline_label(row))}</td>"
            f"<td class='timing-value'>{candidate_timing}</td>"
            f"<td class='timing-value'>{baseline_timing}</td>"
            f"<td class='timing-value'>{bundle_preparation}</td>"
            f"<td>{timing_scope}</td>"
            f"<td class='light'>{_light(status)}</td>"
            f"<td>{reason}</td>"
            f"<td>{commands}</td>"
            "</tr>"
        )
    generated = html.escape(str(results.get("finished_at", results.get("started_at", ""))))
    campaign_seconds = _wall_time_seconds(results)
    campaign_duration = _duration_html(campaign_seconds) if campaign_seconds is not None else "—"
    campaign_seconds_label = (
        f"{campaign_seconds:,.3f} s" if campaign_seconds is not None else "unavailable"
    )
    campaign_started = html.escape(str(results.get("started_at", "—")))
    campaign_finished = html.escape(str(results.get("finished_at", "—")))
    summary = f"🟢 {counts['green']} &nbsp; 🟡 {counts['yellow']} &nbsp; 🔴 {counts['red']} &nbsp; ⚪ {len(rows) - sum(counts.values())}"
    repeated_note = (
        " " + html.escape("; ".join(repeated_families)) + " contribute multiple rows."
        if repeated_families
        else ""
    )
    preflight = results.get("timing_preflight", {})
    preflight_count = int(preflight.get("case_count", 0)) if isinstance(preflight, Mapping) else 0
    catalog = results.get("catalog_coverage", {})
    ready_profiles = int(catalog.get("ready_profiles", len(rows)))
    release_profiles = int(catalog.get("release_profiles", len(rows)))
    excluded_l0_profiles = int(catalog.get("excluded_l0_profiles", 0))
    explicitly_excluded_profiles = int(catalog.get("explicitly_excluded_profiles", 0))
    distributed_profiles = int(catalog.get("distributed_profiles", 0))
    other_profiles = int(catalog.get("other_profiles", 0))
    explicit_exclusion_label = f"{explicitly_excluded_profiles} explicitly excluded profile" + (
        "s" if explicitly_excluded_profiles != 1 else ""
    )
    bundle_preparation_summary = html.escape(_bundle_preparation_summary(results, rows))
    report_filters = render_report_filters(
        row_count=len(rows),
        search_placeholder="Model type, operation, model, or task type",
        filters=(
            ReportFilter(
                "model-type",
                "Model type",
                sorted_filter_values(row.get("family", "") for row in rows),
            ),
            ReportFilter(
                "operation",
                "Operation",
                sorted_filter_values(row.get("operation", "") for row in rows),
            ),
            ReportFilter(
                "task-type",
                "Task type",
                sorted_filter_values(_report_task_metadata(row)[0] for row in rows),
            ),
            ReportFilter(
                "status",
                "Status",
                tuple(
                    (status, label)
                    for status, label in (
                        ("green", "Green"),
                        ("yellow", "Yellow"),
                        ("red", "Red"),
                        ("white", "White"),
                    )
                    if any(
                        (
                            str(row.get("status", "pending"))
                            if row.get("status") in {"green", "yellow", "red"}
                            else "white"
                        )
                        == status
                        for row in rows
                    )
                ),
            ),
            ReportFilter(
                "preparation",
                "Bundle preparation",
                (
                    ("built", "Built"),
                    ("cache_hit", "Cache hit"),
                    ("reused", "Existing bundle"),
                    ("would_build", "Would build"),
                    ("unrecorded", "Not recorded"),
                ),
                token_values=True,
            ),
        ),
    )
    filter_script = render_report_filter_script()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRTMC performance matrix</title>
<style>
{COMMON_REPORT_STYLES}
.table-wrap table{{min-width:1800px}}
.command-label{{font-weight:650;margin-top:8px}} pre{{max-width:720px}}
.light{{font-size:20px;text-align:center}} .timing-value{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.scope-side{{font-size:11px;margin-bottom:7px;min-width:270px}}
.scope-title{{font-size:12px;font-weight:700;margin-bottom:2px}}
</style></head><body>
<header class="report-header">
<p class="report-eyebrow">Performance report</p>
<h1>TRTMC performance matrix</h1>
<p class="purpose">TRTMC and reference latency at aligned public task boundaries.</p>
</header>
<p class="meta">Generated {generated}. {family_count} model types across {len(rows)} model comparisons.{repeated_note}</p>
<p class="campaign-duration"><strong>Total campaign wall time:</strong> {campaign_duration} <span class="timing-meta">({campaign_seconds_label}; {campaign_started} → {campaign_finished})</span></p>
<p class="meta">Release matrix coverage: {release_profiles} single-process profiles. {explicit_exclusion_label} and {excluded_l0_profiles} duplicate L0 profiles are excluded from the {ready_profiles} ready catalog profiles. {distributed_profiles} distributed profiles require a separate multi-process run and are outside this report. {other_profiles} other or unsupported profiles.</p>
<p class="meta">Timing contracts were validated before execution for {preflight_count} comparisons. A row is classified only when its recorded TRTMC and reference policies match that contract.</p>
<p class="meta">Times are the p50 wall time from the recorded timed samples. Measured scope states the exact recorded boundary and the work included and excluded on each side.</p>
<p class="meta">Model-profile wall time spans the complete case from start to finish, including bundle preparation, GPU headroom waits, TRTMC and baseline commands, and orchestration overhead. It is reported for observability and is not used for traffic-light classification.</p>
<p class="meta">{bundle_preparation_summary}</p>
<div class="traffic-summary">{summary}</div>
{report_filters}
<div class="table-wrap"><table><thead><tr><th>Model type</th><th>Operation</th><th>Model</th><th>Task type</th><th>Model wall time</th><th>Baseline</th><th>TRTMC infer p50 (ms)</th><th>Baseline infer p50 (ms)</th><th>TRTMC bundle preparation</th><th>Measured scope</th><th>Status</th><th>Note</th><th>Commands</th></tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<p class="meta">Green: TRTMC is more than the configured margin faster. Yellow: within the margin. Red: TRTMC is more than the margin slower. White: not run, partial, or invalid comparison. Commands are the original recorded argv and must be run from the displayed working directory with the same model cache and dependencies.</p>
{filter_script}
</body></html>"""


def _write_artifacts(
    output: Path,
    results: MutableMapping[str, Any],
    ledger: ExecutionLedger | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    if ledger is not None:
        _sync_perf_results_from_ledger(results, ledger)
    _write_json(output / "results.json", results)
    report = _materialize_public_perf_report(output, results)
    legacy_replay = output / "reproduce.py"
    if legacy_replay.is_file():
        legacy_replay.unlink()
    return report


def _apply_bundle_preparation_receipt(
    results: MutableMapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    if receipt.get("schema_version") != PREPARATION_SCHEMA:
        raise PerfMatrixError(f"preparation receipt schema_version must be {PREPARATION_SCHEMA!r}")
    if receipt.get("scope") != "test_task":
        raise PerfMatrixError("preparation receipt scope must be 'test_task'")
    if receipt.get("git_commit") != results.get("git_commit"):
        raise PerfMatrixError(
            "preparation receipt repository revision does not match the performance run"
        )
    if receipt.get("included_in_performance_metrics") is not False:
        raise PerfMatrixError(
            "preparation receipt must exclude bundle preparation from performance metrics"
        )
    bundles = receipt.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise PerfMatrixError("preparation receipt must contain at least one bundle")

    run_bundles = {
        (str(bundle.get("model", "")), str(bundle.get("bundle", "")))
        for row in results.get("cases", [])
        if isinstance(row, Mapping)
        for bundle in _bundle_preparation_records(row)
    }
    seen: set[tuple[str, str]] = set()
    allowed_statuses = {"built", "cache_hit", "reused"}
    validated: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping):
            raise PerfMatrixError(f"preparation receipt bundle {index} must be a JSON object")
        model = bundle.get("model")
        path = bundle.get("bundle")
        status = bundle.get("status")
        if not isinstance(model, str) or not model:
            raise PerfMatrixError(f"preparation receipt bundle {index} has no model")
        if not isinstance(path, str) or not path:
            raise PerfMatrixError(f"preparation receipt bundle {index} has no bundle path")
        if status not in allowed_statuses:
            raise PerfMatrixError(
                f"preparation receipt bundle {index} has invalid status {status!r}"
            )
        if bundle.get("included_in_performance_metrics") is not False:
            raise PerfMatrixError(
                f"preparation receipt bundle {index} must be excluded from performance metrics"
            )
        build_time = bundle.get("build_time_s")
        if status == "built" and (
            not isinstance(build_time, (int, float))
            or isinstance(build_time, bool)
            or not math.isfinite(float(build_time))
            or float(build_time) < 0.0
        ):
            raise PerfMatrixError(f"preparation receipt bundle {index} has invalid build_time_s")
        key = (model, path)
        if key in seen:
            raise PerfMatrixError(
                f"preparation receipt repeats bundle {path!r} for model {model!r}"
            )
        seen.add(key)
        if key not in run_bundles:
            raise PerfMatrixError(
                f"preparation receipt bundle {path!r} for model {model!r} "
                "does not match a bundle used by the performance run"
            )
        validated.append(deepcopy(dict(bundle)))

    results["bundle_preparation"] = {
        "schema_version": PREPARATION_SCHEMA,
        "scope": "test_task",
        "git_commit": receipt["git_commit"],
        "included_in_performance_metrics": False,
        "bundles": validated,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PerfMatrixError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerfMatrixError(f"{label} must contain a JSON object")
    return value


def write_report(
    output: Path,
    *,
    preparation_receipt: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Rebuild the public report for an existing performance run."""
    output = output.resolve()
    results = _read_json_object(output / "results.json", "performance results")
    if results.get("schema_version") != RESULT_SCHEMA:
        raise PerfMatrixError(f"cannot report non-{RESULT_SCHEMA} results")
    if not isinstance(results.get("cases"), list):
        raise PerfMatrixError("cannot report results without matrix entries")
    if preparation_receipt is not None:
        receipt = _read_json_object(preparation_receipt.resolve(), "preparation receipt")
        _apply_bundle_preparation_receipt(results, receipt)
    try:
        ledger = (
            ExecutionLedger.load(output, task_kind="performance")
            if (output / "ledger" / "campaign.json").is_file()
            else None
        )
    except ExecutionLedgerError as error:
        raise PerfMatrixError(str(error)) from error
    return _write_artifacts(output, results, ledger)


def _report_existing(arguments: argparse.Namespace) -> int:
    output = arguments.run_directory.resolve()
    write_report(
        output,
        preparation_receipt=arguments.preparation_receipt,
    )
    print(f"Results: {output / 'results.json'}")
    print(f"Report data: {output / 'report.json'}")
    print(f"Report: {output / 'report.html'}")
    return 0


def _existing_ancestor(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _environment_preflight(environment: Mapping[str, Any], options: RunOptions) -> dict[str, Any]:
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
    if options.storage_root is not None:
        storage_root = options.storage_root.expanduser().resolve()
        if not storage_root.is_dir():
            raise PerfMatrixError(f"storage root does not exist: {storage_root}")
        for label, path in paths.items():
            resolved = path.expanduser().resolve()
            try:
                resolved.relative_to(storage_root)
            except ValueError as exc:
                raise PerfMatrixError(
                    f"{label} must stay below storage root {storage_root}: {resolved}"
                ) from exc
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
    insufficient = [value for value in filesystems.values() if value["free_bytes"] < minimum_bytes]
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


def _new_run_directory(results_root: Path, suite: Mapping[str, Any]) -> tuple[str, Path]:
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


def _selected_rows(results: Mapping[str, Any], selected_ids: set[str]) -> list[Mapping[str, Any]]:
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
    execution_revision = _execution_revision()
    if results.get("git_commit") != execution_revision:
        raise PerfMatrixError(
            "performance execution revision changed after campaign initialization"
        )
    ledger = _open_perf_ledger(options.output, selected, results)
    ledger.recover_interrupted()
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
    for case in selected:
        case_id = str(case["id"])
        receipt = ledger.receipt(case_id)
        if receipt["state"] == "terminal" or not _should_skip(rows[case_id]):
            continue
        try:
            prior_revision = case_evidence.exact_source_revision(
                rows[case_id].get("source_revision"),
                f"Performance case {case_id} source revision",
            )
        except case_evidence.CaseEvidenceError:
            continue
        if prior_revision != execution_revision:
            continue
        ledger.begin(
            case_id,
            stage="legacy-import",
            evidence={
                "commands": deepcopy(dict(rows[case_id].get("commands", {}))),
                "logs": deepcopy(list(rows[case_id].get("logs", []))),
            },
        )
        state, result = _perf_state_and_result(rows[case_id])
        if state != "terminal" or result is None:
            raise PerfMatrixError(f"cannot import incomplete result for {case_id!r}")
        ledger.finish(
            case_id,
            result=result,
            payload=case_evidence.stamp_case(rows[case_id], prior_revision),
        )
    _sync_perf_results_from_ledger(results, ledger)
    rows = _result_rows(results)
    work_root = options.scratch_root / options.output.name
    work_root.mkdir(parents=True, exist_ok=True)
    for case in selected:
        case_id = str(case["id"])
        failure = preflight_failures.get(case_id)
        receipt = ledger.receipt(case_id)
        if failure is None or receipt["state"] == "terminal":
            continue
        failure_row = case_evidence.stamp_case(
            _resolution_failure_row(case, failure), execution_revision
        )
        ledger.begin(
            case_id,
            stage="preflight",
            evidence={"commands": deepcopy(dict(failure_row.get("commands", {})))},
        )
        ledger.finish(
            case_id,
            result="white",
            payload=failure_row,
            attempt_outcome="failed",
            evidence={"retryable": True},
        )
    _write_artifacts(options.output, results, ledger)
    for case in selected:
        case_id = str(case["id"])
        receipt = ledger.receipt(case_id)
        if receipt["state"] == "terminal":
            existing = receipt["payload"]
            action = "resume: keeping" if case_id in preflight else "preflight: recorded"
            print(f"[{case['id']}] {action} {existing['status']}", flush=True)
            continue
        if case_id not in preflight:
            existing = rows[case_id]
            print(
                f"[{case_id}] skipped: {existing.get('reason', 'preflight failed')}",
                flush=True,
            )
            continue
        case_attempt = len(receipt["attempts"]) + 1
        case_work = work_root / _slug(case_id) / f"attempt-{case_attempt}"
        attempt_evidence = _perf_attempt_evidence(
            options.output,
            case_id,
            case_attempt,
            preflight[case_id][2],
            _case_command_environment(options, case_work),
        )
        ledger.begin(
            case_id,
            stage=("candidate" if _stable_even(case_id) else "reference"),
            evidence=attempt_evidence,
        )
        _write_artifacts(options.output, results, ledger)

        def publish_progress(stage: str, evidence: Mapping[str, Any]) -> None:
            ledger.update_stage(case_id, stage, evidence=evidence)
            _write_artifacts(options.output, results, ledger)

        row = _run_one(
            case,
            options,
            work_root,
            preflight[case_id],
            case_attempt=case_attempt,
            progress=publish_progress,
        )
        if not row.get("logs"):
            row["logs"] = deepcopy(attempt_evidence["logs"])
        passed = row.get("status") in {"green", "yellow", "red"}
        bundle_cleanup = _cleanup_managed_bundle(
            preflight[case_id][0],
            options,
            passed=passed,
        )
        work_cleanup = _cleanup_entry_work(
            work_root / _slug(case_id),
            options,
            passed=passed,
        )
        row["resource_cleanup"] = {
            "bundle": bundle_cleanup,
            "entry_work": work_cleanup,
        }
        cleanup_failures = [
            str(item.get("error", "unknown error"))
            for item in (bundle_cleanup, work_cleanup)
            if item["status"] == "failed"
        ]
        if cleanup_failures:
            row.setdefault("warnings", []).append(
                {
                    "stage": "cleanup",
                    "code": "resource_cleanup_failed",
                    "message": "; ".join(cleanup_failures),
                }
            )
        state, result = _perf_state_and_result(row)
        if state != "terminal" or result is None:
            raise PerfMatrixError(f"Performance case {case_id!r} did not finish")
        row = case_evidence.stamp_case(row, execution_revision)
        ledger.finish(
            case_id,
            result=result,
            payload=row,
            attempt_outcome=(
                "timed_out"
                if any(
                    command.get("exit_code") == 124
                    for command in row.get("commands", {}).values()
                    if isinstance(command, Mapping)
                )
                else "failed"
                if row.get("status") == "failed"
                else "completed"
            ),
            evidence={
                "return_codes": {
                    name: command.get("exit_code")
                    for name, command in row.get("commands", {}).items()
                    if isinstance(command, Mapping) and "exit_code" in command
                },
                "retryable": row.get("status") == "failed",
            },
        )
        _write_artifacts(options.output, results, ledger)
    selected_ids = {str(case["id"]) for case in selected}
    results["finished_at"] = _now()
    results["status"] = _final_status(_selected_rows(results, selected_ids))
    _finish_execution_attempt(results)
    _, _, final_report = _write_artifacts(options.output, results, ledger)
    try:
        work_root.rmdir()
    except OSError:
        pass
    try:
        options.scratch_root.rmdir()
    except OSError:
        pass
    print(f"Results: {options.output / 'results.json'}")
    print(f"Report data: {options.output / 'report.json'}")
    print(f"Report: {options.output / 'report.html'}")
    source_identity_failed = not final_report["model_source_identity"]["consistent"]
    if source_identity_failed:
        print(
            "Performance source identity is mixed, missing, or incomplete; "
            "re-run the affected model with --invalidate-model",
            file=sys.stderr,
        )
    return 1 if _campaign_failed(results, selected_ids) or source_identity_failed else 0


def _load_suite_request(
    arguments: argparse.Namespace,
) -> tuple[
    performance_catalog.PerformanceSuite,
    list[dict[str, Any]],
    dict[str, Any],
]:
    suite = _load_performance_suite(arguments.suite.resolve())
    selection_modes = sum(
        bool(value)
        for value in (
            arguments.entry,
            arguments.model,
            arguments.model_selection,
        )
    )
    if selection_modes > 1:
        raise PerfMatrixError(
            "choose exactly one selection: --entry, --model, or --model-selection"
        )
    requested_families = (
        model_selection.load_model_selection(arguments.model_selection)
        if arguments.model_selection
        else ()
    )
    try:
        selected = suite.select(
            entries=arguments.entry,
            models=tuple(arguments.model),
            families=requested_families,
        )
    except performance_catalog.PerformanceSuiteError as exc:
        raise PerfMatrixError(str(exc)) from exc
    if not selected:
        raise PerfMatrixError("selection contains no entries")
    environment = _read_environment(arguments.environment)
    return suite, selected, environment


def _check(arguments: argparse.Namespace) -> int:
    _, selected, environment = _load_suite_request(arguments)
    options = _run_options(environment, Path(str(environment["storage"]["results_root"])))
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


def _prepare(arguments: argparse.Namespace) -> int:
    _, selected, environment = _load_suite_request(arguments)
    options = _run_options(environment, Path(str(environment["storage"]["results_root"])))
    _environment_preflight(environment, options)
    _preflight_worker(options)
    _preflight, failures = _preflight_candidates(selected, options)
    if failures:
        for case_id, failure in sorted(failures.items()):
            print(f"[{case_id}] {failure['stage']}: {failure['reason']}", file=sys.stderr)
        return 1

    revision = str(_git_commit() or "").strip().lower()
    if EXACT_SOURCE_REVISION_PATTERN.fullmatch(revision) is None:
        raise PerfMatrixError("bundle preparation requires an exact 40-character source revision")
    bundles: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for case in selected:
        case_id = str(case["id"])
        print(f"[{case_id}] bundle preparation", flush=True)
        command = _run_command(
            [*_candidate_base_argv(case, options), "--prepare-only"],
            _command_environment(),
            options.timeout_seconds,
        )
        if command["exit_code"] != 0:
            raise PerfMatrixError(
                f"bundle preparation failed for {case_id}: {command['stderr_tail']}"
            )
        try:
            receipt = json.loads(command["stdout"])
        except json.JSONDecodeError as exc:
            raise PerfMatrixError(
                f"trtmc-bench returned invalid bundle preparation JSON for {case_id}"
            ) from exc
        records = receipt.get("bundles") if isinstance(receipt, Mapping) else None
        if not isinstance(records, list) or not records:
            raise PerfMatrixError(f"trtmc-bench prepared no bundle for {case_id}")
        for record in records:
            if not isinstance(record, Mapping):
                raise PerfMatrixError(f"trtmc-bench returned invalid bundle evidence for {case_id}")
            if record.get("source_revision") != revision:
                raise PerfMatrixError(
                    f"bundle for {case_id} does not match source revision {revision}"
                )
            key = (str(record.get("model", "")), str(record.get("bundle", "")))
            if key not in seen:
                seen.add(key)
                bundles.append(dict(record))
    _write_json(
        arguments.output.resolve(),
        {
            "schema_version": PREPARATION_SCHEMA,
            "scope": "test_task",
            "git_commit": revision,
            "included_in_performance_metrics": False,
            "bundles": bundles,
        },
    )
    print(f"Bundle preparation: {arguments.output.resolve()}")
    return 0


def _run_new(arguments: argparse.Namespace) -> int:
    suite, selected, environment = _load_suite_request(arguments)
    results_root = Path(str(environment["storage"]["results_root"]))
    preliminary_options = _run_options(
        environment,
        results_root,
        verbose=arguments.verbose,
        require_prebuilt_bundles=arguments.no_build,
    )
    storage = _environment_preflight(environment, preliminary_options)
    worker = _preflight_worker(preliminary_options)
    run_id, output = _new_run_directory(results_root, suite.definition)
    options = _run_options(
        environment,
        output,
        verbose=arguments.verbose,
        require_prebuilt_bundles=arguments.no_build,
    )
    results = _initial_results(suite, selected, environment)
    results["run_id"] = run_id
    ledger = _open_perf_ledger(output, selected, results)
    _write_artifacts(output, results, ledger)
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
    suite = _load_performance_suite(suite_path)
    selected_ids = results.get("selected_entry_ids")
    if not isinstance(selected_ids, list) or not all(
        isinstance(value, str) for value in selected_ids
    ):
        raise PerfMatrixError("cannot resume without selected entry ids")
    try:
        selected = suite.select(entries=selected_ids)
    except performance_catalog.PerformanceSuiteError as exc:
        raise PerfMatrixError(str(exc)) from exc
    environment = _read_environment(environment_path)
    if environment != environment_record:
        raise PerfMatrixError("cannot resume because the resolved environment values changed")
    execution_revision = _execution_revision()
    invalidated_models = set(getattr(arguments, "invalidate_model", []))
    selected_models = {str(case["model"]) for case in selected}
    unknown_models = sorted(invalidated_models - selected_models)
    if unknown_models:
        raise PerfMatrixError(
            "cannot invalidate models outside this run: " + ", ".join(unknown_models)
        )
    _record_execution_attempt(results, execution_revision)
    options = _run_options(
        environment,
        output,
        verbose=arguments.verbose,
        require_prebuilt_bundles=arguments.no_build,
    )
    ledger = _open_perf_ledger(output, selected, results)
    ledger.recover_interrupted()
    ledger.reopen_retryable()
    invalidated_case_ids = [
        str(case["id"])
        for case in selected
        if str(case["model"]) in invalidated_models
        and ledger.receipt(str(case["id"]))["state"] == "terminal"
    ]
    if invalidated_case_ids:
        ledger.reopen_cases(
            invalidated_case_ids,
            reason="model_invalidation",
            evidence={
                "models": sorted(invalidated_models),
                "requested_revision": execution_revision,
            },
        )
    if invalidated_models:
        rows = _result_rows(results)
        for case in selected:
            if str(case["model"]) in invalidated_models:
                rows[str(case["id"])] = _pending_perf_row(case)
    _write_artifacts(output, results, ledger)
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
        if arguments.command == "prepare":
            return _prepare(arguments)
        if arguments.command == "run":
            return _run_new(arguments)
        if arguments.command == "resume":
            return _resume(arguments)
        if arguments.command == "report":
            return _report_existing(arguments)
        raise PerfMatrixError(f"unsupported command: {arguments.command}")
    except (PerfMatrixError, case_evidence.CaseEvidenceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

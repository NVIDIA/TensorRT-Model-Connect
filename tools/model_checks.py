#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve canonical models into independent Accuracy and Perf bindings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = REPOSITORY / "python"
for source_root in (REPOSITORY, PYTHON_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from tools import campaign_shards, case_evidence  # noqa: E402
from tools import model_ci  # noqa: E402
from tools import model_selection  # noqa: E402
from tools import perf_matrix  # noqa: E402
from tools import trtmc_validate  # noqa: E402
from tools.execution_ledger import ExecutionLedger, ExecutionLedgerError  # noqa: E402
from tools.ci.context import CiContext  # noqa: E402
from tools.ci.model_reference_cache import (  # noqa: E402
    ModelReferenceCacheWarmer,
    ModelReferenceContract,
    parse_model_reference_contract,
)
from tools.ci.process import CiError  # noqa: E402
from tools.performance import catalog as performance_catalog  # noqa: E402
from tools.validation import catalog as validation_catalog  # noqa: E402
from tensorrt_model_connect.python_profiles import (  # noqa: E402
    PREBUILT_ONLY_ENV,
    PROFILE_ROOT_ENV,
)


PLATFORM_SCHEMA = "trtmc.model-check-platform/v1"
ENVIRONMENT_SCHEMA = "trtmc.model-check-environment/v1"
DEFAULT_PLATFORM_ROOT = REPOSITORY / "tests" / "model_checks" / "platforms"
DEFAULT_ENVIRONMENT_ROOT = REPOSITORY / "tests" / "model_checks" / "environments"
DEFAULT_PERF_SUITE = REPOSITORY / "benchmarks" / "performance" / "release.yaml"
TASKS = ("accuracy", "perf")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
EXACT_GIT_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"TRTMC_[A-Z0-9_]+")
ENVIRONMENT_REFERENCE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
MANAGED_TASK_ENVIRONMENT_VARIABLES = frozenset(
    {
        PROFILE_ROOT_ENV,
        PREBUILT_ONLY_ENV,
        "TRTMC_PERF_SOURCE_REVISION",
        trtmc_validate.REFERENCE_SOURCES_PREBUILT_ONLY_ENV,
        "TRTMC_VALIDATION_SOURCE_REVISION",
    }
)


class ModelCheckError(ValueError):
    """A model-check selection or platform profile is invalid."""


def _add_selection_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--platform", required=True, help="platform ID or profile YAML")
    selection = command.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", action="append", default=[])
    selection.add_argument("--model-selection", type=Path)
    selection.add_argument("--all", action="store_true")
    command.add_argument("--task", action="append", choices=TASKS, default=[])
    accuracy = command.add_mutually_exclusive_group()
    accuracy.add_argument("--accuracy-suite", action="append", default=[])
    accuracy.add_argument(
        "--accuracy-binding",
        action="append",
        default=[],
        metavar="MODEL=SUITE",
        help="exact Accuracy binding; repeatable",
    )
    command.add_argument("--revision", default="HEAD")
    command.add_argument("--catalog", type=Path, default=trtmc_validate.DEFAULT_CATALOG)
    command.add_argument("--suites", type=Path, default=trtmc_validate.DEFAULT_SUITES)
    command.add_argument("--models-dir", type=Path, default=trtmc_validate.DEFAULT_MODELS)
    command.add_argument("--perf-suite", type=Path, default=DEFAULT_PERF_SUITE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="show task bindings without running them")
    _add_selection_arguments(check)
    check.add_argument("--json", action="store_true", help="print the resolved JSON")
    run = commands.add_parser("run", help="run resolved task bindings locally")
    _add_selection_arguments(run)
    run.add_argument(
        "--environment",
        help="execution environment ID or YAML; defaults to the platform ID",
    )
    run.add_argument(
        "--debug",
        action="store_true",
        help="allow missing dependencies to be prepared during this non-qualification run",
    )
    run.add_argument("--run-id", help="stable output directory name")
    run.add_argument("--dry-run", action="store_true", help="write and print commands only")
    run.add_argument(
        "--verbose",
        action="store_true",
        help="print full child commands and enable detailed child-runner output",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume the existing --run-id after verifying its request",
    )
    run.add_argument(
        "--invalidate-model",
        action="append",
        default=[],
        help=(
            "with --resume, re-run every selected Accuracy and Perf case for this "
            "model; repeatable"
        ),
    )
    run.add_argument(
        "--hf-cache-seed-dir",
        type=Path,
        help="existing HF_HOME tree to hard-link into isolated Accuracy caches",
    )
    run.add_argument(
        "--shard",
        metavar="INDEX/COUNT",
        help="run one zero-based deterministic shard without enabling CI orchestration",
    )
    consolidate = commands.add_parser(
        "consolidate",
        help="materialize global reports from a sharded model-check campaign",
    )
    consolidate.add_argument("run_root", type=Path)
    consolidate.add_argument(
        "--watch",
        action="store_true",
        help="continue refreshing until every selected case is terminal",
    )
    consolidate.add_argument(
        "--interval-seconds",
        type=float,
        default=15.0,
        help="refresh interval with --watch (default: 15)",
    )
    return parser


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelCheckError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelCheckError(f"{label} must contain a YAML object: {path}")
    return payload


def _platform_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate.resolve()
    return (DEFAULT_PLATFORM_ROOT / f"{value}.yaml").resolve()


def load_platform(value: str) -> dict[str, Any]:
    path = _platform_path(value)
    profile = _read_yaml(path, "platform profile")
    if profile.get("schema_version") != PLATFORM_SCHEMA:
        raise ModelCheckError(f"platform profile schema_version must be {PLATFORM_SCHEMA}: {path}")
    platform_id = profile.get("id")
    if not isinstance(platform_id, str) or not platform_id.strip():
        raise ModelCheckError(f"platform profile needs a non-empty id: {path}")
    execution = profile.get("execution")
    if not isinstance(execution, Mapping):
        raise ModelCheckError(f"platform profile needs execution settings: {path}")
    task_order = execution.get("task_order")
    if (
        not isinstance(task_order, list)
        or not task_order
        or any(task not in TASKS for task in task_order)
        or len(task_order) != len(set(task_order))
    ):
        raise ModelCheckError(f"platform task_order is invalid: {path}")
    if not isinstance(execution.get("serial_tasks"), bool):
        raise ModelCheckError(f"platform serial_tasks must be boolean: {path}")
    storage = profile.get("storage", {})
    if not isinstance(storage, Mapping):
        raise ModelCheckError(f"platform storage must be an object: {path}")
    root_prefix = storage.get("root_prefix")
    if root_prefix is not None and (
        not isinstance(root_prefix, str) or not Path(root_prefix).is_absolute()
    ):
        raise ModelCheckError(f"platform storage.root_prefix must be an absolute path: {path}")
    required_device = storage.get("device")
    if required_device is not None and (
        not isinstance(required_device, str) or not Path(required_device).is_absolute()
    ):
        raise ModelCheckError(f"platform storage.device must be an absolute path: {path}")
    legacy_exclusions = sorted({"unsupported", "excluded"}.intersection(profile))
    if legacy_exclusions:
        raise ModelCheckError(
            "platform binding-level exclusions are no longer supported; use "
            f"excluded_models instead of {', '.join(legacy_exclusions)}: {path}"
        )
    excluded_models = profile.get("excluded_models", [])
    if not isinstance(excluded_models, list) or any(
        not isinstance(model, str) or not model.strip() for model in excluded_models
    ):
        raise ModelCheckError(f"platform excluded_models must be a list of models: {path}")
    if len(excluded_models) != len(set(excluded_models)):
        raise ModelCheckError(f"platform excluded_models contains duplicates: {path}")
    return {**profile, "source": str(path)}


def audit_platform_exclusions(
    platform: Mapping[str, Any],
    *,
    accuracy_catalog: Mapping[str, Any],
    perf_cases: Sequence[Mapping[str, Any]],
) -> None:
    known_models = set(accuracy_catalog["models"])
    known_models.update(str(case["model"]) for case in perf_cases)
    unknown = sorted(set(platform.get("excluded_models", [])) - known_models)
    if unknown:
        raise ModelCheckError(
            "platform excluded_models names unknown models: " + ", ".join(unknown)
        )


def _environment_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate.resolve()
    return (DEFAULT_ENVIRONMENT_ROOT / f"{value}.yaml").resolve()


def _expand_environment_value(value: str, field: str) -> str:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        configured = os.environ.get(name)
        if configured is None or not configured.strip():
            missing.append(name)
            return ""
        return configured

    expanded = ENVIRONMENT_REFERENCE_PATTERN.sub(replace, value)
    if missing:
        raise ModelCheckError(f"environment {field} requires: {', '.join(sorted(set(missing)))}")
    return expanded


def _expand_environment(value: Any, field: str) -> Any:
    if isinstance(value, str):
        return _expand_environment_value(value, field)
    if isinstance(value, list):
        return [_expand_environment(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {
            str(key): _expand_environment(item, f"{field}.{key}") for key, item in value.items()
        }
    return value


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY / path).resolve()


def _runner_executable(value: str, field: str) -> str:
    expanded = _expand_environment_value(value, field)
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return str(path.absolute())
    if path.parent != Path("."):
        return str((REPOSITORY / path).absolute())
    return expanded


def load_execution_environment(value: str, *, platform_id: str) -> dict[str, Any]:
    path = _environment_path(value)
    raw = _read_yaml(path, "model-check environment")
    environment = _expand_environment(raw, "model-check environment")
    if environment.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise ModelCheckError(
            f"model-check environment schema_version must be {ENVIRONMENT_SCHEMA}: {path}"
        )
    if environment.get("id") != platform_id:
        raise ModelCheckError(
            f"environment platform {environment.get('id')!r} does not match {platform_id!r}"
        )
    library_dirs = environment.get("library_dirs", [])
    if not isinstance(library_dirs, list) or any(
        not isinstance(path, str) or not path for path in library_dirs
    ):
        raise ModelCheckError("model-check environment library_dirs must be a list of paths")
    executable_dirs = environment.get("executable_dirs", [])
    if not isinstance(executable_dirs, list) or any(
        not isinstance(path, str) or not path for path in executable_dirs
    ):
        raise ModelCheckError("model-check environment executable_dirs must be a list of paths")
    python_dirs = environment.get("python_dirs", [])
    if not isinstance(python_dirs, list) or any(
        not isinstance(path, str) or not path for path in python_dirs
    ):
        raise ModelCheckError("model-check environment python_dirs must be a list of paths")
    environment_variables = environment.get("environment_variables", {})
    if not isinstance(environment_variables, Mapping) or any(
        not isinstance(name, str)
        or ENVIRONMENT_VARIABLE_PATTERN.fullmatch(name) is None
        or name in MANAGED_TASK_ENVIRONMENT_VARIABLES
        or not isinstance(value, str)
        or not value
        for name, value in environment_variables.items()
    ):
        raise ModelCheckError(
            "model-check environment environment_variables must map unmanaged "
            "TRTMC_* names to non-empty strings"
        )
    storage = environment.get("storage")
    if not isinstance(storage, Mapping):
        raise ModelCheckError("model-check environment needs storage settings")
    for field in ("root", "results_root"):
        if not isinstance(storage.get(field), str) or not storage[field]:
            raise ModelCheckError(f"model-check environment storage.{field} is required")
    task_config = environment.get("tasks")
    if not isinstance(task_config, Mapping):
        raise ModelCheckError("model-check environment needs task settings")
    accuracy = task_config.get("accuracy")
    perf = task_config.get("perf")
    if not isinstance(accuracy, Mapping) or not isinstance(perf, Mapping):
        raise ModelCheckError("model-check environment needs Accuracy and Perf settings")
    for task, config in (("accuracy", accuracy), ("perf", perf)):
        if not isinstance(config.get("runner_python"), str) or not config["runner_python"]:
            raise ModelCheckError(f"model-check environment {task}.runner_python is required")
    options = accuracy.get("options", {})
    if not isinstance(options, Mapping):
        raise ModelCheckError("model-check environment accuracy.options must be an object")
    forbidden = {
        "all",
        "binding",
        "dry-run",
        "model",
        "model-selection",
        "model-work-dir",
        "output",
        "storage-root",
        "workload",
    }.intersection(str(name).replace("_", "-") for name in options)
    if forbidden:
        raise ModelCheckError(
            "accuracy.options cannot control selection or output: " + ", ".join(sorted(forbidden))
        )
    for field in ("suite", "environment"):
        if not isinstance(perf.get(field), str) or not perf[field]:
            raise ModelCheckError(f"model-check environment perf.{field} is required")
    storage_root = _repo_path(str(storage["root"]))
    python_profiles_root = storage.get("python_profiles_root")
    if python_profiles_root is None:
        resolved_python_profiles_root = storage_root / "python-profiles"
    elif not isinstance(python_profiles_root, str) or not python_profiles_root:
        raise ModelCheckError(
            "model-check environment storage.python_profiles_root must be a non-empty path"
        )
    else:
        resolved_python_profiles_root = _repo_path(python_profiles_root)
    model_reference_cache_root = storage.get("model_reference_cache_root")
    if model_reference_cache_root is None:
        resolved_model_reference_cache_root = storage_root / "references" / "model-sources"
    elif not isinstance(model_reference_cache_root, str) or not model_reference_cache_root:
        raise ModelCheckError(
            "model-check environment storage.model_reference_cache_root must be a non-empty path"
        )
    else:
        resolved_model_reference_cache_root = _repo_path(model_reference_cache_root)
    return {
        **environment,
        "source": str(path),
        "library_dirs": [str(_repo_path(path)) for path in library_dirs],
        "executable_dirs": [str(_repo_path(path)) for path in executable_dirs],
        "python_dirs": [str(_repo_path(path)) for path in python_dirs],
        "environment_variables": dict(environment_variables),
        "storage": {
            **storage,
            "root": str(storage_root),
            "results_root": str(_repo_path(str(storage["results_root"]))),
            "python_profiles_root": str(resolved_python_profiles_root),
            "model_reference_cache_root": str(resolved_model_reference_cache_root),
        },
        "tasks": {
            "accuracy": {
                **accuracy,
                "runner_python": _runner_executable(
                    str(accuracy["runner_python"]),
                    "model-check environment tasks.accuracy.runner_python",
                ),
            },
            "perf": {
                **perf,
                "runner_python": _runner_executable(
                    str(perf["runner_python"]),
                    "model-check environment tasks.perf.runner_python",
                ),
                "suite": str(_repo_path(str(perf["suite"]))),
                "environment": str(_repo_path(str(perf["environment"]))),
            },
        },
    }


def _task_environment(
    environment: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    *,
    source_revision: str = "",
    allow_dependency_creation: bool = True,
) -> dict[str, str]:
    """Build a child environment rooted in this worktree and its managed caches."""
    child = os.environ.copy()
    child[PROFILE_ROOT_ENV] = str(environment["storage"]["python_profiles_root"])
    if allow_dependency_creation:
        child.pop(PREBUILT_ONLY_ENV, None)
        child.pop(trtmc_validate.REFERENCE_SOURCES_PREBUILT_ONLY_ENV, None)
    else:
        child[PREBUILT_ONLY_ENV] = "1"
        child[trtmc_validate.REFERENCE_SOURCES_PREBUILT_ONLY_ENV] = "1"
    library_dirs = [str(path) for path in environment.get("library_dirs", [])]
    missing_library_dirs = [path for path in library_dirs if not Path(path).is_dir()]
    if missing_library_dirs:
        raise ModelCheckError(
            "model-check runtime library directory does not exist: "
            + ", ".join(missing_library_dirs)
        )
    inherited_library_path = child.get("LD_LIBRARY_PATH", "")
    if inherited_library_path:
        library_dirs.append(inherited_library_path)
    if library_dirs:
        child["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    executable_dirs = [str(path) for path in environment.get("executable_dirs", [])]
    missing_executable_dirs = [path for path in executable_dirs if not Path(path).is_dir()]
    if missing_executable_dirs:
        raise ModelCheckError(
            "model-check executable directory does not exist: " + ", ".join(missing_executable_dirs)
        )
    inherited_path = child.get("PATH", "")
    if inherited_path:
        executable_dirs.append(inherited_path)
    if executable_dirs:
        child["PATH"] = os.pathsep.join(executable_dirs)
    configured_python_dirs = [str(path) for path in environment.get("python_dirs", [])]
    missing_python_dirs = [path for path in configured_python_dirs if not Path(path).is_dir()]
    if missing_python_dirs:
        raise ModelCheckError(
            "model-check Python runtime directory does not exist: " + ", ".join(missing_python_dirs)
        )
    python_dirs = [str(PYTHON_SOURCE), str(REPOSITORY), *configured_python_dirs]
    inherited_python_path = child.get("PYTHONPATH", "")
    if inherited_python_path:
        python_dirs.extend(inherited_python_path.split(os.pathsep))
    child["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, python_dirs)))
    child.update(
        {
            str(name): str(value)
            for name, value in environment.get("environment_variables", {}).items()
        }
    )
    if EXACT_GIT_REVISION_PATTERN.fullmatch(source_revision):
        revision = source_revision.lower()
        child["TRTMC_VALIDATION_SOURCE_REVISION"] = revision
        child["TRTMC_PERF_SOURCE_REVISION"] = revision
        child["TRTMC_ENGINE_BUILD_REVISION"] = revision
    child.update(overrides or {})
    return child


def _selected_tasks(profile: Mapping[str, Any], requested: Iterable[str]) -> tuple[str, ...]:
    task_order = tuple(profile["execution"]["task_order"])
    requested = tuple(dict.fromkeys(requested))
    return tuple(task for task in task_order if task in requested) if requested else task_order


def model_profiles_for_owners(
    owners: Iterable[str],
    *,
    tasks: Sequence[str],
    accuracy_models: Mapping[str, Mapping[str, Any]],
    accuracy_catalog: Mapping[str, Any],
    perf_cases: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Expand model_ci owner IDs into the selected tasks' model profiles."""

    selected_owners = model_selection.normalize_models(owners)
    profiles: list[str] = []
    missing: list[str] = []
    for owner in selected_owners:
        matched: set[str] = set()
        if "accuracy" in tasks:
            matched.update(
                model
                for model, record in accuracy_models.items()
                if model in accuracy_catalog["models"] and str(record.get("family", "")) == owner
            )
        if "perf" in tasks:
            matched.update(
                str(case["model"]) for case in perf_cases if str(case["family"]) == owner
            )
        if not matched:
            missing.append(owner)
        profiles.extend(sorted(matched))
    if missing:
        raise ModelCheckError("model owners have no selected task profiles: " + ", ".join(missing))
    return model_selection.normalize_models(profiles)


def _platform_exclusion_reason(profile: Mapping[str, Any], model: str) -> str:
    if model not in profile.get("excluded_models", []):
        return ""
    return f"Model is excluded from platform {profile['id']}"


def _projection_status(bindings: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(binding["status"]) for binding in bindings}
    if len(statuses) == 1:
        status = next(iter(statuses))
        if status == "excluded":
            return status
    return "configured"


def _accuracy_projection(
    model: str,
    *,
    catalog: Mapping[str, Any],
    workloads: Sequence[str],
    missing_is_not_applicable: bool,
    platform: Mapping[str, Any],
) -> dict[str, Any]:
    if model not in catalog["models"]:
        return {
            "status": ("not_applicable" if missing_is_not_applicable else "unconfigured"),
            "reason": (
                "model belongs only to another selected task's complete matrix"
                if missing_is_not_applicable
                else "model has no Accuracy catalog binding"
            ),
            "bindings": [],
        }
    try:
        bindings = trtmc_validate.resolve_bindings(
            catalog,
            [model],
            workloads=workloads,
        )
    except trtmc_validate.ValidationError as exc:
        return {"status": "unconfigured", "reason": str(exc), "bindings": []}
    projected = []
    for binding in bindings:
        binding_id = binding.workload or "not-compared"
        reason = _platform_exclusion_reason(platform, model)
        status = "excluded" if reason else "configured"
        projected.append(
            {
                "id": f"accuracy:{model}:{binding_id}",
                "model": model,
                "workload": binding.workload,
                "status": status,
                **({"reason": reason} if reason else {}),
            }
        )
    status = _projection_status(projected)
    return {"status": status, "bindings": projected}


def _perf_projection(
    model: str,
    *,
    cases: Sequence[Mapping[str, Any]],
    exclusions: Mapping[str, str],
    platform: Mapping[str, Any],
) -> dict[str, Any]:
    matched = sorted(
        (case for case in cases if str(case["model"]) == model),
        key=lambda case: str(case["id"]),
    )
    if not matched:
        if model in exclusions:
            return {
                "status": "excluded",
                "reason": exclusions[model],
                "bindings": [],
            }
        if performance_catalog.is_l0_profile(model):
            return {
                "status": "not_applicable",
                "reason": "L0 profiles are excluded from the release performance matrix",
                "bindings": [],
            }
        return {
            "status": "unconfigured",
            "reason": "model has no Perf release entry",
            "bindings": [],
        }
    projected = []
    for case in matched:
        entry_id = str(case["id"])
        reason = _platform_exclusion_reason(platform, model)
        status = "excluded" if reason else "configured"
        projected.append(
            {
                "id": f"perf:{model}:{entry_id}",
                "model": model,
                "entry": entry_id,
                "family": case.get("family"),
                "operation": case.get("operation"),
                "status": status,
                **({"reason": reason} if reason else {}),
            }
        )
    status = _projection_status(projected)
    return {"status": status, "bindings": projected}


def resolve_plan(
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    platform: Mapping[str, Any],
    accuracy_catalog: Mapping[str, Any],
    accuracy_workloads: Sequence[str],
    accuracy_bindings: Mapping[str, Sequence[str]],
    perf_cases: Sequence[Mapping[str, Any]],
    perf_exclusions: Mapping[str, str],
    complete_task_matrices: bool = False,
) -> dict[str, Any]:
    results = []
    blocker_count = 0
    for model in model_selection.normalize_models(models):
        record: dict[str, Any] = {"model": model, "tasks": {}}
        if "accuracy" in tasks:
            record["tasks"]["accuracy"] = _accuracy_projection(
                model,
                catalog=accuracy_catalog,
                workloads=accuracy_bindings.get(model, accuracy_workloads),
                missing_is_not_applicable=complete_task_matrices,
                platform=platform,
            )
        if "perf" in tasks:
            record["tasks"]["perf"] = _perf_projection(
                model,
                cases=perf_cases,
                exclusions=perf_exclusions,
                platform=platform,
            )
        blocker_count += sum(task["status"] == "unconfigured" for task in record["tasks"].values())
        results.append(record)
    bindings = [
        binding
        for record in results
        for task in record["tasks"].values()
        for binding in task["bindings"]
    ]
    return {
        "schema_version": "trtmc.model-check-selection/v1",
        "platform": platform["id"],
        "platform_source": platform["source"],
        "execution": {
            "task_order": list(tasks),
            "serial_tasks": bool(platform["execution"]["serial_tasks"]),
        },
        "models": results,
        "summary": {
            "model_count": len(results),
            "binding_count": len(bindings),
            "configured_binding_count": sum(
                binding["status"] == "configured" for binding in bindings
            ),
            "excluded_binding_count": sum(binding["status"] == "excluded" for binding in bindings),
            "blocker_count": blocker_count,
        },
    }


def _render(plan: Mapping[str, Any]) -> str:
    execution = plan["execution"]
    lines = [
        f"Platform: {plan['platform']}",
        "Tasks: " + " -> ".join(execution["task_order"]),
        "Execution: " + ("serial" if execution["serial_tasks"] else "independent"),
    ]
    for record in plan["models"]:
        lines.append(f"\n{record['model']}")
        for task_name in execution["task_order"]:
            task = record["tasks"][task_name]
            lines.append(f"  {task_name}: {task['status']}")
            if task.get("reason"):
                lines.append(f"    reason: {task['reason']}")
            for binding in task["bindings"]:
                selected = binding.get("workload") or binding.get("entry") or "not-compared"
                lines.append(f"    - {selected}: {binding['status']}")
                if binding.get("reason"):
                    lines.append(f"      reason: {binding['reason']}")
    summary = plan["summary"]
    lines.extend(
        [
            "",
            f"Summary: {summary['model_count']} models, "
            f"{summary['binding_count']} bindings, "
            f"{summary['excluded_binding_count']} excluded, "
            f"{summary['blocker_count']} blockers",
        ]
    )
    return "\n".join(lines)


def _task_label(task: str) -> str:
    return "Accuracy" if task == "accuracy" else "Perf"


def _render_run_header(
    plan: Mapping[str, Any],
    *,
    run_id: str,
    run_root: Path,
) -> str:
    models = [str(record["model"]) for record in plan["models"]]
    model_summary = ", ".join(models) if len(models) <= 5 else f"{len(models)} selected"
    order = " -> ".join(_task_label(task) for task in plan["execution"]["task_order"])
    return "\n".join(
        (
            f"Run: {run_id}",
            f"Platform: {plan['platform']}",
            f"Models: {model_summary}",
            f"Order: {order}",
            f"Bindings: {plan['summary']['configured_binding_count']}",
            f"Run root: {run_root}",
        )
    )


def _detailed_command(command: Sequence[str], *, verbose: bool) -> list[str]:
    return [*command, "--verbose"] if verbose else list(command)


def _resolve_request(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    platform = load_platform(arguments.platform)
    tasks = _selected_tasks(platform, arguments.task)
    accuracy_catalog = trtmc_validate.load_catalog(arguments.catalog)
    accuracy_suites = validation_catalog.load_suites(arguments.suites)
    accuracy_suite_map = {suite["id"]: suite for suite in accuracy_suites}
    accuracy_models = validation_catalog.load_manifest_records_by_name(arguments.models_dir)
    trtmc_validate.audit_catalog(
        accuracy_catalog,
        ready_models=trtmc_validate.ready_model_names(arguments.models_dir),
        suite_names=accuracy_suite_map,
    )
    trtmc_validate.audit_workload_compatibility(
        accuracy_catalog,
        suites=accuracy_suite_map,
        task_models=accuracy_models,
    )

    try:
        perf_suite = performance_catalog.load_suite(arguments.perf_suite)
    except performance_catalog.PerformanceSuiteError as error:
        raise ModelCheckError(str(error)) from error
    perf_cases = perf_suite.cases
    e2e_only_profiles = {
        entry.name: entry.reason
        for entry in performance_catalog.ManifestCatalog(arguments.models_dir).entries()
        if entry.status == "e2e_only"
    }
    perf_exclusions = {**perf_suite.excluded_profiles, **e2e_only_profiles}
    audit_platform_exclusions(
        platform,
        accuracy_catalog=accuracy_catalog,
        perf_cases=perf_cases,
    )

    known_profiles = set(accuracy_catalog["models"]).union(
        str(case["model"]) for case in perf_cases
    )
    if arguments.all:
        complete_profiles: set[str] = set()
        if "accuracy" in tasks:
            complete_profiles.update(accuracy_catalog["models"])
        if "perf" in tasks:
            complete_profiles.update(str(case["model"]) for case in perf_cases)
        models = tuple(sorted(complete_profiles))
    elif arguments.model_selection:
        owners = model_selection.load_model_selection(arguments.model_selection)
        known_owners = set(model_ci.discover_catalog(REPOSITORY, arguments.revision).models)
        unknown_owners = sorted(set(owners) - known_owners)
        if unknown_owners:
            raise ModelCheckError("unknown model owners: " + ", ".join(unknown_owners))
        models = model_profiles_for_owners(
            owners,
            tasks=tasks,
            accuracy_models=accuracy_models,
            accuracy_catalog=accuracy_catalog,
            perf_cases=perf_cases,
        )
    else:
        models = model_selection.normalize_models(arguments.model)
        unknown = sorted(set(models) - known_profiles)
        if unknown:
            raise ModelCheckError("unknown model profiles: " + ", ".join(unknown))

    accuracy_bindings: dict[str, list[str]] = {}
    for raw_binding in arguments.accuracy_binding:
        model, separator, workload = raw_binding.partition("=")
        model = model.strip()
        workload = workload.strip()
        if not separator or not model or not workload:
            raise ModelCheckError(
                f"invalid --accuracy-binding {raw_binding!r}; expected MODEL=SUITE"
            )
        if model not in models:
            raise ModelCheckError(f"Accuracy binding model {model!r} is not in the selected models")
        accuracy_bindings.setdefault(model, [])
        if workload not in accuracy_bindings[model]:
            accuracy_bindings[model].append(workload)
    if accuracy_bindings and "accuracy" not in tasks:
        raise ModelCheckError("--accuracy-binding requires the Accuracy task")
    missing_accuracy_bindings = sorted(set(models) - set(accuracy_bindings))
    if accuracy_bindings and missing_accuracy_bindings:
        raise ModelCheckError(
            "exact Accuracy selection needs a binding for every selected model: "
            + ", ".join(missing_accuracy_bindings)
        )

    plan = resolve_plan(
        models=models,
        tasks=tasks,
        platform=platform,
        accuracy_catalog=accuracy_catalog,
        accuracy_workloads=tuple(arguments.accuracy_suite),
        accuracy_bindings=accuracy_bindings,
        perf_cases=perf_cases,
        perf_exclusions=perf_exclusions,
        complete_task_matrices=arguments.all,
    )
    trtmc_validate.audit_binding_compatibility(
        (
            trtmc_validate.Binding(
                str(binding["model"]),
                str(binding["workload"]),
            )
            for model in plan["models"]
            for binding in model["tasks"].get("accuracy", {}).get("bindings", [])
            if binding.get("workload")
        ),
        suites=accuracy_suite_map,
        task_models=accuracy_models,
    )
    return plan, platform


def _check(arguments: argparse.Namespace) -> int:
    plan, _ = _resolve_request(arguments)
    print(json.dumps(plan, indent=2) if arguments.json else _render(plan))
    return 2 if plan["summary"]["blocker_count"] else 0


def _default_run_id(platform_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    revision = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    suffix = revision or "unknown"
    return f"model-check-{platform_id}-{timestamp}-{suffix}"


def _require_managed_path(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ModelCheckError(f"{label} must be below storage root {root}: {resolved}") from exc
    if not relative.parts:
        raise ModelCheckError(f"{label} cannot be the storage root itself: {root}")
    return resolved


def _require_platform_storage_root(
    root: Path,
    platform: Mapping[str, Any],
) -> None:
    configured = platform.get("storage", {}).get("root_prefix")
    if configured:
        required = Path(str(configured)).resolve()
        try:
            root.relative_to(required)
        except ValueError as exc:
            raise ModelCheckError(
                f"storage root must be below platform root {required}: {root}"
            ) from exc
    required_device = platform.get("storage", {}).get("device")
    if not required_device:
        return
    device_path = Path(str(required_device)).resolve()
    try:
        root_device = root.stat().st_dev
        device_stat = device_path.stat()
    except OSError as exc:
        raise ModelCheckError(
            f"cannot verify storage device {device_path} for {root}: {exc}"
        ) from exc
    expected_device = (
        device_stat.st_rdev if stat.S_ISBLK(device_stat.st_mode) else device_stat.st_dev
    )
    if root_device != expected_device:
        raise ModelCheckError(f"storage root must be on device {device_path}: {root}")


def _option_arguments(options: Mapping[str, Any]) -> list[str]:
    arguments: list[str] = []
    for name, value in options.items():
        option = "--" + str(name)
        if isinstance(value, bool):
            if value:
                arguments.append(option)
            continue
        if value is None or value == "":
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, (Mapping, list)):
                raise ModelCheckError(f"accuracy option {name} must be scalar or a scalar list")
            arguments.extend([option, str(item)])
    return arguments


def _task_bindings(plan: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    return [
        binding
        for model in plan["models"]
        for binding in model["tasks"].get(task, {}).get("bindings", [])
        if binding["status"] == "configured"
    ]


def _campaign_cases(
    plan: Mapping[str, Any],
    *,
    shard_count: int,
    accuracy_sample_limits: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for task in plan["execution"]["task_order"]:
        for binding in _task_bindings(plan, task):
            if task == "accuracy":
                case_id = f"{binding['model']}::{binding['workload']}"
                report = {
                    "model": binding["model"],
                    "workload": binding["workload"],
                    "samples": {
                        "planned": (accuracy_sample_limits or {}).get(binding["workload"]),
                        "evaluated": None,
                    },
                }
            else:
                case_id = str(binding["entry"])
                report = {
                    "model": binding["model"],
                    "family": binding.get("family"),
                    "operation": binding.get("operation"),
                    "workload": binding["entry"],
                }
            cases.append(
                {
                    "binding_id": str(binding["id"]),
                    "task": task,
                    "id": case_id,
                    "report": report,
                }
            )
    assignments = {
        binding_id: index
        for index in range(shard_count)
        for binding_id in campaign_shards.assign_cases(
            [str(case["binding_id"]) for case in cases],
            index=index,
            count=shard_count,
        )
    }
    for case in cases:
        case["shard"] = assignments[str(case["binding_id"])]
    return cases


def _shard_task_bindings(
    plan: Mapping[str, Any],
    campaign_cases: Sequence[Mapping[str, Any]],
    *,
    task: str,
    shard_index: int,
) -> list[dict[str, Any]]:
    owned = {
        str(case["binding_id"])
        for case in campaign_cases
        if case["task"] == task and case["shard"] == shard_index
    }
    return [binding for binding in _task_bindings(plan, task) if binding["id"] in owned]


def _selected_perf_reference_contracts(
    plan: Mapping[str, Any],
    models_dir: Path,
    *,
    bindings: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ModelReferenceContract, ...]:
    selected_models = {
        str(binding["model"])
        for binding in (bindings if bindings is not None else _task_bindings(plan, "perf"))
    }
    if not selected_models:
        return ()
    records = {
        str(record["name"]): record
        for record in validation_catalog.load_manifest_records(models_dir)
    }
    contracts: list[ModelReferenceContract] = []
    seen: set[str] = set()
    for model in sorted(selected_models):
        record = records.get(model)
        if record is None:
            raise ModelCheckError(f"Perf model has no owner manifest: {model}")
        family = str(record.get("family", "") or "")
        if not family:
            continue
        manifest_path = Path(str(record["manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = REPOSITORY / manifest_path
        owner_path = (
            manifest_path.parent.parent / "MODEL.toml"
            if manifest_path.parent.name == "manifests"
            else manifest_path.parent / "MODEL.toml"
        )
        try:
            owner = tomllib.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ModelCheckError(f"cannot read model owner {owner_path}: {exc}") from exc
        try:
            contract = parse_model_reference_contract(
                owner,
                family,
                owner_path,
                suite=None,
            )
        except CiError as exc:
            raise ModelCheckError(str(exc)) from exc
        if contract is None or not contract.environment_variable or contract.relative_path in seen:
            continue
        seen.add(contract.relative_path)
        contracts.append(contract)
    return tuple(contracts)


def _prepare_perf_reference_dependencies(
    contracts: Sequence[ModelReferenceContract],
    cache_root: Path,
) -> dict[str, str]:
    if not contracts:
        return {}
    cache_root = cache_root.resolve()
    environment = os.environ.copy()
    environment["TRTMC_MODEL_REFERENCE_CACHE_ROOT"] = str(cache_root)
    warmer = ModelReferenceCacheWarmer(CiContext(REPOSITORY, environment))
    child = {"TRTMC_MODEL_REFERENCE_CACHE_ROOT": str(cache_root)}
    for contract in contracts:
        try:
            checkout = warmer.warm_contract(contract)
        except CiError as exc:
            raise ModelCheckError(
                f"could not prepare {contract.family} reference source: {exc}"
            ) from exc
        if contract.environment_variable:
            child[contract.environment_variable] = str(checkout)
    return child


def _write_selected_models(plan: Mapping[str, Any], run_root: Path) -> Path:
    models = sorted(
        {
            str(binding["model"])
            for task in TASKS
            for model in plan["models"]
            for binding in model["tasks"].get(task, {}).get("bindings", [])
            if binding["status"] in {"configured", "excluded"}
        }
    )
    selection = run_root / "selected-models.txt"
    selection.write_text("".join(f"{model}\n" for model in models), encoding="utf-8")
    return selection


def _accuracy_command(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    arguments: argparse.Namespace,
    output: Path,
    *,
    bindings: Sequence[Mapping[str, Any]] | None = None,
) -> list[str] | None:
    bindings = list(bindings) if bindings is not None else _task_bindings(plan, "accuracy")
    if not bindings:
        return None
    config = environment["tasks"]["accuracy"]
    command = [
        str(config["runner_python"]),
        str(REPOSITORY / "tools" / "trtmc_validate.py"),
    ]
    for binding in bindings:
        command.extend(["--binding", f"{binding['model']}={binding['workload']}"])
    command.extend(
        [
            "--catalog",
            str(arguments.catalog),
            "--suites",
            str(arguments.suites),
            "--models-dir",
            str(arguments.models_dir),
            "--output",
            str(output),
            "--storage-root",
            str(environment["storage"]["root"]),
            "--model-work-dir",
            str(output.parent / "work" / "accuracy"),
            "--reference-source-cache-dir",
            str(environment["storage"]["model_reference_cache_root"]),
        ]
    )
    if arguments.hf_cache_seed_dir is not None:
        command.extend(["--hf-cache-seed-dir", str(arguments.hf_cache_seed_dir)])
    command.extend(_option_arguments(config.get("options", {})))
    return command


def _resolved_perf_environment(
    source: Path,
    *,
    destination: Path,
    build_dir: Path,
    storage_root: Path,
    results_root: Path,
    scratch_root: Path,
    bundle_cache: Path | None = None,
) -> Path:
    raw = _read_yaml(source, "performance environment")
    tools = raw.get("tools")
    if not isinstance(tools, dict):
        raise ModelCheckError("performance environment tools must be an object")
    storage = raw.get("storage")
    if not isinstance(storage, dict):
        raise ModelCheckError("performance environment storage must be an object")

    # The unified entry point owns one native build. Keep standalone Perf
    # environments configurable, but do not expose their internal path knobs
    # to model-check users.
    tools["trtmc_worker"] = str(build_dir / "trtmc_benchmark_worker")
    storage["storage_root"] = str(storage_root)
    storage["results_root"] = str(results_root)
    storage["scratch_root"] = str(scratch_root)
    storage["bundle_cache"] = str(bundle_cache or storage_root / "engines" / "perf")
    storage["bundle_roots"] = []
    storage["runtime_dirs"] = [str(build_dir)]
    raw = _expand_environment(raw, "performance environment")
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return destination


def _perf_command(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    resolved_environment: Path,
    *,
    bindings: Sequence[Mapping[str, Any]] | None = None,
    require_prebuilt: bool = False,
) -> list[str] | None:
    bindings = list(bindings) if bindings is not None else _task_bindings(plan, "perf")
    if not bindings:
        return None
    config = environment["tasks"]["perf"]
    command = [
        str(config["runner_python"]),
        str(REPOSITORY / "tools" / "perf_matrix.py"),
        "run",
        str(config["suite"]),
        "--environment",
        str(resolved_environment),
    ]
    for binding in bindings:
        command.extend(["--entry", str(binding["entry"])])
    if require_prebuilt:
        command.append("--no-build")
    return command


def _perf_prepare_command(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    resolved_environment: Path,
    receipt: Path,
    *,
    bindings: Sequence[Mapping[str, Any]] | None = None,
) -> list[str] | None:
    bindings = list(bindings) if bindings is not None else _task_bindings(plan, "perf")
    if not bindings:
        return None
    config = environment["tasks"]["perf"]
    command = [
        str(config["runner_python"]),
        str(REPOSITORY / "tools" / "perf_matrix.py"),
        "prepare",
        str(config["suite"]),
        "--environment",
        str(resolved_environment),
        "--output",
        str(receipt),
    ]
    for binding in bindings:
        command.extend(["--entry", str(binding["entry"])])
    return command


def _perf_resume_command(
    environment: Mapping[str, Any],
    results_root: Path,
    *,
    require_prebuilt: bool = False,
) -> list[str] | None:
    if not results_root.is_dir():
        return None
    candidates = sorted(
        path for path in results_root.iterdir() if (path / "results.json").is_file()
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ModelCheckError(
            f"cannot identify one Perf run to resume below {results_root}: found {len(candidates)}"
        )
    config = environment["tasks"]["perf"]
    command = [
        str(config["runner_python"]),
        str(REPOSITORY / "tools" / "perf_matrix.py"),
        "resume",
        str(candidates[0]),
    ]
    if require_prebuilt:
        command.append("--no-build")
    return command


def _verify_resume_request(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelCheckError(f"cannot resume without request metadata: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCheckError(f"cannot read resume request {path}: {exc}") from exc
    if not isinstance(previous, Mapping):
        raise ModelCheckError(f"resume request must contain a JSON object: {path}")
    previous_revision = previous.get("revision")
    try:
        case_evidence.exact_source_revision(
            previous_revision, label="exact recorded execution revision"
        )
    except case_evidence.CaseEvidenceError as error:
        raise ModelCheckError(str(error)) from error
    for field in (
        "schema_version",
        "run_id",
        "intent",
        "platform",
        "platform_source",
        "platform_config",
        "environment_source",
        "environment_config",
        "perf_environment_config",
        "selection",
        "commands",
    ):
        if previous.get(field) != request.get(field):
            raise ModelCheckError(f"cannot resume because the resolved {field} changed")
    if previous.get("shard") != request.get("shard"):
        raise ModelCheckError("cannot resume because the resolved shard changed")
    return dict(previous)


def _write_request(path: Path, request: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _record_execution_attempt(
    request: dict[str, Any],
    *,
    revision: str,
    source_identity: Mapping[str, Any],
) -> None:
    attempts = request.setdefault("execution_attempts", [])
    if not isinstance(attempts, list):
        raise ModelCheckError("cannot resume with invalid execution attempt history")
    attempts.append(
        {
            "revision": revision,
            "source_identity": dict(source_identity),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": bool(request.get("dry_run")),
        }
    )


def _perf_shard_output(shard_root: Path) -> Path | None:
    results_root = shard_root / "perf" / "results"
    if not results_root.is_dir():
        return None
    candidates = sorted(path.parent.parent for path in results_root.glob("*/ledger/campaign.json"))
    if len(candidates) > 1:
        raise ModelCheckError(f"shard has multiple Performance runs: {shard_root}")
    return candidates[0] if candidates else None


def _resume_preparation_bindings(
    execution_root: Path,
    task_bindings: Mapping[str, Sequence[Mapping[str, Any]]],
    invalidated_models: set[str],
) -> dict[str, list[Mapping[str, Any]]]:
    active: dict[str, list[Mapping[str, Any]]] = {}
    for task, bindings in task_bindings.items():
        bindings = list(bindings)
        output = (
            execution_root / "accuracy"
            if task == "accuracy"
            else _perf_shard_output(execution_root)
        )
        if output is None or not (output / "ledger" / "campaign.json").is_file():
            active[task] = bindings
            continue
        try:
            ledger = ExecutionLedger.load(
                output,
                task_kind="performance" if task == "perf" else "accuracy",
            )
        except ExecutionLedgerError as error:
            raise ModelCheckError(str(error)) from error
        selected: list[Mapping[str, Any]] = []
        for binding in bindings:
            case_id = (
                f"{binding['model']}::{binding['workload']}"
                if task == "accuracy"
                else str(binding["entry"])
            )
            receipt = ledger.receipt(case_id)
            attempts = receipt.get("attempts", [])
            evidence = attempts[-1].get("evidence", {}) if attempts else {}
            retryable = (
                receipt.get("result") == "white"
                and isinstance(evidence, Mapping)
                and evidence.get("retryable") is True
            )
            if (
                str(binding["model"]) in invalidated_models
                or receipt.get("state") != "terminal"
                or retryable
            ):
                selected.append(binding)
        active[task] = selected
    return active


def _model_source_identity(
    execution_root: Path,
    tasks: Iterable[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        output = (
            execution_root / "accuracy"
            if task == "accuracy"
            else _perf_shard_output(execution_root)
        )
        report_path = output / "report.json" if output is not None else None
        if report_path is None or not report_path.is_file():
            raise ModelCheckError(f"completed {_task_label(task)} task has no report.json")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelCheckError(f"cannot read {_task_label(task)} report: {error}") from error
        report_rows = report.get("results") if isinstance(report, Mapping) else None
        if not isinstance(report_rows, list):
            raise ModelCheckError(f"{_task_label(task)} report has no result rows")
        for row in report_rows:
            if not isinstance(row, Mapping):
                raise ModelCheckError(f"{_task_label(task)} report contains an invalid row")
            rows.append({**dict(row), "task": task})
    try:
        return case_evidence.summarize_model_revisions(rows)
    except case_evidence.CaseEvidenceError as error:
        raise ModelCheckError(str(error)) from error


def _refresh_shard_report(task: str, output: Path) -> None:
    report = output / "report.json"
    receipts = list((output / "ledger" / "cases").glob("*/receipt.json"))
    if (
        report.is_file()
        and receipts
        and report.stat().st_mtime_ns >= max(receipt.stat().st_mtime_ns for receipt in receipts)
    ):
        return
    if task == "accuracy":
        trtmc_validate.write_report(output)
        return
    try:
        perf_matrix.write_report(output)
    except perf_matrix.PerfMatrixError as error:
        raise ModelCheckError(str(error)) from error


def _validate_shard_member(
    shard_root: Path,
    *,
    label: str,
    index: int,
    campaign: Mapping[str, Any],
) -> dict[str, Any] | None:
    request_path = shard_root / "request.json"
    if not request_path.is_file():
        return None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCheckError(f"cannot read shard request {request_path}: {error}") from error
    selection = request.get("selection", {}) if isinstance(request, Mapping) else {}
    stable_selection = (
        {key: value for key, value in selection.items() if key != "platform_source"}
        if isinstance(selection, Mapping)
        else None
    )
    expected_shard = {
        "index": index,
        "count": campaign["shard_count"],
        "name": label,
    }
    if (
        not isinstance(request, Mapping)
        or request.get("run_id") != campaign.get("run_id")
        or request.get("platform") != campaign.get("platform")
        or request.get("shard") != expected_shard
        or stable_selection != campaign.get("selection")
    ):
        raise ModelCheckError(f"shard {label} does not belong to this campaign")
    return dict(request)


def _shard_result_status(
    shard_root: Path,
    request: Mapping[str, Any],
) -> str | None:
    result_path = shard_root / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCheckError(f"cannot read shard result {result_path}: {error}") from error
    if not isinstance(result, Mapping) or result.get("run_id") != request.get("run_id"):
        raise ModelCheckError(f"shard result does not belong to its request: {result_path}")
    if result.get("execution_revision") != request.get("revision"):
        return None
    status = result.get("status")
    if status == "running":
        return None
    if status not in {"passed", "failed"}:
        raise ModelCheckError(f"shard result has invalid status: {result_path}")
    return str(status)


def _consolidate_once(run_root: Path) -> bool:
    try:
        campaign = campaign_shards.load_campaign(run_root)
    except campaign_shards.CampaignShardError as error:
        raise ModelCheckError(str(error)) from error
    cases = campaign.get("cases")
    shard_count = campaign.get("shard_count")
    if not isinstance(cases, list) or not isinstance(shard_count, int) or shard_count < 1:
        raise ModelCheckError("sharded campaign inventory is invalid")
    shards = []
    shard_results_complete = True
    shards_passed = True
    for index in range(shard_count):
        label = campaign_shards.shard_name(index, shard_count)
        shard_root = run_root / "shards" / label
        request = None
        if shard_root.exists():
            request = _validate_shard_member(
                shard_root,
                label=label,
                index=index,
                campaign=campaign,
            )
        status = _shard_result_status(shard_root, request) if request is not None else None
        shard_results_complete = shard_results_complete and status is not None
        shards_passed = shards_passed and status == "passed"
        shards.append((index, label, shard_root, request is not None))

    all_terminal = True
    for task, report_kind in (
        ("accuracy", "accuracy"),
        ("perf", "performance"),
    ):
        expected = [
            case for case in cases if isinstance(case, Mapping) and case.get("task") == task
        ]
        if not expected:
            continue
        shard_outputs: list[tuple[str, Path]] = []
        for _index, label, shard_root, ready in shards:
            if not ready:
                output = None
            elif task == "accuracy":
                output = shard_root / "accuracy"
            else:
                output = _perf_shard_output(shard_root)
            if output is None or not (output / "ledger" / "campaign.json").is_file():
                continue
            _refresh_shard_report(task, output)
            shard_outputs.append((label, output))

        try:
            _, _, report = campaign_shards.merge_receipt_reports(
                run_root / task,
                report_kind=report_kind,
                campaign=campaign,
                expected_cases=expected,
                shard_outputs=shard_outputs,
            )
        except campaign_shards.CampaignShardError as error:
            raise ModelCheckError(str(error)) from error
        progress = report["accounting"]["progress"]
        all_terminal = all_terminal and not progress["pending"] and not progress["running"]
        print(
            f"{_task_label(task)}: {progress['terminal']}/{report['accounting']['selected']} "
            f"terminal · {run_root / task / 'report.json'}",
            flush=True,
        )
    all_terminal = all_terminal and shard_results_complete
    if all_terminal:
        combined_rows: list[dict[str, Any]] = []
        for task in TASKS:
            report_path = run_root / task / "report.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for row in report.get("results", []):
                if isinstance(row, Mapping):
                    combined_rows.append({**dict(row), "task": task})
        try:
            source_identity = case_evidence.summarize_model_revisions(combined_rows)
        except case_evidence.CaseEvidenceError as error:
            raise ModelCheckError(str(error)) from error
        _write_request(
            run_root / "result.json",
            {
                "schema_version": "trtmc.model-check-run-result/v1",
                "run_id": campaign.get("run_id"),
                "model_source_identity": source_identity,
                "status": (
                    "passed"
                    if shards_passed and source_identity["consistent"]
                    else "failed"
                ),
            },
        )
    return all_terminal


def _consolidate(arguments: argparse.Namespace) -> int:
    if arguments.interval_seconds <= 0:
        raise ModelCheckError("--interval-seconds must be positive")
    run_root = arguments.run_root.expanduser().resolve()
    try:
        with campaign_shards.consolidator_lock(run_root):
            while True:
                complete = _consolidate_once(run_root)
                if complete or not arguments.watch:
                    if not complete:
                        return 0
                    result = json.loads(
                        (run_root / "result.json").read_text(encoding="utf-8")
                    )
                    return 0 if result.get("status") == "passed" else 1
                time.sleep(arguments.interval_seconds)
    except campaign_shards.CampaignShardError as error:
        raise ModelCheckError(str(error)) from error


def _resolved_revision(revision: str) -> str:
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    value = resolved.stdout.strip().lower()
    if resolved.returncode or not EXACT_GIT_REVISION_PATTERN.fullmatch(value):
        raise ModelCheckError(f"cannot resolve exact source revision: {revision}")
    return value


def _worktree_changes() -> tuple[str, ...]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode:
        raise ModelCheckError("cannot inspect the active worktree state")
    return tuple(line for line in status.stdout.splitlines() if line)


def _source_identity(revision: str, *, require_clean: bool = False) -> dict[str, Any]:
    modules = {
        "model_checks": sys.modules[__name__],
        "perf_matrix": perf_matrix,
        "trtmc_validate": trtmc_validate,
        "performance_catalog": performance_catalog,
        "validation_catalog": validation_catalog,
        "python_profiles": sys.modules["tensorrt_model_connect.python_profiles"],
    }
    imports: dict[str, str] = {}
    for name, module in modules.items():
        module_path = Path(str(module.__file__)).resolve()
        try:
            imports[name] = str(module_path.relative_to(REPOSITORY))
        except ValueError as exc:
            raise ModelCheckError(
                f"{name} was imported from outside the active worktree: {module_path}"
            ) from exc
    head = _resolved_revision("HEAD")
    if revision != head:
        raise ModelCheckError(
            "requested source revision does not match the active worktree HEAD: "
            f"requested={revision}, HEAD={head}"
        )
    if require_clean:
        changes = _worktree_changes()
        if changes:
            raise ModelCheckError(
                "qualification requires a clean worktree; commit or remove local changes"
            )
    return {"revision": revision, "imports": imports}


def _revalidate_qualification_source(
    revision: str,
    expected: Mapping[str, Any],
) -> None:
    current = _source_identity(revision, require_clean=True)
    if current != expected:
        raise ModelCheckError("qualification source identity changed during the run")


def _prepare_accuracy_dependencies(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    arguments: argparse.Namespace,
    *,
    bindings: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    bindings = list(bindings) if bindings is not None else _task_bindings(plan, "accuracy")
    if not bindings:
        return
    task_models = validation_catalog.load_manifest_records_by_name(arguments.models_dir)
    suites = {suite["id"]: suite for suite in validation_catalog.load_suites(arguments.suites)}
    profile_names: list[str] = []
    for binding in bindings:
        selected = trtmc_validate.Binding(
            str(binding["model"]),
            str(binding["workload"]),
        )
        profile_names.extend(
            trtmc_validate.binding_profiles(
                selected,
                task_models=task_models,
                suites=suites,
            )
        )

    config = environment["tasks"]["accuracy"]
    options = config.get("options", {})
    profile_root = str(environment["storage"]["python_profiles_root"])
    previous_profile_root = os.environ.get(PROFILE_ROOT_ENV)
    previous_prebuilt = os.environ.get(PREBUILT_ONLY_ENV)
    os.environ[PROFILE_ROOT_ENV] = profile_root
    os.environ.pop(PREBUILT_ONLY_ENV, None)
    try:
        trtmc_validate.ensure_environments(
            tuple(dict.fromkeys(profile_names)),
            str(options.get("hf-python", config["runner_python"])),
        )
    finally:
        if previous_profile_root is None:
            os.environ.pop(PROFILE_ROOT_ENV, None)
        else:
            os.environ[PROFILE_ROOT_ENV] = previous_profile_root
        if previous_prebuilt is None:
            os.environ.pop(PREBUILT_ONLY_ENV, None)
        else:
            os.environ[PREBUILT_ONLY_ENV] = previous_prebuilt

    prepared_models: set[str] = set()
    for binding in bindings:
        model = str(binding["model"])
        if model in prepared_models:
            continue
        prepared_models.add(model)
        record = task_models[model]
        trtmc_validate.ensure_reference_sources(
            str(record.get("family", "") or ""),
            Path(str(options["reference-cache-dir"])),
            record.get("model_reference_cache"),
            source_cache_root=Path(str(environment["storage"]["model_reference_cache_root"])),
            prebuilt_only=False,
        )


def _prepare_qualification_dependencies(
    plan: Mapping[str, Any],
    environment: Mapping[str, Any],
    arguments: argparse.Namespace,
    *,
    task_bindings: Mapping[str, Sequence[Mapping[str, Any]]],
    perf_environment: Path | None,
    perf_preparation_receipt: Path | None,
    model_reference_cache_root: Path,
) -> dict[str, str]:
    print("\nPreparing qualification dependencies", flush=True)
    _prepare_accuracy_dependencies(
        plan,
        environment,
        arguments,
        bindings=task_bindings.get("accuracy", ()),
    )
    contracts = _selected_perf_reference_contracts(
        plan,
        arguments.models_dir,
        bindings=task_bindings.get("perf", ()),
    )
    reference_environment = _prepare_perf_reference_dependencies(
        contracts,
        model_reference_cache_root,
    )
    if perf_environment is not None and perf_preparation_receipt is not None:
        command = _perf_prepare_command(
            plan,
            environment,
            perf_environment,
            perf_preparation_receipt,
            bindings=task_bindings.get("perf", ()),
        )
        if command is not None:
            completed = subprocess.run(
                _detailed_command(command, verbose=arguments.verbose),
                cwd=REPOSITORY,
                check=False,
                env=_task_environment(
                    environment,
                    reference_environment,
                    source_revision=arguments.revision,
                ),
            )
            if completed.returncode:
                raise ModelCheckError("Perf bundle preparation failed")
    print("Qualification dependencies prepared; starting frozen measurement", flush=True)
    return reference_environment


def _run(arguments: argparse.Namespace) -> int:
    arguments.intent = "debug" if arguments.debug else "qualification"
    shard = campaign_shards.parse_shard(arguments.shard) if arguments.shard else None
    if shard is not None and not arguments.run_id:
        raise ModelCheckError("--shard requires an explicit shared --run-id")
    arguments.revision = _resolved_revision(arguments.revision)
    source_identity = _source_identity(
        arguments.revision,
        require_clean=arguments.intent == "qualification",
    )
    platform = load_platform(arguments.platform)
    environment = load_execution_environment(
        arguments.environment or str(platform["id"]),
        platform_id=str(platform["id"]),
    )
    arguments.perf_suite = Path(environment["tasks"]["perf"]["suite"])
    plan, _ = _resolve_request(arguments)
    if plan["summary"]["blocker_count"]:
        print(_render(plan))
        raise ModelCheckError("selection has unconfigured task bindings")
    invalidated_models = set(arguments.invalidate_model)
    if invalidated_models and not arguments.resume:
        raise ModelCheckError("--invalidate-model requires --resume")
    selected_models = {str(model["model"]) for model in plan["models"]}
    unknown_invalidations = sorted(invalidated_models - selected_models)
    if unknown_invalidations:
        raise ModelCheckError(
            "cannot invalidate models outside this run: "
            + ", ".join(unknown_invalidations)
        )

    run_id = arguments.run_id or _default_run_id(str(platform["id"]))
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ModelCheckError(
            "--run-id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen"
        )
    storage_root = Path(environment["storage"]["root"]).resolve()
    _require_platform_storage_root(storage_root, platform)
    python_profiles_root = _require_managed_path(
        Path(environment["storage"]["python_profiles_root"]),
        storage_root,
        "Python profiles root",
    )
    model_reference_cache_root = _require_managed_path(
        Path(environment["storage"]["model_reference_cache_root"]),
        storage_root,
        "model reference cache root",
    )
    results_root = _require_managed_path(
        Path(environment["storage"]["results_root"]),
        storage_root,
        "results root",
    )
    run_root = _require_managed_path(results_root / run_id, storage_root, "run root")
    if shard is None:
        campaign_cases: list[dict[str, Any]] = []
        task_bindings = {
            task: _task_bindings(plan, task) for task in plan["execution"]["task_order"]
        }
        execution_root = run_root
    else:
        shard_index, shard_count = shard
        accuracy_sample_limits = trtmc_validate.load_catalog(arguments.catalog).get(
            "sample_limits", {}
        )
        campaign_cases = _campaign_cases(
            plan,
            shard_count=shard_count,
            accuracy_sample_limits=accuracy_sample_limits,
        )
        task_bindings = {
            task: _shard_task_bindings(
                plan,
                campaign_cases,
                task=task,
                shard_index=shard_index,
            )
            for task in plan["execution"]["task_order"]
        }
        execution_root = run_root / "shards" / campaign_shards.shard_name(*shard)
    if arguments.hf_cache_seed_dir is not None:
        if not _task_bindings(plan, "accuracy"):
            raise ModelCheckError("--hf-cache-seed-dir requires the Accuracy task")
        if (
            environment["tasks"]["accuracy"]["options"].get("hf-cache-mode", "shared")
            != "per_model"
        ):
            raise ModelCheckError("--hf-cache-seed-dir requires Accuracy hf-cache-mode per_model")
        arguments.hf_cache_seed_dir = _require_managed_path(
            arguments.hf_cache_seed_dir,
            storage_root,
            "Hugging Face cache seed directory",
        )
        if not arguments.hf_cache_seed_dir.is_dir():
            raise ModelCheckError(
                f"Hugging Face cache seed directory does not exist: {arguments.hf_cache_seed_dir}"
            )
    if shard is not None:
        stable_selection = {key: value for key, value in plan.items() if key != "platform_source"}
        try:
            campaign_shards.open_campaign(
                run_root,
                {
                    "run_id": run_id,
                    "platform": platform["id"],
                    "intent": arguments.intent,
                    "shard_count": shard[1],
                    "selection": stable_selection,
                    "cases": campaign_cases,
                },
            )
        except campaign_shards.CampaignShardError as error:
            raise ModelCheckError(str(error)) from error
    if arguments.resume:
        if not execution_root.is_dir():
            raise ModelCheckError(f"cannot resume missing run root: {execution_root}")
    else:
        if execution_root.exists():
            raise ModelCheckError(f"run root already exists: {execution_root}")
        execution_root.mkdir(parents=True)
    _write_selected_models(plan, execution_root)

    perf_environment = None
    if task_bindings.get("perf"):
        build_dir_value = environment["tasks"]["accuracy"]["options"].get("backend-dir")
        if not isinstance(build_dir_value, str) or not build_dir_value:
            raise ModelCheckError(
                "model-check environment accuracy.options.backend-dir is required "
                "as the shared native build directory"
            )
        perf_environment = _resolved_perf_environment(
            Path(environment["tasks"]["perf"]["environment"]),
            destination=execution_root / "perf-environment.yaml",
            build_dir=Path(build_dir_value),
            storage_root=storage_root,
            results_root=execution_root / "perf" / "results",
            scratch_root=execution_root / "work" / "perf",
            bundle_cache=(execution_root / "cache" / "perf" if shard else None),
        )
    perf_preparation_receipt = (
        execution_root / "perf-bundle-preparation.json" if perf_environment is not None else None
    )
    commands: list[tuple[str, list[str] | None]] = []
    for task in plan["execution"]["task_order"]:
        if task == "accuracy":
            command = _accuracy_command(
                plan,
                environment,
                arguments,
                execution_root / "accuracy",
                bindings=task_bindings[task],
            )
        else:
            command = (
                _perf_command(
                    plan,
                    environment,
                    perf_environment,
                    bindings=task_bindings[task],
                    require_prebuilt=arguments.intent == "qualification",
                )
                if perf_environment is not None
                else None
            )
        commands.append((task, command))
    request = {
        "schema_version": "trtmc.model-check-run/v1",
        "run_id": run_id,
        "revision": arguments.revision,
        "intent": arguments.intent,
        "source_identity": source_identity,
        "platform": platform["id"],
        "platform_source": platform["source"],
        "platform_config": platform,
        "environment_source": environment["source"],
        "environment_config": environment,
        "perf_environment_config": (
            _read_yaml(perf_environment, "resolved performance environment")
            if perf_environment is not None
            else None
        ),
        "selection": plan,
        "commands": {task: command for task, command in commands if command is not None},
        "dry_run": bool(arguments.dry_run),
        "shard": (
            {
                "index": shard[0],
                "count": shard[1],
                "name": campaign_shards.shard_name(*shard),
            }
            if shard is not None
            else None
        ),
    }
    request_path = execution_root / "request.json"
    if arguments.resume:
        previous_request = _verify_resume_request(request_path, request)
        previous_attempts = previous_request.get("execution_attempts", [])
        if not isinstance(previous_attempts, list):
            raise ModelCheckError("cannot resume with invalid execution attempt history")
        request["execution_attempts"] = list(previous_attempts)
    else:
        request["execution_attempts"] = []
    _record_execution_attempt(
        request,
        revision=arguments.revision,
        source_identity=source_identity,
    )
    _write_request(request_path, request)

    execution_commands = list(commands)
    if arguments.resume:
        execution_commands = []
        for task, command in commands:
            if command is None:
                execution_commands.append((task, None))
            elif task == "accuracy":
                resumed = [*command, "--resume-existing"]
                task_models = {str(binding["model"]) for binding in task_bindings[task]}
                for model in sorted(invalidated_models & task_models):
                    resumed.extend(["--invalidate-model", model])
                execution_commands.append((task, resumed))
            else:
                resume_command = _perf_resume_command(
                    environment,
                    execution_root / "perf" / "results",
                    require_prebuilt=arguments.intent == "qualification",
                )
                resumed = resume_command or command
                if resume_command is not None:
                    task_models = {
                        str(binding["model"]) for binding in task_bindings[task]
                    }
                    for model in sorted(invalidated_models & task_models):
                        resumed.extend(["--invalidate-model", model])
                execution_commands.append((task, resumed))
    print(_render_run_header(plan, run_id=run_id, run_root=execution_root))
    if shard is not None:
        print(
            f"Shard: {shard[0]}/{shard[1]} · "
            f"{sum(len(bindings) for bindings in task_bindings.values())} bindings"
        )
    profile_policy = (
        "shared; creates missing profiles"
        if arguments.intent == "debug"
        else "prepared before measurement; prebuilt during measurement"
    )
    print(f"Python profiles: {python_profiles_root} ({profile_policy})")
    if arguments.verbose or arguments.dry_run:
        print("\nCommands:")
        for task, command in execution_commands:
            if command is None:
                print(f"  {_task_label(task)}: no configured bindings")
                continue
            rendered = _detailed_command(command, verbose=arguments.verbose)
            print(f"  {_task_label(task)}: {shlex.join(rendered)}")

    if arguments.dry_run:
        return 0

    if shard is not None:
        _write_request(
            execution_root / "result.json",
            {
                "schema_version": "trtmc.model-check-run-result/v1",
                "run_id": run_id,
                "execution_revision": arguments.revision,
                "status": "running",
            },
        )

    preparation_bindings = (
        _resume_preparation_bindings(
            execution_root,
            task_bindings,
            invalidated_models,
        )
        if arguments.resume
        else task_bindings
    )
    if arguments.intent == "qualification":
        reference_environment = _prepare_qualification_dependencies(
            plan,
            environment,
            arguments,
            task_bindings=preparation_bindings,
            perf_environment=perf_environment,
            perf_preparation_receipt=perf_preparation_receipt,
            model_reference_cache_root=model_reference_cache_root,
        )
        _revalidate_qualification_source(arguments.revision, source_identity)
    else:
        reference_contracts = _selected_perf_reference_contracts(
            plan,
            arguments.models_dir,
            bindings=preparation_bindings.get("perf", ()),
        )
        reference_environment = _prepare_perf_reference_dependencies(
            reference_contracts,
            model_reference_cache_root,
        )
        if reference_contracts:
            print(
                f"Prepared external model sources: {len(reference_contracts)} "
                f"under {model_reference_cache_root}",
                flush=True,
            )

    task_results: dict[str, int] = {}
    runnable = [(task, command) for task, command in execution_commands if command is not None]
    for index, (task, command) in enumerate(runnable, start=1):
        label = _task_label(task)
        print(f"\n[{index}/{len(runnable)}] {label}", flush=True)
        if command is None:
            continue
        if arguments.intent == "qualification":
            _revalidate_qualification_source(arguments.revision, source_identity)
        completed = subprocess.run(
            _detailed_command(command, verbose=arguments.verbose),
            cwd=REPOSITORY,
            check=False,
            env=_task_environment(
                environment,
                reference_environment,
                source_revision=arguments.revision,
                allow_dependency_creation=arguments.intent == "debug",
            ),
        )
        if arguments.intent == "qualification":
            _revalidate_qualification_source(arguments.revision, source_identity)
        if (
            task == "perf"
            and completed.returncode == 0
            and preparation_bindings.get("perf")
            and perf_preparation_receipt is not None
            and perf_preparation_receipt.is_file()
        ):
            perf_output = _perf_shard_output(execution_root)
            if perf_output is None:
                raise ModelCheckError("cannot locate completed Perf output")
            try:
                perf_matrix.write_report(
                    perf_output,
                    preparation_receipt=perf_preparation_receipt,
                )
            except perf_matrix.PerfMatrixError as error:
                raise ModelCheckError(str(error)) from error
        task_results[task] = completed.returncode
        status = "PASSED" if completed.returncode == 0 else "FAILED"
        print(f"[{index}/{len(runnable)}] {label}: {status}", flush=True)
    tasks_passed = all(code == 0 for code in task_results.values())
    model_source_identity = (
        _model_source_identity(execution_root, task_results)
        if tasks_passed and task_results
        else None
    )
    identity_passed = (
        model_source_identity is None or model_source_identity["consistent"]
    )
    result = {
        "schema_version": "trtmc.model-check-run-result/v1",
        "run_id": run_id,
        "resumed": bool(arguments.resume),
        "execution_revision": arguments.revision,
        "task_exit_codes": task_results,
        "model_source_identity": model_source_identity,
        "status": "passed" if tasks_passed and identity_passed else "failed",
    }
    _write_request(execution_root / "result.json", result)
    print(f"\nOverall: {result['status'].upper()}")
    for task, returncode in task_results.items():
        status = "PASSED" if returncode == 0 else "FAILED"
        print(f"  {_task_label(task)}: {status}")
    if model_source_identity is not None and not identity_passed:
        inconsistent = [
            model
            for model, evidence in model_source_identity["models"].items()
            if evidence["status"] != "consistent"
        ]
        print("  Source identity: FAILED · " + ", ".join(inconsistent))
    print(f"Run root: {execution_root}")
    return 0 if result["status"] == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "check":
            return _check(arguments)
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "consolidate":
            return _consolidate(arguments)
        raise ModelCheckError(f"unsupported command: {arguments.command}")
    except (
        ModelCheckError,
        campaign_shards.CampaignShardError,
        case_evidence.CaseEvidenceError,
        model_ci.ModelCIError,
        model_selection.ModelSelectionError,
        perf_matrix.PerfMatrixError,
        trtmc_validate.ValidationError,
        CiError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

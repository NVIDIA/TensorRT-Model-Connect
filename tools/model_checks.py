#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve canonical models into independent Accuracy and Perf bindings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = REPOSITORY / "python"
for source_root in (REPOSITORY, PYTHON_SOURCE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from tools import model_ci  # noqa: E402
from tools import model_selection  # noqa: E402
from tools import perf_matrix  # noqa: E402
from tools import trtmc_validate  # noqa: E402
from tools.validation import catalog as validation_catalog  # noqa: E402


PLATFORM_SCHEMA = "trtmc.model-check-platform/v1"
DEFAULT_PLATFORM_ROOT = REPOSITORY / "tests" / "model_checks" / "platforms"
DEFAULT_PERF_SUITE = REPOSITORY / "benchmarks" / "performance" / "release.yaml"
TASKS = ("accuracy", "perf")


class ModelCheckError(ValueError):
    """A model-check selection or platform profile is invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="show task bindings without running them")
    check.add_argument("--platform", required=True, help="platform ID or profile YAML")
    selection = check.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", action="append", default=[])
    selection.add_argument("--model-selection", type=Path)
    selection.add_argument("--all", action="store_true")
    check.add_argument("--task", action="append", choices=TASKS, default=[])
    accuracy = check.add_mutually_exclusive_group()
    accuracy.add_argument("--accuracy-suite", action="append", default=[])
    accuracy.add_argument("--all-accuracy-suites", action="store_true")
    check.add_argument("--json", action="store_true", help="print the resolved JSON")
    check.add_argument("--revision", default="HEAD")
    check.add_argument("--catalog", type=Path, default=trtmc_validate.DEFAULT_CATALOG)
    check.add_argument("--suites", type=Path, default=trtmc_validate.DEFAULT_SUITES)
    check.add_argument("--models-dir", type=Path, default=trtmc_validate.DEFAULT_MODELS)
    check.add_argument("--perf-suite", type=Path, default=DEFAULT_PERF_SUITE)
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
        raise ModelCheckError(
            f"platform profile schema_version must be {PLATFORM_SCHEMA}: {path}"
        )
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
    unsupported = profile.get("unsupported", [])
    if not isinstance(unsupported, list):
        raise ModelCheckError(f"platform unsupported must be a list: {path}")
    for index, item in enumerate(unsupported):
        if not isinstance(item, Mapping):
            raise ModelCheckError(f"platform unsupported[{index}] must be an object")
        if not all(isinstance(item.get(key), str) and item[key] for key in ("model", "task", "reason")):
            raise ModelCheckError(
                f"platform unsupported[{index}] needs model, task, and reason"
            )
        if item["task"] not in TASKS:
            raise ModelCheckError(
                f"platform unsupported[{index}] has unknown task {item['task']!r}"
            )
    return {**profile, "source": str(path)}


def _selected_tasks(profile: Mapping[str, Any], requested: Iterable[str]) -> tuple[str, ...]:
    task_order = tuple(profile["execution"]["task_order"])
    requested = tuple(dict.fromkeys(requested))
    return (
        tuple(task for task in task_order if task in requested)
        if requested
        else task_order
    )


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
                if model in accuracy_catalog["models"]
                and str(record.get("family", "")) == owner
            )
        if "perf" in tasks:
            matched.update(
                str(case["model"])
                for case in perf_cases
                if str(case["family"]) == owner
            )
        if not matched:
            missing.append(owner)
        profiles.extend(sorted(matched))
    if missing:
        raise ModelCheckError(
            "model owners have no selected task profiles: " + ", ".join(missing)
        )
    return model_selection.normalize_models(profiles)


def _unsupported_reason(
    profile: Mapping[str, Any],
    *,
    model: str,
    task: str,
    binding: str,
) -> str:
    for item in profile.get("unsupported", []):
        if item["model"] != model or item["task"] != task:
            continue
        selected_binding = str(item.get("binding", "") or "")
        if not selected_binding or selected_binding == binding:
            return str(item["reason"])
    return ""


def _accuracy_projection(
    model: str,
    *,
    catalog: Mapping[str, Any],
    workloads: Sequence[str],
    all_workloads: bool,
    platform: Mapping[str, Any],
) -> dict[str, Any]:
    if model not in catalog["models"]:
        return {
            "status": "unconfigured",
            "reason": "model has no Accuracy catalog binding",
            "bindings": [],
        }
    try:
        bindings = trtmc_validate.resolve_bindings(
            catalog,
            [model],
            workloads=workloads,
            all_workloads=all_workloads,
        )
    except trtmc_validate.ValidationError as exc:
        return {"status": "unconfigured", "reason": str(exc), "bindings": []}
    projected = []
    for binding in bindings:
        binding_id = binding.workload or "not-compared"
        reason = _unsupported_reason(
            platform,
            model=model,
            task="accuracy",
            binding=binding_id,
        )
        projected.append(
            {
                "id": f"accuracy:{model}:{binding_id}",
                "workload": binding.workload,
                "status": "unsupported" if reason else "configured",
                **({"reason": reason} if reason else {}),
            }
        )
    status = (
        "unsupported"
        if projected and all(item["status"] == "unsupported" for item in projected)
        else "configured"
    )
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
        if perf_matrix._is_l0_profile(model):
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
        reason = _unsupported_reason(
            platform,
            model=model,
            task="perf",
            binding=entry_id,
        )
        projected.append(
            {
                "id": f"perf:{model}:{entry_id}",
                "entry": entry_id,
                "status": "unsupported" if reason else "configured",
                **({"reason": reason} if reason else {}),
            }
        )
    status = (
        "unsupported"
        if projected and all(item["status"] == "unsupported" for item in projected)
        else "configured"
    )
    return {"status": status, "bindings": projected}


def resolve_plan(
    *,
    models: Sequence[str],
    tasks: Sequence[str],
    platform: Mapping[str, Any],
    accuracy_catalog: Mapping[str, Any],
    accuracy_workloads: Sequence[str],
    all_accuracy_workloads: bool,
    perf_cases: Sequence[Mapping[str, Any]],
    perf_exclusions: Mapping[str, str],
) -> dict[str, Any]:
    results = []
    blocker_count = 0
    for model in model_selection.normalize_models(models):
        record: dict[str, Any] = {"model": model, "tasks": {}}
        if "accuracy" in tasks:
            record["tasks"]["accuracy"] = _accuracy_projection(
                model,
                catalog=accuracy_catalog,
                workloads=accuracy_workloads,
                all_workloads=all_accuracy_workloads,
                platform=platform,
            )
        if "perf" in tasks:
            record["tasks"]["perf"] = _perf_projection(
                model,
                cases=perf_cases,
                exclusions=perf_exclusions,
                platform=platform,
            )
        blocker_count += sum(
            task["status"] == "unconfigured" for task in record["tasks"].values()
        )
        results.append(record)
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
            "binding_count": sum(
                len(task["bindings"])
                for record in results
                for task in record["tasks"].values()
            ),
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
            f"{summary['blocker_count']} blockers",
        ]
    )
    return "\n".join(lines)


def _check(arguments: argparse.Namespace) -> int:
    platform = load_platform(arguments.platform)
    tasks = _selected_tasks(platform, arguments.task)
    accuracy_catalog = trtmc_validate.load_catalog(arguments.catalog)
    accuracy_suites = validation_catalog.load_suites(arguments.suites)
    accuracy_suite_map = {suite["id"]: suite for suite in accuracy_suites}
    accuracy_models = trtmc_validate._validation_models(arguments.models_dir)
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

    perf_suite = perf_matrix._read_yaml(arguments.perf_suite)
    perf_cases = perf_matrix._cases(perf_suite)
    perf_exclusions = perf_matrix._excluded_profiles(perf_suite)
    perf_matrix._validate_coverage(perf_cases, perf_exclusions)

    available_profiles = set(accuracy_catalog["models"]).union(
        str(case["model"]) for case in perf_cases
    )
    if arguments.all:
        models = tuple(sorted(available_profiles))
    elif arguments.model_selection:
        owners = model_selection.load_model_selection(arguments.model_selection)
        known_owners = set(
            model_ci.discover_catalog(REPOSITORY, arguments.revision).models
        )
        unknown_owners = sorted(set(owners) - known_owners)
        if unknown_owners:
            raise ModelCheckError(
                "unknown model owners: " + ", ".join(unknown_owners)
            )
        models = model_profiles_for_owners(
            owners,
            tasks=tasks,
            accuracy_models=accuracy_models,
            accuracy_catalog=accuracy_catalog,
            perf_cases=perf_cases,
        )
    else:
        models = model_selection.normalize_models(arguments.model)
        unknown = sorted(set(models) - available_profiles)
        if unknown:
            raise ModelCheckError("unknown model profiles: " + ", ".join(unknown))

    plan = resolve_plan(
        models=models,
        tasks=tasks,
        platform=platform,
        accuracy_catalog=accuracy_catalog,
        accuracy_workloads=tuple(arguments.accuracy_suite),
        all_accuracy_workloads=arguments.all_accuracy_suites,
        perf_cases=perf_cases,
        perf_exclusions=perf_exclusions,
    )
    print(json.dumps(plan, indent=2) if arguments.json else _render(plan))
    return 2 if plan["summary"]["blocker_count"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "check":
            return _check(arguments)
        raise ModelCheckError(f"unsupported command: {arguments.command}")
    except (
        ModelCheckError,
        model_ci.ModelCIError,
        model_selection.ModelSelectionError,
        perf_matrix.PerfMatrixError,
        trtmc_validate.ValidationError,
    ) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

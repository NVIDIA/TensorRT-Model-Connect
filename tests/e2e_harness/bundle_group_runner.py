# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared grouped-bundle execution support for model-owned E2E entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tests.e2e_harness.contracts import E2EStatus, RunContext, StageStatus
from tests.e2e_harness.manifest_loader import get_case_by_name, load_all_manifests
from tests.e2e_harness.model_selection import (
    BUNDLE_GROUP_PREFIX,
    case_matches_e2e_model,
    case_names_from_param,
    select_cases_from_models_file,
)
from tests.e2e_harness.orchestrator import E2EOrchestrator
from tests.e2e_harness.python_profiles import (
    resolve_case_profile_names,
    resolve_case_python_profiles,
)


_GROUP_BY_BUNDLE_METADATA_KEY = "group_by_bundle"


def _parse_e2e_model_filters(values: list[str] | None) -> set[str]:
    filters: set[str] = set()
    for raw in values or []:
        for item in str(raw).split(","):
            item = item.strip()
            if item:
                filters.add(item)
    return filters


def _group_cases_by_bundle(cases) -> list[list]:
    groups: list[list] = []
    bundle_groups: dict[str, list] = {}
    for case in cases:
        if not _case_allows_bundle_group(case):
            groups.append([case])
            continue
        group = bundle_groups.get(case.bundle)
        if group is None:
            group = []
            bundle_groups[case.bundle] = group
            groups.append(group)
        group.append(case)
    return [_sort_bundle_group(group) for group in groups]


def _case_allows_bundle_group(case) -> bool:
    metadata = case.metadata or {}
    return bool(case.bundle and metadata.get(_GROUP_BY_BUNDLE_METADATA_KEY))


def _sort_bundle_group(cases) -> list:
    if not cases:
        return []
    bundle_stem = Path(cases[0].bundle).stem if cases[0].bundle else ""
    return sorted(cases, key=lambda case: (case.name != bundle_stem, case.name))


def _case_group_id(cases) -> str:
    names = [case.name for case in cases]
    if len(names) == 1:
        return names[0]
    return BUNDLE_GROUP_PREFIX + "+".join(names)


def model_case_names_for_dir(
    *,
    config,
    model_dir: Path,
    case_matches_model: Callable,
    is_multi_device_case: Callable,
) -> list[str]:
    strategy_filter = None
    core_only = False
    multi_device_only = False
    excluded_ci_tiers = set()
    model_filters: set[str] = set()
    models_file = None
    group_by_bundle = False

    if config is not None:
        strategy_filter = config.getoption("--e2e-task-strategy", default=None)
        model_filters = _parse_e2e_model_filters(
            config.getoption("--e2e-model", default=[]) or []
        )
        core_only = config.getoption("--e2e-core-only", default=False)
        multi_device_only = config.getoption("--multi-device-only", default=False)
        excluded_ci_tiers = set(
            config.getoption("--e2e-exclude-ci-tier", default=[]) or []
        )
        models_file = config.getoption("--e2e-models-file", default=None)
        group_by_bundle = config.getoption("--e2e-group-by-bundle", default=False)

    cases = load_all_manifests(model_dir, task_strategy_filter=strategy_filter)

    if models_file:
        cases = select_cases_from_models_file(cases, models_file)

    if model_filters:
        cases = [
            case
            for case in cases
            if case_matches_model(case, model_filters)
            and case_matches_e2e_model(case, model_filters)
        ]

    if excluded_ci_tiers:
        cases = [
            case for case in cases
            if str(case.metadata.get("ci_tier", "")) not in excluded_ci_tiers
        ]

    if multi_device_only:
        cases = [case for case in cases if is_multi_device_case(case)]
    else:
        cases = [case for case in cases if not is_multi_device_case(case)]

    if core_only:
        cases = [case for case in cases if case.metadata.get("core", False)]

    if group_by_bundle:
        return [_case_group_id(group) for group in _group_cases_by_bundle(cases)]
    return [case.name for case in cases]


def run_model_e2e_case_or_group(
    *,
    case_name: str,
    request,
    model_dir: Path,
    load_waives: Callable,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
) -> None:
    case_names = case_names_from_param(case_name)
    if len(case_names) == 1:
        _assert_single_case(
            case_names[0],
            request,
            model_dir,
            load_waives,
            resolve_hf_python,
            resolve_artifacts_dir,
            resolve_binary,
            resolve_ld_library_path,
            resolve_engine_dir,
            resolve_model_plugin_dir,
            model_plugin_dir_env,
        )
    else:
        _assert_bundle_group(
            case_name,
            case_names,
            request,
            model_dir,
            load_waives,
            resolve_hf_python,
            resolve_artifacts_dir,
            resolve_binary,
            resolve_ld_library_path,
            resolve_engine_dir,
            resolve_model_plugin_dir,
            model_plugin_dir_env,
        )


def _assert_single_case(
    case_name: str,
    request,
    model_dir: Path,
    load_waives: Callable,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
) -> None:
    outcome = _run_case(
        case_name,
        request,
        model_dir,
        load_waives,
        resolve_hf_python,
        resolve_artifacts_dir,
        resolve_binary,
        resolve_ld_library_path,
        resolve_engine_dir,
        resolve_model_plugin_dir,
        model_plugin_dir_env,
        rebuild_override=None,
        mark_xfail=True,
    )
    if outcome["status"] == "skip":
        pytest.skip(outcome["message"])
    if outcome["status"] == "fail":
        pytest.fail(outcome["message"])


def _assert_bundle_group(
    group_name: str,
    case_names: list[str],
    request,
    model_dir: Path,
    load_waives: Callable,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
) -> None:
    requested_rebuild = request.config.getoption("--rebuild-engines", default=False)
    bundle_rebuilt = False
    outcomes = []

    for case_name in case_names:
        rebuild_case = requested_rebuild and not bundle_rebuilt
        outcome = _run_case(
            case_name,
            request,
            model_dir,
            load_waives,
            resolve_hf_python,
            resolve_artifacts_dir,
            resolve_binary,
            resolve_ld_library_path,
            resolve_engine_dir,
            resolve_model_plugin_dir,
            model_plugin_dir_env,
            rebuild_override=rebuild_case,
            mark_xfail=False,
        )
        outcomes.append(outcome)
        if requested_rebuild and outcome.get("bundle_exists"):
            bundle_rebuilt = True
        if outcome.get("failure_type") == "build_fail":
            break

    failures = [outcome for outcome in outcomes if outcome["status"] == "fail"]
    if failures:
        pytest.fail(_format_group_outcomes(group_name, outcomes, "Grouped E2E entry failed"))

    if any(outcome["status"] == "pass" for outcome in outcomes):
        return

    xfails = [outcome for outcome in outcomes if outcome["status"] == "xfail"]
    if xfails:
        pytest.xfail(_format_group_outcomes(group_name, outcomes, "Grouped E2E entry xfailed"))

    skips = [outcome for outcome in outcomes if outcome["status"] == "skip"]
    if skips:
        pytest.skip(_format_group_outcomes(group_name, outcomes, "Grouped E2E entry skipped"))

    pytest.skip(f"No E2E cases executed for {group_name}")


def _run_case(
    case_name: str,
    request,
    model_dir: Path,
    load_waives: Callable,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
    *,
    rebuild_override: bool | None,
    mark_xfail: bool,
) -> dict:
    if case_name == "__no_models__":
        return {
            "name": case_name,
            "status": "skip",
            "message": "No model manifests found",
            "bundle_exists": False,
        }

    config = request.config
    waives = load_waives(config.getoption("--e2e-platform", default=""))
    xfail_reason = ""
    if case_name in waives:
        action, reason = waives[case_name]
        if action == "SKIP":
            return {
                "name": case_name,
                "status": "skip",
                "message": reason,
                "bundle_exists": False,
            }
        if action == "XFAIL":
            xfail_reason = reason
            if mark_xfail:
                request.node.add_marker(pytest.mark.xfail(reason=reason, strict=False))

    case = get_case_by_name(case_name, model_dir)
    if case is None:
        return {
            "name": case_name,
            "status": "fail",
            "message": f"Case not found in {model_dir}: {case_name}",
            "bundle_exists": False,
        }

    skip_reason = case.metadata.get("skip_reason", "")
    if skip_reason:
        return {
            "name": case_name,
            "status": "skip",
            "message": skip_reason,
            "bundle_exists": False,
        }

    base_python = resolve_hf_python(config)
    profile_names = resolve_case_profile_names(case)
    profile_paths = resolve_case_python_profiles(case, base_python)
    engine_dir = resolve_engine_dir(config)
    rebuild = (
        rebuild_override
        if rebuild_override is not None
        else config.getoption("--rebuild-engines", default=False)
    )

    ctx = RunContext(
        case=case,
        artifacts_dir=resolve_artifacts_dir(config),
        binary_path=resolve_binary(config),
        hf_python=base_python,
        build_python=profile_paths["build"],
        runtime_python=profile_paths["runtime"],
        reference_python=profile_paths["reference"],
        build_profile=profile_names["build"],
        runtime_profile=profile_names["runtime"],
        reference_profile=profile_names["reference"],
        ld_library_path=resolve_ld_library_path(),
        engine_dir=engine_dir,
        model_plugin_dir=resolve_model_plugin_dir(config),
        rebuild=rebuild,
        verbose=config.getoption("verbose", default=0) > 0,
    )

    orchestrator = E2EOrchestrator()
    with model_plugin_dir_env(ctx.model_plugin_dir):
        result = orchestrator.run(case, ctx)

    bundle_exists = bool(case.bundle and (Path(engine_dir) / case.bundle).is_file())
    if result.status == E2EStatus.SKIP.value:
        skip_detail = ""
        if result.determinism and "preflight" in result.determinism:
            failed = [d for d in result.determinism["preflight"] if not d.get("passed")]
            if failed:
                skip_detail = "; ".join(d.get("message", "") for d in failed)
        return {
            "name": case_name,
            "status": "skip",
            "message": (
                f"Case {case_name} skipped: {skip_detail}"
                if skip_detail else f"Case {case_name} skipped"
            ),
            "bundle_exists": bundle_exists,
            "failure_type": result.failure_type,
        }
    if result.status == E2EStatus.PASS.value:
        return {
            "name": case_name,
            "status": "pass",
            "message": "",
            "bundle_exists": bundle_exists,
            "failure_type": result.failure_type,
        }

    failed_stages = [
        f"  {name} [{cr.status}]: {cr.message}"
        for name, cr in result.stages.items()
        if cr.status in (StageStatus.FAILED.value, StageStatus.ERROR.value)
    ]
    failure_msg = (
        f"E2E failed for {case_name} "
        f"(failure_type={result.failure_type}, "
        f"oracle_level={result.oracle_level}):\n"
    )
    failure_msg += "\n".join(failed_stages) if failed_stages else f"  status={result.status}"
    if xfail_reason and not mark_xfail:
        return {
            "name": case_name,
            "status": "xfail",
            "message": f"{xfail_reason}\n{failure_msg}" if xfail_reason else failure_msg,
            "bundle_exists": bundle_exists,
            "failure_type": result.failure_type,
        }
    return {
        "name": case_name,
        "status": "fail",
        "message": failure_msg,
        "bundle_exists": bundle_exists,
        "failure_type": result.failure_type,
    }


def _format_group_outcomes(group_name: str, outcomes: list[dict], header: str) -> str:
    lines = [f"{header} for {group_name}:"]
    for outcome in outcomes:
        line = f"  {outcome['name']} [{outcome['status']}]"
        failure_type = outcome.get("failure_type")
        if failure_type:
            line += f" failure_type={failure_type}"
        message = outcome.get("message") or ""
        if message:
            line += f": {message}"
        lines.append(line)
    return "\n".join(lines)

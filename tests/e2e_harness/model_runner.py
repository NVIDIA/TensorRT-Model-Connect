# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Uniform model-manifest execution for model-owned E2E entrypoints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e_harness.contracts import E2ECase, E2EModel, E2EStatus, RunContext, StageStatus
from tests.e2e_harness.manifest_loader import get_model_by_name, load_all_model_manifests
from tests.e2e_harness.model_selection import (
    case_matches_e2e_model,
    parse_e2e_model_filters,
    read_e2e_models_file,
)
from tests.e2e_harness.orchestrator import (
    BundleResolution,
    E2EOrchestrator,
    run_preflight,
)
from tests.e2e_harness.python_profiles import (
    resolve_case_profile_names,
    resolve_case_python_profiles,
)
from tests.e2e_partition import partition_models


def _testcase_filters(config) -> set[str]:
    if config is None:
        return set()
    return parse_e2e_model_filters(config.getoption("--e2e-testcase", default=[]) or [])


def _model_matches_filter(model: E2EModel, filters: set[str]) -> bool:
    if not filters:
        return True
    fields = {
        model.name,
        model.family,
        Path(model.hf_id).name if model.hf_id else "",
    }
    return bool(filters & {field for field in fields if field})


def selected_testcases(
    model: E2EModel,
    *,
    config,
    case_matches_model: Callable,
    is_multi_device_case: Callable,
) -> list[E2ECase]:
    """Return child testcases selected for one model manifest."""
    cases = list(model.testcases)
    if config is None:
        return cases

    models_file = config.getoption("--e2e-models-file", default=None)
    if models_file and model.name not in read_e2e_models_file(models_file):
        return []

    strategy_filter = config.getoption("--e2e-task-strategy", default=None)
    if strategy_filter:
        cases = [case for case in cases if case.task_strategy == strategy_filter]

    model_filters = parse_e2e_model_filters(config.getoption("--e2e-model", default=[]) or [])
    if model_filters and not _model_matches_filter(model, model_filters):
        cases = [
            case
            for case in cases
            if case_matches_model(case, model_filters)
            and case_matches_e2e_model(case, model_filters)
        ]

    testcase_filters = _testcase_filters(config)
    if testcase_filters:
        cases = [case for case in cases if case.name in testcase_filters]

    category_filter = config.getoption("--e2e-category", default=None)
    if category_filter:
        cases = [
            case
            for case in cases
            if case.metadata.get("test_category", "e2e") == category_filter
        ]

    excluded_ci_tiers = set(config.getoption("--e2e-exclude-ci-tier", default=[]) or [])
    if excluded_ci_tiers:
        cases = [
            case for case in cases if str(case.metadata.get("ci_tier", "")) not in excluded_ci_tiers
        ]

    multi_device_only = config.getoption("--multi-device-only", default=False)
    if multi_device_only:
        cases = [case for case in cases if is_multi_device_case(case)]
    else:
        cases = [case for case in cases if not is_multi_device_case(case)]

    if config.getoption("--e2e-core-only", default=False):
        cases = [case for case in cases if case.metadata.get("core", False)]

    return cases


def model_names_for_dir(
    *,
    config,
    model_dir: Path,
    case_matches_model: Callable,
    is_multi_device_case: Callable,
) -> list[str]:
    """Collect one pytest parameter for each selected model manifest."""
    models = [
        model
        for model in load_all_model_manifests(model_dir)
        if selected_testcases(
            model,
            config=config,
            case_matches_model=case_matches_model,
            is_multi_device_case=is_multi_device_case,
        )
    ]
    partition_id = (
        config.getoption("--e2e-partition-id", default=None)
        if config is not None
        else None
    )
    partition_size = (
        config.getoption("--e2e-partition-size", default=None)
        if config is not None
        else None
    )
    if partition_id is not None or partition_size is not None:
        if partition_id is None or partition_size is None:
            raise ValueError("--e2e-partition-id and --e2e-partition-size must be used together")
        names = partition_models(models, partition_size, partition_id)
    else:
        names = [model.name for model in models]
    return names or ["__no_models__"]


def _make_context(
    case: E2ECase,
    *,
    config,
    rebuild: bool,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
) -> RunContext:
    base_python = resolve_hf_python(config)
    profile_names = resolve_case_profile_names(case)
    profile_paths = resolve_case_python_profiles(case, base_python)
    return RunContext(
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
        engine_dir=resolve_engine_dir(config),
        model_plugin_dir=resolve_model_plugin_dir(config),
        rebuild=rebuild,
        verbose=config.getoption("verbose", default=0) > 0,
    )


def _case_with_platform_thresholds(case: E2ECase, platform: str) -> E2ECase:
    platform_key = str(platform or "").strip()
    if not platform_key:
        return case

    platform_overrides = case.metadata.get("platform_threshold_overrides", {})
    if not isinstance(platform_overrides, dict):
        return case

    overrides = platform_overrides.get(platform_key)
    if not isinstance(overrides, dict) or not overrides:
        return case

    return replace(
        case,
        threshold_overrides={
            **case.threshold_overrides,
            **overrides,
        },
    )


def _skip_detail(result) -> str:
    if result.determinism and "preflight" in result.determinism:
        failed = [detail for detail in result.determinism["preflight"] if not detail.get("passed")]
        return "; ".join(detail.get("message", "") for detail in failed)
    return ""


def _failure_message(case_name: str, result) -> str:
    failed_stages = [
        f"  {name} [{comparison.status}]: {comparison.message}"
        for name, comparison in result.stages.items()
        if comparison.status in (StageStatus.FAILED.value, StageStatus.ERROR.value)
    ]
    message = (
        f"E2E failed for {case_name} "
        f"(failure_type={result.failure_type}, "
        f"oracle_level={result.oracle_level}):\n"
    )
    return message + ("\n".join(failed_stages) if failed_stages else f"  status={result.status}")


def _run_testcase(
    case: E2ECase,
    *,
    config,
    xfail_reason: str,
    prepared_bundle: BundleResolution | None,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
) -> dict:
    ctx = _make_context(
        case,
        config=config,
        rebuild=False,
        resolve_hf_python=resolve_hf_python,
        resolve_artifacts_dir=resolve_artifacts_dir,
        resolve_binary=resolve_binary,
        resolve_ld_library_path=resolve_ld_library_path,
        resolve_engine_dir=resolve_engine_dir,
        resolve_model_plugin_dir=resolve_model_plugin_dir,
    )
    orchestrator = E2EOrchestrator()
    with model_plugin_dir_env(ctx.model_plugin_dir):
        result = orchestrator.run(case, ctx, prepared_bundle)

    if result.status == E2EStatus.SKIP.value:
        detail = _skip_detail(result)
        return {
            "name": case.name,
            "status": "skip",
            "message": (
                f"Case {case.name} skipped: {detail}" if detail else f"Case {case.name} skipped"
            ),
            "failure_type": result.failure_type,
        }
    if result.status == E2EStatus.PASS.value:
        return {"name": case.name, "status": "pass", "message": ""}

    failure_message = _failure_message(case.name, result)
    return {
        "name": case.name,
        "status": "xfail" if xfail_reason else "fail",
        "message": (f"{xfail_reason}\n{failure_message}" if xfail_reason else failure_message),
        "failure_type": result.failure_type,
    }


def _format_outcomes(model_name: str, outcomes: list[dict], header: str) -> str:
    lines = [f"{header} for {model_name}:"]
    for outcome in outcomes:
        line = f"  {outcome['name']} [{outcome['status']}]"
        failure_type = outcome.get("failure_type")
        if failure_type:
            line += f" failure_type={failure_type}"
        if outcome.get("message"):
            line += f": {outcome['message']}"
        lines.append(line)
    return "\n".join(lines)


def run_model_e2e(
    *,
    model_name: str,
    request,
    model_dir: Path,
    load_waives: Callable,
    case_matches_model: Callable,
    is_multi_device_case: Callable,
    resolve_hf_python: Callable,
    resolve_artifacts_dir: Callable,
    resolve_binary: Callable,
    resolve_ld_library_path: Callable,
    resolve_engine_dir: Callable,
    resolve_model_plugin_dir: Callable,
    model_plugin_dir_env: Callable,
) -> None:
    """Build one model once, then execute all selected child testcases."""
    if model_name == "__no_models__":
        pytest.skip("No model manifests found")

    model = get_model_by_name(model_name, model_dir)
    if model is None:
        pytest.fail(f"Model not found in {model_dir}: {model_name}")

    config = request.config
    platform = config.getoption("--e2e-platform", default="")
    cases = selected_testcases(
        model,
        config=config,
        case_matches_model=case_matches_model,
        is_multi_device_case=is_multi_device_case,
    )
    cases = [_case_with_platform_thresholds(case, platform) for case in cases]
    if not cases:
        pytest.skip(f"No selected testcases for model {model_name}")

    waives = load_waives(platform)
    outcomes: list[dict] = []
    runnable: list[tuple[E2ECase, str]] = []
    for case in cases:
        action, reason = waives.get(case.name, ("", ""))
        skip_reason = case.metadata.get("skip_reason", "")
        if action == "SKIP" or skip_reason:
            outcomes.append(
                {
                    "name": case.name,
                    "status": "skip",
                    "message": reason or skip_reason,
                }
            )
            continue
        runnable.append((case, reason if action == "XFAIL" else ""))

    if runnable:
        build_case = model.build_case
        build_ctx = _make_context(
            build_case,
            config=config,
            rebuild=config.getoption("--rebuild-engines", default=False),
            resolve_hf_python=resolve_hf_python,
            resolve_artifacts_dir=resolve_artifacts_dir,
            resolve_binary=resolve_binary,
            resolve_ld_library_path=resolve_ld_library_path,
            resolve_engine_dir=resolve_engine_dir,
            resolve_model_plugin_dir=resolve_model_plugin_dir,
        )
        preflight_passed = False
        for case, _xfail_reason in runnable:
            preflight_ctx = _make_context(
                case,
                config=config,
                rebuild=False,
                resolve_hf_python=resolve_hf_python,
                resolve_artifacts_dir=resolve_artifacts_dir,
                resolve_binary=resolve_binary,
                resolve_ld_library_path=resolve_ld_library_path,
                resolve_engine_dir=resolve_engine_dir,
                resolve_model_plugin_dir=resolve_model_plugin_dir,
            )
            preflight_passed, _details = run_preflight(case, preflight_ctx)
            if preflight_passed:
                break

        prepared: BundleResolution | None = None
        if preflight_passed:
            orchestrator = E2EOrchestrator()
            with model_plugin_dir_env(build_ctx.model_plugin_dir):
                resolution = orchestrator.resolve_model_bundle(build_case, build_ctx)
            if resolution.path is None:
                pytest.fail(
                    f"E2E build failed for {model_name}: "
                    f"{resolution.error or 'bundle was not produced'}"
                )
            prepared = resolution

        for case, xfail_reason in runnable:
            outcomes.append(
                _run_testcase(
                    case,
                    config=config,
                    xfail_reason=xfail_reason,
                    prepared_bundle=prepared,
                    resolve_hf_python=resolve_hf_python,
                    resolve_artifacts_dir=resolve_artifacts_dir,
                    resolve_binary=resolve_binary,
                    resolve_ld_library_path=resolve_ld_library_path,
                    resolve_engine_dir=resolve_engine_dir,
                    resolve_model_plugin_dir=resolve_model_plugin_dir,
                    model_plugin_dir_env=model_plugin_dir_env,
                )
            )
            if prepared is not None:
                prepared = BundleResolution(prepared.path)

    failures = [outcome for outcome in outcomes if outcome["status"] == "fail"]
    if failures:
        pytest.fail(_format_outcomes(model_name, outcomes, "Model E2E entry failed"))
    if any(outcome["status"] == "pass" for outcome in outcomes):
        return
    if any(outcome["status"] == "xfail" for outcome in outcomes):
        pytest.xfail(_format_outcomes(model_name, outcomes, "Model E2E entry xfailed"))
    pytest.skip(_format_outcomes(model_name, outcomes, "Model E2E entry skipped"))

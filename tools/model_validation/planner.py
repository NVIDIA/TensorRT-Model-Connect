# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan compilation for compatibility and future native validation paths."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    SCHEMA_VERSION,
    Assessment,
    CasePlan,
    CompatibilityMode,
    SuiteContract,
    ValidationPlan,
    ValidationRequest,
    WorkloadResolution,
    WorkloadSpec,
    digest_value,
)


class UnsupportedLegacyPerformanceError(ValueError):
    """Raised when Perf is requested for an unmigrated legacy task."""


def compile_legacy_plan(
    request: ValidationRequest,
    *,
    suite: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
) -> ValidationPlan:
    """Compile provenance for a run still executed by ``tools/task_eval.py``.

    Legacy plans intentionally defer ordered sample resolution to the existing
    runtime. They cannot request Performance Evaluation because legacy
    ``wall_ms`` fields do not share a comparable measurement scope.
    """

    if Assessment.PERFORMANCE in request.assessments:
        raise UnsupportedLegacyPerformanceError(
            "Performance Evaluation requires a native task adapter; legacy Task Eval "
            "timing is diagnostic only"
        )
    return _compile_plan(
        request,
        suite=suite,
        models=models,
        compatibility_mode=CompatibilityMode.LEGACY_TASK_EVAL,
        workload_resolution=WorkloadResolution.DEFERRED_TO_LEGACY_RUNTIME,
        task_adapter_kind="",
        task_adapter_version="",
    )


def compile_native_plan(
    request: ValidationRequest,
    *,
    suite: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    task_adapter_kind: str,
    task_adapter_version: str,
) -> ValidationPlan:
    """Compile a plan that will resolve its workload through a native adapter."""

    if not task_adapter_kind or not task_adapter_version:
        raise ValueError("Native plans require a task adapter kind and version")
    return _compile_plan(
        request,
        suite=suite,
        models=models,
        compatibility_mode=CompatibilityMode.NATIVE,
        workload_resolution=WorkloadResolution.DEFERRED_TO_NATIVE_PREPARE,
        task_adapter_kind=task_adapter_kind,
        task_adapter_version=task_adapter_version,
    )


def _compile_plan(
    request: ValidationRequest,
    *,
    suite: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    compatibility_mode: CompatibilityMode,
    workload_resolution: WorkloadResolution,
    task_adapter_kind: str,
    task_adapter_version: str,
) -> ValidationPlan:
    suite_id = str(suite.get("id", ""))
    if suite_id != request.suite_id:
        raise ValueError(f"Request suite {request.suite_id!r} does not match contract {suite_id!r}")
    dataset = suite.get("dataset", {})
    if not isinstance(dataset, Mapping):
        raise ValueError(f"Suite {suite_id!r} dataset must be a mapping")
    dataset_kind = str(dataset.get("kind", ""))
    if not dataset_kind:
        raise ValueError(f"Suite {suite_id!r} dataset.kind must be non-empty")
    if not models:
        raise ValueError(f"Suite {suite_id!r} plan requires at least one selected model")

    suite_contract = SuiteContract(
        suite_id=suite_id,
        user_contract=str(suite.get("user_contract", "")),
        dataset_kind=dataset_kind,
        contract_digest=digest_value(suite),
    )
    dataset_path = request.dataset_override
    if dataset_path is None and dataset.get("default_path"):
        dataset_path = str(dataset["default_path"])
    dataset_revision = dataset.get("source_revision") or dataset.get("sha256")
    workload = WorkloadSpec(
        dataset_path=dataset_path,
        dataset_revision=str(dataset_revision) if dataset_revision else None,
        limit=request.limit,
        seed=request.seed,
        ordered_sample_ids=(),
        resolution=workload_resolution,
    )
    cases = tuple(_case_plan(suite_id, model) for model in models)
    return ValidationPlan(
        schema_version=SCHEMA_VERSION,
        compatibility_mode=compatibility_mode,
        suite=suite_contract,
        request=request,
        workload=workload,
        task_adapter_kind=task_adapter_kind,
        task_adapter_version=task_adapter_version,
        cases=cases,
    )


def _case_plan(suite_id: str, model: Mapping[str, Any]) -> CasePlan:
    model_name = str(model.get("name", ""))
    if not model_name:
        raise ValueError("Selected models must define a non-empty name")
    return CasePlan(
        case_id=f"{suite_id}:{model_name}",
        model_name=model_name,
        hf_id=str(model.get("hf_id", "")),
        bundle=str(model.get("bundle", "")),
        runtime_strategy=str(model.get("runtime_strategy", "")),
        task_strategy=str(model.get("task_strategy", "")),
        reference_family=str(model.get("reference_family", "")),
        user_contract=str(model.get("user_contract", "")),
        manifest=str(model.get("manifest", "")),
        model_contract_digest=digest_value(model),
    )

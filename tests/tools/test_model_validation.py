# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools import task_eval
from tools.model_validation import (
    Assessment,
    CompatibilityMode,
    LegacyTaskEvalFacade,
    PlanIntegrityError,
    TaskAdapterRegistry,
    UnknownTaskAdapterError,
    UnsupportedLegacyPerformanceError,
    ValidationPlan,
    ValidationRequest,
    WorkloadResolution,
    compile_legacy_plan,
)


def _suite() -> dict:
    return {
        "id": "example_suite",
        "description": "Example compatibility suite",
        "user_contract": "example_output",
        "dataset": {
            "kind": "reranking_json",
            "default_path": "/datasets/example.json",
            "source_revision": "dataset-revision",
        },
        "generation": {"seed": 7, "max_new_tokens": 4},
        "gates": {"min_agreement": 1.0},
    }


def _models() -> list[dict]:
    return [
        {
            "name": "example-model",
            "hf_id": "org/example-model",
            "bundle": "example-model.trtfb",
            "runtime_strategy": "example_runtime",
            "task_strategy": "reranking",
            "reference_family": "example",
            "user_contract": "example_output",
            "manifest": "tests/e2e/models/example/MODEL.toml",
        }
    ]


def _request(*, limit: int = 10) -> ValidationRequest:
    return ValidationRequest(
        suite_id="example_suite",
        model_selectors=("example-model",),
        assessments=(Assessment.TASK, Assessment.FIDELITY),
        dataset_override="/datasets/override.json",
        limit=limit,
        seed=20260717,
    )


def test_compile_legacy_plan_is_deterministic_and_immutable() -> None:
    first = compile_legacy_plan(_request(), suite=_suite(), models=_models())
    reordered_suite = {
        "gates": {"min_agreement": 1.0},
        "generation": {"max_new_tokens": 4, "seed": 7},
        "dataset": {
            "source_revision": "dataset-revision",
            "default_path": "/datasets/example.json",
            "kind": "reranking_json",
        },
        "user_contract": "example_output",
        "description": "Example compatibility suite",
        "id": "example_suite",
    }
    reordered_model = dict(reversed(list(_models()[0].items())))
    second = compile_legacy_plan(_request(), suite=reordered_suite, models=[reordered_model])

    assert first.plan_digest == second.plan_digest
    assert first.compatibility_mode is CompatibilityMode.LEGACY_TASK_EVAL
    assert first.workload.resolution is WorkloadResolution.DEFERRED_TO_LEGACY_RUNTIME
    assert first.workload.ordered_sample_ids == ()
    assert first.to_dict()["plan_digest"] == first.plan_digest
    with pytest.raises(FrozenInstanceError):
        first.schema_version = "changed"  # type: ignore[misc]


def test_plan_digest_changes_when_execution_semantics_change() -> None:
    first = compile_legacy_plan(_request(limit=10), suite=_suite(), models=_models())
    second = compile_legacy_plan(_request(limit=11), suite=_suite(), models=_models())
    changed_models = _models()
    changed_models[0]["precision"] = "float16"
    third = compile_legacy_plan(_request(limit=10), suite=_suite(), models=changed_models)

    assert first.plan_digest != second.plan_digest
    assert first.plan_digest != third.plan_digest


def test_validation_request_requires_explicit_performance_profile() -> None:
    with pytest.raises(ValueError, match="performance_profile_id"):
        ValidationRequest(
            suite_id="example_suite",
            assessments=(Assessment.PERFORMANCE,),
        )


def test_legacy_plan_rejects_performance_assessment() -> None:
    request = ValidationRequest(
        suite_id="example_suite",
        assessments=(Assessment.PERFORMANCE,),
        performance_profile_id="gb300_single_stream_v1",
    )

    with pytest.raises(UnsupportedLegacyPerformanceError, match="native task adapter"):
        compile_legacy_plan(request, suite=_suite(), models=_models())


def test_facade_compiles_native_performance_plan_for_registered_adapter() -> None:
    class Adapter:
        kind = "time_series_csv"
        version = "1"

    plan = LegacyTaskEvalFacade().compile_eval_plan(
        suite=_suite(),
        models=_models(),
        performance_profile_id="etth1_process_e2e_observation_v1",
        native_adapter=Adapter(),
    )

    assert plan.compatibility_mode is CompatibilityMode.NATIVE
    assert plan.workload.resolution is WorkloadResolution.DEFERRED_TO_NATIVE_PREPARE
    assert plan.task_adapter_kind == "time_series_csv"
    assert plan.task_adapter_version == "1"
    assert Assessment.PERFORMANCE in plan.request.assessments
    assert plan.request.performance_profile_id == "etth1_process_e2e_observation_v1"


def test_facade_writes_and_validates_versioned_plan(tmp_path: Path) -> None:
    facade = LegacyTaskEvalFacade()
    plan = facade.compile_eval_plan(
        suite=_suite(),
        models=_models(),
        model_selectors=["example-model"],
        dataset_override="/datasets/override.json",
        limit=10,
        seed=20260717,
    )

    path = facade.write_plan(plan, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = ValidationPlan.from_dict(payload)

    assert path == tmp_path / "validation_plan.json"
    assert payload["schema_version"] == "1.0"
    assert payload["kind"] == "model_validation_plan"
    assert restored == plan

    payload["request"]["limit"] = 99
    with pytest.raises(PlanIntegrityError, match="digest"):
        ValidationPlan.from_dict(payload)


def test_eval_parser_accepts_performance_profile_configuration() -> None:
    args = task_eval.build_arg_parser().parse_args(
        [
            "eval",
            "--suite",
            "etth1_time_series_parity",
            "--performance-profile",
            "etth1_process_e2e_observation_v1",
            "--performance-profiles",
            "/profiles.yaml",
            "--performance-baseline",
            "/approved-baseline.json",
        ]
    )

    assert args.performance_profile == "etth1_process_e2e_observation_v1"
    assert args.performance_profiles == "/profiles.yaml"
    assert args.performance_baseline == "/approved-baseline.json"


def test_task_adapter_registry_fails_closed_on_duplicates_and_unknown_kinds() -> None:
    class Adapter:
        kind = "example"
        version = "1"

        def prepare(self, _work_dir, *, suite_id):
            return suite_id

        def fidelity_metrics(self, _reference, _candidate, *, gates):
            return dict(gates)

        def measurement_units(self):
            return ("sample",)

    registry = TaskAdapterRegistry()
    adapter = Adapter()

    registry.register(adapter)
    assert registry.get("example") is adapter
    assert registry.kinds() == ("example",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Adapter())
    with pytest.raises(UnknownTaskAdapterError, match="example-missing"):
        registry.get("example-missing")

    class IncompleteAdapter:
        kind = "incomplete"
        version = "1"

    with pytest.raises(TypeError, match="missing required methods"):
        registry.register(IncompleteAdapter())


def test_cmd_eval_emits_compatibility_plan_without_changing_legacy_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite()
    model = _models()[0]
    monkeypatch.setattr(task_eval, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(task_eval, "load_manifest_records", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        task_eval,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: [model],
    )
    monkeypatch.setattr(
        task_eval,
        "eval_one_model",
        lambda **_kwargs: {
            "suite": suite["id"],
            "model": model["name"],
            "hf_id": model["hf_id"],
            "mode": "reranking_parity",
            "status": "passed",
            "hf_reused": False,
            "bundle_built": False,
            "sample_pass_rate": 1.0,
            "mean_pairwise_ordering_agreement": 1.0,
            "min_pairwise_ordering_agreement": 1.0,
        },
    )
    args = argparse.Namespace(
        suites="",
        suite=suite["id"],
        dataset="/datasets/override.json",
        models_dir="",
        waives="",
        waive_platform="",
        include_waived=False,
        model=[],
        single_device_only=True,
        bundle="",
        work_root=str(tmp_path / "work"),
        engine_dir=str(tmp_path / "bundles"),
        limit=5,
        sample_seed=17,
        fail_fast=False,
        disable_model_process_isolation=True,
    )

    assert task_eval.cmd_eval(args) == 0

    suite_root = tmp_path / "work" / suite["id"]
    plan = ValidationPlan.from_dict(
        json.loads((suite_root / "validation_plan.json").read_text(encoding="utf-8"))
    )
    summary = json.loads((suite_root / "eval_summary.json").read_text(encoding="utf-8"))

    assert plan.request.limit == 5
    assert plan.request.seed == 17
    assert plan.workload.dataset_path == "/datasets/override.json"
    assert [case.model_name for case in plan.cases] == ["example-model"]
    assert summary == {
        "suite": "example_suite",
        "count": 1,
        "passed_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "model_process_isolation": False,
        "results": [
            {
                "suite": "example_suite",
                "model": "example-model",
                "hf_id": "org/example-model",
                "mode": "reranking_parity",
                "status": "passed",
                "hf_reused": False,
                "bundle_built": False,
                "sample_pass_rate": 1.0,
                "mean_pairwise_ordering_agreement": 1.0,
                "min_pairwise_ordering_agreement": 1.0,
            }
        ],
    }


def test_cmd_eval_emits_native_plan_when_etth1_perf_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = {
        "id": "etth1_time_series_parity",
        "user_contract": "time_series_prediction_parity",
        "dataset": {"kind": "time_series_csv", "default_path": "/data/ETTh1.csv"},
    }
    model = _models()[0]
    monkeypatch.setattr(task_eval, "load_suites", lambda *_args, **_kwargs: [suite])
    monkeypatch.setattr(task_eval, "load_manifest_records", lambda *_args, **_kwargs: [model])
    monkeypatch.setattr(
        task_eval,
        "selected_models_for_suite",
        lambda *_args, **_kwargs: [model],
    )
    monkeypatch.setattr(
        task_eval,
        "eval_one_model",
        lambda **_kwargs: {
            "suite": suite["id"],
            "model": model["name"],
            "hf_id": model["hf_id"],
            "status": "passed",
            "hf_reused": False,
            "bundle_built": False,
        },
    )
    monkeypatch.setattr(task_eval, "_format_result_line", lambda *_args: "passed")
    profile_path = Path(__file__).resolve().parents[1] / "task_eval" / "performance_profiles.yaml"
    args = argparse.Namespace(
        suites="",
        suite=suite["id"],
        dataset="/data/ETTh1.csv",
        models_dir="",
        waives="",
        waive_platform="",
        include_waived=False,
        model=[],
        single_device_only=True,
        bundle="",
        work_root=str(tmp_path / "work"),
        engine_dir=str(tmp_path / "bundles"),
        limit=5,
        sample_seed=17,
        fail_fast=False,
        disable_model_process_isolation=True,
        performance_profile="etth1_process_e2e_observation_v1",
        performance_profiles=str(profile_path),
        performance_baseline="",
    )

    assert task_eval.cmd_eval(args) == 0

    plan = ValidationPlan.from_dict(
        json.loads(
            (tmp_path / "work" / suite["id"] / "validation_plan.json").read_text(encoding="utf-8")
        )
    )
    assert plan.compatibility_mode is CompatibilityMode.NATIVE
    assert plan.task_adapter_kind == "time_series_csv"
    assert Assessment.PERFORMANCE in plan.request.assessments

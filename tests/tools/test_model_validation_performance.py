# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import task_eval
from tools.model_validation import AssessmentStatus
from tools.model_validation.adapters.time_series import TimeSeriesTaskAdapter
from tools.model_validation.measurement import (
    MeasurementEngine,
    MeasurementObservation,
    MeasurementOutput,
    MeasurementResult,
    MetricValue,
    NullMemoryMonitor,
    nvidia_smi_device_args,
    percentile_nearest_rank,
    validate_output_digests,
)
from tools.model_validation.performance import (
    PerformanceBaseline,
    EnvironmentFingerprint,
    PerformanceMode,
    evaluate_performance,
    load_baseline,
    load_performance_profile,
)


PROFILE_PATH = Path(__file__).resolve().parents[1] / "task_eval" / "performance_profiles.yaml"


def test_repository_etth1_profile_is_process_scoped_observation() -> None:
    profile = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1")

    assert profile.mode is PerformanceMode.OBSERVATION
    assert profile.scenario.scope.value == "process_e2e"
    assert profile.scenario.synchronization == "process_exit"
    assert profile.scenario.warmup_iterations == 1
    assert profile.scenario.measured_iterations == 5
    assert profile.scenario.process_repetitions == 3
    assert profile.scenario.require_output_match is True
    assert profile.backends == ("hf", "trtfb")
    assert "etth1_time_series_parity" in profile.supported_suite_ids
    assert "request_latency_ms.p50" in profile.required_metric_names
    assert "request_latency_ms.p95" in profile.required_metric_names

    blocking = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_blocking_v1")
    assert blocking.mode is PerformanceMode.BLOCKING
    assert blocking.profile_digest != profile.profile_digest


def test_profile_loader_rejects_invalid_measurement_counts(tmp_path: Path) -> None:
    path = tmp_path / "profiles.yaml"
    path.write_text(
        """
version: 1
profiles:
  invalid:
    mode: observation
    supported_suite_ids: [suite]
    supported_dataset_kinds: [time_series_csv]
    backends: [trtfb]
    scenario:
      scope: process_e2e
      synchronization: process_exit
      warmup_iterations: 0
      measured_iterations: 0
      process_repetitions: 1
      concurrency: 1
    metrics:
      required:
        - name: request_latency_ms.p50
          direction: lower
          max_regression_fraction: 0.05
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="measured_iterations"):
        load_performance_profile(path, "invalid")


@pytest.mark.parametrize(
    ("values", "percentile", "expected"),
    [
        ([1.0], 50, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.0),
        ([1.0, 2.0, 3.0, 4.0], 95, 4.0),
    ],
)
def test_nearest_rank_percentile(values: list[float], percentile: int, expected: float) -> None:
    assert percentile_nearest_rank(values, percentile) == expected


def test_nvidia_smi_sampling_respects_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-example,2")

    assert nvidia_smi_device_args() == ["--id", "GPU-example"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert nvidia_smi_device_args() == ["--id", "-1"]

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert nvidia_smi_device_args() == []


def test_measurement_engine_excludes_warmup_and_preserves_observations() -> None:
    profile = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1")
    profile = replace(
        profile,
        scenario=replace(
            profile.scenario,
            warmup_iterations=1,
            measured_iterations=3,
            process_repetitions=1,
        ),
    )
    # One warmup and three measured invocations, each with a start/end pair.
    clock_values = iter(
        [
            0,
            9_000_000,
            10_000_000,
            11_000_000,
            20_000_000,
            23_000_000,
            30_000_000,
            35_000_000,
        ]
    )
    engine = MeasurementEngine(clock_ns=lambda: next(clock_values))

    result = engine.measure(
        profile=profile,
        backend="trtfb",
        samples=("sample-a",),
        sample_id=lambda sample: sample,
        operation=lambda _invocation: MeasurementOutput(unit_count=2.0),
    )

    assert len(result.observations) == 4
    assert sum(observation.included for observation in result.observations) == 3
    assert result.metric("request_latency_ms.p50") == 3.0
    assert result.metric("request_latency_ms.p95") == 5.0
    assert result.metric("request_latency_ms.mean") == 3.0
    assert result.metric("request_latency_ms.mad") == 2.0
    assert result.metric("request_latency_ms.p50.repetition_mean") == 3.0
    assert result.metric("request_latency_ms.p50.repetition_cv") == 0.0
    assert result.metric("throughput_units_per_second") == pytest.approx(6 / 0.009)
    assert result.metric("error_rate") == 0.0


def test_measurement_engine_records_errors_without_hiding_them() -> None:
    profile = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1")
    profile = replace(
        profile,
        scenario=replace(
            profile.scenario,
            warmup_iterations=0,
            measured_iterations=2,
            process_repetitions=1,
        ),
    )
    clock_values = iter([0, 1_000_000, 2_000_000, 4_000_000])
    calls = 0

    def operation(_invocation):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("backend failed")
        return MeasurementOutput(unit_count=1.0)

    result = MeasurementEngine(clock_ns=lambda: next(clock_values)).measure(
        profile=profile,
        backend="trtfb",
        samples=("sample-a",),
        sample_id=lambda sample: sample,
        operation=operation,
    )

    assert result.metric("error_rate") == 0.5
    assert result.has_errors is True
    assert result.observations[1].error == "backend failed"


def test_measurement_output_must_match_correctness_stage_digest() -> None:
    result = MeasurementResult(
        schema_version=1,
        profile_id="profile",
        backend="trtfb",
        scope="process_e2e",
        observations=(
            MeasurementObservation(
                backend="trtfb",
                sample_id="sample-a",
                process_repetition=0,
                iteration=0,
                warmup=False,
                included=True,
                latency_ms=4.0,
                unit_count=1.0,
                peak_device_memory_mb=None,
                error="",
                output_metadata={"output_digest": "unexpected"},
            ),
        ),
        metrics=(MetricValue("error_rate", 0.0, "fraction"),),
    )

    validated = validate_output_digests(result, {"sample-a": "expected"})

    assert validated.has_errors is True
    assert validated.metric("error_rate") == 1.0
    assert validated.observations[0].unit_count == 0.0
    assert "differs" in validated.observations[0].error


def test_observation_profile_reports_without_an_approved_baseline() -> None:
    profile = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1")
    measurement = _measurement_result()

    assessment = evaluate_performance(
        profile=profile,
        measurement=measurement,
        comparison_key="comparison-key",
        correctness_passed=True,
        environment_compatible=True,
        baseline=None,
    )

    assert assessment.status is AssessmentStatus.OBSERVED
    assert assessment.baseline_status == "missing"
    assert assessment.blocking is False


def test_blocking_profile_requires_approved_comparable_baseline() -> None:
    profile = replace(
        load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1"),
        mode=PerformanceMode.BLOCKING,
    )
    measurement = _measurement_result()

    missing = evaluate_performance(
        profile=profile,
        measurement=measurement,
        comparison_key="comparison-key",
        correctness_passed=True,
        environment_compatible=True,
        baseline=None,
    )
    mismatched = evaluate_performance(
        profile=profile,
        measurement=measurement,
        comparison_key="comparison-key",
        correctness_passed=True,
        environment_compatible=True,
        baseline=PerformanceBaseline(
            schema_version=1,
            approved=True,
            profile_id=profile.profile_id,
            profile_digest=profile.profile_digest,
            comparison_key="different-key",
            metrics=measurement.metric_map,
        ),
    )

    assert missing.status is AssessmentStatus.BLOCKED
    assert mismatched.status is AssessmentStatus.BLOCKED
    assert mismatched.baseline_status == "not_comparable"


def test_baseline_loader_requires_candidate_eligibility_and_two_level_approval(
    tmp_path: Path,
) -> None:
    path = tmp_path / "baseline.json"
    payload = {
        "schema_version": 1,
        "approved": True,
        "eligible_for_approval": False,
        "backends": {
            "trtfb": {
                "schema_version": 1,
                "approved": True,
                "eligible_for_approval": True,
                "profile_id": "profile",
                "profile_digest": "digest",
                "comparison_key": "key",
                "metrics": {"request_latency_ms.p50": 1.0},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_baseline(path, "trtfb").approved is False

    payload["eligible_for_approval"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_baseline(path, "trtfb").approved is True

    payload["backends"]["trtfb"]["metrics"] = {
        "request_latency_ms.p50": float("nan")
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_baseline(path, "trtfb")


def test_blocking_profile_rejects_baseline_missing_required_metric() -> None:
    profile = replace(
        load_performance_profile(
            PROFILE_PATH, "etth1_process_e2e_observation_v1"
        ),
        mode=PerformanceMode.BLOCKING,
    )
    measurement = _measurement_result()
    baseline_metrics = dict(measurement.metric_map)
    baseline_metrics.pop("request_latency_ms.p95")
    baseline = PerformanceBaseline(
        schema_version=1,
        approved=True,
        profile_id=profile.profile_id,
        profile_digest=profile.profile_digest,
        comparison_key="comparison-key",
        metrics=baseline_metrics,
    )

    assessment = evaluate_performance(
        profile=profile,
        measurement=measurement,
        comparison_key="comparison-key",
        correctness_passed=True,
        environment_compatible=True,
        baseline=baseline,
    )

    assert assessment.status is AssessmentStatus.BLOCKED
    assert assessment.baseline_status == "not_comparable"


def test_blocking_profile_detects_latency_regression() -> None:
    profile = replace(
        load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1"),
        mode=PerformanceMode.BLOCKING,
    )
    measurement = _measurement_result()
    baseline_metrics = dict(measurement.metric_map)
    baseline_metrics["request_latency_ms.p50"] = 80.0
    baseline_metrics["request_latency_ms.p95"] = 90.0
    baseline = PerformanceBaseline(
        schema_version=1,
        approved=True,
        profile_id=profile.profile_id,
        profile_digest=profile.profile_digest,
        comparison_key="comparison-key",
        metrics=baseline_metrics,
    )

    assessment = evaluate_performance(
        profile=profile,
        measurement=measurement,
        comparison_key="comparison-key",
        correctness_passed=True,
        environment_compatible=True,
        baseline=baseline,
    )

    assert assessment.status is AssessmentStatus.FAILED
    assert any(not gate.passed for gate in assessment.gates)


def test_correctness_failure_blocks_measurement_assessment() -> None:
    profile = load_performance_profile(PROFILE_PATH, "etth1_process_e2e_observation_v1")

    assessment = evaluate_performance(
        profile=profile,
        measurement=_measurement_result(),
        comparison_key="comparison-key",
        correctness_passed=False,
        environment_compatible=True,
        baseline=None,
    )

    assert assessment.status is AssessmentStatus.BLOCKED
    assert "correctness" in assessment.reason


def test_time_series_adapter_prepares_stable_workload_and_reduces_fidelity(
    tmp_path: Path,
) -> None:
    prompts = [
        {"sample_id": "etth1-1", "inputs": {"branch_input": [1.0, 2.0]}},
        {"sample_id": "etth1-2", "inputs": {"branch_input": [2.0, 3.0]}},
    ]
    (tmp_path / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    adapter = TimeSeriesTaskAdapter()

    first = adapter.prepare(tmp_path, suite_id="etth1_time_series_parity")
    second = adapter.prepare(tmp_path, suite_id="etth1_time_series_parity")
    hf = {
        "responses": [
            {
                "sample_id": "etth1-1",
                "output_values": [1.0, 2.0],
                "output_shape": [2],
            },
            {
                "sample_id": "etth1-2",
                "output_values": [3.0, 4.0],
                "output_shape": [2],
            },
        ]
    }
    trtfb = json.loads(json.dumps(hf))

    fidelity = adapter.fidelity_metrics(
        hf,
        trtfb,
        gates={
            "max_relative_l2": 1e-6,
            "max_absolute_error": 1e-6,
            "min_sample_agreement_rate": 1.0,
        },
    )

    assert first == second
    assert first.ordered_sample_ids == ("etth1-1", "etth1-2")
    assert len(first.workload_digest) == 64
    assert fidelity["status"] == "passed"
    assert fidelity["sample_agreement_rate"] == 1.0
    assert adapter.prediction_output_digests(hf, label="HF") == (
        adapter.prediction_output_digests(trtfb, label="TRTFB")
    )


def test_time_series_adapter_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    row = {"sample_id": "duplicate", "inputs": {"branch_input": [1.0]}}
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        TimeSeriesTaskAdapter().prepare(tmp_path, suite_id="suite")


def test_time_series_process_perf_uses_outer_timer_and_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "etth1-1", "inputs": {"branch_input": [1.0, 2.0]}}) + "\n",
        encoding="utf-8",
    )
    correctness_predictions = {
        "responses": [
            {
                "sample_id": "etth1-1",
                "output_values": [1.0, 2.0],
                "output_shape": [2],
            }
        ]
    }
    for backend in ("hf", "trtfb"):
        (tmp_path / f"{backend}_predictions.json").write_text(
            json.dumps(correctness_predictions),
            encoding="utf-8",
        )
    template = SimpleNamespace(
        name="template",
        inputs={},
        stages=[SimpleNamespace(name="full_inference", required=True)],
        bundle="model.trtfb",
    )

    class Executor:
        def __init__(self) -> None:
            self.calls = 0

        def run_stage(self, _case, _stage, _context):
            self.calls += 1
            return SimpleNamespace(
                data={"output_field": [1.0, 2.0], "output_shape": [2]},
                metadata={"returncode": 0},
                # Deliberately absurd: the Measurement Engine must ignore it.
                timing_s=999.0,
            )

    reference = Executor()
    runner = Executor()
    monkeypatch.setattr(
        task_eval,
        "_load_time_series_task_eval_plugins",
        lambda _work_dir: (template, reference, runner),
    )
    args = SimpleNamespace(
        performance_profile="etth1_process_e2e_observation_v1",
        performance_profiles=str(PROFILE_PATH),
        performance_baseline="",
        hf_python="",
        trtmc_binary="build/trtmc",
        model_plugin_dir="",
    )
    suite = {
        "id": "etth1_time_series_parity",
        "dataset": {"kind": "time_series_csv"},
    }
    model = {
        "name": "chronos",
        "precision": "fp16",
        "runtime_strategy": "chronos_bolt_trt",
    }
    environment = EnvironmentFingerprint(
        gpu_name="NVIDIA GB300",
        driver="1",
        cuda_version="13",
        trt_version="11",
        hostname="runner",
        gpu_utilization_pct=0,
    )

    result = task_eval.run_time_series_performance_evaluation(
        args=args,
        suite=suite,
        model=model,
        work_dir=tmp_path,
        bundle_path=tmp_path / "model.trtfb",
        correctness_passed=True,
        measurement_engine=MeasurementEngine(memory_monitor_factory=NullMemoryMonitor),
        environment=environment,
    )

    assert result["status"] == "observed"
    assert result["measurement_scope"] == "process_e2e"
    assert reference.calls == 18
    assert runner.calls == 18
    assert result["backends"]["hf"]["metrics"]["request_latency_ms.p50"] < 1000
    assert result["backends"]["trtfb"]["metrics"]["error_rate"] == 0.0
    assert (tmp_path / "performance" / "measurements.jsonl").is_file()
    candidate = json.loads(
        (tmp_path / "performance" / "baseline_candidate.json").read_text(encoding="utf-8")
    )
    assert candidate["approved"] is False
    assert candidate["eligible_for_approval"] is True
    assert candidate["backends"]["trtfb"]["approved"] is False


def test_time_series_perf_blocks_without_running_when_correctness_failed(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "etth1-1", "inputs": {"branch_input": [1.0, 2.0]}}) + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        performance_profile="etth1_process_e2e_observation_v1",
        performance_profiles=str(PROFILE_PATH),
        performance_baseline="",
    )
    environment = EnvironmentFingerprint(
        gpu_name="NVIDIA GB300",
        driver="1",
        cuda_version="13",
        trt_version="11",
        hostname="runner",
        gpu_utilization_pct=0,
    )

    result = task_eval.run_time_series_performance_evaluation(
        args=args,
        suite={
            "id": "etth1_time_series_parity",
            "dataset": {"kind": "time_series_csv"},
        },
        model={"name": "chronos", "precision": "fp16"},
        work_dir=tmp_path,
        bundle_path=tmp_path / "model.trtfb",
        correctness_passed=False,
        environment=environment,
    )

    assert result["status"] == "blocked"
    assert result["backends"]["hf"]["observation_count"] == 0
    assert "correctness" in result["backends"]["hf"]["assessment"]["reason"]


def test_time_series_perf_blocks_output_drift_and_rejects_baseline_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"sample_id": "etth1-1", "inputs": {"branch_input": [1.0, 2.0]}}) + "\n",
        encoding="utf-8",
    )
    predictions = {
        "responses": [
            {
                "sample_id": "etth1-1",
                "output_values": [1.0, 2.0],
                "output_shape": [2],
            }
        ]
    }
    for backend in ("hf", "trtfb"):
        (tmp_path / f"{backend}_predictions.json").write_text(
            json.dumps(predictions), encoding="utf-8"
        )
    template = SimpleNamespace(
        name="template",
        inputs={},
        stages=[SimpleNamespace(name="full_inference", required=True)],
        bundle="model.trtfb",
    )

    class DriftingExecutor:
        def run_stage(self, _case, _stage, _context):
            return SimpleNamespace(
                data={"output_field": [9.0, 9.0], "output_shape": [2]},
                metadata={"returncode": 0},
                timing_s=0.0,
            )

    monkeypatch.setattr(
        task_eval,
        "_load_time_series_task_eval_plugins",
        lambda _work_dir: (template, DriftingExecutor(), DriftingExecutor()),
    )
    args = SimpleNamespace(
        performance_profile="etth1_process_e2e_observation_v1",
        performance_profiles=str(PROFILE_PATH),
        performance_baseline="",
        hf_python="",
        trtmc_binary="build/trtmc",
        model_plugin_dir="",
    )
    environment = EnvironmentFingerprint(
        gpu_name="NVIDIA GB300",
        driver="1",
        cuda_version="13",
        trt_version="11",
        hostname="runner",
        gpu_utilization_pct=0,
    )

    result = task_eval.run_time_series_performance_evaluation(
        args=args,
        suite={
            "id": "etth1_time_series_parity",
            "dataset": {"kind": "time_series_csv"},
        },
        model={"name": "chronos", "precision": "fp16"},
        work_dir=tmp_path,
        bundle_path=tmp_path / "model.trtfb",
        correctness_passed=True,
        measurement_engine=MeasurementEngine(memory_monitor_factory=NullMemoryMonitor),
        environment=environment,
    )

    assert result["status"] == "blocked"
    assert result["backends"]["hf"]["metrics"]["error_rate"] == 1.0
    candidate = json.loads(
        (tmp_path / "performance" / "baseline_candidate.json").read_text(encoding="utf-8")
    )
    assert candidate["eligible_for_approval"] is False
    assert candidate["backends"]["trtfb"]["eligible_for_approval"] is False


def _measurement_result():
    from tools.model_validation.measurement import MeasurementResult, MetricValue

    return MeasurementResult(
        schema_version=1,
        profile_id="etth1_process_e2e_observation_v1",
        backend="trtfb",
        scope="process_e2e",
        observations=(),
        metrics=(
            MetricValue("request_latency_ms.p50", 100.0, "ms"),
            MetricValue("request_latency_ms.p95", 120.0, "ms"),
            MetricValue("throughput_units_per_second", 10.0, "sample/s"),
            MetricValue("error_rate", 0.0, "fraction"),
        ),
    )

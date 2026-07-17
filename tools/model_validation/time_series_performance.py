# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ETTh1 process-scoped Performance Evaluation orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def run_time_series_performance_evaluation(
    *,
    args: argparse.Namespace,
    suite: dict[str, Any],
    model: dict[str, Any],
    work_dir: Path,
    bundle_path: Path,
    correctness_passed: bool,
    measurement_engine: Any = None,
    environment: Any = None,
) -> dict[str, Any]:
    """Run explicit process-scoped Perf without consuming legacy ``wall_ms``."""
    from tools import task_eval as legacy_task_eval
    from tests.e2e_harness.contracts import RunContext
    from tools.model_validation.adapters import TimeSeriesTaskAdapter
    from tools.model_validation.contracts import AssessmentStatus, digest_value
    from tools.model_validation.measurement import (
        MeasurementEngine,
        MeasurementOutput,
        MeasurementResult,
        NvidiaSmiMemoryMonitor,
        NullMemoryMonitor,
        validate_output_digests,
    )
    from tools.model_validation.performance import (
        check_environment,
        detect_environment,
        evaluate_performance,
        load_baseline,
        load_performance_profile,
    )

    profile_id = str(getattr(args, "performance_profile", "") or "")
    profile = load_performance_profile(
        Path(getattr(args, "performance_profiles", legacy_task_eval.DEFAULT_PERFORMANCE_PROFILES)),
        profile_id,
    )
    dataset_kind = str(suite.get("dataset", {}).get("kind", ""))
    profile.validate_target(suite_id=str(suite["id"]), dataset_kind=dataset_kind)
    adapter = TimeSeriesTaskAdapter()
    workload = adapter.prepare(work_dir, suite_id=str(suite["id"]))
    fingerprint = environment or detect_environment()
    environment_compatible, environment_reasons = check_environment(
        profile.environment, fingerprint
    )
    environment_class = (
        profile.environment.compatibility_class
        if environment_compatible
        else "unmatched-" + digest_value(fingerprint.to_dict())[:12]
    )
    performance_dir = work_dir / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    if measurement_engine is None:
        needs_memory = any(
            policy.name == "peak_device_memory_mb" for policy in profile.metric_policies
        )
        measurement_engine = MeasurementEngine(
            memory_monitor_factory=(NvidiaSmiMemoryMonitor if needs_memory else NullMemoryMonitor)
        )
    template = reference = runner = None
    if correctness_passed:
        template, reference, runner = legacy_task_eval._load_time_series_task_eval_plugins(work_dir)
    stage = (
        legacy_task_eval._time_series_full_inference_stage(template)
        if template is not None
        else None
    )

    measurements: dict[str, Any] = {}
    assessments: dict[str, Any] = {}
    comparison_keys: dict[str, str] = {}
    execution_digests: dict[str, str] = {}
    measurement_digests: dict[str, str] = {}
    measurement_result_digests: dict[str, str] = {}
    baseline_path = str(getattr(args, "performance_baseline", "") or "")
    bundle_identity: dict[str, Any] = {
        "filename": bundle_path.name,
        "exists": bundle_path.is_file(),
    }
    if bundle_path.is_file():
        bundle_stat = bundle_path.stat()
        bundle_identity.update(
            {"size_bytes": bundle_stat.st_size, "mtime_ns": bundle_stat.st_mtime_ns}
        )
    for backend in profile.backends:
        comparison_key = digest_value(
            {
                "workload_digest": workload.workload_digest,
                "logical_model_id": str(model["name"]),
                "backend": backend,
                "precision": str(model.get("precision", "")),
                "profile_id": profile.profile_id,
                "profile_digest": profile.profile_digest,
                "environment_class": environment_class,
                "task_adapter": f"{adapter.kind}:{adapter.version}",
            }
        )
        comparison_keys[backend] = comparison_key
        execution_identity = {
            "workload_digest": workload.workload_digest,
            "model_contract_digest": digest_value(model),
            "logical_model_id": str(model["name"]),
            "backend": backend,
            "precision": str(model.get("precision", "")),
            "runtime_strategy": str(model.get("runtime_strategy", "")),
            "bundle": bundle_identity if backend == "trtfb" else None,
            "environment": fingerprint.to_dict(),
            "task_adapter": f"{adapter.kind}:{adapter.version}",
        }
        execution_digests[backend] = digest_value(execution_identity)
        measurement_digests[backend] = digest_value(
            {
                "execution_digest": execution_digests[backend],
                "scenario": profile.scenario.to_dict(),
            }
        )
        if correctness_passed:
            operation = _time_series_performance_operation(
                backend=backend,
                template=template,
                executor=reference if backend == "hf" else runner,
                stage=stage,
                model=model,
                args=args,
                bundle_path=bundle_path,
                performance_dir=performance_dir,
                run_context_type=RunContext,
                measurement_output_type=MeasurementOutput,
                legacy_api=legacy_task_eval,
            )
            measurement = measurement_engine.measure(
                profile=profile,
                backend=backend,
                samples=workload.samples,
                sample_id=lambda sample: sample.sample_id,
                operation=operation,
            )
            if profile.scenario.require_output_match:
                prediction_path = work_dir / f"{backend}_predictions.json"
                prediction_data = json.loads(prediction_path.read_text(encoding="utf-8"))
                if not isinstance(prediction_data, dict):
                    raise ValueError(f"{prediction_path} must contain a prediction object")
                expected_digests = adapter.prediction_output_digests(
                    prediction_data,
                    label=backend.upper(),
                )
                if set(expected_digests) != set(workload.ordered_sample_ids):
                    raise ValueError(
                        f"{backend.upper()} correctness outputs do not match "
                        "the prepared performance workload"
                    )
                measurement = validate_output_digests(measurement, expected_digests)
        else:
            measurement = MeasurementResult(
                schema_version=1,
                profile_id=profile.profile_id,
                backend=backend,
                scope=profile.scenario.scope.value,
                observations=(),
                metrics=(),
            )
        baseline = load_baseline(Path(baseline_path), backend) if baseline_path else None
        assessment = evaluate_performance(
            profile=profile,
            measurement=measurement,
            comparison_key=comparison_key,
            correctness_passed=correctness_passed,
            environment_compatible=environment_compatible,
            baseline=baseline,
        )
        measurements[backend] = measurement
        measurement_result_digests[backend] = digest_value(measurement.to_dict())
        assessments[backend] = assessment
        (performance_dir / f"{backend}_measurement.json").write_text(
            json.dumps(measurement.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    with (performance_dir / "measurements.jsonl").open("w", encoding="utf-8") as stream:
        for backend in profile.backends:
            for observation in measurements[backend].observations:
                stream.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")

    statuses = [assessment.status for assessment in assessments.values()]
    if any(status is AssessmentStatus.FAILED for status in statuses):
        overall_status = AssessmentStatus.FAILED
    elif any(status is AssessmentStatus.BLOCKED for status in statuses):
        overall_status = AssessmentStatus.BLOCKED
    elif statuses and all(status is AssessmentStatus.PASSED for status in statuses):
        overall_status = AssessmentStatus.PASSED
    else:
        overall_status = AssessmentStatus.OBSERVED
    metric_maps = {backend: measurements[backend].metric_map for backend in profile.backends}
    speedup: dict[str, float] = {}
    if "hf" in metric_maps and "trtfb" in metric_maps:
        for name in ("request_latency_ms.p50", "request_latency_ms.p95"):
            hf_value = metric_maps["hf"].get(name)
            trtfb_value = metric_maps["trtfb"].get(name)
            if hf_value is not None and trtfb_value and trtfb_value > 0:
                speedup[name] = hf_value / trtfb_value
        hf_throughput = metric_maps["hf"].get("throughput_units_per_second")
        trtfb_throughput = metric_maps["trtfb"].get("throughput_units_per_second")
        if hf_throughput and hf_throughput > 0 and trtfb_throughput is not None:
            speedup["throughput_units_per_second"] = trtfb_throughput / hf_throughput

    result = {
        "schema_version": 1,
        "status": overall_status.value,
        "mode": profile.mode.value,
        "profile_id": profile.profile_id,
        "profile_digest": profile.profile_digest,
        "measurement_scope": profile.scenario.scope.value,
        "workload": {
            "digest": workload.workload_digest,
            "sample_count": len(workload.samples),
            "ordered_sample_ids": list(workload.ordered_sample_ids),
            "adapter_kind": adapter.kind,
            "adapter_version": adapter.version,
        },
        "environment": {
            **fingerprint.to_dict(),
            "compatibility_class": environment_class,
            "compatible": environment_compatible,
            "reasons": list(environment_reasons),
        },
        "backends": {
            backend: {
                "comparison_key": comparison_keys[backend],
                "execution_digest": execution_digests[backend],
                "measurement_digest": measurement_digests[backend],
                "measurement_result_digest": measurement_result_digests[backend],
                "metrics": metric_maps[backend],
                "assessment": assessments[backend].to_dict(),
                "observation_count": len(measurements[backend].observations),
            }
            for backend in profile.backends
        },
        "speedup": speedup,
    }
    (performance_dir / "performance_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    backend_eligibility = {
        backend: (
            correctness_passed
            and environment_compatible
            and not measurements[backend].has_errors
            and all(
                metric in measurements[backend].metric_map
                for metric in profile.required_metric_names
            )
        )
        for backend in profile.backends
    }
    baseline_candidate = {
        "schema_version": 1,
        "approved": False,
        "eligible_for_approval": all(backend_eligibility.values()),
        "profile_id": profile.profile_id,
        "profile_digest": profile.profile_digest,
        "workload_digest": workload.workload_digest,
        "model": str(model["name"]),
        "environment_class": environment_class,
        "backends": {
            backend: {
                "schema_version": 1,
                "approved": False,
                "eligible_for_approval": backend_eligibility[backend],
                "profile_id": profile.profile_id,
                "profile_digest": profile.profile_digest,
                "comparison_key": comparison_keys[backend],
                "execution_digest": execution_digests[backend],
                "measurement_digest": measurement_digests[backend],
                "source_measurement_result_digest": measurement_result_digests[backend],
                "metrics": metric_maps[backend],
            }
            for backend in profile.backends
        },
    }
    (performance_dir / "baseline_candidate.json").write_text(
        json.dumps(baseline_candidate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _time_series_performance_operation(
    *,
    backend: str,
    template: Any,
    executor: Any,
    stage: Any,
    model: dict[str, Any],
    args: argparse.Namespace,
    bundle_path: Path,
    performance_dir: Path,
    run_context_type: Any,
    measurement_output_type: Any,
    legacy_api: Any,
) -> Any:
    if backend not in {"hf", "trtfb"}:
        raise ValueError(f"Unsupported time-series performance backend {backend!r}")

    def operation(invocation: Any) -> Any:
        sample = invocation.sample
        prompt_row = {"sample_id": sample.sample_id, "inputs": dict(sample.inputs)}
        case = legacy_api._time_series_case_for_request(template, prompt_row, 0)
        phase = "warmup" if invocation.warmup else "measured"
        artifact_dir = (
            performance_dir
            / backend
            / f"repetition_{invocation.process_repetition:02d}"
            / phase
            / f"iteration_{invocation.iteration:03d}"
            / sample.sample_id
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if backend == "hf":
            reference_python = legacy_api.model_reference_python(
                model,
                str(getattr(args, "hf_python", "") or sys.executable),
            )
            context = run_context_type(
                case=case,
                artifacts_dir=str(artifact_dir),
                hf_python=reference_python,
                reference_python=reference_python,
            )
            source = "hf"
        else:
            case.bundle = bundle_path.name
            context = run_context_type(
                case=case,
                artifacts_dir=str(artifact_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            source = "trtfb"
        output = executor.run_stage(case, stage, context)
        response = legacy_api._time_series_response(case=case, source=source, output=output)
        if int(response.get("returncode", 0)) != 0:
            raise RuntimeError(
                f"{source} performance invocation failed for {sample.sample_id}: "
                f"returncode={response['returncode']}"
            )
        from tools.model_validation.contracts import digest_value

        output_digest = digest_value(
            {
                "output_values": response["output_values"],
                "output_shape": response["output_shape"],
            }
        )
        return measurement_output_type(
            unit_count=1.0,
            metadata={
                "output_digest": output_digest,
                "output_numel": len(response["output_values"]),
            },
        )

    return operation

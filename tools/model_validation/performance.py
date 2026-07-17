# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance profiles, environment identity, baselines, and gate evaluation."""

from __future__ import annotations

import json
import math
import socket
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .contracts import AssessmentStatus, digest_value
from .measurement import MeasurementResult, nvidia_smi_device_args


PERFORMANCE_PROFILE_SCHEMA_VERSION = 1
PERFORMANCE_BASELINE_SCHEMA_VERSION = 1


class PerformanceMode(str, Enum):
    DISABLED = "disabled"
    OBSERVATION = "observation"
    BLOCKING = "blocking"


class MeasurementScope(str, Enum):
    PROCESS_E2E = "process_e2e"
    WARM_SESSION = "warm_session"


class MetricDirection(str, Enum):
    LOWER = "lower"
    HIGHER = "higher"


@dataclass(frozen=True)
class EnvironmentPolicy:
    compatibility_class: str
    gpu_name_pattern: str
    exclusive_gpu: bool
    max_background_gpu_utilization_pct: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility_class": self.compatibility_class,
            "gpu_name_pattern": self.gpu_name_pattern,
            "exclusive_gpu": self.exclusive_gpu,
            "max_background_gpu_utilization_pct": (self.max_background_gpu_utilization_pct),
        }


@dataclass(frozen=True)
class MeasurementScenario:
    scope: MeasurementScope
    synchronization: str
    warmup_iterations: int
    measured_iterations: int
    process_repetitions: int
    concurrency: int
    quantile_method: str
    require_output_match: bool

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be positive")
        if self.process_repetitions <= 0:
            raise ValueError("process_repetitions must be positive")
        if self.concurrency != 1:
            raise ValueError("Only concurrency=1 is currently supported")
        if self.quantile_method != "nearest_rank":
            raise ValueError("Only quantile_method=nearest_rank is supported")
        if self.scope is MeasurementScope.PROCESS_E2E and self.synchronization != "process_exit":
            raise ValueError("process_e2e scope requires synchronization=process_exit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "synchronization": self.synchronization,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "process_repetitions": self.process_repetitions,
            "concurrency": self.concurrency,
            "quantile_method": self.quantile_method,
            "require_output_match": self.require_output_match,
        }


@dataclass(frozen=True)
class MetricPolicy:
    name: str
    direction: MetricDirection
    required: bool
    max_regression_fraction: float | None = None
    absolute_max: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Performance metric name must be non-empty")
        if self.max_regression_fraction is not None and not (
            0.0 <= self.max_regression_fraction < 1.0
        ):
            raise ValueError("max_regression_fraction must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "required": self.required,
            "max_regression_fraction": self.max_regression_fraction,
            "absolute_max": self.absolute_max,
        }


@dataclass(frozen=True)
class PerformanceProfile:
    schema_version: int
    profile_id: str
    description: str
    mode: PerformanceMode
    supported_suite_ids: tuple[str, ...]
    supported_dataset_kinds: tuple[str, ...]
    backends: tuple[str, ...]
    environment: EnvironmentPolicy
    scenario: MeasurementScenario
    metric_policies: tuple[MetricPolicy, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported performance profile schema {self.schema_version}")
        if not self.profile_id:
            raise ValueError("Performance profile ID must be non-empty")
        if not self.supported_suite_ids or not self.supported_dataset_kinds:
            raise ValueError("Performance profile must declare supported suites and kinds")
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("Performance profile backends must be non-empty and unique")
        names = [policy.name for policy in self.metric_policies]
        if len(set(names)) != len(names):
            raise ValueError("Performance metric policies must have unique names")

    @property
    def required_metric_names(self) -> tuple[str, ...]:
        return tuple(policy.name for policy in self.metric_policies if policy.required)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "description": self.description,
            "mode": self.mode.value,
            "supported_suite_ids": list(self.supported_suite_ids),
            "supported_dataset_kinds": list(self.supported_dataset_kinds),
            "backends": list(self.backends),
            "environment": self.environment.to_dict(),
            "scenario": self.scenario.to_dict(),
            "metric_policies": [policy.to_dict() for policy in self.metric_policies],
        }

    @property
    def profile_digest(self) -> str:
        return digest_value(self.to_dict())

    def validate_target(self, *, suite_id: str, dataset_kind: str) -> None:
        if suite_id not in self.supported_suite_ids:
            raise ValueError(
                f"Performance profile {self.profile_id!r} does not support suite {suite_id!r}"
            )
        if dataset_kind not in self.supported_dataset_kinds:
            raise ValueError(
                f"Performance profile {self.profile_id!r} does not support "
                f"dataset kind {dataset_kind!r}"
            )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    gpu_name: str
    driver: str
    cuda_version: str
    trt_version: str
    hostname: str
    gpu_utilization_pct: int | None
    gpu_temperature_c: float | None = None
    gpu_power_draw_w: float | None = None
    gpu_performance_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "driver": self.driver,
            "cuda_version": self.cuda_version,
            "trt_version": self.trt_version,
            "hostname": self.hostname,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpu_temperature_c": self.gpu_temperature_c,
            "gpu_power_draw_w": self.gpu_power_draw_w,
            "gpu_performance_state": self.gpu_performance_state,
        }


def detect_environment() -> EnvironmentFingerprint:
    gpu_name = ""
    driver = ""
    utilization: int | None = None
    temperature: float | None = None
    power_draw: float | None = None
    performance_state = ""
    nvidia_smi = ["nvidia-smi", *nvidia_smi_device_args()]
    nvidia_smi.extend(
        [
            "--query-gpu=name,driver_version,utilization.gpu,temperature.gpu,power.draw,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        process = subprocess.run(
            nvidia_smi,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if process.returncode == 0 and process.stdout.strip():
            parts = [part.strip() for part in process.stdout.splitlines()[0].split(",")]
            gpu_name = parts[0] if parts else ""
            driver = parts[1] if len(parts) > 1 else ""
            utilization_value = _optional_float(parts[2]) if len(parts) > 2 else None
            utilization = (
                int(utilization_value) if utilization_value is not None else None
            )
            temperature = _optional_float(parts[3]) if len(parts) > 3 else None
            power_draw = _optional_float(parts[4]) if len(parts) > 4 else None
            performance_state = parts[5] if len(parts) > 5 else ""
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    cuda_version = ""
    try:
        process = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in process.stdout.splitlines():
            if "release" in line.lower():
                cuda_version = line.lower().split("release", 1)[1].split(",", 1)[0].strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    trt_version = ""
    try:
        import tensorrt

        trt_version = str(tensorrt.__version__)
    except Exception:
        pass
    return EnvironmentFingerprint(
        gpu_name=gpu_name,
        driver=driver,
        cuda_version=cuda_version,
        trt_version=trt_version,
        hostname=socket.gethostname(),
        gpu_utilization_pct=utilization,
        gpu_temperature_c=temperature,
        gpu_power_draw_w=power_draw,
        gpu_performance_state=performance_state,
    )


def _optional_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def check_environment(
    policy: EnvironmentPolicy, fingerprint: EnvironmentFingerprint
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if policy.gpu_name_pattern.lower() not in fingerprint.gpu_name.lower():
        reasons.append(f"GPU {fingerprint.gpu_name!r} does not match {policy.gpu_name_pattern!r}")
    if (
        policy.exclusive_gpu
        and fingerprint.gpu_utilization_pct is not None
        and fingerprint.gpu_utilization_pct > policy.max_background_gpu_utilization_pct
    ):
        reasons.append(
            f"background GPU utilization exceeds {policy.max_background_gpu_utilization_pct}%"
        )
    return not reasons, tuple(reasons)


@dataclass(frozen=True)
class PerformanceBaseline:
    schema_version: int
    approved: bool
    profile_id: str
    profile_digest: str
    comparison_key: str
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "approved": self.approved,
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "comparison_key": self.comparison_key,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PerformanceBaseline:
        raw_metrics = payload.get("metrics", {})
        if not isinstance(raw_metrics, Mapping):
            raise ValueError("Performance baseline metrics must be a mapping")
        baseline = cls(
            schema_version=int(payload.get("schema_version", 0)),
            approved=bool(payload.get("approved", False)),
            profile_id=str(payload.get("profile_id", "")),
            profile_digest=str(payload.get("profile_digest", "")),
            comparison_key=str(payload.get("comparison_key", "")),
            metrics={str(name): float(value) for name, value in raw_metrics.items()},
        )
        if baseline.schema_version != PERFORMANCE_BASELINE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported performance baseline schema {baseline.schema_version}")
        non_finite = [
            name for name, value in baseline.metrics.items() if not math.isfinite(value)
        ]
        if non_finite:
            raise ValueError(
                "Performance baseline contains non-finite metrics: "
                + ", ".join(sorted(non_finite))
            )
        return baseline


@dataclass(frozen=True)
class MetricGateResult:
    metric: str
    current: float
    baseline: float | None
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current": self.current,
            "baseline": self.baseline,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PerformanceAssessment:
    status: AssessmentStatus
    blocking: bool
    reason: str
    baseline_status: str
    comparison_key: str
    gates: tuple[MetricGateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "blocking": self.blocking,
            "reason": self.reason,
            "baseline_status": self.baseline_status,
            "comparison_key": self.comparison_key,
            "gates": [gate.to_dict() for gate in self.gates],
        }


def evaluate_performance(
    *,
    profile: PerformanceProfile,
    measurement: MeasurementResult,
    comparison_key: str,
    correctness_passed: bool,
    environment_compatible: bool,
    baseline: PerformanceBaseline | None,
) -> PerformanceAssessment:
    blocking = profile.mode is PerformanceMode.BLOCKING
    if profile.mode is PerformanceMode.DISABLED:
        return PerformanceAssessment(
            AssessmentStatus.NOT_RUN, False, "profile disabled", "unused", comparison_key, ()
        )
    if not correctness_passed:
        return PerformanceAssessment(
            AssessmentStatus.BLOCKED,
            blocking,
            "correctness prerequisite failed",
            "unused",
            comparison_key,
            (),
        )
    if blocking and not environment_compatible:
        return PerformanceAssessment(
            AssessmentStatus.BLOCKED,
            True,
            "environment is not compatible with blocking profile",
            "unused",
            comparison_key,
            (),
        )
    metrics = measurement.metric_map
    missing = [name for name in profile.required_metric_names if name not in metrics]
    if missing:
        return PerformanceAssessment(
            AssessmentStatus.BLOCKED,
            blocking,
            f"required metrics unavailable: {', '.join(missing)}",
            "unused",
            comparison_key,
            (),
        )
    if measurement.has_errors or metrics.get("error_rate", 0.0) > 0.0:
        return PerformanceAssessment(
            AssessmentStatus.BLOCKED,
            blocking,
            "one or more performance observations failed",
            "unused",
            comparison_key,
            (),
        )

    baseline_status = "missing"
    comparable_baseline: PerformanceBaseline | None = None
    if baseline is not None:
        if not baseline.approved:
            baseline_status = "not_approved"
        elif baseline.profile_id != profile.profile_id:
            baseline_status = "not_comparable"
        elif baseline.profile_digest != profile.profile_digest:
            baseline_status = "not_comparable"
        elif baseline.comparison_key != comparison_key:
            baseline_status = "not_comparable"
        elif any(
            name not in baseline.metrics for name in profile.required_metric_names
        ):
            baseline_status = "not_comparable"
        else:
            baseline_status = "comparable"
            comparable_baseline = baseline

    gates = _evaluate_metric_gates(profile, metrics, comparable_baseline)
    if not blocking:
        return PerformanceAssessment(
            AssessmentStatus.OBSERVED,
            False,
            "observation profile does not affect the validation result",
            baseline_status,
            comparison_key,
            gates,
        )
    if comparable_baseline is None:
        return PerformanceAssessment(
            AssessmentStatus.BLOCKED,
            True,
            "blocking profile requires an approved comparable baseline",
            baseline_status,
            comparison_key,
            gates,
        )
    if any(not gate.passed for gate in gates):
        return PerformanceAssessment(
            AssessmentStatus.FAILED,
            True,
            "performance regression detected",
            baseline_status,
            comparison_key,
            gates,
        )
    return PerformanceAssessment(
        AssessmentStatus.PASSED,
        True,
        "performance is within approved limits",
        baseline_status,
        comparison_key,
        gates,
    )


def _evaluate_metric_gates(
    profile: PerformanceProfile,
    metrics: Mapping[str, float],
    baseline: PerformanceBaseline | None,
) -> tuple[MetricGateResult, ...]:
    results: list[MetricGateResult] = []
    for policy in profile.metric_policies:
        if policy.name not in metrics:
            continue
        current = metrics[policy.name]
        if policy.absolute_max is not None:
            passed = current <= policy.absolute_max
            results.append(
                MetricGateResult(
                    policy.name,
                    current,
                    None,
                    passed,
                    f"current <= {policy.absolute_max}",
                )
            )
        if (
            baseline is None
            or policy.max_regression_fraction is None
            or policy.name not in baseline.metrics
        ):
            continue
        base = float(baseline.metrics[policy.name])
        if policy.direction is MetricDirection.LOWER:
            limit = base * (1.0 + policy.max_regression_fraction)
            passed = current <= limit
            reason = f"current <= baseline * {1.0 + policy.max_regression_fraction:.3f}"
        else:
            limit = base * (1.0 - policy.max_regression_fraction)
            passed = current >= limit
            reason = f"current >= baseline * {1.0 - policy.max_regression_fraction:.3f}"
        results.append(MetricGateResult(policy.name, current, base, passed, reason))
    return tuple(results)


def load_performance_profile(path: Path, profile_id: str) -> PerformanceProfile:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is required by project
        raise RuntimeError("PyYAML is required for performance profiles") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    version = int(payload.get("version", 0))
    if version != PERFORMANCE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported performance profile file version {version}")
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError(f"{path}: profiles must be a mapping")
    raw = profiles.get(profile_id)
    if not isinstance(raw, Mapping):
        known = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(f"Unknown performance profile {profile_id!r}. Known: {known}")
    return _parse_profile(version, profile_id, raw)


def _parse_profile(version: int, profile_id: str, raw: Mapping[str, Any]) -> PerformanceProfile:
    environment = _require_mapping(raw.get("environment", {}), "environment")
    scenario = _require_mapping(raw.get("scenario", {}), "scenario")
    metrics = _require_mapping(raw.get("metrics", {}), "metrics")
    policies: list[MetricPolicy] = []
    for required, key in ((True, "required"), (False, "optional")):
        raw_policies = metrics.get(key, [])
        if not isinstance(raw_policies, list):
            raise ValueError(f"metrics.{key} must be a list")
        for item in raw_policies:
            mapping = _require_mapping(item, f"metrics.{key} entry")
            policies.append(
                MetricPolicy(
                    name=str(mapping.get("name", "")),
                    direction=MetricDirection(str(mapping.get("direction", ""))),
                    required=required,
                    max_regression_fraction=(
                        float(mapping["max_regression_fraction"])
                        if mapping.get("max_regression_fraction") is not None
                        else None
                    ),
                    absolute_max=(
                        float(mapping["absolute_max"])
                        if mapping.get("absolute_max") is not None
                        else None
                    ),
                )
            )
    return PerformanceProfile(
        schema_version=version,
        profile_id=profile_id,
        description=str(raw.get("description", "")),
        mode=PerformanceMode(str(raw.get("mode", ""))),
        supported_suite_ids=tuple(str(item) for item in raw.get("supported_suite_ids", [])),
        supported_dataset_kinds=tuple(str(item) for item in raw.get("supported_dataset_kinds", [])),
        backends=tuple(str(item) for item in raw.get("backends", [])),
        environment=EnvironmentPolicy(
            compatibility_class=str(environment.get("compatibility_class", "")),
            gpu_name_pattern=str(environment.get("gpu_name_pattern", "")),
            exclusive_gpu=bool(environment.get("exclusive_gpu", False)),
            max_background_gpu_utilization_pct=int(
                environment.get("max_background_gpu_utilization_pct", 100)
            ),
        ),
        scenario=MeasurementScenario(
            scope=MeasurementScope(str(scenario.get("scope", ""))),
            synchronization=str(scenario.get("synchronization", "")),
            warmup_iterations=int(scenario.get("warmup_iterations", 0)),
            measured_iterations=int(scenario.get("measured_iterations", 0)),
            process_repetitions=int(scenario.get("process_repetitions", 0)),
            concurrency=int(scenario.get("concurrency", 0)),
            quantile_method=str(scenario.get("quantile_method", "")),
            require_output_match=bool(scenario.get("require_output_match", False)),
        ),
        metric_policies=tuple(policies),
    )


def load_baseline(path: Path, backend: str) -> PerformanceBaseline:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Performance baseline file must contain a mapping")
    backends = payload.get("backends", {})
    if not isinstance(backends, Mapping) or not isinstance(backends.get(backend), Mapping):
        raise ValueError(f"Performance baseline does not contain backend {backend!r}")
    backend_payload = dict(backends[backend])
    backend_payload["approved"] = (
        bool(payload.get("approved", False))
        and bool(payload.get("eligible_for_approval", False))
        and bool(backend_payload.get("approved", False))
        and bool(backend_payload.get("eligible_for_approval", False))
    )
    return PerformanceBaseline.from_dict(backend_payload)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value

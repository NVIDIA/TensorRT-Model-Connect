# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurement engine with explicit warmup and raw observation retention."""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol, Sequence


MEASUREMENT_SCHEMA_VERSION = 1


def percentile_nearest_rank(values: Sequence[float], percentile: int) -> float:
    """Return a nearest-rank percentile with a stable, documented algorithm."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil(percentile / 100.0 * len(ordered))
    return ordered[rank - 1]


@dataclass(frozen=True)
class MeasurementInvocation:
    sample: Any
    sample_id: str
    process_repetition: int
    iteration: int
    warmup: bool


@dataclass(frozen=True)
class MeasurementOutput:
    """Normalized output returned by one backend invocation."""

    unit_count: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementObservation:
    backend: str
    sample_id: str
    process_repetition: int
    iteration: int
    warmup: bool
    included: bool
    latency_ms: float
    unit_count: float
    peak_device_memory_mb: float | None
    error: str
    output_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "sample_id": self.sample_id,
            "process_repetition": self.process_repetition,
            "iteration": self.iteration,
            "warmup": self.warmup,
            "included": self.included,
            "latency_ms": self.latency_ms,
            "unit_count": self.unit_count,
            "peak_device_memory_mb": self.peak_device_memory_mb,
            "error": self.error,
            "output_metadata": dict(self.output_metadata),
        }


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class MeasurementResult:
    schema_version: int
    profile_id: str
    backend: str
    scope: str
    observations: tuple[MeasurementObservation, ...]
    metrics: tuple[MetricValue, ...]

    @property
    def metric_map(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def metric(self, name: str) -> float:
        try:
            return self.metric_map[name]
        except KeyError as exc:
            raise KeyError(f"Measurement does not provide metric {name!r}") from exc

    @property
    def has_errors(self) -> bool:
        return any(observation.error for observation in self.observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "backend": self.backend,
            "scope": self.scope,
            "observations": [observation.to_dict() for observation in self.observations],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def validate_output_digests(
    measurement: MeasurementResult,
    expected_by_sample: Mapping[str, str],
) -> MeasurementResult:
    """Fail observations whose outputs differ from correctness-stage outputs."""

    validated: list[MeasurementObservation] = []
    for observation in measurement.observations:
        if observation.error:
            validated.append(observation)
            continue
        expected = expected_by_sample.get(observation.sample_id)
        actual = str(observation.output_metadata.get("output_digest", ""))
        if expected is None:
            error = (
                f"measurement sample is absent from correctness outputs: {observation.sample_id}"
            )
        elif not actual:
            error = "measurement output did not provide output_digest"
        elif actual != expected:
            error = "measurement output differs from correctness-stage output"
        else:
            validated.append(observation)
            continue
        validated.append(replace(observation, unit_count=0.0, error=error))
    return replace(
        measurement,
        observations=tuple(validated),
        metrics=_aggregate_metrics(validated),
    )


class MemoryMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> float | None: ...


class NullMemoryMonitor:
    def start(self) -> None:
        return None

    def stop(self) -> float | None:
        return None


class NvidiaSmiMemoryMonitor:
    """Poll maximum memory used on any visible GPU outside the timed window."""

    def __init__(
        self,
        *,
        poll_interval_s: float = 0.05,
        reader: Callable[[], float | None] | None = None,
    ) -> None:
        self._poll_interval_s = poll_interval_s
        self._reader = reader or _read_peak_gpu_memory_mib
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak: float | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._record(self._reader())
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._poll_interval_s * 4))
        self._record(self._reader())
        return self._peak

    def _poll(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            self._record(self._reader())

    def _record(self, value: float | None) -> None:
        if value is not None and (self._peak is None or value > self._peak):
            self._peak = value


def _read_peak_gpu_memory_mib() -> float | None:
    command = ["nvidia-smi", *nvidia_smi_device_args()]
    command.extend(
        [
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    try:
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if process.returncode != 0:
        return None
    try:
        values = [float(line.strip()) for line in process.stdout.splitlines() if line.strip()]
    except ValueError:
        return None
    return max(values) if values else None


def nvidia_smi_device_args() -> list[str]:
    """Select the same physical GPU exposed to the measured backend."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return []
    selector = visible.strip().split(",", 1)[0].strip()
    if not selector:
        selector = "-1"
    return ["--id", selector]


class MeasurementEngine:
    """Measure a backend callable without trusting backend-reported timing."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        memory_monitor_factory: Callable[[], MemoryMonitor] = NullMemoryMonitor,
    ) -> None:
        self._clock_ns = clock_ns
        self._memory_monitor_factory = memory_monitor_factory

    def measure(
        self,
        *,
        profile: Any,
        backend: str,
        samples: Sequence[Any],
        sample_id: Callable[[Any], str],
        operation: Callable[[MeasurementInvocation], MeasurementOutput],
    ) -> MeasurementResult:
        if not samples:
            raise ValueError("Performance measurement requires at least one sample")
        scenario = profile.scenario
        observations: list[MeasurementObservation] = []
        for repetition in range(scenario.process_repetitions):
            for iteration in range(scenario.warmup_iterations):
                observations.extend(
                    self._measure_iteration(
                        backend=backend,
                        samples=samples,
                        sample_id=sample_id,
                        operation=operation,
                        repetition=repetition,
                        iteration=iteration,
                        warmup=True,
                    )
                )
            for iteration in range(scenario.measured_iterations):
                observations.extend(
                    self._measure_iteration(
                        backend=backend,
                        samples=samples,
                        sample_id=sample_id,
                        operation=operation,
                        repetition=repetition,
                        iteration=iteration,
                        warmup=False,
                    )
                )
        return MeasurementResult(
            schema_version=MEASUREMENT_SCHEMA_VERSION,
            profile_id=profile.profile_id,
            backend=backend,
            scope=scenario.scope.value,
            observations=tuple(observations),
            metrics=_aggregate_metrics(observations),
        )

    def _measure_iteration(
        self,
        *,
        backend: str,
        samples: Sequence[Any],
        sample_id: Callable[[Any], str],
        operation: Callable[[MeasurementInvocation], MeasurementOutput],
        repetition: int,
        iteration: int,
        warmup: bool,
    ) -> list[MeasurementObservation]:
        rows: list[MeasurementObservation] = []
        for sample in samples:
            identifier = str(sample_id(sample))
            invocation = MeasurementInvocation(
                sample=sample,
                sample_id=identifier,
                process_repetition=repetition,
                iteration=iteration,
                warmup=warmup,
            )
            monitor = self._memory_monitor_factory()
            monitor.start()
            start_ns = self._clock_ns()
            output = MeasurementOutput(unit_count=0.0)
            error = ""
            try:
                output = operation(invocation)
                if not isinstance(output, MeasurementOutput):
                    raise TypeError("Measurement operation must return MeasurementOutput")
                if not math.isfinite(output.unit_count) or output.unit_count <= 0:
                    raise ValueError("Measurement output unit_count must be positive and finite")
            except Exception as exc:  # Record backend failures as observations.
                error = str(exc) or type(exc).__name__
            end_ns = self._clock_ns()
            peak_memory = monitor.stop()
            rows.append(
                MeasurementObservation(
                    backend=backend,
                    sample_id=identifier,
                    process_repetition=repetition,
                    iteration=iteration,
                    warmup=warmup,
                    included=not warmup,
                    latency_ms=max(0.0, (end_ns - start_ns) / 1_000_000.0),
                    unit_count=output.unit_count if not error else 0.0,
                    peak_device_memory_mb=peak_memory,
                    error=error,
                    output_metadata=dict(output.metadata) if not error else {},
                )
            )
        return rows


def _aggregate_metrics(
    observations: Sequence[MeasurementObservation],
) -> tuple[MetricValue, ...]:
    included = [observation for observation in observations if observation.included]
    successful = [observation for observation in included if not observation.error]
    latencies = [observation.latency_ms for observation in successful]
    metrics: list[MetricValue] = []
    if latencies:
        median = statistics.median(latencies)
        deviations = [abs(value - median) for value in latencies]
        metrics.extend(
            [
                MetricValue("request_latency_ms.mean", statistics.mean(latencies), "ms"),
                MetricValue(
                    "request_latency_ms.stddev",
                    statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
                    "ms",
                ),
                MetricValue("request_latency_ms.min", min(latencies), "ms"),
                MetricValue("request_latency_ms.max", max(latencies), "ms"),
                MetricValue(
                    "request_latency_ms.p50",
                    percentile_nearest_rank(latencies, 50),
                    "ms",
                ),
                MetricValue(
                    "request_latency_ms.p95",
                    percentile_nearest_rank(latencies, 95),
                    "ms",
                ),
                MetricValue("request_latency_ms.mad", statistics.median(deviations), "ms"),
            ]
        )
        elapsed_seconds = sum(latencies) / 1000.0
        metrics.append(
            MetricValue(
                "throughput_units_per_second",
                sum(observation.unit_count for observation in successful) / elapsed_seconds,
                "sample/s",
            )
        )
    memory_values = [
        observation.peak_device_memory_mb
        for observation in successful
        if observation.peak_device_memory_mb is not None
    ]
    if memory_values:
        metrics.append(MetricValue("peak_device_memory_mb", max(memory_values), "MiB"))
    repetition_p50s: list[float] = []
    for repetition in sorted({observation.process_repetition for observation in successful}):
        repetition_latencies = [
            observation.latency_ms
            for observation in successful
            if observation.process_repetition == repetition
        ]
        if repetition_latencies:
            repetition_p50s.append(percentile_nearest_rank(repetition_latencies, 50))
    if repetition_p50s:
        repetition_mean = statistics.mean(repetition_p50s)
        repetition_stddev = statistics.pstdev(repetition_p50s) if len(repetition_p50s) > 1 else 0.0
        metrics.extend(
            [
                MetricValue(
                    "request_latency_ms.p50.repetition_mean",
                    repetition_mean,
                    "ms",
                ),
                MetricValue(
                    "request_latency_ms.p50.repetition_stddev",
                    repetition_stddev,
                    "ms",
                ),
                MetricValue(
                    "request_latency_ms.p50.repetition_cv",
                    repetition_stddev / repetition_mean if repetition_mean > 0 else 0.0,
                    "fraction",
                ),
            ]
        )
    error_count = sum(bool(observation.error) for observation in included)
    error_rate = error_count / len(included) if included else 1.0
    metrics.append(MetricValue("error_rate", error_rate, "fraction"))
    return tuple(metrics)

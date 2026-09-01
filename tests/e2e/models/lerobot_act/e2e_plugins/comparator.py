# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict action, bounds, performance, and 50 Hz replay comparator."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.families.lerobot_act.plugin import ACTION_MAX, ACTION_MIN
from tests.e2e_harness.contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)

_ACTION_MIN = np.asarray(ACTION_MIN, dtype=np.float32)
_ACTION_MAX = np.asarray(ACTION_MAX, dtype=np.float32)
_OPERATORS = {
    "action_max_abs_error": "<=",
    "action_mean_abs_error": "<=",
    "action_rmse": "<=",
    "action_step_capacity_hz": ">=",
    "chunk_inference_p95_ms": "<=",
    "control_effective_hz": ">=",
    "control_p99_abs_jitter_ms": "<=",
    "control_missed_deadlines": "<=",
    "gpu_memory_delta_mib": "<=",
    "peak_resident_memory_mib": "<=",
    "startup_ms": "<=",
    "training_bounds_fraction": ">=",
}


def _metric(value: float, threshold: float, operator: str) -> MetricResult:
    passed = np.isfinite(value) and (value <= threshold if operator == "<=" else value >= threshold)
    return MetricResult(value=value, threshold=threshold, operator=operator, passed=passed)


class RobotActionChunkComparator:
    @property
    def task_strategy(self) -> str:
        return "robot_action_chunk"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        try:
            trt_actions = np.asarray(trt.data["actions"], dtype=np.float32)
            ref_actions = np.asarray(ref.data["actions"], dtype=np.float32)
            if trt_actions.shape != (100, 14) or ref_actions.shape != (100, 14):
                raise ValueError(
                    f"action shapes must both be (100, 14), got {trt_actions.shape} and {ref_actions.shape}"
                )
            if not np.isfinite(trt_actions).all() or not np.isfinite(ref_actions).all():
                raise ValueError("action chunks must be finite")
            summary = trt.data["summary"]
            if int(summary.get("num_actions", 0)) != 100 or int(summary.get("action_dim", 0)) != 14:
                raise ValueError("native summary violates the qualified action shape")
            if float(summary.get("control_frequency_hz", 0.0)) != 50.0:
                raise ValueError("native replay did not use the qualified 50 Hz control rate")
            for field in (
                "action_step_capacity_hz",
                "chunk_inference_p50_ms",
                "chunk_inference_p95_ms",
                "chunk_throughput_per_second",
                "gpu_memory_delta_mib",
                "gpu_memory_total_mib",
                "peak_resident_memory_mib",
                "startup_ms",
            ):
                value = float(summary.get(field, 0.0))
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"native summary has no positive finite {field}")
            for field in (
                "control_effective_hz",
                "control_p99_abs_jitter_ms",
                "control_missed_deadlines",
            ):
                value = float(summary[field])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"native summary has no non-negative finite {field}")
            missing = set(_OPERATORS) - set(threshold.metrics)
            if missing:
                raise ValueError(f"LeRobot ACT thresholds are missing {sorted(missing)}")
        except (KeyError, TypeError, ValueError) as error:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.ERROR.value,
                message=f"Invalid LeRobot ACT action contract: {error}",
            )

        delta = trt_actions.astype(np.float64) - ref_actions.astype(np.float64)
        in_bounds = np.logical_and(trt_actions >= _ACTION_MIN, trt_actions <= _ACTION_MAX)
        numeric = {
            "action_max_abs_error": float(np.max(np.abs(delta))),
            "action_mean_abs_error": float(np.mean(np.abs(delta))),
            "action_rmse": float(np.sqrt(np.mean(np.square(delta)))),
            "action_step_capacity_hz": float(summary["action_step_capacity_hz"]),
            "chunk_inference_p95_ms": float(summary["chunk_inference_p95_ms"]),
            "control_effective_hz": float(summary["control_effective_hz"]),
            "control_p99_abs_jitter_ms": float(summary["control_p99_abs_jitter_ms"]),
            "control_missed_deadlines": float(summary["control_missed_deadlines"]),
            "gpu_memory_delta_mib": float(summary["gpu_memory_delta_mib"]),
            "peak_resident_memory_mib": float(summary["peak_resident_memory_mib"]),
            "startup_ms": float(summary["startup_ms"]),
            "training_bounds_fraction": float(np.mean(in_bounds)),
        }
        metrics = {
            name: _metric(numeric[name], float(threshold.metrics[name]), operator)
            for name, operator in _OPERATORS.items()
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "all 100x14 action parity, training-bound, chunk-horizon, and 50 Hz control gates must pass"
            ),
            message=(
                "LeRobot ACT native recorded control qualification passed"
                if passed
                else "LeRobot ACT native recorded control qualification failed"
            ),
        )


comparator = RobotActionChunkComparator()

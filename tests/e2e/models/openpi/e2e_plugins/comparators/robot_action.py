# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical OpenPI comparator for action chunks and diagnostic stages."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import performance
from ..contracts import (
    CompareResult,
    MetricResult,
    StageOutput,
    StageSpec,
    StageStatus,
    ThresholdProfile,
)


def _flatten(value: Any) -> Iterable[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _flatten(item)
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"tensor value must be numeric, got {type(value).__name__}")
    yield float(value)


def _binary_values(descriptor: Mapping[str, Any]) -> Iterable[float]:
    path = Path(str(descriptor["path"]))
    payload = path.read_bytes()
    dtype = str(descriptor["dtype"])
    if dtype == "bool":
        return (float(value != 0) for value in payload)
    if dtype == "uint8":
        return (float(value) for value in payload)
    if dtype == "int32":
        return (float(item[0]) for item in struct.iter_unpack("<i", payload))
    if dtype == "int64":
        return (float(item[0]) for item in struct.iter_unpack("<q", payload))
    if dtype == "float16":
        return (float(item[0]) for item in struct.iter_unpack("<e", payload))
    if dtype == "float32":
        return (float(item[0]) for item in struct.iter_unpack("<f", payload))
    if dtype == "bfloat16":
        return (
            float(struct.unpack("<f", struct.pack("<I", item[0] << 16))[0])
            for item in struct.iter_unpack("<H", payload)
        )
    raise ValueError(f"unsupported tensor dtype {dtype!r}")


def _tensor_values(data: Mapping[str, Any], name: str) -> Iterable[float]:
    if name in data:
        return _flatten(data[name])
    tensors = data.get("tensors")
    if isinstance(tensors, Mapping) and name in tensors:
        return _flatten(tensors[name])
    files = data.get("tensor_files")
    if isinstance(files, Mapping) and isinstance(files.get(name), Mapping):
        return _binary_values(files[name])
    raise ValueError(f"stage output is missing tensor {name!r}")


def _numeric_error_metrics(actual: Iterable[float], expected: Iterable[float]) -> dict[str, float]:
    count = 0
    dot = 0.0
    actual_sq = 0.0
    expected_sq = 0.0
    squared_error = 0.0
    absolute_error_sum = 0.0
    max_abs = 0.0
    nonfinite = 0
    actual_iter = iter(actual)
    expected_iter = iter(expected)
    sentinel = object()
    while True:
        lhs = next(actual_iter, sentinel)
        rhs = next(expected_iter, sentinel)
        if lhs is sentinel or rhs is sentinel:
            if lhs is not sentinel or rhs is not sentinel:
                raise ValueError("tensor element counts differ")
            break
        lhs_value, rhs_value = float(lhs), float(rhs)
        count += 1
        if not math.isfinite(lhs_value) or not math.isfinite(rhs_value):
            nonfinite += 1
            continue
        error = abs(lhs_value - rhs_value)
        dot += lhs_value * rhs_value
        actual_sq += lhs_value * lhs_value
        expected_sq += rhs_value * rhs_value
        squared_error += error * error
        absolute_error_sum += error
        max_abs = max(max_abs, error)
    if count == 0:
        raise ValueError("cannot compare empty tensors")
    denominator = math.sqrt(actual_sq * expected_sq)
    cosine = dot / denominator if denominator > 0.0 else float(max_abs == 0.0)
    return {
        "count": float(count),
        "cosine": cosine,
        "rmse": math.sqrt(squared_error / count),
        "mae": absolute_error_sum / count,
        "max_abs": max_abs,
        "nonfinite": float(nonfinite),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty tensor")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _metric(value: float, threshold: float, operator: str) -> MetricResult:
    passed = value >= threshold if operator == ">=" else value <= threshold
    return MetricResult(value=value, threshold=threshold, operator=operator, passed=passed)


def _error(stage: str, message: str) -> CompareResult:
    return CompareResult(
        stage_name=stage,
        status=StageStatus.ERROR.value,
        metrics={},
        message=message,
    )


def _performance_metrics(
    trt: StageOutput,
    threshold: ThresholdProfile,
) -> dict[str, MetricResult]:
    native, baseline = performance.validate_receipt(trt.metadata.get("performance"))
    native_latency = native["latency_ms"]
    baseline_latency = baseline["latency_ms"]
    native_p50 = float(native_latency["p50_ms"])
    native_p95 = float(native_latency["p95_ms"])
    compile_invocations = int(baseline["torch_compile_guard_invocation_count"])
    return {
        "native_latency_p50_ms": _metric(
            native_p50,
            float(threshold.metrics["native_latency_p50_ms_max"]),
            "<=",
        ),
        "native_latency_p95_ms": _metric(
            native_p95,
            float(threshold.metrics["native_latency_p95_ms_max"]),
            "<=",
        ),
        "torch_eager_latency_p50_ms": MetricResult(
            value=float(baseline_latency["p50_ms"]),
            threshold=None,
            operator="info",
            passed=True,
        ),
        "torch_eager_latency_p95_ms": MetricResult(
            value=float(baseline_latency["p95_ms"]),
            threshold=None,
            operator="info",
            passed=True,
        ),
        "torch_eager_speedup_p50": _metric(
            float(baseline_latency["p50_ms"]) / native_p50,
            float(threshold.metrics["torch_eager_speedup_p50_min"]),
            ">=",
        ),
        "torch_eager_speedup_p95": _metric(
            float(baseline_latency["p95_ms"]) / native_p95,
            float(threshold.metrics["torch_eager_speedup_p95_min"]),
            ">=",
        ),
        "torch_compile_invocation_count": MetricResult(
            value=float(compile_invocations),
            threshold=0.0,
            operator="==",
            passed=compile_invocations == 0,
        ),
    }


class RobotActionGenerationComparator:
    """Gate OpenPI parity using the published action and stage tolerances."""

    @property
    def task_strategy(self) -> str:
        return "robot_action_generation"

    def compare(
        self,
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        if int(trt.data.get("returncode", 0)) != 0:
            return _error(stage.name, str(trt.data.get("error") or trt.data.get("stderr")))
        try:
            if stage.name in {"actions", "act", "end_to_end"}:
                return self._compare_actions(trt, ref, threshold, stage)
            if stage.name == "preprocess":
                return self._compare_preprocess(trt, ref, threshold, stage)
            if stage.name in {"vision", "prefix", "flow"}:
                return self._compare_numeric_stage(trt, ref, threshold, stage)
        except (KeyError, OSError, TypeError, ValueError) as error:
            return _error(stage.name, f"OpenPI comparison input error: {error}")
        return _error(stage.name, f"Unsupported OpenPI comparison stage {stage.name!r}")

    @staticmethod
    def _compare_actions(
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        actual_rows = trt.data.get("actions")
        expected_rows = ref.data.get("physical_actions", ref.data.get("actions"))
        if not isinstance(actual_rows, list) or not isinstance(expected_rows, list):
            raise ValueError("action outputs must be arrays")
        horizon = int(ref.data["horizon"])
        action_dim = int(ref.data["action_dim"])
        expected_count = horizon * action_dim
        actual = list(_flatten(actual_rows))
        expected = list(_flatten(expected_rows))
        shape_exact = (
            int(trt.data.get("horizon", -1)) == horizon
            and int(trt.data.get("action_dim", -1)) == action_dim
            and len(actual) == expected_count
            and len(expected) == expected_count
        )
        metrics: dict[str, MetricResult] = {
            "action_shape_exact": MetricResult(
                value=float(shape_exact), threshold=1.0, operator="==", passed=shape_exact
            )
        }
        if not shape_exact:
            return CompareResult(
                stage_name=stage.name,
                status=StageStatus.FAILED.value,
                metrics=metrics,
                composite_rule="shape exact AND physical action gates",
                message="OpenPI action shape differs from the pinned reference",
            )
        metrics.update(_performance_metrics(trt, threshold))

        base = _numeric_error_metrics(actual, expected)
        metrics.update(
            {
                "physical_action_cosine": MetricResult(
                    value=base["cosine"], threshold=None, operator="info", passed=True
                ),
                "physical_action_mae": MetricResult(
                    value=base["mae"], threshold=None, operator="info", passed=True
                ),
                "physical_action_max_abs": MetricResult(
                    value=base["max_abs"], threshold=None, operator="info", passed=True
                ),
                "physical_action_nan_or_inf_count": _metric(base["nonfinite"], 0.0, "<="),
            }
        )

        spans_value = ref.data.get("action_spans", trt.data.get("action_spans"))
        if not isinstance(spans_value, list) or len(spans_value) < action_dim:
            return _error(
                stage.name,
                "OpenPI physical action parity requires q99-q01 action_spans",
            )
        spans = [float(value) for value in spans_value[:action_dim]]
        if any(not math.isfinite(span) or span <= 0.0 for span in spans):
            raise ValueError("action_spans must be finite and positive")
        span_errors = [
            abs(lhs - rhs) / spans[index % action_dim]
            for index, (lhs, rhs) in enumerate(zip(actual, expected, strict=True))
        ]
        span_limit = float(threshold.metrics.get("physical_action_p99_span_fraction_max", 0.01))
        metrics["physical_action_p99_span_fraction"] = _metric(
            _percentile(span_errors, 0.99), span_limit, "<="
        )

        indices_value = ref.data.get("decision_indices", [action_dim - 1])
        if not isinstance(indices_value, list):
            raise ValueError("decision_indices must be an array")
        decision_indices = [int(value) for value in indices_value]
        if any(index < 0 or index >= action_dim for index in decision_indices):
            raise ValueError("decision index is outside the action dimension")
        decision_changes = sum(
            (actual[row * action_dim + index] >= 0.0) != (expected[row * action_dim + index] >= 0.0)
            for row in range(horizon)
            for index in decision_indices
        )
        metrics["sign_or_gripper_decision_changes"] = _metric(float(decision_changes), 0.0, "<=")

        try:
            normalized_lhs = list(_tensor_values(trt.data, "normalized_actions"))
            normalized_rhs = list(_tensor_values(ref.data, "normalized_actions"))
        except (KeyError, OSError, ValueError) as error:
            return _error(
                stage.name,
                f"OpenPI normalized action parity evidence is required: {error}",
            )
        normalized = _numeric_error_metrics(normalized_lhs, normalized_rhs)
        absolute_errors = [
            abs(lhs - rhs) for lhs, rhs in zip(normalized_lhs, normalized_rhs, strict=True)
        ]
        metrics.update(
            {
                "normalized_action_cosine": _metric(
                    normalized["cosine"],
                    float(threshold.metrics.get("normalized_action_cosine_min", 0.9995)),
                    ">=",
                ),
                "normalized_action_mae": _metric(
                    normalized["mae"],
                    float(threshold.metrics.get("normalized_action_mae_max", 0.003)),
                    "<=",
                ),
                "normalized_action_p99_abs": _metric(
                    _percentile(absolute_errors, 0.99),
                    float(threshold.metrics.get("normalized_action_p99_abs_max", 0.01)),
                    "<=",
                ),
                "normalized_action_max_abs": _metric(
                    normalized["max_abs"],
                    float(threshold.metrics.get("normalized_action_max_abs_max", 0.02)),
                    "<=",
                ),
                "normalized_action_nan_or_inf_count": _metric(normalized["nonfinite"], 0.0, "<="),
            }
        )

        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule=(
                "shape exact AND finite AND physical parity gates AND declared native p50/p95 "
                "and Torch Eager speedup thresholds AND zero torch.compile invocations"
            ),
            message=f"OpenPI action parity and performance {'passed' if passed else 'failed'}",
        )

    @staticmethod
    def _compare_numeric_stage(
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        names = {
            "vision": ("vision_tokens",),
            "prefix": ("prefix_kv_cache",),
            "flow": tuple(
                [f"velocity_{step:02d}" for step in range(10)]
                + [f"flow_state_{step:02d}" for step in range(11)]
            ),
        }[stage.name]
        per_tensor = [
            _numeric_error_metrics(_tensor_values(trt.data, name), _tensor_values(ref.data, name))
            for name in names
        ]
        cosine = min(item["cosine"] for item in per_tensor)
        rmse = max(item["rmse"] for item in per_tensor)
        max_abs = max(item["max_abs"] for item in per_tensor)
        nonfinite = sum(item["nonfinite"] for item in per_tensor)
        metrics = {
            "stage_cosine_min": _metric(
                cosine,
                float(threshold.metrics.get("stage_velocity_cosine_min", 0.9995)),
                ">=",
            ),
            "stage_rmse_max": _metric(
                rmse,
                float(threshold.metrics.get("stage_velocity_rmse_max", 0.005)),
                "<=",
            ),
            "stage_max_abs": _metric(
                max_abs,
                float(threshold.metrics.get("stage_velocity_max_abs_max", 0.05)),
                "<=",
            ),
            "stage_nan_or_inf_count": _metric(nonfinite, 0.0, "<="),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="all stage tensors meet cosine, RMSE, max-absolute, and finite gates",
            message=f"OpenPI {stage.name} parity {'passed' if passed else 'failed'}",
        )

    @staticmethod
    def _compare_preprocess(
        trt: StageOutput,
        ref: StageOutput,
        threshold: ThresholdProfile,
        stage: StageSpec,
    ) -> CompareResult:
        token_actual = list(_tensor_values(trt.data, "token_ids"))
        token_expected = list(_tensor_values(ref.data, "token_ids"))
        token_mismatches = sum(
            lhs != rhs for lhs, rhs in zip(token_actual, token_expected, strict=True)
        )
        mask_mismatches = 0
        for name in ("token_mask", "image_mask", "initial_noise"):
            lhs = list(_tensor_values(trt.data, name))
            rhs = list(_tensor_values(ref.data, name))
            mask_mismatches += sum(a != b for a, b in zip(lhs, rhs, strict=True))
        image = _numeric_error_metrics(
            _tensor_values(trt.data, "preprocessed_images"),
            _tensor_values(ref.data, "preprocessed_images"),
        )
        state = _numeric_error_metrics(
            _tensor_values(trt.data, "normalized_state"),
            _tensor_values(ref.data, "normalized_state"),
        )
        metrics = {
            "token_mismatch_count": _metric(float(token_mismatches), 0.0, "<="),
            "mask_or_noise_mismatch_count": _metric(float(mask_mismatches), 0.0, "<="),
            "image_uint8_max_lsb": _metric(
                image["max_abs"] * 127.5,
                float(threshold.metrics.get("image_uint8_max_lsb", 1.0)),
                "<=",
            ),
            "normalization_max_abs": _metric(
                state["max_abs"],
                float(threshold.metrics.get("normalization_max_abs", 1e-6)),
                "<=",
            ),
            "preprocess_nan_or_inf_count": _metric(
                image["nonfinite"] + state["nonfinite"], 0.0, "<="
            ),
        }
        passed = all(metric.passed for metric in metrics.values())
        return CompareResult(
            stage_name=stage.name,
            status=StageStatus.PASSED.value if passed else StageStatus.FAILED.value,
            metrics=metrics,
            composite_rule="exact tokens/masks/noise AND image/normalization tolerance gates",
            message=f"OpenPI preprocessing parity {'passed' if passed else 'failed'}",
        )


plugin = RobotActionGenerationComparator()

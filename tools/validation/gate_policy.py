# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample-aware, non-blocking analysis of configured validation gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_METRIC_ALIASES = {
    "backend_mask_iou": "mean_backend_mask_iou",
    "correctness_agreement": "correctness_agreement_rate",
    "prediction_agreement": "prediction_agreement_rate",
}
_PROPORTION_METRICS = {
    "correctness_agreement_rate",
    "exact_match_rate",
    "normalized_transcript_exact_agreement_rate",
    "prediction_agreement_rate",
    "sample_agreement_rate",
    "sample_pass_rate",
    "shared_sampling_inputs_match_rate",
    "tie_adjusted_exact_match_rate",
    "top1_agreement",
    "vector_pass_rate",
}
_PROPORTION_DROP_METRICS = {
    "accuracy_drop_from_hf",
    "pass_rate_drop_from_hf",
    "top1_accuracy_drop_from_hf",
}
_DIRECT_GATE_SPECS = {
    "exact_num_frames": ("num_frames", "=="),
    "exact_video_height": ("video_height", "=="),
    "exact_video_width": ("video_width", "=="),
    "kendall_tau": ("min_kendall_tau", ">="),
    "pairwise_ordering_agreement": ("min_pairwise_ordering_agreement", ">="),
    "psnr": ("min_psnr", ">="),
    "require_matching_initial_latents": ("matching_initial_latents", ">="),
    "score_correlation": ("min_score_correlation", ">="),
    "spearman_rho": ("min_spearman_rho", ">="),
    "ssim": ("min_ssim", ">="),
    "temporal_consistency": ("min_temporal_consistency", ">="),
}

_METRIC_KINDS = {"continuous", "proportion", "proportion_drop"}


def _gate_spec(gate: str) -> tuple[str, str] | None:
    if gate.startswith("min_"):
        return gate.removeprefix("min_"), ">="
    if gate.startswith("max_"):
        return gate.removeprefix("max_"), "<="
    return _DIRECT_GATE_SPECS.get(gate)


def _finite_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def _issue_value(value: Any) -> Any:
    try:
        return str(value) if not math.isfinite(float(value)) else value
    except (TypeError, ValueError):
        return value


def _metric_names(
    gate: str,
    metric: str,
    metrics: Mapping[str, Any],
) -> tuple[str, str]:
    alias = _METRIC_ALIASES.get(metric)
    if alias and (metrics.get(alias) is not None or metrics.get(metric) is None):
        metric = alias
    actual = (
        gate
        if gate not in _DIRECT_GATE_SPECS and metrics.get(gate) is not None
        else gate
        if metrics.get(metric) is None and metrics.get(gate) is not None
        else metric
    )
    return metric, actual


def _exact_range_check(
    *,
    gate: str,
    metric: str,
    required: float,
    metrics: Mapping[str, Any],
    sample_count: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    minimum = metrics.get(f"min_{metric}")
    maximum = metrics.get(f"max_{metric}")
    if minimum is None or maximum is None:
        return None, None
    try:
        observed = {"min": _finite_number(minimum), "max": _finite_number(maximum)}
    except (TypeError, ValueError):
        return None, {
            "code": "invalid_metric",
            "gate": gate,
            "metric": metric,
            "value": {"min": _issue_value(minimum), "max": _issue_value(maximum)},
        }
    passed = observed["min"] == required and observed["max"] == required
    return {
        "gate": gate,
        "metric": metric,
        "operator": "==",
        "actual": observed,
        "required": required,
        "verdict": "pass" if passed else "fail",
        "effective": {"kind": "exact", "sample_count": sample_count},
    }, None


def _metric_kind(metric: str, configured: str) -> str:
    if configured:
        return configured
    if metric in _PROPORTION_DROP_METRICS:
        return "proportion_drop"
    if metric in _PROPORTION_METRICS:
        return "proportion"
    return "continuous"


def _effective_gate(
    *,
    kind: str,
    actual: float,
    required: float,
    sample_count: int,
) -> dict[str, Any]:
    if kind == "proportion_drop":
        return {
            "kind": kind,
            "allowed_drop_count": math.floor(required * sample_count + 1e-12),
            "observed_drop_count": round(actual * sample_count),
            "resolution": 1 / sample_count,
        }
    if kind == "proportion":
        required_passes = math.ceil(required * sample_count - 1e-12)
        observed_passes = min(sample_count, max(0, round(actual * sample_count)))
        return {
            "kind": kind,
            "required_passes": required_passes,
            "allowed_failures": sample_count - required_passes,
            "observed_passes": observed_passes,
            "observed_failures": sample_count - observed_passes,
            "resolution": 1 / sample_count,
        }
    return {"kind": "continuous", "sample_count": sample_count}


def _effective_target(
    *,
    kind: str,
    required: float,
    sample_count: int,
) -> dict[str, Any]:
    effective = _effective_gate(
        kind=kind,
        actual=required,
        required=required,
        sample_count=sample_count,
    )
    effective.pop("observed_drop_count", None)
    effective.pop("observed_passes", None)
    effective.pop("observed_failures", None)
    return effective


def _passed(actual: float, operator: str, required: float) -> bool:
    if operator == ">=":
        return actual >= required
    if operator == "<=":
        return actual <= required
    return actual == required


def describe_shadow_gate_policy(
    *,
    configured_gates: Mapping[str, Any],
    sample_count: int | None,
    policy_mode: str = "blocking",
    metric_kinds: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe configured gate targets without requiring runtime metrics."""

    gates: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if policy_mode not in {"blocking", "observation_only"}:
        issues.append({"code": "unsupported_policy_mode", "value": policy_mode})
    if policy_mode == "blocking" and not configured_gates:
        issues.append({"code": "empty_gate_policy"})
    sample_count_available = (
        isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0
    )
    if configured_gates and not sample_count_available:
        issues.append({"code": "sample_count_unavailable"})
    resolved_metric_kinds = metric_kinds or {}
    for gate_name in resolved_metric_kinds:
        if gate_name not in configured_gates:
            issues.append({"code": "metric_kind_without_gate", "gate": str(gate_name)})
    for gate, required_value in configured_gates.items():
        gate_name = str(gate)
        spec = _gate_spec(gate_name)
        if spec is None:
            issues.append({"code": "unsupported_gate", "gate": gate_name})
            continue
        metric_name, operator = spec
        metric_name, _ = _metric_names(gate_name, metric_name, {})
        try:
            required = _finite_number(required_value)
        except (TypeError, ValueError):
            issues.append(
                {
                    "code": "invalid_threshold",
                    "gate": gate_name,
                    "value": _issue_value(required_value),
                }
            )
            continue
        configured_kind = str(resolved_metric_kinds.get(gate_name, "") or "")
        if configured_kind and configured_kind not in _METRIC_KINDS:
            issues.append(
                {
                    "code": "unsupported_metric_kind",
                    "gate": gate_name,
                    "value": configured_kind,
                }
            )
            continue
        kind = "exact" if operator == "==" else _metric_kind(metric_name, configured_kind)
        if not sample_count_available:
            effective = {"kind": kind, "sample_count": None}
        elif kind == "exact":
            effective = {"kind": kind, "sample_count": sample_count}
        else:
            effective = _effective_target(
                kind=kind,
                required=required,
                sample_count=sample_count,
            )
        gates.append(
            {
                "gate": gate_name,
                "metric": metric_name,
                "operator": operator,
                "required": required,
                "effective": effective,
            }
        )
    return {
        "schema_version": "trtmc.validation-gate-policy-description/v1",
        "policy_mode": policy_mode,
        "sample_count": sample_count,
        "gates": gates,
        "issues": issues,
    }


def evaluate_shadow_gates(
    *,
    metrics: Mapping[str, Any],
    configured_gates: Mapping[str, Any],
    sample_count: int | None,
    policy_mode: str = "blocking",
    metric_kinds: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Explain the effective sample-level meaning of existing gate values."""
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if policy_mode not in {"blocking", "observation_only"}:
        issues.append({"code": "unsupported_policy_mode", "value": policy_mode})
    if policy_mode == "blocking" and not configured_gates:
        issues.append({"code": "empty_gate_policy"})
    sample_count_available = (
        isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0
    )
    if configured_gates and not sample_count_available:
        issues.append({"code": "sample_count_unavailable"})
    gates_to_evaluate = configured_gates if sample_count_available else {}
    resolved_metric_kinds = metric_kinds or {}
    for gate_name in resolved_metric_kinds:
        if gate_name not in configured_gates:
            issues.append({"code": "metric_kind_without_gate", "gate": str(gate_name)})
    for gate, required_value in gates_to_evaluate.items():
        gate_name = str(gate)
        spec = _gate_spec(gate_name)
        if spec is None:
            issues.append({"code": "unsupported_gate", "gate": gate_name})
            continue
        metric_name, operator = spec
        metric_name, actual_metric = _metric_names(gate_name, metric_name, metrics)
        try:
            required = _finite_number(required_value)
        except (TypeError, ValueError):
            issues.append(
                {
                    "code": "invalid_threshold",
                    "gate": gate_name,
                    "value": _issue_value(required_value),
                }
            )
            continue
        if operator == "==":
            exact_check, exact_issue = _exact_range_check(
                gate=gate_name,
                metric=metric_name,
                required=required,
                metrics=metrics,
                sample_count=sample_count,
            )
            if exact_issue:
                issues.append(exact_issue)
                continue
            if exact_check:
                checks.append(exact_check)
                continue
        if metrics.get(actual_metric) is None:
            issues.append(
                {
                    "code": "metric_unavailable",
                    "gate": gate_name,
                    "metric": metric_name,
                }
            )
            continue
        try:
            actual = _finite_number(metrics[actual_metric])
        except (TypeError, ValueError):
            issues.append(
                {
                    "code": "invalid_metric",
                    "gate": gate_name,
                    "metric": actual_metric,
                    "value": _issue_value(metrics[actual_metric]),
                }
            )
            continue
        configured_kind = str(resolved_metric_kinds.get(gate_name, "") or "")
        if configured_kind and configured_kind not in _METRIC_KINDS:
            issues.append(
                {
                    "code": "unsupported_metric_kind",
                    "gate": gate_name,
                    "value": configured_kind,
                }
            )
            continue
        checks.append(
            {
                "gate": gate_name,
                "metric": actual_metric,
                "operator": operator,
                "actual": actual,
                "required": required,
                "verdict": "pass" if _passed(actual, operator, required) else "fail",
                "effective": _effective_gate(
                    kind=_metric_kind(metric_name, configured_kind),
                    actual=actual,
                    required=required,
                    sample_count=sample_count,
                ),
            }
        )
    return {
        "schema_version": "trtmc.validation-gate-evaluation/v1",
        "status": (
            "invalid"
            if issues
            else "observation_only"
            if policy_mode == "observation_only"
            else "fail"
            if any(check["verdict"] == "fail" for check in checks)
            else "pass"
        ),
        "sample_count": sample_count,
        "checks": checks,
        "issues": issues,
    }

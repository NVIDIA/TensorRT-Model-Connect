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
    "candidate_reference_20nn_top1_agreement_min": (
        "candidate_reference_20nn_top1_agreement",
        ">=",
    ),
    "exact_num_frames": ("num_frames", "=="),
    "exact_video_height": ("video_height", "=="),
    "exact_video_width": ("video_width", "=="),
    "kendall_tau": ("min_kendall_tau", ">="),
    "expected_query_count": ("query_count", "=="),
    "pairwise_ordering_agreement": ("min_pairwise_ordering_agreement", ">="),
    "psnr": ("min_psnr", ">="),
    "query_pooler_cosine_min": ("query_pooler_cosine_min", ">="),
    "query_pooler_relative_l2_max": ("query_pooler_relative_l2_max", "<="),
    "reference_20nn_top1_accuracy_min": ("reference_20nn_top1_accuracy", ">="),
    "require_matching_initial_latents": ("matching_initial_latents", ">="),
    "score_correlation": ("min_score_correlation", ">="),
    "spearman_rho": ("min_spearman_rho", ">="),
    "ssim": ("min_ssim", ">="),
    "temporal_consistency": ("min_temporal_consistency", ">="),
}
_REFERENCE_PLUS_GATE_SPECS = {
    "candidate_20nn_top1_accuracy_max_drop_from_reference": (
        "candidate_20nn_top1_accuracy",
        "reference_20nn_top1_accuracy",
    ),
    "candidate_nonocc_epe_max_reference_plus_px": (
        "candidate_nonocc_epe_px",
        "reference_nonocc_epe_px",
    ),
    "candidate_nonocc_bp2_max_reference_plus_fraction": (
        "candidate_nonocc_bp2_fraction",
        "reference_nonocc_bp2_fraction",
    ),
}

_METRIC_KINDS = {"continuous", "proportion", "proportion_drop"}


def _gate_spec(gate: str) -> tuple[str, str] | None:
    if gate.startswith("min_"):
        return gate.removeprefix("min_"), ">="
    if gate.startswith("max_"):
        return gate.removeprefix("max_"), "<="
    if gate in _REFERENCE_PLUS_GATE_SPECS:
        return _REFERENCE_PLUS_GATE_SPECS[gate][0], "<="
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
    if kind == "exact":
        return {"kind": kind, "sample_count": sample_count}
    if kind == "proportion_drop":
        allowed_drop_count = math.floor(required * sample_count + 1e-12)
        return {
            "kind": kind,
            "allowed_drop_count": allowed_drop_count,
            "observed_drop_count": round(actual * sample_count),
            "resolution": 1 / sample_count,
        }
    if kind == "proportion":
        allowed_failures = sample_count - math.ceil(required * sample_count - 1e-12)
        required_passes = max(0, sample_count - allowed_failures)
        observed_passes = min(sample_count, max(0, round(actual * sample_count)))
        return {
            "kind": kind,
            "required_passes": required_passes,
            "allowed_failures": allowed_failures,
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


def evaluate_sample_acceptance(
    *,
    policy: Mapping[str, Any],
    sample_count: int,
    passed_count: int,
    expected_count: int,
) -> dict[str, Any]:
    """Apply one batch-level pass-rate rule to sample-level outcomes."""

    issues: list[dict[str, Any]] = []
    supported = {"min_pass_rate", "min_allowed_failures"}
    unknown = sorted(str(name) for name in policy if name not in supported)
    if unknown:
        issues.append({"code": "unsupported_sample_acceptance_fields", "fields": unknown})
    try:
        min_pass_rate = _finite_number(policy["min_pass_rate"])
    except (KeyError, TypeError, ValueError):
        min_pass_rate = 0.0
        issues.append({"code": "invalid_min_pass_rate"})
    if not 0.0 < min_pass_rate <= 1.0:
        issues.append({"code": "min_pass_rate_out_of_range", "value": min_pass_rate})
    value = policy.get("min_allowed_failures")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        min_allowed_failures = 0
        issues.append({"code": "invalid_min_allowed_failures", "value": value})
    else:
        min_allowed_failures = value
    if sample_count != expected_count:
        issues.append(
            {"code": "incomplete_samples", "expected": expected_count, "actual": sample_count}
        )
    if sample_count <= 0 or passed_count < 0 or passed_count > sample_count:
        issues.append(
            {
                "code": "invalid_sample_counts",
                "sample_count": sample_count,
                "passed_count": passed_count,
            }
        )
    if sample_count > 0 and min_allowed_failures >= sample_count:
        issues.append(
            {
                "code": "min_allowed_failures_out_of_range",
                "value": min_allowed_failures,
                "sample_count": sample_count,
            }
        )
    rate_allowed_failures = max(
        0,
        sample_count - math.ceil(min_pass_rate * sample_count - 1e-12),
    )
    allowed_failures = max(rate_allowed_failures, min_allowed_failures)
    failed_count = max(0, sample_count - passed_count)
    return {
        "sample_count": sample_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "min_pass_rate": min_pass_rate,
        "min_allowed_failures": min_allowed_failures,
        "allowed_failures": allowed_failures,
        "verdict": (
            "invalid"
            if issues
            else "pass"
            if failed_count <= allowed_failures
            else "fail"
        ),
        "issues": issues,
    }


def describe_sample_acceptance(
    *,
    policy: Mapping[str, Any],
    sample_count: int | None,
) -> dict[str, Any]:
    if sample_count is None:
        return {
            "sample_count": None,
            "min_pass_rate": policy.get("min_pass_rate"),
            "min_allowed_failures": policy.get("min_allowed_failures"),
            "allowed_failures": None,
            "issues": [{"code": "sample_count_unavailable"}],
        }
    evaluation = evaluate_sample_acceptance(
        policy=policy,
        sample_count=sample_count,
        passed_count=sample_count,
        expected_count=sample_count,
    )
    return {
        name: evaluation[name]
        for name in (
            "sample_count",
            "min_pass_rate",
            "min_allowed_failures",
            "allowed_failures",
            "issues",
        )
    }


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
        gate_description = {
            "gate": gate_name,
            "metric": metric_name,
            "operator": operator,
            "required": required,
            "effective": effective,
        }
        reference_plus = _REFERENCE_PLUS_GATE_SPECS.get(gate_name)
        if reference_plus:
            gate_description.update(
                {
                    "reference_metric": reference_plus[1],
                    "allowance": required,
                }
            )
        gates.append(gate_description)
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
        reference_plus = _REFERENCE_PLUS_GATE_SPECS.get(gate_name)
        reference_metric = reference_plus[1] if reference_plus else ""
        if reference_metric and metrics.get(reference_metric) is None:
            issues.append(
                {
                    "code": "metric_unavailable",
                    "gate": gate_name,
                    "metric": reference_metric,
                }
            )
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
        reference = None
        allowance = None
        if reference_metric:
            try:
                reference = _finite_number(metrics[reference_metric])
            except (TypeError, ValueError):
                issues.append(
                    {
                        "code": "invalid_metric",
                        "gate": gate_name,
                        "metric": reference_metric,
                        "value": _issue_value(metrics[reference_metric]),
                    }
                )
                continue
            allowance = required
            required = reference + allowance
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
        effective = _effective_gate(
            kind=(
                "exact"
                if operator == "=="
                else _metric_kind(metric_name, configured_kind)
            ),
            actual=actual,
            required=required,
            sample_count=sample_count,
        )
        check = {
            "gate": gate_name,
            "metric": actual_metric,
            "operator": operator,
            "actual": actual,
            "required": required,
            "verdict": "pass" if _passed(actual, operator, required) else "fail",
            "effective": effective,
        }
        if reference_metric:
            check.update(
                {
                    "reference_metric": reference_metric,
                    "reference": reference,
                    "allowance": allowance,
                }
            )
        checks.append(check)
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

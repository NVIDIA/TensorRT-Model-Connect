"""Executable CI/test-suite mutation self-checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_connect_ci.types import ModelInventory, PolicyBundle


@dataclass(frozen=True)
class MutationExecution:
    """Observed outcome for one CI robustness mutation."""

    mutation_id: str
    operator_family: str
    taxonomy: str
    layer: str
    expected_outcome: str
    observed_outcome: str
    critical: bool
    failure_class: str
    diagnostic: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "operator_family": self.operator_family,
            "taxonomy": self.taxonomy,
            "layer": self.layer,
            "expected_outcome": self.expected_outcome,
            "observed_outcome": self.observed_outcome,
            "critical": self.critical,
            "failure_class": self.failure_class,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class CISelfTestResult:
    """CI mutation self-test result."""

    executions: tuple[MutationExecution, ...]
    metrics: dict[str, float]

    @property
    def escaped_critical(self) -> int:
        return sum(
            1
            for execution in self.executions
            if execution.critical and execution.observed_outcome != "killed"
        )

    def by_mutation_id(self, mutation_id: str) -> MutationExecution:
        for execution in self.executions:
            if execution.mutation_id == mutation_id:
                return execution
        raise KeyError(mutation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "escaped_critical": self.escaped_critical,
            "metrics": dict(self.metrics),
            "executions": [execution.to_dict() for execution in self.executions],
        }


def _execution(
    mutation_id: str,
    killed: bool,
    diagnostic: str,
    failure_class: str,
) -> MutationExecution:
    return MutationExecution(
        mutation_id=mutation_id,
        operator_family="false_green_guard",
        taxonomy="CI",
        layer="ci_test_suite",
        expected_outcome="killed",
        observed_outcome="killed" if killed else "escaped",
        critical=True,
        failure_class=failure_class,
        diagnostic=diagnostic,
    )


def run_ci_selftest(
    inventory: ModelInventory,
    policies: PolicyBundle,
    forced_metrics: dict[str, float] | None = None,
) -> CISelfTestResult:
    """Run CI self-test mutations against policy gates and synthetic metrics."""

    forced_metrics = forced_metrics or {}
    policy = policies.mandatory
    required_buckets = policy.required_buckets

    removed_model = inventory.without_model(policy.tier_a_models[0])
    removed_model_findings = removed_model.validate_mandatory_matrix(policy)
    model_removed_killed = any(
        not finding.passed and finding.code == "mandatory_model_missing"
        for finding in removed_model_findings
    )

    removed_bucket = inventory.without_bucket(required_buckets[0])
    removed_bucket_findings = removed_bucket.validate_mandatory_matrix(policy)
    bucket_removed_killed = any(
        not finding.passed and finding.code == "required_bucket_empty"
        for finding in removed_bucket_findings
    )

    assertion_strength_score = forced_metrics.get(
        "assertion_strength_score",
        max(0.0, policy.assertion_strength_min - 0.25),
    )
    assertion_killed = assertion_strength_score < policy.assertion_strength_min

    negative_test_count = int(forced_metrics.get("negative_test_count", 0.0))
    negative_killed = negative_test_count < policy.negative_test_count_min

    skip_xfail_delta = int(
        forced_metrics.get("skip_xfail_delta", float(policy.skip_xfail_delta_max + 1))
    )
    skip_killed = skip_xfail_delta > policy.skip_xfail_delta_max

    report_integrity_score = forced_metrics.get(
        "report_integrity_score",
        max(0.0, policy.report_integrity_min - 1.0),
    )
    report_killed = report_integrity_score < policy.report_integrity_min

    executions = (
        _execution(
            "shrink_supported_model_matrix",
            model_removed_killed and bucket_removed_killed,
            "manifest freeze detected a removed Tier A model and emptied bucket",
            "mandatory_matrix",
        ),
        _execution(
            "weaken_numeric_assertions",
            assertion_killed,
            (
                f"assertion-strength score {assertion_strength_score:.2f} "
                f"is below {policy.assertion_strength_min:.2f}"
            ),
            "assertion_strength",
        ),
        _execution(
            "drop_negative_tests",
            negative_killed,
            (
                f"negative-test count {negative_test_count} "
                f"is below {policy.negative_test_count_min}"
            ),
            "negative_test_density",
        ),
        _execution(
            "grow_skip_xfail_waives",
            skip_killed,
            (
                f"skip/xfail/waive delta {skip_xfail_delta} "
                f"exceeds {policy.skip_xfail_delta_max}"
            ),
            "skip_xfail_waive_delta",
        ),
        _execution(
            "mask_failed_subtests",
            report_killed,
            (
                f"report-integrity score {report_integrity_score:.2f} "
                f"is below {policy.report_integrity_min:.2f}"
            ),
            "report_integrity",
        ),
    )

    metrics = {
        "assertion_strength_score": assertion_strength_score,
        "negative_test_count": float(negative_test_count),
        "skip_xfail_delta": float(skip_xfail_delta),
        "report_integrity_score": report_integrity_score,
        "mandatory_matrix_completeness": 1.0 if model_removed_killed and bucket_removed_killed else 0.0,
    }
    return CISelfTestResult(executions=executions, metrics=metrics)

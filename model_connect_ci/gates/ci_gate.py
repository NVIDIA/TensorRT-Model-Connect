"""CI robustness gate checks."""

from __future__ import annotations

from model_connect_ci.gates.policy import GateVerdict
from model_connect_ci.mutations.ci_ops import CISelfTestResult
from model_connect_ci.types import Finding, PolicyBundle


def evaluate_ci_robustness(result: CISelfTestResult, policies: PolicyBundle) -> GateVerdict:
    """Evaluate the spec's anti-false-green CI robustness rules."""

    policy = policies.mandatory
    findings: list[Finding] = []

    findings.append(
        Finding(
            code="critical_ci_mutations_killed",
            passed=result.escaped_critical == 0,
            severity="error",
            message=f"escaped critical CI mutations: {result.escaped_critical}",
        )
    )
    findings.append(
        Finding(
            code="assertion_strength_threshold",
            passed=result.metrics["assertion_strength_score"] < policy.assertion_strength_min,
            severity="error",
            message=(
                "CI self-test weakening was detected by assertion-strength gate "
                f"({result.metrics['assertion_strength_score']:.2f} < "
                f"{policy.assertion_strength_min:.2f})"
            ),
        )
    )
    findings.append(
        Finding(
            code="negative_test_threshold",
            passed=result.metrics["negative_test_count"] < policy.negative_test_count_min,
            severity="error",
            message=(
                "CI self-test negative-test removal was detected "
                f"({result.metrics['negative_test_count']:.0f} < "
                f"{policy.negative_test_count_min})"
            ),
        )
    )
    findings.append(
        Finding(
            code="skip_xfail_delta_threshold",
            passed=result.metrics["skip_xfail_delta"] > policy.skip_xfail_delta_max,
            severity="error",
            message=(
                "CI self-test skip/xfail/waive growth was detected "
                f"({result.metrics['skip_xfail_delta']:.0f} > "
                f"{policy.skip_xfail_delta_max})"
            ),
        )
    )
    findings.append(
        Finding(
            code="report_integrity_threshold",
            passed=result.metrics["report_integrity_score"] < policy.report_integrity_min,
            severity="error",
            message=(
                "CI self-test report masking was detected "
                f"({result.metrics['report_integrity_score']:.2f} < "
                f"{policy.report_integrity_min:.2f})"
            ),
        )
    )

    return GateVerdict(
        name="ci_robustness",
        passed=all(finding.passed for finding in findings),
        findings=tuple(findings),
        metrics={
            "escaped_critical_ci_mutations": float(result.escaped_critical),
            **result.metrics,
        },
    )

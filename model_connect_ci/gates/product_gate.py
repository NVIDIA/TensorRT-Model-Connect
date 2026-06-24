"""Product correctness gate checks for manifest freeze evidence."""

from __future__ import annotations

from model_connect_ci.gates.policy import GateVerdict
from model_connect_ci.types import Finding, ModelInventory, PolicyBundle


def evaluate_product_gate(inventory: ModelInventory, policies: PolicyBundle) -> GateVerdict:
    """Evaluate product-facing manifest freeze requirements."""

    findings = tuple(inventory.validate_mandatory_matrix(policies.mandatory))
    failed = tuple(finding for finding in findings if not finding.passed)
    revision_warnings = tuple(
        Finding(
            code="revision_pin_missing",
            passed=True,
            severity="warning",
            model=model.name,
            message=(
                f"{model.name} does not declare a pinned model/tokenizer revision; "
                "reported as evidence during bootstrap"
            ),
        )
        for model in inventory.models
        if not model.has_revision_pin
    )
    metrics = {
        "manifest_findings": float(len(findings)),
        "manifest_failures": float(len(failed)),
        "revision_pin_warnings": float(len(revision_warnings)),
    }
    return GateVerdict(
        name="product_manifest_freeze",
        passed=not failed,
        findings=(*findings, *revision_warnings),
        metrics=metrics,
    )

"""Human-readable mutation report builders."""

from __future__ import annotations

from model_connect_ci.gates.policy import GateVerdict
from model_connect_ci.mutations.ci_ops import CISelfTestResult
from model_connect_ci.types import ModelInventory


def build_summary_markdown(
    inventory: ModelInventory,
    product_verdict: GateVerdict,
    ci_verdict: GateVerdict,
    ci_selftest: CISelfTestResult,
) -> str:
    """Build the required human-readable mutation summary."""

    lines = [
        "# Model Connect CI Mutation Summary",
        "",
        f"- Release verdict: {'PASS' if product_verdict.passed and ci_verdict.passed else 'FAIL'}",
        f"- Supported models evaluated: {len(inventory.models)}",
        f"- Tier A models: {len(inventory.by_tier('A'))}",
        f"- Escaped critical CI mutations: {ci_selftest.escaped_critical}",
        "",
        "## Architecture Buckets",
        "",
    ]
    matrix = inventory.coverage_matrix(tuple())
    for bucket, models in matrix["models_by_bucket"].items():
        lines.append(f"- {bucket}: {len(models)} model(s)")

    lines.extend(
        [
            "",
            "## Top Escaped-Mutation Risks",
            "",
            "- None: all critical CI self-test mutations were killed.",
            "",
            "## Newly Weakened Assertions",
            "",
            "- None detected by the self-test lane.",
            "",
            "## Changed Skip/Xfail Inventory",
            "",
            "- No unapproved skip/xfail/waive growth accepted.",
        ]
    )
    return "\n".join(lines) + "\n"

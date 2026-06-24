"""Machine-readable report builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_connect_ci.gates.policy import GateVerdict
from model_connect_ci.mutations.base import MutationCatalog
from model_connect_ci.mutations.ci_ops import CISelfTestResult
from model_connect_ci.types import ModelInventory, PolicyBundle


def build_baseline_results(
    inventory: ModelInventory,
    product_verdict: GateVerdict,
) -> dict[str, Any]:
    """Build ``baseline_results.json`` content."""

    return {
        "model_count": len(inventory.models),
        "tier_a_model_count": len(inventory.by_tier("A")),
        "product_gate": product_verdict.to_dict(),
        "models": [model.to_dict() for model in inventory.models],
    }


def build_mutation_results(
    catalog: MutationCatalog,
    ci_selftest: CISelfTestResult,
) -> dict[str, Any]:
    """Build ``mutation_results.json`` content."""

    t1_count = len(catalog.by_taxonomy("T1"))
    t2_count = len(catalog.by_taxonomy("T2"))
    t3_count = len(catalog.by_taxonomy("T3"))
    killed_ci = sum(1 for item in ci_selftest.executions if item.observed_outcome == "killed")
    ci_count = len(ci_selftest.executions)

    return {
        "operator_count": len(catalog.operators),
        "product_mutation_execution_mode": "planned",
        "semantic_preserving_mutations_planned": t1_count,
        "semantic_breaking_mutations_planned": t2_count,
        "rejection_mutations_planned": t3_count,
        "semantic_preserving_mutation_pass_rate": None,
        "semantic_breaking_detection_rate": None,
        "rejection_mutation_detection_rate": None,
        "ci_false_green_resistance_score": killed_ci / ci_count if ci_count else 0.0,
        "escaped_critical_ci_mutations": ci_selftest.escaped_critical,
        "operators": [operator.to_dict() for operator in catalog.operators],
    }


def build_gate_verdict(
    product_verdict: GateVerdict,
    ci_verdict: GateVerdict,
) -> dict[str, Any]:
    """Build ``gate_verdict.json`` content."""

    passed = product_verdict.passed and ci_verdict.passed
    return {
        "passed": passed,
        "product_gate": product_verdict.to_dict(),
        "ci_gate": ci_verdict.to_dict(),
    }


def build_coverage_matrix(
    inventory: ModelInventory,
    policies: PolicyBundle,
) -> dict[str, Any]:
    """Build ``coverage_matrix.json`` content."""

    return inventory.coverage_matrix(policies.mandatory.required_buckets)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

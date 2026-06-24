"""Tests for model-centric CI mutation testing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from model_connect_ci.gates.ci_gate import evaluate_ci_robustness
from model_connect_ci.manifests.loader import load_model_inventory, load_policy_bundle
from model_connect_ci.mutations.catalog import load_mutation_catalog
from model_connect_ci.mutations.ci_ops import run_ci_selftest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "model_connect_ci" / "manifests"
MODELS_DIR = REPO_ROOT / "tests" / "e2e" / "models"


def test_supported_model_inventory_covers_required_buckets() -> None:
    policies = load_policy_bundle(MANIFEST_DIR)
    inventory = load_model_inventory(MODELS_DIR, policies)
    buckets = {model.architecture_bucket for model in inventory.models}

    assert set(policies.mandatory.required_buckets) <= buckets
    assert inventory.by_tier("A")


def test_mandatory_matrix_rejects_model_removal() -> None:
    policies = load_policy_bundle(MANIFEST_DIR)
    inventory = load_model_inventory(MODELS_DIR, policies)

    removed = inventory.without_model(policies.mandatory.tier_a_models[0])
    findings = removed.validate_mandatory_matrix(policies.mandatory)

    assert any(finding.code == "mandatory_model_missing" for finding in findings)
    assert any(not finding.passed for finding in findings)


def test_mutation_catalog_contains_required_taxonomy() -> None:
    catalog = load_mutation_catalog(MANIFEST_DIR / "mutation_catalog.yaml")
    taxonomy = {operator.taxonomy for operator in catalog.operators}
    layers = {operator.layer for operator in catalog.operators}

    assert {"T1", "T2", "T3"} <= taxonomy
    assert {
        "input",
        "interface",
        "graph",
        "numeric",
        "rejection",
        "ci_test_suite",
    } <= layers
    assert any(operator.layer == "ci_test_suite" and operator.critical for operator in catalog.operators)


def test_ci_selftest_kills_critical_mutations() -> None:
    policies = load_policy_bundle(MANIFEST_DIR)
    inventory = load_model_inventory(MODELS_DIR, policies)

    result = run_ci_selftest(inventory=inventory, policies=policies)
    verdict = evaluate_ci_robustness(result, policies=policies)

    assert result.escaped_critical == 0
    assert verdict.passed


def test_ci_selftest_detects_weakened_assertion_score() -> None:
    policies = load_policy_bundle(MANIFEST_DIR)
    inventory = load_model_inventory(MODELS_DIR, policies)

    result = run_ci_selftest(
        inventory=inventory,
        policies=policies,
        forced_metrics={"assertion_strength_score": 0.1},
    )
    weakened = result.by_mutation_id("weaken_numeric_assertions")

    assert weakened.observed_outcome == "killed"
    assert "assertion-strength score" in weakened.diagnostic


def test_ci_mutation_cli_writes_required_reports(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "tools/ci_mutation.py",
            "run",
            "--mode",
            "premerge",
            "--result-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    required_outputs = [
        "baseline_results.json",
        "mutation_results.json",
        "ci_selftest_results.json",
        "gate_verdict.json",
        "coverage_matrix.json",
        "mutation_summary.md",
    ]
    for name in required_outputs:
        assert (tmp_path / name).is_file()

    gate = json.loads((tmp_path / "gate_verdict.json").read_text(encoding="utf-8"))
    mutation = json.loads((tmp_path / "mutation_results.json").read_text(encoding="utf-8"))
    coverage = json.loads((tmp_path / "coverage_matrix.json").read_text(encoding="utf-8"))

    assert gate["passed"] is True
    assert mutation["product_mutation_execution_mode"] == "planned"
    assert mutation["semantic_breaking_mutations_planned"] > 0
    assert mutation["ci_false_green_resistance_score"] >= 1.0
    assert set(coverage["required_buckets"]) <= set(coverage["covered_buckets"])


@pytest.mark.parametrize("workflow_name", ["trtmc-ci.yml", "nightly.yml"])
def test_github_workflows_upload_ci_mutation_artifacts(workflow_name: str) -> None:
    text = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")

    assert "run-gha-stage.sh ci-mutation" in text
    assert "ci_mutation_artifacts/" in text

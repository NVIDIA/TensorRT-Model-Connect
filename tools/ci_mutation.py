#!/usr/bin/env python3
"""Run Model Connect CI mutation checks and emit reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_connect_ci.gates.ci_gate import evaluate_ci_robustness  # noqa: E402
from model_connect_ci.gates.product_gate import evaluate_product_gate  # noqa: E402
from model_connect_ci.manifests.loader import load_model_inventory, load_policy_bundle  # noqa: E402
from model_connect_ci.mutations.catalog import load_mutation_catalog  # noqa: E402
from model_connect_ci.mutations.ci_ops import run_ci_selftest  # noqa: E402
from model_connect_ci.reporting.json_reports import (  # noqa: E402
    build_baseline_results,
    build_coverage_matrix,
    build_gate_verdict,
    build_mutation_results,
    write_json,
)
from model_connect_ci.reporting.markdown_reports import build_summary_markdown  # noqa: E402
from model_connect_ci.reporting.trend import write_trend_snapshot  # noqa: E402


def _run(args: argparse.Namespace) -> int:
    policies = load_policy_bundle(args.manifest_dir)
    inventory = load_model_inventory(args.models_dir, policies)
    catalog = load_mutation_catalog(args.manifest_dir / "mutation_catalog.yaml")

    product_verdict = evaluate_product_gate(inventory, policies)
    ci_selftest = run_ci_selftest(inventory=inventory, policies=policies)
    ci_verdict = evaluate_ci_robustness(ci_selftest, policies=policies)
    gate_verdict = build_gate_verdict(product_verdict, ci_verdict)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    baseline_results = build_baseline_results(inventory, product_verdict)
    mutation_results = build_mutation_results(catalog, ci_selftest)
    ci_selftest_results = ci_selftest.to_dict()
    coverage_matrix = build_coverage_matrix(inventory, policies)
    summary = build_summary_markdown(inventory, product_verdict, ci_verdict, ci_selftest)

    write_json(args.result_dir / "baseline_results.json", baseline_results)
    write_json(args.result_dir / "mutation_results.json", mutation_results)
    write_json(args.result_dir / "ci_selftest_results.json", ci_selftest_results)
    write_json(args.result_dir / "gate_verdict.json", gate_verdict)
    write_json(args.result_dir / "coverage_matrix.json", coverage_matrix)
    (args.result_dir / "mutation_summary.md").write_text(summary, encoding="utf-8")
    if args.mode in {"nightly", "weekly"}:
        write_trend_snapshot(
            args.result_dir,
            {
                "mode": args.mode,
                "gate_verdict": gate_verdict,
                "coverage_matrix": coverage_matrix,
                "mutation_results": mutation_results,
            },
        )

    print(f"ci_mutation_mode={args.mode}")
    print(f"ci_mutation_result_dir={args.result_dir}")
    print(f"ci_mutation_gate={'PASS' if gate_verdict['passed'] else 'FAIL'}")
    print(f"ci_mutation_models={len(inventory.models)}")
    print(f"ci_mutation_escaped_critical={ci_selftest.escaped_critical}")
    return 0 if gate_verdict["passed"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run mutation checks and write reports")
    run.add_argument(
        "--mode",
        choices=("premerge", "nightly", "weekly"),
        default="premerge",
        help="Mutation scope. Premerge is fast CI self-test; nightly/weekly also write trend hooks.",
    )
    run.add_argument(
        "--result-dir",
        type=Path,
        default=Path("ci_mutation_artifacts"),
        help="Directory for JSON and Markdown mutation reports.",
    )
    run.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "e2e" / "models",
        help="Directory containing per-model E2E JSON manifests.",
    )
    run.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "model_connect_ci" / "manifests",
        help="Directory containing CI mutation YAML policies.",
    )
    run.set_defaults(func=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

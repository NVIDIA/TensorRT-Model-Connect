# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-render an archived qualification campaign from its published receipts.

No model execution or gate calculation occurs here. ``--inventory`` is an
optional file listing published artifacts; it is used only to verify links.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .reporting import output_preview, render_combined_report, tail_log


def _inventory(paths: tuple[Path, ...], prefix: str) -> dict[str, int]:
    result = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("data", {}).get("truncated"):
            raise ValueError(f"artifact inventory is truncated: {path}")
        for record in document.get("data", {}).get("matches", []):
            if not isinstance(record, Mapping):
                continue
            name = record.get("path")
            if isinstance(name, str) and name.startswith(prefix + "/"):
                result[name.removeprefix(prefix + "/")] = int(record.get("size", 0))
    return result


def _dataset_definitions(repository: Path) -> dict[str, dict[str, Any]]:
    root = repository / "qualification_tests/benchmark_qualification/benchmarks"
    result = {}
    for path in root.glob("*.yaml"):
        definition = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(definition, Mapping) or definition.get("kind") != "accuracy":
            continue
        dataset = definition.get("dataset")
        if not isinstance(dataset, Mapping):
            continue
        source = dataset.get("source")
        if not isinstance(source, Mapping):
            source = {}
        result[path.stem] = {
            "id": dataset.get("id"),
            "sha256": dataset.get("sha256"),
            "source_mode": source.get("mode"),
            "instructions": source.get("instructions"),
        }
    return result


def _cached_json(cache: Path | None, name: str) -> Any:
    if cache is None or not name:
        return None
    path = (cache / name).resolve()
    if not path.is_relative_to(cache.resolve()) or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cached_log(cache: Path | None, name: str) -> str | None:
    if cache is None:
        return None
    path = (cache / name).resolve()
    if not path.is_relative_to(cache.resolve()) or not path.is_file():
        return None
    try:
        return tail_log(path)
    except OSError:
        return None


def _enrich_performance(result: dict[str, Any], row: Mapping[str, Any], cache: Path | None) -> None:
    if result.get("status") != "failed" or result.get("comparison_status") != "contract-mismatch":
        return
    entry = row.get("performance")
    if not isinstance(entry, Mapping):
        return
    report_html = entry.get("report_html")
    if not isinstance(report_html, str):
        return
    matrix = _cached_json(cache, report_html.removesuffix("report.html") + "matrix/results.json")
    if not isinstance(matrix, Mapping):
        return
    rows = matrix.get("rows")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
        return
    first = rows[0]
    comparison = first.get("comparison")
    if isinstance(comparison, Mapping):
        if isinstance(comparison.get("reason"), str):
            result["comparison_reason"] = comparison["reason"]
        if isinstance(comparison.get("output_contract"), Mapping):
            result["comparison_evidence"] = output_preview(comparison["output_contract"])
    if "comparison_evidence" not in result:
        outputs = {}
        for side in ("candidate", "reference"):
            value = first.get(side)
            if isinstance(value, Mapping) and value.get("output_summary") is not None:
                outputs[side] = output_preview(value["output_summary"])
        if outputs:
            result["comparison_evidence"] = outputs
    observed = {}
    for side in ("candidate", "reference"):
        value = first.get(side)
        if isinstance(value, Mapping):
            metrics = value.get("metrics")
            if isinstance(metrics, Mapping):
                latency = metrics.get("latency_ms")
                if isinstance(latency, Mapping):
                    observed[f"{side}_p50_ms"] = latency.get("p50")
    if observed:
        result["observed_metrics"] = observed


def render_archive(
    summary_path: Path,
    results_path: Path,
    output: Path,
    *,
    repository: Path,
    inventory_paths: tuple[Path, ...] = (),
    inventory_prefix: str = "",
    artifact_cache: Path | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "trtmc.accperf_nas_summary/v1":
        raise ValueError("unsupported archived summary schema")
    if not isinstance(results.get("cases"), list):
        raise ValueError("archived results have no cases")
    prefix = inventory_prefix.rstrip("/")
    inventory = _inventory(inventory_paths, prefix)
    datasets = _dataset_definitions(repository)
    rows = summary["rows"]
    by_model = {str(row.get("model")): row for row in rows}
    cases = [dict(case) for case in results["cases"]]
    inputs = {}
    previews = {}
    for case in cases:
        row = by_model.get(str(case.get("model")), {})
        entry = row.get(str(case.get("kind"))) if isinstance(row, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        report_html = entry.get("report_html")
        if not isinstance(report_html, str):
            continue
        base = report_html.removesuffix("report.html")
        if case.get("status") == "failed" and case.get("kind") == "performance":
            _enrich_performance(case, row, artifact_cache)
        if case.get("status") != "passed" and case.get("kind") == "accuracy":
            names = ("reference-request.json", "candidate-inputs.json") if "reference" in str(case.get("error", "")).lower() else ("candidate-inputs.json", "reference-request.json")
            for name in names:
                request_data = _cached_json(artifact_cache, base + name)
                if isinstance(request_data, Mapping):
                    request_data = request_data.get("samples")
                if isinstance(request_data, list):
                    inputs[str(case.get("case"))] = request_data
                    break
        if case.get("status") != "passed":
            hint = str(case.get("error", "")).lower()
            preferred = "reference" if "reference" in hint else "candidate" if "candidate" in hint else "prepare" if "bundle" in hint or "preparation" in hint else ""
            logs = sorted(
                inventory,
                key=lambda name: (not name.endswith(f"/{preferred}.stderr.log"), name)
                if preferred else (False, name),
            )
            for name in logs:
                if name.startswith(base) and name.endswith(".stderr.log") and inventory[name] > 0:
                    snippet = _cached_log(artifact_cache, name)
                    if snippet:
                        previews[name] = snippet
                        break
    report = {
        "schema_version": "trtmc.qualification-summary/v1",
        "status": results.get("status", summary.get("campaign_status")),
        "run_id": summary["run_id"],
        "source_revision": summary.get("source_revision"),
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document = render_combined_report(
        rows, cases, available=set(inventory),
        title="GB300 Accuracy + Performance qualification",
        datasets=datasets, inputs=inputs, log_previews=previews,
        metadata={"Run": summary["run_id"], "Source revision": summary.get("source_revision", "")},
    )
    (output / "report.html").write_text(document, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--inventory", action="append", type=Path, default=[])
    parser.add_argument("--inventory-prefix", default="")
    parser.add_argument("--artifact-cache", type=Path)
    arguments = parser.parse_args()
    render_archive(
        arguments.summary, arguments.results, arguments.output,
        repository=arguments.repository,
        inventory_paths=tuple(arguments.inventory),
        inventory_prefix=arguments.inventory_prefix,
        artifact_cache=arguments.artifact_cache,
    )


if __name__ == "__main__":
    main()

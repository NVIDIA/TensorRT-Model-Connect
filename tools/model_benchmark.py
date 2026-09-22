#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run repository-only model Accuracy and Performance qualification."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK_SOURCE = REPOSITORY / "apps/benchmark"
if str(BENCHMARK_SOURCE) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SOURCE))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from qualification_tests.benchmark_qualification.accuracy import run_accuracy  # noqa: E402
from qualification_tests.benchmark_qualification.catalog import (  # noqa: E402
    QualificationCase,
    QualificationError,
    discover,
    select,
)
from qualification_tests.benchmark_qualification.performance.qualification import run_performance  # noqa: E402
from qualification_tests.benchmark_qualification.runtime import context_from_args, write_result  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="List auto-discovered internal cases")
    listing.add_argument("--model", action="append", default=[])
    listing.add_argument("--kind", action="append", choices=("accuracy", "performance"))

    run = commands.add_parser("run", help="Run selected internal qualification cases")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--model", action="append", default=[])
    run.add_argument("--kind", action="append", choices=("accuracy", "performance"))
    run.add_argument("--dataset", action="append", default=[], metavar="ID=PATH")
    run.add_argument("--artifacts", type=Path, default=Path("artifacts/model-benchmark"))
    run.add_argument("--data-root", type=Path)
    run.add_argument("--env-root", type=Path)
    run.add_argument("--bundle-cache", type=Path)
    run.add_argument("--bundle-root", action="append", default=[], type=Path)
    run.add_argument("--runtime-root", type=Path)
    run.add_argument("--trtmc-bench", type=Path)
    run.add_argument("--worker", type=Path)
    run.add_argument(
        "--reference-python",
        action="append",
        default=[],
        metavar="MODEL_OR_FAMILY=PATH",
    )
    run.add_argument("--no-build", action="store_true")
    run.add_argument("--verbose", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        cases = _selected(arguments)
        if arguments.command == "list":
            _list(cases)
            return 0
        return _run(cases, arguments)
    except QualificationError as error:
        print(f"model-benchmark: {error}", file=sys.stderr)
        return 2


def _selected(arguments: argparse.Namespace) -> tuple[QualificationCase, ...]:
    cases = discover(REPOSITORY)
    cases = select(cases, arguments.model)
    kinds = set(arguments.kind or ())
    if kinds:
        cases = tuple(case for case in cases if case.kind in kinds)
    if not cases:
        raise QualificationError("selection contains no benchmark cases")
    return cases


def _list(cases: Sequence[QualificationCase]) -> None:
    print("MODEL\tFAMILY\tKIND\tCASE\tBENCHMARK")
    for case in cases:
        print(f"{case.model}\t{case.family}\t{case.kind}\t{case.name}\t{case.benchmark}")


def _run(cases: Sequence[QualificationCase], arguments: argparse.Namespace) -> int:
    context = context_from_args(arguments, REPOSITORY)
    context.artifacts.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"[{case.kind}] {case.model}/{case.name}", flush=True)
        try:
            result = (
                run_accuracy(case, context)
                if case.kind == "accuracy"
                else run_performance(case, context)
            )
        except QualificationError as error:
            result = {
                "schema_version": "trtmc.qualification-result/v1",
                "case": case.id,
                "kind": case.kind,
                "model": case.model,
                "benchmark": case.benchmark,
                "status": "error",
                "error": str(error),
            }
            write_result(context.case_artifacts(case), result)
        results.append(result)
        print(f"  {result['status']}", flush=True)
    summary = {
        "schema_version": "trtmc.qualification-summary/v1",
        "status": "passed" if all(item.get("status") == "passed" for item in results) else "failed",
        "cases": results,
    }
    _write_summary(context.artifacts, summary)
    print(f"JSON: {context.artifacts / 'report.json'}")
    print(f"HTML: {context.artifacts / 'report.html'}")
    return 0 if summary["status"] == "passed" else 1


def _write_summary(output: Path, summary: Mapping[str, Any]) -> None:
    (output / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for case in summary["cases"]:
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(case.get("model", ""))),
                html.escape(str(case.get("kind", ""))),
                html.escape(str(case.get("case", ""))),
                html.escape(str(case.get("status", ""))),
            )
        )
    document = """<!doctype html><meta charset=\"utf-8\"><title>TRTMC model benchmark</title>
<h1>Internal model benchmark</h1><p>Status: <strong>{status}</strong></p>
<table><thead><tr><th>Model</th><th>Kind</th><th>Case</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table><p><a href=\"report.json\">report.json</a></p>
""".format(status=html.escape(str(summary["status"])), rows="".join(rows))
    (output / "report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

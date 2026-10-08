#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run repository-only model Accuracy and Performance qualification."""

from __future__ import annotations

import argparse
import sys
import traceback
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
from qualification_tests.benchmark_qualification.reporting import write_local_summary  # noqa: E402
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

    aiperf = commands.add_parser(
        "aiperf", help="Run the AIPerf-based TRTMC vs native qualification (apps/aiperf_qual run-all)"
    )
    aiperf.add_argument("--environment", type=Path, required=True, help="machine environment file")
    aiperf.add_argument(
        "--aiperf-python", type=Path, required=True, help="interpreter of the AIPerf environment (setup.sh)"
    )
    aiperf.add_argument("--out-root", type=Path, required=True)
    aiperf.add_argument("--model", action="append", default=[], help="profiles (default: the machine's list)")
    aiperf.add_argument("--shard", help="INDEX/COUNT")
    aiperf.add_argument("--rerun", action="store_true")
    return value


def aiperf_command(arguments: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """The ``trtmc-aiperf-qual run-all`` invocation and its PYTHONPATH for this checkout."""
    command = [str(arguments.aiperf_python), "-m", "trtmc_aiperf_qual", "run-all",
               "--environment", str(arguments.environment), "--out-root", str(arguments.out_root)]
    for model in arguments.model:
        command += ["--profile", model]
    if arguments.shard:
        command += ["--shard", arguments.shard]
    if arguments.rerun:
        command.append("--rerun")
    roots = ("apps/aiperf_qual", "apps/perf_serving", "core/builder", "apps/benchmark", ".")
    return command, {"PYTHONPATH": ":".join(str(REPOSITORY / root) for root in roots)}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "aiperf":
        import os
        import subprocess

        command, env = aiperf_command(arguments)
        return subprocess.run(command, env={**os.environ, **env}, cwd=REPOSITORY / "apps/aiperf_qual").returncode
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
        except Exception as error:
            # A single family must not prevent the campaign from producing a
            # complete report for every other model.
            output = context.case_artifacts(case)
            output.mkdir(parents=True, exist_ok=True)
            (output / "execution.stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
            result = {
                "schema_version": "trtmc.qualification-result/v1",
                "case": case.id,
                "kind": case.kind,
                "model": case.model,
                "benchmark": case.benchmark,
                "status": "error",
                "error": f"unexpected {type(error).__name__}: {error}",
            }
            write_result(output, result)
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
    write_local_summary(output, summary)


if __name__ == "__main__":
    raise SystemExit(main())

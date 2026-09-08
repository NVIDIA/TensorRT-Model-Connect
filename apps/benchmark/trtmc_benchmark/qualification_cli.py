# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line entry point for family-local qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .qualification import (
    SUPPORTED_KINDS,
    QualificationCatalog,
    QualificationError,
    QualificationRunner,
    generate_report,
    load_environment,
    load_plan,
    load_run_configuration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtmc-qualify",
        description="Discover and run family-owned Accuracy and Performance suites.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    _selection_arguments(plan)
    plan.add_argument("-o", "--output", type=Path)

    run = commands.add_parser("run")
    _selection_arguments(run)
    run.add_argument("--environment", type=Path)
    run.add_argument("-o", "--output", required=True, type=Path)

    resume = commands.add_parser("resume")
    resume.add_argument("run_directory", type=Path)

    report = commands.add_parser("report")
    report.add_argument("run_directory", type=Path)
    return parser


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kind", choices=SUPPORTED_KINDS)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--families-root", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command in {"plan", "run"}:
            kind, models, suites, cases, environment_path = _resolve_selection(
                arguments, require_environment=arguments.command == "run"
            )
            plan = QualificationCatalog(arguments.families_root).plan(
                kind,
                models=models,
                suites=suites,
                cases=cases,
            )
            if arguments.command == "plan":
                payload = json.dumps(plan.to_json(), indent=2, sort_keys=True) + "\n"
                if arguments.output:
                    arguments.output.expanduser().resolve().write_text(payload, encoding="utf-8")
                else:
                    print(payload, end="")
                return 0
            assert environment_path is not None
            environment = load_environment(environment_path)
            report = QualificationRunner().run(plan, arguments.output, environment)
            _print_report(report, arguments.output.expanduser().resolve())
            return 0 if report["status"] in {"pass", "observed", "empty"} else 1
        if arguments.command == "resume":
            root = arguments.run_directory.expanduser().resolve()
            plan = load_plan(root / "plan.json")
            environment = load_environment(root / "environment.json")
            report = QualificationRunner().run(plan, root, environment, resume=True)
            _print_report(report, root)
            return 0 if report["status"] in {"pass", "observed", "empty"} else 1
        if arguments.command == "report":
            root = arguments.run_directory.expanduser().resolve()
            report = generate_report(load_plan(root / "plan.json"), root)
            _print_report(report, root)
            return 0 if report["status"] in {"pass", "observed", "empty"} else 1
    except QualificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _resolve_selection(
    arguments: argparse.Namespace, *, require_environment: bool
) -> tuple[str, Sequence[str], Sequence[str], Sequence[str], Path | None]:
    if arguments.run_config:
        if arguments.kind or arguments.model or arguments.suite or arguments.case:
            raise QualificationError(
                "--run-config cannot be combined with --kind, --model, --suite, or --case"
            )
        if getattr(arguments, "environment", None):
            raise QualificationError("--run-config cannot be combined with --environment")
        configured = load_run_configuration(arguments.run_config)
        return (
            configured["kind"],
            configured["models"],
            configured["suites"],
            configured["cases"],
            configured["environment"],
        )
    if not arguments.kind:
        raise QualificationError("provide --kind or --run-config")
    environment = getattr(arguments, "environment", None)
    if require_environment and environment is None:
        raise QualificationError("run requires --environment or --run-config")
    return arguments.kind, arguments.model, arguments.suite, arguments.case, environment


def _print_report(report: dict, root: Path) -> None:
    summary = report["summary"]
    print(f"{report['status']}: {summary['planned']} planned, {summary['results']} result(s)")
    print(f"JSON: {root / 'report.json'}")
    print(f"HTML: {root / 'report.html'}")

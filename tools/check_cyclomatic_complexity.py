#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute and optionally gate C/C++ cyclomatic complexity with lizard."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FunctionMetric:
    nloc: int
    ccn: int
    token: int
    param: int
    length: int
    location: str
    file: str
    function: str
    signature: str
    start_line: int
    end_line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lizard on C/C++ sources and report cyclomatic complexity."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src"],
        help="Paths to scan (default: src).",
    )
    parser.add_argument(
        "--language",
        default="cpp",
        help="Lizard language mode (default: cpp).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many top-complex functions to print (default: 25).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude functions whose file is this path or under this directory. Repeatable.",
    )
    parser.add_argument(
        "--max-ccn",
        type=int,
        default=None,
        help="Fail if any function exceeds this CCN.",
    )
    parser.add_argument(
        "--ccn-threshold",
        type=int,
        default=20,
        help="Threshold used with --max-count-at-or-above (default: 20).",
    )
    parser.add_argument(
        "--max-count-at-or-above",
        type=int,
        default=None,
        help="Fail if count(CCN >= --ccn-threshold) exceeds this value.",
    )
    parser.add_argument(
        "--fail-on-missing-lizard",
        action="store_true",
        help="Fail if lizard is not installed (default: print guidance and return 2).",
    )
    return parser.parse_args()


def run_lizard(language: str, paths: list[str]) -> str:
    cmd = ["lizard", "-l", language, "--csv", *paths]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "lizard is not installed. Install with: pip install lizard"
        ) from exc

    if proc.returncode not in (0, 1):
        # lizard returns 1 when thresholds are violated; with no thresholds it should be 0.
        raise RuntimeError(
            f"lizard failed (rc={proc.returncode}):\n{proc.stderr.strip()}"
        )
    return proc.stdout


def parse_csv(output: str) -> list[FunctionMetric]:
    metrics: list[FunctionMetric] = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 11:
            continue
        try:
            metrics.append(
                FunctionMetric(
                    nloc=int(row[0]),
                    ccn=int(row[1]),
                    token=int(row[2]),
                    param=int(row[3]),
                    length=int(row[4]),
                    location=row[5],
                    file=row[6],
                    function=row[7],
                    signature=row[8],
                    start_line=int(row[9]),
                    end_line=int(row[10]),
                )
            )
        except ValueError:
            continue
    return metrics


def _normalize_path(path: str) -> str:
    return Path(path).as_posix().rstrip("/")


def filter_excluded(metrics: list[FunctionMetric], excludes: list[str]) -> list[FunctionMetric]:
    excluded = [_normalize_path(path) for path in excludes if path]
    if not excluded:
        return metrics

    kept: list[FunctionMetric] = []
    for metric in metrics:
        file_path = _normalize_path(metric.file)
        if any(file_path == path or file_path.startswith(f"{path}/") for path in excluded):
            continue
        kept.append(metric)
    return kept


def print_report(metrics: list[FunctionMetric], top_n: int) -> None:
    if not metrics:
        print("[ccm] No functions found.")
        return

    ccns = [m.ccn for m in metrics]
    nloc_total = sum(m.nloc for m in metrics)
    ccn_sorted = sorted(ccns)
    p90_index = max(0, int(0.9 * len(ccn_sorted)) - 1)

    print("[ccm] Summary")
    print(f"[ccm] Functions: {len(metrics)}")
    print(f"[ccm] NLOC total: {nloc_total}")
    print(f"[ccm] CCN avg: {statistics.mean(ccns):.2f}")
    print(f"[ccm] CCN median: {statistics.median(ccns):.2f}")
    print(f"[ccm] CCN p90: {ccn_sorted[p90_index]}")
    print(f"[ccm] CCN max: {max(ccns)}")
    for threshold in (10, 15, 20, 30, 40, 50):
        count = sum(1 for c in ccns if c >= threshold)
        print(f"[ccm] CCN >= {threshold}: {count}")

    print("[ccm] Top Functions")
    ranked = sorted(metrics, key=lambda m: (m.ccn, m.nloc), reverse=True)
    for idx, metric in enumerate(ranked[:top_n], start=1):
        print(
            f"[ccm] {idx:02d} "
            f"CCN={metric.ccn:>3} NLOC={metric.nloc:>4} "
            f"{metric.file}:{metric.start_line} {metric.function}"
        )


def evaluate_gate(
    metrics: list[FunctionMetric],
    max_ccn: int | None,
    ccn_threshold: int,
    max_count_at_or_above: int | None,
) -> list[str]:
    failures: list[str] = []
    if not metrics:
        return failures

    if max_ccn is not None:
        max_seen = max(m.ccn for m in metrics)
        if max_seen > max_ccn:
            failures.append(f"max CCN {max_seen} exceeds allowed {max_ccn}")

    if max_count_at_or_above is not None:
        count = sum(1 for m in metrics if m.ccn >= ccn_threshold)
        if count > max_count_at_or_above:
            failures.append(
                f"count(CCN >= {ccn_threshold}) {count} exceeds allowed {max_count_at_or_above}"
            )

    return failures


def main() -> int:
    args = parse_args()

    missing_paths = [path for path in args.paths if not Path(path).exists()]
    if missing_paths:
        print(f"[ccm] ERROR: paths do not exist: {missing_paths}", file=sys.stderr)
        return 2

    try:
        csv_output = run_lizard(args.language, args.paths)
    except RuntimeError as exc:
        print(f"[ccm] ERROR: {exc}", file=sys.stderr)
        return 1 if args.fail_on_missing_lizard else 2

    metrics = filter_excluded(parse_csv(csv_output), args.exclude)
    print_report(metrics, args.top)

    failures = evaluate_gate(
        metrics,
        args.max_ccn,
        args.ccn_threshold,
        args.max_count_at_or_above,
    )
    if failures:
        print("[ccm] FAIL")
        for reason in failures:
            print(f"[ccm] - {reason}")
        return 1

    print("[ccm] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

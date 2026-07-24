#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and source-bind the frozen dynamic-memory test manifest.

The implementation plan names both the full CTest baseline and the focused
``dynamic_memory`` CTest/pytest manifests.  This producer owns those exact
commands so a review receipt cannot silently omit a suite, reuse old output,
or claim a dirty tree as the frozen release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA = "trtmc.dynamic-memory-test-manifest/v1"
PYTEST_PATHS = (
    "tests/builder",
    "tests/tools",
    "tests/e2e/test_native_dynamic_memory_graph.py",
)


class ManifestError(RuntimeError):
    """The test manifest was incomplete, failed, or changed source."""


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_dynamic_memory_manifest_boundary", path
    )
    if spec is None or spec.loader is None:
        raise ManifestError(f"cannot load source-state helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _commands(build_dir: Path, python: Path) -> list[tuple[str, list[str]]]:
    pytest_selection = [*PYTEST_PATHS]
    return [
        (
            "build",
            ["cmake", "--build", str(build_dir), "-j"],
        ),
        (
            "build_cpp_tests_and_qualifiers",
            [
                "cmake",
                "--build",
                str(build_dir),
                "-j",
                "--target",
                "trtmc_cpp_tests",
                "trtmc_dynamic_memory_qualify",
                "trtmc_dynamic_memory_surfaces",
                "trtmc_benchmark_worker",
            ],
        ),
        (
            "ctest_manifest_all",
            ["ctest", "--test-dir", str(build_dir), "-N"],
        ),
        (
            "ctest_all",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "--output-on-failure",
            ],
        ),
        (
            "ctest_manifest_dynamic_memory",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-N",
                "-L",
                "dynamic_memory",
            ],
        ),
        (
            "ctest_dynamic_memory",
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "-L",
                "dynamic_memory",
                "--output-on-failure",
            ],
        ),
        (
            "pytest_manifest_dynamic_memory",
            [
                str(python),
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-m",
                "dynamic_memory",
                *pytest_selection,
            ],
        ),
        (
            "pytest_dynamic_memory",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "-m",
                "dynamic_memory",
                *pytest_selection,
            ],
        ),
        (
            "pytest_graph_e2e",
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "tests/e2e/test_native_dynamic_memory_graph.py",
            ],
        ),
    ]


def _manifest_entries(label: str, stdout: str) -> list[str]:
    if label.startswith("ctest_manifest_"):
        return re.findall(r"Test\s+#\d+:\s+(\S+)", stdout)
    if label == "pytest_manifest_dynamic_memory":
        return sorted(
            line.strip()
            for line in stdout.splitlines()
            if "::" in line and not line.lstrip().startswith("=")
        )
    return []


def _run_one(
    label: str,
    argv: Sequence[str],
    *,
    repo_root: Path,
    output_dir: Path,
    environment_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = output_dir / f"{label}.stdout.log"
    stderr_path = output_dir / f"{label}.stderr.log"
    started_ns = time.time_ns()
    environment = os.environ.copy()
    environment.update(environment_overrides or {})
    completed = subprocess.run(
        list(argv),
        cwd=repo_root,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    finished_ns = time.time_ns()
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    entries = _manifest_entries(label, completed.stdout)
    return {
        "label": label,
        "argv": list(argv),
        "cwd": str(repo_root),
        "environment_overrides": dict(environment_overrides or {}),
        "returncode": completed.returncode,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "stdout": str(stdout_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": _sha256(stderr_path),
        "manifest_entries": entries,
        "manifest_count": len(entries),
        "passed": completed.returncode == 0,
    }


def capture(
    *,
    repo_root: Path,
    build_dir: Path,
    python: Path,
    output_dir: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    build_dir = build_dir.resolve()
    # Preserve a virtual environment's interpreter symlink. Resolving
    # ``/opt/venv/bin/python`` to the system executable bypasses the adjacent
    # pyvenv.cfg and silently drops the venv site-packages.
    python = python.absolute()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    boundary = _load_boundary_module()
    source_pre = boundary.source_state_provenance(
        repo_root,
        Path(__file__).resolve(),
        output_dir,
        label="pre",
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "repo_root": str(repo_root),
        "build_dir": str(build_dir),
        "python": str(python),
        "source_state_pre": source_pre,
        "commands": [],
        "passed": False,
    }
    report_path = output_dir / "test-manifest-report.json"
    if not source_pre["exact_head_gate_satisfied"]:
        _write_json(report_path, report)
        raise ManifestError(
            "test-manifest qualification requires a clean exact HEAD"
        )

    failed_label: str | None = None
    for label, argv in _commands(build_dir, python):
        environment_overrides = (
            {
                "TRTMC_BENCH_WORKER": str(
                    build_dir / "trtmc_benchmark_worker"
                )
            }
            if label.startswith("pytest_")
            else None
        )
        result = _run_one(
            label,
            argv,
            repo_root=repo_root,
            output_dir=output_dir,
            environment_overrides=environment_overrides,
        )
        report["commands"].append(result)
        _write_json(report_path, report)
        if not result["passed"]:
            failed_label = label
            break

    source_post = boundary.source_state_provenance(
        repo_root,
        Path(__file__).resolve(),
        output_dir,
        label="post",
    )
    unchanged = (
        source_pre["source_state_sha256"]
        == source_post["source_state_sha256"]
    )
    report["source_state_post"] = source_post
    report["source_state_unchanged"] = unchanged
    report["passed"] = bool(
        failed_label is None
        and
        unchanged
        and source_post["exact_head_gate_satisfied"]
        and all(command["passed"] for command in report["commands"])
    )
    _write_json(report_path, report)
    if failed_label is not None:
        failed = report["commands"][-1]
        raise ManifestError(
            f"test-manifest command failed: {failed_label}; "
            f"see {failed['stderr']}"
        )
    if not report["passed"]:
        raise ManifestError(
            "source state changed while executing the test manifest"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = capture(
        repo_root=args.repo_root,
        build_dir=args.build_dir,
        python=args.python,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "report": str(
                    args.output_dir.resolve() / "test-manifest-report.json"
                ),
                "commands": len(report["commands"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as error:
        print(f"capture_dynamic_memory_test_manifest: {error}", file=sys.stderr)
        raise SystemExit(1)

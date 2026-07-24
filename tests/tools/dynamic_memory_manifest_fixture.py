# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only helpers for strict dynamic-memory build-manifest receipts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


DYNAMIC_TEST_ENTRIES = (
    "tests/builder/test_dynamic_memory_qualification.py::test_manifest_fixture",
    "tests/e2e/test_native_dynamic_memory_graph.py::test_graph_fixture",
)


def load_manifest_module(repo_source_root: Path) -> Any:
    module_path = (
        repo_source_root
        / "tools"
        / "capture_dynamic_memory_test_manifest.py"
    )
    module_name = "_trtmc_test_dynamic_memory_manifest_fixture"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load build-manifest module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def seed_manifest_test_modules(repo_root: Path) -> None:
    """Ensure synthetic repositories have modules referenced by JUnit."""

    for relative in (
        "tests/builder/test_dynamic_memory_qualification.py",
        "tests/e2e/test_native_dynamic_memory_graph.py",
    ):
        path = repo_root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic manifest fixture\n", encoding="utf-8")


def _write_junit(path: Path, entries: tuple[str, ...]) -> None:
    testcases: list[str] = []
    for entry in entries:
        module, *qualifiers = entry.split("::")
        classname = module.removesuffix(".py").replace("/", ".")
        if len(qualifiers) > 1:
            classname = ".".join((classname, *qualifiers[:-1]))
        name = qualifiers[-1]
        testcases.append(
            f'<testcase classname="{classname}" name="{name}" />'
        )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites><testsuite name="pytest">'
        f"{''.join(testcases)}"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )


def complete_command_receipts(
    manifest_module: Any,
    *,
    repo_root: Path,
    build_dir: Path,
    output_dir: Path,
    python: Path,
) -> list[dict[str, Any]]:
    """Create replayable receipts for the producer's complete fixed command set."""

    repo_root = repo_root.resolve()
    build_dir = build_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest_test_modules(repo_root)
    dynamic_entries = list(DYNAMIC_TEST_ENTRIES)
    receipts: list[dict[str, Any]] = []
    for index, (label, argv) in enumerate(
        manifest_module._commands(build_dir, python)
    ):
        stdout = output_dir / f"{label}.stdout.log"
        stderr = output_dir / f"{label}.stderr.log"
        if label.startswith("ctest_manifest_"):
            stdout.write_text(
                "  Test #1: dynamic_memory_fixture\n",
                encoding="utf-8",
            )
        elif label == "pytest_manifest_dynamic_memory":
            stdout.write_text(
                "\n".join(dynamic_entries) + "\n",
                encoding="utf-8",
            )
        else:
            stdout.write_text("ok\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        executed_argv = list(argv)
        expected_entries: list[str] | None = None
        if label == "pytest_dynamic_memory":
            expected_entries = dynamic_entries
        elif label == "pytest_graph_e2e":
            expected_entries = [
                entry
                for entry in dynamic_entries
                if entry.startswith(
                    "tests/e2e/test_native_dynamic_memory_graph.py::"
                )
            ]
        junit: dict[str, Any] | None = None
        if expected_entries is not None:
            junit_path = output_dir / f"{label}.junit.xml"
            _write_junit(junit_path, tuple(expected_entries))
            executed_argv.append(f"--junitxml={junit_path}")
            junit = {
                "path": str(junit_path),
                "sha256": manifest_module._sha256(junit_path),
                "outcomes": manifest_module._pytest_junit_outcomes(
                    junit_path,
                    repo_root=repo_root,
                    expected_entries=expected_entries,
                ),
            }
        started_ns = (index + 1) * 10
        receipt: dict[str, Any] = {
            "label": label,
            "argv": executed_argv,
            "cwd": str(repo_root),
            "environment_overrides": (
                {
                    "TRTMC_BENCH_WORKER": str(
                        build_dir / "trtmc_benchmark_worker"
                    ),
                    "TRTMC_TRT_PLUGIN_LIBRARY": str(
                        build_dir / "libtrtmc_trt_plugins.so"
                    ),
                }
                if label.startswith("pytest_")
                else {}
            ),
            "returncode": 0,
            "started_ns": started_ns,
            "finished_ns": started_ns + 1,
            "stdout": str(stdout),
            "stdout_sha256": manifest_module._sha256(stdout),
            "stderr": str(stderr),
            "stderr_sha256": manifest_module._sha256(stderr),
            "manifest_entries": manifest_module._manifest_entries(
                label, stdout.read_text(encoding="utf-8")
            ),
            "manifest_count": 0,
            "passed": True,
        }
        receipt["manifest_count"] = len(receipt["manifest_entries"])
        if junit is not None:
            receipt["pytest_junit"] = junit
        receipts.append(receipt)
    return receipts

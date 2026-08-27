# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect per-test coverage data from Python tests using coverage.py.

Runs pytest with --cov-context=test, then parses the resulting .coverage
SQLite database to build a {source_file: [test_node_ids]} mapping.
"""

import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

_PYTHON_PACKAGE = "tensorrt_model_connect"
_PYTHON_SOURCE_PREFIX = f"python/{_PYTHON_PACKAGE}"
_SERVER_PACKAGE = "trtmc_server"
_SERVER_SOURCE_PREFIX = f"server/python/{_SERVER_PACKAGE}"


def _normalize_source_path(src_path: str, repo_root: Optional[Path] = None) -> str:
    path_text = str(src_path).replace("\\", "/")
    if repo_root is not None:
        try:
            source_path = Path(src_path)
            if source_path.is_absolute():
                return source_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except (OSError, ValueError):
            pass

    parts = [part for part in path_text.strip("/").split("/") if part]
    for package, source_prefix in (
        (_PYTHON_PACKAGE, _PYTHON_SOURCE_PREFIX),
        (_SERVER_PACKAGE, _SERVER_SOURCE_PREFIX),
    ):
        if package not in parts:
            continue
        package_index = parts.index(package)
        package_tail = "/".join(parts[package_index + 1 :])
        return f"{source_prefix}/{package_tail}" if package_tail else source_prefix

    return path_text.strip("/")


def parse_coverage_db(db_path: Path, repo_root: Optional[Path] = None) -> Dict[str, List[str]]:
    """Parse a coverage.py SQLite database with per-test contexts.

    Returns:
        {source_file_path: [test_node_id, ...]}
        Test node IDs have the |run/|setup/|teardown phase suffix stripped.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Coverage database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT f.path, c.context "
            "FROM line_bits lb "
            "JOIN file f ON lb.file_id = f.id "
            "JOIN context c ON lb.context_id = c.id"
        ).fetchall()
    finally:
        conn.close()

    source_to_tests: Dict[str, set] = {}
    for src_path, context in rows:
        # Strip phase suffix: "test_foo.py::test_bar|run" -> "test_foo.py::test_bar"
        test_id = context.split("|")[0] if "|" in context else context
        if not test_id:
            continue
        normalized_src_path = _normalize_source_path(src_path, repo_root)
        source_to_tests.setdefault(normalized_src_path, set()).add(test_id)

    return {src: sorted(tests) for src, tests in source_to_tests.items()}


def collect_python_coverage(
    repo_root: Path,
    test_targets: Optional[List[str]] = None,
    cov_source: str = "tensorrt_model_connect",
    python_bin: str = "python",
) -> Dict[str, List[str]]:
    """Run Python tests with per-test coverage and return source->tests map.

    Args:
        repo_root: Repository root directory.
        test_targets: pytest target directories (default: builder, tools, and server tests).
        cov_source: Package to measure coverage for.
        python_bin: Python executable to use.

    Returns:
        {source_file_path: [test_node_ids]}
    """
    if test_targets is None:
        test_targets = ["tests/builder", "tests/tools", "server/tests"]

    coverage_file = repo_root / ".coverage-map-gen"

    # Clean up any previous coverage data
    coverage_file.unlink(missing_ok=True)
    for wal in [coverage_file.with_suffix(".db-wal"), coverage_file.with_suffix(".db-shm")]:
        wal.unlink(missing_ok=True)

    coverage_sources = [cov_source]
    if cov_source == _PYTHON_PACKAGE:
        coverage_sources.append(_SERVER_PACKAGE)

    cmd = [
        python_bin, "-m", "pytest",
        *test_targets,
        *(f"--cov={source}" for source in coverage_sources),
        "--cov-context=test",
        "--cov-report=",  # suppress report output
        "-q",
        "--ignore=tests/builder/test_cli.py",
    ]

    env = {
        "COVERAGE_FILE": str(coverage_file),
        "PYTHONPATH": os.pathsep.join(
            (
                str(repo_root / "python"),
                str(repo_root / "server" / "python"),
                str(repo_root),
            )
        ),
    }

    full_env = {**os.environ, **env}

    subprocess.run(
        cmd, cwd=repo_root, env=full_env, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    result = parse_coverage_db(coverage_file, repo_root=repo_root)

    coverage_file.unlink(missing_ok=True)

    return result

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect per-test coverage data from C++ tests using gcov/gcovr.

Runs each ctest binary individually with gcda reset between runs,
captures gcovr JSON output, and builds a {source_file: [test_names]} mapping.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set


def parse_gcovr_json(json_path: Path, repo_root: Path) -> Set[str]:
    """Parse a gcovr JSON report and return set of covered source files.

    Only includes files with at least one covered line.
    Paths are returned relative to repo_root.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    covered = set()
    repo_prefix = str(repo_root).rstrip("/") + "/"
    for file_entry in data.get("files", []):
        if file_entry.get("line_covered", 0) > 0:
            path = file_entry["filename"]
            if path.startswith(repo_prefix):
                path = path[len(repo_prefix):]
            covered.add(path)
    return covered


def build_cpp_map_from_jsons(
    json_dir: Path,
    repo_root: Path,
) -> Dict[str, List[str]]:
    """Build source->test mapping from per-test gcovr JSON files.

    Expects files named <test_name>.json in json_dir.
    """
    source_to_tests: Dict[str, set] = {}
    for json_path in sorted(json_dir.glob("*.json")):
        test_name = json_path.stem
        covered_files = parse_gcovr_json(json_path, repo_root)
        for src in covered_files:
            source_to_tests.setdefault(src, set()).add(test_name)
    return {src: sorted(tests) for src, tests in source_to_tests.items()}


def _list_ctest_names(build_dir: Path) -> List[str]:
    """List all registered ctest names."""
    result = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N", "--quiet"],
        capture_output=True, text=True, check=True,
    )
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Test #"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
    return names


def collect_cpp_coverage(
    repo_root: Path,
    build_dir: Path,
    output_dir: Optional[Path] = None,
    gcovr_filters: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Run each ctest individually with gcov and build source->test mapping.

    Args:
        repo_root: Repository root directory.
        build_dir: CMake build directory (must be built with --coverage).
        output_dir: Where to write per-test gcovr JSONs (default: temp dir).
        gcovr_filters: gcovr --filter values (default: src/, include/, and server/native/).
    """
    import tempfile

    if gcovr_filters is None:
        gcovr_filters = [
            str(repo_root / "src"),
            str(repo_root / "include"),
            str(repo_root / "server" / "native"),
        ]

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="cpp_covmap_"))

    output_dir.mkdir(parents=True, exist_ok=True)
    test_names = _list_ctest_names(build_dir)

    for test_name in test_names:
        for gcda in build_dir.rglob("*.gcda"):
            gcda.unlink()

        subprocess.run(
            ["ctest", "--test-dir", str(build_dir), "-R", f"^{test_name}$",
             "--output-on-failure"],
            capture_output=True, text=True,
        )

        gcovr_cmd = [
            "gcovr",
            "--root", str(repo_root),
            "--object-directory", str(build_dir),
            "--json", "-o", str(output_dir / f"{test_name}.json"),
            "--exclude", str(repo_root / "tests"),
        ]
        for f in gcovr_filters:
            gcovr_cmd.extend(["--filter", f])

        subprocess.run(gcovr_cmd, capture_output=True, text=True)

    return build_cpp_map_from_jsons(output_dir, repo_root)

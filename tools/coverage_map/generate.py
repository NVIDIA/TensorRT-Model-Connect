#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a unified coverage map from Python and C++ test coverage data.

Usage:
    python tools/coverage_map/generate.py --output coverage_map.json
    python tools/coverage_map/generate.py --python-only --output coverage_map.json
    python tools/coverage_map/generate.py --cpp-only --output coverage_map.json
    python tools/coverage_map/generate.py --validate coverage_map.json
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def merge_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge multiple source->tests mappings. Union test lists, deduplicate, sort."""
    merged: Dict[str, set] = {}
    for m in maps:
        for src, tests in m.items():
            merged.setdefault(src, set()).update(tests)
    return {src: sorted(tests) for src, tests in sorted(merged.items())}


def validate_map(
    source_to_tests: Dict[str, List[str]],
    repo_root: Path,
) -> List[str]:
    """Check that source files in the map still exist on disk.

    Returns list of warning strings. Empty list means all clean.
    """
    warnings = []
    for src_path in source_to_tests:
        full_path = repo_root / src_path
        if not full_path.exists():
            warnings.append(f"Source file no longer exists: {src_path}")
    return warnings


def load_coverage_map(path: Path) -> Optional[Dict[str, List[str]]]:
    """Load a coverage_map.json and return the source_to_tests dict.

    Returns None if the file doesn't exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("source_to_tests", {})


def _get_head_commit(repo_root: Path) -> str:
    """Get the current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=repo_root,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def generate(
    repo_root: Path,
    output_path: Path,
    python_only: bool = False,
    cpp_only: bool = False,
    build_dir: Optional[Path] = None,
    python_bin: str = "python",
) -> Dict[str, List[str]]:
    """Generate the unified coverage map."""
    from .python_collector import collect_python_coverage
    from .cpp_collector import collect_cpp_coverage

    py_map: Dict[str, List[str]] = {}
    cpp_map: Dict[str, List[str]] = {}
    py_count = 0
    cpp_count = 0

    if not cpp_only:
        print("[coverage-map] Collecting Python coverage...", file=sys.stderr)
        py_map = collect_python_coverage(repo_root, python_bin=python_bin)
        py_count = len({t for tests in py_map.values() for t in tests})
        print(f"[coverage-map] Python: {len(py_map)} source files, {py_count} tests",
              file=sys.stderr)

    if not python_only:
        if build_dir is None:
            build_dir = repo_root / "build"
        print("[coverage-map] Collecting C++ coverage...", file=sys.stderr)
        cpp_map = collect_cpp_coverage(repo_root, build_dir)
        cpp_count = len({t for tests in cpp_map.values() for t in tests})
        print(f"[coverage-map] C++: {len(cpp_map)} source files, {cpp_count} tests",
              file=sys.stderr)

    merged = merge_maps(py_map, cpp_map)

    output = {
        "meta": {
            "commit": _get_head_commit(repo_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python_tests": py_count,
            "cpp_tests": cpp_count,
        },
        "source_to_tests": merged,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[coverage-map] Written to {output_path} "
          f"({len(merged)} source files)", file=sys.stderr)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate coverage map for test selection.")
    parser.add_argument("--output", "-o", help="Output coverage_map.json path")
    parser.add_argument("--python-only", action="store_true", help="Only collect Python coverage")
    parser.add_argument("--cpp-only", action="store_true", help="Only collect C++ coverage")
    parser.add_argument("--build-dir", default=None, help="CMake build directory")
    parser.add_argument("--python-bin", default="python", help="Python executable")
    parser.add_argument("--repo-root", default=None, help="Repository root (default: auto)")
    parser.add_argument("--validate", metavar="MAP_PATH",
                        help="Validate an existing coverage map")
    args = parser.parse_args()

    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            )
            repo_root = Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            repo_root = Path.cwd()

    if args.validate:
        source_to_tests = load_coverage_map(Path(args.validate))
        if source_to_tests is None:
            print(f"ERROR: Cannot load {args.validate}", file=sys.stderr)
            return 1
        warnings = validate_map(source_to_tests, repo_root)
        if warnings:
            for w in warnings:
                print(f"  WARN: {w}", file=sys.stderr)
            print(f"Validation: {len(warnings)} warnings.", file=sys.stderr)
        else:
            print("Validation passed: all source files exist.", file=sys.stderr)
        return 0

    if not args.output:
        parser.error("--output/-o is required unless --validate is used")

    build_dir = Path(args.build_dir) if args.build_dir else None
    generate(
        repo_root=repo_root,
        output_path=Path(args.output),
        python_only=args.python_only,
        cpp_only=args.cpp_only,
        build_dir=build_dir,
        python_bin=args.python_bin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

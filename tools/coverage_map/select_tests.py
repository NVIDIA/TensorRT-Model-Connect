#!/usr/bin/env python3
"""Select tests to run based on changed files and a coverage map.

Given a list of changed files and a coverage_map.json, determines which
specific unit tests need to run. Falls back to full-tier execution for
files not present in the coverage map (zero false negatives).

Usage:
    python tools/coverage_map/select_tests.py --coverage-map coverage_map.json --files file1,file2
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_NO_IMPACT_PATTERNS = [
    r"^docs/",
    r"^\.gitignore$",
    r"^\.clang-format$",
    r"^\.editorconfig$",
    r"^\.claude/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^recovery-",
    r"^scripts/",
]


@dataclass
class SelectionResult:
    """Per-tier test selection result."""
    cpp_tests: List[str] = field(default_factory=list)
    builder_tests: List[str] = field(default_factory=list)
    tools_tests: List[str] = field(default_factory=list)
    fallback_tiers: List[str] = field(default_factory=list)


def _is_no_impact(path: str) -> bool:
    """Check if a file path has no impact on unit tests."""
    if path == "tools/test_impact.py" or path.startswith(".github/"):
        return False
    if path.endswith(".md"):
        return True
    for pattern in _NO_IMPACT_PATTERNS:
        if re.match(pattern, path):
            return True
    if path.startswith("tools/") and not path.startswith("tools/coverage_map/"):
        return True
    if path.startswith("tests/e2e"):
        return True
    return False


def _classify_tier(path: str) -> Optional[str]:
    """Determine which unit test tier a source file belongs to."""
    if path.startswith("src/") or path.startswith("include/"):
        return "cpp"
    if path == "CMakeLists.txt" or path.startswith("cmake/"):
        return "cpp"
    if path.startswith("python/tensorrt_model_connect/"):
        return "builder"
    if path.startswith("tests/builder/"):
        return "builder"
    if path.startswith("tests/cpp/"):
        return "cpp"
    if path.startswith("tests/tools/"):
        return "tools"
    if path == "tools/test_impact.py" or path.startswith(".github/"):
        return "tools"
    return None


def _direct_python_test_tier(path: str) -> Optional[str]:
    """Return the tier for Python test files that pytest can run directly."""
    if path.startswith("tests/builder/"):
        return "builder"
    if path.startswith("tests/tools/") or path.startswith("tests/e2e_harness/test_"):
        return "tools"
    return None


def select_tests(
    changed_files: List[str],
    source_to_tests: Dict[str, List[str]],
) -> SelectionResult:
    """Select tests based on changed files and coverage map.

    For each changed file:
    - If it's in the coverage map: select the specific tests that cover it.
    - If it's a source file NOT in the map: fall back to running all tests
      in that tier (zero false negatives).
    - If it's a no-impact file (docs, scripts, etc.): skip.
    """
    cpp_tests: set = set()
    builder_tests: set = set()
    tools_tests: set = set()
    fallback_tiers: set = set()

    for path in changed_files:
        path = path.replace("\\", "/").strip("/")

        direct_tier = _direct_python_test_tier(path)
        if direct_tier == "builder":
            builder_tests.add(path)
            continue
        if direct_tier == "tools":
            tools_tests.add(path)
            continue

        if _is_no_impact(path):
            continue

        tier = _classify_tier(path)
        if tier is None:
            continue

        if path in source_to_tests:
            tests = source_to_tests[path]
            for test_id in tests:
                if "::" in test_id or test_id.startswith("tests/"):
                    builder_tests.add(test_id)
                else:
                    cpp_tests.add(test_id)
        else:
            fallback_tiers.add(tier)

    return SelectionResult(
        cpp_tests=sorted(cpp_tests),
        builder_tests=sorted(builder_tests),
        tools_tests=sorted(tools_tests),
        fallback_tiers=sorted(fallback_tiers),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select unit tests based on coverage map and changed files.",
    )
    parser.add_argument("--coverage-map", required=True, help="Path to coverage_map.json")
    parser.add_argument("--files", help="Comma-separated list of changed files")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    args = parser.parse_args()

    from .generate import load_coverage_map

    source_to_tests = load_coverage_map(Path(args.coverage_map))
    if source_to_tests is None:
        print("ERROR: Coverage map not found. Run all tests.", file=sys.stderr)
        return 1

    changed = [f.strip() for f in args.files.split(",") if f.strip()] if args.files else []
    result = select_tests(changed, source_to_tests)

    if args.json_output:
        print(json.dumps({
            "cpp_tests": result.cpp_tests,
            "builder_tests": result.builder_tests,
            "tools_tests": result.tools_tests,
            "fallback_tiers": result.fallback_tiers,
        }, indent=2))
    else:
        if result.cpp_tests:
            print(f"C++ tests ({len(result.cpp_tests)}):")
            for t in result.cpp_tests:
                print(f"  {t}")
        if result.builder_tests:
            print(f"Builder tests ({len(result.builder_tests)}):")
            for t in result.builder_tests:
                print(f"  {t}")
        if result.fallback_tiers:
            print(f"Fallback tiers (run all): {', '.join(result.fallback_tiers)}")
        if not result.cpp_tests and not result.builder_tests and not result.fallback_tiers:
            print("No unit tests affected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

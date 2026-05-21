#!/usr/bin/env python3
"""Verify documentation file references and numerical claims.

ISO 26262-6 §7.4.1 compliance gate: architecture docs must describe only
implemented code. This tool catches phantom file references and stale counts.

Usage:
    python tools/check_doc_file_references.py [--strict] [website/docs/wiki/]

Exit codes:
    0: All checks passed
    1: Errors found (phantom paths or strict mode with warnings)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    level: str  # ERROR or WARNING
    doc_file: str
    line_no: int
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.doc_file}:{self.line_no}: {self.message}"


@dataclass
class CheckReport:
    findings: List[Finding] = field(default_factory=list)
    docs_scanned: int = 0
    paths_checked: int = 0
    claims_checked: int = 0

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "ERROR"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.level == "WARNING"]


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

# Patterns that look like repo-relative file paths.
# Match backtick-quoted paths or bare paths in table cells.
_PATH_PREFIXES = (
    "python/",
    "src/",
    "include/",
    "tests/",
    "tensorrt_model_connect/",
    "tools/",
    "scripts/",
    "docs/",
    "website/",
)

# Regex: captures a path starting with one of the known prefixes.
# Handles:
#   `src/foo/bar.cpp`           (backtick-quoted)
#   | `src/foo/bar.cpp` |       (in markdown table)
#   `src/foo/bar.h/cpp`         (h/cpp shorthand -- we handle this specially)
_PATH_RE = re.compile(
    r"`("
    + "|".join(re.escape(p) for p in _PATH_PREFIXES)
    + r")[^`\s]*`"
)

# Paths containing angle-bracket placeholders like <family> or <model-name>
# are intentional templates, not real file paths.
_PLACEHOLDER_RE = re.compile(r"<[^>]+>")

# Some docs use shorthand like `file.h/cpp` meaning both file.h and file.cpp exist.
_H_CPP_SHORTHAND = re.compile(r"^(.+)\.(h)/cpp$")
_H_HPP_SHORTHAND = re.compile(r"^(.+)\.(h)/hpp$")


def _expand_path(raw: str) -> List[str]:
    """Expand a single extracted path into one or more concrete paths to check."""
    # Handle h/cpp shorthand: foo.h/cpp -> [foo.h, foo.cpp]
    m = _H_CPP_SHORTHAND.match(raw)
    if m:
        base, _ = m.group(1), m.group(2)
        return [f"{base}.h", f"{base}.cpp"]
    m = _H_HPP_SHORTHAND.match(raw)
    if m:
        base, _ = m.group(1), m.group(2)
        return [f"{base}.h", f"{base}.hpp"]
    return [raw]


def extract_path_references(
    content: str, doc_file: str
) -> List[Tuple[int, str]]:
    """Return (line_number, path) pairs found in the document."""
    results: List[Tuple[int, str]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        for match in _PATH_RE.finditer(line):
            raw = match.group(0)[1:-1]  # strip backticks
            # Skip wildcard globs -- they are patterns, not literal paths
            if "*" in raw:
                continue
            # Skip template/placeholder paths like <family>.py
            if _PLACEHOLDER_RE.search(raw):
                continue
            for expanded in _expand_path(raw):
                # Strip trailing punctuation that may leak in
                expanded = expanded.rstrip(".,;:)")
                # Skip directory references that end with /
                # (we still check them, just as directories)
                results.append((line_no, expanded))
    return results


# ---------------------------------------------------------------------------
# Numerical claim extraction
# ---------------------------------------------------------------------------

# Patterns for numerical claims.  Each tuple:
#   (compiled regex, claim_kind, group index for the number)
_CLAIM_PATTERNS: List[Tuple[re.Pattern, str, int]] = [
    # "68 manifests", "68 JSON manifest files", "68 JSON manifests", "50 per-model JSON manifests"
    (re.compile(r"\b(\d+)\s+(?:per-model\s+)?(?:JSON\s+)?manifest(?:s|[\s-]+file)", re.I), "manifests", 1),
    # "50 models" in E2E context (e.g. "All 50 models")
    (re.compile(r"\b(?:All\s+)?(\d+)\s+models\b", re.I), "models_e2e", 1),
    # "53 auto-discovered plugins", "50 plugins"
    (re.compile(r"\b(\d+)\s+(?:auto-discovered\s+)?plugin", re.I), "family_plugins", 1),
    # "74 test files" in builder context
    (re.compile(r"\b(\d+)\s+test\s+file", re.I), "test_files_generic", 1),
    # "61 test executables" in C++ context
    (re.compile(r"\b(\d+)\s+test\s+executable", re.I), "cpp_test_executables", 1),
    # "20 test files" in tools context -- captured by generic
    # "N .cpp files"
    (re.compile(r"\b(\d+)\s+\.cpp\s+file", re.I), "cpp_files", 1),
    # "N family" or "N families"
    (re.compile(r"\b(\d+)\s+famil(?:y|ies)\b", re.I), "families", 1),
    # Pipe-separated table counts: "| 74 |" or "| 61 |"
    # These appear in summary tables; we match them via the surrounding context
]

# Context keywords that disambiguate generic "N test files" claims.
_BUILDER_CONTEXT_KEYWORDS = {"builder", "python builder", "tests/builder"}
_CPP_CONTEXT_KEYWORDS = {"c++", "cpp", "runtime unit", "tests/cpp"}
_TOOLS_CONTEXT_KEYWORDS = {"tools", "self-test", "tests/tools"}


def _get_actual_counts(repo_root: Path) -> dict:
    """Compute actual file counts from the repo."""
    counts = {}

    # E2E manifests
    manifest_dir = repo_root / "tests" / "e2e" / "models"
    if manifest_dir.is_dir():
        counts["manifests"] = len(
            [f for f in os.listdir(manifest_dir) if f.endswith(".json")]
        )
    else:
        counts["manifests"] = 0

    # Family plugins (excluding __init__.py and base.py)
    families_dir = repo_root / "python" / "tensorrt_model_connect" / "families"
    if families_dir.is_dir():
        flat_plugins = [
            f for f in families_dir.iterdir()
            if f.is_file() and f.suffix == ".py" and f.name not in ("__init__.py", "base.py")
        ]
        package_plugins = [
            d / "plugin.py" for d in families_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_") and (d / "plugin.py").is_file()
        ]
        counts["family_plugins"] = len(
            flat_plugins + package_plugins
        )
    else:
        counts["family_plugins"] = 0

    # Total family .py files (for "N Python files in the families directory" claims)
    if families_dir.is_dir():
        counts["families_total_py"] = len(
            [
                p for p in families_dir.rglob("*.py")
                if "__pycache__" not in p.parts
            ]
        )
    else:
        counts["families_total_py"] = 0

    # C++ test files
    cpp_test_dir = repo_root / "tests" / "cpp"
    if cpp_test_dir.is_dir():
        counts["cpp_tests"] = len(
            [f for f in os.listdir(cpp_test_dir) if f.endswith(".cpp")]
        )
    else:
        counts["cpp_tests"] = 0

    # Builder test files
    builder_test_dir = repo_root / "tests" / "builder"
    if builder_test_dir.is_dir():
        counts["builder_tests"] = len(
            [
                f
                for f in os.listdir(builder_test_dir)
                if f.startswith("test_") and f.endswith(".py")
            ]
        )
    else:
        counts["builder_tests"] = 0

    # Tools test files
    tools_test_dir = repo_root / "tests" / "tools"
    if tools_test_dir.is_dir():
        counts["tools_tests"] = len(
            [
                f
                for f in os.listdir(tools_test_dir)
                if f.startswith("test_") and f.endswith(".py")
            ]
        )
    else:
        counts["tools_tests"] = 0

    return counts


def _surrounding_context(lines: List[str], line_no: int, window: int = 5) -> str:
    """Get surrounding lines as lowercase text for disambiguation."""
    start = max(0, line_no - window - 1)
    end = min(len(lines), line_no + window)
    return " ".join(lines[start:end]).lower()


def extract_numerical_claims(
    content: str, doc_file: str, actual_counts: dict
) -> List[Finding]:
    """Check numerical claims in the document against actual file counts."""
    findings: List[Finding] = []
    lines = content.splitlines()

    for line_no, line in enumerate(lines, start=1):
        # Skip markdown headings (e.g. "### 3.4 Plugin Auto-Discovery")
        # which contain section numbers that look like counts.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue

        ctx = _surrounding_context(lines, line_no)

        for pattern, claim_kind, group_idx in _CLAIM_PATTERNS:
            for match in pattern.finditer(line):
                claimed = int(match.group(group_idx))

                # Disambiguate generic claims using context
                if claim_kind == "test_files_generic":
                    if any(kw in ctx for kw in _BUILDER_CONTEXT_KEYWORDS):
                        actual_key = "builder_tests"
                        label = "builder test files (test_*.py)"
                    elif any(kw in ctx for kw in _CPP_CONTEXT_KEYWORDS):
                        actual_key = "cpp_tests"
                        label = "C++ test files (.cpp)"
                    elif any(kw in ctx for kw in _TOOLS_CONTEXT_KEYWORDS):
                        actual_key = "tools_tests"
                        label = "tools test files (test_*.py)"
                    else:
                        continue  # ambiguous context, skip
                elif claim_kind == "manifests":
                    actual_key = "manifests"
                    label = "E2E model manifests (tests/e2e/models/*.json)"
                elif claim_kind == "models_e2e":
                    # "N models" claims -- map to manifest count
                    actual_key = "manifests"
                    label = "E2E model manifests (tests/e2e/models/*.json)"
                elif claim_kind == "family_plugins":
                    actual_key = "family_plugins"
                    label = "family plugins (excluding __init__.py and base.py)"
                elif claim_kind == "cpp_test_executables":
                    actual_key = "cpp_tests"
                    label = "C++ test files (.cpp)"
                elif claim_kind == "cpp_files":
                    actual_key = "cpp_tests"
                    label = "C++ test files (.cpp)"
                elif claim_kind == "families":
                    # "N families" is too ambiguous -- skip
                    continue
                else:
                    continue

                actual = actual_counts.get(actual_key)
                if actual is None:
                    continue

                if claimed != actual:
                    # Check if the claim might refer to total py files
                    # in families dir (including __init__ and base)
                    if claim_kind == "family_plugins":
                        total = actual_counts.get("families_total_py", 0)
                        if claimed == total:
                            # Claim says "N Python files" which includes
                            # __init__ + base -- that is correct
                            continue

                    findings.append(
                        Finding(
                            level="WARNING",
                            doc_file=doc_file,
                            line_no=line_no,
                            message=(
                                f"Numerical claim '{match.group(0).strip()}' "
                                f"says {claimed} but actual count of "
                                f"{label} is {actual}"
                            ),
                        )
                    )

    return findings


# ---------------------------------------------------------------------------
# Main check logic
# ---------------------------------------------------------------------------

def check_docs(
    scan_dir: Path, repo_root: Path
) -> CheckReport:
    """Scan markdown files and verify references."""
    report = CheckReport()
    actual_counts = _get_actual_counts(repo_root)

    # Find all .md files
    md_files: List[Path] = []
    for root_dir, _dirs, files in os.walk(scan_dir):
        for fname in sorted(files):
            if fname.endswith(".md"):
                md_files.append(Path(root_dir) / fname)

    report.docs_scanned = len(md_files)

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8", errors="replace")
        rel_doc = str(md_path.relative_to(repo_root))

        # --- Check file path references ---
        refs = extract_path_references(content, rel_doc)
        for line_no, ref_path in refs:
            report.paths_checked += 1
            full = repo_root / ref_path
            # Check both as file and directory
            if not full.exists():
                report.findings.append(
                    Finding(
                        level="ERROR",
                        doc_file=rel_doc,
                        line_no=line_no,
                        message=f"Path does not exist: {ref_path}",
                    )
                )

        # --- Check numerical claims ---
        claim_findings = extract_numerical_claims(content, rel_doc, actual_counts)
        report.findings.extend(claim_findings)
        report.claims_checked += len(claim_findings)

    # Add actual counts to report for summary
    report._actual_counts = actual_counts  # type: ignore[attr-defined]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify documentation file references and numerical claims.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scan_dir",
        nargs="?",
        default="website/docs/wiki/",
        help="Directory to scan for .md files (default: website/docs/wiki/)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 on warnings too (not just errors)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve repo root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        # tools/check_doc_file_references.py -> repo root is ../
        repo_root = Path(__file__).resolve().parent.parent

    scan_dir = (repo_root / args.scan_dir).resolve()
    if not scan_dir.is_dir():
        print(f"ERROR: Scan directory does not exist: {scan_dir}", file=sys.stderr)
        return 1

    print(f"Repository root: {repo_root}")
    print(f"Scanning: {scan_dir}")
    print()

    report = check_docs(scan_dir, repo_root)

    # Print actual counts
    counts = getattr(report, "_actual_counts", {})
    if counts:
        print("=== Actual file counts ===")
        print(f"  E2E manifests (tests/e2e/models/*.json):           {counts.get('manifests', '?')}")
        print(f"  Family plugins (excl __init__/base):               {counts.get('family_plugins', '?')}")
        print(f"  Family dir total .py files:                        {counts.get('families_total_py', '?')}")
        print(f"  C++ test files (tests/cpp/*.cpp):                  {counts.get('cpp_tests', '?')}")
        print(f"  Builder test files (tests/builder/test_*.py):      {counts.get('builder_tests', '?')}")
        print(f"  Tools test files (tests/tools/test_*.py):          {counts.get('tools_tests', '?')}")
        print()

    # Print findings grouped by level
    errors = report.errors
    warnings = report.warnings

    if errors:
        print(f"=== ERRORS ({len(errors)}) ===")
        for f in errors:
            print(f"  {f}")
        print()

    if warnings:
        print(f"=== WARNINGS ({len(warnings)}) ===")
        for f in warnings:
            print(f"  {f}")
        print()

    # Summary
    print("=== Summary ===")
    print(f"  Documents scanned:    {report.docs_scanned}")
    print(f"  Path references:      {report.paths_checked}")
    print(f"  Errors (phantom paths): {len(errors)}")
    print(f"  Warnings (stale counts): {len(warnings)}")

    if not errors and not warnings:
        print("\nAll checks passed.")
        return 0

    if errors:
        print(f"\nFAILED: {len(errors)} phantom path(s) found.")
        return 1

    if args.strict and warnings:
        print(f"\nFAILED (strict mode): {len(warnings)} stale count(s) found.")
        return 1

    print("\nPassed with warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

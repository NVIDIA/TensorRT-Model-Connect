#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify documentation file references and numerical claims.

ISO 26262-6 §7.4.1 compliance gate: architecture docs must describe only
implemented code. This tool catches phantom file references and stale counts.

Usage:
    python tools/check_doc_file_references.py [--strict] [website/docs/wiki/]
    python tools/check_doc_file_references.py --strict --tracked

Exit codes:
    0: All checks passed
    1: Errors found (phantom paths or strict mode with warnings)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple


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
    ".agents/",
    ".github/",
    "cmake/",
    "examples/",
    "python/",
    "src/",
    "include/",
    "plugins/",
    "reports/",
    "tests/",
    "tensorrt_model_connect/",
    "tools/",
    "scripts/",
    "docs/",
    "website/",
)
_ROOT_PATHS = (
    "AGENTS.md",
    "CMakeLists.txt",
    "CONTRIBUTING.md",
    "Dockerfile",
    "README.md",
    "pyproject.toml",
)

# Regex: captures a backtick span that starts with one of the known prefixes.
# Handles:
#   `src/foo/bar.cpp`           (backtick-quoted)
#   `src/foo/bar.h/cpp`         (h/cpp shorthand -- we handle this specially)
_PATH_RE = re.compile(r"`(" + "|".join(re.escape(p) for p in _PATH_PREFIXES) + r")[^`\s]*`")
_ROOT_PATH_RE = re.compile(r"`(" + "|".join(re.escape(path) for path in _ROOT_PATHS) + r")`")

# Repository paths embedded in shell fences do not normally have an individual
# pair of backticks, for example ``python3 tools/check.py``.  Scan tokens from
# shell fences as a second source without treating arbitrary prose as a shell
# command.
_BARE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])(" + "|".join(re.escape(p) for p in _PATH_PREFIXES) + r")[^\s`'\"|]+"
)
_OPEN_FENCE_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>.*)$")
_SHELL_FENCE_LANGUAGES = {"bash", "sh", "shell"}
_SHELL_OUTPUT_PATH_PREFIX_RE = re.compile(
    r"(?:^|\s)(?:-o|--output|--output-dir|--output-json|--json)(?:=|\s+)$"
)

# Paths containing angle-bracket placeholders like <family> or <model-name>
# are intentional templates, not real file paths.
_PLACEHOLDER_RE = re.compile(r"<[^>]+>|\{[^{}]+\}")

# Some docs use shorthand like `file.h/cpp` meaning both file.h and file.cpp exist.
_H_CPP_SHORTHAND = re.compile(r"^(.+)\.(h)/cpp$")
_H_HPP_SHORTHAND = re.compile(r"^(.+)\.(h)/hpp$")

# Truth surfaces retired by the model-owned layout.  These need a dedicated
# plain-text check because architecture prose and complete command snippets are
# not always enclosed in a path-only code span.
_KNOWN_RETIRED_SURFACES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?<![A-Za-z0-9_])src/runtime/plugins(?:/|\b)"),
        "src/runtime/plugins",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])src/runtime/pipelines(?:/|\b)"),
        "src/runtime/pipelines",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])"
            r"python/tensorrt_model_connect/graph_ops\.py\b"
        ),
        "python/tensorrt_model_connect/graph_ops.py",
    ),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])"
            r"python/tensorrt_model_connect/graph_blocks\.py\b"
        ),
        "python/tensorrt_model_connect/graph_blocks.py",
    ),
    (
        re.compile(r"(?<![A-Za-z0-9_])tools/nsight_collect\.py\b"),
        "tools/nsight_collect.py",
    ),
    (
        re.compile(r"/workspace/users/yifeif(?:/|\b)"),
        "personal /workspace/users/yifeif path",
    ),
)
_NONCURRENT_CONTEXT_RE = re.compile(
    r"\b(?:historical|history-only|archived|deprecated|retired|removed|"
    r"no longer|does not exist|not a replay|non-replayable|"
    r"old path|former path)\b",
    re.IGNORECASE,
)
_NONCURRENT_DOCUMENT_RE = re.compile(
    r"(?:historical (?:evidence )?snapshot|historical record|"
    r"status:\s*(?:archived|deprecated|historical)|not a replay runbook)",
    re.IGNORECASE,
)
_GENERIC_RUNTIME_STRATEGIES = (
    "decoder_kv_cache",
    "decoder_moe",
    "embedding",
    "encoder_only",
    "neural_operator",
    "object_detection",
    "prompted_segmentation",
    "reranking",
    "segmentation",
    "speech_to_text",
    "speech_to_text_rnnt",
    "vision_language",
)
_GENERIC_STRATEGY_RE = re.compile(
    r"`(" + "|".join(re.escape(key) for key in _GENERIC_RUNTIME_STRATEGIES) + r")`"
)
_STRATEGY_WORD_RE = re.compile(r"\b(?:runtime[_ -]?)?strateg(?:y|ies)\b", re.I)
_GENERIC_STRATEGY_EXEMPT_RE = re.compile(
    r"\b(?:do not|does not have|generic|not the current|retired|task[_ -]strategy|"
    r"task label|no model-owned|not a runtime)\b",
    re.I,
)


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


def _clean_extracted_path(raw: str) -> str:
    """Remove Markdown/shell punctuation adjacent to a path token."""
    return raw.split("::", 1)[0].split("#", 1)[0].rstrip("\\.,;:)]}")


def _shell_fence_path_references(content: str) -> List[Tuple[int, str]]:
    """Extract repo-local path tokens from bash/sh/shell fenced blocks."""
    results: List[Tuple[int, str]] = []
    fence_char = ""
    fence_length = 0
    shell_fence = False

    for line_no, line in enumerate(content.splitlines(), start=1):
        if not fence_char:
            match = _OPEN_FENCE_RE.match(line)
            if not match:
                continue
            fence = match.group("fence")
            info = match.group("info").strip()
            language = (
                info.split(maxsplit=1)[0].strip("{}").removeprefix(".").lower() if info else ""
            )
            fence_char = fence[0]
            fence_length = len(fence)
            shell_fence = language in _SHELL_FENCE_LANGUAGES
            continue

        if re.match(
            rf"^[ \t]*{re.escape(fence_char)}{{{fence_length},}}[ \t]*$",
            line,
        ):
            fence_char = ""
            fence_length = 0
            shell_fence = False
            continue

        if not shell_fence:
            continue
        for match in _BARE_PATH_RE.finditer(line):
            # Output destinations need not exist before a documented command
            # runs.  Keep validating the same path when it is mentioned as
            # prose or used as an input, but do not classify an explicit
            # output argument as a phantom input reference.
            if _SHELL_OUTPUT_PATH_PREFIX_RE.search(line[: match.start()]):
                continue
            raw = _clean_extracted_path(match.group(0))
            if "*" in raw or "?" in raw or _PLACEHOLDER_RE.search(raw):
                continue
            for expanded in _expand_path(raw):
                results.append((line_no, expanded))

    return results


def extract_path_references(content: str, doc_file: str) -> List[Tuple[int, str]]:
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
                expanded = _clean_extracted_path(expanded)
                results.append((line_no, expanded))
        for match in _ROOT_PATH_RE.finditer(line):
            results.append((line_no, match.group(1)))

    results.extend(_shell_fence_path_references(content))
    # A path-only code span inside a shell fence can be found by both passes.
    return list(dict.fromkeys(results))


def extract_generic_strategy_findings(
    content: str,
    doc_file: str,
) -> List[Finding]:
    """Reject generic task labels presented as concrete runtime strategies."""
    findings: List[Finding] = []
    lines = content.splitlines()
    if is_explicitly_noncurrent_document(content):
        return findings

    for line_no, line in enumerate(lines, start=1):
        matches = list(_GENERIC_STRATEGY_RE.finditer(line))
        if not matches or not _STRATEGY_WORD_RE.search(line):
            continue
        start = max(0, line_no - 2)
        end = min(len(lines), line_no + 1)
        context = " ".join(lines[start:end])
        if _GENERIC_STRATEGY_EXEMPT_RE.search(context):
            continue
        for match in matches:
            findings.append(
                Finding(
                    level="ERROR",
                    doc_file=doc_file,
                    line_no=line_no,
                    message=(
                        f"Generic task label is presented as a runtime strategy: "
                        f"{match.group(1)}; use a family-owned strategy key"
                    ),
                )
            )
    return findings


def extract_retired_surface_findings(content: str, doc_file: str) -> List[Finding]:
    """Flag retired truth surfaces unless their non-current status is explicit."""
    findings: List[Finding] = []
    lines = content.splitlines()
    if is_explicitly_noncurrent_document(content):
        return findings

    for line_no, line in enumerate(lines, start=1):
        start = max(0, line_no - 3)
        end = min(len(lines), line_no + 2)
        context = " ".join(lines[start:end])
        if _NONCURRENT_CONTEXT_RE.search(context):
            continue
        for pattern, label in _KNOWN_RETIRED_SURFACES:
            if pattern.search(line):
                findings.append(
                    Finding(
                        level="ERROR",
                        doc_file=doc_file,
                        line_no=line_no,
                        message=(
                            f"Retired truth surface is presented as current: {label}; "
                            "mark it historical/retired or replace it"
                        ),
                    )
                )
    return findings


def is_explicitly_noncurrent_document(content: str) -> bool:
    """Return whether the document declares itself historical or archived."""
    return bool(_NONCURRENT_DOCUMENT_RE.search("\n".join(content.splitlines()[:30])))


# ---------------------------------------------------------------------------
# Numerical claim extraction
# ---------------------------------------------------------------------------

# Patterns for numerical claims.  Each tuple:
#   (compiled regex, claim_kind, group index for the number)
_CLAIM_PATTERNS: List[Tuple[re.Pattern, str, int]] = [
    # "68 manifests", "68 JSON manifest files", "68 JSON manifests", "50 per-model JSON manifests"
    (
        re.compile(r"\b(\d+)\s+(?:per-model\s+)?(?:JSON\s+)?manifest(?:s|[\s-]+file)", re.I),
        "manifests",
        1,
    ),
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
            {
                *manifest_dir.glob("*.json"),
                *manifest_dir.glob("*/manifests/*.json"),
            }
        )
    else:
        counts["manifests"] = 0

    # Family plugins.  The authoritative unit is a package root owning either
    # MODEL.toml or plugin.py; count that root once when it owns both.  Flat
    # compatibility modules such as _time_series_trt.py are not family plugins.
    families_dir = repo_root / "python" / "tensorrt_model_connect" / "families"
    if families_dir.is_dir():
        package_roots = {
            candidate.parent
            for pattern in ("*/MODEL.toml", "*/plugin.py")
            for candidate in families_dir.glob(pattern)
            if candidate.parent.is_dir() and not candidate.parent.name.startswith("_")
        }
        counts["family_plugins"] = len(package_roots)
    else:
        counts["family_plugins"] = 0

    # Total family .py files (for "N Python files in the families directory" claims)
    if families_dir.is_dir():
        counts["families_total_py"] = len(
            [p for p in families_dir.rglob("*.py") if "__pycache__" not in p.parts]
        )
    else:
        counts["families_total_py"] = 0

    # C++ test files
    cpp_test_dir = repo_root / "tests" / "cpp"
    if cpp_test_dir.is_dir():
        counts["cpp_tests"] = len([f for f in os.listdir(cpp_test_dir) if f.endswith(".cpp")])
    else:
        counts["cpp_tests"] = 0

    # Builder test files
    builder_test_dir = repo_root / "tests" / "builder"
    if builder_test_dir.is_dir():
        counts["builder_tests"] = len(
            [f for f in os.listdir(builder_test_dir) if f.startswith("test_") and f.endswith(".py")]
        )
    else:
        counts["builder_tests"] = 0

    # Tools test files
    tools_test_dir = repo_root / "tests" / "tools"
    if tools_test_dir.is_dir():
        counts["tools_tests"] = len(
            [f for f in os.listdir(tools_test_dir) if f.startswith("test_") and f.endswith(".py")]
        )
    else:
        counts["tools_tests"] = 0

    return counts


def _surrounding_context(lines: List[str], line_no: int, window: int = 5) -> str:
    """Get surrounding lines as lowercase text for disambiguation."""
    start = max(0, line_no - window - 1)
    end = min(len(lines), line_no + window)
    return " ".join(lines[start:end]).lower()


def extract_numerical_claims(content: str, doc_file: str, actual_counts: dict) -> List[Finding]:
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
                    label = "E2E model manifests (tests/e2e/models/<family>/manifests/*.json)"
                elif claim_kind == "models_e2e":
                    # "N models" claims -- map to manifest count
                    actual_key = "manifests"
                    label = "E2E model manifests (tests/e2e/models/<family>/manifests/*.json)"
                elif claim_kind == "family_plugins":
                    actual_key = "family_plugins"
                    label = "family plugin packages (*/MODEL.toml or */plugin.py)"
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


def _markdown_files_under(scan_path: Path) -> List[Path]:
    """Return Markdown files beneath a directory, or one selected Markdown file."""
    if scan_path.is_file():
        return [scan_path] if scan_path.suffix.lower() in {".md", ".mdx"} else []

    md_files: List[Path] = []
    for root_dir, dirs, files in os.walk(scan_path):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in {".git", ".pytest_cache", "__pycache__", "node_modules"}
        )
        for fname in sorted(files):
            if Path(fname).suffix.lower() in {".md", ".mdx"}:
                md_files.append(Path(root_dir) / fname)
    return md_files


def tracked_markdown_files(repo_root: Path) -> List[Path]:
    """Return every Git-tracked Markdown file in the repository."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z", "--", "*.md", "*.mdx"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or "git ls-files failed")
    return sorted(repo_root / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw)


def check_markdown_files(md_files: Iterable[Path], repo_root: Path) -> CheckReport:
    """Verify references and current truth surfaces in selected Markdown files."""
    report = CheckReport()
    actual_counts = _get_actual_counts(repo_root)
    selected_files = sorted(set(md_files))

    report.docs_scanned = len(selected_files)

    for md_path in selected_files:
        content = md_path.read_text(encoding="utf-8", errors="replace")
        rel_doc = str(md_path.relative_to(repo_root))
        explicitly_noncurrent = is_explicitly_noncurrent_document(content)
        retired_findings = extract_retired_surface_findings(content, rel_doc)
        retired_lines = {finding.line_no for finding in retired_findings}

        # --- Check file path references ---
        refs = extract_path_references(content, rel_doc)
        for line_no, ref_path in refs:
            report.paths_checked += 1
            if explicitly_noncurrent:
                continue
            if line_no in retired_lines and any(
                pattern.search(ref_path) for pattern, _label in _KNOWN_RETIRED_SURFACES
            ):
                # The retired-surface finding is more actionable than a second
                # generic "path does not exist" error for the same token.
                continue
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

        # --- Check known retired truth surfaces in all prose/commands ---
        report.findings.extend(retired_findings)
        report.findings.extend(extract_generic_strategy_findings(content, rel_doc))

        # --- Check numerical claims ---
        claim_findings = (
            []
            if explicitly_noncurrent
            else extract_numerical_claims(content, rel_doc, actual_counts)
        )
        report.findings.extend(claim_findings)
        report.claims_checked += len(claim_findings)

    # Add actual counts to report for summary
    report._actual_counts = actual_counts  # type: ignore[attr-defined]
    return report


def check_docs(scan_path: Path, repo_root: Path) -> CheckReport:
    """Scan a Markdown file or directory and verify references."""
    return check_markdown_files(_markdown_files_under(scan_path), repo_root)


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
        default=None,
        help="Directory to scan for .md files (default: website/docs/wiki/)",
    )
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="Scan all Git-tracked Markdown instead of one directory",
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

    if args.tracked and args.scan_dir is not None:
        print("ERROR: --tracked cannot be combined with scan_dir", file=sys.stderr)
        return 1

    if args.tracked:
        try:
            md_files = tracked_markdown_files(repo_root)
        except RuntimeError as error:
            print(f"ERROR: Could not enumerate tracked Markdown: {error}", file=sys.stderr)
            return 1
        scan_label = "all Git-tracked Markdown"
        report = check_markdown_files(md_files, repo_root)
    else:
        selection = args.scan_dir or "website/docs/wiki/"
        scan_path = (repo_root / selection).resolve()
        if not scan_path.exists():
            print(f"ERROR: Documentation path does not exist: {scan_path}", file=sys.stderr)
            return 1
        scan_label = str(scan_path)
        report = check_docs(scan_path, repo_root)

    print(f"Repository root: {repo_root}")
    print(f"Scanning: {scan_label}")
    print()

    # Print actual counts
    counts = getattr(report, "_actual_counts", {})
    if counts:
        print("=== Actual file counts ===")
        print(
            f"  E2E manifests (tests/e2e/models/<family>/manifests/*.json): {counts.get('manifests', '?')}"
        )
        print(
            f"  Family plugin packages (MODEL.toml or plugin.py):  {counts.get('family_plugins', '?')}"
        )
        print(
            f"  Family dir total .py files:                        {counts.get('families_total_py', '?')}"
        )
        print(
            f"  C++ test files (tests/cpp/*.cpp):                  {counts.get('cpp_tests', '?')}"
        )
        print(
            f"  Builder test files (tests/builder/test_*.py):      {counts.get('builder_tests', '?')}"
        )
        print(
            f"  Tools test files (tests/tools/test_*.py):          {counts.get('tools_tests', '?')}"
        )
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
    print(f"  Errors (phantom/retired paths): {len(errors)}")
    print(f"  Warnings (stale counts): {len(warnings)}")

    if not errors and not warnings:
        print("\nAll checks passed.")
        return 0

    if errors:
        print(f"\nFAILED: {len(errors)} phantom or retired path finding(s).")
        return 1

    if args.strict and warnings:
        print(f"\nFAILED (strict mode): {len(warnings)} stale count(s) found.")
        return 1

    print("\nPassed with warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

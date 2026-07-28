# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the documentation-reference and inventory checks in
``tools/check_doc_file_references.py``.

Intent:
    ``check_doc_file_references.py`` is the CI gate that enforces
    ISO 26262-6 §7.4.1 compliance by catching phantom/stale file references
    in documentation. The pure extraction helpers cover paths, numerical
    claims, and the public family-plugin inventory. A silent regression in
    those helpers would let phantom paths or stale launch-facing support
    claims slip past the gate.

Preconditions:
    ``tools.check_doc_file_references`` is importable (pure-stdlib module
    with no third-party dependencies).

Postconditions:
    Path shorthand and filtering behavior remains stable. Qualified public
    support counts and family inventories are compared with the repository,
    and the two launch-facing support pages remain synchronized.

Trace IDs:
    - ARCH-CI-QUALITY-GATES (ISO 26262-6 §7.4.1 doc-reference gate)
    - UD-TOOLS-DOC-REF-HELPERS (pure helpers: _expand_path,
      extract_path_references)
    - UT-TOOLS-DOC-REF-EXPAND-PATH, UT-TOOLS-DOC-REF-EXTRACT-REFERENCES
"""

from __future__ import annotations

from pathlib import Path

from tools import check_doc_file_references as cdfr


# ---------------------------------------------------------------------------
# _expand_path
# ---------------------------------------------------------------------------


def test_expand_path_plain_cpp_returned_unchanged() -> None:
    # A plain repo-relative .cpp path has no shorthand to expand; _expand_path
    # must pass it through as a single-element list so extract_path_references
    # can feed each concrete path to the filesystem existence check.
    assert cdfr._expand_path("src/foo/bar.cpp") == ["src/foo/bar.cpp"]


def test_expand_path_h_cpp_shorthand_splits_into_header_and_source() -> None:
    # "foo.h/cpp" is the canonical shorthand in docs meaning "both foo.h and
    # foo.cpp exist". _expand_path must produce exactly [foo.h, foo.cpp] in
    # that order so downstream checks exercise both translation units.
    assert cdfr._expand_path("src/foo/bar.h/cpp") == [
        "src/foo/bar.h",
        "src/foo/bar.cpp",
    ]


def test_expand_path_h_hpp_shorthand_splits_into_header_pair() -> None:
    # "foo.h/hpp" is the header-only variant of the shorthand and must expand
    # to [foo.h, foo.hpp] — the .hpp suffix, not .cpp. Guards against a
    # regex regression where H_HPP_SHORTHAND accidentally emits a .cpp.
    assert cdfr._expand_path("include/trtmc/bar.h/hpp") == [
        "include/trtmc/bar.h",
        "include/trtmc/bar.hpp",
    ]


# ---------------------------------------------------------------------------
# extract_path_references
# ---------------------------------------------------------------------------


def test_extract_path_references_captures_backtick_path_with_1_based_line_no() -> None:
    # A backtick-quoted path on a known prefix must be captured with its
    # 1-based line number. Line 2 (not 1) is used here to lock down that the
    # enumerate() start=1 convention is preserved.
    content = (
        "first line of the doc\n"
        "see `src/runtime/models/qwen/plugin.cpp` for the impl\n"
        "third line\n"
    )

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == [(2, "src/runtime/models/qwen/plugin.cpp")]


def test_extract_path_references_skips_wildcard_glob() -> None:
    # Wildcard paths like `src/foo/*.cpp` are glob patterns, not literal
    # paths, and cannot be checked for filesystem existence. They must be
    # filtered out so they do not trigger phantom-path errors.
    content = "intro\n`src/foo/*.cpp` is a glob pattern\nouter\n"

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == []


def test_extract_path_references_skips_angle_bracket_placeholder() -> None:
    # Angle-bracket placeholders like `tools/<family>.py` are templates in
    # scaffolding docs, not real files. They must be skipped so doc authors
    # can write `tools/<family>.py` without tripping the CI gate.
    content = "run `tools/<family>.py` to scaffold a new plugin\n"

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == []


def test_extract_path_references_strips_trailing_punctuation() -> None:
    # A trailing '.' (sentence punctuation) that leaks past the regex must be
    # stripped from the captured path. Otherwise the existence check would
    # look for "src/foo.cpp." (a phantom) instead of the real "src/foo.cpp".
    content = "The module `src/foo.cpp`. closes the paragraph.\n"

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == [(1, "src/foo.cpp")]
    # And explicitly: no entry ends with a trailing dot.
    assert not any(path.endswith(".") for _line, path in refs)


def test_extract_path_references_h_cpp_shorthand_yields_two_entries_same_line() -> None:
    # The h/cpp shorthand must flow end-to-end through extract_path_references:
    # a single backtick-quoted `src/foo.h/cpp` produces two (line, path)
    # entries — src/foo.h and src/foo.cpp — sharing the SAME line number,
    # because they originated from the same textual occurrence.
    content = "header and source live together: `src/foo.h/cpp` on one line\n"

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == [(1, "src/foo.h"), (1, "src/foo.cpp")]
    # The two entries share exactly one line number; guard the invariant.
    assert {line for line, _path in refs} == {1}


# ---------------------------------------------------------------------------
# Public model-support claims
# ---------------------------------------------------------------------------


def test_extract_numerical_claims_checks_qualified_public_counts() -> None:
    content = (
        "The checkout contains 71 Python family plugins, "
        "197 E2E model manifests, and 74 E2E family indexes.\n"
        "It has 36 C++ runtime strategy keys.\n"
    )
    actual_counts = {
        "family_plugins": 78,
        "families_total_py": 500,
        "manifests": 203,
        "family_indexes": 78,
        "runtime_strategy_keys": 79,
    }

    findings = cdfr.extract_numerical_claims(
        content,
        "website/docs/getting-started/model-support.md",
        actual_counts,
    )

    assert len(findings) == 4
    messages = [finding.message for finding in findings]
    assert any("71 Python family plugin" in message for message in messages)
    assert any("197 E2E model manifest" in message for message in messages)
    assert any("74 E2E family indexes" in message for message in messages)
    assert any("36 C++ runtime strategy keys" in message for message in messages)


def test_extract_numerical_claims_accepts_current_public_counts() -> None:
    content = (
        "The checkout contains 78 Python family plugins, "
        "203 E2E model manifests, and 78 E2E family indexes.\n"
        "It has 79 C++ runtime strategy keys.\n"
    )
    actual_counts = {
        "family_plugins": 78,
        "families_total_py": 500,
        "manifests": 203,
        "family_indexes": 78,
        "runtime_strategy_keys": 79,
    }

    findings = cdfr.extract_numerical_claims(
        content,
        "website/docs/getting-started/model-support.md",
        actual_counts,
    )

    assert findings == []


def test_extract_family_inventory_claims_reports_drift() -> None:
    content = (
        "## Family plugin inventory\n\n"
        "The current Python plugin inventory is:\n\n"
        "```text\n"
        "alpha, removed\n"
        "```\n"
    )

    findings = cdfr.extract_family_inventory_claims(
        content,
        "website/docs/getting-started/model-support.md",
        {"alpha", "beta"},
    )

    assert len(findings) == 1
    assert "missing: beta" in findings[0].message
    assert "not registered: removed" in findings[0].message


def test_public_model_support_pages_match_repository_inventory() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    public_support_docs = [
        repo_root / "website" / "docs" / "intro.md",
        repo_root / "website" / "docs" / "getting-started" / "model-support.md",
    ]

    report = cdfr.check_doc_paths(public_support_docs, repo_root)

    assert report.findings == []
    assert report.docs_scanned == 2

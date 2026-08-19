# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure path-extraction helpers in
``tools/check_doc_file_references.py``.

Intent:
    ``check_doc_file_references.py`` is the CI gate that enforces
    ISO 26262-6 §7.4.1 compliance by catching phantom/stale file references
    in documentation. Its two pure helpers — ``_expand_path`` (h/cpp and
    h/hpp shorthand expansion) and ``extract_path_references`` (regex
    extraction, wildcard/placeholder filtering, and trailing-punctuation
    cleanup) — are the extraction front door. A silent regression in
    either would let phantom paths slip past the gate unnoticed. These
    tests lock their documented behavior using raw in-memory strings so
    that no filesystem or subprocess is required.

Preconditions:
    ``tools.check_doc_file_references`` is importable (pure-stdlib module
    with no third-party dependencies).

Postconditions:
    ``_expand_path`` expands the h/cpp and h/hpp shorthand forms and
    otherwise returns the input unchanged; ``extract_path_references``
    captures backtick-quoted paths that start with a known repo prefix
    with 1-based line numbers, skips wildcard and placeholder paths, and
    strips trailing punctuation. The shorthand expansion is preserved
    end-to-end with both entries sharing one line number.

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
        "see `python/tensorrt_model_connect/models/qwen/runtime/plugin.cpp` for the impl\n"
        "third line\n"
    )

    refs = cdfr.extract_path_references(content, "website/docs/wiki/any.md")

    assert refs == [
        (2, "python/tensorrt_model_connect/models/qwen/runtime/plugin.cpp")
    ]


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
    content = "run `tools/<family>.py` to scaffold a new family model\n"

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


def test_actual_counts_use_canonical_family_model_entries(tmp_path: Path) -> None:
    families = tmp_path / "python/tensorrt_model_connect/models"
    for name in ("alpha", "beta"):
        path = families / name / "model.py"
        path.parent.mkdir(parents=True)
        path.write_text("def build(model_dir, output_path, **options):\n    pass\n")
    legacy = families / "legacy/plugin.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("plugin = object()\n")

    counts = cdfr._get_actual_counts(tmp_path)

    assert counts["family_models"] == 2

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for documentation path, drift, and inventory checks in
``tools/check_doc_file_references.py``.

Intent:
    ``check_doc_file_references.py`` is the CI gate that enforces
    ISO 26262-6 §7.4.1 compliance by catching phantom/stale file references
    in documentation. The tests lock shorthand/path extraction, shell-command
    path discovery, retired-surface classification, tracked-file discovery,
    and authoritative family-package counting.

Preconditions:
    ``tools.check_doc_file_references`` is importable (pure-stdlib module
    with no third-party dependencies).

Postconditions:
    ``_expand_path`` expands the h/cpp and h/hpp shorthand forms and
    otherwise returns the input unchanged; ``extract_path_references``
    captures backtick-quoted paths that start with a known repo prefix
    plus paths in shell fences with 1-based line numbers, skips wildcard and
    placeholder paths, and strips trailing punctuation. Retired truth surfaces
    are rejected unless explicitly historical, and family counts use package
    ``MODEL.toml``/``plugin.py`` roots rather than flat helper modules.

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
        "first line of the doc\nsee `src/runtime/models/qwen/plugin.cpp` for the impl\nthird line\n"
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


def test_extract_path_references_strips_pytest_node_selector() -> None:
    content = "```bash\npytest tests/e2e/test_model.py::test_model[small] -q\n```\n"

    assert cdfr.extract_path_references(content, "README.md") == [(2, "tests/e2e/test_model.py")]


def test_extract_path_references_captures_paths_inside_shell_fence() -> None:
    content = (
        "```bash\n"
        "python3 tools/nsight_collect.py --help\n"
        'rg -n "PluginRegistrar" src/runtime/plugins --glob "*.cpp"\n'
        "```\n"
    )

    refs = cdfr.extract_path_references(content, "README.md")

    assert refs == [
        (2, "tools/nsight_collect.py"),
        (3, "src/runtime/plugins"),
    ]


def test_extract_path_references_covers_authoritative_non_source_roots() -> None:
    content = (
        "See `.github/workflows/nightly.yml`, `cmake/plugins.cmake`, "
        "`examples/plan.yaml`, and `CMakeLists.txt`.\n"
    )

    assert cdfr.extract_path_references(content, "README.md") == [
        (1, ".github/workflows/nightly.yml"),
        (1, "cmake/plugins.cmake"),
        (1, "examples/plan.yaml"),
        (1, "CMakeLists.txt"),
    ]


def test_extract_path_references_does_not_scan_non_shell_fence() -> None:
    content = "```text\npython3 tools/example.py --help\n```\n"

    assert cdfr.extract_path_references(content, "README.md") == []


def test_extract_path_references_skips_explicit_shell_output_destination() -> None:
    content = (
        "```bash\n"
        "python3 tools/generate.py --output reports/generated\n"
        "trtmc-bench report results/gb300 results/h100 -o reports/combined\n"
        "```\n"
    )

    assert cdfr.extract_path_references(content, "README.md") == [(2, "tools/generate.py")]


def test_shell_output_destinations_cover_continuations_and_redirections() -> None:
    content = (
        "```bash\n"
        "python3 tools/generate.py \\\n"
        "  --output \\\n"
        "  reports/generated.json\n"
        "python3 tools/generate.py > reports/stdout.txt\n"
        "python3 tools/generate.py 1> reports/fd-one.txt\n"
        "python3 tools/generate.py &>reports/combined.txt\n"
        "```\n"
    )

    assert cdfr.extract_path_references(content, "README.md") == [
        (2, "tools/generate.py"),
        (5, "tools/generate.py"),
        (6, "tools/generate.py"),
        (7, "tools/generate.py"),
    ]


def test_shell_continuation_keeps_non_output_inputs_visible() -> None:
    content = (
        "```bash\n"
        "python3 tools/generate.py \\\n"
        "  tests/fixtures/input.json \\\n"
        "  > reports/generated.json\n"
        "```\n"
    )

    assert cdfr.extract_path_references(content, "README.md") == [
        (2, "tools/generate.py"),
        (3, "tests/fixtures/input.json"),
    ]


# ---------------------------------------------------------------------------
# retired truth surfaces
# ---------------------------------------------------------------------------


def test_retired_surface_is_found_in_plain_prose() -> None:
    content = (
        "Runtime registration is implemented in src/runtime/plugins and src/runtime/pipelines.\n"
    )

    findings = cdfr.extract_retired_surface_findings(content, "README.md")

    assert [finding.message for finding in findings] == [
        "Retired truth surface is presented as current: src/runtime/plugins; "
        "mark it historical/retired or replace it",
        "Retired truth surface is presented as current: src/runtime/pipelines; "
        "mark it historical/retired or replace it",
    ]


def test_retired_missing_path_produces_one_actionable_finding(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(
        "Runtime registration lives in `src/runtime/plugins`.\n",
        encoding="utf-8",
    )

    report = cdfr.check_markdown_files([doc], tmp_path)

    assert len(report.findings) == 1
    assert "Retired truth surface" in report.findings[0].message


def test_retired_surface_is_allowed_when_same_context_marks_it_retired() -> None:
    content = (
        "The retired path `python/tensorrt_model_connect/graph_ops.py` no "
        "longer owns graph construction.\n"
    )

    assert cdfr.extract_retired_surface_findings(content, "README.md") == []


def test_historical_document_allows_personal_evidence_paths() -> None:
    content = (
        "# Historical evidence snapshot — not a replay runbook\n\n"
        "Workspace: `/workspace/users/yifeif/workspaces/agent-4/repo`\n"
    )

    assert cdfr.extract_retired_surface_findings(content, "reports/evidence.md") == []


def test_historical_document_skips_live_path_and_count_checks(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "evidence.md"
    report_path.parent.mkdir()
    report_path.write_text(
        "# Historical evidence snapshot — not a replay runbook\n\n"
        "The old run used `tools/deleted.py` and covered 197 models.\n",
        encoding="utf-8",
    )

    report = cdfr.check_markdown_files([report_path], tmp_path)

    assert report.findings == []


def test_generated_output_path_is_allowed_but_missing_source_path_is_not(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(
        "Docusaurus writes `website/build/`; source lives in `website/missing/`.\n",
        encoding="utf-8",
    )

    report = cdfr.check_markdown_files([doc], tmp_path)

    assert [finding.message for finding in report.findings] == [
        "Path does not exist: website/missing/"
    ]


def test_missing_generated_output_descendant_is_not_blanket_exempted(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "README.md"
    doc.write_text(
        "The generated page is `website/build/missing-page/index.html`.\n",
        encoding="utf-8",
    )

    report = cdfr.check_markdown_files([doc], tmp_path)

    assert [finding.message for finding in report.findings] == [
        "Path does not exist: website/build/missing-page/index.html"
    ]


def test_personal_workspace_path_is_rejected_in_current_document() -> None:
    content = "Build from /workspace/users/yifeif/workspaces/current/repo.\n"

    findings = cdfr.extract_retired_surface_findings(content, "README.md")

    assert len(findings) == 1
    assert "personal /workspace/users/yifeif path" in findings[0].message


def test_generic_task_label_is_rejected_as_runtime_strategy() -> None:
    content = "Inspect the bundle for its `vision_language` runtime strategy.\n"

    findings = cdfr.extract_generic_strategy_findings(content, "README.md")

    assert [finding.message for finding in findings] == [
        "Generic task label is presented as a runtime strategy: "
        "vision_language; use a family-owned strategy key"
    ]


def test_generic_strategy_label_is_allowed_when_explicitly_noncurrent() -> None:
    content = (
        "Do not use the retired generic `decoder_kv_cache` runtime strategy; "
        "use the owning family's key.\n"
    )

    assert cdfr.extract_generic_strategy_findings(content, "README.md") == []


# ---------------------------------------------------------------------------
# inventory and tracked-file selection
# ---------------------------------------------------------------------------


def test_family_plugin_count_uses_unique_package_roots(tmp_path: Path) -> None:
    families = tmp_path / "python" / "tensorrt_model_connect" / "families"
    families.mkdir(parents=True)
    (families / "_time_series_trt.py").write_text("# helper\n", encoding="utf-8")
    (families / "__init__.py").write_text("", encoding="utf-8")

    alpha = families / "alpha"
    alpha.mkdir()
    (alpha / "MODEL.toml").write_text("[model]\n", encoding="utf-8")
    (alpha / "plugin.py").write_text("", encoding="utf-8")

    beta = families / "beta"
    beta.mkdir()
    (beta / "MODEL.toml").write_text("[model]\n", encoding="utf-8")

    gamma = families / "gamma"
    gamma.mkdir()
    (gamma / "plugin.py").write_text("", encoding="utf-8")

    counts = cdfr._get_actual_counts(tmp_path)

    assert counts["family_plugins"] == 3


def test_plugin_claim_never_uses_total_python_file_count() -> None:
    findings = cdfr.extract_numerical_claims(
        "The checkout contains 5 plugins.\n",
        "README.md",
        {"family_plugins": 2, "families_total_py": 5},
    )

    assert len(findings) == 1
    assert "actual count of family plugin packages" in findings[0].message


def test_tracked_markdown_files_uses_git_results(tmp_path: Path, monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = b"README.md\0docs/design.mdx\0"
        stderr = b""

    def fake_run(*_args, **_kwargs):
        return Result()

    monkeypatch.setattr(cdfr.subprocess, "run", fake_run)

    assert cdfr.tracked_markdown_files(tmp_path) == [
        tmp_path / "README.md",
        tmp_path / "docs" / "design.mdx",
    ]

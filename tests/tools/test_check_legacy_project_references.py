# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools import check_legacy_project_references as checker


def test_scan_paths_rejects_retired_project_slug(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    document.write_text("use " + "trt" + "-transformers for this task\n", encoding="utf-8")

    findings = checker.scan_paths(tmp_path, [document])

    assert [(finding.path, finding.line, finding.kind) for finding in findings] == [
        ("guide.md", 1, "retired project slug")
    ]


def test_scan_paths_rejects_singular_full_project_name(tmp_path: Path) -> None:
    document = tmp_path / "guide.md"
    document.write_text(
        "use " + "TensorRT" + " Transformer for this task\n", encoding="utf-8"
    )

    findings = checker.scan_paths(tmp_path, [document])

    assert len(findings) == 1
    assert findings[0].kind == "retired full project display name"


def test_scan_paths_rejects_retired_container_prefix_case_insensitively(
    tmp_path: Path,
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("TRTF" + "-DEV-GB300-AGENT-2\n", encoding="utf-8")

    findings = checker.scan_paths(tmp_path, [document])

    assert len(findings) == 1
    assert findings[0].kind == "retired container prefix"


def test_scan_paths_checks_tracked_path_names(tmp_path: Path) -> None:
    directory = tmp_path / ("trt" + "-transformers-notes")
    directory.mkdir()
    document = directory / "README.md"
    document.write_text("current text\n", encoding="utf-8")

    findings = checker.scan_paths(tmp_path, [document])

    assert len(findings) == 1
    assert findings[0].line == 0
    assert findings[0].kind == "retired project slug"


def test_scan_paths_allows_bundle_extension(tmp_path: Path) -> None:
    document = tmp_path / "bundle.md"
    document.write_text("the output is model.trtfb\n", encoding="utf-8")

    assert checker.scan_paths(tmp_path, [document]) == []


def test_scan_paths_ignores_binary_files(tmp_path: Path) -> None:
    binary = tmp_path / "artifact.bin"
    binary.write_bytes(b"\x00" + ("trt" + "-transformers").encode())

    assert checker.scan_paths(tmp_path, [binary]) == []

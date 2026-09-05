# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for explicit E2E waive platform selection."""

from __future__ import annotations

from pathlib import Path

from tests import test_e2e
from tests.e2e_harness.manifest_loader import get_case_names


def test_load_waives_filters_with_explicit_platform(monkeypatch, tmp_path) -> None:
    waives_file = tmp_path / "waives.txt"
    waives_file.write_text(
        "\n".join(
            [
                "GB300/model-gpu SKIP platform-specific skip",
                "OTHER/model-other SKIP other platform skip",
                "model-shared XFAIL shared waive",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(test_e2e, "_WAIVES_FILE", waives_file)

    waives = test_e2e._load_waives("GB300")

    assert waives["model-gpu"] == ("SKIP", "platform-specific skip")
    assert waives["model-shared"] == ("XFAIL", "shared waive")
    assert "model-other" not in waives


def test_load_waives_without_platform_ignores_prefixed_entries(monkeypatch, tmp_path) -> None:
    waives_file = tmp_path / "waives.txt"
    waives_file.write_text(
        "\n".join(
            [
                "GB300/model-gpu SKIP platform-specific skip",
                "model-shared XFAIL shared waive",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(test_e2e, "_WAIVES_FILE", waives_file)

    waives = test_e2e._load_waives()

    assert waives == {"model-shared": ("XFAIL", "shared waive")}


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WAIVES_FILE = _REPO_ROOT / "tests" / "e2e" / "waives.txt"


def _declared_case_names() -> set[str]:
    return set(get_case_names())


def test_every_waive_names_a_declared_case() -> None:
    """A waive for a name no E2E testcase declares is silently inert.

    The runtime resolves a waive against ``case.name``, so the namespace to
    check against is the child testcase names, not the top-level manifest
    names -- 50 of the former are not the latter. ``_load_waives`` skips any
    line it cannot parse and reports nothing when a name stops matching, so
    such an entry quietly stops doing anything. If the name is ever reused,
    the waive springs back and skips a case no one intended to waive.
    """
    declared = _declared_case_names()
    assert declared, "no E2E testcases were discovered"

    # Platform-prefixed waives are only visible when that platform is
    # selected, so collect across every platform the file mentions.
    platforms = {""}
    for line in _WAIVES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part = line.split(None, 1)[0]
        if "/" in name_part:
            platforms.add(name_part.split("/", 1)[0])

    waived: set[str] = set()
    for platform in platforms:
        waived.update(test_e2e._load_waives(platform))

    orphans = sorted(waived - declared)
    assert not orphans, (
        "waives.txt names cases that no E2E testcase declares, so these waives "
        f"are silently inert: {orphans}"
    )

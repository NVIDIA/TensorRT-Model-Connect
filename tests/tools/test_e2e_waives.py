"""Unit tests for explicit E2E waive platform selection."""

from __future__ import annotations

import json
from pathlib import Path

from tests import test_e2e


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_repo_waives_reference_existing_e2e_cases() -> None:
    manifest_names = set()
    for manifest_path in (REPO_ROOT / "tests/e2e/models").glob("*.json"):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_names.add(raw["name"])

    waives = test_e2e._load_waives()

    assert set(waives).issubset(manifest_names)

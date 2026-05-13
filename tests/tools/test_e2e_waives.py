"""Unit tests for explicit E2E waive platform selection."""

from __future__ import annotations

from pathlib import Path

from tests import test_e2e
from tests.e2e_harness.manifest_loader import load_manifest


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


def test_gemma_gated_model_uses_manifest_preflight_not_global_waive() -> None:
    case = load_manifest(REPO_ROOT / "tests/e2e/models/gemma-2-2b.json")
    waives = test_e2e._load_waives()

    assert "gemma-2-2b" not in waives
    assert any(
        req.kind == "hf_auth_token_present" and req.gating
        for req in case.preflight
    )

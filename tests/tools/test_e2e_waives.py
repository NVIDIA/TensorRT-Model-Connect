"""Unit tests for explicit E2E waive platform selection."""

from __future__ import annotations

from tests import test_e2e


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


def test_non_gating_success_detects_manifest_skip_comparison() -> None:
    case = test_e2e.get_case_by_name("qwen3-moe-tiny-random")

    assert case is not None
    assert "comparison skipped by manifest" in test_e2e._non_gating_success_reason(case)


def test_non_gating_success_detects_missing_reference() -> None:
    case = test_e2e.get_case_by_name("nemotron-h-nano-9b")

    assert case is not None
    assert "no reference backend" in test_e2e._non_gating_success_reason(case)


def test_non_gating_success_keeps_green_invariant_case() -> None:
    case = test_e2e.get_case_by_name("qwen3-0.6b-topp")

    assert case is not None
    assert test_e2e._non_gating_success_reason(case) == ""

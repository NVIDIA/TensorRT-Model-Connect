"""Regression coverage for copied standard decoder builders."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILIES_DIR = (
    ROOT / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
)


def _family_files(name: str) -> list[Path]:
    return sorted(FAMILIES_DIR.glob(f"*/{name}"))


def test_standard_decoder_builder_copies_honor_split_roles():
    """Each copied standard builder must route split prefill by role."""
    missing: list[str] = []
    for path in _family_files("standard_decoder_builder.py"):
        text = path.read_text(encoding="utf-8")
        if (
            "_decoder_engine_role" not in text
            or "_decoder_engine_layout_supported" not in text
            or "profile_mode=(" not in text
        ):
            missing.append(str(path.relative_to(ROOT)))

    assert not missing


def test_dual_profile_builder_copies_support_prefill_only_mode():
    """Each copied dual-profile builder must be able to emit prefill only."""
    missing: list[str] = []
    for path in _family_files("dual_profile_decoder_builder.py"):
        text = path.read_text(encoding="utf-8")
        if (
            'profile_mode: str = "dual_profile"' not in text
            or 'profile_mode == "prefill"' not in text
            or "TRTMC_REVERSE_PROFILE_ORDER" not in text
        ):
            missing.append(str(path.relative_to(ROOT)))

    assert not missing

"""Regression coverage for shared standard decoder builders."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILIES_DIR = (
    ROOT / "python" / "tensorrt_model_connect" / "families"
)
BUILDERS_DIR = (
    ROOT / "python" / "tensorrt_model_connect" / "builders"
)
SHARED_BUILDER_DIRS = (BUILDERS_DIR,)
MODEL_AGNOSTIC_SHARED_FILES = (
    BUILDERS_DIR / "__init__.py",
    BUILDERS_DIR / "default_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder_tp.py",
    BUILDERS_DIR / "utils.py",
)
FAMILY_NAMES = {
    path.name
    for path in FAMILIES_DIR.iterdir()
    if path.is_dir() and not path.name.startswith("_")
}


def _family_files(name: str) -> list[Path]:
    return sorted(FAMILIES_DIR.glob(f"*/{name}"))


def _builder_contract_text(path: Path, *, local_module: str) -> str:
    """Return the source that owns the contract for a builder or shim."""
    text = path.read_text(encoding="utf-8")
    if f"from .{local_module} import" in text:
        target = path.with_name(f"{local_module}.py")
        if target.is_file():
            return target.read_text(encoding="utf-8")

    shared_import = f"builders.{local_module}"
    if shared_import in text:
        target = BUILDERS_DIR / f"{local_module}.py"
        if target.is_file():
            return target.read_text(encoding="utf-8")

    return text


def _python_files(*dirs: Path) -> list[Path]:
    files: list[Path] = []
    for directory in dirs:
        files.extend(sorted(directory.rglob("*.py")))
    return files


def test_shared_standard_decoder_builder_honors_split_roles():
    """The default standard builder must route split prefill by role."""
    text = (BUILDERS_DIR / "default_decoder.py").read_text(encoding="utf-8")
    assert "_decoder_engine_role" in text
    assert "_decoder_engine_layout_supported" in text
    assert "profile_mode=(" in text


def test_family_standard_decoder_builders_are_shims_or_honor_split_roles():
    """Family builders can be tiny shims or owned implementations."""
    missing: list[str] = []
    for path in _family_files("standard_decoder_builder.py"):
        text = _builder_contract_text(path, local_module="default_decoder")
        if (
            "_decoder_engine_role" not in text
            or "_decoder_engine_layout_supported" not in text
            or "profile_mode=(" not in text
        ):
            missing.append(str(path.relative_to(ROOT)))

    assert not missing


def test_shared_dual_profile_builder_supports_prefill_only_mode():
    """The default dual-profile builder must be able to emit prefill only."""
    text = (BUILDERS_DIR / "default_dual_profile_decoder.py").read_text(
        encoding="utf-8")
    assert 'profile_mode: str = "dual_profile"' in text
    assert 'profile_mode == "prefill"' in text
    assert "TRTMC_REVERSE_PROFILE_ORDER" in text


def test_family_dual_profile_builders_are_shims_or_support_prefill_only_mode():
    """Family dual-profile builders can be tiny shims or owned implementations."""
    missing: list[str] = []
    for path in _family_files("dual_profile_decoder_builder.py"):
        text = _builder_contract_text(
            path, local_module="default_dual_profile_decoder")
        if (
            'profile_mode: str = "dual_profile"' not in text
            or 'profile_mode == "prefill"' not in text
            or "TRTMC_REVERSE_PROFILE_ORDER" not in text
        ):
            missing.append(str(path.relative_to(ROOT)))

    assert not missing


def test_shared_builder_modules_do_not_import_families():
    offenders: list[str] = []
    for path in _python_files(*SHARED_BUILDER_DIRS):
        text = path.read_text(encoding="utf-8")
        if ".families" in text or "import families" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders


def test_model_agnostic_shared_builder_modules_do_not_branch_on_family_names():
    offenders: list[str] = []
    for path in MODEL_AGNOSTIC_SHARED_FILES:
        text = path.read_text(encoding="utf-8")
        for family in FAMILY_NAMES:
            quoted = re.escape(family)
            family_branch = (
                rf"\b(model_type|family|family_name)\s*==\s*[\"']{quoted}[\"']"
                rf"|[\"']{quoted}[\"']\s*==\s*\b(model_type|family|family_name)"
                rf"|\b(model_type|family|family_name)\s+in\s+\{{[^}}]*[\"']{quoted}[\"']"
            )
            if re.search(family_branch, text):
                offenders.append(f"{path.relative_to(ROOT)}: {family}")

    assert not offenders

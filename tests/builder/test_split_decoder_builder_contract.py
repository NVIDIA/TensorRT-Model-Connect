"""Regression coverage for family-owned standard decoder builders."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILIES_DIR = ROOT / "python" / "tensorrt_model_connect" / "families"
BUILDERS_DIR = ROOT / "python" / "tensorrt_model_connect" / "builders"

SHARED_BUILDER_STUBS = (
    BUILDERS_DIR / "default_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder_tp.py",
    BUILDERS_DIR / "utils.py",
)


def _family_files(name: str) -> list[Path]:
    return sorted(FAMILIES_DIR.glob(f"*/{name}"))


def _builder_contract_text(path: Path, *, local_module: str) -> str:
    """Return the family-local source that owns a builder contract."""
    text = path.read_text(encoding="utf-8")
    if f"from .{local_module} import" in text:
        target = path.with_name(f"{local_module}.py")
        if target.is_file():
            return target.read_text(encoding="utf-8")
    return text


def test_shared_builder_modules_are_retired_stubs() -> None:
    """Shared builder package must not retain concrete model builder logic."""
    violations: list[str] = []
    forbidden_terms = (
        "import tensorrt",
        "trt_compat",
        "graph_ops",
        "graph_blocks",
        "create_network",
        "add_input",
        "add_output",
        "profile_mode ==",
        "_decoder_engine_role",
        "position_type ==",
        "mlp_type ==",
    )
    forbidden_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)
    for path in SHARED_BUILDER_STUBS:
        text = path.read_text(encoding="utf-8")
        if "RetiredSharedBuilderError" not in text:
            violations.append(f"{path.relative_to(ROOT)}: missing retired-builder error")
        for term in forbidden_terms:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)}: contains {term}")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, forbidden_nodes):
                violations.append(
                    f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)} "
                    f"contains {type(node).__name__}"
                )

    assert not violations


def test_family_standard_decoder_builders_honor_split_roles() -> None:
    """Family builders can be tiny local shims or owned implementations."""
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


def test_family_dual_profile_builders_support_prefill_only_mode() -> None:
    """Family dual-profile builders own the split-prefill contract."""
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


def test_family_code_does_not_import_shared_builder_package() -> None:
    """Concrete family builders must use family-local helper copies."""
    violations: list[str] = []
    for path in sorted(FAMILIES_DIR.glob("*/*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "tensorrt_model_connect.builders" in text:
            violations.append(str(path.relative_to(ROOT)))

    assert not violations

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for family-owned standard decoder builders."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILIES_DIR = ROOT / "python" / "tensorrt_model_connect" / "models"
BUILDERS_DIR = ROOT / "python" / "tensorrt_model_connect" / "builders"

REMOVED_SHARED_BUILDER_MODULES = (
    BUILDERS_DIR / "__init__.py",
    BUILDERS_DIR / "default_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder.py",
    BUILDERS_DIR / "default_dual_profile_decoder_tp.py",
    BUILDERS_DIR / "utils.py",
)

BLOOM_FP32_BUILDERS = (
    FAMILIES_DIR / "bloom" / "standard_decoder_builder.py",
    FAMILIES_DIR / "bloom" / "dual_profile_decoder_builder.py",
    FAMILIES_DIR / "bloom" / "dual_profile_decoder_tp_builder.py",
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


def _selects_prefill_or_dual_profile(text: str) -> bool:
    """Return whether a call chooses between the two supported profile modes."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "profile_mode":
            continue
        values = {
            child.value
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        if {"prefill", "dual_profile"} <= values:
            return True
    return False


def _supports_prefill_profile_mode(text: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "build_dual_profile_decoder_engine":
            continue
        defaults = dict(zip(node.args.kwonlyargs, node.args.kw_defaults))
        default = next(
            (
                value
                for argument, value in defaults.items()
                if argument.arg == "profile_mode"
            ),
            None,
        )
        if not isinstance(default, ast.Constant) or default.value != "dual_profile":
            return False
        return any(
            isinstance(child, ast.Compare)
            and isinstance(child.left, ast.Name)
            and child.left.id == "profile_mode"
            and any(
                isinstance(comparator, ast.Constant)
                and comparator.value == "prefill"
                for comparator in child.comparators
            )
            for child in ast.walk(node)
        )
    return False


def test_shared_builder_modules_are_removed() -> None:
    """Shared builder package must not retain concrete model builder logic."""
    violations = [
        str(path.relative_to(ROOT))
        for path in REMOVED_SHARED_BUILDER_MODULES
        if path.exists()
    ]

    assert not violations


def test_family_standard_decoder_builders_honor_split_roles() -> None:
    """Every remaining standard decoder file owns its implementation."""
    missing: list[str] = []
    for path in _family_files("standard_decoder_builder.py"):
        text = _builder_contract_text(path, local_module="default_decoder")
        if (
            "_decoder_engine_role" not in text
            or "_decoder_engine_layout_supported" not in text
            or not _selects_prefill_or_dual_profile(text)
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
            not _supports_prefill_profile_mode(text)
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


def test_bloom_fp32_builders_disable_tf32() -> None:
    """BLOOM FP32 must not silently use lower-precision TF32 matmuls."""
    missing = [
        str(path.relative_to(ROOT))
        for path in BLOOM_FP32_BUILDERS
        if "trt_config.clear_flag(trt.BuilderFlag.TF32)"
        not in path.read_text(encoding="utf-8")
    ]

    assert not missing

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static E2E-exemption gates that must run without TensorRT imports."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ROOT / "python/tensorrt_model_connect/families"
RUNTIMES = ROOT / "src/runtime/models"
E2E_MODELS = ROOT / "tests/e2e/models"


def _family_metadata() -> dict[str, frozenset[str]]:
    metadata: dict[str, frozenset[str]] = {}
    for path in sorted(FAMILIES.glob("*/MODEL.toml")):
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        family = raw.get("id") or raw.get("plugin") or path.parent.name
        assert isinstance(family, str) and family
        assert family not in metadata, f"duplicate family metadata id: {family}"
        assert (path.parent / "plugin.py").is_file(), f"{family} has no plugin.py"
        capabilities = raw.get("capabilities", [])
        assert isinstance(capabilities, list)
        assert all(isinstance(value, str) and value for value in capabilities)
        metadata[family] = frozenset(capabilities)
    assert metadata, "no family metadata discovered"
    return metadata


def _manifest_families() -> set[str]:
    families: set[str] = set()
    for index_path in sorted(E2E_MODELS.glob("*/MODEL.toml")):
        raw = tomllib.loads(index_path.read_text(encoding="utf-8"))
        entries = raw.get("test_manifests", [])
        assert isinstance(entries, list), f"{index_path}: test_manifests must be a list"
        for entry in entries:
            assert isinstance(entry, str) and entry
            relative = PurePosixPath(entry.replace("\\", "/"))
            assert not relative.is_absolute() and ".." not in relative.parts
            manifest_path = index_path.parent.joinpath(*relative.parts)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            family = manifest.get("family")
            if isinstance(family, str) and family:
                families.add(family)
    return families


def test_build_only_and_zero_pin_e2e_exemptions_are_fail_closed() -> None:
    metadata = _family_metadata()
    manifest_families = _manifest_families()
    build_only: set[str] = set()
    zero_pin: set[str] = set()

    for family, capabilities in metadata.items():
        runtime_dir = RUNTIMES / family
        runtime_manifest = runtime_dir / "MODEL.toml"
        if "complete_bundle_builder" in capabilities and not runtime_manifest.is_file():
            build_only.add(family)

        if "production_runtime_unpinned" not in capabilities:
            continue
        assert "complete_bundle_builder" in capabilities, (
            f"{family} production_runtime_unpinned requires complete_bundle_builder"
        )
        assert runtime_manifest.is_file(), (
            f"{family} declares production_runtime_unpinned without a runtime manifest"
        )
        pin_sources = sorted(runtime_dir.glob("*_production_qualification_pins.cpp"))
        assert len(pin_sources) == 1, (
            f"{family} production_runtime_unpinned requires exactly one production pin source"
        )
        pin_source = pin_sources[0].read_text(encoding="utf-8")
        pin_arrays = re.findall(
            r"std::array\s*<\s*NativeQualificationStaticPin\s*,\s*([0-9]+)\s*>",
            pin_source,
        )
        assert pin_arrays == ["0"], (
            f"{family} production_runtime_unpinned requires one audited zero-sized "
            f"NativeQualificationStaticPin array; found sizes {pin_arrays}"
        )
        zero_pin.add(family)

    uncovered = set(metadata) - manifest_families - build_only - zero_pin
    assert not uncovered, (
        f"Families without E2E manifest coverage: {sorted(uncovered)}. Add a declared E2E "
        "manifest, or use only the fail-closed build-only/zero-pin exemption. Remove "
        "production_runtime_unpinned and add E2E coverage when a pin is activated."
    )

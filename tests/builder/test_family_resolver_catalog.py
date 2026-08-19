# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only coverage for the family resolver's current model catalog."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tensorrt_model_connect.models as families


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "website/data/hf-model-metadata.json"
E2E_MODELS_ROOT = REPO_ROOT / "python/tensorrt_model_connect/models"

# This checkpoint is metadata for a family-owned test dependency rather than an
# E2E manifest root, so it intentionally has no direct manifest of its own.
CATALOG_ONLY_FAMILIES = {
    "yujiepan/deepseek-v3-tiny-random": "deepseek_v2",
}


def _catalog() -> list[dict[str, object]]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return list(payload["checkpoints"])


def _e2e_families_by_hf_id() -> dict[str, str]:
    owners: dict[str, str] = {}
    paths = set(E2E_MODELS_ROOT.glob("*/tests/manifests/*.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        hf_id = payload.get("hf_id")
        family = payload.get("family")
        if not isinstance(hf_id, str) or not isinstance(family, str):
            continue
        previous = owners.setdefault(hf_id, family)
        assert previous == family, f"{hf_id} is owned by both {previous} and {family}"
    return owners


def _forbid_model_imports(monkeypatch) -> None:
    def fail(module_name: str):
        raise AssertionError(f"metadata resolution imported {module_name}")

    monkeypatch.setattr(families.importlib, "import_module", fail)


def test_catalog_config_metadata_resolves_to_declared_family(monkeypatch) -> None:
    _forbid_model_imports(monkeypatch)
    expected_by_hf_id = {
        **_e2e_families_by_hf_id(),
        **CATALOG_ONLY_FAMILIES,
    }

    checked = 0
    for entry in _catalog():
        model_type = entry.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            continue
        hf_id = str(entry["hf_id"])
        expected = expected_by_hf_id.get(hf_id)
        assert expected is not None, f"catalog entry has no declared owner: {hf_id}"
        architectures = entry.get("architectures") or []
        config = SimpleNamespace(
            model_type=model_type,
            architectures=architectures,
            raw={"architectures": architectures},
        )
        assert families.resolve_family_id_from_config(config) == expected, hf_id
        checked += 1

    assert checked


def test_family_metadata_routes_are_unambiguous_without_model_imports(
    monkeypatch,
) -> None:
    _forbid_model_imports(monkeypatch)
    pipeline_owners: dict[str, str] = {}

    for metadata in families._load_family_metadata():
        for alias in metadata.aliases:
            candidates = families._candidate_module_names(alias)
            assert candidates and candidates[0] == metadata.import_module, (
                metadata.id,
                alias,
                candidates,
            )

        for prefix in metadata.prefixes:
            candidates = families._candidate_module_names(f"{prefix}_probe")
            assert candidates and candidates[0] == metadata.import_module, (
                metadata.id,
                prefix,
                candidates,
            )

        for pattern in metadata.architecture_patterns:
            config = SimpleNamespace(
                model_type="unclaimed_model_type",
                architectures=[pattern],
                raw={"architectures": [pattern]},
            )
            assert families.resolve_family_id_from_config(config) == metadata.id

        for pipeline_class in metadata.diffusion_pipeline_classes:
            previous = pipeline_owners.setdefault(pipeline_class, metadata.id)
            assert previous == metadata.id, (
                f"{pipeline_class} is owned by both {previous} and {metadata.id}"
            )
            assert (
                families.resolve_diffusion_family_id(pipeline_class)
                == metadata.id
            )

    assert pipeline_owners

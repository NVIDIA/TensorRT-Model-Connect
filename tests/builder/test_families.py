# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-wide contracts for model-owned family metadata."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import tensorrt_model_connect.models as families
from tests.e2e_harness.manifest_loader import iter_manifest_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python/tensorrt_model_connect/models"


def test_candidate_names_use_the_cached_metadata_index(monkeypatch) -> None:
    metadata = [
        families._FamilyMetadata(
            id="alpha",
            import_module="alpha",
            aliases=frozenset({"alpha"}),
            compact_aliases=frozenset({"alpha"}),
            prefixes=frozenset({"alpha"}),
            compact_prefixes=frozenset({"alpha"}),
            capabilities=frozenset(),
            architecture_patterns=frozenset(),
            diffusion_pipeline_classes=frozenset(),
            nemo_target_patterns=frozenset(),
            nemo_model_type="",
        ),
        families._FamilyMetadata(
            id="alpha_vl",
            import_module="alpha_vl",
            aliases=frozenset({"alpha_vl"}),
            compact_aliases=frozenset({"alphavl"}),
            prefixes=frozenset({"alpha_vl"}),
            compact_prefixes=frozenset({"alphavl"}),
            capabilities=frozenset(),
            architecture_patterns=frozenset(),
            diffusion_pipeline_classes=frozenset(),
            nemo_target_patterns=frozenset(),
            nemo_model_type="",
        ),
    ]
    monkeypatch.setattr(families, "_METADATA_CACHE", metadata)
    monkeypatch.setattr(families, "_METADATA_INDEX_CACHE", None)

    assert families._candidate_module_names("alpha3") == ["alpha"]
    assert families._candidate_module_names("alpha_vl") == ["alpha_vl", "alpha"]

    monkeypatch.setattr(
        families,
        "_load_family_metadata",
        lambda: (_ for _ in ()).throw(
            AssertionError("candidate lookup did not use its index cache")
        ),
    )
    assert families._candidate_module_names("alpha3") == ["alpha"]


def test_family_metadata_owns_capabilities_and_nemo_resolution() -> None:
    for metadata in families._load_family_metadata():
        alias = next(iter(sorted(metadata.aliases)), None)
        if alias is not None:
            for capability in metadata.capabilities:
                assert families.family_has_capability(alias, capability)
        for pattern in metadata.nemo_target_patterns:
            assert families.resolve_nemo_model_type(
                {"target": f"example.{pattern}.Model"}
            ) == metadata.nemo_model_type

    assert families.resolve_nemo_model_type(
        {"model_type": "custom_nemo_model"}
    ) == "custom_nemo_model"


def test_nemo_archive_resolution_uses_family_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    metadata = families._FamilyMetadata(
        id="example_nemo",
        import_module="example_nemo",
        aliases=frozenset({"example_nemo"}),
        compact_aliases=frozenset({"examplenemo"}),
        prefixes=frozenset({"example_nemo"}),
        compact_prefixes=frozenset({"examplenemo"}),
        capabilities=frozenset(),
        architecture_patterns=frozenset(),
        diffusion_pipeline_classes=frozenset(),
        nemo_target_patterns=frozenset(),
        nemo_model_type="",
        nemo_archive_adapter="adapter.py|resolve",
    )

    def adapter(path: Path) -> Path:
        resolved = tmp_path / "resolved"
        resolved.mkdir()
        (resolved / "config.json").write_text(
            json.dumps(
                {"model_type": "example_nemo", "_nemo_archive_path": str(path)}
            ),
            encoding="utf-8",
        )
        return resolved

    monkeypatch.setattr(families, "_load_family_metadata", lambda: [metadata])
    monkeypatch.setattr(
        families, "_load_metadata_callable_from_file", lambda *_args: adapter
    )
    archive = tmp_path / "example.nemo"
    payload = b"target: example.Target\n"
    info = tarfile.TarInfo("model_config.yaml")
    info.size = len(payload)
    with tarfile.open(archive, "w") as output:
        output.addfile(info, io.BytesIO(payload))

    resolved = families.resolve_nemo_archive_model_dir(archive)
    assert resolved is not None
    assert json.loads((Path(resolved) / "config.json").read_text())[
        "_nemo_archive_path"
    ] == str(archive)


def test_every_family_has_an_e2e_manifest_and_every_manifest_family_exists() -> None:
    family_ids = set(families.available_family_ids())
    manifested: set[str] = set()
    unknown: list[tuple[str, str]] = []
    for manifest_path in iter_manifest_paths(REPO_ROOT / "python/tensorrt_model_connect/models"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = str(payload.get("family", ""))
        if family:
            manifested.add(family)
            if family not in family_ids:
                unknown.append((str(manifest_path), family))

    assert unknown == []
    assert family_ids <= manifested


def test_family_models_do_not_import_other_families() -> None:
    for family_id in families.available_family_ids():
        model_path = FAMILIES_ROOT / family_id / "model.py"
        if not model_path.is_file():
            continue
        source = model_path.read_text(encoding="utf-8")
        foreign_prefix = "tensorrt_model_connect.models."
        for line in source.splitlines():
            if foreign_prefix not in line:
                continue
            assert f"{foreign_prefix}{family_id}" in line, (family_id, line)

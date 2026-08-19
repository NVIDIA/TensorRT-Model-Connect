# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the metadata-bounded family model resolver."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import tensorrt_model_connect.models as families


def _model(*, owns: bool = True):
    return types.SimpleNamespace(
        matches=lambda _config: owns,
        build=lambda _model_dir, _output_path, **_options: None,
    )


def test_diffusion_family_resolution_is_metadata_only(monkeypatch) -> None:
    metadata = types.SimpleNamespace(
        id="synthetic_family",
        diffusion_pipeline_classes=frozenset({"SyntheticPipeline"}),
    )
    monkeypatch.setattr(families, "_load_family_metadata", lambda: [metadata])

    assert families.resolve_diffusion_family_id("SyntheticPipeline") == (
        "synthetic_family"
    )
    assert families.resolve_diffusion_family_id("UnknownPipeline") is None


def test_load_model_by_id_imports_only_the_declared_model_module(
    monkeypatch,
    tmp_path,
) -> None:
    family_dir = tmp_path / "models" / "example_family"
    family_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text(
        'id = "example_family"\n', encoding="utf-8"
    )
    imported = []
    model = _model()
    monkeypatch.setattr(families, "__file__", str(tmp_path / "models/__init__.py"))
    monkeypatch.setattr(
        families.importlib,
        "import_module",
        lambda name: imported.append(name) or model,
    )

    assert families.load_model_by_id("example_family") is model
    assert imported == ["tensorrt_model_connect.models.example_family.model"]


def test_load_model_by_id_requires_build_and_matches(monkeypatch, tmp_path) -> None:
    family_dir = tmp_path / "models" / "example_family"
    family_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text(
        'id = "example_family"\n', encoding="utf-8"
    )
    monkeypatch.setattr(families, "__file__", str(tmp_path / "models/__init__.py"))
    monkeypatch.setattr(
        families.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(build=lambda *_args, **_kwargs: None),
    )

    with pytest.raises(TypeError, match=r"must define matches\(\)"):
        families.load_model_by_id("example_family")


@pytest.mark.parametrize("descriptor", ("", 'id = "other_family"\n'))
def test_load_model_by_id_rejects_missing_or_mismatched_owner_id(
    monkeypatch, tmp_path, descriptor
) -> None:
    family_dir = tmp_path / "models" / "example_family"
    family_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text(descriptor, encoding="utf-8")
    monkeypatch.setattr(families, "__file__", str(tmp_path / "models/__init__.py"))

    with pytest.raises(ValueError, match="model id|Model id"):
        families.load_model_by_id("example_family")


def test_find_model_checks_only_metadata_candidates(monkeypatch) -> None:
    first = _model(owns=False)
    second = _model(owns=True)
    loaded = []
    monkeypatch.setattr(
        families,
        "_candidate_module_names_from_config",
        lambda _config: ["first"],
    )
    monkeypatch.setattr(
        families,
        "_candidate_module_names",
        lambda _model_type: ["second"],
    )
    monkeypatch.setattr(
        families,
        "load_model_by_id",
        lambda family_id: loaded.append(family_id)
        or {"first": first, "second": second}[family_id],
    )

    config = types.SimpleNamespace(model_type="synthetic", raw={})
    assert families.find_model(config) is second
    assert loaded == ["first", "second"]


def test_unknown_family_id_and_model_return_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(families, "__file__", str(tmp_path / "models/__init__.py"))
    monkeypatch.setattr(families, "_candidate_module_names", lambda _value: [])
    monkeypatch.setattr(
        families, "_candidate_module_names_from_config", lambda _value: []
    )

    assert families.load_model_by_id("unknown") is None
    assert families.find_model("unknown") is None


def test_string_lookup_trusts_declared_metadata_without_rechecking_model(
    monkeypatch,
) -> None:
    model = _model(owns=False)
    monkeypatch.setattr(families, "resolve_family_id", lambda _value: "example_family")
    monkeypatch.setattr(families, "load_model_by_id", lambda _family: model)

    assert families.find_model("declared_alias") is model


def test_current_repository_declares_all_family_ids() -> None:
    family_root = Path(families.__file__).parent
    declared = {
        manifest.parent.name for manifest in family_root.glob("*/MODEL.toml")
    }
    assert set(families.available_family_ids()) == declared

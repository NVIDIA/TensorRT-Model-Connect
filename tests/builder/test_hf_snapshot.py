# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tensorrt_model_connect.engine_builder import _text_bundle_model_id
from tensorrt_model_connect.hf_snapshot import hf_cache_snapshot_identity


def _snapshot(
    root: Path,
    *,
    model_id: str = "ExampleOrg/ExampleModel",
    revision: str = "ABCDEF0123456789",
) -> Path:
    organization, repository = model_id.split("/", 1)
    snapshot = (
        root
        / f"models--{organization}--{repository}"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    return snapshot


def test_hf_cache_snapshot_identity_recovers_canonical_source(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)

    assert hf_cache_snapshot_identity(snapshot) == (
        "ExampleOrg/ExampleModel",
        "ABCDEF0123456789",
    )


def test_hf_cache_snapshot_identity_resolves_snapshot_symlink(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path / "cache")
    alias = tmp_path / "model-alias"
    alias.symlink_to(snapshot, target_is_directory=True)

    assert hf_cache_snapshot_identity(alias) == (
        "ExampleOrg/ExampleModel",
        "ABCDEF0123456789",
    )


def test_hf_cache_snapshot_identity_rejects_regular_local_directory(
    tmp_path: Path,
) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()

    assert hf_cache_snapshot_identity(local_model) is None


def test_text_bundle_identity_uses_hf_source_and_local_basename(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path / "cache")
    local_model = tmp_path / "local-model"
    local_model.mkdir()

    assert _text_bundle_model_id(snapshot, None) == (
        "ExampleOrg/ExampleModel"
    )
    assert _text_bundle_model_id(local_model, None) == "local-model"


def test_text_bundle_identity_keeps_qualified_model_authoritative(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    qualification = SimpleNamespace(
        qualified_model_id="QualifiedOrg/QualifiedModel"
    )

    assert _text_bundle_model_id(
        snapshot,
        qualification,
    ) == "QualifiedOrg/QualifiedModel"

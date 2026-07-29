# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline cache resolution checks for Eagle VLM HF references."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub
from huggingface_hub import constants

from tests.e2e.models.eagle_vlm.e2e_plugins.references.hf_transformers import (
    _resolve_cached_model_ref,
)


def test_partial_cached_snapshot_resolves_to_exact_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Runtime-complete partial snapshots remain reproducibly offline."""
    cache_root = tmp_path / "hub"
    commit = "9c20c4aedf9ec87b6b7346c3bc4754ea030dab35"
    repo_cache = cache_root / "models--nvidia--example-reranker"
    snapshot = repo_cache / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(commit, encoding="utf-8")

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache_root))

    def _incomplete_snapshot(*args, **kwargs):
        raise RuntimeError("snapshot is missing non-runtime repository files")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _incomplete_snapshot)

    assert (
        _resolve_cached_model_ref("nvidia/example-reranker")
        == str(snapshot)
    )


def test_missing_cached_snapshot_keeps_hub_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A genuinely absent model remains visible as a missing Hub reference."""
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def _missing_snapshot(*args, **kwargs):
        raise RuntimeError("model is not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _missing_snapshot)

    assert _resolve_cached_model_ref("nvidia/missing-model") == "nvidia/missing-model"

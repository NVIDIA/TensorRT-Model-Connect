# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned E2E checkpoints from the cache staged by offline CI."""

from pathlib import Path

import huggingface_hub
from huggingface_hub import constants
from huggingface_hub.utils import LocalEntryNotFoundError
import pytest

from .test_e2e import _checkpoint


_REVISION = "a" * 40
_MANIFEST = {"hf_id": "example/llama", "hf_revision": _REVISION}


@pytest.fixture
def staged_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    # Older staging clients create snapshots without the newer tree metadata.
    snapshot = tmp_path / "models--example--llama" / "snapshots" / _REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_offline_checkpoint_uses_staged_pinned_snapshot(staged_snapshot: Path) -> None:
    assert _checkpoint(_MANIFEST) == staged_snapshot


def test_offline_checkpoint_rejects_unstaged_revision(staged_snapshot: Path) -> None:
    manifest = {**_MANIFEST, "hf_revision": "b" * 40}
    with pytest.raises(LocalEntryNotFoundError):
        _checkpoint(manifest)


def test_offline_checkpoint_requires_config(staged_snapshot: Path) -> None:
    (staged_snapshot / "config.json").unlink()
    with pytest.raises(AssertionError):
        _checkpoint(_MANIFEST)


def test_online_checkpoint_allows_download(
    staged_snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)

    def download(*, repo_id: str, revision: str, local_files_only: bool = False) -> str:
        assert repo_id == _MANIFEST["hf_id"]
        assert revision == _REVISION
        assert not local_files_only
        return str(staged_snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert _checkpoint(_MANIFEST) == staged_snapshot

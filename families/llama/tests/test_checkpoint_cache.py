# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged checkpoint lookup without Hub metadata or network access."""

import pytest
from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError

from .test_e2e import _checkpoint


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return tmp_path


def test_pinned_checkpoint_uses_snapshot_without_cached_tree(offline_cache):
    revision = "a" * 40
    snapshot = offline_cache / "models--trtmc-test--llama" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert _checkpoint({"hf_id": "trtmc-test/llama", "hf_revision": revision}) == snapshot


def test_missing_checkpoint_is_not_replaced_by_another_revision(offline_cache):
    cached = offline_cache / "models--trtmc-test--llama" / "snapshots" / ("a" * 40)
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LocalEntryNotFoundError):
        _checkpoint({"hf_id": "trtmc-test/llama", "hf_revision": "b" * 40})


def test_staged_snapshot_still_requires_config(offline_cache):
    revision = "a" * 40
    snapshot = offline_cache / "models--trtmc-test--llama" / "snapshots" / revision
    snapshot.mkdir(parents=True)

    with pytest.raises(AssertionError):
        _checkpoint({"hf_id": "trtmc-test/llama", "hf_revision": revision})

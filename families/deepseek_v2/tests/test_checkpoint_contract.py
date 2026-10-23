# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline E2E checkpoint lookup must consume the already staged Hub snapshot."""

import json

import pytest

hub = pytest.importorskip("huggingface_hub")


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def metadata_forbidden(*args, **kwargs):
        pytest.fail("an offline staged checkpoint must not request Hub metadata")

    monkeypatch.setattr(hub.HfApi, "repo_info", metadata_forbidden)
    monkeypatch.setattr(hub.HfApi, "list_repo_tree", metadata_forbidden)
    manifest = {"hf_id": "trtmc-test/deepseek-v2", "hf_revision": "a" * 40}
    snapshot = cache / "models--trtmc-test--deepseek-v2" / "snapshots" / manifest["hf_revision"]
    return manifest, snapshot


@pytest.mark.parametrize("offline_value", ["1", "TRUE", "YES", "ON"])
def test_staged_snapshot_is_read_without_hub_metadata(offline_cache, monkeypatch, offline_value):
    from families.deepseek_v2.tests import test_e2e

    manifest, snapshot = offline_cache
    monkeypatch.setenv("HF_HUB_OFFLINE", offline_value)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "deepseek_v2"}))
    assert test_e2e._checkpoint(manifest) == snapshot


def test_missing_offline_snapshot_fails_without_network(offline_cache):
    from huggingface_hub.utils import LocalEntryNotFoundError

    from families.deepseek_v2.tests import test_e2e

    manifest, _ = offline_cache
    with pytest.raises(LocalEntryNotFoundError):
        test_e2e._checkpoint(manifest)


def test_snapshot_without_config_keeps_the_required_checkpoint_assertion(offline_cache):
    from families.deepseek_v2.tests import test_e2e

    manifest, snapshot = offline_cache
    snapshot.mkdir(parents=True)
    with pytest.raises(AssertionError):
        test_e2e._checkpoint(manifest)

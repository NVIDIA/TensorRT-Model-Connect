# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume host-staged checkpoints without contacting Hub from the GPU container."""

import pytest

hub = pytest.importorskip("huggingface_hub")


@pytest.fixture
def offline_checkpoint(tmp_path, monkeypatch):
    from huggingface_hub import constants

    from families.clef.tests import test_e2e

    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRTMC_CLEF_CHECKPOINT", raising=False)
    bundle = tmp_path / "existing.bundle"
    bundle.touch()
    monkeypatch.setenv("TRTMC_CLEF_BUNDLE", str(bundle))
    manifest = {
        "tensor_parallel_size": 1,
        "hf_id": "trtmc-test/clef",
        "hf_revision": "a" * 40,
    }
    monkeypatch.setattr(test_e2e, "MANIFEST", manifest)

    def metadata_forbidden(*args, **kwargs):
        pytest.fail("an offline staged checkpoint must not request Hub metadata")

    monkeypatch.setattr(hub.HfApi, "repo_info", metadata_forbidden)
    monkeypatch.setattr(hub.HfApi, "list_repo_tree", metadata_forbidden)
    snapshot = cache / "models--trtmc-test--clef" / "snapshots" / manifest["hf_revision"]
    return test_e2e, snapshot, bundle


@pytest.mark.parametrize("offline_value", ["1", "TRUE", "YES", "ON"])
def test_staged_checkpoint_never_requests_hub_metadata(
    offline_checkpoint, monkeypatch, offline_value
):
    test_e2e, snapshot, bundle = offline_checkpoint
    monkeypatch.setenv("HF_HUB_OFFLINE", offline_value)
    snapshot.mkdir(parents=True)
    assert test_e2e.clef_bundle.__wrapped__(None) == (snapshot, bundle)


def test_missing_offline_checkpoint_fails_without_network(offline_checkpoint):
    from huggingface_hub.errors import LocalEntryNotFoundError

    test_e2e, _, _ = offline_checkpoint
    with pytest.raises(LocalEntryNotFoundError):
        test_e2e.clef_bundle.__wrapped__(None)


def test_explicit_checkpoint_override_still_avoids_hub(offline_checkpoint, monkeypatch, tmp_path):
    test_e2e, _, bundle = offline_checkpoint
    checkpoint = tmp_path / "explicit-checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("TRTMC_CLEF_CHECKPOINT", str(checkpoint))
    assert test_e2e.clef_bundle.__wrapped__(None) == (checkpoint, bundle)

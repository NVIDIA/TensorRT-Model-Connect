# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification rejects mismatched checkpoint and prebuilt-bundle provenance."""

import copy

import pytest

from families.qwen3_next.tests.checkpoint import (
    assert_bundle_checkpoint, assert_pinned_checkpoint, manifest,
)


def snapshot(tmp_path, repository="Qwen--Qwen3-Coder-Next", revision=None):
    path = tmp_path / ("models--" + repository) / "snapshots" / (revision or manifest()["hf_revision"])
    path.mkdir(parents=True)
    return path


def test_correct_snapshot_and_alias_are_accepted(tmp_path):
    path = snapshot(tmp_path)
    assert_pinned_checkpoint(path)
    alias = tmp_path / "checkpoint"
    alias.symlink_to(path, target_is_directory=True)
    assert_pinned_checkpoint(alias)


@pytest.mark.parametrize("options", [{"revision": "0" * 40}, {"repository": "other--checkpoint"}])
def test_wrong_snapshot_identity_is_rejected(tmp_path, options):
    with pytest.raises(AssertionError, match="pinned checkpoint"):
        assert_pinned_checkpoint(snapshot(tmp_path, **options))


@pytest.mark.parametrize("changed", [None, "config.json", "model.safetensors.index.json", "part.safetensors"])
def test_each_local_asset_receipt_must_match_the_pin(tmp_path, changed):
    root = tmp_path / "local"
    receipts = root / ".cache/huggingface/download"
    receipts.mkdir(parents=True)
    for name in ("config.json", "model.safetensors.index.json", "part.safetensors"):
        (root / name).write_bytes(b"")
        revision = "0" * 40 if name == changed else manifest()["hf_revision"]
        (receipts / (name + ".metadata")).write_text(revision + "\nunused\n")
    if changed is None:
        assert_pinned_checkpoint(root)
    else:
        with pytest.raises(AssertionError, match="wrong revision"):
            assert_pinned_checkpoint(root)


def test_missing_checkpoint_receipts_are_rejected(tmp_path):
    with pytest.raises(AssertionError, match="lacks a pinned receipt"):
        assert_pinned_checkpoint(tmp_path)


def valid_bundle():
    selected = manifest()
    return {"provenance": {"version": 1, "checkpoint": {
        "hf_id": selected["hf_id"], "revision": selected["hf_revision"],
    }}}


def test_matching_prebuilt_checkpoint_is_accepted():
    assert_bundle_checkpoint(valid_bundle(), manifest())


@pytest.mark.parametrize("identity", [None, {}, {"hf_id": "other/model", "revision": manifest()["hf_revision"]},
    {"hf_id": manifest()["hf_id"], "revision": "0" * 40}])
def test_wrong_or_unknown_prebuilt_identity_fails(identity):
    info = copy.deepcopy(valid_bundle())
    info["provenance"]["checkpoint"] = identity
    with pytest.raises(AssertionError, match="checkpoint differs"):
        assert_bundle_checkpoint(info, manifest())


def test_missing_prebuilt_provenance_fails():
    with pytest.raises(AssertionError, match="lacks provenance"):
        assert_bundle_checkpoint({}, manifest())
    info = valid_bundle()
    del info["provenance"]["checkpoint"]
    with pytest.raises(AssertionError, match="checkpoint differs"):
        assert_bundle_checkpoint(info, manifest())


def test_layer_revision_check_precedes_engine_construction(tmp_path, monkeypatch):
    from families.qwen3_next.tests.test_graph import test_checkpoint_layer
    path = snapshot(tmp_path, revision="0" * 40)
    monkeypatch.setenv("TRTMC_QWEN3_NEXT_CHECKPOINT", str(path))
    with pytest.raises(AssertionError, match="pinned checkpoint"):
        test_checkpoint_layer(0)

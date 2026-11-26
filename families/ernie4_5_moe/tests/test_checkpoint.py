# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint provenance must distinguish local overrides from pinned snapshots."""

import pytest

from ._checkpoint import checkpoint_record


@pytest.fixture
def manifest():
    return {"hf_id": "owner/model", "hf_revision": "a" * 40, "precision": "fp32"}


def test_local_checkpoint_does_not_claim_the_requested_revision(tmp_path, manifest):
    record = checkpoint_record(tmp_path, manifest, local_override=True)
    assert record["pinned"] is False
    assert record["source"] == "local_override"
    assert record["path"] == str(tmp_path.resolve())
    assert "revision" not in record and "repo_id" not in record
    assert record["requested_checkpoint"]["revision"] == manifest["hf_revision"]


def test_cached_checkpoint_reports_its_verified_snapshot_identity(tmp_path, manifest):
    directory = tmp_path / "models--owner--model" / "snapshots" / manifest["hf_revision"]
    directory.mkdir(parents=True)
    record = checkpoint_record(directory, manifest, local_override=False)
    assert record["pinned"] is True
    assert record["repo_id"] == manifest["hf_id"]
    assert record["revision"] == directory.name


@pytest.mark.parametrize(
    "repo,revision", [("models--other--model", "a" * 40), ("models--owner--model", "b" * 40)]
)
def test_cached_checkpoint_rejects_wrong_snapshot_identity(tmp_path, manifest, repo, revision):
    directory = tmp_path / repo / "snapshots" / revision
    directory.mkdir(parents=True)
    with pytest.raises(AssertionError, match="snapshot identity"):
        checkpoint_record(directory, manifest, local_override=False)

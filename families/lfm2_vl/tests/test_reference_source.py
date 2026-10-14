# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from .reference_source import CHECKPOINT, REVISION, reference_source


def test_remote_reference_requests_the_recorded_commit():
    options, metadata = reference_source(CHECKPOINT)
    assert options == {"revision": REVISION}
    assert metadata["checkpoint"] == CHECKPOINT
    assert metadata["revision"] == REVISION
    assert metadata["provenance"]["revision_verified"]


def test_local_reference_does_not_claim_an_unverified_revision(tmp_path):
    options, metadata = reference_source(str(tmp_path))
    assert options == {"local_files_only": True}
    assert metadata["checkpoint"] == "local" and metadata["revision"] is None
    assert metadata["provenance"] == {"source": "local", "revision_verified": False}
    assert str(tmp_path) not in json.dumps(metadata)


def test_unknown_repository_does_not_acquire_the_pinned_identity():
    with pytest.raises(ValueError, match="pinned repository"):
        reference_source("another-owner/another-model")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from .reference_source import CHECKPOINT, REVISION, reference_source


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_bytes(
        (Path(__file__).parent / "fixtures/config.json").read_bytes()
    )
    return tmp_path


def test_remote_reference_requests_the_recorded_commit(monkeypatch, checkpoint):
    import huggingface_hub
    from huggingface_hub import constants

    calls = []

    def config(*args, **kwargs):
        calls.append((args, kwargs))
        return str(checkpoint / "config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", config)
    options, metadata = reference_source(CHECKPOINT)
    assert calls == [
        (
            (CHECKPOINT, "config.json"),
            {"revision": REVISION, "local_files_only": constants.HF_HUB_OFFLINE},
        )
    ]
    assert options == {"revision": REVISION}
    assert metadata["checkpoint"] == CHECKPOINT
    assert metadata["revision"] == REVISION
    assert metadata["provenance"]["revision_verified"]


def test_local_reference_does_not_claim_an_unverified_revision(checkpoint):
    options, metadata = reference_source(str(checkpoint))
    assert options == {"local_files_only": True}
    assert metadata["checkpoint"] == "local" and metadata["revision"] is None
    assert metadata["provenance"] == {"source": "local", "revision_verified": False}
    assert str(checkpoint) not in json.dumps(metadata)


def test_unknown_repository_does_not_acquire_the_pinned_identity():
    with pytest.raises(ValueError, match="pinned repository"):
        reference_source("another-owner/another-model")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve staged checkpoints with the actual Hub cache contract on CPU."""

import socket
from pathlib import Path

import pytest

from families.lfm2_moe.tests import test_e2e as e2e

hub = pytest.importorskip("huggingface_hub")
constants = pytest.importorskip("huggingface_hub.constants")
LocalEntryNotFoundError = pytest.importorskip("huggingface_hub.errors").LocalEntryNotFoundError


MANIFEST = next(
    manifest for manifest, _ in e2e._CASES.values() if manifest["hf_id"] == "LiquidAI/LFM2.5-8B-A1B"
)


def deny_remote_request(*args, **kwargs):
    raise AssertionError("staged checkpoint resolution must not query the network")


def offline_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub.HfApi, "repo_info", deny_remote_request)
    monkeypatch.setattr(hub.HfApi, "list_repo_tree", deny_remote_request)
    monkeypatch.setattr(socket.socket, "connect", deny_remote_request)
    return (
        tmp_path
        / ("models--" + MANIFEST["hf_id"].replace("/", "--"))
        / "snapshots"
        / MANIFEST["hf_revision"]
    )


def test_actual_hub_resolves_staged_snapshot_without_remote_tree(monkeypatch, tmp_path):
    snapshot = offline_cache(monkeypatch, tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    # The isolated layout contains snapshot files, not the Hub's optional tree
    # metadata. Execute actual snapshot_download, with all remote paths denied.
    assert e2e._checkpoint(MANIFEST) == snapshot


def test_missing_offline_snapshot_fails_without_network(monkeypatch, tmp_path):
    offline_cache(monkeypatch, tmp_path)
    with pytest.raises(LocalEntryNotFoundError):
        e2e._checkpoint(MANIFEST)


@pytest.mark.parametrize("offline,raw_flag", [(True, "yes"), (False, "0"), (False, "false")])
def test_checkpoint_uses_parsed_hub_mode_and_preserves_online_lookup(
    monkeypatch, tmp_path, offline, raw_flag
):
    monkeypatch.setenv("HF_HUB_OFFLINE", raw_flag)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    (tmp_path / "config.json").write_text("{}")
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", resolve)
    assert e2e._checkpoint(MANIFEST) == Path(tmp_path)
    assert calls == [
        {
            "repo_id": MANIFEST["hf_id"],
            "revision": MANIFEST["hf_revision"],
            "local_files_only": offline,
        }
    ]

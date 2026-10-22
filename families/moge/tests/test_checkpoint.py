# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint lookup must work with a pre-staged offline Hub cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("cache_state", ["complete", "missing_model", "wrong_revision"])
def test_checkpoint_uses_exact_cached_file_offline(tmp_path: Path, cache_state: str) -> None:
    revision = "a" * 40
    cached_revision = "b" * 40 if cache_state == "wrong_revision" else revision
    cache = tmp_path / "hub"
    snapshot = cache / "models--fixture--moge" / "snapshots" / cached_revision
    snapshot.mkdir(parents=True)
    filename = "other.pt" if cache_state == "missing_model" else "model.pt"
    (snapshot / filename).write_bytes(b"cached checkpoint fixture")

    repository = Path(__file__).resolve().parents[3]
    # Initialize Hub constants in a fresh interpreter. The cache deliberately
    # has no repository-tree metadata, as in older pre-staged CI caches.
    script = """
import json
import socket

def reject_network(*args, **kwargs):
    raise AssertionError("offline checkpoint lookup attempted a network connection")

socket.socket.connect = reject_network
socket.socket.connect_ex = reject_network

from families.moge.tests.test_e2e import _checkpoint

path = _checkpoint({"hf_id": "fixture/moge", "hf_revision": "a" * 40})
assert (path / "model.pt").read_bytes() == b"cached checkpoint fixture"
print(json.dumps(str(path)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env={
            **os.environ,
            "HF_HOME": str(tmp_path / "hf-home"),
            "HF_HUB_CACHE": str(cache),
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join([str(repository / "core/builder"), str(repository)]),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if cache_state == "complete":
        assert result.returncode == 0, result.stderr
        assert Path(json.loads(result.stdout)) == snapshot
    else:
        assert result.returncode != 0
        assert "LocalEntryNotFoundError" in result.stderr
    assert "attempted a network connection" not in result.stderr

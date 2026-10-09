# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Hub cache readers with tiny files and a mocked metadata server.

Run with the Hub version pinned by requirements/community-gpu-linux-amd64.lock.
No checkpoint weights, external service or GPU are needed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import community_gpu_ci

hub = pytest.importorskip("huggingface_hub", minversion="1.33.0")
httpx = pytest.importorskip("httpx")


def _offline_call(cache: Path, repo_id: str, revision: str):
    """Use a fresh interpreter so the result requires on-disk metadata."""
    env = {key: os.environ[key] for key in ("PATH", "PYTHONPATH") if key in os.environ}
    env.update(
        HF_HOME=str(cache.parent),
        HF_HUB_CACHE=str(cache),
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,socket,sys; "
            "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Network forbidden')); "
            "from huggingface_hub import snapshot_download; "
            "print(json.dumps(snapshot_download(sys.argv[1], revision=sys.argv[2])))",
            repo_id,
            revision,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_preparation_repairs_legacy_snapshot_for_unchanged_offline_call(tmp_path, monkeypatch):
    """Repair an offline API failure without changing the caller's arguments."""
    repo_id, revision = "example/checkpoint", "a" * 40
    cache = tmp_path / "hub"
    repository_cache = cache / ("models--" + repo_id.replace("/", "--"))
    snapshot = repository_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    content = b"{}\n"
    (snapshot / "config.json").write_bytes(content)

    before = _offline_call(cache, repo_id, revision)
    assert before.returncode != 0
    assert "OfflineModeIsEnabled" in before.stderr
    assert f"/api/models/{repo_id}/tree/{revision}" in before.stderr

    requests = []
    blob_id = hashlib.sha1(b"blob 3\0" + content).hexdigest()

    def metadata(request):
        requests.append(request.url.path)
        if request.url.path == f"/api/models/{repo_id}/revision/{revision}":
            return httpx.Response(200, json={"id": repo_id, "sha": revision, "siblings": []})
        if request.url.path == f"/api/models/{repo_id}/tree/{revision}":
            return httpx.Response(
                200,
                json=[
                    {"type": "file", "path": "config.json", "size": len(content), "oid": blob_id}
                ],
            )
        raise AssertionError(f"Unexpected preparation request: {request.url}")

    from huggingface_hub.utils import _http

    factory = _http._GLOBAL_CLIENT_FACTORY
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.setattr(hub.constants, "HF_HUB_DISABLE_TELEMETRY", True)
    # A transport fixture replaces only HTTP; the real Hub cache writer runs.
    hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(metadata)))
    try:
        plan = community_gpu_ci.FamilyPlan("fixture", ("smoke",), ((repo_id, revision),))
        community_gpu_ci._stage_checkpoints((plan,), cache)
    finally:
        hub.set_client_factory(factory)

    assert requests == [
        f"/api/models/{repo_id}/revision/{revision}",
        f"/api/models/{repo_id}/tree/{revision}",
    ]
    assert (repository_cache / "trees" / f"{revision}.json").is_file()
    after = _offline_call(cache, repo_id, revision)
    assert after.returncode == 0, after.stderr
    assert Path(json.loads(after.stdout)) == snapshot

    # Metadata must not turn missing checkpoint bytes into a successful result.
    (snapshot / "config.json").unlink()
    incomplete = _offline_call(cache, repo_id, revision)
    assert incomplete.returncode != 0

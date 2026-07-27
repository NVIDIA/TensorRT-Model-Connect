# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security-contract tests for the public-to-private CI bridge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BRIDGE_PATH = REPO_ROOT / ".github" / "scripts" / "private_ci_bridge.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml"


def _bridge():
    spec = importlib.util.spec_from_file_location("private_ci_bridge", BRIDGE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pull(*, head: str = "2" * 40, merge: str = "3" * 40) -> dict:
    return {
        "number": 17,
        "state": "open",
        "draft": False,
        "merge_commit_sha": merge,
        "head": {
            "sha": head,
            "repo": {"full_name": "contributor/fork"},
        },
        "base": {
            "sha": "1" * 40,
            "ref": "main",
            "repo": {
                "id": 1216320259,
                "full_name": "NVIDIA/TensorRT-Model-Connect",
            },
        },
    }


def test_live_snapshot_requires_exact_refs_and_two_merge_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge()
    pull = _pull()

    def request(_token, _method, path, **_kwargs):
        if path.endswith("/pulls/17"):
            return pull
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "1" * 40}}
        if path.endswith("/git/ref/pull/17/head"):
            return {"object": {"sha": "2" * 40}}
        if path.endswith("/git/ref/pull/17/merge"):
            return {"object": {"sha": "3" * 40}}
        if path.endswith(f"/commits/{'3' * 40}"):
            return {
                "parents": [
                    {"sha": "1" * 40},
                    {"sha": "2" * 40},
                ]
            }
        raise AssertionError(path)

    monkeypatch.setattr(bridge, "_request", request)
    assert bridge._resolve_live_snapshot("not-a-real-token", 17) == {
        "pr_number": "17",
        "head_sha": "2" * 40,
        "base_sha": "1" * 40,
        "merge_sha": "3" * 40,
    }


def test_live_snapshot_rejects_parent_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge()
    pull = _pull()

    def request(_token, _method, path, **_kwargs):
        if path.endswith("/pulls/17"):
            return pull
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "1" * 40}}
        if path.endswith("/git/ref/pull/17/head"):
            return {"object": {"sha": "2" * 40}}
        if path.endswith("/git/ref/pull/17/merge"):
            return {"object": {"sha": "3" * 40}}
        if path.endswith(f"/commits/{'3' * 40}"):
            return {
                "parents": [
                    {"sha": "9" * 40},
                    {"sha": "2" * 40},
                ]
            }
        raise AssertionError(path)

    monkeypatch.setattr(bridge, "_request", request)
    with pytest.raises(bridge.BridgeError, match="merge-parent-mismatch"):
        bridge._resolve_live_snapshot("not-a-real-token", 17)


def test_dispatch_sends_only_fixed_structured_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bridge = _bridge()
    request = {
        "base_sha": "1" * 40,
        "head_sha": "2" * 40,
        "merge_sha": "3" * 40,
        "pr_number": "17",
        "request_id": f"trtmc-premerge:17:{'2' * 40}:{'3' * 40}",
        "source_repository": "NVIDIA/TensorRT-Model-Connect",
        "source_repository_id": "1216320259",
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token-placeholder")
    monkeypatch.setenv("SOURCE_CHECK_TOKEN", "source-token-placeholder")
    monkeypatch.setenv("CI_DISPATCH_TOKEN", "ci-token-placeholder")
    monkeypatch.setenv("PRIVATE_CI_OWNER", "internal-owner")
    monkeypatch.setenv("PRIVATE_CI_REPOSITORY", "internal-ci")

    calls: list[tuple[str, str, str, dict | None]] = []

    def api(token, method, path, *, payload=None, expect_json=True):
        calls.append((token, method, path, payload))
        if path.endswith("/check-runs") and method == "POST":
            return {"id": 456}
        return None

    monkeypatch.setattr(bridge, "_request", api)
    bridge.dispatch(request_path, consume_label=True)

    assert calls[0][1:] == (
        "DELETE",
        "/repos/NVIDIA/TensorRT-Model-Connect/issues/17/labels/run-ci",
        None,
    )
    assert calls[1][0] == "source-token-placeholder"
    assert calls[1][3]["head_sha"] == "3" * 40
    dispatch_call = calls[2]
    assert dispatch_call[0] == "ci-token-placeholder"
    assert dispatch_call[1] == "POST"
    assert dispatch_call[2].endswith(
        "/internal-owner/internal-ci/actions/workflows/premerge.yml/dispatches"
    )
    assert dispatch_call[3] == {
        "ref": "main",
        "inputs": {**request, "check_run_id": "456"},
    }
    serialized = json.dumps(dispatch_call[3], sort_keys=True)
    assert "placeholder" not in serialized


def test_public_workflow_never_checks_out_or_executes_pull_request_code() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "pull_request_target:" in workflow
    assert "types: [labeled]" in workflow
    assert "github.event.pull_request.base.sha" in workflow
    assert "github.event.pull_request.base.sha || 'main'" in workflow
    assert "github.event.pull_request.base.sha || github.sha" not in workflow
    assert "github.event.pull_request.head.sha" not in workflow
    assert "github.event.pull_request.merge_commit_sha" not in workflow
    assert "refs/pull/" not in workflow
    assert "persist-credentials: false" in workflow
    assert "permission-actions: write" in workflow
    assert "permission-checks: write" in workflow
    assert "secrets: inherit" not in workflow
    assert "self-hosted" not in workflow
    assert "owner: ${{ secrets.TRTMC_PRIVATE_CI_OWNER }}" in workflow
    assert (
        "repositories: ${{ secrets.TRTMC_PRIVATE_CI_REPOSITORY }}"
        in workflow
    )


def test_private_target_must_be_bounded_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge()
    monkeypatch.setenv("PRIVATE_CI_OWNER", "internal-owner")
    monkeypatch.setenv("PRIVATE_CI_REPOSITORY", "../unsafe")
    with pytest.raises(bridge.BridgeError, match="private-ci-target-invalid"):
        bridge._private_ci_repository()

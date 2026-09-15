# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the stable/dev dispatch and status isolation boundaries."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
STABLE_CONTEXT = "TRTMC Internal CI / Automated premerge gate"
DEV_CONTEXT = "TRTMC Internal CI / Dev premerge (non-blocking)"


def workflow(lane: str) -> dict:
    filename = "internal-ci-bridge.yml" if lane == "stable" else "internal-ci-dev.yml"
    return yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())


def test_stable_does_not_wait_for_dev_or_read_its_verdict() -> None:
    stable, dev = workflow("stable"), workflow("dev")
    assert stable["concurrency"]["group"] != dev["concurrency"]["group"]
    jobs = stable["jobs"]
    assert jobs["launch-dev"]["continue-on-error"] is True
    assert jobs["launch-dev"]["needs"] == "authorize"
    assert jobs["publish"]["needs"] == ["authorize", "announce", "dispatch"]
    assert jobs["dispatch"]["needs"] == ["authorize", "announce"]
    assert jobs["dispatch"].get("continue-on-error", False) is False
    for lane in ("stable", "dev"):
        config = workflow(lane)
        assert "github.ref == 'refs/heads/main'" in config["jobs"]["authorize"]["if"]
        text = json.dumps(config)
        assert (DEV_CONTEXT if lane == "stable" else STABLE_CONTEXT) not in text
        assert "actions/checkout" not in text
    assert "TRTMC_PREMERGE_DEV_REF" not in json.dumps(stable)
    assert "TRTMC_PREMERGE_DEV_REF || 'main'" in json.dumps(dev)


@pytest.mark.parametrize(
    "lane,ci_ref", [("stable", "main"), ("dev", "main"), ("dev", "ci/developer")]
)
def test_dispatch_preserves_the_snapshot_and_lane(tmp_path: Path, lane: str, ci_ref: str) -> None:
    steps = workflow(lane)["jobs"]["dispatch"]["steps"]
    script = next(step["run"] for step in steps if step["id"] == "private_run")
    fake = tmp_path / "gh"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
record = Path(os.environ['PAYLOAD'])
if '--input' in args:
    payload = json.loads(Path(args[args.index('--input') + 1]).read_text())
    record.write_text(json.dumps(payload))
else:
    payload = json.loads(record.read_text())
    data = payload['inputs']
    title = f"Source PR #{data['pr_number']} · {data['head_sha']} · dispatch {data['dispatch_nonce']}"
    print(json.dumps({'workflow_runs': [
        {'id': 42, 'display_title': title, 'created_at': '2026-01-01'},
        {'id': 99, 'display_title': title + '-unrelated', 'created_at': '2026-01-02'}
    ]}))
"""
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    expected = {
        "pr_number": "17",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "tested_sha": "c" * 40,
        "source_tree": "d" * 40,
        "lane": lane,
    }
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            **{key.upper(): value for key, value in expected.items()},
            "CI_REF": ci_ref,
            "PRIVATE_CI_OWNER": "example",
            "PRIVATE_CI_REPOSITORY": "ci",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
            "PAYLOAD": str(tmp_path / "payload"),
        },
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "payload").read_text())
    nonce = payload["inputs"].pop("dispatch_nonce")
    assert len(nonce) == 32
    assert payload == {"ref": ci_ref, "inputs": expected}
    assert "run_id=42\n" in output.read_text()


@pytest.mark.parametrize(
    "dispatch_result,conclusion,expected",
    [
        ("success", "success", "success"),
        ("success", "failure", "failure"),
        ("success", "cancelled", "failure"),
        ("success", "timed_out", "failure"),
        ("failure", "", "failure"),
        ("failure", "success", "failure"),
    ],
)
def test_dev_terminal_results_cannot_write_the_required_context(
    tmp_path: Path, dispatch_result: str, conclusion: str, expected: str
) -> None:
    script = workflow("dev")["jobs"]["publish"]["steps"][0]["run"]
    fake = tmp_path / "gh"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$STATUS_CALL"\n')
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "HEAD_SHA": "a" * 40,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "42",
            "DISPATCH_RESULT": dispatch_result,
            "PRIVATE_CONCLUSION": conclusion,
            "STATUS_CALL": str(calls),
        },
    )
    assert result.returncode == (0 if expected == "success" else 1), result.stderr
    arguments = calls.read_text().splitlines()
    assert f"state={expected}" in arguments
    assert f"context={DEV_CONTEXT}" in arguments
    assert STABLE_CONTEXT not in calls.read_text()


def test_dev_bridge_receives_every_authorized_snapshot_field() -> None:
    steps = workflow("stable")["jobs"]["launch-dev"]["steps"]
    launch = steps[0]
    for key in ("pr_number", "head_sha", "base_sha", "tested_sha", "source_tree"):
        assert launch["env"][key.upper()] == "${{ needs.authorize.outputs." + key + " }}"
        assert f"{key}: ${key}" in launch["run"]
    assert steps[1]["with"]["name"] == "premerge-dev-snapshot-${{ github.run_attempt }}"
    assert 'ref: "main"' in steps[2]["run"]
    assert "internal-ci-dev.yml/dispatches" in steps[2]["run"]
    assert "sleep" not in steps[2]["run"]
    download = workflow("dev")["jobs"]["authorize"]["steps"][1]
    assert download["with"]["run-id"] == "${{ inputs.bridge_run_id }}"
    assert download["with"]["name"] == "premerge-dev-snapshot-${{ inputs.bridge_run_attempt }}"


@pytest.mark.parametrize(
    "event,branch,path,attempt,role,accepted",
    [
        (
            "pull_request_target",
            "contributor-branch",
            "internal-ci-bridge.yml",
            1,
            "maintain",
            True,
        ),
        ("workflow_dispatch", "main", "internal-ci-bridge.yml", 1, "admin", True),
        ("workflow_dispatch", "topic", "internal-ci-bridge.yml", 1, "admin", False),
        ("pull_request", "main", "internal-ci-bridge.yml", 1, "admin", False),
        ("pull_request_target", "topic", "other.yml", 1, "admin", False),
        ("workflow_dispatch", "main", "internal-ci-bridge.yml", 2, "admin", False),
        ("workflow_dispatch", "main", "internal-ci-bridge.yml", 1, "write", False),
        ("workflow_dispatch", "main", "internal-ci-bridge.yml", 1, "none", False),
    ],
)
def test_dev_requires_an_authorized_bridge_run(
    tmp_path: Path, event: str, branch: str, path: str, attempt: int, role: str, accepted: bool
) -> None:
    fake = tmp_path / "gh"
    fake.write_text(
        '#!/bin/bash\ncase "$*" in\n'
        '  *collaborators*) printf "%s\\n" "$ROLE" ;;\n'
        '  *) printf "%s\\n" "$RUN_JSON" ;;\nesac\n'
    )
    fake.chmod(0o755)
    script = workflow("dev")["jobs"]["authorize"]["steps"][0]["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "BRIDGE_RUN_ID": "42",
            "BRIDGE_RUN_ATTEMPT": "1",
            "GITHUB_REPOSITORY": "example/source",
            "ROLE": role,
            "RUN_JSON": json.dumps(
                {
                    "id": 42,
                    "run_attempt": attempt,
                    "event": event,
                    "path": f".github/workflows/{path}",
                    "head_branch": branch,
                    "actor": {"login": "maintainer"},
                }
            ),
        },
    )
    assert (result.returncode == 0) is accepted, result.stderr

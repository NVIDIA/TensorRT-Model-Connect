# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the real alert shell with fake GitHub and Slack boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/community-activity-slack-alert.yml"
)
INTERNAL_GATE = "TRTMC Internal CI / Automated premerge gate"


def scenario() -> dict:
    return {
        "pull": {
            "number": 1190,
            "state": "open",
            "merged": False,
            "draft": False,
            "mergeable": True,
            "mergeable_state": "clean",
            "head": {"sha": "a" * 40},
            "base": {"ref": "main", "sha": "b" * 40},
            "title": "A model <test> & <!here>",
            "user": {"login": "contributor"},
            "author_association": "CONTRIBUTOR",
            "html_url": "https://github.com/NVIDIA/TensorRT-Model-Connect/pull/1190",
        },
        "statuses": [{"id": 10, "context": INTERNAL_GATE, "state": "success"}],
        "checks": [
            {"id": index, "name": name, "status": "completed", "conclusion": "success"}
            for index, name in enumerate(
                ("Community CPU / Required", "PR Metadata / Required", "DCO"), 1
            )
        ],
    }


FAKE_BOUNDARY = r"""
import json, os, pathlib, sys
root = pathlib.Path(os.environ["FAKE_ROOT"])
data = json.loads((root / "scenario.json").read_text())
args = sys.argv[1:]
log = root / "calls.jsonl"
with log.open("a") as stream:
    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *args]) + "\n")
if pathlib.Path(sys.argv[0]).name == "curl":
    payload = json.loads(args[args.index("--data") + 1])
    with (root / "payloads.jsonl").open("a") as stream:
        stream.write(json.dumps(payload) + "\n")
    sys.exit(22 if data.get("slack_failure") else 0)
endpoint = next(arg for arg in args if arg.startswith("repos/"))
counter = root / "counts.json"
counts = json.loads(counter.read_text()) if counter.exists() else {}
counts[endpoint] = counts.get(endpoint, 0) + 1
counter.write_text(json.dumps(counts))
markers_path = root / "markers.json"
markers = json.loads(markers_path.read_text()) if markers_path.exists() else []
if "--method" in args and args[args.index("--method") + 1] == "POST":
    assert endpoint.endswith("/statuses/" + data["pull"]["head"]["sha"])
    assert (root / "payloads.jsonl").exists(), "Delivery must precede the marker"
    fields = dict(arg.split("=", 1) for arg in args if "=" in arg)
    markers.append({"head": endpoint.rsplit("/", 1)[1], "id": 1000 + len(markers), **fields})
    markers_path.write_text(json.dumps(markers))
elif "/pulls?" in endpoint:
    print(data["pull"]["number"])
elif "/pulls/" in endpoint:
    print(json.dumps(data.get("pull_after", data["pull"]) if counts[endpoint] > 1 else data["pull"]))
elif "/statuses?" in endpoint:
    values = data.get("statuses_after", data["statuses"]) if counts[endpoint] > 1 else data["statuses"]
    head = endpoint.split("/commits/")[1].split("/")[0]
    print(json.dumps(values + [m for m in markers if m["head"] == head]))
    if data.get("partial_status_error"):
        sys.exit(1)
elif "/check-runs?" in endpoint:
    # Separate page objects exercise gh --paginate aggregation.
    for check in data["checks"]:
        print(json.dumps({"check_runs": [check]}))
else:
    raise AssertionError(endpoint)
"""


def run_alert(tmp_path: Path, data: dict, *, dry_run: bool = False) -> subprocess.CompletedProcess:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    for name in ("gh", "curl"):
        path = fake_bin / name
        path.write_text(f"#!{sys.executable}\n{FAKE_BOUNDARY}")
        path.chmod(0o755)
    (tmp_path / "scenario.json").write_text(json.dumps(data))
    (tmp_path / "counts.json").write_text("{}")
    step = yaml.safe_load(WORKFLOW.read_text())["jobs"]["notify-merge-ready-pr"]["steps"][0]
    return subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_ROOT": str(tmp_path),
            "GH_TOKEN": "fake-token",
            "SLACK_WEBHOOK_URL": "" if dry_run else "https://hooks.slack.test/merge-ready",
            "REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
            "DRY_RUN": str(dry_run).lower(),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_merge_ready_alert_posts_green_pass_for_external_contributor(tmp_path: Path) -> None:
    data = scenario()
    result = run_alert(tmp_path, data)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "payloads.jsonl").read_text())
    assert payload["text"].startswith("✅ PASS · PR #1190 ready to merge")
    assert "&lt;!here&gt;" in json.dumps(payload)
    assert "<!here>" not in json.dumps(payload)
    assert "Internal CI" in json.dumps(payload)
    assert json.loads((tmp_path / "markers.json").read_text())[0]["state"] == "success"


@pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
def test_repository_members_do_not_receive_merge_ready_alerts(
    tmp_path: Path, association: str
) -> None:
    data = scenario()
    data["pull"]["author_association"] = association
    result = run_alert(tmp_path, data)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "payloads.jsonl").exists()


@pytest.mark.parametrize("state", ["pending", "failure", "error", "missing"])
def test_internal_ci_must_pass_on_the_current_head(tmp_path: Path, state: str) -> None:
    data = scenario()
    data["statuses"] = (
        []
        if state == "missing"
        else [{"id": 11, "context": INTERNAL_GATE, "state": state}, *data["statuses"]]
    )
    result = run_alert(tmp_path, data)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "payloads.jsonl").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("draft", True),
        ("merged", True),
        ("state", "closed"),
        ("mergeable", None),
        ("mergeable", False),
        ("mergeable_state", "blocked"),
        ("mergeable_state", "dirty"),
    ],
)
def test_unmergeable_prs_do_not_notify(tmp_path: Path, field: str, value: object) -> None:
    data = scenario()
    data["pull"][field] = value
    assert run_alert(tmp_path, data).returncode == 0
    assert not (tmp_path / "payloads.jsonl").exists()


@pytest.mark.parametrize("pending", [False, True])
def test_newer_failed_or_pending_public_check_overrides_old_success(
    tmp_path: Path, pending: bool
) -> None:
    data = scenario()
    data["checks"].append(
        {
            "id": 100,
            "name": "DCO",
            "status": "in_progress" if pending else "completed",
            "conclusion": None if pending else "failure",
        }
    )
    assert run_alert(tmp_path, data).returncode == 0
    assert not (tmp_path / "payloads.jsonl").exists()


@pytest.mark.parametrize("change", ["head", "base", "closed", "gate"])
def test_readiness_is_rechecked_before_delivery(tmp_path: Path, change: str) -> None:
    data = scenario()
    data["pull_after"] = json.loads(json.dumps(data["pull"]))
    if change in ("head", "base"):
        data["pull_after"][change]["sha"] = "c" * 40
    elif change == "closed":
        data["pull_after"]["state"] = "closed"
    else:
        data["statuses_after"] = [{"id": 12, "context": INTERNAL_GATE, "state": "pending"}]
    assert run_alert(tmp_path, data).returncode == 0
    assert not (tmp_path / "payloads.jsonl").exists()


def test_delivery_is_deduplicated_per_pr_head(tmp_path: Path) -> None:
    data = scenario()
    assert run_alert(tmp_path, data).returncode == 0
    assert run_alert(tmp_path, data).returncode == 0
    assert len((tmp_path / "payloads.jsonl").read_text().splitlines()) == 1
    data["pull"]["head"]["sha"] = "c" * 40
    assert run_alert(tmp_path, data).returncode == 0
    assert len((tmp_path / "payloads.jsonl").read_text().splitlines()) == 2


def test_failed_slack_delivery_does_not_record_success(tmp_path: Path) -> None:
    data = scenario()
    data["slack_failure"] = True
    assert run_alert(tmp_path, data).returncode != 0
    assert not (tmp_path / "markers.json").exists()


def test_partial_github_response_cannot_authorize_an_alert(tmp_path: Path) -> None:
    data = scenario()
    data["partial_status_error"] = True
    assert run_alert(tmp_path, data).returncode == 0
    assert not (tmp_path / "payloads.jsonl").exists()


def test_dry_run_previews_without_webhook_or_status_writes(tmp_path: Path) -> None:
    result = run_alert(tmp_path, scenario(), dry_run=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["text"].startswith("✅ PASS")
    assert not (tmp_path / "payloads.jsonl").exists()
    assert not (tmp_path / "markers.json").exists()


def test_workflow_runs_only_trusted_metadata_and_serializes_delivery() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    events = workflow.get("on", workflow.get(True))
    assert set(events) == {
        "issues",
        "issue_comment",
        "discussion",
        "discussion_comment",
        "workflow_run",
        "schedule",
        "workflow_dispatch",
    }
    assert "TensorRT-Model-Connect Internal CI Bridge" in events["workflow_run"]["workflows"]
    job = workflow["jobs"]["notify-merge-ready-pr"]
    assert workflow["permissions"] == {}
    assert job["permissions"] == {"checks": "read", "pull-requests": "read", "statuses": "write"}
    assert job["concurrency"]["cancel-in-progress"] is False
    assert "github.repository == 'NVIDIA/TensorRT-Model-Connect'" in job["if"]
    assert all("uses" not in step for step in job["steps"])
    assert "${{" not in job["steps"][0]["run"]
    assert "git checkout" not in WORKFLOW.read_text()
    assert "gh run download" not in WORKFLOW.read_text()

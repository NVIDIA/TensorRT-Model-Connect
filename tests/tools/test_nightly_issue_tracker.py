# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scheduled Nightly failure issue state machine."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tools.nightly_issue_tracker import (
    EXPECTED_REF,
    EXPECTED_REPOSITORY,
    EXPECTED_WORKFLOW_REF,
    EXPECTED_API_URL,
    EXPECTED_SERVER_URL,
    FAILURE_TITLE,
    RECOVERY_TITLE,
    TRACKER_MARKER,
    GitHubApi,
    NightlyRun,
    TrackerError,
    parse_args,
    reconcile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "nightly_issue_tracker.py"
BASE_ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_API_URL": EXPECTED_API_URL,
    "GITHUB_EVENT_NAME": "schedule",
    "GITHUB_REF": EXPECTED_REF,
    "GITHUB_REPOSITORY": EXPECTED_REPOSITORY,
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_RUN_ID": "777",
    "GITHUB_SERVER_URL": EXPECTED_SERVER_URL,
    "GITHUB_SHA": "a" * 40,
    "GITHUB_WORKFLOW": "TensorRT-Model-Connect Nightly CI",
    "GITHUB_WORKFLOW_REF": EXPECTED_WORKFLOW_REF,
    "NIGHTLY_RELEASE_RESULT": "skipped",
    "NIGHTLY_REQUIRED_RESULT": "failure",
}


class FakeApi:
    """In-memory API that records every read and write operation."""

    def __init__(
        self,
        *,
        issues: list[dict[str, object]] | None = None,
        jobs: list[dict[str, object]] | None = None,
        artifacts: list[dict[str, object]] | None = None,
    ) -> None:
        self.issues = list(issues or [])
        self.jobs = list(jobs or [])
        self.artifacts = list(artifacts or [])
        self.calls: list[tuple[object, ...]] = []

    def list_open_issues(self) -> list[dict[str, object]]:
        self.calls.append(("list_open_issues",))
        return self.issues

    def list_run_jobs(self, run_id: int, run_attempt: int) -> list[dict[str, object]]:
        self.calls.append(("list_run_jobs", run_id, run_attempt))
        return self.jobs

    def list_run_artifacts(self, run_id: int) -> list[dict[str, object]]:
        self.calls.append(("list_run_artifacts", run_id))
        return self.artifacts

    def create_issue(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_issue", payload))
        return {
            **payload,
            "number": 101,
            "state": "open",
            "html_url": "https://github.com/NVIDIA/TensorRT-Model-Connect/issues/101",
        }

    def update_issue(self, issue_number: int, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("update_issue", issue_number, payload))
        return {
            **payload,
            "number": issue_number,
            "html_url": (f"https://github.com/NVIDIA/TensorRT-Model-Connect/issues/{issue_number}"),
        }

    @property
    def mutations(self) -> list[tuple[object, ...]]:
        return [call for call in self.calls if call[0] in {"create_issue", "update_issue"}]


def _run(**overrides: str) -> NightlyRun:
    return NightlyRun.from_environ({**BASE_ENV, **overrides})


def _tracker_issue(
    *, number: int = 55, body: str = TRACKER_MARKER, state: str = "open"
) -> dict[str, object]:
    return {
        "number": number,
        "state": state,
        "body": body,
        "html_url": (f"https://github.com/NVIDIA/TensorRT-Model-Connect/issues/{number}"),
    }


@pytest.mark.parametrize(
    "field,value",
    (
        ("github_actions", "false"),
        ("repository", "someone/TensorRT-Model-Connect"),
        ("event_name", "workflow_dispatch"),
        ("ref", "refs/heads/feature"),
        (
            "workflow_ref",
            "NVIDIA/TensorRT-Model-Connect/.github/workflows/other.yml@refs/heads/main",
        ),
        ("server_url", "https://github.example.com"),
        ("api_url", "https://api.github.example.com"),
    ),
)
def test_non_production_contexts_never_call_github(field: str, value: str) -> None:
    api = FakeApi()
    result = reconcile(replace(_run(), **{field: value}), api, apply=True)

    assert result.action == "disabled-non-production-context"
    assert api.calls == []


def test_default_mode_plans_a_failure_without_mutating() -> None:
    api = FakeApi(
        jobs=[
            {
                "name": "4 / Model / qwen",
                "conclusion": "failure",
                "html_url": "https://github.com/NVIDIA/TensorRT-Model-Connect/"
                "actions/runs/777/job/9",
            }
        ]
    )

    result = reconcile(_run(), api, apply=False)

    assert result.action == "would-create"
    assert api.mutations == []
    assert parse_args([]).apply is False
    assert parse_args(["--apply"]).apply is True


def test_first_failure_creates_one_bug_with_current_attempt_evidence() -> None:
    run = _run()
    api = FakeApi(
        issues=[
            {
                "number": 4,
                "state": "open",
                "body": TRACKER_MARKER,
                "pull_request": {"url": "https://api.github.com/pulls/4"},
            }
        ],
        jobs=[
            {
                "name": "4 / Model / qwen",
                "conclusion": "failure",
                "html_url": f"{run.run_url}/job/9",
                "steps": [{"name": "Build + reference test", "conclusion": "failure"}],
            },
            {
                "name": "4 / Model / llama",
                "conclusion": "success",
                "html_url": f"{run.run_url}/job/10",
            },
            {
                "name": "7 / Nightly CI",
                "conclusion": "cancelled",
                "html_url": f"{run.run_url}/job/11",
            },
        ],
        artifacts=[
            {
                "id": 700,
                "name": "trtmc-nightly-html-report-777-1",
            },
            {
                "id": 701,
                "name": "trtmc-nightly-html-report-777-2",
                "expired": False,
                "size_in_bytes": 12345,
                "digest": "sha256:abc123",
            },
        ],
    )

    result = reconcile(run, api, apply=True)

    assert result.action == "created"
    assert ("list_run_jobs", 777, 2) in api.calls
    assert len(api.mutations) == 1
    operation, payload = api.mutations[0]
    assert operation == "create_issue"
    assert payload["title"] == FAILURE_TITLE
    assert payload["labels"] == ["bug"]
    body = str(payload["body"])
    assert TRACKER_MARKER in body
    assert run.run_marker in body
    assert run.run_url in body and run.sha in body
    assert "4 / Model / qwen" in body
    assert "Build + reference test" in body
    assert "7 / Nightly CI" in body
    assert "4 / Model / llama" not in body
    assert "trtmc-nightly-html-report-777-2" in body
    assert "sha256:abc123" in body
    assert "`12345` bytes" in body
    assert "trtmc-nightly-html-report-777-1" not in body


def test_later_failure_updates_the_open_tracker_and_same_attempt_is_idempotent() -> None:
    run = _run(GITHUB_RUN_ID="778", GITHUB_RUN_ATTEMPT="1")
    tracker = _tracker_issue(body=f"{TRACKER_MARKER}\n<!-- old run -->")
    api = FakeApi(issues=[tracker])

    result = reconcile(run, api, apply=True)

    assert result.action == "updated"
    assert len(api.mutations) == 1
    operation, issue_number, payload = api.mutations[0]
    assert operation == "update_issue"
    assert issue_number == 55
    assert payload["state"] == "open"
    assert run.run_marker in str(payload["body"])

    reconciled = _tracker_issue(body=str(payload["body"]))
    second_api = FakeApi(issues=[reconciled])
    second = reconcile(run, second_api, apply=True)
    assert second.action == "already-reconciled"
    assert second_api.calls == [("list_open_issues",)]
    assert second_api.mutations == []


def test_success_closes_one_open_tracker_without_erasing_failure_evidence() -> None:
    run = _run(
        NIGHTLY_REQUIRED_RESULT="success",
        NIGHTLY_RELEASE_RESULT="success",
    )
    api = FakeApi(issues=[_tracker_issue(body=f"{TRACKER_MARKER}\nlast failure evidence")])

    result = reconcile(run, api, apply=True)

    assert result.action == "closed"
    operation, issue_number, payload = api.mutations[0]
    assert operation == "update_issue" and issue_number == 55
    assert payload["title"] == RECOVERY_TITLE
    assert payload["state"] == "closed"
    assert payload["state_reason"] == "completed"
    assert run.run_marker in str(payload["body"])
    assert "last failure evidence" in str(payload["body"])


def test_success_without_an_open_tracker_is_a_noop() -> None:
    run = _run(
        NIGHTLY_REQUIRED_RESULT="success",
        NIGHTLY_RELEASE_RESULT="success",
    )

    empty_api = FakeApi()
    empty = reconcile(run, empty_api, apply=True)
    assert empty.action == "no-open-tracker"
    assert empty_api.mutations == []


def test_multiple_open_trackers_fail_without_an_additional_mutation() -> None:
    api = FakeApi(issues=[_tracker_issue(number=4), _tracker_issue(number=5)])

    with pytest.raises(TrackerError, match="multiple open Nightly tracker issues"):
        reconcile(_run(), api, apply=True)

    assert api.mutations == []


def test_rest_client_refuses_mutation_without_explicit_apply() -> None:
    api = GitHubApi(
        api_url="https://api.github.com",
        repository=EXPECTED_REPOSITORY,
        token="not-a-real-token",
        allow_mutations=False,
    )

    with pytest.raises(TrackerError, match="without --apply"):
        api.create_issue({"title": "must not be sent"})


def test_rest_client_refuses_to_send_token_to_another_host_or_repository() -> None:
    with pytest.raises(TrackerError, match="non-GitHub API URL"):
        GitHubApi(
            api_url="https://api.github.example.com",
            repository=EXPECTED_REPOSITORY,
            token="not-a-real-token",
            allow_mutations=True,
        )
    with pytest.raises(TrackerError, match="outside the official repository"):
        GitHubApi(
            api_url=EXPECTED_API_URL,
            repository="someone/TensorRT-Model-Connect",
            token="not-a-real-token",
            allow_mutations=True,
        )


def test_rest_client_reads_jobs_from_only_the_current_attempt() -> None:
    class RecordingApi(GitHubApi):
        def __init__(self) -> None:
            super().__init__(
                api_url=EXPECTED_API_URL,
                repository=EXPECTED_REPOSITORY,
                token="not-a-real-token",
                allow_mutations=False,
            )
            self.paths: list[str] = []

        def _request(
            self,
            method: str,
            path: str,
            payload: dict[str, object] | None = None,
        ) -> object:
            assert method == "GET" and payload is None
            self.paths.append(path)
            return {"jobs": []}

    api = RecordingApi()

    assert api.list_run_jobs(777, 2) == []
    assert api.paths == [
        "/repos/NVIDIA/TensorRT-Model-Connect/actions/runs/777/attempts/2/jobs?per_page=100&page=1"
    ]


def test_cli_manual_branch_guard_exits_before_token_or_network() -> None:
    env = os.environ.copy()
    env.update(
        {
            **BASE_ENV,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/agent/nightly-issue-automation",
            "GITHUB_WORKFLOW_REF": (
                "NVIDIA/TensorRT-Model-Connect/.github/workflows/nightly.yml@"
                "refs/heads/agent/nightly-issue-automation"
            ),
        }
    )
    env.pop("GITHUB_TOKEN", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--apply"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"action": "disabled-non-production-context"' in result.stdout

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create one GitHub issue per scheduled Nightly failure streak."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_REPOSITORY = "NVIDIA/TensorRT-Model-Connect"
EXPECTED_REF = "refs/heads/main"
EXPECTED_WORKFLOW_REF = (
    "NVIDIA/TensorRT-Model-Connect/.github/workflows/nightly.yml@refs/heads/main"
)
EXPECTED_SERVER_URL = "https://github.com"
EXPECTED_API_URL = "https://api.github.com"
TRACKER_MARKER = "<!-- trtmc-nightly-failure-tracker:v1 -->"
FAILURE_TITLE = "[Nightly] Scheduled validation needs attention"
RECOVERY_TITLE = "[Nightly] Scheduled validation recovered"
FAILURE_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "stale",
    "startup_failure",
    "timed_out",
}
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


class TrackerError(RuntimeError):
    """Raised when tracker inputs or GitHub responses are unsafe or ambiguous."""


@dataclass(frozen=True)
class NightlyRun:
    """Immutable GitHub Actions context used for issue reconciliation."""

    repository: str
    run_id: int
    run_attempt: int
    sha: str
    ref: str
    event_name: str
    workflow_name: str
    workflow_ref: str
    github_actions: str
    server_url: str
    api_url: str
    required_result: str
    release_result: str

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> NightlyRun:
        """Build and validate a run context from GitHub's default variables."""

        def required(name: str) -> str:
            value = environ.get(name, "").strip()
            if not value:
                raise TrackerError(f"required environment variable {name} is empty")
            if any(character in value for character in "\r\n"):
                raise TrackerError(f"environment variable {name} contains a newline")
            return value

        repository = required("GITHUB_REPOSITORY")
        if _REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise TrackerError(f"invalid GITHUB_REPOSITORY: {repository!r}")
        sha = required("GITHUB_SHA")
        if _SHA_PATTERN.fullmatch(sha) is None:
            raise TrackerError("GITHUB_SHA must be a 40-character hexadecimal commit")
        try:
            run_id = int(required("GITHUB_RUN_ID"))
            run_attempt = int(required("GITHUB_RUN_ATTEMPT"))
        except ValueError as error:
            raise TrackerError("GitHub run ID and attempt must be integers") from error
        if run_id < 1 or run_attempt < 1:
            raise TrackerError("GitHub run ID and attempt must be positive")
        server_url = required("GITHUB_SERVER_URL").rstrip("/")
        api_url = required("GITHUB_API_URL").rstrip("/")
        if not server_url.startswith("https://") or not api_url.startswith("https://"):
            raise TrackerError("GitHub server and API URLs must use HTTPS")
        return cls(
            repository=repository,
            run_id=run_id,
            run_attempt=run_attempt,
            sha=sha.lower(),
            ref=required("GITHUB_REF"),
            event_name=required("GITHUB_EVENT_NAME"),
            workflow_name=required("GITHUB_WORKFLOW"),
            workflow_ref=required("GITHUB_WORKFLOW_REF"),
            github_actions=required("GITHUB_ACTIONS").lower(),
            server_url=server_url,
            api_url=api_url,
            required_result=required("NIGHTLY_REQUIRED_RESULT").lower(),
            release_result=required("NIGHTLY_RELEASE_RESULT").lower(),
        )

    @property
    def issue_writes_allowed(self) -> bool:
        """Return true only for the official scheduled workflow on main."""

        return (
            self.github_actions == "true"
            and self.repository == EXPECTED_REPOSITORY
            and self.event_name == "schedule"
            and self.ref == EXPECTED_REF
            and self.workflow_ref == EXPECTED_WORKFLOW_REF
            and self.server_url == EXPECTED_SERVER_URL
            and self.api_url == EXPECTED_API_URL
        )

    @property
    def successful(self) -> bool:
        """Return true only when validation and publication both succeeded."""

        return self.required_result == "success" and self.release_result == "success"

    @property
    def run_url(self) -> str:
        return f"{self.server_url}/{self.repository}/actions/runs/{self.run_id}"

    @property
    def commit_url(self) -> str:
        return f"{self.server_url}/{self.repository}/commit/{self.sha}"

    @property
    def run_marker(self) -> str:
        return f"<!-- trtmc-nightly-run:{self.run_id}:{self.run_attempt} -->"


class TrackerApi(Protocol):
    """Small GitHub API boundary used by the reconciler and in-memory tests."""

    def list_open_issues(self) -> list[dict[str, object]]: ...

    def list_run_jobs(self, run_id: int, run_attempt: int) -> list[dict[str, object]]: ...

    def list_run_artifacts(self, run_id: int) -> list[dict[str, object]]: ...

    def create_issue(self, payload: dict[str, object]) -> dict[str, object]: ...

    def update_issue(self, issue_number: int, payload: dict[str, object]) -> dict[str, object]: ...


class GitHubApi:
    """Minimal versioned REST client with mutation disabled unless requested."""

    def __init__(
        self,
        *,
        api_url: str,
        repository: str,
        token: str,
        allow_mutations: bool,
    ) -> None:
        if not token or any(character in token for character in "\r\n"):
            raise TrackerError("GITHUB_TOKEN must be a non-empty single-line token")
        if api_url.rstrip("/") != EXPECTED_API_URL:
            raise TrackerError("refusing to send GITHUB_TOKEN to a non-GitHub API URL")
        if repository != EXPECTED_REPOSITORY:
            raise TrackerError("refusing to mutate issues outside the official repository")
        self.api_url = api_url.rstrip("/")
        self.repository = repository
        self.token = token
        self.allow_mutations = allow_mutations

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "trtmc-nightly-issue-tracker",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.api_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                body = response.read()
        except HTTPError as error:
            details = error.read(2_000).decode("utf-8", errors="replace")
            raise TrackerError(
                f"GitHub API {method} {path} returned {error.code}: {details}"
            ) from error
        except URLError as error:
            raise TrackerError(f"GitHub API {method} {path} failed: {error.reason}") from error
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise TrackerError(f"GitHub API {method} {path} returned invalid JSON") from error

    def _paginate(self, path: str, *, field: str | None = None) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        page = 1
        while True:
            separator = "&" if "?" in path else "?"
            response = self._request("GET", f"{path}{separator}per_page=100&page={page}")
            page_items: object
            if field is None:
                page_items = response
            elif isinstance(response, dict):
                page_items = response.get(field)
            else:
                page_items = None
            if not isinstance(page_items, list) or not all(
                isinstance(item, dict) for item in page_items
            ):
                raise TrackerError(f"GitHub API pagination payload is invalid for {path}")
            items.extend(page_items)
            if len(page_items) < 100:
                return items
            page += 1

    def list_open_issues(self) -> list[dict[str, object]]:
        return self._paginate(
            f"/repos/{self.repository}/issues?state=open&sort=created&direction=desc"
        )

    def list_run_jobs(self, run_id: int, run_attempt: int) -> list[dict[str, object]]:
        return self._paginate(
            f"/repos/{self.repository}/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
            field="jobs",
        )

    def list_run_artifacts(self, run_id: int) -> list[dict[str, object]]:
        return self._paginate(
            f"/repos/{self.repository}/actions/runs/{run_id}/artifacts",
            field="artifacts",
        )

    def _mutate(self, method: str, path: str, payload: dict[str, object]) -> dict[str, object]:
        if not self.allow_mutations:
            raise TrackerError("GitHub issue mutation was attempted without --apply")
        response = self._request(method, path, payload)
        if not isinstance(response, dict):
            raise TrackerError(f"GitHub API {method} {path} returned no issue object")
        return response

    def create_issue(self, payload: dict[str, object]) -> dict[str, object]:
        return self._mutate("POST", f"/repos/{self.repository}/issues", payload)

    def update_issue(self, issue_number: int, payload: dict[str, object]) -> dict[str, object]:
        return self._mutate("PATCH", f"/repos/{self.repository}/issues/{issue_number}", payload)


@dataclass(frozen=True)
class ReconcileResult:
    """Machine-readable description of the performed or planned action."""

    action: str
    issue_number: int | None = None
    issue_url: str | None = None

    def as_json(self) -> str:
        return json.dumps(
            {
                "action": self.action,
                "issue_number": self.issue_number,
                "issue_url": self.issue_url,
            },
            sort_keys=True,
        )


def _single_line(value: object, *, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    return text[:300] if text else fallback


def _code(value: object) -> str:
    return _single_line(value, fallback="unknown").replace("`", "'")


def _issue_number(issue: Mapping[str, object]) -> int:
    number = issue.get("number")
    if not isinstance(number, int) or number < 1:
        raise TrackerError("GitHub issue response has no positive issue number")
    return number


def _issue_url(issue: Mapping[str, object]) -> str | None:
    value = issue.get("html_url")
    return value if isinstance(value, str) and value.startswith("https://") else None


def _find_tracker(issues: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    matches = [
        issue
        for issue in issues
        if issue.get("state") == "open"
        and "pull_request" not in issue
        and TRACKER_MARKER in str(issue.get("body") or "")
    ]
    if len(matches) > 1:
        numbers = ", ".join(str(_issue_number(issue)) for issue in matches)
        raise TrackerError(f"multiple open Nightly tracker issues are ambiguous: {numbers}")
    return matches[0] if matches else None


def _failing_job_lines(run: NightlyRun, jobs: Sequence[Mapping[str, object]]) -> list[str]:
    failures: list[tuple[str, str, str, str]] = []
    for job in jobs:
        conclusion = str(job.get("conclusion") or "").lower()
        if conclusion not in FAILURE_CONCLUSIONS:
            continue
        name = _single_line(job.get("name"), fallback="Unnamed job")
        url = str(job.get("html_url") or "")
        if not url.startswith(f"{run.run_url}/job/"):
            url = run.run_url
        raw_steps = job.get("steps")
        failed_steps = []
        if isinstance(raw_steps, list):
            failed_steps = [
                _single_line(step.get("name"), fallback="Unnamed step")
                for step in raw_steps
                if isinstance(step, dict)
                and str(step.get("conclusion") or "").lower() in FAILURE_CONCLUSIONS
            ]
        step_text = ", ".join(failed_steps[:5])
        failures.append((name, conclusion, url, step_text))
    return [
        f"- [{name}]({url}) — `{conclusion}`"
        + (f"; failing step: `{_code(step_text)}`" if step_text else "")
        for name, conclusion, url, step_text in sorted(failures)
    ] or [f"- No terminal failed job was returned; inspect the [full run]({run.run_url})."]


def _artifact_lines(run: NightlyRun, artifacts: Sequence[Mapping[str, object]]) -> list[str]:
    expected_name = f"trtmc-nightly-html-report-{run.run_id}-{run.run_attempt}"
    reports: list[tuple[str, str, str, str, str]] = []
    for artifact in artifacts:
        name = str(artifact.get("name") or "")
        artifact_id = artifact.get("id")
        if name != expected_name or not isinstance(artifact_id, int):
            continue
        expired = str(bool(artifact.get("expired"))).lower()
        size = artifact.get("size_in_bytes")
        size_text = str(size) if isinstance(size, int) and size >= 0 else "unknown"
        digest = _single_line(artifact.get("digest"), fallback="unavailable")
        reports.append(
            (
                _single_line(name, fallback="Nightly report"),
                str(artifact_id),
                expired,
                size_text,
                digest,
            )
        )
    return [
        f"- `{name}` (artifact `{artifact_id}`, `{size}` bytes, digest `{_code(digest)}`, "
        f"expired `{expired}`; [open run artifacts]({run.run_url}#artifacts))"
        for name, artifact_id, expired, size, digest in sorted(reports)
    ] or [f"- No combined report artifact is available; inspect the [run]({run.run_url})."]


def render_failure_body(
    run: NightlyRun,
    jobs: Sequence[Mapping[str, object]],
    artifacts: Sequence[Mapping[str, object]],
) -> str:
    """Render the latest failure evidence for the single open tracker."""

    job_lines = "\n".join(_failing_job_lines(run, jobs))
    artifact_lines = "\n".join(_artifact_lines(run, artifacts))
    return f"""{TRACKER_MARKER}
{run.run_marker}

## Scheduled Nightly failure

**Action required:** investigate the scheduled TensorRT-Model-Connect Nightly and
fix the failing validation or publication stage.

- Workflow: `{_code(run.workflow_name)}`
- Run: [{run.run_id} attempt {run.run_attempt}]({run.run_url})
- Commit: [`{run.sha}`]({run.commit_url})
- Required validation gate: `{_code(run.required_result)}`
- Nightly wheel publication: `{_code(run.release_result)}`

### Failing jobs

{job_lines}

### Diagnostic report

{artifact_lines}

This issue is maintained automatically. Later failures in the same streak update this
body; the next fully successful scheduled Nightly closes the issue.
"""


def render_recovery_body(run: NightlyRun, previous_body: str) -> str:
    """Render recovery evidence before atomically closing the tracker."""

    return f"""{previous_body.rstrip()}

---

{run.run_marker}

## Scheduled Nightly recovered

The scheduled TensorRT-Model-Connect Nightly is green again.

- Workflow: `{_code(run.workflow_name)}`
- Recovery run: [{run.run_id} attempt {run.run_attempt}]({run.run_url})
- Commit: [`{run.sha}`]({run.commit_url})
- Required validation gate: `{_code(run.required_result)}`
- Nightly wheel publication: `{_code(run.release_result)}`

This tracker was closed automatically after both validation and publication succeeded.
"""


def _result(action: str, issue: Mapping[str, object] | None = None) -> ReconcileResult:
    if issue is None:
        return ReconcileResult(action=action)
    return ReconcileResult(
        action=action,
        issue_number=_issue_number(issue),
        issue_url=_issue_url(issue),
    )


def reconcile(run: NightlyRun, api: TrackerApi, *, apply: bool) -> ReconcileResult:
    """Create, update, close, or plan the one issue for this failure streak."""

    if not run.issue_writes_allowed:
        return _result("disabled-non-production-context")

    tracker = _find_tracker(api.list_open_issues())
    if run.successful:
        if tracker is None:
            return _result("no-open-tracker")
        payload: dict[str, object] = {
            "title": RECOVERY_TITLE,
            "body": render_recovery_body(run, str(tracker.get("body") or "")),
            "state": "closed",
            "state_reason": "completed",
        }
        if not apply:
            return _result("would-close", tracker)
        return _result("closed", api.update_issue(_issue_number(tracker), payload))

    if tracker is not None and run.run_marker in str(tracker.get("body") or ""):
        return _result("already-reconciled", tracker)

    jobs = api.list_run_jobs(run.run_id, run.run_attempt)
    artifacts = api.list_run_artifacts(run.run_id)
    body = render_failure_body(run, jobs, artifacts)
    if tracker is None:
        payload = {"title": FAILURE_TITLE, "body": body, "labels": ["bug"]}
        if not apply:
            return _result("would-create")
        return _result("created", api.create_issue(payload))

    payload = {"title": FAILURE_TITLE, "body": body, "state": "open"}
    if not apply:
        return _result("would-update", tracker)
    return _result("updated", api.update_issue(_issue_number(tracker), payload))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile the scheduled main-branch Nightly failure issue."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="allow issue creation/update/closure after all production guards pass",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run = NightlyRun.from_environ(os.environ)
        if not run.issue_writes_allowed:
            print(_result("disabled-non-production-context").as_json())
            return 0
        token = os.environ.get("GITHUB_TOKEN", "")
        api = GitHubApi(
            api_url=run.api_url,
            repository=run.repository,
            token=token,
            allow_mutations=args.apply,
        )
        result = reconcile(run, api, apply=args.apply)
    except TrackerError as error:
        prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") == "true" else "error: "
        print(f"{prefix}{error}", file=sys.stderr)
        return 1
    print(result.as_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

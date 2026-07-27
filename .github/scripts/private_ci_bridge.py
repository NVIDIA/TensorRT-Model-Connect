#!/usr/bin/env python3
"""Resolve and dispatch one exact pull-request snapshot to private CI.

This file is executed only from the protected base branch by the
``pull_request_target`` bridge. It must never import or execute pull-request
code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_ROOT = "https://api.github.com"
API_VERSION = "2022-11-28"
SOURCE_REPOSITORY = "NVIDIA/TensorRT-Model-Connect"
SOURCE_REPOSITORY_ID = 1216320259
SOURCE_BASE_BRANCH = "main"
CI_WORKFLOW = "premerge.yml"
CI_WORKFLOW_REF = "main"
CHECK_NAME = "TensorRT-Model-Connect CI"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PR_RE = re.compile(r"^[1-9][0-9]{0,9}$")
OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class BridgeError(RuntimeError):
    """A stable, non-sensitive bridge failure."""


def _required_token(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise BridgeError(f"{name.lower()}-missing")
    return value


def _private_ci_repository() -> str:
    owner = os.environ.get("PRIVATE_CI_OWNER", "")
    repository = os.environ.get("PRIVATE_CI_REPOSITORY", "")
    if (
        OWNER_RE.fullmatch(owner) is None
        or REPOSITORY_RE.fullmatch(repository) is None
    ):
        raise BridgeError("private-ci-target-invalid")
    return f"{owner}/{repository}"


def _request(
    token: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    expect_json: bool = True,
) -> Any:
    data = None
    if payload is not None:
        data = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "trtmc-private-ci-bridge",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        raise BridgeError("github-api-request-failed") from error
    if len(body) > MAX_RESPONSE_BYTES:
        raise BridgeError("github-api-response-too-large")
    if not expect_json:
        return None
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError("github-api-invalid-response") from error


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BridgeError(code)
    return value


def _sha(value: Any, code: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise BridgeError(code)
    return value


def _pr_number(value: Any) -> int:
    text = str(value)
    if PR_RE.fullmatch(text) is None:
        raise BridgeError("pr-number-invalid")
    return int(text)


def _ref_sha(token: str, ref: str) -> str:
    value = _mapping(
        _request(
            token,
            "GET",
            f"/repos/{SOURCE_REPOSITORY}/git/ref/{ref}",
        ),
        "github-ref-invalid",
    )
    return _sha(
        _mapping(value.get("object"), "github-ref-invalid").get("sha"),
        "github-ref-invalid",
    )


def _pull_snapshot(value: Any) -> dict[str, Any]:
    pull = _mapping(value, "pull-request-invalid")
    number = _pr_number(pull.get("number"))
    if pull.get("state") != "open" or pull.get("draft") is not False:
        raise BridgeError("pull-request-not-runnable")

    base = _mapping(pull.get("base"), "pull-request-base-invalid")
    base_repo = _mapping(base.get("repo"), "pull-request-base-invalid")
    if (
        base_repo.get("id") != SOURCE_REPOSITORY_ID
        or base_repo.get("full_name") != SOURCE_REPOSITORY
        or base.get("ref") != SOURCE_BASE_BRANCH
    ):
        raise BridgeError("pull-request-base-invalid")

    head = _mapping(pull.get("head"), "pull-request-head-invalid")
    return {
        "pr_number": str(number),
        "head_sha": _sha(head.get("sha"), "pull-request-head-invalid"),
        "base_sha": _sha(base.get("sha"), "pull-request-base-invalid"),
        "merge_sha": _sha(
            pull.get("merge_commit_sha"), "pull-request-merge-invalid"
        ),
    }


def _event_snapshot(event: dict[str, Any]) -> dict[str, Any] | None:
    pull = event.get("pull_request")
    if pull is None:
        return None
    repository = _mapping(event.get("repository"), "event-repository-invalid")
    if (
        repository.get("id") != SOURCE_REPOSITORY_ID
        or repository.get("full_name") != SOURCE_REPOSITORY
    ):
        raise BridgeError("event-repository-invalid")
    if event.get("action") != "labeled":
        raise BridgeError("event-action-invalid")
    label = _mapping(event.get("label"), "event-label-invalid")
    if label.get("name") != "run-ci":
        raise BridgeError("event-label-invalid")
    return _pull_snapshot(pull)


def _resolve_live_snapshot(token: str, pr_number: int) -> dict[str, Any]:
    first = _pull_snapshot(
        _request(
            token,
            "GET",
            f"/repos/{SOURCE_REPOSITORY}/pulls/{pr_number}",
        )
    )
    if first["pr_number"] != str(pr_number):
        raise BridgeError("pull-request-number-mismatch")

    if _ref_sha(token, f"heads/{SOURCE_BASE_BRANCH}") != first["base_sha"]:
        raise BridgeError("pull-request-base-stale")
    if _ref_sha(token, f"pull/{pr_number}/head") != first["head_sha"]:
        raise BridgeError("pull-request-head-stale")
    if _ref_sha(token, f"pull/{pr_number}/merge") != first["merge_sha"]:
        raise BridgeError("pull-request-merge-stale")

    merge = _mapping(
        _request(
            token,
            "GET",
            f"/repos/{SOURCE_REPOSITORY}/commits/{first['merge_sha']}",
        ),
        "merge-commit-invalid",
    )
    parents = merge.get("parents")
    if not isinstance(parents, list) or len(parents) != 2:
        raise BridgeError("merge-commit-invalid")
    parent_shas = [
        _sha(_mapping(parent, "merge-commit-invalid").get("sha"), "merge-commit-invalid")
        for parent in parents
    ]
    if parent_shas != [first["base_sha"], first["head_sha"]]:
        raise BridgeError("merge-parent-mismatch")

    second = _pull_snapshot(
        _request(
            token,
            "GET",
            f"/repos/{SOURCE_REPOSITORY}/pulls/{pr_number}",
        )
    )
    if second != first:
        raise BridgeError("pull-request-changed")
    return first


def _load_event(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise BridgeError("event-read-failed") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise BridgeError("event-too-large")
    try:
        return _mapping(json.loads(raw), "event-invalid")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BridgeError("event-invalid") from error


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def resolve(event_path: Path, requested_pr: str | None, output: Path) -> None:
    token = _required_token("GITHUB_TOKEN")
    event = _load_event(event_path)
    event_value = _event_snapshot(event)
    if event_value is not None:
        if requested_pr is not None:
            raise BridgeError("manual-pr-unexpected")
        pr_number = _pr_number(event_value["pr_number"])
    else:
        if requested_pr is None:
            raise BridgeError("manual-pr-missing")
        pr_number = _pr_number(requested_pr)

    live = _resolve_live_snapshot(token, pr_number)
    if event_value is not None and live != event_value:
        raise BridgeError("event-snapshot-stale")
    live.update(
        {
            "request_id": (
                f"trtmc-premerge:{live['pr_number']}:"
                f"{live['head_sha']}:{live['merge_sha']}"
            ),
            "source_repository": SOURCE_REPOSITORY,
            "source_repository_id": str(SOURCE_REPOSITORY_ID),
        }
    )
    _write_canonical(output, live)


def _load_request(path: Path) -> dict[str, str]:
    value = _load_event(path)
    expected = {
        "base_sha",
        "head_sha",
        "merge_sha",
        "pr_number",
        "request_id",
        "source_repository",
        "source_repository_id",
    }
    if set(value) != expected or any(
        not isinstance(item, str) for item in value.values()
    ):
        raise BridgeError("request-invalid")
    if (
        value["source_repository"] != SOURCE_REPOSITORY
        or value["source_repository_id"] != str(SOURCE_REPOSITORY_ID)
    ):
        raise BridgeError("request-repository-invalid")
    _pr_number(value["pr_number"])
    for name in ("base_sha", "head_sha", "merge_sha"):
        _sha(value[name], f"request-{name}-invalid")
    expected_request_id = (
        f"trtmc-premerge:{value['pr_number']}:"
        f"{value['head_sha']}:{value['merge_sha']}"
    )
    if value["request_id"] != expected_request_id:
        raise BridgeError("request-id-invalid")
    return value


def _create_check(token: str, request: dict[str, str]) -> int:
    value = _mapping(
        _request(
            token,
            "POST",
            f"/repos/{SOURCE_REPOSITORY}/check-runs",
            payload={
                "name": CHECK_NAME,
                # Check Runs calls this field ``head_sha`` even when the
                # tested object is GitHub's synthetic PR merge commit.
                "head_sha": request["merge_sha"],
                "status": "queued",
                "external_id": request["request_id"],
                "output": {
                    "title": "Private CI queued",
                    "summary": (
                        "The exact pull-request merge snapshot has been "
                        "accepted for private CI."
                    ),
                },
            },
        ),
        "check-create-invalid",
    )
    check_id = value.get("id")
    if not isinstance(check_id, int) or check_id <= 0:
        raise BridgeError("check-create-invalid")
    return check_id


def _complete_dispatch_failure(token: str, check_id: int) -> None:
    _request(
        token,
        "PATCH",
        f"/repos/{SOURCE_REPOSITORY}/check-runs/{check_id}",
        payload={
            "status": "completed",
            "conclusion": "failure",
            "output": {
                "title": "Private CI dispatch failed",
                "summary": "No test code was executed. Reapply the run-ci label.",
            },
        },
    )


def _consume_label(token: str, pr_number: str) -> None:
    _request(
        token,
        "DELETE",
        (
            f"/repos/{SOURCE_REPOSITORY}/issues/{pr_number}/labels/"
            "run-ci"
        ),
        expect_json=False,
    )


def dispatch(request_path: Path, *, consume_label: bool) -> None:
    request = _load_request(request_path)
    ci_repository = _private_ci_repository()
    github_token = _required_token("GITHUB_TOKEN")
    source_token = _required_token("SOURCE_CHECK_TOKEN")
    ci_token = _required_token("CI_DISPATCH_TOKEN")
    if consume_label:
        _consume_label(github_token, request["pr_number"])

    check_id = _create_check(source_token, request)
    inputs = dict(request)
    inputs["check_run_id"] = str(check_id)
    try:
        _request(
            ci_token,
            "POST",
            (
                f"/repos/{ci_repository}/actions/workflows/"
                f"{CI_WORKFLOW}/dispatches"
            ),
            payload={"ref": CI_WORKFLOW_REF, "inputs": inputs},
            expect_json=False,
        )
    except BridgeError:
        try:
            _complete_dispatch_failure(source_token, check_id)
        except BridgeError:
            pass
        raise


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--event", type=Path, required=True)
    resolve_parser.add_argument("--pr-number")
    resolve_parser.add_argument("--output", type=Path, required=True)

    dispatch_parser = subparsers.add_parser("dispatch")
    dispatch_parser.add_argument("--request", type=Path, required=True)
    dispatch_parser.add_argument("--consume-label", action="store_true")

    options = parser.parse_args(arguments)
    try:
        if options.command == "resolve":
            resolve(options.event, options.pr_number or None, options.output)
        else:
            dispatch(options.request, consume_label=options.consume_label)
    except BridgeError as error:
        print(f"private CI bridge rejected request: {error}", file=sys.stderr)
        return 1
    print(f"private CI bridge {options.command}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

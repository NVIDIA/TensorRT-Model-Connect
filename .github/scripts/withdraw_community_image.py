# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Withdraw only the reviewed dependency-image publication and its OCI children."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PACKAGE = "tensorrt-model-connect-community/nemotron_h"
PACKAGE_ID = 15714309
REPOSITORY_ID = 1216320259
ROOT_VERSION_ID = 1356372450
ROOT_DIGEST = "sha256:5c7111005a2d012b51cf0ecd7a14df0227b81e42720c8882e3842b2d01ab0272"
# These are the two child descriptors from the digest-verified OCI index.
DIGESTS = frozenset(
    {
        ROOT_DIGEST,
        "sha256:f8e0e6d165997cb4563d2242d90e95d1acdc9ccc0d89ef996b53376142403bd8",
        "sha256:906596fa0e3fbd98eb3244c31da712643d8a2d6afa967168d03d72f116d8eb95",
    }
)
ENDPOINT = "https://api.github.com/orgs/NVIDIA/packages/container/" + urllib.parse.quote(
    PACKAGE, safe=""
)
LAST_TAGGED_MESSAGE = (
    "You cannot delete the last tagged version of a package. You must delete the package instead."
)


class PackageAPIError(RuntimeError):
    def __init__(self, method, status, message="", *, policy_message=None):
        self.status = status
        policy = message if policy_message is None else policy_message
        self.api_message = policy if policy == LAST_TAGGED_MESSAGE else ""
        detail = f": {message}" if message else ""
        super().__init__(f"Package API {method} failed with HTTP {status}{detail}")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def api(token: str, suffix: str = "", *, method: str = "GET", control: bool = False):
    if control and (suffix or method != "GET"):
        raise RuntimeError("Organization control is read-only")
    endpoint = (
        "https://api.github.com/orgs/NVIDIA/packages?package_type=container&per_page=1"
        if control
        else ENDPOINT + suffix
    )
    request = urllib.request.Request(
        endpoint,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            if method == "DELETE":
                if response.status != 204:
                    raise RuntimeError("Package deletion did not return HTTP 204")
                return None
            if response.status != 200:
                raise RuntimeError("Package inventory did not return HTTP 200")
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise RuntimeError("Package inventory exceeds the response limit")
            return json.loads(payload)
    except urllib.error.HTTPError as error:
        # GitHub can reject a deletion for a policy reason even after reads
        # succeed. Retain only its bounded message, never headers or raw bodies.
        message = ""
        policy_message = ""
        try:
            body = json.loads(error.read(4096))
            value = body.get("message") if isinstance(body, dict) else None
            if isinstance(value, str):
                policy_message = value if value == LAST_TAGGED_MESSAGE else ""
                message = value.replace(token, "[redacted]") if token else value
                message = re.sub(
                    r"gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+", "[redacted]", message
                )
                message = " ".join(message.split())[:500]
        except (OSError, ValueError):
            pass
        raise PackageAPIError(method, error.code, message, policy_message=policy_message) from None
    except (OSError, ValueError):
        raise RuntimeError("Package API evidence is unavailable or invalid") from None


def inventory(token: str, *, state: str = "active") -> list[dict]:
    if state not in {"active", "deleted"}:
        raise RuntimeError("Unknown package version state")
    rows = []
    for page in range(1, 11):
        batch = api(token, f"/versions?state={state}&per_page=100&page={page}")
        if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
            raise RuntimeError("Package version inventory is not an array of objects")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise RuntimeError("Package version inventory exceeds the page limit")


def validated_package(token: str) -> None:
    package = api(token)
    if (
        not isinstance(package, dict)
        or package.get("id") != PACKAGE_ID
        or package.get("name") != PACKAGE
        or not isinstance(package.get("repository"), dict)
        or package["repository"].get("id") != REPOSITORY_ID
    ):
        raise RuntimeError("Package identity does not match the reviewed publication")


def withdraw_whole_package(token: str) -> dict:
    validated_package(token)
    active_rows = inventory(token)
    # Whole-package deletion also covers recoverable version history.
    rows = active_rows + inventory(token, state="deleted")
    ids = [row.get("id") for row in rows]
    if (
        not active_rows
        or any(not isinstance(row.get("name"), str) or row["name"] not in DIGESTS for row in rows)
        or any(type(identity) is not int or identity <= 0 for identity in ids)
        or len(set(ids)) != len(ids)
        or any(row["name"] == ROOT_DIGEST and row["id"] != ROOT_VERSION_ID for row in rows)
    ):
        raise RuntimeError("Whole-package withdrawal would include unreviewed versions")
    api(token, method="DELETE")
    print(
        json.dumps({"package_id": PACKAGE_ID, "whole_package": True, "http_status": 204}),
        flush=True,
    )
    for _ in range(2):
        try:
            api(token)
        except PackageAPIError as error:
            if error.status != 404:
                raise
        else:
            raise RuntimeError("Withdrawn package remains visible")
        control = api(token, control=True)
        if not isinstance(control, list) or any(not isinstance(row, dict) for row in control):
            raise RuntimeError("Authenticated organization control is unavailable")
        if any(row.get("id") == PACKAGE_ID or row.get("name") == PACKAGE for row in control):
            raise RuntimeError("Withdrawn package remains in organization control")
        time.sleep(2)
    return {
        "package": PACKAGE,
        "reviewed_digests": sorted(DIGESTS),
        "whole_package_deleted": True,
        "package_delete_http_status": 204,
        "deleted": [
            {"id": row["id"], "digest": row["name"], "scope": "whole_package"} for row in rows
        ],
        "remaining_other_versions": 0,
        "authenticated_absence_confirmations": 2,
    }


def withdraw(token: str) -> dict:
    if not token:
        raise RuntimeError("Package administration credentials are unavailable")
    validated_package(token)
    before = inventory(token)
    targets = [row for row in before if row.get("name") in DIGESTS]
    ids = [row.get("id") for row in targets]
    if any(type(value) is not int or value <= 0 for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Package version identities are invalid")
    for row in targets:
        if row["name"] == ROOT_DIGEST and row["id"] != ROOT_VERSION_ID:
            raise RuntimeError("Root version identity does not match the reviewed publication")
    deleted = []
    # Remove the index first, then both directly addressable child manifests.
    for row in sorted(targets, key=lambda row: row["name"] != ROOT_DIGEST):
        current = api(token, f"/versions/{row['id']}")
        if current.get("id") != row["id"] or current.get("name") != row["name"]:
            raise RuntimeError("Package version identity changed before deletion")
        try:
            api(token, f"/versions/{row['id']}", method="DELETE")
        except PackageAPIError as error:
            if error.status != 400 or error.api_message != LAST_TAGGED_MESSAGE:
                raise
            return withdraw_whole_package(token)
        deleted.append({"id": row["id"], "digest": row["name"], "http_status": 204})
        print(json.dumps(deleted[-1]), flush=True)
    for _ in range(2):
        if any(row.get("name") in DIGESTS for row in inventory(token)):
            raise RuntimeError("Withdrawn image remains in the authenticated active inventory")
        time.sleep(2)
    return {
        "package": PACKAGE,
        "reviewed_digests": sorted(DIGESTS),
        "deleted": deleted,
        "remaining_other_versions": len([row for row in before if row.get("name") not in DIGESTS]),
        "authenticated_absence_confirmations": 2,
    }


if __name__ == "__main__":
    receipt = withdraw(os.environ.get("GH_TOKEN", ""))
    output = Path(os.environ["RUNNER_TEMP"]) / "community-image-withdrawal.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt.get("whole_package_deleted"):
        print("Reviewed package deletion confirmed by two authenticated absence checks.")
    else:
        print("Reviewed image versions are absent from two authenticated active inventories.")

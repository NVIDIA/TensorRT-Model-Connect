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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def api(token: str, suffix: str = "", *, method: str = "GET"):
    request = urllib.request.Request(
        ENDPOINT + suffix,
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
        try:
            body = json.loads(error.read(4096))
            value = body.get("message") if isinstance(body, dict) else None
            if isinstance(value, str):
                message = value.replace(token, "[redacted]") if token else value
                message = re.sub(
                    r"gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+", "[redacted]", message
                )
                message = " ".join(message.split())[:500]
        except (OSError, ValueError):
            pass
        detail = f": {message}" if message else ""
        raise RuntimeError(f"Package API {method} failed with HTTP {error.code}{detail}") from None
    except (OSError, ValueError):
        raise RuntimeError("Package API evidence is unavailable or invalid") from None


def inventory(token: str) -> list[dict]:
    rows = []
    for page in range(1, 11):
        batch = api(token, f"/versions?state=active&per_page=100&page={page}")
        if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
            raise RuntimeError("Package version inventory is not an array of objects")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise RuntimeError("Package version inventory exceeds the page limit")


def withdraw(token: str) -> dict:
    if not token:
        raise RuntimeError("Package administration credentials are unavailable")
    package = api(token)
    if (
        not isinstance(package, dict)
        or package.get("id") != PACKAGE_ID
        or package.get("name") != PACKAGE
        or package.get("repository", {}).get("id") != REPOSITORY_ID
    ):
        raise RuntimeError("Package identity does not match the reviewed publication")
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
        api(token, f"/versions/{row['id']}", method="DELETE")
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
    print("Reviewed image versions are absent from two authenticated active inventories.")

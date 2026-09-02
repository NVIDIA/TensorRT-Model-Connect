# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind a private failure payload to one authorized Source commit graph."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence


class PublicFailureIdentityError(ValueError):
    """The failure payload does not match the authorized Source snapshot."""


CommitParents = Callable[[str], Sequence[str]]
IsAncestor = Callable[[str, str], bool]


def validate_failure_identity(
    report: Mapping[str, object],
    *,
    expected_dispatch_nonce: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_base_sha: str,
    commit_parents: CommitParents,
    is_ancestor: IsAncestor,
) -> None:
    """Validate exact run identity plus the authorized base-to-merge lineage."""
    expected = {
        "dispatch_nonce": expected_dispatch_nonce,
        "head_sha": expected_head_sha,
        "pr_number": expected_pr_number,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise PublicFailureIdentityError("the failure payload does not match this authorized run")

    kind = report.get("tested_revision_kind")
    tested_revision = report.get("tested_revision")
    resolved_base = report.get("base_sha")
    if kind == "head":
        if tested_revision != expected_head_sha or resolved_base != expected_base_sha:
            raise PublicFailureIdentityError(
                "the head failure payload does not match the authorized snapshot"
            )
        return
    if (
        kind != "merge"
        or not isinstance(tested_revision, str)
        or not isinstance(resolved_base, str)
    ):
        raise PublicFailureIdentityError("the failure payload has an invalid revision kind")

    if not is_ancestor(expected_base_sha, resolved_base):
        raise PublicFailureIdentityError(
            "the resolved merge parent is outside the authorized base lineage"
        )
    if tuple(commit_parents(tested_revision)) != (resolved_base, expected_head_sha):
        raise PublicFailureIdentityError(
            "the tested merge does not have the expected commit parents"
        )


class GitHubCommitGraph:
    """Read the minimal public commit graph required for identity validation."""

    def __init__(self, repository: str) -> None:
        self.repository = repository

    def _api(self, endpoint: str, jq_filter: str) -> Mapping[str, object]:
        result = subprocess.run(
            ["gh", "api", "--method", "GET", endpoint, "--jq", jq_filter],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise PublicFailureIdentityError(
                "GitHub could not verify the failure payload commit graph"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise PublicFailureIdentityError(
                "GitHub returned an invalid commit graph response"
            ) from error
        if not isinstance(payload, Mapping):
            raise PublicFailureIdentityError("GitHub returned an invalid commit graph response")
        return payload

    def commit_parents(self, revision: str) -> tuple[str, ...]:
        payload = self._api(
            f"/repos/{self.repository}/commits/{revision}",
            "{sha: .sha, parents: [.parents[].sha]}",
        )
        if payload.get("sha") != revision:
            raise PublicFailureIdentityError("GitHub returned the wrong tested revision")
        parents = payload.get("parents")
        if not isinstance(parents, list):
            raise PublicFailureIdentityError("GitHub returned invalid commit parents")
        resolved: list[str] = []
        for parent in parents:
            if not isinstance(parent, str):
                raise PublicFailureIdentityError("GitHub returned invalid commit parents")
            resolved.append(parent)
        return tuple(resolved)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        if ancestor == descendant:
            return True
        comparison = self._api(
            f"/repos/{self.repository}/compare/{ancestor}...{descendant}",
            "{status: .status, merge_base_sha: .merge_base_commit.sha}",
        )
        return (
            comparison.get("status") == "ahead"
            and comparison.get("merge_base_sha") == ancestor
        )

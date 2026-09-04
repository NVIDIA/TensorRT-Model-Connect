# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from tools.public_failure.identity import (
    GitHubCommitGraph,
    PublicFailureIdentityError,
    validate_failure_identity,
)


APPROVED_BASE_SHA = "a" * 40
CURRENT_MERGE_PARENT_SHA = "b" * 40
HEAD_SHA = "c" * 40
TESTED_MERGE_SHA = "d" * 40
DISPATCH_NONCE = "e" * 32


def _report(*, tested_revision_kind: str = "merge") -> dict[str, object]:
    if tested_revision_kind == "head":
        return {
            "dispatch_nonce": DISPATCH_NONCE,
            "pr_number": 1059,
            "head_sha": HEAD_SHA,
            "base_sha": APPROVED_BASE_SHA,
            "tested_revision": HEAD_SHA,
            "tested_revision_kind": "head",
        }
    return {
        "dispatch_nonce": DISPATCH_NONCE,
        "pr_number": 1059,
        "head_sha": HEAD_SHA,
        "base_sha": CURRENT_MERGE_PARENT_SHA,
        "tested_revision": TESTED_MERGE_SHA,
        "tested_revision_kind": "merge",
    }


def _validate(
    report: dict[str, object],
    *,
    parents: Sequence[str] = (CURRENT_MERGE_PARENT_SHA, HEAD_SHA),
    approved_base_is_ancestor: bool = True,
) -> None:
    validate_failure_identity(
        report,
        expected_dispatch_nonce=DISPATCH_NONCE,
        expected_pr_number=1059,
        expected_head_sha=HEAD_SHA,
        expected_base_sha=APPROVED_BASE_SHA,
        commit_parents=lambda _revision: parents,
        is_ancestor=lambda _ancestor, _descendant: approved_base_is_ancestor,
    )


def test_accepts_merge_payload_based_on_a_newer_main_descendant() -> None:
    _validate(_report())


def test_rejects_merge_payload_outside_the_authorized_base_lineage() -> None:
    with pytest.raises(PublicFailureIdentityError, match="authorized base lineage"):
        _validate(_report(), approved_base_is_ancestor=False)


def test_rejects_merge_payload_with_the_wrong_commit_parents() -> None:
    with pytest.raises(PublicFailureIdentityError, match="expected commit parents"):
        _validate(_report(), parents=(APPROVED_BASE_SHA, HEAD_SHA))


def test_accepts_head_payload_when_snapshot_resolution_failed() -> None:
    def unexpected_graph_lookup(*_args: str) -> object:
        raise AssertionError("head payload must not query the commit graph")

    validate_failure_identity(
        _report(tested_revision_kind="head"),
        expected_dispatch_nonce=DISPATCH_NONCE,
        expected_pr_number=1059,
        expected_head_sha=HEAD_SHA,
        expected_base_sha=APPROVED_BASE_SHA,
        commit_parents=unexpected_graph_lookup,
        is_ancestor=unexpected_graph_lookup,
    )


def test_github_graph_requests_only_bounded_identity_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if "/commits/" in arguments[4]:
            output = (
                f'{{"sha":"{TESTED_MERGE_SHA}",'
                f'"parents":["{CURRENT_MERGE_PARENT_SHA}","{HEAD_SHA}"]}}'
            )
        else:
            output = f'{{"status":"ahead","merge_base_sha":"{APPROVED_BASE_SHA}"}}'
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr="")

    monkeypatch.setattr("tools.public_failure.identity.subprocess.run", fake_run)
    graph = GitHubCommitGraph("NVIDIA/TensorRT-Model-Connect")

    assert graph.commit_parents(TESTED_MERGE_SHA) == (
        CURRENT_MERGE_PARENT_SHA,
        HEAD_SHA,
    )
    assert graph.is_ancestor(APPROVED_BASE_SHA, CURRENT_MERGE_PARENT_SHA)
    assert all("--jq" in call for call in calls)

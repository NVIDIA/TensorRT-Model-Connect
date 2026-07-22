# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for per-node cache readiness receipts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.ci.cache_warm_receipt import create_receipt, verify_receipts
from tools.ci.process import CiError


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
REVISION = "1" * 40


def _summary(mode: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": REVISION,
        "mode": mode,
        "status": "passed",
        "cache_root": "/cache/hub",
        "cache_plan_digest": DIGEST_A,
        "resolved_cache_digest": DIGEST_B,
        "expected_count": 10,
        "present_count": 10,
        "missing_count": 0,
        "cached_count": 10,
        "downloaded_count": 0,
        "started_at": "2026-07-21T00:00:00+00:00",
        "completed_at": "2026-07-21T00:01:00+00:00",
    }


def _environment(node: str, runner: str) -> dict[str, str]:
    return {
        "TRTMC_NODE_ID": node,
        "RUNNER_NAME": runner,
        "GITHUB_RUN_ID": "123",
        "GITHUB_JOB": "cache-warm",
        "TRTMC_CI_WORKSPACE": "/work/nightly-cache",
        "TRTMC_CACHE_SOURCE_REVISION": REVISION,
        "TRTMC_HF_CACHE": "/cache",
        "TRTMC_HF_HUB_CACHE": "/cache/hub",
        "TRTMC_HF_CACHE_LOCK_FILE": "/cache.lock",
    }


def _receipt(node: str) -> dict[str, object]:
    receipt = create_receipt(
        _summary("warm"),
        _summary("local-only"),
        _environment(node, f"{node}-proof-00"),
    )
    receipt["hostname"] = f"host-{node}"
    return receipt


def _matrix() -> dict[str, object]:
    return {
        "include": [
            {
                "node_label": "trtmc-node-node-a",
                "anchor_runner": "node-a-proof-00",
            },
            {
                "node_label": "trtmc-node-node-b",
                "anchor_runner": "node-b-proof-00",
            },
        ]
    }


def test_receipt_requires_successful_zero_download_local_verification() -> None:
    verify = _summary("local-only")
    verify["downloaded_count"] = 1
    verify["cached_count"] = 9

    with pytest.raises(CiError, match="download zero"):
        create_receipt(_summary("warm"), verify, _environment("node-a", "node-a-proof-00"))


def test_receipt_rejects_hub_outside_root_or_summary_mismatch() -> None:
    environment = _environment("node-a", "node-a-proof-00")
    environment["TRTMC_HF_HUB_CACHE"] = "/other/hub"
    with pytest.raises(CiError, match="unsafe cache root or hub"):
        create_receipt(_summary("warm"), _summary("local-only"), environment)

    environment = _environment("node-a", "node-a-proof-00")
    warm = _summary("warm")
    warm["cache_root"] = "/cache/different-hub"
    with pytest.raises(CiError, match="does not describe the configured Hub cache"):
        create_receipt(warm, _summary("local-only"), environment)

    environment = _environment("node-a", "node-a-proof-00")
    environment.update(
        {
            "TRTMC_CI_WORKSPACE": "/workspace/repository",
            "TRTMC_HF_CACHE": "/workspace",
            "TRTMC_HF_HUB_CACHE": "/workspace/repository/hub",
        }
    )
    warm = _summary("warm")
    warm["cache_root"] = "/workspace/repository/hub"
    with pytest.raises(CiError, match="must not overlap"):
        create_receipt(warm, _summary("local-only"), environment)


def test_receipt_verifier_requires_every_expected_node_and_matching_digests() -> None:
    result = verify_receipts(
        [_receipt("node-a"), _receipt("node-b")],
        _matrix(),
        expected_run_id="123",
        expected_revision=REVISION,
    )

    assert result == {
        "schema_version": 1,
        "status": "ready",
        "node_count": 2,
        "run_id": "123",
        "source_revision": REVISION,
        "cache_plan_digest": DIGEST_A,
        "resolved_cache_digest": DIGEST_B,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda receipts: receipts.pop(), "node set mismatch"),
        (
            lambda receipts: receipts[1].update({"cache_plan_digest": "c" * 64}),
            "disagree on cache_plan_digest",
        ),
        (
            lambda receipts: receipts[1].update({"anchor_runner": "wrong"}),
            "identity does not match",
        ),
        (
            lambda receipts: receipts[1].update({"run_id": "456"}),
            "wrong workflow run",
        ),
        (
            lambda receipts: receipts[1].update({"present_count": 9}),
            "is incomplete",
        ),
        (
            lambda receipts: receipts.append(deepcopy(receipts[0])),
            "duplicate cache-warm receipt",
        ),
        (
            lambda receipts: receipts[1].update({"hostname": "host-node-a"}),
            "hostname .* maps to multiple node IDs",
        ),
    ],
)
def test_receipt_verifier_fails_closed(mutate, message: str) -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    mutate(receipts)

    with pytest.raises(CiError, match=message):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_revision=REVISION,
        )


@pytest.mark.parametrize(
    ("expected_run_id", "expected_revision", "message"),
    [
        ("456", REVISION, "wrong workflow run"),
        ("123", "2" * 40, "wrong source revision"),
    ],
)
def test_receipt_verifier_binds_current_run_and_revision(
    expected_run_id: str, expected_revision: str, message: str
) -> None:
    with pytest.raises(CiError, match=message):
        verify_receipts(
            [_receipt("node-a"), _receipt("node-b")],
            _matrix(),
            expected_run_id=expected_run_id,
            expected_revision=expected_revision,
        )

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
SUMMARY_COUNT_FIELDS = (
    "expected_count",
    "present_count",
    "missing_count",
    "cached_count",
    "downloaded_count",
)
RECEIPT_COUNT_FIELDS = SUMMARY_COUNT_FIELDS + (
    "verification_present_count",
    "verification_missing_count",
    "verification_cached_count",
    "verification_downloaded_count",
)


def _summary(mode: str) -> dict[str, object]:
    if mode == "warm":
        started_at = "2026-07-21T00:00:00+00:00"
        completed_at = "2026-07-21T00:01:00+00:00"
    else:
        started_at = "2026-07-21T00:02:00+00:00"
        completed_at = "2026-07-21T00:03:00+00:00"
    return {
        "schema_version": 1,
        "source_revision": REVISION,
        "mode": mode,
        "status": "passed",
        "cache_root": "/cache/hub" if mode == "warm" else "/hf-cache/hub",
        "cache_plan_digest": DIGEST_A,
        "resolved_cache_digest": DIGEST_B,
        "expected_count": 10,
        "present_count": 10,
        "missing_count": 0,
        "cached_count": 10,
        "downloaded_count": 0,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _environment(node: str, runner: str) -> dict[str, str]:
    return {
        "TRTMC_NODE_ID": node,
        "RUNNER_NAME": runner,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
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
            {"node_label": "trtmc-node-node-a"},
            {"node_label": "trtmc-node-node-b"},
        ]
    }


def test_receipt_maps_host_warm_path_to_read_only_container_verify_path() -> None:
    warm = _summary("warm")
    verify = _summary("local-only")

    receipt = create_receipt(warm, verify, _environment("node-a", "node-a-proof-00"))

    assert warm["cache_root"] == "/cache/hub"
    assert verify["cache_root"] == "/hf-cache/hub"
    assert receipt["cache_root"] == "/cache"
    assert receipt["hub_cache"] == "/cache/hub"


def test_receipt_requires_successful_zero_download_local_verification() -> None:
    verify = _summary("local-only")
    verify["downloaded_count"] = 1
    verify["cached_count"] = 9

    with pytest.raises(CiError, match="download zero"):
        create_receipt(_summary("warm"), verify, _environment("node-a", "node-a-proof-00"))


@pytest.mark.parametrize(
    "key",
    (
        "schema_version",
        "source_revision",
        "mode",
        "status",
        "cache_root",
        "cache_plan_digest",
        "resolved_cache_digest",
        *SUMMARY_COUNT_FIELDS,
        "started_at",
        "completed_at",
    ),
)
def test_receipt_creation_rejects_every_absent_summary_field(key: str) -> None:
    warm = _summary("warm")
    del warm[key]

    with pytest.raises(CiError):
        create_receipt(warm, _summary("local-only"), _environment("node-a", "runner-a"))


@pytest.mark.parametrize("key", SUMMARY_COUNT_FIELDS)
def test_receipt_creation_rejects_boolean_summary_counts(key: str) -> None:
    warm = _summary("warm")
    warm[key] = False

    with pytest.raises(CiError, match=key):
        create_receipt(warm, _summary("local-only"), _environment("node-a", "runner-a"))


@pytest.mark.parametrize(
    ("summary_key", "value", "message"),
    [
        ("schema_version", True, "unsupported schema or mode"),
        ("source_revision", "main", "source revision is invalid"),
        ("cache_plan_digest", "abc", "cache_plan_digest is invalid"),
        ("resolved_cache_digest", "b" * 63, "resolved_cache_digest is invalid"),
        ("cache_root", "relative/hub", "cache root is invalid"),
        ("started_at", 123, "started_at is invalid"),
        ("started_at", "2026-07-21T00:00:00", "started_at is timezone-free"),
        ("completed_at", "not-a-time", "completed_at is invalid"),
    ],
)
def test_receipt_creation_rejects_malformed_summary_evidence(
    summary_key: str, value: object, message: str
) -> None:
    warm = _summary("warm")
    warm[summary_key] = value

    with pytest.raises(CiError, match=message):
        create_receipt(warm, _summary("local-only"), _environment("node-a", "runner-a"))


def test_receipt_creation_requires_strict_timestamp_order() -> None:
    warm = _summary("warm")
    warm["completed_at"] = "2026-07-21T00:00:00+00:00"
    warm["started_at"] = "2026-07-21T00:01:00+00:00"
    with pytest.raises(CiError, match="completes before it starts"):
        create_receipt(warm, _summary("local-only"), _environment("node-a", "runner-a"))

    verify = _summary("local-only")
    verify["started_at"] = "2026-07-21T00:00:30+00:00"
    with pytest.raises(CiError, match="starts before cache warm completes"):
        create_receipt(_summary("warm"), verify, _environment("node-a", "runner-a"))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("TRTMC_NODE_ID", "bad/node", "node ID is invalid"),
        ("RUNNER_NAME", "", "environment is missing"),
        ("GITHUB_RUN_ID", "0", "workflow run ID is invalid"),
        ("GITHUB_RUN_ATTEMPT", "01", "workflow run attempt is invalid"),
        ("GITHUB_JOB", "cache warm", "job ID is invalid"),
        ("TRTMC_CACHE_SOURCE_REVISION", "main", "source revision is invalid"),
        ("TRTMC_CI_WORKSPACE", "relative/work", "workspace is invalid"),
        ("TRTMC_HF_CACHE", "relative/cache", "cache root is invalid"),
        ("TRTMC_HF_HUB_CACHE", "/cache/../hub", "Hub cache is invalid"),
        ("TRTMC_HF_CACHE_LOCK_FILE", "relative.lock", "cache lock path is invalid"),
    ],
)
def test_receipt_creation_rejects_malformed_identity_or_path_environment(
    key: str, value: str, message: str
) -> None:
    environment = _environment("node-a", "runner-a")
    environment[key] = value

    with pytest.raises(CiError, match=message):
        create_receipt(_summary("warm"), _summary("local-only"), environment)


@pytest.mark.parametrize("key", sorted(_environment("node-a", "runner-a")))
def test_receipt_creation_rejects_every_absent_environment_field(key: str) -> None:
    environment = _environment("node-a", "runner-a")
    del environment[key]

    with pytest.raises(CiError, match="environment is missing"):
        create_receipt(_summary("warm"), _summary("local-only"), environment)


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

    verify = _summary("local-only")
    verify["cache_root"] = "/cache/other-hub"
    with pytest.raises(CiError, match="does not describe the read-only container Hub cache"):
        create_receipt(_summary("warm"), verify, _environment("node-a", "runner-a"))

    environment = _environment("node-a", "runner-a")
    environment["TRTMC_HF_CACHE_LOCK_FILE"] = "/cache/hub/cache.lock"
    with pytest.raises(CiError, match="unsafe cache lock path"):
        create_receipt(_summary("warm"), _summary("local-only"), environment)


def test_receipt_verifier_requires_every_expected_node_and_matching_digests() -> None:
    result = verify_receipts(
        [_receipt("node-a"), _receipt("node-b")],
        _matrix(),
        expected_run_id="123",
        expected_run_attempt="1",
        expected_job_id="cache-warm",
        expected_revision=REVISION,
    )

    assert result == {
        "schema_version": 2,
        "status": "ready",
        "node_count": 2,
        "run_id": "123",
        "run_attempt": "1",
        "job_id": "cache-warm",
        "source_revision": REVISION,
        "cache_plan_digest": DIGEST_A,
        "resolved_cache_digest": DIGEST_B,
        "expected_count": 10,
        "cache_root": "/cache",
        "hub_cache": "/cache/hub",
        "cache_lock_file": "/cache.lock",
    }


def test_receipt_verifier_accepts_any_safe_runner_selected_on_the_declared_node() -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0]["anchor_runner"] = "node-a-proof-09"

    result = verify_receipts(
        receipts,
        _matrix(),
        expected_run_id="123",
        expected_run_attempt="1",
        expected_job_id="cache-warm",
        expected_revision=REVISION,
    )

    assert result["status"] == "ready"
    assert result["node_count"] == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda receipts: receipts.pop(), "node set mismatch"),
        (
            lambda receipts: receipts[1].update({"cache_plan_digest": "c" * 64}),
            "disagree on cache_plan_digest",
        ),
        (
            lambda receipts: receipts[1].update({"anchor_runner": "node-a-proof-00"}),
            "runner .* maps to multiple node IDs",
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
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


@pytest.mark.parametrize("key", sorted(_receipt("node-a")))
def test_receipt_verifier_rejects_every_absent_receipt_field(key: str) -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    del receipts[0][key]

    with pytest.raises(CiError, match="field set is invalid"):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


@pytest.mark.parametrize("key", RECEIPT_COUNT_FIELDS)
def test_receipt_verifier_rejects_boolean_counts(key: str) -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0][key] = True

    with pytest.raises(CiError, match=key):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 1, "unsupported schema"),
        ("node_id", "bad/node", "node ID is invalid"),
        ("hostname", "bad host", "hostname is invalid"),
        ("anchor_runner", "bad runner", "anchor runner is invalid"),
        ("run_id", 123, "run ID is invalid"),
        ("run_attempt", "2", "wrong workflow run attempt"),
        ("job_id", "other-job", "wrong job"),
        ("source_revision", "main", "source revision is invalid"),
        ("cache_plan_digest", "a" * 63, "cache_plan_digest is invalid"),
        ("resolved_cache_digest", "B" * 64, "resolved_cache_digest is invalid"),
        ("verification_status", "failed", "unsupported schema"),
        ("warm_started_at", 123, "warm_started_at is invalid"),
        ("warm_completed_at", "2026-07-21", "warm_completed_at is timezone-free"),
        ("cache_root", "relative", "cache root is invalid"),
        ("hub_cache", "/other/hub", "unsafe cache paths"),
        ("cache_lock_file", "/", "cache lock file is invalid"),
        ("cache_lock_file", "/cache/hub/cache.lock", "unsafe cache lock path"),
    ],
)
def test_receipt_verifier_rejects_malformed_evidence(key: str, value: object, message: str) -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0][key] = value

    with pytest.raises(CiError, match=message):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


def test_receipt_verifier_rejects_invalid_timestamp_order_and_count_totals() -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0]["verification_started_at"] = "2026-07-20T23:59:00+00:00"
    with pytest.raises(CiError, match="invalid timestamp order"):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )

    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0]["cached_count"] = 9
    with pytest.raises(CiError, match="is incomplete"):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


def test_receipt_verifier_rejects_unexpected_fields_and_cross_node_path_drift() -> None:
    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[0]["legacy_verified_at"] = receipts[0]["verification_completed_at"]
    with pytest.raises(CiError, match="unexpected=.*legacy_verified_at"):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )

    receipts = [_receipt("node-a"), _receipt("node-b")]
    receipts[1].update(
        {
            "cache_root": "/other-cache",
            "hub_cache": "/other-cache/hub",
            "cache_lock_file": "/other-cache.lock",
        }
    )
    with pytest.raises(CiError, match="disagree on cache_root"):
        verify_receipts(
            receipts,
            _matrix(),
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
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
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=expected_revision,
        )


@pytest.mark.parametrize(
    ("expected_run_attempt", "expected_job_id", "message"),
    [
        ("2", "cache-warm", "wrong workflow run attempt"),
        ("1", "other-job", "wrong job"),
        ("0", "cache-warm", "expected cache-warm workflow run attempt is invalid"),
        ("1", "bad job", "expected cache-warm job ID is invalid"),
    ],
)
def test_receipt_verifier_binds_current_attempt_and_job(
    expected_run_attempt: str, expected_job_id: str, message: str
) -> None:
    with pytest.raises(CiError, match=message):
        verify_receipts(
            [_receipt("node-a"), _receipt("node-b")],
            _matrix(),
            expected_run_id="123",
            expected_run_attempt=expected_run_attempt,
            expected_job_id=expected_job_id,
            expected_revision=REVISION,
        )


def test_receipt_verifier_rejects_duplicate_node_in_expected_matrix() -> None:
    matrix = _matrix()
    matrix["include"][1]["node_label"] = "trtmc-node-node-a"

    with pytest.raises(CiError, match="repeats node"):
        verify_receipts(
            [_receipt("node-a"), _receipt("node-b")],
            matrix,
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


def test_receipt_verifier_rejects_legacy_runner_identity_in_expected_matrix() -> None:
    matrix = _matrix()
    matrix["include"][0]["anchor_runner"] = "node-a-proof-00"

    with pytest.raises(CiError, match="entry fields are invalid"):
        verify_receipts(
            [_receipt("node-a"), _receipt("node-b")],
            matrix,
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )


def test_receipt_verifier_rejects_non_object_expected_matrix() -> None:
    with pytest.raises(CiError, match="matrix must be a JSON object"):
        verify_receipts(
            [_receipt("node-a"), _receipt("node-b")],
            [],
            expected_run_id="123",
            expected_run_attempt="1",
            expected_job_id="cache-warm",
            expected_revision=REVISION,
        )

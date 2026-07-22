# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and verify fail-closed per-node Nightly cache-warm receipts.

Boundary: evidence validation only; topology planning and cache warming stay elsewhere.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import socket
import sys
from pathlib import Path, PurePosixPath

from .process import CiError


_DIGEST = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_RECEIPT_SCHEMA_VERSION = 2
_LOCAL_ONLY_HUB_CACHE = PurePosixPath("/hf-cache/hub")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "node_id",
        "hostname",
        "anchor_runner",
        "run_id",
        "run_attempt",
        "job_id",
        "source_revision",
        "cache_plan_digest",
        "resolved_cache_digest",
        "expected_count",
        "present_count",
        "missing_count",
        "cached_count",
        "downloaded_count",
        "warm_started_at",
        "warm_completed_at",
        "verification_status",
        "verification_present_count",
        "verification_missing_count",
        "verification_cached_count",
        "verification_downloaded_count",
        "verification_started_at",
        "verification_completed_at",
        "cache_root",
        "hub_cache",
        "cache_lock_file",
    }
)


def _load_object(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CiError(f"{description} is unavailable or invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CiError(f"{description} must be a JSON object: {path}")
    return payload


def _validated_digest(value: object, description: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CiError(f"{description} is invalid")
    return value


def _validated_revision(value: object, description: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise CiError(f"{description} is invalid")
    return value


def _validated_safe_id(value: object, description: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CiError(f"{description} is invalid")
    return value


def _validated_positive_decimal(value: object, description: str) -> str:
    if not isinstance(value, str) or _POSITIVE_DECIMAL.fullmatch(value) is None:
        raise CiError(f"{description} is invalid")
    return value


def _validated_count(
    value: object,
    description: str,
    *,
    positive: bool = False,
) -> int:
    # bool is an int subclass; evidence must not accept true/false as a count.
    if type(value) is not int or value < (1 if positive else 0):
        raise CiError(f"{description} is invalid")
    return value


def _validated_timestamp(value: object, description: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CiError(f"{description} is invalid")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise CiError(f"{description} is invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise CiError(f"{description} is timezone-free")
    return timestamp


def _validated_path(
    value: object,
    description: str,
    *,
    file_path: bool = False,
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CiError(f"{description} is invalid")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or ".." in path.parts
        or str(path) != value
        or (file_path and (value.endswith("/") or not path.name))
    ):
        raise CiError(f"{description} is invalid")
    return path


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validated_summary(payload: dict[str, object], mode: str) -> dict[str, object]:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("mode") != mode
    ):
        raise CiError(f"cache {mode} summary has an unsupported schema or mode")
    if payload.get("status") != "passed":
        raise CiError(f"cache {mode} summary did not pass")
    for key in ("cache_plan_digest", "resolved_cache_digest"):
        _validated_digest(payload.get(key), f"cache {mode} summary {key}")
    _validated_revision(payload.get("source_revision"), f"cache {mode} summary source revision")
    _validated_path(payload.get("cache_root"), f"cache {mode} summary cache root")
    for key in (
        "expected_count",
        "present_count",
        "missing_count",
        "cached_count",
        "downloaded_count",
    ):
        _validated_count(
            payload.get(key),
            f"cache {mode} summary {key}",
            positive=key == "expected_count",
        )
    if payload["missing_count"] != 0 or payload["present_count"] != payload["expected_count"]:
        raise CiError(f"cache {mode} summary is incomplete")
    if payload["cached_count"] + payload["downloaded_count"] != payload["expected_count"]:
        raise CiError(f"cache {mode} summary item counts are inconsistent")
    started_at = _validated_timestamp(payload.get("started_at"), f"cache {mode} summary started_at")
    completed_at = _validated_timestamp(
        payload.get("completed_at"), f"cache {mode} summary completed_at"
    )
    if completed_at < started_at:
        raise CiError(f"cache {mode} summary completes before it starts")
    return payload


def create_receipt(
    warm_summary: dict[str, object],
    verify_summary: dict[str, object],
    environment: dict[str, str],
) -> dict[str, object]:
    warm = _validated_summary(warm_summary, "warm")
    verify = _validated_summary(verify_summary, "local-only")
    for key in (
        "source_revision",
        "cache_plan_digest",
        "resolved_cache_digest",
    ):
        if warm.get(key) != verify.get(key):
            raise CiError(f"warm and local-only summaries disagree on {key}")
    for key in ("expected_count", "present_count", "missing_count"):
        if warm.get(key) != verify.get(key):
            raise CiError(f"warm and local-only summaries disagree on {key}")
    if verify["downloaded_count"] != 0:
        raise CiError("local-only cache verification must download zero items")
    warm_completed_at = _validated_timestamp(
        warm["completed_at"], "cache warm summary completed_at"
    )
    verification_started_at = _validated_timestamp(
        verify["started_at"], "cache local-only summary started_at"
    )
    if verification_started_at < warm_completed_at:
        raise CiError("local-only cache verification starts before cache warm completes")

    required_environment = (
        "TRTMC_NODE_ID",
        "RUNNER_NAME",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "TRTMC_CI_WORKSPACE",
        "TRTMC_CACHE_SOURCE_REVISION",
        "TRTMC_HF_CACHE",
        "TRTMC_HF_HUB_CACHE",
        "TRTMC_HF_CACHE_LOCK_FILE",
    )
    missing = [name for name in required_environment if not environment.get(name)]
    if missing:
        raise CiError(f"cache receipt environment is missing: {missing!r}")
    node_id = _validated_safe_id(environment["TRTMC_NODE_ID"], "cache receipt node ID")
    anchor_runner = _validated_safe_id(environment["RUNNER_NAME"], "cache receipt anchor runner")
    run_id = _validated_positive_decimal(
        environment["GITHUB_RUN_ID"], "cache receipt workflow run ID"
    )
    run_attempt = _validated_positive_decimal(
        environment["GITHUB_RUN_ATTEMPT"], "cache receipt workflow run attempt"
    )
    job_id = _validated_safe_id(environment["GITHUB_JOB"], "cache receipt job ID")
    hostname = _validated_safe_id(socket.gethostname(), "cache receipt hostname")
    source_revision = _validated_revision(
        environment["TRTMC_CACHE_SOURCE_REVISION"],
        "cache receipt source revision",
    )
    _validated_path(environment["TRTMC_CI_WORKSPACE"], "cache receipt workspace")
    _validated_path(environment["TRTMC_HF_CACHE"], "cache receipt cache root")
    _validated_path(environment["TRTMC_HF_HUB_CACHE"], "cache receipt Hub cache")
    _validated_path(
        environment["TRTMC_HF_CACHE_LOCK_FILE"],
        "cache receipt cache lock path",
        file_path=True,
    )
    lock_file = Path(environment["TRTMC_HF_CACHE_LOCK_FILE"]).resolve()
    if warm["source_revision"] != environment["TRTMC_CACHE_SOURCE_REVISION"]:
        raise CiError("cache summary source revision does not match the workflow")
    cache_root = Path(environment["TRTMC_HF_CACHE"]).resolve()
    hub_cache = Path(environment["TRTMC_HF_HUB_CACHE"]).resolve()
    workspace = Path(environment["TRTMC_CI_WORKSPACE"]).resolve()
    if cache_root == Path("/") or hub_cache == cache_root or cache_root not in hub_cache.parents:
        raise CiError("cache receipt environment has an unsafe cache root or hub path")
    if lock_file == cache_root or _paths_overlap(lock_file, hub_cache):
        raise CiError("cache receipt environment has an unsafe cache lock path")
    if any(_paths_overlap(path, workspace) for path in (cache_root, hub_cache, lock_file)):
        raise CiError("cache receipt paths must not overlap the workflow checkout")
    if Path(str(warm["cache_root"])).resolve() != hub_cache:
        raise CiError("cache warm summary does not describe the configured Hub cache")
    if PurePosixPath(str(verify["cache_root"])) != _LOCAL_ONLY_HUB_CACHE:
        raise CiError(
            "cache local-only summary does not describe the read-only container Hub cache"
        )

    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "node_id": node_id,
        "hostname": hostname,
        "anchor_runner": anchor_runner,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "job_id": job_id,
        "source_revision": source_revision,
        "cache_plan_digest": warm["cache_plan_digest"],
        "resolved_cache_digest": warm["resolved_cache_digest"],
        "expected_count": warm["expected_count"],
        "present_count": verify["present_count"],
        "missing_count": verify["missing_count"],
        "cached_count": warm["cached_count"],
        "downloaded_count": warm["downloaded_count"],
        "warm_started_at": warm["started_at"],
        "warm_completed_at": warm["completed_at"],
        "verification_status": verify["status"],
        "verification_present_count": verify["present_count"],
        "verification_missing_count": verify["missing_count"],
        "verification_cached_count": verify["cached_count"],
        "verification_downloaded_count": verify["downloaded_count"],
        "verification_started_at": verify["started_at"],
        "verification_completed_at": verify["completed_at"],
        "cache_root": str(cache_root),
        "hub_cache": str(hub_cache),
        "cache_lock_file": str(lock_file),
        "status": "ready",
    }


def verify_receipts(
    receipts: list[dict[str, object]],
    expected_matrix: object,
    *,
    expected_run_id: str,
    expected_run_attempt: str,
    expected_job_id: str,
    expected_revision: str,
) -> dict[str, object]:
    _validated_positive_decimal(expected_run_id, "expected cache-warm workflow run ID")
    _validated_positive_decimal(expected_run_attempt, "expected cache-warm workflow run attempt")
    _validated_safe_id(expected_job_id, "expected cache-warm job ID")
    _validated_revision(expected_revision, "expected cache-warm source revision")
    if not isinstance(expected_matrix, dict):
        raise CiError("expected cache-warm matrix must be a JSON object")
    if set(expected_matrix) != {"include"}:
        raise CiError("expected cache-warm matrix fields are invalid")
    raw_entries = expected_matrix.get("include")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CiError("expected cache-warm matrix is invalid or empty")
    expected: set[str] = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise CiError("expected cache-warm matrix contains a non-object entry")
        if set(entry) != {"node_label"}:
            raise CiError("expected cache-warm matrix entry fields are invalid")
        node_label = entry.get("node_label")
        if not isinstance(node_label, str) or not node_label.startswith("trtmc-node-"):
            raise CiError("expected cache-warm matrix contains an invalid node label")
        node_id = node_label.removeprefix("trtmc-node-")
        _validated_safe_id(node_id, "expected cache-warm matrix node ID")
        if node_id in expected:
            raise CiError(f"expected cache-warm matrix repeats node {node_id!r}")
        expected.add(node_id)

    actual: dict[str, dict[str, object]] = {}
    hostname_nodes: dict[str, str] = {}
    runner_nodes: dict[str, str] = {}
    for receipt in receipts:
        if set(receipt) != _RECEIPT_FIELDS:
            missing_fields = sorted(_RECEIPT_FIELDS - set(receipt))
            unexpected_fields = sorted(set(receipt) - _RECEIPT_FIELDS)
            raise CiError(
                "cache-warm receipt field set is invalid: "
                f"missing={missing_fields!r}, unexpected={unexpected_fields!r}"
            )
        if (
            receipt.get("schema_version") != _RECEIPT_SCHEMA_VERSION
            or receipt.get("status") != "ready"
            or receipt.get("verification_status") != "passed"
        ):
            raise CiError("cache-warm receipt has an unsupported schema or is not ready")
        node_id = _validated_safe_id(receipt.get("node_id"), "cache-warm receipt node ID")
        anchor = _validated_safe_id(
            receipt.get("anchor_runner"), "cache-warm receipt anchor runner"
        )
        if node_id in actual:
            raise CiError(f"duplicate cache-warm receipt for node {node_id!r}")
        if node_id not in expected:
            raise CiError(f"cache-warm receipt node {node_id!r} is not declared")
        previous_runner_node = runner_nodes.setdefault(anchor, node_id)
        if previous_runner_node != node_id:
            raise CiError(f"cache-warm runner {anchor!r} maps to multiple node IDs")
        hostname = _validated_safe_id(
            receipt.get("hostname"), f"cache-warm receipt for {node_id!r} hostname"
        )
        previous_node = hostname_nodes.setdefault(hostname, node_id)
        if previous_node != node_id:
            raise CiError(f"cache-warm hostname {hostname!r} maps to multiple node IDs")
        run_id = _validated_positive_decimal(
            receipt.get("run_id"), f"cache-warm receipt for {node_id!r} run ID"
        )
        run_attempt = _validated_positive_decimal(
            receipt.get("run_attempt"),
            f"cache-warm receipt for {node_id!r} run attempt",
        )
        job_id = _validated_safe_id(
            receipt.get("job_id"), f"cache-warm receipt for {node_id!r} job ID"
        )
        source_revision = _validated_revision(
            receipt.get("source_revision"),
            f"cache-warm receipt for {node_id!r} source revision",
        )
        if run_id != expected_run_id:
            raise CiError(f"cache-warm receipt for {node_id!r} is from the wrong workflow run")
        if run_attempt != expected_run_attempt:
            raise CiError(
                f"cache-warm receipt for {node_id!r} is from the wrong workflow run attempt"
            )
        if job_id != expected_job_id:
            raise CiError(f"cache-warm receipt for {node_id!r} is from the wrong job")
        if source_revision != expected_revision:
            raise CiError(f"cache-warm receipt for {node_id!r} is from the wrong source revision")
        for key in ("cache_plan_digest", "resolved_cache_digest"):
            _validated_digest(receipt.get(key), f"cache-warm receipt {key}")
        counts = {
            key: _validated_count(
                receipt.get(key),
                f"cache-warm receipt for {node_id!r} {key}",
                positive=key == "expected_count",
            )
            for key in (
                "expected_count",
                "present_count",
                "missing_count",
                "cached_count",
                "downloaded_count",
                "verification_present_count",
                "verification_missing_count",
                "verification_cached_count",
                "verification_downloaded_count",
            )
        }
        expected_count = counts["expected_count"]
        if (
            counts["present_count"] != expected_count
            or counts["missing_count"] != 0
            or counts["cached_count"] + counts["downloaded_count"] != expected_count
            or counts["verification_present_count"] != expected_count
            or counts["verification_missing_count"] != 0
            or counts["verification_cached_count"] != expected_count
            or counts["verification_downloaded_count"] != 0
        ):
            raise CiError(f"cache-warm receipt for {node_id!r} is incomplete")
        warm_started_at = _validated_timestamp(
            receipt.get("warm_started_at"),
            f"cache-warm receipt for {node_id!r} warm_started_at",
        )
        warm_completed_at = _validated_timestamp(
            receipt.get("warm_completed_at"),
            f"cache-warm receipt for {node_id!r} warm_completed_at",
        )
        verification_started_at = _validated_timestamp(
            receipt.get("verification_started_at"),
            f"cache-warm receipt for {node_id!r} verification_started_at",
        )
        verification_completed_at = _validated_timestamp(
            receipt.get("verification_completed_at"),
            f"cache-warm receipt for {node_id!r} verification_completed_at",
        )
        if not (
            warm_started_at
            <= warm_completed_at
            <= verification_started_at
            <= verification_completed_at
        ):
            raise CiError(f"cache-warm receipt for {node_id!r} has invalid timestamp order")
        cache_root = _validated_path(
            receipt.get("cache_root"),
            f"cache-warm receipt for {node_id!r} cache root",
        )
        hub_cache = _validated_path(
            receipt.get("hub_cache"),
            f"cache-warm receipt for {node_id!r} Hub cache",
        )
        cache_lock_file = _validated_path(
            receipt.get("cache_lock_file"),
            f"cache-warm receipt for {node_id!r} cache lock file",
            file_path=True,
        )
        if hub_cache == cache_root or cache_root not in hub_cache.parents:
            raise CiError(f"cache-warm receipt for {node_id!r} has unsafe cache paths")
        if cache_lock_file == cache_root or _paths_overlap(cache_lock_file, hub_cache):
            raise CiError(f"cache-warm receipt for {node_id!r} has unsafe cache lock path")
        actual[node_id] = receipt

    if set(actual) != expected:
        raise CiError(
            f"cache-warm receipt node set mismatch: expected {sorted(expected)!r}, "
            f"found {sorted(actual)!r}"
        )
    for key in (
        "run_id",
        "run_attempt",
        "job_id",
        "source_revision",
        "cache_plan_digest",
        "resolved_cache_digest",
        "expected_count",
        "cache_root",
        "hub_cache",
        "cache_lock_file",
    ):
        values = {str(receipt[key]) for receipt in actual.values()}
        if len(values) != 1:
            raise CiError(f"cache-warm receipts disagree on {key}: {sorted(values)!r}")

    first = next(iter(actual.values()))
    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "status": "ready",
        "node_count": len(actual),
        "run_id": first["run_id"],
        "run_attempt": first["run_attempt"],
        "job_id": first["job_id"],
        "source_revision": first["source_revision"],
        "cache_plan_digest": first["cache_plan_digest"],
        "resolved_cache_digest": first["resolved_cache_digest"],
        "expected_count": first["expected_count"],
        "cache_root": first["cache_root"],
        "hub_cache": first["hub_cache"],
        "cache_lock_file": first["cache_lock_file"],
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--warm-summary", type=Path, required=True)
    create.add_argument("--verify-summary", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipts-dir", type=Path, required=True)
    verify.add_argument("--expected-matrix-json", required=True)
    verify.add_argument("--expected-run-id", required=True)
    verify.add_argument("--expected-run-attempt", required=True)
    verify.add_argument("--expected-job-id", required=True)
    verify.add_argument("--expected-revision", required=True)
    verify.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.command == "create":
        payload = create_receipt(
            _load_object(arguments.warm_summary, "cache warm summary"),
            _load_object(arguments.verify_summary, "cache local-only summary"),
            dict(os.environ),
        )
    else:
        try:
            expected_matrix = json.loads(arguments.expected_matrix_json)
        except json.JSONDecodeError as error:
            raise CiError(f"expected cache-warm matrix is invalid: {error}") from error
        paths = sorted(arguments.receipts_dir.rglob("cache-warm-receipt.json"))
        if not paths:
            raise CiError("no cache-warm receipts were downloaded")
        payload = verify_receipts(
            [_load_object(path, "cache-warm receipt") for path in paths],
            expected_matrix,
            expected_run_id=arguments.expected_run_id,
            expected_run_attempt=arguments.expected_run_attempt,
            expected_job_id=arguments.expected_job_id,
            expected_revision=arguments.expected_revision,
        )
    _write_json_atomic(arguments.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CiError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error

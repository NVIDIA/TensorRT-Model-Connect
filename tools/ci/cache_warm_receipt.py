# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and verify fail-closed per-node Nightly cache-warm receipts.

Boundary: evidence validation only; runner discovery and cache warming stay elsewhere.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import socket
import sys
from pathlib import Path

from .process import CiError


_DIGEST = re.compile(r"[0-9a-f]{64}")


def _load_object(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CiError(f"{description} is unavailable or invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CiError(f"{description} must be a JSON object: {path}")
    return payload


def _validated_summary(payload: dict[str, object], mode: str) -> dict[str, object]:
    if payload.get("schema_version") != 1 or payload.get("mode") != mode:
        raise CiError(f"cache {mode} summary has an unsupported schema or mode")
    if payload.get("status") != "passed":
        raise CiError(f"cache {mode} summary did not pass")
    for key in ("cache_plan_digest", "resolved_cache_digest"):
        value = payload.get(key)
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise CiError(f"cache {mode} summary has invalid {key}")
    if not isinstance(payload.get("source_revision"), str) or not payload["source_revision"]:
        raise CiError(f"cache {mode} summary has no source revision")
    cache_root = payload.get("cache_root")
    if not isinstance(cache_root, str) or not Path(cache_root).is_absolute():
        raise CiError(f"cache {mode} summary has an invalid cache root")
    for key in (
        "expected_count",
        "present_count",
        "missing_count",
        "cached_count",
        "downloaded_count",
    ):
        if not isinstance(payload.get(key), int) or int(payload[key]) < 0:
            raise CiError(f"cache {mode} summary has invalid {key}")
    if payload["missing_count"] != 0 or payload["present_count"] != payload["expected_count"]:
        raise CiError(f"cache {mode} summary is incomplete")
    if payload["cached_count"] + payload["downloaded_count"] != payload["expected_count"]:
        raise CiError(f"cache {mode} summary item counts are inconsistent")
    timestamps: list[datetime] = []
    for key in ("started_at", "completed_at"):
        try:
            value = datetime.fromisoformat(str(payload.get(key, "")))
        except ValueError as error:
            raise CiError(f"cache {mode} summary has invalid {key}") from error
        if value.tzinfo is None:
            raise CiError(f"cache {mode} summary has timezone-free {key}")
        timestamps.append(value)
    if timestamps[1] < timestamps[0]:
        raise CiError(f"cache {mode} summary completes before it starts")
    return payload


def create_receipt(
    warm_summary: dict[str, object],
    verify_summary: dict[str, object],
    environment: dict[str, str],
) -> dict[str, object]:
    warm = _validated_summary(warm_summary, "warm")
    verify = _validated_summary(verify_summary, "local-only")
    for key in ("source_revision", "cache_plan_digest", "resolved_cache_digest"):
        if warm.get(key) != verify.get(key):
            raise CiError(f"warm and local-only summaries disagree on {key}")
    for key in ("expected_count", "present_count", "missing_count"):
        if warm.get(key) != verify.get(key):
            raise CiError(f"warm and local-only summaries disagree on {key}")
    if verify["downloaded_count"] != 0:
        raise CiError("local-only cache verification must download zero items")

    required_environment = (
        "TRTMC_NODE_ID",
        "RUNNER_NAME",
        "GITHUB_RUN_ID",
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
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", environment["TRTMC_NODE_ID"]) is None:
        raise CiError("cache receipt has an unsafe node ID")
    if not environment["GITHUB_RUN_ID"].isdigit():
        raise CiError("cache receipt has an invalid workflow run ID")
    lock_file = Path(environment["TRTMC_HF_CACHE_LOCK_FILE"])
    if not lock_file.is_absolute() or lock_file == Path("/"):
        raise CiError("cache receipt has an unsafe cache lock path")
    if warm["source_revision"] != environment["TRTMC_CACHE_SOURCE_REVISION"]:
        raise CiError("cache summary source revision does not match the workflow")
    cache_root = Path(environment["TRTMC_HF_CACHE"]).resolve()
    hub_cache = Path(environment["TRTMC_HF_HUB_CACHE"]).resolve()
    workspace = Path(environment["TRTMC_CI_WORKSPACE"]).resolve()
    if cache_root == Path("/") or hub_cache == cache_root or cache_root not in hub_cache.parents:
        raise CiError("cache receipt environment has an unsafe cache root or hub path")
    if any(
        path == workspace or path in workspace.parents or workspace in path.parents
        for path in (cache_root, hub_cache)
    ):
        raise CiError("cache receipt paths must not overlap the workflow checkout")
    if Path(str(warm["cache_root"])).resolve() != hub_cache:
        raise CiError("cache warm summary does not describe the configured Hub cache")

    return {
        "schema_version": 1,
        "node_id": environment["TRTMC_NODE_ID"],
        "hostname": socket.gethostname(),
        "anchor_runner": environment["RUNNER_NAME"],
        "run_id": environment["GITHUB_RUN_ID"],
        "job_id": environment["GITHUB_JOB"],
        "source_revision": warm["source_revision"],
        "cache_plan_digest": warm["cache_plan_digest"],
        "resolved_cache_digest": warm["resolved_cache_digest"],
        "expected_count": warm["expected_count"],
        "present_count": verify["present_count"],
        "missing_count": verify["missing_count"],
        "cached_count": warm["cached_count"],
        "downloaded_count": warm["downloaded_count"],
        "warm_started_at": warm["started_at"],
        "warm_completed_at": warm["completed_at"],
        "verified_at": verify["completed_at"],
        "verification_downloaded_count": verify["downloaded_count"],
        "cache_root": str(cache_root),
        "hub_cache": str(hub_cache),
        "cache_lock_file": environment["TRTMC_HF_CACHE_LOCK_FILE"],
        "status": "ready",
    }


def verify_receipts(
    receipts: list[dict[str, object]],
    expected_matrix: dict[str, object],
) -> dict[str, object]:
    raw_entries = expected_matrix.get("include")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CiError("expected cache-anchor matrix is invalid or empty")
    expected: dict[str, str] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            raise CiError("expected cache-anchor matrix contains a non-object entry")
        node_label = entry.get("node_label")
        anchor = entry.get("anchor_runner")
        if not isinstance(node_label, str) or not node_label.startswith("trtmc-node-"):
            raise CiError("expected cache-anchor matrix contains an invalid node label")
        if not isinstance(anchor, str) or not anchor:
            raise CiError("expected cache-anchor matrix contains an invalid anchor")
        node_id = node_label.removeprefix("trtmc-node-")
        if node_id in expected:
            raise CiError(f"expected cache-anchor matrix repeats node {node_id!r}")
        expected[node_id] = anchor

    actual: dict[str, dict[str, object]] = {}
    for receipt in receipts:
        if receipt.get("schema_version") != 1 or receipt.get("status") != "ready":
            raise CiError("cache-warm receipt has an unsupported schema or is not ready")
        node_id = receipt.get("node_id")
        anchor = receipt.get("anchor_runner")
        if not isinstance(node_id, str) or not isinstance(anchor, str):
            raise CiError("cache-warm receipt has invalid node or anchor identity")
        if node_id in actual:
            raise CiError(f"duplicate cache-warm receipt for node {node_id!r}")
        if expected.get(node_id) != anchor:
            raise CiError(f"cache-warm receipt identity does not match matrix for {node_id!r}")
        if not isinstance(receipt.get("run_id"), str) or not str(receipt["run_id"]).isdigit():
            raise CiError(f"cache-warm receipt for {node_id!r} has invalid run identity")
        if not isinstance(receipt.get("source_revision"), str) or not receipt["source_revision"]:
            raise CiError(f"cache-warm receipt for {node_id!r} has no source revision")
        for key in ("cache_plan_digest", "resolved_cache_digest"):
            value = receipt.get(key)
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise CiError(f"cache-warm receipt has invalid {key}")
        expected_count = receipt.get("expected_count")
        if (
            not isinstance(expected_count, int)
            or expected_count < 0
            or receipt.get("present_count") != expected_count
            or receipt.get("missing_count") != 0
            or receipt.get("verification_downloaded_count") != 0
        ):
            raise CiError(f"cache-warm receipt for {node_id!r} is incomplete")
        actual[node_id] = receipt

    if set(actual) != set(expected):
        raise CiError(
            f"cache-warm receipt node set mismatch: expected {sorted(expected)!r}, "
            f"found {sorted(actual)!r}"
        )
    for key in ("run_id", "source_revision", "cache_plan_digest", "resolved_cache_digest"):
        values = {str(receipt[key]) for receipt in actual.values()}
        if len(values) != 1:
            raise CiError(f"cache-warm receipts disagree on {key}: {sorted(values)!r}")

    first = next(iter(actual.values()))
    return {
        "schema_version": 1,
        "status": "ready",
        "node_count": len(actual),
        "run_id": first["run_id"],
        "source_revision": first["source_revision"],
        "cache_plan_digest": first["cache_plan_digest"],
        "resolved_cache_digest": first["resolved_cache_digest"],
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
            raise CiError(f"expected cache-anchor matrix is invalid: {error}") from error
        paths = sorted(arguments.receipts_dir.rglob("cache-warm-receipt.json"))
        if not paths:
            raise CiError("no cache-warm receipts were downloaded")
        payload = verify_receipts(
            [_load_object(path, "cache-warm receipt") for path in paths],
            expected_matrix,
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

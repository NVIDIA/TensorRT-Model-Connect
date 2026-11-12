# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint bounded Community GPU recovery without executing PR code."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile

from tools import community_gpu_lease as evidence
from tools.brev_exec import ProvisionError, _load_lease, _save_lease, cleanup


WORKFLOW = ".github/workflows/community-gpu-recovery.yml"
JOB = "Community GPU / Recover leases"
ARTIFACT = "community-gpu-recovery-journal"
JOURNAL_FILE = "journal.json"
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 10_000
HEARTBEAT_SECONDS = 3600
LOOKBACK_DAYS = 35
LEASE_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "sku",
        "provider",
        "organization_id",
        "host_ram_gib",
        "requested_disk_gb",
        "instance_id",
        "phase",
        "allocation_pending",
        "create_started",
        "create_accepted",
        "run_id",
        "run_attempt",
        "intent_prepared",
        "rate_limit_retry_policy",
        "stock_bootstrap",
        "create_attempts",
        "create_diagnostics",
        "create_submission",
        "cleanup_confirmed",
        "cleanup_cursor",
        "cleanup_instances",
        "delete_accepted",
        "delete_attempted",
        "delete_outcome_unknown",
        "owned_instance_observed",
        "ownership_blocked",
    }
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _key(run_id: int, attempt: int) -> str:
    return f"{run_id}/{attempt}"


def _integer(value: object) -> bool:
    return type(value) is int and value > 0


def _runs(path: str, *, latest_only: bool = False) -> list[dict]:
    # Actions run objects are large: 100 per page exceeds the evidence
    # reader's 1 MiB bound. Paginate at 30 and never silently hit its 1,000 cap.
    rows = []
    separator = "&" if "?" in path else "?"
    for page in range(1, 35):
        response = evidence._api(f"{path}{separator}per_page=30&page={page}")
        batch = response.get("workflow_runs") if isinstance(response, dict) else None
        if not isinstance(batch, list) or not all(isinstance(row, dict) for row in batch):
            raise evidence.RecoveryError("Invalid workflow run inventory")
        if response.get("total_count", 0) >= 1000 and not latest_only:
            raise evidence.RecoveryError(
                "The workflow discovery window exceeds the API search limit"
            )
        rows.extend(batch)
        if latest_only or len(batch) < 30:
            return rows
    raise evidence.RecoveryError("The workflow run inventory is incomplete")


def _safe_lease(lease: dict) -> dict:
    """The journal contains recovery authority, never arbitrary provider output."""
    result = {key: copy.deepcopy(value) for key, value in lease.items() if key in LEASE_FIELDS}
    if "cleanup_instances" in result:
        result["cleanup_instances"] = {
            identity: _safe_lease(child) for identity, child in result["cleanup_instances"].items()
        }
    # Existing diagnostics are enum/hash-only. Keep the minimal retry history
    # required by the deletion contract, not diagnostics or probe output.
    result.pop("create_diagnostics", None)
    if "create_attempts" in result:
        result["create_attempts"] = [
            {key: item[key] for key in ("attempt", "outcome")} for item in result["create_attempts"]
        ]
    return result


def validate_journal(document: object, repository: str, repository_id: int) -> dict:
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or document.get("repository") != repository
        or not _integer(document.get("repository_id"))
        or document.get("repository_id") != repository_id
        or type(document.get("scan_complete")) is not bool
        or document.get("mode") not in {"observe", "apply"}
        or not isinstance(document.get("entries"), dict)
        or len(document["entries"]) > MAX_ENTRIES
        or not isinstance(document.get("seen_runs"), dict)
        or len(document["seen_runs"]) > MAX_ENTRIES
    ):
        raise evidence.RecoveryError("The recovery journal has an invalid schema or repository")
    evidence._time(document.get("updated_at"))
    with tempfile.TemporaryDirectory(prefix="trtmc-journal-validation-") as directory:
        for key, record in document["entries"].items():
            if not isinstance(record, dict):
                raise evidence.RecoveryError("The recovery journal has an invalid record")
            run, attempt = record.get("run_id"), record.get("run_attempt")
            if (
                not _integer(run)
                or not _integer(attempt)
                or key != _key(run, attempt)
                or not _integer(record.get("producer_id"))
                or not re.fullmatch(r"[0-9a-f]{40}", str(record.get("workflow_sha", "")))
                or record.get("state") not in {"unresolved", "confirmed", "missing_evidence"}
            ):
                raise evidence.RecoveryError(
                    "The recovery journal has conflicting producer identity"
                )
            evidence._time(record.get("producer_completed_at"))
            evidence._time(record.get("last_checked_at"))
            if record.get("lease") is None:
                if record["state"] != "missing_evidence" or record.get("origin_digest") is not None:
                    raise evidence.RecoveryError("A recovery record omitted its original lease")
                continue
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(record.get("origin_digest", ""))):
                raise evidence.RecoveryError(
                    "A recovery record omitted its original artifact digest"
                )
            lease = record["lease"]
            path = Path(directory) / evidence.LEASE_FILE
            _save_lease(path, lease)
            _load_lease(path, f"trtmc-gpu-ci-{run}-{attempt}")
            if _safe_lease(lease) != lease:
                raise evidence.RecoveryError(
                    "A recovery record contains unexpected operational fields"
                )
            if record["state"] == "confirmed" and lease.get("cleanup_confirmed") is not True:
                raise evidence.RecoveryError("A recovery receipt does not confirm deletion")
    return document


def _decode_checkpoint(archive: bytes, artifact: dict, repository: str, repository_id: int) -> dict:
    if len(archive) > MAX_JOURNAL_BYTES or "sha256:" + hashlib.sha256(
        archive
    ).hexdigest() != artifact.get("digest"):
        raise evidence.RecoveryError("The recovery journal archive failed its digest check")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            members = bundle.infolist()
            if (
                len(members) != 1
                or members[0].filename != JOURNAL_FILE
                or members[0].file_size > MAX_JOURNAL_BYTES
                or (members[0].external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise evidence.RecoveryError("The recovery journal archive has unexpected members")
            document = json.loads(bundle.read(members[0]))
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError, RecursionError):
        raise evidence.RecoveryError("The recovery journal archive is invalid") from None
    return validate_journal(document, repository, repository_id)


def load_journal(repository: str, *, bootstrap: bool = False) -> dict:
    """Read only a completed default-branch recovery job's immutable checkpoint.

    Original owner artifacts keep their existing provenance rule. Recovery
    receipts have this separate authority: a trusted main workflow and a
    successful upload in the completed recovery job's own execution window.
    """
    info = evidence._api(f"repos/{repository}")
    if (
        not isinstance(info, dict)
        or info.get("full_name") != repository
        or not _integer(info.get("id"))
    ):
        raise evidence.RecoveryError("The recovery repository identity is unavailable")

    def trusted(run):
        return (
            isinstance(run, dict)
            and run.get("path") == WORKFLOW
            and run.get("head_branch") == info.get("default_branch")
            and run.get("event") in {"schedule", "workflow_dispatch"}
            and run.get("repository", {}).get("id") == info["id"]
            and run.get("head_repository", {}).get("id") == info["id"]
        )

    # Artifact creation order also covers a new attempt of an old workflow
    # run. Ordering only by run ID would silently miss that newer receipt.
    for page in range(1, 21):
        response = evidence._api(
            f"repos/{repository}/actions/artifacts?name={ARTIFACT}&per_page=100&page={page}"
        )
        batch = response.get("artifacts") if isinstance(response, dict) else None
        if not isinstance(batch, list) or not all(isinstance(row, dict) for row in batch):
            raise evidence.RecoveryError("The recovery checkpoint inventory is invalid")
        for artifact in sorted(batch, key=lambda item: item.get("id", 0), reverse=True):
            if artifact.get("name") != ARTIFACT:
                continue
            provenance = artifact.get("workflow_run", {})
            if not _integer(provenance.get("id")):
                raise evidence.RecoveryError("The recovery checkpoint has no workflow identity")
            run = evidence._api(f"repos/{repository}/actions/runs/{provenance['id']}")
            if not trusted(run):
                continue
            jobs = evidence._rows(
                f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=all", "jobs"
            )
            producers = [
                job
                for job in jobs
                if (
                    job.get("name") == JOB
                    and job.get("status") == "completed"
                    and job.get("run_attempt") == 1
                )
            ]
            if not producers:
                # The upload can become visible just before the job completes.
                continue
            if len(producers) != 1:
                raise evidence.RecoveryError("The recovery checkpoint producer is ambiguous")
            producer = producers[0]
            uploads = [
                step
                for step in producer.get("steps", [])
                if step.get("name") == "Publish recovery journal"
            ]
            if (
                len(uploads) != 1
                or uploads[0].get("conclusion") != "success"
                or producer.get("run_id") != run["id"]
                or producer.get("head_sha") != run.get("head_sha")
                or artifact.get("expired") is not False
                or provenance.get("head_sha") != run.get("head_sha")
                or provenance.get("repository_id") != info["id"]
                or provenance.get("head_repository_id") != info["id"]
                or not _integer(artifact.get("id"))
                or not (
                    evidence._time(producer.get("started_at"))
                    <= evidence._time(artifact.get("created_at"))
                    <= evidence._time(producer.get("completed_at"))
                )
            ):
                raise evidence.RecoveryError("The recovery checkpoint provenance is invalid")
            archive = evidence._api(
                f"repos/{repository}/actions/artifacts/{artifact['id']}/zip",
                binary=True,
            )
            return _decode_checkpoint(archive, artifact, repository, info["id"])
        if len(batch) < 100:
            break
    else:
        raise evidence.RecoveryError(
            "The recovery checkpoint lookup exceeded its bounded inventory"
        )

    runs = _runs(
        f"repos/{repository}/actions/workflows/community-gpu-recovery.yml/runs",
        latest_only=True,
    )
    initialized_history = False
    for run in runs:
        if not trusted(run):
            continue
        jobs = evidence._rows(
            f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=all", "jobs"
        )
        for producer in jobs:
            if producer.get("name") != JOB or producer.get("status") != "completed":
                continue
            initialized_history |= any(
                step.get("name") == "Prepare recovery journal"
                and step.get("conclusion") == "success"
                for step in producer.get("steps", [])
            )
    if initialized_history or len(runs) >= 30 or not bootstrap:
        raise evidence.RecoveryError(
            "Recovery history is unavailable; refusing to forget outstanding leases"
        )
    return {
        "schema_version": 1,
        "repository": repository,
        "repository_id": info["id"],
        "updated_at": _stamp(_now()),
        "scan_complete": False,
        "mode": "observe",
        "entries": {},
        "seen_runs": {},
    }


def restore_record(journal: dict, producer: dict, destination: Path) -> bool:
    record = journal["entries"].get(_key(producer["run_id"], producer["run_attempt"]))
    if record is None or record.get("lease") is None:
        return False
    if (
        any(
            record.get(key) != producer.get(source)
            for key, source in (
                ("run_id", "run_id"),
                ("run_attempt", "run_attempt"),
                ("producer_id", "id"),
                ("workflow_sha", "head_sha"),
            )
        )
        or producer.get("status") != "completed"
    ):
        raise evidence.RecoveryError("The recovery receipt belongs to a different producer")
    temporary = destination.with_suffix(".unverified")
    _save_lease(temporary, record["lease"])
    _load_lease(temporary, f"trtmc-gpu-ci-{producer['run_id']}-{producer['run_attempt']}")
    authority = {
        key: record[key]
        for key in (
            "run_id",
            "run_attempt",
            "workflow_sha",
            "producer_id",
            "producer_completed_at",
            "origin_digest",
        )
    }
    authority.update(repository=journal["repository"], repository_id=journal["repository_id"])
    authority.update(
        base_lease_digest=hashlib.sha256(
            json.dumps(record["lease"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        base_checked_at=record["last_checked_at"],
    )
    try:
        _save_lease(destination.with_suffix(".authority.json"), authority)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _trusted_source(run: dict, repository: str, repository_id: int) -> bool:
    return (
        isinstance(run, dict)
        and run.get("event") == "workflow_dispatch"
        and run.get("path") == ".github/workflows/community-ci.yml"
        and run.get("head_branch") in {"main", "ci/developer"}
        and run.get("repository", {}).get("id") == repository_id
        and run.get("repository", {}).get("full_name") == repository
        and run.get("head_repository", {}).get("id") == repository_id
        and _integer(run.get("id"))
        and re.fullmatch(r"[0-9a-f]{40}", str(run.get("head_sha", ""))) is not None
    )


def _source_runs(repository: str, now: datetime) -> list[dict]:
    runs = {}
    day = (now - timedelta(days=LOOKBACK_DAYS)).date()
    while day <= now.date():
        path = f"repos/{repository}/actions/workflows/community-ci.yml/runs?event=workflow_dispatch&created={day}"
        rows = _runs(path)
        if len(rows) >= 1000:
            raise evidence.RecoveryError("The source inventory needs a narrower discovery window")
        runs.update((run["id"], run) for run in rows)
        day += timedelta(days=1)
    return sorted(runs.values(), key=lambda run: run["id"])


def _capture_producer(
    repository: str, run: dict, producer: dict, previous: dict | None
) -> dict | None:
    evidence.select_producer([producer], run["id"], producer["run_attempt"], run["head_sha"])
    reservations = [
        step for step in producer.get("steps", []) if step.get("name") == "Reserve a GPU instance"
    ]
    if len(reservations) == 1 and reservations[0].get("conclusion") == "skipped":
        return None
    if previous is not None and previous.get("lease") is not None:
        return previous
    record = {
        "run_id": run["id"],
        "run_attempt": producer["run_attempt"],
        "workflow_sha": run["head_sha"],
        "producer_id": producer["id"],
        "producer_completed_at": producer["completed_at"],
        "origin_digest": None,
        "lease": None,
        "state": "missing_evidence",
        "last_checked_at": "1970-01-01T00:00:00Z",
    }
    try:
        artifacts = evidence._rows(
            f"repos/{repository}/actions/runs/{run['id']}/artifacts", "artifacts"
        )
        artifact = evidence.select_artifact(artifacts, producer, run["repository"]["id"])
        archive = evidence._api(
            f"repos/{repository}/actions/artifacts/{artifact['id']}/zip", binary=True
        )
        lease = evidence.decode_lease(archive, artifact, producer, repository)
        with tempfile.TemporaryDirectory(prefix="trtmc-owner-lease-") as directory:
            path = Path(directory) / evidence.LEASE_FILE
            _save_lease(path, lease)
            _load_lease(path, lease["name"])
        record.update(
            lease=_safe_lease(lease),
            origin_digest=artifact["digest"],
            state="unresolved",
        )
    except (evidence.RecoveryError, ProvisionError, OSError):
        # A missing owner artifact is outstanding work, never absence proof.
        pass
    return record


def _compatible(original: dict, update: dict) -> bool:
    if any(
        original.get(key) != update.get(key)
        for key in (
            "run_id",
            "run_attempt",
            "producer_id",
            "workflow_sha",
            "producer_completed_at",
        )
    ):
        return False
    before, after = original.get("lease"), update.get("lease")
    if after is None:
        return False
    if before is None:
        return True
    if any(before.get(key) != after.get(key) for key in ("name", "sku", "organization_id")):
        return False
    if (
        before.get("instance_id") is not None
        and after.get("instance_id") is not None
        and before["instance_id"] != after["instance_id"]
    ):
        return False
    return True


def _merge_lease(before: dict, after: dict, *, newer: bool) -> dict:
    for field in ("name", "sku", "organization_id"):
        if before.get(field) != after.get(field):
            raise evidence.RecoveryError("Recovery progress changes original resource ownership")
    if (
        before.get("instance_id")
        and after.get("instance_id")
        and before["instance_id"] != after["instance_id"]
    ):
        raise evidence.RecoveryError("Recovery progress changes an authenticated instance ID")
    merged = copy.deepcopy(after if newer else before)
    merged["instance_id"] = before.get("instance_id") or after.get("instance_id")
    if merged["instance_id"]:
        merged["allocation_pending"] = False
    for field in (
        "delete_accepted",
        "delete_attempted",
        "owned_instance_observed",
        "cleanup_confirmed",
    ):
        if before.get(field) is True or after.get(field) is True:
            merged[field] = True
    if "cleanup_cursor" in before or "cleanup_cursor" in after:
        merged["cleanup_cursor"] = max(
            before.get("cleanup_cursor", 0), after.get("cleanup_cursor", 0)
        )
    children = copy.deepcopy(before.get("cleanup_instances", {}))
    for identity, child in after.get("cleanup_instances", {}).items():
        children[identity] = (
            _merge_lease(children[identity], child, newer=newer)
            if identity in children
            else copy.deepcopy(child)
        )
    if children:
        merged["cleanup_instances"] = children
        # A newly authenticated, different owned ID needs recovery even if an
        # earlier subset of this allocation had already been deleted.
        if any(not child.get("cleanup_confirmed") for child in children.values()):
            merged["cleanup_confirmed"] = False
    if merged.get("cleanup_confirmed"):
        receipt = before if before.get("cleanup_confirmed") else after
        merged["phase"] = receipt.get("phase", "deleted")
    return merged


def _merge_record(original: dict, update: dict) -> dict:
    if not _compatible(original, update):
        raise evidence.RecoveryError("The backstop receipt changes original resource ownership")
    if original.get("lease") is None:
        merged = copy.deepcopy(update)
    else:
        newer = evidence._time(update["last_checked_at"]) >= evidence._time(
            original["last_checked_at"]
        )
        merged = copy.deepcopy(update if newer else original)
        merged["lease"] = _merge_lease(original["lease"], update["lease"], newer=newer)
    with tempfile.TemporaryDirectory(prefix="trtmc-merged-progress-") as directory:
        path = Path(directory) / evidence.LEASE_FILE
        _save_lease(path, merged["lease"])
        _load_lease(path, f"trtmc-gpu-ci-{merged['run_id']}-{merged['run_attempt']}")
    merged["state"] = "confirmed" if merged["lease"].get("cleanup_confirmed") else "unresolved"
    return merged


def _read_backstop_updates(repository: str, run: dict, jobs: list[dict], journal: dict) -> None:
    backstops = [
        job
        for job in jobs
        if job.get("name") == "Community GPU / Cleanup" and job.get("status") == "completed"
    ]
    if not backstops:
        return
    artifacts = None
    for job in sorted(backstops, key=lambda item: item["run_attempt"]):
        uploads = [
            step for step in job.get("steps", []) if step.get("name") == "Publish recovery progress"
        ]
        if len(uploads) != 1 or uploads[0].get("conclusion") != "success":
            continue
        if artifacts is None:
            artifacts = evidence._rows(
                f"repos/{repository}/actions/runs/{run['id']}/artifacts", "artifacts"
            )
        name = f"community-gpu-recovery-{run['id']}-{job['run_attempt']}"
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        if not matches:
            # upload-artifact can succeed with if-no-files-found=ignore when
            # Reserve was skipped. An optional delta never replaces the
            # original producer's ownership evidence.
            continue
        if len(matches) != 1:
            raise evidence.RecoveryError("The backstop recovery receipt is missing or ambiguous")
        artifact = matches[0]
        provenance = artifact.get("workflow_run", {})
        if (
            artifact.get("expired") is not False
            or job.get("run_id") != run["id"]
            or job.get("head_sha") != run["head_sha"]
            or provenance.get("id") != run["id"]
            or provenance.get("head_sha") != run["head_sha"]
            or provenance.get("repository_id") != journal["repository_id"]
            or provenance.get("head_repository_id") != journal["repository_id"]
            or not _integer(artifact.get("id"))
            or not (
                evidence._time(job.get("started_at"))
                <= evidence._time(artifact.get("created_at"))
                <= evidence._time(job.get("completed_at"))
            )
        ):
            raise evidence.RecoveryError("The backstop recovery receipt has invalid provenance")
        archive = evidence._api(
            f"repos/{repository}/actions/artifacts/{artifact['id']}/zip", binary=True
        )
        update = _decode_checkpoint(archive, artifact, repository, journal["repository_id"])
        for key, record in update["entries"].items():
            original = journal["entries"].get(key)
            if (
                record["run_id"] != run["id"]
                or original is None
                or not _compatible(original, record)
            ):
                raise evidence.RecoveryError(
                    "The backstop receipt changes original resource ownership"
                )
            journal["entries"][key] = _merge_record(original, record)


def export_progress(
    repository: str, run_id: int, sha: str, directory: Path, destination: Path
) -> None:
    """Checkpoint locally authenticated evidence even when GitHub is unavailable."""
    records = {}
    repository_id = None
    for path in directory.glob(f"*/{evidence.LEASE_FILE}"):
        authority = json.loads(path.with_suffix(".authority.json").read_text())
        attempt = int(path.parent.name)
        if (
            authority.get("repository") != repository
            or authority.get("run_id") != run_id
            or authority.get("run_attempt") != attempt
            or authority.get("workflow_sha") != sha
        ):
            raise evidence.RecoveryError("Local recovery progress belongs to a different producer")
        if repository_id is not None and repository_id != authority.get("repository_id"):
            raise evidence.RecoveryError(
                "Local recovery progress has conflicting repository identity"
            )
        repository_id = authority["repository_id"]
        lease = _load_lease(path, f"trtmc-gpu-ci-{run_id}-{attempt}")
        record = {
            key: value
            for key, value in authority.items()
            if key not in {"repository", "repository_id", "base_lease_digest", "base_checked_at"}
        }
        unchanged = hashlib.sha256(
            json.dumps(lease, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() == authority.get("base_lease_digest")
        record.update(
            lease=_safe_lease(lease),
            last_checked_at=(
                authority.get("base_checked_at", "1970-01-01T00:00:00Z")
                if unchanged
                else _stamp(_now())
            ),
            state="confirmed" if lease.get("cleanup_confirmed") else "unresolved",
        )
        records[_key(run_id, attempt)] = record
    if not records:
        return
    document = {
        "schema_version": 1,
        "repository": repository,
        "repository_id": repository_id,
        "updated_at": _stamp(_now()),
        "scan_complete": False,
        "mode": "apply",
        "entries": records,
        "seen_runs": {},
    }
    validate_journal(document, repository, repository_id)
    _save_lease(destination, document)


def discover(journal: dict, runs: list[dict], *, limit: int = 100) -> None:
    repository = journal["repository"]
    runs = list(runs)
    present = {run["id"] for run in runs}
    errors = journal.setdefault("discovery_errors", {})
    # An error must remain retryable after the normal discovery lookback.
    # Do not clear an evidence gap merely because its original run is old.
    for identity in errors:
        if int(identity) in present:
            continue
        budget = evidence.EVIDENCE_DEADLINE.get()
        if budget is not None and time.monotonic() >= budget:
            break
        try:
            candidate = evidence._api(f"repos/{repository}/actions/runs/{identity}")
            if _trusted_source(candidate, repository, journal["repository_id"]):
                runs.append(candidate)
        except (evidence.RecoveryError, OSError):
            pass
    todo = []
    active = False
    for run in runs:
        if not _trusted_source(run, repository, journal["repository_id"]):
            continue
        marker = {
            key: run.get(key) for key in ("updated_at", "run_attempt", "status", "created_at")
        }
        missing = any(
            record["run_id"] == run["id"] and record["lease"] is None
            for record in journal["entries"].values()
        )
        if (
            all(
                journal["seen_runs"].get(str(run["id"]), {}).get(key) == value
                for key, value in marker.items()
            )
            and not missing
            and str(run["id"]) not in journal.get("discovery_errors", {})
            and run.get("status") == "completed"
        ):
            continue
        todo.append((run, marker))
    todo.sort(
        key=lambda item: (
            journal["seen_runs"]
            .get(str(item[0]["id"]), {})
            .get("checked_at", "1970-01-01T00:00:00Z"),
            item[0]["id"],
        )
    )
    journal["scan_complete"] = len(todo) <= limit
    for run, marker in todo[:limit]:
        budget = evidence.EVIDENCE_DEADLINE.get()
        if budget is not None and time.monotonic() >= budget:
            journal["scan_complete"] = False
            break
        try:
            _, current, jobs = evidence._context(
                repository, run["id"], run["run_attempt"], run["head_sha"]
            )
            if not _trusted_source(current, repository, journal["repository_id"]):
                raise evidence.RecoveryError("The source run changed its trusted identity")
            attempts = [
                job.get("run_attempt")
                for job in jobs
                if job.get("name") == evidence.OWNER_JOB and job.get("conclusion") != "skipped"
            ]
            if len(attempts) != len(set(attempts)):
                raise evidence.RecoveryError("The original GPU producer is ambiguous")
            for producer in jobs:
                if (
                    producer.get("name") != evidence.OWNER_JOB
                    or producer.get("conclusion") == "skipped"
                ):
                    continue
                if producer.get("status") != "completed":
                    active = True
                    continue
                key = _key(run["id"], producer["run_attempt"])
                record = _capture_producer(
                    repository, current, producer, journal["entries"].get(key)
                )
                if record is not None:
                    journal["entries"][key] = record
            _read_backstop_updates(repository, current, jobs, journal)
            errors.pop(str(run["id"]), None)
        except (evidence.RecoveryError, ProvisionError, OSError):
            errors[str(run["id"])] = "unconfirmed"
        finally:
            journal["seen_runs"][str(run["id"])] = {**marker, "checked_at": _stamp(_now())}
    journal["scan_complete"] = journal["scan_complete"] and not errors
    # A live owner is revisited on the next pass, but is never deleted here.
    journal["active_owners"] = active
    cutoff = _now() - timedelta(days=LOOKBACK_DAYS + 1)
    journal["seen_runs"] = {
        key: value
        for key, value in journal["seen_runs"].items()
        if key in errors or evidence._time(value["created_at"]) >= cutoff
    }
    journal["entries"] = {
        key: value
        for key, value in journal["entries"].items()
        if value["state"] != "confirmed" or evidence._time(value["producer_completed_at"]) >= cutoff
    }


def reconcile(journal: dict, directory: Path, deadline: float, *, apply: bool) -> None:
    records = sorted(journal["entries"].values(), key=lambda record: record["last_checked_at"])
    for record in records:
        if record["state"] == "confirmed" or record["lease"] is None:
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        if not apply:
            continue
        path = directory / str(record["run_id"]) / str(record["run_attempt"]) / evidence.LEASE_FILE
        # Spend little time on an undiscovered instance, and preserve the
        # cursor/IDs even if the API does not become healthy during this turn.
        per_lease = 90 if record["lease"].get("instance_id") else 5
        try:
            _save_lease(path, record["lease"])
            cleanup(
                record["lease"]["name"],
                lease_file=path,
                timeout=min(per_lease, remaining),
            )
        except (ProvisionError, OSError):
            record["state"] = "unresolved"
        finally:
            try:
                learned = _load_lease(path, record["lease"]["name"])
                if learned is not None:
                    record["lease"] = _safe_lease(learned)
            except (ProvisionError, OSError):
                record["last_error"] = "progress_unavailable"
            record["state"] = (
                "confirmed" if record["lease"].get("cleanup_confirmed") else "unresolved"
            )
            record["last_checked_at"] = _stamp(_now())


def check_admission(journal: dict, now: datetime, *, maximum: int = 5, cooldown: int = 900) -> None:
    if maximum < 1 or cooldown < 0:
        raise evidence.RecoveryError("Invalid GPU allocation limits")
    age = (now - evidence._time(journal["updated_at"])).total_seconds()
    if (
        age < -60
        or age > HEARTBEAT_SECONDS
        or not journal["scan_complete"]
        or journal["mode"] != "apply"
    ):
        raise evidence.RecoveryError(
            "GPU recovery is not ready or its heartbeat is stale; no allocation requested"
        )
    outstanding = [
        record for record in journal["entries"].values() if record["state"] != "confirmed"
    ]
    if len(outstanding) >= maximum:
        raise evidence.RecoveryError(
            "The unresolved GPU allocation limit was reached; no allocation requested"
        )
    if any(
        (now - evidence._time(record["producer_completed_at"])).total_seconds() < cooldown
        for record in outstanding
    ):
        raise evidence.RecoveryError("GPU allocation is cooling down after an unresolved request")


def check_pull(repository: str, pr: int, head: str) -> None:
    if not _integer(pr) or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise evidence.RecoveryError("Invalid PR allocation identity")
    pull = evidence._api(f"repos/{repository}/pulls/{pr}")
    if (
        not isinstance(pull, dict)
        or pull.get("state") != "open"
        or pull.get("head", {}).get("sha") != head
        or pull.get("base", {}).get("ref") != "main"
        or pull.get("base", {}).get("repo", {}).get("full_name") != repository
    ):
        raise evidence.RecoveryError(
            "This PR snapshot was superseded or closed; no allocation requested"
        )


def _apply_prior_receipts(
    journal: dict, directory: Path | None, current: int, attempt: int
) -> None:
    if directory is None:
        return
    for key, record in list(journal["entries"].items()):
        if record["run_id"] != current or record["run_attempt"] >= attempt:
            continue
        path = directory / str(record["run_attempt"]) / evidence.LEASE_FILE
        if not path.exists():
            continue
        authority = json.loads(path.with_suffix(".authority.json").read_text())
        expected = {
            "repository": journal["repository"],
            "repository_id": journal["repository_id"],
            **{
                field: record[field]
                for field in (
                    "run_id",
                    "run_attempt",
                    "producer_id",
                    "workflow_sha",
                    "origin_digest",
                )
            },
        }
        if any(authority.get(field) != value for field, value in expected.items()):
            raise evidence.RecoveryError("Prior cleanup receipt changes original ownership")
        lease = _load_lease(path, f"trtmc-gpu-ci-{current}-{record['run_attempt']}")
        if lease.get("cleanup_confirmed") is True:
            update = {
                **record,
                "lease": _safe_lease(lease),
                "last_checked_at": _stamp(_now()),
                "state": "confirmed",
            }
            journal["entries"][key] = _merge_record(record, update)


def admit(
    repository: str,
    pr: int,
    head: str,
    *,
    maximum: int,
    cooldown: int,
    prior_directory: Path | None = None,
) -> None:
    check_pull(repository, pr, head)
    journal = load_journal(repository)
    now = _now()
    current = int(os.environ["GITHUB_RUN_ID"])
    _apply_prior_receipts(
        journal, prior_directory, current, int(os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    )
    check_admission(journal, now, maximum=maximum, cooldown=cooldown)
    candidates = {}
    for day in ((now - timedelta(days=1)).date(), now.date()):
        path = f"repos/{repository}/actions/workflows/community-ci.yml/runs?event=workflow_dispatch&created={day}"
        candidates.update((run["id"], run) for run in _runs(path))
    for status in ("in_progress", "queued", "waiting", "pending"):
        path = f"repos/{repository}/actions/workflows/community-ci.yml/runs?event=workflow_dispatch&status={status}"
        candidates.update((run["id"], run) for run in _runs(path))
    # Long-running older owners also matter; they must not disappear merely
    # because their creation date is outside the small admission lookback.
    for identity, seen in journal["seen_runs"].items():
        if seen.get("status") != "completed" and int(identity) not in candidates:
            run = evidence._api(f"repos/{repository}/actions/runs/{identity}")
            candidates[run["id"]] = run
    for run in sorted(candidates.values(), key=lambda item: item["id"]):
        if run["id"] == current or not _trusted_source(run, repository, journal["repository_id"]):
            continue
        seen = journal["seen_runs"].get(str(run["id"]), {})
        if run.get("status") == "completed" and all(
            run.get(key) == seen.get(key) for key in ("updated_at", "run_attempt", "status")
        ):
            continue
        _, actual, jobs = evidence._context(
            repository, run["id"], run["run_attempt"], run["head_sha"]
        )
        for producer in jobs:
            if (
                producer.get("name") != evidence.OWNER_JOB
                or producer.get("conclusion") == "skipped"
            ):
                continue
            if producer.get("status") == "completed":
                key = _key(run["id"], producer["run_attempt"])
                record = _capture_producer(
                    repository, actual, producer, journal["entries"].get(key)
                )
                if record is not None:
                    journal["entries"][key] = record
                continue
            reservations = [
                step
                for step in producer.get("steps", [])
                if step.get("name") == "Reserve a GPU instance"
            ]
            if len(reservations) != 1:
                continue
            reservation = reservations[0]
            if reservation.get("conclusion") == "failure":
                raise evidence.RecoveryError(
                    "An earlier GPU request is unresolved; allocation is cooling down"
                )
    check_admission(journal, now, maximum=maximum, cooldown=cooldown)
    check_pull(repository, pr, head)


def run_round(repository: str, destination: Path, *, timeout: float, apply: bool) -> dict:
    if not math.isfinite(timeout) or timeout <= 0:
        raise evidence.RecoveryError("Recovery needs a finite positive budget")
    deadline = time.monotonic() + timeout
    token = evidence.EVIDENCE_DEADLINE.set(deadline)
    journal = None
    try:
        if destination.exists():
            prepared = json.loads(destination.read_text())
            repository_id = os.environ.get("GITHUB_REPOSITORY_ID")
            if repository_id is None:
                repository_id = evidence._api(f"repos/{repository}")["id"]
            journal = validate_journal(prepared, repository, int(repository_id))
        else:
            journal = load_journal(repository, bootstrap=True)
        journal["mode"] = "apply" if apply else "observe"
        journal["scan_complete"] = False
        # Preserve time for known resource deletion even when GitHub discovery
        # is degraded. A discovery outage cannot erase attested ownership.
        discovery_deadline = min(deadline, time.monotonic() + timeout * 0.4)
        discovery_token = evidence.EVIDENCE_DEADLINE.set(discovery_deadline)
        try:
            discover(journal, _source_runs(repository, _now()))
        except (evidence.RecoveryError, OSError):
            journal["scan_complete"] = False
            journal["discovery_degraded"] = True
        finally:
            evidence.EVIDENCE_DEADLINE.reset(discovery_token)
            # Checkpoint discovery before attempting provider operations.
            journal["updated_at"] = _stamp(_now())
            _save_lease(destination, journal)
        reconcile(journal, destination.parent / "leases", deadline, apply=apply)
        return journal
    finally:
        if journal is not None:
            journal["updated_at"] = _stamp(_now())
            _save_lease(destination, journal)
        evidence.EVIDENCE_DEADLINE.reset(token)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admit", action="store_true")
    parser.add_argument("--check-pr", action="store_true")
    parser.add_argument("--export-progress", type=Path)
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--prior-directory", type=Path)
    parser.add_argument("--head-sha")
    parser.add_argument("--journal-file", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--timeout", type=float, default=480)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    token = evidence.EVIDENCE_DEADLINE.set(time.monotonic() + args.timeout)
    try:
        repository = os.environ["GITHUB_REPOSITORY"]
        if args.export_progress:
            if args.journal_file is None:
                raise evidence.RecoveryError("A progress receipt path is required")
            export_progress(
                repository,
                int(os.environ["GITHUB_RUN_ID"]),
                os.environ["GITHUB_SHA"],
                args.export_progress,
                args.journal_file,
            )
        elif args.admit:
            admit(
                repository,
                args.pr_number,
                args.head_sha,
                maximum=int(os.environ.get("TRTMC_GPU_MAX_UNRESOLVED", "5")),
                cooldown=int(os.environ.get("TRTMC_GPU_ALLOCATION_COOLDOWN", "900")),
                prior_directory=args.prior_directory,
            )
        elif args.check_pr:
            check_pull(repository, args.pr_number, args.head_sha)
        else:
            if args.journal_file is None:
                raise evidence.RecoveryError("A journal checkpoint path is required")
            if args.initialize:
                _save_lease(args.journal_file, load_journal(repository, bootstrap=True))
                return 0
            document = run_round(
                repository, args.journal_file, timeout=args.timeout, apply=args.apply
            )
            outstanding = sum(
                record["state"] != "confirmed" for record in document["entries"].values()
            )
            print(
                f"Recovery checkpoint: unresolved={outstanding}, discovery_complete={document['scan_complete']}, mode={document['mode']}"
            )
            if outstanding:
                print(
                    "::warning::GPU allocations remain unresolved; the journal is preserved for the next bounded pass."
                )
        return 0
    except evidence.RecoveryError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 75
    except (KeyError, ValueError, TypeError, RuntimeError, OSError):
        print(
            "::error::GPU recovery or admission remains unconfirmed; inspect the recovery journal and runbook.",
            file=sys.stderr,
        )
        return 75
    finally:
        evidence.EVIDENCE_DEADLINE.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())

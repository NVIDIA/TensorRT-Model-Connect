# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover a GPU lease from its original trusted Actions producer."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

from tools.brev_exec import _load_lease, _save_lease


OWNER_JOB = "Community GPU / Provision and test"
LEASE_FILE = "trtmc-gpu-ci-lease.json"
MAX_ARCHIVE_BYTES = 1024 * 1024


class RecoveryError(RuntimeError):
    """Recovery did not establish an original allocation identity."""


def _api(path: str, *, binary: bool = False):
    try:
        result = subprocess.run(["gh", "api", path], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise RecoveryError("GitHub lease evidence is temporarily unavailable") from None
    if result.returncode or len(result.stdout) > MAX_ARCHIVE_BYTES:
        raise RecoveryError("GitHub lease evidence could not be read")
    if binary:
        return result.stdout
    try:
        return json.loads(result.stdout)
    except (ValueError, RecursionError):
        raise RecoveryError("GitHub lease evidence is invalid") from None


def _rows(path: str, key: str) -> list[dict]:
    rows = []
    separator = "&" if "?" in path else "?"
    for page in range(1, 101):
        document = _api(f"{path}{separator}per_page=100&page={page}")
        batch = document.get(key) if isinstance(document, dict) else None
        if not isinstance(batch, list) or not all(isinstance(row, dict) for row in batch):
            raise RecoveryError("GitHub lease evidence has an invalid list")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise RecoveryError("GitHub lease evidence exceeded the bounded inventory")


def _time(value: object) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is not None:
            return result
    except ValueError:
        pass
    raise RecoveryError("GitHub producer timestamps are unavailable")


def select_producer(jobs: list[dict], run_id: int, attempt: int, sha: str) -> dict:
    """A cleanup-only rerun must use the attempt of the job that allocated."""
    candidates = []
    for job in jobs:
        if job.get("name") != OWNER_JOB or job.get("conclusion") == "skipped":
            continue
        if (
            type(job.get("run_attempt")) is not int
            or job["run_attempt"] < 1
            or job.get("run_id") != run_id
            or job.get("head_sha") != sha
        ):
            raise RecoveryError("GPU producer provenance does not match this workflow")
        if job["run_attempt"] <= attempt:
            candidates.append(job)
    if not candidates:
        raise RecoveryError("The original GPU producer is unavailable")
    producer_attempt = max(job["run_attempt"] for job in candidates)
    selected = [job for job in candidates if job["run_attempt"] == producer_attempt]
    if len(selected) != 1 or selected[0].get("status") != "completed":
        raise RecoveryError("The original GPU producer is ambiguous or still running")
    return selected[0]


def select_artifact(artifacts: list[dict], producer: dict, repository_id: int) -> dict:
    run_id, attempt = producer["run_id"], producer["run_attempt"]
    # The final receipt has a separate immutable artifact. Replacing the initial
    # artifact would create an avoidable gap if the runner died during upload.
    for name in (f"gpu-ci-lease-final-{run_id}-{attempt}", f"gpu-ci-lease-{run_id}-{attempt}"):
        matches = [artifact for artifact in artifacts if artifact.get("name") == name]
        if not matches:
            continue
        if len(matches) != 1:
            raise RecoveryError("The original GPU lease artifact is ambiguous")
        artifact = matches[0]
        provenance = artifact.get("workflow_run", {})
        if (
            artifact.get("expired") is not False
            or type(artifact.get("id")) is not int
            or artifact["id"] < 1
            or not isinstance(provenance, dict)
            or provenance.get("id") != run_id
            or provenance.get("head_sha") != producer["head_sha"]
            or provenance.get("repository_id") != repository_id
            or provenance.get("head_repository_id") != repository_id
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(artifact.get("digest", "")))
            or not (
                _time(producer.get("started_at"))
                <= _time(artifact.get("created_at"))
                <= _time(producer.get("completed_at"))
            )
        ):
            raise RecoveryError("GPU lease artifact provenance is invalid")
        return artifact
    raise RecoveryError("The original GPU lease artifact is unavailable")


def decode_lease(archive: bytes, artifact: dict, producer: dict) -> dict:
    if (
        len(archive) > MAX_ARCHIVE_BYTES
        or "sha256:" + hashlib.sha256(archive).hexdigest() != artifact["digest"]
    ):
        raise RecoveryError("GPU lease archive digest does not match GitHub")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            entries = bundle.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != LEASE_FILE
                or entries[0].file_size > 64 * 1024
                or (entries[0].external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise RecoveryError("GPU lease archive has an unexpected member")
            lease = json.loads(bundle.read(entries[0]))
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError, RecursionError):
        raise RecoveryError("GPU lease archive is invalid") from None
    run_id, attempt = producer["run_id"], producer["run_attempt"]
    if not isinstance(lease, dict) or lease.get("name") != f"trtmc-gpu-ci-{run_id}-{attempt}":
        raise RecoveryError("GPU lease does not belong to its producer attempt")
    for field, expected in (("run_id", run_id), ("run_attempt", attempt)):
        if field in lease and str(lease[field]) != str(expected):
            raise RecoveryError("GPU lease has conflicting original attempt metadata")
        lease[field] = expected
    return lease


def recover(repository: str, run_id: int, attempt: int, sha: str, destination: Path) -> dict:
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or run_id < 1
        or attempt < 1
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
    ):
        raise RecoveryError("Invalid trusted workflow identity")
    prefix = f"repos/{repository}/actions/runs/{run_id}"
    run = _api(prefix)
    if (
        not isinstance(run, dict)
        or run.get("id") != run_id
        or run.get("head_sha") != sha
        or run.get("path") != ".github/workflows/community-ci.yml"
        or not isinstance(run.get("repository"), dict)
        or run["repository"].get("full_name") != repository
        or type(run["repository"].get("id")) is not int
    ):
        raise RecoveryError("The workflow run does not match the trusted cleanup context")
    producer = select_producer(_rows(prefix + "/jobs?filter=all", "jobs"), run_id, attempt, sha)
    steps = producer.get("steps", [])
    reservations = [step for step in steps if step.get("name") == "Reserve a GPU instance"]
    if len(reservations) == 1 and reservations[0].get("conclusion") == "skipped":
        return {"cleanup_confirmed": "true"}
    artifact = select_artifact(
        _rows(prefix + "/artifacts", "artifacts"), producer, run["repository"]["id"]
    )
    archive = _api(f"repos/{repository}/actions/artifacts/{artifact['id']}/zip", binary=True)
    lease = decode_lease(archive, artifact, producer)
    # Reuse the deletion helper's full schema validation before publishing any
    # output. The temporary file is never used as authority after a failed check.
    temporary = destination.with_name(destination.name + ".unverified")
    try:
        _save_lease(temporary, lease)
        _load_lease(temporary, lease["name"])
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"instance_name": lease["name"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease-file", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = recover(
            os.environ["GITHUB_REPOSITORY"],
            int(os.environ["GITHUB_RUN_ID"]),
            int(os.environ["GITHUB_RUN_ATTEMPT"]),
            os.environ["GITHUB_SHA"],
            args.lease_file,
        )
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            for key, value in result.items():
                output.write(f"{key}={value}\n")
        return 0
    except (KeyError, ValueError, RuntimeError, OSError):
        print("ERROR: Original GPU lease recovery remains unconfirmed.", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())

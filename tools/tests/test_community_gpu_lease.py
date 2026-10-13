# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial recovery tests for cancelled owners and cleanup-only reruns."""

import copy
import hashlib
import io
import json
import zipfile

import pytest

from tools import community_gpu_lease as recovery


SHA = "a" * 40
REPOSITORY = "example/project"


def _producer(attempt=1):
    return {
        "id": 100 + attempt,
        "run_id": 123,
        "run_attempt": attempt,
        "name": recovery.OWNER_JOB,
        "head_sha": SHA,
        "status": "completed",
        "conclusion": "cancelled",
        "started_at": "2026-01-01T01:00:00Z",
        "completed_at": "2026-01-01T07:00:00Z",
        "steps": [{"name": "Reserve a GPU instance", "conclusion": "failure"}],
    }


def _archive(lease, filename=recovery.LEASE_FILE):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(filename, json.dumps(lease))
    return stream.getvalue()


def _fixture(attempt=1, *, final=False):
    lease = {
        "schema_version": 1,
        "name": f"trtmc-gpu-ci-123-{attempt}",
        "sku": "gpu-fixture.1",
        "organization_id": "org-fixture",
        "instance_id": "vm-fixture",
        "create_started": True,
        "create_accepted": True,
        "allocation_pending": False,
    }
    archive = _archive(lease)
    artifact = {
        "id": 200 + attempt,
        "name": f"gpu-ci-lease-{'final-' if final else ''}123-{attempt}",
        "expired": False,
        "created_at": "2026-01-01T02:00:00Z",
        "digest": "sha256:" + hashlib.sha256(archive).hexdigest(),
        "workflow_run": {
            "id": 123,
            "head_sha": SHA,
            "repository_id": 5,
            "head_repository_id": 5,
        },
    }
    return lease, artifact, archive


def _api_fixture(monkeypatch, *, jobs=None, artifacts=None, archives=None):
    _, default_artifact, default_archive = _fixture()
    calls = []
    responses = {
        f"repos/{REPOSITORY}/actions/runs/123": {
            "id": 123,
            "head_sha": SHA,
            "path": ".github/workflows/community-ci.yml",
            "repository": {"id": 5, "full_name": REPOSITORY},
        },
        f"repos/{REPOSITORY}/actions/runs/123/jobs?filter=all&per_page=100&page=1": {
            "jobs": jobs if jobs is not None else [_producer()]
        },
        f"repos/{REPOSITORY}/actions/runs/123/artifacts?per_page=100&page=1": {
            "artifacts": artifacts if artifacts is not None else [default_artifact]
        },
        f"repos/{REPOSITORY}/actions/artifacts/201/zip": default_archive,
    }
    if archives:
        responses.update(archives)

    def api(path, *, binary=False):
        calls.append(path)
        return copy.deepcopy(responses[path])

    monkeypatch.setattr(recovery, "_api", api)
    return calls, responses


@pytest.mark.parametrize("attempt", [1, 2, 7])
def test_timeout_without_job_outputs_recovers_original_producer(monkeypatch, tmp_path, attempt):
    calls, _ = _api_fixture(monkeypatch)
    path = tmp_path / "lease.json"
    # Cleanup reruns receive a later GITHUB_RUN_ATTEMPT, but this workflow's
    # cancelled allocation job actually ran on attempt 1. No needs outputs exist.
    result = recovery.recover(REPOSITORY, 123, attempt, SHA, path)
    assert result == {"instance_name": "trtmc-gpu-ci-123-1"}
    lease = json.loads(path.read_text())
    assert lease["run_id"] == 123 and lease["run_attempt"] == 1
    assert lease["instance_id"] == "vm-fixture"
    assert calls[-1].endswith("/201/zip")


def test_missing_latest_lease_never_falls_back_to_an_older_allocation(monkeypatch, tmp_path):
    _api_fixture(monkeypatch, jobs=[_producer(1), _producer(2)])
    path = tmp_path / "lease.json"
    with pytest.raises(recovery.RecoveryError, match="unavailable"):
        recovery.recover(REPOSITORY, 123, 3, SHA, path)
    assert not path.exists()


def test_cleanup_rerun_ignores_a_skipped_owner_placeholder(monkeypatch, tmp_path):
    skipped = _producer(2)
    skipped.update(conclusion="skipped", steps=[])
    _api_fixture(monkeypatch, jobs=[_producer(1), skipped])
    result = recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")
    assert result == {"instance_name": "trtmc-gpu-ci-123-1"}


def test_newer_artifact_cannot_select_the_allocation_attempt(monkeypatch, tmp_path):
    _, artifact, _ = _fixture(2)
    _api_fixture(monkeypatch, artifacts=[artifact])
    with pytest.raises(recovery.RecoveryError, match="unavailable"):
        recovery.recover(REPOSITORY, 123, 3, SHA, tmp_path / "lease.json")


def test_final_receipt_is_preferred_without_replacing_original(monkeypatch, tmp_path):
    _, initial, _ = _fixture()
    _, final, archive = _fixture(final=True)
    final["id"] = 301
    calls, _ = _api_fixture(
        monkeypatch,
        artifacts=[initial, final],
        archives={f"repos/{REPOSITORY}/actions/artifacts/301/zip": archive},
    )
    recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")
    assert calls[-1].endswith("/301/zip")


@pytest.mark.parametrize(
    "mutation",
    [
        {"expired": True},
        {"digest": None},
        {"created_at": "2026-01-01T00:59:59Z"},
        {"created_at": "2026-01-01T07:00:01Z"},
        {"workflow_run": {"id": 999}},
    ],
)
def test_artifact_provenance_must_match_its_owner(monkeypatch, tmp_path, mutation):
    _, artifact, _ = _fixture()
    artifact.update(mutation)
    _api_fixture(monkeypatch, artifacts=[artifact])
    path = tmp_path / "lease.json"
    with pytest.raises(recovery.RecoveryError, match="provenance"):
        recovery.recover(REPOSITORY, 123, 2, SHA, path)
    assert not path.exists()


def test_ambiguous_lease_artifacts_fail_closed(monkeypatch, tmp_path):
    _, artifact, _ = _fixture()
    duplicate = {**artifact, "id": 999}
    _api_fixture(monkeypatch, artifacts=[artifact, duplicate])
    with pytest.raises(recovery.RecoveryError, match="ambiguous"):
        recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")


@pytest.mark.parametrize(
    "mutation",
    [{"head_sha": "b" * 40}, {"run_id": 999}, {"status": "in_progress"}],
)
def test_foreign_or_running_producer_is_not_cleanup_authority(monkeypatch, tmp_path, mutation):
    producer = _producer()
    producer.update(mutation)
    _api_fixture(monkeypatch, jobs=[producer])
    with pytest.raises(recovery.RecoveryError):
        recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")


def test_ambiguous_producers_fail_closed(monkeypatch, tmp_path):
    _api_fixture(monkeypatch, jobs=[_producer(), _producer()])
    with pytest.raises(recovery.RecoveryError, match="ambiguous"):
        recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")


def test_skipped_reservation_is_proven_by_github_when_outputs_are_lost(monkeypatch, tmp_path):
    producer = _producer()
    producer["steps"][0]["conclusion"] = "skipped"
    calls, _ = _api_fixture(monkeypatch, jobs=[producer])
    result = recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json")
    assert result == {"cleanup_confirmed": "true"}
    assert not any("artifacts" in call for call in calls)


@pytest.mark.parametrize(
    "mutation",
    [
        {"name": "trtmc-gpu-ci-999-1"},
        {"name": "trtmc-gpu-ci-123-2"},
        {"run_id": 999},
        {"run_attempt": 2},
        {"schema_version": 99},
        {"organization_id": "org\ninjection"},
        {"create_started": "true"},
    ],
)
def test_corrupt_lease_never_becomes_a_cleanup_input(monkeypatch, tmp_path, mutation):
    lease, artifact, _ = _fixture()
    lease.update(mutation)
    archive = _archive(lease)
    artifact["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
    _api_fixture(
        monkeypatch,
        artifacts=[artifact],
        archives={f"repos/{REPOSITORY}/actions/artifacts/201/zip": archive},
    )
    destination = tmp_path / "lease.json"
    with pytest.raises(RuntimeError):
        recovery.recover(REPOSITORY, 123, 2, SHA, destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_archive_integrity_is_checked_before_reading_lease():
    _, artifact, archive = _fixture()
    with pytest.raises(recovery.RecoveryError, match="digest"):
        recovery.decode_lease(archive + b"altered", artifact, _producer())


@pytest.mark.parametrize("filename", ["../lease.json", "nested/trtmc-gpu-ci-lease.json"])
def test_archive_members_are_never_extracted_to_arbitrary_paths(filename):
    lease, artifact, _ = _fixture()
    archive = _archive(lease, filename)
    artifact["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
    with pytest.raises(recovery.RecoveryError):
        recovery.decode_lease(archive, artifact, _producer())


def test_recovery_failure_cannot_publish_an_instance_name(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr("sys.argv", ["recovery", "--lease-file", str(tmp_path / "lease")])
    _api_fixture(monkeypatch, artifacts=[])
    assert recovery.main() == 75
    assert not output.exists()

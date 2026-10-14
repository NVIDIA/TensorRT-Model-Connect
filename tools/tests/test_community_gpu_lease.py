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


def _intent_fixture(attempt=1):
    lease, artifact, _ = _fixture(attempt)
    lease.update(
        phase="prepared",
        intent_prepared=True,
        instance_id=None,
        create_started=True,
        create_accepted=False,
        allocation_pending=True,
    )
    archive = _archive(lease)
    artifact.update(
        name=f"gpu-ci-intent-123-{attempt}",
        digest="sha256:" + hashlib.sha256(archive).hexdigest(),
    )
    return lease, artifact, archive


def test_hard_killed_reservation_recovers_preexisting_intent(monkeypatch, tmp_path):
    lease, artifact, archive = _intent_fixture()
    calls, _ = _api_fixture(
        monkeypatch,
        artifacts=[artifact],
        archives={f"repos/{REPOSITORY}/actions/artifacts/201/zip": archive},
    )
    destination = tmp_path / "lease.json"
    result = recovery.recover(REPOSITORY, 123, 2, SHA, destination)
    assert result == {"instance_name": "trtmc-gpu-ci-123-1"}
    recovered = json.loads(destination.read_text())
    assert recovered["create_started"] and recovered["allocation_pending"]
    assert recovered["instance_id"] is None and not recovered.get("cleanup_confirmed", False)
    assert calls[-1].endswith("/201/zip")


@pytest.mark.parametrize("intent_uploaded", [False, True])
def test_cancelled_before_post_is_closed_only_by_authoritative_skipped_step(
    monkeypatch, tmp_path, intent_uploaded
):
    producer = _producer()
    producer["steps"][0]["conclusion"] = "skipped"
    _, artifact, archive = _intent_fixture()
    calls, _ = _api_fixture(
        monkeypatch,
        jobs=[producer],
        artifacts=[artifact] if intent_uploaded else [],
        archives={f"repos/{REPOSITORY}/actions/artifacts/201/zip": archive},
    )
    cleanups = []
    monkeypatch.setattr(recovery, "cleanup", lambda *a, **kw: cleanups.append((a, kw)))
    result = recovery.cleanup_attempts(REPOSITORY, 123, 2, SHA, tmp_path, prior_only=False)
    assert result == {"cleanup_confirmed": "true"}
    assert not cleanups and not any("artifacts" in call for call in calls)


@pytest.mark.parametrize("conclusion", ["cancelled", "failure", None])
def test_reservation_that_may_have_started_never_turns_intent_into_no_submission(
    monkeypatch, tmp_path, conclusion
):
    producer = _producer()
    producer["steps"][0]["conclusion"] = conclusion
    _, artifact, archive = _intent_fixture()
    _api_fixture(
        monkeypatch,
        jobs=[producer],
        artifacts=[artifact],
        archives={f"repos/{REPOSITORY}/actions/artifacts/201/zip": archive},
    )
    calls = []

    def unknown(name, *, lease_file, timeout):
        calls.append(name)
        lease = json.loads(lease_file.read_text())
        assert timeout == 90 and lease["allocation_pending"] and lease["create_started"]
        raise RuntimeError("unknown submission remains unresolved")

    monkeypatch.setattr(recovery, "cleanup", unknown)
    with pytest.raises(RuntimeError, match="unresolved"):
        recovery.cleanup_attempts(REPOSITORY, 123, 2, SHA, tmp_path, prior_only=True)
    assert calls == ["trtmc-gpu-ci-123-1"]


def test_full_owner_rerun_releases_every_older_producer_before_new_allocation(
    monkeypatch, tmp_path
):
    artifacts, archives = [], {}
    for attempt in (1, 2):
        _, artifact, archive = _fixture(attempt)
        artifacts.append(artifact)
        archives[f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"] = archive
    current = _producer(3)
    current.update(status="in_progress", conclusion=None, completed_at=None)
    _api_fixture(
        monkeypatch,
        jobs=[_producer(2), current, _producer(1)],
        artifacts=artifacts,
        archives=archives,
    )
    calls = []

    def deleted(name, *, lease_file, timeout):
        assert timeout == 90 and json.loads(lease_file.read_text())["name"] == name
        calls.append(name)

    monkeypatch.setattr(recovery, "cleanup", deleted)
    result = recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=True)
    assert result == {"cleanup_confirmed": "true"}
    assert calls == ["trtmc-gpu-ci-123-1", "trtmc-gpu-ci-123-2"]


def test_cleanup_only_rerun_releases_all_original_attempts_not_only_latest(monkeypatch, tmp_path):
    artifacts, archives = [], {}
    for attempt in (1, 2):
        _, artifact, archive = _fixture(attempt)
        artifacts.append(artifact)
        archives[f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"] = archive
    placeholder = _producer(3)
    placeholder.update(conclusion="skipped", steps=[])
    _api_fixture(
        monkeypatch,
        jobs=[_producer(1), _producer(2), placeholder],
        artifacts=artifacts,
        archives=archives,
    )
    calls = []
    monkeypatch.setattr(recovery, "cleanup", lambda name, **kw: calls.append(name))
    assert recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=False) == {
        "cleanup_confirmed": "true"
    }
    assert calls == ["trtmc-gpu-ci-123-1", "trtmc-gpu-ci-123-2"]


def test_newer_skipped_reservation_cannot_hide_an_older_unconfirmed_allocation(
    monkeypatch, tmp_path
):
    latest = _producer(2)
    latest["steps"][0]["conclusion"] = "skipped"
    _, artifact, archive = _intent_fixture(1)
    _api_fixture(
        monkeypatch,
        jobs=[_producer(1), latest],
        artifacts=[artifact],
        archives={f"repos/{REPOSITORY}/actions/artifacts/201/zip": archive},
    )
    monkeypatch.setattr(
        recovery,
        "cleanup",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("older allocation unconfirmed")),
    )
    with pytest.raises(RuntimeError, match="older allocation"):
        recovery.cleanup_attempts(REPOSITORY, 123, 2, SHA, tmp_path, prior_only=False)


@pytest.mark.parametrize("fault", ["running", "different_sha", "duplicate", "missing_artifact"])
def test_any_unverified_previous_owner_blocks_new_allocation(monkeypatch, tmp_path, fault):
    older = _producer(1)
    _, latest_artifact, latest_archive = _fixture(2)
    _, older_artifact, _ = _fixture(1)
    jobs = [older, _producer(2)]
    artifacts = [older_artifact, latest_artifact]
    if fault == "running":
        older["status"] = "in_progress"
    elif fault == "different_sha":
        older["head_sha"] = "b" * 40
    elif fault == "duplicate":
        jobs.append(copy.deepcopy(older))
    else:
        artifacts = [latest_artifact]
    _api_fixture(
        monkeypatch,
        jobs=jobs,
        artifacts=artifacts,
        archives={f"repos/{REPOSITORY}/actions/artifacts/202/zip": latest_archive},
    )
    calls = []
    monkeypatch.setattr(recovery, "cleanup", lambda name, **kw: calls.append(name))

    def stop_observing(_seconds):
        raise recovery.RecoveryError("fixture observation ended with cleanup unconfirmed")

    monkeypatch.setattr(recovery.time, "sleep", stop_observing)
    with pytest.raises(recovery.RecoveryError):
        recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=True)
    # Bad evidence blocks a new POST, but cannot strand an independently
    # verified VM from another completed attempt.
    assert calls == ["trtmc-gpu-ci-123-2"]


def test_initial_attempt_has_no_previous_allocation_to_reconcile(monkeypatch, tmp_path):
    def forbidden(*a, **kw):
        raise AssertionError("first attempt must not query or delete any older instance")

    monkeypatch.setattr(recovery, "_api", forbidden)
    monkeypatch.setattr(recovery, "cleanup", forbidden)
    assert recovery.cleanup_attempts(REPOSITORY, 123, 1, SHA, tmp_path, prior_only=True) == {
        "cleanup_confirmed": "true"
    }


def test_known_owned_vm_is_released_before_waiting_for_an_older_unknown_create(
    monkeypatch, tmp_path
):
    _, older, older_archive = _intent_fixture(1)
    _, later, later_archive = _fixture(2)
    _api_fixture(
        monkeypatch,
        jobs=[_producer(1), _producer(2)],
        artifacts=[older, later],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": older_archive,
            f"repos/{REPOSITORY}/actions/artifacts/202/zip": later_archive,
        },
    )
    calls = []

    def cleanup(name, *, lease_file, timeout):
        calls.append(name)
        assert timeout == 90
        if json.loads(lease_file.read_text())["instance_id"] is None:
            raise RuntimeError("unknown allocation remains unresolved")

    monkeypatch.setattr(recovery, "cleanup", cleanup)
    with pytest.raises(RuntimeError, match="unresolved"):
        recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=False)
    assert calls == ["trtmc-gpu-ci-123-2", "trtmc-gpu-ci-123-1"]


def test_transient_github_inventory_failure_is_retried_before_releasing_the_original_owner(
    monkeypatch, tmp_path
):
    calls, _ = _api_fixture(monkeypatch)
    api = recovery._api
    failures = 1
    sleeps = []

    def flaky(path, *, binary=False):
        nonlocal failures
        if failures:
            failures -= 1
            raise recovery.RecoveryError("temporary provider error")
        return api(path, binary=binary)

    monkeypatch.setattr(recovery, "_api", flaky)
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    deleted = []
    monkeypatch.setattr(recovery, "cleanup", lambda name, **kw: deleted.append(name))
    assert recovery.cleanup_attempts(REPOSITORY, 123, 2, SHA, tmp_path, prior_only=True) == {
        "cleanup_confirmed": "true"
    }
    assert sleeps == [recovery.POLL_INTERVAL] and calls
    assert deleted == ["trtmc-gpu-ci-123-1"]


def test_one_unresponsive_known_identity_cannot_starve_another_owned_vm(monkeypatch, tmp_path):
    _, first, first_archive = _fixture(1)
    _, second, second_archive = _fixture(2)
    calls, _ = _api_fixture(
        monkeypatch,
        jobs=[_producer(1), _producer(2)],
        artifacts=[first, second],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": first_archive,
            f"repos/{REPOSITORY}/actions/artifacts/202/zip": second_archive,
        },
    )
    observed, sleeps = [], []

    def cleanup(name, *, lease_file, timeout):
        assert timeout == 90
        observed.append(name)
        if len(observed) == 1:
            # The bounded helper may learn the ID during its first pass. That
            # state must survive later metadata outages and further passes.
            lease = json.loads(lease_file.read_text())
            lease["first_pass_evidence"] = True
            lease_file.write_text(json.dumps(lease))
            raise recovery.ProvisionError("deletion remains unconfirmed")
        if name.endswith("-1"):
            assert json.loads(lease_file.read_text())["first_pass_evidence"]

    api = recovery._api

    def failing_after_initial_download(path, *, binary=False):
        if len(observed) >= 2:
            raise recovery.RecoveryError("later GitHub outage")
        return api(path, binary=binary)

    monkeypatch.setattr(recovery, "_api", failing_after_initial_download)
    monkeypatch.setattr(recovery, "cleanup", cleanup)
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    assert recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=False) == {
        "cleanup_confirmed": "true"
    }
    assert observed == ["trtmc-gpu-ci-123-1", "trtmc-gpu-ci-123-2", "trtmc-gpu-ci-123-1"]
    assert sleeps == [recovery.POLL_INTERVAL]
    assert sum(path.endswith("/201/zip") for path in calls) == 1


def test_unavailable_artifact_is_retried_without_redeleting_a_confirmed_other_attempt(
    monkeypatch, tmp_path
):
    _, first, first_archive = _fixture(1)
    _, second, second_archive = _fixture(2)
    calls, responses = _api_fixture(
        monkeypatch,
        jobs=[_producer(1), _producer(2)],
        artifacts=[second],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": first_archive,
            f"repos/{REPOSITORY}/actions/artifacts/202/zip": second_archive,
        },
    )
    observed = []
    monkeypatch.setattr(recovery, "cleanup", lambda name, **kw: observed.append(name))

    def restore_metadata(_seconds):
        responses[f"repos/{REPOSITORY}/actions/runs/123/artifacts?per_page=100&page=1"][
            "artifacts"
        ] = [first, second]

    monkeypatch.setattr(recovery.time, "sleep", restore_metadata)
    assert recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=False) == {
        "cleanup_confirmed": "true"
    }
    assert observed == ["trtmc-gpu-ci-123-2", "trtmc-gpu-ci-123-1"]
    assert sum(path.endswith("/202/zip") for path in calls) == 1

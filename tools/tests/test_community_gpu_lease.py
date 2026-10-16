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
        "steps": [
            {"name": "Reserve a GPU instance", "conclusion": "failure"},
            {"name": "Publish the HTML report", "conclusion": "success"},
        ],
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


def _report_fixture(*, cleanup=True, attempt=1):
    lease, artifact, _ = _fixture(attempt, final=True)
    lease["cleanup_confirmed"] = cleanup
    receipt = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "run_id": 123,
        "run_attempt": attempt,
        "workflow_sha": SHA,
        "leases": [lease],
    }
    html = (
        "<!doctype html><html><body>Human-readable results"
        '<script type="application/json" id="trtmc-community-recovery">'
        + json.dumps(receipt).replace("<", "\\u003c")
        + "</script></body></html>"
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("report.html", html)
    archive = stream.getvalue()
    artifact.update(
        id=901,
        name=f"community-gpu-report-123-{attempt}",
        digest="sha256:" + hashlib.sha256(archive).hexdigest(),
    )
    return receipt, artifact, archive


def test_compacted_html_preserves_original_recovery_on_cleanup_only_rerun(monkeypatch, tmp_path):
    receipt, report, archive = _report_fixture()
    _api_fixture(
        monkeypatch,
        artifacts=[report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
        },
    )
    result = recovery.recover(REPOSITORY, 123, 4, SHA, tmp_path / "lease.json")
    restored = json.loads((tmp_path / "lease.json").read_text())
    assert result == {"instance_name": "trtmc-gpu-ci-123-1"}
    assert restored["organization_id"] == receipt["leases"][0]["organization_id"]
    assert restored["instance_id"] == receipt["leases"][0]["instance_id"]
    assert restored["cleanup_confirmed"] is True
    assert restored["run_attempt"] == 1


def test_verified_html_is_required_before_pruning_operational_artifacts(monkeypatch):
    receipt, report, archive = _report_fixture()
    _, final, _ = _fixture(final=True)
    final_archive = _archive(receipt["leases"][0])
    final["digest"] = "sha256:" + hashlib.sha256(final_archive).hexdigest()
    _api_fixture(
        monkeypatch,
        artifacts=[final, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": final_archive,
        },
    )
    deleted = []

    def delete(args, **kwargs):
        assert args[:4] == ["gh", "api", "--method", "DELETE"]
        deleted.append(args[-1])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(recovery.subprocess, "run", delete)
    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA) == {"removed_artifacts": "1"}
    assert deleted == [f"repos/{REPOSITORY}/actions/artifacts/201"]


@pytest.mark.parametrize("report_uploaded,cleanup_confirmed", [(False, True), (True, False)])
def test_missing_upload_or_unconfirmed_cleanup_keeps_every_journal(
    monkeypatch, report_uploaded, cleanup_confirmed
):
    _, initial, _ = _fixture()
    _, report, archive = _report_fixture(cleanup=cleanup_confirmed)
    _api_fixture(
        monkeypatch,
        artifacts=[initial] + ([report] if report_uploaded else []),
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Unconfirmed report/release must not delete recovery artifacts")

    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA) == {"removed_artifacts": "0"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "foreign/repo"),
        ("run_id", 999),
        ("run_attempt", 2),
        ("workflow_sha", "b" * 40),
        ("schema_version", True),
    ],
)
def test_report_receipt_cannot_change_original_authority(field, value):
    receipt, _, _ = _report_fixture()
    receipt[field] = value
    with pytest.raises(recovery.RecoveryError):
        recovery.validate_report_receipt(receipt, REPOSITORY, 123, 1, SHA)


@pytest.mark.parametrize(
    "bad_html",
    [
        "<html>No recovery receipt</html>",
        '<script id="trtmc-community-recovery" type="text/javascript">{}</script>',
        '<script id="trtmc-community-recovery" type="application/json">{}</script>' * 2,
        '<script id="trtmc-community-recovery" type="application/json">not json</script>',
    ],
)
def test_malformed_html_receipt_cannot_destroy_recovery_evidence(monkeypatch, bad_html):
    _, initial, _ = _fixture()
    _, report, _ = _report_fixture()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("report.html", bad_html)
    archive = stream.getvalue()
    report["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
    _api_fixture(
        monkeypatch,
        artifacts=[initial, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed report must not delete journals")

    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    with pytest.raises(recovery.RecoveryError):
        recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA)


def test_bad_report_does_not_block_existing_emergency_journal_recovery(monkeypatch, tmp_path):
    _, initial, _ = _fixture()
    _, report, _ = _report_fixture()
    report["digest"] = "invalid"
    _api_fixture(monkeypatch, artifacts=[initial, report])
    assert recovery.recover(REPOSITORY, 123, 2, SHA, tmp_path / "lease.json") == {
        "instance_name": "trtmc-gpu-ci-123-1"
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("organization_id", "different-org"),
        ("instance_id", "different-vm"),
        ("sku", "different-type"),
        ("cleanup_instances", [{"instance_id": "another-child"}]),
    ],
)
def test_report_must_preserve_exact_final_journal_before_any_deletion(monkeypatch, field, value):
    receipt, report, archive = _report_fixture()
    original = copy.deepcopy(receipt["leases"][0])
    original[field] = value
    _, final, _ = _fixture(final=True)
    final_archive = _archive(original)
    final["digest"] = "sha256:" + hashlib.sha256(final_archive).hexdigest()
    _api_fixture(
        monkeypatch,
        artifacts=[final, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": final_archive,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A changed original journal must never be pruned")

    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    with pytest.raises(recovery.RecoveryError, match="exact final"):
        recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA)


def test_missing_final_upload_retains_initial_journal_even_with_valid_html(monkeypatch):
    _, initial, _ = _fixture()
    _, report, archive = _report_fixture()
    _api_fixture(
        monkeypatch,
        artifacts=[initial, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Missing final journal must keep initial recovery evidence")

    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA) == {"removed_artifacts": "0"}


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", None])
def test_report_without_successful_owner_upload_cannot_replace_a_journal(outcome):
    _, report, _ = _report_fixture()
    producer = _producer()
    producer["steps"][1]["conclusion"] = outcome
    with pytest.raises(recovery.RecoveryError, match="confirm its HTML upload"):
        recovery.select_artifact([report], producer, 5)


def test_compaction_deletes_final_authority_last_for_safe_interruption(monkeypatch):
    receipt, report, archive = _report_fixture()
    _, final, _ = _fixture(final=True)
    final_archive = _archive(receipt["leases"][0])
    final["digest"] = "sha256:" + hashlib.sha256(final_archive).hexdigest()
    initial = {**final, "id": 202, "name": "gpu-ci-lease-123-1"}
    intent = {**final, "id": 203, "name": "gpu-ci-intent-123-1"}
    output = {**final, "id": 204, "name": "gpu-ci-output-123-1"}
    _api_fixture(
        monkeypatch,
        artifacts=[final, report, initial, intent, output],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive,
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": final_archive,
        },
    )
    deleted = []

    def delete(args, **kwargs):
        deleted.append(args[-1].split("/")[-1])
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(recovery.subprocess, "run", delete)
    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA) == {"removed_artifacts": "4"}
    assert deleted == ["202", "203", "204", "201"]


def test_interrupted_compaction_resumes_without_losing_original_recovery(monkeypatch, tmp_path):
    receipt, report, report_archive = _report_fixture()
    _, final, _ = _fixture(final=True)
    final_archive = _archive(receipt["leases"][0])
    final["digest"] = "sha256:" + hashlib.sha256(final_archive).hexdigest()
    initial = {**final, "id": 202, "name": "gpu-ci-lease-123-1"}
    intent = {**final, "id": 203, "name": "gpu-ci-intent-123-1"}
    output = {**final, "id": 204, "name": "gpu-ci-output-123-1"}
    _, responses = _api_fixture(
        monkeypatch,
        artifacts=[initial, intent, output, final, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": report_archive,
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": final_archive,
        },
    )
    inventory = responses[f"repos/{REPOSITORY}/actions/runs/123/artifacts?per_page=100&page=1"][
        "artifacts"
    ]
    deleted = []
    interrupted = False

    def delete(args, **kwargs):
        nonlocal interrupted
        assert args[:4] == ["gh", "api", "--method", "DELETE"]
        identity = int(args[-1].rsplit("/", 1)[1])
        if identity == 203 and not interrupted:
            interrupted = True
            return type("Result", (), {"returncode": 1})()
        deleted.append(identity)
        inventory[:] = [item for item in inventory if item["id"] != identity]
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(recovery.subprocess, "run", delete)
    with pytest.raises(recovery.RecoveryError, match="verified report is preserved"):
        recovery.prune_reported_artifacts(REPOSITORY, 123, 1, SHA)
    assert deleted == [202]
    assert {item["id"] for item in inventory} == {203, 204, 201, 901}

    interrupted_path = tmp_path / "interrupted.json"
    assert recovery.recover(REPOSITORY, 123, 2, SHA, interrupted_path) == {
        "instance_name": "trtmc-gpu-ci-123-1"
    }
    assert json.loads(interrupted_path.read_text())["instance_id"] == "vm-fixture"

    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 2, SHA) == {"removed_artifacts": "3"}
    assert deleted == [202, 203, 204, 201]
    assert [item["id"] for item in inventory] == [901]

    # A later cleanup-only rerun must still invoke authenticated provider
    # reconciliation, even when the compacted original journal says deleted.
    reconciled = []

    def cleanup(name, *, lease_file, timeout):
        restored = json.loads(lease_file.read_text())
        assert restored == {**receipt["leases"][0], "run_id": 123, "run_attempt": 1}
        assert timeout == 90
        reconciled.append(name)

    monkeypatch.setattr(recovery, "cleanup", cleanup)
    assert recovery.cleanup_attempts(
        REPOSITORY, 123, 3, SHA, tmp_path / "rerun", prior_only=False
    ) == {"cleanup_confirmed": "true"}
    assert reconciled == ["trtmc-gpu-ci-123-1"]


@pytest.mark.parametrize("skipped_job", [False, True])
@pytest.mark.parametrize("prior_only", [False, True])
def test_new_skipped_owner_does_not_hide_compacted_historical_allocation(
    monkeypatch, tmp_path, skipped_job, prior_only
):
    receipt, report, archive = _report_fixture()
    newer = _producer(2)
    if skipped_job:
        newer.update(conclusion="skipped", steps=[])
    else:
        newer["steps"][0]["conclusion"] = "skipped"
    _api_fixture(
        monkeypatch,
        jobs=[_producer(), newer],
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    reconciled = []

    def cleanup(name, *, lease_file, timeout):
        lease = json.loads(lease_file.read_text())
        assert lease == {**receipt["leases"][0], "run_id": 123, "run_attempt": 1}
        assert timeout == 90
        reconciled.append(name)

    monkeypatch.setattr(recovery, "cleanup", cleanup)
    assert recovery.cleanup_attempts(REPOSITORY, 123, 3, SHA, tmp_path, prior_only=prior_only) == {
        "cleanup_confirmed": "true"
    }
    assert reconciled == ["trtmc-gpu-ci-123-1"]
    assert not (tmp_path / "2").exists()
    assert not (tmp_path / "3").exists()


@pytest.mark.parametrize("upload_outcome", ["failure", "cancelled", "skipped", None])
def test_failed_report_upload_keeps_journal_recovery_working(monkeypatch, tmp_path, upload_outcome):
    lease, initial, initial_archive = _fixture()
    _, report, report_archive = _report_fixture()
    producer = _producer()
    producer["steps"][1]["conclusion"] = upload_outcome
    _api_fixture(
        monkeypatch,
        jobs=[producer],
        artifacts=[initial, report],
        archives={
            f"repos/{REPOSITORY}/actions/artifacts/201/zip": initial_archive,
            f"repos/{REPOSITORY}/actions/artifacts/901/zip": report_archive,
        },
    )
    reconciled = []

    def cleanup(name, *, lease_file, timeout):
        restored = json.loads(lease_file.read_text())
        assert restored == {**lease, "run_id": 123, "run_attempt": 1}
        reconciled.append(name)

    def forbidden(*args, **kwargs):
        pytest.fail("An unconfirmed HTML upload must not delete the working journal")

    monkeypatch.setattr(recovery, "cleanup", cleanup)
    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    assert recovery.cleanup_attempts(REPOSITORY, 123, 2, SHA, tmp_path, prior_only=False) == {
        "cleanup_confirmed": "true"
    }
    assert reconciled == ["trtmc-gpu-ci-123-1"]
    with pytest.raises(recovery.RecoveryError, match="confirm its HTML upload"):
        recovery.prune_reported_artifacts(REPOSITORY, 123, 2, SHA)


def test_report_check_validates_only_the_original_normal_html(monkeypatch):
    _, initial, _ = _fixture()
    initial["digest"] = "invalid journal does not certify report availability"
    _, report, archive = _report_fixture()
    calls, _ = _api_fixture(
        monkeypatch,
        artifacts=[initial, report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "true",
        "report_check_unconfirmed": "false",
    }
    assert calls[-1].endswith("/901/zip")
    assert not any(call.endswith("/201/zip") for call in calls)


@pytest.mark.parametrize("other_name", ["gpu-ci-lease-123-1", "community-gpu-diagnostic-123-1"])
def test_missing_normal_report_is_unavailable_even_with_other_artifacts(monkeypatch, other_name):
    _, report, archive = _report_fixture()
    report["name"] = other_name
    calls, _ = _api_fixture(
        monkeypatch,
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "false",
    }
    assert not any(call.endswith("/zip") for call in calls)


def test_diagnostic_html_cannot_become_cleanup_authority(monkeypatch, tmp_path):
    _, artifact, _ = _report_fixture()
    artifact["name"] = "community-gpu-diagnostic-123-1"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("report.html", "<html>Original allocation recovery remains unknown</html>")
    archive = stream.getvalue()
    artifact["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
    calls, _ = _api_fixture(
        monkeypatch,
        artifacts=[artifact],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Diagnostic HTML must never authorize cleanup or artifact deletion")

    monkeypatch.setattr(recovery, "cleanup", forbidden)
    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    destination = tmp_path / "lease.json"
    with pytest.raises(recovery.RecoveryError, match="unavailable"):
        recovery.recover(REPOSITORY, 123, 2, SHA, destination)
    assert not destination.exists()
    assert recovery.prune_reported_artifacts(REPOSITORY, 123, 2, SHA) == {"removed_artifacts": "0"}
    assert not any(call.endswith("/zip") for call in calls)


@pytest.mark.parametrize("upload_outcome", ["failure", "cancelled", "skipped", None])
def test_report_check_requires_successful_original_owner_upload(monkeypatch, upload_outcome):
    _, report, archive = _report_fixture()
    producer = _producer()
    producer["steps"][1]["conclusion"] = upload_outcome
    calls, _ = _api_fixture(
        monkeypatch,
        jobs=[producer],
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "true",
    }
    assert not any(call.endswith("/zip") for call in calls)


def test_report_check_uses_original_owner_on_cleanup_only_rerun(monkeypatch):
    _, report, archive = _report_fixture()
    skipped = _producer(2)
    skipped.update(conclusion="skipped", steps=[])
    unrelated_report = {**report, "id": 902, "name": "community-gpu-report-123-2"}
    calls, _ = _api_fixture(
        monkeypatch,
        jobs=[_producer(), skipped],
        artifacts=[unrelated_report, report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 4, SHA) == {
        "report_available": "true",
        "report_check_unconfirmed": "false",
    }
    assert calls[-1].endswith("/901/zip")
    assert not any(call.endswith("/902/zip") for call in calls)


def test_report_check_never_substitutes_an_older_owners_report(monkeypatch):
    _, report, archive = _report_fixture()
    calls, _ = _api_fixture(
        monkeypatch,
        jobs=[_producer(), _producer(2)],
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 3, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "false",
    }
    assert not any(call.endswith("/zip") for call in calls)


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "in_progress"},
        {"head_sha": "b" * 40},
        {"steps": None},
        {"steps": ["invalid step metadata"]},
    ],
)
def test_report_check_fails_closed_on_invalid_owner_evidence(monkeypatch, mutation):
    _, report, archive = _report_fixture()
    producer = {**_producer(), **mutation}
    _api_fixture(
        monkeypatch,
        jobs=[producer],
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "true",
    }


@pytest.mark.parametrize("fault", ["expired", "digest", "foreign_run", "malformed_html"])
def test_report_check_requires_artifact_and_embedded_receipt_integrity(monkeypatch, fault):
    _, report, archive = _report_fixture()
    if fault == "expired":
        report["expired"] = True
    elif fault == "digest":
        archive += b"changed after GitHub computed its digest"
    elif fault == "foreign_run":
        report["workflow_run"]["id"] = 999
    else:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            bundle.writestr(
                "report.html", "<html>Diagnostic HTML without recovery authority</html>"
            )
        archive = stream.getvalue()
        report["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
    _api_fixture(
        monkeypatch,
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "true",
    }


@pytest.mark.parametrize("failed_endpoint", ["/123", "/jobs?", "/artifacts?", "/901/zip"])
def test_report_check_api_failure_is_uncertainty_without_resource_inference(
    monkeypatch, failed_endpoint
):
    _, report, archive = _report_fixture()
    _api_fixture(
        monkeypatch,
        artifacts=[report],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    original_api = recovery._api

    def api(path, *, binary=False):
        if path.endswith(failed_endpoint) or failed_endpoint in path:
            raise recovery.RecoveryError("Fixture metadata outage")
        return original_api(path, binary=binary)

    monkeypatch.setattr(recovery, "_api", api)
    assert recovery.check_report(REPOSITORY, 123, 1, SHA) == {
        "report_available": "false",
        "report_check_unconfirmed": "true",
    }


@pytest.mark.parametrize("present,invalid", [(True, False), (False, False), (True, True)])
def test_check_report_cli_writes_only_read_only_report_outputs(
    monkeypatch, tmp_path, present, invalid
):
    _, report, archive = _report_fixture()
    if invalid:
        report["digest"] = "invalid"
    _api_fixture(
        monkeypatch,
        artifacts=[report] if present else [],
        archives={f"repos/{REPOSITORY}/actions/artifacts/901/zip": archive},
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", REPOSITORY)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    monkeypatch.setenv("GITHUB_SHA", SHA)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr("sys.argv", ["recovery", "--check-report"])

    def forbidden(*args, **kwargs):
        pytest.fail("Checking report availability must not mutate provider or GitHub resources")

    monkeypatch.setattr(recovery, "cleanup", forbidden)
    monkeypatch.setattr(recovery.subprocess, "run", forbidden)
    assert recovery.main() == 0
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == {
        "report_available": str(present and not invalid).lower(),
        "report_check_unconfirmed": str(invalid).lower(),
    }

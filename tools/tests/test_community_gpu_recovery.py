# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for bounded recovery and cross-run lease handoff."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import zipfile

import pytest

from tools import brev_exec
from tools import community_gpu_lease
from tools import community_gpu_recovery as service


REPOSITORY = "example/project"
SHA = "a" * 40
NOW = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)


def lease():
    return {
        "schema_version": 1,
        "name": "trtmc-gpu-ci-123-1",
        "sku": "gpu-fixture.1",
        "organization_id": "org-fixture",
        "instance_id": None,
        "create_started": True,
        "create_accepted": False,
        "allocation_pending": True,
        "phase": "provision_failed",
    }


def test_cleanup_deadline_does_not_poison_ownership(monkeypatch, tmp_path):
    original = lease()
    path = tmp_path / "lease.json"
    brev_exec._save_lease(path, original)
    monkeypatch.setattr(brev_exec.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(brev_exec, "_cleanup_request", lambda *args: (200, {"id": "org-fixture"}))

    def expire(*args, **kwargs):
        brev_exec._remaining(-1)

    monkeypatch.setattr(brev_exec, "_instance", expire)
    with pytest.raises(brev_exec.ProvisionError):
        brev_exec.cleanup(original["name"], lease_file=path, timeout=1)
    saved = json.loads(path.read_text())
    assert not saved.get("ownership_blocked", False)
    assert saved["cleanup_confirmed"] is False
    assert saved["phase"] == "unresolved"


def test_recovery_of_unavailable_evidence_exits_on_its_budget(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(community_gpu_lease.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        community_gpu_lease.time,
        "sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )

    def unavailable(*args):
        raise community_gpu_lease.RecoveryError("unavailable")

    monkeypatch.setattr(community_gpu_lease, "_context", unavailable)
    with pytest.raises(community_gpu_lease.RecoveryError, match="budget"):
        community_gpu_lease.cleanup_attempts(
            "example/project", 123, 2, "a" * 40, tmp_path, prior_only=True, timeout=5
        )
    assert clock[0] <= 5


def record():
    return {
        "run_id": 123,
        "run_attempt": 1,
        "producer_id": 101,
        "workflow_sha": SHA,
        "producer_completed_at": service._stamp(NOW - timedelta(days=1)),
        "origin_digest": "sha256:" + "b" * 64,
        "lease": lease(),
        "state": "unresolved",
        "last_checked_at": "1970-01-01T00:00:00Z",
    }


def journal(*, populated=True):
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "repository_id": 5,
        "updated_at": service._stamp(NOW),
        "scan_complete": True,
        "mode": "apply",
        "entries": {"123/1": record()} if populated else {},
        "seen_runs": {},
    }


def producer():
    return {
        "id": 101,
        "run_id": 123,
        "run_attempt": 1,
        "head_sha": SHA,
        "name": community_gpu_lease.OWNER_JOB,
        "status": "completed",
        "conclusion": "failure",
        "started_at": "2026-11-11T11:00:00Z",
        "completed_at": "2026-11-11T12:00:00Z",
        "steps": [{"name": "Reserve a GPU instance", "conclusion": "failure"}],
    }


def source_run():
    return {
        "id": 123,
        "run_attempt": 1,
        "head_sha": SHA,
        "repository": {"id": 5, "full_name": REPOSITORY},
        "head_repository": {"id": 5},
        "path": ".github/workflows/community-ci.yml",
        "event": "workflow_dispatch",
        "head_branch": "ci/developer",
        "status": "completed",
        "created_at": "2026-11-11T11:00:00Z",
        "updated_at": "2026-11-11T12:00:00Z",
    }


def archive(document):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(service.JOURNAL_FILE, json.dumps(document))
    return stream.getvalue()


def checkpoint_api(monkeypatch):
    document = journal()
    data = archive(document)
    run = {
        **source_run(),
        "id": 777,
        "head_branch": "main",
        "path": service.WORKFLOW,
        "event": "schedule",
        "status": "completed",
    }
    job = {
        "id": 778,
        "run_id": 777,
        "head_sha": SHA,
        "run_attempt": 1,
        "name": service.JOB,
        "status": "completed",
        "started_at": "2026-11-12T11:55:00Z",
        "completed_at": "2026-11-12T12:01:00Z",
        "steps": [
            {"name": "Prepare recovery journal", "conclusion": "success"},
            {"name": "Publish recovery journal", "conclusion": "success"},
        ],
    }
    artifact = {
        "id": 900,
        "name": service.ARTIFACT,
        "expired": False,
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "created_at": "2026-11-12T12:00:00Z",
        "workflow_run": {
            "id": 777,
            "head_sha": SHA,
            "repository_id": 5,
            "head_repository_id": 5,
        },
    }
    responses = {
        f"repos/{REPOSITORY}": {
            "id": 5,
            "full_name": REPOSITORY,
            "default_branch": "main",
        },
        f"repos/{REPOSITORY}/actions/artifacts?name={service.ARTIFACT}&per_page=100&page=1": {
            "artifacts": [artifact]
        },
        f"repos/{REPOSITORY}/actions/runs/777": run,
        f"repos/{REPOSITORY}/actions/runs/777/jobs?filter=all&per_page=100&page=1": {"jobs": [job]},
        f"repos/{REPOSITORY}/actions/artifacts/900/zip": data,
        f"repos/{REPOSITORY}/actions/workflows/community-gpu-recovery.yml/runs?per_page=30&page=1": {
            "workflow_runs": [run]
        },
    }
    monkeypatch.setattr(
        community_gpu_lease,
        "_api",
        lambda path, **kwargs: copy.deepcopy(responses[path]),
    )
    return document, responses, artifact, job, run


def test_checkpoint_is_read_with_a_separate_trusted_provenance_rule(monkeypatch):
    expected, *_ = checkpoint_api(monkeypatch)
    assert service.load_journal(REPOSITORY) == expected


@pytest.mark.parametrize(
    "mutation", ["digest", "expired", "time", "repository", "branch", "upload"]
)
def test_foreign_or_invalid_checkpoint_cannot_authorize_recovery(monkeypatch, mutation):
    _, responses, artifact, job, run = checkpoint_api(monkeypatch)
    if mutation == "digest":
        artifact["digest"] = "sha256:" + "0" * 64
    elif mutation == "expired":
        artifact["expired"] = True
    elif mutation == "time":
        artifact["created_at"] = "2026-11-12T12:02:00Z"
    elif mutation == "repository":
        artifact["workflow_run"]["head_repository_id"] = 99
    elif mutation == "branch":
        run["head_branch"] = "untrusted-pr"
    elif mutation == "upload":
        job["steps"][-1]["conclusion"] = "failure"
    with pytest.raises(community_gpu_lease.RecoveryError):
        service.load_journal(REPOSITORY)


def test_cleanup_rerun_can_read_its_own_previous_completed_checkpoint(monkeypatch):
    expected, _, _, job, run = checkpoint_api(monkeypatch)
    run.update(status="in_progress", run_attempt=2)
    monkeypatch.setenv("GITHUB_RUN_ID", "777")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_WORKFLOW", "Community GPU recovery")
    assert job["run_attempt"] == 1
    assert service.load_journal(REPOSITORY) == expected


def test_missing_checkpoint_does_not_rebootstrap_known_history(monkeypatch):
    _, responses, _, _, _ = checkpoint_api(monkeypatch)
    responses[f"repos/{REPOSITORY}/actions/artifacts?name={service.ARTIFACT}&per_page=100&page=1"][
        "artifacts"
    ] = []
    with pytest.raises(community_gpu_lease.RecoveryError, match="forget"):
        service.load_journal(REPOSITORY, bootstrap=True)


def test_recovery_receipt_restores_discovered_id_and_rerun_reads_it(monkeypatch, tmp_path):
    document = journal()
    document["entries"]["123/1"]["lease"]["instance_id"] = "late-vm"
    document["entries"]["123/1"]["lease"]["allocation_pending"] = False
    name = lease()["name"]
    destination = tmp_path / "1" / community_gpu_lease.LEASE_FILE
    assert service.restore_record(document, producer(), destination)
    assert json.loads(destination.read_text())["instance_id"] == "late-vm"
    metadata = json.loads(destination.with_suffix(".authority.json").read_text())
    assert metadata["origin_digest"] == record()["origin_digest"]
    monkeypatch.setattr(
        community_gpu_lease,
        "_context",
        lambda *args: ("prefix", {"repository": {"id": 5}}, [producer()]),
    )
    calls = []

    def confirm(instance, *, lease_file, timeout):
        assert json.loads(lease_file.read_text())["instance_id"] == "late-vm"
        calls.append(instance)

    monkeypatch.setattr(community_gpu_lease, "cleanup", confirm)
    assert community_gpu_lease.cleanup_attempts(
        REPOSITORY, 123, 2, SHA, tmp_path, prior_only=True, journal=document
    ) == {"cleanup_confirmed": "true"}
    assert calls == [name]


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 999),
        ("head_sha", "f" * 40),
        ("run_attempt", 2),
        ("status", "in_progress"),
    ],
)
def test_receipt_cannot_be_restored_for_a_foreign_or_live_owner(tmp_path, field, value):
    owner = producer()
    owner[field] = value
    if field == "run_attempt":
        assert not service.restore_record(journal(), owner, tmp_path / "lease.json")
    else:
        with pytest.raises(community_gpu_lease.RecoveryError):
            service.restore_record(journal(), owner, tmp_path / "lease.json")
    assert not (tmp_path / "lease.json").exists()


def test_backstop_progress_survives_a_later_job_and_strips_provider_text(monkeypatch, tmp_path):
    document = journal()
    path = tmp_path / "1" / community_gpu_lease.LEASE_FILE
    service.restore_record(document, producer(), path)
    learned = json.loads(path.read_text())
    learned.update(
        instance_id="late-vm",
        allocation_pending=False,
        provider_text="SECRET-PROVIDER-BODY",
    )
    learned["create_diagnostics"] = {"raw_error": "SECRET-TOKEN"}
    brev_exec._save_lease(path, learned)
    output = tmp_path / service.JOURNAL_FILE
    monkeypatch.setattr(service, "_now", lambda: NOW)
    service.export_progress(REPOSITORY, 123, SHA, tmp_path, output)
    exported = json.loads(output.read_text())
    assert "SECRET" not in output.read_text()
    assert exported["entries"]["123/1"]["lease"]["instance_id"] == "late-vm"
    assert service.validate_journal(exported, REPOSITORY, 5) == exported


def test_live_owner_is_never_captured_or_deleted(monkeypatch):
    document = journal(populated=False)
    owner = producer()
    owner.update(status="in_progress", conclusion=None)
    run = source_run()
    run["status"] = "in_progress"
    monkeypatch.setattr(community_gpu_lease, "_context", lambda *args: ("prefix", run, [owner]))
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(
        service,
        "_capture_producer",
        lambda *args: pytest.fail("live owner used as recovery authority"),
    )
    service.discover(document, [run])
    assert document["entries"] == {}
    assert document["active_owners"] is True


def test_unknown_creation_is_bounded_then_a_late_vm_is_eventually_deleted(monkeypatch, tmp_path):
    document = journal()
    clock = [0.0]
    monkeypatch.setattr(brev_exec.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        brev_exec.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    monkeypatch.setattr(service, "_now", lambda: NOW + timedelta(seconds=clock[0]))
    available = [False]
    deleted = [False]
    body = {
        "id": "late-vm",
        "name": lease()["name"],
        "organizationId": "org-fixture",
        "instanceType": "gpu-fixture.1",
    }
    monkeypatch.setattr(
        brev_exec,
        "_instance",
        lambda *args, **kwargs: {"id": "late-vm"} if available[0] else None,
    )
    operations = []

    def request(method, identity, deadline):
        operations.append((method, identity))
        if method == "AUTH":
            return 200, {"id": "org-fixture"}
        if method == "LIST":
            return 200, {"items": []}
        if method == "GET":
            return (404, None) if deleted[0] else (200, body)
        if method == "DELETE":
            deleted[0] = True
            return 202, body
        pytest.fail("unexpected operation")

    monkeypatch.setattr(brev_exec, "_cleanup_request", request)
    service.reconcile(document, tmp_path, 5, apply=True)
    assert document["entries"]["123/1"]["state"] == "unresolved"
    assert clock[0] == 5
    assert not any(method == "DELETE" for method, _ in operations)
    available[0] = True
    service.reconcile(document, tmp_path, clock[0] + 5, apply=True)
    saved = document["entries"]["123/1"]["lease"]
    assert saved["instance_id"] == "late-vm"
    service.reconcile(document, tmp_path, clock[0] + 90, apply=True)
    assert document["entries"]["123/1"]["state"] == "confirmed"
    assert document["entries"]["123/1"]["lease"]["cleanup_confirmed"] is True
    assert all(identity == "late-vm" for method, identity in operations if method == "DELETE")


@pytest.mark.parametrize(
    "mutation", ["stale", "future", "incomplete", "observe", "cap", "cooldown"]
)
def test_recovery_health_and_resource_limits_block_new_requests(mutation):
    document = journal()
    maximum = 5
    if mutation == "stale":
        document["updated_at"] = service._stamp(NOW - timedelta(hours=2))
    elif mutation == "future":
        document["updated_at"] = service._stamp(NOW + timedelta(minutes=5))
    elif mutation == "incomplete":
        document["scan_complete"] = False
    elif mutation == "observe":
        document["mode"] = "observe"
    elif mutation == "cap":
        maximum = 1
    elif mutation == "cooldown":
        document["entries"]["123/1"]["producer_completed_at"] = service._stamp(
            NOW - timedelta(seconds=10)
        )
    with pytest.raises(community_gpu_lease.RecoveryError):
        service.check_admission(document, NOW, maximum=maximum)


@pytest.mark.parametrize("state,head", [("closed", SHA), ("open", "b" * 40)])
def test_stale_pull_never_reaches_allocation(monkeypatch, state, head):
    pull = {
        "state": state,
        "head": {"sha": head},
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
    }
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: pull)
    with pytest.raises(community_gpu_lease.RecoveryError, match="superseded"):
        service.check_pull(REPOSITORY, 123, SHA)


def test_admission_allows_other_healthy_concurrent_reservations(monkeypatch):
    document = journal(populated=False)
    run = source_run()
    run["id"] = 122
    owner = producer()
    owner.update(status="in_progress", conclusion=None)
    owner["steps"][0].update(status="in_progress", conclusion=None)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(service, "load_journal", lambda *args: document)
    monkeypatch.setattr(service, "check_pull", lambda *args: None)
    monkeypatch.setattr(service, "_runs", lambda *args: [run])
    monkeypatch.setattr(community_gpu_lease, "_context", lambda *args: ("prefix", run, [owner]))
    service.admit(REPOSITORY, 123, SHA, maximum=5, cooldown=900)


def test_discovery_outage_does_not_strand_a_previously_authenticated_vm(monkeypatch, tmp_path):
    document = journal()
    document["entries"]["123/1"]["lease"].update(instance_id="known-vm", allocation_pending=False)
    output = tmp_path / service.JOURNAL_FILE
    brev_exec._save_lease(output, document)
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "5")
    monkeypatch.setattr(service, "_now", lambda: NOW)

    def unavailable(*args):
        raise community_gpu_lease.RecoveryError("GitHub unavailable")

    monkeypatch.setattr(community_gpu_lease, "_api", unavailable)

    calls = []

    def remove(name, *, lease_file, timeout):
        saved = json.loads(lease_file.read_text())
        assert saved["instance_id"] == "known-vm"
        calls.append(name)
        saved.update(cleanup_confirmed=True)
        brev_exec._save_lease(lease_file, saved)

    monkeypatch.setattr(service, "_source_runs", unavailable)
    monkeypatch.setattr(service, "cleanup", remove)
    result = service.run_round(REPOSITORY, output, timeout=30, apply=True)
    assert calls == [lease()["name"]]
    assert result["scan_complete"] is False
    assert result["entries"]["123/1"]["state"] == "confirmed"
    assert service.validate_journal(json.loads(output.read_text()), REPOSITORY, 5)
    assert community_gpu_lease.EVIDENCE_DEADLINE.get() is None


def test_observation_never_calls_provider_deletion(monkeypatch, tmp_path):
    document = journal()
    monkeypatch.setattr(
        service,
        "cleanup",
        lambda *args, **kwargs: pytest.fail("observation deleted a resource"),
    )
    service.reconcile(document, tmp_path, 999999999, apply=False)
    assert document["entries"]["123/1"]["state"] == "unresolved"


def test_backstop_update_is_consumed_only_with_matching_owner_and_resource(monkeypatch):
    document = journal()
    update = copy.deepcopy(document)
    update["entries"]["123/1"]["lease"].update(instance_id="late-vm", allocation_pending=False)
    update["entries"]["123/1"]["last_checked_at"] = service._stamp(NOW)
    data = archive(update)
    item = {
        "id": 900,
        "name": "community-gpu-recovery-123-2",
        "expired": False,
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "created_at": "2026-11-12T12:00:00Z",
        "workflow_run": {
            "id": 123,
            "head_sha": SHA,
            "repository_id": 5,
            "head_repository_id": 5,
        },
    }
    job = {
        "id": 102,
        "run_id": 123,
        "head_sha": SHA,
        "run_attempt": 2,
        "name": "Community GPU / Cleanup",
        "status": "completed",
        "started_at": "2026-11-12T11:00:00Z",
        "completed_at": "2026-11-12T13:00:00Z",
        "steps": [{"name": "Publish recovery progress", "conclusion": "success"}],
    }
    monkeypatch.setattr(community_gpu_lease, "_rows", lambda *args: [item])
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: data)
    service._read_backstop_updates(REPOSITORY, source_run(), [job], document)
    assert document["entries"]["123/1"]["lease"]["instance_id"] == "late-vm"
    update["entries"]["123/1"]["lease"]["organization_id"] = "someone-else"
    changed = archive(update)
    item["digest"] = "sha256:" + hashlib.sha256(changed).hexdigest()
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: changed)
    with pytest.raises(community_gpu_lease.RecoveryError, match="ownership"):
        service._read_backstop_updates(REPOSITORY, source_run(), [job], document)


def test_missing_evidence_does_not_starve_newer_runs(monkeypatch):
    document = journal(populated=False)
    runs = [source_run(), {**source_run(), "id": 124}]
    monkeypatch.setattr(service, "_now", lambda: NOW)

    def context(repository, run_id, attempt, sha):
        run = next(run for run in runs if run["id"] == run_id)
        owner = {**producer(), "id": run_id + 100, "run_id": run_id}
        return "prefix", run, [owner]

    def missing(repository, run, owner, previous):
        return {
            **record(),
            "run_id": run["id"],
            "producer_id": owner["id"],
            "lease": None,
            "origin_digest": None,
            "state": "missing_evidence",
        }

    monkeypatch.setattr(community_gpu_lease, "_context", context)
    monkeypatch.setattr(service, "_capture_producer", missing)
    service.discover(document, runs, limit=1)
    assert set(document["entries"]) == {"123/1"}
    service.discover(document, runs, limit=1)
    assert set(document["entries"]) == {"123/1", "124/1"}


def test_recovery_workflow_uses_trusted_main_entry_and_persists_on_failure():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / service.WORKFLOW).read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    job = workflow["jobs"]["recover"]
    assert "refs/heads/main" in job["if"]
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    steps = {step["name"]: step for step in job["steps"]}
    checkout = steps["Check out the trusted recovery worker"]
    assert checkout["with"]["ref"] == "${{ steps.worker.outputs.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert "main|ci/developer" in steps["Resolve the trusted recovery worker"]["run"]
    assert "github.event.workflow_run.head_sha" not in str(workflow)
    assert "always()" in steps["Publish recovery journal"]["if"]
    assert steps["Publish recovery journal"]["with"]["retention-days"] == 90
    assert steps["Publish recovery journal"]["with"]["name"] == service.ARTIFACT
    assert "GITHUB_RUN_ATTEMPT" in steps["Require a fresh recovery run"]["run"]


def test_new_source_lease_is_attested_sanitized_and_restorable_without_original_artifact(
    monkeypatch, tmp_path
):
    document = journal(populated=False)
    original = {**lease(), "provider_text": "SECRET-PROVIDER-RESPONSE"}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr(community_gpu_lease.LEASE_FILE, json.dumps(original))
    data = stream.getvalue()
    item = {
        "id": 201,
        "name": "gpu-ci-lease-final-123-1",
        "expired": False,
        "created_at": "2026-11-11T11:30:00Z",
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        "workflow_run": {"id": 123, "head_sha": SHA, "repository_id": 5, "head_repository_id": 5},
    }
    run, owner = source_run(), producer()
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(community_gpu_lease, "_context", lambda *args: ("prefix", run, [owner]))
    monkeypatch.setattr(community_gpu_lease, "_rows", lambda *args: [item])
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: data)
    service.discover(document, [run])
    assert document["entries"]["123/1"]["origin_digest"] == item["digest"]
    assert "SECRET" not in json.dumps(document)
    assert service.validate_journal(document, REPOSITORY, 5)
    monkeypatch.setattr(
        community_gpu_lease,
        "_api",
        lambda *args, **kwargs: pytest.fail("original artifact was required again"),
    )
    output = tmp_path / community_gpu_lease.LEASE_FILE
    assert service.restore_record(document, owner, output)
    assert json.loads(output.read_text())["name"] == lease()["name"]


def test_failed_or_missing_original_evidence_is_retained_as_unresolved_work(monkeypatch):
    run, owner = source_run(), producer()

    def missing(*args):
        raise community_gpu_lease.RecoveryError("artifact unavailable")

    monkeypatch.setattr(community_gpu_lease, "_rows", missing)
    saved = service._capture_producer(REPOSITORY, run, owner, None)
    assert saved["state"] == "missing_evidence"
    assert saved["lease"] is None
    assert saved["producer_id"] == owner["id"]
    document = journal(populated=False)
    document["entries"]["123/1"] = saved
    assert service.validate_journal(document, REPOSITORY, 5)


@pytest.mark.parametrize(
    "response", [{"workflow_runs": None}, {"workflow_runs": [{}], "total_count": 1000}]
)
def test_source_inventory_cannot_silently_drop_runs(monkeypatch, response):
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: response)
    with pytest.raises(community_gpu_lease.RecoveryError):
        service._runs("repos/example/project/actions/workflows/community-ci.yml/runs")


def test_github_request_uses_the_remaining_recovery_budget(monkeypatch):
    import subprocess

    monkeypatch.setattr(community_gpu_lease.time, "monotonic", lambda: 10.0)
    observed = []

    def expired(command, **kwargs):
        observed.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(community_gpu_lease.subprocess, "run", expired)
    token = community_gpu_lease.EVIDENCE_DEADLINE.set(10.25)
    try:
        with pytest.raises(community_gpu_lease.RecoveryError):
            community_gpu_lease._api("repos/example/project")
    finally:
        community_gpu_lease.EVIDENCE_DEADLINE.reset(token)
    assert observed == [0.25]


def test_bad_discovery_run_does_not_block_other_owners_and_can_be_retried(monkeypatch):
    document = journal(populated=False)
    runs = [source_run(), {**source_run(), "id": 124}]
    fail = [True]
    calls = []
    monkeypatch.setattr(service, "_now", lambda: NOW)

    def context(repository, identity, *args):
        calls.append(identity)
        if identity == 123 and fail[0]:
            raise community_gpu_lease.RecoveryError("temporary outage")
        return "prefix", next(run for run in runs if run["id"] == identity), []

    monkeypatch.setattr(community_gpu_lease, "_context", context)
    service.discover(document, runs)
    assert calls == [123, 124]
    assert document["discovery_errors"] == {"123": "unconfirmed"}
    assert document["scan_complete"] is False
    assert document["seen_runs"]["123"]["checked_at"] == service._stamp(NOW)
    fail[0] = False
    calls.clear()
    service.discover(document, runs)
    assert calls == [123]
    assert not document["discovery_errors"]
    assert document["scan_complete"] is True


def test_backstop_can_supply_original_evidence_that_was_temporarily_missing():
    missing = {**record(), "lease": None, "origin_digest": None, "state": "missing_evidence"}
    learned = record()
    learned["lease"].update(instance_id="late-vm", allocation_pending=False)
    result = service._merge_record(missing, learned)
    assert result["lease"]["instance_id"] == "late-vm"
    assert result["origin_digest"] == learned["origin_digest"]


def test_stale_backstop_cannot_erase_an_id_cursor_or_confirmed_deletion():
    original = record()
    original["lease"].update(
        instance_id="known-vm",
        allocation_pending=False,
        cleanup_cursor=7,
        delete_accepted=True,
        cleanup_confirmed=True,
        phase="deleted",
    )
    original["state"] = "confirmed"
    stale = record()
    stale["last_checked_at"] = service._stamp(NOW)
    stale["lease"].update(cleanup_cursor=1, delete_accepted=False)
    merged = service._merge_record(original, stale)
    assert merged["lease"]["instance_id"] == "known-vm"
    assert merged["lease"]["cleanup_cursor"] == 7
    assert merged["lease"]["delete_accepted"] is True
    assert merged["state"] == "confirmed"
    assert merged["lease"]["phase"] == "deleted"


def test_merge_preserves_all_owned_child_ids_and_recovers_new_late_children():
    original = record()
    first = {
        **lease(),
        "instance_id": "vm-one",
        "allocation_pending": False,
        "cleanup_confirmed": True,
        "phase": "deleted",
    }
    second = {
        **lease(),
        "instance_id": "vm-two",
        "allocation_pending": False,
        "cleanup_confirmed": False,
    }
    original["lease"].update(cleanup_instances={"vm-one": first}, cleanup_confirmed=True)
    original["state"] = "confirmed"
    update = record()
    update["lease"].update(cleanup_instances={"vm-two": second})
    merged = service._merge_record(original, update)
    assert set(merged["lease"]["cleanup_instances"]) == {"vm-one", "vm-two"}
    assert merged["lease"]["cleanup_instances"]["vm-one"]["cleanup_confirmed"] is True
    assert merged["state"] == "unresolved"


def test_export_does_not_invent_a_fresh_check_for_untouched_restored_progress(
    monkeypatch, tmp_path
):
    document = journal()
    original_time = "2026-11-11T12:00:00Z"
    document["entries"]["123/1"]["last_checked_at"] = original_time
    service.restore_record(document, producer(), tmp_path / "1" / community_gpu_lease.LEASE_FILE)
    monkeypatch.setattr(service, "_now", lambda: NOW)
    destination = tmp_path / service.JOURNAL_FILE
    service.export_progress(REPOSITORY, 123, SHA, tmp_path, destination)
    assert (
        json.loads(destination.read_text())["entries"]["123/1"]["last_checked_at"] == original_time
    )


def test_one_unreadable_progress_file_does_not_starve_another_known_owner(monkeypatch, tmp_path):
    document = journal()
    another = copy.deepcopy(record())
    another.update(run_id=124, producer_id=102)
    another["lease"].update(
        name="trtmc-gpu-ci-124-1", instance_id="second-vm", allocation_pending=False
    )
    document["entries"]["124/1"] = another
    calls = []
    monkeypatch.setattr(service, "_now", lambda: NOW)

    def remove(name, *, lease_file, timeout):
        calls.append(name)
        if name == lease()["name"]:
            lease_file.write_text("{broken")
            raise brev_exec.ProvisionError("local progress interrupted")
        saved = json.loads(lease_file.read_text())
        saved["cleanup_confirmed"] = True
        brev_exec._save_lease(lease_file, saved)

    monkeypatch.setattr(service, "cleanup", remove)
    service.reconcile(document, tmp_path, 999999999, apply=True)
    assert calls == [lease()["name"], "trtmc-gpu-ci-124-1"]
    assert document["entries"]["123/1"]["state"] == "unresolved"
    assert document["entries"]["123/1"]["lease"] == lease()
    assert document["entries"]["124/1"]["state"] == "confirmed"


def test_a_current_rerun_can_use_its_authenticated_prior_cleanup_immediately(monkeypatch, tmp_path):
    document = journal()
    document["entries"]["123/1"]["lease"].update(instance_id="known-vm", allocation_pending=False)
    path = tmp_path / "1" / community_gpu_lease.LEASE_FILE
    service.restore_record(document, producer(), path)
    confirmed = json.loads(path.read_text())
    confirmed.update(cleanup_confirmed=True, delete_accepted=True, phase="deleted")
    brev_exec._save_lease(path, confirmed)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(service, "load_journal", lambda *args: copy.deepcopy(document))
    monkeypatch.setattr(service, "check_pull", lambda *args: None)
    monkeypatch.setattr(service, "_runs", lambda *args: [])
    service.admit(REPOSITORY, 123, SHA, maximum=1, cooldown=900, prior_directory=tmp_path)
    assert document["entries"]["123/1"]["state"] == "unresolved"


def test_admission_sees_an_old_run_restarted_since_its_completed_checkpoint(monkeypatch):
    document = journal(populated=False)
    old = source_run()
    old.update(id=122, run_attempt=2, status="in_progress", created_at="2026-11-01T11:00:00Z")
    document["seen_runs"]["122"] = {"status": "completed"}
    owner = producer()
    owner.update(run_id=122, run_attempt=2, status="in_progress", conclusion=None)
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(service, "load_journal", lambda *args: document)
    monkeypatch.setattr(service, "check_pull", lambda *args: None)
    monkeypatch.setattr(
        service, "_runs", lambda path: [old] if "status=in_progress" in path else []
    )
    monkeypatch.setattr(community_gpu_lease, "_context", lambda *args: ("prefix", old, [owner]))
    with pytest.raises(community_gpu_lease.RecoveryError, match="unresolved"):
        service.admit(REPOSITORY, 123, SHA, maximum=5, cooldown=900)


def test_empty_successful_backstop_upload_cannot_close_admission(monkeypatch, tmp_path):
    document = journal(populated=False)
    owner = producer()
    owner["steps"][0]["conclusion"] = "skipped"
    backstop = {
        "id": 102,
        "run_attempt": 1,
        "name": "Community GPU / Cleanup",
        "status": "completed",
        "steps": [{"name": "Publish recovery progress", "conclusion": "success"}],
    }
    run = source_run()
    monkeypatch.setattr(service, "_now", lambda: NOW)
    monkeypatch.setattr(
        community_gpu_lease, "_context", lambda *args: ("prefix", run, [owner, backstop])
    )
    monkeypatch.setattr(community_gpu_lease, "_rows", lambda *args: [])
    output = tmp_path / service.JOURNAL_FILE
    service.export_progress(REPOSITORY, 123, SHA, tmp_path, output)
    assert not output.exists()
    service.discover(document, [run])
    assert not document["discovery_errors"]
    assert document["scan_complete"] is True
    assert not document["entries"]


def test_discovery_gap_is_retried_after_its_source_run_ages_out(monkeypatch):
    document = journal(populated=False)
    run = source_run()
    document["discovery_errors"] = {"123": "unconfirmed"}
    document["seen_runs"]["123"] = {
        key: run[key] for key in ("created_at", "updated_at", "run_attempt", "status")
    }
    monkeypatch.setattr(service, "_now", lambda: NOW + timedelta(days=41))

    def unavailable(*args, **kwargs):
        raise community_gpu_lease.RecoveryError("temporarily unavailable")

    monkeypatch.setattr(community_gpu_lease, "_api", unavailable)
    service.discover(document, [])
    assert "123" in document["discovery_errors"]
    assert "123" in document["seen_runs"]
    assert document["scan_complete"] is False
    monkeypatch.setattr(community_gpu_lease, "_api", lambda *args, **kwargs: run)
    monkeypatch.setattr(community_gpu_lease, "_context", lambda *args: ("prefix", run, []))
    service.discover(document, [])
    assert not document["discovery_errors"]
    assert "123" not in document["seen_runs"]
    assert document["scan_complete"] is True


def test_expired_discovery_budget_does_not_mark_unattempted_runs_as_errors(monkeypatch):
    document = journal(populated=False)
    monkeypatch.setattr(community_gpu_lease.time, "monotonic", lambda: 10)
    monkeypatch.setattr(
        community_gpu_lease,
        "_context",
        lambda *args: pytest.fail("budget expired before this run was attempted"),
    )
    token = community_gpu_lease.EVIDENCE_DEADLINE.set(10)
    try:
        service.discover(document, [source_run()])
    finally:
        community_gpu_lease.EVIDENCE_DEADLINE.reset(token)
    assert not document["discovery_errors"]
    assert not document["seen_runs"]
    assert document["scan_complete"] is False


def test_invalid_merged_progress_is_rejected_before_checkpointing():
    original = record()
    update = record()
    update["last_checked_at"] = service._stamp(NOW)
    update["lease"].update(
        create_started=False, create_attempts=[{"attempt": 1, "outcome": "unknown"}]
    )
    with pytest.raises(brev_exec.ProvisionError):
        service._merge_record(original, update)
    assert original == record()


@pytest.mark.parametrize("from_checkpoint", [False, True])
def test_a_lease_is_not_published_before_its_authority_sidecar(
    monkeypatch, tmp_path, from_checkpoint
):
    path = tmp_path / "lease.json"
    module = service if from_checkpoint else community_gpu_lease
    original_save = module._save_lease

    def unavailable(destination, value):
        if destination.name.endswith(".authority.json"):
            raise OSError("cannot persist authority")
        return original_save(destination, value)

    monkeypatch.setattr(module, "_save_lease", unavailable)
    with pytest.raises(OSError):
        if from_checkpoint:
            service.restore_record(journal(), producer(), path)
        else:
            from tools.tests.test_community_gpu_lease import _api_fixture

            _api_fixture(monkeypatch)
            community_gpu_lease.recover(REPOSITORY, 123, 1, SHA, path)
    assert not path.exists()

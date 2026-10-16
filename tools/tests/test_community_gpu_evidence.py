# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the untrusted-container evidence boundary and cleanup ordering."""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from tools import community_gpu_ci as ci
from tools.e2e_evidence import Evidence


SOURCE = "a" * 40
IDENTITY = {"source_revision": SOURCE, "workflow_run_id": 42, "workflow_run_attempt": 2}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQMgkDAAD4AJ3MaiF4AAAAAElFTkSuQmCC"
)


def _index(*families):
    return ci._new_evidence_index(
        {family: ci.FamilyPlan(family, ("case",), ()) for family in families}, IDENTITY
    )


def _record(capture, family="alpha", *, status="failed"):
    case = capture / "evidence" / family / "case"
    case.mkdir(parents=True)
    value = {
        "schema_version": 1,
        "family": family,
        "case": "case",
        **IDENTITY,
        "status": status,
        "artifacts": [
            {"path": "artifacts/input.png", "media_type": "image/png"},
        ],
    }
    (case / "evidence.json").write_text(json.dumps(value))
    (case / "artifacts").mkdir()
    (case / "artifacts/input.png").write_bytes(PNG)
    (case / "report.html").write_text("<script>untrusted HTML must never be copied</script>")
    return case


def _metadata(results, index, *, status="failed", phase="validation", entered=True):
    ci._save_json(results / "evidence-index.json", index)
    ci._summary(
        {
            "alpha": {
                "family": "alpha",
                "status": status,
                "phase": phase,
                "entrypoint_started": entered,
                "failure_class": "pr_failure" if entered else "infra_failure",
                "requested_cases": ["case"],
                "cases": {"case": status if entered else "not_run"},
            }
        },
        {"TRTMC_GPU_RESULTS_DIR": str(results)},
        time.monotonic(),
    )


def test_existing_recorder_data_is_collected_but_its_html_is_never_exported(tmp_path):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    input_image = tmp_path / "input.png"
    input_image.write_bytes(PNG)
    recorder = Evidence(
        capture / "evidence/alpha/case",
        family="alpha",
        case="case",
        source_revision=SOURCE,
        roots=(tmp_path,),
    )
    recorder.data["workflow_run_attempt"] = 2
    recorder.record("inputs", {"image": input_image})
    recorder.finish("failed", failure="recorded model comparison failed")
    assert (capture / "evidence/alpha/case/report.html").exists()
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    _metadata(results, index)
    archive = tmp_path / "report.tar.gz"
    ci.export_report_archive(results, archive)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert all(member.isfile() for member in members)
        assert len(members) == 4
        assert all(not member.name.endswith(".html") for member in members)
        evidence = json.load(bundle.extractfile("evidence/alpha/case/evidence.json"))
        assert evidence["status"] == "failed" and evidence["inputs"]["image"]["artifact"]
    entry = index["cases"][0]
    assert entry["status"] == "collected" and len(entry["artifacts"]) == 1
    assert entry["artifacts"][0]["sha256"] == hashlib.sha256(PNG).hexdigest()
    assert (results / entry["artifacts"][0]["path"]).read_bytes() == PNG


@pytest.mark.parametrize(
    "attack",
    [
        "root",
        "family",
        "case",
        "json_symlink",
        "json_hardlink",
        "json_fifo",
        "artifact_parent",
        "artifact_symlink",
        "artifact_hardlink",
        "artifact_fifo",
    ],
)
def test_container_paths_cannot_read_host_files(tmp_path, attack):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    case = _record(capture)
    private = tmp_path / "host-private"
    private.mkdir()
    secret = private / "secret"
    secret.write_text("HOST_TOKEN_FIXTURE_MUST_NOT_ESCAPE")
    if attack in {"root", "family", "case"}:
        target = {"root": capture / "evidence", "family": capture / "evidence/alpha", "case": case}[
            attack
        ]
        shutil.copytree(target, private / "subtree")
        for path in (private / "subtree").rglob("evidence.json"):
            value = json.loads(path.read_text())
            value["private"] = secret.read_text()
            path.write_text(json.dumps(value))
        shutil.rmtree(target)
        target.symlink_to(private / "subtree", target_is_directory=True)
    elif attack.startswith("json_"):
        target = case / "evidence.json"
        value = json.loads(target.read_text())
        value["private"] = secret.read_text()
        secret.write_text(json.dumps(value))
        target.unlink()
        if attack == "json_symlink":
            target.symlink_to(secret)
        elif attack == "json_hardlink":
            os.link(secret, target)
        else:
            os.mkfifo(target)
    elif attack == "artifact_parent":
        shutil.rmtree(case / "artifacts")
        (private / "input.png").write_text(secret.read_text())
        (case / "artifacts").symlink_to(private, target_is_directory=True)
    else:
        target = case / "artifacts/input.png"
        target.unlink()
        if attack == "artifact_symlink":
            target.symlink_to(secret)
        elif attack == "artifact_hardlink":
            os.link(secret, target)
        else:
            os.mkfifo(target)
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    assert index["cases"][0]["status"] in {"rejected", "limited"}
    assert not index["cases"][0]["artifacts"]
    assert all(
        b"HOST_TOKEN_FIXTURE" not in path.read_bytes()
        for path in results.rglob("*")
        if path.is_file()
    )


def test_open_directory_descriptor_survives_a_concurrent_symlink_swap(tmp_path, monkeypatch):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    _record(capture)
    private = tmp_path / "private"
    _record(private)
    value = json.loads((private / "evidence/alpha/case/evidence.json").read_text())
    value["secret"] = "HOST_TOKEN_FIXTURE"
    (private / "evidence/alpha/case/evidence.json").write_text(json.dumps(value))
    original = os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = original(path, flags, *args, **kwargs)
        if path == "alpha" and flags & os.O_DIRECTORY and not swapped:
            swapped = True
            (capture / "evidence/alpha").rename(capture / "detached")
            (capture / "evidence/alpha").symlink_to(
                private / "evidence/alpha", target_is_directory=True
            )
        return descriptor

    monkeypatch.setattr(os, "open", racing_open)
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    assert swapped
    assert b"HOST_TOKEN_FIXTURE" not in (results / "evidence/alpha/case/evidence.json").read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "b" * 40),
        ("workflow_run_attempt", 1),
        ("family", "beta"),
        ("case", "other"),
    ],
)
def test_evidence_cannot_claim_another_source_attempt_or_case(tmp_path, field, value):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    case = _record(capture)
    record = json.loads((case / "evidence.json").read_text())
    record[field] = value
    (case / "evidence.json").write_text(json.dumps(record))
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    assert index["cases"][0]["status"] == "rejected"
    assert not list(results.rglob("evidence.json"))


def test_payload_file_and_case_budgets_leave_explicit_missing_details(tmp_path, monkeypatch):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    case = _record(capture)
    json_size = (case / "evidence.json").stat().st_size
    monkeypatch.setattr(ci, "REPORT_MAX_PAYLOAD_BYTES", json_size + 1)
    monkeypatch.setattr(ci, "REPORT_MAX_CASES", 1)
    index = _index("alpha", "beta")
    ci.collect_family_evidence(capture, results, "alpha", index)
    assert len(index["cases"]) == 1 and index["issues"] == ["case_count_limit"]
    assert index["total_bytes"] == json_size and index["file_count"] == 1
    assert index["cases"][0]["status"] == "limited"
    assert not index["cases"][0]["artifacts"]


@pytest.mark.parametrize(
    "relative",
    ["artifacts/page.html", "artifacts/image.svg", "../../host-token", "/tmp/host-token"],
)
def test_declared_html_and_escaping_artifacts_are_not_collected(tmp_path, relative):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    case = _record(capture)
    record = json.loads((case / "evidence.json").read_text())
    record["artifacts"].append({"path": relative, "media_type": "image/png"})
    (case / "evidence.json").write_text(json.dumps(record))
    (case / "artifacts/page.html").write_text("<script>forbidden</script>")
    (case / "artifacts/image.svg").write_text("<svg onload='forbidden()'></svg>")
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    assert index["cases"][0]["status"] == "limited"
    assert [Path(item["path"]).name for item in index["cases"][0]["artifacts"]] == ["input.png"]
    assert not list(results.rglob("*.html")) and not list(results.rglob("*.svg"))


def test_unsupported_case_names_do_not_expand_or_escape_the_index():
    index = ci._new_evidence_index(
        {
            "alpha": ci.FamilyPlan(
                "alpha", ("../outside", "a" * (2 * ci.REPORT_MAX_METADATA_BYTES), "case"), ()
            )
        },
        IDENTITY,
    )
    assert [entry["case"] for entry in index["cases"]] == ["case"]
    assert index["issues"] == ["unsupported_case_name"]
    assert len(json.dumps(index).encode()) < ci.REPORT_MAX_METADATA_BYTES


def test_frozen_source_overrides_an_environment_supplied_revision(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": SOURCE,
                "source_tree": "b" * 40,
                "selection": {
                    "scope": "families",
                    "families": ["alpha"],
                    "direct_families": ["alpha"],
                    "added_families": [],
                    "selected_families": ["alpha"],
                },
                "families": [],
                "active_families": ["alpha"],
            }
        )
    )
    identity = ci._report_identity(
        {
            "TRTMC_E2E_SOURCE_REVISION": "c" * 40,
            "TRTMC_E2E_WORKFLOW_RUN_ID": "42",
            "TRTMC_E2E_WORKFLOW_ATTEMPT": "2",
        },
        plan,
    )
    assert identity == IDENTITY


@pytest.mark.parametrize("attack", ["html", "traversal", "symlink", "changed_digest"])
def test_export_rejects_unsafe_or_changed_collector_paths(tmp_path, attack):
    capture, results = tmp_path / "capture", tmp_path / "results"
    capture.mkdir()
    results.mkdir()
    _record(capture)
    index = _index("alpha")
    ci.collect_family_evidence(capture, results, "alpha", index)
    record = index["cases"][0]["artifacts"][0]
    if attack == "html":
        record["path"] = "evidence/alpha/case/artifacts/report.html"
    elif attack == "traversal":
        record["path"] = "evidence/alpha/case/../../token"
    elif attack == "changed_digest":
        (results / record["path"]).write_bytes(b"tampered")
    else:
        path = results / record["path"]
        path.unlink()
        path.symlink_to(tmp_path / "token")
        (tmp_path / "token").write_bytes(PNG)
    _metadata(results, index)
    with pytest.raises((ci.CommunityGpuError, OSError, ValueError)):
        ci.export_report_archive(results, tmp_path / "report.tar.gz")
    assert not (tmp_path / "report.tar.gz").exists()


def test_standalone_export_retains_preparation_failure_without_case_evidence(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _metadata(results, _index("alpha"), phase="environment", entered=False)
    wrapper = tmp_path / "community_gpu_ci.py"
    wrapper.write_bytes(Path(ci.__file__).read_bytes())
    archive = tmp_path / "report.tar.gz"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(wrapper),
            "--export-report-archive",
            str(archive),
            "--results-directory",
            str(results),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    with tarfile.open(archive) as bundle:
        assert bundle.getnames() == ["summary.json", "evidence-index.json"]
        summary = json.load(bundle.extractfile("summary.json"))
        assert not summary["passed"]
        assert summary["families"][0]["entrypoint_started"] is False
        assert summary["families"][0]["cases"] == {"case": "not_run"}


@pytest.mark.parametrize("failure", ["test", "oom", "timeout", "preparation", "collector"])
def test_family_collection_precedes_removal_and_never_promotes_failures(
    tmp_path, monkeypatch, failure
):
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    owner = repository / "families/alpha/tests/manifests"
    owner.mkdir(parents=True)
    (repository / "families/alpha/model.py").touch()
    (repository / "families/alpha/tests/test_e2e.py").touch()
    (owner / "case.json").write_text(
        json.dumps({"family": "alpha", "testcases": [{"name": "case", "premerge": True}]})
    )
    results = tmp_path / "results"
    image = "sha256:" + "c" * 64
    calls = []
    original = subprocess.run

    def prepare(*args):
        if failure == "preparation":
            raise RuntimeError("dependency preparation failed")
        return image

    def transport(command, **kwargs):
        if command[0] == "git":
            return original(command, **kwargs)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=image)
        if "--stage-family" in command:
            return subprocess.CompletedProcess(command, 0)
        if command[:2] == ["docker", "run"]:
            mounts = {
                v.rsplit(":", 1)[1]: Path(v.rsplit(":", 1)[0])
                for i, v in enumerate(command)
                if i and command[i - 1] == "--volume" and not v.endswith(":ro")
            }
            capture = mounts["/tmp/trtmc-community-evidence"]
            cache = mounts["/tmp/trtmc-community-huggingface"]
            assert capture != cache
            assert "TRTMC_E2E_ARTIFACT_DIR=/tmp/trtmc-community-evidence" in command
            assert f"TRTMC_E2E_SOURCE_REVISION={SOURCE}" in command
            assert "TRTMC_E2E_WORKFLOW_ATTEMPT=2" in command
            assert "TRTMC_E2E_WORKFLOW_RUN_ID=42" in command
            (cache / "result.json.entrypoint").write_text("alpha\n")
            (cache / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "family": "alpha",
                        "status": "failed",
                        "phase": "validation",
                        "failure_class": "pr_failure",
                        "requested_cases": ["case"],
                        "cases": {"case": "failed"},
                    }
                )
            )
            if failure != "oom":
                _record(capture)
            calls.append("run")
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1)
            return subprocess.CompletedProcess(command, 1)
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"OOMKilled": failure == "oom"})
            )
        assert command[:3] == ["docker", "rm", "--force"]
        index = json.loads((results / "evidence-index.json").read_text())
        assert index["cases"][0]["status"] == (
            "missing" if failure in {"oom", "collector"} else "collected"
        )
        calls.append("remove")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ci.subprocess, "run", transport)
    monkeypatch.setattr(
        ci, "_image_preparation", lambda: SimpleNamespace(ensure_family_image=prepare)
    )
    if failure == "collector":

        def failed_collection(*args):
            raise OSError("report destination unavailable")

        monkeypatch.setattr(ci, "collect_family_evidence", failed_collection)
    environment = {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
        "TRTMC_GPU_RESULTS_DIR": str(results),
        "TRTMC_E2E_SOURCE_REVISION": SOURCE,
        "TRTMC_E2E_WORKFLOW_RUN_ID": "42",
        "TRTMC_E2E_WORKFLOW_ATTEMPT": "2",
    }
    with pytest.raises(ci.CommunityGpuError):
        ci.run_containers(repository, environment, "base")
    summary = json.loads((results / "summary.json").read_text())
    assert not summary["passed"] and summary["families"][0]["status"] == "failed"
    assert calls == ([] if failure == "preparation" else ["run", "remove"])
    if failure == "preparation":
        assert summary["families"][0]["failure_class"] == "infra_failure"
        assert (
            json.loads((results / "evidence-index.json").read_text())["cases"][0]["status"]
            == "missing"
        )
    else:
        assert summary["families"][0]["failure_class"] == "pr_failure"
    if failure == "collector":
        assert summary["families"][0]["report_collection_error"] == "evidence_collection_failed"

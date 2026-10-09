# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate image-producer admission, credential lifetime and cleanup boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import signal
import stat
import subprocess
import sys
import time
import urllib.response
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".github/scripts/community_dependency_image.py"
SPEC = importlib.util.spec_from_file_location("community_dependency_image", SOURCE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WORKFLOW = yaml.load(
    (ROOT / ".github/workflows/community-dependency-image.yml").read_text(),
    Loader=yaml.BaseLoader,
)


def test_registered_entry_keeps_image_production_out_of_pr_statuses() -> None:
    caller = yaml.load(
        (ROOT / ".github/workflows/community-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    inputs = caller["on"]["workflow_dispatch"]["inputs"]
    assert inputs["task"]["default"] == "test"
    assert inputs["task"]["options"] == [
        "test",
        "dependency-image",
        "dependency-image-audit",
        "dependency-image-withdraw",
        "dependency-image-access-check",
    ]
    jobs = caller["jobs"]
    producer = jobs["produce-dependency-image"]
    assert producer["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.task == 'dependency-image' }}"
    )
    assert producer["uses"] == "./.github/workflows/community-dependency-image.yml"
    assert producer["with"] == {
        "family": "${{ inputs.audit_family }}",
        "digest": "${{ inputs.audit_digest }}",
    }
    assert set(producer["secrets"]) == {
        "BREV_API_KEY",
        "HF_TOKEN",
        "TRTMC_COMMUNITY_REGISTRY_READ_TOKEN",
        "TRTMC_COMMUNITY_REGISTRY",
    }
    assert all(
        value == "${{ secrets." + name + " }}" for name, value in producer["secrets"].items()
    )
    assert set(WORKFLOW["on"]["workflow_call"]["secrets"]) == {
        "BREV_API_KEY",
        "HF_TOKEN",
        "TRTMC_COMMUNITY_REGISTRY_READ_TOKEN",
        "TRTMC_COMMUNITY_REGISTRY",
    }
    assert set(producer["secrets"]) == set(WORKFLOW["on"]["workflow_call"]["secrets"])
    for name in ("snapshot", "authorize", "required"):
        assert "inputs.task != 'dependency-image'" in jobs[name]["if"]
    for name, job in jobs.items():
        if name != "withdraw-dependency-image":
            assert job.get("permissions", {}).get("packages") != "write"
    withdrawal = jobs["withdraw-dependency-image"]
    assert withdrawal["permissions"] == {"contents": "read", "packages": "write"}
    assert "uses" not in withdrawal and "secrets" not in withdrawal
    authorization = withdrawal["steps"][0]["run"]
    assert 'test "$GITHUB_EVENT_NAME" = workflow_dispatch' in authorization
    assert "refs/heads/main|refs/heads/ci/developer" in authorization
    assert "admin|maintain" in authorization
    assert caller["concurrency"]["cancel-in-progress"] == "false"


def workflow_condition(expression: str, *, task: str, snapshot: str = "") -> bool:
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = expression.replace("!cancelled()", "not cancelled()")
    return bool(
        eval(
            expression,
            {"__builtins__": {}},
            {
                "github": SimpleNamespace(
                    event_name="workflow_dispatch", repository="NVIDIA/TensorRT-Model-Connect"
                ),
                "inputs": SimpleNamespace(task=task, source_snapshot=snapshot),
                "cancelled": lambda: False,
                "always": lambda: True,
                "contains": lambda sequence, item: item in sequence,
                "fromJSON": json.loads,
                "true": True,
                "false": False,
            },
        )
    )


def test_registered_audit_cannot_enter_test_or_producer_jobs() -> None:
    caller = yaml.load(
        (ROOT / ".github/workflows/community-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    jobs = caller["jobs"]
    audit = jobs["audit-dependency-image"]
    assert "uses" not in audit
    assert audit["runs-on"] == "ubuntu-24.04"
    assert audit["permissions"] == {"contents": "read", "packages": "read"}
    assert not audit.get("secrets")
    for task in (
        "test",
        "dependency-image",
        "dependency-image-audit",
        "dependency-image-withdraw",
        "dependency-image-access-check",
    ):
        assert workflow_condition(jobs["withdraw-dependency-image"]["if"], task=task) == (
            task == "dependency-image-withdraw"
        )
        assert workflow_condition(audit["if"], task=task) == (task == "dependency-image-audit")
        assert workflow_condition(jobs["produce-dependency-image"]["if"], task=task) == (
            task == "dependency-image"
        )
        assert workflow_condition(jobs["check-dependency-image-access"]["if"], task=task) == (
            task == "dependency-image-access-check"
        )
        for name, snapshot in (("snapshot", ""), ("authorize", "frozen"), ("required", "frozen")):
            assert workflow_condition(jobs[name]["if"], task=task, snapshot=snapshot) == (
                task == "test"
            )
    assert WORKFLOW["jobs"]["produce"]["if"] == "${{ needs.authorize.outputs.allowed == 'true' }}"
    assert WORKFLOW["jobs"]["cleanup"]["if"] == (
        "${{ always() && needs.authorize.result == 'success' && needs.produce.outputs.instance_name != '' }}"
    )
    assert "audit" not in WORKFLOW["jobs"]


def test_audit_job_has_no_cloud_or_production_credentials() -> None:
    caller = yaml.load(
        (ROOT / ".github/workflows/community-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    job = caller["jobs"]["audit-dependency-image"]
    assert not job.get("needs")
    assert not job.get("environment")
    assert not job.get("secrets")
    steps = job["steps"]
    authorization = next(
        step for step in steps if "collaborators/$REQUEST_ACTOR/permission" in step.get("run", "")
    )
    assert authorization["env"]["REQUEST_ACTOR"] == "${{ github.triggering_actor }}"
    assert "collaborators/$REQUEST_ACTOR/permission" in authorization["run"]
    assert "admin|maintain" in authorization["run"]
    checkout = next(step for step in steps if "actions/checkout@" in step.get("uses", ""))
    assert checkout["with"] == {"ref": "${{ github.sha }}", "persist-credentials": "false"}
    assert steps[0]["env"]["REQUEST_ACTOR"] == "${{ github.triggering_actor }}"
    metadata = next(
        step for step in steps if "community_dependency_image.py audit" in step.get("run", "")
    )
    assert steps.index(checkout) < steps.index(authorization) < steps.index(metadata)
    commands = "\n".join(step.get("run", "") for step in steps)
    assert all(word not in commands for word in ("brev ", "docker ", "--token-file", " publish "))
    assert all(
        "BREV_API_KEY" not in step.get("env", {}) and "HF_TOKEN" not in step.get("env", {})
        for step in steps
    )


def candidate(directory: Path, native: object = True, family: object = True) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "family": "nemotron_h",
                "native_byok_passed": native,
                "family_e2e_passed": family,
                "source_sha": "a" * 40,
                "resolved_dependencies_sha256": "b" * 64,
                "base_local_image": "local-base",
                "base_local_image_id": "sha256:" + "d" * 64,
                "local_image": "local-family",
            }
        )
    )
    return path


@pytest.mark.parametrize("native,family", [(False, True), (True, False), ("true", True)])
def test_unqualified_publication_never_contacts_registry_and_scrubs_token(
    tmp_path: Path, native: object, family: object
) -> None:
    candidate(tmp_path, native, family)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    with patch.object(MODULE, "run") as run:
        with pytest.raises(RuntimeError, match="qualification"):
            MODULE.publish(
                tmp_path,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
    run.assert_not_called()
    assert not token.exists()
    assert not (tmp_path / "published-candidate.json").exists()


def test_push_failure_scrubs_token_and_docker_credentials(tmp_path: Path) -> None:
    candidate(tmp_path)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    directories: list[Path] = []

    def run(command: list[str], **kwargs: object) -> str:
        assert "sensitive-token" not in " ".join(command)
        directory = Path(command[command.index("--config") + 1])
        directories.append(directory)
        assert directory.stat().st_mode & 0o777 == 0o700
        if "login" in command:
            assert kwargs["stdin"] == "sensitive-token"
            (directory / "config.json").write_text("sensitive-token")
        if "push" in command:
            raise subprocess.CalledProcessError(1, command)
        return ""

    with (
        patch.object(MODULE, "run", side_effect=run),
        patch.object(MODULE, "require_private_package"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            MODULE.publish(
                tmp_path,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
    assert not token.exists()
    assert directories and all(not directory.exists() for directory in directories)
    assert not (tmp_path / "published-candidate.json").exists()


def test_publication_records_only_immutable_digests(tmp_path: Path) -> None:
    path = candidate(tmp_path)
    initial = json.loads(path.read_text())
    initial["base_image"] = "unqualified-base-must-not-be-admitted"
    path.write_text(json.dumps(initial))
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    pushed: list[str] = []

    def run(command: list[str], **kwargs: object) -> str:
        assert "sensitive-token" not in " ".join(command)
        if "push" in command:
            pushed.append(command[-1])
        if "inspect" in command:
            return json.dumps([command[-1].rsplit(":", 1)[0] + "@sha256:" + "c" * 64])
        return ""

    with (
        patch.object(MODULE, "run", side_effect=run),
        patch.object(MODULE, "require_private_package"),
    ):
        MODULE.publish(
            tmp_path,
            "ghcr.io/nvidia/tensorrt-model-connect-community",
            "actor",
            token,
            family="nemotron_h",
        )
    receipt = json.loads((tmp_path / "published-candidate.json").read_text())
    assert receipt["image"].endswith("@sha256:" + "c" * 64)
    assert "base_image" not in receipt
    assert receipt["base_local_image_id"] == "sha256:" + "d" * 64
    assert len(pushed) == 1 and "/nemotron_h:" in pushed[0]
    assert "sensitive-token" not in json.dumps(receipt)
    assert not token.exists()


def test_publication_exports_only_the_nonsecret_receipt_with_private_parent_ownership(
    tmp_path,
    monkeypatch,
):
    """Separate proof/auth directories exercise the real transfer permission boundary."""
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    previous = os.umask(0o077)
    real_chown = os.chown
    chowns = []

    def chown(path, uid, gid):
        chowns.append((Path(path), uid, gid))
        real_chown(path, uid, gid)

    def run(command, **kwargs):
        if "inspect" in command:
            return json.dumps(
                ["ghcr.io/nvidia/tensorrt-model-connect-community/nemotron_h@sha256:" + "c" * 64]
            )
        return ""

    try:
        candidate(proof)
        auth.mkdir(mode=0o700)
        token = auth / "token"
        token.write_text("sensitive-token")
        owner = auth.stat()
        proof_before = proof.stat()
        monkeypatch.setattr(MODULE.os, "chown", chown)
        with (
            patch.object(MODULE, "run", side_effect=run),
            patch.object(MODULE, "require_private_package"),
        ):
            MODULE.publish(
                proof,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
        private_receipt, export = (
            proof / "published-candidate.json",
            auth / "published-candidate.json",
        )
        assert json.loads(private_receipt.read_text()) == json.loads(export.read_text())
        assert stat.S_IMODE(export.stat().st_mode) == 0o600
        assert (export.stat().st_uid, export.stat().st_gid) == (owner.st_uid, owner.st_gid)
        assert chowns == [(export, owner.st_uid, owner.st_gid)]
        assert (proof.stat().st_uid, proof.stat().st_mode) == (
            proof_before.st_uid,
            proof_before.st_mode,
        )
        assert (auth.stat().st_uid, auth.stat().st_gid, auth.stat().st_mode) == (
            owner.st_uid,
            owner.st_gid,
            owner.st_mode,
        )
        assert stat.S_IMODE(private_receipt.stat().st_mode) == 0o600
        assert "sensitive-token" not in export.read_text() and not token.exists()
    finally:
        os.umask(previous)


@pytest.mark.parametrize(
    "failure", ["unqualified", "previsibility", "push", "postvisibility", "digest"]
)
def test_failed_publication_never_exports_a_receipt_or_relaxes_private_gates(tmp_path, failure):
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    candidate(proof, native=failure != "unqualified")
    auth.mkdir(mode=0o700)
    token = auth / "token"
    token.write_text("sensitive-token")
    directories = []

    def run(command, **kwargs):
        directory = Path(command[command.index("--config") + 1])
        directories.append(directory)
        if "login" in command:
            (directory / "config.json").write_text("sensitive-token")
        if "push" in command and failure == "push":
            raise subprocess.CalledProcessError(1, command)
        if "inspect" in command:
            return json.dumps(["mutable:latest"])
        return ""

    visibility = (
        [RuntimeError("visibility unavailable")]
        if failure == "previsibility"
        else [None, RuntimeError("visibility unavailable")]
        if failure == "postvisibility"
        else [None, None]
    )
    with (
        patch.object(MODULE, "run", side_effect=run),
        patch.object(MODULE, "require_private_package", side_effect=visibility),
        patch.object(MODULE.os, "chown") as chown,
    ):
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            MODULE.publish(
                proof,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
    assert not token.exists() and all(not directory.exists() for directory in directories)
    assert not (proof / "published-candidate.json").exists()
    assert not (auth / "published-candidate.json").exists()
    chown.assert_not_called()


def test_qualification_uses_common_hub_pin_and_complete_trusted_checkout():
    steps = WORKFLOW["jobs"]["produce"]["steps"]
    access = next(step for step in steps if step.get("id") == "access")
    assert "--staging-hub-requirement" in access["run"]
    assert '--ci-sha "$GITHUB_SHA"' in access["run"]
    setup = next(step for step in steps if "HUB_REQUIREMENT" in step.get("env", {}))
    assert setup["env"]["HUB_REQUIREMENT"] == "${{ steps.access.outputs.hub_requirement }}"
    assert "'$HUB_REQUIREMENT'" in setup["run"]
    assert "huggingface-hub==0.36.0" not in setup["run"]
    assert "test -f '$repo/tools/community_gpu_ci.py'" in setup["run"]
    assert "test -f '$repo/tools/community_gpu_images.py'" in setup["run"]


def test_qualification_download_uses_only_the_ssh_owned_auth_receipt():
    steps = WORKFLOW["jobs"]["produce"]["steps"]
    proof = next(step for step in steps if step.get("id") == "proof")
    assert "$INSTANCE_NAME:$REMOTE_AUTH/qualification.json" in proof["run"]
    assert "$INSTANCE_NAME:$REMOTE_PROOF/qualification.json" not in proof["run"]
    assert "export-qualification" in proof["run"]
    assert proof["timeout-minutes"] == "5"


@pytest.mark.parametrize(
    "xml",
    [
        "<testsuites/>",
        '<testsuite><testcase name="byok_tvm_ffi"><skipped/></testcase></testsuite>',
        '<testsuite><testcase name="byok_tvm_ffi"><failure/></testcase></testsuite>',
        '<testsuite><testcase name="unrelated"/></testsuite>',
    ],
)
def test_native_abi_gate_requires_actual_unskipped_target(tmp_path: Path, xml: str) -> None:
    report = tmp_path / "native.xml"
    report.write_text(xml)
    with pytest.raises(RuntimeError):
        MODULE.require_native_pass(report)


def test_family_e2e_does_not_run_before_native_abi_gate(tmp_path: Path) -> None:
    candidate(tmp_path, False, False)
    with patch.object(MODULE.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match="Native ABI"):
            MODULE.qualify(tmp_path, Path("/stage/python"), None, family="nemotron_h")
    run.assert_not_called()


def completed_family_summary(family: str = "nemotron_h") -> dict:
    return {
        "schema_version": 1,
        "complete": True,
        "passed": True,
        "families": [
            {
                "family": family,
                "status": "passed",
                "phase": "complete",
                "failure_class": None,
                "requested_cases": ["unchanged_a"],
                "cases": {"unchanged_a": "passed"},
                "deferred_cases": [],
            }
        ],
    }


def write_family_summary(environment: dict, summary: dict) -> None:
    directory = Path(environment["TRTMC_GPU_RESULTS_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary))


def test_producer_requires_coverage_of_its_qualified_family(tmp_path: Path) -> None:
    candidate(tmp_path)

    def coordinator(command, *, env, check):
        summary = completed_family_summary()
        summary["families"][0]["dependency_image"] = "ghcr.io/private/unexported"
        summary["families"][0]["secret_payload"] = "never-export-row-payload"
        write_family_summary(env, summary)

    with patch.object(MODULE.subprocess, "run", side_effect=coordinator) as run:
        MODULE.qualify(
            tmp_path,
            Path("/stage/python"),
            None,
            repository=Path("/protected/model"),
            family="nemotron_h",
        )
    assert "--require-family-coverage" in run.call_args.args[0]
    assert "--dependencies-prepared" in run.call_args.args[0]
    command = run.call_args.args[0]
    assert command[command.index("--repository") + 1] == "/protected/model"
    assert command[command.index("--image") + 1] == "local-family"
    environment = run.call_args.kwargs["env"]
    assert json.loads(environment["TRTMC_GPU_FAMILIES"]) == ["nemotron_h"]
    assert json.loads(environment["TRTMC_GPU_DIRECT_FAMILIES"]) == ["nemotron_h"]
    assert json.loads(environment["TRTMC_GPU_ADDED_FAMILIES"]) == []
    receipt = json.loads((tmp_path / "candidate.json").read_text())
    assert receipt["cases"] == {"unchanged_a": "passed"}
    assert "ghcr.io" not in json.dumps(receipt)
    assert "never-export-row-payload" not in json.dumps(receipt)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "incomplete",
        "failed",
        "wrong-owner",
        "empty",
        "skipped",
        "missing-requested",
        "stale",
    ],
)
def test_family_qualification_requires_fresh_complete_actual_owner_e2e_evidence(
    tmp_path: Path, failure: str
):
    path = candidate(tmp_path, family=False)
    summary = completed_family_summary()
    row = summary["families"][0]
    if failure == "incomplete":
        summary["complete"] = False
    elif failure == "failed":
        summary["passed"] = False
    elif failure == "wrong-owner":
        row["family"] = "another_family"
    elif failure == "empty":
        row["requested_cases"] = []
        row["cases"] = {}
    elif failure == "skipped":
        row["cases"] = {"unchanged_a": "skipped"}
    elif failure == "missing-requested":
        row["requested_cases"] = ["unchanged_a", "unchanged_b"]
    if failure == "stale":
        directory = tmp_path / "family-results"
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps(summary))

    def coordinator(command, *, env, check):
        if failure not in {"missing", "stale"}:
            write_family_summary(env, summary)

    with patch.object(MODULE.subprocess, "run", side_effect=coordinator):
        with pytest.raises(RuntimeError):
            MODULE.qualify(tmp_path, Path("/stage/python"), None, family="nemotron_h")
    assert json.loads(path.read_text())["family_e2e_passed"] is False


def test_failed_original_e2e_clears_any_previous_pass_and_blocks_export(tmp_path):
    path = candidate(tmp_path)
    auth = tmp_path / "auth"
    auth.mkdir(mode=0o700)
    directory = tmp_path / "family-results"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps(completed_family_summary()))
    with patch.object(
        MODULE.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(17, ["original-family-runner"]),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            MODULE.qualify(tmp_path, Path("/stage/python"), None, family="nemotron_h")
    assert json.loads(path.read_text())["family_e2e_passed"] is False
    assert not (directory / "summary.json").exists()
    with pytest.raises(RuntimeError):
        MODULE.export_qualification(tmp_path, auth)
    assert not (auth / "qualification.json").exists()


def admission_program(entry: str) -> str:
    if entry == "audit":
        caller = yaml.load(
            (ROOT / ".github/workflows/community-ci.yml").read_text(), Loader=yaml.BaseLoader
        )
        program = caller["jobs"]["audit-dependency-image"]["steps"][0]["run"]
    else:
        program = WORKFLOW["jobs"]["authorize"]["steps"][0]["run"]
    return program.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize("entry", ["producer", "audit"])
@pytest.mark.parametrize(
    "repo,event,ref",
    [
        ("foreign/repo", "workflow_dispatch", "refs/heads/main"),
        ("NVIDIA/TensorRT-Model-Connect", "pull_request_target", "refs/heads/main"),
        ("NVIDIA/TensorRT-Model-Connect", "workflow_dispatch", "refs/pull/1/merge"),
        ("NVIDIA/TensorRT-Model-Connect", "workflow_dispatch", "refs/heads/unprotected-topic"),
    ],
)
def test_unauthorized_entry_fails_before_permission_or_vm_calls(
    tmp_path: Path, repo: str, event: str, ref: str, entry: str
) -> None:
    program = admission_program(entry)
    environment = {
        "GITHUB_REPOSITORY": repo,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_REF": ref,
        "GITHUB_SHA": "a" * 40,
        "GITHUB_ACTOR": "actor",
        "REQUEST_ACTOR": "actor",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    with patch.dict(os.environ, environment), patch("subprocess.check_output") as lookup:
        with pytest.raises(SystemExit):
            exec(compile(program, "producer-authorization", "exec"), {})
    lookup.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_workflow_keeps_credentials_late_and_both_cleanup_paths() -> None:
    jobs = WORKFLOW["jobs"]
    produce = jobs["produce"]
    steps = produce["steps"]
    prepare = next(step for step in steps if " prepare-candidate " in step.get("run", ""))
    qualify = next(step for step in steps if " qualify --family " in step.get("run", ""))
    proof = next(step for step in steps if step.get("id") == "proof")
    release = next(step for step in steps if step.get("id") == "release")
    assert prepare["env"]["REGISTRY_TOKEN"] == "${{ secrets.TRTMC_COMMUNITY_REGISTRY_READ_TOKEN }}"
    assert "HF_TOKEN" not in prepare.get("env", {})
    assert "REGISTRY_TOKEN" not in qualify.get("env", {})
    assert "REGISTRY_TOKEN" not in proof.get("env", {})
    assert steps.index(prepare) < steps.index(qualify) < steps.index(proof) < steps.index(release)
    commands = "\n".join(step.get("run", "") for step in steps)
    assert '" build --family' not in commands and '" publish --family' not in commands
    assert "docker build" not in commands and "docker push" not in commands
    assert '--auth-file "$REMOTE_AUTH/registry-auth.json"' in prepare["run"]
    assert "unset REGISTRY_TOKEN REGISTRY_PREFIX REGISTRY_USERNAME" in prepare["run"]
    assert "trap 'rm -f \"$auth_file\"' EXIT" in prepare["run"]
    assert '--repository "$REMOTE_MODEL"' in prepare["run"]
    assert '--repository "$REMOTE_MODEL"' in qualify["run"]
    assert "unset HF_TOKEN" in qualify["run"]
    assert "always()" in release["if"] and "--until-deleted" in release["run"]
    backup = jobs["cleanup"]
    assert "always()" in backup["if"]
    backup_release = next(
        step for step in backup["steps"] if "--until-deleted" in step.get("run", "")
    )
    assert "always()" in backup_release["if"]
    assert "GITHUB_RUN_ATTEMPT" not in backup_release["run"]
    assert WORKFLOW["concurrency"]["cancel-in-progress"] == "false"
    assert produce["timeout-minutes"] == "360" and backup["timeout-minutes"] == "360"
    assert "packages" not in produce["permissions"]
    assert "packages" not in backup["permissions"]
    copies = [
        line.strip()
        for step in steps
        for line in step.get("run", "").splitlines()
        if "brev copy " in line
    ]
    assert len(copies) == 3
    assert all(
        line.startswith("timeout --signal=TERM --kill-after=10s 120s brev copy") for line in copies
    )
    assert all(
        step["timeout-minutes"] == "5"
        for step in steps
        if "upload-artifact@" in step.get("uses", "")
    )
    before_cleanup = steps[: steps.index(release)]
    assert sum(int(step["timeout-minutes"]) for step in before_cleanup) <= 300
    assert (
        int(produce["timeout-minutes"])
        - sum(int(step["timeout-minutes"]) for step in before_cleanup)
        >= 60
    )


def test_authorization_uses_triggering_actor_and_freezes_protected_main(tmp_path: Path) -> None:
    program = WORKFLOW["jobs"]["authorize"]["steps"][0]["run"]
    program = program.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    environment = {
        "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/ci/developer",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_ACTOR": "original",
        "REQUEST_ACTOR": "current-maintainer",
        "CANDIDATE_FAMILY": "nemotron_h",
        "CANDIDATE_DIGEST": "sha256:" + "c" * 64,
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    responses = [
        json.dumps({"role_name": "maintain", "permission": "write"}),
        json.dumps({"object": {"sha": "d" * 40}}),
    ]
    with (
        patch.dict(os.environ, environment),
        patch("subprocess.check_output", side_effect=responses) as lookup,
    ):
        exec(compile(program, "producer-authorization", "exec"), {})
    assert "current-maintainer/permission" in lookup.call_args_list[0].args[0][-1]
    assert lookup.call_args_list[1].args[0][-1].endswith("git/ref/heads/main")
    assert (tmp_path / "output").read_text() == (
        "allowed=true\nmodel_sha="
        + "d" * 40
        + "\nfamily=nemotron_h\ndigest=sha256:"
        + "c" * 64
        + "\n"
    )


@pytest.mark.parametrize("entry", ["producer", "audit"])
@pytest.mark.parametrize("role", ["read", "triage", "write"])
def test_rerun_cannot_inherit_original_actors_permission(
    tmp_path: Path, role: str, entry: str
) -> None:
    program = admission_program(entry)
    environment = {
        "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/ci/developer",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_ACTOR": "original-maintainer",
        "REQUEST_ACTOR": "rerunning-reader",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    with (
        patch.dict(os.environ, environment),
        patch("subprocess.check_output", return_value=json.dumps({"role_name": role})) as lookup,
    ):
        with pytest.raises(SystemExit):
            exec(compile(program, "producer-authorization", "exec"), {})
    assert lookup.call_count == 1
    assert lookup.call_args.args[0][-1].endswith("collaborators/rerunning-reader/permission")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("ref", ["refs/heads/main", "refs/heads/ci/developer"])
def test_audit_admission_checks_current_actor_before_protected_checkout(ref: str) -> None:
    environment = {
        "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": ref,
        "GITHUB_SHA": "a" * 40,
        "GITHUB_ACTOR": "original-reader",
        "REQUEST_ACTOR": "current-maintainer",
    }
    with (
        patch.dict(os.environ, environment),
        patch(
            "subprocess.check_output", return_value=json.dumps({"role_name": "maintain"})
        ) as lookup,
    ):
        exec(compile(admission_program("audit"), "audit-authorization", "exec"), {})
    assert lookup.call_count == 1
    assert lookup.call_args.args[0][-1].endswith("collaborators/current-maintainer/permission")


def test_audit_admission_rejects_mutable_source_before_any_permission_lookup() -> None:
    environment = {
        "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/ci/developer",
        "GITHUB_SHA": "ci/developer",
        "REQUEST_ACTOR": "maintainer",
    }
    with patch.dict(os.environ, environment), patch("subprocess.check_output") as lookup:
        with pytest.raises(SystemExit):
            exec(compile(admission_program("audit"), "audit-authorization", "exec"), {})
    lookup.assert_not_called()


def test_wrong_registry_scrubs_private_token_before_any_contact(tmp_path: Path) -> None:
    candidate(tmp_path)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    with patch.object(MODULE, "run") as run:
        with pytest.raises(RuntimeError, match="namespace"):
            MODULE.publish(tmp_path, "ghcr.io/foreign/project", "actor", token, family="nemotron_h")
    run.assert_not_called()
    assert not token.exists()


def test_arm_host_cannot_start_x86_production(tmp_path: Path) -> None:
    with (
        patch.object(MODULE.platform, "machine", return_value="aarch64"),
        patch.object(MODULE, "run") as run,
    ):
        with pytest.raises(RuntimeError, match="real x86_64"):
            MODULE.build(tmp_path, ROOT, family="nemotron_h")
    run.assert_not_called()


def test_family_path_cannot_escape_recipe_root_or_start_build(tmp_path: Path) -> None:
    with patch.object(MODULE, "run") as run:
        with pytest.raises(RuntimeError, match="family directory"):
            MODULE.build(tmp_path, ROOT, family="../another_recipe")
    run.assert_not_called()


def test_different_family_candidate_cannot_run_or_publish(tmp_path: Path) -> None:
    candidate(tmp_path)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    with patch.object(MODULE.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match="different family"):
            MODULE.qualify(tmp_path, Path("/stage/python"), None, family="other_family")
        with pytest.raises(RuntimeError, match="different family"):
            MODULE.publish(
                tmp_path,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="other_family",
            )
    run.assert_not_called()
    assert not token.exists()


def test_existing_public_registry_namespace_blocks_all_pushes(tmp_path: Path) -> None:
    candidate(tmp_path)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    with (
        patch.object(MODULE, "run") as run,
        patch.object(MODULE, "require_private_package", side_effect=RuntimeError("public package")),
    ):
        with pytest.raises(RuntimeError, match="public"):
            MODULE.publish(
                tmp_path,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
    run.assert_not_called()
    assert not token.exists()
    assert not (tmp_path / "published-candidate.json").exists()


def test_postpush_private_visibility_is_required_for_catalog_receipt(tmp_path: Path) -> None:
    candidate(tmp_path)
    token = tmp_path / "token"
    token.write_text("sensitive-token")
    with (
        patch.object(MODULE, "run", return_value=""),
        patch.object(
            MODULE,
            "require_private_package",
            side_effect=[None, RuntimeError("visibility unknown")],
        ),
    ):
        with pytest.raises(RuntimeError, match="visibility"):
            MODULE.publish(
                tmp_path,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
    assert not token.exists()
    assert not (tmp_path / "published-candidate.json").exists()


def test_registry_api_redirect_never_forwards_authorization() -> None:
    requests: list[str] = []

    class RedirectTransport(MODULE.urllib.request.HTTPSHandler):
        def https_open(self, request):
            requests.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://another-origin.invalid/credentials"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "Found"
            return response

    real_opener = MODULE.urllib.request.build_opener
    with patch.object(
        MODULE.urllib.request,
        "build_opener",
        side_effect=lambda *handlers: real_opener(*handlers, RedirectTransport()),
    ):
        with pytest.raises(RuntimeError, match="visibility") as error:
            MODULE.require_private_package(
                "ghcr.io/nvidia/tensorrt-model-connect-community/nemotron_h",
                "sensitive-token",
            )
    assert len(requests) == 1 and requests[0].startswith("https://api.github.com/")
    assert "sensitive-token" not in str(error.value)


@pytest.mark.parametrize("value", [None, [], "private"])
def test_registry_api_non_object_json_fails_safely(value: object) -> None:
    response = io.BytesIO(json.dumps(value).encode())
    response.status = 200
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", return_value=response):
        with pytest.raises(RuntimeError, match="visibility"):
            MODULE.require_private_package(
                "ghcr.io/nvidia/tensorrt-model-connect-community/nemotron_h",
                "sensitive-token",
            )


@pytest.mark.parametrize(
    "lookup",
    [
        404,
        403,
        401,
        500,
        "public",
        "internal",
        "missing",
        "invalid",
        "network",
        "non200",
        "private",
    ],
)
def test_full_image_push_requires_authenticated_200_private_before_any_docker_command(
    tmp_path,
    lookup,
    capsys,
):
    """Run the real publication gate; first-push absence is never private evidence."""
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    candidate(proof)
    auth.mkdir(mode=0o700)
    token = auth / "token"
    token.write_text("sensitive-token")
    events = []
    requests = []

    def api(request, **kwargs):
        requests.append(request)
        events.append("package-get")
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer sensitive-token"
        assert kwargs["timeout"] == 30
        if isinstance(lookup, int):
            raise MODULE.urllib.error.HTTPError(
                request.full_url, lookup, "private body", {}, io.BytesIO(b"sensitive-token")
            )
        if lookup == "network":
            raise OSError("sensitive-token network failure")
        payload = (
            b"not JSON"
            if lookup == "invalid"
            else json.dumps(
                {
                    "visibility": None
                    if lookup == "missing"
                    else "private"
                    if lookup == "non200"
                    else lookup
                }
            ).encode()
        )
        response = io.BytesIO(payload)
        response.status = 202 if lookup == "non200" else 200
        return response

    def docker(command, **kwargs):
        assert "sensitive-token" not in str(command)
        for action in ("login", "tag", "push", "inspect"):
            if action in command:
                events.append(action)
        if "inspect" in command:
            return json.dumps(
                ["ghcr.io/nvidia/tensorrt-model-connect-community/nemotron_h@sha256:" + "c" * 64]
            )
        return ""

    with (
        patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=api),
        patch.object(MODULE, "run", side_effect=docker) as run,
    ):
        if lookup == "private":
            MODULE.publish(
                proof,
                "ghcr.io/nvidia/tensorrt-model-connect-community",
                "actor",
                token,
                family="nemotron_h",
            )
            assert events == ["package-get", "login", "tag", "push", "package-get", "inspect"]
            assert (
                json.loads((auth / "published-candidate.json").read_text())["registry_visibility"]
                == "private"
            )
        else:
            with pytest.raises(RuntimeError) as failure:
                MODULE.publish(
                    proof,
                    "ghcr.io/nvidia/tensorrt-model-connect-community",
                    "actor",
                    token,
                    family="nemotron_h",
                )
            run.assert_not_called()
            assert events == ["package-get"]
            assert (
                not (proof / "published-candidate.json").exists()
                and not (auth / "published-candidate.json").exists()
            )
            if lookup in (404, 403, 401, 500, "network", "invalid", "non200"):
                assert "Bootstrap a known-private package" in str(failure.value)
            assert "sensitive-token" not in str(failure.value)
    assert not token.exists()
    output = capsys.readouterr()
    assert "sensitive-token" not in output.out + output.err


def test_failed_native_probe_preserves_bounded_real_subprocess_diagnostics(capsys) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; print('probe-start'); "
        "sys.stderr.write('x' * 20000 + '\\nNative ABI: missing_symbol\\n'); sys.exit(17)",
    ]
    with pytest.raises(subprocess.CalledProcessError) as error:
        MODULE.run(command, capture=True)
    assert error.value.returncode == 17
    output = capsys.readouterr()
    assert "probe-start" in output.err and "Native ABI: missing_symbol" in output.err
    assert output.out == "" and len(output.err) < 16500


@pytest.mark.parametrize("capture", [True, False])
def test_failed_real_credential_command_never_echoes_response_body(capsys, capture: bool) -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; token = sys.stdin.read(); print('rejected:' + token); "
        "sys.stderr.write('response:' + token); sys.exit(9)",
    ]
    with pytest.raises(RuntimeError, match="Credential command failed") as error:
        MODULE.run(command, capture=capture, stdin="sensitive-token")
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert "sensitive-token" not in str(error.value)


AUDIT_DIGEST = "sha256:" + "e" * 64
AUDIT_PACKAGE = "tensorrt-model-connect-community/nemotron_h"


def audit_package_response(visibility: str = "private") -> dict:
    return {
        "name": AUDIT_PACKAGE,
        "visibility": visibility,
        "id": 123,
        "repository": {"full_name": "NVIDIA/TensorRT-Model-Connect", "id": 456, "private": True},
        "secret_payload": "sensitive-token",
        "owner": {"login": "do-not-export-owner"},
    }


def audit_version_response(name: str = AUDIT_DIGEST) -> dict:
    return {
        "id": 789,
        "name": name,
        "created_at": "2026-10-08T00:00:00Z",
        "updated_at": "2026-10-08T01:00:00Z",
        "metadata": {"container": {"tags": ["qualified-candidate"]}, "secret": "sensitive-token"},
        "url": "https://api.github.com/do-not-export-url",
    }


def expected_audit(visibility: str = "private", *, matches: bool = True) -> dict:
    return {
        "package_name": AUDIT_PACKAGE,
        "visibility": visibility,
        "id": 123,
        "repository": {"full_name": "NVIDIA/TensorRT-Model-Connect", "id": 456},
        "root_keys": sorted(audit_package_response()),
        "matching_versions": [
            {
                "id": 789,
                "name": AUDIT_DIGEST,
                "created_at": "2026-10-08T00:00:00Z",
                "updated_at": "2026-10-08T01:00:00Z",
                "tags": ["qualified-candidate"],
            }
        ]
        if matches
        else [],
    }


@pytest.mark.parametrize("visibility", ["private", "public", "internal"])
def test_package_audit_is_get_only_and_exports_only_safe_fields(visibility: str, capsys) -> None:
    requests = []
    responses = [audit_package_response(visibility), [audit_version_response()]]

    def open_response(request, *, timeout):
        requests.append(request)
        assert request.get_method() == "GET"
        assert request.full_url.startswith(
            "https://api.github.com/orgs/NVIDIA/packages/container/"
            "tensorrt-model-connect-community%2Fnemotron_h"
        )
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["authorization"] == "Bearer sensitive-token"
        assert headers["x-github-api-version"] == "2022-11-28"
        assert "sensitive-token" not in request.full_url
        assert 0 < timeout <= 30
        return io.BytesIO(json.dumps(responses.pop(0)).encode())

    with (
        patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=open_response),
        patch.object(MODULE, "run") as run,
        patch.object(MODULE, "save") as save,
    ):
        receipt = MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert receipt == expected_audit(visibility)
    assert len(requests) == 2 and "/versions?" in requests[-1].full_url
    assert "sensitive-token" not in json.dumps(receipt)
    assert all(key not in receipt for key in ("image", "native_byok_passed", "family_e2e_passed"))
    run.assert_not_called()
    save.assert_not_called()
    assert capsys.readouterr().out == ""


def test_package_audit_pagination_keeps_digest_matching_exact() -> None:
    urls: list[str] = []
    nonmatch = audit_version_response(AUDIT_DIGEST + "-suffix")

    def open_response(request, *, timeout):
        urls.append(request.full_url)
        if "/versions?" not in request.full_url:
            return io.BytesIO(json.dumps(audit_package_response()).encode())
        query = MODULE.urllib.parse.parse_qs(MODULE.urllib.parse.urlsplit(request.full_url).query)
        assert query["per_page"] == ["100"]
        assert query["page"] == [str(len(urls) - 1)]
        rows = [nonmatch] * 100 if len(urls) == 2 else [audit_version_response()]
        return io.BytesIO(json.dumps(rows).encode())

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=open_response):
        receipt = MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert receipt == expected_audit()
    assert len(urls) == 3


def test_package_audit_fails_closed_at_bounded_version_inventory_limit() -> None:
    urls: list[str] = []

    def open_response(request, *, timeout):
        urls.append(request.full_url)
        if "/versions?" not in request.full_url:
            return io.BytesIO(json.dumps(audit_package_response()).encode())
        query = MODULE.urllib.parse.parse_qs(MODULE.urllib.parse.urlsplit(request.full_url).query)
        assert query["per_page"] == ["100"]
        assert query["page"] == [str(len(urls) - 1)]
        return io.BytesIO(json.dumps([audit_version_response("unrelated-digest")] * 100).encode())

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=open_response):
        with pytest.raises(RuntimeError, match="bounded"):
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert len(urls) == 11


def test_package_audit_rejects_oversized_response_before_export() -> None:
    body = io.BytesIO(b" " * (1024 * 1024 + 1))
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", return_value=body):
        with pytest.raises(RuntimeError):
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")


@pytest.mark.parametrize(
    "family,digest",
    [
        ("../nemotron_h", AUDIT_DIGEST),
        ("nemotron_h", "e" * 64),
        ("nemotron_h", "sha256:" + "e" * 63),
    ],
)
def test_invalid_audit_target_never_contacts_github(family: str, digest: str) -> None:
    with patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact:
        with pytest.raises((RuntimeError, ValueError)):
            MODULE.audit_package(family, digest, "sensitive-token")
    contact.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_package_audit_http_failure_never_echoes_credentials_or_response_body(
    status: int, capsys
) -> None:
    error = MODULE.urllib.error.HTTPError(
        "https://api.github.com/packages",
        status,
        "sensitive-token",
        Message(),
        io.BytesIO(b"private-response-sensitive-token"),
    )
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=error):
        with pytest.raises(RuntimeError) as caught:
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert "sensitive-token" not in str(caught.value)
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize("body", [b"not-json", b"null", b"[]", b'"private"'])
def test_package_audit_rejects_malformed_package_response_without_body_disclosure(
    body: bytes, capsys
) -> None:
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", return_value=io.BytesIO(body)):
        with pytest.raises(RuntimeError):
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("body", [b"not-json", b"null", b"{}", b"[null]"])
def test_package_audit_rejects_malformed_versions_response(body: bytes) -> None:
    responses = [io.BytesIO(json.dumps(audit_package_response()).encode()), io.BytesIO(body)]
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=responses):
        with pytest.raises(RuntimeError):
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")


def test_package_audit_redirect_cannot_send_token_to_another_origin() -> None:
    requests: list[str] = []

    class RedirectTransport(MODULE.urllib.request.HTTPSHandler):
        def https_open(self, request):
            requests.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://another-origin.invalid/credentials"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "Found"
            return response

    real_opener = MODULE.urllib.request.build_opener
    with patch.object(
        MODULE.urllib.request,
        "build_opener",
        side_effect=lambda *handlers: real_opener(*handlers, RedirectTransport()),
    ):
        with pytest.raises(RuntimeError) as caught:
            MODULE.audit_package("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert len(requests) == 1 and requests[0].startswith("https://api.github.com/")
    assert "sensitive-token" not in str(caught.value)


@pytest.mark.parametrize("visibility", ["private", "public"])
def test_audit_cli_uses_environment_token_and_preserves_evidence_without_admission(
    tmp_path: Path, visibility: str, capsys
) -> None:
    receipt = expected_audit(visibility)
    arguments = [
        str(SOURCE),
        "audit",
        "--family",
        "nemotron_h",
        "--digest",
        AUDIT_DIGEST,
        "--output",
        str(tmp_path),
    ]
    with (
        patch.dict(os.environ, {"GH_TOKEN": "sensitive-token"}),
        patch.object(sys, "argv", arguments),
        patch.object(MODULE, "audit_package", return_value=receipt) as audit,
        patch.object(MODULE, "run") as run,
        patch.object(MODULE, "publish") as publish,
    ):
        if visibility == "private":
            MODULE.main()
        else:
            with pytest.raises((SystemExit, RuntimeError)):
                MODULE.main()
    audit.assert_called_once_with("nemotron_h", AUDIT_DIGEST, "sensitive-token")
    assert json.loads((tmp_path / "dependency-image-audit.json").read_text()) == receipt
    assert not (tmp_path / "published-candidate.json").exists()
    assert "sensitive-token" not in capsys.readouterr().out
    run.assert_not_called()
    publish.assert_not_called()


def test_audit_cli_without_environment_token_fails_before_contact(tmp_path: Path) -> None:
    arguments = [
        str(SOURCE),
        "audit",
        "--family",
        "nemotron_h",
        "--digest",
        AUDIT_DIGEST,
        "--output",
        str(tmp_path),
    ]
    with (
        patch.dict(os.environ, {"GH_TOKEN": ""}),
        patch.object(sys, "argv", arguments),
        patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact,
    ):
        with pytest.raises((SystemExit, RuntimeError)):
            MODULE.main()
    contact.assert_not_called()
    assert not (tmp_path / "dependency-image-audit.json").exists()


class RegistryResponse(io.BytesIO):
    status = 200


@pytest.mark.parametrize(
    "anonymous,repository,denied",
    [
        (401, 403, True),
        (404, 404, True),
        (200, 403, False),
        (403, 200, False),
        (500, 403, False),
        (403, 503, False),
        (0, 403, False),
    ],
)
def test_registry_denial_requires_both_real_manifest_checks(anonymous, repository, denied, capsys):
    requests = []
    statuses = iter((anonymous, repository))

    def transport(request, *, timeout):
        requests.append(request)
        assert timeout == 15 and request.full_url.startswith("https://ghcr.io/")
        if "/token?" in request.full_url:
            return RegistryResponse(b'{"token":"ephemeral-bearer-secret"}')
        assert request.get_method() == "HEAD"
        assert request.get_header("Authorization") == "Bearer ephemeral-bearer-secret"
        status = next(statuses)
        if status == 200:
            return RegistryResponse(b"")
        if status == 0:
            raise TimeoutError("sensitive-token and private registry URL")
        raise MODULE.urllib.error.HTTPError(
            request.full_url, status, "sensitive-token", {}, io.BytesIO(b"private response")
        )

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=transport):
        receipt = MODULE.check_registry_access(
            "test_family",
            AUDIT_DIGEST,
            "ghcr.io/test-owner/private-dependencies",
            "public-repo-token",
            "test-actor",
        )
    assert receipt["denied"] is denied and len(requests) == 4
    assert requests[0].get_header("Authorization") is None
    encoded = requests[2].get_header("Authorization").removeprefix("Basic ")
    assert MODULE.base64.b64decode(encoded).decode() == "test-actor:public-repo-token"
    captured = capsys.readouterr()
    output = captured.out + captured.err + json.dumps(receipt)
    assert not any(
        secret in output
        for secret in (
            "ghcr.io",
            "private-dependencies",
            "public-repo-token",
            "ephemeral-bearer-secret",
            "sensitive-token",
        )
    )
    if anonymous in (0, 500):
        assert receipt["anonymous"]["status"] == "unknown"


@pytest.mark.parametrize("status", [401, 403, 404, 500, 302])
def test_registry_token_endpoint_denial_is_distinct_from_unknown(status):
    def transport(request, *, timeout):
        assert "/token?" in request.full_url
        raise MODULE.urllib.error.HTTPError(request.full_url, status, "secret", {}, None)

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=transport) as calls:
        receipt = MODULE.check_registry_access(
            "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "token", "test-actor"
        )
    assert calls.call_count == 2 and receipt["denied"] is (status in (401, 403, 404))


@pytest.mark.parametrize(
    "media_type",
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    ],
)
def test_readable_manifest_formats_cannot_be_mistaken_for_denial(media_type):
    def registry(request, *, timeout):
        if "/token?" in request.full_url:
            return RegistryResponse(b'{"token":"manifest-reader"}')
        # This registry serves a known readable digest only when its format is
        # negotiated. An incomplete Accept header would incorrectly get 404.
        if media_type not in request.get_header("Accept", "").split(", "):
            raise MODULE.urllib.error.HTTPError(request.full_url, 404, "not acceptable", {}, None)
        return RegistryResponse(b"")

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=registry):
        receipt = MODULE.check_registry_access(
            "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "token", "test-actor"
        )
    assert not receipt["denied"] and all(
        receipt[actor]["status"] == "accessible"
        for actor in ("anonymous", "public_repository_token")
    )


@pytest.mark.parametrize("body", [b"{}", b"[]", b'{"token":null}', b"not-json", b"x" * 65537])
def test_invalid_registry_exchange_is_unknown_without_response_disclosure(body):
    with patch.object(
        MODULE.urllib.request.OpenerDirector,
        "open",
        side_effect=lambda *a, **kw: RegistryResponse(body),
    ) as calls:
        receipt = MODULE.check_registry_access(
            "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "token", "test-actor"
        )
    assert calls.call_count == 2 and receipt["denied"] is False
    assert all(
        receipt[actor]["status"] == "unknown" for actor in ("anonymous", "public_repository_token")
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "https://ghcr.io/test-owner",
        "ghcr.io/test-owner?secret=value",
        "ghcr.io/credential@owner",
        "foreign.example/test-owner",
        "ghcr.io/test-owner/../other",
    ],
)
def test_access_check_rejects_untrusted_registry_before_contact(prefix):
    with patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact:
        with pytest.raises(RuntimeError) as failure:
            MODULE.check_registry_access(
                "alpha", AUDIT_DIGEST, prefix, "sensitive-token", "test-actor"
            )
    contact.assert_not_called()
    assert prefix not in str(failure.value) and "sensitive-token" not in str(failure.value)


@pytest.mark.parametrize("username", ["", "credential:token", "login\nheader"])
def test_access_check_cannot_use_an_invalid_public_token_username(username):
    with patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact:
        with pytest.raises(RuntimeError, match="username is unavailable"):
            MODULE.check_registry_access(
                "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "token", username
            )
    contact.assert_not_called()


def test_registry_exchange_redirect_cannot_forward_a_credential():
    requested = []

    class RedirectTransport(MODULE.urllib.request.HTTPSHandler):
        def https_open(self, request):
            requested.append(request.full_url)
            headers = Message()
            headers["Location"] = "https://foreign.example/steal"
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "Found"
            return response

    opener = MODULE.urllib.request.build_opener
    with patch.object(
        MODULE.urllib.request,
        "build_opener",
        side_effect=lambda *handlers: opener(*handlers, RedirectTransport()),
    ):
        receipt = MODULE.check_registry_access(
            "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "sensitive-token", "test-actor"
        )
    assert len(requested) == 2 and all(
        url.startswith("https://ghcr.io/token?") for url in requested
    )
    assert not receipt["denied"] and "sensitive-token" not in json.dumps(receipt)


@pytest.mark.parametrize("denied", [True, False])
def test_access_check_cli_uses_only_public_token_and_preserves_safe_result(
    tmp_path, denied, capsys
):
    receipt = {"family": "alpha", "digest": AUDIT_DIGEST, "denied": denied}
    argv = [
        str(SOURCE),
        "access-check",
        "--family",
        "alpha",
        "--digest",
        AUDIT_DIGEST,
        "--output",
        str(tmp_path),
    ]
    with (
        patch.object(sys, "argv", argv),
        patch.dict(
            os.environ,
            {
                "GH_TOKEN": "public-repo-token",
                "GITHUB_ACTOR": "test-actor",
                "TRTMC_COMMUNITY_REGISTRY": "ghcr.io/test-owner/cache",
                "TRTMC_COMMUNITY_REGISTRY_READ_TOKEN": "must-not-be-used",
            },
        ),
        patch.object(MODULE, "check_registry_access", return_value=receipt) as check,
    ):
        if denied:
            MODULE.main()
        else:
            with pytest.raises(RuntimeError, match="denial was not established"):
                MODULE.main()
    check.assert_called_once_with(
        "alpha", AUDIT_DIGEST, "ghcr.io/test-owner/cache", "public-repo-token", "test-actor"
    )
    assert json.loads((tmp_path / "dependency-image-access-check.json").read_text()) == receipt
    assert "must-not-be-used" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "event,ref,role,allowed",
    [
        ("workflow_dispatch", "refs/heads/ci/developer", "maintain", True),
        ("workflow_dispatch", "refs/heads/main", "admin", True),
        ("workflow_dispatch", "refs/heads/main", "write", False),
        ("workflow_dispatch", "refs/heads/topic", "maintain", False),
        ("pull_request_target", "refs/heads/main", "maintain", False),
    ],
)
def test_access_check_job_authorizes_current_actor_without_private_reader(
    tmp_path, event, ref, role, allowed
):
    caller = yaml.load(
        (ROOT / ".github/workflows/community-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    job = caller["jobs"]["check-dependency-image-access"]
    assert job["permissions"] == {"contents": "read", "packages": "read"}
    assert job["environment"]["name"] == "gpu-ci-dispatch"
    assert "BREV" not in json.dumps(job) and "REGISTRY_READ_TOKEN" not in json.dumps(job)
    assert not job.get("needs")
    step = job["steps"][0]
    assert step["env"]["REQUEST_ACTOR"] == "${{ github.triggering_actor }}"
    check = next(s for s in job["steps"] if "access-check \\" in s.get("run", ""))
    assert check["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert check["env"]["TRTMC_COMMUNITY_REGISTRY"] == "${{ secrets.TRTMC_COMMUNITY_REGISTRY }}"
    assert "collaborators/$REQUEST_ACTOR/permission" in check["run"]
    trace = tmp_path / "authorization"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'gh() { printf "%s\\n" "$*" >> "$TRACE"; printf "%s" "$ROLE"; }\n' + step["run"],
        ],
        env={
            **os.environ,
            "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF": ref,
            "GITHUB_SHA": "a" * 40,
            "REQUEST_ACTOR": "current-rerun-actor",
            "ROLE": role,
            "TRACE": str(trace),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) is allowed, result.stderr
    if trace.exists():
        assert "current-rerun-actor" in trace.read_text()


PRIVATE_PREFIX = "ghcr.io/test-owner/private-dependencies"
PRIVATE_TOKEN = "candidate-read-secret"
CANDIDATE_DIGEST = "sha256:" + "c" * 64
LOCAL_CANDIDATE_ID = "sha256:" + "d" * 64


def candidate_input_paths(family: str = "nemotron_h") -> tuple[str, ...]:
    """Keep the producer's complete public input contract explicit in this test."""
    return (
        "Dockerfile.dev.x86-gpu",
        "requirements/community-ci.txt",
        "requirements/image-environment.py",
        "requirements/community-gpu-linux-amd64.lock",
        "requirements/community-gpu-linux-amd64.json",
        f"families/{family}/requirements.txt",
        f"families/{family}/ci/Dockerfile.dependencies",
        f"families/{family}/ci/build-dependencies.sh",
        f"families/{family}/ci/constraints-linux-amd64.txt",
        f"families/{family}/ci/environment-linux-amd64.lock",
        f"families/{family}/ci/environment-linux-amd64.json",
    )


def fixture_git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def fixture_commit(repository: Path, message: str) -> str:
    fixture_git(repository, "add", ".")
    fixture_git(repository, "commit", "--quiet", "-m", message)
    return fixture_git(repository, "rev-parse", "HEAD")


@pytest.fixture
def private_candidate_fixture(tmp_path: Path, monkeypatch):
    environment = tmp_path / "environment"
    model = tmp_path / "model"
    for repository in (environment, model):
        repository.mkdir()
        fixture_git(repository, "init", "--quiet")
        fixture_git(repository, "config", "user.name", "Candidate fixture")
        fixture_git(repository, "config", "user.email", "fixture@example.invalid")
    for index, relative in enumerate(candidate_input_paths()):
        target = environment / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"public input {index}\n")
    input_hashes = {
        relative: hashlib.sha256((environment / relative).read_bytes()).hexdigest()
        for relative in candidate_input_paths()
    }
    environment_sha = fixture_commit(environment, "Complete public environment inputs")
    (environment / "README.md").write_text("Later protected CI commit\n")
    (environment / "requirements/community-ci.txt").write_text("Later public dependency input\n")
    protected_sha = fixture_commit(environment, "Protected branch advances")
    fixture_git(environment, "update-ref", "refs/remotes/origin/main", protected_sha)
    requirement = model / "families/nemotron_h/requirements.txt"
    requirement.parent.mkdir(parents=True)
    requirement.write_bytes((environment / "families/nemotron_h/requirements.txt").read_bytes())
    model_sha = fixture_commit(model, "Frozen protected model snapshot")
    fixture_git(model, "update-ref", "refs/remotes/origin/main", model_sha)
    auth_directory = tmp_path / "auth"
    auth_directory.mkdir(mode=0o700)
    auth_file = auth_directory / "registry.json"
    auth_file.write_text(
        json.dumps(
            {"registry_prefix": PRIVATE_PREFIX, "username": "test-actor", "token": PRIVATE_TOKEN}
        )
    )
    auth_file.chmod(0o600)
    metadata = {
        "schema_version": 1,
        "source_sha": environment_sha,
        "mode": "locked",
        "inputs": input_hashes,
        "native_byok_passed": False,
        "family_e2e_passed": False,
    }
    monkeypatch.setattr(MODULE, "SOURCE", environment)
    return SimpleNamespace(
        environment=environment,
        environment_sha=environment_sha,
        protected_sha=protected_sha,
        model=model,
        model_sha=model_sha,
        metadata=metadata,
        auth_directory=auth_directory,
        auth_file=auth_file,
        output=tmp_path / "proof",
    )


def prepare_private_candidate(fixture) -> None:
    MODULE.prepare_candidate(
        fixture.output,
        fixture.model,
        fixture.auth_file,
        family="nemotron_h",
        candidate_digest=CANDIDATE_DIGEST,
    )


def candidate_transport(fixture, events: list[str], *, visibility: str = "private"):
    def transport(request, *, timeout):
        assert timeout == 30
        assert request.full_url.startswith("https://api.github.com/")
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer " + PRIVATE_TOKEN
        assert not fixture.auth_file.exists()
        if request.full_url == "https://api.github.com/users/test-owner":
            events.append("owner-get")
            body = {"login": "test-owner", "type": "Organization"}
        else:
            assert request.full_url == (
                "https://api.github.com/orgs/test-owner/packages/container/"
                "private-dependencies%2Fnemotron_h"
            )
            events.append("package-get")
            body = {"name": "private-dependencies/nemotron_h", "visibility": visibility}
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        return response

    return transport


def candidate_docker_boundary(fixture, events: list[str], *, metadata: str | None = None):
    """Fake Docker only, preserving real Git ancestry and blob reads."""
    configurations: list[Path] = []
    original_run = MODULE.run

    def private_docker(command, *, stdin=None, **kwargs):
        assert command[0] == "docker" and PRIVATE_TOKEN not in str(command)
        assert not fixture.auth_file.exists()
        if "/opt/trtmc-ci/build-inputs.json" in command:
            assert "--config" not in command and stdin is None
            assert kwargs["stdout_limit"] == 65536
            return host_run(command, capture=True)
        assert "--config" in command
        configuration = Path(command[command.index("--config") + 1])
        assert configuration.is_dir()
        configurations.append(configuration)
        if "login" in command:
            events.append("login")
            assert stdin == PRIVATE_TOKEN
            assert command[-1] == "--password-stdin"
            (configuration / "config.json").write_text("private credential simulation")
            return ""
        assert stdin is None
        if "pull" in command:
            events.append("pull")
            assert command[-1] == PRIVATE_PREFIX + "/nemotron_h@" + CANDIDATE_DIGEST
            assert command[command.index("--platform") + 1] == "linux/amd64"
            return ""
        if "inspect" in command:
            events.append("inspect")
            assert command[-1] == PRIVATE_PREFIX + "/nemotron_h@" + CANDIDATE_DIGEST
            return LOCAL_CANDIDATE_ID
        pytest.fail(f"Unexpected private Docker operation: {command[1:3]}")

    def host_run(command, *, capture=False, stdin=None):
        if command[0] == "git":
            return original_run(command, capture=capture, stdin=stdin)
        assert command[:3] == ["docker", "run", "--rm"]
        assert not fixture.auth_file.exists()
        assert configurations and all(not directory.exists() for directory in configurations)
        assert PRIVATE_PREFIX not in str(command) and PRIVATE_TOKEN not in str(command)
        assert "--network" in command and command[command.index("--network") + 1] == "none"
        assert LOCAL_CANDIDATE_ID in command
        if "/opt/trtmc-ci/build-inputs.json" in command:
            events.append("metadata")
            assert command == [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "/usr/bin/head",
                LOCAL_CANDIDATE_ID,
                "-c",
                "65537",
                "/opt/trtmc-ci/build-inputs.json",
            ]
            return metadata if metadata is not None else json.dumps(fixture.metadata)
        if MODULE.PROBE in command:
            events.append("probe")
            return "TRTMC_IMAGE_PROFILE=" + json.dumps(
                {
                    "platform": "linux/amd64",
                    "python_abi": "cp312",
                    "torch": "2.12.0+cu130",
                    "cuda": "13.0",
                    "tensorrt": "11.1.0.106",
                    "cxx11abi": True,
                    "apache_tvm_ffi": "0.1.7",
                    "gpu": "NVIDIA L4",
                    "sm": [8, 9],
                    "resolved_dependencies": ["apache-tvm-ffi==0.1.7", "torch==2.12.0+cu130"],
                }
            )
        if command[-4:] == ["python", "-m", "pip", "check"]:
            events.append("pip-check")
            return ""
        assert command[-3:-1] == ["bash", "-c"]
        assert "test_byok_tvm_ffi" in command[-1]
        assert "--output-junit /proof/native-byok.xml" in command[-1]
        assert "-R '^byok_tvm_ffi$'" in command[-1]
        events.append("native-byok")
        fixture.output.mkdir(parents=True, exist_ok=True)
        (fixture.output / "native-byok.xml").write_text(
            '<testsuite><testcase name="byok_tvm_ffi"/></testsuite>'
        )
        return ""

    return private_docker, host_run


def test_private_candidate_uses_real_protected_ancestor_and_erases_auth_before_image_code(
    private_candidate_fixture, capsys
):
    fixture = private_candidate_fixture
    events: list[str] = []
    private_docker, host_run = candidate_docker_boundary(fixture, events)
    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        prepare_private_candidate(fixture)
    assert events[:6] == ["owner-get", "package-get", "login", "pull", "inspect", "metadata"]
    assert set(events[6:]) == {"probe", "pip-check", "native-byok"}
    receipt = json.loads((fixture.output / "candidate.json").read_text())
    assert receipt["source_sha"] == receipt["environment_source_sha"] == fixture.environment_sha
    assert receipt["model_source_sha"] == fixture.model_sha
    assert fixture.environment_sha != fixture.protected_sha
    assert receipt["digest"] == CANDIDATE_DIGEST
    assert receipt["local_image"] == LOCAL_CANDIDATE_ID
    assert receipt["native_byok_passed"] is True and receipt["family_e2e_passed"] is False
    assert not fixture.auth_file.exists()
    captured = capsys.readouterr()
    public = captured.out + captured.err + json.dumps(receipt)
    assert all(secret not in public for secret in (PRIVATE_PREFIX, PRIVATE_TOKEN, "test-actor"))


@pytest.mark.parametrize(
    "field,relative",
    [
        ("environment_recorder_sha256", "requirements/image-environment.py"),
        ("base_environment_receipt_sha256", "requirements/community-gpu-linux-amd64.json"),
        (
            "family_environment_receipt_sha256",
            "families/nemotron_h/ci/environment-linux-amd64.json",
        ),
    ],
)
def test_complete_environment_hashes_survive_sanitized_qualification_export(
    private_candidate_fixture, field, relative, capsys
):
    fixture = private_candidate_fixture
    unrelated = {"registry_prefix": PRIVATE_PREFIX, "token": PRIVATE_TOKEN}
    fixture.metadata["unrelated_metadata"] = unrelated
    events: list[str] = []
    private_docker, host_run = candidate_docker_boundary(fixture, events)
    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        prepare_private_candidate(fixture)
    candidate_path = fixture.output / "candidate.json"
    prepared = json.loads(candidate_path.read_text())
    expected = fixture.metadata["inputs"][relative]
    assert prepared[field] == expected
    assert "unrelated_metadata" not in prepared

    def coordinator(command, *, env, check):
        write_family_summary(env, completed_family_summary())

    with patch.object(MODULE.subprocess, "run", side_effect=coordinator):
        MODULE.qualify(
            fixture.output, Path("/stage/python"), None, fixture.model, family="nemotron_h"
        )
    qualified = json.loads(candidate_path.read_text())
    qualified["unrelated_metadata"] = unrelated
    candidate_path.write_text(json.dumps(qualified))
    with patch.object(MODULE.os, "chown"):
        MODULE.export_qualification(fixture.output, fixture.auth_directory)
    exported = json.loads((fixture.auth_directory / "qualification.json").read_text())
    assert exported[field] == expected
    assert exported["cases"] == {"unchanged_a": "passed"}
    assert "unrelated_metadata" not in exported and "local_image" not in exported
    captured = capsys.readouterr()
    public = captured.out + captured.err + json.dumps(prepared) + json.dumps(exported)
    assert PRIVATE_PREFIX not in public and PRIVATE_TOKEN not in public


@pytest.mark.parametrize(
    "mutation",
    [
        "bootstrap",
        "boolean-schema",
        "missing-input",
        "extra-input",
        "tampered-input",
        "mutable-source",
        "unprotected-source",
        "model-requirements",
        "native-proof-true",
        "e2e-proof-true",
        "nonboolean-native-proof",
        "nonboolean-e2e-proof",
        "oversized-json",
        "oversized-padded-json",
        "duplicate-key",
        "malformed-json",
    ],
)
def test_private_candidate_rejects_untrusted_embedded_provenance_before_probe_or_native(
    private_candidate_fixture, mutation: str
):
    fixture = private_candidate_fixture
    metadata = fixture.metadata
    text = None
    if mutation == "bootstrap":
        metadata["mode"] = "bootstrap"
    elif mutation == "boolean-schema":
        metadata["schema_version"] = True
    elif mutation == "missing-input":
        metadata["inputs"].pop("requirements/image-environment.py")
    elif mutation == "extra-input":
        metadata["inputs"]["private/source.cpp"] = "f" * 64
    elif mutation == "tampered-input":
        metadata["inputs"]["requirements/community-ci.txt"] = "f" * 64
    elif mutation == "mutable-source":
        metadata["source_sha"] = "main"
    elif mutation == "unprotected-source":
        fixture_git(fixture.environment, "checkout", "--quiet", "--orphan", "unprotected")
        fixture_git(fixture.environment, "rm", "--quiet", "-r", "--cached", ".")
        metadata["source_sha"] = fixture_commit(fixture.environment, "Unrelated contributor tree")
    elif mutation == "model-requirements":
        (fixture.model / "families/nemotron_h/requirements.txt").write_text(
            "different public inputs\n"
        )
        fixture_commit(fixture.model, "Model requirements differ")
    elif mutation == "native-proof-true":
        metadata["native_byok_passed"] = True
    elif mutation == "e2e-proof-true":
        metadata["family_e2e_passed"] = True
    elif mutation == "nonboolean-native-proof":
        metadata["native_byok_passed"] = 0
    elif mutation == "nonboolean-e2e-proof":
        metadata["family_e2e_passed"] = 0
    elif mutation == "oversized-json":
        text = "x" * 65537
    elif mutation == "oversized-padded-json":
        text = json.dumps(metadata)
        text += " " * (65537 - len(text.encode()))
    elif mutation == "duplicate-key":
        text = json.dumps(metadata).replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    elif mutation == "malformed-json":
        text = "sensitive-token malformed metadata"
    events: list[str] = []
    private_docker, host_run = candidate_docker_boundary(fixture, events, metadata=text)
    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError):
            prepare_private_candidate(fixture)
    assert events[-1] == "metadata"
    assert not any(event in events for event in ("probe", "pip-check", "native-byok"))
    assert not (fixture.output / "candidate.json").exists()
    assert not fixture.auth_file.exists()


@pytest.mark.parametrize("protected_ref", ["main", "ci/developer"])
def test_environment_provenance_accepts_either_protected_branch_ancestor(
    private_candidate_fixture, protected_ref: str
):
    fixture = private_candidate_fixture
    fixture_git(fixture.environment, "update-ref", "-d", "refs/remotes/origin/main")
    fixture_git(
        fixture.environment,
        "update-ref",
        "refs/remotes/origin/" + protected_ref,
        fixture.protected_sha,
    )
    events: list[str] = []
    private_docker, host_run = candidate_docker_boundary(fixture, events)
    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        prepare_private_candidate(fixture)
    assert json.loads((fixture.output / "candidate.json").read_text())["native_byok_passed"] is True


@pytest.mark.parametrize(
    "lookup",
    [
        401,
        403,
        404,
        500,
        "public",
        "internal",
        "missing",
        "invalid",
        "network",
        "non200",
        "duplicate-visibility",
        "duplicate-name",
    ],
)
def test_private_candidate_requires_authenticated_200_private_before_any_docker(
    private_candidate_fixture, lookup, capsys
):
    fixture = private_candidate_fixture
    requests = []

    def transport(request, *, timeout):
        requests.append(request)
        assert request.get_header("Authorization") == "Bearer " + PRIVATE_TOKEN
        assert not fixture.auth_file.exists()
        if request.full_url.endswith("/users/test-owner"):
            response = io.BytesIO(b'{"login":"test-owner","type":"Organization"}')
            response.status = 200
            return response
        if isinstance(lookup, int):
            raise MODULE.urllib.error.HTTPError(
                request.full_url, lookup, PRIVATE_TOKEN, {}, io.BytesIO(PRIVATE_PREFIX.encode())
            )
        if lookup == "network":
            raise OSError(PRIVATE_TOKEN + " " + PRIVATE_PREFIX)
        payload = (
            b'{"name":"private-dependencies/nemotron_h","visibility":"public","visibility":"private"}'
            if lookup == "duplicate-visibility"
            else b'{"name":"foreign-package","name":"private-dependencies/nemotron_h","visibility":"private"}'
            if lookup == "duplicate-name"
            else b"not-json"
            if lookup == "invalid"
            else json.dumps(
                {
                    "visibility": None
                    if lookup == "missing"
                    else "private"
                    if lookup == "non200"
                    else lookup
                }
            ).encode()
        )
        response = io.BytesIO(payload)
        response.status = 202 if lookup == "non200" else 200
        return response

    with (
        patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=transport),
        patch.object(MODULE, "_private_docker") as docker,
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError) as failure:
            prepare_private_candidate(fixture)
    docker.assert_not_called()
    assert len(requests) == 2
    assert not fixture.auth_file.exists()
    captured = capsys.readouterr()
    public = str(failure.value) + captured.out + captured.err
    assert all(secret not in public for secret in (PRIVATE_TOKEN, PRIVATE_PREFIX))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-token",
        "missing-username",
        "invalid-prefix",
        "duplicate-key",
        "malformed",
        "oversized",
        "world-readable",
        "symlink",
        "fifo",
    ],
)
def test_candidate_auth_is_bounded_regular_private_json_before_any_contact(
    private_candidate_fixture, mutation: str
):
    fixture = private_candidate_fixture
    auth = json.loads(fixture.auth_file.read_text())
    if mutation == "missing-token":
        auth["token"] = ""
    elif mutation == "missing-username":
        auth["username"] = ""
    elif mutation == "invalid-prefix":
        auth["registry_prefix"] = "ghcr.io/credential@owner?token=" + PRIVATE_TOKEN
    fixture.auth_file.write_text(json.dumps(auth))
    if mutation == "duplicate-key":
        fixture.auth_file.write_text(
            json.dumps(auth).replace(
                '"token": "' + PRIVATE_TOKEN + '"',
                '"token": "' + PRIVATE_TOKEN + '", "token": "' + PRIVATE_TOKEN + '"',
            )
        )
    elif mutation == "malformed":
        fixture.auth_file.write_text(PRIVATE_TOKEN)
    elif mutation == "oversized":
        fixture.auth_file.write_text("x" * 65537)
    elif mutation == "world-readable":
        fixture.auth_file.chmod(0o644)
    elif mutation in ("symlink", "fifo"):
        fixture.auth_file.unlink()
        if mutation == "symlink":
            target = fixture.auth_directory / "private-target"
            target.write_text(json.dumps(auth))
            target.chmod(0o600)
            fixture.auth_file.symlink_to(target)
        else:
            os.mkfifo(fixture.auth_file, 0o600)
    with (
        patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact,
        patch.object(MODULE, "_private_docker") as docker,
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError):
            prepare_private_candidate(fixture)
    contact.assert_not_called()
    docker.assert_not_called()
    assert not fixture.auth_file.exists()


@pytest.mark.parametrize(
    "digest",
    [
        "latest",
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "ghcr.io/private/image@sha256:" + "a" * 64,
    ],
)
def test_prepare_candidate_rejects_mutable_or_coordinate_digest_before_contact(
    private_candidate_fixture, digest: str
):
    fixture = private_candidate_fixture
    with (
        patch.object(MODULE.urllib.request.OpenerDirector, "open") as contact,
        patch.object(MODULE, "_private_docker") as docker,
    ):
        with pytest.raises(RuntimeError):
            MODULE.prepare_candidate(
                fixture.output,
                fixture.model,
                fixture.auth_file,
                family="nemotron_h",
                candidate_digest=digest,
            )
    contact.assert_not_called()
    docker.assert_not_called()
    assert not fixture.auth_file.exists()


@pytest.mark.parametrize("failure", ["process", "timeout", "os-error"])
def test_private_pull_boundary_never_discloses_registry_errors(failure: str, capsys):
    command = ["docker", "pull", PRIVATE_PREFIX + "/nemotron_h@" + CANDIDATE_DIGEST]
    if failure == "process":
        error = subprocess.CalledProcessError(
            17,
            command,
            output=PRIVATE_TOKEN + " " + PRIVATE_PREFIX,
            stderr=PRIVATE_PREFIX + " " + PRIVATE_TOKEN,
        )
    elif failure == "timeout":
        error = subprocess.TimeoutExpired(
            command,
            30,
            output=PRIVATE_TOKEN,
            stderr=PRIVATE_PREFIX,
        )
    else:
        error = OSError(PRIVATE_PREFIX + " " + PRIVATE_TOKEN)
    with patch.object(MODULE, "_bounded_capture", side_effect=error):
        with pytest.raises(RuntimeError) as failure:
            MODULE._private_docker(command, stdin=PRIVATE_TOKEN)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert PRIVATE_PREFIX not in str(failure.value) and PRIVATE_TOKEN not in str(failure.value)
    assert failure.value.__cause__ is None and failure.value.__suppress_context__


def test_metadata_transport_keeps_bytes_for_the_bounded_manifest_check():
    content = "{}" + " " * 65535
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('{}' + ' ' * 65535)",
    ]
    assert MODULE._private_docker(command) == content
    with pytest.raises(RuntimeError, match="output is suppressed"):
        MODULE._private_docker(command, stdout_limit=65536)


@pytest.mark.parametrize("failure", ["login", "pull", "inspect", "mutable-local-image"])
def test_failed_private_pull_cleans_auth_and_config_without_executing_image_code(
    private_candidate_fixture, failure: str
):
    fixture = private_candidate_fixture
    events: list[str] = []
    base_docker, host_run = candidate_docker_boundary(fixture, events)
    configurations = []

    def private_docker(command, **kwargs):
        configurations.append(Path(command[command.index("--config") + 1]))
        result = base_docker(command, **kwargs)
        if failure in command:
            raise RuntimeError("Private candidate transport failed; output is suppressed")
        if failure == "mutable-local-image" and "inspect" in command:
            return "local-candidate:latest"
        return result

    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError):
            prepare_private_candidate(fixture)
    assert configurations and all(not configuration.exists() for configuration in configurations)
    assert not fixture.auth_file.exists()
    assert "metadata" not in events and "probe" not in events and "native-byok" not in events
    assert not (fixture.output / "candidate.json").exists()


def test_prepare_candidate_cli_passes_only_auth_path_not_secret_or_coordinate(
    private_candidate_fixture,
):
    fixture = private_candidate_fixture
    argv = [
        str(SOURCE),
        "prepare-candidate",
        "--family",
        "nemotron_h",
        "--digest",
        CANDIDATE_DIGEST,
        "--repository",
        str(fixture.model),
        "--auth-file",
        str(fixture.auth_file),
        "--output",
        str(fixture.output),
    ]
    with patch.object(sys, "argv", argv), patch.object(MODULE, "prepare_candidate") as prepare:
        MODULE.main()
    prepare.assert_called_once_with(
        fixture.output,
        fixture.model,
        fixture.auth_file,
        family="nemotron_h",
        candidate_digest=CANDIDATE_DIGEST,
    )
    assert PRIVATE_PREFIX not in str(argv) and PRIVATE_TOKEN not in str(argv)


@pytest.mark.parametrize("proof", ["skipped", "failed", "missing", "unrelated"])
def test_preparation_cannot_create_native_pass_from_missing_or_skipped_ctest(
    private_candidate_fixture, proof: str
):
    fixture = private_candidate_fixture
    events: list[str] = []
    private_docker, successful_host_run = candidate_docker_boundary(fixture, events)

    def host_run(command, **kwargs):
        result = successful_host_run(command, **kwargs)
        if command[0] == "docker" and "native-byok" in events:
            report = fixture.output / "native-byok.xml"
            if proof == "missing":
                report.unlink()
            else:
                report.write_text(
                    '<testsuite><testcase name="'
                    + ("unrelated" if proof == "unrelated" else "byok_tvm_ffi")
                    + '">'
                    + {"skipped": "<skipped/>", "failed": "<failure/>", "unrelated": ""}[proof]
                    + "</testcase></testsuite>"
                )
        return result

    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector,
            "open",
            side_effect=candidate_transport(fixture, events),
        ),
        patch.object(MODULE, "_private_docker", side_effect=private_docker),
        patch.object(MODULE, "run", side_effect=host_run),
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises((RuntimeError, FileNotFoundError)):
            prepare_private_candidate(fixture)
    receipt_path = fixture.output / "candidate.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["native_byok_passed"] is False and receipt["family_e2e_passed"] is False
    assert not fixture.auth_file.exists()


def test_qualification_export_chowns_only_sanitized_nonsecret_pending_receipt(tmp_path):
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    proof.mkdir(mode=0o700)
    auth.mkdir(mode=0o700)
    path = candidate(proof)
    record = json.loads(path.read_text())
    record.update(
        {
            "schema_version": 1,
            "environment_source_sha": "e" * 40,
            "model_source_sha": "f" * 40,
            "digest": CANDIDATE_DIGEST,
            "cases": {"unchanged_a": "passed"},
            "inputs": {"Dockerfile.dev.x86-gpu": "d" * 64},
            "abi": {"python_abi": "cp312", "cuda": "13.0", "tensorrt": "11.1.0.106"},
            "local_image": PRIVATE_PREFIX + "/nemotron_h@" + CANDIDATE_DIGEST,
            "registry_prefix": PRIVATE_PREFIX,
            "token": PRIVATE_TOKEN,
            "cleanup_confirmed": True,
            "admitted": True,
        }
    )
    path.write_text(json.dumps(record))
    original_mode = stat.S_IMODE(proof.stat().st_mode)
    original_owner = proof.stat().st_uid, proof.stat().st_gid
    with patch.object(MODULE.os, "chown") as chown:
        MODULE.export_qualification(proof, auth)
    exported = auth / "qualification.json"
    chown.assert_called_once_with(exported, auth.stat().st_uid, auth.stat().st_gid)
    receipt = json.loads(exported.read_text())
    assert receipt["cases"] == {"unchanged_a": "passed"}
    assert receipt["source_sha"] == "a" * 40
    assert receipt["environment_source_sha"] == "e" * 40
    assert receipt["model_source_sha"] == "f" * 40
    assert receipt["digest"] == CANDIDATE_DIGEST
    assert receipt["native_byok_passed"] is True and receipt["family_e2e_passed"] is True
    assert receipt["cleanup_confirmed"] is False and receipt["admitted"] is False
    assert stat.S_IMODE(exported.stat().st_mode) == 0o600
    assert stat.S_IMODE(proof.stat().st_mode) == original_mode
    assert (proof.stat().st_uid, proof.stat().st_gid) == original_owner
    assert all(key not in receipt for key in ("local_image", "registry_prefix", "token"))
    assert all(secret not in exported.read_text() for secret in (PRIVATE_PREFIX, PRIVATE_TOKEN))


@pytest.mark.parametrize("native,e2e", [(False, True), (True, False), (1, True), (True, 1)])
def test_qualification_export_requires_both_exact_actual_passes(tmp_path, native, e2e):
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    auth.mkdir(mode=0o700)
    candidate(proof, native=native, family=e2e)
    with patch.object(MODULE.os, "chown") as chown:
        with pytest.raises(RuntimeError):
            MODULE.export_qualification(proof, auth)
    chown.assert_not_called()
    assert not (auth / "qualification.json").exists()


@pytest.mark.parametrize("coordinate", [PRIVATE_PREFIX, "https://private.example/receipt"])
def test_qualification_export_rejects_nested_nonpublic_coordinates(tmp_path, coordinate):
    proof, auth = tmp_path / "proof", tmp_path / "auth"
    auth.mkdir(mode=0o700)
    path = candidate(proof)
    record = json.loads(path.read_text())
    record["abi"] = {"unexpected": coordinate}
    path.write_text(json.dumps(record))
    with patch.object(MODULE.os, "chown") as chown:
        with pytest.raises(RuntimeError):
            MODULE.export_qualification(proof, auth)
    chown.assert_not_called()
    assert not (auth / "qualification.json").exists()


@pytest.mark.parametrize(
    "owner",
    [
        {"type": "Organization", "login": None},
        {"type": "Organization", "login": []},
        {"type": "Organization", "login": "another-owner"},
        {"type": "Repository", "login": "test-owner"},
        {"type": [], "login": "test-owner"},
        {"login": "test-owner"},
    ],
)
def test_private_candidate_owner_must_be_verified_before_package_or_docker(
    private_candidate_fixture, owner
):
    fixture = private_candidate_fixture
    response = io.BytesIO(json.dumps(owner).encode())
    response.status = 200
    with (
        patch.object(
            MODULE.urllib.request.OpenerDirector, "open", return_value=response
        ) as contact,
        patch.object(MODULE, "_private_docker") as docker,
        patch.object(MODULE.platform, "machine", return_value="x86_64"),
    ):
        with pytest.raises(RuntimeError):
            prepare_private_candidate(fixture)
    assert contact.call_count == 1
    docker.assert_not_called()
    assert not fixture.auth_file.exists()


def test_private_candidate_supports_a_verified_user_owned_private_package():
    requests = []

    def transport(request, *, timeout):
        requests.append(request)
        assert timeout == 30 and request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer " + PRIVATE_TOKEN
        if len(requests) == 1:
            assert request.full_url == "https://api.github.com/users/test-owner"
            body = {"login": "test-owner", "type": "User"}
        else:
            assert request.full_url == (
                "https://api.github.com/users/test-owner/packages/container/"
                "private-dependencies%2Fnemotron_h"
            )
            body = {"name": "private-dependencies/nemotron_h", "visibility": "private"}
        response = io.BytesIO(json.dumps(body).encode())
        response.status = 200
        return response

    with patch.object(MODULE.urllib.request.OpenerDirector, "open", side_effect=transport):
        MODULE._require_private_candidate(PRIVATE_PREFIX, "nemotron_h", PRIVATE_TOKEN)
    assert len(requests) == 2


@pytest.mark.parametrize("copy_fails", [False, True])
def test_workflow_copies_private_auth_as_json_and_erases_it_even_on_copy_failure(
    tmp_path, copy_fails
):
    prepare = next(
        step
        for step in WORKFLOW["jobs"]["produce"]["steps"]
        if " prepare-candidate " in step.get("run", "")
    )
    trace, copied_auth = tmp_path / "trace", tmp_path / "copied-auth.json"
    script = r"""
    brev() {
      test "$1" = copy
      cp "$2" "$CAPTURE_AUTH"
      printf '%s\n' "$*" >> "$TRACE"
      if [ "$COPY_FAILS" = true ]; then return 17; fi
    }
    timeout() { shift 3; "$@"; }
    python3() {
      if [ "$1" = - ]; then command python3 "$@"; return; fi
      test "${REGISTRY_TOKEN+x}" != x
      test "${REGISTRY_PREFIX+x}" != x
      test "${REGISTRY_USERNAME+x}" != x
      printf '%s\n' "$*" >> "$TRACE"
    }
    """
    result = subprocess.run(
        ["bash", "-c", script + prepare["run"]],
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "INSTANCE_NAME": "fake-instance",
            "REMOTE_REPO": "/tmp/protected-ci",
            "REMOTE_MODEL": "/tmp/protected-model",
            "REMOTE_PROOF": "/tmp/public-proof",
            "REMOTE_AUTH": "/tmp/private-auth",
            "CANDIDATE_FAMILY": "nemotron_h",
            "CANDIDATE_DIGEST": CANDIDATE_DIGEST,
            "REGISTRY_TOKEN": PRIVATE_TOKEN + "\n'quoted-value",
            "REGISTRY_PREFIX": PRIVATE_PREFIX,
            "REGISTRY_USERNAME": "test-actor",
            "CAPTURE_AUTH": str(copied_auth),
            "TRACE": str(trace),
            "COPY_FAILS": "true" if copy_fails else "false",
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == (17 if copy_fails else 0), result.stderr
    assert not (tmp_path / "community-image-read-auth.json").exists()
    assert json.loads(copied_auth.read_text()) == {
        "registry_prefix": PRIVATE_PREFIX,
        "username": "test-actor",
        "token": PRIVATE_TOKEN + "\n'quoted-value",
    }
    assert stat.S_IMODE(copied_auth.stat().st_mode) == 0o600
    output = result.stdout + result.stderr + trace.read_text()
    assert PRIVATE_TOKEN not in output and PRIVATE_PREFIX not in output
    assert ("prepare-candidate" in trace.read_text()) is (not copy_fails)


@pytest.mark.parametrize(
    "token,prefix,identity_available,allowed",
    [
        ("read-secret", PRIVATE_PREFIX, True, True),
        ("", PRIVATE_PREFIX, True, False),
        ("read-secret", "https://ghcr.io/test-owner/cache", True, False),
        ("read-secret", PRIVATE_PREFIX, False, False),
    ],
)
def test_private_reader_is_resolved_before_vm_allocation(
    tmp_path, token, prefix, identity_available, allowed
):
    from tools import community_gpu_images as images

    steps = WORKFLOW["jobs"]["produce"]["steps"]
    gate = next(
        step
        for step in steps
        if step["name"] == "Require protected candidate access before allocation"
    )
    reserve = next(step for step in steps if step.get("id") == "reserve")
    prepare = next(step for step in steps if " prepare-candidate " in step.get("run", ""))
    assert steps.index(gate) < steps.index(reserve)
    assert gate["env"]["REGISTRY_TOKEN"] == "${{ secrets.TRTMC_COMMUNITY_REGISTRY_READ_TOKEN }}"
    assert "REGISTRY_USERNAME" not in gate["env"]
    assert prepare["env"]["REGISTRY_USERNAME"] == "${{ steps.access.outputs.registry_username }}"
    program = gate["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "access-output"
    with (
        patch.dict(
            os.environ,
            {
                "REGISTRY_TOKEN": token,
                "REGISTRY_PREFIX": prefix,
                "GITHUB_OUTPUT": str(output),
            },
            clear=True,
        ),
        patch.object(
            images,
            "registry_reader_login",
            return_value="test-actor",
            side_effect=None if identity_available else images.ImagePreparationError("unavailable"),
        ) as resolve,
    ):
        if allowed:
            exec(compile(program, "protected-candidate-reader", "exec"), {})
            assert output.read_text() == "registry_username=test-actor\n"
            resolve.assert_called_once_with(token)
        else:
            with pytest.raises(SystemExit):
                exec(compile(program, "protected-candidate-reader", "exec"), {})
            assert not output.exists()
            if not token or prefix != PRIVATE_PREFIX:
                resolve.assert_not_called()
            else:
                resolve.assert_called_once_with(token)


def final_gpu_proof_program() -> str:
    step = next(
        step
        for step in WORKFLOW["jobs"]["cleanup"]["steps"]
        if step.get("name") == "Confirm cleanup in the final GPU qualification proof"
    )
    return step["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize("backstop", ["true", "false", ""])
def test_final_gpu_proof_requires_actual_backstop_confirmation(tmp_path, backstop):
    directory = tmp_path / "image-gpu-proof"
    directory.mkdir()
    pending = {
        "native_byok_passed": True,
        "family_e2e_passed": True,
        "cleanup_confirmed": False,
        "admitted": False,
        "cases": {"unchanged_a": "passed"},
    }
    (directory / "qualification-pending.json").write_text(json.dumps(pending))
    with patch.dict(
        os.environ,
        {"RUNNER_TEMP": str(tmp_path), "GITHUB_RUN_ID": "123", "BACKSTOP_CONFIRMED": backstop},
    ):
        if backstop == "true":
            exec(compile(final_gpu_proof_program(), "final-gpu-proof", "exec"), {})
        else:
            with pytest.raises(SystemExit):
                exec(compile(final_gpu_proof_program(), "final-gpu-proof", "exec"), {})
    final = directory / "qualification.json"
    assert final.exists() is (backstop == "true")
    if final.exists():
        proof = json.loads(final.read_text())
        assert proof["owner_cleanup_confirmed"] is True
        assert proof["backstop_cleanup_confirmed"] is True
        assert proof["cleanup_confirmed"] is True
        assert proof["admitted"] is False
        assert proof["resources"] == {"host_ram_gib": 128}
        assert proof["qualification_host"] == {
            "ram_gib": 128,
            "gpu_count": 1,
            "arch": "x86_64",
            "run_id": "123",
        }


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_byok_passed", False),
        ("native_byok_passed", 1),
        ("family_e2e_passed", False),
        ("family_e2e_passed", 1),
        ("cleanup_confirmed", True),
        ("cleanup_confirmed", 0),
        ("admitted", True),
        ("admitted", 0),
    ],
)
def test_pending_gpu_proof_cannot_prematurely_claim_admission_or_cleanup(tmp_path, field, value):
    directory = tmp_path / "image-gpu-proof"
    directory.mkdir()
    pending = {
        "native_byok_passed": True,
        "family_e2e_passed": True,
        "cleanup_confirmed": False,
        "admitted": False,
    }
    pending[field] = value
    (directory / "qualification-pending.json").write_text(json.dumps(pending))
    with patch.dict(
        os.environ,
        {"RUNNER_TEMP": str(tmp_path), "GITHUB_RUN_ID": "123", "BACKSTOP_CONFIRMED": "true"},
    ):
        with pytest.raises(SystemExit):
            exec(compile(final_gpu_proof_program(), "final-gpu-proof", "exec"), {})
    assert not (directory / "qualification.json").exists()


def test_final_gpu_artifact_is_gated_by_both_cleanup_paths_and_never_admits_catalog():
    jobs = WORKFLOW["jobs"]
    owner_steps = jobs["produce"]["steps"]
    owner_release = next(step for step in owner_steps if step.get("id") == "release")
    pending_upload = next(
        step
        for step in owner_steps
        if step.get("name") == "Preserve sanitized pending proof after owner cleanup"
    )
    assert owner_steps.index(owner_release) < owner_steps.index(pending_upload)
    assert "steps.release.outputs.cleanup_confirmed == 'true'" in pending_upload["if"]
    backup = jobs["cleanup"]
    assert backup["needs"] == ["authorize", "produce"]
    backup_steps = backup["steps"]
    backup_release = next(step for step in backup_steps if "--until-deleted" in step.get("run", ""))
    finalizer = next(
        step
        for step in backup_steps
        if step.get("name") == "Confirm cleanup in the final GPU qualification proof"
    )
    final_upload = next(
        step
        for step in backup_steps
        if step.get("name") == "Preserve final GPU proof only after both cleanup confirmations"
    )
    assert backup_release["id"] == "backstop_release"
    assert (
        finalizer["env"]["BACKSTOP_CONFIRMED"]
        == "${{ steps.backstop_release.outputs.cleanup_confirmed }}"
    )
    assert (
        backup_steps.index(backup_release)
        < backup_steps.index(finalizer)
        < backup_steps.index(final_upload)
    )
    for step in (finalizer, final_upload):
        assert "success()" in step["if"]
        assert "needs.produce.result == 'success'" in step["if"]
        assert "needs.produce.outputs.cleanup_confirmed == 'true'" in step["if"]
    assert final_upload["with"]["path"] == "${{ runner.temp }}/image-gpu-proof/qualification.json"
    assert all(job.get("permissions", {}).get("packages") != "write" for job in jobs.values())
    assert "dependency-image.json" not in json.dumps(WORKFLOW)


@pytest.fixture
def bounded_children(monkeypatch):
    children = []
    original_popen = MODULE.subprocess.Popen

    def popen(*arguments, **kwargs):
        child = original_popen(*arguments, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(MODULE.subprocess, "Popen", popen)
    yield children
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)


def assert_capture_closed_and_reaped(children) -> None:
    assert children
    for child in children:
        assert child.poll() is not None
        assert child.stdout.closed and child.stderr.closed
        assert child.stdin is None or child.stdin.closed


@pytest.mark.parametrize("stream,descriptor", [("stdout", 1), ("stderr", 2)])
def test_receiver_limits_each_actual_child_pipe_before_accumulation(
    bounded_children, monkeypatch, stream, descriptor
):
    command = [
        sys.executable,
        "-c",
        f"import os, time; os.write({descriptor}, b'x' * 8192); time.sleep(10)",
    ]
    requested_reads = []
    original_read = MODULE.os.read

    def read(descriptor, size):
        if any(
            not stream.closed and stream.fileno() == descriptor
            for child in bounded_children
            for stream in (child.stdout, child.stderr)
        ):
            requested_reads.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(MODULE.os, "read", read)
    started = time.monotonic()
    with pytest.raises(MODULE.CapturedOutputLimit, match=stream) as failure:
        MODULE._bounded_capture(command, timeout=2, stdout_limit=4096, stderr_limit=4096)
    assert time.monotonic() - started < 2
    assert requested_reads and max(requested_reads) <= 4097
    assert len(failure.value.stdout.encode()) <= 4096
    assert len(failure.value.stderr.encode()) <= 4096
    assert_capture_closed_and_reaped(bounded_children)


@pytest.mark.parametrize("descriptor", [1, 2])
def test_public_probe_default_capture_is_bounded_and_diagnostics_keep_only_tail(
    bounded_children, descriptor, capsys
):
    command = [
        sys.executable,
        "-c",
        f"import os, time; os.write({descriptor}, b'x' * (1024 * 1024 + 65536)); time.sleep(10)",
    ]
    with pytest.raises(MODULE.CapturedOutputLimit):
        MODULE.run(command, capture=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Captured" in captured.err and "x" * 16384 in captured.err
    assert len(captured.err) < 16500
    assert_capture_closed_and_reaped(bounded_children)


@pytest.mark.parametrize("capture", [False, True])
def test_actual_credential_output_flood_is_suppressed_and_reaped(bounded_children, capture, capsys):
    command = [
        sys.executable,
        "-c",
        "import sys, time; token=sys.stdin.read(); sys.stdout.write(token * 100000); "
        "sys.stdout.flush(); time.sleep(10)",
    ]
    with pytest.raises(RuntimeError, match="Credential command failed") as failure:
        MODULE.run(command, capture=capture, stdin=PRIVATE_TOKEN)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert PRIVATE_TOKEN not in str(failure.value)
    assert_capture_closed_and_reaped(bounded_children)


def test_private_metadata_receiver_rejects_the_65537th_actual_byte_during_read(
    bounded_children, capsys
):
    command = [
        sys.executable,
        "-c",
        "import os,time; os.write(1,b'x'*65537); time.sleep(10)",
    ]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="output is suppressed"):
        MODULE._private_docker(command, stdout_limit=65536, timeout=2)
    assert time.monotonic() - started < 2
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert_capture_closed_and_reaped(bounded_children)


@pytest.mark.parametrize("stdin", [None, "x" * 65536])
def test_private_timeout_closes_and_reaps_even_when_child_never_reads_token(
    bounded_children, stdin, capsys
):
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="output is suppressed"):
        MODULE._private_docker(command, stdin=stdin, timeout=0.15)
    assert time.monotonic() - started < 2
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert_capture_closed_and_reaped(bounded_children)


@pytest.mark.parametrize("escaped", [False, True])
def test_capture_deadline_survives_parent_exit_and_descendant_held_pipes(
    tmp_path, bounded_children, monkeypatch, escaped
):
    descendant_pid = tmp_path / "descendant.pid"
    command = [
        sys.executable,
        "-c",
        "import os, pathlib, sys, time\n"
        "if os.fork() == 0:\n"
        " if sys.argv[2] == 'escaped': os.setsid()\n"
        " pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        " time.sleep(5)\n"
        "else: os._exit(0)\n",
        str(descendant_pid),
        "escaped" if escaped else "same-group",
    ]
    killed_groups = []
    original_killpg = MODULE.os.killpg

    def killpg(group, sig):
        killed_groups.append(group)
        return original_killpg(group, sig)

    monkeypatch.setattr(MODULE.os, "killpg", killpg)
    started = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            MODULE._bounded_capture(command, timeout=0.25)
        assert time.monotonic() - started < 2
        assert descendant_pid.is_file()
        assert bounded_children[0].returncode == 0
        assert bounded_children[0].pid in killed_groups
        assert_capture_closed_and_reaped(bounded_children)
    finally:
        if descendant_pid.exists():
            try:
                os.kill(int(descendant_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("stdin", ["x" * 65537, "\U0001f680" * 20000])
def test_oversized_credential_stdin_is_rejected_before_any_child(stdin, capsys):
    with patch.object(MODULE.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="output is suppressed"):
            MODULE._private_docker([sys.executable, "-c", "pass"], stdin=stdin)
    popen.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""

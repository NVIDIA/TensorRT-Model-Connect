# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate image-producer admission, credential lifetime and cleanup boundaries."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
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
    ]
    jobs = caller["jobs"]
    producer = jobs["produce-dependency-image"]
    assert producer["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.task == 'dependency-image' }}"
    )
    assert producer["uses"] == "./.github/workflows/community-dependency-image.yml"
    assert set(producer["secrets"]) == set(WORKFLOW["on"]["workflow_call"]["secrets"])
    assert set(producer["secrets"]) == {"BREV_API_KEY", "HF_TOKEN"}
    for name in ("snapshot", "authorize", "required"):
        assert "inputs.task != 'dependency-image'" in jobs[name]["if"]
    for name, job in jobs.items():
        if name not in {"produce-dependency-image", "withdraw-dependency-image"}:
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
    for task in ("test", "dependency-image", "dependency-image-audit", "dependency-image-withdraw"):
        assert workflow_condition(jobs["withdraw-dependency-image"]["if"], task=task) == (
            task == "dependency-image-withdraw"
        )
        assert workflow_condition(audit["if"], task=task) == (task == "dependency-image-audit")
        assert workflow_condition(jobs["produce-dependency-image"]["if"], task=task) == (
            task == "dependency-image"
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


def test_publication_download_uses_only_the_ssh_owned_auth_receipt():
    steps = WORKFLOW["jobs"]["produce"]["steps"]
    publish = next(step for step in steps if step.get("id") == "publish")
    assert "$INSTANCE_NAME:$REMOTE_AUTH/published-candidate.json" in publish["run"]
    assert "$INSTANCE_NAME:$REMOTE_PROOF/published-candidate.json" not in publish["run"]
    assert publish["timeout-minutes"] == "20"


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


def test_producer_requires_coverage_of_its_qualified_family(tmp_path: Path) -> None:
    candidate(tmp_path)
    with patch.object(MODULE.subprocess, "run") as run:
        MODULE.qualify(tmp_path, Path("/stage/python"), None, family="nemotron_h")
    assert "--require-family-coverage" in run.call_args.args[0]


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
    build = next(
        step
        for step in steps
        if step.get("name") == "Build full dependencies and validate native ABI"
    )
    qualify = next(
        step
        for step in steps
        if step.get("name") == "Qualify the unchanged Nemotron-H workloads on L4"
    )
    publish = next(step for step in steps if step.get("id") == "publish")
    release = next(step for step in steps if step.get("id") == "release")
    assert "REGISTRY_TOKEN" not in build.get("env", {})
    assert "HF_TOKEN" not in build.get("env", {})
    assert "REGISTRY_TOKEN" not in qualify.get("env", {})
    assert steps.index(build) < steps.index(qualify) < steps.index(publish) < steps.index(release)
    assert "always()" in release["if"] and "--until-deleted" in release["run"]
    backup = jobs["cleanup"]
    assert "always()" in backup["if"]
    assert "--until-deleted" in backup["steps"][-1]["run"]
    assert "GITHUB_RUN_ATTEMPT" not in backup["steps"][-1]["run"]
    assert WORKFLOW["concurrency"]["cancel-in-progress"] == "false"
    assert produce["timeout-minutes"] == "360" and backup["timeout-minutes"] == "360"
    assert produce["permissions"]["packages"] == "write"
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
    assert (tmp_path / "output").read_text() == "allowed=true\nmodel_sha=" + "d" * 40 + "\n"


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

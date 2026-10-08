# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate image-producer admission, credential lifetime and cleanup boundaries."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import urllib.response
from email.message import Message
from pathlib import Path
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
    assert inputs["task"]["options"] == ["test", "dependency-image"]
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
        if name != "produce-dependency-image":
            assert job.get("permissions", {}).get("packages") != "write"
    assert caller["concurrency"]["cancel-in-progress"] == "false"


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


@pytest.mark.parametrize(
    "repo,event,ref",
    [
        ("foreign/repo", "workflow_dispatch", "refs/heads/main"),
        ("NVIDIA/TensorRT-Model-Connect", "pull_request_target", "refs/heads/main"),
        ("NVIDIA/TensorRT-Model-Connect", "workflow_dispatch", "refs/pull/1/merge"),
    ],
)
def test_unauthorized_entry_fails_before_permission_or_vm_calls(
    tmp_path: Path, repo: str, event: str, ref: str
) -> None:
    program = WORKFLOW["jobs"]["authorize"]["steps"][0]["run"]
    program = program.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
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
                allow_missing=True,
            )
    assert len(requests) == 1 and requests[0].startswith("https://api.github.com/")
    assert "sensitive-token" not in str(error.value)


@pytest.mark.parametrize("value", [None, [], "private"])
def test_registry_api_non_object_json_fails_safely(value: object) -> None:
    response = io.BytesIO(json.dumps(value).encode())
    with patch.object(MODULE.urllib.request.OpenerDirector, "open", return_value=response):
        with pytest.raises(RuntimeError, match="visibility"):
            MODULE.require_private_package(
                "ghcr.io/nvidia/tensorrt-model-connect-community/nemotron_h",
                "sensitive-token",
                allow_missing=True,
            )


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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GitHub Actions CI wiring."""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from tools.ci.container import CiContainer
from tools.ci.environment import OPTIONAL_TUNING_ENVIRONMENT
from tools.ci.process import CiError
from tools.ci.quality import UnitTestRunner
from tools.ci.stage import ContainerStageRunner


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_fake_jq(fake_bin: Path) -> None:
    fake_jq = fake_bin / "jq"
    fake_jq.write_text(
        """#!/usr/bin/env python3
import json
import sys
from urllib.parse import quote

arguments = sys.argv[1:]
variables = {}
expression = ""
null_input = False
raw_output = False
exit_status = False
index = 0
while index < len(arguments):
    argument = arguments[index]
    if argument == "--arg":
        variables[arguments[index + 1]] = arguments[index + 2]
        index += 3
        continue
    if argument.startswith("-"):
        null_input = null_input or "n" in argument[1:]
        raw_output = raw_output or "r" in argument[1:]
        exit_status = exit_status or "e" in argument[1:]
    else:
        expression = argument
    index += 1

if null_input:
    if expression != "$value | @uri":
        raise SystemExit(f"unsupported null-input jq expression: {expression}")
    result = quote(variables["value"], safe="")
else:
    value = json.load(sys.stdin)
    if expression.startswith("[.[] | select(.body | contains("):
        marker = variables["marker"]
        result = next(
            (
                item["id"]
                for item in value
                if marker in str(item.get("body", ""))
            ),
            "",
        )
    else:
        optional_empty = expression.endswith(" // empty")
        path = expression.removesuffix(" // empty").removeprefix(".").split(".")
        result = value
        for part in path:
            if not isinstance(result, dict) or part not in result:
                result = None
                break
            result = result[part]
        if optional_empty and result is None:
            result = ""

if exit_status and result is None:
    raise SystemExit(1)
if raw_output:
    if isinstance(result, bool):
        print(str(result).lower())
    elif result is not None:
        print(result)
else:
    print(json.dumps(result))
""",
        encoding="utf-8",
    )
    fake_jq.chmod(0o755)


def _internal_ci_snapshot_script() -> str:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["authorize"]["steps"]
    return next(
        step["run"] for step in steps if step["name"] == "Capture the exact pull-request snapshot"
    )


def _run_internal_ci_snapshot(
    tmp_path: Path,
    *,
    event_head_sha: str,
    pr_head_sha: str,
    event_name: str = "pull_request_target",
    actor_role: str = "maintain",
    community_conclusion: str = "success",
    community_head_sha: str | None = None,
    community_merge_sha: str | None = None,
    system_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_jq(fake_bin)
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from urllib.parse import parse_qs, urlsplit

arguments = sys.argv[1:]
endpoint = next(
    (argument for argument in arguments if argument.startswith("/repos/")),
    "",
)
if "/collaborators/" in endpoint:
    print(os.environ["FAKE_ACTOR_ROLE"])
elif "/pulls/" in endpoint:
    print(os.environ["FAKE_PULL_JSON"])
elif "/actions/workflows/community-cpu.yml/runs" in endpoint:
    query = arguments[arguments.index("--jq") + 1]
    old_merge = os.environ["FAKE_COMMUNITY_MERGE_SHA"]
    requested_head = parse_qs(urlsplit(endpoint).query).get("head_sha", [""])[0]
    if (
        os.environ["FAKE_COMMUNITY_CONCLUSION"] == "success"
        and requested_head == os.environ["FAKE_COMMUNITY_HEAD_SHA"]
        and (not old_merge or "display_title" not in query or old_merge in query)
    ):
        print("12345")
    else:
        print("")
else:
    print(f"unexpected gh invocation: {arguments}", file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    output = tmp_path / "github-output"
    pull = {
        "state": "open",
        "base": {
            "ref": "main",
            "sha": "d" * 40,
            "repo": {"full_name": "NVIDIA/TensorRT-Model-Connect"},
        },
        "head": {"sha": pr_head_sha},
        "merge_commit_sha": "a" * 40,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "ACTOR": "trusted-maintainer",
            "EVENT_HEAD_SHA": event_head_sha,
            "EVENT_NAME": event_name,
            "FAKE_ACTOR_ROLE": actor_role,
            "FAKE_COMMUNITY_CONCLUSION": community_conclusion,
            "FAKE_COMMUNITY_HEAD_SHA": community_head_sha or pr_head_sha,
            "FAKE_COMMUNITY_MERGE_SHA": community_merge_sha or "",
            "FAKE_PULL_JSON": json.dumps(pull),
            "GH_TOKEN": "test-token",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
            "PATH": f"{fake_bin}:{system_path or environment['PATH']}",
            "POLICY_SHA": "e" * 40,
            "PR_NUMBER": "715",
        }
    )
    return subprocess.run(
        ["bash", "-c", _internal_ci_snapshot_script()],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _ci_source(*filenames: str) -> str:
    """Return the review surface for one or more class-based CI modules."""
    return "\n".join(
        (REPO_ROOT / "tools" / "ci" / filename).read_text(encoding="utf-8")
        for filename in filenames
    )


def _single_default_model_config(filename: str) -> tuple[Path, dict]:
    configs = sorted((REPO_ROOT / "tests" / "e2e" / "models").glob(f"*/{filename}"))
    defaults = []
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append((path, data))
    assert len(defaults) == 1
    return defaults[0]


def test_source_workflow_inventory_does_not_repeat_premerge_after_merge() -> None:
    workflows = REPO_ROOT / ".github" / "workflows"
    workflow_files = {
        *workflows.glob("*.yml"),
        *workflows.glob("*.yaml"),
    }
    assert sorted(path.name for path in workflow_files) == [
        "community-cpu.yml",
        "internal-ci-bridge.yml",
        "pages.yml",
        "pr-metadata.yml",
    ]

    bridge = (workflows / "internal-ci-bridge.yml").read_text(encoding="utf-8")
    pages = (workflows / "pages.yml").read_text(encoding="utf-8")

    for forbidden_trigger in (
        "push",
        "schedule",
        "workflow_run",
        "repository_dispatch",
    ):
        assert f"\n  {forbidden_trigger}:" not in bridge

    assert "\n  push:" in pages
    assert '      - "website/**"' in pages
    assert "python3 -m tools.ci" not in pages
    assert "actions/workflows/premerge.yml/dispatches" not in pages


def test_only_pages_workflow_creates_deployment_objects() -> None:
    deployments = []
    workflows = REPO_ROOT / ".github" / "workflows"
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            environment = job.get("environment")
            if environment is None:
                continue
            if isinstance(environment, str):
                deployments.append((path.name, job_name, environment))
            elif environment.get("deployment", True):
                deployments.append((path.name, job_name, environment["name"]))

    assert deployments == [("pages.yml", "deploy", "github-pages")]


def test_community_docs_gate_matches_pages_predeploy_contract() -> None:
    workflows = REPO_ROOT / ".github" / "workflows"
    pages = yaml.safe_load((workflows / "pages.yml").read_text(encoding="utf-8"))
    community = yaml.safe_load((workflows / "community-cpu.yml").read_text(encoding="utf-8"))

    pages_build = pages["jobs"]["build"]
    pages_steps = {step["name"]: step for step in pages_build["steps"]}
    docs = community["jobs"]["docs"]
    docs_steps = {step["name"]: step for step in docs["steps"]}

    assert str(pages_steps["Set up Node"]["with"]["node-version"]) == "20"
    assert docs_steps["Set up Node"] == {
        "name": "Set up Node",
        "uses": "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
        "with": {"node-version": "20"},
    }
    assert pages_build["defaults"]["run"]["working-directory"] == "website"
    assert docs_steps["Test generated model support inventory"] == {
        "name": "Test generated model support inventory",
        "working-directory": "website",
        "run": pages_steps["Test generated model support inventory"]["run"],
    }
    assert pages_steps["Test generated model support inventory"]["run"] == (
        "npm run test:model-support"
    )
    assert docs_steps["Install website dependencies"]["run"] == "npm ci"
    assert docs_steps["Build production documentation"]["run"] == "npm run build"
    assert pages_steps["Build site"]["run"].strip().endswith("npm run build")


def test_internal_ci_bridge_only_dispatches_an_exact_trusted_head() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml").read_text(
        encoding="utf-8"
    )
    workflow_config = yaml.safe_load(workflow)
    authorize = workflow.split("\n  authorize:", maxsplit=1)[1].split("\n  announce:", maxsplit=1)[0]
    announce = workflow.split("\n  announce:", maxsplit=1)[1].split("\n  dispatch:", maxsplit=1)[0]
    dispatch = workflow.split("\n  dispatch:", maxsplit=1)[1].split("\n  publish:", maxsplit=1)[0]
    publish = workflow.split("\n  publish:", maxsplit=1)[1]

    assert "pull_request_target:" in workflow
    assert "name: TensorRT-Model-Connect Internal CI Bridge" in workflow
    assert "types: [labeled]" in workflow
    assert "github.event.label.name == 'run-internal-ci'" in workflow
    assert "/labels/run-ci" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "github.repository == 'NVIDIA/TensorRT-Model-Connect'" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.event_name == 'pull_request_target'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "permissions: {}" in workflow
    assert workflow_config["jobs"]["authorize"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }
    assert workflow_config["jobs"]["announce"]["permissions"] == {
        "pull-requests": "write",
        "statuses": "write",
    }
    assert workflow_config["jobs"]["dispatch"]["permissions"] == {"contents": "read"}
    assert workflow_config["jobs"]["dispatch"]["timeout-minutes"] == 360
    assert workflow_config["jobs"]["publish"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
        "statuses": "write",
    }

    assert "collaborators/$ACTOR/permission" in authorize
    assert "--jq '.role_name'" in authorize
    assert "--jq '.permission'" not in authorize
    assert "maintain|admin)" in authorize
    assert "write|maintain|admin)" not in authorize
    assert "Only actors with maintain or admin access" in authorize
    assert "environment:" not in authorize
    assert "secrets." not in authorize
    assert "statuses: write" not in authorize
    assert 'pull="$(gh api --method GET' in authorize
    assert 'test "$state" = "open"' in authorize
    assert 'test "$base_repo" = "$GITHUB_REPOSITORY"' in authorize
    assert 'test "$base_ref" = "main"' in authorize
    assert 'if [ "$EVENT_NAME" = "pull_request_target" ]; then' in authorize
    assert '"/repos/$head_repo/branches/$head_ref_uri"' not in authorize
    assert "head_repo" not in authorize
    assert "head_ref" not in authorize
    assert '[[ "$head_sha" =~ ^[0-9a-f]{40}$ ]]' in authorize
    assert "Community CPU / Required" in authorize
    assert (
        "/actions/workflows/community-cpu.yml/runs?event=pull_request&head_sha=$head_sha"
        in authorize
    )
    assert '.head_sha == \\"$head_sha\\"' not in authorize
    assert '.conclusion == "success"' in authorize
    assert "display_title" not in authorize
    assert 'if ! [[ "$community_cpu_run" =~ ^[1-9][0-9]*$ ]]; then' in authorize
    assert 'echo "head_sha=$head_sha"' in authorize
    assert "pr_number=$PR_NUMBER" in authorize
    assert 'echo "base_sha=$base_sha"' in authorize
    assert 'echo "policy_sha=$POLICY_SHA"' in authorize
    assert ("/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/labels/run-internal-ci") in authorize
    assert "gh api --silent --method DELETE" in authorize
    assert "success() && github.event_name == 'pull_request_target'" in authorize

    assert "needs:" in dispatch
    assert workflow_config["jobs"]["dispatch"]["environment"] == {
        "name": "ci-dispatch",
        "deployment": False,
    }
    secret_references = set(re.findall(r"secrets\.([A-Z][A-Z0-9_]*)", workflow))
    assert secret_references == {
        "TRTMC_CI_DISPATCH_TOKEN",
        "TRTMC_PRIVATE_CI_OWNER",
        "TRTMC_PRIVATE_CI_REPOSITORY",
    }
    for secret in secret_references:
        assert secret not in authorize
        assert secret in dispatch
        assert secret not in announce
        assert secret not in publish
    assert workflow.count("${{ secrets.TRTMC_CI_DISPATCH_TOKEN }}") == 3

    assert "actions/create-github-app-token@" not in workflow
    assert "permission-checks:" not in workflow
    assert "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803" in workflow
    assert "persist-credentials: false" in dispatch
    assert "private_ci_bridge.py" not in workflow
    assert "self-hosted" not in workflow
    assert "secrets: inherit" not in workflow
    assert "report-guard-failure" not in workflow
    assert workflow.count("/statuses/") == 2
    assert "/comments" in workflow
    assert (
        "/repos/$PRIVATE_CI_OWNER/$PRIVATE_CI_REPOSITORY/actions/workflows/premerge.yml/dispatches"
    ) in workflow
    assert re.search(r'"/repos/[A-Za-z0-9]', workflow) is None
    assert '[[ "$PRIVATE_CI_OWNER" =~ ^[A-Za-z0-9]' in workflow
    assert '[[ "$PRIVATE_CI_REPOSITORY" =~ ^[A-Za-z0-9]' in workflow
    assert 'echo "$PRIVATE_CI_' not in dispatch
    assert "GITHUB_STEP_SUMMARY" not in dispatch

    assert dispatch.count("HEAD_SHA: ${{ needs.authorize.outputs.head_sha }}") >= 1

    assert 'ref: "main"' in dispatch
    for name in ("pr_number", "head_sha", "dispatch_nonce"):
        assert f"{name}: ${name}" in dispatch
    assert "umask 077" in dispatch
    assert "openssl rand -hex 16" in dispatch
    assert '[[ "$dispatch_nonce" =~ ^[0-9a-f]{32}$ ]]' in dispatch
    assert (
        'expected_title="Source PR #$PR_NUMBER · $HEAD_SHA · dispatch $dispatch_nonce"' in dispatch
    )
    assert ".display_title == $title" in dispatch
    assert ".created_at >= $dispatched_at" not in dispatch
    assert "actions/workflows/premerge.yml/runs?event=workflow_dispatch" in dispatch
    assert "actions/runs/$RUN_ID" in dispatch
    assert "while true; do" in dispatch
    assert "seq 1 220" not in dispatch
    assert "--name public-failure-payload" in dispatch
    assert "--log" not in dispatch
    assert "validate_public_failure(report)" in dispatch
    assert "EXPECTED_DISPATCH_NONCE" in dispatch
    assert '"dispatch_nonce": os.environ["EXPECTED_DISPATCH_NONCE"]' in dispatch
    assert "assert_public_payload_safe(report, document)" in dispatch
    assert "name: public-failure-log" in dispatch
    assert "Automated internal CI failed; open the public failure log" in publish
    assert workflow.count("TRTMC Internal CI / Automated premerge gate") == 2
    assert "always() && needs.authorize.result == 'success'" in workflow
    assert "cancelled|timed_out|skipped|neutral|action_required" in workflow
    assert 'payload_size" -gt 65536' in dispatch
    assert '"base_sha": os.environ["EXPECTED_BASE_SHA"]' in dispatch
    assert publish.index("- name: Publish the terminal automated status") < publish.index(
        "- name: Print public-failure.log"
    )
    assert "github-actions[bot]" in publish
    assert "<!-- trtmc-internal-ci-result -->" in publish
    assert "trap 'rm -f \"$payload\"' EXIT" in dispatch
    assert "if: ${{ failure() }}" not in workflow


def test_internal_ci_bridge_rejects_a_new_push_after_label(
    tmp_path: Path,
) -> None:
    event_head = "f7b48712c82318ded4e41c0dd7003379e1790198"
    current_head = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=event_head,
        pr_head_sha=current_head,
    )

    assert result.returncode != 0
    assert "superseded by a newer PR head" in result.stdout + result.stderr


def test_internal_ci_bridge_uses_upstream_pr_metadata_without_fork_access(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_ci_bridge_accepts_a_green_head_after_the_base_moves(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
        community_merge_sha="b" * 40,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_ci_bridge_requires_current_community_cpu_success(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
        community_conclusion="failure",
    )

    assert result.returncode != 0
    assert "Community CPU / Required must pass" in result.stdout + result.stderr


def test_internal_ci_bridge_rejects_community_cpu_from_another_head(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
        community_head_sha="f7b48712c82318ded4e41c0dd7003379e1790198",
    )

    assert result.returncode != 0
    assert "Community CPU / Required must pass" in result.stdout + result.stderr


def test_internal_ci_bridge_only_allows_maintainers_and_admins(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    for permission in ("maintain", "admin"):
        run_dir = tmp_path / permission
        run_dir.mkdir()
        result = _run_internal_ci_snapshot(
            run_dir,
            event_head_sha=head_sha,
            pr_head_sha=head_sha,
            actor_role=permission,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    write_dir = tmp_path / "write"
    write_dir.mkdir()
    result = _run_internal_ci_snapshot(
        write_dir,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
        actor_role="write",
    )
    assert result.returncode != 0
    assert "Only actors with maintain or admin access" in result.stdout + result.stderr


def test_internal_ci_bridge_protects_manual_dispatch_without_event_head(
    tmp_path: Path,
) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha="",
        pr_head_sha=head_sha,
        event_name="workflow_dispatch",
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_ci_bridge_tests_do_not_depend_on_host_jq(tmp_path: Path) -> None:
    head_sha = "c8844445a1c630aef586b45daf7dfb31d4168c5a"
    tool_bin = tmp_path / "system-bin"
    tool_bin.mkdir()
    (tool_bin / "bash").symlink_to("/bin/bash")
    (tool_bin / "python3").symlink_to(sys.executable)
    (tool_bin / "sleep").symlink_to("/bin/sleep")

    result = _run_internal_ci_snapshot(
        tmp_path,
        event_head_sha=head_sha,
        pr_head_sha=head_sha,
        system_path=str(tool_bin),
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_ci_trigger_label_is_consumed_only_after_authorization() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml").read_text(encoding="utf-8")
    )
    step = next(
        item
        for item in workflow["jobs"]["authorize"]["steps"]
        if item["name"] == "Consume the trusted trigger label"
    )

    assert step["if"] == ("${{ success() && github.event_name == 'pull_request_target' }}")
    assert step["env"]["PR_NUMBER"] == "${{ github.event.pull_request.number }}"


def test_ci_orchestration_uses_the_class_based_python_entrypoint() -> None:
    legacy_scripts = (
        ".github/scripts/ensure-ci-docker-image.sh",
        ".github/scripts/start-gha-container.sh",
        ".github/scripts/run-gha-stage.sh",
        ".github/scripts/run-trtmc-ci.sh",
        ".github/scripts/run-model-proof.sh",
        "scripts/run_e2e_parallel.sh",
        "tools/coverage_ci/run_cpp_coverage.sh",
        "tools/coverage_ci/run_python_coverage.sh",
        "tools/coverage/cpp_coverage.sh",
        "tools/coverage/python_coverage.sh",
        "tools/coverage/run_coverage_all.sh",
    )
    assert not [path for path in legacy_scripts if (REPO_ROOT / path).exists()]

    source = _ci_source(
        "container.py",
        "coverage.py",
        "docker_image.py",
        "e2e_scheduler.py",
        "model_proof.py",
        "pipeline.py",
        "stage.py",
    )
    for class_name in (
        "CiContainer",
        "CoverageRunner",
        "DockerImageManager",
        "E2EParallelRunner",
        "ModelProofRunner",
        "CiPipeline",
        "ContainerStageRunner",
    ):
        assert f"class {class_name}" in source


def test_ci_modules_have_minimal_role_comments_and_a_complete_tutorial() -> None:
    ci_directory = REPO_ROOT / "tools" / "ci"
    modules = sorted(ci_directory.glob("*.py"))
    for module in modules:
        docstring = ast.get_docstring(ast.parse(module.read_text(encoding="utf-8")))
        assert docstring, f"{module.name} has no module documentation"
        assert "Boundary:" in docstring, f"{module.name} does not state its responsibility boundary"

    readme = (ci_directory / "README.md").read_text(encoding="utf-8")
    missing_modules = [module.name for module in modules if f"`{module.name}`" not in readme]
    assert missing_modules == []
    contracts = readme.split("## Component contracts", 1)[1].split(
        "## Data passed between stages", 1
    )[0]
    for module in modules:
        match = re.search(
            rf"^### `{re.escape(module.name)}`$(.*?)(?=^### `|^## |\Z)",
            contracts,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match, f"{module.name} has no component contract"
        for field in ("Functionality / units", "Inputs", "Outputs", "Boundary"):
            assert f"**{field}:**" in match.group(1), f"{module.name} has no {field} contract"
    for section in (
        "## The system at a glance",
        "## Pre-merge, step by step",
        "## What nightly adds",
        "## Module map",
        "## Component contracts",
        "## Making a CI change",
        "## Reading a failure",
    ):
        assert section in readme


def test_source_quality_pipeline_keeps_the_full_static_gate() -> None:
    source = _ci_source("pipeline.py", "quality.py")
    source_quality = source.split('"source-quality":', maxsplit=1)[1].split(
        '"cpp-unit":', maxsplit=1
    )[0]

    assert '"Check cyclomatic complexity"' in source_quality
    assert "self.quality.complexity" in source_quality
    assert '"Lint changed files"' in source_quality
    assert "self.quality.lint_changed_files" in source_quality
    assert '"Check model architecture contracts"' in source_quality
    assert "self.quality.architecture_contracts" in source_quality

    architecture_contract = source.split("def architecture_contracts", maxsplit=1)[1].split(
        "def _changed_files", maxsplit=1
    )[0]
    assert '"pytest"' in architecture_contract
    assert "tests/tools/test_model_plugin_encapsulation_static.py" in architecture_contract
    assert '"-q"' in architecture_contract
    assert '"no:cacheprovider"' in architecture_contract


def test_source_ci_image_uses_common_and_parameterized_tensorrt_overlay() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "ARG CUDA_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu24.04"
        "@sha256:ef2203909e80b8b976cfc672f7e2ae2b00bc0e25c404ee86d89e10a3802f1c52" in dockerfile
    )
    assert dockerfile.count("ARG TENSORRT_VERSION=11.1.0.106") == 1
    assert dockerfile.count("ARG TENSORRT_APT_VERSION=11.1.0.106-1+cuda13.3") == 1
    assert "ARG TENSORRT_VERSION\n" in dockerfile
    assert "ARG TENSORRT_APT_VERSION\n" in dockerfile
    assert "RUN python3.12 -m venv $VIRTUAL_ENV" in dockerfile
    assert "--system-site-packages" not in dockerfile
    assert "ENV TRT_ROOT=" not in dockerfile
    assert "ENV PIP_FIND_LINKS=" not in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs" in dockerfile
    assert "ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu" in dockerfile
    assert "ghcr.io" not in dockerfile
    assert "TENSORRT_SDK_IMAGE" not in dockerfile
    assert "/opt/tensorrt/python" not in dockerfile

    from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert from_lines == [
        "FROM ${CUDA_IMAGE} AS ci-common-base",
        "FROM ci-common-base AS python-profile-builder",
        "FROM ci-common-base AS ci-common",
        "FROM ci-common AS ci-runtime",
    ]

    common = dockerfile.split("FROM ci-common-base AS ci-common", maxsplit=1)[1].split(
        "FROM ci-common AS ci-runtime", maxsplit=1
    )[0]
    assert "COPY --from=python-profile-builder" in common
    assert "TRTMC_PYTHON_PROFILE_PREBUILT_ONLY=1" in common
    assert 'find_spec("tensorrt") is None' in common
    assert "NvInferVersion.h" in common
    assert "NvOnnxParser.h" in common
    assert "ENV PYTHONPATH=/opt/trtmc-profile-source" not in dockerfile

    overlay = dockerfile.split("FROM ci-common AS ci-runtime", maxsplit=1)[1]
    for package in (
        "libnvinfer-dev",
        "libnvinfer-headers-dev",
        "libnvinfer-headers-plugin-dev",
        "libnvinfer-safe-headers-dev",
        "libnvinfer11",
        "libnvonnxparsers-dev",
        "libnvonnxparsers11",
    ):
        assert f'"{package}=${{TENSORRT_APT_VERSION}}"' in overlay
    for distribution in (
        "tensorrt",
        "tensorrt_cu13",
        "tensorrt_cu13_bindings",
        "tensorrt_cu13_libs",
    ):
        assert f'"{distribution}"' in overlay
    assert 'pip install --no-cache-dir "tensorrt==${TENSORRT_VERSION}"' in overlay
    assert 'ln -s "libnvinfer.so.$TENSORRT_MAJOR"' in overlay
    assert 'ln -s "libnvonnxparser.so.$TENSORRT_MAJOR"' in overlay
    assert "NvInferVersion.h" in overlay
    assert "NvOnnxParser.h" in overlay
    assert "libnvonnxparser.so" in overlay
    assert "libnvinfer_builder_resource_sm110.so" in overlay
    assert "getInferLibVersion" in overlay
    assert "-x none" in overlay
    assert '"$TRT_LIB_DIR/libnvinfer.so" "$TRT_LIB_DIR/libnvonnxparser.so"' in overlay
    assert "-lnvinfer -lnvonnxparser" not in overlay
    assert "#include <NvInferRuntime.h>" in overlay
    assert "-I/usr/local/cuda/include" in overlay
    assert "c++ -x c++" in overlay

    source_dockerfiles = {
        "aarch64": (REPO_ROOT / "Dockerfile.dev.aarch64").read_text(encoding="utf-8"),
        "x86_64": (REPO_ROOT / "Dockerfile.dev.x86").read_text(encoding="utf-8"),
    }
    assert (
        "@sha256:f794a79e8b996d16dbc2e5884e19d8e2269a51c960106c9b49b0061a6926c541"
        in source_dockerfiles["aarch64"]
    )
    assert (
        "@sha256:b82db1abc23750ab0069abc99bbe4ea29138dbdc23ea39861199e2346638b48a"
        in source_dockerfiles["x86_64"]
    )
    for architecture, source_dockerfile in source_dockerfiles.items():
        assert "FROM ${TENSORRT_IMAGE}" in source_dockerfile
        assert "COPY community-ci.txt /tmp/trtmc-community-ci.txt" in source_dockerfile
        assert "pip install --requirement /tmp/trtmc-community-ci.txt" in source_dockerfile
        assert "pre-commit>=" not in source_dockerfile
        assert "https://download.pytorch.org/whl/cpu" in source_dockerfile
        assert "torch.version.cuda is None" in source_dockerfile
        assert "TRTMC_TORCH_CUDA_ARCH_LIST" not in source_dockerfile
        assert "python-profile-builder" not in source_dockerfile
        assert "nemo_toolkit" not in source_dockerfile
        assert "ln -s /usr/bin/cmake ${VIRTUAL_ENV}/bin/cmake" in source_dockerfile
        assert "RUN cmake --version" in source_dockerfile
        assert f"ENV TRT_LIB_DIR=/usr/lib/{architecture}-linux-gnu" in source_dockerfile
        assert f"ENV TRT_INC_DIR=/usr/include/{architecture}-linux-gnu" in source_dockerfile
    assert "x86_64-linux-gnu" not in dockerfile

    source_build = (REPO_ROOT / "website/docs/getting-started/source-build.md").read_text(
        encoding="utf-8"
    )
    assert "Dockerfile.dev.aarch64" in source_build
    assert "Dockerfile.dev.x86" in source_build
    assert "trtmc_model_qwen" in source_build
    assert "trtmc_model_plugins" not in source_build
    assert "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL=OFF" in source_build
    assert "TRTMC_TORCH_CUDA_ARCH_LIST" not in source_build

    ci_docker_build = (REPO_ROOT / "scripts/docker_build_gb300.sh").read_text(encoding="utf-8")
    assert '"$REPO_ROOT/Dockerfile"' in ci_docker_build
    assert "Dockerfile.dev.aarch64" not in ci_docker_build
    assert "Dockerfile.dev.x86" not in ci_docker_build
    assert "--target" not in ci_docker_build

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version", "dependencies"]' in pyproject
    assert 'base-version = "0.1.0"' in pyproject
    assert 'default-tensorrt-version = "11.1.0.106"' in pyproject

    package = _ci_source("package.py")
    assert "_install_tensorrt_sdk" not in package
    assert 'PACKAGE_TENSORRT_VERSION_ENV = "TRTMC_PACKAGE_TENSORRT_VERSION"' in package
    assert "_package_variant_version" in package
    assert "_validate_package_variant" in package
    assert "_validate_backend_files" in package
    assert "_validate_backend_identity" in package


def test_hardened_unit_container_is_unprivileged_offline_and_cpu_only() -> None:
    source = _ci_source("container.py", "environment.py")
    for option in (
        '"--network"',
        '"none"',
        "--read-only",
        "/tmp:rw,exec,nosuid,nodev,size=16g",
        "--cap-drop",
        "--security-opt",
        "no-new-privileges",
        'f"{os.getuid()}:{os.getgid()}"',
        "--ipc",
        "HOME=/tmp",
        "USER=trtmc-ci",
        "LOGNAME=trtmc-ci",
        "TMPDIR=/work/tmp",
        "TEMP=/work/tmp",
        "TMP=/work/tmp",
        "XDG_CACHE_HOME=/work/cache",
        "TORCHINDUCTOR_CACHE_DIR=/work/torch-cache",
        "PIP_NO_INDEX=1",
        "TRTMC_CI_SCRATCH_DIR=/work",
        "NVIDIA_VISIBLE_DEVICES=void",
        "CUDA_VISIBLE_DEVICES=",
        "--runtime",
        'f"{mount}:ro"',
        'f"{scratch}:/work"',
    ):
        assert option in source
    assert "if self.config.hardened" in source
    assert "if not self.config.hardened" in source
    assert "Path('/dev').glob('nvidia*')" in source
    assert "Hardened unit scratch must be inside RUNNER_TEMP" in source
    assert "Hardened unit scratch must not be a symlink" in source
    assert 'env.get("TRTMC_CI_WORKSPACE")' in source
    assert 'env.get("GITHUB_WORKSPACE", "")' in source
    assert 'mount = f"{self.config.workspace}:{self.config.workspace}"' in source

    common = source.split("COMMON_ENVIRONMENT =", maxsplit=1)[1].split(
        "TRUSTED_ENVIRONMENT =", maxsplit=1
    )[0]
    assert "TRTMC_PREMERGE_UNIT_SCOPE" in common
    assert "TRTMC_PREMERGE_PYTHON_TEST_TARGETS" in common
    for name in (
        "TRTMC_PACKAGE_PYTHON_TAGS",
        "TRTMC_PACKAGE_TENSORRT_VERSION",
        "TRTMC_PACKAGE_WHEEL_ARCH",
    ):
        assert name in common
    assert "TRTMC_PACKAGE_BUILD_ROOT" not in common
    assert "HF_TOKEN" not in common
    assert "HUGGING_FACE_HUB_TOKEN" not in common

    stage = _ci_source("stage.py")
    assert "COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT" in stage


def test_github_stage_wrapper_mounts_and_exports_hf_cache_env() -> None:
    stage_text = _ci_source("stage.py", "environment.py")
    start_text = _ci_source("container.py", "environment.py")
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert name in start_text
        assert name in stage_text
    assert 'self.env.get("TRTMC_CI_HOST_MOUNTS", "")' in start_text
    assert 'f"{host_path}:{host_path}"' in start_text
    assert '"docker"' in stage_text
    assert '"exec"' in stage_text


def test_trusted_container_mounts_only_explicit_cache_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "storage"
    engine_dir = storage / "engines"
    hf_home = storage / "hf"
    extra = tmp_path / "extra"
    for path in (workspace, engine_dir, hf_home, extra):
        path.mkdir(parents=True)

    container = CiContainer(
        {
            "TRTMC_CI_WORKSPACE": str(workspace),
            "TRTMC_CI_IMAGE": "example.invalid/trtmc:test",
            "TRTMC_STORAGE_ROOT": str(storage),
            "ENGINE_DIR": str(engine_dir),
            "HF_HOME": str(hf_home),
            "TRTMC_CI_HOST_MOUNTS": os.pathsep.join(
                (
                    str(extra),
                    str(storage),
                    str(tmp_path),
                    "/",
                    "relative-path-is-ignored",
                )
            ),
        }
    )

    options, mounts = container._runtime_boundary()

    assert options == []
    assert mounts == [
        "-v",
        f"{extra}:{extra}",
        "-v",
        f"{storage}:{storage}",
    ]


def test_github_stage_wrapper_removes_exact_container_on_cancellation(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    exec_started = tmp_path / "exec-started"
    container_removed = tmp_path / "container-removed"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'case "${1:-}" in\n'
        "  inspect) printf 'true\\n' ;;\n"
        "  exec) touch \"$DOCKER_EXEC_STARTED\"; trap '' INT TERM; "
        'while [ ! -f "$DOCKER_REMOVED" ]; do sleep 0.1; done ;;\n'
        '  rm) touch "$DOCKER_REMOVED"; exit 0 ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    container_name = "trtmc-nightly-package-4242-1"
    env = os.environ.copy()
    env.update(
        {
            "BASH_ENV": "",
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "DOCKER_EXEC_STARTED": str(exec_started),
            "DOCKER_REMOVED": str(container_removed),
            "TRTMC_CI_CONTAINER_NAME": container_name,
            "TRTMC_CI_WORKSPACE": str(workspace),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tools.ci", "stage", "python-builder"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not exec_started.is_file():
            assert process.poll() is None
            time.sleep(0.05)
        assert exec_started.is_file()
        started = time.monotonic()
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)

    assert process.returncode == 143, stdout + stderr
    assert elapsed < 2
    assert container_removed.is_file()
    assert f"rm -f {container_name}" in docker_log.read_text(encoding="utf-8")


def test_github_container_only_exports_nonempty_hf_transport_controls() -> None:
    stage_text = _ci_source("stage.py")
    start_text = _ci_source("container.py", "environment.py")

    assert "OPTIONAL_HUGGING_FACE_ENVIRONMENT" in start_text
    assert 'if self.env.get(name, "")' in start_text
    for name in ("HF_HUB_DISABLE_XET", "HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT"):
        assert name not in stage_text


def test_github_container_only_exports_nonempty_tuning_controls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base_env = {
        "TRTMC_CI_WORKSPACE": str(workspace),
        "TRTMC_CI_IMAGE": "example.invalid/trtmc:test",
    }

    for name in OPTIONAL_TUNING_ENVIRONMENT:
        for value in (None, "", "   ", "3"):
            env = dict(base_env)
            if value is not None:
                env[name] = value
            arguments = CiContainer(env)._environment_arguments()
            forwarded = [
                arguments[index + 1] for index, item in enumerate(arguments[:-1]) if item == "-e"
            ]
            stage_command = ContainerStageRunner("package", env)._docker_command()
            if value == "3":
                assert f"{name}=3" in forwarded
                assert name in stage_command
            else:
                assert not any(item.startswith(f"{name}=") for item in forwarded)
                assert name not in stage_command


def test_github_stage_wrapper_exports_e2e_gpu_controls() -> None:
    text = _ci_source("environment.py")
    assert "TRTMC_E2E_EXCLUDE_GPU0" in text
    assert "TRTMC_E2E_DEPRIORITIZE_GPU0" in text


def test_github_stage_wrapper_exports_premerge_unit_parallelism() -> None:
    stage = _ci_source("stage.py", "environment.py")
    start = _ci_source("container.py", "environment.py")
    for name in (
        "TRTMC_UNIT_BUILD_JOBS",
        "TRTMC_UNIT_TEST_JOBS",
        "TRTMC_PREMERGE_UNIT_SCOPE",
    ):
        assert name in stage
        assert name in start


def test_github_stage_wrapper_exports_cpp_coverage_scope() -> None:
    stage = _ci_source("stage.py", "environment.py")
    start = _ci_source("container.py", "environment.py")
    assert "CPP_COVERAGE_SCOPE" in stage
    assert "CPP_COVERAGE_SCOPE" in start


def test_github_stage_wrapper_exports_diffusion_vlm_config() -> None:
    text = _ci_source("stage.py", "environment.py")
    start_text = _ci_source("container.py", "environment.py")
    assert "DIFFUSION_VLM_CONFIG" in text
    assert "DIFFUSION_VLM_CONFIG" in start_text


def test_github_stage_wrapper_exports_package_smoke_controls() -> None:
    text = _ci_source("environment.py")
    for name in (
        "TRTMC_PACKAGE_PYTHON_TAGS",
        "TRTMC_PACKAGE_WHEEL_ARCH",
        "TRTMC_PACKAGE_BUILD_ROOT",
        "TRTMC_PACKAGE_TENSORRT_VERSION",
        "TRTMC_WHEEL_SMOKE_CONFIG",
        "TRTMC_WHEEL_SMOKE_MODEL_ID",
        "TRTMC_WHEEL_SMOKE_MAX_CACHE",
        "TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS",
        "TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL",
        "TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT",
        "TRTMC_WHEEL_SMOKE_RUN_TIMEOUT",
    ):
        assert name in text
    assert "TRTMC_PACKAGE_VERSION" not in text


def test_github_stage_wrapper_does_not_export_diffusion_vlm_waives_file() -> None:
    text = _ci_source("stage.py", "environment.py")
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text


def test_diffusion_vlm_gate_failures_are_not_waived() -> None:
    text = _ci_source("e2e.py")
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text
    assert "--waives" not in text


def test_diffusion_vlm_pair_count_uses_helper() -> None:
    text = _ci_source("e2e.py")
    vlm_block = text.split("def diffusion_vlm_assessment", maxsplit=1)[1].split(
        "def _prepare_plugins", maxsplit=1
    )[0]
    assert "tools/count_diffusion_frame_pairs.py" in vlm_block
    assert '"--config"' in vlm_block
    assert "config_path" in vlm_block


def test_diffusion_vlm_assessment_default_is_model_owned() -> None:
    path, data = _single_default_model_config("diffusion_vlm_assessment.json")
    assert path.parent.parent == REPO_ROOT / "tests" / "e2e" / "models"
    for key in ("model_id", "max_side", "max_new_tokens", "timeout"):
        assert data.get(key)


def test_diffusion_vlm_shared_ci_has_no_model_owned_default() -> None:
    shared_paths = (
        REPO_ROOT / "tools" / "ci" / "e2e.py",
        REPO_ROOT / "tools" / "evaluate_diffusion_vlm_similarity.py",
    )
    _, data = _single_default_model_config("diffusion_vlm_assessment.json")
    forbidden = (str(data["model_id"]),)
    violations = [
        (path, needle)
        for path in shared_paths
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not violations


def test_full_python_builder_preserves_parallel_and_allocator_coverage() -> None:
    text = _ci_source("coverage.py")
    builder = text.split("def python_builder_tests", maxsplit=1)[1].split("def cpp", maxsplit=1)[0]
    builder_conftest = (REPO_ROOT / "tests" / "builder" / "conftest.py").read_text(encoding="utf-8")

    assert 'glob("test_*.py")' in builder
    assert "--ignore=tests/builder/test_cli.py" not in builder
    assert '"-n", "auto"' in builder
    assert '"not model_proof_allocator and not gpu and not trt"' in builder
    assert '"tests/tools/test_model_proof_runner.py"' in builder
    assert '"model_proof_allocator"' in builder
    assert '"--cov-append"' in builder
    assert '"TRTMC_TEST_INSTALLED_WHEEL": "1"' in builder
    assert "source_pkgs =" in text
    assert "tensorrt_model_connect" in text
    assert 'os.environ.get("TRTMC_TEST_INSTALLED_WHEEL") == "1"' in builder_conftest
    assert "imported tensorrt_model_connect" in builder_conftest
    assert builder.index('"-n", "auto"') < builder.index('"tests/tools/test_model_proof_runner.py"')


def test_selective_python_always_runs_static_ci_smoke_tests() -> None:
    text = _ci_source("coverage.py")
    for test_path in (
        "tests/tools/test_github_actions_ci.py",
        "tests/tools/test_model_plugin_encapsulation_static.py",
        "tests/tools/test_schedule_e2e.py",
        "tests/tools/test_test_impact.py",
    ):
        assert test_path in text


def test_python_package_coverage_gate_excludes_family_owned_modules() -> None:
    text = _ci_source("coverage.py")
    assert "_write_python_config" in text
    assert "*/tensorrt_model_connect/families/*" in text
    assert 'self.directory / "python-package-gate.coveragerc"' in text
    assert 'f"--cov-config={config}"' in text
    assert "PYTHON_COVERAGE_MIN_LINE" in text
    assert "PYTHON_COVERAGE_MIN_BRANCH" in text


def test_full_e2e_collection_uses_model_e2e_files_with_visible_errors() -> None:
    text = _ci_source("e2e_scheduler.py")
    full_mode = text.split("def _collect_tests", maxsplit=1)[1].split(
        "def _model_name", maxsplit=1
    )[0]
    assert 'glob("*/test_*_e2e.py")' in full_mode
    assert '"--co"' in full_mode
    assert '"-q"' in full_mode
    assert '"test_model_e2e[" in line' in full_mode


def test_qwen_flashinfer_scripts_skip_pytest_collection() -> None:
    for relpath in (
        "tests/e2e/models/qwen/test_flashinfer_plugin.py",
        "tests/e2e/models/qwen/test_flashinfer_trt_attention.py",
        "tests/e2e/models/qwen/test_qwen3_flashinfer.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert 'if __name__ != "__main__":' in text
        assert "pytest.skip(" in text
        assert "allow_module_level=True" in text
        assert text.index("pytest.skip(") < text.index("import tvm_ffi")


def test_source_quality_lint_uses_resolved_ci_base_ref() -> None:
    text = _ci_source("quality.py")
    assert "f\"origin/{self.context.env.get('GITHUB_REF_NAME', 'main')}\"" in text


def test_premerge_unit_stage_runs_all_cpu_tests_without_native_wheel() -> None:
    script = _ci_source("quality.py")
    stage = script.split("def premerge", maxsplit=1)[1].split("def _premerge_scope", maxsplit=1)[0]
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    qwen_manifest = (REPO_ROOT / "src" / "runtime" / "models" / "qwen" / "MODEL.toml").read_text()

    assert "pip install" not in stage
    assert "source / 'python'" in stage
    assert '"TRTMC_CI_SCRATCH_DIR", "/tmp"' in stage
    assert "TRTMC_PREMERGE_UNIT_BUILD_DIR" in stage
    assert '"not gpu and not trt and not e2e and not model_proof_allocator"' in stage
    assert '["tests/builder/", "tests/tools/", "tests/e2e_harness/"]' in script
    assert "tests/e2e/models" in script
    assert "python/tensorrt_model_connect/families" in script
    assert "tests/test_e2e_selection.py" in script
    assert "tests/e2e/test_diffusion_image_parity_inputs.py" in script
    assert "tests/e2e/test_error_handling.py" in stage
    assert '"--ignore-glob=*_e2e.py"' in stage
    assert "_mixed_e2e_cpu_contract_files" in script
    assert 'f"--deselect={path}::test_model_e2e"' in stage
    assert '"-q"' in stage and '"-x"' in stage
    assert '"--dist=worksteal"' in stage
    assert '"--import-mode=importlib"' in stage
    assert 'if scope == "community-all"' in stage
    assert "test_distinct_explicit_hf_cache_paths_reach_both_containers" in stage
    assert 'not model_proof_allocator"' in stage
    assert '"-m"' in stage and '"model_proof_allocator"' in stage
    assert '["trtmc", "test_cli_args", "test_config_cli_support"]' in script
    assert '["trtmc", "trtmc_cpu_cpp_tests"]' in script
    assert '["-L", "cpu"]' in script
    assert '"TRTMC_PREMERGE_UNIT_SCOPE", "all"' in stage
    assert "tests/builder/test_cli.py" in script
    assert 'if scope == "builder"' in script
    assert '["tests/builder/"]' in script
    assert "if native_targets:" in stage
    assert '[build / "trtmc", "version"]' in stage
    assert '[build / "trtmc", "--help"]' in stage
    assert "--stop-on-failure" in stage
    assert "-DTRTMC_ENABLE_TRT=OFF" not in stage
    assert "-DTRTMC_BUILD_BACKEND_TRT=OFF" not in stage
    assert "-DTRTMC_ENABLE_TVM_FFI=OFF" not in stage
    assert "conan " not in stage
    assert "build_pip_package" not in stage
    assert "--ignore=tests/builder/test_flashinfer_benchmark.py" in stage
    assert "--ignore=tests/builder/test_tvm_ffi_plugin.py" in stage
    assert "trtmc_model_plugins" not in stage
    assert "add_custom_target(trtmc_platform_cpp_tests)" in cmake
    assert "add_custom_target(trtmc_cpu_cpp_tests)" in cmake
    assert "if(ARG_UNPARSED_ARGUMENTS)" in cmake
    assert (
        "add_dependencies(trtmc_cpu_cpp_tests test_optimized_runtime_bundle_contract)" in cmake
    )
    assert len(re.findall(r"(?m)^\s*add_test\(", cmake)) == 2
    assert "add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})" in cmake
    assert "NAME test_optimized_runtime_bundle_contract" in cmake
    assert '"${_trtmc_test_owner_label};${_trtmc_test_resource_label}"' in cmake
    assert "trtmc_add_test(test_model_plugin_loader MODEL_OWNED)" in cmake
    assert "test_c_abi_runtime_regression" not in cmake
    assert (
        "test_c_abi_runtime_regression|test_c_abi_runtime_regression.cpp|"
        "trtmc_model_qwen|_|REQUIRES_GPU"
        in qwen_manifest
    )
    assert "MODEL_OWNED\n        ${_trtmc_test_options}" in cmake
    for gpu_test in (
        "test_trt_runtime_lifetime REQUIRES_TRT REQUIRES_GPU",
        "test_trt_module REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_buffer REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_stream REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_graph REQUIRES_TRT REQUIRES_GPU",
        "test_device_tensor REQUIRES_GPU",
        "test_tvm_ffi_plugin REQUIRES_TRT REQUIRES_GPU",
        "test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE REQUIRES_TRT REQUIRES_GPU",
        "test_tvm_ffi_module_loader REQUIRES_TRT REQUIRES_GPU",
    ):
        assert f"trtmc_add_test({gpu_test})" in cmake


def test_builder_unit_scope_runs_python_without_native_build(tmp_path: Path) -> None:
    selected_test = "tests/e2e/models/qwen/test_qwen_native_kv_routing.py"
    selected_path = tmp_path / selected_test
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text("def test_selected(): pass\n", encoding="utf-8")

    class RecordingContext:
        repository = tmp_path
        env = {
            "GITHUB_WORKSPACE": str(tmp_path),
            "TRTMC_PREMERGE_UNIT_SCOPE": "builder",
            "TRTMC_PREMERGE_PYTHON_TEST_TARGETS": json.dumps([selected_test]),
            "TRTMC_UNIT_BUILD_JOBS": "8",
            "TRTMC_UNIT_TEST_JOBS": "8",
        }

        def __init__(self) -> None:
            self.commands: list[list[object]] = []

        def positive_integer(self, value: str, _name: str) -> int:
            return int(value)

        def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = RecordingContext()
    UnitTestRunner(context).premerge()

    pytest_commands = [
        command for command in context.commands if command[:3] == ["python", "-m", "pytest"]
    ]
    assert len(pytest_commands) == 1
    assert "tests/builder/" in pytest_commands[0]
    assert selected_test in pytest_commands[0]
    assert not [command for command in context.commands if command[0] in {"cmake", "ctest"}]


def test_all_unit_scope_isolates_shared_and_family_python_suites(tmp_path: Path) -> None:
    mixed_file = (
        tmp_path / "tests/e2e/models/example/optimized_adapter/test_example_e2e.py"
    )
    mixed_file.parent.mkdir(parents=True)
    mixed_file.write_text(
        "def test_model_e2e(): pass\n"
        "def test_manifest_contract(): pass\n",
        encoding="utf-8",
    )

    class RecordingContext:
        repository = tmp_path
        env = {
            "GITHUB_WORKSPACE": str(tmp_path),
            "TRTMC_PREMERGE_UNIT_SCOPE": "community-all",
            "TRTMC_UNIT_BUILD_JOBS": "8",
            "TRTMC_UNIT_TEST_JOBS": "8",
        }

        def __init__(self) -> None:
            self.commands: list[list[object]] = []

        def positive_integer(self, value: str, _name: str) -> int:
            return int(value)

        def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = RecordingContext()
    UnitTestRunner(context).premerge()

    pytest_commands = [
        command for command in context.commands if command[:3] == ["python", "-m", "pytest"]
    ]
    source_commands = [
        command
        for command in pytest_commands
        if "model_proof_allocator" not in command
        and "tests/e2e/test_error_handling.py" not in command
    ]
    assert len(source_commands) == 4
    shared, models, family, mixed = source_commands
    assert all(
        target in shared for target in ("tests/builder/", "tests/tools/", "tests/e2e_harness/")
    )
    assert "tests/e2e/models/" not in shared
    assert "--ignore-glob=*_e2e.py" not in shared
    assert all(
        target in models
        for target in (
            "tests/e2e/models/",
            "tests/test_e2e_selection.py",
            "tests/e2e/test_diffusion_image_parity_inputs.py",
        )
    )
    assert "python/tensorrt_model_connect/families/" not in models
    assert "--ignore-glob=*_e2e.py" in models
    assert "python/tensorrt_model_connect/families/" in family
    assert "tests/e2e/models/" not in family
    assert "--ignore-glob=*_e2e.py" not in family
    relative_mixed = "tests/e2e/models/example/optimized_adapter/test_example_e2e.py"
    assert relative_mixed in mixed
    assert f"--deselect={relative_mixed}::test_model_e2e" in mixed
    assert "--ignore-glob=*_e2e.py" not in mixed
    for command in source_commands:
        assert "--import-mode=importlib" in command
        assert "not gpu and not trt and not e2e and not model_proof_allocator" in command

    build_command = next(
        command for command in context.commands if command[:2] == ["cmake", "--build"]
    )
    assert "trtmc_cpu_cpp_tests" in build_command
    ctest_command = next(command for command in context.commands if command[0] == "ctest")
    label_index = ctest_command.index("-L")
    assert ctest_command[label_index : label_index + 2] == ["-L", "cpu"]


def test_mixed_e2e_cpu_contract_inventory_is_not_hidden() -> None:
    class InventoryContext:
        repository = REPO_ROOT

    selected = set(UnitTestRunner(InventoryContext())._mixed_e2e_cpu_contract_files())

    assert {
        "tests/e2e/models/fast_foundation_stereo/test_fast_foundation_stereo_e2e.py",
        "tests/e2e/models/minimax_h3/test_minimax_h3_e2e.py",
    } <= selected


@pytest.mark.parametrize(
    "selected_test",
    [
        "../outside/test_bad.py",
        "tests/e2e/models/qwen/test_qwen_e2e.py",
        "-k",
    ],
)
def test_premerge_rejects_an_unsafe_selected_python_test(
    tmp_path: Path,
    selected_test: str,
) -> None:
    class RecordingContext:
        repository = tmp_path
        env = {
            "GITHUB_WORKSPACE": str(tmp_path),
            "TRTMC_PREMERGE_UNIT_SCOPE": "builder",
            "TRTMC_PREMERGE_PYTHON_TEST_TARGETS": json.dumps([selected_test]),
            "TRTMC_UNIT_BUILD_JOBS": "8",
            "TRTMC_UNIT_TEST_JOBS": "8",
        }

        def positive_integer(self, value: str, _name: str) -> int:
            return int(value)

        def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(CiError, match="selected Python test target"):
        UnitTestRunner(RecordingContext()).premerge()


def test_unowned_gpu_only_builder_suites_are_excluded_from_cpu_units() -> None:
    stage = _ci_source("quality.py")
    for relative in (
        "tests/builder/test_flashinfer_benchmark.py",
        "tests/builder/test_tvm_ffi_plugin.py",
    ):
        assert f"--ignore={relative}" in stage

    ffi_architecture = (REPO_ROOT / "tests/builder/test_ffi_architecture.py").read_text()
    flashinfer_section = ffi_architecture.split("class TestFlashInferKernelSetup:", maxsplit=1)[
        1
    ].split("class TestEngineBuilderKernelArtifacts:", maxsplit=1)[0]
    assert flashinfer_section.count("@pytest.mark.gpu") == 3
    assert flashinfer_section.count("@pytest.mark.trt") == 3


def test_package_stage_builds_py310_and_py312_wheels() -> None:
    text = _ci_source("package.py", "pipeline.py")
    assert '"TRTMC_PACKAGE_PYTHON_TAGS", "py310 py312"' in text
    assert '"WHEEL_PYVER": tag' in text
    assert '"build"' in text and '"--wheel"' in text and '"--outdir"' in text
    assert 'f"build-dir={tag_root}"' in text
    assert "manylinux_2_39_aarch64" in text
    assert '"wheel-model-smoke":' in text
    assert "Model smoke test from trtmc pip package" in text


def test_package_reuses_conan_cmake_build_directory(tmp_path: Path) -> None:
    from tools.ci.package import WheelPackageManager

    release = tmp_path / "conan_out" / "build" / "Release"
    release.mkdir(parents=True)
    (release / "CMakeCache.txt").touch()

    assert WheelPackageManager(object())._conan_cmake_build_dir(tmp_path / "conan_out") == release


def test_package_smoke_default_is_model_owned() -> None:
    path, data = _single_default_model_config("package_smoke.json")
    assert path.parent.parent == REPO_ROOT / "tests" / "e2e" / "models"
    for key in (
        "name",
        "model_id",
        "bundle",
        "timing_cache",
        "max_cache",
        "max_new_tokens",
        "optimization_level",
        "build_timeout",
        "run_timeout",
        "precision",
        "prompt",
    ):
        assert data.get(key)
    assert isinstance(data.get("run_args", []), list)


def test_package_smoke_ci_surface_has_no_model_owned_names() -> None:
    shared_paths = (
        REPO_ROOT / "tools" / "ci" / "stage.py",
        REPO_ROOT / "tools" / "ci" / "container.py",
        REPO_ROOT / "tools" / "ci" / "package.py",
        REPO_ROOT / "tools" / "ci" / "pipeline.py",
    )
    config_path, data = _single_default_model_config("package_smoke.json")
    family = config_path.parent.name
    model_name = str(data["name"])
    model_prefix = model_name.split("-", maxsplit=1)[0]
    family_tokens = {family, model_prefix}
    forbidden = {
        str(data[key]) for key in ("model_id", "name", "bundle", "timing_cache") if data.get(key)
    }
    for token in family_tokens:
        forbidden.update(
            {
                f"TRTMC_WHEEL_{token.upper()}",
                f"wheel-{token}-smoke",
                f"{token.title()} smoke test from trtmc pip package",
                f"trtmc-wheel-{token}-smoke",
            }
        )
    violations = [
        (path, needle)
        for path in shared_paths
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not violations


def test_package_stage_requires_manylinux_aarch64_wheels() -> None:
    text = _ci_source("package.py")
    assert '"TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64"' in text
    assert "self.platform = platform" in text
    assert "native wheel must not contain .data/purelib entries" in text
    assert ".data/scripts/trtmc" in text
    assert "native trtmc must be installed directly, not via console_scripts" in text
    assert '"auditwheel>=6.2"' in text
    assert 'sys.executable, "-m", "auditwheel", "show", wheel' in text
    assert 'f"*-{tag}-none-{platform}.whl"' in text
    assert "_validate_build_platform" in text
    assert "build_glibc" in text


def test_package_stage_uses_conan_py_build_inputs() -> None:
    text = _ci_source("package.py")
    assert '"CONAN_PY_BUILD_PROFILE_AUTODETECT": "1"' in text
    assert '"TRTMC_TRT_INCLUDE_DIR": trt_include' in text
    assert '"TRTMC_TRT_LIBRARY": trt_library' in text
    assert '"TRTMC_CUDA_INCLUDE_DIR": cuda_include' in text
    assert '"TRTMC_CUDART_LIBRARY": cudart' in text


def test_impact_stage_reuses_cached_json_for_summary() -> None:
    text = _ci_source("quality.py")
    assert 'arguments.append("--json")' in text
    assert "write_text(result.stdout" in text
    assert "ImpactResult(**impact)" in text
    assert '"--verbose"' not in text


def test_python_builder_fallback_is_per_tier() -> None:
    script = _ci_source("coverage.py")
    assert 'if {"builder", "tools"}.issubset(fallback)' in script
    assert '["tests/builder/"] if "builder" in fallback' in script
    assert 'if "tools" in fallback' in script
    assert 'add(["tests/tools/"])' in script
    assert 'glob("test_*.py")' in script


def test_release_wheel_build_disables_libtorch_linkage() -> None:
    text = (REPO_ROOT / "conanfile.py").read_text()
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in text


def test_model_plugins_are_staged_for_installed_trtmc() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    loader = (REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_plugin_loader.cpp").read_text()

    assert "install(TARGETS trtmc_model_${_trtmc_model}" in cmake
    assert "${CMAKE_INSTALL_LIBDIR}/trtmc/models/${_trtmc_model}" in cmake
    assert 'cmake.build(target="trtmc_model_plugins")' in conanfile
    assert '"libtrtmc_model_*.so*"' in conanfile
    assert 'rglob("libtrtmc_model_*.so*")' in conanfile
    assert "src=str(model_plugin.parent)" in conanfile
    assert "model_plugins = sorted(package_bin.glob" in conanfile
    assert "TRTMC model plugin DSOs were not staged" in conanfile
    assert '"site-packages" / "tensorrt_model_connect" / "bin"' in loader
    assert '"trtmc" / "models"' in loader


def test_release_wheel_stages_core_runtime_and_uses_origin_rpath() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    script = _ci_source("package.py")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert "set_target_properties(trtmc PROPERTIES" in cmake
    assert "set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)" in cmake
    assert "TRTMC_DISTRIBUTABLE_BUILD" in cmake
    assert 'toolchain.cache_variables["TRTMC_DISTRIBUTABLE_BUILD"]' in conanfile
    assert "BUILD_RPATH_USE_ORIGIN TRUE" in cmake
    assert 'INSTALL_RPATH "\\$ORIGIN"' in cmake
    assert '"libtrtmc_core.so*"' in conanfile
    assert "for destination in (package_bin, wheel_data_scripts):" in conanfile
    assert "TRTMC core DSO was not staged beside the wheel script" in conanfile
    assert "_set_wheel_runpath" in conanfile
    assert '"$ORIGIN:$ORIGIN/../../tensorrt_libs:/usr/local/cuda/lib64"' in conanfile
    assert "apache-tvm-ffi==0.1.12" in pyproject
    assert "script_cores" in script
    assert 'if "$ORIGIN" not in dynamic' in script
    assert "installed trtmc RUNPATH leaks the CI build directory" in script
    assert '"TRTMC_DISTRIBUTABLE_BUILD": "1"' in script
    assert "wheel embeds its CI checkout path" in script


def test_ci_source_build_defaults_to_packaged_libtorch_mode() -> None:
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    wrapper = _ci_source("environment.py")
    coverage = _ci_source("coverage.py")
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in conanfile
    assert "self._value('TRTMC_ENABLE_LIBTORCH_MULTINOMIAL', 'OFF')" in coverage
    assert '"-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="' in coverage
    assert "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL" in wrapper


def test_ci_cpp_test_build_reuses_wheel_conan_tree() -> None:
    script = _ci_source("quality.py", "package.py")
    assert '"TRTMC_CONAN_ENABLE_TEST_TARGETS": "1"' in script
    assert '"TRTMC_CONAN_BUILD_TARGETS": "\\n".join(targets)' in script
    assert '"conan", "build", ".", "-of", metadata["conan_out_dir"]' in script
    assert 'arguments = ["ctest", "--test-dir", build_dir]' in script
    assert "build_metadata" in script


def test_selective_e2e_builds_and_runs_single_family_source_projections() -> None:
    selective = _ci_source("e2e.py")
    group_runner = _ci_source("isolation.py")
    script = selective + group_runner

    assert '"tools/model_plugin_isolation.py"' in group_runner
    assert '"plan"' in group_runner
    assert "tools/model_plugin_isolation.py" in selective
    assert "IsolatedModelRunner" in selective
    assert "impact-models" in selective
    assert "e2e_isolation_models.txt" in selective
    assert "E2EParallelRunner" in selective
    assert '"--exclude-ci-tier"' in selective
    assert '"nightly_only"' in selective
    assert '"multi_device"' in selective
    assert "if standard_rc" in selective
    assert "strict model-owned isolation E2E" in selective
    assert "_prepare_plugins" in selective
    assert '"tools/model_plugin_isolation.py"' in group_runner
    assert '"schedule"' in group_runner
    assert "_run_queue" in group_runner
    assert "concurrent.futures" in group_runner
    assert '"stage-source"' in group_runner
    assert "def _configure" in group_runner
    assert "CMAKE_TOOLCHAIN_FILE" in script
    assert "FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON" in script
    assert 'str(group["runtime_plugin"]["target"])' in group_runner
    assert '"PYTHONPATH": f"{source / \'python\'}:{source}"' in group_runner
    assert '"LD_LIBRARY_PATH": ":".join(library_path)' in group_runner
    assert '"--trtmc-binary"' in group_runner and 'build / "trtmc"' in group_runner
    assert '"--engine-dir"' in group_runner and "engines" in group_runner
    assert '"--model-plugin-dir"' in group_runner and "plugins" in group_runner
    assert '"CUDA_VISIBLE_DEVICES": str(gpu)' in group_runner
    assert '"--rootdir"' in group_runner and "source" in group_runner
    assert '"--e2e-models-file"' in group_runner and "models" in group_runner
    assert '"--e2e-exclude-ci-tier"' in group_runner and '"nightly_only"' in group_runner
    assert '"SELECTIVE_E2E_GROUP_TIMEOUT", "90m"' in group_runner
    assert "for model in selected" in group_runner
    assert "verify-results" in group_runner
    assert "expected exactly 1" in group_runner
    assert "prepare_model_plugin_dir" not in group_runner


def test_full_e2e_stages_all_runtime_plugins_from_reusable_build() -> None:
    script = _ci_source("e2e.py")
    assert 'self._prepare_plugins(plugins, ["--all"])' in script
    assert '"--model-plugin-dir"' in script


def test_cpp_coverage_builds_excluded_test_target() -> None:
    coverage = _ci_source("coverage.py")
    assert '"CPP_COVERAGE_BUILD_TARGET", "trtmc_cpp_tests"' in coverage
    assert '["cmake", "--build", build_dir, "--target", build_target, *parallel]' in coverage
    assert "CommandRunner(cwd=build_dir" in coverage


def test_cpp_coverage_gate_excludes_model_owned_runtime_plugins() -> None:
    coverage = _ci_source("coverage.py")
    assert 'self._words("GCOVR_EXCLUDES")' in coverage
    assert 'str(self.repository / "src/runtime/models")' in coverage
    assert 'gcovr_base.extend(("--exclude", value))' in coverage


def test_cpp_coverage_engine_runs_tools_directly_without_shell(tmp_path: Path, monkeypatch) -> None:
    from tools.ci.coverage import CppCoverageEngine

    commands: list[list[str]] = []
    gcovr_commands: list[list[str]] = []

    class Context:
        repository = tmp_path
        env = {"PATH": os.environ["PATH"], "BUILD_DIR": "relative-build"}

        @staticmethod
        def executable(name: str) -> str:
            return name

        @staticmethod
        def output(command: list[str]) -> str:
            assert command == ["gcovr", "--help"]
            return "--fail-under-function"

        @staticmethod
        def run(command, **_kwargs):
            commands.append([str(item) for item in command])
            return subprocess.CompletedProcess(command, 0, "", "")

    def fake_gcovr(_runner, command, **_kwargs):
        rendered = [str(item) for item in command]
        gcovr_commands.append(rendered)
        stdout = "lines: 100.0%\nfunctions: 100.0%\nbranches: 100.0%\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("tools.ci.coverage.CommandRunner.run", fake_gcovr)
    report = tmp_path / "reports"
    result = CppCoverageEngine(Context(), report).run(
        ["-L", "platform"],
        build_target="trtmc_platform_cpp_tests",
        limit=None,
    )

    assert result == 0
    assert [
        "cmake",
        "--build",
        str(tmp_path / "relative-build"),
        "--target",
        "trtmc_platform_cpp_tests",
        "--parallel",
    ] in commands
    assert [
        "ctest",
        "--test-dir",
        str(tmp_path / "relative-build"),
        "--output-on-failure",
        "-L",
        "platform",
    ] in commands
    assert len(gcovr_commands) == 3
    assert all(
        "--gcov-ignore-parse-errors" in command
        and command[command.index("--gcov-ignore-parse-errors") + 1]
        == "negative_hits.warn_once_per_file"
        for command in gcovr_commands
    )
    gate = gcovr_commands[-1]
    assert gate[gate.index("--fail-under-line") + 1] == "100"
    assert gate[gate.index("--fail-under-function") + 1] == "100"
    assert gate[gate.index("--fail-under-branch") + 1] == "100"
    assert all(command[0] != "bash" for command in commands + gcovr_commands)
    assert (report / "cpp-coverage-summary.txt").is_file()


def test_python_coverage_engine_runs_coverage_directly(tmp_path: Path) -> None:
    from tools.ci.coverage import PythonCoverageEngine

    commands: list[list[str]] = []

    class Context:
        repository = tmp_path
        env = {"PATH": os.environ["PATH"]}

        @staticmethod
        def executable(name: str) -> str:
            return name

        @staticmethod
        def run(command, **_kwargs):
            rendered = [str(item) for item in command]
            commands.append(rendered)
            stdout = ""
            if rendered[2:4] == ["coverage", "report"]:
                stdout = "TOTAL 10 0 100%\n"
            if rendered[2:4] == ["coverage", "xml"]:
                destination = Path(rendered[rendered.index("-o") + 1])
                destination.write_text(
                    '<coverage line-rate="1.0" branch-rate="1.0"/>\n',
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")

    report = tmp_path / "reports"
    PythonCoverageEngine(Context(), report).run(["-q"])

    coverage_run = next(command for command in commands if command[2:4] == ["coverage", "run"])
    assert coverage_run[-3:] == ["tests/builder", "tests/tools", "-q"]
    assert all(command[0] != "bash" for command in commands)
    assert (report / "python-cobertura.xml").is_file()
    assert (report / "python-coverage.txt").read_text() == "TOTAL 10 0 100%\n"


def test_root_pyproject_configures_conan_py_build_wheel() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    backend_text = (REPO_ROOT / "_pyproject_backend.py").read_text()
    assert 'build-backend = "_pyproject_backend"' in text
    assert "return [_CONAN_PY_BUILD_REQUIREMENT]" in backend_text
    assert "conan_build.build_wheel" in backend_text
    assert "conan_build.build_sdist" in backend_text
    assert "_py_only_enabled" in backend_text
    assert 'packages = ["python/tensorrt_model_connect"]' in text
    assert "[project.scripts]" not in text


def test_wheel_model_smoke_checks_py312_wheel_only() -> None:
    text = _ci_source("package.py")
    smoke_block = text.split("def model_smoke", maxsplit=1)[1].split(
        "def _clean_venv_smoke", maxsplit=1
    )[0]
    assert 'self.select_wheel("py312")' in smoke_block
    assert "sys.version_info[:2] != (3, 12)" in smoke_block
    assert "TRTMC_WHEEL_SMOKE_PYTHON" not in smoke_block
    assert "select_compatible_wheel" not in smoke_block
    assert '"PATH"' not in smoke_block
    assert 'trtmc,\n                "build"' in smoke_block
    assert "InstalledWheelValidator.require_elf(trtmc)" in smoke_block
    assert (
        smoke_block.index("self._create_venv(venv, wheel)")
        < smoke_block.index("self._install_model_smoke_dependencies(python)")
        < smoke_block.index('self.context.run([python, "-m", "pip", "check"])')
    )


def test_wheel_model_smoke_installs_pinned_cpu_torch() -> None:
    from tools.ci.package import (
        PACKAGE_SMOKE_TORCH_INDEX,
        PACKAGE_SMOKE_TORCH_VERSION,
        WheelPackageManager,
    )

    class RecordingContext:
        def __init__(self) -> None:
            self.commands: list[list[object]] = []

        def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = RecordingContext()
    python = Path("venv/bin/python")
    WheelPackageManager(context)._install_model_smoke_dependencies(python)

    assert context.commands[0] == [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--index-url",
        PACKAGE_SMOKE_TORCH_INDEX,
        f"torch=={PACKAGE_SMOKE_TORCH_VERSION}",
    ]
    assert context.commands[1][:3] == [python, "-I", "-c"]
    assert PACKAGE_SMOKE_TORCH_VERSION in str(context.commands[1][3])
    assert "torch.version.cuda is None" in str(context.commands[1][3])

    package_text = _ci_source("package.py")
    clean_smoke_block = package_text.split("def _clean_venv_smoke", maxsplit=1)[1].split(
        "def _create_venv", maxsplit=1
    )[0]
    assert "_install_model_smoke_dependencies" not in clean_smoke_block

    for dockerfile_name in ("Dockerfile.dev.aarch64", "Dockerfile.dev.x86"):
        dockerfile = (REPO_ROOT / dockerfile_name).read_text()
        assert f"ARG TORCH_VERSION={PACKAGE_SMOKE_TORCH_VERSION}" in dockerfile
        assert f"ARG PYTORCH_CPU_INDEX={PACKAGE_SMOKE_TORCH_INDEX}" in dockerfile


def test_selective_e2e_zero_model_path_still_generates_report_input_dir() -> None:
    text = _ci_source("e2e.py")
    zero_model_block = text.split(
        'print("No E2E models affected by this change -- skipping E2E tests")',
        maxsplit=1,
    )[1].split("return", maxsplit=1)[0]
    assert '"e2e_artifacts/artifacts"' in zero_model_block
    assert "mkdir(parents=True, exist_ok=True)" in zero_model_block


def test_etth1_model_proofs_use_the_single_validation_engine_entry_point() -> None:
    stage = _ci_source("validation.py", "model_proof.py", "model_proof_inner.py")
    validation_engine = (REPO_ROOT / "tools" / "validation" / "engine.py").read_text()

    assert '"/src/tools/validation/engine.py"' in stage
    assert '"prepare-ci-dataset"' in stage
    assert '"eval"' in stage
    assert "validation_engine_ci.py" not in stage
    for argument in (
        "--suite",
        "etth1_time_series_parity",
        "--ci-lane",
        "nightly",
        "--engine-dir",
        "/work/engines",
        "--model-plugin-dir",
        "/work/model-plugins",
        "--require-prebuilt-bundles",
    ):
        assert f'"{argument}"' in stage
    assert "ETTh1 validation requires a GB300 GPU" in stage
    assert '"--network"' in stage and '"none"' in stage
    assert "validate_eval_summary" in validation_engine
    assert 'result.get("status") == "passed"' in validation_engine
    assert "return complete and all" in validation_engine
    assert (
        '"work_dir"'
        not in validation_engine.split("def _public_ci_result", maxsplit=1)[1].split(
            ")", maxsplit=1
        )[0]
    )


def test_etth1_dataset_preparation_imports_the_projected_python_package(
    tmp_path: Path,
) -> None:
    from tools.ci.validation import ValidationDatasetPreparer

    projection = tmp_path / "projection"
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    destination = work / "validation-data" / ValidationDatasetPreparer.DATASET
    projection.mkdir()
    artifacts.mkdir()
    docker_commands: list[list[str]] = []

    class Commands:
        @staticmethod
        def run(command, **_kwargs):
            rendered = [str(item) for item in command]
            docker_commands.append(rendered)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("date,HUFL\n", encoding="utf-8")
            return subprocess.CompletedProcess(rendered, 0, "", "")

    class Context:
        commands = Commands()

        @staticmethod
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "", "")

    result = ValidationDatasetPreparer(
        Context(),
        "nightly",
        "timesfm",
        projection,
        work,
        artifacts,
        "ci-image",
        "dataset-container",
        [],
    ).prepare()

    assert result == destination.parents[1]
    assert len(docker_commands) == 1
    command = docker_commands[0]
    image_index = command.index("ci-image")
    assert command.index("PYTHONPATH=/src/python:/src") < image_index
    assert command.index("PYTHONNOUSERSITE=1") < image_index
    assert command[image_index + 1 : image_index + 4] == [
        "/opt/venv/bin/python",
        "/src/tools/validation/engine.py",
        "prepare-ci-dataset",
    ]

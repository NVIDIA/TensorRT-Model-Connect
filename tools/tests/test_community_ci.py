# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CI entrypoint."""

from __future__ import annotations

import shlex

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import community_ci, legal_headers


REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow_step_script(workflow_name: str, job_name: str, step_name: str) -> str:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
    )
    return next(
        step["run"] for step in workflow["jobs"][job_name]["steps"] if step["name"] == step_name
    )


def test_pre_commit_config_installs_only_lightweight_commit_hooks() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    assert "default_install_hook_types" not in config

    repositories = {repository["repo"]: repository for repository in config["repos"]}
    assert repositories["https://github.com/astral-sh/ruff-pre-commit"]["rev"] == "v0.16.4"
    assert repositories["https://github.com/pre-commit/mirrors-clang-format"]["rev"] == "v22.1.8"

    hooks = {hook["id"]: hook for repository in config["repos"] for hook in repository["hooks"]}
    for hook_id in ("trailing-whitespace", "end-of-file-fixer", "check-yaml"):
        assert hooks[hook_id]["stages"] == ["pre-commit"]
    assert hooks["ruff-check"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["entry"] == "clang-format --dry-run --Werror"
    assert all(hook["stages"] == ["pre-commit"] for hook in hooks.values())

    source = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "python3 -m tools.community_ci format-" not in source
    assert "pre-push" not in source


def test_contributor_guide_matches_the_live_ci_flow() -> None:
    source = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    ordered = [
        "pre-commit install --install-hooks",
        "git commit --signoff",
        "git push --set-upstream origin",
        "### 9. Run contributor-visible Community CI",
        "### 10. Hand off for NVIDIA maintainer review",
    ]
    positions = [source.index(marker) for marker in ordered]
    assert positions == sorted(positions)
    for marker in (
        "Stable Community GPU CI",
        "single automated premerge",
        "current PR head",
        "five directly changed or added families",
        "confirmed VM cleanup",
        "NVIDIA employee maintainer",
        "does not approve",
        "Private Nightly",
        "no repository, Hub, Brev or registry credentials",
    ):
        assert marker in source
    assert "run-internal-ci" not in source
    assert "trtmc/premerge/required" not in source


def test_internal_premerge_bridge_is_retired() -> None:
    workflows = REPO_ROOT / ".github/workflows"
    assert not (workflows / "internal-ci-bridge.yml").exists()
    for path in workflows.glob("*.yml"):
        source = path.read_text(encoding="utf-8")
        assert "/actions/workflows/premerge.yml/dispatches" not in source
        if "TRTMC_CI_DISPATCH_TOKEN" in source:
            workflow = yaml.safe_load(source)
            assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}
            assert "Automated premerge gate" not in source


def test_website_contributing_page_points_to_the_canonical_guide() -> None:
    source = (REPO_ROOT / "website/docs/extend/contributing.md").read_text(encoding="utf-8")
    assert "CONTRIBUTING.md" in source


def test_impact_publishes_only_the_public_cpu_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_output = tmp_path / "github-output"
    github_summary = tmp_path / "github-summary"
    runner = community_ci.CommunityCI(
        REPO_ROOT,
        {
            **os.environ,
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_STEP_SUMMARY": str(github_summary),
        },
    )
    monkeypatch.setattr(runner, "resolve_base", lambda _base: "base-sha")
    monkeypatch.setattr(community_ci.test_impact, "validate", lambda _repo: None)
    monkeypatch.setattr(
        community_ci.test_impact,
        "changed_files",
        lambda *_args: ["families/qwen/model.py"],
    )
    monkeypatch.setattr(
        community_ci.test_impact,
        "classify",
        lambda *_args: community_ci.test_impact.Impact(
            scope="families",
            families=("qwen",),
            direct_families=("qwen",),
            changed_files=("families/qwen/model.py",),
            run_core_tests=True,
            run_docs=False,
        ),
    )

    result = runner.impact(None)

    assert result["families"] == ["qwen"]
    assert github_output.read_text(encoding="utf-8") == 'families=["qwen"]\n'
    summary = github_summary.read_text(encoding="utf-8")
    assert "families/qwen/model.py" in summary


@pytest.mark.parametrize("change", ["valid", "missing-header", "LICENSE", "NOTICE"])
def test_public_source_quality_enforces_legal_compliance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    change: str,
) -> None:
    header = legal_headers.HASH_STYLE.render(b"\n").decode() + "\n\n"
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "tools/legal_headers.py", tools_dir / "legal_headers.py")
    (tools_dir / "legal_header_exceptions.toml").write_text(
        header + "schema_version = 1\n", encoding="utf-8"
    )
    for name in ("LICENSE", "NOTICE"):
        (tmp_path / name).write_text("Original legal document.\n", encoding="utf-8")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "Base fixture")
    base = git("rev-parse", "HEAD")
    source = tmp_path / "families/example/support.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        ("" if change == "missing-header" else header) + '"""Example family support."""\n',
        encoding="utf-8",
    )
    if change in ("LICENSE", "NOTICE"):
        (tmp_path / change).write_text("Changed legal document.\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "Contribution fixture")

    # Keep the real public entrypoint, header audit, and Git comparison; unrelated
    # architecture and formatter checks need the full project and toolchain.
    for name in ("family_coverage", "complexity", "lint_changed_files", "architecture_contracts"):
        monkeypatch.setattr(community_ci.SourceQualityChecks, name, lambda _self: None)
    runner = community_ci.CommunityCI(tmp_path, dict(os.environ))
    if change == "valid":
        runner.source_quality(base)
        assert "findings=0" in capfd.readouterr().out
    else:
        with pytest.raises(community_ci.CiError):
            runner.source_quality(base)
        captured = capfd.readouterr()
        output = captured.out + captured.err
        if change == "missing-header":
            assert (
                "[missing] families/example/support.py: missing required hash SPDX header" in output
            )
        else:
            assert change in output


def test_public_workflow_is_one_exact_merge_cpu_then_gpu_authorization():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    assert set(workflow[True]) == {"pull_request", "pull_request_target", "workflow_dispatch"}
    assert workflow[True]["workflow_dispatch"]["inputs"]["ci_lane"]["options"] == [
        "stable",
        "dev",
    ]
    assert "Stable" in workflow["run-name"] and "Dev" in workflow["run-name"]
    assert workflow["permissions"] == {}
    assert "paths" not in workflow[True]["pull_request"]
    assert "paths" not in workflow[True]["pull_request_target"]
    jobs = workflow["jobs"]
    authorization = jobs["authorize"]["steps"][0]["run"]
    assert 'stable) test "$CI_REF" = refs/heads/main' in authorization
    assert "Unknown Community CI lane" in authorization
    for name in ("source-quality", "docs", "ownership-impact", "unit"):
        job = jobs[name]
        assert job["needs"] == "authorize"
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["permissions"] == {"contents": "read"}
        assert "secrets." not in json.dumps(job)
        assert "environment" not in job
        assert job["steps"][0]["with"] == {
            "ref": "${{ needs.authorize.outputs.merge_sha }}",
            "fetch-depth": 0,
            "persist-credentials": False,
        }
    # Metadata-only entry runs skip the gate; failed authorization on an
    # executor must still run it so skipped CPU stages produce failure.
    assert jobs["required"]["if"] == (
        "${{ !cancelled() && (github.event_name == 'pull_request' || "
        "(github.event_name == 'workflow_dispatch' && inputs.task != 'dependency-image' "
        "&& inputs.task != 'dependency-image-audit' && inputs.task != 'dependency-image-withdraw' "
        "&& inputs.task != 'dependency-image-access-check' "
        "&& inputs.source_snapshot != '')) }}"
    )
    assert jobs["required"]["needs"] == [
        "authorize",
        "source-quality",
        "docs",
        "ownership-impact",
        "unit",
    ]
    assert jobs["gpu-authorize"]["needs"] == ["authorize", "required"]
    assert "needs.required.result == 'success'" in jobs["gpu-authorize"]["if"]
    assert jobs["provision-and-test"]["needs"] == "gpu-authorize"
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in jobs["provision-and-test"]["if"]
    assert jobs["publish"]["needs"] == [
        "authorize",
        "required",
        "gpu-authorize",
        "provision-and-test",
        "cleanup",
    ]
    assert "allow-unsafe-pr-checkout" not in json.dumps(workflow)
    assert workflow["env"]["COMMUNITY_GPU_EXECUTION_ENABLED"] == "true"
    assert workflow[True]["workflow_dispatch"]["inputs"]["run_gpu_smoke"]["default"] is False
    assert jobs["unit"]["steps"][-1]["run"] == "python3 -m tools.community_ci unit"
    docs = {step["name"]: step for step in jobs["docs"]["steps"]}
    assert all("if" not in step for step in docs.values())
    assert docs["Install website dependencies"]["run"] == "npm ci"
    assert docs["Test generated model support inventory"]["run"] == "npm run test:model-support"
    assert docs["Build production documentation"]["run"] == "npm run build"


@pytest.mark.parametrize("event_head", ["", "a" * 40])
def test_community_authorize_pins_the_exact_merge_and_uses_its_base_parent(tmp_path, event_head):
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport os,sys\n"
        "print(os.environ['PULL' if any('/pulls/' in arg for arg in sys.argv) else 'MERGE'])\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    head, base, merge, tree = (value * 40 for value in "abcd")
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Capture the exact pull-request snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "EVENT_HEAD_SHA": event_head,
            "STABLE_MERGE_SHA": merge if event_head else "",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "example/source",
            "PULL": json.dumps(
                {
                    "state": "open",
                    "head": {"sha": head},
                    "base": {"ref": "main", "repo": {"full_name": "example/source"}},
                    "merge_commit_sha": "f" * 40 if event_head else merge,
                }
            ),
            "MERGE": json.dumps(
                {"sha": merge, "parents": [{"sha": base}, {"sha": head}], "tree": {"sha": tree}}
            ),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["source_snapshot"]) == {
        "head_sha": head,
        "base_sha": base,
        "merge_sha": merge,
        "source_tree": tree,
    }


@pytest.mark.parametrize(
    ("source_quality", "docs", "ownership_impact", "unit", "expected_returncode"),
    [
        ("success", "success", "success", "success", 0),
        ("failure", "success", "success", "success", 1),
        ("success", "failure", "success", "success", 1),
        ("success", "skipped", "success", "success", 1),
        ("success", "success", "failure", "failure", 1),
    ],
)
def test_public_required_job_fails_closed(
    source_quality: str,
    docs: str,
    ownership_impact: str,
    unit: str,
    expected_returncode: int,
) -> None:
    environment = {
        **os.environ,
        "SOURCE_QUALITY_RESULT": source_quality,
        "DOCS_RESULT": docs,
        "OWNERSHIP_IMPACT_RESULT": ownership_impact,
        "UNIT_RESULT": unit,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml",
                "required",
                "Require every public CPU stage",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    assert f"Source quality: {source_quality}" in result.stdout
    assert f"Docs: {docs}" in result.stdout
    assert f"Ownership and impact: {ownership_impact}" in result.stdout
    assert f"Unit / C++ and Python: {unit}" in result.stdout


def test_cpu_image_installs_the_same_pinned_community_requirements() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.community-cpu").read_text(encoding="utf-8")
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "-base-ubuntu24.04@sha256:" in dockerfile
    assert "COPY community-ci.txt" in dockerfile
    assert "pip install --requirement /tmp/trtmc-community-ci.txt" in dockerfile
    assert '"libnvinfer11=${TENSORRT_APT_VERSION}"' in dockerfile
    assert '"libnvinfer-safe-headers-dev=${TENSORRT_APT_VERSION}"' in dockerfile
    assert "libcurand-dev-13-3" in dockerfile
    assert "cuda-nvrtc-dev-13-3" in dockerfile
    assert "      jq \\\n" in dockerfile
    assert "2.12.0+cu130" in dockerfile
    assert "torch.version.cuda == '13.0'" in dockerfile
    assert "ENV TORCH_CUDA_ARCH_LIST=10.0" in dockerfile
    assert "pip install --no-deps" in dockerfile
    assert '"tensorrt_cu13_bindings==${TENSORRT_VERSION}"' in dockerfile
    assert '"tensorrt==${TENSORRT_VERSION}"' not in dockerfile
    assert 'multiarch="$(gcc -dumpmachine)"' in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/trtmc-tensorrt-lib" in dockerfile
    assert "ENV TRT_INC_DIR=/opt/trtmc-tensorrt-include" in dockerfile
    assert "/usr/lib/x86_64-linux-gnu" not in dockerfile
    assert "/usr/include/x86_64-linux-gnu" not in dockerfile
    assert "NVIDIA_VISIBLE_DEVICES" not in dockerfile
    assert "!requirements/" in dockerignore
    assert "requirements/*" in dockerignore
    assert "!requirements/base.txt" in dockerignore
    assert "!requirements/community-ci.txt" not in dockerignore


def test_gpu_image_verifies_the_native_byok_dependency() -> None:
    """The GPU image fails during construction if the native TVM-FFI input is absent."""
    dockerfile = (REPO_ROOT / "Dockerfile.dev.x86-gpu").read_text(encoding="utf-8")

    assert "import onnx, tensorrt, torch, tvm_ffi" in dockerfile
    assert "metadata.version('apache-tvm-ffi') == '0.1.12'" in dockerfile


def test_cpu_image_builds_from_the_minimal_requirements_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = community_ci.CommunityCI(REPO_ROOT, dict(os.environ))
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.commands, "run", run)

    runner._ensure_cpu_image()

    assert calls[0][:3] == ["docker", "build", "--file"]
    assert calls[0][-1] == "requirements"


def test_cpu_source_policy_changes_only_the_registry_for_one_immutable_pin(tmp_path):
    script = _workflow_step_script(
        "community-ci.yml", "unit", "Use the official NVIDIA source for the pinned CUDA base"
    )
    runner = tmp_path / "runner"
    runner.mkdir()
    source = tmp_path / "Dockerfile.community-cpu"
    source.write_text("FROM example.invalid/unrelated:unchanged\n")
    environment = tmp_path / "environment"
    # No Docker operation is needed to check the workflow's generated policy;
    # a separate real BuildKit experiment verifies conversion and DENY controls.
    result = subprocess.run(
        ["bash", "-c", 'docker() { test "$*" = "buildx version"; };\n' + script],
        cwd=tmp_path,
        env={**os.environ, "RUNNER_TEMP": str(runner), "GITHUB_ENV": str(environment)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    values = dict(line.split("=", 1) for line in environment.read_text().splitlines())
    assert values["DOCKER_BUILDKIT"] == "1"
    policy = json.loads(Path(values["EXPERIMENTAL_BUILDKIT_SOURCE_POLICY"]).read_text())
    assert len(policy["rules"]) == 1
    rule = policy["rules"][0]
    assert rule["action"] == "CONVERT" and rule["selector"]["matchType"] == "EXACT"
    original, mirrored = rule["selector"]["identifier"], rule["updates"]["identifier"]
    assert original.startswith("docker-image://docker.io/nvidia/cuda:")
    assert mirrored == original.replace("docker-image://docker.io/", "docker-image://nvcr.io/", 1)
    assert (
        original.split("@", 1)[1]
        == "sha256:bcf7d05f0b13b9bbb86d9a4cd039d331894b8f1145ad009d1af75023bcd1dc5c"
    )
    assert "*" not in original
    assert source.read_text() == "FROM example.invalid/unrelated:unchanged\n"
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    steps = workflow["jobs"]["unit"]["steps"]
    assert steps[-2]["run"] == script
    assert steps[-1]["run"] == "python3 -m tools.community_ci unit"


@pytest.mark.parametrize(
    ("job_status", "test_outcome", "test_conclusion", "expected"),
    [
        ("success", "success", "success", "success"),
        ("failure", "skipped", "", "failure"),
        ("failure", "failure", "", "failure"),
        ("failure", "success", "success", "failure"),
        ("success", "skipped", "", "failure"),
        ("success", "success", "", "failure"),
        ("success", "failure", "success", "failure"),
        ("cancelled", "cancelled", "", "cancelled"),
        ("cancelled", "success", "success", "cancelled"),
        ("", "", "", "failure"),
    ],
)
@pytest.mark.parametrize("cleanup_confirmed", ["true", "false", ""])
def test_gpu_step_conclusion_requires_completed_success(
    tmp_path: Path,
    job_status: str,
    test_outcome: str,
    test_conclusion: str,
    expected: str,
    cleanup_confirmed: str,
) -> None:
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "provision-and-test", "Record the step conclusion"
            ),
        ],
        env={
            **os.environ,
            "JOB_STATUS": job_status,
            "TEST_OUTCOME": test_outcome,
            "TEST_CONCLUSION": test_conclusion,
            "CLEANUP_CONFIRMED": cleanup_confirmed,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if expected == "success" and cleanup_confirmed != "true":
        expected = "failure"
    assert output.read_text(encoding="utf-8") == f"conclusion={expected}\n"


@pytest.mark.parametrize(
    "failed",
    [
        "",
        "AUTHORIZED_RESULT",
        "CPU_RESULT",
        "GPU_AUTHORIZED",
        "GPU_RESULT",
        "TEST_RESULT",
        "GPU_FAILURE_CLASS",
        "CLEANUP_RESULT",
        "BACKSTOP_RELEASE_CONFIRMED",
    ],
)
@pytest.mark.parametrize("bad_result", ["failure", "cancelled", "skipped", ""])
def test_complete_pipeline_requires_every_cpu_and_gpu_stage(failed, bad_result):
    states = dict.fromkeys(
        (
            "AUTHORIZED_RESULT",
            "CPU_RESULT",
            "GPU_AUTHORIZED",
            "GPU_RESULT",
            "TEST_RESULT",
            "CLEANUP_RESULT",
        ),
        "success",
    )
    states["GPU_FAILURE_CLASS"] = "none"
    states["BACKSTOP_RELEASE_CONFIRMED"] = "true"
    if failed:
        states[failed] = bad_result
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "publish", "Require the complete Community CI result"
            ),
        ],
        env={**os.environ, **states, "EVENT_NAME": "workflow_dispatch", "RUN_GPU": "true"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == (1 if failed else 0), result.stderr


@pytest.mark.parametrize("failure", ["", "copy", "coordinate", "exit"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("registry_cache", [False, True])
def test_gpu_status_and_cleanup_fail_closed(
    tmp_path: Path, failure: str, cleanup_fails: bool, registry_cache: bool
) -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/community-ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["provision-and-test"]
    steps = {step["name"]: step for step in job["steps"]}
    reserve = steps["Reserve a GPU instance"]
    assert reserve["id"] == "reserve"
    assert "python3 -m tools.brev_exec provision --from-intent" in reserve["run"]
    prepare = steps["Prepare the GPU allocation intent"]
    assert 'echo "instance_name=$instance_name" >> "$GITHUB_OUTPUT"' in prepare["run"]
    test_step = steps["Build the GPU image and validate the exact PR merge"]
    assert "sudo docker build -f Dockerfile.dev.x86-gpu" not in test_step["run"]
    assert "/tmp/community_gpu_images.py --pull-base" in test_step["run"]
    assert "'$HUB_REQUIREMENT'" in test_step["run"]
    assert "git show $CI_SHA:tools/community_gpu_images.py" in test_step["run"]
    assert (
        "/tmp/trtmc-community-stage-venv/bin/python -I /tmp/community_gpu_ci.py "
        "--containers --repository /tmp/model_connect" in test_step["run"]
    )
    result = steps["Record the step conclusion"]
    assert result["id"] == "result"
    assert result["if"] == "always()"
    assert result["env"] == {
        "JOB_STATUS": "${{ job.status }}",
        "TEST_OUTCOME": "${{ steps.test.outcome }}",
        "TEST_CONCLUSION": "${{ steps.test.outputs.conclusion }}",
        "CLEANUP_CONFIRMED": "${{ steps.release.outputs.cleanup_confirmed || steps.no_allocation.outputs.cleanup_confirmed }}",
    }
    assert "${{" not in result["run"]
    cleanup = steps["Always tear down the GPU instance"]
    assert cleanup["if"] == (
        "${{ always() && steps.prepare.outcome == 'success' && steps.reserve.outcome != 'skipped' }}"
    )
    assert cleanup["env"] == {"INSTANCE_NAME": "${{ steps.prepare.outputs.instance_name }}"}
    assert test_step["env"]["INSTANCE_NAME"] == "${{ steps.reserve.outputs.instance_name }}"
    assert "python3 -m tools.brev_exec cleanup" in cleanup["run"]
    assert '--lease-file "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"' in cleanup["run"]
    assert "--until-deleted" in cleanup["run"]
    assert "|| true" not in cleanup["run"]
    assert job["outputs"] == {
        "conclusion": "${{ steps.result.outputs.conclusion }}",
        "failure_class": "${{ steps.classification.outputs.failure_class }}",
        "instance_name": "${{ steps.prepare.outputs.instance_name }}",
        "lease_artifact_name": "${{ steps.prepare.outputs.lease_artifact_name }}",
        "instance_type": "${{ steps.prepare.outputs.instance_type }}",
        "organization_id": "${{ steps.prepare.outputs.organization_id }}",
        "allocation_requested": "${{ steps.reserve.outputs.allocation_requested }}",
        "reserve_outcome": "${{ steps.reserve.outcome }}",
        "no_allocation_confirmed": "${{ steps.no_allocation.outputs.no_allocation_confirmed }}",
        "cleanup_confirmed": "${{ steps.release.outputs.cleanup_confirmed || steps.no_allocation.outputs.cleanup_confirmed }}",
    }
    cleanup_job = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup_job["if"]
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in cleanup_job["if"]
    cleanup_steps = {step["name"]: step for step in cleanup_job["steps"]}
    cleanup_script = cleanup_steps["Delete every original GPU allocation"]["run"]
    lease = tmp_path / "gpu-ci-lease/trtmc-gpu-ci-lease.json"
    lease.parent.mkdir()
    lease.write_text("{}")
    cleanup_result = subprocess.run(
        [
            "bash",
            "-c",
            'python3() { printf "%s\\n" "$*"; return 73; }\n' + cleanup_script,
        ],
        env={
            **os.environ,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "RUNNER_TEMP": str(tmp_path),
            "INSTANCE_NAME": "trtmc-gpu-ci-123-1",
        },
        capture_output=True,
        text=True,
    )
    assert cleanup_result.returncode == 73, cleanup_result.stderr
    assert cleanup_result.stdout.splitlines() == [
        "-m tools.community_gpu_lease --cleanup-all "
        f"--lease-directory {tmp_path}/gpu-ci-all-leases",
    ]
    _workflow_plan(tmp_path)
    # Execute the real workflow script with remote operations stubbed. A copy
    # can leave a partial token even when it reports failure. The ready VM must
    # receive token cleanup on every exit; test failures must not replace it.
    trace = tmp_path / "auth-trace"
    catalog = tmp_path / "trtmc-gpu-host-profiles.json"
    catalog.write_text(json.dumps({"families": {"alpha": {}} if registry_cache else {}}))
    catalog.chmod(0o600)
    stubs = r"""
sleep() { :; }
timeout() { shift; "$@"; }
brev() {
  printf 'argv %s\n' "$*" >> "$AUTH_TRACE"
  test -z "${HF_TOKEN+x}" || exit 99
  case "$1" in
    copy)
      printf 'copy %s\n' "${3%%:*}" >> "$AUTH_TRACE"
      if [[ "$3" == */registry-token ]]; then
        test "${CI_TEST_TRUSTED_MODULES_STAGED:-0}" = 1 || return 96
        CI_TEST_REGISTRY_COPIED=1
        test "$(cat "$2")" = test-registry-secret || return 97
        printf 'registry-copy %s\n' "${3%%:*}" >> "$AUTH_TRACE"
      fi
      test "$AUTH_FAILURE" != copy || return 1
      ;;
    exec)
      if [[ "$3" == *'git show '* && "$3" == *'community_gpu_images.py'* ]]; then
        CI_TEST_TRUSTED_MODULES_STAGED=1
      fi
      if [[ "$3" == *'--pull-base'* ]]; then
        test "${CI_TEST_REGISTRY_COPIED:-0}" = 1 || return 96
        CI_TEST_REGISTRY_COPIED=0
        CI_TEST_BASE_PULLED=1
      fi
      if [[ "$3" == "rm -f -- "* ]]; then
        printf 'cleanup %s\n' "$2" >> "$AUTH_TRACE"
        test "$AUTH_CLEANUP_FAILS" != true || return 1
      fi
      ;;
    delete)
      printf 'delete %s\n' "$2" >> "$AUTH_TRACE"
      return 1
      ;;
    create)
      printf 'create %s\n' "$2" >> "$AUTH_TRACE"
      return 1
      ;;
  esac
}
python3() {
  test "$1" = -m && test "$2" = tools.brev_exec || exit 98
  test "$3" != provision || exit 98
  test -z "${HF_TOKEN+x}" || exit 99
  test -z "${REGISTRY_TOKEN+x}" || exit 97
  test "${CI_TEST_BASE_PULLED:-0}" = 1 || exit 96
  test "${CI_TEST_REGISTRY_COPIED:-0}" = 0 || exit 96
  printf 'coordinate %s\n' "$INSTANCE_NAME" >> "$AUTH_TRACE"
  test "$AUTH_FAILURE" != exit || exit 17
  test "$AUTH_FAILURE" != coordinate
}
"""
    auth_result = subprocess.run(
        ["bash", "-c", stubs + test_step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "AUTH_TRACE": str(trace),
            "AUTH_FAILURE": failure,
            "AUTH_CLEANUP_FAILS": str(cleanup_fails).lower(),
            "PYTHONPATH": str(REPO_ROOT),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path / "outputs"),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_REPOSITORY": "example/repository",
            "INSTANCE_NAME": "trtmc-gpu-ci-123-2",
            "GPU_TYPE": "test",
            "CI_SHA": "a" * 40,
            "MERGE_SHA": "b" * 40,
            "FAMILIES": '["bert"]',
            "DIRECT_FAMILIES": '["bert"]',
            "ADDED_FAMILIES": "[]",
            "SCOPE": "families",
            "CUDA_ARCHITECTURES": "89",
            "HF_TOKEN": "test-checkpoint-secret",
            "REGISTRY_TOKEN": "test-registry-secret",
            "DEPENDENCY_IMAGES": str(registry_cache).lower(),
            "REGISTRY_USERNAME": "test-reader",
            "HUB_REQUIREMENT": "huggingface-hub==1.33.0",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert auth_result.returncode == (17 if failure == "exit" else 1 if failure else 0)
    events = trace.read_text(encoding="utf-8").splitlines()
    copied = {event.removeprefix("copy ") for event in events if event.startswith("copy ")}
    assert copied == {"trtmc-gpu-ci-123-2"}
    assert len([event for event in events if event.startswith("coordinate ")]) == (
        0 if failure == "copy" else 1
    )
    assert not any(event.startswith(("create ", "delete ")) for event in events)
    for instance in copied:
        cleanup_index = events.index(f"cleanup {instance}")
        assert cleanup_index > events.index(f"copy {instance}")
    assert not list(tmp_path.glob("trtmc-checkpoint-token.*"))
    assert not list(tmp_path.glob("trtmc-registry-token.*"))
    assert any(event.startswith("registry-copy ") for event in events) is (failure != "copy")
    assert "test-checkpoint-secret" not in (
        auth_result.stdout + auth_result.stderr + trace.read_text(encoding="utf-8")
    )
    assert "test-registry-secret" not in (
        auth_result.stdout + auth_result.stderr + trace.read_text(encoding="utf-8")
    )
    assert ("VM teardown is still required" in auth_result.stderr) is cleanup_fails
    assert 'timeout 30s brev exec "$checkpoint_instance"' in test_step["run"]
    assert '--timeout "$((execution_budget + 120))"' in test_step["run"]
    assert test_step["timeout-minutes"] == 240
    assert "--execution-budget" in test_step["run"]
    publish = workflow["jobs"]["publish"]["steps"][0]
    assert publish["env"]["CPU_RESULT"] == "${{ needs.required.result }}"
    assert publish["env"]["GPU_RESULT"] == "${{ needs.provision-and-test.result }}"
    assert publish["env"]["CLEANUP_RESULT"] == "${{ needs.cleanup.result }}"

    for install_step in (
        steps["Install the Brev CLI"],
        cleanup_steps["Install the pinned Brev CLI"],
    ):
        assert install_step["env"] == {
            "BREV_VERSION": "0.6.335",
            "BREV_ARCHIVE_SHA256": (
                "89d778e6f1e5e52495f3e0f10393f1666a1b16d180001b13faec6f906955e6f8"
            ),
        }
        assert "raw.githubusercontent.com" not in install_step["run"]
        assert "sha256sum --check --strict" in install_step["run"]


@pytest.mark.parametrize(
    (
        "changed_paths",
        "expected_scope",
        "expected_families",
        "expected_direct_families",
        "expected_added_families",
    ),
    [
        (("families/bert/model.py",), "families", ["bert"], ["bert"], []),
        (("families/new_family/model.py",), "families", [], [], ["new_family"]),
        (
            ("families/new_family/model.py", "website/docs/new-family.md"),
            "families",
            [],
            [],
            ["new_family"],
        ),
        (
            ("families/new_family/model.py", "families/bert/model.py"),
            "families",
            ["bert"],
            ["bert"],
            ["new_family"],
        ),
        (
            ("families/new_family/model.py", "CMakeLists.txt"),
            "all",
            ["bert", "gpt2"],
            [],
            ["new_family"],
        ),
        (
            ("families/new_family/model.py", "families/bert/model.py", "CMakeLists.txt"),
            "all",
            ["bert", "gpt2"],
            ["bert"],
            ["new_family"],
        ),
        (("README.md",), "docs", [], [], []),
    ],
)
def test_gpu_impact_executes_only_trusted_base_code(
    tmp_path: Path,
    changed_paths: tuple[str, ...],
    expected_scope: str,
    expected_families: list[str],
    expected_direct_families: list[str],
    expected_added_families: list[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for family in ("bert", "gpt2"):
        root = repository / "families" / family
        root.mkdir(parents=True)
        (root / "model.py").write_text("# trusted base\n", encoding="utf-8")
    tools = repository / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (tools / "test_impact.py").write_text(
        (REPO_ROOT / "tools/test_impact.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repository / "README.md").write_text("Trusted documentation\n", encoding="utf-8")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "CI Test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "CI Test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "trusted fixture")
    base = git("rev-parse", "HEAD")
    for changed_path in changed_paths:
        changed = repository / changed_path
        changed.parent.mkdir(parents=True, exist_ok=True)
        changed.write_text("# pull-request content\n", encoding="utf-8")
        git("add", changed_path)
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "untrusted fixture")
    head = git("rev-parse", "HEAD")

    sentinel = tmp_path / "untrusted-code-executed"
    (tools / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
        "raise RuntimeError('untrusted')\n",
        encoding="utf-8",
    )
    git("add", "tools/__init__.py")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "poison fixture")
    poisoned_head = git("rev-parse", "HEAD")
    git("checkout", "--detach", base)

    output = tmp_path / "output"
    script = _workflow_step_script(
        "community-ci.yml", "gpu-authorize", "Resolve the changed model families"
    )
    for revision in (head, poisoned_head):
        output.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=repository,
            env={
                **os.environ,
                "PYTHONPATH": "",
                "BASE_SHA": base,
                "HEAD_SHA": revision,
                "GPU_EXECUTION_ENABLED": "false",
                "MANUAL_GPU_EXECUTION_ENABLED": "false",
                "EVENT_NAME": "pull_request_target",
                "RUNNER_TEMP": str(tmp_path),
                "GITHUB_OUTPUT": str(output),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not sentinel.exists()
        assert git("rev-parse", "HEAD") == base
        summary = json.loads(result.stdout)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        if revision == poisoned_head:
            assert summary["scope"] == "all"
            assert summary["families"] == ["bert", "gpt2"]
        else:
            assert summary["scope"] == expected_scope
            assert summary["families"] == expected_families
        assert json.loads(values["families"]) == summary["families"]
        assert summary["direct_families"] == expected_direct_families
        assert json.loads(values["direct_families"]) == expected_direct_families
        assert json.loads(values["added_families"]) == expected_added_families
        assert values["scope"] == summary["scope"]
        assert values["gpu_enabled"] == "false"
        assert values["run_gpu"] == "false"
        validation = subprocess.run(
            [
                "bash",
                "-c",
                _workflow_step_script(
                    "community-ci.yml", "gpu-authorize", "Validate the GPU selection"
                ),
            ],
            env={**os.environ, **{key.upper(): value for key, value in values.items()}},
            capture_output=True,
            text=True,
            check=False,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr

    output.write_text("", encoding="utf-8")
    manual = subprocess.run(
        ["bash", "-c", script],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONPATH": "",
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "GPU_EXECUTION_ENABLED": "false",
            "MANUAL_GPU_EXECUTION_ENABLED": "true",
            "EVENT_NAME": "workflow_dispatch",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert manual.returncode == 0, manual.stdout + manual.stderr
    manual_summary = json.loads(manual.stdout)
    manual_values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert manual_values["gpu_enabled"] == "true"
    assert manual_values["run_gpu"] == (
        "true" if manual_summary["scope"] in {"all", "families"} else "false"
    )


@pytest.mark.parametrize(
    ("source", "expected_scope", "expected_direct"),
    [
        ("core/runtime/shared.py", "all", []),
        ("core/runtime/shared\nname.py", "all", []),
        ("CMakeLists.txt", "all", []),
        ("families/bert/runtime/helper.py", "families", ["bert"]),
        ("README.md", "families", []),
    ],
)
def test_gpu_impact_retains_both_rename_owners(
    tmp_path: Path, source: str, expected_scope: str, expected_direct: list[str]
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for family in ("bert", "gpt2"):
        root = repository / "families" / family
        root.mkdir(parents=True)
        (root / "model.py").write_text("# trusted model\n")
    tools = repository / "tools"
    tools.mkdir()
    (tools / "__init__.py").touch()
    shutil.copyfile(REPO_ROOT / "tools/test_impact.py", tools / "test_impact.py")
    origin = repository / source
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text("# exact rename content\n")

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=CI Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            cwd=repository,
            text=True,
        ).strip()

    git("init", "-q")
    git("config", "diff.renames", "true")
    git("add", ".")
    git("commit", "-qm", "trusted fixture")
    base = git("rev-parse", "HEAD")
    destination = "families/new_family/model.py"
    (repository / destination).parent.mkdir()
    git("mv", source, destination)
    git("commit", "-qm", "rename fixture")
    head = git("rev-parse", "HEAD")
    assert git("diff", "--name-only", base, head).splitlines() == [destination]
    git("checkout", "--detach", base)
    output = tmp_path / "output"
    script = _workflow_step_script(
        "community-ci.yml", "gpu-authorize", "Resolve the changed model families"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONPATH": "",
            "BASE_SHA": base,
            "HEAD_SHA": head,
            "GPU_EXECUTION_ENABLED": "false",
            "MANUAL_GPU_EXECUTION_ENABLED": "false",
            "EVENT_NAME": "pull_request_target",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert set(summary["changed_files"]) == {source, destination}
    assert summary["scope"] == expected_scope
    assert summary["direct_families"] == expected_direct
    assert summary["families"] == (["bert", "gpt2"] if expected_scope == "all" else expected_direct)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["added_families"]) == ["new_family"]


@pytest.mark.parametrize("provision_exitcode", [0, 1])
def test_gpu_cleanup_can_delete_instance_after_reservation_failure(
    tmp_path: Path,
    provision_exitcode: int,
) -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["provision-and-test"])
    prepared = harness.run("prepare")
    assert prepared.returncode == 0, prepared.stderr
    identity = dict(
        line.split("=", 1) for line in (tmp_path / "prepare-output").read_text().splitlines()
    )
    assert identity["instance_name"] == "trtmc-gpu-ci-123-2"
    assert identity["lease_artifact_name"] == "gpu-ci-lease-123-2"
    assert not harness.events("provision")
    reserved = harness.run(
        "reserve", PROVISION_EXIT=str(provision_exitcode), PROVISION_EMIT_NAME="false"
    )
    assert reserved.returncode == provision_exitcode, reserved.stderr
    # The pre-upload identity survives even if create fails before publishing
    # any reserve-step output. Owner cleanup must use that prepared identity.
    assert not (tmp_path / "reserve-output").exists()
    released = harness.run("release", INSTANCE_NAME=identity["instance_name"])
    assert released.returncode == 0, released.stderr
    assert harness.events("provision") == [
        "-m tools.brev_exec provision --from-intent --instance trtmc-gpu-ci-123-2 "
        "--provider aws --host-ram-gib 64 --disk-gb 500 --min-free-disk-gb 200 "
        f"--lease-file {tmp_path}/trtmc-gpu-ci-lease.json --timeout 2700 --attempts 1",
    ]
    assert harness.events("cleanup") == [
        "-m tools.brev_exec cleanup --instance trtmc-gpu-ci-123-2 "
        f"--lease-file {tmp_path}/trtmc-gpu-ci-lease.json --until-deleted",
    ]


def test_community_premerge_has_independent_lanes_and_public_only_execution():
    control = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    executor = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    dispatch = control["jobs"]["dispatch"]
    assert dispatch["strategy"] == {
        "fail-fast": False,
        "matrix": {"lane": "${{ fromJSON(needs.snapshot.outputs.lanes) }}"},
    }
    assert "matrix.lane" in dispatch["concurrency"]["group"]
    assert "inputs.source_snapshot == ''" in control["jobs"]["snapshot"]["if"]
    assert set(control[True]) == {"pull_request", "pull_request_target", "workflow_dispatch"}
    assert not (REPO_ROOT / ".github/workflows/community-premerge.yml").exists()
    assert "needs.snapshot.result == 'success'" in dispatch["if"]
    assert dispatch["permissions"] == {"actions": "write", "statuses": "write"}
    assert (
        dispatch["continue-on-error"]
        == "${{ matrix.lane == 'dev' && contains(fromJSON(needs.snapshot.outputs.lanes), 'stable') }}"
    )
    assert "actions/checkout" not in json.dumps(control["jobs"]["snapshot"])
    assert "actions/checkout" not in json.dumps(dispatch)
    step = next(step for step in dispatch["steps"] if step.get("id") == "execution")
    assert (
        step["env"]["CI_REF"]
        == "${{ matrix.lane == 'stable' && 'main' || github.ref == 'refs/heads/ci/developer' && 'ci/developer' || vars.TRTMC_COMMUNITY_CI_DEV_REF || 'main' }}"
    )
    assert "sleep" not in step["run"]
    gpu = executor["jobs"]["provision-and-test"]
    test = next(step for step in gpu["steps"] if step.get("id") == "test")
    assert gpu["permissions"] == {"contents": "read", "actions": "read"}
    assert test["env"]["REGISTRY_TOKEN"] == "${{ secrets.TRTMC_COMMUNITY_REGISTRY_READ_TOKEN }}"
    assert "github.token" not in json.dumps(test)
    assert test["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    assert """printf '%s' "$HF_TOKEN" > "$checkpoint_token" """.strip() in test["run"]
    assert "unset HF_TOKEN" in test["run"]
    assert "unset REGISTRY_TOKEN" in test["run"]
    profile = next(step for step in gpu["steps"] if step.get("id") == "host_profile")
    assert '--ci-sha "$GITHUB_SHA"' in profile["run"]
    assert profile["env"]["PROTECTED_DEPENDENCY_CATALOG"] == (
        "${{ secrets.TRTMC_COMMUNITY_DEPENDENCY_CATALOG }}"
    )
    assert "sudo docker build" not in test["run"]
    assert test["run"].index("unset REGISTRY_TOKEN") < test["run"].index(
        "retry_backoff pull_shared_base"
    )
    assert (
        """trap 'rm -f "$checkpoint_token" "$registry_token"; cleanup_checkpoint_token' EXIT"""
        in test["run"]
    )
    assert "install -d -m 0700 $remote_auth" in test["run"]
    assert 'retry brev copy "$checkpoint_token" "$INSTANCE_NAME:$remote_auth/token"' in test["run"]
    assert '--checkpoint-token-file "$remote_auth/token"' in test["run"]
    assert "HF_TOKEN=" not in test["run"]
    assert "git show $CI_SHA:tools/community_gpu_ci.py" in test["run"]
    assert "git fetch --depth 2 origin $MERGE_SHA" in test["run"]
    assert "huggingface-hub==0.36.0" not in test["run"]
    assert "'$HUB_REQUIREMENT'" in test["run"]
    assert test["env"]["HUB_REQUIREMENT"] == "${{ steps.host_profile.outputs.hub_requirement }}"
    assert "--staging-hub-requirement" in profile["run"]
    assert "git show $CI_SHA:tools/community_gpu_images.py" in test["run"]
    assert (
        "/tmp/trtmc-community-stage-venv/bin/python -I /tmp/community_gpu_ci.py "
        "--containers --repository /tmp/model_connect" in test["run"]
    )
    assert gpu["environment"]["name"] == "gpu-ci-dispatch"
    assert gpu["concurrency"]["cancel-in-progress"] is False
    assert executor["concurrency"]["cancel-in-progress"] is False


@pytest.mark.parametrize(
    "dual_run,expected_lanes",
    [
        ("", ["stable"]),
        ("false", ["stable"]),
        ("true", ["stable"]),
        ("TRUE", ["stable"]),
        ("1", ["stable"]),
        ('["stable","dev"]', ["stable"]),
    ],
)
def test_ordinary_pr_uses_only_stable_even_with_legacy_dual_run_variable(
    tmp_path, dual_run, expected_lanes
):
    control = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    snapshot = control["jobs"]["snapshot"]
    selector = next(step for step in snapshot["steps"] if step.get("id") == "lanes")
    assert "DUAL_RUN" not in selector["env"]
    assert snapshot["outputs"]["lanes"] == "${{ steps.lanes.outputs.lanes }}"
    assert control["jobs"]["dispatch"]["strategy"]["matrix"]["lane"] == (
        "${{ fromJSON(needs.snapshot.outputs.lanes) }}"
    )

    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-c", selector["run"]],
        env={
            **os.environ,
            "DUAL_RUN": dual_run,
            "CI_ENTRY_REF": "refs/heads/main",
            "EVENT_NAME": "pull_request_target",
            "REQUESTED_LANE": "stable",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    # This output is the actual job matrix. With the switch off there is no
    # dev job to publish a status, request credentials, or provision a VM.
    assert json.loads(values["lanes"]) == expected_lanes


@pytest.mark.parametrize("fault", ["", "head", "parent", "base-repo", "closed"])
def test_community_trigger_rejects_stale_or_invalid_pr_metadata(tmp_path, fault):
    head, base, merge, tree = (value * 40 for value in "abcd")
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport os,sys\nprint(os.environ['PULL' if any('/pulls/' in arg for arg in sys.argv) else 'MERGE'])\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Capture the exact pull-request snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "EVENT_HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "example/source",
            "PULL": json.dumps(
                {
                    "state": "closed" if fault == "closed" else "open",
                    "base": {
                        "repo": {
                            "full_name": "other/repo" if fault == "base-repo" else "example/source"
                        },
                        "ref": "main",
                    },
                    "head": {"sha": "f" * 40 if fault == "head" else head},
                    "merge_commit_sha": merge,
                }
            ),
            "MERGE": json.dumps(
                {
                    "sha": merge,
                    "parents": [{"sha": base}, {"sha": "f" * 40 if fault == "parent" else head}],
                    "tree": {"sha": tree},
                }
            ),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (not fault), result.stderr
    if fault == "parent":
        assert output.read_text() == f"head_sha={head}\n"
    elif fault:
        assert not output.exists()


@pytest.mark.parametrize(
    "lane,ref,gpu_provider",
    [
        ("stable", "main", "auto"),
        ("dev", "main", "auto"),
        ("dev", "ci/developer", "auto"),
        ("dev", "ci/developer", ""),
        ("stable", "main", "aws"),
        ("stable", "main", "nebius"),
        ("dev", "ci/developer", "aws"),
        ("dev", "ci/developer", "nebius"),
    ],
)
def test_community_lane_dispatch_preserves_snapshot_and_request_identity(
    tmp_path, lane, ref, gpu_provider
):
    fake = tmp_path / "gh"
    fake.write_text(
        """#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
record=Path(os.environ['PAYLOAD'])
if '--input' in args:
    record.write_text(Path(args[args.index('--input')+1]).read_text())
    print(json.dumps({'workflow_run_id':42}))
else:
    assert any('/statuses/' in a for a in args)
"""
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    snapshot = json.dumps(
        {"head_sha": "a" * 40, "base_sha": "b" * 40, "merge_sha": "c" * 40, "source_tree": "d" * 40}
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml",
                "dispatch",
                "Dispatch the selected Community CI implementation",
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "HEAD_SHA": "a" * 40,
            "LANE": lane,
            "CI_REF": ref,
            "CI_ENTRY_BRANCH": "main",
            "AUTOMATIC_GPU": "true",
            "GPU_PROVIDER": gpu_provider,
            "STABLE_RUN_ID": "99" if lane == "stable" else "",
            "STATUS_CONTEXT": (
                "Stable Community GPU CI" if lane == "stable" else "Dev Community CI"
            ),
            "GITHUB_SERVER_URL": "https://github.com",
            "SOURCE_SNAPSHOT": snapshot,
            "GITHUB_REPOSITORY": "example/source",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
            "PAYLOAD": str(tmp_path / "payload"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "payload").read_text())
    assert payload["ref"] == ref
    assert payload["inputs"]["ci_lane"] == lane
    assert payload["inputs"]["source_snapshot"] == snapshot
    assert payload["inputs"]["pr_number"] == "17"
    assert len(payload["inputs"]["request_id"]) == 32
    assert payload["return_run_details"] is True
    if lane == "dev" and gpu_provider in {"aws", "nebius"}:
        assert payload["inputs"]["gpu_provider"] == gpu_provider
    else:
        # Stable/main has not declared this Dev-only input. Automatic and
        # default requests must also retain the original provider selection.
        assert "gpu_provider" not in payload["inputs"]
    assert output.read_text() == f"run_id=42\nci_ref={ref}\n"


@pytest.mark.parametrize(
    "lane,branch,conclusion,expected,wrong_title",
    [
        ("stable", "main", "success", "success", False),
        ("stable", "main", "failure", "failure", False),
        ("dev", "ci/developer", "failure", "failure", False),
        ("dev", "ci/developer", "success", "success", False),
        ("dev", "main", "cancelled", "failure", False),
        ("stable", "ci/developer", "success", None, False),
        ("dev", "ci/developer", "success", None, True),
    ],
)
@pytest.mark.parametrize("queued_first", [False, True])
def test_dispatched_pipeline_observer_never_overwrites_stable_or_dev_results(
    tmp_path, lane, branch, conclusion, expected, wrong_title, queued_first
):
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "calls=Path(os.environ['CALLS'])\n"
        "if any(a.startswith('/repos/') and '/actions/runs/' in a for a in sys.argv):\n"
        " counter=Path(os.environ['COUNTER'])\n"
        " count=int(counter.read_text())+1 if counter.exists() else 1\n"
        " counter.write_text(str(count))\n"
        " data=json.loads(os.environ['RUN'])\n"
        " if os.environ['QUEUED_FIRST']=='true' and count==1:\n"
        "  assert not calls.exists()\n"
        "  data['status']='queued'; data['conclusion']=None\n"
        "  data['display_title']='Community CI'\n"
        " print(json.dumps(data))\n"
        "else: calls.write_text('\\n'.join(sys.argv[1:]))\n"
    )
    fake.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    head, merge = "a" * 40, "b" * 40
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "dispatch", "Publish the complete workflow conclusion"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PIPELINE_RUN_ID": "42",
            "CI_BRANCH": branch,
            "LANE": lane,
            "PR_NUMBER": "17",
            "HEAD_SHA": head,
            "STATUS_CONTEXT": (
                "Stable Community GPU CI" if lane == "stable" else "Dev Community CI"
            ),
            "SOURCE_SNAPSHOT": json.dumps({"merge_sha": merge}),
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "CALLS": str(tmp_path / "calls"),
            "COUNTER": str(tmp_path / "counter"),
            "QUEUED_FIRST": str(queued_first).lower(),
            "RUN": json.dumps(
                {
                    "path": ".github/workflows/community-ci.yml",
                    "event": "workflow_dispatch",
                    "head_branch": branch,
                    "display_title": (
                        "Unexpected run"
                        if wrong_title
                        else f"{lane.title()} Community CI · PR #17 · head {head} · merge {merge}"
                    ),
                    "status": "completed",
                    "conclusion": conclusion,
                }
            ),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (expected == "success"), result.stderr
    assert not (tmp_path / "calls").exists()
    if expected is not None:
        assert (tmp_path / "output").read_text() == "reported=true\n"
        assert int((tmp_path / "counter").read_text()) == (2 if queued_first else 1)


@pytest.mark.parametrize("api_unavailable", [False, True])
@pytest.mark.parametrize("lane,branch", [("stable", "main"), ("dev", "ci/developer")])
def test_observer_timeout_never_finalizes_the_resource_owner(
    tmp_path, api_unavailable, lane, branch
):
    script = r"""
gh() {
  if [[ "$*" == *"/actions/runs/"* ]]; then
    if [ "$API_UNAVAILABLE" = true ]; then return 1; fi
    printf '%s\n' '{"status":"in_progress"}'
  else
    printf '%s\n' "$*" >> "$STATUS_WRITES"
  fi
}
sleep() { SECONDS=$((SECONDS + 18001)); }
""" + _workflow_step_script(
        "community-ci.yml", "dispatch", "Publish the complete workflow conclusion"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "PIPELINE_RUN_ID": "42",
            "CI_BRANCH": branch,
            "LANE": lane,
            "PR_NUMBER": "17",
            "HEAD_SHA": "a" * 40,
            "SOURCE_SNAPSHOT": json.dumps({"merge_sha": "b" * 40}),
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "API_UNAVAILABLE": str(api_unavailable).lower(),
            "STATUS_WRITES": str(tmp_path / "writes"),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert not (tmp_path / "writes").exists()
    assert (tmp_path / "output").read_text() == "reported=true\n"
    assert "inner run will publish after release" in result.stdout
    fallback = _workflow_step_script("community-ci.yml", "dispatch", "Report a failed CI request")
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$*" >> "$STATUS_WRITES"; }\n' + fallback],
        env={**os.environ, "LANE": lane, "STATUS_WRITES": str(tmp_path / "writes")},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0 and not (tmp_path / "writes").exists()


@pytest.mark.parametrize(
    "owner,backstop,gpu,complete,expected",
    [
        ("true", "true", "success", "success", "success"),
        ("true", "true", "failure", "failure", "failure"),
        ("", "true", "failure", "failure", "failure"),
        ("true", "", "failure", "failure", "pending"),
        ("true", "", "success", "success", "pending"),
        ("", "", "failure", "failure", "pending"),
        ("", "", "success", "success", "pending"),
        ("", "", "skipped", "failure", "failure"),
    ],
)
def test_trusted_inner_status_requires_release_before_a_gpu_verdict(
    tmp_path, owner, backstop, gpu, complete, expected, execution_publisher
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    publisher = workflow["jobs"]["publish"]
    assert {"provision-and-test", "cleanup"} <= set(publisher["needs"])
    assert publisher["permissions"] == {"statuses": "write"}
    step = next(
        s for s in publisher["steps"] if s["name"] == "Publish the released workflow conclusion"
    )
    assert "always()" in step["if"] and "workflow_dispatch" in step["if"]
    assert "inputs.ci_lane" not in step["if"]
    assert step["env"]["HEAD_SHA"] == "${{ needs.authorize.outputs.head_sha }}"
    assert step["env"]["CPU_RESULT"] == "${{ needs.required.result }}"
    assert step["env"]["RUN_GPU"] == "${{ needs.gpu-authorize.outputs.run_gpu }}"
    assert step["env"]["GPU_FAILURE_CLASS"] == (
        "${{ needs.provision-and-test.outputs.failure_class }}"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + step["run"]],
        env={
            **execution_publisher,
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success" if gpu != "skipped" else "failure",
            "RUN_GPU": "true" if gpu != "skipped" else "false",
            "GPU_RESULT": gpu,
            "GPU_FAILURE_CLASS": "none" if gpu == "success" else "pr_failure",
            "OWNER_RELEASE_CONFIRMED": owner,
            "BACKSTOP_RELEASE_CONFIRMED": backstop,
            "ORIGINAL_RUN_ATTEMPT": "1",
            "COMPLETE_RESULT": complete,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "42",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    arguments = writes.read_text().splitlines()
    assert f"state={expected}" in arguments
    assert (
        "context=Stable Community GPU CI"
        if execution_publisher["CI_LANE"] == "stable"
        else "context=Dev Community CI"
    ) in arguments
    assert "target_url=https://github.com/example/source/actions/runs/42" in arguments
    if expected == "pending":
        assert "description=Community CI release remains unconfirmed" in arguments


@pytest.mark.parametrize("outcome", ["skipped", "failure", "cancelled", "success", ""])
def test_backstop_requires_explicit_trusted_no_allocation_proof(tmp_path, outcome):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    owner = workflow["jobs"]["provision-and-test"]
    marker = next(s for s in owner["steps"] if s.get("id") == "no_allocation")
    assert marker["if"] == "${{ always() && steps.reserve.outcome == 'skipped' }}"
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    driver = _cleanup_evidence(
        tmp_path,
        [1],
        skipped={1} if outcome == "skipped" else set(),
        artifacts=False,
        reserve_outcome=outcome,
    )
    result = harness.run(
        "release",
        LEASE_DRIVER=str(driver),
        NO_ALLOCATION_CONFIRMED="true",
        OWNER_CLEANUP_CONFIRMED="true",
    )
    # Missing/cancelled owner outputs never count as evidence. Only original
    # GitHub job metadata may prove that this attempt did not reach Reserve.
    assert result.returncode == (0 if outcome == "skipped" else 75), result.stderr
    assert not harness.events("cleanup")
    output = tmp_path / "release-output"
    assert output.exists() is (outcome == "skipped")
    if output.exists():
        assert output.read_text() == "cleanup_confirmed=true\n"
    assert "needs.provision-and-test.outputs" not in str(harness.steps["release"])


@pytest.mark.parametrize("preparation_failure", ["configuration", "external_assets"])
def test_preparation_failure_finalizes_infra_without_brev_credentials(
    tmp_path, execution_publisher, gpu_job, preparation_failure
):
    calls = tmp_path / "brev-calls"
    if preparation_failure == "external_assets":
        from tools.tests.test_community_gpu_plan import commit, external_assets

        repository, catalog, environment = _host_selection_fixture(tmp_path)
        external_assets(repository)
        (repository / "families/alpha/ci/Dockerfile.dependencies").unlink()
        environment["SOURCE_SHA"] = commit(repository)
        environment["PROTECTED_DEPENDENCY_CATALOG"] = json.dumps(catalog)
        environment["BREV_CALLS"] = str(calls)
        profile = next(step for step in gpu_job["steps"] if step.get("id") == "host_profile")
        preparation = subprocess.run(
            [
                "bash",
                "-c",
                'brev() { echo called >> "$BREV_CALLS"; return 91; }\n' + profile["run"],
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert preparation.returncode != 0
        assert "external_files require a regular family-owned ci/Dockerfile.dependencies" in (
            preparation.stderr
        )
        assert not (tmp_path / "trtmc-gpu-plan.json").exists()
        assert not (tmp_path / "host-output").exists() and not calls.exists()
    owner = _workflow_step_script(
        "community-ci.yml", "provision-and-test", "Confirm that reservation was skipped"
    )
    output = tmp_path / "owner-output"
    result = subprocess.run(
        ["bash", "-c", 'brev() { printf "%s\\n" "$*" >> "$BREV_CALLS"; return 91; }\n' + owner],
        env={
            "PATH": os.environ["PATH"],
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "BREV_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0 and not calls.exists()
    assert output.read_text().splitlines() == [
        "no_allocation_confirmed=true",
        "cleanup_confirmed=true",
    ]
    assert "Creation was not requested" in (tmp_path / "summary").read_text()
    backstop = _workflow_step_script(
        "community-ci.yml", "cleanup", "Confirm an initial skipped allocation"
    )
    result = subprocess.run(
        ["bash", "-c", backstop],
        env={
            "PATH": os.environ["PATH"],
            "GITHUB_OUTPUT": str(tmp_path / "backstop-output"),
            "GITHUB_RUN_ATTEMPT": "1",
            "RESERVE_OUTCOME": "skipped",
            "NO_ALLOCATION_CONFIRMED": "true",
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert (tmp_path / "backstop-output").read_text() == "cleanup_confirmed=true\n"
    conclusion = _workflow_step_script(
        "community-ci.yml", "provision-and-test", "Record the step conclusion"
    )
    result = subprocess.run(
        ["bash", "-c", conclusion],
        env={
            **execution_publisher,
            "PATH": os.environ["PATH"],
            "GITHUB_OUTPUT": str(tmp_path / "conclusion"),
            "JOB_STATUS": "failure",
            "TEST_OUTCOME": "skipped",
            "TEST_CONCLUSION": "",
            "CLEANUP_CONFIRMED": "true",
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert (tmp_path / "conclusion").read_text() == "conclusion=failure\n"
    publish = _workflow_step_script(
        "community-ci.yml", "publish", "Publish the released workflow conclusion"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            **execution_publisher,
            "PATH": os.environ["PATH"],
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success",
            "RUN_GPU": "true",
            "GPU_RESULT": "failure",
            "GPU_FAILURE_CLASS": "infra_failure",
            "OWNER_RELEASE_CONFIRMED": "true",
            "BACKSTOP_RELEASE_CONFIRMED": "true",
            "ORIGINAL_RUN_ATTEMPT": "1",
            "NO_ALLOCATION_CONFIRMED": "true",
            "COMPLETE_RESULT": "failure",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "42",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0 and not calls.exists()
    assert "state=error" in writes.read_text().splitlines()
    assert "description=Complete Community CI: infra_failure" in writes.read_text().splitlines()


@pytest.mark.parametrize(
    "receipt,expected_class",
    [
        ("missing", "infra_failure"),
        ("null", "infra_failure"),
        ("malformed", "infra_failure"),
        ("oversized", "infra_failure"),
        ("wrong-schema", "infra_failure"),
        ("pre-entry", "infra_failure"),
        ("interrupted", "infra_failure"),
        ("missing-entry-marker", "infra_failure"),
        ("mixed", "infra_failure"),
        ("false-success", "infra_failure"),
        ("pr-failure", "pr_failure"),
        ("post-entry-oom", "pr_failure"),
        ("success", "none"),
    ],
)
@pytest.mark.parametrize("released", [False, True])
def test_gpu_receipt_classification_reaches_only_a_released_final_status(
    tmp_path, receipt, expected_class, released, execution_publisher
):
    """Run the real workflow scripts, including missing pre-coordinator output."""
    row = {
        "family": "example",
        "status": "failed",
        "phase": "validation",
        "failure_class": "pr_failure",
        "entrypoint_started": True,
        "requested_cases": ["one"],
        "cases": {"one": "failed"},
    }
    report = {"schema_version": 1, "families": [row], "passed": True, "complete": True}
    if receipt == "wrong-schema":
        report["schema_version"] = 2
    elif receipt == "pre-entry":
        row.update(phase="checkpoints", failure_class="infra_failure", entrypoint_started=False)
        row["cases"]["one"] = "not_run"
    elif receipt == "interrupted":
        row.update(status="running", phase="container")
        row["cases"]["one"] = "not_run"
    elif receipt == "missing-entry-marker":
        row.pop("entrypoint_started")
    elif receipt == "mixed":
        report["families"].append(
            {
                **row,
                "family": "other",
                "failure_class": "infra_failure",
                "entrypoint_started": False,
            }
        )
    elif receipt == "post-entry-oom":
        row.update(phase="container", evidence="Docker reported OOMKilled for this container")
        row["cases"]["one"] = "not_run"
    elif receipt in {"success", "false-success"}:
        row.update(status="passed", phase="complete", failure_class=None)
        row["cases"]["one"] = "passed" if receipt == "success" else "not_run"
    path = tmp_path / "summary.json"
    if receipt != "missing":
        payload = {
            "null": "null\n",
            "malformed": "{\n",
            "oversized": " " * (4 * 1024 * 1024 + 1),
        }.get(receipt, json.dumps(report))
        path.write_text(payload)
    output = tmp_path / "output"
    classify = _workflow_step_script(
        "community-ci.yml", "provision-and-test", "Classify the GPU result"
    )
    result = subprocess.run(
        ["bash", "-c", classify],
        cwd=REPO_ROOT,
        env={**os.environ, "SUMMARY_FILE": str(path), "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"failure_class={expected_class}\n"

    publish = _workflow_step_script(
        "community-ci.yml", "publish", "Publish the released workflow conclusion"
    )
    writes = tmp_path / "writes"
    success = receipt in {"success", "false-success"}
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            **execution_publisher,
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success",
            "RUN_GPU": "true",
            "GPU_RESULT": "success" if success else "failure",
            "GPU_FAILURE_CLASS": expected_class,
            "OWNER_RELEASE_CONFIRMED": "true" if released else "",
            "BACKSTOP_RELEASE_CONFIRMED": "true" if released else "",
            "ORIGINAL_RUN_ATTEMPT": "1",
            "NO_ALLOCATION_CONFIRMED": "",
            "COMPLETE_RESULT": "success" if success else "failure",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "42",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    arguments = writes.read_text().splitlines()
    expected_state = {"none": "success", "infra_failure": "error", "pr_failure": "failure"}[
        expected_class
    ]
    if not released:
        assert "state=pending" in arguments
        assert "description=Community CI release remains unconfirmed" in arguments
    else:
        assert f"state={expected_state}" in arguments
        description = "success" if expected_class == "none" else expected_class
        assert f"description=Complete Community CI: {description}" in arguments


@pytest.mark.parametrize("category", ["", "unrecognized", "infra_failure"])
@pytest.mark.parametrize("cpu", ["success", "failure"])
def test_absent_classification_output_defaults_to_infra_only_after_cpu_passes(
    tmp_path, category, cpu, execution_publisher
):
    """A skipped/failed classifier cannot blame the PR or relabel CPU failures."""
    publish = _workflow_step_script(
        "community-ci.yml", "publish", "Publish the released workflow conclusion"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            **execution_publisher,
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": cpu,
            "RUN_GPU": "true",
            "GPU_RESULT": "failure",
            "GPU_FAILURE_CLASS": category,
            "OWNER_RELEASE_CONFIRMED": "true",
            "BACKSTOP_RELEASE_CONFIRMED": "true",
            "ORIGINAL_RUN_ATTEMPT": "1",
            "NO_ALLOCATION_CONFIRMED": "",
            "COMPLETE_RESULT": "failure",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "42",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    arguments = writes.read_text().splitlines()
    expected_state, expected_class = (
        ("error", "infra_failure") if cpu == "success" else ("failure", "failure")
    )
    assert f"state={expected_state}" in arguments
    assert f"description=Complete Community CI: {expected_class}" in arguments


@pytest.mark.parametrize("ready_after", [1, 2, 7])
def test_pr_trigger_waits_for_github_merge_generation_with_a_bounded_retry(tmp_path, ready_after):
    head, base, merge, tree = (value * 40 for value in "abcd")
    fake = tmp_path / "gh"
    fake.write_text(
        """#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
if any('/pulls/' in arg for arg in sys.argv):
    counter=Path(os.environ['COUNTER'])
    attempt=int(counter.read_text())+1 if counter.exists() else 1
    counter.write_text(str(attempt))
    data=json.loads(os.environ['PULL'])
    if attempt < int(os.environ['READY_AFTER']): data['merge_commit_sha']=None
else:
    data=json.loads(os.environ['MERGE'])
print(json.dumps(data))
"""
    )
    fake.chmod(0o755)
    pause = tmp_path / "sleep"
    pause.write_text("#!/bin/sh\nexit 0\n")
    pause.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Capture the exact pull-request snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "EVENT_HEAD_SHA": head,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "example/source",
            "READY_AFTER": str(ready_after),
            "COUNTER": str(tmp_path / "counter"),
            "PULL": json.dumps(
                {
                    "state": "open",
                    "head": {"sha": head},
                    "base": {"ref": "main", "repo": {"full_name": "example/source"}},
                    "merge_commit_sha": merge,
                }
            ),
            "MERGE": json.dumps(
                {"sha": merge, "parents": [{"sha": base}, {"sha": head}], "tree": {"sha": tree}}
            ),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (ready_after <= 6), result.stderr
    assert int((tmp_path / "counter").read_text()) == min(6, ready_after)
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["head_sha"] == head
    assert ("source_snapshot" in values) is (ready_after <= 6)


@pytest.mark.parametrize("fault", ["", "head", "base", "tree", "lane", "stable-ref"])
def test_community_executor_keeps_the_captured_snapshot_when_merge_ref_advances(tmp_path, fault):
    head, base, merge, tree = (value * 40 for value in "abcd")
    snapshot = {"head_sha": head, "base_sha": base, "merge_sha": merge, "source_tree": tree}
    if fault in {"base", "tree"}:
        snapshot["base_sha" if fault == "base" else "source_tree"] = "f" * 40
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport os,sys\n"
        "key='PULL' if any('/pulls/' in arg for arg in sys.argv) else 'MERGE'\n"
        "print(os.environ[key])\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "authorize", "Capture the exact pull-request snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "EVENT_NAME": "workflow_dispatch",
            "ACTOR": "github-actions[bot]",
            "CI_LANE": "unknown" if fault == "lane" else "stable",
            "CI_REF": "refs/heads/ci/developer" if fault == "stable-ref" else "refs/heads/main",
            "PR_NUMBER": "17",
            "REQUEST_ID": "1" * 32,
            "SOURCE_SNAPSHOT": json.dumps(snapshot),
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(output),
            "PULL": json.dumps(
                {
                    "state": "open",
                    "base": {"repo": {"full_name": "example/source"}, "ref": "main"},
                    "head": {"sha": "f" * 40 if fault == "head" else head},
                    "merge_commit_sha": "e" * 40,
                }
            ),
            "MERGE": json.dumps(
                {"sha": merge, "parents": [{"sha": base}, {"sha": head}], "tree": {"sha": tree}}
            ),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (not fault), result.stderr
    if not fault:
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert values == {
            "enabled": "true",
            "reuse_stable_cpu": "false",
            "pr_number": "17",
            "head_sha": head,
            "base_sha": base,
            "merge_sha": merge,
        }
    else:
        assert not output.exists()


@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("base_gpu_enabled", [False, True])
def test_stable_pr_cpu_switches_only_after_promotion_is_in_the_base(
    tmp_path, promoted, base_gpu_enabled
):
    head, base, merge = (value * 40 for value in "abc")
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport base64,json,os,sys\n"
        "if any('/contents/' in arg for arg in sys.argv):\n"
        " text='# Community CI GPU promotion v1' if os.environ['PROMOTED']=='true' else 'name: Community CI'\n"
        " text += '\\n  COMMUNITY_GPU_EXECUTION_ENABLED: ' + json.dumps(os.environ['BASE_GPU_ENABLED'])\n"
        " text += '\\n# A quoted policy example must not enable GPU: COMMUNITY_GPU_EXECUTION_ENABLED: \"true\"'\n"
        " print(base64.b64encode(text.encode()).decode())\n"
        "else: print(os.environ['PULL' if any('/pulls/' in arg for arg in sys.argv) else 'MERGE'])\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "authorize", "Capture the exact pull-request snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "EVENT_NAME": "pull_request",
            "PROMOTED": str(promoted).lower(),
            "BASE_GPU_ENABLED": str(base_gpu_enabled).lower(),
            "EVENT_HEAD_SHA": head,
            "EVENT_BASE_SHA": base,
            "EVENT_MERGE_SHA": merge,
            "PR_NUMBER": "17",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(output),
            "PULL": json.dumps(
                {
                    "state": "open",
                    "head": {"sha": head},
                    "base": {"ref": "main", "repo": {"full_name": "example/source"}},
                }
            ),
            "MERGE": json.dumps({"sha": merge, "parents": [{"sha": base}, {"sha": head}]}),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {
        "enabled": "true",
        "reuse_stable_cpu": str(promoted and base_gpu_enabled).lower(),
        "pr_number": "17",
        "head_sha": head,
        "base_sha": base,
        "merge_sha": merge,
    }


@pytest.mark.parametrize("head", ["", "invalid", "a" * 40])
def test_failed_snapshot_reports_only_a_validated_head(tmp_path, head):
    fake = tmp_path / "gh"
    fake.write_text('#!/bin/bash\nprintf "%s\n" "$@" > "$CALLS"\n')
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script("community-ci.yml", "snapshot", "Report a failed snapshot"),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "HEAD_SHA": head,
            "STATUS_CONTEXT": "Stable Community GPU CI",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_RUN_ID": "42",
            "CALLS": str(calls),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if len(head) == 40:
        arguments = calls.read_text().splitlines()
        assert f"/repos/example/source/statuses/{head}" in arguments
        assert "state=failure" in arguments
        assert "context=Stable Community GPU CI" in arguments
    else:
        assert not calls.exists()


@pytest.mark.parametrize("available", [True, False])
def test_pairing_selects_only_the_existing_stable_pr_run(tmp_path, available):
    head, merge = "a" * 40, "b" * 40
    valid = {
        "id": 42,
        "event": "pull_request",
        "head_sha": head,
        "path": ".github/workflows/community-ci.yml",
        "display_title": f"PR #17 · community CI · head {head} · merge {merge}",
    }
    candidates = [
        {**valid, "id": 90, "event": "workflow_dispatch"},
        {**valid, "id": 91, "head_sha": "c" * 40},
        {**valid, "id": 92, "path": ".github/workflows/unrelated.yml"},
        {**valid, "id": 93, "display_title": "PR #18 · unrelated run"},
    ]
    if available:
        candidates.extend([{**valid, "id": 41}, valid])
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport os,sys\n"
        "print(os.environ['PULL' if any('/pulls/' in a for a in sys.argv) else 'RUNS'])\n"
    )
    gh.chmod(0o755)
    pause = tmp_path / "sleep"
    pause.write_text("#!/bin/sh\nexit 0\n")
    pause.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Find the existing Stable PR snapshot"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "EVENT_HEAD_SHA": head,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "PULL": json.dumps({"head": {"sha": head}}),
            "RUNS": json.dumps({"workflow_runs": candidates}),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is available, result.stderr
    if available:
        assert (tmp_path / "output").read_text() == f"run_id=42\nmerge_sha={merge}\n"
    else:
        assert "No Stable PR snapshot" in result.stdout
        assert not (tmp_path / "output").exists()


def test_stable_pairing_does_not_dispatch_or_repeat_the_existing_pipeline(tmp_path):
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\necho 'unexpected API mutation' >&2\nexit 99\n")
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "dispatch", "Dispatch the selected Community CI implementation"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "LANE": "stable",
            "CI_REF": "main",
            "STABLE_RUN_ID": "42",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output").read_text() == "run_id=42\nci_ref=main\nexisting_stable=true\n"


@pytest.mark.parametrize("fault", ["", "head", "merge", "event", "conclusion"])
def test_existing_stable_verdict_is_bound_to_the_selected_head_and_merge(tmp_path, fault):
    head, merge = "a" * 40, "b" * 40
    run = {
        "path": ".github/workflows/community-ci.yml",
        "event": "workflow_dispatch" if fault == "event" else "pull_request",
        "head_sha": "c" * 40 if fault == "head" else head,
        "display_title": f"PR #17 · community CI · head {head} · merge {('c' * 40) if fault == 'merge' else merge}",
        "status": "completed",
        "conclusion": "failure" if fault == "conclusion" else "success",
    }
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n"
        "if any(a.startswith('/repos/') and '/actions/runs/' in a for a in sys.argv): print(os.environ['RUN'])\n"
        "else: Path(os.environ['CALLS']).write_text('\\n'.join(sys.argv))\n"
    )
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "dispatch", "Publish the complete workflow conclusion"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RUN": json.dumps(run),
            "CALLS": str(tmp_path / "calls"),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "PIPELINE_RUN_ID": "42",
            "EXISTING_STABLE": "true",
            "LANE": "stable",
            "CI_BRANCH": "main",
            "PR_NUMBER": "17",
            "HEAD_SHA": head,
            "SOURCE_SNAPSHOT": json.dumps({"merge_sha": merge}),
            "STATUS_CONTEXT": "Stable Community GPU CI",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_REPOSITORY": "example/source",
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (fault == ""), result.stderr
    if fault in {"head", "merge", "event"}:
        assert not (tmp_path / "calls").exists()
    else:
        state = "failure" if fault == "conclusion" else "success"
        assert f"state={state}" in (tmp_path / "calls").read_text().splitlines()


@pytest.mark.parametrize("cpu_result", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize("queued_first", [False, True])
def test_promoted_cpu_compatibility_uses_the_actual_stable_result(
    tmp_path, cpu_result, queued_first
):
    head, merge = "a" * 40, "b" * 40
    valid = {
        "id": 42,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "path": ".github/workflows/community-ci.yml",
        "display_title": f"Stable Community CI · PR #17 · head {head} · merge {merge}",
    }
    runs = [
        valid,
        {**valid, "id": 91, "head_branch": "ci/developer"},
        {**valid, "id": 92, "event": "pull_request"},
        {**valid, "id": 93, "display_title": valid["display_title"].replace(head, "c" * 40)},
        {**valid, "id": 94, "display_title": valid["display_title"].replace(merge, "c" * 40)},
    ]
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "assert '--method' not in sys.argv\n"
        "if any('/workflows/' in a for a in sys.argv): print(os.environ['RUNS'])\n"
        "else:\n"
        " assert any('/actions/runs/42/jobs?' in a for a in sys.argv)\n"
        " counter=Path(os.environ['COUNTER'])\n"
        " count=int(counter.read_text())+1 if counter.exists() else 1\n"
        " counter.write_text(str(count))\n"
        " pending=os.environ['QUEUED_FIRST']=='true' and count==1\n"
        " print(json.dumps({'status':'in_progress' if pending else 'completed', 'conclusion':None if pending else os.environ['CPU_RESULT']}))\n"
    )
    gh.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "required", "Read the promoted Stable CPU result"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RUNS": json.dumps({"workflow_runs": runs}),
            "COUNTER": str(tmp_path / "counter"),
            "QUEUED_FIRST": str(queued_first).lower(),
            "CPU_RESULT": cpu_result,
            "PR_NUMBER": "17",
            "HEAD_SHA": head,
            "MERGE_SHA": merge,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (cpu_result == "success"), result.stderr
    assert int((tmp_path / "counter").read_text()) == (2 if queued_first else 1)
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    for name in ["source-quality", "docs", "ownership-impact", "unit"]:
        assert (
            workflow["jobs"][name]["if"]
            == "${{ needs.authorize.outputs.reuse_stable_cpu != 'true' }}"
        )
    assert (
        workflow["jobs"]["required"]["timeout-minutes"]
        > workflow["jobs"]["unit"]["timeout-minutes"]
    )
    steps = {step["name"]: step for step in workflow["jobs"]["required"]["steps"]}
    assert (
        steps["Read the promoted Stable CPU result"]["if"]
        == "${{ needs.authorize.outputs.reuse_stable_cpu == 'true' }}"
    )
    assert (
        steps["Require every public CPU stage"]["if"]
        == "${{ needs.authorize.outputs.reuse_stable_cpu != 'true' }}"
    )


def _workflow_plan(directory):
    path = directory / "trtmc-gpu-plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha": "b" * 40,
                "source_tree": "c" * 40,
                "selection": {
                    "scope": "families",
                    "selected_families": ["bert"],
                    "families": ["bert"],
                    "direct_families": ["bert"],
                    "added_families": [],
                },
                "active_families": ["bert"],
                "families": [
                    {"family": "bert", "cases": ["bert"], "deferred_cases": [], "checkpoints": []}
                ],
            }
        )
    )
    path.chmod(0o600)
    return path


# Execute the checked-in shell scripts. These functions replace only external
# provider and coordinator operations, and preserve their real shell exit codes.
STUBS = r"""
sleep() { :; }
timeout() { shift; "$@"; }
brev() {
  printf 'brev\t%s\n' "$*" >> "$TRACE"
  case "$1" in
    copy)
      test -z "${HF_TOKEN+x}" || return 98
      test "$(stat -c '%a' "$2")" = 600 || return 97
      if [[ "$3" == */dependency-catalog.json ]]; then
        command python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["families"] == {}' "$2" || return 96
        printf 'catalog-private\t%s\n' "${3%%:*}" >> "$TRACE"
      elif [[ "$3" == */gpu-plan.json ]]; then
        command python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["active_families"] == ["bert"]' "$2" || return 96
        printf 'plan-private\t%s\n' "${3%%:*}" >> "$TRACE"
      elif [[ "$3" == */registry-token ]]; then
        test "$(cat "$2")" = "$EXPECTED_REGISTRY_TOKEN" || return 96
        REGISTRY_COPIED=1
        printf 'registry-private\t%s\n' "${3%%:*}" >> "$TRACE"
      else
        test "$(cat "$2")" = "$EXPECTED_TOKEN" || return 96
        printf 'token-private\t%s\n' "${3%%:*}" >> "$TRACE"
      fi
      test "$COPY_FAILS" != true || return 9
      ;;
    exec)
      if [[ "$3" == "rm -f -- "* ]]; then
        printf 'token-cleanup\t%s\n' "$2" >> "$TRACE"
        test "$CLEANUP_FAILS" != true || return 8
      elif [[ "$3" == *"--pull-base"* ]]; then
        test "${REGISTRY_COPIED:-0}" = 1 || return 96
        REGISTRY_COPIED=0
        printf 'base-pull\t%s\n' "$2" >> "$TRACE"
        return "${BASE_PULL_EXIT:-0}"
      elif [[ "$3" == "git init "* && -n "${LOCAL_GIT_SETUP:-}" ]]; then
        bash -c "$LOCAL_GIT_SETUP"
        return "$?"
      elif [[ "$3" == "git init "* && "$FETCH_FAILS_ONCE" == true ]]; then
        if [ ! -f "$FETCH_ATTEMPT" ]; then
          touch "$FETCH_ATTEMPT"
          return 7
        fi
      fi
      ;;
    create|delete) return 0 ;;
    *) return 95 ;;
  esac
}
python3() {
  if [ "$1" = - ]; then command python3 "$@"; return "$?"; fi
  test "$1" = -m || return 94
  case "$2:$3" in
    tools.brev_exec:provision)
      if [ "$4" = --prepare-only ]; then
        printf 'prepare\t%s\n' "$*" >> "$TRACE"
        printf '{"name":"%s","create_started":true,"allocation_pending":true}\n' "$RESERVED_INSTANCE" > "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"
        return "${PREPARE_EXIT:-0}"
      fi
      printf 'provision\t%s\n' "$*" >> "$TRACE"
      if [ "${PROVISION_EMIT_NAME:-true}" = true ]; then
        printf 'instance_name=%s\n' "$RESERVED_INSTANCE" >> "$GITHUB_OUTPUT"
      fi
      printf '{"name":"%s","instance_id":"test123"}\n' "$RESERVED_INSTANCE" > "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"
      return "$PROVISION_EXIT"
      ;;
    tools.community_gpu_lease:*)
      if [ -n "${LEASE_DRIVER:-}" ]; then
        command python3 "$LEASE_DRIVER" "${@:3}"
        return "$?"
      fi
      printf 'reconcile\t%s\n' "$*" >> "$TRACE"
      test "$VM_CLEANUP_EXIT" = 0 || return "$VM_CLEANUP_EXIT"
      printf 'cleanup_confirmed=true\n' >> "$GITHUB_OUTPUT"
      ;;
    tools.brev_exec:cleanup)
      printf 'cleanup\t%s\n' "$*" >> "$TRACE"
      return "$VM_CLEANUP_EXIT"
      ;;
    tools.brev_exec:*)
      test -z "${HF_TOKEN+x}" || return 93
      test "${REGISTRY_COPIED:-0}" = 0 || return 96
      printf 'coordinate\t%s\n' "$*" >> "$TRACE"
      return "$COORDINATOR_EXIT"
      ;;
    *) return 92 ;;
  esac
}
"""


@pytest.fixture
def gpu_job():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    return workflow["jobs"]["provision-and-test"]


class JobHarness:
    def __init__(self, tmp_path: Path, job: dict):
        self.directory = tmp_path
        self.steps = {step.get("id", step["name"]): step for step in job["steps"]}
        self.steps.update({step["name"]: step for step in job["steps"]})
        self.trace = tmp_path / "trace"
        catalog = tmp_path / "trtmc-gpu-host-profiles.json"
        if not catalog.exists():
            catalog.write_text('{"families":{}}')
        catalog.chmod(0o600)
        _workflow_plan(tmp_path)
        self.environment = {
            **os.environ,
            "TRACE": str(self.trace),
            "FETCH_ATTEMPT": str(tmp_path / "fetch-attempt"),
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_REPOSITORY": "example/repository",
            "GITHUB_SHA": "a" * 40,
            "GPU_TYPE": "L40S",
            "GPU_PROVIDER": "aws",
            "HOST_RAM_GIB": "64",
            "RESERVED_INSTANCE": "trtmc-gpu-ci-123-2",
            "INSTANCE_NAME": "trtmc-gpu-ci-123-2",
            "PROVISION_EXIT": "0",
            "COORDINATOR_EXIT": "0",
            "VM_CLEANUP_EXIT": "0",
            "COPY_FAILS": "false",
            "CLEANUP_FAILS": "false",
            "FETCH_FAILS_ONCE": "false",
            "CI_SHA": "a" * 40,
            "MERGE_SHA": "b" * 40,
            "FAMILIES": '["bert"]',
            "DIRECT_FAMILIES": '["bert"]',
            "ADDED_FAMILIES": "[]",
            "SCOPE": "families",
            "CUDA_ARCHITECTURES": "89",
            "HF_TOKEN": "workflow-test-checkpoint-token",
            "EXPECTED_TOKEN": "workflow-test-checkpoint-token",
            "REGISTRY_TOKEN": "workflow-test-registry-token",
            "EXPECTED_REGISTRY_TOKEN": "workflow-test-registry-token",
            "REGISTRY_USERNAME": "test-reader",
            "HUB_REQUIREMENT": "huggingface-hub==1.33.0",
        }

    def run(self, step: str, **updates: str) -> subprocess.CompletedProcess:
        output = self.directory / f"{step.replace(' ', '-')}-output"
        return subprocess.run(
            ["bash", "-c", STUBS + self.steps[step]["run"]],
            cwd=REPO_ROOT,
            env={**self.environment, "GITHUB_OUTPUT": str(output), **updates},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def events(self, kind: str) -> list[str]:
        if not self.trace.exists():
            return []
        prefix = f"{kind}\t"
        return [
            line.removeprefix(prefix)
            for line in self.trace.read_text().splitlines()
            if line.startswith(prefix)
        ]

    def reserve_output(self) -> str:
        values = dict(
            line.split("=", 1)
            for line in (self.directory / "reserve-output").read_text().splitlines()
        )
        return values["instance_name"]


def _host_selection_fixture(tmp_path, profile=64):
    """Committed common inputs plus hostile uncommitted PR image overrides."""
    from tools import community_gpu_images as images
    from tools.tests.test_community_gpu_ci import _dependency_entry
    from tools.tests.test_community_gpu_plan import owner

    repository = tmp_path / "source"
    repository.mkdir()
    entry = _dependency_entry(repository, "alpha")
    prefix = "ghcr.io/test-owner/private-dependencies"
    entry["lock"]["image"] = f"{prefix}/alpha@sha256:{'a' * 64}"
    entry["lock"]["resources"] = {"host_ram_gib": 64 if profile == 64 else 128}
    if profile != 64:
        entry["lock"]["qualification_host"] = {
            "ram_gib": 64 if profile == "invalid" else 128,
            "gpu_count": 1,
            "arch": "x86_64",
            "run_id": "12345",
        }
    lock = repository / "families/alpha/ci/dependency-image.json"
    lock.write_text(json.dumps(entry["lock"]))
    owner(repository)
    for relative in images.BASE_INPUTS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("huggingface-hub==1.33.0\n" if relative.endswith(".lock") else "{}\n")
    inputs = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in images.BASE_INPUTS
    }

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.test",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "trusted host inputs",
    )
    ci_sha = git("rev-parse", "HEAD")
    git("remote", "add", "origin", str(repository))
    catalog = {
        "schema_version": 1,
        "ci_sha": "ignored",
        "registry_prefix": prefix,
        "families": {"alpha": entry},
        "base": {
            "schema_version": 1,
            "kind": "community-base",
            "platform": "linux/amd64",
            "registry_visibility": "private",
            "image": prefix + "/base@sha256:" + "c" * 64,
            "environment_source_sha": ci_sha,
            "inputs": inputs,
            "input_key": images.base_key(inputs),
            "cpu_environment_verified": True,
            "local_image_id": "sha256:" + "d" * 64,
        },
    }
    lock.write_text('{"resources":{"host_ram_gib":256},"image":"attacker:latest"}')
    # The actual trusted Python admission runs in a subprocess. Supply a fixed
    # identity/package responses at its HTTP boundary; parsers have unit coverage.
    http_fixture = tmp_path / "identity-http-fixture"
    http_fixture.mkdir()
    (http_fixture / "sitecustomize.py").write_text(
        "import io, json, os, urllib.request, urllib.error\n"
        "from types import SimpleNamespace\n"
        "def identity(request, timeout):\n"
        "    assert request.get_header('Authorization') == 'Bearer dedicated-test-reader-token'\n"
        "    assert timeout > 0\n"
        "    if request.full_url == 'https://api.github.com/orgs/test-owner/packages/container/private-dependencies%2Fbase':\n"
        "        if os.environ.get('PACKAGE_FIXTURE_MODE') == 'denied':\n"
        "            raise urllib.error.HTTPError(request.full_url, 403, 'Forbidden', {}, io.BytesIO(b'{}'))\n"
        "        response = io.BytesIO(json.dumps({'name':'private-dependencies/base','package_type':'container','visibility':'private'}).encode())\n"
        "        response.status = 200\n"
        "        return response\n"
        "    assert request.full_url == 'https://api.github.com/user'\n"
        "    if os.environ.get('IDENTITY_FIXTURE_MODE') == 'error':\n"
        "        raise OSError('dedicated-test-reader-token upstream error')\n"
        "    login = 'test-reader' if os.environ.get('IDENTITY_FIXTURE_MODE') != 'invalid' else 'invalid\\nlogin'\n"
        "    response = io.BytesIO(json.dumps({'login': login}).encode())\n"
        "    response.status = 200\n"
        "    return response\n"
        "urllib.request.build_opener = lambda *handlers: SimpleNamespace(open=identity)\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(http_fixture) + os.pathsep + str(REPO_ROOT),
        "GITHUB_WORKSPACE": str(repository),
        "GITHUB_SHA": ci_sha,
        "SOURCE_SHA": ci_sha,
        "GITHUB_OUTPUT": str(tmp_path / "host-output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "host-summary"),
        "RUNNER_TEMP": str(tmp_path),
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
        "REGISTRY_TOKEN": "dedicated-test-reader-token",
    }
    return repository, catalog, environment


def test_oversized_pr_is_rejected_by_the_actual_preallocation_step(tmp_path, gpu_job):
    repository, catalog, environment = _host_selection_fixture(tmp_path)
    families = ["alpha", "beta", "delta", "epsilon", "gamma", "zeta"]
    environment.update(
        TRTMC_GPU_FAMILIES=json.dumps(families),
        TRTMC_GPU_DIRECT_FAMILIES=json.dumps(families),
        PROTECTED_DEPENDENCY_CATALOG=json.dumps(catalog),
    )
    calls = tmp_path / "brev-calls"
    environment["BREV_CALLS"] = str(calls)
    step = next(item for item in gpu_job["steps"] if item.get("id") == "host_profile")
    result = subprocess.run(
        ["bash", "-c", 'brev() { echo called >> "$BREV_CALLS"; return 91; }\n' + step["run"]],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "split" in result.stderr.lower()
    assert "GPU plan rejected before allocation" in (tmp_path / "host-summary").read_text()
    assert not calls.exists()
    assert not (tmp_path / "host-output").exists()
    assert not (tmp_path / "trtmc-gpu-plan.json").exists()


@pytest.mark.parametrize("profile", [64, 128, "invalid"])
@pytest.mark.parametrize("protected", [False, True])
def test_trusted_host_selection_step_runs_before_any_allocation(
    tmp_path, gpu_job, profile, protected
):
    """Execute actual Git-object planning and protected admission before allocation."""
    repository, catalog, environment = _host_selection_fixture(tmp_path, profile)
    secret = json.dumps(catalog) if protected else ""
    environment["PROTECTED_DEPENDENCY_CATALOG"] = secret
    step = next(step for step in gpu_job["steps"] if step.get("id") == "host_profile")
    assert step.get("continue-on-error", False) is False
    assert "REGISTRY_USERNAME" not in step["env"]
    steps = gpu_job["steps"]
    execution = next(item for item in steps if item.get("id") == "test")
    assert execution["env"]["REGISTRY_USERNAME"] == (
        "${{ steps.host_profile.outputs.registry_username }}"
    )
    assert steps.index(step) < next(
        i for i, item in enumerate(steps) if item["name"] == "Install the Brev CLI"
    )
    output = tmp_path / "host-output"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert secret == "" or secret not in result.stdout + result.stderr
    assert environment["REGISTRY_TOKEN"] not in result.stdout + result.stderr
    assert not list(tmp_path.glob("trtmc-protected-dependency-catalog.*"))
    harness = JobHarness(tmp_path, gpu_job)
    if profile == "invalid" or not protected:
        assert result.returncode != 0 and not output.exists() and not harness.events("provision")
        if not protected:
            assert "Configure the protected shared base catalog" in result.stdout
        return
    assert result.returncode == 0, result.stdout + result.stderr
    exported = tmp_path / "trtmc-gpu-host-profiles.json"
    assert exported.stat().st_mode & 0o777 == 0o600
    assert json.loads(exported.read_text())["ci_sha"] == environment["GITHUB_SHA"]
    assert output.read_text().splitlines() == [
        "registry_username=test-reader",
        f"host_ram_gib={profile}",
        "hub_requirement=huggingface-hub==1.33.0",
        "dependency_images=true",
    ]
    reserved = harness.run("reserve", HOST_RAM_GIB=str(profile))
    assert (
        reserved.returncode == 0 and f"--host-ram-gib {profile}" in harness.events("provision")[0]
    )


@pytest.mark.parametrize(
    "fault",
    ["registry", "base", "token", "identity", "invalid_identity", "base_inputs", "package_access"],
)
def test_invalid_protected_catalog_stops_the_real_preallocation_step(tmp_path, gpu_job, fault):
    repository, catalog, environment = _host_selection_fixture(tmp_path)
    if fault == "registry":
        catalog["registry_prefix"] = "credential-sensitive-prefix"
    elif fault == "base":
        del catalog["base"]
    elif fault in {"identity", "invalid_identity"}:
        environment["IDENTITY_FIXTURE_MODE"] = "error" if fault == "identity" else "invalid"
    elif fault == "package_access":
        environment["PACKAGE_FIXTURE_MODE"] = "denied"
    elif fault == "base_inputs":
        from tools import community_gpu_images as images

        catalog["base"]["inputs"][images.BASE_INPUTS[0]] = "0" * 64
        catalog["base"]["input_key"] = images.base_key(catalog["base"]["inputs"])
    else:
        environment["REGISTRY_" + fault.upper()] = ""
    secret = json.dumps(catalog)
    environment["PROTECTED_DEPENDENCY_CATALOG"] = secret
    step = next(step for step in gpu_job["steps"] if step.get("id") == "host_profile")
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and not (tmp_path / "host-output").exists()
    assert "credential-sensitive-prefix" not in result.stdout + result.stderr
    assert secret not in result.stdout + result.stderr
    assert "dedicated-test-reader-token" not in result.stdout + result.stderr
    assert not list(tmp_path.glob("trtmc-protected-dependency-catalog.*"))
    if fault == "package_access":
        assert "HTTP 403" in result.stderr


def test_failed_blocking_reservation_stops_normal_test_admission(tmp_path, gpu_job):
    harness = JobHarness(tmp_path, gpu_job)
    reserve, test_step = harness.steps["reserve"], harness.steps["test"]
    # GitHub gives ordinary steps an implicit success() condition. A failed
    # readiness helper must not be hidden or overridden by an always() test.
    assert reserve.get("continue-on-error", False) is False
    assert test_step.get("continue-on-error", False) is False
    assert test_step.get("if") in (None, "success()", "${{ success() }}")
    assert test_step["env"]["INSTANCE_NAME"] == "${{ steps.reserve.outputs.instance_name }}"
    result = harness.run("reserve", PROVISION_EXIT="23")
    if result.returncode == 0:
        harness.run("test", INSTANCE_NAME=harness.reserve_output())
    assert result.returncode == 23, result.stdout + result.stderr
    provisioned = harness.events("provision")
    assert len(provisioned) == 1
    assert "--instance trtmc-gpu-ci-123-2" in provisioned[0]
    assert "--provider aws --host-ram-gib 64" in provisioned[0]
    assert "--attempts 1" in provisioned[0]
    assert not harness.events("coordinate")
    assert not harness.events("brev")
    # Even an unsuccessful helper leaves the actual attempted name available
    # for the job's always-run teardown.
    cleanup = harness.steps["release"]
    assert cleanup["env"]["INSTANCE_NAME"] == "${{ steps.prepare.outputs.instance_name }}"
    assert "always()" in cleanup["if"]
    result = harness.run("release", INSTANCE_NAME=harness.reserve_output())
    assert result.returncode == 0
    assert not harness.events("brev")
    assert len(harness.events("cleanup")) == 1
    assert "--instance trtmc-gpu-ci-123-2" in harness.events("cleanup")[0]


@pytest.mark.parametrize(
    "provider,ram,expected",
    [
        ("auto", "64", "aws"),
        ("auto", "128", "aws"),
        ("aws", "64", "aws"),
        ("aws", "128", "aws"),
        ("nebius", "64", "nebius"),
        ("nebius", "128", "nebius"),
        ("AWS", "64", False),
        ("aws", "96", False),
        ("aws", "", False),
        ("aws;$(printf unexpected-provider-command)", "128", False),
    ],
)
def test_gpu_provider_selects_one_qualified_exact_type(tmp_path, gpu_job, provider, ram, expected):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    choice = workflow[True]["workflow_dispatch"]["inputs"]["gpu_provider"]
    assert choice["type"] == "choice"
    assert choice["default"] == "auto"
    assert choice["options"] == ["auto", "aws", "nebius"]
    reserve = next(step for step in gpu_job["steps"] if step.get("id") == "prepare")
    assert reserve["env"]["HOST_RAM_GIB"] == "${{ steps.host_profile.outputs.host_ram_gib }}"
    assert reserve["env"]["GPU_PROVIDER"] == "${{ inputs.gpu_provider || 'auto' }}"
    arguments = tmp_path / "helper-arguments"
    script = r"""
python3() {
  printf '%s\0' "$@" > "$HELPER_ARGUMENTS"
}
""" + reserve["run"]
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "HELPER_ARGUMENTS": str(arguments),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "RUNNER_TEMP": str(tmp_path),
            "GPU_TYPE": "L40S",
            "GPU_PROVIDER": provider,
            "HOST_RAM_GIB": ram,
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    if expected is False:
        assert result.returncode != 0
        assert not arguments.exists()
        assert "unexpected-provider-command" not in result.stdout
        return
    assert result.returncode == 0, result.stdout + result.stderr
    argv = arguments.read_bytes().decode().split("\0")[:-1]
    assert argv[:4] == ["-m", "tools.brev_exec", "provision", "--prepare-only"]
    assert "--gpu" not in argv and "--instance-type" not in argv
    assert "--fallback-provider" not in argv
    assert argv.count("--provider") == 1
    assert argv[argv.index("--provider") + 1] == expected
    assert argv[argv.index("--host-ram-gib") + 1] == ram
    assert argv[argv.index("--disk-gb") + 1] == "500"
    assert argv[argv.index("--min-free-disk-gb") + 1] == "200"
    assert argv[argv.index("--lease-file") + 1] == str(tmp_path / "trtmc-gpu-ci-lease.json")
    assert argv[argv.index("--timeout") + 1] == "2700"
    assert argv[argv.index("--attempts") + 1] == "1"
    assert "--recover-nebius-start-limit" not in argv
    assert argv[argv.index("--instance") + 1] == "trtmc-gpu-ci-123-2"


@pytest.mark.parametrize("coordinator_exit", [0, 17])
def test_gpu_coordinator_runs_once_on_the_ready_instance(tmp_path, gpu_job, coordinator_exit):
    harness = JobHarness(tmp_path, gpu_job)
    reserve = harness.run("reserve")
    assert reserve.returncode == 0, reserve.stdout + reserve.stderr
    instance = harness.reserve_output()
    result = harness.run("test", INSTANCE_NAME=instance, COORDINATOR_EXIT=str(coordinator_exit))
    assert result.returncode == coordinator_exit, result.stdout + result.stderr
    assert len(harness.events("provision")) == 1
    coordinated = harness.events("coordinate")
    assert len(coordinated) == 1
    assert f"--instance {instance}" in coordinated[0]
    assert "--timeout 10920" in coordinated[0]
    remote = harness.events("brev")
    assert not any(call.startswith(("create ", "delete ")) for call in remote)
    assert not any(call == f"exec {instance} true" for call in remote)
    assert harness.events("token-private") == [instance]
    assert harness.events("token-cleanup") == [instance]
    assert not list(tmp_path.glob("trtmc-checkpoint-token.*"))
    assert harness.environment["EXPECTED_TOKEN"] not in (
        result.stdout + result.stderr + harness.trace.read_text()
    )
    output = tmp_path / "test-output"
    assert (output.exists() and "conclusion=success" in output.read_text()) == (
        coordinator_exit == 0
    )


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_token_copy_cleans_up_without_model_execution_or_reprovision(
    tmp_path, gpu_job, cleanup_fails
):
    harness = JobHarness(tmp_path, gpu_job)
    instance = harness.environment["RESERVED_INSTANCE"]
    result = harness.run(
        "test", INSTANCE_NAME=instance, COPY_FAILS="true", CLEANUP_FAILS=str(cleanup_fails).lower()
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert not harness.events("coordinate")
    assert not harness.events("provision")
    remote = harness.events("brev")
    assert not any(call.startswith(("create ", "delete ")) for call in remote)
    assert harness.events("token-private")
    assert set(harness.events("token-private")) == {instance}
    assert harness.events("token-cleanup") == [instance]
    assert not list(tmp_path.glob("trtmc-checkpoint-token.*"))
    assert harness.environment["EXPECTED_TOKEN"] not in (
        result.stdout + result.stderr + harness.trace.read_text()
    )
    assert ("VM teardown is still required" in result.stderr) is cleanup_fails


@pytest.mark.parametrize("pull_exit,attempts", [(0, 1), (42, 6)])
def test_shared_base_pull_restores_scrubbed_reader_on_each_existing_attempt(
    tmp_path, gpu_job, pull_exit, attempts
):
    harness = JobHarness(tmp_path, gpu_job)
    instance = harness.environment["RESERVED_INSTANCE"]
    result = harness.run("test", INSTANCE_NAME=instance, BASE_PULL_EXIT=str(pull_exit))
    assert (result.returncode == 0) is (pull_exit == 0), result.stdout + result.stderr
    assert harness.events("registry-private") == [instance] * attempts
    assert harness.events("base-pull") == [instance] * attempts
    assert len(harness.events("coordinate")) == (1 if pull_exit == 0 else 0)
    assert harness.events("token-cleanup") == [instance]
    assert not harness.events("provision")
    assert not any(call.startswith(("create ", "delete ")) for call in harness.events("brev"))
    assert not list(tmp_path.glob("trtmc-registry-token.*"))
    assert harness.environment["EXPECTED_REGISTRY_TOKEN"] not in (
        result.stdout + result.stderr + harness.trace.read_text()
    )


def test_dependency_network_retry_does_not_repeat_gpu_test_or_replace_vm(tmp_path, gpu_job):
    harness = JobHarness(tmp_path, gpu_job)
    instance = harness.environment["RESERVED_INSTANCE"]
    result = harness.run("test", INSTANCE_NAME=instance, FETCH_FAILS_ONCE="true")
    assert result.returncode == 0, result.stdout + result.stderr
    fetches = [
        call for call in harness.events("brev") if call.startswith(f"exec {instance} git init ")
    ]
    assert len(fetches) == 2
    assert len(harness.events("coordinate")) == 1
    assert not harness.events("provision")
    assert not any(call.startswith(("create ", "delete ")) for call in harness.events("brev"))


def test_fetch_retry_recovers_after_origin_was_already_configured(tmp_path, gpu_job):
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(origin)], capture_output=True, check=True, text=True
    )
    subprocess.run(
        ["git", "-C", str(origin), "fast-import", "--quiet"],
        input=(
            "blob\nmark :1\ndata 8\nfixture\n\n"
            "commit refs/heads/main\n"
            "committer Test <test@example.com> 1 +0000\n"
            "data 7\nfixture\nM 100644 :1 fixture.txt\n\ndone\n"
        ),
        capture_output=True,
        check=True,
        text=True,
    )
    merge_sha = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "refs/heads/main"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    harness = JobHarness(tmp_path, gpu_job)
    setup_line = next(
        line.strip()
        for line in harness.steps["test"]["run"].splitlines()
        if line.strip().startswith('retry_backoff brev exec "$INSTANCE_NAME" "git init ')
    )
    # Run the workflow's actual remote setup against a local Git repository.
    # The first fetch fails after init/origin configuration has already happened.
    clone = tmp_path / "clone"
    setup_command = subprocess.run(
        [
            "bash",
            "-c",
            'retry_backoff() { "$@"; }\nbrev() { printf \'%s\\n\' "$3"; }\n' + setup_line,
        ],
        env={**harness.environment, "MERGE_SHA": merge_sha},
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    setup_command = setup_command.replace("/tmp/model_connect", shlex.quote(str(clone)))
    setup_command = setup_command.replace(
        f"https://github.com/{harness.environment['GITHUB_REPOSITORY']}.git",
        shlex.quote(origin.as_uri()),
    )
    git_shim = r"""
git() {
  if [ "$1" = fetch ] && [ ! -f "$FETCH_ATTEMPT" ]; then
    touch "$FETCH_ATTEMPT"
    return 7
  fi
  command git "$@"
}
"""
    instance = harness.environment["RESERVED_INSTANCE"]
    result = harness.run(
        "test",
        INSTANCE_NAME=instance,
        MERGE_SHA=merge_sha,
        LOCAL_GIT_SETUP=git_shim + setup_command,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual_sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert actual_sha == merge_sha
    assert (clone / "fixture.txt").read_text() == "fixture\n"
    fetches = [
        call for call in harness.events("brev") if call.startswith(f"exec {instance} git init ")
    ]
    assert len(fetches) == 2
    assert len(harness.events("coordinate")) == 1
    assert not harness.events("provision")
    assert not any(call.startswith(("create ", "delete ")) for call in harness.events("brev"))


@pytest.mark.parametrize("cleanup_exit", [0, 73])
def test_independent_cleanup_verifies_every_original_attempt_and_propagates_failure(
    tmp_path, gpu_job, cleanup_exit
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    cleanup = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup["if"]
    assert "provision-and-test" in cleanup["needs"]
    harness = JobHarness(tmp_path, cleanup)
    result = harness.run("release", VM_CLEANUP_EXIT=str(cleanup_exit))
    assert result.returncode == cleanup_exit, result.stdout + result.stderr
    assert harness.events("reconcile") == [
        "-m tools.community_gpu_lease --cleanup-all "
        f"--lease-directory {tmp_path}/gpu-ci-all-leases",
    ]
    assert not harness.events("brev")
    assert not harness.events("provision")
    assert not harness.events("coordinate")
    output = tmp_path / "release-output"
    assert output.exists() is (cleanup_exit == 0)


@pytest.mark.parametrize("phase,exit_code", [("reserve", 23), ("test", 1), ("test", 130)])
def test_gpu_failure_reaches_release_before_original_result(tmp_path, gpu_job, phase, exit_code):
    harness = JobHarness(tmp_path, gpu_job)
    reserve = harness.run("reserve", PROVISION_EXIT=str(exit_code if phase == "reserve" else 0))
    instance = harness.reserve_output()
    test_result = None
    if reserve.returncode == 0:
        test_result = harness.run("test", INSTANCE_NAME=instance, COORDINATOR_EXIT=str(exit_code))
        assert test_result.returncode == exit_code
    else:
        assert reserve.returncode == exit_code
    assert "always()" in harness.steps["release"]["if"]
    released = harness.run("release", INSTANCE_NAME=instance)
    assert released.returncode == 0, released.stderr
    assert len(harness.events("cleanup")) == 1
    confirmed = dict(
        line.split("=", 1) for line in (tmp_path / "release-output").read_text().splitlines()
    )
    state = "cancelled" if exit_code == 130 else "failure"
    result = harness.run(
        "result",
        JOB_STATUS=state,
        TEST_OUTCOME=state if test_result else "skipped",
        TEST_CONCLUSION="",
        CLEANUP_CONFIRMED=confirmed["cleanup_confirmed"],
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result-output").read_text() == f"conclusion={state}\n"


def test_interrupted_gpu_evidence_survives_blocked_cleanup_and_reruns(tmp_path, gpu_job):
    """Execute diagnostics and retain each attempt while deletion is unconfirmed."""
    report = {
        "schema_version": 1,
        "families": [
            {
                "family": "example",
                "status": "running",
                "phase": "environment",
                "failure_class": "infra_failure",
                "entrypoint_started": False,
                "requested_cases": ["one"],
                "cases": {"one": "not_run"},
            }
        ],
    }
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for attempt in (1, 2):
        directory = tmp_path / str(attempt)
        directory.mkdir()
        harness = JobHarness(directory, gpu_job)
        log = directory / "gpu-ci-output.log"
        log.write_text("TRTMC_GPU_SUMMARY=" + json.dumps(report) + "\npartial build output\n")
        summary = directory / "trtmc-community-gpu-summary.json"
        summarize = harness.steps["Summarize every selected GPU family"]
        result = subprocess.run(
            [
                "bash",
                "-e",
                "-c",
                summarize["run"].replace("/tmp/gpu-ci-output.log", shlex.quote(str(log))),
            ],
            cwd=REPO_ROOT,
            env={**harness.environment, "GITHUB_STEP_SUMMARY": str(directory / "rendered")},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        result = harness.run("classification", SUMMARY_FILE=str(summary))
        assert result.returncode == 0, result.stderr
        assert (directory / "classification-output").read_text() == "failure_class=infra_failure\n"

        # Replace only the artifact service with an immutable local store. A
        # rerun must not collide with or replace the earlier attempt's output.
        upload = harness.steps["Upload the smoke-test log"]
        name = (
            upload["with"]["name"]
            .replace("${{ github.run_id }}", "123")
            .replace("${{ github.run_attempt }}", str(attempt))
        )
        destination = artifacts / name
        destination.mkdir()
        for path in upload["with"]["path"].splitlines():
            source = Path(
                path.replace("/tmp/gpu-ci-output.log", str(log)).replace(
                    "${{ runner.temp }}", str(directory)
                )
            )
            shutil.copyfile(source, destination / source.name)
        released = harness.run("release", VM_CLEANUP_EXIT="75")
        assert released.returncode == 75
        assert not (directory / "release-output").exists()
        assert not (directory / "result-output").exists()
        assert (destination / log.name).read_text() == log.read_text()
        receipt = json.loads((destination / summary.name).read_text())
        assert receipt["families"][0]["phase"] == "environment"
        assert receipt["passed"] is False and receipt["complete"] is False
    assert len(list(artifacts.iterdir())) == 2


@pytest.mark.parametrize("diagnostic", ["Summarize every selected GPU family", "classification"])
def test_gpu_diagnostic_failure_still_releases_and_cannot_report_success(
    tmp_path, gpu_job, diagnostic
):
    harness = JobHarness(tmp_path, gpu_job)
    step = harness.steps[diagnostic]
    assert step["if"] == "always()"
    assert not step.get("continue-on-error", False)
    assert 0 < step["timeout-minutes"] <= 2
    # An unwritable output is a real script failure, even after successful GPU
    # execution. It must not turn into a passing conclusion after release.
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=REPO_ROOT,
        env={
            **harness.environment,
            "GITHUB_STEP_SUMMARY": str(tmp_path),
            "GITHUB_OUTPUT": str(tmp_path),
            "SUMMARY_FILE": str(tmp_path / "missing"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "always()" in harness.steps["release"]["if"]
    assert "classification" not in harness.steps["release"]["if"]
    released = harness.run("release")
    assert released.returncode == 0, released.stderr
    result = harness.run(
        "result",
        JOB_STATUS="failure",
        TEST_OUTCOME="success",
        TEST_CONCLUSION="success",
        CLEANUP_CONFIRMED="true",
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result-output").read_text() == "conclusion=failure\n"


def _cleanup_evidence(
    tmp_path,
    attempts,
    *,
    skipped=(),
    artifacts=True,
    foreign=False,
    reserve_outcome="failure",
    intent_only=False,
):
    """Execute the real recovery CLI; replace only provider and GitHub I/O."""
    from tools.tests.test_community_gpu_lease import _archive, _fixture, _producer

    repository = "example/repository"
    prefix = f"repos/{repository}/actions/runs/123"
    responses = {
        prefix: {
            "id": 123,
            "head_sha": "a" * 40,
            "path": ".github/workflows/community-ci.yml",
            "repository": {"id": 5, "full_name": repository},
        },
        prefix + "/jobs?filter=all&per_page=100&page=1": {"jobs": []},
        prefix + "/artifacts?per_page=100&page=1": {"artifacts": []},
    }
    for attempt in attempts:
        producer = _producer(attempt)
        producer["steps"][0]["conclusion"] = reserve_outcome
        if attempt in skipped:
            producer["steps"][0]["conclusion"] = "skipped"
        responses[prefix + "/jobs?filter=all&per_page=100&page=1"]["jobs"].append(producer)
        if not artifacts:
            continue
        lease, artifact, archive = _fixture(attempt)
        if intent_only:
            artifact["name"] = f"gpu-ci-intent-123-{attempt}"
            lease.update(instance_id=None, create_accepted=False, allocation_pending=True)
            lease["intent_prepared"] = True
        if foreign:
            lease["name"] = "trtmc-gpu-ci-999-1"
        if intent_only or foreign:
            archive = _archive(lease)
            artifact["digest"] = "sha256:" + hashlib.sha256(archive).hexdigest()
        responses[prefix + "/artifacts?per_page=100&page=1"]["artifacts"].append(artifact)
        responses[f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"] = {
            "binary": archive.hex()
        }
    data = tmp_path / "github-evidence.json"
    data.write_text(json.dumps(responses))
    driver = tmp_path / "reconcile.py"
    driver.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from tools import community_gpu_lease as recovery\n"
        f"responses = json.loads(Path({str(data)!r}).read_text())\n"
        "def api(path, *, binary=False):\n"
        "    value = responses[path]\n"
        "    return bytes.fromhex(value['binary']) if binary else value\n"
        "def cleanup(name, *, lease_file, timeout):\n"
        "    lease = json.loads(lease_file.read_text())\n"
        "    assert lease['name'] == name and lease['organization_id'] == 'org-fixture'\n"
        "    assert timeout == 90\n"
        "    with open(os.environ['TRACE'], 'a') as trace:\n"
        "        trace.write('cleanup\\t' + name + '\\n')\n"
        "    if os.environ['VM_CLEANUP_EXIT'] != '0':\n"
        "        raise recovery.ProvisionError('deletion remains unconfirmed')\n"
        "recovery._api = api\n"
        "recovery.cleanup = cleanup\n"
        "def stop(_delay):\n"
        "    raise recovery.RecoveryError('fixture observation ended after incomplete reconciliation')\n"
        "recovery.time.sleep = stop\n"
        "raise SystemExit(recovery.main())\n"
    )
    return driver


@pytest.mark.parametrize("fault", ["missing-artifact", "foreign-name", "unknown-delete"])
def test_backstop_never_guesses_an_allocation_from_the_current_attempt(tmp_path, fault):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    driver = _cleanup_evidence(
        tmp_path, [1], artifacts=fault != "missing-artifact", foreign=fault == "foreign-name"
    )
    result = harness.run(
        "release",
        LEASE_DRIVER=str(driver),
        GITHUB_RUN_ATTEMPT="3",
        INSTANCE_NAME="trtmc-gpu-ci-123-3",
        ALLOCATION_REQUESTED="false",
        OWNER_CLEANUP_CONFIRMED="true",
        VM_CLEANUP_EXIT="73" if fault == "unknown-delete" else "0",
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert harness.events("cleanup") == (
        ["trtmc-gpu-ci-123-1"] if fault == "unknown-delete" else []
    )
    assert not harness.events("provision")
    assert not (tmp_path / "release-output").exists()


def test_backstop_uses_every_recovered_identity_when_cancelled_owner_outputs_are_empty(tmp_path):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    driver = _cleanup_evidence(tmp_path, [1, 2])
    result = harness.run(
        "release",
        LEASE_DRIVER=str(driver),
        GITHUB_RUN_ATTEMPT="3",
        INSTANCE_NAME="",
        INSTANCE_TYPE="",
        ORGANIZATION_ID="",
        ALLOCATION_REQUESTED="",
        OWNER_CLEANUP_CONFIRMED="",
    )
    assert result.returncode == 0, result.stderr
    assert harness.events("cleanup") == ["trtmc-gpu-ci-123-1", "trtmc-gpu-ci-123-2"]
    for attempt in (1, 2):
        recovered = json.loads(
            (tmp_path / f"gpu-ci-all-leases/{attempt}/trtmc-gpu-ci-lease.json").read_text()
        )
        assert recovered["run_attempt"] == attempt
        assert recovered["instance_id"] == "vm-fixture"
    assert not (tmp_path / "gpu-ci-all-leases/3").exists()
    assert (tmp_path / "release-output").read_text() == "cleanup_confirmed=true\n"


def test_backstop_skipped_new_attempt_still_releases_older_allocation(tmp_path):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    driver = _cleanup_evidence(tmp_path, [1, 2], skipped={2})
    result = harness.run("release", LEASE_DRIVER=str(driver), NO_ALLOCATION_CONFIRMED="true")
    assert result.returncode == 0, result.stderr
    assert harness.events("cleanup") == ["trtmc-gpu-ci-123-1"]
    assert (tmp_path / "release-output").read_text() == "cleanup_confirmed=true\n"
    assert not harness.events("provision")


def test_backstop_recovers_durable_intent_after_runner_dies_before_lease_upload(tmp_path):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    driver = _cleanup_evidence(tmp_path, [1], intent_only=True)
    result = harness.run("release", LEASE_DRIVER=str(driver), GITHUB_RUN_ATTEMPT="3")
    assert result.returncode == 0, result.stderr
    assert harness.events("cleanup") == ["trtmc-gpu-ci-123-1"]
    lease = json.loads((tmp_path / "gpu-ci-all-leases/1/trtmc-gpu-ci-lease.json").read_text())
    assert lease["intent_prepared"] and lease["allocation_pending"]
    assert lease["instance_id"] is None
    assert (tmp_path / "release-output").read_text() == "cleanup_confirmed=true\n"


@pytest.mark.parametrize(
    "attempt,outcome,proof,confirmed",
    [
        ("1", "skipped", "true", True),
        ("1", "skipped", "", False),
        ("1", "failure", "true", False),
        ("1", "cancelled", "true", False),
        ("1", "success", "true", False),
        ("1", "", "true", False),
        ("2", "skipped", "true", False),
        ("3", "skipped", "true", False),
    ],
)
def test_credential_free_no_allocation_proof_only_closes_initial_attempt(
    tmp_path, attempt, outcome, proof, confirmed
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    result = harness.run(
        "no_allocation",
        GITHUB_RUN_ATTEMPT=attempt,
        RESERVE_OUTCOME=outcome,
        NO_ALLOCATION_CONFIRMED=proof,
    )
    assert result.returncode == 0, result.stderr
    assert not harness.trace.exists()
    output = tmp_path / "no_allocation-output"
    assert output.exists() is confirmed
    if confirmed:
        assert output.read_text() == "cleanup_confirmed=true\n"
    for name in (
        "Check out trusted GPU cleanup",
        "Install the pinned Brev CLI",
        "Log in to Brev",
    ):
        assert harness.steps[name]["if"] == (
            "${{ steps.no_allocation.outputs.cleanup_confirmed != 'true' }}"
        )
    assert (
        "steps.no_allocation.outputs.cleanup_confirmed != 'true'" in harness.steps["release"]["if"]
    )


@pytest.mark.parametrize("cleanup_exit", [0, 73])
def test_owning_job_cleanup_failure_is_not_swallowed(tmp_path, gpu_job, cleanup_exit):
    harness = JobHarness(tmp_path, gpu_job)
    result = harness.run(
        "release",
        INSTANCE_NAME="trtmc-gpu-ci-123-2",
        VM_CLEANUP_EXIT=str(cleanup_exit),
    )
    assert result.returncode == cleanup_exit
    assert len(harness.events("cleanup")) == 1
    assert "--until-deleted" in harness.events("cleanup")[0]
    output = tmp_path / "release-output"
    assert (output.exists() and "cleanup_confirmed=true" in output.read_text()) is (
        cleanup_exit == 0
    )
    assert not harness.events("provision")
    assert not harness.events("coordinate")


def test_gpu_lease_survives_reservation_and_cleanup_with_a_trusted_backstop(gpu_job):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    steps = {step["name"]: step for step in gpu_job["steps"]}
    names = [step["name"] for step in gpu_job["steps"]]
    ordered = [
        "Release every earlier allocation attempt",
        "Prepare the GPU allocation intent",
        "Preserve the GPU allocation intent before creation",
        "Reserve a GPU instance",
        "Preserve the GPU instance lease",
        "Build the GPU image and validate the exact PR merge",
        "Summarize every selected GPU family",
        "Classify the GPU result",
        "Upload the smoke-test log",
        "Always tear down the GPU instance",
        "Record the step conclusion",
        "Preserve the final GPU instance lease",
    ]
    assert [names.index(name) for name in ordered] == sorted(names.index(name) for name in ordered)
    assert names.index("Always tear down the GPU instance") < names.index(
        "Record the step conclusion"
    )
    output = steps["Upload the smoke-test log"]
    assert output["if"] == "always()"
    assert output["continue-on-error"] is True
    assert 0 < output["timeout-minutes"] <= 2
    assert not output["with"].get("overwrite")
    intent = steps["Preserve the GPU allocation intent before creation"]
    assert intent.get("if") in (None, "success()", "${{ success() }}")
    assert intent["with"]["name"] == "gpu-ci-intent-${{ github.run_id }}-${{ github.run_attempt }}"
    assert steps["Reserve a GPU instance"]["if"] == (
        "${{ success() && steps.intent.outcome == 'success' && steps.intent.outputs.artifact-id != '' }}"
    )
    for name in (
        "Preserve the GPU allocation intent before creation",
        "Preserve the GPU instance lease",
        "Preserve the final GPU instance lease",
    ):
        step = steps[name]
        if step is not intent:
            assert "always()" in step["if"]
        assert step["uses"].startswith("actions/upload-artifact@")
        assert step["with"]["path"] == "${{ runner.temp }}/trtmc-gpu-ci-lease.json"
        assert step["with"]["if-no-files-found"] == "error"
        assert not step["with"].get("overwrite")
        assert not step.get("continue-on-error", False)
    assert (
        steps["Preserve the GPU instance lease"]["with"]["name"]
        == "${{ steps.prepare.outputs.lease_artifact_name }}"
    )
    assert steps["Preserve the final GPU instance lease"]["with"]["name"] == (
        "gpu-ci-lease-final-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    prior = steps["Release every earlier allocation attempt"]
    assert "--cleanup-prior" in prior["run"]
    assert prior.get("if") in (None, "success()", "${{ success() }}")
    assert not prior.get("continue-on-error", False)
    cleanup = workflow["jobs"]["cleanup"]
    assert gpu_job["timeout-minutes"] == 360
    assert cleanup["timeout-minutes"] == 360
    assert cleanup["permissions"] == {"contents": "read", "actions": "read"}
    assert gpu_job["permissions"] == {"contents": "read", "actions": "read"}
    cleanup_steps = {step["name"]: step for step in cleanup["steps"]}
    assert cleanup_steps["Check out trusted GPU cleanup"]["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }
    delete = cleanup_steps["Delete every original GPU allocation"]
    assert delete["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "needs.provision-and-test.outputs" not in str(delete)
    assert "always()" in delete["if"]
    assert "steps.cleanup-checkout.outcome == 'success'" in delete["if"]
    assert "steps.cleanup-login.outcome == 'success'" in delete["if"]
    assert "python3 -m tools.community_gpu_lease --cleanup-all" in delete["run"]
    assert "|| true" not in delete["run"]
    assert "-r2" not in delete["run"] and "-r3" not in delete["run"]


@pytest.mark.parametrize(
    "prepare_exit,upload_outcome,artifact_id,expected_posts",
    [
        (0, "success", "17", 1),
        (23, "skipped", "", 0),
        (0, "failure", "", 0),
        (0, "cancelled", "", 0),
        (0, "success", "", 0),
        (0, "failure", "17", 0),
    ],
)
def test_no_allocation_post_before_confirmed_immutable_intent_upload(
    tmp_path, gpu_job, prepare_exit, upload_outcome, artifact_id, expected_posts
):
    harness = JobHarness(tmp_path, gpu_job)
    prepared = harness.run("prepare", PREPARE_EXIT=str(prepare_exit))
    assert prepared.returncode == prepare_exit
    assert not harness.events("provision")
    assert len(harness.events("prepare")) == 1
    # Evaluate the checked-in Actions admission expression in a shell after
    # substituting its three step-result inputs. Upload itself is a hosted action.
    expression = harness.steps["reserve"]["if"].removeprefix("${{ ").removesuffix(" }}")
    replacements = {
        "success()": "${PRIOR_SUCCESS}",
        "steps.intent.outcome": '"${UPLOAD_OUTCOME}"',
        "steps.intent.outputs.artifact-id": '"${ARTIFACT_ID}"',
    }
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    expression = expression.replace("${PRIOR_SUCCESS}", '"${PRIOR_SUCCESS}" == true')
    script = STUBS + f"if [[ {expression} ]]; then\n" + harness.steps["reserve"]["run"] + "\nfi\n"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        env={
            **harness.environment,
            "GITHUB_OUTPUT": str(tmp_path / "reserve-output"),
            "PRIOR_SUCCESS": str(prepare_exit == 0 and upload_outcome == "success").lower(),
            "UPLOAD_OUTCOME": upload_outcome,
            "ARTIFACT_ID": artifact_id,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert len(harness.events("provision")) == expected_posts
    assert not harness.events("coordinate")


@pytest.mark.parametrize("prior_cleanup_exit", [0, 73])
def test_prior_attempt_cleanup_blocks_new_create_until_confirmed(
    tmp_path, gpu_job, prior_cleanup_exit
):
    harness = JobHarness(tmp_path, gpu_job)
    driver = _cleanup_evidence(tmp_path, [1])
    prior = harness.run(
        "Release every earlier allocation attempt",
        LEASE_DRIVER=str(driver),
        VM_CLEANUP_EXIT=str(prior_cleanup_exit),
    )
    assert prior.returncode == (0 if prior_cleanup_exit == 0 else 75), prior.stderr
    assert harness.events("cleanup") == ["trtmc-gpu-ci-123-1"]
    if prior.returncode == 0:
        assert harness.run("prepare").returncode == 0
        assert harness.run("reserve").returncode == 0
        kinds = [line.split("\t", 1)[0] for line in harness.trace.read_text().splitlines()]
        assert kinds == ["cleanup", "prepare", "provision"]
    else:
        assert not harness.events("prepare")
        assert not harness.events("provision")


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, True),
        ({"github.run_attempt": 1}, False),
        ({"needs.gpu-authorize.outputs.run_gpu": "false"}, True),
        ({"needs.gpu-authorize.result": "cancelled"}, True),
        ({"needs.gpu-authorize.result": "failure"}, True),
        ({"github.ref": "refs/heads/main"}, True),
        ({"github.ref": "refs/heads/contributor-change"}, False),
        ({"github.ref": "refs/tags/ci/developer"}, False),
        ({"github.event_name": "pull_request"}, False),
        ({"github.event_name": "pull_request_target"}, False),
        ({"inputs.source_snapshot": ""}, False),
        ({"inputs.task": "dependency-image-access"}, False),
        (
            {
                "github.run_attempt": 1,
                "needs.gpu-authorize.result": "success",
                "needs.gpu-authorize.outputs.run_gpu": "true",
            },
            True,
        ),
    ],
)
def test_cleanup_job_admits_trusted_reruns_after_current_cpu_failure(tmp_path, updates, expected):
    """Evaluate actual job admission; running its shell alone misses skipped jobs."""
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    condition = workflow["jobs"]["cleanup"]["if"].strip().removeprefix("${{").removesuffix("}}")
    context = {
        "github.event_name": "workflow_dispatch",
        "github.run_attempt": 2,
        "github.ref": "refs/heads/ci/developer",
        "inputs.source_snapshot": '{"merge_sha":"' + "b" * 40 + '"}',
        "inputs.task": "premerge",
        "needs.gpu-authorize.result": "skipped",
        "needs.gpu-authorize.outputs.run_gpu": "",
        **updates,
    }
    for key, value in context.items():
        condition = condition.replace(key, repr(value))
    condition = (
        condition.replace("&&", " and ")
        .replace("||", " or ")
        .replace("!startsWith", "not startsWith")
    )
    admitted = eval(
        condition.strip(),
        {"__builtins__": {}},
        {"always": lambda: True, "startsWith": lambda value, prefix: value.startswith(prefix)},
    )
    assert admitted is expected
    if admitted:
        harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
        driver = _cleanup_evidence(tmp_path, [1])
        result = harness.run("release", LEASE_DRIVER=str(driver))
        assert result.returncode == 0, result.stderr
        assert harness.events("cleanup") == ["trtmc-gpu-ci-123-1"]


@pytest.mark.parametrize("run_gpu", ["", "false"])
@pytest.mark.parametrize("cpu", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_rerun_without_current_gpu_result_remains_pending_until_old_allocations_released(
    tmp_path, run_gpu, cpu, confirmed, execution_publisher
):
    script = _workflow_step_script(
        "community-ci.yml", "publish", "Publish the released workflow conclusion"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + script],
        env={
            **execution_publisher,
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": cpu,
            "RUN_GPU": run_gpu,
            "GPU_RESULT": "skipped",
            "GPU_FAILURE_CLASS": "",
            "OWNER_RELEASE_CONFIRMED": "true",
            "BACKSTOP_RELEASE_CONFIRMED": "true" if confirmed else "",
            "ORIGINAL_RUN_ATTEMPT": "2",
            "NO_ALLOCATION_CONFIRMED": "true",
            "COMPLETE_RESULT": "success" if cpu == "success" and run_gpu == "false" else "failure",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "123",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "pending"
        if not confirmed
        else "success"
        if cpu == "success" and run_gpu == "false"
        else "failure"
    )
    assert f"state={expected}" in writes.read_text().splitlines()


@pytest.mark.parametrize(
    "attempt,cleanup,confirmed,expected",
    [
        ("1", "skipped", "", 0),
        ("1", "success", "true", 1),
        ("2", "success", "true", 0),
        ("2", "success", "", 1),
        ("2", "skipped", "true", 1),
        ("2", "failure", "true", 1),
        ("2", "cancelled", "", 1),
        ("3", "success", "true", 0),
    ],
)
def test_empty_current_gpu_plan_requires_historical_release_on_rerun(
    attempt, cleanup, confirmed, expected
):
    script = _workflow_step_script(
        "community-ci.yml", "publish", "Require the complete Community CI result"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "EVENT_NAME": "workflow_dispatch",
            "AUTHORIZED_RESULT": "success",
            "CPU_RESULT": "success",
            "GPU_AUTHORIZED": "success",
            "RUN_GPU": "false",
            "GPU_RESULT": "skipped",
            "CLEANUP_RESULT": cleanup,
            "BACKSTOP_RELEASE_CONFIRMED": confirmed,
            "ORIGINAL_RUN_ATTEMPT": attempt,
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize("dual_run", ["", "false", "true"])
@pytest.mark.parametrize("entry_ref", ["refs/heads/main", "refs/heads/ci/developer"])
def test_manual_dev_request_never_allocates_a_stable_publisher(tmp_path, dual_run, entry_ref):
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Select the Community CI branches"
            ),
        ],
        env={
            **os.environ,
            "DUAL_RUN": dual_run,
            "CI_ENTRY_REF": entry_ref,
            "EVENT_NAME": "workflow_dispatch",
            "REQUESTED_LANE": "dev",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'lanes=["dev"]\n'


@pytest.mark.parametrize("stable_state", ["failure", "pending"])
@pytest.mark.parametrize("entry_ref", ["refs/heads/main", "refs/heads/ci/developer"])
def test_manual_dev_preserves_stable_gpu_verdict_despite_successful_cpu(
    tmp_path, stable_state, entry_ref
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    steps = {step["name"]: step for step in workflow["jobs"]["dispatch"]["steps"]}
    head, merge = "a" * 40, "c" * 40
    states = tmp_path / "states.json"
    states.write_text(json.dumps({"Stable Community GPU CI": stable_state}))
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "gh"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['CALLS']).open('a') as record:
    record.write(json.dumps(args) + '\\n')
if '--input' in args:
    payload = json.loads(Path(args[args.index('--input') + 1]).read_text())
    assert payload['inputs']['ci_lane'] == 'dev'
    print(json.dumps({'workflow_run_id': 43}))
elif any(a.startswith('/repos/') and '/actions/runs/' in a for a in args):
    path = next(a for a in args if a.startswith('/repos/'))
    print(json.dumps(json.loads(os.environ['RUNS'])[path.rsplit('/', 1)[1]]))
else:
    assert any('/statuses/' in a for a in args)
    fields = dict(a.split('=', 1) for a in args if '=' in a)
    path = Path(os.environ['STATES'])
    states = json.loads(path.read_text())
    states[fields['context']] = fields['state']
    path.write_text(json.dumps(states))
"""
    )
    fake.chmod(0o755)
    common = {
        "path": ".github/workflows/community-ci.yml",
        "head_sha": head,
        "status": "completed",
        "conclusion": "success",
    }
    # After promotion, a successful legacy PR run proves CPU only. The full
    # Stable GPU run is failed or pending and must retain ownership of its verdict.
    runs = {
        "42": {
            **common,
            "event": "pull_request",
            "display_title": f"PR #17 · community CI · head {head} · merge {merge}",
        },
        "43": {
            **common,
            "event": "workflow_dispatch",
            "head_branch": "ci/developer",
            "display_title": f"Dev Community CI · PR #17 · head {head} · merge {merge}",
        },
        "53": {
            **common,
            "event": "workflow_dispatch",
            "head_branch": "main",
            "display_title": f"Stable Community CI · PR #17 · head {head} · merge {merge}",
            "status": "in_progress" if stable_state == "pending" else "completed",
            "conclusion": None if stable_state == "pending" else "failure",
        },
    }
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CI_BRANCH": "ci/developer",
        "CI_ENTRY_BRANCH": "ci/developer",
        "CI_ENTRY_REF": entry_ref,
        "EVENT_NAME": "workflow_dispatch",
        "REQUESTED_LANE": "dev",
        "DUAL_RUN": "true",
        "AUTOMATIC_GPU": "true",
        "STABLE_RUN_ID": "42",
        "PR_NUMBER": "17",
        "HEAD_SHA": head,
        "SOURCE_SNAPSHOT": json.dumps({"merge_sha": merge}),
        "GITHUB_REPOSITORY": "example/source",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "100",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "selection"),
        "RUNS": json.dumps(runs),
        "STATES": str(states),
        "CALLS": str(calls),
    }

    def execute(script, environment):
        result = subprocess.run(
            ["bash", "-c", script], env=environment, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr

    execute(
        _workflow_step_script("community-ci.yml", "snapshot", "Select the Community CI branches"),
        env,
    )
    lanes = json.loads((tmp_path / "selection").read_text().strip().split("=", 1)[1])
    for lane in lanes:
        output = tmp_path / f"{lane}-output"
        lane_env = {
            **env,
            "LANE": lane,
            "CI_REF": "main" if lane == "stable" else "ci/developer",
            "STATUS_CONTEXT": (
                "Stable Community GPU CI" if lane == "stable" else "Dev Community CI"
            ),
            "GITHUB_OUTPUT": str(output),
        }
        # Also execute against the pre-fix workflow to reproduce the original
        # bad status write rather than merely checking for a new guard's text.
        if "Authorize the result publisher" in steps:
            execute(steps["Authorize the result publisher"]["run"], lane_env)
        execute(steps["Dispatch the selected Community CI implementation"]["run"], lane_env)
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        if outputs.get("existing_stable") == "true":
            execute(steps["Mark the selected CI pending"]["run"], lane_env)
        execute(
            steps["Publish the complete workflow conclusion"]["run"],
            {
                **lane_env,
                "PIPELINE_RUN_ID": outputs["run_id"],
                "CI_BRANCH": outputs["ci_ref"],
                "EXISTING_STABLE": outputs.get("existing_stable", ""),
            },
        )
    assert json.loads(states.read_text()) == {"Stable Community GPU CI": stable_state}
    assert all(
        not any("/statuses/" in arg for arg in args)
        for args in map(json.loads, calls.read_text().splitlines())
    )
    for arguments in map(json.loads, calls.read_text().splitlines()):
        assert "context=Stable Community GPU CI" not in arguments


@pytest.mark.parametrize(
    "lane,entry_ref,context,allowed",
    [
        ("stable", "refs/heads/main", "Stable Community GPU CI", True),
        ("dev", "refs/heads/main", "Dev Community CI", True),
        ("dev", "refs/heads/ci/developer", "Dev Community CI", True),
        ("stable", "refs/heads/ci/developer", "Stable Community GPU CI", False),
        ("stable", "refs/tags/main", "Stable Community GPU CI", False),
        ("stable", "refs/pull/17/merge", "Stable Community GPU CI", False),
        ("stable", "refs/heads/main", "Dev Community CI", False),
        ("dev", "refs/heads/ci/developer", "Stable Community GPU CI", False),
        ("unknown", "refs/heads/main", "Dev Community CI", False),
    ],
)
def test_publisher_authorization_precedes_all_status_writes(
    tmp_path, lane, entry_ref, context, allowed
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    steps = workflow["jobs"]["dispatch"]["steps"]
    assert steps[0]["id"] == "publisher"
    assert steps[0]["env"]["CI_ENTRY_REF"] == "${{ github.ref }}"
    result = subprocess.run(
        ["bash", "-c", steps[0]["run"]],
        env={
            **os.environ,
            "LANE": lane,
            "CI_ENTRY_REF": entry_ref,
            "EVENT_NAME": "workflow_dispatch",
            "REQUESTED_LANE": "dev",
            "STATUS_CONTEXT": context,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is allowed, result.stderr
    if allowed:
        assert (tmp_path / "output").read_text() == "authorized=true\n"
    else:
        assert not (tmp_path / "output").exists()
    # Ordinary steps require prior success. The failure callback needs its
    # own authorization check so denying a publisher cannot turn Stable red.
    for step in steps[1:-1]:
        if step["name"] == "Mark the selected CI pending":
            assert step["if"] == "${{ steps.execution.outputs.existing_stable == 'true' }}"
        else:
            assert "if" not in step
    assert "failure()" in steps[-1]["if"]
    assert "steps.publisher.outputs.authorized == 'true'" in steps[-1]["if"]
    reporter = workflow["jobs"]["snapshot"]["steps"][-1]
    assert reporter["env"]["STATUS_CONTEXT"] == (
        "${{ steps.lanes.outputs.lanes == '[\"dev\"]' && 'Dev Community CI' || 'Stable Community GPU CI' }}"
    )


@pytest.mark.parametrize(
    "entry_ref", ["refs/heads/contributor", "refs/tags/main", "refs/tags/ci/developer"]
)
def test_unapproved_coordinator_is_rejected_before_selecting_lanes(tmp_path, entry_ref):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    for name in ("snapshot", "dispatch"):
        assert "github.ref == 'refs/heads/main'" in workflow["jobs"][name]["if"]
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "snapshot", "Select the Community CI branches"
            ),
        ],
        env={
            **os.environ,
            "CI_ENTRY_REF": entry_ref,
            "EVENT_NAME": "workflow_dispatch",
            "REQUESTED_LANE": "dev",
            "DUAL_RUN": "true",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()
    assert "inputs.ci_lane == 'dev'" in workflow["concurrency"]["group"]


@pytest.mark.parametrize(
    "lane,ref", [("dev", "unprotected"), ("dev", "refs/tags/main"), ("stable", "ci/developer")]
)
def test_dispatch_rejects_unapproved_implementation_refs(tmp_path, lane, ref):
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\ntouch "$CALLS"\nexit 99\n')
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "dispatch", "Dispatch the selected Community CI implementation"
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "LANE": lane,
            "CI_REF": ref,
            "STABLE_RUN_ID": "42",
            "CALLS": str(tmp_path / "calls"),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert "approved" in result.stdout
    assert not (tmp_path / "calls").exists()
    assert not (tmp_path / "output").exists()


@pytest.fixture(params=[("stable", "main"), ("dev", "ci/developer"), ("dev", "main")])
def execution_publisher(request):
    lane, branch = request.param
    return {"CI_LANE": lane, "CI_REF": f"refs/heads/{branch}"}


@pytest.mark.parametrize(
    "step_name", ["Mark the trusted execution pending", "Publish the released workflow conclusion"]
)
@pytest.mark.parametrize(
    "event,lane,ref,allowed",
    [
        ("workflow_dispatch", "stable", "refs/heads/main", True),
        ("workflow_dispatch", "dev", "refs/heads/ci/developer", True),
        ("workflow_dispatch", "dev", "refs/heads/main", True),
        ("workflow_dispatch", "stable", "refs/heads/ci/developer", False),
        ("workflow_dispatch", "stable", "refs/tags/main", False),
        ("workflow_dispatch", "dev", "refs/heads/contributor", False),
        ("workflow_dispatch", "dev", "refs/pull/17/merge", False),
        ("workflow_dispatch", "unknown", "refs/heads/main", False),
        ("pull_request", "stable", "refs/heads/main", False),
        ("pull_request_target", "stable", "refs/heads/main", False),
    ],
)
def test_execution_pending_and_final_publishers_reject_untrusted_contexts(
    tmp_path, step_name, event, lane, ref, allowed
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    job = workflow["jobs"]["authorize" if step_name.startswith("Mark") else "publish"]
    step = next(step for step in job["steps"] if step["name"] == step_name)
    assert step["env"]["CI_LANE"] == "${{ inputs.ci_lane }}"
    assert step["env"]["CI_REF"] == "${{ github.ref }}"
    # Exercise the checked-in event admission before its independent shell guard.
    condition = step["if"].removeprefix("${{ ").removesuffix(" }}")
    condition = condition.replace("github.event_name", repr(event)).replace("always()", "True")
    admitted = eval(condition.replace("&&", " and "), {"__builtins__": {}})
    writes = tmp_path / "writes"
    if not admitted:
        assert not allowed
        assert not writes.exists()
        return
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + step["run"]],
        env={
            **os.environ,
            "CI_LANE": lane,
            "CI_REF": ref,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success",
            "RUN_GPU": "false",
            "GPU_RESULT": "skipped",
            "GPU_FAILURE_CLASS": "",
            "ORIGINAL_RUN_ATTEMPT": "1",
            "COMPLETE_RESULT": "success",
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "123",
            "STATUS_WRITES": str(writes),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) is allowed, result.stderr
    assert writes.exists() is allowed
    if allowed:
        arguments = writes.read_text().splitlines()
        assert (
            "context=Stable Community GPU CI" if lane == "stable" else "context=Dev Community CI"
        ) in arguments
        assert f"state={'pending' if step_name.startswith('Mark') else 'success'}" in arguments


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, "success"),
        ({"RUN_GPU": "false", "GPU_RESULT": "skipped", "CLEANUP_RESULT": "skipped"}, "success"),
        ({"CPU_RESULT": "failure", "RUN_GPU": "false", "GPU_RESULT": "skipped"}, "failure"),
        (
            {"GPU_RESULT": "failure", "TEST_RESULT": "failure", "GPU_FAILURE_CLASS": "pr_failure"},
            "failure",
        ),
        (
            {
                "GPU_RESULT": "failure",
                "TEST_RESULT": "failure",
                "GPU_FAILURE_CLASS": "infra_failure",
            },
            "error",
        ),
        ({"GPU_RESULT": "cancelled", "TEST_RESULT": "", "GPU_FAILURE_CLASS": ""}, "error"),
        ({"GPU_RESULT": "success", "TEST_RESULT": "failure"}, "error"),
        ({"GPU_RESULT": "failure", "TEST_RESULT": "success"}, "error"),
        ({"CLEANUP_RESULT": "failure", "BACKSTOP_RELEASE_CONFIRMED": ""}, "pending"),
        ({"CLEANUP_RESULT": "success", "BACKSTOP_RELEASE_CONFIRMED": ""}, "pending"),
        (
            {
                "GPU_RESULT": "failure",
                "GPU_FAILURE_CLASS": "infra_failure",
                "BACKSTOP_RELEASE_CONFIRMED": "",
            },
            "pending",
        ),
        (
            {
                "ORIGINAL_RUN_ATTEMPT": "2",
                "RUN_GPU": "false",
                "GPU_RESULT": "skipped",
                "BACKSTOP_RELEASE_CONFIRMED": "",
            },
            "pending",
        ),
        ({"ORIGINAL_RUN_ATTEMPT": "2", "RUN_GPU": "false", "GPU_RESULT": "skipped"}, "success"),
    ],
)
def test_complete_gate_and_status_share_the_same_release_contract(
    tmp_path, execution_publisher, updates, expected
):
    """A green job alone cannot replace test evidence or confirmed resource release."""
    environment = {
        **os.environ,
        **execution_publisher,
        "EVENT_NAME": "workflow_dispatch",
        "AUTHORIZED_RESULT": "success",
        "CPU_RESULT": "success",
        "GPU_AUTHORIZED": "success",
        "RUN_GPU": "true",
        "GPU_RESULT": "success",
        "TEST_RESULT": "success",
        "GPU_FAILURE_CLASS": "none",
        "CLEANUP_RESULT": "success",
        "BACKSTOP_RELEASE_CONFIRMED": "true",
        "ORIGINAL_RUN_ATTEMPT": "1",
        "NO_ALLOCATION_CONFIRMED": "",
        "HEAD_SHA": "a" * 40,
        "GITHUB_REPOSITORY": "example/source",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "123",
        "STATUS_WRITES": str(tmp_path / "writes"),
        **updates,
    }
    gate = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "publish", "Require the complete Community CI result"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (gate.returncode == 0) is (expected == "success"), gate.stderr
    publish = subprocess.run(
        [
            "bash",
            "-c",
            'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n'
            + _workflow_step_script(
                "community-ci.yml", "publish", "Publish the released workflow conclusion"
            ),
        ],
        env={**environment, "COMPLETE_RESULT": "success" if gate.returncode == 0 else "failure"},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert publish.returncode == 0, publish.stderr
    arguments = (tmp_path / "writes").read_text().splitlines()
    assert f"state={expected}" in arguments
    assert (
        "context=Stable Community GPU CI"
        if execution_publisher["CI_LANE"] == "stable"
        else "context=Dev Community CI"
    ) in arguments


@pytest.mark.parametrize("event", ["pull_request", "pull_request_target", "workflow_dispatch"])
@pytest.mark.parametrize("cpu", ["success", "failure", "cancelled", "skipped"])
def test_gpu_admission_requires_trusted_dispatch_and_passing_cpu(event, cpu):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    expression = workflow["jobs"]["gpu-authorize"]["if"].removeprefix("${{ ").removesuffix(" }}")
    for key, value in {
        "github.event_name": event,
        "needs.authorize.result": "success",
        "needs.required.result": cpu,
    }.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace("always()", "True").replace("&&", " and ")
    assert eval(expression, {"__builtins__": {}}) is (
        event == "workflow_dispatch" and cpu == "success"
    )


@pytest.mark.parametrize(
    "fault,queued_reads,running_reads,clock_step,conclusion,expected_rc",
    [
        ("discovery-once", 0, 0, 30, "success", 0),
        ("job-once", 0, 0, 30, "success", 0),
        ("none", 116, 0, 30, "success", 0),
        ("none", 30, 4, 600, "success", 0),
        ("job-once", 2, 1, 30, "failure", 1),
        ("none", -1, 0, 3600, "success", 75),
        ("job-always", 0, 0, 3600, "success", 75),
        ("discovery-always", 0, 0, 3600, "success", 75),
    ],
)
def test_promoted_cpu_observer_retains_the_same_child_through_queueing_and_read_failures(
    tmp_path, fault, queued_reads, running_reads, clock_step, conclusion, expected_rc
):
    """A read outage or long queue is not evidence that the actual CPU gate failed."""
    fake = tmp_path / "gh"
    fake.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[0] == 'api'
assert '--method' not in args and '--input' not in args
assert not any('/statuses/' in arg or '/dispatches' in arg or '/cancel' in arg or '/rerun' in arg for arg in args)
path = next(arg for arg in args if arg.startswith('/repos/'))
record = Path(os.environ['OBSERVER_CALLS'])
with record.open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
counter = Path(os.environ['OBSERVER_STATE'])
state = json.loads(counter.read_text()) if counter.exists() else {'discovery': 0, 'jobs': 0}
kind = 'discovery' if '/workflows/' in path else 'jobs'
state[kind] += 1
counter.write_text(json.dumps(state))
fault = os.environ['OBSERVER_FAULT']
if (fault == 'discovery-always' and kind == 'discovery'
    or fault == 'discovery-once' and kind == 'discovery' and state[kind] == 1
    or fault == 'job-always' and kind == 'jobs'
    or fault == 'job-once' and kind == 'jobs' and state[kind] == 1):
    print('gh: Bad Gateway (HTTP 502)', file=sys.stderr)
    sys.exit(1)
if kind == 'discovery':
    # A newer request exists on a later inventory response. After first
    # discovery the observer must keep run 42 instead of switching children.
    first_valid = 2 if fault == 'discovery-once' else 1
    run_id = 42 if state[kind] == first_valid else 99
    print(json.dumps({'workflow_runs': [{
        'id': run_id, 'event': 'workflow_dispatch', 'head_branch': 'main',
        'path': '.github/workflows/community-ci.yml', 'display_title': os.environ['EXPECTED_TITLE'],
        'status': 'queued', 'conclusion': None,
    }]}))
else:
    assert path == '/repos/example/source/actions/runs/42/jobs?filter=latest&per_page=100'
    count = state[kind] - (1 if fault == 'job-once' else 0)
    queued = int(os.environ['QUEUED_READS'])
    running = int(os.environ['RUNNING_READS'])
    if queued == -1 or count <= queued:
        print('{}')
    elif count <= queued + running:
        print(json.dumps({'status': 'in_progress', 'conclusion': None}))
    else:
        print(json.dumps({'status': 'completed', 'conclusion': os.environ['CPU_CONCLUSION']}))
"""
    )
    fake.chmod(0o755)
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    gate = workflow["jobs"]["required"]
    step = next(
        step for step in gate["steps"] if step["name"] == "Read the promoted Stable CPU result"
    )
    assert 60 < step["timeout-minutes"] < gate["timeout-minutes"] <= 360
    assert gate["timeout-minutes"] > workflow["jobs"]["unit"]["timeout-minutes"]
    head, merge = "a" * 40, "b" * 40
    result = subprocess.run(
        ["bash", "-c", "sleep() { SECONDS=$((SECONDS + CLOCK_STEP)); }\n" + step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CLOCK_STEP": str(clock_step),
            "OBSERVER_FAULT": fault,
            "OBSERVER_CALLS": str(tmp_path / "calls"),
            "OBSERVER_STATE": str(tmp_path / "state"),
            "QUEUED_READS": str(queued_reads),
            "RUNNING_READS": str(running_reads),
            "CPU_CONCLUSION": conclusion,
            "EXPECTED_TITLE": f"Stable Community CI · PR #17 · head {head} · merge {merge}",
            "PR_NUMBER": "17",
            "HEAD_SHA": head,
            "MERGE_SHA": merge,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_SERVER_URL": "https://github.com",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected_rc, result.stdout + result.stderr
    counts = json.loads((tmp_path / "state").read_text())
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert all(arguments[0] == "api" and "--method" not in arguments for arguments in calls)
    if fault != "discovery-always":
        assert counts["discovery"] == (2 if fault == "discovery-once" else 1)
        assert "Observing promoted Stable CPU run 42" in result.stdout
    if queued_reads == 116:
        assert counts["jobs"] == 117  # More reads than the old 55-minute observer permitted.
    if expected_rc == 0:
        assert (
            "Stable CPU result: https://github.com/example/source/actions/runs/42" in result.stdout
        )
        assert "budget exhausted" not in result.stdout
    elif expected_rc == 1:
        assert "CPU gate completed with failure" in result.stdout
    else:
        assert "Infrastructure observation budget exhausted" in result.stdout
        assert "existing execution remains authoritative" in result.stdout
        assert "Stable CPU result:" not in result.stdout


@pytest.mark.parametrize(
    "ref,event,lane,snapshot,repository,allowed",
    [
        ("refs/heads/main", "pull_request_target", "", "", "NVIDIA/TensorRT-Model-Connect", True),
        (
            "refs/heads/main",
            "workflow_dispatch",
            "stable",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            True,
        ),
        ("refs/heads/main", "workflow_dispatch", "dev", "", "NVIDIA/TensorRT-Model-Connect", True),
        (
            "refs/heads/ci/developer",
            "workflow_dispatch",
            "dev",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            True,
        ),
        (
            "refs/heads/ci/developer",
            "workflow_dispatch",
            "stable",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        (
            "refs/heads/ci/developer",
            "workflow_dispatch",
            "",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        (
            "refs/heads/ci/developer",
            "workflow_dispatch",
            "dev",
            "frozen",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        (
            "refs/heads/ci/developer",
            "pull_request_target",
            "dev",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        (
            "refs/heads/ci/developer",
            "pull_request",
            "dev",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        ("refs/heads/main", "pull_request", "stable", "", "NVIDIA/TensorRT-Model-Connect", False),
        (
            "refs/tags/ci/developer",
            "workflow_dispatch",
            "dev",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        (
            "refs/heads/contributor",
            "workflow_dispatch",
            "dev",
            "",
            "NVIDIA/TensorRT-Model-Connect",
            False,
        ),
        ("refs/heads/ci/developer", "workflow_dispatch", "dev", "", "example/fork", False),
    ],
)
def test_dev_manual_coordinator_admission_is_exact_and_cannot_select_stable(
    ref, event, lane, snapshot, repository, allowed
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    context = {
        "github.ref": ref,
        "github.event_name": event,
        "github.repository": repository,
        "inputs.ci_lane": lane,
        "inputs.source_snapshot": snapshot,
        "inputs.task": "premerge",
    }

    def admission(name, extra=None):
        expression = workflow["jobs"][name]["if"].strip().removeprefix("${{ ").removesuffix(" }}")
        for key, value in {**context, **(extra or {})}.items():
            expression = expression.replace(key, repr(value))
        return eval(
            "(" + expression.replace("&&", " and ").replace("||", " or ") + ")",
            {"__builtins__": {}},
        )

    assert admission("snapshot") is allowed
    assert (
        admission("dispatch", {"needs.snapshot.result": "success" if allowed else "skipped"})
        is allowed
    )


@pytest.mark.parametrize(
    "event,lane",
    [
        ("workflow_dispatch", "stable"),
        ("workflow_dispatch", ""),
        ("pull_request_target", "dev"),
        ("pull_request", "dev"),
    ],
)
def test_dev_entry_shell_rejects_default_stable_and_automatic_events(tmp_path, event, lane):
    selector = _workflow_step_script(
        "community-ci.yml", "snapshot", "Select the Community CI branches"
    )
    publisher = _workflow_step_script(
        "community-ci.yml", "dispatch", "Authorize the result publisher"
    )
    for index, script in enumerate((selector, publisher)):
        output = tmp_path / f"output-{index}"
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "CI_ENTRY_REF": "refs/heads/ci/developer",
                "EVENT_NAME": event,
                "REQUESTED_LANE": lane,
                "LANE": lane or "stable",
                "STATUS_CONTEXT": "Dev Community CI"
                if lane == "dev"
                else "Stable Community GPU CI",
                "DUAL_RUN": "true",
                "GITHUB_OUTPUT": str(output),
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0
        assert not output.exists()


@pytest.mark.parametrize("role", ["maintain", "admin", "write", "read", "none", "api-error"])
def test_dev_manual_entry_requires_maintainer_before_any_dispatch_or_status(tmp_path, role):
    selector = _workflow_step_script(
        "community-ci.yml", "snapshot", "Select the Community CI branches"
    )
    authorization = _workflow_step_script(
        "community-ci.yml", "snapshot", "Authorize a manual request"
    )
    calls, admitted = tmp_path / "calls", tmp_path / "admitted"
    script = (
        r"""
gh() {
  printf '%s\n' "$*" >> "$CALLS"
  test "$*" = 'api /repos/example/source/collaborators/example-user/permission --jq .role_name' || return 99
  if [ "$ROLE" = api-error ]; then return 1; fi
  printf '%s\n' "$ROLE"
}
"""
        + selector
        + "\n"
        + authorization
        + '\ntouch "$ADMITTED"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "CI_ENTRY_REF": "refs/heads/ci/developer",
            "EVENT_NAME": "workflow_dispatch",
            "REQUESTED_LANE": "dev",
            "DUAL_RUN": "true",
            "ACTOR": "example-user",
            "ROLE": role,
            "GITHUB_REPOSITORY": "example/source",
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "CALLS": str(calls),
            "ADMITTED": str(admitted),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) is (role in {"maintain", "admin"})
    assert admitted.exists() is (role in {"maintain", "admin"})
    assert (tmp_path / "output").read_text() == 'lanes=["dev"]\n'
    assert calls.read_text().splitlines() == [
        "api /repos/example/source/collaborators/example-user/permission --jq .role_name"
    ]


@pytest.mark.parametrize("configured_ref", ["", "main", "ci/developer", "unapproved"])
def test_dev_manual_coordinator_selects_its_own_implementation(configured_ref):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    step = next(
        step for step in workflow["jobs"]["dispatch"]["steps"] if step.get("id") == "execution"
    )
    expression = step["env"]["CI_REF"].removeprefix("${{ ").removesuffix(" }}")
    for key, value in {
        "matrix.lane": "dev",
        "github.ref": "refs/heads/ci/developer",
        "vars.TRTMC_COMMUNITY_CI_DEV_REF": configured_ref,
    }.items():
        expression = expression.replace(key, repr(value))
    assert (
        eval(expression.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}})
        == "ci/developer"
    )


def test_documentation_review_is_a_manual_metadata_only_bridge() -> None:
    source = (REPO_ROOT / ".github/workflows/doc-check.yml").read_text()
    workflow = yaml.safe_load(source)
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}
    assert workflow["permissions"] == {}
    job = workflow["jobs"]["dispatch"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert job["environment"]["name"] == "ci-dispatch"
    assert "actions/checkout" not in source
    assert "NVIDIA_INFERENCE_API_KEY" not in source
    assert "maintain|admin)" in source
    assert "SOURCE_SHA: ${{ github.sha }}" in source
    assert "statuses: write" not in source
    assert "deploy-pages" not in source

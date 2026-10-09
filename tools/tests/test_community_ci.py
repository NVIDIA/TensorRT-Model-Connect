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
    path = REPO_ROOT / "CONTRIBUTING.md"
    source = path.read_text(encoding="utf-8")
    ordered_markers = [
        "pre-commit install --install-hooks",
        "git commit --signoff",
        "git push --set-upstream origin",
        "Community CPU / Required",
        "Community GPU",
        "run-internal-ci",
        "TRTMC Internal CI / Automated premerge gate",
    ]

    positions = [source.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    for marker in (
        "automatically",
        "GitHub-hosted",
        "ubuntu-24.04",
        "read-only\nrepository permission",
        "no access to private runners, secrets, or GPUs",
        "Only after `Community CPU / Required`",
        "Stable\nCommunity CI",
        "Dev Community CI",
        "TRTMC_COMMUNITY_CI_DUAL_RUN=true",
        "isolated GPU instance",
        "pull-request checks",
        "Public Actions logs",
        "py -3 -m pip",
    ):
        assert marker in source
    assert "/run-ci" not in source
    assert "status comment" not in source


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


@pytest.mark.parametrize(
    ("cpu_state", "candidate_identity", "expected_returncode"),
    [
        ("success", "exact", 0),
        ("success", "regenerated", 0),
        ("success", "advanced-base", 0),
        ("success", "older-base-same-tree", 0),
        ("missing", "regenerated", 1),
        ("in_progress", "advanced-base", 1),
        ("failure", "advanced-base", 1),
        ("cancelled", "advanced-base", 1),
        ("skipped", "advanced-base", 1),
        ("success", "unrelated-base", 1),
        ("success", "newer-base", 1),
        ("success", "wrong-merge-base", 1),
        ("success", "invalid-base", 1),
        ("success", "invalid-parents", 1),
        ("success", "wrong-resolved", 1),
        ("success", "missing-tree", 1),
        ("success", "stale-head", 1),
        ("success", "different-tree", 1),
        ("success", "invalid-title", 1),
        ("success", "runs-api-error", 1),
        ("success", "merge-api-error", 1),
        ("success", "compare-api-error", 1),
        ("success", "jobs-api-error", 1),
        ("success", "unauthorized-actor", 1),
        ("success", "superseded-trigger", 1),
        ("success", "missing-merge", 1),
    ],
)
def test_internal_label_bridge_accepts_cpu_gate_across_main_advancement(
    tmp_path: Path,
    cpu_state: str,
    candidate_identity: str,
    expected_returncode: int,
) -> None:
    head_sha = "a" * 40
    base_sha = "b" * 40
    live_merge_sha = "c" * 40
    candidate_merge_sha = live_merge_sha if candidate_identity == "exact" else "d" * 40
    merge_tree_sha = "e" * 40
    older_base_cases = {
        "advanced-base",
        "older-base-same-tree",
        "unrelated-base",
        "newer-base",
        "wrong-merge-base",
        "compare-api-error",
    }
    candidate_base_sha = "f" * 40 if candidate_identity in older_base_cases else base_sha
    if candidate_identity == "invalid-base":
        candidate_base_sha = "invalid"
    candidate_head_sha = "f" * 40 if candidate_identity == "stale-head" else head_sha
    candidate_tree_sha = (
        "f" * 40
        if candidate_identity == "different-tree" or candidate_identity in older_base_cases
        else merge_tree_sha
    )
    if candidate_identity == "older-base-same-tree":
        candidate_tree_sha = merge_tree_sha
    if candidate_identity == "missing-tree":
        candidate_tree_sha = ""
    candidate_parents = [{"sha": candidate_base_sha}, {"sha": candidate_head_sha}]
    if candidate_identity == "invalid-parents":
        candidate_parents.append({"sha": "1" * 40})
    candidate_merge = {
        "sha": "1" * 40 if candidate_identity == "wrong-resolved" else candidate_merge_sha,
        "tree": {"sha": candidate_tree_sha},
        "parents": candidate_parents,
    }
    comparison = {
        "status": {"unrelated-base": "diverged", "newer-base": "behind"}.get(
            candidate_identity, "ahead"
        ),
        "merge_base_commit": {
            "sha": "1" * 40 if candidate_identity == "wrong-merge-base" else candidate_base_sha
        },
    }
    cpu_job = {
        "id": 33,
        "name": "Community CPU / Required",
        "status": "in_progress" if cpu_state == "in_progress" else "completed",
        "conclusion": None if cpu_state == "in_progress" else cpu_state,
    }
    cpu_jobs = {
        "jobs": [] if cpu_state == "missing" else [cpu_job],
    }
    run_title = (
        "PR #17 · stale Community CI"
        if candidate_identity == "invalid-title"
        else f"PR #17 · community CI · head {head_sha} · merge {candidate_merge_sha}"
    )
    github_output = tmp_path / "github-output"
    gh = tmp_path / "gh"
    gh.write_text(
        """#!/bin/bash
set -euo pipefail
arguments="$*"
case "$arguments" in
  *collaborators/tester/permission*) printf '%s\n' "$ACTOR_ROLE" ;;
  *pulls/17*)
    printf '{"state":"open","base":{"repo":{"full_name":"example/repo"},"ref":"main","sha":"%s"},"head":{"sha":"%s"},"merge_commit_sha":"%s"}\n' "$BASE_SHA" "$HEAD_SHA" "$MERGE_SHA"
    ;;
  *git/commits/*)
    requested_sha="${arguments##*/}"
    if [ "$requested_sha" = "$MERGE_SHA" ]; then
      printf '{"sha":"%s","tree":{"sha":"%s"},"parents":[{"sha":"%s"},{"sha":"%s"}]}\n' "$MERGE_SHA" "$MERGE_TREE_SHA" "$BASE_SHA" "$HEAD_SHA"
    elif [ "$requested_sha" = "$CANDIDATE_MERGE_SHA" ]; then
      [ "$CANDIDATE_IDENTITY" != merge-api-error ] || exit 1
      printf '%s\n' "$CANDIDATE_MERGE"
    else
      printf 'unexpected merge commit: %s\n' "$requested_sha" >&2
      exit 99
    fi
    ;;
  *community-ci.yml*)
    [ "$CANDIDATE_IDENTITY" != runs-api-error ] || exit 1
    printf '{"workflow_runs":[{"id":22,"event":"pull_request","head_sha":"%s","display_title":"%s","updated_at":"2026-01-01T00:00:00Z"}]}\n' "$HEAD_SHA" "$RUN_TITLE"
    ;;
  *compare/$CANDIDATE_BASE_SHA...$BASE_SHA?per_page=1)
    [ "$CANDIDATE_IDENTITY" != compare-api-error ] || exit 1
    printf '%s\n' "$COMPARISON"
    ;;
  *actions/runs/22/jobs*)
    [ "$CANDIDATE_IDENTITY" != jobs-api-error ] || exit 1
    printf '%s\n' "$CPU_JOBS" | jq "${@: -1}"
    ;;
  *) printf 'unexpected gh call: %s\n' "$arguments" >&2; exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "internal-ci-bridge.yml",
                "authorize",
                "Capture the exact pull-request snapshot",
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "ACTOR": "tester",
            "ACTOR_ROLE": "write" if candidate_identity == "unauthorized-actor" else "maintain",
            "PR_NUMBER": "17",
            "EVENT_NAME": "pull_request_target",
            "EVENT_HEAD_SHA": "1" * 40 if candidate_identity == "superseded-trigger" else head_sha,
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_OUTPUT": str(github_output),
            "HEAD_SHA": head_sha,
            "BASE_SHA": base_sha,
            "MERGE_SHA": "" if candidate_identity == "missing-merge" else live_merge_sha,
            "MERGE_TREE_SHA": merge_tree_sha,
            "CANDIDATE_MERGE_SHA": candidate_merge_sha,
            "CANDIDATE_BASE_SHA": candidate_base_sha,
            "CANDIDATE_IDENTITY": candidate_identity,
            "CANDIDATE_MERGE": json.dumps(candidate_merge),
            "COMPARISON": json.dumps(comparison),
            "CPU_JOBS": json.dumps(cpu_jobs),
            "RUN_TITLE": run_title,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    if expected_returncode == 0:
        assert github_output.read_text(encoding="utf-8") == (
            f"trigger_authorized=true\npr_number=17\nhead_sha={head_sha}\nbase_sha={base_sha}\n"
        )
    elif candidate_identity.endswith("-api-error"):
        assert "Community CPU / Required must pass" not in result.stdout + result.stderr
        assert "::error::Unable to" in result.stdout
    elif candidate_identity == "unauthorized-actor":
        assert "Only actors with maintain or admin access" in result.stdout
        assert not github_output.exists()
        return
    elif candidate_identity == "superseded-trigger":
        assert "superseded by a newer PR head" in result.stdout
    elif candidate_identity == "missing-merge":
        assert "has no testable merge commit" in result.stdout
    else:
        assert "Community CPU / Required must pass" in result.stdout + result.stderr
        if cpu_state == "missing":
            assert "status=missing" in result.stdout
        elif cpu_state == "in_progress":
            assert "status=in_progress" in result.stdout
        elif cpu_state != "success":
            assert f"conclusion={cpu_state}" in result.stdout
    assert "trigger_authorized=true" in github_output.read_text(encoding="utf-8")
    if expected_returncode != 0:
        assert github_output.read_text(encoding="utf-8") == "trigger_authorized=true\n"


def test_internal_label_bridge_consumes_authorized_trigger_after_snapshot_failure() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/internal-ci-bridge.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["authorize"]["steps"]
    consume = next(step for step in steps if step["name"] == "Consume the trusted trigger label")
    assert consume["if"] == (
        "${{ always() && github.event_name == 'pull_request_target' "
        "&& steps.snapshot.outputs.trigger_authorized == 'true' }}"
    )
    assert "--method DELETE" in consume["run"]
    assert "/labels/run-internal-ci" in consume["run"]


@pytest.mark.parametrize(
    ("private_conclusion", "current_head_matches", "expected_output"),
    [
        (
            "success",
            True,
            "state=success\ndescription=Automated internal CI passed\npublish_comment=false\n",
        ),
        (
            "failure",
            True,
            "state=failure\n"
            "description=Automated internal CI failed; details withheld\n"
            "publish_comment=true\n",
        ),
        (
            "success",
            False,
            "state=failure\n"
            "description=Automated internal CI result was superseded by a newer PR head\n"
            "publish_comment=false\n",
        ),
    ],
)
def test_internal_bridge_publishes_downstream_result_when_rest_base_differs(
    tmp_path: Path,
    private_conclusion: str,
    current_head_matches: bool,
    expected_output: str,
) -> None:
    head_sha = "a" * 40
    current_head_sha = head_sha if current_head_matches else "d" * 40
    tested_base_sha = "b" * 40
    stale_rest_base_sha = "c" * 40
    github_output = tmp_path / "github-output"
    gh = tmp_path / "gh"
    gh.write_text(
        """#!/bin/bash
set -euo pipefail
arguments="$*"
case "$arguments" in
  *pulls/17*)
    printf '{"state":"open","base":{"repo":{"full_name":"example/repo"},"ref":"main","sha":"%s"},"head":{"sha":"%s"}}\n' "$STALE_REST_BASE_SHA" "$CURRENT_HEAD_SHA"
    ;;
  *) printf 'unexpected gh call: %s\n' "$arguments" >&2; exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "internal-ci-bridge.yml",
                "publish",
                "Resolve the contributor-visible result",
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "PR_NUMBER": "17",
            "HEAD_SHA": head_sha,
            "BASE_SHA": tested_base_sha,
            "DISPATCH_RESULT": "success",
            "PRIVATE_CONCLUSION": private_conclusion,
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_OUTPUT": str(github_output),
            "STALE_REST_BASE_SHA": stale_rest_base_sha,
            "CURRENT_HEAD_SHA": current_head_sha,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert github_output.read_text(encoding="utf-8") == expected_output


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
    assert "python3 -m tools.brev_exec provision" in reserve["run"]
    assert 'echo "instance_name=$instance_name" >> "$GITHUB_OUTPUT"' in reserve["run"]
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
    assert cleanup["if"] == "${{ always() && steps.reserve.outputs.instance_name != '' }}"
    assert cleanup["env"] == {"INSTANCE_NAME": "${{ steps.reserve.outputs.instance_name }}"}
    assert test_step["env"]["INSTANCE_NAME"] == "${{ steps.reserve.outputs.instance_name }}"
    assert "python3 -m tools.brev_exec cleanup" in cleanup["run"]
    assert '--lease-file "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"' in cleanup["run"]
    assert "--until-deleted" in cleanup["run"]
    assert "|| true" not in cleanup["run"]
    assert job["outputs"] == {
        "conclusion": "${{ steps.result.outputs.conclusion }}",
        "failure_class": "${{ steps.classification.outputs.failure_class }}",
        "instance_name": "${{ steps.reserve.outputs.instance_name }}",
        "lease_artifact_name": "${{ steps.reserve.outputs.lease_artifact_name }}",
        "instance_type": "${{ steps.reserve.outputs.instance_type }}",
        "organization_id": "${{ steps.reserve.outputs.organization_id }}",
        "allocation_requested": "${{ steps.reserve.outputs.allocation_requested }}",
        "reserve_outcome": "${{ steps.reserve.outcome }}",
        "no_allocation_confirmed": "${{ steps.no_allocation.outputs.no_allocation_confirmed }}",
        "cleanup_confirmed": "${{ steps.release.outputs.cleanup_confirmed || steps.no_allocation.outputs.cleanup_confirmed }}",
    }
    cleanup_job = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup_job["if"]
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in cleanup_job["if"]
    cleanup_steps = {step["name"]: step for step in cleanup_job["steps"]}
    cleanup_script = cleanup_steps["Delete the deterministic GPU instance"]["run"]
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
        "-m tools.brev_exec cleanup --instance trtmc-gpu-ci-123-1 "
        f"--lease-file {tmp_path}/gpu-ci-lease/trtmc-gpu-ci-lease.json --until-deleted",
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
        "changed_path",
        "expected_scope",
        "expected_families",
        "expected_direct_families",
        "expected_added_families",
    ),
    [
        ("families/bert/model.py", "families", ["bert"], ["bert"], []),
        ("families/new_family/model.py", "all", ["bert", "gpt2"], [], ["new_family"]),
        ("README.md", "docs", [], [], []),
    ],
)
def test_gpu_impact_executes_only_trusted_base_code(
    tmp_path: Path,
    changed_path: str,
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


@pytest.mark.parametrize("provision_exitcode", [0, 1])
def test_gpu_cleanup_can_delete_instance_after_reservation_failure(
    tmp_path: Path,
    provision_exitcode: int,
) -> None:
    output = tmp_path / "output"
    calls = tmp_path / "brev-calls"
    brev = tmp_path / "brev"
    brev.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$BREV_CALLS"\n',
        encoding="utf-8",
    )
    brev.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "BREV_CALLS": str(calls),
        "PROVISION_EXITCODE": str(provision_exitcode),
        "GPU_TYPE": "L40",
        "GPU_PROVIDER": "auto",
        "HOST_RAM_GIB": "64",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(output),
        "RUNNER_TEMP": str(tmp_path),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
python3() {
  printf '%s %s\n' "$3" "$*" >> "$BREV_CALLS"
  test "$1" = -m && test "$2" = tools.brev_exec || return 99
  if [ "$3" = provision ]; then
    # Publish the attempted identity even if the single allocation fails.
    printf 'instance_name=trtmc-gpu-ci-123-2\n' >> "$GITHUB_OUTPUT"
    printf '{"name":"trtmc-gpu-ci-123-2","instance_id":"test123"}\n' > "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"
    return "$PROVISION_EXITCODE"
  fi
  test "$3" = cleanup || return 98
}
"""
            + _workflow_step_script(
                "community-ci.yml", "provision-and-test", "Reserve a GPU instance"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == provision_exitcode, result.stdout + result.stderr
    instance_name = "trtmc-gpu-ci-123-2"
    outputs = output.read_text(encoding="utf-8").splitlines()
    assert outputs == [
        "instance_name=trtmc-gpu-ci-123-2",
        "lease_artifact_name=gpu-ci-lease-123-2",
        "allocation_requested=false",
        f"instance_name={instance_name}",
    ]
    assert dict(line.split("=", 1) for line in outputs)["instance_name"] == instance_name
    cleanup = subprocess.run(
        [
            "bash",
            "-c",
            r"""
python3() {
  printf '%s %s\n' "$3" "$*" >> "$BREV_CALLS"
  test "$3" = cleanup || return 99
  test -f "$RUNNER_TEMP/trtmc-gpu-ci-lease.json" || return 98
}
"""
            + _workflow_step_script(
                "community-ci.yml",
                "provision-and-test",
                "Always tear down the GPU instance",
            ),
        ],
        env={**environment, "INSTANCE_NAME": instance_name},
        capture_output=True,
        text=True,
        check=False,
    )
    assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "provision -m tools.brev_exec provision --instance trtmc-gpu-ci-123-2 "
        "--provider aws --host-ram-gib 64 --disk-gb 500 --min-free-disk-gb 200 "
        f"--lease-file {tmp_path}/trtmc-gpu-ci-lease.json --timeout 2700 --attempts 1",
        f"cleanup -m tools.brev_exec cleanup --instance {instance_name} "
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
        == "${{ matrix.lane == 'stable' && 'main' || github.ref_name != 'main' && github.ref_name || vars.TRTMC_COMMUNITY_CI_DEV_REF || 'main' }}"
    )
    assert "sleep" not in step["run"]
    gpu = executor["jobs"]["provision-and-test"]
    test = next(step for step in gpu["steps"] if step.get("id") == "test")
    assert gpu["permissions"] == {"contents": "read"}
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
        ("true", ["stable", "dev"]),
        ("TRUE", ["stable"]),
        ("1", ["stable"]),
        ('["stable","dev"]', ["stable"]),
    ],
)
def test_community_dual_run_switch_controls_job_allocation(tmp_path, dual_run, expected_lanes):
    control = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    snapshot = control["jobs"]["snapshot"]
    selector = next(step for step in snapshot["steps"] if step.get("id") == "lanes")
    assert selector["env"]["DUAL_RUN"] == "${{ vars.TRTMC_COMMUNITY_CI_DUAL_RUN }}"
    assert snapshot["outputs"]["lanes"] == "${{ steps.lanes.outputs.lanes }}"
    assert control["jobs"]["dispatch"]["strategy"]["matrix"]["lane"] == (
        "${{ fromJSON(needs.snapshot.outputs.lanes) }}"
    )

    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-c", selector["run"]],
        env={**os.environ, "DUAL_RUN": dual_run, "CI_BRANCH": "main", "GITHUB_OUTPUT": str(output)},
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
            "STATUS_CONTEXT": f"{lane.title()} Community CI",
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
def test_only_complete_pipeline_publishes_stable_and_dev_results(
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
            "STATUS_CONTEXT": f"{lane.title()} Community CI",
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
    if expected is None or lane == "dev":
        assert not (tmp_path / "calls").exists()
    if expected is not None:
        assert (tmp_path / "output").read_text() == "reported=true\n"
        assert int((tmp_path / "counter").read_text()) == (2 if queued_first else 1)
    if expected is not None and lane == "stable":
        calls = (tmp_path / "calls").read_text().splitlines()
        assert f"state={expected}" in calls
        assert f"context={lane.title()} Community CI" in calls


@pytest.mark.parametrize("api_unavailable", [False, True])
def test_dev_observer_timeout_never_finalizes_the_resource_owner(tmp_path, api_unavailable):
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
            "CI_BRANCH": "ci/developer",
            "LANE": "dev",
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
        env={**os.environ, "LANE": "dev", "STATUS_WRITES": str(tmp_path / "writes")},
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
        ("true", "", "failure", "failure", "failure"),
        ("", "", "failure", "failure", "pending"),
        ("", "", "success", "success", "pending"),
        ("", "", "skipped", "failure", "failure"),
    ],
)
def test_trusted_inner_status_requires_release_before_a_gpu_verdict(
    tmp_path, owner, backstop, gpu, complete, expected
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    publisher = workflow["jobs"]["publish"]
    assert {"provision-and-test", "cleanup"} <= set(publisher["needs"])
    assert publisher["permissions"] == {"statuses": "write"}
    step = next(
        s for s in publisher["steps"] if s["name"] == "Publish the released Dev workflow conclusion"
    )
    assert "always()" in step["if"] and "workflow_dispatch" in step["if"]
    assert "inputs.ci_lane == 'dev'" in step["if"]
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
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success" if gpu != "skipped" else "failure",
            "RUN_GPU": "true" if gpu != "skipped" else "false",
            "GPU_RESULT": gpu,
            "GPU_FAILURE_CLASS": "none" if gpu == "success" else "pr_failure",
            "OWNER_RELEASE_CONFIRMED": owner,
            "BACKSTOP_RELEASE_CONFIRMED": backstop,
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
    assert "context=Dev Community CI" in arguments
    assert "target_url=https://github.com/example/source/actions/runs/42" in arguments
    if expected == "pending":
        assert "description=Community CI release remains unconfirmed" in arguments


@pytest.mark.parametrize(
    "outcome,proof,confirmed",
    [
        ("skipped", "true", True),
        ("skipped", "", False),
        ("failure", "true", False),
        ("cancelled", "true", False),
        ("success", "true", False),
        ("", "true", False),
    ],
)
def test_backstop_requires_explicit_trusted_no_allocation_proof(
    tmp_path, outcome, proof, confirmed
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    owner = workflow["jobs"]["provision-and-test"]
    marker = next(s for s in owner["steps"] if s.get("id") == "no_allocation")
    assert marker["if"] == "${{ always() && steps.reserve.outcome == 'skipped' }}"
    assert owner["outputs"]["reserve_outcome"] == "${{ steps.reserve.outcome }}"
    backup = workflow["jobs"]["cleanup"]
    step = backup["steps"][0]
    assert step["id"] == "no_allocation"
    assert step["env"] == {
        "RESERVE_OUTCOME": "${{ needs.provision-and-test.outputs.reserve_outcome }}",
        "NO_ALLOCATION_CONFIRMED": "${{ needs.provision-and-test.outputs.no_allocation_confirmed }}",
    }
    output = tmp_path / "output"
    calls = tmp_path / "brev-calls"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'brev() { printf "%s\\n" "$*" >> "$BREV_CALLS"; return 91; }\n' + step["run"],
        ],
        env={
            **os.environ,
            "RESERVE_OUTCOME": outcome,
            "NO_ALLOCATION_CONFIRMED": proof,
            "GITHUB_OUTPUT": str(output),
            "BREV_CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0 and not calls.exists()
    assert output.exists() is confirmed
    if confirmed:
        assert output.read_text() == "cleanup_confirmed=true\n"
    for name in (
        "Check out trusted GPU cleanup",
        "Install the pinned Brev CLI",
        "Log in to Brev",
        "Recover the GPU instance lease",
    ):
        guarded = next(s for s in backup["steps"] if s["name"] == name)
        assert guarded["if"] == "${{ steps.no_allocation.outputs.cleanup_confirmed != 'true' }}"
    # Without proof these guards stay eligible; normal owned-ID cleanup is retained.
    delete = next(s for s in backup["steps"] if s.get("id") == "release")
    assert "cleanup-login.outcome == 'success'" in delete["if"]
    assert "--until-deleted" in delete["run"]


def test_preparation_failure_finalizes_infra_without_brev_credentials(tmp_path):
    owner = _workflow_step_script(
        "community-ci.yml", "provision-and-test", "Confirm that reservation was skipped"
    )
    output = tmp_path / "owner-output"
    calls = tmp_path / "brev-calls"
    result = subprocess.run(
        ["bash", "-c", 'brev() { printf "%s\\n" "$*" >> "$BREV_CALLS"; return 91; }\n' + owner],
        env={"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(output), "BREV_CALLS": str(calls)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0 and not calls.exists()
    assert output.read_text().splitlines() == [
        "no_allocation_confirmed=true",
        "cleanup_confirmed=true",
    ]
    conclusion = _workflow_step_script(
        "community-ci.yml", "provision-and-test", "Record the step conclusion"
    )
    result = subprocess.run(
        ["bash", "-c", conclusion],
        env={
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
        "community-ci.yml", "publish", "Publish the released Dev workflow conclusion"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            "PATH": os.environ["PATH"],
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success",
            "RUN_GPU": "true",
            "GPU_RESULT": "failure",
            "GPU_FAILURE_CLASS": "infra_failure",
            "OWNER_RELEASE_CONFIRMED": "true",
            "BACKSTOP_RELEASE_CONFIRMED": "true",
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
    tmp_path, receipt, expected_class, released
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
        "community-ci.yml", "publish", "Publish the released Dev workflow conclusion"
    )
    writes = tmp_path / "writes"
    success = receipt in {"success", "false-success"}
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": "success",
            "RUN_GPU": "true",
            "GPU_RESULT": "success" if success else "failure",
            "GPU_FAILURE_CLASS": expected_class,
            "OWNER_RELEASE_CONFIRMED": "true" if released else "",
            "BACKSTOP_RELEASE_CONFIRMED": "",
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
    tmp_path, category, cpu
):
    """A skipped/failed classifier cannot blame the PR or relabel CPU failures."""
    publish = _workflow_step_script(
        "community-ci.yml", "publish", "Publish the released Dev workflow conclusion"
    )
    writes = tmp_path / "writes"
    result = subprocess.run(
        ["bash", "-c", 'gh() { printf "%s\\n" "$@" > "$STATUS_WRITES"; }\n' + publish],
        env={
            **os.environ,
            "HEAD_SHA": "a" * 40,
            "CPU_RESULT": cpu,
            "RUN_GPU": "true",
            "GPU_RESULT": "failure",
            "GPU_FAILURE_CLASS": category,
            "OWNER_RELEASE_CONFIRMED": "true",
            "BACKSTOP_RELEASE_CONFIRMED": "",
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
            "STATUS_CONTEXT": "Stable Community CI",
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
        assert "context=Stable Community CI" in arguments
    else:
        assert not calls.exists()


@pytest.mark.parametrize("dual_run", ["", "false", "true"])
def test_manual_dev_branch_obeys_the_comparison_switch(tmp_path, dual_run):
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
            "CI_BRANCH": "ci/developer",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = '["stable","dev"]' if dual_run == "true" else '["dev"]'
    assert output.read_text() == f"lanes={expected}\n"


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
            "STATUS_CONTEXT": "Stable Community CI",
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
      printf 'provision\t%s\n' "$*" >> "$TRACE"
      printf 'instance_name=%s\n' "$RESERVED_INSTANCE" >> "$GITHUB_OUTPUT"
      printf '{"name":"%s","instance_id":"test123"}\n' "$RESERVED_INSTANCE" > "$RUNNER_TEMP/trtmc-gpu-ci-lease.json"
      return "$PROVISION_EXIT"
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
            "GPU_TYPE": "L40S",
            "GPU_PROVIDER": "auto",
            "HOST_RAM_GIB": "64",
            "RESERVED_INSTANCE": "trtmc-gpu-ci-123-2",
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
        "RUNNER_TEMP": str(tmp_path),
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": '["alpha"]',
        "TRTMC_GPU_DIRECT_FAMILIES": '["alpha"]',
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
        "REGISTRY_TOKEN": "dedicated-test-reader-token",
    }
    return repository, catalog, environment


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
    assert cleanup["env"]["INSTANCE_NAME"] == "${{ steps.reserve.outputs.instance_name }}"
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
    reserve = next(step for step in gpu_job["steps"] if step.get("id") == "reserve")
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
    assert argv[:3] == ["-m", "tools.brev_exec", "provision"]
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
def test_independent_cleanup_verifies_the_single_instance_and_propagates_failure(
    tmp_path, gpu_job, cleanup_exit
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    cleanup = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup["if"]
    assert "provision-and-test" in cleanup["needs"]
    job = {"steps": cleanup["steps"]}
    harness = JobHarness(tmp_path, job)
    result = harness.run(
        "Delete the deterministic GPU instance",
        VM_CLEANUP_EXIT=str(cleanup_exit),
        INSTANCE_NAME="trtmc-gpu-ci-123-1",
        INSTANCE_TYPE="g6.4xlarge",
        ORGANIZATION_ID="org-original",
        ALLOCATION_REQUESTED="true",
        OWNER_CLEANUP_CONFIRMED="",
    )
    assert result.returncode == cleanup_exit, result.stdout + result.stderr
    assert harness.events("cleanup") == [
        "-m tools.brev_exec cleanup --instance trtmc-gpu-ci-123-1 "
        f"--lease-file {tmp_path}/gpu-ci-lease/trtmc-gpu-ci-lease.json --until-deleted",
    ]
    assert not harness.events("brev")
    assert not harness.events("provision")
    assert not harness.events("coordinate")
    recovered = json.loads((tmp_path / "gpu-ci-lease/trtmc-gpu-ci-lease.json").read_text())
    assert recovered["name"] == "trtmc-gpu-ci-123-1"
    assert recovered["organization_id"] == "org-original"
    assert recovered["create_started"] and recovered["allocation_pending"]


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


@pytest.mark.parametrize(
    "name,requested,confirmed,expected",
    [
        ("", "true", "", 75),
        ("trtmc-gpu-ci-999-1", "true", "", 1),
        ("trtmc-gpu-ci-123-1", "", "", 1),
        ("trtmc-gpu-ci-123-1", "true", "true", 0),
    ],
)
def test_backstop_never_guesses_an_allocation_from_the_current_attempt(
    tmp_path, name, requested, confirmed, expected
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    harness = JobHarness(tmp_path, workflow["jobs"]["cleanup"])
    result = harness.run(
        "Delete the deterministic GPU instance",
        INSTANCE_NAME=name,
        INSTANCE_TYPE="g6.4xlarge",
        ALLOCATION_REQUESTED=requested,
        OWNER_CLEANUP_CONFIRMED=confirmed,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert not harness.events("cleanup")
    assert not harness.events("provision")


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
    assert names.index("Reserve a GPU instance") < names.index("Preserve the GPU instance lease")
    assert names.index("Preserve the GPU instance lease") < names.index(
        "Build the GPU image and validate the exact PR merge"
    )
    assert names.index("Always tear down the GPU instance") < names.index(
        "Preserve the final GPU instance lease"
    )
    assert names.index("Always tear down the GPU instance") < names.index(
        "Record the step conclusion"
    )
    for name in ("Preserve the GPU instance lease", "Preserve the final GPU instance lease"):
        step = steps[name]
        assert "always()" in step["if"]
        assert step["with"]["path"] == "${{ runner.temp }}/trtmc-gpu-ci-lease.json"
        assert step["with"]["name"] == "${{ steps.reserve.outputs.lease_artifact_name }}"
        assert step["with"]["if-no-files-found"] == "error"
    assert steps["Preserve the final GPU instance lease"]["with"]["overwrite"] is True
    cleanup = workflow["jobs"]["cleanup"]
    assert gpu_job["timeout-minutes"] == 360
    assert cleanup["timeout-minutes"] == 360
    assert cleanup["permissions"] == {"contents": "read", "actions": "read"}
    cleanup_steps = {step["name"]: step for step in cleanup["steps"]}
    assert cleanup_steps["Check out trusted GPU cleanup"]["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }
    download = cleanup_steps["Recover the GPU instance lease"]
    assert download["continue-on-error"] is True
    assert download["timeout-minutes"] == 1
    assert 'gh run download "$GITHUB_RUN_ID"' in download["run"]
    assert download["env"]["LEASE_ARTIFACT_NAME"] == (
        "${{ needs.provision-and-test.outputs.lease_artifact_name }}"
    )
    assert '--name "$LEASE_ARTIFACT_NAME"' in download["run"]
    assert "GITHUB_RUN_ATTEMPT" not in download["run"]
    delete = cleanup_steps["Delete the deterministic GPU instance"]
    assert "always()" in delete["if"]
    assert "steps.cleanup-checkout.outcome == 'success'" in delete["if"]
    assert "steps.cleanup-login.outcome == 'success'" in delete["if"]
    assert "--until-deleted" in delete["run"]
    assert "|| true" not in delete["run"]
    assert "-r2" not in delete["run"] and "-r3" not in delete["run"]

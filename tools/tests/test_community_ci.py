# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CI entrypoint."""

from __future__ import annotations

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
        "(github.event_name == 'workflow_dispatch' && inputs.source_snapshot != '')) }}"
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
    assert jobs["provision-and-test"]["needs"] == ["gpu-authorize", "announce"]
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in jobs["provision-and-test"]["if"]
    assert jobs["publish"]["needs"] == [
        "authorize",
        "gpu-authorize",
        "announce",
        "provision-and-test",
        "cleanup",
    ]
    assert "allow-unsafe-pr-checkout" not in json.dumps(workflow)
    assert workflow["env"]["COMMUNITY_GPU_EXECUTION_ENABLED"] == "false"
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
def test_gpu_step_conclusion_requires_completed_success(
    tmp_path: Path,
    job_status: str,
    test_outcome: str,
    test_conclusion: str,
    expected: str,
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
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == f"conclusion={expected}\n"


@pytest.mark.parametrize(
    ("run_gpu", "job_result", "conclusion", "cleanup_result", "expected_state"),
    [
        ("false", "skipped", "", "skipped", "success"),
        ("false", "success", "success", "success", "failure"),
        ("true", "success", "success", "success", "success"),
        ("true", "success", "success", "failure", "failure"),
        ("true", "failure", "success", "success", "failure"),
        ("true", "cancelled", "success", "success", "failure"),
        ("true", "skipped", "", "skipped", "failure"),
        ("true", "success", "failure", "success", "failure"),
        ("true", "success", "cancelled", "success", "failure"),
        ("true", "success", "", "success", "failure"),
    ],
)
def test_gpu_published_status_requires_job_and_test_success(
    tmp_path: Path,
    run_gpu: str,
    job_result: str,
    conclusion: str,
    cleanup_result: str,
    expected_state: str,
) -> None:
    output = tmp_path / "status-args"
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$STATUS_ARGS"\n', encoding="utf-8")
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script("community-ci.yml", "publish", "Publish the terminal status"),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "STATUS_ARGS": str(output),
            "RUN_GPU": run_gpu,
            "JOB_RESULT": job_result,
            "CONCLUSION": conclusion,
            "CLEANUP_RESULT": cleanup_result,
            "GITHUB_REPOSITORY": "example/model-connect",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "123",
            "HEAD_SHA": "a" * 40,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if expected_state == "success" else 1), (
        result.stdout + result.stderr
    )
    assert f"state={expected_state}" in output.read_text(encoding="utf-8").splitlines()


def test_gpu_status_and_cleanup_fail_closed() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/community-ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["provision-and-test"]
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Reserve a GPU instance"]["id"] == "reserve"
    test_step = steps["Build the GPU image, check out the exact PR merge, and run the smoke test"]
    assert "sudo docker build -f Dockerfile.dev.x86-gpu" in test_step["run"]
    assert "sudo docker run --rm --gpus all" in test_step["run"]
    result = steps["Record the step conclusion"]
    assert result["id"] == "result"
    assert result["if"] == "always()"
    assert result["env"] == {
        "JOB_STATUS": "${{ job.status }}",
        "TEST_OUTCOME": "${{ steps.test.outcome }}",
        "TEST_CONCLUSION": "${{ steps.test.outputs.conclusion }}",
    }
    assert "${{" not in result["run"]
    cleanup = steps["Always tear down the GPU instance"]
    assert cleanup["if"] == "${{ always() && steps.reserve.outputs.instance_name != '' }}"
    assert cleanup["env"] == {"INSTANCE_NAME": "${{ steps.reserve.outputs.instance_name }}"}
    assert cleanup["run"] == 'brev delete "$INSTANCE_NAME" || true'
    assert job["outputs"] == {"conclusion": "${{ steps.result.outputs.conclusion }}"}
    cleanup_job = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup_job["if"]
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in cleanup_job["if"]
    cleanup_steps = {step["name"]: step for step in cleanup_job["steps"]}
    assert cleanup_steps["Delete the deterministic GPU instance"]["run"] == (
        'brev delete "trtmc-gpu-ci-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" || true'
    )
    publish = workflow["jobs"]["publish"]["steps"][0]
    assert publish["env"]["RUN_GPU"] == "${{ needs.gpu-authorize.outputs.run_gpu }}"
    assert publish["env"]["JOB_RESULT"] == "${{ needs.provision-and-test.result }}"
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


@pytest.mark.parametrize("create_exitcode", [0, 1])
def test_gpu_cleanup_can_delete_instance_after_reservation_failure(
    tmp_path: Path,
    create_exitcode: int,
) -> None:
    output = tmp_path / "output"
    calls = tmp_path / "brev-calls"
    brev = tmp_path / "brev"
    brev.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$BREV_CALLS"\n'
        'if [ "$1" = "create" ]; then exit "$CREATE_EXITCODE"; fi\n',
        encoding="utf-8",
    )
    brev.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "BREV_CALLS": str(calls),
        "CREATE_EXITCODE": str(create_exitcode),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "provision-and-test", "Reserve a GPU instance"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == create_exitcode, result.stdout + result.stderr
    instance_name = "trtmc-gpu-ci-123-2"
    assert output.read_text(encoding="utf-8") == f"instance_name={instance_name}\n"
    cleanup = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
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
        f"create {instance_name} -g L40 --timeout 600",
        f"delete {instance_name}",
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
        == "${{ matrix.lane == 'stable' && 'main' || vars.TRTMC_COMMUNITY_CI_DEV_REF || 'main' }}"
    )
    assert "sleep" not in step["run"]
    gpu = executor["jobs"]["provision-and-test"]
    test = next(step for step in gpu["steps"] if step.get("id") == "test")
    # Stable retains the existing manual GPU implementation. Dev carries the
    # automatic public-only GPU experiment as a separate branch commit.
    assert test["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"
    assert gpu["environment"]["name"] == "gpu-ci-dispatch"
    assert gpu["concurrency"]["cancel-in-progress"] is True


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
@pytest.mark.parametrize("event_name", ["pull_request_target", "workflow_dispatch"])
def test_community_dual_run_switch_controls_job_allocation(
    tmp_path, dual_run, expected_lanes, event_name
):
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
        env={
            **os.environ,
            "DUAL_RUN": dual_run,
            "CI_ENTRY_REF": "refs/heads/main",
            "EVENT_NAME": event_name,
            "REQUESTED_LANE": "stable" if event_name == "workflow_dispatch" else "",
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


@pytest.mark.parametrize("lane,ref", [("stable", "main"), ("dev", "main"), ("dev", "ci/developer")])
def test_community_lane_dispatch_preserves_snapshot_and_request_identity(tmp_path, lane, ref):
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
    if expected is None:
        assert not (tmp_path / "calls").exists()
    else:
        calls = (tmp_path / "calls").read_text().splitlines()
        assert f"state={expected}" in calls
        assert f"context={lane.title()} Community CI" in calls
        assert (tmp_path / "output").read_text() == "reported=true\n"
        assert int((tmp_path / "counter").read_text()) == (2 if queued_first else 1)


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
            "pr_number": "17",
            "head_sha": head,
            "base_sha": base,
            "merge_sha": merge,
        }
    else:
        assert not output.exists()


def test_stable_pr_cpu_runs_without_a_cutover_or_internal_bridge_change(tmp_path):
    head, base, merge = (value * 40 for value in "abc")
    fake = tmp_path / "gh"
    fake.write_text(
        "#!/usr/bin/env python3\nimport base64,os,sys\n"
        "if any('/contents/' in arg for arg in sys.argv):\n"
        " if os.environ['DISPATCHER']=='api-error': sys.exit(1)\n"
        " text='# Community CI branch dispatch v1' if os.environ['DISPATCHER']=='true' else 'name: Community CI'\n"
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
def test_main_manual_dev_request_never_allocates_a_stable_publisher(tmp_path, dual_run):
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
            "CI_ENTRY_REF": "refs/heads/main",
            "EVENT_NAME": "workflow_dispatch",
            "REQUESTED_LANE": "dev",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'lanes=["dev"]\n'


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


@pytest.mark.parametrize("stable_state", ["failure", "pending"])
def test_manual_dev_preserves_stable_gpu_verdict_despite_successful_cpu(tmp_path, stable_state):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/community-ci.yml").read_text())
    steps = {step["name"]: step for step in workflow["jobs"]["dispatch"]["steps"]}
    head, merge = "a" * 40, "c" * 40
    states = tmp_path / "states.json"
    states.write_text(json.dumps({"Stable Community CI": stable_state}))
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
        "CI_ENTRY_REF": "refs/heads/main",
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
            "STATUS_CONTEXT": f"{lane.title()} Community CI",
            "GITHUB_OUTPUT": str(output),
        }
        # Also execute against the pre-fix workflow to reproduce the original
        # bad status write rather than merely checking for a new guard's text.
        if "Authorize the result publisher" in steps:
            execute(steps["Authorize the result publisher"]["run"], lane_env)
        execute(steps["Mark the selected CI pending"]["run"], lane_env)
        execute(steps["Dispatch the selected Community CI implementation"]["run"], lane_env)
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        execute(
            steps["Publish the complete workflow conclusion"]["run"],
            {
                **lane_env,
                "PIPELINE_RUN_ID": outputs["run_id"],
                "CI_BRANCH": outputs["ci_ref"],
                "EXISTING_STABLE": outputs.get("existing_stable", ""),
            },
        )
    assert json.loads(states.read_text()) == {
        "Stable Community CI": stable_state,
        "Dev Community CI": "success",
    }
    for arguments in map(json.loads, calls.read_text().splitlines()):
        assert "context=Stable Community CI" not in arguments


@pytest.mark.parametrize(
    "lane,entry_ref,context,allowed",
    [
        ("stable", "refs/heads/main", "Stable Community CI", True),
        ("dev", "refs/heads/main", "Dev Community CI", True),
        ("dev", "refs/heads/ci/developer", "Dev Community CI", False),
        ("stable", "refs/heads/ci/developer", "Stable Community CI", False),
        ("stable", "refs/tags/main", "Stable Community CI", False),
        ("stable", "refs/pull/17/merge", "Stable Community CI", False),
        ("stable", "refs/heads/main", "Dev Community CI", False),
        ("dev", "refs/heads/ci/developer", "Stable Community CI", False),
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
        assert "if" not in step
    assert "failure()" in steps[-1]["if"]
    assert "steps.publisher.outputs.authorized == 'true'" in steps[-1]["if"]
    reporter = workflow["jobs"]["snapshot"]["steps"][-1]
    assert reporter["env"]["STATUS_CONTEXT"] == (
        "${{ steps.lanes.outputs.lanes == '[\"dev\"]' && 'Dev Community CI' || 'Stable Community CI' }}"
    )


@pytest.mark.parametrize("entry_ref", ["refs/heads/ci/developer", "refs/tags/main"])
def test_non_main_coordinator_is_rejected_before_selecting_lanes(tmp_path, entry_ref):
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

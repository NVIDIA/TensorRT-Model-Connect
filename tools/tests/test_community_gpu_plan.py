# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise frozen Git-object planning before cloud allocation."""

import json
import subprocess

import pytest

from tools import community_gpu_ci as ci


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def owner(repo, family="alpha", deferred=False):
    root = repo / "families" / family
    (root / "tests/manifests").mkdir(parents=True)
    (root / "model.py").write_text("raise RuntimeError('model must not import during planning')\n")
    (root / "tests/test_e2e.py").write_text(
        "raise RuntimeError('tests must not import during planning')\n"
    )
    (root / "tests/manifests/smoke.json").write_text(
        json.dumps(
            {
                "family": family,
                "hf_id": "example/" + family,
                "hf_revision": "a" * 40,
                "testcases": [
                    {"name": family + "-smoke", "premerge": True, "community_gpu": not deferred}
                ],
            }
        )
    )


def commit(repo):
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.test",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "static fixture",
    )
    return git(repo, "rev-parse", "HEAD")


def environment(family="alpha"):
    return {
        "TRTMC_GPU_SCOPE": "families",
        "TRTMC_GPU_FAMILIES": json.dumps([family]),
        "TRTMC_GPU_DIRECT_FAMILIES": json.dumps([family]),
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
    }


def repository(tmp_path, deferred=False):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init", "-q")
    owner(repo, deferred=deferred)
    lock = repo / "requirements/community-gpu-linux-amd64.lock"
    lock.parent.mkdir()
    lock.write_text("huggingface-hub==1.33.0\n")
    return repo


def test_static_plan_uses_git_blobs_without_any_pr_import(tmp_path, monkeypatch):
    repo = repository(tmp_path)
    poison = repo / "tools"
    poison.mkdir()
    sentinel = tmp_path / "imported"
    (poison / "__init__.py").write_text(
        f"from pathlib import Path; Path({str(sentinel)!r}).touch()\n"
    )
    sha = commit(repo)
    (repo / "families/alpha/tests/manifests/smoke.json").write_text("uncommitted invalid data")
    actual = ci.family_plan
    observed = []

    def inspect_shadow(root, family):
        files = list(root.rglob("*"))
        assert all(
            path.is_dir() or path.suffix == ".json" or path.name in {"model.py", "test_e2e.py"}
            for path in files
        )
        assert (root / f"families/{family}/model.py").read_bytes() == b""
        assert (root / f"families/{family}/tests/test_e2e.py").read_bytes() == b""
        observed.append(family)
        return actual(root, family)

    monkeypatch.setattr(ci, "family_plan", inspect_shadow)
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, environment(), path)
    plan = json.loads(path.read_text())
    assert not sentinel.exists() and observed == ["alpha"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert plan["source_sha"] == sha and plan["source_tree"] == git(
        repo, "rev-parse", sha + "^{tree}"
    )
    assert plan["active_families"] == ["alpha"]
    assert plan["families"] == [
        {
            "family": "alpha",
            "cases": ["alpha-smoke"],
            "deferred_cases": [],
            "checkpoints": [{"repo_id": "example/alpha", "revision": "a" * 40}],
        }
    ]


def test_frozen_plan_contains_effective_baseline_and_no_deferred_checkpoint(tmp_path):
    repo = repository(tmp_path, deferred=True)
    for family in ci.SHARED_SMOKE_FAMILIES:
        owner(repo, family)
    sha = commit(repo)
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, environment(), path)
    plans, active = ci.verify_gpu_plan(repo, environment(), path)
    assert active == ci.SHARED_SMOKE_FAMILIES and len(plans) == 6
    assert plans["alpha"].checkpoints == () and plans["alpha"].testcases == ()
    assert plans["alpha"].deferred_testcases == ("alpha-smoke",)
    assert all(plans[family].testcases == (family + "-smoke",) for family in active)
    assert ci.execution_budget_seconds(environment(), path) == min(
        ci.MAX_EXECUTION_SECONDS,
        len(active)
        * (ci.FAMILY_PREPARATION_SECONDS + ci.STAGING_TIMEOUT_SECONDS + ci.FAMILY_TIMEOUT_SECONDS),
    )


@pytest.mark.parametrize(
    "kind", ["model_missing", "model_link", "manifest_link", "no_selected_cases"]
)
def test_invalid_static_owner_fails_before_any_execution(tmp_path, kind):
    repo = repository(tmp_path)
    model = repo / "families/alpha/model.py"
    manifest = repo / "families/alpha/tests/manifests/smoke.json"
    if kind == "model_missing":
        model.unlink()
    elif kind == "model_link":
        model.unlink()
        model.symlink_to("tests/test_e2e.py")
    elif kind == "manifest_link":
        target = manifest.with_suffix(".txt")
        manifest.rename(target)
        manifest.symlink_to(target.name)
    else:
        data = json.loads(manifest.read_text())
        data["testcases"][0]["premerge"] = False
        manifest.write_text(json.dumps(data))
    sha = commit(repo)
    path = tmp_path / "plan.json"
    with pytest.raises(ci.CommunityGpuError):
        ci.export_gpu_plan(repo, sha, environment(), path)
    assert not path.exists()


@pytest.mark.parametrize("mutation", ["head", "dirty", "cases", "checkpoint", "tree", "selection"])
def test_vm_rejects_drift_from_frozen_source_and_selection(tmp_path, mutation):
    repo = repository(tmp_path)
    sha = commit(repo)
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, environment(), path)
    assert ci.verify_gpu_plan(repo, environment(), path)[1] == ("alpha",)
    data = json.loads(path.read_text())
    if mutation in {"head", "dirty"}:
        (repo / "README.md").write_text("modified tracked source\n")
        if mutation == "head":
            commit(repo)
        else:
            git(repo, "add", "README.md")
    elif mutation == "cases":
        data["families"][0]["cases"] = ["easy"]
    elif mutation == "checkpoint":
        data["families"][0]["checkpoints"][0]["revision"] = "b" * 40
    elif mutation == "tree":
        data["source_tree"] = "b" * 40
    else:
        data["selection"]["scope"] = "all"
    path.write_text(json.dumps(data))
    with pytest.raises(ci.CommunityGpuError):
        ci.verify_gpu_plan(repo, environment(), path)


@pytest.mark.parametrize(
    "lock", ["huggingface-hub>=1.33.0\n", "huggingface-hub==1.33.0\nhuggingface-hub==0.36.0\n"]
)
def test_staging_hub_requires_one_exact_trusted_pin(tmp_path, lock):
    repo = repository(tmp_path)
    (repo / "requirements/community-gpu-linux-amd64.lock").write_text(lock)
    sha = commit(repo)
    with pytest.raises(ci.CommunityGpuError, match="exactly one"):
        ci.staging_hub_requirement(repo, sha)


def test_staging_hub_ignores_working_tree_override(tmp_path):
    repo = repository(tmp_path)
    sha = commit(repo)
    (repo / "requirements/community-gpu-linux-amd64.lock").write_text("huggingface-hub==0.36.0\n")
    assert ci.staging_hub_requirement(repo, sha) == "huggingface-hub==1.33.0"


@pytest.mark.parametrize("prepared", [False, True])
def test_verified_candidate_skips_rebuilding_dependencies_but_still_stages_assets(
    tmp_path, monkeypatch, prepared
):
    """Qualification reuses the candidate while retaining pre-entry asset admission."""
    from types import SimpleNamespace

    repo = repository(tmp_path)
    commit(repo)
    image = "sha256:" + "a" * 64
    built = []
    staged = []
    original_run = subprocess.run

    def prepare(root, family, base, deadline):
        assert root == repo and base == image
        built.append(family)
        return image

    def transport(command, **options):
        if command[0] == "git":
            return original_run(command, **options)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=image)
        if "--stage-family" in command:
            staged.append(command[command.index("--stage-family") + 1])
            return subprocess.CompletedProcess(command, 7)
        pytest.fail(f"Unexpected command before asset admission: {command}")

    monkeypatch.setattr(
        ci, "_image_preparation", lambda: SimpleNamespace(ensure_family_image=prepare)
    )
    monkeypatch.setattr(ci.subprocess, "run", transport)
    env = environment()
    env["TRTMC_GPU_RESULTS_DIR"] = str(tmp_path / "results")
    with pytest.raises(ci.CommunityGpuError, match="checkpoint staging exited 7"):
        ci.run_containers(repo, env, "verified-candidate", dependencies_prepared=prepared)
    assert built == ([] if prepared else ["alpha"])
    assert staged == ["alpha"]
    assert json.loads((repo / "impact.json").read_text())["families"] == ["alpha"]
    result = json.loads((tmp_path / "results/summary.json").read_text())["families"][0]
    assert result["failure_class"] == "infra_failure"
    assert result["entrypoint_started"] is False

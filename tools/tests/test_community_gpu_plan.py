# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise frozen Git-object planning before cloud allocation."""

import json
import subprocess
import sys
from pathlib import Path

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


def test_new_owner_frozen_plan_retains_every_premerge_case_without_baselines(tmp_path):
    repo = repository(tmp_path)
    manifest = repo / "families/alpha/tests/manifests/smoke.json"
    data = json.loads(manifest.read_text())
    data["testcases"].extend(
        [
            {"name": "alpha-second", "premerge": True},
            {"name": "alpha-nightly", "premerge": False},
        ]
    )
    manifest.write_text(json.dumps(data))
    sha = commit(repo)
    selection = {
        **environment(),
        "TRTMC_GPU_FAMILIES": "[]",
        "TRTMC_GPU_DIRECT_FAMILIES": "[]",
        "TRTMC_GPU_ADDED_FAMILIES": '["alpha"]',
    }
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, selection, path)
    plans, active = ci.verify_gpu_plan(repo, selection, path)
    assert active == ("alpha",) and tuple(plans) == ("alpha",)
    assert plans["alpha"].testcases == ("alpha-second", "alpha-smoke")
    assert plans["alpha"].checkpoints == (("example/alpha", "a" * 40),)


@pytest.mark.parametrize("added", [False, True])
def test_frozen_plan_contains_effective_baseline_and_no_deferred_checkpoint(tmp_path, added):
    repo = repository(tmp_path, deferred=True)
    for family in ci.SHARED_SMOKE_FAMILIES:
        owner(repo, family)
    sha = commit(repo)
    path = tmp_path / "plan.json"
    selection = environment()
    if added:
        selection.update(
            TRTMC_GPU_FAMILIES="[]",
            TRTMC_GPU_DIRECT_FAMILIES="[]",
            TRTMC_GPU_ADDED_FAMILIES='["alpha"]',
        )
    ci.export_gpu_plan(repo, sha, selection, path)
    plans, active = ci.verify_gpu_plan(repo, selection, path)
    assert active == ci.SHARED_SMOKE_FAMILIES and len(plans) == 6
    assert plans["alpha"].checkpoints == () and plans["alpha"].testcases == ()
    assert plans["alpha"].deferred_testcases == ("alpha-smoke",)
    assert all(plans[family].testcases == (family + "-smoke",) for family in active)
    assert ci.execution_budget_seconds(selection, path) == min(
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


def external_assets(repo, *, value=None):
    manifest = repo / "families/alpha/tests/manifests/smoke.json"
    data = json.loads(manifest.read_text())
    data["external_files"] = (
        [{"path": "weights.bin", "url": "https://example.test/weights.bin"}]
        if value is None
        else value
    )
    manifest.write_text(json.dumps(data))


def asset_recipe(repo):
    recipe = repo / "families/alpha/ci/Dockerfile.dependencies"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    # This must remain inert during plan admission, even in the secret-bearing job.
    recipe.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nRUN exit 97\n")
    return recipe


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory", "oversize"])
def test_selected_external_assets_require_a_regular_family_recipe(tmp_path, kind):
    repo = repository(tmp_path)
    external_assets(repo)
    recipe = asset_recipe(repo)
    if kind == "missing":
        recipe.unlink()
    elif kind == "symlink":
        target = recipe.with_suffix(".real")
        recipe.rename(target)
        recipe.symlink_to(target.name)
    elif kind == "directory":
        recipe.unlink()
        recipe.mkdir()
    else:
        recipe.write_bytes(b"#" * (ci.MAX_DEPENDENCY_CATALOG_BYTES + 1))
    sha = commit(repo)
    path = tmp_path / "plan.json"
    with pytest.raises(ci.CommunityGpuError, match="external_files.*Dockerfile.dependencies"):
        ci.family_plan(repo, "alpha")
    with pytest.raises(ci.CommunityGpuError, match="external_files.*Dockerfile.dependencies"):
        ci.export_gpu_plan(repo, sha, environment(), path)
    assert not path.exists()


def test_external_asset_recipe_parent_cannot_alias_another_owner(tmp_path):
    repo = repository(tmp_path)
    external_assets(repo)
    recipe = asset_recipe(repo)
    source = recipe.parent
    target = repo / "families/other/ci"
    target.parent.mkdir()
    source.rename(target)
    source.symlink_to("../other/ci", target_is_directory=True)
    assert (source / recipe.name).is_file()
    sha = commit(repo)
    with pytest.raises(ci.CommunityGpuError, match="external_files.*Dockerfile.dependencies"):
        ci.family_plan(repo, "alpha")
    with pytest.raises(ci.CommunityGpuError, match="external_files.*Dockerfile.dependencies"):
        ci.export_gpu_plan(repo, sha, environment(), tmp_path / "plan.json")


@pytest.mark.parametrize("declaration", ["weights.bin", {}, False])
def test_selected_external_asset_declaration_must_be_a_list(tmp_path, declaration):
    repo = repository(tmp_path)
    external_assets(repo, value=declaration)
    asset_recipe(repo)
    sha = commit(repo)
    with pytest.raises(ci.CommunityGpuError, match="external_files must be a list"):
        ci.export_gpu_plan(repo, sha, environment(), tmp_path / "plan.json")


def test_frozen_external_asset_admission_uses_source_recipe_not_working_tree(tmp_path):
    repo = repository(tmp_path)
    external_assets(repo)
    missing_sha = commit(repo)
    recipe = asset_recipe(repo)
    # An untracked working-tree recipe cannot repair an immutable source plan.
    with pytest.raises(ci.CommunityGpuError, match="external_files.*Dockerfile.dependencies"):
        ci.export_gpu_plan(repo, missing_sha, environment(), tmp_path / "missing.json")
    ready_sha = commit(repo)
    original = recipe.read_bytes()
    recipe.unlink()
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, ready_sha, environment(), path)
    assert json.loads(path.read_text())["active_families"] == ["alpha"]
    recipe.write_bytes(original)
    plans, active = ci.verify_gpu_plan(repo, environment(), path)
    assert active == ("alpha",) and plans["alpha"].testcases == ("alpha-smoke",)
    recipe.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    with pytest.raises(ci.CommunityGpuError):
        ci.verify_gpu_plan(repo, environment(), path)


def test_external_asset_plan_runs_in_the_standalone_container_wrapper(tmp_path):
    repo = repository(tmp_path)
    external_assets(repo)
    asset_recipe(repo)
    # The family container receives only this wrapper, not its host-only sibling.
    wrapper = tmp_path / "community_gpu_ci.py"
    wrapper.write_bytes(Path(ci.__file__).read_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; from pathlib import Path; "
            "runner = runpy.run_path(sys.argv[1]); "
            "print(runner['family_plan'](Path(sys.argv[2]), 'alpha').testcases)",
            str(wrapper),
            str(repo),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "('alpha-smoke',)"


@pytest.mark.parametrize("selection", ["other_owner", "nightly", "deferred", "empty"])
def test_unselected_or_empty_external_assets_do_not_require_a_recipe(tmp_path, selection):
    repo = repository(tmp_path, deferred=selection == "deferred")
    external_assets(repo, value=[] if selection == "empty" else None)
    chosen = environment()
    if selection == "other_owner":
        owner(repo, "beta")
        chosen = environment("beta")
    elif selection == "nightly":
        path = repo / "families/alpha/tests/manifests/smoke.json"
        data = json.loads(path.read_text())
        data["testcases"][0]["premerge"] = False
        path.write_text(json.dumps(data))
        (path.parent / "other.json").write_text(json.dumps({
            "family": "alpha", "testcases": [{"name": "alpha-other", "premerge": True}],
        }))
    elif selection == "deferred":
        for family in ci.SHARED_SMOKE_FAMILIES:
            owner(repo, family)
    sha = commit(repo)
    path = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, chosen, path)
    plans, active = ci.verify_gpu_plan(repo, chosen, path)
    assert active == (
        ("beta",) if selection == "other_owner"
        else ci.SHARED_SMOKE_FAMILIES if selection == "deferred"
        else ("alpha",)
    )
    assert plans


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


@pytest.mark.parametrize("scope", ["families", "all"])
def test_oversize_plan_fails_before_reading_or_importing_pr_source(tmp_path, monkeypatch, scope):
    changed = sorted(f"owner_{index}" for index in range(86))
    env = {
        "TRTMC_GPU_SCOPE": scope,
        "TRTMC_GPU_FAMILIES": json.dumps(changed),
        "TRTMC_GPU_DIRECT_FAMILIES": json.dumps(changed),
        "TRTMC_GPU_ADDED_FAMILIES": "[]",
    }
    monkeypatch.setattr(ci, "_git_plan", lambda *args: pytest.fail("PR source was accessed"))
    output = tmp_path / "plan.json"
    with pytest.raises(ci.CommunityGpuError, match="split this PR"):
        ci.export_gpu_plan(tmp_path, "a" * 40, env, output)
    assert not output.exists()


@pytest.mark.parametrize("scope", ["families", "all"])
def test_old_frozen_plan_cannot_bypass_changed_family_admission(tmp_path, scope):
    repo = repository(tmp_path)
    sha = commit(repo)
    output = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, environment(), output)
    frozen = json.loads(output.read_text())
    changed = sorted(f"owner_{index}" for index in range(86))
    frozen["selection"] = {
        "scope": scope,
        "selected_families": changed,
        "families": changed,
        "direct_families": changed,
        "added_families": [],
    }
    output.write_text(json.dumps(frozen))
    with pytest.raises(ci.CommunityGpuError, match="split this PR"):
        ci._read_gpu_plan(output)


def test_frozen_active_families_cannot_expand_beyond_admitted_selection(tmp_path):
    repo = repository(tmp_path)
    sha = commit(repo)
    output = tmp_path / "plan.json"
    ci.export_gpu_plan(repo, sha, environment(), output)
    frozen = json.loads(output.read_text())
    frozen["active_families"] = sorted(f"owner_{index}" for index in range(86))
    output.write_text(json.dumps(frozen))
    with pytest.raises(ci.CommunityGpuError, match="active families"):
        ci._read_gpu_plan(output)

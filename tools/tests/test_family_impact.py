# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import test_impact


def _repo(tmp_path: Path) -> Path:
    for family in ("alpha", "beta"):
        root = tmp_path / "families" / family
        (root / "runtime").mkdir(parents=True)
        (root / "tests/manifests").mkdir(parents=True)
        (root / "model.py").write_text("def build(request, writer):\n    pass\n")
        (root / "support.py").write_text("def describe(metadata):\n    return None\n")
        (root / "runtime/CMakeLists.txt").write_text("# owned\n")
        (root / "runtime/plugin.h").write_text("#pragma once\n")
        (root / "tests/test_e2e.py").write_text("def test_e2e():\n    pass\n")
    return tmp_path


def test_family_path_selects_only_its_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["families/alpha/model.py"])
    assert impact.scope == "families"
    assert impact.families == ("alpha",)


def test_family_runtime_selects_owner_and_cpp(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["families/beta/runtime/plugin.cpp"])
    assert impact.families == ("beta",)


def test_family_requirements_select_only_the_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["families/alpha/requirements.txt"])
    assert impact.scope == "families"
    assert impact.families == ("alpha",)


def test_base_requirements_select_all_families(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["requirements/base.txt"])
    assert impact.scope == "all"
    assert impact.families == ("alpha", "beta")


def test_shared_contract_selects_all_directly(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["core/runtime/include/trtmc/task.h"])
    assert impact.scope == "all"
    assert impact.families == ("alpha", "beta")


def test_families_package_selects_all_directly(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(repo, ["families/__init__.py"])
    assert impact.scope == "all"
    assert impact.families == ("alpha", "beta")


@pytest.mark.parametrize(
    "path",
    [
        ".coderabbit.yaml",
        ".gitignore",
        "ASSET_LICENSES.md",
        "examples/removed.py",
        "plugins/removed/SKILL.md",
        "requirements/community-ci.txt",
        "conftest.py",
        "retired-root/deleted.py",
    ],
)
def test_shared_and_deleted_infrastructure_is_classified(tmp_path: Path, path: str) -> None:
    impact = test_impact.classify(_repo(tmp_path), [path])
    assert impact.scope == "all"
    assert impact.families == ("alpha", "beta")


def test_docs_do_not_select_model_proofs(tmp_path: Path) -> None:
    impact = test_impact.classify(_repo(tmp_path), ["website/docs/architecture/overview.md"])
    assert impact.scope == "docs"
    assert impact.families == ()
    assert impact.run_docs is True


def test_unclassified_path_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    path = repo / "mystery/file.xyz"
    path.parent.mkdir()
    path.write_text("unknown\n")
    with pytest.raises(ValueError, match="unclassified"):
        test_impact.classify(repo, ["mystery/file.xyz"])


def test_validation_requires_no_metadata_registry(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert not (repo / "families/alpha/tests/thresholds").exists()
    test_impact.validate(repo)
    (repo / "families/alpha/MODEL.toml").write_text("id='alpha'\n")
    with pytest.raises(ValueError, match="MODEL.toml"):
        test_impact.validate(repo)


def test_validation_allows_isolated_family_cpp_tests(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = repo / "families/alpha/tests/cpp/test_alpha.cpp"
    source.parent.mkdir()
    source.write_text('#include "families/alpha/runtime/plugin.h"\n')
    test_impact.validate(repo)

    source.write_text('#include "families/beta/runtime/plugin.h"\n')
    with pytest.raises(ValueError, match="imports sibling beta"):
        test_impact.validate(repo)


def test_validation_rejects_central_model_cpp_tests(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    source = repo / "tests/cpp/models/test_alpha.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("int main() {}\n")
    with pytest.raises(ValueError, match="central tests/cpp/models"):
        test_impact.validate(repo)


def test_changed_files_includes_deletions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        output = "base\n" if command[1] == "merge-base" else "families/alpha/old.py\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(test_impact.subprocess, "run", run)
    assert test_impact.changed_files(tmp_path, "main", "HEAD") == ["families/alpha/old.py"]
    assert "--diff-filter=ACMRTD" in calls[1]

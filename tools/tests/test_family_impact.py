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


def test_shared_change_preserves_directly_changed_family(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(
        repo,
        [
            "core/runtime/include/trtmc/task.h",
            "families/alpha/model.py",
        ],
    )

    assert impact.scope == "all"
    assert impact.families == ("alpha", "beta")
    assert impact.direct_families == ("alpha",)


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


def test_asset_license_record_selects_only_the_fixture_owner(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    alone = test_impact.classify(repo, ["ASSET_LICENSES.md"])
    assert alone.scope == "docs"
    assert alone.families == ()

    impact = test_impact.classify(
        repo,
        ["ASSET_LICENSES.md", "families/alpha/tests/data/frame.png"],
    )
    assert impact.scope == "families"
    assert impact.families == ("alpha",)
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


def test_validation_rejects_aiperf_settings_in_a_family(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # Family code whose name starts with aiperf is not AIPerf settings.
    (repo / "families/alpha/tests/aiperf_requests.py").write_text("PACKAGE = None\n")
    test_impact.validate(repo)
    settings = repo / "families/alpha/tests/aiperf/alpha.yaml"
    settings.parent.mkdir()
    settings.write_text("reference: {mode: eager}\n")
    with pytest.raises(ValueError, match="alpha: obsolete tests/aiperf"):
        test_impact.validate(repo)


def test_isolated_application_selects_no_family(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    impact = test_impact.classify(
        repo,
        ["apps/aiperf_qual/config/models/alpha.yaml", "apps/aiperf_qual/trtmc_aiperf_qual/models.py"],
    )
    assert impact.scope == "none"
    assert impact.families == ()
    assert impact.run_core_tests is True

    with_family = test_impact.classify(repo, ["apps/aiperf_qual/README.md", "families/alpha/model.py"])
    assert with_family.scope == "families"
    assert with_family.families == ("alpha",)
    # A sibling application stays shared: family references and examples may import it.
    assert test_impact.classify(repo, ["apps/aiperf_qual/README.md", "apps/benchmark/x.py"]).scope == "all"


@pytest.mark.parametrize(
    "path, source",
    [
        ("families/alpha/tests/test_alpha.py", "from trtmc_aiperf_qual.models import resolve_model\n"),
        ("families/alpha/reference/inputs.py", "CONFIG = 'apps/aiperf_qual/config/models/alpha.yaml'\n"),
        ("core/builder/helper.py", "import trtmc_aiperf_qual\n"),
        ("apps/benchmark/runner.py", "import trtmc_aiperf_qual\n"),
    ],
)
def test_validation_rejects_dependents_of_an_isolated_application(
    tmp_path: Path, path: str, source: str
) -> None:
    repo = _repo(tmp_path)
    app = repo / "apps/aiperf_qual/trtmc_aiperf_qual/models.py"
    app.parent.mkdir(parents=True)
    app.write_text("from trtmc_aiperf_qual.config import CONFIG_ROOT\n")
    dependent = repo / path
    dependent.parent.mkdir(parents=True, exist_ok=True)
    dependent.write_text("VALUE = 1\n")
    test_impact.validate(repo)
    dependent.write_text(source)
    with pytest.raises(ValueError, match=f"{path} depends on isolated apps/aiperf_qual"):
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

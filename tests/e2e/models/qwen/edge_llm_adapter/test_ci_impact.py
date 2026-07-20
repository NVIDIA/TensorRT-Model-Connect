# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for selective Qwen EdgeLLM CI routing."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SELECTOR = Path(__file__).resolve().parent / "ci_impact.py"


def _load_selector():
    name = f"trtmc_qwen_edgellm_ci_impact_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, SELECTOR)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: str, content: str = "base\n") -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _profile(repo: Path, leaf: str) -> None:
    _write(
        repo,
        f"python/tensorrt_model_connect/families/qwen/edge_llm_adapter/{leaf}/IMPLEMENTATION.toml",
        f'[implementation]\nid = "{leaf}"\n',
    )
    _write(repo, f"src/runtime/models/qwen/edge_llm_adapter/{leaf}/CMakeLists.txt")
    _write(repo, f"src/runtime/models/qwen/edge_llm_adapter/{leaf}/adapter.cpp")
    _write(repo, f"tests/e2e/models/qwen/edge_llm_adapter/{leaf}/build_runners.py")
    _write(repo, f"tests/e2e/models/qwen/edge_llm_adapter/{leaf}/test_a100_e2e.py")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _profile(repo, "profile_a")
    _profile(repo, "profile_b")
    _write(repo, "python/tensorrt_model_connect/families/qwen/plugin.py")
    _write(repo, "src/runtime/providers/optimized_runtime_factory.h")
    _write(repo, "docs/readme.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit(repo: Path, relative: str, content: str = "changed\n") -> str:
    _write(repo, relative, content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", f"change {relative}")
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/profile_a/adapter.py",
        "src/runtime/models/qwen/edge_llm_adapter/profile_a/adapter.cpp",
        "tests/e2e/models/qwen/edge_llm_adapter/profile_a/test_runtime_contract.py",
    ),
)
def test_leaf_change_selects_only_that_leaf(
    repository: tuple[Path, str], relative: str
) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(repo, relative)

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "leaf"
    assert result["run"] is True
    assert result["profiles"] == ["profile_a"]
    assert result["matrix"] == {
        "include": [{"scope": "leaf", "profile": "profile_a"}]
    }


def test_two_leaf_changes_select_each_leaf_once(repository: tuple[Path, str]) -> None:
    selector = _load_selector()
    repo, base = repository
    _write(
        repo,
        "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/profile_b/adapter.py",
    )
    head = _commit(
        repo,
        "src/runtime/models/qwen/edge_llm_adapter/profile_a/adapter.cpp",
    )

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "leaf"
    assert result["profiles"] == ["profile_a", "profile_b"]
    assert result["matrix"]["include"] == [
        {"scope": "leaf", "profile": "profile_a"},
        {"scope": "leaf", "profile": "profile_b"},
    ]


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/qwen/plugin.py",
        "tests/e2e/models/qwen/edge_llm_adapter/qualify_a100.py",
        "tests/e2e/models/qwen/edge_llm_adapter/coexistence/test_a100_coexistence.py",
        ".github/workflows/qwen-edgellm-a100.yml",
    ),
)
def test_qwen_common_change_selects_one_all_profile_job(
    repository: tuple[Path, str], relative: str
) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(repo, relative)

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "family"
    assert result["profiles"] == ["profile_a", "profile_b"]
    assert result["matrix"] == {
        "include": [{"scope": "family", "profile": ""}]
    }


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/runtime_provider/orchestrator.py",
        "include/trtmc/pipeline.h",
        "src/runtime/providers/optimized_runtime_factory.h",
        "src/runtime/registry/pipeline_factory.cpp",
        "tests/cpp/test_optimized_runtime_host.cpp",
        "CMakeLists.txt",
    ),
)
def test_provider_or_private_abi_change_selects_broad_contracts(
    repository: tuple[Path, str], relative: str
) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(repo, relative)

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "provider"
    assert result["profiles"] == ["profile_a", "profile_b"]
    assert result["matrix"] == {
        "include": [{"scope": "provider", "profile": ""}]
    }


def test_unrelated_change_uses_a_valid_skipped_matrix(repository: tuple[Path, str]) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(repo, "docs/readme.md")

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "none"
    assert result["run"] is False
    assert result["profiles"] == ["profile_a", "profile_b"]
    assert result["matrix"] == {
        "include": [{"scope": "none", "profile": ""}]
    }


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/qwen/vllm_adapter/profile_x/adapter.py",
        "src/runtime/models/qwen/vllm_adapter/profile_x/adapter.cpp",
        "tests/e2e/models/qwen/vllm_adapter/profile_x/test_contract.py",
    ),
)
def test_different_runtime_adapter_does_not_select_edgellm(
    repository: tuple[Path, str], relative: str
) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(repo, relative)

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "none"
    assert result["run"] is False


def test_deleted_leaf_expands_to_remaining_family(repository: tuple[Path, str]) -> None:
    selector = _load_selector()
    repo, base = repository
    for path in repo.glob("**/edge_llm_adapter/profile_a"):
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "remove profile a")
    head = _git(repo, "rev-parse", "HEAD")

    result = selector.calculate(repo, base, head)

    assert result["mode"] == "family"
    assert result["profiles"] == ["profile_b"]
    assert result["matrix"] == {
        "include": [{"scope": "family", "profile": ""}]
    }


def test_incomplete_profile_fails_closed(repository: tuple[Path, str]) -> None:
    selector = _load_selector()
    repo, base = repository
    head = _commit(
        repo,
        "python/tensorrt_model_connect/families/qwen/edge_llm_adapter/profile_c/IMPLEMENTATION.toml",
    )

    with pytest.raises(selector.ImpactError, match="profile_c.*incomplete"):
        selector.calculate(repo, base, head)


def test_cli_writes_compact_github_outputs(
    repository: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base = repository
    head = _commit(
        repo,
        "tests/e2e/models/qwen/edge_llm_adapter/profile_b/test_runtime_contract.py",
    )
    output = tmp_path / "github-output"

    process = subprocess.run(
        [
            sys.executable,
            SELECTOR,
            "--repo-root",
            repo,
            "--base",
            base,
            "--head",
            head,
            "--github-output",
            output,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(process.stdout)
    assert payload["mode"] == "leaf"
    assert output.read_text(encoding="utf-8").splitlines() == [
        "qwen_edgellm_mode=leaf",
        "qwen_edgellm_run=true",
        'qwen_edgellm_profiles=["profile_b"]',
        'qwen_edgellm_matrix={"include":[{"scope":"leaf","profile":"profile_b"}]}',
    ]

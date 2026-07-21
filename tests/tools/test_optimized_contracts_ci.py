# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for generic model-owned optimized-runtime CI dispatch."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


DISPATCHER = Path(__file__).resolve().parents[2] / "tools/ci/optimized_contracts.py"


def _load_dispatcher():
    name = f"trtmc_optimized_contracts_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, DISPATCHER)
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


def _write(repo: Path, relative: str, content: str = "owned\n") -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _payload(mode: str = "none", profiles: tuple[str, ...] = ("profile_a",)) -> dict:
    selected = list(profiles)
    if mode == "leaf":
        entries = [{"scope": "leaf", "profile": profile} for profile in selected]
    else:
        entries = [{"scope": mode, "profile": ""}]
    return {
        "mode": mode,
        "run": mode != "none",
        "profiles": selected,
        "matrix": {"include": entries},
    }


def _adapter(repo: Path, family: str, name: str) -> str:
    roots = (
        f"python/tensorrt_model_connect/families/{family}/{name}",
        f"src/runtime/models/{family}/{name}",
        f"tests/e2e/models/{family}/{name}",
    )
    _write(repo, f"{roots[0]}/profile_a/IMPLEMENTATION.toml")
    _write(repo, f"{roots[1]}/profile_a/adapter.cpp")
    test_root = roots[2]
    _write(
        repo,
        f"{test_root}/ci_impact.py",
        """#!/usr/bin/env python3
import json
from pathlib import Path
print(Path(__file__).with_name("result.json").read_text(encoding="utf-8"))
""",
    )
    _write(repo, f"{test_root}/ci_run.py", "# model-owned runner\n")
    _write(repo, f"{test_root}/test_ci_impact.py", "# model-owned selector tests\n")
    _write(repo, f"{test_root}/result.json", json.dumps(_payload()) + "\n")
    return test_root


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str, tuple[str, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    roots = (
        _adapter(repo, "model_a", "fast_adapter"),
        _adapter(repo, "model_b", "vendor_adapter"),
    )
    _write(repo, "docs/readme.md")
    _write(repo, ".github/workflows/trtmc-ci.yml")
    _write(repo, "tools/ci/optimized_contracts.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD"), roots


def _select(repo: Path, root: str, payload: dict) -> None:
    _write(repo, f"{root}/result.json", json.dumps(payload) + "\n")


def _commit(repo: Path, message: str = "change") -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def test_leaf_dispatch_points_only_at_the_selected_model_runner(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, second) = repository
    _select(repo, second, _payload("leaf"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["run"] is True
    assert result["matrix"] == {
        "include": [
            {
                "adapter_root": second,
                "runner": f"{second}/ci_run.py",
                "scope": "leaf",
                "profile": "profile_a",
            }
        ]
    }
    assert result["selection"] == [
        {"adapter_root": first, "mode": "none", "profiles": ["profile_a"]},
        {"adapter_root": second, "mode": "leaf", "profiles": ["profile_a"]},
    ]


def test_family_dispatch_does_not_leak_to_a_sibling_model(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, second) = repository
    _select(repo, first, _payload("family"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["matrix"]["include"] == [
        {
            "adapter_root": first,
            "runner": f"{first}/ci_run.py",
            "scope": "family",
            "profile": "",
        }
    ]
    assert not any(second in json.dumps(entry) for entry in result["matrix"]["include"])


def test_provider_dispatch_aggregates_every_model_owned_runner(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, roots = repository
    for root in roots:
        _select(repo, root, _payload("provider"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert [entry["adapter_root"] for entry in result["matrix"]["include"]] == list(roots)
    assert {entry["scope"] for entry in result["matrix"]["include"]} == {"provider"}


def test_dispatcher_change_forces_every_adapter_to_provider_scope(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, roots = repository
    _write(repo, "tools/ci/optimized_contracts.py", "changed\n")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert [entry["adapter_root"] for entry in result["matrix"]["include"]] == list(roots)
    assert {entry["scope"] for entry in result["matrix"]["include"]} == {"provider"}


def test_unrelated_change_returns_a_valid_skipped_matrix(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "docs/readme.md", "changed\n")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["run"] is False
    assert result["matrix"] == {
        "include": [{"adapter_root": "", "runner": "", "scope": "none", "profile": ""}]
    }


def test_nightly_all_mode_dispatches_every_adapter_once(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, revision, roots = repository

    result = dispatcher.calculate(repo, revision, revision, all_adapters=True)

    assert [entry["adapter_root"] for entry in result["matrix"]["include"]] == list(roots)
    assert {entry["scope"] for entry in result["matrix"]["include"]} == {"family"}
    assert {entry["profile"] for entry in result["matrix"]["include"]} == {""}


def test_incomplete_three_root_adapter_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "python/tensorrt_model_connect/families/model_c/new_adapter/adapter.py")
    _write(repo, "tests/e2e/models/model_c/new_adapter/ci_impact.py")
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="lacks ownership roots"):
        dispatcher.calculate(repo, base, head)


def test_missing_model_owned_runner_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    (repo / first / "ci_run.py").unlink()
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="lacks ci_run.py"):
        dispatcher.calculate(repo, base, head)


def test_malformed_selector_output_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    _write(repo, f"{first}/result.json", "not json\n")
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="invalid JSON"):
        dispatcher.calculate(repo, base, head)


def test_cli_writes_only_generic_outputs(repository, tmp_path: Path) -> None:
    repo, base, (_, second) = repository
    _select(repo, second, _payload("leaf"))
    head = _commit(repo)
    output = tmp_path / "github-output"

    process = subprocess.run(
        [
            sys.executable,
            DISPATCHER,
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

    assert json.loads(process.stdout)["run"] is True
    names = [line.split("=", 1)[0] for line in output.read_text().splitlines()]
    assert names == ["run", "matrix", "selection"]


def test_premerge_workflow_contains_only_generic_dispatch_contract() -> None:
    workflow = (DISPATCHER.parents[2] / ".github/workflows/trtmc-ci.yml").read_text(
        encoding="utf-8"
    )
    impact = workflow.split("\n  impact:", 1)[1].split("\n  source-quality:", 1)[0]
    contracts = workflow.split("\n  optimized-runtime-contracts:", 1)[1].split(
        "\n  model-proof:", 1
    )[0]
    required = workflow.split("\n  required:", 1)[1]

    assert "tools/ci/optimized_contracts.py" in impact
    assert "optimized_contracts_matrix:" in impact
    assert (
        "matrix: ${{ fromJSON(needs.impact.outputs.optimized_contracts_matrix) }}"
        in contracts
    )
    assert "ADAPTER_ROOT: ${{ matrix.adapter_root }}" in contracts
    assert "RUNNER: ${{ matrix.runner }}" in contracts
    assert 'test "$RUNNER" = "$ADAPTER_ROOT/ci_run.py"' in contracts
    assert 'python "$RUNNER"' in contracts
    assert not re.search(r"qwen|edge.?llm", workflow, re.IGNORECASE)
    assert "- optimized-runtime-contracts" in required
    assert 'test "$OPTIMIZED_CONTRACT_RESULT" = "success"' in required
    assert 'test "$OPTIMIZED_CONTRACT_RESULT" = "skipped"' in required


def test_nightly_generically_dispatches_every_registered_adapter() -> None:
    workflow = (DISPATCHER.parents[2] / ".github/workflows/nightly.yml").read_text(
        encoding="utf-8"
    )
    inventory = workflow.split("\n  inventory:", 1)[1].split("\n  source-quality:", 1)[0]
    contracts = workflow.split("\n  optimized-runtime-contracts:", 1)[1].split(
        "\n  required:", 1
    )[0]
    required = workflow.split("\n  required:", 1)[1].split("\n  release:", 1)[0]

    assert "tools/ci/optimized_contracts.py" in inventory
    assert "--all" in inventory
    assert "optimized_contracts_matrix:" in inventory
    assert "fail-fast: false" in contracts
    assert (
        "matrix: ${{ fromJSON(needs.inventory.outputs.optimized_contracts_matrix) }}"
        in contracts
    )
    assert "ADAPTER_ROOT: ${{ matrix.adapter_root }}" in contracts
    assert "RUNNER: ${{ matrix.runner }}" in contracts
    assert 'test "$RUNNER" = "$ADAPTER_ROOT/ci_run.py"' in contracts
    assert 'python "$RUNNER"' in contracts
    assert not re.search(r"qwen|edge.?llm", workflow, re.IGNORECASE)
    assert "- optimized-runtime-contracts" in required
    assert 'true) test "$OPTIMIZED_CONTRACT_RESULT" = "success"' in required
    assert 'false) test "$OPTIMIZED_CONTRACT_RESULT" = "skipped"' in required

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for generic, bounded, model-owned optimized-runtime CI."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
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


def _payload(
    mode: str = "none",
    profiles: tuple[str, ...] = ("profile_a",),
    *,
    schema_version: object = 1,
) -> dict[str, object]:
    selected = list(profiles)
    if mode == "leaf":
        entries = [{"scope": "leaf", "profile": profile} for profile in selected]
    else:
        entries = [{"scope": mode, "profile": ""}]
    return {
        "schema_version": schema_version,
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
    _write(repo, ".github/workflows/nightly.yml")
    _write(repo, "tools/ci/optimized_contracts.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    return repo, _git(repo, "rev-parse", "HEAD"), roots


def _select(repo: Path, root: str, payload: dict[str, object]) -> None:
    _write(repo, f"{root}/result.json", json.dumps(payload) + "\n")


def _commit(repo: Path, message: str = "change") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _row(
    kind: str,
    *,
    root: str = "",
    family: str = "",
    scope: str = "none",
    profile: str = "",
    index: int = 0,
    count: int = 1,
) -> dict[str, object]:
    return {
        "kind": kind,
        "adapter_root": root,
        "family": family,
        "scope": scope,
        "profile": profile,
        "shard_index": index,
        "shard_count": count,
    }


def test_leaf_dispatch_points_only_at_the_selected_model_runner(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, second) = repository
    _select(repo, second, _payload("leaf"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["matrix"]["include"] == [
        _row("direct", root=second, scope="leaf", profile="profile_a")
    ]
    assert result["selection"] == [
        {"adapter_root": second, "mode": "leaf", "profile_count": 1}
    ]
    assert first not in json.dumps(result["matrix"])


def test_family_dispatch_does_not_leak_to_a_sibling_model(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, second) = repository
    _select(repo, first, _payload("family"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["matrix"]["include"] == [
        _row("direct", root=first, scope="family")
    ]
    assert second not in json.dumps(result["matrix"])


def test_provider_selector_rows_remain_model_owned(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, roots = repository
    for root in roots:
        _select(repo, root, _payload("provider"))
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert [entry["adapter_root"] for entry in result["matrix"]["include"]] == list(roots)
    assert {entry["kind"] for entry in result["matrix"]["include"]} == {"direct"}
    assert {entry["scope"] for entry in result["matrix"]["include"]} == {"provider"}


def test_dispatcher_change_uses_bounded_provider_shards_and_one_structural_row(
    repository,
) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "tools/ci/optimized_contracts.py", "changed\n")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)
    rows = result["matrix"]["include"]

    assert len(rows) == 2
    assert sum(row["kind"] == "structural" for row in rows) == 1
    shards = [row for row in rows if row["kind"] == "shard"]
    assert {row["scope"] for row in shards} == {"provider"}
    assert {row["shard_count"] for row in shards} == {1}
    assert all(not row["adapter_root"] for row in shards)


def test_unrelated_change_returns_a_valid_skipped_matrix(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "docs/readme.md", "changed\n")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["run"] is False
    assert result["matrix"] == {"include": [_row("none")]}
    assert result["selection"] == []


def test_nightly_all_mode_is_selector_free_and_bounded(repository, monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    repo, revision, _ = repository
    monkeypatch.setattr(
        dispatcher,
        "_select",
        lambda *_args, **_kwargs: pytest.fail("nightly invoked an adapter selector"),
    )

    result = dispatcher.calculate(repo, revision, revision, all_adapters=True)

    rows = result["matrix"]["include"]
    assert len(rows) == 1
    assert {entry["kind"] for entry in rows} == {"shard"}
    assert {entry["scope"] for entry in rows} == {"family"}
    assert all(not entry["adapter_root"] for entry in rows)


def test_incomplete_three_root_adapter_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "python/tensorrt_model_connect/families/model_c/new_adapter/adapter.py")
    _write(repo, "tests/e2e/models/model_c/new_adapter/ci_impact.py")
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="lacks ownership roots"):
        dispatcher.calculate(repo, base, head)


def test_builder_and_runtime_without_test_contracts_fail_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "python/tensorrt_model_connect/families/model_c/new_adapter/adapter.py")
    _write(repo, "src/runtime/models/model_c/new_adapter/adapter.cpp")
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


def test_new_adapter_cannot_select_itself_out_of_ci(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    root = _adapter(repo, "model_c", "new_adapter")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["matrix"]["include"] == [
        _row("direct", root=root, scope="family")
    ]


def test_completing_a_preexisting_unregistered_layout_forces_family(
    repository,
) -> None:
    dispatcher = _load_dispatcher()
    repo, _, _ = repository
    _write(repo, "python/tensorrt_model_connect/families/model_c/new_adapter/adapter.py")
    _write(repo, "src/runtime/models/model_c/new_adapter/adapter.cpp")
    _write(repo, "tests/e2e/models/model_c/new_adapter/placeholder.txt")
    base = _commit(repo, "partial layout")
    root = _adapter(repo, "model_c", "new_adapter")
    head = _commit(repo, "register adapter")

    result = dispatcher.calculate(repo, base, head)

    assert result["inventory"]["added"] == 1
    assert result["matrix"]["include"] == [
        _row("direct", root=root, scope="family")
    ]


def test_changed_adapter_returning_none_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    _write(repo, f"{first}/profile_a/test_contract.py")
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="selected no optimized-runtime CI"):
        dispatcher.calculate(repo, base, head)


def test_full_deletion_emits_explicit_structural_validation(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    for owner in dispatcher.OWNER_ROOTS:
        shutil.rmtree(repo / owner / "model_a/fast_adapter")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["run"] is True
    assert result["inventory"]["removed"] == 1
    assert result["matrix"]["include"] == [_row("structural", scope="removal")]


def test_deleting_the_last_adapters_never_becomes_a_skip(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    for owner in dispatcher.OWNER_ROOTS:
        shutil.rmtree(repo / owner / "model_a/fast_adapter")
        shutil.rmtree(repo / owner / "model_b/vendor_adapter")
    head = _commit(repo)

    result = dispatcher.calculate(repo, base, head)

    assert result["run"] is True
    assert result["inventory"] == {"current": 0, "added": 0, "removed": 2}
    assert result["matrix"]["include"] == [_row("structural", scope="removal")]


def test_parent_symlink_cannot_escape_source_ownership(repository, tmp_path: Path) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    outside = tmp_path / "outside"
    outside.mkdir()
    family = repo / "python/tensorrt_model_connect/families/escaped"
    family.symlink_to(outside, target_is_directory=True)
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="ownership path is a symlink"):
        dispatcher.calculate(repo, base, head)


@pytest.mark.parametrize("schema", (True, 1.0, 2, None))
def test_selector_schema_version_is_exact(schema: object, repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    _select(repo, first, _payload("leaf", schema_version=schema))
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="invalid schema_version"):
        dispatcher.calculate(repo, base, head)


def test_malformed_selector_output_fails_closed(repository) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    _write(repo, f"{first}/result.json", "not json\n")
    head = _commit(repo)

    with pytest.raises(dispatcher.ContractError, match="invalid JSON"):
        dispatcher.calculate(repo, base, head)


def test_unrelated_broken_selector_is_not_invoked(repository, monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    repo, base, _ = repository
    _write(repo, "docs/readme.md", "changed\n")
    head = _commit(repo)
    monkeypatch.setattr(
        dispatcher,
        "_select",
        lambda *_args, **_kwargs: pytest.fail("unrelated selector was invoked"),
    )

    assert dispatcher.calculate(repo, base, head)["run"] is False


def test_affected_hung_selector_has_a_bounded_timeout(repository, monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    repo, _, (first, _) = repository
    _write(repo, f"{first}/ci_impact.py", "import time\ntime.sleep(60)\n")
    base = _commit(repo, "hang selector")
    _write(repo, f"{first}/profile_a/test_contract.py")
    head = _commit(repo)
    monkeypatch.setattr(dispatcher, "SELECTOR_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(dispatcher.ContractError, match="exceeded .* seconds"):
        dispatcher.calculate(repo, base, head)


def test_selector_output_is_bounded(repository, monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    repo, base, (first, _) = repository
    _select(repo, first, _payload("leaf"))
    head = _commit(repo)
    monkeypatch.setattr(dispatcher, "SELECTOR_OUTPUT_LIMIT", 8)

    with pytest.raises(dispatcher.ContractError, match="output limit"):
        dispatcher.calculate(repo, base, head)


def test_thousand_adapter_nightly_inventory_emits_bounded_shards(monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    adapters = tuple(
        dispatcher.Adapter(f"model_{index:04d}", "fast_adapter")
        for index in range(1_000)
    )
    monkeypatch.setattr(dispatcher, "_revision", lambda _repo, revision: revision)
    monkeypatch.setattr(
        dispatcher,
        "_inventory",
        lambda _repo, _revision, **_kwargs: adapters,
    )

    result = dispatcher.calculate(Path("."), "base", "head", all_adapters=True)

    rows = result["matrix"]["include"]
    assert len(rows) == 125
    assert len(rows) <= dispatcher.MAX_MATRIX_ROWS == 256
    assert all(row["kind"] == "shard" for row in rows)
    assert all(not row["adapter_root"] for row in rows)
    assert len(json.dumps(result["matrix"])) < 100_000


def test_hash_ordered_shards_are_deterministic_bounded_and_complete() -> None:
    dispatcher = _load_dispatcher()
    adapters = [
        dispatcher.Adapter(f"model_{index:04d}", "fast_adapter")
        for index in range(1_000)
    ]

    first = sorted(adapters, key=dispatcher._shard_order)
    second = sorted(adapters, key=dispatcher._shard_order)
    shards = [
        first[index : index + dispatcher.ADAPTERS_PER_SHARD]
        for index in range(0, len(first), dispatcher.ADAPTERS_PER_SHARD)
    ]

    assert first == second
    assert all(len(shard) <= dispatcher.ADAPTERS_PER_SHARD for shard in shards)
    assert [adapter for shard in shards for adapter in shard] == first
    assert set(first) == set(adapters)


def test_fanout_above_bounded_capacity_fails_closed() -> None:
    dispatcher = _load_dispatcher()

    with pytest.raises(dispatcher.ContractError, match="bounded shard capacity"):
        dispatcher._shard_rows(
            dispatcher.MAX_MATRIX_ROWS * dispatcher.ADAPTERS_PER_SHARD + 1,
            "family",
        )


def test_shard_continues_after_failure_and_aggregates(repository, monkeypatch) -> None:
    dispatcher = _load_dispatcher()
    repo, revision, _ = repository
    adapters = (
        dispatcher.Adapter("model_a", "fast_adapter"),
        dispatcher.Adapter("model_b", "vendor_adapter"),
    )
    calls = []
    monkeypatch.setattr(dispatcher, "discover", lambda _repo, _head: adapters)
    monkeypatch.setattr(
        dispatcher,
        "_run_adapter",
        lambda _repo, adapter, _scope, _profile: calls.append(adapter.key)
        or ("failed" if adapter.family == "model_a" else None),
    )

    result = dispatcher.run_matrix_row(
        repo,
        revision,
        revision,
        kind="shard",
        adapter_root="",
        family="",
        scope="family",
        profile="",
        shard_index=0,
        shard_count=1,
    )

    assert result == 1
    assert set(calls) == {adapter.key for adapter in adapters}
    assert len(calls) == len(adapters)


def test_cli_writes_only_compact_generic_outputs(repository, tmp_path: Path) -> None:
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
    assert output.stat().st_size < 10_000


def test_premerge_workflow_contains_only_bounded_generic_dispatch() -> None:
    dispatcher = _load_dispatcher()
    workflow = (DISPATCHER.parents[2] / ".github/workflows/trtmc-ci.yml").read_text(
        encoding="utf-8"
    )
    impact = workflow.split("\n  impact:", 1)[1].split("\n  source-quality:", 1)[0]
    contracts = workflow.split("\n  optimized-runtime-contracts:", 1)[1].split(
        "\n  model-proof:", 1
    )[0]

    assert "tools/ci/optimized_contracts.py" in impact
    assert "matrix.kind" in contracts
    assert "--run-kind \"$KIND\"" in contracts
    assert "--shard-index \"$SHARD_INDEX\"" in contracts
    assert "pytest==9.1.1" in contracts
    assert "if: ${{ matrix.kind == 'structural' }}" in contracts
    assert "numpy==2.4.6" in contracts
    assert "timeout-minutes: 100" in contracts
    assert "nlohmann" not in contracts
    assert "matrix.runner" not in contracts
    assert not re.search(r"qwen|edge.?llm", workflow, re.IGNORECASE)
    assert (
        dispatcher.ADAPTERS_PER_SHARD * dispatcher.RUNNER_TIMEOUT_SECONDS
        + 20 * 60
        <= 100 * 60
    )


def test_nightly_generically_runs_bounded_selector_free_shards() -> None:
    workflow = (DISPATCHER.parents[2] / ".github/workflows/nightly.yml").read_text(
        encoding="utf-8"
    )
    inventory = workflow.split("\n  inventory:", 1)[1].split("\n  source-quality:", 1)[0]
    contracts = workflow.split("\n  optimized-runtime-contracts:", 1)[1].split(
        "\n  required:", 1
    )[0]

    assert "tools/ci/optimized_contracts.py" in inventory
    assert "--all" in inventory
    assert "fail-fast: false" in contracts
    assert "--run-kind \"$KIND\"" in contracts
    assert "--shard-count \"$SHARD_COUNT\"" in contracts
    assert "pytest==9.1.1" in contracts
    assert "timeout-minutes: 100" in contracts
    assert "nlohmann" not in contracts
    assert "numpy" not in contracts
    assert "matrix.runner" not in contracts
    assert not re.search(r"qwen|edge.?llm", workflow, re.IGNORECASE)

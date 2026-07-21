#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route model-owned optimized-runtime CI without importing adapter logic.

The dispatcher owns only discovery, change isolation, bounded fan-out, and
contract validation. Every model/runtime adapter owns its selector and runner.

Boundary: discover, validate, and aggregate CI contracts; never implement an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
OWNER_ROOTS = (
    Path("python/tensorrt_model_connect/families"),
    Path("src/runtime/models"),
    Path("tests/e2e/models"),
)
TEST_OWNER_ROOT = OWNER_ROOTS[2]
CONTRACT_FILES = ("ci_impact.py", "ci_run.py", "test_ci_impact.py")
PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SCOPES = frozenset({"none", "leaf", "family", "provider"})
DIRECT_ROW_LIMIT = 16
MAX_MATRIX_ROWS = 256
ADAPTERS_PER_SHARD = 8
SELECTOR_TIMEOUT_SECONDS = 30.0
SELECTOR_OUTPUT_LIMIT = 1_048_576
RUNNER_TIMEOUT_SECONDS = 600.0

# These files and roots define the runtime-neutral optimized-runtime protocol.
# They deliberately contain no model or third-party runtime names.
SHARED_PROVIDER_EXACT = frozenset(
    {
        ".github/workflows/nightly.yml",
        ".github/workflows/trtmc-ci.yml",
        "CMakeLists.txt",
        "include/trtmc/pipeline.h",
        "python/tensorrt_model_connect/build_cli.py",
        "python/tensorrt_model_connect/engine_builder.py",
        "src/runtime/registry/pipeline_factory.cpp",
        "tests/builder/test_manifest_validation.py",
        "tests/builder/test_optimized_runtime_capsules.py",
        "tests/builder/test_optimized_runtime_orchestrator.py",
        "tests/builder/test_optimized_runtime_public_routing.py",
        "tests/cpp/test_optimized_runtime_host.cpp",
        "tests/tools/test_optimized_contracts_ci.py",
        "tests/tools/test_optimized_runtime_packaging.py",
        "tools/ci/model_proof_selection.py",
        "tools/ci/optimized_contracts.py",
    }
)
SHARED_PROVIDER_PREFIXES = (
    "python/tensorrt_model_connect/runtime_provider/",
    "src/runtime/providers/",
    "tests/cpp/fakes/fake_optimized_runtime_",
)
STRUCTURAL_TESTS = (
    "tests/builder/test_manifest_validation.py",
    "tests/builder/test_optimized_runtime_capsules.py",
    "tests/builder/test_optimized_runtime_orchestrator.py",
    "tests/builder/test_optimized_runtime_public_routing.py",
    "tests/tools/test_optimized_runtime_packaging.py",
)


class ContractError(RuntimeError):
    """A registered model-owned CI contract is unsafe or invalid."""


@dataclass(frozen=True, order=True)
class Adapter:
    family: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"

    @property
    def root(self) -> str:
        return (TEST_OWNER_ROOT / self.family / self.name).as_posix()

    @property
    def selector(self) -> str:
        return f"{self.root}/ci_impact.py"

    @property
    def runner(self) -> str:
        return f"{self.root}/ci_run.py"

    def owner_root(self, owner: Path) -> Path:
        return owner / self.family / self.name


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None


def _git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if process.returncode:
        error = (
            process.stderr.decode(errors="replace") if binary else process.stderr
        ).strip()
        raise ContractError(f"git {' '.join(arguments)} failed: {error}")
    return process.stdout


def _revision(repo: Path, revision: str) -> str:
    resolved = str(_git(repo, "rev-parse", f"{revision}^{{commit}}")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ContractError(f"Git did not resolve {revision!r} to an exact commit")
    return resolved


def _safe_path(raw: bytes) -> str:
    path = os.fsdecode(raw)
    if path.startswith("/") or ".." in Path(path).parts:
        raise ContractError(f"unsafe Git tree path: {path!r}")
    return path


def _tree_entries(repo: Path, revision: str) -> dict[str, str]:
    raw = _git(repo, "ls-tree", "-r", "-z", revision, binary=True)
    assert isinstance(raw, bytes)
    entries: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode = metadata.split(b" ", 1)[0].decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ContractError("Git returned a malformed tree entry") from exc
        path = _safe_path(encoded_path)
        if path in entries:
            raise ContractError(f"Git returned duplicate tree path: {path}")
        entries[path] = mode
    return entries


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _owned_location(path: str) -> tuple[str, str | None] | None:
    for owner in OWNER_ROOTS:
        root = owner.as_posix()
        if not path.startswith(f"{root}/"):
            continue
        relative = Path(path[len(root) + 1 :])
        if not relative.parts:
            return None
        family = relative.parts[0]
        adapter = (
            relative.parts[1]
            if len(relative.parts) >= 2 and relative.parts[1].endswith("_adapter")
            else None
        )
        return family, adapter
    return None


def _adapter_keys(entries: dict[str, str]) -> frozenset[Adapter]:
    adapters = set()
    for path in entries:
        location = _owned_location(path)
        if location is not None and location[1] is not None:
            adapters.add(Adapter(location[0], location[1]))
    return frozenset(adapters)


def _assert_no_owned_symlinks(entries: dict[str, str], revision: str) -> None:
    for path, mode in entries.items():
        if mode == "120000" and any(_under(path, root.as_posix()) for root in OWNER_ROOTS):
            raise ContractError(
                f"optimized-runtime ownership path is a symlink at {revision}: {path}"
            )


def _tree_has_root(entries: dict[str, str], root: Path) -> bool:
    prefix = f"{root.as_posix()}/"
    return any(path.startswith(prefix) for path in entries)


def _validate_tree_adapter(
    entries: dict[str, str], adapter: Adapter, revision: str
) -> None:
    missing_roots = [
        owner.as_posix()
        for owner in OWNER_ROOTS
        if not _tree_has_root(entries, adapter.owner_root(owner))
    ]
    if missing_roots:
        raise ContractError(
            f"optimized-runtime adapter {adapter.key} lacks ownership roots at "
            f"{revision}: {missing_roots}"
        )
    for contract in CONTRACT_FILES:
        path = f"{adapter.root}/{contract}"
        if entries.get(path) != "100644" and entries.get(path) != "100755":
            raise ContractError(
                f"optimized-runtime adapter {adapter.key} lacks {contract} at {revision}"
            )


def _assert_real_contained_directory(repo: Path, owner: Path, relative: Path) -> Path:
    owner_path = repo / owner
    candidate = owner_path / relative
    current = repo
    for component in candidate.relative_to(repo).parts:
        current /= component
        if current.is_symlink():
            raise ContractError(f"optimized-runtime ownership path is a symlink: {current}")
    if not candidate.is_dir():
        raise ContractError(f"optimized-runtime ownership directory is unavailable: {candidate}")
    resolved_owner = owner_path.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_owner):
        raise ContractError(f"optimized-runtime ownership escapes {owner_path}: {candidate}")
    return resolved


def _validate_worktree_adapter(repo: Path, adapter: Adapter) -> None:
    relative = Path(adapter.family) / adapter.name
    for owner in OWNER_ROOTS:
        _assert_real_contained_directory(repo, owner, relative)
    test_root = repo / adapter.root
    for contract in CONTRACT_FILES:
        path = test_root / contract
        if path.is_symlink() or not path.is_file():
            raise ContractError(
                f"optimized-runtime adapter {adapter.key} lacks real {contract}"
            )
        if not path.resolve(strict=True).is_relative_to(test_root.resolve(strict=True)):
            raise ContractError(f"optimized-runtime contract escapes its owner: {path}")


def _inventory(
    repo: Path,
    revision: str,
    *,
    allow_incomplete: bool = False,
) -> tuple[Adapter, ...]:
    entries = _tree_entries(repo, revision)
    _assert_no_owned_symlinks(entries, revision)
    adapters = []
    for adapter in sorted(_adapter_keys(entries)):
        try:
            _validate_tree_adapter(entries, adapter, revision)
        except ContractError:
            if not allow_incomplete:
                raise
            continue
        adapters.append(adapter)
    result = tuple(adapters)
    if revision == _revision(repo, "HEAD"):
        for adapter in result:
            _validate_worktree_adapter(repo, adapter)
    return result


def discover(repo: Path, revision: str = "HEAD") -> tuple[Adapter, ...]:
    """Return the complete, contained adapter inventory at a Git revision."""

    return _inventory(repo, _revision(repo, revision))


def _diff(repo: Path, base: str, head: str) -> tuple[DiffEntry, ...]:
    merge_base = str(_git(repo, "merge-base", base, head)).strip()
    raw = _git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        merge_base,
        head,
        binary=True,
    )
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    entries = []
    index = 0
    while index < len(fields) and fields[index]:
        try:
            status = fields[index].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ContractError("Git returned a non-ASCII diff status") from exc
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise ContractError("truncated Git rename/copy record")
            old_path = _safe_path(fields[index])
            new_path = _safe_path(fields[index + 1])
            index += 2
        else:
            if index >= len(fields):
                raise ContractError("truncated Git diff record")
            path = _safe_path(fields[index])
            index += 1
            old_path = None if code == "A" else path
            new_path = None if code == "D" else path
        if code not in {"A", "C", "D", "M", "R", "T"}:
            raise ContractError(f"unsupported Git diff status: {status}")
        entries.append(DiffEntry(status, old_path, new_path))
    return tuple(entries)


def _changed_paths(repo: Path, base: str, head: str) -> frozenset[str]:
    return frozenset(
        path
        for entry in _diff(repo, base, head)
        for path in (entry.old_path, entry.new_path)
        if path is not None
    )


def _is_shared_provider(path: str) -> bool:
    return path in SHARED_PROVIDER_EXACT or any(
        path.startswith(prefix) for prefix in SHARED_PROVIDER_PREFIXES
    )


def _select(repo: Path, adapter: Adapter, base: str, head: str) -> dict[str, object]:
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            [
                sys.executable,
                adapter.selector,
                "--repo-root",
                str(repo),
                "--base",
                base,
                "--head",
                head,
            ],
            cwd=repo,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            process.wait(timeout=SELECTOR_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            raise ContractError(
                f"selector {adapter.selector} exceeded {SELECTOR_TIMEOUT_SECONDS:g} seconds"
            ) from exc
        stdout_size = stdout_file.tell()
        stderr_size = stderr_file.tell()
        if stdout_size > SELECTOR_OUTPUT_LIMIT or stderr_size > SELECTOR_OUTPUT_LIMIT:
            raise ContractError(f"selector {adapter.selector} exceeded its output limit")
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()
    output = stdout.decode("utf-8", errors="replace")
    error = stderr.decode("utf-8", errors="replace")
    if process.returncode:
        raise ContractError(
            f"selector {adapter.selector} failed: {(error.strip() or output.strip())}"
        )
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ContractError(f"selector {adapter.selector} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"selector {adapter.selector} result must be an object")
    return payload


def _validate(
    adapter: Adapter,
    payload: dict[str, object],
) -> tuple[str, list[str], list[dict[str, str]]]:
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ContractError(f"selector {adapter.selector} returned invalid schema_version")
    mode, run = payload.get("mode"), payload.get("run")
    profiles, matrix = payload.get("profiles"), payload.get("matrix")
    if mode not in SCOPES or type(run) is not bool or run != (mode != "none"):
        raise ContractError(f"selector {adapter.selector} returned invalid mode/run")
    if (
        not isinstance(profiles, list)
        or not all(
            isinstance(profile, str) and PROFILE_RE.fullmatch(profile)
            for profile in profiles
        )
        or profiles != sorted(set(profiles))
    ):
        raise ContractError(f"selector {adapter.selector} returned invalid profiles")
    if not isinstance(matrix, dict) or set(matrix) != {"include"}:
        raise ContractError(f"selector {adapter.selector} returned invalid matrix")
    raw_entries = matrix["include"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ContractError(f"selector {adapter.selector} returned an empty matrix")

    entries = []
    for raw in raw_entries:
        if not isinstance(raw, dict) or set(raw) != {"scope", "profile"}:
            raise ContractError(f"selector {adapter.selector} returned invalid matrix entry")
        scope, profile = raw["scope"], raw["profile"]
        if scope != mode or not isinstance(profile, str):
            raise ContractError(f"selector {adapter.selector} matrix contradicts its mode")
        if mode == "leaf":
            if not PROFILE_RE.fullmatch(profile) or profile not in profiles:
                raise ContractError(f"selector {adapter.selector} returned invalid leaf")
        elif profile:
            raise ContractError(f"selector {adapter.selector} returned an unexpected profile")
        entries.append({"scope": str(scope), "profile": profile})
    if mode == "leaf" and sorted(entry["profile"] for entry in entries) != profiles:
        raise ContractError(f"selector {adapter.selector} leaf matrix is incomplete")
    if mode != "leaf" and len(entries) != 1:
        raise ContractError(f"selector {adapter.selector} emitted duplicate {mode} rows")
    return str(mode), profiles, entries


def _matrix_row(
    kind: str,
    *,
    adapter: Adapter | None = None,
    family: str = "",
    scope: str = "none",
    profile: str = "",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, object]:
    return {
        "kind": kind,
        "adapter_root": adapter.root if adapter is not None else "",
        "family": family,
        "scope": scope,
        "profile": profile,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def _direct_rows(
    adapter: Adapter, scope: str, profiles: Sequence[str] = ()
) -> list[dict[str, object]]:
    if scope == "leaf":
        return [
            _matrix_row("direct", adapter=adapter, scope=scope, profile=profile)
            for profile in profiles
        ]
    return [_matrix_row("direct", adapter=adapter, scope=scope)]


def _shard_rows(
    adapter_count: int,
    scope: str,
    *,
    family: str = "",
    capacity: int = MAX_MATRIX_ROWS,
) -> list[dict[str, object]]:
    if adapter_count <= 0 or capacity <= 0:
        return []
    count = (adapter_count + ADAPTERS_PER_SHARD - 1) // ADAPTERS_PER_SHARD
    if count > capacity:
        raise ContractError(
            f"{adapter_count} adapters exceed the bounded shard capacity of "
            f"{capacity * ADAPTERS_PER_SHARD}"
        )
    return [
        _matrix_row(
            "shard",
            family=family,
            scope=scope,
            shard_index=index,
            shard_count=count,
        )
        for index in range(count)
    ]


def _selection(adapter: Adapter, mode: str, profile_count: int) -> dict[str, object]:
    return {
        "adapter_root": adapter.root,
        "mode": mode,
        "profile_count": profile_count,
    }


def calculate(
    repo: Path,
    base: str,
    head: str,
    *,
    all_adapters: bool = False,
) -> dict[str, object]:
    base, head = _revision(repo, base), _revision(repo, head)
    # A base tree may contain a pre-existing, non-registered directory whose
    # registration is completed by this change. Only complete base contracts
    # constitute an adapter that can be removed or modified.
    base_adapters = frozenset(_inventory(repo, base, allow_incomplete=True))
    head_adapters = frozenset(_inventory(repo, head))
    added = head_adapters - base_adapters
    removed = base_adapters - head_adapters

    if all_adapters:
        rows = _shard_rows(len(head_adapters), "family")
        selected = []
    else:
        paths = _changed_paths(repo, base, head)
        shared_provider = any(_is_shared_provider(path) for path in paths)
        exact: set[Adapter] = set()
        family_common: set[str] = set()
        forced_family = set(added)

        for path in paths:
            location = _owned_location(path)
            if location is None:
                continue
            family, name = location
            if name is None:
                family_common.add(family)
                continue
            adapter = Adapter(family, name)
            exact.add(adapter)
            relative = path.split(f"/{name}/", 1)
            if len(relative) == 2 and relative[1] in CONTRACT_FILES:
                forced_family.add(adapter)

        selected = []
        if shared_provider:
            rows = [_matrix_row("structural", scope="provider")]
            rows.extend(
                _shard_rows(
                    len(head_adapters),
                    "provider",
                    capacity=MAX_MATRIX_ROWS - 1,
                )
            )
            selected.append(
                {
                    "mode": "provider",
                    "adapter_count": len(head_adapters),
                    "reason": "shared-provider",
                }
            )
        else:
            rows = []
            if removed:
                rows.append(_matrix_row("structural", scope="removal"))
                selected.append(
                    {
                        "mode": "removal",
                        "adapter_count": len(removed),
                        "reason": "atomic-removal",
                    }
                )

            family_owned = {
                adapter
                for adapter in head_adapters
                if adapter.family in family_common
            }
            forced_family.update(family_owned)
            affected = (exact | forced_family) & head_adapters
            for adapter in sorted(affected):
                if adapter in forced_family:
                    mode, profiles, adapter_rows = "family", [], _direct_rows(
                        adapter, "family"
                    )
                else:
                    mode, profiles, raw_rows = _validate(
                        adapter, _select(repo, adapter, base, head)
                    )
                    if mode == "none":
                        raise ContractError(
                            f"changed adapter {adapter.key} selected no optimized-runtime CI"
                        )
                    adapter_rows = _direct_rows(adapter, mode, profiles)
                rows.extend(adapter_rows)
                selected.append(_selection(adapter, mode, len(profiles)))

            if len(rows) > MAX_MATRIX_ROWS:
                reserve = 1 if removed else 0
                families = {adapter.family for adapter in affected}
                family = next(iter(families)) if len(families) == 1 else ""
                eligible = [
                    adapter
                    for adapter in head_adapters
                    if not family or adapter.family == family
                ]
                rows = ([_matrix_row("structural", scope="removal")] if removed else [])
                rows.extend(
                    _shard_rows(
                        len(eligible),
                        "family",
                        family=family,
                        capacity=MAX_MATRIX_ROWS - reserve,
                    )
                )
                selected = [
                    {
                        "mode": "family",
                        "adapter_count": len(eligible),
                        "reason": "bounded-fanout",
                    }
                ]
            elif len(rows) > DIRECT_ROW_LIMIT:
                # Keep normal changes direct, but bound unusually broad family edits.
                non_structural = [row for row in rows if row["kind"] != "structural"]
                families = {adapter.family for adapter in affected}
                if non_structural and len(families) == 1:
                    family = next(iter(families))
                    eligible = [
                        adapter for adapter in head_adapters if adapter.family == family
                    ]
                    rows = [row for row in rows if row["kind"] == "structural"]
                    rows.extend(
                        _shard_rows(
                            len(eligible),
                            "family",
                            family=family,
                            capacity=MAX_MATRIX_ROWS - len(rows),
                        )
                    )

    rows.sort(
        key=lambda row: (
            str(row["kind"]),
            str(row["adapter_root"]),
            str(row["family"]),
            int(row["shard_index"]),
            str(row["profile"]),
        )
    )
    if not rows:
        rows = [_matrix_row("none")]
    return {
        "schema_version": 1,
        "base_revision": base,
        "head_revision": head,
        "run": rows[0]["kind"] != "none",
        "matrix": {"include": rows},
        "selection": selected,
        "inventory": {
            "current": len(head_adapters),
            "added": len(added),
            "removed": len(removed),
        },
    }


def _shard_order(adapter: Adapter) -> tuple[bytes, str]:
    return hashlib.sha256(adapter.key.encode("utf-8")).digest(), adapter.key


def _run_adapter(repo: Path, adapter: Adapter, scope: str, profile: str) -> str | None:
    try:
        result = subprocess.run(
            [
                sys.executable,
                adapter.runner,
                "--scope",
                scope,
                "--profile",
                profile,
            ],
            cwd=repo,
            check=False,
            timeout=RUNNER_TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return f"{adapter.key}: exceeded {RUNNER_TIMEOUT_SECONDS:g} seconds"
    return None if result.returncode == 0 else f"{adapter.key}: exit {result.returncode}"


def _run_structural(repo: Path) -> int:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *STRUCTURAL_TESTS,
        ],
        cwd=repo,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    ).returncode


def run_matrix_row(
    repo: Path,
    base: str,
    head: str,
    *,
    kind: str,
    adapter_root: str,
    family: str,
    scope: str,
    profile: str,
    shard_index: int,
    shard_count: int,
) -> int:
    base, head = _revision(repo, base), _revision(repo, head)
    adapters = discover(repo, head)
    if kind == "structural":
        result = calculate(repo, base, head)
        if not any(row["kind"] == "structural" for row in result["matrix"]["include"]):
            raise ContractError("structural matrix row is not selected for this change")
        return _run_structural(repo)
    if kind == "direct":
        matches = [adapter for adapter in adapters if adapter.root == adapter_root]
        if len(matches) != 1 or family or shard_index != 0 or shard_count != 1:
            raise ContractError("invalid direct optimized-runtime matrix row")
        if scope not in {"leaf", "family", "provider"}:
            raise ContractError("invalid direct optimized-runtime scope")
        if scope == "leaf":
            if not PROFILE_RE.fullmatch(profile):
                raise ContractError("invalid direct optimized-runtime profile")
        elif profile:
            raise ContractError("non-leaf optimized-runtime row has a profile")
        failure = _run_adapter(repo, matches[0], scope, profile)
        if failure:
            print(f"optimized-contracts: {failure}", file=sys.stderr)
            return 1
        return 0
    if kind != "shard":
        raise ContractError(f"unsupported optimized-runtime matrix kind: {kind!r}")
    if adapter_root or profile or scope not in {"family", "provider"}:
        raise ContractError("invalid optimized-runtime shard row")
    if shard_count < 1 or shard_count > MAX_MATRIX_ROWS:
        raise ContractError("invalid optimized-runtime shard count")
    if shard_index < 0 or shard_index >= shard_count:
        raise ContractError("invalid optimized-runtime shard index")
    if family and not PROFILE_RE.fullmatch(family):
        raise ContractError("invalid optimized-runtime shard family")
    eligible = sorted(
        (
            adapter
            for adapter in adapters
            if not family or adapter.family == family
        ),
        key=_shard_order,
    )
    expected_count = (
        len(eligible) + ADAPTERS_PER_SHARD - 1
    ) // ADAPTERS_PER_SHARD
    if shard_count != expected_count:
        raise ContractError(
            f"optimized-runtime shard count {shard_count} does not match "
            f"inventory-derived count {expected_count}"
        )
    start = shard_index * ADAPTERS_PER_SHARD
    assigned = eligible[start : start + ADAPTERS_PER_SHARD]
    failures = [
        failure
        for adapter in assigned
        if (failure := _run_adapter(repo, adapter, scope, "")) is not None
    ]
    if failures:
        print(
            "optimized-contracts: shard failures:\n- " + "\n- ".join(failures),
            file=sys.stderr,
        )
        return 1
    print(
        f"optimized-contracts: shard {shard_index + 1}/{shard_count} "
        f"completed {len(assigned)} adapter(s)"
    )
    return 0


def _write_outputs(path: Path, result: dict[str, object]) -> None:
    summary = {
        "selected": result["selection"],
        "inventory": result["inventory"],
    }
    values = {
        "run": str(result["run"]).lower(),
        "matrix": json.dumps(result["matrix"], separators=(",", ":")),
        "selection": json.dumps(summary, separators=(",", ":")),
    }
    with path.open("a", encoding="utf-8") as output:
        for name, value in values.items():
            output.write(f"{name}={value}\n")


def _repository(value: str) -> Path:
    path = Path(value).resolve()
    if not (path / ".git").exists():
        raise argparse.ArgumentTypeError(f"not a Git worktree: {path}")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_repository, default=REPOSITORY)
    parser.add_argument("--base")
    parser.add_argument("--head", required=True)
    parser.add_argument("--all", action="store_true", dest="all_adapters")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--run-kind", choices=("direct", "shard", "structural"))
    parser.add_argument("--adapter-root", default="")
    parser.add_argument("--family", default="")
    parser.add_argument("--scope", default="none")
    parser.add_argument("--profile", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        if arguments.run_kind:
            if not arguments.base:
                raise ContractError("--base is required to execute a matrix row")
            return run_matrix_row(
                arguments.repo_root,
                arguments.base,
                arguments.head,
                kind=arguments.run_kind,
                adapter_root=arguments.adapter_root,
                family=arguments.family,
                scope=arguments.scope,
                profile=arguments.profile,
                shard_index=arguments.shard_index,
                shard_count=arguments.shard_count,
            )
        if not arguments.all_adapters and not arguments.base:
            raise ContractError("--base is required unless --all is selected")
        base = arguments.head if arguments.all_adapters else arguments.base
        result = calculate(
            arguments.repo_root,
            base,
            arguments.head,
            all_adapters=arguments.all_adapters,
        )
        if arguments.github_output:
            _write_outputs(arguments.github_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ContractError as exc:
        print(f"optimized-contracts: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

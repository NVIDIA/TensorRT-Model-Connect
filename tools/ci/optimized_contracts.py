#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route optimized-runtime CPU CI to model-owned contracts.

Boundary: discover, validate, and aggregate CI contracts; never implement an adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[2]
TEST_ROOT = Path("tests/e2e/models")
OWNER_ROOTS = (
    Path("python/tensorrt_model_connect/families"),
    Path("src/runtime/models"),
)
CONTRACT_FILES = ("ci_impact.py", "ci_run.py", "test_ci_impact.py")
PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SCOPES = frozenset({"none", "leaf", "family", "provider"})
DISPATCHER_FILES = frozenset(
    {
        ".github/workflows/trtmc-ci.yml",
        "tools/ci/model_proof_selection.py",
        "tools/ci/optimized_contracts.py",
    }
)


class ContractError(RuntimeError):
    """A registered model-owned CI contract is unsafe or invalid."""


@dataclass(frozen=True, order=True)
class Adapter:
    root: str

    @property
    def selector(self) -> str:
        return f"{self.root}/ci_impact.py"

    @property
    def runner(self) -> str:
        return f"{self.root}/ci_run.py"


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


def _changed_paths(repo: Path, base: str, head: str) -> frozenset[str]:
    merge_base = str(_git(repo, "merge-base", base, head)).strip()
    raw = _git(
        repo,
        "diff",
        "--name-only",
        "-z",
        "--find-renames",
        merge_base,
        head,
        binary=True,
    )
    assert isinstance(raw, bytes)
    return frozenset(os.fsdecode(path) for path in raw.split(b"\0") if path)


def discover(repo: Path) -> tuple[Adapter, ...]:
    """Return complete adapters that opt in with model-owned CI files."""

    base = repo / TEST_ROOT
    if not base.is_dir():
        raise ContractError(f"missing model test ownership root: {base}")
    adapters = []
    for root in sorted(base.glob("*/*_adapter")):
        if not any((root / name).exists() for name in CONTRACT_FILES):
            continue
        if root.is_symlink() or not root.is_dir():
            raise ContractError(f"adapter test root must be a real directory: {root}")
        family, name = root.parent.name, root.name
        missing_roots = [
            owner.as_posix()
            for owner in OWNER_ROOTS
            if (repo / owner / family / name).is_symlink()
            or not (repo / owner / family / name).is_dir()
        ]
        if missing_roots:
            raise ContractError(
                f"optimized-runtime adapter {family}/{name} lacks ownership roots: "
                f"{missing_roots}"
            )
        for contract in CONTRACT_FILES:
            path = root / contract
            if path.is_symlink() or not path.is_file():
                raise ContractError(
                    f"optimized-runtime adapter {family}/{name} lacks {contract}"
                )
        adapters.append(Adapter(root.relative_to(repo).as_posix()))
    return tuple(adapters)


def _select(repo: Path, adapter: Adapter, base: str, head: str) -> dict[str, object]:
    process = subprocess.run(
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
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ContractError(f"selector {adapter.selector} failed: {detail}")
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"selector {adapter.selector} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"selector {adapter.selector} result must be an object")
    return payload


def _validate(
    adapter: Adapter,
    payload: dict[str, object],
) -> tuple[str, list[str], list[dict[str, str]]]:
    mode, run = payload.get("mode"), payload.get("run")
    profiles, matrix = payload.get("profiles"), payload.get("matrix")
    if mode not in SCOPES or not isinstance(run, bool) or run != (mode != "none"):
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
        entries.append(
            {
                "adapter_root": adapter.root,
                "runner": adapter.runner,
                "scope": str(scope),
                "profile": profile,
            }
        )
    if mode == "leaf" and sorted(entry["profile"] for entry in entries) != profiles:
        raise ContractError(f"selector {adapter.selector} leaf matrix is incomplete")
    if mode != "leaf" and len(entries) != 1:
        raise ContractError(f"selector {adapter.selector} emitted duplicate {mode} rows")
    return str(mode), profiles, entries


def _row(adapter: Adapter, scope: str) -> dict[str, str]:
    return {
        "adapter_root": adapter.root,
        "runner": adapter.runner,
        "scope": scope,
        "profile": "",
    }


def calculate(
    repo: Path,
    base: str,
    head: str,
    *,
    all_adapters: bool = False,
) -> dict[str, object]:
    base, head = _revision(repo, base), _revision(repo, head)
    force_provider = bool(_changed_paths(repo, base, head) & DISPATCHER_FILES)
    rows, selection = [], []
    for adapter in discover(repo):
        mode, profiles, selected = _validate(adapter, _select(repo, adapter, base, head))
        if all_adapters:
            mode, selected = "family", [_row(adapter, "family")]
        elif force_provider:
            mode, selected = "provider", [_row(adapter, "provider")]
        if mode != "none":
            rows.extend(selected)
        selection.append({"adapter_root": adapter.root, "mode": mode, "profiles": profiles})

    rows.sort(key=lambda row: (row["adapter_root"], row["profile"], row["scope"]))
    if not rows:
        rows = [{"adapter_root": "", "runner": "", "scope": "none", "profile": ""}]
    return {
        "schema_version": 1,
        "base_revision": base,
        "head_revision": head,
        "run": rows[0]["scope"] != "none",
        "matrix": {"include": rows},
        "selection": selection,
    }


def _write_outputs(path: Path, result: dict[str, object]) -> None:
    values = {
        "run": str(result["run"]).lower(),
        "matrix": json.dumps(result["matrix"], separators=(",", ":")),
        "selection": json.dumps(result["selection"], separators=(",", ":")),
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
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

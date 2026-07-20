#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the smallest sound Qwen EdgeLLM premerge qualification scope."""

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


REPOSITORY = Path(__file__).resolve().parents[5]
FAMILY = "qwen"
ADAPTER = "edge_llm_adapter"
PROFILE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

BUILDER_ROOT = f"python/tensorrt_model_connect/families/{FAMILY}"
RUNTIME_ROOT = f"src/runtime/models/{FAMILY}"
TEST_ROOT = f"tests/e2e/models/{FAMILY}"
ADAPTER_ROOTS = (
    f"{BUILDER_ROOT}/{ADAPTER}",
    f"{RUNTIME_ROOT}/{ADAPTER}",
    f"{TEST_ROOT}/{ADAPTER}",
)
FAMILY_ROOTS = (BUILDER_ROOT, RUNTIME_ROOT, TEST_ROOT)

# These files participate directly in optimized-runtime selection, packaging,
# loading, or private SDK compilation even though they do not live below the
# provider directories.
PROVIDER_EXACT = frozenset(
    {
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
        "tests/tools/test_optimized_runtime_packaging.py",
    }
)
PROVIDER_PREFIXES = (
    "python/tensorrt_model_connect/runtime_provider/",
    "src/runtime/providers/",
    "tests/cpp/fakes/fake_optimized_runtime_",
)
FAMILY_WORKFLOW_EXACT = frozenset(
    {
        ".github/workflows/qwen-edgellm-a100.yml",
    }
)


class ImpactError(RuntimeError):
    """The revision or model-owned adapter layout cannot be routed safely."""


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None


def _git(repo: Path, arguments: Sequence[str], *, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", errors="replace")
            if binary
            else result.stderr
        )
        raise ImpactError(f"git {' '.join(arguments)} failed: {stderr.strip()}")
    return result.stdout


def _revision(repo: Path, revision: str) -> str:
    resolved = str(_git(repo, ["rev-parse", f"{revision}^{{commit}}"])).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ImpactError(f"Git did not resolve an exact commit for {revision!r}")
    return resolved


def _tree_paths(repo: Path, revision: str) -> frozenset[str]:
    raw = _git(repo, ["ls-tree", "-r", "--name-only", "-z", revision], binary=True)
    assert isinstance(raw, bytes)
    paths = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        path = os.fsdecode(encoded)
        if path.startswith("/") or ".." in Path(path).parts:
            raise ImpactError(f"unsafe Git tree path: {path!r}")
        paths.append(path)
    return frozenset(paths)


def _profiles(repo: Path, revision: str) -> tuple[str, ...]:
    paths = _tree_paths(repo, revision)
    manifest_prefix = f"{ADAPTER_ROOTS[0]}/"
    suffix = "/IMPLEMENTATION.toml"
    profiles = []
    for path in sorted(paths):
        if not path.startswith(manifest_prefix) or not path.endswith(suffix):
            continue
        relative = path[len(manifest_prefix) : -len(suffix)]
        if "/" in relative or not PROFILE_RE.fullmatch(relative):
            raise ImpactError(f"invalid Qwen EdgeLLM profile directory: {relative!r}")
        required = (
            f"{ADAPTER_ROOTS[1]}/{relative}/CMakeLists.txt",
            f"{ADAPTER_ROOTS[1]}/{relative}/adapter.cpp",
            f"{ADAPTER_ROOTS[2]}/{relative}/build_runners.py",
            f"{ADAPTER_ROOTS[2]}/{relative}/test_a100_e2e.py",
        )
        missing = [candidate for candidate in required if candidate not in paths]
        if missing:
            raise ImpactError(
                f"Qwen EdgeLLM profile {relative!r} is incomplete: {missing}"
            )
        profiles.append(relative)
    if len(profiles) != len(set(profiles)):
        raise ImpactError("Qwen EdgeLLM profile discovery produced duplicates")
    return tuple(profiles)


def _diff(repo: Path, base: str, head: str) -> tuple[DiffEntry, ...]:
    merge_base = str(_git(repo, ["merge-base", base, head])).strip()
    raw = _git(
        repo,
        ["diff", "--name-status", "-z", "--find-renames", merge_base, head],
        binary=True,
    )
    assert isinstance(raw, bytes)
    fields = raw.split(b"\0")
    entries = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode("ascii")
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise ImpactError("truncated Git rename/copy record")
            old_path = os.fsdecode(fields[index])
            new_path = os.fsdecode(fields[index + 1])
            index += 2
        else:
            if index >= len(fields):
                raise ImpactError("truncated Git diff record")
            path = os.fsdecode(fields[index])
            index += 1
            old_path = None if code == "A" else path
            new_path = None if code == "D" else path
        if code not in {"A", "C", "D", "M", "R", "T"}:
            raise ImpactError(f"unsupported Git diff status: {status}")
        entries.append(DiffEntry(status, old_path, new_path))
    return tuple(entries)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def _profile_for_path(path: str, profiles: frozenset[str]) -> str | None:
    for root in ADAPTER_ROOTS:
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        first = relative.split("/", 1)[0]
        return first if first in profiles and "/" in relative else None
    return None


def _family_path_kind(path: str) -> str | None:
    """Return Edge adapter, Qwen common, or unrelated sibling-adapter ownership."""

    for root in FAMILY_ROOTS:
        if not _under(path, root):
            continue
        if path == root:
            return "common"
        relative = path[len(root) + 1 :]
        namespace = relative.split("/", 1)[0]
        if namespace == ADAPTER:
            return "edge"
        if namespace.endswith("_adapter"):
            return "other_adapter"
        return "common"
    return None


def _is_provider(path: str) -> bool:
    return path in PROVIDER_EXACT or any(path.startswith(prefix) for prefix in PROVIDER_PREFIXES)


def calculate(repo: Path, base: str, head: str) -> dict[str, object]:
    resolved_base = _revision(repo, base)
    resolved_head = _revision(repo, head)
    profiles = _profiles(repo, resolved_head)
    profile_set = frozenset(profiles)
    selected: set[str] = set()
    family = False
    provider = False
    changed_paths: set[str] = set()

    for entry in _diff(repo, resolved_base, resolved_head):
        for path in (entry.old_path, entry.new_path):
            if path is None or path in changed_paths:
                continue
            changed_paths.add(path)
            if _is_provider(path):
                provider = True
                continue
            if path in FAMILY_WORKFLOW_EXACT:
                family = True
                continue
            family_kind = _family_path_kind(path)
            if family_kind in {None, "other_adapter"}:
                continue
            if family_kind == "common":
                family = True
                continue
            leaf = _profile_for_path(path, profile_set)
            if leaf is not None:
                selected.add(leaf)
            else:
                family = True

    if provider:
        mode = "provider"
        matrix_entries = [{"scope": "provider", "profile": ""}]
    elif family:
        mode = "family"
        matrix_entries = [{"scope": "family", "profile": ""}]
    elif selected:
        mode = "leaf"
        matrix_entries = [
            {"scope": "leaf", "profile": profile} for profile in sorted(selected)
        ]
    else:
        mode = "none"
        # GitHub validates matrix syntax before evaluating a job-level if.
        matrix_entries = [{"scope": "none", "profile": ""}]

    run = mode != "none"
    return {
        "schema_version": 1,
        "base_revision": resolved_base,
        "head_revision": resolved_head,
        "mode": mode,
        "run": run,
        "profiles": sorted(selected) if mode == "leaf" else list(profiles),
        "all_profiles": list(profiles),
        "matrix": {"include": matrix_entries},
        "changed_paths": sorted(changed_paths),
    }


def _write_github_output(path: Path, result: dict[str, object]) -> None:
    outputs = {
        "qwen_edgellm_mode": str(result["mode"]),
        "qwen_edgellm_run": str(bool(result["run"])).lower(),
        "qwen_edgellm_profiles": json.dumps(result["profiles"], separators=(",", ":")),
        "qwen_edgellm_matrix": json.dumps(result["matrix"], separators=(",", ":")),
    }
    with path.open("a", encoding="utf-8") as stream:
        for name, value in outputs.items():
            stream.write(f"{name}={value}\n")


def _repository(value: str) -> Path:
    path = Path(value).resolve()
    if not (path / ".git").exists():
        raise argparse.ArgumentTypeError(f"not a Git worktree: {path}")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=_repository, default=REPOSITORY)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        result = calculate(arguments.repo_root, arguments.base, arguments.head)
        if arguments.github_output is not None:
            _write_github_output(arguments.github_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ImpactError as exc:
        print(f"qwen-edgellm-impact: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

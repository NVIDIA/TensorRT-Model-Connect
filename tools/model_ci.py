#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact model ownership, impact, and positive source projection for CI.

The directory name below a ``MODEL.toml`` ownership root is its physical owner.
E2E ``runtime_strategy`` metadata normalizes the rare logical/runtime name
split into one independently testable module. Impact analysis selects only
modules touched by a diff. A model projection starts empty and materializes
tracked Git blobs for one module plus an explicit platform allowlist; sibling
model files are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


MODEL_ROOTS = (
    "python/tensorrt_model_connect/families",
    "src/runtime/models",
    "tests/e2e/models",
    "tests/cpp/models",
)

# These are the only non-model source surfaces made visible to an isolated
# build.  Model-root handling takes precedence, so a sibling below e.g. src/
# is excluded even though src/ is otherwise an approved platform root.
PLATFORM_PROJECTION_EXACT = frozenset(
    {
        ".clang-format",
        ".github/scripts/run-model-proof.sh",
        "CMakeLists.txt",
        "Dockerfile",
        "LICENSE",
        "NOTICE",
        "README.md",
        "_pyproject_backend.py",
        "conanfile.py",
        "conftest.py",
        "pyproject.toml",
        "ruff.toml",
        "tests/__init__.py",
    }
)
PLATFORM_PROJECTION_PREFIXES = (
    "cmake/",
    "include/",
    "python/tensorrt_model_connect/",
    "scripts/",
    "src/",
    "tensorrt_model_connect/",
    "tests/cpp/",
    "tests/assets/",
    "tests/e2e/",
    "tests/e2e_harness/",
    "third_party/",
    "tools/",
)

LEGAL_OR_DOC_EXACT = frozenset(
    {
        "AGENTS.md",
        "CODEOWNERS",
        "CONTRIBUTING.md",
        "LICENSE",
        "NOTICE",
        "README.md",
        ".github/workflows/legal.yml",
        "tools/legal_header_exceptions.toml",
        "tools/legal_headers.py",
    }
)
LEGAL_OR_DOC_PREFIXES = ("website/",)

PLATFORM_EXACT = frozenset(
    {
        ".clang-format",
        ".dockerignore",
        ".gitignore",
        "CMakeLists.txt",
        "Dockerfile",
        "_pyproject_backend.py",
        "conanfile.py",
        "conftest.py",
        "docker-compose.yml",
        "hf_links_wave1.txt",
        "pyproject.toml",
        "ruff.toml",
        "verify_encoder.py",
    }
)
PLATFORM_PREFIXES = (
    "cmake/",
    "examples/",
    "include/",
    "python/",
    "src/",
    "tensorrt_model_connect/",
    "tests/",
    "third_party/",
)
CI_OR_TOOLING_PREFIXES = (
    ".agents/",
    ".ci/",
    ".codex/",
    ".github/",
    "agent_bench/",
    "plugins/",
    "reports/",
    "scripts/",
    "tools/",
)

_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_MANIFEST_ID_RE = re.compile(r'(?m)^\s*id\s*=\s*"([^"]+)"\s*$')


class ModelCIError(RuntimeError):
    """A fail-closed ownership or projection error."""


@dataclass(frozen=True)
class TreeEntry:
    mode: str
    object_type: str
    object_id: str
    path: str


@dataclass(frozen=True)
class OwnershipCatalog:
    revision: str
    entries: tuple[TreeEntry, ...]
    models: tuple[str, ...]
    manifests: dict[str, tuple[str, ...]]
    owners_by_root: dict[str, dict[str, str]]
    runtime_models: dict[str, tuple[str, ...]]
    e2e_families: dict[str, tuple[str, ...]]
    legacy_shared_runtime: tuple[str, ...]


@dataclass(frozen=True)
class DiffEntry:
    status: str
    old_path: str | None
    new_path: str | None


def _run_git(repo_root: Path, args: Sequence[str], *, text: bool = False):
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=text,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr if isinstance(exc.stderr, str) else os.fsdecode(exc.stderr)
            detail = f": {stderr.strip()}"
        raise ModelCIError(f"git {' '.join(args)} failed{detail}") from exc


def _resolve_revision(repo_root: Path, revision: str) -> str:
    return str(_run_git(repo_root, ["rev-parse", f"{revision}^{{commit}}"], text=True)).strip()


def _read_tree(repo_root: Path, revision: str) -> tuple[TreeEntry, ...]:
    raw = _run_git(repo_root, ["ls-tree", "-rz", "--full-tree", revision])
    entries: list[TreeEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
        except ValueError as exc:
            raise ModelCIError(f"could not parse git tree entry at {revision}") from exc
        path = os.fsdecode(raw_path)
        _validate_git_path(path)
        entries.append(TreeEntry(mode, object_type, object_id, path))
    return tuple(entries)


def _validate_git_path(path: str) -> None:
    pure = PurePosixPath(path)
    if not path or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ModelCIError(f"unsafe path in Git tree: {path!r}")


def _read_blob(repo_root: Path, object_id: str) -> bytes:
    return bytes(_run_git(repo_root, ["cat-file", "blob", object_id]))


def _toml_strings(text: str, key: str) -> tuple[str, ...]:
    list_match = re.search(rf"(?ms)^\s*{re.escape(key)}\s*=\s*\[([^]]*)\]", text)
    if list_match is not None:
        return tuple(re.findall(r'"([^"]+)"', list_match.group(1)))
    scalar_match = re.search(rf'(?m)^\s*{re.escape(key)}\s*=\s*"([^"]+)"\s*$', text)
    return (scalar_match.group(1),) if scalar_match is not None else ()


def _manifest_location(path: str) -> tuple[str, str] | None:
    for root in MODEL_ROOTS[:-1]:
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :].split("/")
        if len(relative) == 2 and relative[1] == "MODEL.toml":
            return root, relative[0]
    return None


def _validate_model_roots() -> None:
    roots = [PurePosixPath(root) for root in MODEL_ROOTS]
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ModelCIError(f"overlapping model ownership roots: {left} and {right}")


def discover_catalog(
    repo_root: Path,
    revision: str,
    *,
    allow_legacy_shared_runtime: bool = False,
) -> OwnershipCatalog:
    """Discover model IDs from MODEL.toml blobs at one Git revision."""
    _validate_model_roots()
    resolved = _resolve_revision(repo_root, revision)
    entries = _read_tree(repo_root, resolved)
    manifest_roots_by_physical: dict[str, list[str]] = {}
    manifest_text: dict[tuple[str, str], str] = {}
    seen_locations: set[tuple[str, str]] = set()
    for entry in entries:
        location = _manifest_location(entry.path)
        if location is None:
            continue
        if entry.object_type != "blob":
            raise ModelCIError(f"model manifest is not a blob: {entry.path}")
        root, directory_id = location
        try:
            text = _read_blob(repo_root, entry.object_id).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelCIError(f"model manifest is not UTF-8: {entry.path}") from exc
        match = _MANIFEST_ID_RE.search(text)
        if match is None:
            raise ModelCIError(f"model manifest has no string id: {entry.path}")
        declared_id = match.group(1)
        if not _MODEL_ID_RE.fullmatch(declared_id):
            raise ModelCIError(f"model manifest has unsafe id {declared_id!r}: {entry.path}")
        if declared_id != directory_id:
            raise ModelCIError(
                f"model manifest id {declared_id!r} does not match directory "
                f"{directory_id!r}: {entry.path}"
            )
        if location in seen_locations:
            raise ModelCIError(f"duplicate model ownership manifest: {entry.path}")
        seen_locations.add(location)
        manifest_roots_by_physical.setdefault(declared_id, []).append(root)
        manifest_text[(root, declared_id)] = text
    if not manifest_roots_by_physical:
        raise ModelCIError(f"no MODEL.toml ownership manifests found at {resolved}")

    runtime_root = "src/runtime/models"
    e2e_root = "tests/e2e/models"
    runtime_ids = {
        model for model, roots in manifest_roots_by_physical.items() if runtime_root in roots
    }
    e2e_ids = {model for model, roots in manifest_roots_by_physical.items() if e2e_root in roots}
    strategy_owner: dict[str, str] = {}
    for runtime_id in sorted(runtime_ids):
        text = manifest_text[(runtime_root, runtime_id)]
        strategies = _toml_strings(text, "runtime_strategies") or _toml_strings(
            text, "runtime_strategy"
        )
        for strategy in strategies:
            previous = strategy_owner.get(strategy)
            if previous is not None and previous != runtime_id:
                raise ModelCIError(
                    f"runtime strategy {strategy!r} is owned by both {previous!r} "
                    f"and {runtime_id!r}"
                )
            strategy_owner[strategy] = runtime_id

    e2e_runtime_candidates: dict[str, set[str]] = {model: set() for model in e2e_ids}
    e2e_prefix = f"{e2e_root}/"
    for entry in entries:
        if not entry.path.startswith(e2e_prefix) or "/manifests/" not in entry.path:
            continue
        relative = entry.path[len(e2e_prefix) :]
        family = relative.split("/", 1)[0]
        if family not in e2e_ids or not entry.path.endswith(".json"):
            continue
        try:
            payload = json.loads(_read_blob(repo_root, entry.object_id))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelCIError(f"invalid E2E manifest JSON: {entry.path}") from exc
        strategy = payload.get("runtime_strategy")
        if strategy is None:
            continue
        runtime_id = strategy_owner.get(str(strategy))
        if runtime_id is None:
            raise ModelCIError(
                f"E2E manifest uses unowned runtime strategy {strategy!r}: {entry.path}"
            )
        e2e_runtime_candidates[family].add(runtime_id)

    e2e_to_runtime: dict[str, str] = {}
    for family in sorted(e2e_ids):
        candidates = e2e_runtime_candidates[family]
        if family in runtime_ids:
            candidates.add(family)
        if len(candidates) > 1:
            raise ModelCIError(
                f"E2E family {family!r} depends on multiple runtime models: {sorted(candidates)}"
            )
        if candidates:
            e2e_to_runtime[family] = next(iter(candidates))

    runtime_to_logical: dict[str, str] = {}
    legacy_shared_runtime: set[str] = set()
    for runtime_id in sorted(runtime_ids):
        logical_owners = sorted(
            family for family, candidate in e2e_to_runtime.items() if candidate == runtime_id
        )
        if len(logical_owners) > 1:
            if not allow_legacy_shared_runtime:
                raise ModelCIError(
                    f"runtime model {runtime_id!r} is shared by multiple model modules: "
                    f"{logical_owners}"
                )
            # Old merge bases can predate the one-runtime-owner invariant. Keep
            # their shared runtime root as a conservative platform-like owner;
            # a change to it fans out below because it is absent from the head
            # catalog's logical modules.
            runtime_to_logical[runtime_id] = runtime_id
            legacy_shared_runtime.add(runtime_id)
        else:
            runtime_to_logical[runtime_id] = logical_owners[0] if logical_owners else runtime_id

    owners_by_root: dict[str, dict[str, str]] = {root: {} for root in MODEL_ROOTS}
    for physical_id, roots in manifest_roots_by_physical.items():
        for root in roots:
            if root == runtime_root:
                logical = runtime_to_logical[physical_id]
            else:
                logical = physical_id
            owners_by_root[root][physical_id] = logical

    cpp_root = "tests/cpp/models"
    cpp_prefix = f"{cpp_root}/"
    cpp_children = {
        entry.path[len(cpp_prefix) :].split("/", 1)[0]
        for entry in entries
        if entry.path.startswith(cpp_prefix) and "/" in entry.path[len(cpp_prefix) :]
    }
    logical_ids = set(e2e_ids) | set(manifest_roots_by_physical)
    for child in cpp_children:
        if child in runtime_to_logical:
            owners_by_root[cpp_root][child] = runtime_to_logical[child]
        elif child in logical_ids:
            owners_by_root[cpp_root][child] = child

    models = sorted(logical for owners in owners_by_root.values() for logical in owners.values())
    models = sorted(set(models) - legacy_shared_runtime)
    manifests: dict[str, list[str]] = {model: [] for model in models}
    for root, owners in owners_by_root.items():
        for physical, logical in owners.items():
            if logical not in manifests:
                continue
            prefix = f"{root}/{physical}/"
            if any(entry.path.startswith(prefix) for entry in entries):
                manifests[logical].append(f"{root}/{physical}")
    runtime_models: dict[str, tuple[str, ...]] = {}
    for runtime_id, logical in runtime_to_logical.items():
        runtime_models.setdefault(logical, ())
        runtime_models[logical] = tuple(sorted({*runtime_models[logical], runtime_id}))
    e2e_families = {model: (model,) for model in sorted(e2e_ids) if model in models}
    return OwnershipCatalog(
        resolved,
        entries,
        tuple(models),
        {model: tuple(sorted(roots)) for model, roots in manifests.items()},
        owners_by_root,
        runtime_models,
        e2e_families,
        tuple(sorted(legacy_shared_runtime)),
    )


def _owner_for_path(path: str, catalog: OwnershipCatalog) -> tuple[str | None, bool]:
    """Return (owner, under_model_root). An unregistered child has no owner."""
    matches: list[str] = []
    under_model_root = False
    for root in MODEL_ROOTS:
        prefix = f"{root}/"
        if not path.startswith(prefix):
            continue
        under_model_root = True
        relative = path[len(prefix) :]
        if not relative or "/" not in relative:
            continue
        candidate = relative.split("/", 1)[0]
        owner = catalog.owners_by_root[root].get(candidate)
        if owner is not None:
            matches.append(owner)
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ModelCIError(f"path has overlapping model owners: {path}: {unique}")
    return (unique[0] if unique else None), under_model_root


def _is_legal_or_docs(path: str) -> bool:
    return (
        path in LEGAL_OR_DOC_EXACT
        or path.endswith(".md")
        or any(path.startswith(prefix) for prefix in LEGAL_OR_DOC_PREFIXES)
        or path.startswith("tests/tools/test_legal_")
    )


def _classify_path(path: str, catalog: OwnershipCatalog) -> tuple[str, str | None]:
    owner, under_model_root = _owner_for_path(path, catalog)
    if owner is not None:
        return "model", owner
    if under_model_root:
        raise ModelCIError(f"path is under a model root but has no MODEL.toml owner: {path}")
    if _is_legal_or_docs(path):
        return "legal_docs", None
    if path in PLATFORM_EXACT or any(path.startswith(prefix) for prefix in PLATFORM_PREFIXES):
        return "platform", None
    if any(path.startswith(prefix) for prefix in CI_OR_TOOLING_PREFIXES):
        return "ci_tooling", None
    raise ModelCIError(f"changed path has no model, platform, CI, legal, or docs owner: {path}")


def _diff_entries(repo_root: Path, base: str, head: str) -> tuple[DiffEntry, ...]:
    base_sha = _resolve_revision(repo_root, base)
    head_sha = _resolve_revision(repo_root, head)
    try:
        merge_base = str(_run_git(repo_root, ["merge-base", base_sha, head_sha], text=True)).strip()
    except ModelCIError:
        merge_base = base_sha
    raw = _run_git(
        repo_root,
        ["diff", "--name-status", "-z", "--find-renames", merge_base, head_sha],
    )
    fields = raw.split(b"\0")
    entries: list[DiffEntry] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index].decode("ascii")
        index += 1
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(fields):
                raise ModelCIError("truncated rename/copy record in git diff")
            old_path = os.fsdecode(fields[index])
            new_path = os.fsdecode(fields[index + 1])
            index += 2
        else:
            if index >= len(fields):
                raise ModelCIError("truncated path record in git diff")
            path = os.fsdecode(fields[index])
            index += 1
            old_path = None if code == "A" else path
            new_path = None if code == "D" else path
        if code not in {"A", "C", "D", "M", "R", "T"}:
            raise ModelCIError(f"unsupported git diff status: {status}")
        for path in (old_path, new_path):
            if path is not None:
                _validate_git_path(path)
        entries.append(DiffEntry(status, old_path, new_path))
    return tuple(entries)


def _result(
    models: Iterable[str], *, mode: str, changes: list[dict[str, object]]
) -> dict[str, object]:
    selected = sorted(set(models))
    return {
        "schema_version": 1,
        "mode": mode,
        "has_models": bool(selected),
        "expected_count": len(selected),
        "affected_models": selected,
        "matrix": {"include": [{"model": model} for model in selected]},
        "changes": changes,
    }


def calculate_impact(
    repo_root: Path,
    base: str,
    head: str,
    *,
    platform_change_policy: str,
) -> dict[str, object]:
    base_sha = _resolve_revision(repo_root, base)
    head_sha = _resolve_revision(repo_root, head)
    try:
        comparison_base = str(
            _run_git(repo_root, ["merge-base", base_sha, head_sha], text=True)
        ).strip()
    except ModelCIError:
        comparison_base = base_sha
    base_catalog = discover_catalog(repo_root, comparison_base, allow_legacy_shared_runtime=True)
    head_catalog = discover_catalog(repo_root, head, allow_legacy_shared_runtime=True)
    affected: set[str] = set()
    broad_change = False
    serialized_changes: list[dict[str, object]] = []
    for change in _diff_entries(repo_root, comparison_base, head_sha):
        classifications: list[dict[str, str]] = []
        path_catalogs = (
            (change.old_path, base_catalog),
            (change.new_path, head_catalog),
        )
        seen_path_revision: set[tuple[str, str]] = set()
        for path, catalog in path_catalogs:
            if path is None or (path, catalog.revision) in seen_path_revision:
                continue
            seen_path_revision.add((path, catalog.revision))
            kind, owner = _classify_path(path, catalog)
            item = {"path": path, "kind": kind}
            if owner is not None:
                item["model"] = owner
                affected.add(owner)
            elif kind in {"platform", "ci_tooling"}:
                broad_change = True
            classifications.append(item)
        serialized_changes.append(
            {
                "status": change.status,
                "old_path": change.old_path,
                "new_path": change.new_path,
                "classifications": classifications,
            }
        )
    if (
        affected - set(head_catalog.models)
        or affected.intersection(base_catalog.legacy_shared_runtime)
        or affected.intersection(head_catalog.legacy_shared_runtime)
    ):
        broad_change = True
    if broad_change:
        if platform_change_policy != "all":
            raise ModelCIError(
                "platform or CI/tooling change requires --platform-change-policy all"
            )
        affected.update(head_catalog.models)
        mode = "all"
    elif affected:
        mode = "models"
    else:
        mode = "none"
    result = _result(affected, mode=mode, changes=serialized_changes)
    result["base_revision"] = base_catalog.revision
    result["head_revision"] = head_catalog.revision
    return result


def _write_github_output(path: Path, result: dict[str, object]) -> None:
    outputs = {
        "matrix": json.dumps(result["matrix"], separators=(",", ":")),
        "has_models": str(bool(result["has_models"])).lower(),
        "affected_models": json.dumps(result["affected_models"], separators=(",", ":")),
        "expected_count": str(result["expected_count"]),
        "mode": str(result["mode"]),
    }
    with path.open("a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")


def _is_platform_projection_path(path: str) -> bool:
    return path in PLATFORM_PROJECTION_EXACT or any(
        path.startswith(prefix) for prefix in PLATFORM_PROJECTION_PREFIXES
    )


def _prepare_output(repo_root: Path, output_dir: Path, *, clean: bool) -> None:
    resolved_repo = repo_root.resolve()
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_repo or resolved_output in resolved_repo.parents:
        raise ModelCIError("projection output must not contain the repository")
    if output_dir.exists() or output_dir.is_symlink():
        if not clean:
            raise ModelCIError(f"projection output already exists: {output_dir}")
        if output_dir.is_symlink() or output_dir.is_file():
            output_dir.unlink()
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _validate_included_symlinks(
    repo_root: Path,
    included: Sequence[TreeEntry],
) -> dict[str, str]:
    included_paths = {entry.path for entry in included}
    targets: dict[str, str] = {}
    for entry in included:
        if entry.mode != "120000":
            continue
        raw_target = _read_blob(repo_root, entry.object_id)
        try:
            target = raw_target.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelCIError(f"symlink target is not UTF-8: {entry.path}") from exc
        if not target or PurePosixPath(target).is_absolute():
            raise ModelCIError(f"symlink escapes projection: {entry.path} -> {target!r}")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry.path), target))
        if resolved == ".." or resolved.startswith("../") or resolved.startswith("/"):
            raise ModelCIError(f"symlink escapes projection: {entry.path} -> {target!r}")
        target_is_present = resolved in included_paths or any(
            path.startswith(f"{resolved}/") for path in included_paths
        )
        if not target_is_present:
            raise ModelCIError(
                f"symlink points outside the positive allowlist: {entry.path} -> {target!r}"
            )
        targets[entry.path] = target
    return targets


def create_projection(
    repo_root: Path,
    revision: str,
    model: str,
    output_dir: Path,
    *,
    clean: bool,
) -> dict[str, object]:
    catalog = discover_catalog(repo_root, revision)
    if model not in catalog.models:
        raise ModelCIError(f"unknown model at {catalog.revision}: {model}")
    runtime_models = catalog.runtime_models.get(model, ())
    if len(runtime_models) != 1:
        raise ModelCIError(
            f"model must resolve to exactly one runtime model: {model}: {list(runtime_models)}"
        )
    runtime_model = runtime_models[0]
    e2e_families = catalog.e2e_families.get(model, ())
    if len(e2e_families) > 1:
        raise ModelCIError(
            f"model resolves to multiple E2E families: {model}: {list(e2e_families)}"
        )
    included: list[TreeEntry] = []
    model_files = 0
    platform_files = 0
    excluded_model_files = 0
    for entry in catalog.entries:
        owner, under_model_root = _owner_for_path(entry.path, catalog)
        if owner is not None:
            if owner == model:
                included.append(entry)
                model_files += 1
            else:
                excluded_model_files += 1
            continue
        if under_model_root:
            # Legacy/unregistered model-root content is unavailable too.
            excluded_model_files += 1
            continue
        if _is_platform_projection_path(entry.path):
            included.append(entry)
            platform_files += 1
    if model_files == 0:
        raise ModelCIError(f"model has no owned files at {catalog.revision}: {model}")
    unsupported = [
        entry.path for entry in included if entry.mode not in {"100644", "100755", "120000"}
    ]
    if unsupported:
        raise ModelCIError(f"unsupported Git entry type in projection: {unsupported[0]}")
    symlink_targets = _validate_included_symlinks(repo_root, included)
    _prepare_output(repo_root, output_dir, clean=clean)
    manifest_entries: list[dict[str, str]] = []
    for entry in included:
        destination = output_dir / PurePosixPath(entry.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry.mode == "120000":
            destination.symlink_to(symlink_targets[entry.path])
            digest = hashlib.sha256(symlink_targets[entry.path].encode("utf-8")).hexdigest()
        else:
            content = _read_blob(repo_root, entry.object_id)
            destination.write_bytes(content)
            destination.chmod(0o755 if entry.mode == "100755" else 0o644)
            digest = hashlib.sha256(content).hexdigest()
        owner, _ = _owner_for_path(entry.path, catalog)
        manifest_entries.append(
            {
                "path": entry.path,
                "mode": entry.mode,
                "blob": entry.object_id,
                "sha256": digest,
                "kind": "model" if owner == model else "platform",
            }
        )
    manifest = {
        "schema_version": 1,
        "revision": catalog.revision,
        "model": model,
        "runtime_model": runtime_model,
        "build_target": f"trtmc_model_{runtime_model}",
        "e2e_family": e2e_families[0] if e2e_families else None,
        "model_roots": list(catalog.manifests[model]),
        "model_files": model_files,
        "platform_files": platform_files,
        "excluded_model_files": excluded_model_files,
        "files": manifest_entries,
    }
    manifest_path = output_dir / ".trtmc-model-projection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _repo_root(value: str) -> Path:
    path = Path(value).resolve()
    if not (path / ".git").exists():
        raise argparse.ArgumentTypeError(f"not a Git worktree: {path}")
    return path


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--github-output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_root = Path(__file__).resolve().parents[1]

    validate = subparsers.add_parser("validate", help="validate MODEL.toml ownership")
    validate.add_argument("--repo-root", type=_repo_root, default=default_root)
    validate.add_argument("--revision", default="HEAD")

    impact = subparsers.add_parser("impact", help="calculate exact model impact")
    impact.add_argument("--repo-root", type=_repo_root, default=default_root)
    impact.add_argument("--base", required=True)
    impact.add_argument("--head", required=True)
    impact.add_argument("--platform-change-policy", choices=("all", "fail"), default="all")
    _add_common_output(impact)

    all_models = subparsers.add_parser("all", help="emit every model as a matrix")
    all_models.add_argument("--repo-root", type=_repo_root, default=default_root)
    all_models.add_argument("--revision", default="HEAD")
    _add_common_output(all_models)

    project = subparsers.add_parser("project", help="materialize one positive source projection")
    project.add_argument("--repo-root", type=_repo_root, default=default_root)
    project.add_argument("--revision", default="HEAD")
    project.add_argument("--model", required=True)
    project.add_argument("--output-dir", type=Path, required=True)
    project.add_argument("--clean", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            catalog = discover_catalog(args.repo_root, args.revision)
            result: dict[str, object] = {
                "schema_version": 1,
                "revision": catalog.revision,
                "model_count": len(catalog.models),
                "models": list(catalog.models),
            }
        elif args.command == "impact":
            result = calculate_impact(
                args.repo_root,
                args.base,
                args.head,
                platform_change_policy=args.platform_change_policy,
            )
            if args.github_output is not None:
                _write_github_output(args.github_output, result)
        elif args.command == "all":
            catalog = discover_catalog(args.repo_root, args.revision)
            result = _result(catalog.models, mode="all", changes=[])
            result["revision"] = catalog.revision
            if args.github_output is not None:
                _write_github_output(args.github_output, result)
        elif args.command == "project":
            result = create_projection(
                args.repo_root,
                args.revision,
                args.model,
                args.output_dir,
                clean=args.clean,
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise ModelCIError(f"unsupported command: {args.command}")
    except ModelCIError as exc:
        print(f"model-ci: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

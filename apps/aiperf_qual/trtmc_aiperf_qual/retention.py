# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk retention of bundles and checkpoints after a model is qualified.

``retention`` in the environment (both default to ``retain``):

- ``bundle``: ``retain`` | ``delete_on_pass`` | ``delete_unless_error`` | ``delete_built_unless_error``.
  A bundle is deleted after its model's run when the verdict allows; an ``error`` or ``smoke-fail``
  verdict (a failure to fix and rerun) always keeps it. ``delete_built_unless_error`` deletes only a bundle the run itself
  built (a bundle that existed before the run is kept).
- ``hf_cache``: ``retain`` | ``delete_unused``. ``run-all`` deletes a checkpoint repository from
  ``hf_hub_cache`` once no remaining profile of the batch uses it (profiles sharing one run together).

Deletion is confined to the configured roots: one model directory below ``bundle_root`` and one
``models--<org>--<name>`` repository below ``hf_hub_cache``. Reference environments and reports
are kept: re-judging needs only the reports.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, Environment

BUNDLE_POLICIES = ("retain", "delete_on_pass", "delete_unless_error", "delete_built_unless_error")
HF_CACHE_POLICIES = ("retain", "delete_unused")
TEMPORARY_POLICIES = ("retain", "delete_on_pass", "delete_unless_error")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def policies(environment: Environment) -> tuple[str, str]:
    configured = environment.values.get("retention") or {}
    unknown = set(configured) - {"bundle", "hf_cache", "temporary_files"}
    if unknown:
        raise ConfigError(f"unknown retention options: {', '.join(sorted(unknown))}")
    bundle, hf_cache = configured.get("bundle", "retain"), configured.get("hf_cache", "retain")
    if bundle not in BUNDLE_POLICIES:
        raise ConfigError(f"retention.bundle must be one of {', '.join(BUNDLE_POLICIES)}")
    if hf_cache not in HF_CACHE_POLICIES:
        raise ConfigError(f"retention.hf_cache must be one of {', '.join(HF_CACHE_POLICIES)}")
    if hf_cache != "retain" and not environment.values.get("hf_hub_cache"):
        raise ConfigError("retention.hf_cache requires hf_hub_cache (the checkpoint cache to manage)")
    if configured.get("temporary_files", "retain") not in TEMPORARY_POLICIES:
        raise ConfigError(f"retention.temporary_files must be one of {', '.join(TEMPORARY_POLICIES)}")
    return bundle, hf_cache


def should_delete_bundle(policy: str, category: str, built: bool = False) -> bool:
    """``built``: this run built the bundle (it did not exist before)."""
    if category in ("error", "build-failed", "smoke-fail"):
        return False
    return (policy == "delete_unless_error" or (policy == "delete_on_pass" and category == "pass")
            or (policy == "delete_built_unless_error" and built))


def _size(path: Path) -> int:
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def delete_bundle(environment: Environment, model: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the model's directory under bundle_root (the bundle and its build receipt)."""
    root = environment.path("bundle_root").resolve()
    directory = (root / model["candidate"]["bundle"]).parent.resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ConfigError(f"bundle {model['candidate']['bundle']!r} is not a model directory under {root}")
    if not directory.is_dir():
        return {"status": "absent", "path": str(directory)}
    size = _size(directory)
    shutil.rmtree(directory)
    return {"status": "deleted", "path": str(directory), "bytes": size}


def delete_checkpoint(hub_cache: Path, repo_id: str) -> dict[str, Any]:
    """Remove one model repository (all revisions) from a Hugging Face hub cache."""
    if not _REPO_ID.match(repo_id) or ".." in repo_id:
        raise ConfigError(f"not a Hugging Face model id: {repo_id!r}")
    folder = "models--" + repo_id.replace("/", "--")
    directory = hub_cache / folder
    locks = hub_cache / ".locks" / folder
    root = hub_cache.resolve()
    if (directory.is_symlink() or locks.is_symlink() or not directory.resolve().is_relative_to(root)
            or not locks.resolve().is_relative_to(root)):
        raise ConfigError(f"checkpoint cache entry must not be a symlink: {repo_id}")
    if not directory.is_dir():
        return {"status": "absent", "repo": repo_id}
    size = _size(directory)
    shutil.rmtree(directory)
    shutil.rmtree(locks, ignore_errors=True)
    return {"status": "deleted", "repo": repo_id, "bytes": size}


def cleanup_temporary(environment: Environment, out: Path, category: str) -> dict[str, Any]:
    """Delete generated request files only after scoring and server shutdown; never reports or raw exports."""
    policies(environment)
    policy = (environment.values.get("retention") or {}).get("temporary_files", "retain")
    receipt: dict[str, Any] = {"policy": policy, "status": "retained", "bytes": 0, "paths": []}
    if not should_delete_bundle(policy, category):
        return receipt
    root = out.resolve()
    work = out / "artifacts"
    paths = sorted(work.rglob("scratch")) if work.is_dir() else []
    # Validate every target before deleting any. A symlink may name shared weights or another run.
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ConfigError(f"unsafe scratch directory: {path}")
    for path in paths:
        if not path.is_dir():
            continue
        # The worker also writes startup diagnostics beside its per-request files.
        for log in path.glob("*.log"):
            if log.is_file() and not log.is_symlink():
                shutil.copy2(log, path.parent / log.name)
        receipt["bytes"] += _size(path)
        shutil.rmtree(path)
        receipt["paths"].append(str(path.relative_to(out)))
    receipt["status"] = "deleted" if receipt["paths"] else "absent"
    receipt["media_recheck"] = "regenerate" if receipt["paths"] else "available"
    return receipt

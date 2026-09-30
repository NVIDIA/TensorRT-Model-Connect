# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disk retention of bundles and checkpoints after a model is qualified.

``retention`` in the environment (both default to ``retain``):

- ``bundle``: ``retain`` | ``delete_on_pass`` | ``delete_unless_error``. A bundle is deleted after its
  model's run when the verdict allows; an ``error`` verdict (a harness failure to rerun) always keeps it.
- ``hf_cache``: ``retain`` | ``delete_unused``. ``run-all`` deletes a checkpoint repository from
  ``hf_hub_cache`` once no remaining profile of the batch uses it (profiles sharing one run together).

Deletion is confined to the configured roots: one model directory below ``bundle_root`` and one
``models--<org>--<name>`` repository below ``hf_hub_cache``. Reference environments, goldens, and
reports are kept: re-judging needs only the reports.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, Environment

BUNDLE_POLICIES = ("retain", "delete_on_pass", "delete_unless_error")
HF_CACHE_POLICIES = ("retain", "delete_unused")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


def policies(environment: Environment) -> tuple[str, str]:
    configured = environment.values.get("retention") or {}
    bundle, hf_cache = configured.get("bundle", "retain"), configured.get("hf_cache", "retain")
    if bundle not in BUNDLE_POLICIES:
        raise ConfigError(f"retention.bundle must be one of {', '.join(BUNDLE_POLICIES)}")
    if hf_cache not in HF_CACHE_POLICIES:
        raise ConfigError(f"retention.hf_cache must be one of {', '.join(HF_CACHE_POLICIES)}")
    if hf_cache != "retain" and not environment.values.get("hf_hub_cache"):
        raise ConfigError("retention.hf_cache requires hf_hub_cache (the checkpoint cache to manage)")
    return bundle, hf_cache


def should_delete_bundle(policy: str, category: str) -> bool:
    if category in ("error", "build-failed"):
        return False
    return policy == "delete_unless_error" or (policy == "delete_on_pass" and category == "pass")


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
    if not directory.is_dir():
        return {"status": "absent", "repo": repo_id}
    size = _size(directory)
    shutil.rmtree(directory)
    shutil.rmtree(hub_cache / ".locks" / folder, ignore_errors=True)
    return {"status": "deleted", "repo": repo_id, "bytes": size}

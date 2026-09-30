# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A machine's model list: ``models`` in its environment file (``run-all`` without ``--profile``).

    models:
      include: all                     # all ready catalog profiles (default), or names / glob patterns
      exclude:                         # a reason is required; excluded models are listed in the summary
        - {profile: "flux-2-dev*", reason: "does not fit in 80 GB"}
      max_checkpoint_gib: 30           # optional: also exclude larger checkpoints (unknown sizes are kept)
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import ConfigError

KEYS = {"include", "exclude", "max_checkpoint_gib"}


@dataclass(frozen=True)
class Profile:
    name: str
    task: str
    checkpoint: str
    revision: str | None


def catalog(repository: Path) -> list[Profile]:
    """Ready catalog profiles (the qualification scope) with their Task and checkpoint."""
    from .models import catalog_profiles

    from trtmc_benchmark.catalog import ManifestCatalog

    resolver = ManifestCatalog(repository / "families")
    profiles = []
    for entry in catalog_profiles(repository):
        model = resolver.resolve(entry.name)
        profiles.append(Profile(entry.name, model.task, model.hf_id, model.hf_revision or None))
    return profiles


def checkpoint_gib(profile: Profile) -> float | None:
    """Checkpoint weight size: the cached safetensors index, else the Hub's file sizes (None if unknown)."""
    from .models import checkpoint_bytes

    cached = checkpoint_bytes(profile.checkpoint, profile.revision)
    if cached:
        return cached / 2**30
    if not profile.checkpoint or profile.checkpoint.startswith("/"):
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(profile.checkpoint, revision=profile.revision, files_metadata=True)
    except Exception:  # noqa: BLE001 - offline or gated: the size is unknown, the model is kept
        return None
    files = info.siblings or []
    weights = [item for item in files if item.rfilename.endswith(".safetensors")] or [
        item for item in files if item.rfilename.endswith((".bin", ".pt", ".pth", ".ckpt"))]
    total = sum(item.size or 0 for item in weights)
    return total / 2**30 if total else None


def _patterns(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"models.{key} must be 'all' or a list of profile names / glob patterns")
    return value


def select(config: Mapping[str, Any], profiles: Sequence[Profile],
           size_gib: Callable[[Profile], float | None] = checkpoint_gib) -> tuple[list[Profile], list[dict]]:
    """(selected profiles in catalog order, excluded [{profile, task, reason}])."""
    unknown = set(config) - KEYS
    if unknown:
        raise ConfigError(f"unknown models keys: {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(KEYS))})")
    include = config.get("include", "all")
    names = [profile.name for profile in profiles]
    if include != "all":
        for pattern in _patterns(include, "include"):
            if not fnmatch.filter(names, pattern):
                raise ConfigError(f"models.include {pattern!r} matches no ready catalog profile")
    rules = config.get("exclude") or []
    if not isinstance(rules, list) or not all(isinstance(rule, Mapping) for rule in rules):
        raise ConfigError("models.exclude must be a list of {profile, reason}")
    for rule in rules:
        if not isinstance(rule.get("profile"), str) or not str(rule.get("reason") or "").strip():
            raise ConfigError(f"models.exclude entry {dict(rule)} needs a profile pattern and a reason")
        if not fnmatch.filter(names, rule["profile"]):
            print(f"trtmc-aiperf-qual: models.exclude {rule['profile']!r} matches no ready catalog profile",
                  file=sys.stderr)
    limit = config.get("max_checkpoint_gib")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0):
        raise ConfigError("models.max_checkpoint_gib must be a positive number")

    selected, excluded = [], []
    for profile in profiles:
        reason = _exclusion(profile, include, rules, limit, size_gib)
        if reason:
            excluded.append({"profile": profile.name, "task": profile.task, "reason": reason})
        else:
            selected.append(profile)
    return selected, excluded


def _exclusion(profile: Profile, include: Any, rules: Sequence[Mapping[str, Any]], limit: float | None,
               size_gib: Callable[[Profile], float | None]) -> str | None:
    if include != "all" and not any(fnmatch.fnmatchcase(profile.name, pattern) for pattern in include):
        return "not in models.include"
    for rule in rules:
        if fnmatch.fnmatchcase(profile.name, rule["profile"]):
            return str(rule["reason"]).strip()
    if limit is not None:
        size = size_gib(profile)
        if size is not None and size > limit:
            return f"checkpoint {size:.0f} GiB > models.max_checkpoint_gib {limit:g}"
    return None

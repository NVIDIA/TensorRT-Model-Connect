# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which profiles a run qualifies, and why each other one is left out.

The scope is every ready catalog profile (L0 smoke and regression variants aside). Each exclusion keeps its
reason and its ``source``, the file that decides it, checked in this order:

1. ``config/tasks.yaml``: the profile's Task has no defaults, so the qualification cannot qualify it.
2. ``config/exclusions.yaml``: no machine runs it (a native reference that cannot serve the profile as cataloged,
   a test model).
3. ``config/models``: an image-generation profile whose catalog request leaves a size, step or guidance control to
   the pipeline's default, which each side would choose for itself; its ``request`` must state the control.
4. The machine's ``models`` (its environment file): ``include`` and ``exclude``, then its platform's exclusions
   (``config/platforms/<platform>.yaml``), then ``max_checkpoint_gib``:

    models:
      include: all                     # all ready catalog profiles (default), or names / glob patterns
      exclude:                         # a reason is required; exclusions remain in the run audit
        - {profile: "flux-2-dev*", reason: "does not fit in 80 GB"}
      max_checkpoint_gib: 30           # optional: also exclude larger checkpoints (unknown sizes are kept)

Profiles named on the command line (``--profile``, names or glob patterns) replace ``include`` and are held to the
same exclusions; ``--ignore-exclusions`` runs them regardless (diagnostics).
"""

from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .config import CONFIG_ROOT, EXCLUSIONS, ConfigError, Environment, exclusions

KEYS = {"include", "exclude", "max_checkpoint_gib"}
TASKS_SOURCE = "config/tasks.yaml"
EXCLUSIONS_SOURCE = f"config/{EXCLUSIONS}"
MODELS_SOURCE = "config/models"
MACHINE_SOURCE = "environment"


@dataclass(frozen=True)
class Profile:
    name: str
    task: str
    checkpoint: str
    revision: str | None
    excluded: str | None = None  # why no machine runs it (config/tasks.yaml, config/exclusions.yaml or config/models)
    source: str | None = None  # the file of that reason


def catalog(repository: Path, root: Path = CONFIG_ROOT) -> list[Profile]:
    """Ready catalog profiles (the qualification scope) with their Task, checkpoint, and why no machine runs one."""
    from .models import catalog_profiles

    tasks = yaml.safe_load((root / "tasks.yaml").read_text())["tasks"]
    rules = exclusions(root)
    profiles = []
    for entry in catalog_profiles(repository):
        model = entry.model  # what ManifestCatalog.resolve returns, without reading every manifest again
        reason, source = None, None
        rule = next((rule for rule in rules if fnmatch.fnmatchcase(entry.name, rule["profile"])), None)
        if model.task not in tasks:
            reason, source = f"no Task defaults for {model.task!r}", TASKS_SOURCE
        elif rule:
            reason, source = rule["reason"], EXCLUSIONS_SOURCE
        elif entry.operation == "generate_image" and (missing := _unstated_image_controls(entry, repository, root)):
            reason, source = (f"its catalog request leaves {', '.join(missing)} to the pipeline's default; its "
                              "configuration's request states none of them"), MODELS_SOURCE
        profiles.append(Profile(entry.name, model.task, model.hf_id, model.hf_revision or None, reason, source))
    return profiles


def _unstated_image_controls(entry: Any, repository: Path, root: Path) -> list[str]:
    """The image controls the profile's catalog request, with its ``config/models`` ``request``, leaves unstated.

    A profile whose configuration or catalog request cannot resolve is kept: its run reports what it finds (its
    suites may not use the catalog request, as an image-to-video profile without a prompt)."""
    from trtmc_benchmark.catalog import resolve_case
    from trtmc_benchmark.types import BenchmarkError

    from .models import resolve_model
    from .suites import unstated_defaults

    try:
        model = resolve_model(entry.name, Environment({"repo": str(repository)}), root)
        request = {**dict(resolve_case(entry.model, Path(f"{entry.name}.bundle")).request),
                   **model.get("catalog_request", {})}
    except (BenchmarkError, ConfigError):
        return []
    return unstated_defaults(request, model["operation"])


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


def expand(patterns: Sequence[str], profiles: Sequence[Profile]) -> list[str]:
    """Named profiles in run order: each pattern's ready profiles in catalog order, else the pattern itself (its run
    then reports that it is not a ready catalog profile)."""
    names = [profile.name for profile in profiles]
    return list(dict.fromkeys(name for pattern in patterns
                              for name in ([name for name in names if fnmatch.fnmatchcase(name, pattern)] or [pattern])))


def select(config: Mapping[str, Any], profiles: Sequence[Profile],
           size_gib: Callable[[Profile], float | None] = checkpoint_gib,
           names: Sequence[str] | None = None) -> tuple[list[Profile], list[dict]]:
    """(selected profiles, excluded [{profile, task, reason, source}]): the machine's list in catalog order, or the
    ``names`` (names or glob patterns, in run order) in place of its ``include``."""
    unknown = set(config) - KEYS
    if unknown:
        raise ConfigError(f"unknown models keys: {', '.join(sorted(unknown))} (allowed: {', '.join(sorted(KEYS))})")
    include = config.get("include", "all")
    all_names = [profile.name for profile in profiles]
    if include != "all":
        for pattern in _patterns(include, "include"):
            if not fnmatch.filter(all_names, pattern):
                raise ConfigError(f"models.include {pattern!r} matches no ready catalog profile")
    rules = config.get("exclude") or []
    if not isinstance(rules, list) or not all(isinstance(rule, Mapping) for rule in rules):
        raise ConfigError("models.exclude must be a list of {profile, reason}")
    for rule in rules:
        if not isinstance(rule.get("profile"), str) or not str(rule.get("reason") or "").strip():
            raise ConfigError(f"models.exclude entry {dict(rule)} needs a profile pattern and a reason")
        if not fnmatch.filter(all_names, rule["profile"]):
            print(f"trtmc-aiperf-qual: models.exclude {rule['profile']!r} matches no ready catalog profile",
                  file=sys.stderr)
    limit = config.get("max_checkpoint_gib")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit <= 0):
        raise ConfigError("models.max_checkpoint_gib must be a positive number")
    candidates = list(profiles)
    if names is not None:  # the named profiles replace the machine's include, not its exclusions
        by_name = {profile.name: profile for profile in profiles}
        candidates, include = [by_name[name] for name in expand(names, profiles) if name in by_name], "all"

    selected, excluded = [], []
    for profile in candidates:
        found = _exclusion(profile, include, rules, limit, size_gib)
        if found:
            excluded.append({"profile": profile.name, "task": profile.task, "reason": found[0], "source": found[1]})
        else:
            selected.append(profile)
    return selected, excluded


def run_list(config: Mapping[str, Any], profiles: Sequence[Profile], names: Sequence[str] | None = None, *,
             ignore_exclusions: bool = False,
             size_gib: Callable[[Profile], float | None] = checkpoint_gib) -> tuple[list[str], list[dict]]:
    """(the profiles a run qualifies, in order; the excluded ones): the machine's list, or ``names`` held to the same
    exclusions. ``ignore_exclusions`` runs every named profile (a profile without Task defaults then reports a
    configuration error)."""
    if not names:
        if ignore_exclusions:
            raise ConfigError("--ignore-exclusions applies to the profiles named with --profile")
        selected, excluded = select(config, profiles, size_gib)
        return [profile.name for profile in selected], excluded
    wanted = expand(names, profiles)
    if ignore_exclusions:
        return wanted, []
    _, excluded = select(config, profiles, size_gib, names=names)
    left_out = {item["profile"] for item in excluded}
    return [name for name in wanted if name not in left_out], excluded


def machine_run_list(environment: Any, names: Sequence[str] | None = None, *,
                     ignore_exclusions: bool = False) -> tuple[list[str], list[dict]]:
    """``run_list`` on a machine: its repository's catalog and its environment's ``models``."""
    return run_list(environment.values.get("models") or {}, catalog(environment.path("repo")), names,
                    ignore_exclusions=ignore_exclusions)


def _exclusion(profile: Profile, include: Any, rules: Sequence[Mapping[str, Any]], limit: float | None,
               size_gib: Callable[[Profile], float | None]) -> tuple[str, str] | None:
    """(reason, source) of the first exclusion that applies, else None."""
    if profile.excluded:  # config/tasks.yaml or config/exclusions.yaml, whatever a machine's list includes
        return profile.excluded, profile.source or EXCLUSIONS_SOURCE
    if include != "all" and not any(fnmatch.fnmatchcase(profile.name, pattern) for pattern in include):
        return "not in models.include", MACHINE_SOURCE
    for rule in rules:
        if fnmatch.fnmatchcase(profile.name, rule["profile"]):
            return str(rule["reason"]).strip(), str(rule.get("source") or MACHINE_SOURCE)
    if limit is not None:
        size = size_gib(profile)
        if size is not None and size > limit:
            return f"checkpoint {size:.0f} GiB > models.max_checkpoint_gib {limit:g}", MACHINE_SOURCE
    return None

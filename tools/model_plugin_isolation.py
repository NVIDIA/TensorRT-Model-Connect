#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and prepare isolated runtime model plugin directories for E2E.

The E2E model family is not always the same as the runtime plugin owner. This
tool maps selected E2E models to the owning ``src/runtime/models/<id>`` plugin
and can copy only those DSOs out of a CMake build tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class E2EManifest:
    name: str
    family: str
    runtime_strategy: str
    path: Path
    bundle: str = ""
    result_case: str = ""
    ci_tier: str = ""


@dataclass(frozen=True)
class RuntimePlugin:
    model_id: str
    library: str
    strategies: tuple[str, ...]

    @property
    def target(self) -> str:
        return f"trtmc_model_{self.model_id}"


_NODE_ID_MODEL_RE = re.compile(r"::test_model_e2e\[([^\]]+)\]")

_MODEL_OWNED_ROOTS = {
    Path("python/tensorrt_model_connect/families"): "builder_families",
    Path("src/runtime/models"): "runtime_plugins",
    Path("tests/e2e/models"): "e2e_families",
    Path("tests/cpp/models"): "runtime_plugins",
}

_MODEL_OWNED_IMPACT_RULES = frozenset(
    {
        "manifest",
        "e2e_model_index",
        "e2e_model_threshold",
        "e2e_model_owned_test",
        "family_model_index",
        "family_package",
        "family_plugin",
        "python_profile_requirements",
        "cpp_runtime_model",
    }
)

_ORACLE_PROOF_KINDS = {
    "L1_external_reference": "reference",
    "L2_internal_reference": "reference",
    "L3_snapshot_regression": "snapshot_regression",
    "L4_invariants": "functional_invariant",
}

_RECOVERY_FIELDS = frozenset(
    {
        "attempt",
        "returncode",
        "signal",
        "builder_pid",
        "started_at",
        "recovered_at",
    }
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        return None
    return timestamp


def _positive_pid(value: object) -> bool:
    return type(value) is int and value > 0


def _toml_string(text: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*\"([^\"]+)\"", text)
    return match.group(1) if match else ""


def _toml_list(text: str, key: str) -> tuple[str, ...]:
    match = re.search(rf"(?ms)^\s*{re.escape(key)}\s*=\s*\[([^\]]*)\]", text)
    if not match:
        return ()
    return tuple(re.findall(r"\"([^\"]+)\"", match.group(1)))


def discover_e2e_manifests(repo_root: Path) -> dict[str, E2EManifest]:
    models_dir = repo_root / "tests" / "e2e" / "models"
    manifests: dict[str, E2EManifest] = {}
    paths = sorted({*models_dir.glob("*.json"), *models_dir.glob("*/manifests/*.json")})
    for path in paths:
        try:
            raw = json.loads(_read_text(path))
        except (OSError, json.JSONDecodeError):
            continue
        name = str(raw.get("name") or path.stem)
        family = str(raw.get("family") or "")
        if not family and path.parent.name == "manifests":
            family = path.parent.parent.name
        runtime_strategy = str(raw.get("runtime_strategy") or "")
        testcases = raw.get("testcases", [])
        testcase_names = [
            str(testcase.get("name") or "")
            for testcase in testcases
            if isinstance(testcase, dict) and testcase.get("name")
        ]
        result_case = (
            name
            if name in testcase_names
            else testcase_names[0]
            if testcase_names
            else name
        )
        result_testcase = next(
            (
                testcase
                for testcase in testcases
                if isinstance(testcase, dict)
                and str(testcase.get("name") or "") == result_case
            ),
            {},
        )
        if name and family and runtime_strategy:
            manifests[name] = E2EManifest(
                name=name,
                family=family,
                runtime_strategy=runtime_strategy,
                path=path,
                bundle=str(raw.get("bundle") or f"{name}.trtfb"),
                result_case=result_case,
                ci_tier=str(
                    result_testcase.get("ci_tier") or raw.get("ci_tier") or ""
                ),
            )
    return manifests


def discover_runtime_plugins(repo_root: Path) -> dict[str, RuntimePlugin]:
    runtime_dir = repo_root / "src" / "runtime" / "models"
    plugins: dict[str, RuntimePlugin] = {}
    for manifest in sorted(runtime_dir.glob("*/MODEL.toml")):
        text = _read_text(manifest)
        model_id = _toml_string(text, "id") or manifest.parent.name
        library = _toml_string(text, "runtime_library") or f"libtrtmc_model_{model_id}.so"
        strategies = _toml_list(text, "runtime_strategies")
        single_strategy = _toml_string(text, "runtime_strategy")
        if not strategies and single_strategy:
            strategies = (single_strategy,)
        if strategies:
            plugins[model_id] = RuntimePlugin(model_id, library, strategies)
    return plugins


def _selected_models_from_file(path: Path) -> set[str]:
    selected: set[str] = set()
    for raw in _read_text(path).splitlines():
        item = raw.strip()
        if not item:
            continue
        match = _NODE_ID_MODEL_RE.search(item)
        selected.add(match.group(1) if match else item)
    return selected


def selected_models(args: argparse.Namespace, manifests: dict[str, E2EManifest]) -> set[str]:
    selected: set[str] = set()
    if args.all:
        selected.update(manifests)
    for model in args.model:
        selected.add(model)
    for models_file in args.models_file:
        selected.update(_selected_models_from_file(models_file))
    for tests_file in args.tests_file:
        selected.update(_selected_models_from_file(tests_file))
    if args.impact_json is not None:
        impact = json.loads(_read_text(args.impact_json))
        selected.update(str(model) for model in impact.get("e2e_models", []))
        for node_id in impact.get("e2e_test_ids", []):
            match = _NODE_ID_MODEL_RE.search(str(node_id))
            if match:
                selected.add(match.group(1))
    return {model for model in selected if model}


def plugins_for_models(
    model_names: set[str],
    manifests: dict[str, E2EManifest],
    runtime_plugins: dict[str, RuntimePlugin],
) -> list[RuntimePlugin]:
    strategy_to_plugin = _plugins_by_strategy(runtime_plugins)

    selected: dict[str, RuntimePlugin] = {}
    missing_models: list[str] = []
    missing_strategies: list[str] = []
    for model in sorted(model_names):
        manifest = manifests.get(model)
        if manifest is None:
            missing_models.append(model)
            continue
        plugin = strategy_to_plugin.get(manifest.runtime_strategy)
        if plugin is None:
            missing_strategies.append(f"{model}:{manifest.runtime_strategy}")
            continue
        selected[plugin.model_id] = plugin

    if missing_models:
        raise SystemExit(
            "No E2E manifest found for selected model(s): " + ", ".join(missing_models)
        )
    if missing_strategies:
        raise SystemExit(
            "No runtime model plugin owns selected runtime_strategy value(s): "
            + ", ".join(missing_strategies)
        )
    return [selected[key] for key in sorted(selected)]


def _plugins_by_strategy(
    runtime_plugins: dict[str, RuntimePlugin],
) -> dict[str, RuntimePlugin]:
    strategy_to_plugin: dict[str, RuntimePlugin] = {}
    for plugin in runtime_plugins.values():
        for strategy in plugin.strategies:
            if strategy in strategy_to_plugin:
                other = strategy_to_plugin[strategy]
                raise SystemExit(
                    f"runtime_strategy {strategy!r} is owned by both "
                    f"{other.model_id!r} and {plugin.model_id!r}"
                )
            strategy_to_plugin[strategy] = plugin
    return strategy_to_plugin


def isolation_groups(
    model_names: set[str],
    manifests: dict[str, E2EManifest],
    runtime_plugins: dict[str, RuntimePlugin],
) -> list[dict[str, object]]:
    """Group cases that can share one single-family source projection."""
    strategy_to_plugin = _plugins_by_strategy(runtime_plugins)
    grouped: dict[tuple[str, str], list[str]] = {}
    missing_models: list[str] = []
    missing_strategies: list[str] = []
    for model_name in sorted(model_names):
        manifest = manifests.get(model_name)
        if manifest is None:
            missing_models.append(model_name)
            continue
        plugin = strategy_to_plugin.get(manifest.runtime_strategy)
        if plugin is None:
            missing_strategies.append(f"{model_name}:{manifest.runtime_strategy}")
            continue
        grouped.setdefault((manifest.family, plugin.model_id), []).append(model_name)

    if missing_models:
        raise SystemExit(
            "No E2E manifest found for selected model(s): " + ", ".join(missing_models)
        )
    if missing_strategies:
        raise SystemExit(
            "No runtime model plugin owns selected runtime_strategy value(s): "
            + ", ".join(missing_strategies)
        )

    groups: list[dict[str, object]] = []
    for (family, runtime_id), models in sorted(grouped.items()):
        plugin = runtime_plugins[runtime_id]
        group_id = f"{family}--{runtime_id}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", group_id):
            raise SystemExit(f"Isolation group has an unsafe identifier: {group_id!r}")
        groups.append(
            {
                "id": group_id,
                "family": family,
                "runtime_plugin": {
                    "model_id": plugin.model_id,
                    "library": plugin.library,
                    "strategies": list(plugin.strategies),
                    "target": plugin.target,
                },
                "models": models,
            }
        )
    return groups


def model_owned_impact_models(impact: dict[str, object]) -> list[str]:
    """Return final impacted cases reached through model-owned rules."""
    selected = {
        str(model) for model in impact.get("e2e_models", []) if str(model)
    }
    for node_id in impact.get("e2e_test_ids", []):
        match = _NODE_ID_MODEL_RE.search(str(node_id))
        if match:
            selected.add(match.group(1))

    owned: set[str] = set()
    for match in impact.get("matched_rules", []):
        if not isinstance(match, dict):
            continue
        if match.get("rule") not in _MODEL_OWNED_IMPACT_RULES:
            continue
        owned.update(str(model) for model in match.get("models", []) if str(model))
    for replacement in impact.get("l0_replacements", []):
        if not isinstance(replacement, dict):
            continue
        if replacement.get("model") in owned and replacement.get("replacement"):
            owned.add(str(replacement["replacement"]))
    return sorted(selected & owned)


def _git_paths(repo_root: Path, *, include_untracked: bool) -> list[Path]:
    cmd = ["git", "-C", str(repo_root), "ls-files", "-z", "--cached"]
    if include_untracked:
        cmd.extend(["--others", "--exclude-standard"])
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Could not list source files with git ls-files: {exc}") from exc
    return sorted(
        Path(os.fsdecode(raw))
        for raw in result.stdout.split(b"\0")
        if raw
    )


def _owner_under(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) <= 1:
        return ""
    return relative.parts[0]


def _include_source_path(path: Path, owners: dict[str, set[str]]) -> bool:
    for root, owner_group in _MODEL_OWNED_ROOTS.items():
        owner = _owner_under(path, root)
        if owner is None or owner == "":
            continue
        return owner in owners[owner_group]
    return True


def _copy_source_files(
    repo_root: Path,
    output_dir: Path,
    paths: Iterable[Path],
    owners: dict[str, set[str]],
) -> tuple[int, dict[str, int]]:
    copied = 0
    excluded = {root.as_posix(): 0 for root in _MODEL_OWNED_ROOTS}
    for relative in paths:
        source = repo_root / relative
        if not source.exists() and not source.is_symlink():
            continue
        if not _include_source_path(relative, owners):
            for root in _MODEL_OWNED_ROOTS:
                if _owner_under(relative, root) is not None:
                    excluded[root.as_posix()] += 1
                    break
            continue

        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            shutil.copy2(source, destination)
        copied += 1
    return copied, excluded


def _git_revision(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _prepare_output_dir(output_dir: Path, *, clean: bool) -> None:
    if output_dir.exists():
        if not clean:
            raise SystemExit(
                f"Isolation source output already exists: {output_dir}; pass --clean to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _reject_output_containing_repo(repo_root: Path, output_dir: Path) -> None:
    try:
        repo_root.relative_to(output_dir)
    except ValueError:
        return
    raise SystemExit("Isolation output must not be the repository root or one of its parents")


def command_stage_source(args: argparse.Namespace) -> int:
    """Create a source projection containing only selected model ownership roots."""
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    _reject_output_containing_repo(repo_root, output_dir)

    manifests = discover_e2e_manifests(repo_root)
    runtime_plugins = discover_runtime_plugins(repo_root)
    model_names = selected_models(args, manifests)
    if not model_names and not args.allow_empty:
        raise SystemExit("No E2E models selected")

    missing_models = sorted(model_names - manifests.keys())
    if missing_models:
        raise SystemExit(
            "No E2E manifest found for selected model(s): " + ", ".join(missing_models)
        )

    selected_manifests = [manifests[name] for name in sorted(model_names)]
    selected_plugins = plugins_for_models(model_names, manifests, runtime_plugins)
    families = {manifest.family for manifest in selected_manifests}
    runtime_ids = {plugin.model_id for plugin in selected_plugins}
    owners = {
        "builder_families": set(families),
        "runtime_plugins": runtime_ids,
        "e2e_families": set(families),
    }

    for family in sorted(families):
        family_dir = repo_root / "python" / "tensorrt_model_connect" / "families" / family
        e2e_dir = repo_root / "tests" / "e2e" / "models" / family
        if not family_dir.is_dir():
            raise SystemExit(f"Selected builder family directory does not exist: {family_dir}")
        if not e2e_dir.is_dir():
            raise SystemExit(f"Selected E2E family directory does not exist: {e2e_dir}")
    for runtime_id in sorted(runtime_ids):
        runtime_dir = repo_root / "src" / "runtime" / "models" / runtime_id
        if not runtime_dir.is_dir():
            raise SystemExit(f"Selected runtime plugin directory does not exist: {runtime_dir}")

    _prepare_output_dir(output_dir, clean=args.clean)
    paths = _git_paths(repo_root, include_untracked=args.include_untracked)
    copied_files, excluded_files = _copy_source_files(
        repo_root, output_dir, paths, owners
    )

    manifest = {
        "schema_version": 1,
        "source_root": str(repo_root),
        "source_revision": _git_revision(repo_root),
        "selected_models": sorted(model_names),
        "builder_families": sorted(families),
        "e2e_families": sorted(families),
        "runtime_plugins": [
            {
                "model_id": plugin.model_id,
                "library": plugin.library,
                "strategies": list(plugin.strategies),
                "target": plugin.target,
            }
            for plugin in selected_plugins
        ],
        "tracked_only": not args.include_untracked,
        "copied_files": copied_files,
        "excluded_model_files": excluded_files,
    }
    manifest_path = output_dir / ".trtmc-isolation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Staged {copied_files} files for models={','.join(sorted(model_names))} "
        f"families={','.join(sorted(families))} "
        f"runtime_plugins={','.join(sorted(runtime_ids))} at {output_dir}"
    )
    print(manifest_path)
    return 0


def command_plan(args: argparse.Namespace) -> int:
    """Write deterministic single-family build groups for selected E2E cases."""
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    _reject_output_containing_repo(repo_root, output_dir)

    manifests = discover_e2e_manifests(repo_root)
    runtime_plugins = discover_runtime_plugins(repo_root)
    model_names = selected_models(args, manifests)
    if not model_names and not args.allow_empty:
        raise SystemExit("No E2E models selected")
    groups = isolation_groups(model_names, manifests, runtime_plugins)

    _prepare_output_dir(output_dir, clean=args.clean)
    serialized_groups: list[dict[str, object]] = []
    for group in groups:
        group_id = str(group["id"])
        group_dir = output_dir / "groups" / group_id
        group_dir.mkdir(parents=True)
        models = [str(model) for model in group["models"]]
        models_file = group_dir / "models.txt"
        models_file.write_text("".join(f"{model}\n" for model in models), encoding="utf-8")
        group_manifest = dict(group)
        group_manifest["models_file"] = str(models_file.relative_to(output_dir))
        group_manifest_path = group_dir / "group.json"
        group_manifest_path.write_text(
            json.dumps(group_manifest, indent=2) + "\n", encoding="utf-8"
        )
        serialized_groups.append(group_manifest)

    plan = {
        "schema_version": 1,
        "source_root": str(repo_root),
        "source_revision": _git_revision(repo_root),
        "selected_models": sorted(model_names),
        "groups": serialized_groups,
    }
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"Planned {len(model_names)} model(s) in {len(serialized_groups)} "
        f"single-family isolation group(s)"
    )
    print(plan_path)
    return 0


def command_schedule(args: argparse.Namespace) -> int:
    """Balance isolation groups across fixed single-GPU worker queues."""
    plan_path = args.plan.resolve()
    output_dir = args.output_dir.resolve()
    plan = json.loads(_read_text(plan_path))
    groups = plan.get("groups")
    if not isinstance(groups, list) or not groups:
        raise SystemExit(f"Isolation plan has no groups: {plan_path}")
    gpu_ids = list(dict.fromkeys(args.gpu_id))
    if not gpu_ids:
        raise SystemExit("At least one --gpu-id is required")
    if any(not re.fullmatch(r"[0-9]+", gpu_id) for gpu_id in gpu_ids):
        raise SystemExit(f"GPU IDs must be non-negative integers: {gpu_ids}")
    if args.default_estimate_seconds < 0 or args.build_overhead_seconds < 0:
        raise SystemExit("Schedule estimates must be non-negative")

    estimates: dict[str, float] = {}
    if args.timing_estimates is not None and args.timing_estimates.is_file():
        timing_data = json.loads(_read_text(args.timing_estimates))
        raw_estimates = timing_data.get("estimates_s", {})
        if isinstance(raw_estimates, dict):
            estimates = {
                str(model): float(seconds)
                for model, seconds in raw_estimates.items()
                if isinstance(seconds, (int, float)) and seconds >= 0
            }

    scheduled_groups: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise SystemExit(f"Isolation plan contains a non-object group: {group!r}")
        group_id = str(group.get("id") or "")
        models = group.get("models")
        if not group_id or not isinstance(models, list) or not models:
            raise SystemExit(f"Invalid isolation group in {plan_path}: {group!r}")
        model_names = [str(model) for model in models]
        estimated_seconds = args.build_overhead_seconds + sum(
            estimates.get(model, args.default_estimate_seconds)
            for model in model_names
        )
        scheduled_groups.append(
            {
                "group_id": group_id,
                "group_manifest": str(
                    plan_path.parent / "groups" / group_id / "group.json"
                ),
                "models": model_names,
                "estimated_seconds": estimated_seconds,
            }
        )

    assignments: dict[str, list[dict[str, object]]] = {
        gpu_id: [] for gpu_id in gpu_ids
    }
    queue_totals = {gpu_id: 0.0 for gpu_id in gpu_ids}
    for group in sorted(
        scheduled_groups,
        key=lambda item: (-float(item["estimated_seconds"]), str(item["group_id"])),
    ):
        gpu_id = min(gpu_ids, key=lambda item: (queue_totals[item], int(item)))
        assignments[gpu_id].append(group)
        queue_totals[gpu_id] += float(group["estimated_seconds"])

    _prepare_output_dir(output_dir, clean=args.clean)
    for gpu_id, queue in assignments.items():
        queue_path = output_dir / f"gpu-{gpu_id}.txt"
        queue_path.write_text(
            "".join(f"{item['group_manifest']}\n" for item in queue),
            encoding="utf-8",
        )
    schedule = {
        "schema_version": 1,
        "plan": str(plan_path),
        "timing_estimates": (
            str(args.timing_estimates.resolve())
            if args.timing_estimates is not None
            else None
        ),
        "default_estimate_seconds": args.default_estimate_seconds,
        "build_overhead_seconds": args.build_overhead_seconds,
        "assignments": assignments,
        "queue_estimated_seconds": queue_totals,
    }
    schedule_path = output_dir / "schedule.json"
    schedule_path.write_text(
        json.dumps(schedule, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Scheduled {len(scheduled_groups)} isolation group(s) across "
        f"{len(gpu_ids)} GPU queue(s)"
    )
    for gpu_id in gpu_ids:
        print(
            f"  GPU {gpu_id}: {len(assignments[gpu_id])} group(s), "
            f"estimated {queue_totals[gpu_id] / 60:.1f}m"
        )
    print(schedule_path)
    return 0


def _returncode_failures(value: object, path: str = "result") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key == "returncode" and item != 0:
                failures.append(f"{item_path} is {item!r}, expected 0")
            failures.extend(_returncode_failures(item, item_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(_returncode_failures(item, f"{path}[{index}]"))
    return failures


def _optional_stage_names(result: dict[str, object]) -> set[str]:
    case_config = result.get("case_config")
    if not isinstance(case_config, dict):
        return set()
    stage_specs = case_config.get("stages")
    if not isinstance(stage_specs, list):
        return set()
    return {
        str(stage_spec["name"])
        for stage_spec in stage_specs
        if isinstance(stage_spec, dict)
        and stage_spec.get("name")
        and stage_spec.get("required") is False
    }


def _load_verified_build_records(
    report_path: Path,
    expected_models: set[str],
) -> dict[str, dict[str, object]]:
    try:
        report = json.loads(_read_text(report_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Build verification report could not be read: {report_path}: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise SystemExit("Build verification report is not a passing object")
    raw_records = report.get("records")
    if not isinstance(raw_records, list):
        raise SystemExit("Build verification report records is not a list")

    records: dict[str, dict[str, object]] = {}
    for index, raw_record in enumerate(raw_records):
        if (
            not isinstance(raw_record, dict)
            or raw_record.get("passed") is not True
            or not isinstance(raw_record.get("record"), dict)
        ):
            raise SystemExit(
                f"Build verification report records[{index}] is not a passing record"
            )
        record = raw_record["record"]
        identity = record.get("identity")
        if (
            not isinstance(identity, str)
            or not identity
            or raw_record.get("identity") != identity
            or identity in records
        ):
            raise SystemExit(
                f"Build verification report records[{index}] has an invalid identity"
            )
        records[identity] = record
    if set(records) != expected_models:
        raise SystemExit(
            "Build verification report identities do not match selected models"
        )
    return records


def _command_verification_errors(
    commands: object,
    build_record: dict[str, object] | None,
) -> list[str]:
    if commands is None and build_record is None:
        return []
    if not isinstance(commands, list):
        return ["commands is not a list"]

    errors: list[str] = []
    verified_prefix_length = 0
    if build_record is not None:
        build_command = build_record.get("command")
        if (
            not isinstance(build_command, list)
            or not build_command
            or not all(isinstance(item, str) and item for item in build_command)
        ):
            errors.append("verified build command is not a non-empty string list")
            build_command = []
        attempt_count = build_record.get("attempt_count")
        expected_prefix = (
            [
                ("build_recovery_attempt_1", -signal.SIGSEGV),
                ("build", 0),
            ]
            if type(attempt_count) is int and attempt_count == 2
            else [("build", 0)]
            if type(attempt_count) is int and attempt_count == 1
            else []
        )
        if not expected_prefix:
            errors.append(
                f"verified build attempt_count is {attempt_count!r}, expected 1 or 2"
            )
        verified_prefix_length = len(expected_prefix)
        for index, (expected_label, expected_returncode) in enumerate(expected_prefix):
            command = commands[index] if index < len(commands) else None
            valid = (
                isinstance(command, dict)
                and command.get("label") == expected_label
                and command.get("command") == build_command
                and type(command.get("returncode")) is int
                and command["returncode"] == expected_returncode
            )
            if not valid:
                errors.append(
                    f"commands[{index}] does not match verified "
                    f"{expected_label!r} build evidence"
                )

    for index, command in enumerate(
        commands[verified_prefix_length:],
        start=verified_prefix_length,
    ):
        if not isinstance(command, dict):
            errors.append(f"commands[{index}] is not an object")
            continue
        if command.get("label") in {
            "build",
            "build_recovery_attempt_1",
        }:
            errors.append(
                f"commands[{index}].label is reserved for verified build evidence"
            )
        returncode = command.get("returncode")
        if type(returncode) is not int or returncode != 0:
            errors.append(
                f"commands[{index}].returncode is {returncode!r}, expected 0"
            )
    return errors


def _verify_model_result(
    model_name: str,
    result_case: str,
    artifacts_dir: Path,
    build_record: dict[str, object] | None = None,
) -> dict[str, object]:
    result_path = artifacts_dir / result_case / "result.json"
    errors: list[str] = []
    if not result_path.is_file():
        return {
            "model": model_name,
            "result_path": str(result_path),
            "proof_kind": "invalid",
            "passed": False,
            "errors": ["result.json is missing"],
        }
    try:
        result = json.loads(_read_text(result_path))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "model": model_name,
            "result_path": str(result_path),
            "proof_kind": "invalid",
            "passed": False,
            "errors": [f"result.json could not be read: {exc}"],
        }
    if not isinstance(result, dict):
        errors.append("result.json root is not an object")
        result = {}

    if result.get("case_name") != result_case:
        errors.append(f"case_name is {result.get('case_name')!r}, expected {result_case!r}")
    if result.get("status") != "pass":
        errors.append(f"status is {result.get('status')!r}, expected 'pass'")
    if result.get("failure_type") not in (None, ""):
        errors.append(f"failure_type is {result.get('failure_type')!r}")

    oracle_level = result.get("oracle_level")
    proof_kind = (
        _ORACLE_PROOF_KINDS.get(oracle_level, "invalid")
        if isinstance(oracle_level, str)
        else "invalid"
    )
    if proof_kind == "invalid":
        errors.append(
            f"oracle_level is {oracle_level!r}, expected one of {sorted(_ORACLE_PROOF_KINDS)}"
        )

    optional_stage_names = _optional_stage_names(result)
    stages = result.get("stages")
    if not isinstance(stages, dict) or not stages:
        errors.append("stages is missing or empty")
    else:
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict):
                errors.append(f"stage {stage_name!r} is not an object")
                continue
            stage_status = stage.get("status")
            optional_skip = (
                stage_name in optional_stage_names and stage_status == "skipped"
            )
            if stage_status != "passed" and not optional_skip:
                errors.append(
                    f"stage {stage_name!r} status is {stage_status!r}, expected 'passed'"
                )
            metrics = stage.get("metrics", {})
            if not isinstance(metrics, dict):
                errors.append(f"stage {stage_name!r} metrics is not an object")

    errors.extend(
        _command_verification_errors(result.get("commands"), build_record)
    )

    errors.extend(_returncode_failures(result.get("stage_outputs", {}), "stage_outputs"))
    errors = list(dict.fromkeys(errors))
    return {
        "model": model_name,
        "result_path": str(result_path),
        "proof_kind": proof_kind,
        "passed": not errors,
        "errors": errors,
    }


def command_verify_results(args: argparse.Namespace) -> int:
    """Require a complete passing E2E artifact for every selected model."""
    repo_root = args.repo_root.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    manifests = discover_e2e_manifests(repo_root)
    model_names = selected_models(args, manifests)
    if not model_names and not args.allow_empty:
        raise SystemExit("No E2E models selected")
    missing_models = sorted(model_names - manifests.keys())
    if missing_models:
        raise SystemExit(
            "No E2E manifest found for selected model(s): " + ", ".join(missing_models)
        )
    build_records = (
        _load_verified_build_records(
            args.build_verification_report.resolve(),
            model_names,
        )
        if args.build_verification_report is not None
        else {}
    )

    results = [
        _verify_model_result(
            model_name,
            manifests[model_name].result_case,
            artifacts_dir,
            build_records.get(model_name),
        )
        for model_name in sorted(model_names)
    ]
    report = {
        "schema_version": 1,
        "artifacts_dir": str(artifacts_dir),
        "selected_models": sorted(model_names),
        "passed": all(bool(result["passed"]) for result in results),
        "results": results,
    }
    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report_path)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['model']}")
        for error in result["errors"]:
            print(f"  {error}", file=sys.stderr)
    return 0 if report["passed"] else 1


def command_verify_builds(args: argparse.Namespace) -> int:
    """Require exactly one completed full-bundle build per selected model."""
    expected_models = selected_models(args, {})
    if not expected_models and not args.allow_empty:
        raise SystemExit("No E2E models selected")

    ledger_dir = args.ledger_dir.resolve()
    expected_revision = str(args.source_revision or "")
    records: list[dict[str, object]] = []
    errors: list[str] = []
    identities: list[str] = []
    record_paths = sorted(ledger_dir.glob("*.json")) if ledger_dir.is_dir() else []
    for record_path in record_paths:
        try:
            record = json.loads(_read_text(record_path))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"could not read build ledger {record_path}: {exc}")
            continue
        if not isinstance(record, dict):
            errors.append(f"build ledger is not an object: {record_path}")
            continue
        identity = str(record.get("identity") or "")
        identities.append(identity)
        record_errors: list[str] = []
        if not identity:
            record_errors.append("identity is missing")
        schema_version = record.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            record_errors.append(
                f"schema_version is {schema_version!r}, expected 1"
            )
        invocation_count = record.get("invocation_count")
        if type(invocation_count) is not int or invocation_count != 1:
            record_errors.append(
                f"invocation_count is {invocation_count!r}, expected 1"
            )
        attempt_count = record.get("attempt_count")
        valid_attempt_count = (
            type(attempt_count) is int and 1 <= attempt_count <= 2
        )
        if not valid_attempt_count:
            record_errors.append(f"attempt_count is {attempt_count!r}, expected 1 or 2")
        builder_pid = record.get("builder_pid")
        if not _positive_pid(builder_pid):
            record_errors.append(
                f"builder_pid is {builder_pid!r}, expected a positive integer"
            )
        started_at = record.get("started_at")
        started_at_time = _utc_timestamp(started_at)
        if started_at_time is None:
            record_errors.append(
                f"started_at is {started_at!r}, expected a UTC timestamp"
            )
        recovery_attempts = record.get("recovery_attempts")
        expected_recoveries = attempt_count - 1 if valid_attempt_count else -1
        if not isinstance(recovery_attempts, list) or len(recovery_attempts) != expected_recoveries:
            record_errors.append("recovery_attempts does not match attempt_count")
        elif any(
            not isinstance(recovery, dict)
            or type(recovery.get("attempt")) is not int
            or recovery["attempt"] != index
            or type(recovery.get("returncode")) is not int
            or recovery["returncode"] != -signal.SIGSEGV
            or type(recovery.get("signal")) is not int
            or recovery["signal"] != signal.SIGSEGV
            for index, recovery in enumerate(recovery_attempts, start=1)
        ):
            record_errors.append("recovery_attempts must contain only ordered SIGSEGV recoveries")
        elif recovery_attempts:
            recovery = recovery_attempts[0]
            recovery_started_at = _utc_timestamp(recovery.get("started_at"))
            recovered_at = recovery.get("recovered_at")
            recovered_at_time = _utc_timestamp(recovered_at)
            complete_fresh_process_evidence = (
                frozenset(recovery) == _RECOVERY_FIELDS
                and _positive_pid(recovery.get("builder_pid"))
                and _positive_pid(builder_pid)
                and recovery.get("builder_pid") != builder_pid
                and recovery_started_at is not None
                and recovered_at_time is not None
                and recovered_at == started_at
                and started_at_time == recovered_at_time
                and recovery_started_at <= recovered_at_time
            )
            if not complete_fresh_process_evidence:
                record_errors.append(
                    "recovery_attempts must contain complete ordered "
                    "fresh-process SIGSEGV evidence"
                )
        if record.get("status") != "passed":
            record_errors.append(
                f"status is {record.get('status')!r}, expected 'passed'"
            )
        returncode = record.get("returncode")
        if type(returncode) is not int or returncode != 0:
            record_errors.append(
                f"returncode is {returncode!r}, expected 0"
            )
        if expected_revision and record.get("source_revision") != expected_revision:
            record_errors.append(
                "source_revision is "
                f"{record.get('source_revision')!r}, expected {expected_revision!r}"
            )
        bundle_path = Path(str(record.get("bundle_path") or ""))
        if not bundle_path.is_file():
            record_errors.append(f"bundle is missing: {bundle_path}")
        timing_path = Path(str(record.get("build_timing_path") or ""))
        if not timing_path.is_file():
            record_errors.append(f"build timing is missing: {timing_path}")
        records.append(
            {
                "path": str(record_path),
                "identity": identity,
                "passed": not record_errors,
                "errors": record_errors,
                "record": record,
            }
        )

    observed_models = {identity for identity in identities if identity}
    duplicate_models = sorted(
        identity for identity in observed_models if identities.count(identity) != 1
    )
    missing_models = sorted(expected_models - observed_models)
    unexpected_models = sorted(observed_models - expected_models)
    if duplicate_models:
        errors.append("duplicate build ledgers: " + ", ".join(duplicate_models))
    if missing_models:
        errors.append("missing build ledgers: " + ", ".join(missing_models))
    if unexpected_models:
        errors.append("unexpected build ledgers: " + ", ".join(unexpected_models))
    for record in records:
        for error in record["errors"]:
            errors.append(f"{record['identity'] or record['path']}: {error}")

    report = {
        "schema_version": 1,
        "ledger_dir": str(ledger_dir),
        "expected_models": sorted(expected_models),
        "observed_models": sorted(observed_models),
        "source_revision": expected_revision,
        "builds_per_model": 1,
        "passed": not errors,
        "errors": errors,
        "records": records,
    }
    if args.report is not None:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report_path)

    passing_records = {
        str(record["identity"])
        for record in records
        if bool(record["passed"])
    }
    for model in sorted(expected_models | observed_models):
        passed = (
            model in expected_models
            and model in passing_records
            and model not in duplicate_models
        )
        print(f"{'PASS' if passed else 'FAIL'} {model}")
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    return 0 if report["passed"] else 1


def command_impact_models(args: argparse.Namespace) -> int:
    """Print impacted cases that came from model-owned classification rules."""
    impact = json.loads(_read_text(args.impact_json))
    if not isinstance(impact, dict):
        raise SystemExit(f"Impact JSON root is not an object: {args.impact_json}")
    excluded_ci_tiers = set(args.exclude_ci_tier)
    manifests = discover_e2e_manifests(args.repo_root.resolve())
    for model_name in model_owned_impact_models(impact):
        manifest = manifests.get(model_name)
        if manifest is not None and manifest.ci_tier in excluded_ci_tiers:
            continue
        print(model_name)
    return 0


def command_targets(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    manifests = discover_e2e_manifests(repo_root)
    runtime_plugins = discover_runtime_plugins(repo_root)
    model_names = selected_models(args, manifests)
    if not model_names and not args.allow_empty:
        raise SystemExit("No E2E models selected")
    for plugin in plugins_for_models(model_names, manifests, runtime_plugins):
        print(plugin.target)
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifests = discover_e2e_manifests(repo_root)
    runtime_plugins = discover_runtime_plugins(repo_root)
    model_names = selected_models(args, manifests)
    if not model_names and not args.allow_empty:
        raise SystemExit("No E2E models selected")

    prepared = plugins_for_models(model_names, manifests, runtime_plugins)
    output_dir.mkdir(parents=True, exist_ok=True)
    for plugin in prepared:
        src = build_dir / "models" / plugin.model_id / plugin.library
        if not src.is_file():
            raise SystemExit(f"Runtime model plugin library not found: {src}")
        dst_dir = output_dir / plugin.model_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / plugin.library
        shutil.copy2(src, dst)
        print(f"{plugin.target} {dst}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_selection_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-root", type=Path, default=Path.cwd())
        p.add_argument("--impact-json", type=Path)
        p.add_argument("--models-file", type=Path, action="append", default=[])
        p.add_argument("--tests-file", type=Path, action="append", default=[])
        p.add_argument("--model", action="append", default=[])
        p.add_argument("--all", action="store_true")
        p.add_argument("--allow-empty", action="store_true")

    targets = subparsers.add_parser("targets", help="Print required CMake targets")
    add_selection_options(targets)
    targets.set_defaults(func=command_targets)

    prepare = subparsers.add_parser(
        "prepare",
        help="Copy required runtime plugin DSOs into an isolated directory",
    )
    add_selection_options(prepare)
    prepare.add_argument("--build-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(func=command_prepare)

    stage_source = subparsers.add_parser(
        "stage-source",
        help="Copy a filtered source tree containing only selected model ownership roots",
    )
    add_selection_options(stage_source)
    stage_source.add_argument("--output-dir", type=Path, required=True)
    stage_source.add_argument(
        "--clean",
        action="store_true",
        help="Replace an existing generated isolation source directory",
    )
    stage_source.add_argument(
        "--include-untracked",
        action="store_true",
        help="Include untracked, non-ignored source files in the projection",
    )
    stage_source.set_defaults(func=command_stage_source)

    plan = subparsers.add_parser(
        "plan",
        help="Group selected E2E cases into single-family build projections",
    )
    add_selection_options(plan)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument(
        "--clean",
        action="store_true",
        help="Replace an existing generated isolation plan directory",
    )
    plan.set_defaults(func=command_plan)

    schedule = subparsers.add_parser(
        "schedule",
        help="Balance an isolation plan across fixed single-GPU queues",
    )
    schedule.add_argument("--plan", type=Path, required=True)
    schedule.add_argument("--output-dir", type=Path, required=True)
    schedule.add_argument("--timing-estimates", type=Path)
    schedule.add_argument("--gpu-id", action="append", default=[])
    schedule.add_argument("--default-estimate-seconds", type=float, default=600.0)
    schedule.add_argument("--build-overhead-seconds", type=float, default=60.0)
    schedule.add_argument(
        "--clean",
        action="store_true",
        help="Replace an existing generated isolation schedule directory",
    )
    schedule.set_defaults(func=command_schedule)

    verify_results = subparsers.add_parser(
        "verify-results",
        help="Require complete passing E2E artifacts for every selected model",
    )
    add_selection_options(verify_results)
    verify_results.add_argument("--artifacts-dir", type=Path, required=True)
    verify_results.add_argument("--build-verification-report", type=Path)
    verify_results.add_argument("--report", type=Path)
    verify_results.set_defaults(func=command_verify_results)

    verify_builds = subparsers.add_parser(
        "verify-builds",
        help="Require one completed full-bundle build per selected model",
    )
    add_selection_options(verify_builds)
    verify_builds.add_argument("--ledger-dir", type=Path, required=True)
    verify_builds.add_argument("--source-revision", default="")
    verify_builds.add_argument("--report", type=Path)
    verify_builds.set_defaults(func=command_verify_builds)

    impact_models = subparsers.add_parser(
        "impact-models",
        help="Print final impacted E2E cases reached through model-owned rules",
    )
    impact_models.add_argument("--repo-root", type=Path, default=Path.cwd())
    impact_models.add_argument("--impact-json", type=Path, required=True)
    impact_models.add_argument("--exclude-ci-tier", action="append", default=[])
    impact_models.set_defaults(func=command_impact_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

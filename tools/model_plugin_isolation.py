#!/usr/bin/env python3
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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class E2EManifest:
    name: str
    family: str
    runtime_strategy: str
    path: Path
    bundle: str = ""


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
        if name and family and runtime_strategy:
            manifests[name] = E2EManifest(
                name=name,
                family=family,
                runtime_strategy=runtime_strategy,
                path=path,
                bundle=str(raw.get("bundle") or f"{name}.trtfb"),
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


def command_stage_source(args: argparse.Namespace) -> int:
    """Create a source projection containing only selected model ownership roots."""
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    try:
        repo_root.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise SystemExit(
            "Isolation source output must not be the repository root or one of its parents"
        )

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

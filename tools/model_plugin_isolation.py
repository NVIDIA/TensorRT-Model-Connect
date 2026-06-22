#!/usr/bin/env python3
"""Resolve and prepare isolated runtime model plugin directories for E2E.

The E2E model family is not always the same as the runtime plugin owner. For
example, many text families use the ``text_generation`` runtime plugin. This
tool maps selected E2E models to the owning ``src/runtime/models/<id>`` plugin
and can copy only those DSOs out of a CMake build tree.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

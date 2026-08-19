#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize a source tree containing one model family.

The filtered tree keeps generic repository infrastructure, but narrows each
model-owned root to the selected Python family and the runtime model plugins
required by that family's E2E manifests.  Building and testing from the result
therefore catches undeclared sibling-family dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

try:
    from tools import model_plugin_isolation
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    import model_plugin_isolation


PYTHON_FAMILIES = PurePosixPath("python/tensorrt_model_connect/families")
RUNTIME_MODELS = PurePosixPath("src/runtime/models")
CPP_MODEL_TESTS = PurePosixPath("tests/cpp/models")
E2E_MODELS = PurePosixPath("tests/e2e/models")
PUBLIC_MODEL_HEADERS = PurePosixPath("include/trtmc/models")
FAMILY_TOOLS = PurePosixPath("tools/families")
BUILDER_FAMILY_TESTS = PurePosixPath("tests/builder/families")

# These files are registry/protocol infrastructure.  Any additional shared
# family module must be reviewed and added explicitly instead of silently
# becoming available to every isolated family.
APPROVED_SHARED_FAMILY_FILES = frozenset({"__init__.py", "base.py"})


@dataclass(frozen=True)
class FamilySourceSelection:
    family: str
    runtime_models: tuple[str, ...]
    e2e_models: tuple[str, ...]
    public_headers: tuple[str, ...]


def resolve_selection(repo_root: Path, family: str) -> FamilySourceSelection:
    repo_root = repo_root.resolve()
    family_dir = repo_root / PYTHON_FAMILIES / family
    if not (family_dir / "plugin.py").is_file():
        raise SystemExit(f"Unknown Python model family: {family}")

    manifests = model_plugin_isolation.discover_e2e_manifests(repo_root)
    selected_models = {
        name for name, manifest in manifests.items() if manifest.family == family
    }
    runtime_plugins = model_plugin_isolation.discover_runtime_plugins(repo_root)
    owners = {
        plugin.model_id
        for plugin in model_plugin_isolation.plugins_for_models(
            selected_models, manifests, runtime_plugins
        )
    } if selected_models else set()

    # A family without an E2E manifest can still own a same-named runtime
    # plugin.  Keeping it makes the filtered tree useful while manifest
    # coverage is being added; normal covered families use the mapping above.
    if not owners and family in runtime_plugins:
        owners.add(family)
    if not owners:
        raise SystemExit(
            f"Family {family!r} has no E2E runtime mapping or same-named runtime plugin"
        )

    public_headers: set[str] = set()
    for owner in owners:
        manifest = repo_root / RUNTIME_MODELS / owner / "MODEL.toml"
        with manifest.open("rb") as source:
            declared = tomllib.load(source).get("public_headers", [])
        if not isinstance(declared, list) or any(
            not isinstance(name, str)
            or not name
            or PurePosixPath(name).is_absolute()
            or len(PurePosixPath(name).parts) != 1
            or name in {".", ".."}
            for name in declared
        ):
            raise SystemExit(f"Runtime model {owner!r} has invalid public_headers")
        for name in declared:
            header = repo_root / PUBLIC_MODEL_HEADERS / name
            if header.is_symlink() or not header.is_file():
                raise SystemExit(f"Runtime model {owner!r} has missing public header {name!r}")
            public_headers.add(name)

    return FamilySourceSelection(
        family=family,
        runtime_models=tuple(sorted(owners)),
        e2e_models=tuple(sorted(selected_models)),
        public_headers=tuple(sorted(public_headers)),
    )


def _owned_child(path: PurePosixPath, root: PurePosixPath) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    return relative.parts[0] if len(relative.parts) > 1 else ""


def include_path(path: PurePosixPath, selection: FamilySourceSelection) -> bool:
    try:
        public_header = path.relative_to(PUBLIC_MODEL_HEADERS)
    except ValueError:
        public_header = None
    if public_header is not None:
        if not public_header.parts:
            return True
        return len(public_header.parts) == 1 and public_header.name in selection.public_headers

    child = _owned_child(path, PYTHON_FAMILIES)
    if child is not None:
        if not child:
            return path.name in APPROVED_SHARED_FAMILY_FILES
        return child == selection.family

    child = _owned_child(path, RUNTIME_MODELS)
    if child is not None:
        return not child or child in selection.runtime_models

    child = _owned_child(path, CPP_MODEL_TESTS)
    if child is not None:
        return not child or child in selection.runtime_models

    child = _owned_child(path, E2E_MODELS)
    if child is not None:
        return not child or child == selection.family

    for root in (FAMILY_TOOLS, BUILDER_FAMILY_TESTS):
        child = _owned_child(path, root)
        if child is not None:
            return not child or child == selection.family

    return True


def tracked_files(repo_root: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "ls-files",
            "-z",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(
        PurePosixPath(os.fsdecode(raw))
        for raw in result.stdout.split(b"\0")
        if raw
    )


def selected_worktree_files(
    repo_root: Path,
    selection: FamilySourceSelection,
) -> tuple[PurePosixPath, ...]:
    """Include new files only from roots owned by the selected family."""
    roots = (
        PYTHON_FAMILIES / selection.family,
        FAMILY_TOOLS / selection.family,
        BUILDER_FAMILY_TESTS / selection.family,
        E2E_MODELS / selection.family,
        *(RUNTIME_MODELS / model for model in selection.runtime_models),
        *(CPP_MODEL_TESTS / model for model in selection.runtime_models),
        *(PUBLIC_MODEL_HEADERS / name for name in selection.public_headers),
    )
    files: set[PurePosixPath] = set()
    for relative_root in roots:
        root = repo_root / relative_root
        if not root.is_dir():
            continue
        files.update(
            PurePosixPath(path.relative_to(repo_root).as_posix())
            for path in root.rglob("*")
            if (path.is_file() or path.is_symlink())
            and "__pycache__" not in path.parts
        )
    return tuple(sorted(files))


def materialize(
    repo_root: Path,
    output_dir: Path,
    selection: FamilySourceSelection,
    *,
    force: bool = False,
) -> int:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == repo_root:
        raise SystemExit("Output directory must differ from the repository root")
    if output_dir.exists():
        if not force:
            raise SystemExit(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied = 0
    source_files = set(tracked_files(repo_root))
    source_files.update(selected_worktree_files(repo_root, selection))
    for relative in sorted(source_files):
        if not include_path(relative, selection):
            continue
        source = repo_root / relative
        # Respect working-tree deletions while still sourcing the file list from
        # Git so unrelated untracked files cannot leak into the isolated tree.
        if not source.exists() and not source.is_symlink():
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, destination)
        copied += 1

    metadata = asdict(selection)
    metadata["copied_files"] = copied
    (output_dir / ".trtmc-family-source.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the resolved ownership plan")
    plan.add_argument("--family", required=True)

    create = subparsers.add_parser("create", help="Create the filtered source tree")
    create.add_argument("--family", required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection = resolve_selection(args.repo_root, args.family)
    if args.command == "plan":
        print(json.dumps(asdict(selection), indent=2, sort_keys=True))
        return 0
    copied = materialize(
        args.repo_root,
        args.output_dir,
        selection,
        force=args.force,
    )
    print(
        f"family={selection.family} runtime_models="
        f"{','.join(selection.runtime_models)} copied_files={copied} "
        f"output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

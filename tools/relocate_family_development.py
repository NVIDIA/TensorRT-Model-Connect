#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move development-only modules out of family runtime packages."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tools import migrate_family_layout as layout_tools
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    import migrate_family_layout as layout_tools


@dataclass(frozen=True)
class Relocation:
    family: str
    source: str
    destination: str


RELOCATIONS = (
    Relocation("bark", "model/debug_runner.py", "tools/families/bark/debug_runner.py"),
    Relocation("deepseek_ocr", "model/vl_debug_runner.py", "tools/families/deepseek_ocr/vl_debug_runner.py"),
    Relocation("elf_flow", "model/debug_runner.py", "tools/families/elf_flow/debug_runner.py"),
    Relocation("flux", "model/diffusion_runner.py", "tools/families/flux/diffusion_runner.py"),
    Relocation("flux", "model/schedulers", "tools/families/flux/schedulers"),
    Relocation("internvl", "model/vl_debug_runner.py", "tools/families/internvl/vl_debug_runner.py"),
    Relocation("lance", "model/vl_debug_runner.py", "tools/families/lance/vl_debug_runner.py"),
    Relocation("locateanything", "model/vl_debug_runner.py", "tools/families/locateanything/vl_debug_runner.py"),
    Relocation("phi4_multimodal", "model/vl_debug_runner.py", "tools/families/phi4_multimodal/vl_debug_runner.py"),
    Relocation("pixart", "model/diffusion_runner.py", "tools/families/pixart/diffusion_runner.py"),
    Relocation("pixart", "model/schedulers", "tools/families/pixart/schedulers"),
    Relocation("qwen_vl", "model/onnx_vision_builder.py", "tools/families/qwen_vl/onnx_vision_builder.py"),
    Relocation("qwen_vl", "model/vl_debug_runner.py", "tools/families/qwen_vl/vl_debug_runner.py"),
    Relocation("segformer", "model/debug_runner.py", "tools/families/segformer/debug_runner.py"),
    Relocation("wan_t2v", "model/diffusion_runner.py", "tools/families/wan_t2v/diffusion_runner.py"),
    Relocation("wan_t2v", "model/schedulers", "tools/families/wan_t2v/schedulers"),
    Relocation("whisper", "model/debug_runner.py", "tools/families/whisper/debug_runner.py"),
    Relocation("z_image", "model/diffusion_runner.py", "tools/families/z_image/diffusion_runner.py"),
    Relocation("z_image", "model/schedulers", "tools/families/z_image/schedulers"),
)


def _source(repo_root: Path, relocation: Relocation) -> Path:
    return (
        repo_root
        / "python/tensorrt_model_connect/families"
        / relocation.family
        / relocation.source
    )


def pending_relocations(
    repo_root: Path,
    families: frozenset[str],
) -> list[Relocation]:
    return [
        relocation
        for relocation in RELOCATIONS
        if (not families or relocation.family in families)
        and _source(repo_root, relocation).exists()
    ]


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())


def apply_relocations(repo_root: Path, families: frozenset[str]) -> None:
    repo_root = repo_root.resolve()
    pending = pending_relocations(repo_root, families)
    mapping: dict[str, str] = {}
    reverse_paths: dict[Path, Path] = {}

    for relocation in pending:
        source = _source(repo_root, relocation)
        destination = repo_root / relocation.destination
        if destination.exists():
            raise SystemExit(f"Relocation destination exists: {destination}")
        source_files = _files(source)
        for source_file in source_files:
            relative = source_file.relative_to(source) if source.is_dir() else Path()
            destination_file = destination / relative if source.is_dir() else destination
            old_module = layout_tools._module_name(repo_root, source_file)
            new_module = layout_tools._module_name(repo_root, destination_file)
            if old_module and new_module:
                mapping[old_module] = new_module
            reverse_paths[destination_file.resolve()] = source_file

        destination.parent.mkdir(parents=True, exist_ok=True)
        family_tools = repo_root / "tools/families" / relocation.family
        family_tools.mkdir(parents=True, exist_ok=True)
        init = family_tools / "__init__.py"
        if not init.exists():
            init.write_text('"""Family-owned development tools."""\n', encoding="utf-8")
        shutil.move(str(source), str(destination))

    if not mapping:
        return
    mapping = layout_tools._collapse_mapping(mapping)
    for path in layout_tools._rewrite_python_paths(repo_root):
        current = layout_tools._module_name(repo_root, path)
        old_path = reverse_paths.get(path.resolve(), path)
        old = layout_tools._module_name(repo_root, old_path)
        if not current or not old:
            continue
        layout_tools._rewrite_imports(
            path,
            old_module=old,
            new_module=current,
            old_is_init=old_path.name == "__init__.py",
            new_is_init=path.name == "__init__.py",
            mapping=mapping,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    families = frozenset(args.family)
    pending = pending_relocations(repo_root, families)
    if args.command == "check":
        for relocation in pending:
            print(f"{relocation.family}: {relocation.source} -> {relocation.destination}")
        return 1 if pending else 0
    apply_relocations(repo_root, families)
    print(f"relocated_family_development_modules={len(pending)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Migrate and validate model families against the canonical component layout."""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


CANONICAL_TOP_LEVEL = frozenset(
    {"MODEL.toml", "__init__.py", "plugin.py", "config.py", "weights", "model"}
)
OPTIONAL_TOP_LEVEL = frozenset({"runtime_config_schema.py", "profiles"})
ALLOWED_TOP_LEVEL = CANONICAL_TOP_LEVEL | OPTIONAL_TOP_LEVEL
CORE_MODEL_MODULES = ("graph_ops.py", "graph_blocks.py", "utils.py")
WEIGHT_MODULES = frozenset(
    {
        "bundle_sections.py",
        "magpie_tokenizer.py",
        "nemo_archive.py",
        "tokenizer_json.py",
    }
)
TOOL_MODULES = frozenset(
    {
        "bench_flashinfer_e2e.py",
        "bench_flux2_perf.py",
        "benchmark_qwen3_8b_aime25_vs_hf.py",
        "build_fp8_onnx_monolithic.py",
        "build_wan14b.py",
        "cpu_profile_matrix.py",
        "debug_diffusion_pipeline.py",
        "diff_audio.py",
        "diff_logits.py",
        "diff_personaplex.py",
        "diff_segmentation.py",
        "diff_t5.py",
        "diff_vl.py",
        "gen_fp8_bf16.py",
        "inject_fp8_qdq_proto.py",
        "make_replay_artifact.py",
        "mk_fp8_bf16_bundle.py",
        "perf_hooks.py",
        "prepare_model.py",
        "prepare_model_dir.py",
        "profile.py",
        "quantize_flux2_fp8.py",
        "validate_dit.py",
        "validate_replay_artifact.py",
        "validate_t5.py",
    }
)
MINIMAL_INIT = 'from .plugin import plugin\n\n__all__ = ["plugin"]\n'
MODEL_INIT = '"""Family-owned TensorRT model construction components."""\n'
TOOLS_INIT = '"""Family-owned development tools."""\n'
TESTS_INIT = '"""Family-specific builder tests."""\n'


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path
    kind: str


def family_dirs(
    repo_root: Path,
    families: tuple[str, ...] = (),
) -> list[Path]:
    root = repo_root / "python/tensorrt_model_connect/families"
    available = {
        path.name: path
        for path in root.iterdir()
        if (path / "plugin.py").is_file()
    }
    if not families:
        return [available[name] for name in sorted(available)]
    unknown = sorted(set(families) - set(available))
    if unknown:
        raise SystemExit(f"Unknown model families: {', '.join(unknown)}")
    return [available[name] for name in sorted(set(families))]


def _module_name(repo_root: Path, path: Path) -> str | None:
    if path.suffix != ".py":
        return None
    relative = path.relative_to(repo_root)
    parts = list(relative.parts)
    if parts[0] == "python":
        parts.pop(0)
    parts[-1] = Path(parts[-1]).stem
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and "__pycache__" not in candidate.parts
    )


def planned_moves(
    repo_root: Path,
    families: tuple[str, ...] = (),
) -> list[Move]:
    moves: list[Move] = []
    tests_root = repo_root / "tests/builder/families"
    tools_root = repo_root / "tools/families"

    for family_dir in family_dirs(repo_root, families):
        family = family_dir.name
        for path in sorted(family_dir.iterdir()):
            if path.name in ALLOWED_TOP_LEVEL or path.name == "__pycache__":
                continue
            if path.name == "checkpoint_mapper.py":
                destination = family_dir / "weights/__init__.py"
                kind = "weights"
            elif path.name == "tests" and path.is_dir():
                destination = tests_root / family
                kind = "tests"
            elif path.is_file() and path.name in TOOL_MODULES:
                destination = tools_root / family / path.name
                kind = "tools"
            elif path.is_file() and path.name in WEIGHT_MODULES:
                destination = family_dir / "weights" / path.name
                kind = "weights"
            else:
                destination = family_dir / "model" / path.name
                kind = "model"
            moves.append(Move(path, destination, kind))
    return moves


def _module_mapping(repo_root: Path, moves: list[Move]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for move in moves:
        source_files = _iter_files(move.source)
        for source_file in source_files:
            relative = source_file.relative_to(move.source) if move.source.is_dir() else Path()
            destination_file = move.destination / relative if move.source.is_dir() else move.destination
            old_module = _module_name(repo_root, source_file)
            new_module = _module_name(repo_root, destination_file)
            if old_module and new_module and old_module != new_module:
                mapping[old_module] = new_module
    return mapping


def _map_module(module: str, mapping: dict[str, str]) -> str:
    for old in sorted(mapping, key=len, reverse=True):
        if module == old:
            return mapping[old]
        if module.startswith(f"{old}."):
            return f"{mapping[old]}{module[len(old):]}"
    return module


def _collapse_mapping(mapping: dict[str, str]) -> dict[str, str]:
    collapsed: dict[str, str] = {}
    for old, initial in mapping.items():
        current = initial
        seen = {old}
        while current not in seen:
            seen.add(current)
            updated = _map_module(current, mapping)
            if updated == current:
                break
            current = updated
        collapsed[old] = current
    return collapsed


def _package_for(module: str, is_init: bool) -> str:
    return module if is_init else module.rpartition(".")[0]


def _resolve_import_from(
    source_module: str,
    source_is_init: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    package = _package_for(source_module, source_is_init)
    parts = package.split(".") if package else []
    parents = node.level - 1
    base = parts[: max(0, len(parts) - parents)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _relative_spec(source_module: str, source_is_init: bool, target: str) -> str:
    package = _package_for(source_module, source_is_init)
    source_parts = package.split(".") if package else []
    target_parts = target.split(".") if target else []
    common = 0
    for source_part, target_part in zip(source_parts, target_parts):
        if source_part != target_part:
            break
        common += 1
    if common == 0:
        return target
    up = len(source_parts) - common
    suffix = ".".join(target_parts[common:])
    return "." * (up + 1) + suffix


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", text):
        offsets.append(match.end())
    return offsets


def _rewrite_imports(
    path: Path,
    *,
    old_module: str,
    new_module: str,
    old_is_init: bool,
    new_is_init: bool,
    mapping: dict[str, str],
) -> None:
    original = path.read_text(encoding="utf-8")
    text = original
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return
    offsets = _line_offsets(text)
    replacements: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[(node.end_lineno or node.lineno) - 1] + (node.end_col_offset or 0)
        statement = text[start:end]
        match = re.match(r"from\s+([.A-Za-z0-9_]+)\s+import\b", statement, re.DOTALL)
        if not match:
            continue

        old_target = _resolve_import_from(old_module, old_is_init, node)
        if node.module is None:
            candidates = [f"{old_target}.{alias.name}" for alias in node.names]
            mapped = [_map_module(candidate, mapping) for candidate in candidates]
            if mapped and all(candidate != result for candidate, result in zip(candidates, mapped)):
                parents = {candidate.rpartition(".")[0] for candidate in mapped}
                if len(parents) == 1:
                    new_target = parents.pop()
                else:
                    new_target = _map_module(old_target, mapping)
            else:
                new_target = _map_module(old_target, mapping)
        else:
            new_target = _map_module(old_target, mapping)

        if node.level > 0:
            new_spec = _relative_spec(new_module, new_is_init, new_target)
        else:
            new_spec = new_target
        spec_start = start + match.start(1)
        spec_end = start + match.end(1)
        if text[spec_start:spec_end] != new_spec:
            replacements.append((spec_start, spec_end, new_spec))

        for alias in node.names:
            candidate = f"{old_target}.{alias.name}" if old_target else alias.name
            mapped_candidate = _map_module(candidate, mapping)
            new_name = mapped_candidate.rsplit(".", 1)[-1]
            if mapped_candidate == candidate or new_name == alias.name:
                continue
            alias_start = offsets[alias.lineno - 1] + alias.col_offset
            alias_end = alias_start + len(alias.name)
            replacement = new_name
            if alias.asname is None:
                replacement = f"{new_name} as {alias.name}"
            replacements.append((alias_start, alias_end, replacement))

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]

    for old, new in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    if text != original:
        path.write_text(text, encoding="utf-8")


def _move_paths(moves: list[Move]) -> None:
    for move in moves:
        if not move.source.exists():
            continue
        if move.destination.exists():
            raise SystemExit(f"Migration destination already exists: {move.destination}")
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))


def _without_module_preamble(text: str, *, remove_graph_ops_import: bool) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    for index, node in enumerate(tree.body):
        remove = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        remove = remove or (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        )
        if remove_graph_ops_import and isinstance(node, ast.ImportFrom):
            remove = remove or (
                node.level == 1
                and node.module is None
                and {alias.name for alias in node.names} == {"graph_ops"}
            )
        if remove:
            removals.append((node.lineno - 1, node.end_lineno or node.lineno))
    for start, end in reversed(removals):
        del lines[start:end]
    body = "".join(lines).strip()
    if remove_graph_ops_import:
        body = re.sub(r"\bgraph_ops\.", "", body)
    return body


def _hoist_top_level_imports(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    imports: list[str] = []
    removals: list[tuple[int, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        imports.append("".join(lines[node.lineno - 1:node.end_lineno]).strip())
        removals.append((node.lineno - 1, node.end_lineno or node.lineno))
    for start, end in reversed(removals):
        del lines[start:end]

    unique_imports = list(dict.fromkeys(imports))
    body = "".join(lines)
    marker = "from __future__ import annotations\n"
    insertion = body.index(marker) + len(marker)
    import_block = "\n\n" + "\n".join(unique_imports) + "\n"
    return body[:insertion] + import_block + body[insertion:].lstrip("\n")


def _normalize_consolidated_models(
    repo_root: Path,
    families: tuple[str, ...] = (),
) -> None:
    for family_dir in family_dirs(repo_root, families):
        path = family_dir / "model/model.py"
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _hoist_top_level_imports(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")


def _consolidate_model_cores(
    repo_root: Path,
    reverse_paths: dict[Path, Path],
    families: tuple[str, ...] = (),
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family_dir in family_dirs(repo_root, families):
        model_dir = family_dir / "model"
        sources = [model_dir / name for name in CORE_MODEL_MODULES]
        existing = [path for path in sources if path.is_file()]
        if not existing:
            continue
        destination = model_dir / "model.py"
        if destination.exists():
            raise SystemExit(f"Model consolidation destination exists: {destination}")

        sections = []
        for source in existing:
            body = _without_module_preamble(
                source.read_text(encoding="utf-8"),
                remove_graph_ops_import=source.name != "graph_ops.py",
            )
            if body:
                title = source.stem.replace("_", " ").title()
                sections.append(f"# {title}\n\n{body}")
            old_module = _module_name(repo_root, source)
            new_module = _module_name(repo_root, destination)
            if old_module and new_module:
                mapping[old_module] = new_module

        destination.write_text(
            '"""Family-owned TensorRT model graph and utility implementation."""\n\n'
            "from __future__ import annotations\n\n"
            + "\n\n\n".join(sections)
            + "\n",
            encoding="utf-8",
        )
        source_context = reverse_paths.get(existing[0].resolve(), existing[0])
        for source in existing:
            reverse_paths.pop(source.resolve(), None)
            source.unlink()
        reverse_paths[destination.resolve()] = source_context
    return mapping


def _minimal_manifest(family: str) -> str:
    return (
        f'id = "{family}"\n'
        f'plugin = "{family}"\n'
        'module = "plugin"\n'
        f'aliases = ["{family}"]\n'
        f'prefixes = ["{family}"]\n'
    )


def _rewrite_manifest(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    text = original
    text = text.replace('debug_runner = "debug_runner.py|', 'debug_runner = "model/runtime.py|')
    text = text.replace('debug_runner = "model/debug_runner.py|', 'debug_runner = "model/runtime.py|')
    text = text.replace(
        'nemo_archive_adapter = "nemo_archive.py|',
        'nemo_archive_adapter = "weights/nemo_archive.py|',
    )
    text = text.replace(
        'config_adapter = "model_config.py|',
        'config_adapter = "model/model_config.py|',
    )
    family = path.parent.name
    text = text.replace(
        f"families/{family}/python_profile_requirements/",
        f"families/{family}/model/python_profile_requirements/",
    )
    text = text.replace(
        f"families/{family}/python_profile_verify.py",
        f"families/{family}/model/python_profile_verify.py",
    )
    if text != original:
        path.write_text(text, encoding="utf-8")


def _rewrite_python_paths(repo_root: Path) -> list[Path]:
    """Return project Python sources without traversing vendored repositories."""
    paths = set(repo_root.glob("*.py"))
    for relative_root in (".github", "python", "scripts", "tests", "tools"):
        root = repo_root / relative_root
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    return sorted(
        path
        for path in paths
        if "__pycache__" not in path.parts and ".venv" not in path.parts
    )


def migrate(repo_root: Path, families: tuple[str, ...] = ()) -> None:
    moves = planned_moves(repo_root, families)
    mapping = _module_mapping(repo_root, moves)
    reverse_paths: dict[Path, Path] = {}
    for move in moves:
        for source_file in _iter_files(move.source):
            relative = source_file.relative_to(move.source) if move.source.is_dir() else Path()
            destination = move.destination / relative if move.source.is_dir() else move.destination
            reverse_paths[destination.resolve()] = source_file

    _move_paths(moves)
    mapping.update(_consolidate_model_cores(repo_root, reverse_paths, families))
    _normalize_consolidated_models(repo_root, families)
    mapping = _collapse_mapping(mapping)

    tools_family_root = repo_root / "tools/families"
    tests_family_root = repo_root / "tests/builder/families"
    if tools_family_root.exists():
        (tools_family_root / "__init__.py").write_text(TOOLS_INIT, encoding="utf-8")
        for directory in sorted(path for path in tools_family_root.iterdir() if path.is_dir()):
            (directory / "__init__.py").write_text(TOOLS_INIT, encoding="utf-8")
    if tests_family_root.exists():
        (tests_family_root / "__init__.py").write_text(TESTS_INIT, encoding="utf-8")
        for directory in sorted(path for path in tests_family_root.iterdir() if path.is_dir()):
            (directory / "__init__.py").write_text(TESTS_INIT, encoding="utf-8")

    selected_families = family_dirs(repo_root, families)
    for family_dir in selected_families:
        family = family_dir.name
        model_dir = family_dir / "model"
        weights_dir = family_dir / "weights"
        model_dir.mkdir(exist_ok=True)
        weights_dir.mkdir(exist_ok=True)
        model_init = model_dir / "__init__.py"
        if not model_init.exists():
            model_init.write_text(MODEL_INIT, encoding="utf-8")
        weights_init = weights_dir / "__init__.py"
        if not weights_init.exists():
            weights_init.write_text('"""Family-owned weight mapping."""\n', encoding="utf-8")
        (family_dir / "__init__.py").write_text(MINIMAL_INIT, encoding="utf-8")
        manifest = family_dir / "MODEL.toml"
        if not manifest.exists():
            manifest.write_text(_minimal_manifest(family), encoding="utf-8")
        _rewrite_manifest(manifest)

    for path in _rewrite_python_paths(repo_root):
        resolved = path.resolve()
        old_path = reverse_paths.get(resolved, path)
        old_module = _module_name(repo_root, old_path)
        new_module = _module_name(repo_root, path)
        if not old_module or not new_module:
            continue
        _rewrite_imports(
            path,
            old_module=old_module,
            new_module=new_module,
            old_is_init=old_path.name == "__init__.py",
            new_is_init=path.name == "__init__.py",
            mapping=mapping,
        )

    print(
        f"migrated_families={len(selected_families)} "
        f"moved_entries={len(moves)} module_rewrites={len(mapping)}"
    )


def layout_violations(
    repo_root: Path,
    families: tuple[str, ...] = (),
) -> list[str]:
    violations: list[str] = []
    for family_dir in family_dirs(repo_root, families):
        names = {
            path.name for path in family_dir.iterdir() if path.name != "__pycache__"
        }
        missing = CANONICAL_TOP_LEVEL - names
        extra = names - ALLOWED_TOP_LEVEL
        if missing:
            violations.append(f"{family_dir.name}: missing {sorted(missing)}")
        if extra:
            violations.append(f"{family_dir.name}: extra {sorted(extra)}")
        if (family_dir / "__init__.py").read_text(encoding="utf-8") != MINIMAL_INIT:
            violations.append(f"{family_dir.name}: non-canonical __init__.py")
        for component in ("model", "weights"):
            path = family_dir / component
            if not path.is_dir() or not (path / "__init__.py").is_file():
                violations.append(f"{family_dir.name}: invalid {component}/ component")
        model_dir = family_dir / "model"
        if not (model_dir / "model.py").is_file():
            violations.append(f"{family_dir.name}: missing model/model.py")
        legacy_core = [name for name in CORE_MODEL_MODULES if (model_dir / name).exists()]
        if legacy_core:
            violations.append(
                f"{family_dir.name}: unconsolidated model modules {legacy_core}"
            )
    return violations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Limit migration or validation to one family; repeat for multiple families",
    )
    parser.add_argument("command", choices=("check", "migrate"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    families = tuple(args.family)
    if args.command == "migrate":
        migrate(repo_root, families)
    violations = layout_violations(repo_root, families)
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print(f"canonical_family_layouts={len(family_dirs(repo_root, families))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

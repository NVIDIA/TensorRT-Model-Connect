#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply reviewed, family-local source specialization layouts."""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tools import migrate_family_layout as layout_tools
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    import migrate_family_layout as layout_tools


@dataclass(frozen=True)
class SpecializationLayout:
    model_modules: tuple[str, ...] = ()
    parallel_modules: tuple[str, ...] = ()
    runtime_modules: tuple[str, ...] = ()
    component_modules: tuple[tuple[str, str], ...] = ()
    profile_paths: tuple[tuple[str, str], ...] = ()


MODEL_MODULES = {
    "albert": ("encoder_builder.py",),
    "bert": ("encoder_builder.py",),
    "convbert": ("builder.py",),
    "distilbert": ("encoder_builder.py",),
    "dpr": ("encoder_builder.py",),
    "electra": ("encoder_builder.py",),
    "fnet": ("fnet_encoder_builder.py",),
    "mpnet": ("encoder_builder.py",),
    "roberta": ("encoder_builder.py",),
    "xlnet": ("xlnet_builder.py",),
}

MODEL_MODULES.update(
    {
        "bark": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "bloom": ("dual_profile_decoder_builder.py", "standard_decoder_builder.py"),
        "chronos_bolt": ("time_series_trt.py",),
        "codegen": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "deepseek_ocr": ("default_decoder.py",),
        "deepseek_v2": ("default_decoder.py",),
        "elf_flow": ("builder.py",),
        "falcon": ("dual_profile_decoder_builder.py", "standard_decoder_builder.py"),
        "gemma": ("dual_profile_decoder_builder.py", "standard_decoder_builder.py"),
        "glm": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "gpt2": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "gpt_neo": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "gpt_neox": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "gpt_oss": ("default_decoder.py",),
        "granite": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "internlm": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "internvl": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "lance": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "llama": ("dual_profile_decoder_builder.py", "standard_decoder_builder.py"),
        "locateanything": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "ltx_video": ("ltx_dit_builder.py",),
        "mistral": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "mixtral": ("default_decoder.py",),
        "nemotron": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "nemotron_labs_diffusion": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "nemotron_speech_streaming": ("canary_encoder_helpers.py",),
        "olmo": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "opt": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "patchtsmixer": ("time_series_trt.py",),
        "patchtst": ("time_series_trt.py",),
        "personaplex": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "phi": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "phi4_multimodal": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "phi_moe": ("default_decoder.py",),
        "pixart": ("standard_dit_builder.py",),
        "qwen": ("dual_profile_decoder_builder.py", "standard_decoder_builder.py"),
        "qwen3_omni": ("default_decoder.py",),
        "qwen_image": ("qwen_image_dit_builder.py",),
        "qwen_moe": ("default_decoder.py",),
        "qwen_vl": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "sam3": ("core_builder.py",),
        "stablelm": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "starcoder2": ("default_decoder.py", "default_dual_profile_decoder.py"),
        "timesfm": ("time_series_trt.py",),
        "wan_t2v": ("standard_dit_builder.py",),
        "xglm": ("default_decoder.py", "default_dual_profile_decoder.py"),
    }
)

COMPONENT_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    "bark": (("encodec_builder.py", "codec.py"),),
    "elf_flow": (
        ("model_config.py", "config.py"),
        ("t5_encoder_builder.py", "text_encoder.py"),
    ),
    "flux": (
        ("clip_encoder_builder.py", "clip_encoder.py"),
        ("flux2_dit_builder.py", "flux2.py"),
        ("flux2_dit_tp_builder.py", "flux2_parallel.py"),
        ("flux_dit_builder.py", "flux.py"),
        ("flux_dit_tp_builder.py", "flux_parallel.py"),
        ("flux_vae_builder.py", "vae.py"),
        ("mistral_encoder_builder.py", "mistral_encoder.py"),
        ("t5_encoder_builder.py", "t5_encoder.py"),
    ),
    "internvl": (("internvit_vision_builder.py", "vision.py"),),
    "lance": (("qwen_vl_vision_builder.py", "vision.py"),),
    "locateanything": (
        ("onnx_vision_builder.py", "vision_onnx.py"),
        ("vision_builder.py", "vision.py"),
    ),
    "ltx_video": (
        ("ltx_vae_builder.py", "vae.py"),
        ("t5_encoder_builder.py", "text_encoder.py"),
    ),
    "magpie_tts": (("nanocodec_builder.py", "codec.py"),),
    "pixart": (
        ("t5_encoder_builder.py", "text_encoder.py"),
        ("vae_2d_builder.py", "vae.py"),
    ),
    "qwen3_omni": (("qwen_vl_vision_builder.py", "vision.py"),),
    "qwen_image": (
        ("qwen25_vl_text_encoder_builder.py", "text_encoder.py"),
        ("qwen_image_bundle_config.py", "bundle_config.py"),
        ("qwen_image_preprocessor.py", "preprocessor.py"),
        ("qwen_image_vae_builder.py", "vae.py"),
        ("qwen_vl_vision_builder.py", "vision.py"),
    ),
    "qwen_vl": (("qwen_vl_vision_builder.py", "vision.py"),),
    "sam3": (
        ("text_encoder_builder.py", "text_encoder.py"),
        ("vision_encoder_builder.py", "vision_encoder.py"),
    ),
    "wan_t2v": (
        ("causal_vae_3d_builder.py", "vae.py"),
        ("t5_encoder_builder.py", "text_encoder.py"),
    ),
    "z_image": (
        ("qwen3_encoder_builder.py", "text_encoder.py"),
        ("vae_2d_builder.py", "vae.py"),
        ("z_image_dit_builder.py", "dit.py"),
    ),
}

PROFILE_PATHS = {
    family: (
        ("python_profile_requirements", "requirements"),
        ("python_profile_verify.py", "verify.py"),
    )
    for family in ("chronos_bolt", "internlm")
}

PARALLEL_MODULES = {
    "albert": "tp_builder.py",
    "bark": "decoder_tp_builder.py",
    "bart": "decoder_tp_builder.py",
    "bert": "tp_builder.py",
    "bloom": "dual_profile_decoder_tp_builder.py",
    "canary": "decoder_tp_builder.py",
    "codegen": "dual_profile_decoder_tp_builder.py",
    "convbert": "tp_builder.py",
    "deberta": "tp_builder.py",
    "deepseek_ocr": "tp_builder.py",
    "deepseek_v2": "tp_builder.py",
    "distilbert": "tp_builder.py",
    "dpr": "tp_builder.py",
    "eagle_vlm": "tp_builder.py",
    "electra": "tp_builder.py",
    "falcon": "dual_profile_decoder_tp_builder.py",
    "fnet": "tp_builder.py",
    "gemma": "dual_profile_decoder_tp_builder.py",
    "glm": "dual_profile_decoder_tp_builder.py",
    "gpt2": "default_dual_profile_decoder_tp.py",
    "gpt_neo": "default_dual_profile_decoder_tp.py",
    "gpt_neox": "default_dual_profile_decoder_tp.py",
    "gpt_oss": "tp_builder.py",
    "granite": "dual_profile_decoder_tp_builder.py",
    "internlm": "dual_profile_decoder_tp_builder.py",
    "internvl": "tp_builder.py",
    "locateanything": "decoder_tp_builder.py",
    "magpie_tts": "decoder_tp_builder.py",
    "mamba": "tp_builder.py",
    "marian": "decoder_tp_builder.py",
    "mixtral": "tp_builder.py",
    "modernbert": "tp_builder.py",
    "mpnet": "tp_builder.py",
    "nemotron_h": "tp_builder.py",
    "nemotron_speech_streaming": "predictor_tp_builder.py",
    "olmo": "dual_profile_decoder_tp_builder.py",
    "olmo2": "tp_builder.py",
    "opt": "default_dual_profile_decoder_tp.py",
    "personaplex": "decoder_tp_builder.py",
    "phi": "dual_profile_decoder_tp_builder.py",
    "phi_moe": "tp_builder.py",
    "pixart": "standard_dit_tp_builder.py",
    "qwen": "dual_profile_decoder_tp_builder.py",
    "qwen_moe": "tp_builder.py",
    "qwen_vl": "decoder_tp_builder.py",
    "roberta": "tp_builder.py",
    "rwkv": "tp_builder.py",
    "sam": "sam_tp_builder.py",
    "segformer": "segformer_tp_builder.py",
    "stablelm": "dual_profile_decoder_tp_builder.py",
    "starcoder2": "dual_profile_decoder_tp_builder.py",
    "t5": "decoder_tp_builder.py",
    "timm_vit": "timm_vit_tp_builder.py",
    "wan_t2v": "standard_dit_tp_builder.py",
    "whisper": "decoder_tp_builder.py",
    "xglm": "dual_profile_decoder_tp_builder.py",
    "xlnet": "tp_builder.py",
    "z_image": "z_image_dit_tp_builder.py",
}

RUNTIME_FAMILIES = {
    "bart",
    "bloom",
    "codegen",
    "deepseek_v2",
    "falcon",
    "gemma",
    "glm",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gpt_oss",
    "granite",
    "internlm",
    "llama",
    "m2m_100",
    "mamba",
    "marian",
    "mistral",
    "mixtral",
    "nemotron",
    "nemotron_h",
    "nemotron_labs_diffusion",
    "olmo",
    "olmo2",
    "opt",
    "phi",
    "phi_moe",
    "qwen",
    "qwen3_5",
    "qwen_moe",
    "rwkv",
    "stablelm",
    "starcoder2",
    "t5",
    "xglm",
}

FAMILY_LAYOUTS: dict[str, SpecializationLayout] = {
    family: SpecializationLayout(
        model_modules=MODEL_MODULES.get(family, ()),
        parallel_modules=(PARALLEL_MODULES[family],)
        if family in PARALLEL_MODULES
        else (),
        runtime_modules=("debug_runner.py",) if family in RUNTIME_FAMILIES else (),
        component_modules=COMPONENT_MODULES.get(family, ()),
        profile_paths=PROFILE_PATHS.get(family, ()),
    )
    for family in sorted(
        MODEL_MODULES.keys()
        | PARALLEL_MODULES.keys()
        | RUNTIME_FAMILIES
        | COMPONENT_MODULES.keys()
        | PROFILE_PATHS.keys()
    )
}


def _strip_merged_preamble(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    module_aliases: set[str] = set()
    name_aliases: dict[str, str] = {}

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
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module is None:
                merged_aliases = [alias for alias in node.names if alias.name == "model"]
                if merged_aliases and len(merged_aliases) == len(node.names):
                    module_aliases.update(alias.asname or alias.name for alias in merged_aliases)
                    remove = True
            elif node.module == "model":
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if local_name != alias.name:
                        name_aliases[local_name] = alias.name
                remove = True
        if remove:
            removals.append((node.lineno - 1, node.end_lineno or node.lineno))

    for start, end in reversed(removals):
        del lines[start:end]
    body = "".join(lines).strip()
    for alias in sorted(module_aliases, key=len, reverse=True):
        body = re.sub(rf"\b{re.escape(alias)}\.", "", body)
    for alias, target in sorted(name_aliases.items(), key=lambda item: -len(item[0])):
        body = re.sub(rf"\b{re.escape(alias)}\b", target, body)
    return body


def _normalize_merged_model(path: Path) -> None:
    """Remove merge-created self imports and repair known local aliases."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    removals: list[tuple[int, int]] = []
    aliases: dict[str, str] = {}

    for node in tree.body:
        if not (
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == "model"
        ):
            continue
        removals.append((node.lineno - 1, node.end_lineno or node.lineno))
        for alias in node.names:
            local_name = alias.asname or alias.name
            if local_name != alias.name:
                aliases[local_name] = alias.name

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for alias, target in {
        "_const_in_work_dtype": "const_in_work_dtype",
        "_norm_multi": "norm_multi",
    }.items():
        if target in defined and alias not in defined:
            aliases[alias] = target

    for start, end in reversed(removals):
        del lines[start:end]
    updated = "".join(lines)
    for alias, target in sorted(aliases.items(), key=lambda item: -len(item[0])):
        updated = re.sub(rf"\b{re.escape(alias)}\b", target, updated)
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _merge_into(
    repo_root: Path,
    family_dir: Path,
    sources: tuple[str, ...],
    destination_name: str,
) -> dict[str, str]:
    model_dir = family_dir / "model"
    destination = model_dir / destination_name
    existing = [model_dir / name for name in sources if (model_dir / name).is_file()]
    if not existing:
        return {}
    if not destination.is_file():
        raise SystemExit(f"Missing specialization destination: {destination}")

    text = destination.read_text(encoding="utf-8").rstrip()
    mapping: dict[str, str] = {}
    destination_module = layout_tools._module_name(repo_root, destination)
    for source in existing:
        body = _strip_merged_preamble(source.read_text(encoding="utf-8"))
        if body:
            title = source.stem.replace("_", " ").title()
            text += f"\n\n\n# {title}\n\n{body}"
        source_module = layout_tools._module_name(repo_root, source)
        if source_module and destination_module:
            mapping[source_module] = destination_module
        source.unlink()
    destination.write_text(text + "\n", encoding="utf-8")
    return mapping


def _move_single_module(
    repo_root: Path,
    family_dir: Path,
    sources: tuple[str, ...],
    destination_name: str,
) -> tuple[dict[str, str], dict[Path, Path]]:
    model_dir = family_dir / "model"
    existing = [model_dir / name for name in sources if (model_dir / name).is_file()]
    if not existing:
        return {}, {}
    destination = model_dir / destination_name
    if destination.exists():
        raise SystemExit(f"Specialization destination exists: {destination}")
    if len(existing) != 1:
        raise SystemExit(
            f"{destination_name} consolidation for {family_dir.name} requires one source; "
            f"found {[path.name for path in existing]}"
        )
    source = existing[0]
    source_module = layout_tools._module_name(repo_root, source)
    destination_module = layout_tools._module_name(repo_root, destination)
    shutil.move(str(source), str(destination))
    mapping = (
        {source_module: destination_module}
        if source_module and destination_module
        else {}
    )
    return mapping, {destination.resolve(): source}


def _move_mapped_paths(
    repo_root: Path,
    family_dir: Path,
    moves: tuple[tuple[str, str], ...],
    *,
    source_root: Path,
    destination_root: Path,
) -> tuple[dict[str, str], dict[Path, Path]]:
    mapping: dict[str, str] = {}
    reverse_paths: dict[Path, Path] = {}
    moved = False
    for source_name, destination_name in moves:
        source = source_root / source_name
        if not source.exists():
            continue
        destination = destination_root / destination_name
        if destination.exists():
            raise SystemExit(f"Specialization destination exists: {destination}")
        source_files = [source] if source.is_file() else sorted(source.rglob("*.py"))
        for source_file in source_files:
            relative = source_file.relative_to(source) if source.is_dir() else Path()
            destination_file = destination / relative if source.is_dir() else destination
            old_module = layout_tools._module_name(repo_root, source_file)
            new_module = layout_tools._module_name(repo_root, destination_file)
            if old_module and new_module:
                mapping[old_module] = new_module
            reverse_paths[destination_file.resolve()] = source_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved = True

    if moved:
        destination_root.mkdir(parents=True, exist_ok=True)
        init = destination_root / "__init__.py"
        if not init.exists():
            init.write_text('"""Family-owned specialized components."""\n', encoding="utf-8")
    return mapping, reverse_paths


def _rewrite_sources(
    repo_root: Path,
    mapping: dict[str, str],
    reverse_paths: dict[Path, Path],
) -> None:
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


def _normalize_manifests(
    repo_root: Path,
    specializations: dict[str, SpecializationLayout],
) -> None:
    for family, specialization in specializations.items():
        manifest = (
            repo_root
            / "python/tensorrt_model_connect/families"
            / family
            / "MODEL.toml"
        )
        text = manifest.read_text(encoding="utf-8")
        updated = text
        if specialization.runtime_modules:
            updated = updated.replace("model/debug_runner.py|", "model/runtime.py|")
        for source_name, destination_name in specialization.component_modules:
            updated = updated.replace(
                f"model/{source_name}|",
                f"model/components/{destination_name}|",
            )
        if specialization.profile_paths:
            updated = updated.replace(
                "/model/python_profile_requirements/",
                "/profiles/requirements/",
            )
            updated = updated.replace(
                "/model/python_profile_verify.py",
                "/profiles/verify.py",
            )
        if updated != text:
            manifest.write_text(updated, encoding="utf-8")


def specialize_family(
    repo_root: Path,
    family: str,
    specialization: SpecializationLayout,
) -> None:
    specialize_families(repo_root, {family: specialization})


def specialize_families(
    repo_root: Path,
    specializations: dict[str, SpecializationLayout],
) -> None:
    repo_root = repo_root.resolve()
    mapping: dict[str, str] = {}
    reverse_paths: dict[Path, Path] = {}
    for family, specialization in specializations.items():
        family_dir = repo_root / "python/tensorrt_model_connect/families" / family
        if not (family_dir / "plugin.py").is_file():
            raise SystemExit(f"Unknown model family: {family}")

        mapping.update(
            _merge_into(
                repo_root,
                family_dir,
                specialization.model_modules,
                "model.py",
            )
        )
        parallel_mapping, parallel_reverse = _move_single_module(
            repo_root,
            family_dir,
            specialization.parallel_modules,
            "parallel.py",
        )
        mapping.update(parallel_mapping)
        reverse_paths.update(parallel_reverse)
        runtime_mapping, runtime_reverse = _move_single_module(
            repo_root,
            family_dir,
            specialization.runtime_modules,
            "runtime.py",
        )
        mapping.update(runtime_mapping)
        reverse_paths.update(runtime_reverse)
        component_mapping, component_reverse = _move_mapped_paths(
            repo_root,
            family_dir,
            specialization.component_modules,
            source_root=family_dir / "model",
            destination_root=family_dir / "model/components",
        )
        mapping.update(component_mapping)
        reverse_paths.update(component_reverse)
        profile_mapping, profile_reverse = _move_mapped_paths(
            repo_root,
            family_dir,
            specialization.profile_paths,
            source_root=family_dir / "model",
            destination_root=family_dir / "profiles",
        )
        mapping.update(profile_mapping)
        reverse_paths.update(profile_reverse)
    _normalize_manifests(repo_root, specializations)
    if mapping:
        _rewrite_sources(repo_root, mapping, reverse_paths)
        layout_tools._normalize_consolidated_models(repo_root)
    for family in specializations:
        model_path = (
            repo_root
            / "python/tensorrt_model_connect/families"
            / family
            / "model/model.py"
        )
        if model_path.is_file():
            _normalize_merged_model(model_path)


def pending_paths(repo_root: Path, family: str) -> list[str]:
    specialization = FAMILY_LAYOUTS[family]
    model_dir = repo_root / "python/tensorrt_model_connect/families" / family / "model"
    pending = [
        name
        for name in (
            *specialization.model_modules,
            *specialization.parallel_modules,
            *specialization.runtime_modules,
        )
        if (model_dir / name).is_file()
    ]
    pending.extend(
        source for source, _ in specialization.component_modules
        if (model_dir / source).exists()
    )
    pending.extend(
        source for source, _ in specialization.profile_paths
        if (model_dir / source).exists()
    )
    return sorted(pending)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.family:
        raise SystemExit("Specify --family or --all")
    selected = sorted(FAMILY_LAYOUTS) if args.all else args.family
    unknown = sorted(set(selected) - FAMILY_LAYOUTS.keys())
    if unknown:
        raise SystemExit("No reviewed specialization layout for: " + ", ".join(unknown))
    repo_root = args.repo_root.resolve()
    pending = {
        family: pending_paths(repo_root, family)
        for family in selected
    }
    if args.command == "check":
        for family, paths in pending.items():
            if paths:
                print(f"{family}: pending {', '.join(paths)}")
        return 1 if any(pending.values()) else 0

    specialize_families(
        repo_root,
        {family: FAMILY_LAYOUTS[family] for family in selected},
    )
    for family in selected:
        print(f"specialized_family={family}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

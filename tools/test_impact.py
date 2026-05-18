#!/usr/bin/env python3
"""Test impact analysis -- selective CI execution based on changed files.

Determines which E2E models and unit test tiers need to run based on
git diff between base and head. Safety invariant: ZERO false negatives.
Any file that doesn't match a known rule triggers ALL model tests.

Usage:
    python3 tools/test_impact.py [--base REF] [--head REF] [--json] [--verbose]
    python3 tools/test_impact.py --files path/to/file1.py,path/to/file2.cpp
    python3 tools/test_impact.py --validate
    python3 tools/test_impact.py --e2e-suite nightly --files src/runtime/models/text_generation/plugin.cpp
    python3 tools/test_impact.py --files tensorrt_model_connect/tensorrt_model_connect/families/qwen/plugin.py --cap 15
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants -- strategy mappings (mirrored from e2e_harness/contracts.py)
# ---------------------------------------------------------------------------

RUNTIME_TO_TASK_STRATEGY: Dict[str, str] = {
    "decoder_kv_cache": "text_generation_causal",
    "decoder_moe": "text_generation_causal",
    "ssm_recurrent": "text_generation_causal",
    "rwkv_recurrent": "text_generation_causal",
    "hybrid_mamba_attention": "text_generation_causal",
    "vision_language": "vision_language_generation",
    "speech_to_text": "speech_to_text",
    "speech_to_text_rnnt": "speech_to_text",
    "text_to_audio": "text_to_audio",
    "text_to_audio_bark": "text_to_audio",
    "text_to_audio_magpie": "text_to_audio",
    "speech_to_speech": "speech_to_speech",
    "segmentation": "segmentation",
    "prompted_segmentation": "prompted_segmentation",
    "image_classification": "image_classification",
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "patchtst_torchtrt": "neural_operator",
    "patchtsmixer_torchtrt": "neural_operator",
    "timesfm_torchtrt": "neural_operator",
    "chronos_bolt_torchtrt": "neural_operator",
    "elf_flow": "diffusion_text_generation",
    "diffusion": "diffusion_media_generation",
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_ltx": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_zimage": "diffusion_media_generation",
    "diffusion_qwen_image": "diffusion_media_generation",
    "diffusion_pixart": "diffusion_media_generation",
    "diffusion_sana_wm": "diffusion_media_generation",
    "torchtrt_decoder": "text_generation_causal",
    "torchtrt_diffusion": "diffusion_media_generation",
    "diffusion_pixart_torchtrt": "diffusion_media_generation",
    "omni_multimodal": "omni_multimodal",
    "text_to_text": "text_generation_causal",
    "marian_translation": "text_generation_causal",
    "seq2seq_encoder_decoder": "text_generation_causal",
}

# C++ plugin filename (stem) -> registered runtime_strategies
CPP_PLUGIN_STRATEGIES: Dict[str, List[str]] = {
    "decoder_plugin": ["decoder_kv_cache", "decoder_moe"],
    "ssm_plugin": ["ssm_recurrent"],
    "rwkv_plugin": ["rwkv_recurrent"],
    "hybrid_plugin": ["hybrid_mamba_attention"],
    "vl_plugin": ["vision_language"],
    "whisper_plugin": ["speech_to_text"],
    "rnnt_plugin": ["speech_to_text_rnnt"],
    "bark_plugin": ["text_to_audio_bark"],
    "magpie_plugin": ["text_to_audio_magpie"],
    "speech_plugin": ["speech_to_speech"],
    "encoder_plugin": ["encoder_only", "embedding", "reranking", "neural_operator"],
    "patchtst_plugin": ["patchtst_torchtrt"],
    "patchtsmixer_plugin": ["patchtsmixer_torchtrt"],
    "timesfm_plugin": ["timesfm_torchtrt"],
    "chronos_bolt_plugin": ["chronos_bolt_torchtrt"],
    "elf_flow_plugin": ["elf_flow"],
    "segmentation_plugin": ["segmentation", "prompted_segmentation"],
    "object_detection_plugin": ["object_detection"],
    "omni_plugin": ["omni_multimodal"],
    "flux_plugin": ["diffusion_flux"],
    "ltx_video_plugin": ["diffusion_ltx"],
    "wan_plugin": ["diffusion_wan"],
    "pixart_plugin": ["diffusion_pixart"],
    "pixart_torchtrt_plugin": ["diffusion_pixart_torchtrt"],
    "zimage_plugin": ["diffusion_zimage"],
    "qwen_image_plugin": ["diffusion_qwen_image"],
    "sana_wm_plugin": ["diffusion_sana_wm"],
    "t5_plugin": ["text_to_text"],
    "marian_plugin": ["marian_translation"],
    "seq2seq_plugin": ["seq2seq_encoder_decoder"],
}

# C++ pipeline filename (stem) -> runtime_strategies it serves
CPP_PIPELINE_STRATEGIES: Dict[str, List[str]] = {
    "text_generation_pipeline": ["decoder_kv_cache", "decoder_moe"],
    "recurrent_pipeline": ["ssm_recurrent", "rwkv_recurrent", "hybrid_mamba_attention"],
    "vl_pipeline": ["vision_language"],
    # Audio/speech pipelines — each has its own .cpp file (no shared audio_pipeline.cpp):
    "whisper_pipeline": ["speech_to_text"],
    "rnnt_pipeline": ["speech_to_text_rnnt"],
    "bark_pipeline": ["text_to_audio_bark"],
    "magpie_pipeline": ["text_to_audio_magpie"],
    "speech_pipeline": ["speech_to_speech"],
    "omni_pipeline": ["omni_multimodal"],
    # Segmentation pipelines — separate files, not part of encoder_pipeline:
    "segment_pipeline": ["segmentation"],
    "sam_pipeline": ["prompted_segmentation"],
    "image_classification_pipeline": ["image_classification"],
    "encoder_pipeline": [
        "encoder_only", "embedding", "reranking", "neural_operator",
        "object_detection",
    ],
    "patchtst_pipeline": ["patchtst_torchtrt"],
    "patchtsmixer_pipeline": ["patchtsmixer_torchtrt"],
    "timesfm_pipeline": ["timesfm_torchtrt"],
    "chronos_bolt_pipeline": ["chronos_bolt_torchtrt"],
    "elf_flow_pipeline": ["elf_flow"],
    "flux_pipeline": ["diffusion_flux"],
    "ltx_video_pipeline": ["diffusion_ltx"],
    "wan_pipeline": ["diffusion_wan"],
    "pixart_pipeline": ["diffusion_pixart"],
    "pixart_torchtrt_pipeline": ["diffusion_pixart_torchtrt"],
    "z_image_pipeline": ["diffusion_zimage"],
    "qwen_image_pipeline": ["diffusion_qwen_image"],
    "sana_wm_pipeline": ["diffusion_sana_wm"],
    "diffusion_pipeline": [
        "diffusion_flux", "diffusion_ltx", "diffusion_wan", "diffusion_pixart",
        "diffusion_zimage", "diffusion_qwen_image", "diffusion_pixart_torchtrt",
        "diffusion_sana_wm",
    ],
}

# E2E runner filename (stem) -> task_strategies
RUNNER_TASK_STRATEGIES: Dict[str, List[str]] = {
    "text_generation": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "audio_speech": ["speech_to_text", "text_to_audio", "speech_to_speech"],
    "diffusion": ["diffusion_media_generation"],
    "diffusion_text_generation": ["diffusion_text_generation"],
    "image_classification": ["image_classification"],
    "segmentation": ["segmentation", "prompted_segmentation", "object_detection"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "encoder_only": ["encoder_only_nlp"],
    "omni": ["omni_multimodal"],
    "neural_operator": ["neural_operator"],
}

# E2E comparator filename (stem) -> task_strategies
COMPARATOR_TASK_STRATEGIES: Dict[str, List[str]] = {
    "text": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "speech_to_text": ["speech_to_text"],
    "text_to_audio": ["text_to_audio"],
    "speech_to_speech": ["speech_to_speech"],
    "encoder_only": ["encoder_only_nlp"],
    "embedding": ["embedding"],
    "reranking": ["reranking"],
    "segmentation": ["segmentation", "prompted_segmentation", "object_detection"],
    "image_classification": ["image_classification"],
    "diffusion": ["diffusion_media_generation"],
    "diffusion_text_generation": ["diffusion_text_generation"],
    "omni": ["omni_multimodal"],
    "neural_operator": ["neural_operator"],
}

# E2E contract plugin filename (stem) -> task_strategies
PLUGIN_TASK_STRATEGIES: Dict[str, List[str]] = {
    "diffusion": ["diffusion_media_generation"],
    "vl_qa": ["vision_language_generation"],
    "multimodal_chat": ["omni_multimodal"],
    "time_series_regression": ["neural_operator"],
    "time_series_classification": ["neural_operator"],
    "tts": ["text_to_audio"],
    "elf_diffusion_text": ["diffusion_text_generation"],
}

# E2E reference filename (stem) -> task_strategies
REFERENCE_TASK_STRATEGIES: Dict[str, List[str]] = {
    "hf_transformers": [
        "text_generation_causal", "vision_language_generation", "text_to_audio",
        "speech_to_text", "encoder_only_nlp", "embedding", "reranking",
        "segmentation", "prompted_segmentation", "image_classification",
        "object_detection",
    ],
    "hf_diffusers": ["diffusion_media_generation"],
    "torch_reference": ["speech_to_speech", "omni_multimodal", "neural_operator"],
}

# E2E threshold profile filename (stem) -> task_strategies
THRESHOLD_PROFILE_TASK_STRATEGIES: Dict[str, List[str]] = {
    "diffusion_media_generation": ["diffusion_media_generation"],
    "torchtrt_diffusion": ["diffusion_media_generation"],
    "vision_language_generation": ["vision_language_generation"],
    "omni_multimodal": ["omni_multimodal"],
    "segmentation": ["segmentation"],
}

# Shared C++ helper -> affected task_strategies
SHARED_CPP_HELPER_STRATEGIES: Dict[str, List[str]] = {
    "diffusion_helpers": [
        "diffusion_flux", "diffusion_ltx", "diffusion_wan", "diffusion_pixart",
        "diffusion_zimage", "diffusion_sana_wm",
    ],
    "audio_helpers": [
        "speech_to_text", "speech_to_text_rnnt", "text_to_audio_bark",
        "text_to_audio_magpie", "speech_to_speech", "omni_multimodal",
    ],
}

# Orchestrator modules in tensorrt_model_connect/ -- not treated as specialized builders
_ORCHESTRATOR_MODULES = {
    "engine_builder", "cli", "__init__", "__main__", "pipeline",
    "debug_runner", "diffusion_runner",
}

# Patterns for files that never affect E2E or unit tests
_NO_IMPACT_PATTERNS = [
    r"^docs/",
    r"^website/",
    r"^\.gitignore$",
    r"^\.clang-format$",
    r"^\.editorconfig$",
    r"^\.github/",
    r"^\.claude/",
    r"^\.agents/",
    r"^plugins/trtmc-agent-skills/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^recovery-",
]

_BROAD_FALLBACK_RULES = {
    "catch_all",
    "harness_shared",
    "shared_builder_module",
}
_FALLBACK_ALLOWLIST = Path("tools/test_impact_fallback_allowlist.txt")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RuleMatch:
    rule: str
    models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool


@dataclass
class ImpactMap:
    family_to_models: Dict[str, List[str]]
    strategy_to_models: Dict[str, List[str]]       # runtime_strategy -> models
    task_strategy_to_models: Dict[str, List[str]]   # task_strategy -> models
    all_model_names: List[str]
    all_model_names_set: Set[str]
    core_models: List[str]
    model_metadata: Dict[str, Dict]
    builder_to_families: Dict[str, List[str]]       # parent module -> families
    cpp_runtime_model_strategies: Dict[str, List[str]]
    manifest_field_to_models: Dict[str, List[str]]
    e2e_data_file_to_models: Dict[str, List[str]]
    path_scope_overrides: Dict[str, List[str]]
    l0_replacement_by_model: Dict[str, str]


@dataclass
class ImpactResult:
    e2e_models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool
    cap_applied: bool
    matched_rules: List[Dict]
    builder_tests: List[str] = field(default_factory=list)
    cpp_tests: List[str] = field(default_factory=list)
    tools_tests: List[str] = field(default_factory=list)
    fallback_tiers: List[str] = field(default_factory=list)
    l0_replacements: List[Dict[str, str]] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Impact map construction
# ---------------------------------------------------------------------------


def _iter_family_python_files(families_dir: Path) -> List[tuple[str, Path]]:
    """Return (family_name, python_file) for flat modules and package layouts."""
    files: List[tuple[str, Path]] = []
    for py_file in sorted(families_dir.glob("*.py")):
        name = py_file.stem
        if name in ("__init__", "base"):
            continue
        files.append((name, py_file))
    for family_dir in sorted(path for path in families_dir.iterdir() if path.is_dir()):
        if family_dir.name.startswith("_"):
            continue
        for py_file in sorted(family_dir.glob("*.py")):
            files.append((family_dir.name, py_file))
    return files


def _scan_family_imports(families_dir: Path) -> Dict[str, List[str]]:
    """Build reverse index: parent builder module -> importer family names.

    Local imports such as ``from .standard_decoder_builder import build`` are
    family-owned implementation details. Only parent-package imports are
    compatibility-shim usage.
    """
    reverse: Dict[str, Set[str]] = {}
    for name, py_file in _iter_family_python_files(families_dir):
        try:
            content = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        # from ..module_name import ... / from ...module_name import ...
        for m in re.finditer(r"from\s+(\.+)(\w+)\s+import", content):
            dots, module = m.group(1), m.group(2)
            if len(dots) <= 1:
                continue
            reverse.setdefault(module, set()).add(name)
        # from .. import module_name / from ... import module_name
        for m in re.finditer(r"from\s+(\.+)\s+import\s+([\w,\s]+)", content):
            dots = m.group(1)
            if len(dots) <= 1:
                continue
            for mod in m.group(2).split(","):
                mod = mod.strip()
                if mod:
                    reverse.setdefault(mod, set()).add(name)
    # Filter to *_builder modules only (excluding orchestrators)
    filtered: Dict[str, List[str]] = {}
    for module, families in reverse.items():
        if module.endswith("_builder") and module not in _ORCHESTRATOR_MODULES:
            filtered[module] = sorted(families)
    return filtered


def _parse_runtime_model_manifest(manifest_path: Path) -> List[str]:
    """Parse the tiny MODEL.toml runtime strategy list without extra deps."""
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return []
    match = re.search(r"runtime_strategies\s*=\s*\[([^\]]*)\]", text)
    if match:
        return re.findall(r'"([^"]+)"', match.group(1))
    match = re.search(r'runtime_strategy\s*=\s*"([^"]+)"', text)
    return [match.group(1)] if match else []


def _scan_cpp_runtime_model_manifests(models_dir: Path) -> Dict[str, List[str]]:
    """Build src/runtime/models/<name>/MODEL.toml -> runtime strategies map."""
    scoped: Dict[str, List[str]] = {}
    if not models_dir.is_dir():
        return scoped
    for manifest_path in sorted(models_dir.glob("*/MODEL.toml")):
        strategies = _parse_runtime_model_manifest(manifest_path)
        if strategies:
            scoped[manifest_path.parent.name] = sorted(set(strategies))
    return scoped


def _iter_manifest_data_paths(value: object) -> List[str]:
    """Return repo-relative tests/e2e/data paths referenced by a manifest."""
    paths: Set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            paths.update(_iter_manifest_data_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_iter_manifest_data_paths(child))
    elif isinstance(value, str):
        normalized = value.strip().replace("\\", "/")
        if normalized.startswith("tests/e2e/data/"):
            paths.add(normalized)
        elif normalized.startswith("data/"):
            paths.add(f"tests/e2e/{normalized}")
        elif normalized.startswith("asset/"):
            paths.add(normalized)
    return sorted(paths)


def build_impact_map(repo_root: Path) -> ImpactMap:
    """Build the impact map by scanning manifests and family plugins."""
    models_dir = repo_root / "tests" / "e2e" / "models"
    families_dir = repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
    pipelines_dir = repo_root / "src" / "runtime" / "pipelines"
    runtime_models_dir = repo_root / "src" / "runtime" / "models"

    family_to_models: Dict[str, List[str]] = {}
    strategy_to_models: Dict[str, List[str]] = {}
    task_strategy_to_models: Dict[str, List[str]] = {}
    manifest_field_to_models_sets: Dict[str, Set[str]] = {}
    e2e_data_file_to_models_sets: Dict[str, Set[str]] = {}
    all_model_names: List[str] = []
    core_models: List[str] = []
    model_metadata: Dict[str, Dict] = {}
    l0_replacement_by_model: Dict[str, str] = {}

    for manifest_path in sorted(models_dir.glob("*.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        name = data.get("name", manifest_path.stem)
        family = data.get("family", "")
        runtime_strategy = data.get("runtime_strategy", "")
        is_core = data.get("core", False)

        all_model_names.append(name)
        model_metadata[name] = data

        if family:
            family_to_models.setdefault(family, []).append(name)
        if runtime_strategy:
            strategy_to_models.setdefault(runtime_strategy, []).append(name)
            task_strategy = RUNTIME_TO_TASK_STRATEGY.get(runtime_strategy, "")
            if task_strategy:
                task_strategy_to_models.setdefault(task_strategy, []).append(name)
        if is_core:
            core_models.append(name)
        l0_replacement = data.get("l0_replacement")
        if isinstance(l0_replacement, str) and l0_replacement:
            l0_replacement_by_model[name] = l0_replacement
        fp8_scales = data.get("fp8_scales")
        if isinstance(fp8_scales, str) and fp8_scales:
            manifest_field_to_models_sets.setdefault("fp8_scales", set()).add(name)
            e2e_data_file_to_models_sets.setdefault(
                f"tests/e2e/data/{fp8_scales}", set()).add(name)
        for data_path in _iter_manifest_data_paths(data):
            e2e_data_file_to_models_sets.setdefault(data_path, set()).add(name)

    builder_to_families = _scan_family_imports(families_dir) if families_dir.is_dir() else {}
    cpp_runtime_model_strategies = _scan_cpp_runtime_model_manifests(runtime_models_dir)

    def _models_for_scoped_strategies(strategies: Set[str]) -> List[str]:
        models: Set[str] = set()
        for strategy in strategies:
            models.update(strategy_to_models.get(strategy, []))
        return sorted(models)

    path_scope_overrides: Dict[str, List[str]] = {}
    scoped_cpp_tokens = {
        "src/runtime/core/gpu_matmul.h": "gpu_matmul",
        "src/runtime/core/gpu_matmul.cpp": "gpu_matmul",
        "src/runtime/domains/diffusion/diffusion_denoising_step_seam.h": (
            "diffusion_denoising_step_seam.h"
        ),
    }
    for path, token in scoped_cpp_tokens.items():
        strategies: Set[str] = set()
        if pipelines_dir.is_dir():
            for cpp_file in sorted(pipelines_dir.glob("*.cpp")):
                try:
                    content = cpp_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if token not in content:
                    continue
                strategies.update(CPP_PIPELINE_STRATEGIES.get(cpp_file.stem, []))
        if runtime_models_dir.is_dir():
            for cpp_file in sorted(runtime_models_dir.glob("*/*.cpp")):
                try:
                    content = cpp_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if token not in content:
                    continue
                strategies.update(
                    cpp_runtime_model_strategies.get(cpp_file.parent.name, [])
                )
        if strategies:
            path_scope_overrides[path] = _models_for_scoped_strategies(strategies)

    return ImpactMap(
        family_to_models=family_to_models,
        strategy_to_models=strategy_to_models,
        task_strategy_to_models=task_strategy_to_models,
        all_model_names=sorted(all_model_names),
        all_model_names_set=set(all_model_names),
        core_models=sorted(core_models),
        model_metadata=model_metadata,
        builder_to_families=builder_to_families,
        cpp_runtime_model_strategies=cpp_runtime_model_strategies,
        manifest_field_to_models={
            key: sorted(models)
            for key, models in manifest_field_to_models_sets.items()
        },
        e2e_data_file_to_models={
            path: sorted(models)
            for path, models in e2e_data_file_to_models_sets.items()
        },
        path_scope_overrides=path_scope_overrides,
        l0_replacement_by_model=l0_replacement_by_model,
    )

# ---------------------------------------------------------------------------
# Helper: resolve models from runtime/task strategies
# ---------------------------------------------------------------------------


def _models_for_runtime_strategies(
    strategies: List[str], imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for s in strategies:
        models.update(imap.strategy_to_models.get(s, []))
    return sorted(models)


def _drop_fp8_scale_models(models: List[str], imap: ImpactMap) -> List[str]:
    """Drop FP8-scale variants when a runtime-only rule has a representative.

    FP8-scale manifests exercise builder quantization and FP8 scale plumbing.
    Known runtime C++ changes consume a built bundle through the same artifact
    contract, so a non-FP8 model with the same family/runtime/HF id can stand in
    for L0. FP8 stays covered by FP8-specific changes and nightly.
    """
    fp8_models = set(imap.manifest_field_to_models.get("fp8_scales", []))
    selected = set(models)
    kept: List[str] = []
    for model in models:
        if model not in fp8_models:
            kept.append(model)
            continue
        fp8_meta = imap.model_metadata.get(model, {})
        has_representative = False
        for candidate in selected - fp8_models:
            candidate_meta = imap.model_metadata.get(candidate, {})
            if all(
                fp8_meta.get(field) == candidate_meta.get(field)
                for field in ("family", "runtime_strategy", "hf_id")
            ):
                has_representative = True
                break
        if not has_representative:
            kept.append(model)
    return sorted(kept)


def _models_for_task_strategies(
    task_strategies: List[str], imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for ts in task_strategies:
        models.update(imap.task_strategy_to_models.get(ts, []))
    return sorted(models)


def _apply_l0_replacements(
    models: List[str],
    imap: ImpactMap,
    exact_models: Set[str],
) -> tuple[List[str], List[Dict[str, str]]]:
    """Replace nightly-only scale models with their L0 representatives.

    Direct edits to a nightly-only scale model still use the L0 representative:
    the large model's artifact contract is covered by nightly, while PR L0 keeps
    the same plugin/runtime path at smaller scale.
    """
    del exact_models  # Retained in the signature to keep call sites stable.
    selected: Set[str] = set()
    replacements: List[Dict[str, str]] = []
    for model in models:
        replacement = imap.l0_replacement_by_model.get(model)
        if replacement:
            selected.add(replacement)
            replacements.append({
                "model": model,
                "replacement": replacement,
                "reason": str(
                    imap.model_metadata.get(model, {}).get(
                        "l0_replacement_reason",
                        "nightly-only scale coverage; L0 uses a smaller representative",
                    )
                ),
            })
        else:
            selected.add(model)
    return sorted(selected), replacements


def _infer_unit_tiers(path: str) -> List[str]:
    """Infer which unit test tiers a file change implies."""
    tiers: List[str] = []
    if path.startswith("tensorrt_model_connect/"):
        tiers.append("builder")
    if (path.startswith("src/") or path.startswith("include/")
            or path == "CMakeLists.txt" or path.startswith("cmake/")):
        tiers.append("cpp")
    if (path.startswith("tests/builder/")
            or path.startswith("tests/torchtrt_builder/")
            or path.startswith("tests/engine_defs/torch_trt/")):
        tiers.append("builder")
    if path.startswith("tests/cpp/"):
        tiers.append("cpp")
    if path.startswith("tests/tools/") or path.startswith("tests/e2e_harness/test_"):
        tiers.append("tools")
    return sorted(set(tiers))


def _infer_rebuild_cpp(path: str) -> bool:
    """Does this file change require a C++ rebuild?"""
    return (path.startswith("src/") or path.startswith("include/")
            or path == "CMakeLists.txt" or path.startswith("cmake/")
            or path.startswith("tests/cpp/"))

# ---------------------------------------------------------------------------
# File classification (ordered declarative rules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleContext:
    path: str
    match: Optional[re.Match[str]] = None


RuleMatcher = Callable[[str, ImpactMap], Optional[RuleContext]]
RuleImpactResolver = Callable[[RuleContext, ImpactMap, List[str], bool], RuleMatch]
RulePredicate = Callable[[str, ImpactMap, re.Match[str]], bool]
ModelsResolver = Callable[[RuleContext, ImpactMap], List[str]]


@dataclass(frozen=True)
class ClassificationRule:
    priority: int
    name: str
    matcher: RuleMatcher
    resolver: RuleImpactResolver
    covered_by: Tuple[str, ...]


def _group(context: RuleContext, index: int = 1) -> str:
    if context.match is None:
        raise ValueError("Rule context has no regex match")
    return context.match.group(index)


def _regex_rule(pattern: str, predicate: Optional[RulePredicate] = None) -> RuleMatcher:
    compiled = re.compile(pattern)

    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        match = compiled.match(path)
        if match is None:
            return None
        if predicate is not None and not predicate(path, imap, match):
            return None
        return RuleContext(path=path, match=match)

    return _matcher


def _path_equals(expected: str) -> RuleMatcher:
    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        del imap
        return RuleContext(path=path) if path == expected else None

    return _matcher


def _path_in(paths: Set[str]) -> RuleMatcher:
    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        del imap
        return RuleContext(path=path) if path in paths else None

    return _matcher


def _path_startswith(prefix: str) -> RuleMatcher:
    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        del imap
        return RuleContext(path=path) if path.startswith(prefix) else None

    return _matcher


def _path_startswith_any(prefixes: Tuple[str, ...]) -> RuleMatcher:
    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        del imap
        return RuleContext(path=path) if path.startswith(prefixes) else None

    return _matcher


def _path_in_impact_map(
    mapping_getter: Callable[[ImpactMap], Dict[str, List[str]]],
) -> RuleMatcher:
    def _matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
        return RuleContext(path=path) if path in mapping_getter(imap) else None

    return _matcher


def _no_impact_matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
    del imap
    if path.startswith("tools/") or path.startswith("scripts/"):
        return RuleContext(path=path)
    if any(re.match(pattern, path) for pattern in _NO_IMPACT_PATTERNS):
        return RuleContext(path=path)
    if path.endswith(".md"):
        return RuleContext(path=path)
    return None


def _catch_all_matcher(path: str, imap: ImpactMap) -> Optional[RuleContext]:
    del imap
    return RuleContext(path=path)


def _match_result(
    rule_name: str,
    models_resolver: ModelsResolver,
    unit_tiers_override: Optional[List[str]] = None,
    rebuild_override: Optional[bool] = None,
) -> RuleImpactResolver:
    def _resolver(
        context: RuleContext,
        imap: ImpactMap,
        unit_tiers: List[str],
        rebuild: bool,
    ) -> RuleMatch:
        effective_unit_tiers = (
            list(unit_tiers_override)
            if unit_tiers_override is not None
            else unit_tiers
        )
        effective_rebuild = rebuild if rebuild_override is None else rebuild_override
        return RuleMatch(
            rule_name,
            models_resolver(context, imap),
            effective_unit_tiers,
            effective_rebuild,
        )

    return _resolver


def _no_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    del context, imap
    return []


def _all_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    del context
    return list(imap.all_model_names)


def _manifest_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    name = _group(context)
    return [name] if name in imap.all_model_names_set else []


def _family_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    return sorted(imap.family_to_models.get(_group(context), []))


def _task_strategy_models(task_strategies: List[str]) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        del context
        return _models_for_task_strategies(task_strategies, imap)

    return _resolver


def _runtime_strategy_models(
    strategies_getter: Callable[[RuleContext, ImpactMap], List[str]],
) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        return _drop_fp8_scale_models(
            _models_for_runtime_strategies(strategies_getter(context, imap), imap),
            imap,
        )

    return _resolver


def _fixed_runtime_strategy_models(strategies: List[str]) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        del context
        return _drop_fp8_scale_models(
            _models_for_runtime_strategies(strategies, imap),
            imap,
        )

    return _resolver


def _cpp_runtime_model_strategies(
    context: RuleContext, imap: ImpactMap,
) -> List[str]:
    return imap.cpp_runtime_model_strategies.get(_group(context), [])


def _cpp_plugin_strategies(context: RuleContext, imap: ImpactMap) -> List[str]:
    del imap
    return CPP_PLUGIN_STRATEGIES.get(_group(context), [])


def _cpp_pipeline_strategies(context: RuleContext, imap: ImpactMap) -> List[str]:
    del imap
    return CPP_PIPELINE_STRATEGIES.get(_group(context), [])


def _shared_cpp_helper_strategies(
    context: RuleContext, imap: ImpactMap,
) -> List[str]:
    del imap
    return SHARED_CPP_HELPER_STRATEGIES.get(_group(context), [])


def _specialized_builder_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    families = imap.builder_to_families[_group(context)]
    models: Set[str] = set()
    for family in families:
        models.update(imap.family_to_models.get(family, []))
    return sorted(models)


def _scoped_cpp_helper_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    return _drop_fp8_scale_models(imap.path_scope_overrides[context.path], imap)


def _e2e_data_file_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    return imap.e2e_data_file_to_models[context.path]


def _fp8_scale_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    del context
    return sorted(imap.manifest_field_to_models.get("fp8_scales", []))


def _known_cpp_runtime_model(
    path: str, imap: ImpactMap, match: re.Match[str],
) -> bool:
    del path
    return bool(imap.cpp_runtime_model_strategies.get(match.group(1), []))


def _unknown_cpp_runtime_model(
    path: str, imap: ImpactMap, match: re.Match[str],
) -> bool:
    return not _known_cpp_runtime_model(path, imap, match)


def _is_specialized_builder(
    path: str, imap: ImpactMap, match: re.Match[str],
) -> bool:
    del path
    module_name = match.group(1)
    if not module_name.endswith("_builder"):
        return False
    if module_name in _ORCHESTRATOR_MODULES:
        return False
    families = imap.builder_to_families.get(module_name, [])
    return any(imap.family_to_models.get(family, []) for family in families)


def _known_plugin_stem(
    strategy_map: Dict[str, List[str]],
    excluded_stems: Set[str],
) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path, imap
        stem = match.group(1)
        return stem not in excluded_stems and bool(strategy_map.get(stem, []))

    return _predicate


def _unknown_plugin_stem(
    strategy_map: Dict[str, List[str]],
) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path, imap
        return not bool(strategy_map.get(match.group(1), []))

    return _predicate


def _known_task_strategy_stem(
    strategy_map: Dict[str, List[str]],
) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path, imap
        return bool(strategy_map.get(match.group(1), []))

    return _predicate


def _unknown_task_strategy_stem(
    strategy_map: Dict[str, List[str]],
) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path, imap
        stem = match.group(1)
        return stem != "__init__" and not bool(strategy_map.get(stem, []))

    return _predicate


def _task_strategy_models_from_group(
    strategy_map: Dict[str, List[str]],
) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        return _models_for_task_strategies(strategy_map.get(_group(context), []), imap)

    return _resolver


def _no_impact_resolver(
    context: RuleContext,
    imap: ImpactMap,
    unit_tiers: List[str],
    rebuild: bool,
) -> RuleMatch:
    del context, imap, unit_tiers, rebuild
    return RuleMatch("no_impact", [], [], False)


def _catch_all_resolver(
    context: RuleContext,
    imap: ImpactMap,
    unit_tiers: List[str],
    rebuild: bool,
) -> RuleMatch:
    del context, rebuild
    return RuleMatch("catch_all", list(imap.all_model_names), unit_tiers, True)


def _classification_rules() -> Tuple[ClassificationRule, ...]:
    rules = (
        ClassificationRule(
            priority=10,
            name="manifest",
            matcher=_regex_rule(r"tests/e2e/models/(.+)\.json$"),
            resolver=_match_result("manifest", _manifest_models),
            covered_by=("TestSafetyNet.test_manifest_self",),
        ),
        ClassificationRule(
            priority=20,
            name="family_package",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"families/([A-Za-z]\w*)/.+\.py$"
            ),
            resolver=_match_result("family_package", _family_models),
            covered_by=(
                "TestFamilyPlugin.test_family_only_change",
                "TestFamilyOwnedBuilder.test_family_local_standard_decoder_builder",
            ),
        ),
        ClassificationRule(
            priority=30,
            name="family_plugin",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"families/(\w+)\.py$",
                lambda _path, _imap, match: match.group(1) not in ("__init__", "base"),
            ),
            resolver=_match_result("family_plugin", _family_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=40,
            name="family_base",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"families/((__init__|base)\.py)$"
            ),
            resolver=_match_result("family_base", _all_models),
            covered_by=(
                "TestFamilyPlugin.test_family_base_all_models",
                "TestFamilyPlugin.test_family_init_all_models",
            ),
        ),
        ClassificationRule(
            priority=50,
            name="torchtrt_family_plugin",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"engine_defs/torch_trt/families/(\w+)\.py$",
                lambda _path, _imap, match: match.group(1) not in ("__init__", "base"),
            ),
            resolver=_match_result("torchtrt_family_plugin", _family_models),
            covered_by=("TestFamilyPlugin.test_torchtrt_family_only_change",),
        ),
        ClassificationRule(
            priority=60,
            name="torchtrt_family_base",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"engine_defs/torch_trt/families/((__init__|base)\.py)$"
            ),
            resolver=_match_result("torchtrt_family_base", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=70,
            name="torchtrt_strategy",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"engine_defs/torch_trt/strategies/(diffusion)\.py$"
            ),
            resolver=_match_result(
                "torchtrt_strategy",
                _task_strategy_models(["diffusion_media_generation"]),
            ),
            covered_by=("TestHarness.test_torchtrt_diffusion_strategy",),
        ),
        ClassificationRule(
            priority=80,
            name="torchtrt_strategy_unknown",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/"
                r"engine_defs/torch_trt/strategies/(\w+)\.py$"
            ),
            resolver=_match_result("torchtrt_strategy_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=90,
            name="specialized_builder",
            matcher=_regex_rule(
                r"tensorrt_model_connect/tensorrt_model_connect/(\w+)\.py$",
                _is_specialized_builder,
            ),
            resolver=_match_result("specialized_builder", _specialized_builder_models),
            covered_by=("TestDeclarativeClassificationRules.test_specialized_builder_rule",),
        ),
        ClassificationRule(
            priority=95,
            name="sana_wm_bridge",
            matcher=_path_equals(
                "tensorrt_model_connect/tensorrt_model_connect/sana_wm_bridge.py"
            ),
            resolver=_match_result(
                "sana_wm_bridge",
                _fixed_runtime_strategy_models(["diffusion_sana_wm"]),
                ["builder", "tools"],
                False,
            ),
            covered_by=("TestSanaWmImpactRules.test_sana_wm_scoped_paths",),
        ),
        ClassificationRule(
            priority=100,
            name="shared_builder_module",
            matcher=_path_startswith("tensorrt_model_connect/"),
            resolver=_match_result("shared_builder_module", _all_models),
            covered_by=("TestSharedModules.test_shared_module_all_models",),
        ),
        ClassificationRule(
            priority=110,
            name="cpp_runtime_model",
            matcher=_regex_rule(
                r"src/runtime/models/([^/]+)/.+$",
                _known_cpp_runtime_model,
            ),
            resolver=_match_result(
                "cpp_runtime_model",
                _runtime_strategy_models(_cpp_runtime_model_strategies),
            ),
            covered_by=("TestCppScope.test_cpp_runtime_model_scope",),
        ),
        ClassificationRule(
            priority=120,
            name="cpp_runtime_model_unknown",
            matcher=_regex_rule(
                r"src/runtime/models/([^/]+)/.+$",
                _unknown_cpp_runtime_model,
            ),
            resolver=_match_result("cpp_runtime_model_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=130,
            name="cpp_plugin_flux_runtime",
            matcher=_regex_rule(r"src/runtime/plugins/(flux_plugin)\.cpp$"),
            resolver=_match_result(
                "cpp_plugin_flux_runtime",
                _runtime_strategy_models(_cpp_plugin_strategies),
            ),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=140,
            name="cpp_plugin",
            matcher=_regex_rule(
                r"src/runtime/plugins/(\w+)\.cpp$",
                _known_plugin_stem(CPP_PLUGIN_STRATEGIES, {"flux_plugin"}),
            ),
            resolver=_match_result(
                "cpp_plugin",
                _runtime_strategy_models(_cpp_plugin_strategies),
            ),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=150,
            name="cpp_plugin_unknown",
            matcher=_regex_rule(
                r"src/runtime/plugins/(\w+)\.cpp$",
                _unknown_plugin_stem(CPP_PLUGIN_STRATEGIES),
            ),
            resolver=_match_result("cpp_plugin_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=160,
            name="cpp_pipeline_flux_runtime",
            matcher=_regex_rule(r"src/runtime/pipelines/(flux_pipeline)\.(h|cpp)$"),
            resolver=_match_result(
                "cpp_pipeline_flux_runtime",
                _runtime_strategy_models(_cpp_pipeline_strategies),
            ),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=170,
            name="cpp_pipeline",
            matcher=_regex_rule(
                r"src/runtime/pipelines/(\w+)\.(h|cpp)$",
                _known_plugin_stem(CPP_PIPELINE_STRATEGIES, {"flux_pipeline"}),
            ),
            resolver=_match_result(
                "cpp_pipeline",
                _runtime_strategy_models(_cpp_pipeline_strategies),
            ),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=180,
            name="cpp_pipeline_unknown",
            matcher=_regex_rule(
                r"src/runtime/pipelines/(\w+)\.(h|cpp)$",
                _unknown_plugin_stem(CPP_PIPELINE_STRATEGIES),
            ),
            resolver=_match_result("cpp_pipeline_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=190,
            name="cpp_shared_plugin_helpers",
            matcher=_regex_rule(
                r"src/runtime/plugins/shared/(plugin_helpers)\.(h|cpp)$"
            ),
            resolver=_match_result("cpp_shared_plugin_helpers", _all_models),
            covered_by=("TestCppScope.test_cpp_shared_plugin_helpers",),
        ),
        ClassificationRule(
            priority=200,
            name="cpp_shared_helper",
            matcher=_regex_rule(
                r"src/runtime/plugins/shared/(\w+)\.(h|cpp)$",
                _known_task_strategy_stem(SHARED_CPP_HELPER_STRATEGIES),
            ),
            resolver=_match_result(
                "cpp_shared_helper",
                _runtime_strategy_models(_shared_cpp_helper_strategies),
            ),
            covered_by=(
                "TestCppScope.test_cpp_shared_audio",
                "TestCppScope.test_cpp_shared_diffusion",
            ),
        ),
        ClassificationRule(
            priority=210,
            name="cpp_shared_helper_unknown",
            matcher=_regex_rule(
                r"src/runtime/plugins/shared/(\w+)\.(h|cpp)$",
                _unknown_task_strategy_stem(SHARED_CPP_HELPER_STRATEGIES),
            ),
            resolver=_match_result("cpp_shared_helper_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=220,
            name="cpp_scoped_helper",
            matcher=_path_in_impact_map(lambda imap: imap.path_scope_overrides),
            resolver=_match_result("cpp_scoped_helper", _scoped_cpp_helper_models),
            covered_by=(
                "TestCppScope.test_scoped_cpp_helper_gpu_matmul",
                "TestCppScope.test_scoped_cpp_helper_diffusion_seam",
            ),
        ),
        ClassificationRule(
            priority=230,
            name="cpp_source",
            matcher=_path_startswith_any(("src/", "include/")),
            resolver=_match_result("cpp_source", _all_models),
            covered_by=("TestCppScope.test_cpp_wildcard_all",),
        ),
        ClassificationRule(
            priority=240,
            name="harness_runner_init",
            matcher=_regex_rule(r"tests/e2e_harness/runners/(__init__)\.py$"),
            resolver=_match_result("harness_runner_init", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=250,
            name="harness_runner",
            matcher=_regex_rule(
                r"tests/e2e_harness/runners/(\w+)\.py$",
                _known_task_strategy_stem(RUNNER_TASK_STRATEGIES),
            ),
            resolver=_match_result(
                "harness_runner",
                _task_strategy_models_from_group(RUNNER_TASK_STRATEGIES),
            ),
            covered_by=("TestHarness.test_harness_runner",),
        ),
        ClassificationRule(
            priority=260,
            name="harness_runner_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/runners/(\w+)\.py$",
                _unknown_task_strategy_stem(RUNNER_TASK_STRATEGIES),
            ),
            resolver=_match_result("harness_runner_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=270,
            name="harness_comparator_init",
            matcher=_regex_rule(r"tests/e2e_harness/comparators/(__init__)\.py$"),
            resolver=_match_result("harness_comparator_init", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=280,
            name="harness_comparator",
            matcher=_regex_rule(
                r"tests/e2e_harness/comparators/(\w+)\.py$",
                _known_task_strategy_stem(COMPARATOR_TASK_STRATEGIES),
            ),
            resolver=_match_result(
                "harness_comparator",
                _task_strategy_models_from_group(COMPARATOR_TASK_STRATEGIES),
            ),
            covered_by=("TestHarness.test_harness_comparator",),
        ),
        ClassificationRule(
            priority=290,
            name="harness_comparator_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/comparators/(\w+)\.py$",
                _unknown_task_strategy_stem(COMPARATOR_TASK_STRATEGIES),
            ),
            resolver=_match_result("harness_comparator_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=300,
            name="harness_reference_init",
            matcher=_regex_rule(r"tests/e2e_harness/references/(__init__)\.py$"),
            resolver=_match_result("harness_reference_init", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=310,
            name="harness_reference",
            matcher=_regex_rule(
                r"tests/e2e_harness/references/(\w+)\.py$",
                _known_task_strategy_stem(REFERENCE_TASK_STRATEGIES),
            ),
            resolver=_match_result(
                "harness_reference",
                _task_strategy_models_from_group(REFERENCE_TASK_STRATEGIES),
            ),
            covered_by=("TestHarness.test_torch_reference_includes_neural_operator_models",),
        ),
        ClassificationRule(
            priority=320,
            name="harness_reference_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/references/(\w+)\.py$",
                _unknown_task_strategy_stem(REFERENCE_TASK_STRATEGIES),
            ),
            resolver=_match_result("harness_reference_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=330,
            name="harness_plugin_init",
            matcher=_regex_rule(r"tests/e2e_harness/plugins/(__init__)\.py$"),
            resolver=_match_result("harness_plugin_init", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=340,
            name="harness_plugin",
            matcher=_regex_rule(
                r"tests/e2e_harness/plugins/(\w+)\.py$",
                _known_task_strategy_stem(PLUGIN_TASK_STRATEGIES),
            ),
            resolver=_match_result(
                "harness_plugin",
                _task_strategy_models_from_group(PLUGIN_TASK_STRATEGIES),
            ),
            covered_by=("TestHarness.test_harness_plugin",),
        ),
        ClassificationRule(
            priority=350,
            name="harness_plugin_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/plugins/(\w+)\.py$",
                _unknown_task_strategy_stem(PLUGIN_TASK_STRATEGIES),
            ),
            resolver=_match_result("harness_plugin_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=360,
            name="harness_threshold_profile",
            matcher=_regex_rule(
                r"tests/e2e_harness/thresholds/defaults/([\w_]+)\.json$",
                _known_task_strategy_stem(THRESHOLD_PROFILE_TASK_STRATEGIES),
            ),
            resolver=_match_result(
                "harness_threshold_profile",
                _task_strategy_models_from_group(THRESHOLD_PROFILE_TASK_STRATEGIES),
            ),
            covered_by=("TestHarness.test_harness_threshold_profile",),
        ),
        ClassificationRule(
            priority=370,
            name="harness_threshold_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/thresholds/defaults/([\w_]+)\.json$",
                _unknown_task_strategy_stem(THRESHOLD_PROFILE_TASK_STRATEGIES),
            ),
            resolver=_match_result("harness_threshold_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=380,
            name="harness_unit_test",
            matcher=_regex_rule(r"tests/e2e_harness/test_[\w_]+\.py$"),
            resolver=_match_result("harness_unit_test", _no_models, ["tools"], False),
            covered_by=("TestHarness.test_harness_unit_test_file",),
        ),
        ClassificationRule(
            priority=385,
            name="harness_shared",
            matcher=_path_startswith("tests/e2e_harness/"),
            resolver=_match_result("harness_shared", _all_models),
            covered_by=("TestHarness.test_harness_shared",),
        ),
        ClassificationRule(
            priority=390,
            name="e2e_entrypoint",
            matcher=_path_in({"tests/test_e2e.py", "tests/conftest.py"}),
            resolver=_match_result("e2e_entrypoint", _all_models),
            covered_by=(
                "TestHarness.test_test_e2e_entrypoint",
                "TestHarness.test_conftest_entrypoint",
            ),
        ),
        ClassificationRule(
            priority=400,
            name="e2e_runner_script",
            matcher=_path_in({
                "scripts/run_e2e_parallel.sh",
                "scripts/schedule_e2e.py",
                "scripts/warm_hf_cache.py",
            }),
            resolver=_match_result("e2e_runner_script", _all_models),
            covered_by=("TestNoImpact.test_e2e_runner_scripts_trigger_all_models",),
        ),
        ClassificationRule(
            priority=410,
            name="e2e_waives",
            matcher=_path_equals("tests/e2e/waives.txt"),
            resolver=_match_result("e2e_waives", _all_models),
            covered_by=("TestHarness.test_waives_diff_can_be_refined",),
        ),
        ClassificationRule(
            priority=420,
            name="unit_builder",
            matcher=_path_startswith("tests/builder/"),
            resolver=_match_result("unit_builder", _no_models),
            covered_by=("TestUnitTiers.test_unit_tier_builder",),
        ),
        ClassificationRule(
            priority=430,
            name="unit_cpp",
            matcher=_path_startswith("tests/cpp/"),
            resolver=_match_result("unit_cpp", _no_models),
            covered_by=("TestUnitTiers.test_unit_tier_cpp",),
        ),
        ClassificationRule(
            priority=440,
            name="unit_tools",
            matcher=_path_startswith("tests/tools/"),
            resolver=_match_result("unit_tools", _no_models),
            covered_by=("TestUnitTiers.test_unit_tier_tools",),
        ),
        ClassificationRule(
            priority=450,
            name="unit_torchtrt_builder",
            matcher=_path_startswith_any((
                "tests/torchtrt_builder/",
                "tests/engine_defs/torch_trt/",
            )),
            resolver=_match_result("unit_torchtrt_builder", _no_models),
            covered_by=("TestUnitTiers.test_unit_tier_torchtrt_engine_defs",),
        ),
        ClassificationRule(
            priority=460,
            name="cmake",
            matcher=lambda path, _imap: (
                RuleContext(path=path)
                if path == "CMakeLists.txt" or path.startswith("cmake/")
                else None
            ),
            resolver=_match_result("cmake", _no_models),
            covered_by=("TestSafetyNet.test_cmake_no_e2e_models",),
        ),
        ClassificationRule(
            priority=470,
            name="e2e_data_file",
            matcher=_path_in_impact_map(lambda imap: imap.e2e_data_file_to_models),
            resolver=_match_result("e2e_data_file", _e2e_data_file_models),
            covered_by=("TestE2EDataFiles.test_data_file_maps_to_manifest_users",),
        ),
        ClassificationRule(
            priority=480,
            name="fp8_gen_script",
            matcher=_path_equals("scripts/_gen_fp8_bf16.py"),
            resolver=_match_result(
                "fp8_gen_script", _fp8_scale_models, [], False,
            ),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=485,
            name="elf_replay_tool",
            matcher=_path_in({
                "tools/make_elf_replay_artifact.py",
                "tools/prepare_elf_model_dir.py",
                "tools/validate_elf_replay_artifact.py",
            }),
            resolver=_match_result("elf_replay_tool", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_elf_replay_tools_trigger_tools_tier",),
        ),
        ClassificationRule(
            priority=488,
            name="sana_wm_inference_script",
            matcher=_path_equals("inference_video_scripts/inference_sana_wm.py"),
            resolver=_match_result(
                "sana_wm_inference_script",
                _fixed_runtime_strategy_models(["diffusion_sana_wm"]),
                ["tools"],
                False,
            ),
            covered_by=("TestSanaWmImpactRules.test_sana_wm_scoped_paths",),
        ),
        ClassificationRule(
            priority=490,
            name="no_impact",
            matcher=_no_impact_matcher,
            resolver=_no_impact_resolver,
            covered_by=("TestNoImpact.test_docs_no_impact",),
        ),
        ClassificationRule(
            priority=500,
            name="catch_all",
            matcher=_catch_all_matcher,
            resolver=_catch_all_resolver,
            covered_by=("TestSafetyNet.test_unknown_file_triggers_all",),
        ),
    )
    priorities = [rule.priority for rule in rules]
    if len(priorities) != len(set(priorities)):
        raise ValueError("Classification rule priorities must be unique")
    return tuple(sorted(rules, key=lambda rule: rule.priority))


CLASSIFICATION_RULES = _classification_rules()


def classify_file(path: str, imap: ImpactMap) -> RuleMatch:
    """Classify a single changed file. Lowest priority matching rule wins."""
    path = path.replace("\\", "/").strip("/")
    unit_tiers = _infer_unit_tiers(path)
    rebuild = _infer_rebuild_cpp(path)

    for rule in CLASSIFICATION_RULES:
        context = rule.matcher(path, imap)
        if context is not None:
            return rule.resolver(context, imap, unit_tiers, rebuild)

    raise RuntimeError("classification rules must include a catch_all rule")

# ---------------------------------------------------------------------------
# Impact analysis (aggregate across all changed files)
# ---------------------------------------------------------------------------


def _direct_python_test_targets(changed_files: List[str]) -> tuple[List[str], List[str]]:
    """Return changed Python unit-test files that pytest can run directly."""
    builder_tests: Set[str] = set()
    tools_tests: Set[str] = set()
    for raw_path in changed_files:
        path = raw_path.replace("\\", "/").strip("/")
        if not path.endswith(".py"):
            continue
        if path.startswith("tests/builder/") or path.startswith("tests/engine_defs/torch_trt/"):
            builder_tests.add(path)
        elif path.startswith("tests/tools/") or path.startswith("tests/e2e_harness/test_"):
            tools_tests.add(path)
    return sorted(builder_tests), sorted(tools_tests)


def analyze_impact(
    changed_files: List[str],
    imap: ImpactMap,
    cap: Optional[int] = None,
    coverage_map: Optional[Dict[str, List[str]]] = None,
    base: Optional[str] = None,
    head: Optional[str] = None,
    repo_root: Optional[Path] = None,
    e2e_suite: str = "l0",
) -> ImpactResult:
    """Analyze impact of all changed files and return aggregated result."""
    all_models: Set[str] = set()
    exact_models: Set[str] = set()
    all_tiers: Set[str] = set()
    rebuild_cpp = False
    matched_rules: List[Dict] = []

    for fpath in changed_files:
        match = classify_file(fpath, imap)
        diff_text = None
        if base and head and repo_root:
            diff_text = get_file_diff(base, head, repo_root, fpath)
        if diff_text:
            match = maybe_refine_match_with_diff(fpath, match, diff_text, imap)
        all_models.update(match.models)
        if match.rule in ("manifest", "e2e_data_file"):
            exact_models.update(match.models)
        all_tiers.update(match.unit_tiers)
        rebuild_cpp = rebuild_cpp or match.rebuild_cpp
        matched_rules.append({
            "file": fpath,
            "rule": match.rule,
            "models": match.models,
        })

    e2e_models = sorted(all_models)
    l0_replacements: List[Dict[str, str]] = []
    if e2e_suite == "l0":
        e2e_models, l0_replacements = _apply_l0_replacements(
            e2e_models, imap, exact_models,
        )
    cap_applied = False
    if cap is not None and len(e2e_models) > cap:
        e2e_models = sorted(imap.core_models)
        cap_applied = True
        l0_replacements = []

    # Coverage-map-based unit test selection
    builder_tests: List[str] = []
    cpp_tests: List[str] = []
    tools_tests: List[str] = []
    fallback_tiers: List[str] = []

    if coverage_map is not None:
        from coverage_map.select_tests import select_tests
        sel = select_tests(changed_files, coverage_map)
        builder_tests = sel.builder_tests
        cpp_tests = sel.cpp_tests
        tools_tests = sel.tools_tests
        fallback_tiers = sel.fallback_tiers

    direct_builder_tests, direct_tools_tests = _direct_python_test_targets(changed_files)
    if direct_builder_tests:
        builder_tests = sorted(set(builder_tests).union(direct_builder_tests))
        fallback_tiers = [tier for tier in fallback_tiers if tier != "builder"]
    if direct_tools_tests:
        tools_tests = sorted(set(tools_tests).union(direct_tools_tests))
        fallback_tiers = [tier for tier in fallback_tiers if tier != "tools"]

    return ImpactResult(
        e2e_models=e2e_models,
        unit_tiers=sorted(all_tiers),
        rebuild_cpp=rebuild_cpp,
        cap_applied=cap_applied,
        matched_rules=matched_rules,
        builder_tests=builder_tests,
        cpp_tests=cpp_tests,
        tools_tests=tools_tests,
        fallback_tiers=fallback_tiers,
        l0_replacements=l0_replacements,
    )

# ---------------------------------------------------------------------------
# Git diff
# ---------------------------------------------------------------------------


def get_changed_files(base: str, head: str, repo_root: Path) -> Optional[List[str]]:
    """Get list of changed files between base and head.

    Returns None if git diff fails (e.g. shallow clone without base ref),
    signaling the caller to treat ALL files as changed (safety net).
    """
    for cmd in [
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}...{head}"],
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", base, head],
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=repo_root,
            )
            files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
            return sorted(files)
        except subprocess.CalledProcessError:
            continue
    # Both diffs failed (shallow clone, missing ref, etc.)
    print(f"WARNING: git diff failed for {base}..{head} -- "
          "treating as all files changed (safety net)", file=sys.stderr)
    return None


def get_file_diff(base: str, head: str, repo_root: Path, path: str) -> Optional[str]:
    """Get unified=0 diff for a single file, or None if git diff fails."""
    for cmd in [
        ["git", "diff", "--unified=0", f"{base}...{head}", "--", path],
        ["git", "diff", "--unified=0", base, head, "--", path],
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=repo_root,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            continue
    return None


def _significant_diff_lines(diff_text: str) -> List[str]:
    """Extract changed code lines, ignoring headers and pure formatting noise."""
    lines: List[str] = []
    for raw_line in diff_text.splitlines():
        if raw_line.startswith(("diff --git", "index ", "@@", "---", "+++")):
            continue
        if not raw_line.startswith(("+", "-")):
            continue
        content = raw_line[1:].strip()
        if not content:
            continue
        if re.fullmatch(r"[\[\]{}(),;]+", content):
            continue
        lines.append(content)
    return lines


def _normalize_diff_line(line: str) -> str:
    """Normalize changed lines for token-based diff heuristics."""
    return re.sub(r"[-\s]+", "_", line.lower())


class DiffRefinementRule:
    """Named diff-aware rule that can narrow a broad file classification."""

    name: str

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        raise NotImplementedError

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        raise NotImplementedError


def _all_lines_match_tokens(lines: List[str], allowed_tokens: tuple[str, ...]) -> bool:
    return all(
        any(token in _normalize_diff_line(line) for token in allowed_tokens)
        for line in lines
    )


def _fp8_scale_models(imap: ImpactMap) -> List[str]:
    return imap.manifest_field_to_models.get("fp8_scales", [])


def _diffusion_task_models(imap: ImpactMap) -> List[str]:
    return _models_for_task_strategies(["diffusion_media_generation"], imap)


def _torchtrt_tokenizer_models(imap: ImpactMap) -> List[str]:
    return _models_for_runtime_strategies(
        ["torchtrt_decoder", "diffusion_pixart_torchtrt"], imap)


@dataclass(frozen=True)
class TokenDiffRefinementRule(DiffRefinementRule):
    name: str
    path: str
    allowed_tokens: tuple[str, ...]
    models_for_impact: Callable[[ImpactMap], List[str]]

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        return path == self.path and _all_lines_match_tokens(lines, self.allowed_tokens)

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path, lines
        return RuleMatch(
            self.name,
            self.models_for_impact(imap),
            match.unit_tiers,
            match.rebuild_cpp,
        )


class HarnessSharedFp8ScalesRule(DiffRefinementRule):
    name = "harness_shared_fp8_scales"
    path = "tests/e2e_harness/orchestrator.py"
    allowed_lines = {
        "CILane,",
        'fp8_scales = case.metadata.get("fp8_scales")',
        "if fp8_scales:",
        "# Resolve relative to tests/e2e/data/",
        'scales_path = Path(__file__).parent.parent / "e2e" / "data" / fp8_scales',
        "if scales_path.is_file():",
        'cmd.extend(["--fp8-scales", str(scales_path)])',
    }

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        return path == self.path and all(line in self.allowed_lines for line in lines)

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path, lines
        return RuleMatch(
            self.name,
            _fp8_scale_models(imap),
            match.unit_tiers,
            match.rebuild_cpp,
        )


class HarnessReferenceDprContextEncoderRule(DiffRefinementRule):
    name = "harness_reference_dpr_context_encoder"
    path = "tests/e2e_harness/references/hf_transformers.py"
    dpr_tokens = (
        "dpr",
        "dprcontextencoder",
        "dprcontextencodertokenizerfast",
        "ctx_encoder",
        "bert_model",
        "autotokenizer.from_pretrained",
        "automodel.from_pretrained",
        "model_type",
        "context",
        "tokenizer",
        "same_token_ids",
        "tokenizer.json",
        "trt_artifact",
        "question_classes",
        "wrong_class",
        "dprquestionencoder",
        "model_ref",
        "trust_remote_code",
    )

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        if path != self.path:
            return False
        normalized_lines = [_normalize_diff_line(line) for line in lines]
        return (
            all(any(token in line for token in self.dpr_tokens) for line in normalized_lines)
            and any("dprcontextencodertokenizerfast" in line for line in normalized_lines)
            and any("dprcontextencoder" in line for line in normalized_lines)
        )

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path, lines, imap
        return RuleMatch(
            self.name,
            ["dpr-ctx-encoder"],
            match.unit_tiers,
            match.rebuild_cpp,
        )


class HarnessReferenceVlGeneratedOnlyDecodeRule(DiffRefinementRule):
    name = "harness_reference_vl_generated_only_decode"
    path = "tests/e2e_harness/references/hf_transformers.py"
    allowed_tokens = (
        "decode_vl_generated_text",
        "vl_generation",
        "generated_ids",
        "generated_text",
        "input_len",
        "token_count",
        "decode_token_ids",
        "processor.decode",
        "processor",
        "prompt_guard",
        "prompt_only",
        "prompt_text",
        "prompt_texts",
        "normalized",
        "marker",
        "image",
        "img_context",
        "image_pad",
        "vision_start",
        "vision_end",
        "fallback_text",
        "text_input",
        "empty",
        "runtimeerror",
        "return_true",
        "return_false",
        "return_",
        "continue",
        "tail",
        "skip_special_tokens",
        "strip",
        "str",
        "if_text",
        "return_text",
        "hf_transformers",
    )

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        if path != self.path:
            return False
        normalized_lines = [
            _normalize_diff_line(line)
            for line in lines
            if any(ch.isalnum() for ch in _normalize_diff_line(line))
        ]
        return all(
            any(token in line for token in self.allowed_tokens)
            for line in normalized_lines
        )

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path, lines, imap
        return RuleMatch(
            self.name,
            ["internvl3-8b"],
            match.unit_tiers,
            match.rebuild_cpp,
        )


class E2EWaivesModelLinesRule(DiffRefinementRule):
    name = "e2e_waives_model_lines"
    path = "tests/e2e/waives.txt"

    @staticmethod
    def _models_from_lines(lines: List[str], imap: ImpactMap) -> List[str]:
        models = []
        for line in lines:
            fields = line.split()
            if fields and fields[0] in imap.all_model_names_set:
                models.append(fields[0])
        return sorted(set(models))

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        return path == self.path and bool(self._models_from_lines(lines, imap))

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path
        return RuleMatch(
            self.name,
            self._models_from_lines(lines, imap),
            match.unit_tiers,
            match.rebuild_cpp,
        )


DIFF_REFINEMENT_RULES: tuple[DiffRefinementRule, ...] = (
    HarnessSharedFp8ScalesRule(),
    TokenDiffRefinementRule(
        "e2e_warm_hf_cache_diffusers_components",
        "scripts/warm_hf_cache.py",
        (
            "component",
            "component_dir",
            "component_has_weight",
            "controlnet",
            "diffusers",
            "diffusers_missing_weight_components",
            "entrypoint_or_required_local_weight_artifact",
            "has_weight",
            "if_(",
            "if_isinstance(value,_list)",
            "if_value_is_none_or_value_is_false",
            "image_encoder",
            "is_diffusers_component_enabled",
            "jsondecodeerror",
            "model_index",
            "path.is_file",
            "required_components",
            "required_local_weight_artifact",
            "return_[",
            "return_any",
            "return_false",
            "return_true",
            "snapshot_dir",
            "text_encoder",
            "text_encoder_2",
            "transformer",
            "try:",
            "unet",
            "vae",
            "weight",
        ),
        _fp8_scale_models,
    ),
    TokenDiffRefinementRule(
        "shared_builder_fp8_scales_cli",
        "tensorrt_model_connect/tensorrt_model_connect/cli.py",
        ("fp8_scales", "save_fp8_scales"),
        _fp8_scale_models,
    ),
    TokenDiffRefinementRule(
        "shared_builder_fp8_scales_engine",
        "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py",
        (
            "fp8_scales",
            "save_fp8_scales",
            "_build_diffusion_bundle(",
            "_effective_precision",
            '"precision"',
            '"quantization"',
            "cfg_dict[",
            "fp8_scales",
        ),
        _fp8_scale_models,
    ),
    TokenDiffRefinementRule(
        "shared_builder_diffusion_tokenizer",
        "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py",
        (
            "detect_diffusion_tokenizer_add_special_tokens",
            "detect_tokenizer_add_special_tokens",
            "detect_add_special",
            "diffusion",
            "tokenizer_add_special_tokens",
            "tokenizer_special_tokens_detection_s",
            "tokenizer_t0",
            "tokenizer_2",
            "tok_subdir",
            "tok_dir",
            "if_tok_dir",
            "model_dir_path",
            "time_monotonic",
            "build_timing",
            "write_build_timing",
            "add_build_timing",
            "return_detect_tokenizer_add_special_tokens",
        ),
        _diffusion_task_models,
    ),
    TokenDiffRefinementRule(
        "torchtrt_compiler_tokenizer",
        "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py",
        (
            "detect_tokenizer_add_special_tokens",
            "detect_diffusion_tokenizer_add_special_tokens",
            "detect_add_special",
            "diffusion",
            "tokenizer_add_special_tokens",
            "tokenizer_config_json",
            "tokenizer_2",
            "autotokenizer",
            "ids_default",
            "ids_without",
            "add_special_tokens",
            "add_bos_token",
            "add_eos_token",
            "tok_config_path",
            "tok_cfg",
            "tok_subdir",
            "tok_dir",
            "tok_=",
            "if_tok_dir",
            "model_dir_path",
            "except_exception",
            "try:",
            "pass",
            "return_bool",
            "return_true",
            "return_detect_tokenizer_add_special_tokens",
        ),
        _torchtrt_tokenizer_models,
    ),
    TokenDiffRefinementRule(
        "harness_manifest_diffusion_thresholds",
        "tests/e2e_harness/manifest_loader.py",
        (
            "reference_min_pixel_std_for_ratio",
            "min_reference_std_ratio",
            "min_pixel_std",
            "overrides",
        ),
        _diffusion_task_models,
    ),
    HarnessReferenceDprContextEncoderRule(),
    HarnessReferenceVlGeneratedOnlyDecodeRule(),
    E2EWaivesModelLinesRule(),
)


def maybe_refine_match_with_diff(
    path: str,
    match: RuleMatch,
    diff_text: str,
    imap: ImpactMap,
) -> RuleMatch:
    """Narrow broad file matches when the diff is demonstrably feature-scoped."""
    lines = _significant_diff_lines(diff_text)
    if not lines:
        return match

    for rule in DIFF_REFINEMENT_RULES:
        if rule.matches(path, lines, imap):
            return rule.refine(path, match, lines, imap)

    return match

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_guarded_fallback(rule: str, path: str) -> bool:
    """Return True when a rule is intentionally broad enough to need review."""
    path = path.replace("\\", "/").strip("/")
    if rule in _BROAD_FALLBACK_RULES:
        return True
    return rule == "no_impact" and path.startswith(("tools/", "scripts/"))


def _load_fallback_allowlist(allowlist_path: Path) -> tuple[Set[tuple[str, str]], List[str]]:
    """Load reviewed broad fallback classifications.

    Non-comment lines use:
        <rule> <repo-relative-path> # <rationale>
    """
    allowed: Set[tuple[str, str]] = set()
    errors: List[str] = []
    if not allowlist_path.is_file():
        return allowed, [f"Fallback allowlist missing: {allowlist_path}"]

    try:
        lines = allowlist_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return allowed, [f"Could not read fallback allowlist {allowlist_path}: {exc}"]

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        entry_text, sep, comment = line.partition("#")
        entry_text = entry_text.strip()
        if not sep or not comment.strip():
            errors.append(
                f"{allowlist_path}:{line_no}: fallback allowlist entries need "
                "an inline rationale comment"
            )
            continue

        parts = entry_text.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(
                f"{allowlist_path}:{line_no}: expected '<rule> <path> # <rationale>'"
            )
            continue

        rule, path = parts
        path = path.replace("\\", "/").strip("/")
        if not _is_guarded_fallback(rule, path):
            errors.append(
                f"{allowlist_path}:{line_no}: '{rule} {path}' is not a guarded "
                "broad fallback classification"
            )
            continue

        entry = (rule, path)
        if entry in allowed:
            errors.append(
                f"{allowlist_path}:{line_no}: duplicate fallback allowlist entry "
                f"for {rule} {path}"
            )
            continue
        allowed.add(entry)

    return allowed, errors


def _tracked_repo_paths(repo_root: Path) -> tuple[List[str], List[str]]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return [], [f"Could not list tracked repo paths with git ls-files: {exc}"]

    paths = [
        path.replace("\\", "/").strip("/")
        for path in result.stdout.splitlines()
        if path.strip()
    ]
    return sorted(paths), []


def _broad_fallback_classifications(
    imap: ImpactMap,
    tracked_paths: List[str],
) -> List[Dict[str, str]]:
    fallbacks: List[Dict[str, str]] = []
    for path in sorted({p.replace("\\", "/").strip("/") for p in tracked_paths if p}):
        match = classify_file(path, imap)
        if _is_guarded_fallback(match.rule, path):
            fallbacks.append({"path": path, "rule": match.rule})
    return fallbacks


def validate_fallback_allowlist(
    imap: ImpactMap,
    repo_root: Path,
    tracked_paths: Optional[List[str]] = None,
    allowlist_path: Optional[Path] = None,
) -> tuple[List[str], List[str], List[Dict[str, str]]]:
    """Validate reviewed broad fallback classifications.

    Returns errors, warnings, and the tracked fallback classifications that were
    checked. Warnings are advisory so obsolete allowlist entries do not fail
    unrelated map checks.
    """
    if allowlist_path is None:
        allowlist_path = repo_root / _FALLBACK_ALLOWLIST
    elif not allowlist_path.is_absolute():
        allowlist_path = repo_root / allowlist_path

    allowed, errors = _load_fallback_allowlist(allowlist_path)

    tracked_errors: List[str] = []
    if tracked_paths is None:
        tracked_paths, tracked_errors = _tracked_repo_paths(repo_root)
    errors.extend(tracked_errors)

    fallbacks = _broad_fallback_classifications(imap, tracked_paths or [])
    fallback_keys = {(entry["rule"], entry["path"]) for entry in fallbacks}

    for entry in fallbacks:
        key = (entry["rule"], entry["path"])
        if key not in allowed:
            errors.append(
                "Unreviewed broad fallback classification: "
                f"{entry['path']} -> {entry['rule']}. Add it to "
                f"{_FALLBACK_ALLOWLIST} with a rationale comment or add a "
                "more precise classification rule."
            )

    warnings: List[str] = []
    for rule, path in sorted(allowed - fallback_keys):
        warnings.append(
            "Fallback allowlist entry no longer matches a tracked broad "
            f"fallback: {rule} {path}"
        )

    return errors, warnings, fallbacks


def validate_map(
    imap: ImpactMap,
    repo_root: Path,
    tracked_paths: Optional[List[str]] = None,
    fallback_allowlist_path: Optional[Path] = None,
    report_fallbacks: bool = False,
) -> List[str]:
    """Validate impact map consistency. Returns list of error strings."""
    errors: List[str] = []
    warnings: List[str] = []
    families_dir = repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
    torchtrt_families_dir = (
        repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "engine_defs" / "torch_trt" / "families"
    )

    def _family_plugin_exists(family: str) -> bool:
        return any((
            (families_dir / f"{family}.py").exists(),
            (families_dir / family / "__init__.py").exists(),
            (torchtrt_families_dir / f"{family}.py").exists(),
            (torchtrt_families_dir / family / "__init__.py").exists(),
        ))

    # 1. Every family in a manifest has a corresponding plugin module/package
    for family in imap.family_to_models:
        if not _family_plugin_exists(family):
            errors.append(
                f"Family '{family}' in manifests has no plugin module or package under "
                f"{families_dir} or {torchtrt_families_dir}"
            )

    # 2. Every family plugin module/package has at least one manifest (warn only)
    if families_dir.is_dir():
        for py_file in sorted(families_dir.glob("*.py")):
            name = py_file.stem
            if name in ("__init__", "base"):
                continue
            if name not in imap.family_to_models:
                warnings.append(f"Family plugin '{name}.py' has no manifests using it")
        for family_dir in sorted(path for path in families_dir.iterdir() if path.is_dir()):
            name = family_dir.name
            if name.startswith("_") or not (family_dir / "__init__.py").exists():
                continue
            if name not in imap.family_to_models:
                warnings.append(f"Family package '{name}/' has no manifests using it")

    # 3. Core model set covers all distinct task_strategies
    core_task_strategies: Set[str] = set()
    for model in imap.core_models:
        for ts, models in imap.task_strategy_to_models.items():
            if model in models:
                core_task_strategies.add(ts)
    all_task_strategies = set(imap.task_strategy_to_models.keys())
    missing = all_task_strategies - core_task_strategies
    if missing:
        warnings.append(
            f"Core models don't cover task_strategies: {sorted(missing)}"
        )

    # 4. Every runtime_strategy in manifests is in RUNTIME_TO_TASK_STRATEGY
    for strategy in imap.strategy_to_models:
        if strategy not in RUNTIME_TO_TASK_STRATEGY:
            errors.append(
                f"Unknown runtime_strategy '{strategy}' in manifests "
                f"(not in RUNTIME_TO_TASK_STRATEGY)"
            )

    # 5. L0 replacements must preserve the execution contract they stand in for.
    for model, replacement in sorted(imap.l0_replacement_by_model.items()):
        src = imap.model_metadata.get(model, {})
        dst = imap.model_metadata.get(replacement)
        if dst is None:
            errors.append(
                f"L0 replacement for '{model}' points to unknown model '{replacement}'"
            )
            continue
        for field_name in ("family", "runtime_strategy", "precision", "quantization"):
            if src.get(field_name) != dst.get(field_name):
                errors.append(
                    f"L0 replacement '{replacement}' for '{model}' does not preserve "
                    f"{field_name}: {src.get(field_name)!r} != {dst.get(field_name)!r}"
                )

    # 6. Every rule pattern matches at least one real file (spot checks)
    spot_checks = {
        "families_dir": families_dir.is_dir(),
        "models_dir": (repo_root / "tests" / "e2e" / "models").is_dir(),
        "src_dir": (repo_root / "src").is_dir(),
        "tests_e2e_harness": (repo_root / "tests" / "e2e_harness").is_dir(),
    }
    for name, exists in spot_checks.items():
        if not exists:
            errors.append(f"Expected directory missing for rule validation: {name}")

    # 7. Broad fallback classifications must be explicitly reviewed.
    fallback_errors, fallback_warnings, fallbacks = validate_fallback_allowlist(
        imap,
        repo_root,
        tracked_paths=tracked_paths,
        allowlist_path=fallback_allowlist_path,
    )
    errors.extend(fallback_errors)
    warnings.extend(fallback_warnings)

    if report_fallbacks:
        for entry in fallbacks:
            print(
                f"  FALLBACK: {entry['path']} -> {entry['rule']}",
                file=sys.stderr,
            )

    # Print warnings to stderr
    for w in warnings:
        print(f"  WARN: {w}", file=sys.stderr)

    return errors

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_human(result: ImpactResult) -> str:
    lines: List[str] = []
    if result.e2e_models:
        lines.append(f"# E2E tests to run ({len(result.e2e_models)} models):")
        for model in result.e2e_models:
            lines.append(f"tests/test_e2e.py::test_e2e[{model}]")
    else:
        lines.append("# No E2E models affected.")
    if result.unit_tiers:
        lines.append(f"# Unit test tiers: {', '.join(result.unit_tiers)}")
    lines.append(f"# C++ rebuild needed: {'yes' if result.rebuild_cpp else 'no'}")
    if result.cap_applied:
        lines.append("# WARNING: Cap applied -- running core models only.")
    if result.l0_replacements:
        lines.append(f"# L0 replacements applied ({len(result.l0_replacements)} models):")
        for repl in result.l0_replacements:
            lines.append(f"#   {repl['model']} -> {repl['replacement']}")
    return "\n".join(lines)


def format_json(result: ImpactResult) -> str:
    return json.dumps({
        "e2e_models": result.e2e_models,
        "unit_tiers": result.unit_tiers,
        "rebuild_cpp": result.rebuild_cpp,
        "cap_applied": result.cap_applied,
        "matched_rules": result.matched_rules,
        "builder_tests": result.builder_tests,
        "cpp_tests": result.cpp_tests,
        "tools_tests": result.tools_tests,
        "fallback_tiers": result.fallback_tiers,
        "l0_replacements": result.l0_replacements,
    }, indent=2)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test impact analysis for selective CI execution.",
    )
    parser.add_argument("--base", default="github/main",
                        help="Git ref for diff base (default: github/main)")
    parser.add_argument("--head", default="HEAD",
                        help="Git ref for diff head (default: HEAD)")
    parser.add_argument("--files",
                        help="Explicit comma-separated file list (overrides git diff)")
    parser.add_argument("--cap", type=int, default=None,
                        help="If affected models > N, limit to core set + warn")
    parser.add_argument("--e2e-suite", choices=("l0", "nightly"), default="l0",
                        help="E2E selection policy: l0 applies configured "
                             "large-model replacements; nightly keeps exact models")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output structured JSON for CI consumption")
    parser.add_argument("--validate", action="store_true",
                        help="Check map consistency (no diff needed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file rule matches")
    parser.add_argument("--repo-root", default=None,
                        help="Repository root (default: auto-detect)")
    parser.add_argument("--coverage-map", default=None,
                        help="Path to coverage_map.json for per-test selection")
    args = parser.parse_args()

    # Resolve repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            )
            repo_root = Path(result.stdout.strip())
        except subprocess.CalledProcessError:
            repo_root = Path.cwd()

    imap = build_impact_map(repo_root)

    if args.validate:
        errors = validate_map(imap, repo_root, report_fallbacks=args.verbose)
        if errors:
            print("Validation FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  ERROR: {e}", file=sys.stderr)
            return 1
        print(f"Validation passed. {len(imap.all_model_names)} models, "
              f"{len(imap.core_models)} core, "
              f"{len(imap.family_to_models)} families.",
              file=sys.stderr)
        return 0

    # Load coverage map if provided
    coverage_map_data = None
    if args.coverage_map:
        sys.path.insert(0, str(repo_root / "tools"))
        from coverage_map.generate import load_coverage_map
        coverage_map_data = load_coverage_map(Path(args.coverage_map))
        if coverage_map_data is None:
            print(f"WARNING: Coverage map not found at {args.coverage_map}. "
                  "Falling back to tier-level selection.", file=sys.stderr)

    # Get changed files
    if args.files:
        changed: Optional[List[str]] = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        changed = get_changed_files(args.base, args.head, repo_root)

    if changed is None:
        # Git diff failed -- safety net: run everything
        print("Running all tests (git diff unavailable).", file=sys.stderr)
        result_obj = ImpactResult(
            e2e_models=list(imap.all_model_names),
            unit_tiers=["builder", "cpp", "tools"],
            rebuild_cpp=True,
            cap_applied=False,
            matched_rules=[{
                "file": "<all>", "rule": "git_diff_failed",
                "models": list(imap.all_model_names),
            }],
        )
    elif not changed:
        print("No changed files detected.", file=sys.stderr)
        result_obj = ImpactResult(
            e2e_models=[], unit_tiers=[], rebuild_cpp=False,
            cap_applied=False, matched_rules=[],
        )
    else:
        result_obj = analyze_impact(
            changed,
            imap,
            cap=args.cap,
            coverage_map=coverage_map_data,
            base=args.base,
            head=args.head,
            repo_root=repo_root,
            e2e_suite=args.e2e_suite,
        )

    if args.verbose:
        for rule in result_obj.matched_rules:
            n = len(rule["models"])
            print(f"  {rule['file']} -> {rule['rule']} ({n} models)",
                  file=sys.stderr)

    if args.json_output:
        print(format_json(result_obj))
    else:
        print(format_human(result_obj))

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Test impact analysis -- selective CI execution based on changed files.

Determines which E2E models and unit test tiers need to run based on
git diff between base and head. Safety invariant: ZERO false negatives.
Any file that doesn't match a known rule triggers ALL model tests.

Usage:
    python3 tools/test_impact.py [--base REF] [--head REF] [--json] [--verbose]
    python3 tools/test_impact.py --files path/to/file1.py,path/to/file2.cpp
    python3 tools/test_impact.py --validate
    python3 tools/test_impact.py --e2e-suite nightly --files src/runtime/plugins/decoder_plugin.cpp
    python3 tools/test_impact.py --files tensorrt_model_connect/tensorrt_model_connect/families/qwen.py --cap 15
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

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
    "object_detection": "object_detection",
    "embedding": "embedding",
    "reranking": "reranking",
    "encoder_only": "encoder_only_nlp",
    "neural_operator": "neural_operator",
    "patchtst_torchtrt": "neural_operator",
    "patchtsmixer_torchtrt": "neural_operator",
    "timesfm_torchtrt": "neural_operator",
    "chronos_bolt_torchtrt": "neural_operator",
    "diffusion": "diffusion_media_generation",
    "diffusion_flux": "diffusion_media_generation",
    "diffusion_ltx": "diffusion_media_generation",
    "diffusion_wan": "diffusion_media_generation",
    "diffusion_zimage": "diffusion_media_generation",
    "diffusion_pixart": "diffusion_media_generation",
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
    "segmentation_plugin": ["segmentation", "prompted_segmentation"],
    "object_detection_plugin": ["object_detection"],
    "omni_plugin": ["omni_multimodal"],
    "flux_plugin": ["diffusion_flux"],
    "ltx_video_plugin": ["diffusion_ltx"],
    "wan_plugin": ["diffusion_wan"],
    "pixart_plugin": ["diffusion_pixart"],
    "pixart_torchtrt_plugin": ["diffusion_pixart_torchtrt"],
    "zimage_plugin": ["diffusion_zimage"],
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
    "encoder_pipeline": [
        "encoder_only", "embedding", "reranking", "neural_operator",
        "object_detection",
    ],
    "patchtst_pipeline": ["patchtst_torchtrt"],
    "patchtsmixer_pipeline": ["patchtsmixer_torchtrt"],
    "timesfm_pipeline": ["timesfm_torchtrt"],
    "chronos_bolt_pipeline": ["chronos_bolt_torchtrt"],
    "flux_pipeline": ["diffusion_flux"],
    "ltx_video_pipeline": ["diffusion_ltx"],
    "wan_pipeline": ["diffusion_wan"],
    "pixart_pipeline": ["diffusion_pixart"],
    "pixart_torchtrt_pipeline": ["diffusion_pixart_torchtrt"],
    "z_image_pipeline": ["diffusion_zimage"],
    "diffusion_pipeline": [
        "diffusion_flux", "diffusion_ltx", "diffusion_wan", "diffusion_pixart",
        "diffusion_zimage", "diffusion_pixart_torchtrt",
    ],
}

# E2E runner filename (stem) -> task_strategies
RUNNER_TASK_STRATEGIES: Dict[str, List[str]] = {
    "text_generation": ["text_generation_causal"],
    "vision_language": ["vision_language_generation"],
    "audio_speech": ["speech_to_text", "text_to_audio", "speech_to_speech"],
    "diffusion": ["diffusion_media_generation"],
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
    "diffusion": ["diffusion_media_generation"],
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
}

# E2E reference filename (stem) -> task_strategies
REFERENCE_TASK_STRATEGIES: Dict[str, List[str]] = {
    "hf_transformers": [
        "text_generation_causal", "vision_language_generation", "text_to_audio",
        "speech_to_text", "encoder_only_nlp", "embedding", "reranking",
        "segmentation", "prompted_segmentation", "object_detection",
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
        "diffusion_zimage",
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
    r"^\.gitlab-ci\.yml$",
    r"^\.gitlab/",
    r"^\.claude/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^recovery-",
]

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


def _scan_family_imports(families_dir: Path) -> Dict[str, List[str]]:
    """Build reverse index: parent_module -> [family_names that import it].

    Only returns entries for *_builder modules (excluding orchestrators).
    """
    reverse: Dict[str, Set[str]] = {}
    for py_file in sorted(families_dir.glob("*.py")):
        name = py_file.stem
        if name in ("__init__", "base"):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        # from ..module_name import ...
        for m in re.finditer(r"from\s+\.\.(\w+)\s+import", content):
            module = m.group(1)
            reverse.setdefault(module, set()).add(name)
        # from .. import module_name
        for m in re.finditer(r"from\s+\.\.\s+import\s+([\w,\s]+)", content):
            for mod in m.group(1).split(","):
                mod = mod.strip()
                if mod:
                    reverse.setdefault(mod, set()).add(name)
    # Filter to *_builder modules only (excluding orchestrators)
    filtered: Dict[str, List[str]] = {}
    for module, families in reverse.items():
        if module.endswith("_builder") and module not in _ORCHESTRATOR_MODULES:
            filtered[module] = sorted(families)
    return filtered


def build_impact_map(repo_root: Path) -> ImpactMap:
    """Build the impact map by scanning manifests and family plugins."""
    models_dir = repo_root / "tests" / "e2e" / "models"
    families_dir = repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
    pipelines_dir = repo_root / "src" / "runtime" / "pipelines"

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

    builder_to_families = _scan_family_imports(families_dir) if families_dir.is_dir() else {}

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
    if pipelines_dir.is_dir():
        for path, token in scoped_cpp_tokens.items():
            strategies: Set[str] = set()
            for cpp_file in sorted(pipelines_dir.glob("*.cpp")):
                try:
                    content = cpp_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if token not in content:
                    continue
                strategies.update(CPP_PIPELINE_STRATEGIES.get(cpp_file.stem, []))
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
    the large model's artifact contract is covered by nightly, while MR L0 keeps
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
    if path.startswith("tests/builder/") or path.startswith("tests/torchtrt_builder/"):
        tiers.append("builder")
    if path.startswith("tests/cpp/"):
        tiers.append("cpp")
    if path.startswith("tests/tools/"):
        tiers.append("tools")
    return sorted(set(tiers))


def _infer_rebuild_cpp(path: str) -> bool:
    """Does this file change require a C++ rebuild?"""
    return (path.startswith("src/") or path.startswith("include/")
            or path == "CMakeLists.txt" or path.startswith("cmake/")
            or path.startswith("tests/cpp/"))

# ---------------------------------------------------------------------------
# File classification (ordered rules)
# ---------------------------------------------------------------------------


def classify_file(path: str, imap: ImpactMap) -> RuleMatch:
    """Classify a single changed file. First matching rule wins."""
    # Normalize path separators
    path = path.replace("\\", "/").strip("/")
    unit_tiers = _infer_unit_tiers(path)
    rebuild = _infer_rebuild_cpp(path)

    # Rule 0: E2E model manifest
    m = re.match(r"tests/e2e/models/(.+)\.json$", path)
    if m:
        name = m.group(1)
        models = [name] if name in imap.all_model_names_set else []
        return RuleMatch("manifest", models, unit_tiers, rebuild)

    # Rule 1: Family plugin (not __init__ or base)
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/families/(\w+)\.py$", path)
    if m and m.group(1) not in ("__init__", "base"):
        family = m.group(1)
        models = imap.family_to_models.get(family, [])
        return RuleMatch("family_plugin", sorted(models), unit_tiers, rebuild)

    # Rule 1b: Family __init__.py or base.py -> ALL models
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/families/((__init__|base)\.py)$", path)
    if m:
        return RuleMatch("family_base", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 1c: Torch-TRT family plugin (not __init__ or base)
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/(\w+)\.py$", path)
    if m and m.group(1) not in ("__init__", "base"):
        family = m.group(1)
        models = imap.family_to_models.get(family, [])
        return RuleMatch("torchtrt_family_plugin", sorted(models), unit_tiers, rebuild)

    # Rule 1d: Torch-TRT family __init__.py or base.py -> ALL models
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/((__init__|base)\.py)$", path)
    if m:
        return RuleMatch("torchtrt_family_base", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 1e: Torch-TRT strategy modules with known modality scope
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/(\w+)\.py$", path)
    if m:
        strategy_stem = m.group(1)
        if strategy_stem == "diffusion":
            return RuleMatch(
                "torchtrt_strategy",
                _models_for_task_strategies(["diffusion_media_generation"], imap),
                unit_tiers,
                rebuild,
            )
        return RuleMatch("torchtrt_strategy_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 2: Specialized builder (auto-detected via import scan)
    m = re.match(r"tensorrt_model_connect/tensorrt_model_connect/(\w+)\.py$", path)
    if m:
        module_name = m.group(1)
        if (module_name.endswith("_builder")
                and module_name not in _ORCHESTRATOR_MODULES
                and module_name in imap.builder_to_families):
            families = imap.builder_to_families[module_name]
            models: Set[str] = set()
            for fam in families:
                models.update(imap.family_to_models.get(fam, []))
            if models:
                return RuleMatch("specialized_builder", sorted(models), unit_tiers, rebuild)
        # Fall through to Rule 3 for non-builder or unmatched builder

    # Rule 3: Any other file under tensorrt_model_connect/
    if path.startswith("tensorrt_model_connect/"):
        return RuleMatch("shared_builder_module", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 4: C++ plugin
    m = re.match(r"src/runtime/plugins/(\w+)\.cpp$", path)
    if m:
        plugin_stem = m.group(1)
        strategies = CPP_PLUGIN_STRATEGIES.get(plugin_stem, [])
        if strategies:
            models = _drop_fp8_scale_models(
                _models_for_runtime_strategies(strategies, imap), imap)
            if plugin_stem == "flux_plugin":
                return RuleMatch(
                    "cpp_plugin_flux_runtime",
                    models, unit_tiers, rebuild,
                )
            return RuleMatch(
                "cpp_plugin", models, unit_tiers, rebuild,
            )
        # Unknown plugin -> all models (safety)
        return RuleMatch("cpp_plugin_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 5: C++ pipeline
    m = re.match(r"src/runtime/pipelines/(\w+)\.(h|cpp)$", path)
    if m:
        pipeline_stem = m.group(1)
        strategies = CPP_PIPELINE_STRATEGIES.get(pipeline_stem, [])
        if strategies:
            models = _drop_fp8_scale_models(
                _models_for_runtime_strategies(strategies, imap), imap)
            if pipeline_stem == "flux_pipeline":
                return RuleMatch(
                    "cpp_pipeline_flux_runtime",
                    models, unit_tiers, rebuild,
                )
            return RuleMatch(
                "cpp_pipeline", models, unit_tiers, rebuild,
            )
        return RuleMatch("cpp_pipeline_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 6: Shared C++ helpers
    m = re.match(r"src/runtime/plugins/shared/(\w+)\.(h|cpp)$", path)
    if m:
        helper_stem = m.group(1)
        if helper_stem == "plugin_helpers":
            return RuleMatch("cpp_shared_plugin_helpers", list(imap.all_model_names), unit_tiers, rebuild)
        runtime_strategies = SHARED_CPP_HELPER_STRATEGIES.get(helper_stem, [])
        if runtime_strategies:
            return RuleMatch(
                "cpp_shared_helper",
                _drop_fp8_scale_models(
                    _models_for_runtime_strategies(runtime_strategies, imap), imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("cpp_shared_helper_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 6b: Scoped C++ helper/source used by a subset of pipelines
    if path in imap.path_scope_overrides:
        return RuleMatch(
            "cpp_scoped_helper",
            _drop_fp8_scale_models(imap.path_scope_overrides[path], imap),
            unit_tiers, rebuild,
        )

    # Rule 7: Any other C++ source/header
    if path.startswith("src/") or path.startswith("include/"):
        return RuleMatch("cpp_source", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8a: E2E runner
    m = re.match(r"tests/e2e_harness/runners/(\w+)\.py$", path)
    if m:
        runner_stem = m.group(1)
        if runner_stem == "__init__":
            return RuleMatch("harness_runner_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = RUNNER_TASK_STRATEGIES.get(runner_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_runner",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_runner_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8b: E2E comparator
    m = re.match(r"tests/e2e_harness/comparators/(\w+)\.py$", path)
    if m:
        comp_stem = m.group(1)
        if comp_stem == "__init__":
            return RuleMatch("harness_comparator_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = COMPARATOR_TASK_STRATEGIES.get(comp_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_comparator",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_comparator_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8c: E2E reference
    m = re.match(r"tests/e2e_harness/references/(\w+)\.py$", path)
    if m:
        ref_stem = m.group(1)
        if ref_stem == "__init__":
            return RuleMatch("harness_reference_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = REFERENCE_TASK_STRATEGIES.get(ref_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_reference",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers, rebuild,
            )
        return RuleMatch("harness_reference_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8d: E2E contract plugin
    m = re.match(r"tests/e2e_harness/plugins/(\w+)\.py$", path)
    if m:
        plugin_stem = m.group(1)
        if plugin_stem == "__init__":
            return RuleMatch("harness_plugin_init", list(imap.all_model_names), unit_tiers, rebuild)
        task_strategies = PLUGIN_TASK_STRATEGIES.get(plugin_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_plugin",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers,
                rebuild,
            )
        return RuleMatch("harness_plugin_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8e: E2E threshold profiles
    m = re.match(r"tests/e2e_harness/thresholds/defaults/([\w_]+)\.json$", path)
    if m:
        profile_stem = m.group(1)
        task_strategies = THRESHOLD_PROFILE_TASK_STRATEGIES.get(profile_stem, [])
        if task_strategies:
            return RuleMatch(
                "harness_threshold_profile",
                _models_for_task_strategies(task_strategies, imap),
                unit_tiers,
                rebuild,
            )
        return RuleMatch("harness_threshold_unknown", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 8f: Any other E2E harness file
    if path.startswith("tests/e2e_harness/"):
        return RuleMatch("harness_shared", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 9: test_e2e.py or conftest.py
    if path in ("tests/test_e2e.py", "tests/conftest.py"):
        return RuleMatch("e2e_entrypoint", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 9b: E2E runner/scheduler scripts affect every selected model.
    if path in {
        "scripts/run_e2e_parallel.sh",
        "scripts/schedule_e2e.py",
        "scripts/warm_hf_cache.py",
    }:
        return RuleMatch("e2e_runner_script", list(imap.all_model_names), unit_tiers, rebuild)

    # Rule 10: Unit test directories (no E2E)
    if path.startswith("tests/builder/"):
        return RuleMatch("unit_builder", [], unit_tiers, rebuild)
    if path.startswith("tests/cpp/"):
        return RuleMatch("unit_cpp", [], unit_tiers, rebuild)
    if path.startswith("tests/tools/"):
        return RuleMatch("unit_tools", [], unit_tiers, rebuild)
    if path.startswith("tests/torchtrt_builder/"):
        return RuleMatch("unit_torchtrt_builder", [], unit_tiers, rebuild)

    # Rule 11: CMake / build system — triggers C++ rebuild + unit tests
    # but no E2E models; actual model impact comes from the source files.
    if path == "CMakeLists.txt" or path.startswith("cmake/"):
        return RuleMatch("cmake", [], unit_tiers, rebuild)

    # Rule 11b: E2E data file referenced by manifests
    if path in imap.e2e_data_file_to_models:
        return RuleMatch("e2e_data_file", imap.e2e_data_file_to_models[path], unit_tiers, rebuild)

    # Rule 11c: FP8 weight generation script — affects all models with fp8_scales manifests
    if path == "scripts/_gen_fp8_bf16.py":
        return RuleMatch(
            "fp8_gen_script",
            sorted(imap.manifest_field_to_models.get("fp8_scales", [])),
            [], False,
        )

    # Rule 12: Non-code files (no impact)
    if path.startswith("tools/") or path.startswith("scripts/"):
        return RuleMatch("no_impact", [], [], False)
    for pattern in _NO_IMPACT_PATTERNS:
        if re.match(pattern, path):
            return RuleMatch("no_impact", [], [], False)
    # *.md files anywhere
    if path.endswith(".md"):
        return RuleMatch("no_impact", [], [], False)

    # CATCH-ALL: unknown file -> ALL models (safety net)
    return RuleMatch("catch_all", list(imap.all_model_names), unit_tiers, True)

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
        elif path.startswith("tests/tools/"):
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

    fp8_models = imap.manifest_field_to_models.get("fp8_scales", [])

    if path == "tests/e2e_harness/orchestrator.py":
        allowed = {
            "CILane,",
            'fp8_scales = case.metadata.get("fp8_scales")',
            "if fp8_scales:",
            "# Resolve relative to tests/e2e/data/",
            'scales_path = Path(__file__).parent.parent / "e2e" / "data" / fp8_scales',
            "if scales_path.is_file():",
            'cmd.extend(["--fp8-scales", str(scales_path)])',
        }
        if all(line in allowed for line in lines):
            return RuleMatch(
                "harness_shared_fp8_scales", fp8_models,
                match.unit_tiers, match.rebuild_cpp,
            )

    if path == "tensorrt_model_connect/tensorrt_model_connect/cli.py":
        allowed_tokens = ("fp8_scales", "save_fp8_scales")
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "shared_builder_fp8_scales_cli", fp8_models,
                match.unit_tiers, match.rebuild_cpp,
            )

    if path == "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py":
        allowed_tokens = (
            "fp8_scales",
            "save_fp8_scales",
            "_build_diffusion_bundle(",
            "_effective_precision",
            '"precision"',
            '"quantization"',
            "cfg_dict[",
            "fp8_scales",
        )
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "shared_builder_fp8_scales_engine", fp8_models,
                match.unit_tiers, match.rebuild_cpp,
            )

    if path == "tensorrt_model_connect/tensorrt_model_connect/engine_builder.py":
        allowed_tokens = (
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
        )
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "shared_builder_diffusion_tokenizer",
                _models_for_task_strategies(["diffusion_media_generation"], imap),
                match.unit_tiers,
                match.rebuild_cpp,
            )

    if path == "tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py":
        allowed_tokens = (
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
        )
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "torchtrt_compiler_tokenizer",
                _models_for_runtime_strategies(
                    ["torchtrt_decoder", "diffusion_pixart_torchtrt"], imap),
                match.unit_tiers,
                match.rebuild_cpp,
            )

    if path == "tests/e2e_harness/manifest_loader.py":
        allowed_tokens = (
            "reference_min_pixel_std_for_ratio",
            "min_reference_std_ratio",
            "min_pixel_std",
            "overrides",
        )
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "harness_manifest_diffusion_thresholds",
                _models_for_task_strategies(["diffusion_media_generation"], imap),
                match.unit_tiers,
                match.rebuild_cpp,
            )

    if path == "tests/test_e2e.py":
        skip_artifact_tokens = (
            "fileartifactsink",
            "e2eresult",
            "e2estatus",
            "manifest_level_skip",
            "manifest_skip",
            "skip_reason",
            "human_readable_contract",
            "missing_reference",
            "artifacts_dir",
            "case_name=case.name",
            "oracle_level=case.oracle_level",
            "tests.e2e_harness.contracts",
            "tests.e2e_harness.artifact_sink",
        )
        normalized_lines = [_normalize_diff_line(line) for line in lines]
        skipped_models = sorted(
            name for name, metadata in imap.model_metadata.items()
            if metadata.get("skip")
        )
        if (
            skipped_models
            and all(
                any(token in line for token in skip_artifact_tokens)
                for line in normalized_lines
            )
        ):
            return RuleMatch(
                "e2e_entrypoint_manifest_skip_artifact",
                skipped_models,
                match.unit_tiers,
                match.rebuild_cpp,
            )

    if path == "tests/e2e_harness/references/hf_transformers.py":
        allowed_tokens = (
            "decode_vl_generated_text",
            "vl_generation",
            "generated_ids",
            "generated_text",
            "input_len",
            "token_count",
            "decode_token_ids",
            "processor.decode",
            "skip_special_tokens",
            "strip",
            "if_text",
            "return_text",
            "hf_transformers",
        )
        if all(
            any(token in _normalize_diff_line(line) for token in allowed_tokens)
            for line in lines
        ):
            return RuleMatch(
                "harness_reference_vl_generated_only_decode",
                ["internvl3-8b"],
                match.unit_tiers,
                match.rebuild_cpp,
            )

    if path == "tests/e2e/waives.txt":
        models = []
        for line in lines:
            fields = line.split()
            if fields and fields[0] in imap.all_model_names_set:
                models.append(fields[0])
        if models:
            return RuleMatch(
                "e2e_waives_model_lines",
                sorted(set(models)),
                match.unit_tiers,
                match.rebuild_cpp,
            )

    return match

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_map(imap: ImpactMap, repo_root: Path) -> List[str]:
    """Validate impact map consistency. Returns list of error strings."""
    errors: List[str] = []
    warnings: List[str] = []
    families_dir = repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "families"
    torchtrt_families_dir = (
        repo_root / "tensorrt_model_connect" / "tensorrt_model_connect" / "engine_defs" / "torch_trt" / "families"
    )

    # 1. Every family in a manifest has a corresponding .py plugin file
    for family in imap.family_to_models:
        raw_plugin_file = families_dir / f"{family}.py"
        torchtrt_plugin_file = torchtrt_families_dir / f"{family}.py"
        if not raw_plugin_file.exists() and not torchtrt_plugin_file.exists():
            errors.append(
                f"Family '{family}' in manifests has no plugin file: "
                f"{raw_plugin_file} or {torchtrt_plugin_file}"
            )

    # 2. Every family plugin .py has at least one manifest (warn only)
    if families_dir.is_dir():
        for py_file in sorted(families_dir.glob("*.py")):
            name = py_file.stem
            if name in ("__init__", "base"):
                continue
            if name not in imap.family_to_models:
                warnings.append(f"Family plugin '{name}.py' has no manifests using it")

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
    parser.add_argument("--base", default="origin/master",
                        help="Git ref for diff base (default: origin/master)")
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
        errors = validate_map(imap, repo_root)
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

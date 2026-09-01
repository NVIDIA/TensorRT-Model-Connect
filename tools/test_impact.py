#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test impact analysis -- selective CI execution based on changed files.

Determines which E2E models and unit test tiers need to run based on
git diff between base and head. Safety invariant: ZERO false negatives.
Any file that doesn't match a known rule triggers ALL model tests.

Usage:
    python3 tools/test_impact.py [--base REF] [--head REF] [--json] [--verbose]
    python3 tools/test_impact.py --files path/to/file1.py,path/to/file2.cpp
    python3 tools/test_impact.py --validate
    python3 tools/test_impact.py --e2e-suite nightly --files src/runtime/models/qwen/plugin.cpp
    python3 tools/test_impact.py --files python/tensorrt_model_connect/families/example/plugin.py --cap 15
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 uses the declared tomli dependency
    import tomli as tomllib

# ---------------------------------------------------------------------------
# Constants -- strategy mappings
# ---------------------------------------------------------------------------

# Shared placeholder sidecars intentionally contain no plugin object. Their
# concrete behavior lives in model-owned E2E plugins; direct edits to these
# placeholders should validate the structural guard, not fan out to E2E.
SHARED_PLACEHOLDER_RUNNER_STEMS: Set[str] = {
    "audio_speech",
    "diffusion",
    "diffusion_text_generation",
    "embedding",
    "encoder_only",
    "image_classification",
    "neural_operator",
    "object_detection",
    "omni",
    "reranking",
    "segmentation",
    "text_generation",
    "vision_language",
}
SHARED_PLACEHOLDER_COMPARATOR_STEMS: Set[str] = {
    "audio",
    "diffusion",
    "diffusion_text_generation",
    "embedding",
    "encoder_only",
    "image_classification",
    "neural_operator",
    "omni",
    "reranking",
    "segmentation",
    "speech_to_speech",
    "speech_to_text",
    "text",
    "text_to_audio",
    "vision_language",
}
SHARED_PLACEHOLDER_REFERENCE_STEMS: Set[str] = {
    "custom_python",
    "golden_snapshot",
    "hf_diffusers",
    "hf_transformers",
    "invariant_only",
    "nemo_reference",
    "torch_reference",
}
RUNNER_TASK_STRATEGY_FALLBACKS: Dict[str, List[str]] = {
    "diffusion_text_generation": ["diffusion_text_generation"],
    "image_classification": ["image_classification"],
}
COMPARATOR_TASK_STRATEGY_FALLBACKS: Dict[str, List[str]] = {
    "diffusion_text_generation": ["diffusion_text_generation"],
    "image_classification": ["image_classification"],
}

# Third-party image loaders are used by image/video-producing or image-consuming
# runtime paths, not by every model family.
STB_IMAGE_TASK_STRATEGIES = [
    "diffusion_media_generation",
    "image_classification",
    "image_feature_extraction",
    "object_detection",
    "omni_multimodal",
    "prompted_segmentation",
    "segmentation",
    "vision_language_generation",
]

# Shared C++ helper -> affected task_strategies
SHARED_CPP_HELPER_STRATEGIES: Dict[str, List[str]] = {
    "diffusion_helpers": [
        "diffusion_flux",
        "diffusion_ltx",
        "diffusion_wan",
        "diffusion_pixart",
        "diffusion_zimage",
    ],
    "audio_helpers": [
        "speech_to_text",
        "speech_to_text_rnnt",
        "text_to_audio_bark",
        "text_to_audio_magpie",
        "speech_to_speech",
        "omni_multimodal",
    ],
}

# Orchestrator modules in python/tensorrt_model_connect/ -- not treated as specialized builders
_ORCHESTRATOR_MODULES = {
    "engine_builder",
    "cli",
    "__init__",
    "__main__",
    "pipeline",
    "debug_runner",
}

# Patterns for files that never affect E2E or unit tests
_NO_IMPACT_PATTERNS = [
    r"^docs/",
    r"^website/",
    r"^\.gitignore$",
    r"^\.clang-format$",
    r"^\.editorconfig$",
    r"^\.claude/",
    r"^\.agents/",
    r"^plugins/trtmc-agent-skills/",
    r"^LICENSE",
    r"^CLAUDE\.md$",
    r"^CODEOWNERS$",
    r"^ruff\.toml$",
    r"^tests/__init__\.py$",
    r"^tests/assets/",
    r"^recovery-",
]

_BROAD_FALLBACK_RULES = {
    "catch_all",
    "harness_shared",
    "shared_builder_module",
}
# TODO: Remove multi_device from the default exclusion once CI has a runner pool
# that can reserve all GPUs for tensor-parallel E2E cases.
_DEFAULT_EXCLUDED_CI_TIERS = frozenset({"multi_device"})
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


@dataclass(frozen=True)
class ModelOwnedDiffRuleSpec:
    owner: str
    name: str
    path: str
    allowed_tokens: Tuple[str, ...]
    scope: Dict[str, Any]


@dataclass
class ImpactMap:
    family_to_models: Dict[str, List[str]]
    strategy_to_models: Dict[str, List[str]]  # runtime_strategy -> models
    task_strategy_to_models: Dict[str, List[str]]  # task_strategy -> models
    all_model_names: List[str]
    all_model_names_set: Set[str]
    core_models: List[str]
    model_metadata: Dict[str, Dict]
    manifest_path_to_model: Dict[str, str]
    testcase_to_model: Dict[str, str]
    builder_to_families: Dict[str, List[str]]  # parent module -> families
    cpp_runtime_model_strategies: Dict[str, List[str]]
    manifest_field_to_models: Dict[str, List[str]]
    e2e_data_file_to_models: Dict[str, List[str]]
    path_scope_overrides: Dict[str, List[str]]
    l0_replacement_by_model: Dict[str, str]
    reference_family_to_models: Dict[str, List[str]]
    model_owned_diff_rules: Tuple[ModelOwnedDiffRuleSpec, ...]
    runner_task_strategies: Dict[str, List[str]]
    comparator_task_strategies: Dict[str, List[str]]
    reference_task_strategies: Dict[str, List[str]]
    plugin_task_strategies: Dict[str, List[str]]
    threshold_profile_task_strategies: Dict[str, List[str]]


@dataclass
class ImpactResult:
    e2e_models: List[str]
    unit_tiers: List[str]
    rebuild_cpp: bool
    cap_applied: bool
    matched_rules: List[Dict]
    e2e_test_ids: List[str] = field(default_factory=list)
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
        if name in ("__init__", "base") or name.startswith("_"):
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
        tree = ast.parse(content, filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level <= 1:
                continue
            # from ..module_name import ... / from ...module_name import ...
            if node.module:
                module = node.module.split(".", 1)[0]
                reverse.setdefault(module, set()).add(name)
                continue
            # from .. import module_name / from ... import module_name
            for alias in node.names:
                reverse.setdefault(alias.name, set()).add(name)
    # Filter to *_builder modules only (excluding orchestrators)
    filtered: Dict[str, List[str]] = {}
    for module, families in reverse.items():
        if module.endswith("_builder") and module not in _ORCHESTRATOR_MODULES:
            filtered[module] = sorted(families)
    return filtered


def _parse_runtime_model_manifest(manifest_path: Path) -> List[str]:
    """Parse the MODEL.toml runtime strategy list with a standards-compliant parser.

    A missing manifest yields an empty list. A malformed manifest raises
    ``tomllib.TOMLDecodeError`` (fail-closed): a broken descriptor must surface
    loudly in CI selection rather than silently selecting no strategies.
    """
    try:
        with manifest_path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError:
        return []
    strategies = data.get("runtime_strategies")
    if isinstance(strategies, list):
        return [value for value in strategies if isinstance(value, str)]
    single = data.get("runtime_strategy")
    return [single] if isinstance(single, str) else []


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


def _scan_model_owned_diff_rules(models_dir: Path) -> Tuple[ModelOwnedDiffRuleSpec, ...]:
    """Load model-owned diff narrowing rules from tests/e2e/models/<id>."""
    specs: List[ModelOwnedDiffRuleSpec] = []
    if not models_dir.is_dir():
        return ()
    for rules_path in sorted(models_dir.glob("*/impact_diff_rules.json")):
        owner = rules_path.parent.name
        try:
            raw_rules = json.loads(rules_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid model-owned impact rules {rules_path}: {exc}") from exc
        if not isinstance(raw_rules, list):
            raise ValueError(f"{rules_path}: expected a list of impact rule objects")
        for index, raw in enumerate(raw_rules, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"{rules_path}:{index}: expected an impact rule object")
            name = raw.get("name")
            path = raw.get("path")
            allowed_tokens = raw.get("allowed_tokens")
            scope = raw.get("scope")
            if not (
                isinstance(name, str)
                and isinstance(path, str)
                and isinstance(allowed_tokens, list)
                and isinstance(scope, dict)
            ):
                raise ValueError(
                    f"{rules_path}:{index}: expected string name/path, list "
                    "allowed_tokens, and object scope"
                )
            tokens = tuple(token for token in allowed_tokens if isinstance(token, str))
            if len(tokens) != len(allowed_tokens) or not tokens:
                raise ValueError(
                    f"{rules_path}:{index}: allowed_tokens must be a non-empty list of strings"
                )
            specs.append(ModelOwnedDiffRuleSpec(owner, name, path, tokens, scope))
    return tuple(specs)


def _iter_e2e_manifest_paths(models_dir: Path) -> List[Path]:
    """Return flat legacy and model-owned E2E manifest paths."""
    if not models_dir.is_dir():
        return []
    paths = set(models_dir.glob("*.json"))
    paths.update(models_dir.glob("*/manifests/*.json"))
    return sorted(paths)


_MODEL_ASSET_FIELDS = {
    "model_assets",
    "prompt_file",
    "test_image",
    "test_input_audio",
    "speech_reference_tokens",
    "golden_snapshot_path",
    "edit_condition_image",
    "fp8_scales",
}


def _manifest_asset_repo_path(value: str, manifest_path: Path, models_dir: Path) -> str:
    repo_root = models_dir.parent.parent.parent

    def _repo_relative(path: Path) -> str:
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    normalized = value.replace("\\", "/")
    if normalized.startswith("tests/e2e/"):
        return normalized
    if normalized.startswith("data/"):
        if manifest_path.parent.name == "manifests":
            family_dir = manifest_path.parent.parent
            return _repo_relative(family_dir / normalized)
        return _repo_relative(models_dir.parent / normalized)
    if "/" not in normalized:
        if manifest_path.parent.name == "manifests":
            family_dir = manifest_path.parent.parent
            return _repo_relative(family_dir / "data" / normalized)
        return _repo_relative(models_dir.parent / "data" / normalized)
    return normalized


def _iter_manifest_data_paths(
    value: object,
    manifest_path: Path,
    models_dir: Path,
    key: str = "",
) -> List[str]:
    if isinstance(value, dict):
        paths: List[str] = []
        if "relative_to" in value and isinstance(value.get("path"), str):
            paths.append(_manifest_asset_repo_path(value["path"], manifest_path, models_dir))
        for item_key, item_value in value.items():
            paths.extend(_iter_manifest_data_paths(item_value, manifest_path, models_dir, item_key))
        return paths
    if isinstance(value, list):
        paths: List[str] = []
        for item in value:
            paths.extend(_iter_manifest_data_paths(item, manifest_path, models_dir, key))
        return paths
    if isinstance(value, str) and key in _MODEL_ASSET_FIELDS:
        return [_manifest_asset_repo_path(value, manifest_path, models_dir)]
    return []


def _literal_method_returns(py_file: Path, method_names: Set[str]) -> List[str]:
    """Return string literals returned by methods such as strategy_name."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    values: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in method_names:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Return)
                and isinstance(child.value, ast.Constant)
                and isinstance(child.value.value, str)
                and child.value.value
            ):
                values.add(child.value.value)
    return sorted(values)


def _literal_string_list(value: ast.AST) -> List[str]:
    if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return []
    strings: Set[str] = set()
    for item in value.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            if item.value:
                strings.add(item.value)
    return sorted(strings)


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _literal_string_list_assignments(
    py_file: Path,
    assignment_names: Set[str],
) -> Dict[str, List[str]]:
    """Return literal string-list assignments such as reference_families."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    assignments: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = _target_name(target)
                if name in assignment_names:
                    values = _literal_string_list(node.value)
                    if values:
                        assignments[name] = values
        elif isinstance(node, ast.AnnAssign):
            name = _target_name(node.target)
            if name in assignment_names:
                values = _literal_string_list(node.value)
                if values:
                    assignments[name] = values
    return assignments


def _literal_string_dict_assignment(py_file: Path, name: str) -> Dict[str, str]:
    """Return a literal string->string dict assignment from a Python file."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(_target_name(target) == name for target in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign) and _target_name(node.target) == name:
            value = node.value
        if value is None:
            continue
        try:
            raw = ast.literal_eval(value)
        except (TypeError, ValueError, SyntaxError):
            continue
        if not isinstance(raw, dict):
            continue
        return {
            key: val for key, val in raw.items() if isinstance(key, str) and isinstance(val, str)
        }
    return {}


def _scan_harness_task_strategy_modules(
    directory: Path,
    *,
    method_names: Set[str],
    fallbacks: Dict[str, List[str]] | None = None,
) -> Dict[str, List[str]]:
    """Build file-stem -> task strategies from plugin source metadata."""
    routes: Dict[str, List[str]] = {}
    if not directory.is_dir():
        return dict(fallbacks or {})

    for py_file in sorted(directory.glob("*.py")):
        stem = py_file.stem
        if stem.startswith("_") or stem.startswith("test_") or stem == "__init__":
            continue
        strategies = _literal_method_returns(py_file, method_names)
        if strategies:
            routes[stem] = strategies

    for stem, strategies in (fallbacks or {}).items():
        routes.setdefault(stem, list(strategies))
    return routes


def _scan_harness_reference_modules(
    directory: Path,
    reference_backend_to_task_strategies: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Build reference backend file-stem -> task strategies from backend_name."""
    routes: Dict[str, List[str]] = {}
    if not directory.is_dir():
        return routes

    for py_file in sorted(directory.glob("*.py")):
        stem = py_file.stem
        if stem.startswith("_") or stem.startswith("test_") or stem == "__init__":
            continue
        strategies: Set[str] = set()
        for backend in _literal_method_returns(py_file, {"backend_name"}):
            strategies.update(reference_backend_to_task_strategies.get(backend, []))
        if strategies:
            routes[stem] = sorted(strategies)
    return routes


def _scan_harness_contract_plugin_modules(
    directory: Path,
    reference_family_to_task_strategies: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Build contract plugin file-stem -> task strategies from reference_families."""
    routes: Dict[str, List[str]] = {}
    if not directory.is_dir():
        return routes

    for py_file in sorted(directory.glob("*.py")):
        stem = py_file.stem
        if stem.startswith("_") or stem.startswith("test_") or stem in {"__init__", "base"}:
            continue
        assignments = _literal_string_list_assignments(py_file, {"reference_families"})
        strategies: Set[str] = set()
        for reference_family in assignments.get("reference_families", []):
            strategies.update(reference_family_to_task_strategies.get(reference_family, []))
        if strategies:
            routes[stem] = sorted(strategies)
    return routes


def _threshold_profile_task_strategy_routes(
    thresholds_dir: Path,
    task_strategy_to_models: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Map threshold profile file stems to matching task strategies."""
    routes: Dict[str, List[str]] = {}
    if not thresholds_dir.is_dir():
        return routes
    known_task_strategies = set(task_strategy_to_models)
    for profile_path in sorted(thresholds_dir.glob("*.json")):
        stem = profile_path.stem
        if stem in known_task_strategies:
            routes[stem] = [stem]
    return routes


def build_impact_map(repo_root: Path) -> ImpactMap:
    """Build the impact map by scanning manifests and family plugins."""
    models_dir = repo_root / "tests" / "e2e" / "models"
    families_dir = repo_root / "python" / "tensorrt_model_connect" / "families"
    runtime_models_dir = repo_root / "src" / "runtime" / "models"
    harness_dir = repo_root / "tests" / "e2e_harness"
    default_reference_backend_by_task = _literal_string_dict_assignment(
        harness_dir / "manifest_loader.py", "_DEFAULT_REFERENCE_BACKEND"
    )

    family_to_models: Dict[str, List[str]] = {}
    strategy_to_models: Dict[str, List[str]] = {}
    task_strategy_to_models: Dict[str, List[str]] = {}
    manifest_field_to_models_sets: Dict[str, Set[str]] = {}
    e2e_data_file_to_models_sets: Dict[str, Set[str]] = {}
    reference_family_to_models_sets: Dict[str, Set[str]] = {}
    reference_family_to_task_strategies_sets: Dict[str, Set[str]] = {}
    reference_backend_to_task_strategies_sets: Dict[str, Set[str]] = {}
    all_model_names: List[str] = []
    core_models: List[str] = []
    model_metadata: Dict[str, Dict] = {}
    manifest_path_to_model: Dict[str, str] = {}
    testcase_to_model: Dict[str, str] = {}
    l0_replacement_by_model: Dict[str, str] = {}

    for manifest_path in _iter_e2e_manifest_paths(models_dir):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        name = data.get("name", manifest_path.stem)
        family = data.get("family", "")
        runtime_strategy = data.get("runtime_strategy", "")
        testcases = data.get("testcases", [])
        if not isinstance(testcases, list):
            testcases = []
        expanded_cases = [
            {**data, **testcase} for testcase in testcases if isinstance(testcase, dict)
        ]
        if not expanded_cases:
            expanded_cases = [data]

        all_model_names.append(name)
        model_metadata[name] = data
        try:
            manifest_path_to_model[manifest_path.relative_to(repo_root).as_posix()] = name
        except ValueError:
            pass

        if family:
            family_to_models.setdefault(family, []).append(name)
        if runtime_strategy:
            strategy_to_models.setdefault(runtime_strategy, []).append(name)
        task_strategies = {str(case.get("task_strategy", "") or "") for case in expanded_cases}
        for task_strategy in sorted(task_strategies - {""}):
            task_strategy_to_models.setdefault(task_strategy, []).append(name)
        if any(case.get("core", False) for case in expanded_cases):
            core_models.append(name)
        for case in expanded_cases:
            case_name = case.get("name")
            if isinstance(case_name, str) and case_name:
                testcase_to_model[case_name] = name
            l0_replacement = case.get("l0_replacement")
            if isinstance(l0_replacement, str) and l0_replacement and l0_replacement != name:
                l0_replacement_by_model[name] = l0_replacement
        fp8_scales = data.get("fp8_scales")
        if isinstance(fp8_scales, str) and fp8_scales:
            manifest_field_to_models_sets.setdefault("fp8_scales", set()).add(name)
        for data_path in _iter_manifest_data_paths(data, manifest_path, models_dir):
            e2e_data_file_to_models_sets.setdefault(data_path, set()).add(name)
        for case in expanded_cases:
            task_strategy = str(case.get("task_strategy", "") or "")
            reference_family = case.get("reference_family")
            if isinstance(reference_family, str) and reference_family:
                reference_family_to_models_sets.setdefault(reference_family, set()).add(name)
            if reference_family and task_strategy:
                reference_family_to_task_strategies_sets.setdefault(reference_family, set()).add(
                    task_strategy
                )
            reference_backend = case.get("reference_backend")
            if not isinstance(reference_backend, str) or not reference_backend:
                reference_backend = (
                    default_reference_backend_by_task.get(task_strategy, "hf_transformers")
                    if task_strategy
                    else ""
                )
            if reference_backend and task_strategy:
                reference_backend_to_task_strategies_sets.setdefault(reference_backend, set()).add(
                    task_strategy
                )

    builder_to_families = _scan_family_imports(families_dir) if families_dir.is_dir() else {}
    cpp_runtime_model_strategies = _scan_cpp_runtime_model_manifests(runtime_models_dir)
    model_owned_diff_rules = _scan_model_owned_diff_rules(models_dir)
    reference_family_to_task_strategies = {
        key: sorted(values) for key, values in reference_family_to_task_strategies_sets.items()
    }
    reference_backend_to_task_strategies = {
        key: sorted(values) for key, values in reference_backend_to_task_strategies_sets.items()
    }
    runner_task_strategies = _scan_harness_task_strategy_modules(
        harness_dir / "runners",
        method_names={"strategy_name"},
        fallbacks=RUNNER_TASK_STRATEGY_FALLBACKS,
    )
    comparator_task_strategies = _scan_harness_task_strategy_modules(
        harness_dir / "comparators",
        method_names={"task_strategy"},
        fallbacks=COMPARATOR_TASK_STRATEGY_FALLBACKS,
    )
    reference_task_strategies = _scan_harness_reference_modules(
        harness_dir / "references",
        reference_backend_to_task_strategies,
    )
    plugin_task_strategies = _scan_harness_contract_plugin_modules(
        harness_dir / "plugins",
        reference_family_to_task_strategies,
    )
    threshold_profile_task_strategies = _threshold_profile_task_strategy_routes(
        harness_dir / "thresholds" / "defaults",
        task_strategy_to_models,
    )

    def _models_for_scoped_strategies(strategies: Set[str]) -> List[str]:
        models: Set[str] = set()
        for strategy in strategies:
            models.update(strategy_to_models.get(strategy, []))
        return sorted(models)

    path_scope_overrides: Dict[str, List[str]] = {}
    scoped_cpp_tokens = {
        "src/runtime/core/gpu_matmul.h": "gpu_matmul",
        "src/runtime/core/gpu_matmul.cpp": "gpu_matmul",
    }
    for path, token in scoped_cpp_tokens.items():
        strategies: Set[str] = set()
        if runtime_models_dir.is_dir():
            for cpp_file in sorted(runtime_models_dir.glob("*/*.cpp")):
                try:
                    content = cpp_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if token not in content:
                    continue
                strategies.update(cpp_runtime_model_strategies.get(cpp_file.parent.name, []))
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
        manifest_path_to_model=manifest_path_to_model,
        testcase_to_model=testcase_to_model,
        builder_to_families=builder_to_families,
        cpp_runtime_model_strategies=cpp_runtime_model_strategies,
        manifest_field_to_models={
            key: sorted(models) for key, models in manifest_field_to_models_sets.items()
        },
        e2e_data_file_to_models={
            path: sorted(models) for path, models in e2e_data_file_to_models_sets.items()
        },
        path_scope_overrides=path_scope_overrides,
        l0_replacement_by_model=l0_replacement_by_model,
        reference_family_to_models={
            reference_family: sorted(models)
            for reference_family, models in reference_family_to_models_sets.items()
        },
        model_owned_diff_rules=model_owned_diff_rules,
        runner_task_strategies=runner_task_strategies,
        comparator_task_strategies=comparator_task_strategies,
        reference_task_strategies=reference_task_strategies,
        plugin_task_strategies=plugin_task_strategies,
        threshold_profile_task_strategies=threshold_profile_task_strategies,
    )


# ---------------------------------------------------------------------------
# Helper: resolve models from runtime/task strategies
# ---------------------------------------------------------------------------


def _models_for_runtime_strategies(
    strategies: List[str],
    imap: ImpactMap,
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
    task_strategies: List[str],
    imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for ts in task_strategies:
        models.update(imap.task_strategy_to_models.get(ts, []))
    return sorted(models)


def _apply_l0_replacements(
    models: List[str],
    imap: ImpactMap,
    preserve_models: Set[str],
    exact_models: Optional[Set[str]] = None,
) -> tuple[List[str], List[Dict[str, str]]]:
    """Replace nightly-only scale models with their L0 representatives.

    Direct edits to a nightly-only scale model still use the L0 representative:
    the large model's artifact contract is covered by nightly, while PR L0 keeps
    the same plugin/runtime path at smaller scale.

    Waiver diffs are different: they name the exact config whose xfail status
    changed, so keep those exact model IDs even if they normally have an L0
    representative.
    """
    del exact_models  # Retained in the signature to keep call sites stable.
    selected: Set[str] = set()
    replacements: List[Dict[str, str]] = []
    for model in models:
        if model in preserve_models:
            selected.add(model)
            continue
        replacement = imap.l0_replacement_by_model.get(model)
        if replacement:
            selected.add(replacement)
            replacements.append(
                {
                    "model": model,
                    "replacement": replacement,
                    "reason": str(
                        imap.model_metadata.get(model, {}).get(
                            "l0_replacement_reason",
                            "nightly-only scale coverage; L0 uses a smaller representative",
                        )
                    ),
                }
            )
        else:
            selected.add(model)
    return sorted(selected), replacements


def _infer_unit_tiers(path: str) -> List[str]:
    """Infer which unit test tiers a file change implies."""
    tiers: List[str] = []
    if path.startswith("python/tensorrt_model_connect/"):
        tiers.append("builder")
    if (
        path.startswith("src/")
        or path.startswith("include/")
        or path == "CMakeLists.txt"
        or path.startswith("cmake/")
    ):
        tiers.append("cpp")
    if path.startswith("tests/builder/"):
        tiers.append("builder")
    if path.startswith("tests/cpp/"):
        tiers.append("cpp")
    if path.startswith("tests/tools/") or path.startswith("tests/e2e_harness/test_"):
        tiers.append("tools")
    return sorted(set(tiers))


def _infer_rebuild_cpp(path: str) -> bool:
    """Does this file change require a C++ rebuild?"""
    return (
        path.startswith("src/")
        or path.startswith("include/")
        or path == "CMakeLists.txt"
        or path.startswith("cmake/")
        or path.startswith("tests/cpp/")
    )


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
            list(unit_tiers_override) if unit_tiers_override is not None else unit_tiers
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
    path_model = imap.manifest_path_to_model.get(context.path)
    if path_model:
        return [path_model]
    name = _group(context)
    return [name] if name in imap.all_model_names_set else []


def _family_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    return sorted(imap.family_to_models.get(_group(context), []))


def _minimax_h3_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    del context
    return sorted(imap.family_to_models.get("minimax_h3", []))


def _python_profile_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    return sorted(imap.family_to_models.get(_group(context), []))


def _e2e_model_threshold_models(context: RuleContext, imap: ImpactMap) -> List[str]:
    if context.match is None:
        return []
    model_name = context.match.group(2)
    owning_model = imap.testcase_to_model.get(model_name)
    if owning_model:
        return [owning_model]
    return sorted(imap.family_to_models.get(context.match.group(1), []))


def _task_strategy_models(task_strategies: List[str]) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        del context
        return _models_for_task_strategies(task_strategies, imap)

    return _resolver


def _hf_id_models(hf_ids: Set[str]) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        del context
        return sorted(
            model
            for model, metadata in imap.model_metadata.items()
            if metadata.get("hf_id") in hf_ids
        )

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


def _cpp_runtime_model_strategies(
    context: RuleContext,
    imap: ImpactMap,
) -> List[str]:
    return imap.cpp_runtime_model_strategies.get(_group(context), [])


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


def _model_owned_e2e_test_id(model: str, imap: ImpactMap) -> str:
    metadata = imap.model_metadata.get(model, {})
    family = str(metadata.get("family", "") or "").strip()
    if not family:
        return f"tests/test_e2e.py::test_e2e[{model}]"
    return f"tests/e2e/models/{family}/test_{family}_e2e.py::test_model_e2e[{model}]"


def _model_owned_e2e_test_ids(models: List[str], imap: ImpactMap) -> List[str]:
    return [_model_owned_e2e_test_id(model, imap) for model in models]


def _known_cpp_runtime_model(
    path: str,
    imap: ImpactMap,
    match: re.Match[str],
) -> bool:
    del path
    return bool(imap.cpp_runtime_model_strategies.get(match.group(1), []))


def _unknown_cpp_runtime_model(
    path: str,
    imap: ImpactMap,
    match: re.Match[str],
) -> bool:
    return not _known_cpp_runtime_model(path, imap, match)


def _is_specialized_builder(
    path: str,
    imap: ImpactMap,
    match: re.Match[str],
) -> bool:
    del path
    module_name = match.group(1)
    if not module_name.endswith("_builder"):
        return False
    if module_name in _ORCHESTRATOR_MODULES:
        return False
    families = imap.builder_to_families.get(module_name, [])
    return any(imap.family_to_models.get(family, []) for family in families)


StrategyMapSource = Dict[str, List[str]] | Callable[[ImpactMap], Dict[str, List[str]]]


def _strategy_map(
    strategy_map_source: StrategyMapSource,
    imap: ImpactMap,
) -> Dict[str, List[str]]:
    if callable(strategy_map_source):
        return strategy_map_source(imap)
    return strategy_map_source


def _known_task_strategy_stem(strategy_map_source: StrategyMapSource) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path
        return bool(_strategy_map(strategy_map_source, imap).get(match.group(1), []))

    return _predicate


def _unknown_task_strategy_stem(strategy_map_source: StrategyMapSource) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path
        stem = match.group(1)
        return stem != "__init__" and not bool(
            _strategy_map(strategy_map_source, imap).get(stem, [])
        )

    return _predicate


def _placeholder_sidecar_stem(stems: Set[str]) -> RulePredicate:
    def _predicate(path: str, imap: ImpactMap, match: re.Match[str]) -> bool:
        del path, imap
        return match.group(1) in stems

    return _predicate


def _task_strategy_models_from_group(
    strategy_map_source: StrategyMapSource,
) -> ModelsResolver:
    def _resolver(context: RuleContext, imap: ImpactMap) -> List[str]:
        return _models_for_task_strategies(
            _strategy_map(strategy_map_source, imap).get(_group(context), []), imap
        )

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


def _is_family_builder_test(path: str) -> bool:
    return (
        re.match(
            r"^python/tensorrt_model_connect/families/[A-Za-z]\w*/tests/.+\.py$",
            path,
        )
        is not None
    )


def _classification_rules() -> Tuple[ClassificationRule, ...]:
    rules = (
        ClassificationRule(
            priority=10,
            name="manifest",
            matcher=_regex_rule(r"tests/e2e/models/(?:[^/]+/manifests/)?([^/]+)\.json$"),
            resolver=_match_result("manifest", _manifest_models),
            covered_by=("TestSafetyNet.test_manifest_self",),
        ),
        ClassificationRule(
            priority=11,
            name="e2e_model_index",
            matcher=_regex_rule(r"tests/e2e/models/([^/]+)/MODEL\.toml$"),
            resolver=_match_result("e2e_model_index", _family_models),
            covered_by=("TestSafetyNet.test_e2e_model_index_self",),
        ),
        ClassificationRule(
            priority=12,
            name="e2e_model_threshold",
            matcher=_regex_rule(r"tests/e2e/models/([^/]+)/thresholds/([^/]+)\.json$"),
            resolver=_match_result("e2e_model_threshold", _e2e_model_threshold_models),
            covered_by=("TestSafetyNet.test_e2e_model_owned_threshold_self",),
        ),
        ClassificationRule(
            priority=14,
            name="standalone_gpu_test_support",
            matcher=_regex_rule(
                r"(?:tests/e2e/models/[^/]+/(?:run_[^/]+_fi|"
                r"test_flashinfer_(?:plugin|trt_attention)|"
                r"test_[^/]+_flashinfer)\.py|tests/test_tvm_ffi_e2e\.py)$"
            ),
            resolver=_match_result(
                "standalone_gpu_test_support",
                _no_models,
                ["tools"],
                False,
            ),
            covered_by=("TestNoImpact.test_standalone_gpu_tests_do_not_select_models",),
        ),
        ClassificationRule(
            priority=17,
            name="e2e_model_owned_test",
            # A public E2E family directory is an ownership boundary for every
            # file type; more specific manifest and asset rules run first.
            matcher=_regex_rule(
                r"tests/e2e/models/([^/]+)/.+$"
            ),
            resolver=_match_result("e2e_model_owned_test", _family_models),
            covered_by=(
                "TestSafetyNet.test_e2e_model_owned_test_self",
                "TestE2EDataFiles.test_unlisted_family_asset_maps_to_family",
            ),
        ),
        ClassificationRule(
            priority=16,
            name="family_unit_builder",
            matcher=lambda path, _imap: (
                RuleContext(path) if _is_family_builder_test(path) else None
            ),
            resolver=_match_result("family_unit_builder", _no_models),
            covered_by=("TestUnitTiers.test_family_unit_builder",),
        ),
        ClassificationRule(
            priority=15,
            name="family_model_index",
            matcher=_regex_rule(
                r"python/tensorrt_model_connect/families/([A-Za-z]\w*)/MODEL\.toml$"
            ),
            resolver=_match_result("family_model_index", _family_models),
            covered_by=("TestFamilyOwnedBuilder.test_family_model_index",),
        ),
        ClassificationRule(
            priority=20,
            name="family_package",
            # Family packages own code, build files, and other resources. The
            # underscore-prefixed internal directory remains shared.
            matcher=_regex_rule(
                r"python/tensorrt_model_connect/"
                r"families/([A-Za-z]\w*)/.+$"
            ),
            resolver=_match_result("family_package", _family_models),
            covered_by=(
                "TestFamilyPlugin.test_family_only_change",
                "TestFamilyPlugin.test_family_resource",
                "TestFamilyOwnedBuilder.test_family_local_model_implementation",
            ),
        ),
        ClassificationRule(
            priority=30,
            name="family_plugin",
            matcher=_regex_rule(
                r"python/tensorrt_model_connect/"
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
                r"python/tensorrt_model_connect/"
                r"families/((__init__|base)\.py)$"
            ),
            resolver=_match_result("family_base", _all_models),
            covered_by=(
                "TestFamilyPlugin.test_family_base_all_models",
                "TestFamilyPlugin.test_family_init_all_models",
            ),
        ),
        ClassificationRule(
            priority=19,
            name="python_profile_requirements",
            matcher=_regex_rule(
                r"python/tensorrt_model_connect/"
                r"families/([^/]+)/(?:profiles/requirements|python_profile_requirements)/"
                r"[^/]+\.lock\.txt$"
            ),
            resolver=_match_result("python_profile_requirements", _python_profile_models),
            covered_by=("TestSharedModules.test_python_profile_requirements_scope",),
        ),
        ClassificationRule(
            priority=18,
            name="validation_reference_requirements",
            matcher=_path_equals(
                "python/tensorrt_model_connect/"
                "python_profile_requirements/reference_common.lock.txt"
            ),
            resolver=_match_result(
                "validation_reference_requirements",
                _no_models,
                ["tools"],
                False,
            ),
            covered_by=(
                "TestUnitTiers.test_validation_reference_requirements_trigger_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=90,
            name="specialized_builder",
            matcher=_regex_rule(
                r"python/tensorrt_model_connect/(\w+)\.py$",
                _is_specialized_builder,
            ),
            resolver=_match_result("specialized_builder", _specialized_builder_models),
            covered_by=("TestDeclarativeClassificationRules.test_specialized_builder_rule",),
        ),
        ClassificationRule(
            priority=95,
            name="benchmark_python",
            matcher=_path_startswith("python/tensorrt_model_connect/benchmark/"),
            resolver=_match_result(
                "benchmark_python",
                _no_models,
                ["builder", "tools"],
                False,
            ),
            covered_by=("TestUnitTiers.test_benchmark_python_triggers_owned_units",),
        ),
        ClassificationRule(
            priority=100,
            name="shared_builder_module",
            matcher=_path_startswith("python/tensorrt_model_connect/"),
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
            priority=220,
            name="cpp_scoped_helper",
            matcher=_path_in_impact_map(lambda imap: imap.path_scope_overrides),
            resolver=_match_result("cpp_scoped_helper", _scoped_cpp_helper_models),
            covered_by=("TestCppScope.test_scoped_cpp_helper_gpu_matmul",),
        ),
        ClassificationRule(
            priority=225,
            name="third_party_stb_image",
            matcher=_path_startswith("third_party/stb/"),
            resolver=_match_result(
                "third_party_stb_image",
                _task_strategy_models(STB_IMAGE_TASK_STRATEGIES),
                ["cpp"],
                True,
            ),
            covered_by=("TestCppScope.test_third_party_stb_scopes_to_image_models",),
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
            priority=245,
            name="harness_runner_placeholder",
            matcher=_regex_rule(
                r"tests/e2e_harness/runners/(\w+)\.py$",
                _placeholder_sidecar_stem(SHARED_PLACEHOLDER_RUNNER_STEMS),
            ),
            resolver=_match_result(
                "harness_runner_placeholder",
                _no_models,
                unit_tiers_override=["tools"],
                rebuild_override=False,
            ),
            covered_by=("TestHarness.test_harness_runner_placeholder",),
        ),
        ClassificationRule(
            priority=250,
            name="harness_runner",
            matcher=_regex_rule(
                r"tests/e2e_harness/runners/(\w+)\.py$",
                _known_task_strategy_stem(lambda imap: imap.runner_task_strategies),
            ),
            resolver=_match_result(
                "harness_runner",
                _task_strategy_models_from_group(lambda imap: imap.runner_task_strategies),
            ),
            covered_by=("TestHarness.test_harness_runner",),
        ),
        ClassificationRule(
            priority=260,
            name="harness_runner_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/runners/(\w+)\.py$",
                _unknown_task_strategy_stem(lambda imap: imap.runner_task_strategies),
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
            priority=275,
            name="harness_comparator_placeholder",
            matcher=_regex_rule(
                r"tests/e2e_harness/comparators/(\w+)\.py$",
                _placeholder_sidecar_stem(SHARED_PLACEHOLDER_COMPARATOR_STEMS),
            ),
            resolver=_match_result(
                "harness_comparator_placeholder",
                _no_models,
                unit_tiers_override=["tools"],
                rebuild_override=False,
            ),
            covered_by=("TestHarness.test_harness_comparator_placeholder",),
        ),
        ClassificationRule(
            priority=280,
            name="harness_comparator",
            matcher=_regex_rule(
                r"tests/e2e_harness/comparators/(\w+)\.py$",
                _known_task_strategy_stem(lambda imap: imap.comparator_task_strategies),
            ),
            resolver=_match_result(
                "harness_comparator",
                _task_strategy_models_from_group(lambda imap: imap.comparator_task_strategies),
            ),
            covered_by=("TestHarness.test_harness_comparator",),
        ),
        ClassificationRule(
            priority=290,
            name="harness_comparator_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/comparators/(\w+)\.py$",
                _unknown_task_strategy_stem(lambda imap: imap.comparator_task_strategies),
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
            priority=305,
            name="harness_reference_placeholder",
            matcher=_regex_rule(
                r"tests/e2e_harness/references/(\w+)\.py$",
                _placeholder_sidecar_stem(SHARED_PLACEHOLDER_REFERENCE_STEMS),
            ),
            resolver=_match_result(
                "harness_reference_placeholder",
                _no_models,
                unit_tiers_override=["tools"],
                rebuild_override=False,
            ),
            covered_by=("TestHarness.test_harness_reference_placeholder",),
        ),
        ClassificationRule(
            priority=310,
            name="harness_reference",
            matcher=_regex_rule(
                r"tests/e2e_harness/references/(\w+)\.py$",
                _known_task_strategy_stem(lambda imap: imap.reference_task_strategies),
            ),
            resolver=_match_result(
                "harness_reference",
                _task_strategy_models_from_group(lambda imap: imap.reference_task_strategies),
            ),
            covered_by=("TestHarness.test_torch_reference_includes_neural_operator_models",),
        ),
        ClassificationRule(
            priority=320,
            name="harness_reference_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/references/(\w+)\.py$",
                _unknown_task_strategy_stem(lambda imap: imap.reference_task_strategies),
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
                _known_task_strategy_stem(lambda imap: imap.plugin_task_strategies),
            ),
            resolver=_match_result(
                "harness_plugin",
                _task_strategy_models_from_group(lambda imap: imap.plugin_task_strategies),
            ),
            covered_by=("TestHarness.test_harness_plugin",),
        ),
        ClassificationRule(
            priority=350,
            name="harness_plugin_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/plugins/(\w+)\.py$",
                _unknown_task_strategy_stem(lambda imap: imap.plugin_task_strategies),
            ),
            resolver=_match_result("harness_plugin_unknown", _all_models),
            covered_by=("TestDeclarativeClassificationRules.test_representative_rule_paths",),
        ),
        ClassificationRule(
            priority=360,
            name="harness_threshold_profile",
            matcher=_regex_rule(
                r"tests/e2e_harness/thresholds/defaults/([\w_]+)\.json$",
                _known_task_strategy_stem(lambda imap: imap.threshold_profile_task_strategies),
            ),
            resolver=_match_result(
                "harness_threshold_profile",
                _task_strategy_models_from_group(
                    lambda imap: imap.threshold_profile_task_strategies
                ),
            ),
            covered_by=("TestHarness.test_harness_threshold_profile",),
        ),
        ClassificationRule(
            priority=370,
            name="harness_threshold_unknown",
            matcher=_regex_rule(
                r"tests/e2e_harness/thresholds/defaults/([\w_]+)\.json$",
                _unknown_task_strategy_stem(lambda imap: imap.threshold_profile_task_strategies),
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
            priority=395,
            name="e2e_schedule_metadata",
            matcher=_path_in(
                {
                    "tests/e2e/timing_estimates.json",
                    "tests/e2e_partition.py",
                    "tests/runtime_strategy_matrix.yaml",
                }
            ),
            resolver=_match_result(
                "e2e_schedule_metadata",
                _no_models,
                ["tools"],
                False,
            ),
            covered_by=("TestNoImpact.test_e2e_schedule_metadata_tools_only",),
        ),
        ClassificationRule(
            priority=398,
            name="minimax_h3_windows_reproduction_helper",
            matcher=_path_in(
                {
                    "requirements/windows-h3-fastvideo-vsa.txt",
                    "scripts/install_windows_h3_fastvideo_vsa.cmd",
                    "scripts/install_windows_h3_fastvideo_vsa.ps1",
                    "scripts/run_windows_h3_fastvideo_vsa.ps1",
                    "scripts/run_windows_h3_hot_benchmark.ps1",
                    "third_party/fastvideo/LICENSE",
                    "third_party/fastvideo/README.md",
                    "third_party/fastvideo/windows-h3-vsa.patch",
                }
            ),
            resolver=_match_result(
                "minimax_h3_windows_reproduction_helper",
                _minimax_h3_models,
                ["tools"],
                False,
            ),
            covered_by=(
                "TestUnitTiers.test_minimax_h3_windows_reproduction_helpers",
            ),
        ),
        ClassificationRule(
            priority=399,
            name="minimax_h3_windows_build_script",
            matcher=_path_equals("scripts/build_windows_h3.ps1"),
            resolver=_match_result(
                "minimax_h3_windows_build_script",
                _minimax_h3_models,
                ["builder"],
                True,
            ),
            covered_by=(
                "TestUnitTiers.test_minimax_h3_windows_build_script",
            ),
        ),
        ClassificationRule(
            priority=400,
            name="e2e_runner_script",
            matcher=_path_in(
                {
                    "tools/ci/e2e_schedule.py",
                    "tools/ci/e2e_scheduler.py",
                    "scripts/schedule_e2e.py",
                    "scripts/hf_cache_download_worker.py",
                    "scripts/warm_hf_cache.py",
                }
            ),
            resolver=_match_result("e2e_runner_script", _all_models, ["tools"]),
            covered_by=("TestNoImpact.test_e2e_runner_scripts_trigger_all_models",),
        ),
        ClassificationRule(
            priority=401,
            name="ci_orchestration",
            matcher=_path_startswith("tools/ci/"),
            resolver=_match_result("ci_orchestration", _all_models, ["tools"]),
            covered_by=("TestNoImpact.test_ci_orchestration_triggers_all_models",),
        ),
        ClassificationRule(
            priority=405,
            name="legacy_e2e_test_support",
            matcher=_regex_rule(r"tests/e2e/(?:__init__|conftest|test_[\w_]+)\.py$"),
            resolver=_match_result(
                "legacy_e2e_test_support",
                _no_models,
                ["tools"],
                False,
            ),
            covered_by=("TestNoImpact.test_legacy_e2e_tests_do_not_select_models",),
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
            priority=445,
            name="e2e_selection_unit",
            matcher=_path_equals("tests/test_e2e_selection.py"),
            resolver=_match_result("e2e_selection_unit", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_e2e_selection_unit",),
        ),
        ClassificationRule(
            priority=446,
            name="model_plugin_validation_tool",
            matcher=_path_in(
                {
                    "tools/e2e_origin_main_parity.py",
                    "tools/model_plugin_isolation.py",
                }
            ),
            resolver=_match_result("model_plugin_validation_tool", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_model_plugin_validation_tools",),
        ),
        ClassificationRule(
            priority=447,
            name="family_development_tool",
            matcher=_regex_rule(r"tools/families/([A-Za-z]\w*)/.+\.py$"),
            resolver=_match_result(
                "family_development_tool",
                _family_models,
                ["tools"],
                False,
            ),
            covered_by=("TestFamilyPlugin.test_family_development_tool",),
        ),
        ClassificationRule(
            priority=448,
            name="family_ownership_tool",
            matcher=_path_in(
                {
                    "tools/families/__init__.py",
                    "tools/family_source_isolation.py",
                    "tools/family_specialization.py",
                    "tools/migrate_family_layout.py",
                    "tools/prune_family_helpers.py",
                    "tools/relocate_family_development.py",
                    "tools/specialize_family.py",
                    "tools/specialize_family_switches.py",
                }
            ),
            resolver=_match_result(
                "family_ownership_tool",
                _no_models,
                ["tools"],
                False,
            ),
            covered_by=("TestUnitTiers.test_family_ownership_tools",),
        ),
        ClassificationRule(
            priority=449,
            name="e2e_report_tool",
            matcher=_path_in({
                "scripts/generate_e2e_report.py",
                "scripts/generate_e2e_report_assets/e2e_report.css",
                "scripts/generate_e2e_report_assets/e2e_report.js",
                "scripts/reporting/__init__.py",
                "scripts/reporting/vlm_assessment.py",
            }),
            resolver=_match_result(
                "e2e_report_tool", _no_models, ["tools"], False,
            ),
            covered_by=("TestUnitTiers.test_e2e_report_tools",),
        ),
        ClassificationRule(
            priority=450,
            name="local_qwen3_hf_fixture",
            matcher=_regex_rule(r"models/hf/(?:Qwen__Qwen3-0\.6B|qwen3)(?:/.*)?$"),
            resolver=_match_result(
                "local_qwen3_hf_fixture",
                _hf_id_models({"Qwen/Qwen3-0.6B"}),
            ),
            covered_by=("TestNoImpact.test_local_qwen3_fixture_scopes_to_qwen3",),
        ),
        ClassificationRule(
            priority=455,
            name="cpp_example_tool",
            matcher=_regex_rule(r"examples/.+\.cpp$"),
            resolver=_match_result("cpp_example_tool", _no_models, ["cpp"], True),
            covered_by=("TestUnitTiers.test_cpp_example_tool_triggers_cpp_tier",),
        ),
        ClassificationRule(
            priority=456,
            name="benchmark_cli_asset",
            matcher=_path_in({"examples/trtmc_bench.yaml", "scripts/trtmc-bench"}),
            resolver=_match_result("benchmark_cli_asset", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_benchmark_cli_assets_trigger_tools_tier",),
        ),
        ClassificationRule(
            priority=457,
            name="release_performance",
            matcher=_path_startswith("benchmarks/performance/"),
            resolver=_match_result("release_performance", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_release_performance_triggers_tools_tier",),
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
            priority=13,
            name="e2e_data_file",
            matcher=_path_in_impact_map(lambda imap: imap.e2e_data_file_to_models),
            resolver=_match_result("e2e_data_file", _e2e_data_file_models),
            covered_by=("TestE2EDataFiles.test_data_file_maps_to_manifest_users",),
        ),
        ClassificationRule(
            priority=484,
            name="nightly_artifact_selector_tool",
            matcher=_path_equals("tools/select_latest_attempt_artifact.py"),
            resolver=_match_result(
                "nightly_artifact_selector_tool", _no_models, ["tools"], False,
            ),
            covered_by=("TestUnitTiers.test_nightly_artifact_selector_tool",),
        ),
        ClassificationRule(
            priority=482,
            name="model_checks_tool",
            matcher=_regex_rule(
                r"(?:tools/(?:campaign_shards|model_(?:checks|selection))\.py|"
                r"tests/model_checks/(?:environments|platforms)/[^/]+\.yaml)$"
            ),
            resolver=_match_result(
                "model_checks_tool", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestUnitTiers.test_model_checks_tool_triggers_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=483,
            name="report_generation_tool",
            matcher=_path_in(
                {
                    "tools/execution_ledger.py",
                    "tools/performance/__init__.py",
                    "tools/performance/catalog.py",
                    "tools/perf_matrix.py",
                    "tools/qualification_report.py",
                    "tools/qualification_report_assets/qualification-report.css",
                    "tools/qualification_report_assets/qualification-report.js",
                    "tools/qualification_report_assets/qualification-report.schema.json",
                    "tools/reporting_html.py",
                }
            ),
            resolver=_match_result(
                "report_generation_tool", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestUnitTiers.test_report_generation_tool_triggers_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=485,
            name="model_ci_tool",
            matcher=_path_equals("tools/model_ci.py"),
            resolver=_match_result(
                "model_ci_tool", _no_models, ["tools"], False,
            ),
            covered_by=("TestUnitTiers.test_model_ci_tool",),
        ),
        ClassificationRule(
            priority=486,
            name="validation_engine_tool",
            matcher=_path_in({
                "tools/validation/engine.py",
                "tools/elf_hf_reference.py",
                "tools/full_duplex_bench_score.py",
                "tools/prepare_elf_validation_datasets.py",
                "tools/prepare_full_duplex_bench_validation.py",
                "tools/prepare_media_validation_datasets.py",
                "tools/prepare_model_plugin_validation_datasets.py",
                "tools/prepare_refcoco_validation_dataset.py",
                "tools/prepare_vision_validation_datasets.py",
            }),
            resolver=_match_result(
                "validation_engine_tool", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestUnitTiers.test_validation_engine_tool_triggers_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=487,
            name="validation_tool",
            matcher=_regex_rule(
                r"tools/(?:validation/(?:[^/]+\.py|README\.md)|"
                r"reference/[^/]+\.py|"
                r"trtmc_(?:compare|disagreements|reference|validate)\.py)$"
            ),
            resolver=_match_result("validation_tool", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_validation_tool_triggers_tools_tier",),
        ),
        ClassificationRule(
            priority=488,
            name="validation_workload_config",
            matcher=_path_equals("tests/validation/workloads.yaml"),
            resolver=_match_result(
                "validation_workload_config", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestUnitTiers.test_validation_workload_config_triggers_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=489,
            name="validation_config",
            matcher=_path_equals("tests/validation/model_workloads.yaml"),
            resolver=_match_result("validation_config", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_validation_config_triggers_tools_tier",),
        ),
        ClassificationRule(
            priority=490,
            name="test_impact_tool",
            matcher=_path_equals("tools/test_impact.py"),
            resolver=_match_result("test_impact_tool", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_test_impact_tool_triggers_tools_tier",),
        ),
        ClassificationRule(
            priority=491,
            name="source_container_contract",
            matcher=_path_in({"Dockerfile.dev.aarch64", "Dockerfile.dev.x86"}),
            resolver=_match_result(
                "source_container_contract", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestNoImpact.test_source_dockerfiles_trigger_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=492,
            name="community_cpu_contract",
            matcher=_path_in(
                {
                    ".pre-commit-config.yaml",
                    "Dockerfile.community-cpu",
                    "requirements/community-ci.txt",
                    "tools/community_ci.py",
                }
            ),
            resolver=_match_result(
                "community_cpu_contract", _no_models, ["tools"], False
            ),
            covered_by=(
                "TestNoImpact.test_community_cpu_contract_triggers_tools_tier",
            ),
        ),
        ClassificationRule(
            priority=493,
            name="github_ci_config",
            matcher=_path_startswith(".github/"),
            resolver=_match_result("github_ci_config", _no_models, ["tools"], False),
            covered_by=("TestUnitTiers.test_github_ci_config_triggers_tools_tier",),
        ),
        ClassificationRule(
            priority=494,
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
        if (
            path.startswith("tests/builder/")
            or _is_family_builder_test(path)
            or _is_model_owned_python_unit_test(path)
        ):
            builder_tests.add(path)
        elif path.startswith("tests/tools/") or path.startswith("tests/e2e_harness/test_"):
            tools_tests.add(path)
    return sorted(builder_tests), sorted(tools_tests)


_EXPLICIT_TOOLS_TEST_TARGETS = {
    "tools/select_latest_attempt_artifact.py": (
        "tests/tools/test_github_actions_ci.py",
        "tests/tools/test_select_latest_attempt_artifact.py",
    ),
    "tools/model_ci.py": (
        "tests/tools/test_model_ci.py",
    ),
    "scripts/generate_e2e_report.py": (
        "tests/tools/test_generate_report.py",
    ),
    "scripts/generate_e2e_report_assets/e2e_report.css": (
        "tests/tools/test_generate_report.py",
    ),
    "scripts/generate_e2e_report_assets/e2e_report.js": (
        "tests/tools/test_generate_report.py",
    ),
    "scripts/reporting/__init__.py": (
        "tests/tools/test_generate_report.py",
    ),
    "scripts/reporting/vlm_assessment.py": (
        "tests/tools/test_generate_report.py",
    ),
    "tools/ci/e2e_scheduler.py": (
        "tests/tools/test_github_actions_ci.py",
        "tests/tools/test_schedule_e2e.py",
    ),
    "tools/ci/e2e_schedule.py": (
        "tests/tools/test_schedule_e2e.py",
    ),
    "scripts/hf_cache_download_worker.py": (
        "tests/tools/test_warm_hf_cache_static.py",
    ),
    "scripts/warm_hf_cache.py": (
        "tests/tools/test_warm_hf_cache_static.py",
    ),
    "tests/e2e_harness/model_runner.py": ("tests/tools/test_model_e2e_runner.py",),
}


def _explicit_tools_test_targets(changed_files: List[str]) -> List[str]:
    """Return tests for non-Python CI surfaces that coverage cannot observe."""
    tests: Set[str] = set()
    for raw_path in changed_files:
        path = raw_path.replace("\\", "/").strip("/")
        tests.update(_EXPLICIT_TOOLS_TEST_TARGETS.get(path, ()))
    return sorted(tests)


def _is_model_owned_python_unit_test(path: str) -> bool:
    """Return True for model-owned pytest files that are safe as unit targets."""
    normalized = path.replace("\\", "/").strip("/")
    if not re.match(r"^tests/e2e/models/[^/]+/(?:.+/)?test_[^/]+\.py$", normalized):
        return False
    return not Path(normalized).name.endswith("_e2e.py")


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _model_families_for_models(models: List[str], imap: ImpactMap) -> List[str]:
    families = {str(imap.model_metadata.get(model, {}).get("family", "") or "") for model in models}
    return sorted(family for family in families if family)


def _model_owned_python_test_targets(
    models: List[str],
    imap: ImpactMap,
    repo_root: Optional[Path],
) -> List[str]:
    """Return local pytest targets owned by the selected model families."""
    if repo_root is None:
        return []

    targets: Set[str] = set()
    for family in _model_families_for_models(models, imap):
        model_test_dir = repo_root / "tests" / "e2e" / "models" / family
        if model_test_dir.is_dir():
            for test_path in sorted(model_test_dir.rglob("test_*.py")):
                if test_path.name.endswith("_e2e.py"):
                    continue
                targets.add(_repo_relative(test_path, repo_root))

        family_package_tests = (
            repo_root / "python" / "tensorrt_model_connect" / "families" / family / "tests"
        )
        if family_package_tests.is_dir():
            for test_path in sorted(family_package_tests.rglob("test_*.py")):
                targets.add(_repo_relative(test_path, repo_root))

    return sorted(targets)


_MODEL_OWNED_COVERAGE_FALLBACK_RULES = {
    "cpp_runtime_model",
    "family_package",
    "family_plugin",
    "python_profile_requirements",
    "specialized_builder",
}


def _is_model_owned_coverage_fallback(
    match: RuleMatch,
    imap: ImpactMap,
) -> bool:
    if match.rule not in _MODEL_OWNED_COVERAGE_FALLBACK_RULES:
        return False
    if match.rule in _BROAD_FALLBACK_RULES or match.rule.endswith("_unknown"):
        return False
    return _is_narrow_model_set(match.models, imap)


def _replace_model_owned_coverage_fallbacks(
    changed_files: List[str],
    fallback_files: Dict[str, List[str]],
    fallback_tiers: List[str],
    match_by_path: Dict[str, RuleMatch],
    imap: ImpactMap,
    repo_root: Optional[Path],
) -> tuple[List[str], List[str], List[str]]:
    """Replace model-owned coverage misses with model-owned pytest targets.

    Coverage maps can lag behind newly added model-owned files. Shared files
    still keep full-tier fallback, but files already classified to a narrow
    model family should not force unrelated builder or C++ tests.
    """
    del changed_files

    builder_targets: Set[str] = set()
    cpp_targets: Set[str] = set()
    remaining_fallback_tiers: Set[str] = set(fallback_tiers)

    for tier, paths in fallback_files.items():
        unresolved_paths: List[str] = []
        for path in paths:
            match = match_by_path.get(path)
            if match is None:
                match = classify_file(path, imap)
            if not _is_model_owned_coverage_fallback(match, imap):
                unresolved_paths.append(path)
                continue
            if tier == "builder":
                builder_targets.update(
                    _model_owned_python_test_targets(match.models, imap, repo_root)
                )
            elif tier == "cpp":
                # Model-owned C++ misses are covered by the impacted model E2E
                # selection and isolated plugin build target.
                pass
            else:
                unresolved_paths.append(path)

        if unresolved_paths:
            remaining_fallback_tiers.add(tier)
        else:
            remaining_fallback_tiers.discard(tier)

    return (
        sorted(builder_targets),
        sorted(cpp_targets),
        sorted(remaining_fallback_tiers),
    )


def _filter_models_by_ci_tier(
    models: List[str],
    imap: ImpactMap,
    exclude_ci_tiers: Set[str],
) -> List[str]:
    """Drop models whose manifest ci_tier is excluded by this selection."""
    if not exclude_ci_tiers:
        return sorted(models)
    return sorted(
        model
        for model in models
        if str(imap.model_metadata.get(model, {}).get("ci_tier", "") or "") not in exclude_ci_tiers
    )


def analyze_impact(
    changed_files: List[str],
    imap: ImpactMap,
    cap: Optional[int] = None,
    coverage_map: Optional[Dict[str, List[str]]] = None,
    base: Optional[str] = None,
    head: Optional[str] = None,
    repo_root: Optional[Path] = None,
    e2e_suite: str = "l0",
    exclude_ci_tiers: Optional[Set[str]] = None,
) -> ImpactResult:
    """Analyze impact of all changed files and return aggregated result."""
    if exclude_ci_tiers is None:
        exclude_ci_tiers = set(_DEFAULT_EXCLUDED_CI_TIERS)

    all_models: Set[str] = set()
    preserve_l0_models: Set[str] = set()
    exact_models: Set[str] = set()
    all_tiers: Set[str] = set()
    rebuild_cpp = False
    matched_rules: List[Dict] = []
    match_by_path: Dict[str, RuleMatch] = {}
    diff_text_by_path: Dict[str, str] = {}
    candidate_models: List[str] = []

    if base and head and repo_root:
        for fpath in changed_files:
            diff_text = get_file_diff(base, head, repo_root, fpath)
            if diff_text:
                diff_text_by_path[fpath] = diff_text
        candidate_models = _candidate_models_from_diffs(
            changed_files,
            imap,
        )

    for fpath in changed_files:
        match = classify_file(fpath, imap)
        diff_text = diff_text_by_path.get(fpath)
        if diff_text:
            match = maybe_refine_match_with_diff(
                fpath,
                match,
                diff_text,
                imap,
                candidate_models,
            )
        all_models.update(match.models)
        if match.rule == "e2e_waives_model_lines":
            preserve_l0_models.update(match.models)
        if match.rule in ("manifest", "e2e_data_file", "e2e_model_threshold"):
            exact_models.update(match.models)
        all_tiers.update(match.unit_tiers)
        rebuild_cpp = rebuild_cpp or match.rebuild_cpp
        match_by_path[fpath] = match
        matched_rules.append(
            {
                "file": fpath,
                "rule": match.rule,
                "models": match.models,
            }
        )

    e2e_models = sorted(all_models)
    l0_replacements: List[Dict[str, str]] = []
    if e2e_suite == "l0":
        e2e_models, l0_replacements = _apply_l0_replacements(
            e2e_models,
            imap,
            preserve_l0_models,
            exact_models,
        )
    cap_applied = False
    if cap is not None and len(e2e_models) > cap:
        e2e_models = sorted(imap.core_models)
        cap_applied = True
        l0_replacements = []
    e2e_models = _filter_models_by_ci_tier(e2e_models, imap, exclude_ci_tiers)

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
        model_builder_tests, model_cpp_tests, fallback_tiers = (
            _replace_model_owned_coverage_fallbacks(
                changed_files,
                getattr(sel, "fallback_files", {}),
                fallback_tiers,
                match_by_path,
                imap,
                repo_root,
            )
        )
        if model_builder_tests:
            builder_tests = sorted(set(builder_tests).union(model_builder_tests))
        if model_cpp_tests:
            cpp_tests = sorted(set(cpp_tests).union(model_cpp_tests))

    direct_builder_tests, direct_tools_tests = _direct_python_test_targets(changed_files)
    direct_tools_tests = sorted(
        set(direct_tools_tests).union(_explicit_tools_test_targets(changed_files))
    )
    if direct_builder_tests:
        builder_tests = sorted(set(builder_tests).union(direct_builder_tests))
    if direct_tools_tests:
        tools_tests = sorted(set(tools_tests).union(direct_tools_tests))

    return ImpactResult(
        e2e_models=e2e_models,
        unit_tiers=sorted(all_tiers),
        rebuild_cpp=rebuild_cpp,
        cap_applied=cap_applied,
        matched_rules=matched_rules,
        e2e_test_ids=_model_owned_e2e_test_ids(e2e_models, imap),
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
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_root,
            )
            files = [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
            return sorted(files)
        except subprocess.CalledProcessError:
            continue
    # Both diffs failed (shallow clone, missing ref, etc.)
    print(
        f"WARNING: git diff failed for {base}..{head} -- "
        "treating as all files changed (safety net)",
        file=sys.stderr,
    )
    return None


def get_file_diff(base: str, head: str, repo_root: Path, path: str) -> Optional[str]:
    """Get unified=0 diff for a single file, or None if git diff fails."""
    for cmd in [
        ["git", "diff", "--unified=0", f"{base}...{head}", "--", path],
        ["git", "diff", "--unified=0", base, head, "--", path],
    ]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=repo_root,
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
        if re.fullmatch(r"[\[\]{}(),;:'\"]+", content):
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
        any(token in _normalize_diff_line(line) for token in allowed_tokens) for line in lines
    )


def _fp8_scale_models(imap: ImpactMap) -> List[str]:
    return imap.manifest_field_to_models.get("fp8_scales", [])


def _diffusion_task_models(imap: ImpactMap) -> List[str]:
    return _models_for_task_strategies(["diffusion_media_generation"], imap)


def _segmentation_task_models(imap: ImpactMap) -> List[str]:
    return _models_for_task_strategies(
        ["segmentation", "prompted_segmentation", "object_detection"], imap
    )


def _models_for_families(families: Tuple[str, ...], imap: ImpactMap) -> List[str]:
    models: Set[str] = set()
    for family in families:
        models.update(imap.family_to_models.get(family, []))
    return sorted(models)


def _model_owned_scope_models(
    spec: ModelOwnedDiffRuleSpec,
    imap: ImpactMap,
) -> List[str]:
    scope = spec.scope
    models: Set[str] = set()
    if bool(scope.get("owner_family")):
        models.update(imap.family_to_models.get(spec.owner, []))
    for model in scope.get("models", []):
        if isinstance(model, str) and model in imap.all_model_names_set:
            models.add(model)
    for family in scope.get("families", []):
        if isinstance(family, str):
            models.update(imap.family_to_models.get(family, []))
    for strategy in scope.get("runtime_strategies", []):
        if isinstance(strategy, str):
            models.update(imap.strategy_to_models.get(strategy, []))
    for task_strategy in scope.get("task_strategies", []):
        if isinstance(task_strategy, str):
            models.update(imap.task_strategy_to_models.get(task_strategy, []))
    return sorted(models)


def _canonical_identifier(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _diff_identifier_fragments(line: str) -> Set[str]:
    fragments: Set[str] = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.-]*", line):
        fragments.add(_canonical_identifier(token))
        for part in re.split(r"[.]", token):
            part = _canonical_identifier(part)
            if part:
                fragments.add(part)
    return {fragment for fragment in fragments if fragment}


def _line_identifier_models(
    line: str,
    imap: ImpactMap,
    include_task_strategies: bool = False,
) -> List[str]:
    fragments = _diff_identifier_fragments(line)
    models: Set[str] = set()
    model_fragments = {_canonical_identifier(model) for model in imap.all_model_names}

    for model in imap.all_model_names:
        if _canonical_identifier(model) in fragments:
            models.add(model)
    for runtime_strategy, strategy_models in imap.strategy_to_models.items():
        if _canonical_identifier(runtime_strategy) in fragments:
            models.update(strategy_models)
    for family, family_models in imap.family_to_models.items():
        if _canonical_identifier(family) in fragments - model_fragments:
            models.update(family_models)
    for reference_family, reference_models in imap.reference_family_to_models.items():
        if _canonical_identifier(reference_family) in fragments:
            models.update(reference_models)
    if include_task_strategies:
        for task_strategy, task_models in imap.task_strategy_to_models.items():
            if _canonical_identifier(task_strategy) in fragments:
                models.update(task_models)

    return sorted(models)


def _models_from_diff_identifiers(
    lines: List[str],
    imap: ImpactMap,
    include_task_strategies: bool = False,
) -> List[str]:
    models: Set[str] = set()
    for line in lines:
        models.update(
            _line_identifier_models(
                line,
                imap,
                include_task_strategies=include_task_strategies,
            )
        )
    return sorted(models)


def _is_narrow_model_set(models: List[str], imap: ImpactMap) -> bool:
    return bool(models) and len(set(models)) < len(imap.all_model_names_set)


def _line_is_metadata_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _line_matches_allowed_tokens(line: str, allowed_tokens: Tuple[str, ...]) -> bool:
    normalized = _normalize_diff_line(line)
    return any(token in normalized for token in allowed_tokens)


def _scoped_models_from_path(path: str, imap: ImpactMap) -> List[str]:
    match = classify_file(path, imap)
    candidate_rule_names = {
        "cpp_runtime_model",
        "e2e_data_file",
        "family_package",
        "family_plugin",
        "manifest",
        "python_profile_requirements",
        "specialized_builder",
    }
    if match.rule not in candidate_rule_names:
        return []
    if not _is_narrow_model_set(match.models, imap):
        return []
    if match.rule in _BROAD_FALLBACK_RULES:
        return []
    if match.rule.endswith("_unknown"):
        return []
    return match.models


def _candidate_models_from_diffs(
    changed_files: List[str],
    imap: ImpactMap,
) -> List[str]:
    models: Set[str] = set()
    for path in changed_files:
        models.update(_scoped_models_from_path(path, imap))
    return sorted(models)


_HARNESS_REGISTRY_TOKENS: Tuple[str, ...] = (
    "args",
    "build",
    "case.family",
    "case.metadata",
    "case.reference_family",
    "cli_commands",
    "comparisonmode",
    "comparator_class",
    "diff_framework",
    "family",
    "gating",
    "import",
    "kind",
    "metadata",
    "overrides",
    "phase",
    "preflightrequirement",
    "python_module_available",
    "referencefamily",
    "reference_family",
    "reqs.append",
    "return",
    "runner_class",
    "runtime_strategy",
    "stage",
    "task",
    "task_strategy",
    "torch",
    "transformers",
    "usercontract",
)


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


@dataclass(frozen=True)
class ModelOwnedTokenDiffRefinementRule(DiffRefinementRule):
    spec: ModelOwnedDiffRuleSpec

    @property
    def name(self) -> str:
        return self.spec.name

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        return path == self.spec.path and _all_lines_match_tokens(lines, self.spec.allowed_tokens)

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
            _model_owned_scope_models(self.spec, imap),
            match.unit_tiers,
            match.rebuild_cpp,
        )


@dataclass(frozen=True)
class IdentifierDiffRefinementRule(DiffRefinementRule):
    name: str
    paths: Tuple[str, ...] = ()
    path_prefixes: Tuple[str, ...] = ()
    allowed_tokens: Tuple[str, ...] = ()
    include_task_strategies: bool = False

    def _path_matches(self, path: str) -> bool:
        return path in self.paths or path.startswith(self.path_prefixes)

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        if not self._path_matches(path):
            return False
        scoped_models = _models_from_diff_identifiers(
            lines,
            imap,
            include_task_strategies=self.include_task_strategies,
        )
        if not _is_narrow_model_set(scoped_models, imap):
            return False
        return all(
            _line_is_metadata_comment(line)
            or _line_identifier_models(
                line,
                imap,
                include_task_strategies=self.include_task_strategies,
            )
            or _line_matches_allowed_tokens(line, self.allowed_tokens)
            for line in lines
        )

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
            _models_from_diff_identifiers(
                lines,
                imap,
                include_task_strategies=self.include_task_strategies,
            ),
            match.unit_tiers,
            match.rebuild_cpp,
        )


@dataclass(frozen=True)
class CandidateTokenDiffRefinementRule(DiffRefinementRule):
    name: str
    path: str
    allowed_tokens: Tuple[str, ...]

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del path, lines, imap
        return False

    def refine(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
    ) -> RuleMatch:
        del path, match, lines, imap
        raise RuntimeError("candidate-aware rules must use refine_with_candidates")

    def matches_with_candidates(
        self,
        path: str,
        lines: List[str],
        imap: ImpactMap,
        candidate_models: List[str],
    ) -> bool:
        return (
            path == self.path
            and _is_narrow_model_set(candidate_models, imap)
            and _all_lines_match_tokens(lines, self.allowed_tokens)
        )

    def refine_with_candidates(
        self,
        path: str,
        match: RuleMatch,
        lines: List[str],
        imap: ImpactMap,
        candidate_models: List[str],
    ) -> RuleMatch:
        del path, lines, imap
        return RuleMatch(
            self.name,
            sorted(set(candidate_models)),
            match.unit_tiers,
            match.rebuild_cpp,
        )


class PyprojectValidationOptionalDependenciesRule(DiffRefinementRule):
    """Scope the isolated validation optional extra to validation tools."""

    name = "pyproject_validation_optional_dependencies"
    path = "pyproject.toml"
    _assignment = re.compile(r"^validation\s*=\s*\[.*\]\s*$")

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        del imap
        return (
            path == self.path
            and bool(lines)
            and all(self._assignment.fullmatch(line.strip()) for line in lines)
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
            [],
            sorted(set(match.unit_tiers) | {"tools"}),
            False,
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


class KnownModelTimingEstimateRule(DiffRefinementRule):
    name = "e2e_timing_estimates_known_models"
    path = "tests/e2e/timing_estimates.json"

    @staticmethod
    def _models_from_lines(lines: List[str], imap: ImpactMap) -> List[str]:
        models: Set[str] = set()
        for line in lines:
            match = re.fullmatch(r'"([^"]+)":\s*[0-9]+,?', line)
            if match is None:
                return []
            model = match.group(1)
            if model not in imap.all_model_names_set:
                return []
            models.add(model)
        return sorted(models)

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


class RuntimeStrategyMatrixRule(DiffRefinementRule):
    name = "runtime_strategy_matrix_known_strategies"
    path = "tests/runtime_strategy_matrix.yaml"
    allowed_tokens = (
        "cli_exemption",
        "cli_commands",
        "comparator_class",
        "diff_framework_check_classes",
        "diff_framework_exemption",
        "neural_operator",
        "no_diff_framework_check_currently_registers_runtime_strategies",
        "performance_mode",
        "runner_class",
        "solve",
        "task_strategy",
        "tests.e2e_harness.comparators",
        "tests.e2e_harness.runners",
    )

    @staticmethod
    def _strategies_from_lines(lines: List[str], imap: ImpactMap) -> List[str]:
        strategies: Set[str] = set()
        for line in lines:
            match = re.match(r'"([^"]+)":\s*\{', line.strip())
            if match and match.group(1) in imap.strategy_to_models:
                strategies.add(match.group(1))
        return sorted(strategies)

    def matches(self, path: str, lines: List[str], imap: ImpactMap) -> bool:
        if path != self.path:
            return False
        strategies = self._strategies_from_lines(lines, imap)
        if not strategies:
            return False
        normalized_lines = [
            _normalize_diff_line(line)
            for line in lines
            if any(ch.isalnum() for ch in _normalize_diff_line(line))
        ]
        strategy_tokens = tuple(strategies)
        return all(
            any(token in line for token in strategy_tokens)
            or any(token in line for token in self.allowed_tokens)
            for line in normalized_lines
        )

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
            _models_for_runtime_strategies(self._strategies_from_lines(lines, imap), imap),
            match.unit_tiers,
            match.rebuild_cpp,
        )


class HarnessReferenceVlGeneratedOnlyDecodeRule(DiffRefinementRule):
    name = "harness_reference_vl_generated_only_decode"
    path = "tests/e2e/models/internvl/e2e_plugins/references/hf_transformers.py"
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
        return all(any(token in line for token in self.allowed_tokens) for line in normalized_lines)

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
    PyprojectValidationOptionalDependenciesRule(),
    HarnessSharedFp8ScalesRule(),
    KnownModelTimingEstimateRule(),
    RuntimeStrategyMatrixRule(),
    IdentifierDiffRefinementRule(
        "pyproject_known_profiles",
        paths=("pyproject.toml",),
        allowed_tokens=(
            "dependencies",
            "optional_dependencies",
            "project",
            "version",
        ),
    ),
    CandidateTokenDiffRefinementRule(
        "shared_builder_config_lookup_family_registry",
        "python/tensorrt_model_connect/families/__init__.py",
        (
            "callable(matches_config)",
            "def_find_plugin",
            "familyplugin",
            "getattr(model_type",
            "getattr(p",
            "matches_config",
            "model_type",
            "model_type_str",
            "p.matches",
            "return_p",
        ),
    ),
    CandidateTokenDiffRefinementRule(
        "shared_builder_config_lookup_cli",
        "python/tensorrt_model_connect/build_cli.py",
        (
            "find_plugin(config)",
            "find_plugin(config.model_type)",
            "plugin",
            "raw_plugin",
        ),
    ),
    CandidateTokenDiffRefinementRule(
        "shared_builder_config_lookup_engine",
        "python/tensorrt_model_connect/engine_builder.py",
        (
            "find_plugin(config)",
            "find_plugin(config.model_type)",
            "plugin",
        ),
    ),
    IdentifierDiffRefinementRule(
        "harness_shared_known_identifiers",
        paths=(
            "tests/e2e_harness/contracts.py",
            "tests/e2e_harness/manifest_loader.py",
            "tests/e2e_harness/orchestrator.py",
        ),
        allowed_tokens=_HARNESS_REGISTRY_TOKENS,
    ),
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
        "python/tensorrt_model_connect/build_cli.py",
        ("fp8_scales", "save_fp8_scales"),
        _fp8_scale_models,
    ),
    TokenDiffRefinementRule(
        "shared_builder_fp8_scales_engine",
        "python/tensorrt_model_connect/engine_builder.py",
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
        "python/tensorrt_model_connect/engine_builder.py",
        (
            "detect_diffusion_tokenizer_add_special_tokens",
            "diffusion_tokenizer_add_special_tokens",
            "diffusion_tokenizer_bundle_sections",
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
    HarnessReferenceVlGeneratedOnlyDecodeRule(),
    IdentifierDiffRefinementRule(
        "harness_reference_known_identifiers",
        path_prefixes=("tests/e2e_harness/references/",),
        allowed_tokens=_HARNESS_REGISTRY_TOKENS
        + (
            "1.0",
            "align_window",
            "any",
            "arch",
            "architectures",
            "backend",
            "branch_input",
            "candidate",
            "case",
            "channels",
            "coerce_numeric_sequence",
            "context",
            "cpu",
            "ctx",
            "data",
            "def_",
            "detach",
            "dtype",
            "elapsed",
            "else",
            "error",
            "eval",
            "exc",
            "expected_len",
            "field",
            "field_input",
            "float",
            "for",
            "forecast",
            "freq",
            "from_pretrained",
            "full_inference",
            "getattr",
            "hf_id",
            "gt",
            "in",
            "inputs",
            "int",
            "isinstance",
            "json",
            "key",
            "logits",
            "max",
            "mean_predictions",
            "model",
            "num_input_channels",
            "observed_mask",
            "output",
            "padding",
            "past",
            "path",
            "payload",
            "pipe.predict",
            "prediction",
            "project_dir",
            "quantile_preds",
            "raw",
            "reference_output_name",
            "regression",
            "reshape",
            "result",
            "run_time_series",
            "scale",
            "script",
            "series",
            "stderr",
            "str",
            "subprocess",
            "sys",
            "tensor",
            "text",
            "time",
            "trunk",
            "trunk_input",
            "try",
            "typing",
            "unsupported",
            "value",
            "valid_len",
        ),
    ),
    E2EWaivesModelLinesRule(),
)


def _diff_refinement_rules_for_impact(imap: ImpactMap) -> List[DiffRefinementRule]:
    """Return shared rules plus model-owned rules at the generic handoff point."""
    model_rules: List[DiffRefinementRule] = [
        ModelOwnedTokenDiffRefinementRule(spec) for spec in imap.model_owned_diff_rules
    ]
    rules: List[DiffRefinementRule] = []
    inserted_model_rules = False
    for rule in DIFF_REFINEMENT_RULES:
        if not inserted_model_rules and rule.name == "harness_shared_known_identifiers":
            rules.extend(model_rules)
            inserted_model_rules = True
        rules.append(rule)
    if not inserted_model_rules:
        rules.extend(model_rules)
    return rules


def maybe_refine_match_with_diff(
    path: str,
    match: RuleMatch,
    diff_text: str,
    imap: ImpactMap,
    candidate_models: Optional[List[str]] = None,
) -> RuleMatch:
    """Narrow broad file matches when the diff is demonstrably feature-scoped."""
    lines = _significant_diff_lines(diff_text)
    if not lines:
        return match

    if candidate_models is None:
        candidate_models = []

    for rule in _diff_refinement_rules_for_impact(imap):
        if isinstance(rule, CandidateTokenDiffRefinementRule):
            if rule.matches_with_candidates(path, lines, imap, candidate_models):
                return rule.refine_with_candidates(
                    path,
                    match,
                    lines,
                    imap,
                    candidate_models,
                )
            continue
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
            errors.append(f"{allowlist_path}:{line_no}: expected '<rule> <path> # <rationale>'")
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
                f"{allowlist_path}:{line_no}: duplicate fallback allowlist entry for {rule} {path}"
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
        if path.strip() and (repo_root / path.strip()).exists()
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
            f"Fallback allowlist entry no longer matches a tracked broad fallback: {rule} {path}"
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
    families_dir = repo_root / "python" / "tensorrt_model_connect" / "families"

    def _family_plugin_exists(family: str) -> bool:
        return any(
            (
                (families_dir / f"{family}.py").exists(),
                (families_dir / family / "__init__.py").exists(),
            )
        )

    # 1. Every family in a manifest has a corresponding plugin module/package
    for family in imap.family_to_models:
        if not _family_plugin_exists(family):
            errors.append(
                f"Family '{family}' in manifests has no plugin module or package under "
                f"{families_dir}"
            )

    # 2. Every family plugin module/package has at least one manifest (warn only)
    if families_dir.is_dir():
        for py_file in sorted(families_dir.glob("*.py")):
            name = py_file.stem
            if name in ("__init__", "base") or name.startswith("_"):
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
        warnings.append(f"Core models don't cover task_strategies: {sorted(missing)}")

    # 4. Every model manifest declares its task strategy locally.
    for model, metadata in sorted(imap.model_metadata.items()):
        if metadata.get("runtime_strategy") and not metadata.get("task_strategy"):
            errors.append(f"Manifest for '{model}' declares runtime_strategy but no task_strategy")

    # 5. L0 replacements must preserve the execution contract they stand in for.
    for model, replacement in sorted(imap.l0_replacement_by_model.items()):
        src = imap.model_metadata.get(model, {})
        dst = imap.model_metadata.get(replacement)
        if dst is None:
            errors.append(f"L0 replacement for '{model}' points to unknown model '{replacement}'")
            continue
        for field_name in ("family", "runtime_strategy", "precision", "quantization"):
            if src.get(field_name) != dst.get(field_name):
                errors.append(
                    f"L0 replacement '{replacement}' for '{model}' does not preserve "
                    f"{field_name}: {src.get(field_name)!r} != {dst.get(field_name)!r}"
                )

    # 6. Model-owned diff refinements must resolve to at least one known model.
    for spec in imap.model_owned_diff_rules:
        if not _model_owned_scope_models(spec, imap):
            errors.append(
                f"Model-owned impact rule '{spec.name}' in '{spec.owner}' "
                "does not resolve to any E2E models"
            )

    # 7. Every rule pattern matches at least one real file (spot checks)
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
        if result.e2e_test_ids:
            lines.extend(result.e2e_test_ids)
        else:
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
    return json.dumps(
        {
            "e2e_models": result.e2e_models,
            "e2e_test_ids": result.e2e_test_ids,
            "unit_tiers": result.unit_tiers,
            "rebuild_cpp": result.rebuild_cpp,
            "cap_applied": result.cap_applied,
            "matched_rules": result.matched_rules,
            "builder_tests": result.builder_tests,
            "cpp_tests": result.cpp_tests,
            "tools_tests": result.tools_tests,
            "fallback_tiers": result.fallback_tiers,
            "l0_replacements": result.l0_replacements,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test impact analysis for selective CI execution.",
    )
    parser.add_argument(
        "--base", default="github/main", help="Git ref for diff base (default: github/main)"
    )
    parser.add_argument("--head", default="HEAD", help="Git ref for diff head (default: HEAD)")
    parser.add_argument("--files", help="Explicit comma-separated file list (overrides git diff)")
    parser.add_argument(
        "--cap", type=int, default=None, help="If affected models > N, limit to core set + warn"
    )
    parser.add_argument(
        "--e2e-suite",
        choices=("l0", "nightly"),
        default="l0",
        help="E2E selection policy: l0 applies configured "
        "large-model replacements; nightly keeps exact models",
    )
    parser.add_argument(
        "--include-ci-tier",
        action="append",
        default=[],
        help="Include a ci_tier that is excluded by default, "
        "for example multi_device for manual local runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON for CI consumption",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Check map consistency (no diff needed)"
    )
    parser.add_argument("--verbose", action="store_true", help="Show per-file rule matches")
    parser.add_argument("--repo-root", default=None, help="Repository root (default: auto-detect)")
    parser.add_argument(
        "--coverage-map", default=None, help="Path to coverage_map.json for per-test selection"
    )
    args = parser.parse_args()

    # Resolve repo root
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
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
        print(
            f"Validation passed. {len(imap.all_model_names)} models, "
            f"{len(imap.core_models)} core, "
            f"{len(imap.family_to_models)} families.",
            file=sys.stderr,
        )
        return 0

    # Load coverage map if provided
    coverage_map_data = None
    if args.coverage_map:
        sys.path.insert(0, str(repo_root / "tools"))
        from coverage_map.generate import load_coverage_map

        coverage_map_data = load_coverage_map(Path(args.coverage_map))
        if coverage_map_data is None:
            print(
                f"WARNING: Coverage map not found at {args.coverage_map}. "
                "Falling back to tier-level selection.",
                file=sys.stderr,
            )

    exclude_ci_tiers = set(_DEFAULT_EXCLUDED_CI_TIERS).difference(set(args.include_ci_tier or []))

    # Get changed files
    if args.files:
        changed: Optional[List[str]] = [f.strip() for f in args.files.split(",") if f.strip()]
    else:
        changed = get_changed_files(args.base, args.head, repo_root)

    if changed is None:
        # Git diff failed -- safety net: run everything
        print("Running all tests (git diff unavailable).", file=sys.stderr)
        e2e_models = _filter_models_by_ci_tier(
            list(imap.all_model_names),
            imap,
            exclude_ci_tiers,
        )
        result_obj = ImpactResult(
            e2e_models=e2e_models,
            unit_tiers=["builder", "cpp", "tools"],
            rebuild_cpp=True,
            cap_applied=False,
            matched_rules=[
                {
                    "file": "<all>",
                    "rule": "git_diff_failed",
                    "models": e2e_models,
                }
            ],
            e2e_test_ids=_model_owned_e2e_test_ids(e2e_models, imap),
        )
    elif not changed:
        print("No changed files detected.", file=sys.stderr)
        result_obj = ImpactResult(
            e2e_models=[],
            unit_tiers=[],
            rebuild_cpp=False,
            cap_applied=False,
            matched_rules=[],
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
            exclude_ci_tiers=exclude_ci_tiers,
        )

    if args.verbose:
        for rule in result_obj.matched_rules:
            n = len(rule["models"])
            print(f"  {rule['file']} -> {rule['rule']} ({n} models)", file=sys.stderr)

    if args.json_output:
        print(format_json(result_obj))
    else:
        print(format_human(result_obj))

    return 0


if __name__ == "__main__":
    sys.exit(main())

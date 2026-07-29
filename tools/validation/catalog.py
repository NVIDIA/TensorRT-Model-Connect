# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve validation workloads and model manifests behind one catalog interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import warnings

from tests.e2e_harness.manifest_loader import iter_manifest_paths, load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "tests" / "e2e" / "models"


def load_structured_file(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML catalog with a mapping at its root."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        raise RuntimeError("PyYAML is required for validation YAML files") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def load_suites(path: Path = DEFAULT_SUITES) -> list[dict[str, Any]]:
    """Load validated, independent copies of every workload suite."""
    path = Path(path)
    data = load_structured_file(path)
    suites = data.get("suites", [])
    if not isinstance(suites, list):
        raise ValueError(f"{path}: 'suites' must be a list")
    suites = copy.deepcopy(suites)
    seen: set[str] = set()
    for suite in suites:
        suite_id = suite.get("id")
        if not suite_id or not isinstance(suite_id, str):
            raise ValueError(f"{path}: every suite needs a string id")
        if suite_id in seen:
            raise ValueError(f"{path}: duplicate suite id {suite_id!r}")
        seen.add(suite_id)
    return suites


def suite_by_id(suites: list[dict[str, Any]], suite_id: str) -> dict[str, Any]:
    """Resolve one workload suite or report all known IDs."""
    for suite in suites:
        if suite["id"] == suite_id:
            return suite
    known = ", ".join(sorted(suite["id"] for suite in suites))
    raise ValueError(f"Unknown suite {suite_id!r}. Known suites: {known}")


def infer_reference_family(raw: dict[str, Any]) -> str:
    return str(raw.get("reference_family", "") or "")


def infer_user_contract(raw: dict[str, Any], reference_family: str) -> str:
    del reference_family
    return str(raw.get("user_contract", "") or "")


def manifest_record(path: Path) -> dict[str, Any]:
    """Project one E2E manifest onto the fields needed by validation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    model_name = str(raw.get("name", path.stem))
    testcases = raw.pop("testcases", [])
    if isinstance(testcases, list) and testcases:
        canonical = next(
            (
                testcase
                for testcase in testcases
                if isinstance(testcase, dict) and testcase.get("name") == model_name
            ),
            testcases[0],
        )
        if isinstance(canonical, dict):
            raw = {**raw, **canonical, "name": model_name}
    build_args = raw.get("build_args", {})
    task_eval_config = raw.get("task_eval", {})
    if not isinstance(task_eval_config, dict):
        task_eval_config = {}
    runtime_strategy = str(raw.get("runtime_strategy") or "")
    task_strategy = str(raw.get("task_strategy") or runtime_strategy)
    reference_family = str(task_eval_config.get("reference_family") or infer_reference_family(raw))
    reference_backend = str(
        task_eval_config.get("reference_backend") or raw.get("reference_backend", "") or ""
    )
    if not reference_backend:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                reference_backend = load_manifest(path).reference_backend
        except Exception:
            reference_backend = "hf_transformers"
    user_contract = str(
        task_eval_config.get("user_contract") or infer_user_contract(raw, reference_family)
    )
    distributed = raw.get("distributed_runtime", {})
    requires_multi_device = bool(distributed.get("enabled")) or (
        str(raw.get("ci_tier", "")) == "multi_device"
    )
    return {
        "name": model_name,
        "manifest": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "hf_id": raw.get("hf_id") or raw.get("model_id", ""),
        "hf_revision": str(raw.get("hf_revision", "") or ""),
        "bundle": raw.get("bundle", f"{path.stem}.trtfb"),
        "family": raw.get("family", ""),
        "runtime_strategy": runtime_strategy,
        "task_strategy": task_strategy,
        "reference_family": reference_family,
        "reference_backend": reference_backend,
        "execution_profiles": (
            raw.get("execution_profiles", {})
            if isinstance(raw.get("execution_profiles", {}), dict)
            else {}
        ),
        "user_contract": user_contract,
        "ci_tier": raw.get("ci_tier", "default"),
        "core": bool(raw.get("core", False)),
        "skip": raw.get("skip", ""),
        "requires_multi_device": requires_multi_device,
        "l0_replacement": raw.get("l0_replacement", ""),
        "gated": bool(raw.get("gated", False)),
        "trust_remote_code": bool(raw.get("trust_remote_code", False)),
        "max_cache_length": raw.get(
            "max_cache_length",
            (build_args.get("max_cache_length", 256) if isinstance(build_args, dict) else 256),
        ),
        "precision": raw.get("precision", "fp32"),
        "fp32_layers": raw.get("fp32_layers", []),
        "quantization": raw.get("quantization", {}),
        "build_args": build_args if isinstance(build_args, dict) else {},
        "fp8_scales": raw.get("fp8_scales", ""),
        "image_height": raw.get("image_height"),
        "image_width": raw.get("image_width"),
        "video_num_frames": raw.get("video_num_frames"),
        "video_height": raw.get("video_height"),
        "video_width": raw.get("video_width"),
        "num_inference_steps": raw.get("num_inference_steps"),
        "task_eval": task_eval_config,
        "runtime_config": (
            raw.get("runtime_config", {}) if isinstance(raw.get("runtime_config", {}), dict) else {}
        ),
        "max_new_tokens": raw.get("max_new_tokens"),
    }


def load_manifest_records(
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> list[dict[str, Any]]:
    """Load the validation projection for every discovered model manifest."""
    return [manifest_record(path) for path in iter_manifest_paths(models_dir)]


def _selector_values(selectors: dict[str, Any], key: str) -> set[str]:
    values = selectors.get(key, [])
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def suite_match_reason(suite: dict[str, Any], model: dict[str, Any]) -> tuple[bool, str]:
    """Explain whether a model satisfies a workload suite's selectors."""
    selectors = suite.get("selectors", {})
    model_names = _selector_values(selectors, "model_names")
    exclude_model_names = _selector_values(selectors, "exclude_model_names")
    task_strategies = _selector_values(selectors, "task_strategies")
    runtime_strategies = _selector_values(selectors, "runtime_strategies")
    user_contracts = _selector_values(selectors, "user_contracts")
    exclude_user_contracts = _selector_values(selectors, "exclude_user_contracts")
    families = _selector_values(selectors, "families")
    exclude_families = _selector_values(selectors, "exclude_families")

    if model_names and model["name"] not in model_names:
        return False, f"model={model['name']} not selected"
    if exclude_model_names and model["name"] in exclude_model_names:
        return False, f"model={model['name']} excluded"
    if task_strategies and model["task_strategy"] not in task_strategies:
        return False, f"task_strategy={model['task_strategy']} not selected"
    if runtime_strategies and model["runtime_strategy"] not in runtime_strategies:
        return False, (f"runtime_strategy={model['runtime_strategy']} not selected")
    if user_contracts and model["user_contract"] not in user_contracts:
        return False, (f"user_contract={model['user_contract'] or '<empty>'} not selected")
    if exclude_user_contracts and model["user_contract"] in exclude_user_contracts:
        return False, f"user_contract={model['user_contract']} excluded"
    if families and model["family"] not in families:
        return False, f"family={model['family']} not selected"
    if exclude_families and model["family"] in exclude_families:
        return False, f"family={model['family']} excluded"
    if model.get("skip"):
        return False, f"manifest skip: {model['skip']}"
    return True, "selected"


def resolve_suite_for_model(suite: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Resolve manifest dimensions and family/model profiles for one run."""
    resolved = copy.deepcopy(suite)
    generation = dict(resolved.get("generation", {}))
    for key in (
        "image_height",
        "image_width",
        "video_num_frames",
        "video_height",
        "video_width",
        "num_inference_steps",
    ):
        value = model.get(key)
        if value is not None:
            generation[key] = value

    family_profiles = resolved.get("family_profiles", {})
    if not isinstance(family_profiles, dict):
        raise ValueError(f"Suite {suite['id']} family_profiles must be a mapping")
    family_profile = family_profiles.get(str(model.get("family", "")), {})
    if not isinstance(family_profile, dict):
        raise ValueError(
            f"Suite {suite['id']} family profile for {model.get('family')} must be a mapping"
        )

    profiles = resolved.get("model_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Suite {suite['id']} model_profiles must be a mapping")
    profile = profiles.get(str(model.get("name", "")), {})
    if not isinstance(profile, dict):
        raise ValueError(f"Suite {suite['id']} profile for {model.get('name')} must be a mapping")
    for owner, source in (
        (model.get("family"), family_profile),
        (model.get("name"), profile),
    ):
        profile_generation = source.get("generation", {})
        if not isinstance(profile_generation, dict):
            raise ValueError(
                f"Suite {suite['id']} generation profile for {owner} must be a mapping"
            )
        generation.update(profile_generation)
    resolved["generation"] = generation

    gates = dict(resolved.get("gates", {}))
    for owner, source in (
        (model.get("family"), family_profile),
        (model.get("name"), profile),
    ):
        profile_gates = source.get("gates", {})
        if not isinstance(profile_gates, dict):
            raise ValueError(f"Suite {suite['id']} gate profile for {owner} must be a mapping")
        gates.update(profile_gates)
    resolved["gates"] = gates
    return resolved

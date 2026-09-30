# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification configuration of a catalog model, derived rather than written.

A model's configuration is its catalog entry (checkpoint, revision, precision, bundle, Task,
sequence limit) combined with the defaults of its Task (config/tasks.yaml). The benchmark
qualification case, when the family has one, only tells whether the native reference needs the
family's own script. config/models/<profile>.yaml holds exceptions and is deep-merged last.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import CONFIG_ROOT, ConfigError, Environment, load_suite

# Operations the generic Hugging Face reference adapters serve (trtmc_perf_serving.backends.reference).
HF_OPERATIONS = {"generate", "translate", "encode", "embed", "rerank", "classify", "detect", "segment",
                 "segment_prompted", "extract_features", "transcribe", "generate_audio", "generate_image",
                 "solve", "regress"}
# Tasks whose inputs the generic adapters would silently ignore (an edit image, a world-model
# context, a text prompt for masks): only the family's declared reference is valid for them.
SCRIPT_ONLY_TASKS = {"image_edit", "world_model_generation", "text_prompted_segmentation"}
PRECISIONS = ("fp16", "bf16", "fp32")
# Checkpoints above this size (bytes) measure fewer requests per run: native references of large models
# take seconds per request, and three runs still give the confidence interval.
LARGE_CHECKPOINT_BYTES = 16 * 2**30
LARGE_MODEL_MEASUREMENT = {"warmup": 1, "requests": 5, "runs": 3}


def checkpoint_bytes(hf_id: str, revision: str | None) -> int | None:
    """Total tensor size from the cached safetensors index (None when unknown)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        index = try_to_load_from_cache(hf_id, "model.safetensors.index.json", revision=revision)
        if isinstance(index, str):
            return int(json.loads(Path(index).read_text()).get("metadata", {}).get("total_size", 0)) or None
    except Exception:  # noqa: BLE001 - size is only a measurement hint
        return None
    return None
GENERATION_MARGIN = 8  # tokens kept free for BOS/template tokens when bounding prompts


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _import_repository(repository: Path) -> None:
    for root in (repository, repository / "core/builder", repository / "apps/benchmark"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def catalog_profiles(repository: Path) -> list[Any]:
    """Ready catalog profiles, excluding L0 smoke and regression variants (release coverage rule)."""
    _import_repository(repository)
    from trtmc_benchmark.catalog import ManifestCatalog

    return [entry for entry in ManifestCatalog(repository / "families").entries()
            if entry.status == "ready" and "-l0" not in f"-{entry.name}-" and "-regression-" not in f"-{entry.name}-"]


def _qualification_cases(repository: Path, profile: str, kind: str) -> list[Mapping[str, Any]]:
    _import_repository(repository)
    from qualification_tests.benchmark_qualification.catalog import discover

    return [{**dict(case.values), "candidate": dict(case.candidate)}
            for case in discover(repository) if case.model == profile and case.kind == kind]


def _qualification_case(repository: Path, profile: str) -> Mapping[str, Any] | None:
    cases = _qualification_cases(repository, profile, "performance")
    return cases[0] if cases else None


def _reference_backend(operation: str, task: str, declared: Mapping[str, Any] | None) -> tuple[str, str | None]:
    """(backend, fallback). The generic adapter serves every operation it supports; when it cannot
    load or run the model, the family's declared qualification reference takes over."""
    script = "script" if declared is not None else None
    if operation not in HF_OPERATIONS or task in SCRIPT_ONLY_TASKS:
        return (script or "unsupported"), None
    return "reference", script


def _suite(reference: str | Mapping[str, Any], profile: str, repository: Path) -> dict[str, Any]:
    if isinstance(reference, Mapping):
        suite = dict(reference)
    elif reference == "catalog":
        return {"suite": f"{profile}-catalog", "version": 1,
                "source": {"kind": "catalog_testcase", "profile": profile},
                "selection": {"method": "first", "count": 1}}
    else:
        suite = load_suite(reference)
    if suite.pop("base", None) == "catalog":  # samples override the profile's catalog request
        suite["base_profile"] = profile
    if suite["source"].get("kind") == "etth1_windows":
        # The family's qualification case declares the window its model forecasts from.
        declared = [case["window"] for case in _qualification_cases(repository, profile, "accuracy")
                    if isinstance(case.get("window"), Mapping)]
        if declared:
            suite["source"] = {**suite["source"], "window": dict(declared[0])}
    return suite


def _bound_generation(suite: dict[str, Any], item: Mapping[str, Any], catalog_model: Any,
                      trust_remote_code: bool) -> dict[str, Any]:
    """Fit text prompts and generated tokens into the bundle's sequence limit."""
    limit = catalog_model.build_settings.get("max_sequence_length")
    wanted = item.get("max_new_tokens")
    if not wanted:
        return suite
    new_tokens = int(wanted) if not limit else max(4, min(int(wanted), int(limit) // 4))
    suite["request"] = {**suite.get("request", {}), "max_new_tokens": new_tokens}
    if limit:
        suite["truncate_prompt"] = {"tokenizer": catalog_model.hf_id, "revision": catalog_model.hf_revision or None,
                                    "trust_remote_code": trust_remote_code,
                                    "max_tokens": int(limit) - new_tokens - GENERATION_MARGIN}
    return suite


def resolve_model(profile: str, environment: Environment, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    repository = environment.path("repo")
    entries = {entry.name: entry for entry in catalog_profiles(repository)}
    if profile not in entries:
        raise ConfigError(f"{profile} is not a ready catalog profile")
    _import_repository(repository)
    from trtmc_benchmark.catalog import ManifestCatalog

    catalog_model = ManifestCatalog(repository / "families").resolve(profile)
    manifest = json.loads(Path(catalog_model.manifest_path).read_text())
    tasks = yaml.safe_load((root / "tasks.yaml").read_text())
    if catalog_model.task not in tasks["tasks"]:
        raise ConfigError(f"no Task defaults for {catalog_model.task!r} ({profile})")
    override_path = root / "models" / f"{profile}.yaml"
    override = (yaml.safe_load(override_path.read_text()) if override_path.is_file() else None) or {}
    task = dict(tasks["tasks"][catalog_model.task])
    # Quantized candidates (the catalog's `quantization`, or a quantized checkpoint declared in
    # config/models) are held to the Task's quantization tolerance instead of exact parity.
    quantization = (override.get("candidate") or {}).get("quantization") or manifest.get("quantization")
    tolerance = task.pop("quantized", None)
    config = deep_merge(deep_merge(tasks["defaults"], task), tolerance if quantization and tolerance else {})
    config = deep_merge(config, override)

    operation = config.get("operation") or entries[profile].operation
    trust_remote_code = bool(manifest.get("trust_remote_code", False))
    case = _qualification_case(repository, profile)
    declared = dict(case.get("reference", {})) if case else None
    # Remote code the family's accepted qualification already trusts may run in the generic adapters too.
    trust_remote_code = trust_remote_code or bool((case or {}).get("candidate", {}).get("trust_remote_code"))
    backend, fallback = _reference_backend(operation, catalog_model.task, declared)
    candidate_precision = catalog_model.precision
    perf_precision = candidate_precision if candidate_precision in PRECISIONS else "bf16"
    reference = config.get("reference", {})
    if reference.get("backend", "auto") != "auto":
        backend, fallback = reference["backend"], reference.get("fallback")
    # fp32 goldens make every candidate precision (fp16, bf16) comparable to a precision-independent
    # truth and give it a noise floor; the declared reference precision is the fallback.
    golden_precision = reference.get("precision") or "fp32"
    declared_precision = (declared or {}).get("precision")

    accuracy = []
    for item in config.get("accuracy", []):
        suite = _suite(item["suite"], profile, repository)
        if item.get("request"):  # per-model request settings, e.g. a runtime's decoding convention
            suite = {**suite, "request": {**suite.get("request", {}), **item["request"]}}
        suite = _bound_generation(suite, item, catalog_model, trust_remote_code)
        accuracy.append({**{key: value for key, value in item.items() if key not in ("max_new_tokens", "request")},
                         "suite": suite, "reference_mode": "eager"})
    l1 = dict(config["performance"]["l1"])
    l1["suite"] = _suite(l1["suite"], profile, repository)
    size = checkpoint_bytes(catalog_model.hf_id, catalog_model.hf_revision or None)
    if size and size > LARGE_CHECKPOINT_BYTES and l1["measurement"]["requests"] > LARGE_MODEL_MEASUREMENT["requests"]:
        l1["measurement"] = dict(LARGE_MODEL_MEASUREMENT)
        l1["measurement_reason"] = f"checkpoint {size / 2**30:.0f} GiB > {LARGE_CHECKPOINT_BYTES / 2**30:.0f} GiB"
    if backend == "script" and case:
        # Family reference scripts validate request settings the catalog testcase may omit (for
        # example MoGe's num_tokens): add the scalar settings of the qualification request.
        extra = {key: value for key, value in (case.get("request") or {}).items()
                 if not key.endswith("_path") and not isinstance(value, (dict, list))}
        for item in [*accuracy, l1]:
            if item["suite"]["source"].get("kind") == "catalog_testcase":
                item["suite"] = {**item["suite"], "request": {**extra, **item["suite"].get("request", {})}}
    # candidate.build overrides catalog manifest fields for the qualified bundle (a new name keeps it
    # apart from the catalog bundle); a family-prepared model directory replaces the checkpoint.
    candidate = dict(config.get("candidate", {}))
    build = dict(candidate.pop("build", None) or {})
    bundle = f"{build.get('name', profile)}/{build.get('bundle', catalog_model.bundle_name)}"
    return {
        "model": profile, "catalog_profile": profile, "operation": operation, "family": catalog_model.family,
        "task": catalog_model.task,
        "candidate": {"bundle": bundle, "precision": candidate_precision, "build": build,
                      "manifest": str(catalog_model.manifest_path), "checkpoint": catalog_model.hf_id,
                      "revision": catalog_model.hf_revision or None, "quantization": quantization,
                      "model_directory": (case or {}).get("candidate", {}).get("model_directory"), **candidate},
        "reference": {**{key: value for key, value in reference.items() if key not in ("backend", "fallback")},
                      "backend": backend, "fallback": fallback, "precision": golden_precision,
                      "declared_precision": declared_precision if declared_precision != golden_precision else None,
                      "perf_precision": perf_precision, "trust_remote_code": trust_remote_code},
        "noise_floor": bool(config.get("noise_floor", True)) and perf_precision != golden_precision,
        "accuracy": accuracy,
        "performance": {"l1": l1},
    }


def checkpoints(model: Mapping[str, Any]) -> set[str]:
    """Hugging Face repositories a model's run downloads: its checkpoint and a separate reference model."""
    names = {model["candidate"].get("checkpoint"), model.get("reference", {}).get("model")}
    return {name for name in names if name and not str(name).startswith("/")}

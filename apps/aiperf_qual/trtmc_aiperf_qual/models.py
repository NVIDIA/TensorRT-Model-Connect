# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification configuration of a catalog model, derived rather than written.

A model's configuration is its catalog entry (checkpoint, revision, precision, bundle, Task) combined
with the defaults of its Task (config/tasks.yaml). The candidate is the catalog bundle as shipped;
accuracy comes from the Task's gold-labelled benchmarks (``absolute``) and whole-output checks
(``supplementary``); a model with neither declares ``accuracy_source: none`` (Perf only). The native
reference is the generic Hugging Face adapter of the operation. config/models/<profile>.yaml holds the
exceptions (a reference environment, remote code, another native checkpoint, a build override) and is
deep-merged last.
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
# Tasks whose inputs the generic adapters would silently ignore (a world-model context, a text prompt for
# masks): only a family's own native adapter (``reference.adapter``) serves them.
NO_GENERIC_TASKS = {"world_model_generation", "text_prompted_segmentation"}
PRECISIONS = ("fp16", "bf16", "fp32")
# Checkpoints above this size (bytes) measure fewer requests per run: native references of large models
# take seconds per request, and three runs still give the confidence interval.
LARGE_CHECKPOINT_BYTES = 16 * 2**30
LARGE_MODEL_MEASUREMENT = {"settle_s": 0, "warmup": 1, "requests": 3, "runs": 5}


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


def _reference_backend(operation: str, task: str) -> str:
    """``reference`` where a generic adapter serves the operation, else ``unsupported``."""
    return "reference" if operation in HF_OPERATIONS and task not in NO_GENERIC_TASKS else "unsupported"


def _suite(reference: str | Mapping[str, Any], profile: str, catalog_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A suite definition; ``catalog_request`` (config/models ``request``) states controls the catalog request
    leaves to each side's default, in every suite built on that request (the suite's own request wins)."""
    if isinstance(reference, Mapping):
        suite = dict(reference)
    elif reference == "catalog":
        suite = {"suite": f"{profile}-catalog", "version": 1,
                 "source": {"kind": "catalog_testcase", "profile": profile},
                 "selection": {"method": "first", "count": 1}}
    else:
        suite = load_suite(reference)
    if suite.pop("base", None) == "catalog":  # samples override the profile's catalog request
        suite["base_profile"] = profile
    on_catalog = suite.get("base_profile") or (suite.get("source") or {}).get("kind") == "catalog_testcase"
    if catalog_request and on_catalog:
        suite["request"] = {**dict(catalog_request), **dict(suite.get("request") or {})}
    return suite


def model_suite(reference: str | Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    """``_suite`` for a resolved model (its catalog profile and stated catalog request)."""
    return _suite(reference, model["catalog_profile"], model.get("catalog_request"))


def _absolute(names: list[Any], definitions: Mapping[str, Any], testcase: Mapping[str, Any],
              quantized: bool) -> list[dict[str, Any]]:
    """Absolute-accuracy benchmarks (config/tasks.yaml ``benchmarks``; a model may name one with
    overrides, ``{name: gsm8k, limit: 300}``) as the model runs them: the chat route when its catalog
    request uses the chat template, the quantization or sampling gate, and one repetition per seed
    when the catalog request samples (top_k 1 is greedy)."""
    sampled = float(testcase.get("temperature") or 0.0) > 0.0 and int(testcase.get("top_k") or 0) != 1
    items = []
    for entry in names:
        overrides = dict(entry) if isinstance(entry, Mapping) else {"name": entry}
        name = overrides.pop("name", None)
        if name not in definitions:
            raise ConfigError(f"unknown absolute-accuracy benchmark {name!r}")
        item = {**definitions[name], **overrides}
        gates = {"quantized": item.pop("quantized_gate", None), "sampled": item.pop("sampled_gate", None)}
        seeds = item.pop("seeds_when_sampled", [1, 2, 3])
        if sampled:
            item.update(gate=gates["sampled"] or item["gate"], seeds=list(seeds))
        elif quantized and gates["quantized"]:
            item["gate"] = gates["quantized"]
        if item.get("window"):  # a forecast window (ETTh1): the model's columns, context, and horizon
            suite = item["suite_definition"]
            suite = load_suite(suite) if isinstance(suite, str) else dict(suite)
            window = {**dict(suite["source"].get("window") or {}), **item.pop("window")}
            item["suite_definition"] = {**suite, "source": {**suite["source"], "window": window}}
        if "min_native" in item:  # below it the native score is no baseline (half of chance)
            item["gate"] = {**item["gate"], "min_native": item.pop("min_native")}
        item.setdefault("endpoint", "chat" if testcase.get("use_chat_template") else "completions")
        items.append(item)
    return items


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
    config = deep_merge(deep_merge(tasks["defaults"], tasks["tasks"][catalog_model.task]), override)
    candidate = dict(config.get("candidate", {}))
    # Quantized candidates (the catalog's `quantization`, or a quantized checkpoint declared in
    # config/models) are held to the benchmarks' quantization gates.
    quantization = candidate.get("quantization") or manifest.get("quantization")
    operation = config.get("operation") or entries[profile].operation
    reference = dict(config.get("reference", {}))
    backend = reference.pop("backend", "auto")
    if backend == "auto":  # a family's own native pipeline serves any operation
        # ``not_covered``: a native pipeline the harness cannot run yet (the reason is reported): not-covered.
        backend = ("unsupported" if reference.get("not_covered") else
                   "reference" if reference.get("adapter") else _reference_backend(operation, catalog_model.task))
    trust_remote_code = bool(manifest.get("trust_remote_code", False) or reference.pop("trust_remote_code", False))
    candidate_precision = catalog_model.precision
    # One immutable checkpoint for the bundle and the native reference: the catalog pin, else
    # config/models' (the build then uses it too); without either both sides load the cached snapshot.
    revision = catalog_model.hf_revision or candidate.pop("revision", None) or None
    build = dict(candidate.pop("build", None) or {})
    if revision and not manifest.get("hf_revision"):
        build["hf_revision"] = revision
    if build:  # a bundle other than the catalog's: its own name, so the catalog bundle is never served instead
        build.setdefault("name", f"{profile}-qual")

    absolute = _absolute(list(config.get("absolute") or []), tasks.get("benchmarks", {}),
                         (manifest.get("testcases") or [{}])[0], bool(quantization))
    for item in absolute:  # gold suites: the suite definition, with the benchmark's request settings
        if item.get("metric"):
            suite = _suite(item["suite_definition"], profile, config.get("request"))
            if item.get("request"):
                suite = {**suite, "request": {**suite.get("request", {}), **item.pop("request")}}
            item["suite_definition"] = suite
    supplementary = [dict(item) for item in config.get("supplementary", [])]
    if config.get("accuracy_source") not in (None, "none"):
        raise ConfigError(f"{profile}: accuracy_source may only be `none` (Perf only)")
    if config.get("accuracy_source") == "none":
        if not config.get("accuracy_note"):
            raise ConfigError(f"{profile}: accuracy_source none needs an accuracy_note (why no accuracy applies)")
        accuracy_source, absolute, supplementary = "none", [], []
    elif absolute or supplementary:
        accuracy_source = "absolute"
    else:  # reported as an error: the Task's contract is not implemented yet
        accuracy_source = "missing"

    l1 = dict(config["performance"]["l1"])
    if isinstance(l1["suite"], Mapping) and l1["suite"].get("from_benchmark"):
        # The first problem of one of the model's benchmarks (a catalog testcase that is no workload,
        # for example a 28-value time series against a 512-step model).
        name = l1["suite"]["from_benchmark"]
        source = next((item for item in absolute if item.get("suite") == name or item.get("name") == name), None)
        if source is None or not isinstance(source.get("suite_definition"), Mapping):
            raise ConfigError(f"{profile}: performance.l1.suite.from_benchmark {name!r} is not one of its gold suites")
        l1["suite"] = {**source["suite_definition"], "suite": f"{profile}-{name}-first", "selection": {"method": "first", "count": 1}}
    l1["suite"] = _suite(l1["suite"], profile, config.get("request"))
    size = checkpoint_bytes(catalog_model.hf_id, revision)
    task_measurement = deep_merge(tasks["defaults"], tasks["tasks"][catalog_model.task])["performance"]["l1"]["measurement"]
    if size and size > LARGE_CHECKPOINT_BYTES and task_measurement["requests"] > LARGE_MODEL_MEASUREMENT["requests"]:
        # The class replaces the Task's measurement; the profile's own measurement settings still apply on top.
        stated = ((override.get("performance") or {}).get("l1") or {}).get("measurement") or {}
        l1["measurement"] = deep_merge(LARGE_MODEL_MEASUREMENT, stated)
        l1["measurement_reason"] = f"checkpoint {size / 2**30:.0f} GiB > {LARGE_CHECKPOINT_BYTES / 2**30:.0f} GiB"
    bundle = f"{build.get('name', profile)}/{build.get('bundle', catalog_model.bundle_name)}"
    reference_model = reference.pop("model", None)
    reference_revision = reference.pop("revision", None) or (None if reference_model else revision)
    return {
        "model": profile, "catalog_profile": profile, "operation": operation, "family": catalog_model.family,
        **({"catalog_request": dict(config["request"])} if config.get("request") else {}),
        "task": catalog_model.task,
        "candidate": {"bundle": bundle, "precision": candidate_precision, "build": build,
                      "manifest": str(catalog_model.manifest_path), "checkpoint": catalog_model.hf_id,
                      "revision": revision, "quantization": quantization,
                      "max_sequence_length": build.get("max_sequence_length")
                      or catalog_model.build_settings.get("max_sequence_length"), **candidate},
        "reference": {**reference, **({"model": reference_model} if reference_model else {}),
                      "revision": reference_revision, "backend": backend,
                      "precision": reference.get("precision") or "fp32",
                      "perf_precision": candidate_precision if candidate_precision in PRECISIONS else "bf16",
                      "trust_remote_code": trust_remote_code},
        "accuracy_source": accuracy_source,
        **({"accuracy_note": str(config["accuracy_note"])} if accuracy_source == "none" else {}),
        "absolute": absolute, "supplementary": supplementary,
        **({"coverage": str(config["coverage"])} if config.get("coverage") else {}),
        "performance": {"l1": l1, **({"l2": dict(config["performance"]["l2"])}
                                     if config["performance"].get("l2") else {})},
    }


def checkpoints(model: Mapping[str, Any]) -> set[str]:
    """Hugging Face repositories a model's run downloads: its checkpoint and a separate reference model."""
    names = {model["candidate"].get("checkpoint"), model.get("reference", {}).get("model")}
    return {name for name in names if name and not str(name).startswith("/")}

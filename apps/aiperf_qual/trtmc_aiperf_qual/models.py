# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification configuration of a catalog model, derived rather than written.

A model's configuration is its catalog entry (checkpoint, revision, precision, bundle, Task,
sequence limit) combined with the defaults of its Task (config/tasks.yaml). Accuracy comes from the
Task's gold-labelled benchmarks (``absolute``), else the family's own accuracy cases; a model with
neither declares ``accuracy_source: none`` (Perf only). The family's benchmark qualification case
also tells whether the native reference needs the family's own script. config/models/<profile>.yaml
holds exceptions and is deep-merged last.
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
# Tasks whose inputs the generic adapters would silently ignore (a world-model context, a text
# prompt for masks): only the family's declared reference is valid for them. Image edits run on the
# Diffusers adapter (it rejects an edit pipeline-less checkpoint, and the family reference takes over).
SCRIPT_ONLY_TASKS = {"world_model_generation", "text_prompted_segmentation"}
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

    return [{**dict(case.values), "candidate": dict(case.candidate), "name": case.name, "benchmark": case.benchmark}
            for case in discover(repository) if case.model == profile and case.kind == kind]


def _qualified_build(profile: str, manifest: Mapping[str, Any], case: Mapping[str, Any] | None) -> dict[str, Any]:
    """Manifest overrides of the family qualification's candidate (its own bundle when they differ)."""
    if not case:
        return {}
    candidate = case["candidate"]
    changed = {key: value for key, value in (candidate.get("build") or {}).items() if manifest.get(key) != value}
    if not changed and candidate.get("bundle", manifest.get("bundle")) == manifest.get("bundle"):
        return {}
    return {"name": f"{profile}-qual", "bundle": candidate.get("bundle", manifest.get("bundle")), **changed}


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
        if declared:  # ``window_overrides`` (for example every hour's window) apply on top
            window = {**dict(declared[0]), **dict(suite["source"].get("window_overrides") or {})}
            suite["source"] = {**suite["source"], "window": window}
    return suite


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
    task = dict(tasks["tasks"][catalog_model.task])
    # Quantized candidates (the catalog's `quantization`, or a quantized checkpoint declared in
    # config/models) are held to the benchmarks' quantization gates.
    quantization = (override.get("candidate") or {}).get("quantization") or manifest.get("quantization")
    config = deep_merge(deep_merge(tasks["defaults"], task), override)

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
    # The last precision the native model is timed at when the candidate's fails (timing_precisions).
    reference_precision = reference.get("precision") or "fp32"
    declared_precision = (declared or {}).get("precision")

    # A family's own accuracy cases (dataset, reference, metric, gate) qualify models whose Task has no
    # gold-labelled benchmark.
    family_cases = _qualification_cases(repository, profile, "accuracy")
    # Absolute accuracy: both sides scored against gold answers.
    absolute = _absolute(list(config.get("absolute") or []), tasks.get("benchmarks", {}),
                         (manifest.get("testcases") or [{}])[0], bool(quantization))
    for item in absolute:  # gold suites: the suite definition, with the benchmark's request settings
        if item.get("metric"):
            suite = _suite(item["suite_definition"], profile, repository)
            if item.get("request"):
                suite = {**suite, "request": {**suite.get("request", {}), **item.pop("request")}}
            item["suite_definition"] = suite
    if config.get("accuracy_source") not in (None, "none"):
        raise ConfigError(f"{profile}: accuracy_source may only be `none` (Perf only)")
    if config.get("accuracy_source") == "none":
        if not config.get("accuracy_note"):
            raise ConfigError(f"{profile}: accuracy_source none needs an accuracy_note (why no accuracy applies)")
        accuracy_source = "none"
    elif absolute or family_cases:
        accuracy_source = "absolute" if absolute else "family"
    else:
        raise ConfigError(f"{profile}: no accuracy scheme: its Task has no `absolute` benchmark and its family no "
                          "accuracy case (set accuracy_source: none with an accuracy_note for a Perf-only model)")
    if accuracy_source == "none":
        absolute = []
    l1 = dict(config["performance"]["l1"])
    if l1["suite"] == "catalog" and case and case.get("request") and not case["candidate"].get("model_directory"):
        # The family's performance workload (the request benchmark qualification times).
        l1["suite"] = {"suite": f"{profile}-qualification-perf", "version": 1,
                       "source": {"kind": "qualification_perf", "profile": profile},
                       "selection": {"method": "first", "count": 1}}
    l1["suite"] = _suite(l1["suite"], profile, repository)
    # One immutable checkpoint for the bundle and the native reference: the catalog pin, else the revision the family qualification pins for the same checkpoint.
    family_candidate = ((family_cases[0] if family_cases else case) or {}).get("candidate", {})
    revision = catalog_model.hf_revision or (
        family_candidate.get("revision") if family_candidate.get("checkpoint", catalog_model.hf_id) == catalog_model.hf_id
        else None) or None
    size = checkpoint_bytes(catalog_model.hf_id, revision)
    if size and size > LARGE_CHECKPOINT_BYTES and l1["measurement"]["requests"] > LARGE_MODEL_MEASUREMENT["requests"]:
        l1["measurement"] = dict(LARGE_MODEL_MEASUREMENT)
        l1["measurement_reason"] = f"checkpoint {size / 2**30:.0f} GiB > {LARGE_CHECKPOINT_BYTES / 2**30:.0f} GiB"
    if backend == "script" and case:
        # Family reference scripts validate request settings the catalog testcase may omit (for
        # example MoGe's num_tokens): add the scalar settings of the qualification request.
        extra = {key: value for key, value in (case.get("request") or {}).items()
                 if not key.endswith("_path") and not isinstance(value, (dict, list))}
        if l1["suite"]["source"].get("kind") in ("catalog_testcase", "qualification_perf"):
            l1["suite"] = {**l1["suite"], "request": {**extra, **l1["suite"].get("request", {})}}
    # candidate.build overrides catalog manifest fields for the qualified bundle (a new name keeps it
    # apart from the catalog bundle); a family-prepared model directory replaces the checkpoint.
    candidate = dict(config.get("candidate", {}))
    build = dict(candidate.pop("build", None) or _qualified_build(
        profile, manifest, family_cases[0] if family_cases else case))
    if revision and not manifest.get("hf_revision") and not build.get("hf_revision"):
        build = {"name": f"{profile}-qual", **build, "hf_revision": revision}  # build the pinned checkpoint
    # The benchmarks' prompts and answers must fit the bundle: a longer one is built under the -qual name.
    needed = max((int(item.get("sequence_length", 0)) for item in absolute), default=0)
    if needed and config.get("absolute_sequence_length"):  # a model whose longer bundles cannot run
        needed = min(needed, int(config["absolute_sequence_length"]))
    current = build.get("max_sequence_length") or catalog_model.build_settings.get("max_sequence_length")
    if needed and (not current or int(current) < needed):
        # A capped length gets its own name: bundles are reused by path, whatever their length.
        name = f"{profile}-qual-{needed}" if config.get("absolute_sequence_length") else f"{profile}-qual"
        build = {**build, "name": name, "max_sequence_length": needed}
    bundle = f"{build.get('name', profile)}/{build.get('bundle', catalog_model.bundle_name)}"
    # The native model the family declares (another checkpoint, e.g. a base model, or a Diffusers export)
    # and its pin; otherwise the candidate's own checkpoint at the candidate's revision.
    options = (declared or {}).get("adapter_options") or {}
    declared_model = (declared or {}).get("model") or options.get("model_id")
    declared_revision = (declared or {}).get("revision") or options.get("model_revision")
    reference_model = reference.get("model") or (declared_model if declared_model not in (None, catalog_model.hf_id)
                                                 else None)
    reference_revision = reference.get("revision") or (
        declared_revision if reference_model and reference_model == declared_model
        else None if reference_model else declared_revision if declared_model == catalog_model.hf_id and declared_revision
        else revision)
    return {
        "model": profile, "catalog_profile": profile, "operation": operation, "family": catalog_model.family,
        "task": catalog_model.task,
        "candidate": {"bundle": bundle, "precision": candidate_precision, "build": build,
                      "manifest": str(catalog_model.manifest_path), "checkpoint": catalog_model.hf_id,
                      "revision": revision, "quantization": quantization,
                      "max_sequence_length": build.get("max_sequence_length")
                      or catalog_model.build_settings.get("max_sequence_length"),
                      "model_directory": (case or {}).get("candidate", {}).get("model_directory"), **candidate},
        "reference": {**{key: value for key, value in reference.items() if key not in ("backend", "fallback")},
                      **({"model": reference_model} if reference_model else {}), "revision": reference_revision,
                      "backend": backend, "fallback": fallback, "precision": reference_precision,
                      "declared_precision": declared_precision if declared_precision != reference_precision else None,
                      "perf_precision": perf_precision, "trust_remote_code": trust_remote_code},
        "accuracy_source": accuracy_source,
        **({"accuracy_note": str(config["accuracy_note"])} if accuracy_source == "none" else {}),
        "absolute": absolute,
        "family_accuracy": [item["name"] for item in family_cases] if accuracy_source == "family" else [],
        # The family cases are reported but not judged where gold-referenced checks decide (generated media).
        "family_informational": bool(config.get("family_cases_informational")) and accuracy_source == "family",
        "supplementary": [dict(item) for item in config.get("supplementary", [])],
        **({"coverage": str(config["coverage"])} if config.get("coverage") else {}),
        "performance": {"l1": l1, **({"l2": dict(config["performance"]["l2"])}
                                     if config["performance"].get("l2") else {})},
    }


def checkpoints(model: Mapping[str, Any]) -> set[str]:
    """Hugging Face repositories a model's run downloads: its checkpoint and a separate reference model."""
    names = {model["candidate"].get("checkpoint"), model.get("reference", {}).get("model")}
    return {name for name in names if name and not str(name).startswith("/")}

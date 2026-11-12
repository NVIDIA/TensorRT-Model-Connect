# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification configuration of a catalog model, derived rather than written.

A model's configuration is its catalog entry (checkpoint, revision, precision, bundle, Task) combined
with the defaults of its Task (config/tasks.yaml). The candidate is the catalog bundle as shipped;
accuracy comes from the Task's gold-labelled benchmarks (``absolute``) and whole-output checks
(``supplementary``); a model with neither declares ``accuracy_source: none`` (Perf only). The native
reference is the generic Hugging Face adapter of the operation. Family-owned exceptions live in
families/<family>/tests/aiperf/<profile>.yaml. Legacy config/models/<profile>.yaml remains supported;
the family's definition is merged last. A quantized profile is qualified as its model's highest-precision
configuration (``qualified_as``): the same benchmarks, requests and native reference, its own candidate.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import CONFIG_ROOT, ConfigError, Environment, load_suite
from .compat import configuration

# Operations the generic Hugging Face reference adapters serve (trtmc_perf_serving.backends.reference).
HF_OPERATIONS = {"generate", "translate", "encode", "embed", "rerank", "classify", "detect", "segment",
                 "segment_prompted", "extract_features", "transcribe", "generate_audio", "generate_image",
                 "solve", "regress"}
# Tasks whose inputs the generic adapters would silently ignore (a world-model context, a text prompt for
# masks): only a family's own native adapter (``reference.adapter``) serves them.
NO_GENERIC_TASKS = {"world_model_generation", "text_prompted_segmentation"}
PRECISIONS = ("fp16", "bf16", "fp32")
# A configuration's numeric precision (its quantization, else its precision): a quantized checkpoint ranks below
# every unquantized one, and bf16, the usual checkpoint dtype, above fp16.
PRECISION_RANK = {"int4": 1, "nvfp4": 1, "int8": 2, "fp8": 2, "fp16": 3, "bf16": 4, "fp32": 5}
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
        item.setdefault("endpoint", "chat" if testcase.get("use_chat_template") else "completions")
        items.append(item)
    return items


def _definition(path: Path) -> Any:
    """A definition file's YAML. One that cannot be read or parsed is a ConfigError: an error of its profile alone,
    which the catalog listing (``selection.catalog``) and a campaign's other profiles survive."""
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ConfigError(f"{path}: {error}") from error


def profile_override(profile: str, family: str, repository: Path, root: Path = CONFIG_ROOT) -> dict[str, Any]:
    """A profile's own configuration: legacy config/models/<profile>.yaml, then the family's definition."""
    override_path = root / "models" / f"{profile}.yaml"
    override = configuration((_definition(override_path) if override_path.is_file() else None) or {})
    family_path = repository / "families" / family / "tests" / "aiperf" / f"{profile}.yaml"
    if family_path.is_file():
        owned = _definition(family_path)
        if not isinstance(owned, dict):
            raise ConfigError(f"{family_path} must contain a mapping")
        override = deep_merge(override, configuration(owned))
    return override


def _catalog_manifest(entry: Any) -> dict[str, Any]:
    return json.loads(Path(entry.model.manifest_path).read_text())


def qualified_as(profile: str, entries: Mapping[str, Any]) -> str:
    """The configuration a quantized profile is qualified as: its model's ready profile of the highest precision
    (``PRECISION_RANK``; the same family and Task, and the same name apart from precision and quantization,
    so qwen35-9b for qwen35-9b-nvfp4), else the profile itself (not quantized, or nothing ranks higher)."""
    def stem(name: str) -> str:
        return "-".join(part for part in name.split("-") if part not in PRECISION_RANK)

    def rank(manifest: Mapping[str, Any]) -> int:
        return PRECISION_RANK.get(manifest.get("quantization") or manifest.get("precision"), 0)

    own, manifest = entries[profile].model, _catalog_manifest(entries[profile])
    if not manifest.get("quantization"):
        return profile
    best, best_rank = profile, rank(manifest)
    for name, entry in sorted(entries.items()):
        if name == profile or (entry.model.family, entry.model.task, stem(name)) != (own.family, own.task, stem(profile)):
            continue
        candidate = rank(_catalog_manifest(entry))
        if candidate > best_rank:
            best, best_rank = name, candidate
    return best


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
    override = profile_override(profile, catalog_model.family, repository, root)
    base = qualified_as(profile, entries)
    if base != profile:  # the base's configuration under the profile's own; the candidate stays the profile's
        inherited = profile_override(base, catalog_model.family, repository, root)
        inherited.pop("candidate", None)
        override = deep_merge(inherited, override)
    config = deep_merge(deep_merge(configuration(tasks["defaults"]),
                        configuration(tasks["tasks"][catalog_model.task])), override)
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
    native_manifest = manifest
    if base != profile:
        native_manifest = _catalog_manifest(entries[base])
        if entries[base].model.hf_id != catalog_model.hf_id and not reference.get("model"):
            # A pre-quantized checkpoint (an -FP8 or -NVFP4 repository): the native reference loads the base's.
            reference.update(model=entries[base].model.hf_id, revision=entries[base].model.hf_revision or None)
    trust_remote_code = bool(native_manifest.get("trust_remote_code", False) or reference.pop("trust_remote_code", False))
    candidate_precision = catalog_model.precision
    # One immutable checkpoint for the bundle and the native reference: the catalog pin, else
    # config/models' (the build then uses it too); without either both sides load the cached snapshot.
    revision = catalog_model.hf_revision or candidate.pop("revision", None) or None
    build = dict(candidate.pop("build", None) or {})
    if revision and not manifest.get("hf_revision"):
        build["hf_revision"] = revision
    if build:  # a bundle other than the catalog's: its own name, so the catalog bundle is never served instead
        build.setdefault("name", f"{profile}-qual")

    definitions = deep_merge(tasks.get("benchmarks", {}), config.get("benchmarks", {}))
    # A quantized profile answers the benchmarks on its base's route (chat or completions), as the base does; the
    # sampling stays its own catalog request's, the request its server starts from.
    testcase = {**(manifest.get("testcases") or [{}])[0],
                "use_chat_template": (native_manifest.get("testcases") or [{}])[0].get("use_chat_template")}
    absolute = _absolute(list(config.get("absolute") or []), definitions, testcase, bool(quantization))
    for item in absolute:  # gold suites: the suite definition, with the benchmark's request settings
        if item.get("metric"):
            suite = _suite(item["suite_definition"], profile, config.get("request"))
            if item.get("request"):  # a null removes the suite's field (a Task that declares no such config)
                request = {**suite.get("request", {}), **item.pop("request")}
                suite = {**suite, "request": {key: value for key, value in request.items() if value is not None}}
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

    policy = dict(config["performance"])
    # The fixed request is timed in the native mode the quality runs use, unless the profile names its own.
    policy.setdefault("reference_modes", [reference.get("mode", "eager")])
    if isinstance(policy["suite"], Mapping) and policy["suite"].get("from_benchmark"):
        # The first problem of one of the model's benchmarks (a catalog testcase that is no workload,
        # for example a 28-value time series against a 512-step model).
        name = policy["suite"]["from_benchmark"]
        source = next((item for item in absolute if item.get("suite") == name or item.get("name") == name), None)
        if source is None or not isinstance(source.get("suite_definition"), Mapping):
            raise ConfigError(f"{profile}: performance.suite.from_benchmark {name!r} is not one of its gold suites")
        policy["suite"] = {**source["suite_definition"], "suite": f"{profile}-{name}-first", "selection": {"method": "first", "count": 1}}
    policy["suite"] = _suite(policy["suite"], profile, config.get("request"))
    reference_model = reference.pop("model", None)
    reference_revision = reference.pop("revision", None) or (None if reference_model else revision)
    size = checkpoint_bytes(reference_model or catalog_model.hf_id, reference_revision)  # the native's checkpoint
    task_measurement = deep_merge(configuration(tasks["defaults"]),
                                  configuration(tasks["tasks"][catalog_model.task]))["performance"]["measurement"]
    if size and size > LARGE_CHECKPOINT_BYTES and task_measurement["requests"] > LARGE_MODEL_MEASUREMENT["requests"]:
        # The class replaces the Task's measurement; the profile's own measurement settings still apply on top.
        stated = (override.get("performance") or {}).get("measurement") or {}
        policy["measurement"] = deep_merge(LARGE_MODEL_MEASUREMENT, stated)
        policy["measurement_reason"] = f"checkpoint {size / 2**30:.0f} GiB > {LARGE_CHECKPOINT_BYTES / 2**30:.0f} GiB"
    bundle = f"{build.get('name', profile)}/{build.get('bundle', catalog_model.bundle_name)}"
    return {
        "model": profile, "catalog_profile": profile, "operation": operation, "family": catalog_model.family,
        **({"qualified_as": base} if base != profile else {}),
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
        "performance": policy,
        **({"service_metrics": dict(config["service_metrics"])} if config.get("service_metrics") else {}),
    }


def checkpoints(model: Mapping[str, Any]) -> set[str]:
    """Hugging Face repositories a model's run downloads: its checkpoint and a separate reference model."""
    names = {model["candidate"].get("checkpoint"), model.get("reference", {}).get("model")}
    return {name for name in names if name and not str(name).startswith("/")}

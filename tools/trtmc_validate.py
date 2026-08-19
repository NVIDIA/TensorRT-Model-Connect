#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-first TRTMC reference validation for Dev and QA."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid as uuidlib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tensorrt_model_connect.python_profiles import (  # noqa: E402
    DEFAULT_PROFILE,
    normalize_execution_profiles,
    profile_env_var,
    resolve_profile_python,
)
from tensorrt_model_connect.benchmark.task_adapters import (  # noqa: E402
    adapter_for_task_strategy,
)
from tensorrt_model_connect.benchmark.types import BenchmarkError  # noqa: E402
from tensorrt_model_connect.benchmark.worker import worker_metadata  # noqa: E402
from tools.reporting_html import (  # noqa: E402
    COMMON_REPORT_STYLES,
    TASK_TYPE_BY_USER_CONTRACT,
    ReportFilter,
    render_report_filter_script,
    render_report_filters,
    sorted_filter_values,
)
from tools.validation import catalog as validation_catalog  # noqa: E402
from tools.validation.gate_census import build_gate_census  # noqa: E402
from tools import trtmc_disagreements  # noqa: E402
from tools.execution_ledger import ExecutionLedger, ExecutionLedgerError  # noqa: E402
from tools import model_selection  # noqa: E402
from tools import qualification_report  # noqa: E402
from tools.validation.gate_policy import evaluate_shadow_gates  # noqa: E402


DEFAULT_CATALOG = REPO_ROOT / "tests" / "validation" / "model_workloads.yaml"
DEFAULT_SUITES = REPO_ROOT / "tests" / "validation" / "workloads.yaml"
DEFAULT_MODELS = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "trtmc-validate"
DEFAULT_ENGINE_DIR = DEFAULT_OUTPUT / "engines"
DEFAULT_REFERENCE_CACHE = DEFAULT_OUTPUT / "references"
HF_WARM_SCRIPT = REPO_ROOT / "scripts" / "warm_hf_cache.py"
COMMON_REFERENCE_PROFILE = "reference_common"
NOT_COMPARED_DIRECTORY = "not-compared"
DEFAULT_REUSED_BUNDLE_REVALIDATION_LIMIT = 1
LEGACY_E2E_REASON = (
    "E2E execution does not compare aligned reference and TRTMC outputs."
)
HF_CACHE_ENVIRONMENT_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_ASSETS_CACHE",
    "HF_XET_CACHE",
)
RETENTION_POLICIES = ("retain", "delete_on_pass", "delete_always")


class ValidationError(RuntimeError):
    """The requested validation cannot be resolved or executed."""


@dataclass(frozen=True)
class Binding:
    model: str
    workload: str | None
    not_compared_reason: str = ""
    reference_cache_identity: str = ""

    @property
    def runnable(self) -> bool:
        return self.workload is not None


def _required_workload(binding: Binding) -> str:
    if binding.workload is None:
        raise ValidationError(f"model {binding.model} has no reference-consistency workload")
    return binding.workload


def _case_directory(output: Path, binding: Binding) -> Path:
    return (
        output
        / binding.model
        / (binding.workload if binding.workload is not None else NOT_COMPARED_DIRECTORY)
    )


@dataclass(frozen=True)
class EnvironmentSelection:
    base_python: str
    names_and_paths: tuple[tuple[str, str], ...]
    overrides: Mapping[str, str]


@dataclass(frozen=True)
class ReferenceSource:
    name: str
    repository: str
    revision: str
    relative_checkout: Path
    entrypoint: Path


@dataclass(frozen=True)
class ReferenceSourceSelection:
    environment: Mapping[str, str]
    elf_reference_repo: Path | None = None


ELF_SOURCE = ReferenceSource(
    name="ELF",
    repository="https://github.com/lillian039/ELF.git",
    revision="b29d8833609e9ab7f67cd9da39435ac5cea04837",
    relative_checkout=Path("elf/reference/ELF-b29d8833609e"),
    entrypoint=Path("src"),
)
SANA_WM_SOURCE = ReferenceSource(
    name="SANA-WM",
    repository="https://github.com/NVlabs/Sana.git",
    revision="59629fdf790850797cb657bad014fce432bd713d",
    relative_checkout=Path("sana_wm/reference/Sana-59629fdf7908"),
    entrypoint=Path("inference_video_scripts/wm/inference_sana_wm.py"),
)


def _validate_model_spec(path: Path, name: Any, spec: Any) -> None:
    if not isinstance(name, str) or not isinstance(spec, dict):
        raise ValidationError(f"{path}: invalid model binding {name!r}")
    obsolete_fields = sorted(
        {"default", "additional_workloads", "diagnostic_workloads"}.intersection(spec)
    )
    if obsolete_fields:
        raise ValidationError(
            f"{path}: {name} uses obsolete fields: {', '.join(obsolete_fields)}; "
            "list every normally selected benchmark under workloads"
        )
    not_compared_reason = spec.get("not_compared_reason")
    if not_compared_reason is not None:
        if not isinstance(not_compared_reason, str) or not not_compared_reason.strip():
            raise ValidationError(f"{path}: {name}.not_compared_reason must be a non-empty string")
        if "workloads" in spec:
            raise ValidationError(
                f"{path}: {name} cannot declare workloads while marked not compared"
            )
        return
    workloads = spec.get("workloads")
    valid_workloads = (
        isinstance(workloads, list)
        and bool(workloads)
        and all(isinstance(item, str) and item for item in workloads)
    )
    if not valid_workloads:
        raise ValidationError(f"{path}: {name}.workloads must contain names")
    if "e2e" in workloads:
        raise ValidationError(
            f"{path}: {name} workloads cannot use e2e; reference consistency "
            "requires aligned reference and TRTMC outputs"
        )
    reference_cache_identity = spec.get("reference_cache_identity")
    if reference_cache_identity is not None and (
        not isinstance(reference_cache_identity, str) or not reference_cache_identity.strip()
    ):
        raise ValidationError(f"{path}: {name}.reference_cache_identity must be a non-empty string")


def declared_workloads(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the benchmarks selected for a model's Accuracy matrix."""

    return tuple(dict.fromkeys(spec.get("workloads", [])))


def _validate_sample_limits(path: Path, raw: Mapping[str, Any]) -> None:
    sample_limits = raw.get("sample_limits")
    if not isinstance(sample_limits, dict) or not sample_limits:
        raise ValidationError(f"{path}: sample_limits must be a non-empty mapping")
    for workload, limit in sample_limits.items():
        if not isinstance(workload, str) or not workload:
            raise ValidationError(f"{path}: invalid sample-limit workload {workload!r}")
        if isinstance(limit, bool) or not isinstance(limit, int) or (limit != -1 and limit <= 0):
            raise ValidationError(
                f"{path}: sample_limits.{workload} must be -1 or a positive integer"
            )


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValidationError(f"{path}: expected version: 1")
    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValidationError(f"{path}: models must be a non-empty mapping")
    _validate_sample_limits(path, raw)
    for name, spec in models.items():
        _validate_model_spec(path, name, spec)
    return raw


def _suite_task_metadata(suite: Mapping[str, Any]) -> tuple[str, str]:
    user_contract = str(suite.get("user_contract", "") or "")
    task_type = str(suite.get("task_type", "") or "")
    if not task_type:
        task_type = TASK_TYPE_BY_USER_CONTRACT.get(user_contract, "")
    return task_type, user_contract


@cache
def _default_suite_task_metadata() -> dict[str, tuple[str, str]]:
    return {
        str(suite["id"]): _suite_task_metadata(suite)
        for suite in validation_catalog.load_suites(DEFAULT_SUITES)
    }


def _result_task_metadata(result: Mapping[str, Any]) -> tuple[str, str]:
    task_type = str(result.get("task_type", "") or "")
    user_contract = str(result.get("user_contract", "") or "")
    if task_type:
        return task_type, user_contract
    workload = str(result.get("workload", "") or "")
    return _default_suite_task_metadata().get(workload, ("", user_contract))


def _operation_for_task_strategy(task_strategy: str) -> str:
    if not task_strategy:
        return ""
    try:
        return adapter_for_task_strategy(task_strategy).operation
    except BenchmarkError:
        return ""


@cache
def _default_model_report_metadata() -> dict[str, tuple[str, str, str]]:
    metadata = {}
    for model in validation_catalog.load_manifest_records(DEFAULT_MODELS):
        task_strategy = str(model.get("task_strategy", "") or "")
        metadata[str(model["name"])] = (
            str(model.get("family", "") or ""),
            _operation_for_task_strategy(task_strategy),
            task_strategy,
        )
    return metadata


def _result_model_report_metadata(result: Mapping[str, Any]) -> tuple[str, str, str]:
    fallback = _default_model_report_metadata().get(
        str(result.get("model", "")),
        ("", "", ""),
    )
    task_strategy = str(result.get("task_strategy", "") or fallback[2])
    return (
        str(result.get("family", "") or fallback[0]),
        str(
            result.get("operation", "")
            or fallback[1]
            or _operation_for_task_strategy(task_strategy)
        ),
        task_strategy,
    )


def ready_model_names(models_root: Path = DEFAULT_MODELS) -> tuple[str, ...]:
    models = validation_catalog.load_manifest_records(models_root)
    return tuple(
        sorted(
            str(model["name"])
            for model in models
            if not model["requires_multi_device"]
            and not model.get("skip")
            and model.get("ci_tier") != "l0_only"
            and model.get("test_category", "e2e") != "regression"
        )
    )


def audit_catalog(
    catalog: Mapping[str, Any],
    *,
    ready_models: Iterable[str],
    suite_names: Iterable[str],
) -> None:
    models = catalog["models"]
    ready = set(ready_models)
    configured = set(models)
    missing = sorted(ready - configured)
    stale = sorted(configured - ready)
    if missing or stale:
        details = []
        if missing:
            details.append(f"missing ready models: {', '.join(missing)}")
        if stale:
            details.append(f"non-ready or unknown models: {', '.join(stale)}")
        raise ValidationError("; ".join(details))

    known_workloads = set(suite_names)
    unknown = sorted(
        {
            workload
            for spec in models.values()
            for workload in declared_workloads(spec)
            if workload not in known_workloads
        }
    )
    if unknown:
        raise ValidationError(f"unknown workloads: {', '.join(unknown)}")

    declared_sampled = {
        workload for spec in models.values() for workload in declared_workloads(spec)
    }
    configured_sampled = set(catalog["sample_limits"])
    missing_limits = sorted(declared_sampled - configured_sampled)
    unknown_limits = sorted(configured_sampled - known_workloads)
    if missing_limits:
        raise ValidationError(f"missing sample limits: {', '.join(missing_limits)}")
    if unknown_limits:
        raise ValidationError(f"unknown sample-limit workloads: {', '.join(unknown_limits)}")


def audit_workload_compatibility(
    catalog: Mapping[str, Any],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> None:
    incompatible = []
    reference_cache_contracts: dict[str, set[tuple[str, ...]]] = {}
    for model_name, spec in catalog["models"].items():
        for workload in declared_workloads(spec):
            matched, reason = validation_catalog.suite_match_reason(
                suites[workload],
                task_models[model_name],
            )
            if not matched:
                incompatible.append(f"{model_name}/{workload}: {reason}")
            reference_cache_identity = str(spec.get("reference_cache_identity", "") or "")
            if reference_cache_identity:
                model = task_models[model_name]
                contract = (
                    str(model.get("hf_id", "") or ""),
                    str(model.get("hf_revision", "") or ""),
                    str(model.get("family", "") or ""),
                    str(model.get("reference_backend", "") or ""),
                    str(model.get("reference_family", "") or ""),
                    workload,
                )
                reference_cache_contracts.setdefault(
                    reference_cache_identity,
                    set(),
                ).add(contract)
    for identity, contracts in sorted(reference_cache_contracts.items()):
        if len(contracts) > 1:
            incompatible.append(
                f"reference cache identity {identity!r} spans different reference contracts"
            )
    if incompatible:
        raise ValidationError("incompatible model/workload bindings: " + "; ".join(incompatible))


def audit_binding_compatibility(
    bindings: Iterable[Binding],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> None:
    """Validate explicitly selected bindings against suite selectors."""

    incompatible = []
    for binding in bindings:
        if not binding.runnable:
            continue
        assert binding.workload is not None
        suite = suites.get(binding.workload)
        model = task_models.get(binding.model)
        if suite is None or model is None:
            incompatible.append(
                f"{binding.model}/{binding.workload}: missing suite or model metadata"
            )
            continue
        matched, reason = validation_catalog.suite_match_reason(suite, model)
        if not matched:
            incompatible.append(f"{binding.model}/{binding.workload}: {reason}")
    if incompatible:
        raise ValidationError(
            "incompatible selected model/workload bindings: " + "; ".join(incompatible)
        )


def resolve_binding(
    catalog: Mapping[str, Any],
    model: str,
    workload: str | None = None,
) -> Binding:
    models = catalog["models"]
    if model not in models:
        raise ValidationError(f"unknown or unsupported model: {model}")
    spec = models[model]
    not_compared_reason = str(spec.get("not_compared_reason", "") or "")
    if not_compared_reason:
        if workload:
            raise ValidationError(
                f"model {model} has no reference-consistency workloads: {not_compared_reason}"
            )
        return Binding(
            model=model,
            workload=None,
            not_compared_reason=not_compared_reason,
        )
    available_workloads = declared_workloads(spec)
    if workload is None:
        if len(available_workloads) != 1:
            raise ValidationError(
                f"model {model} selects {len(available_workloads)} workloads; "
                "resolve the model matrix or name one workload explicitly"
            )
        selected = available_workloads[0]
    else:
        selected = workload
    if selected not in catalog["sample_limits"]:
        available = ", ".join(sorted(catalog["sample_limits"]))
        raise ValidationError(f"unknown workload {selected}; available: {available}")
    return Binding(
        model=model,
        workload=selected,
        reference_cache_identity=str(spec.get("reference_cache_identity", "") or ""),
    )


def resolve_bindings(
    catalog: Mapping[str, Any],
    models: Iterable[str],
    *,
    workloads: Iterable[str] = (),
) -> list[Binding]:
    """Resolve model selection into independent model/workload bindings."""

    selected_models = model_selection.normalize_models(models)
    selected_workloads = model_selection.normalize_models(workloads)

    bindings: list[Binding] = []
    for model in selected_models:
        spec = catalog["models"].get(model)
        if not isinstance(spec, Mapping):
            raise ValidationError(f"unknown or unsupported model: {model}")
        not_compared_reason = str(spec.get("not_compared_reason", "") or "")
        if not_compared_reason:
            if selected_workloads:
                raise ValidationError(
                    f"model {model} has no reference-consistency workloads: {not_compared_reason}"
                )
            bindings.append(resolve_binding(catalog, model))
            continue

        model_workloads = (
            selected_workloads
            if selected_workloads
            else model_selection.normalize_models(spec["workloads"])
        )
        bindings.extend(resolve_binding(catalog, model, workload) for workload in model_workloads)
    return bindings


def model_profiles_for_families(
    task_models: Mapping[str, Mapping[str, Any]],
    ready_models: Iterable[str],
    families: Iterable[str],
) -> tuple[str, ...]:
    """Expand model_ci owner/family IDs into ready Accuracy model profiles."""

    selected_families = model_selection.normalize_models(families)
    ready = set(ready_models)
    profiles: list[str] = []
    missing: list[str] = []
    for family in selected_families:
        matched = sorted(
            model
            for model, record in task_models.items()
            if model in ready and str(record.get("family", "")) == family
        )
        if not matched:
            missing.append(family)
        profiles.extend(matched)
    if missing:
        raise ValidationError("model owners have no ready Accuracy profiles: " + ", ".join(missing))
    return tuple(profiles)


def resolve_sample_limit(
    catalog: Mapping[str, Any],
    binding: Binding,
    explicit_limit: int | None,
) -> int:
    if explicit_limit is not None and explicit_limit < -1:
        raise ValidationError("--limit must be -1 or greater")
    if not binding.runnable:
        return 0
    if explicit_limit is not None:
        return 0 if explicit_limit == -1 else explicit_limit
    assert binding.workload is not None
    configured = int(catalog["sample_limits"][binding.workload])
    return 0 if configured == -1 else configured


def _validation_models(models_root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(model["name"]): model for model in validation_catalog.load_manifest_records(models_root)
    }


def _declared_profile(
    *,
    family: str,
    runtime_strategy: str,
    reference_backend: str,
    execution_profiles: Mapping[str, str] | None,
) -> str:
    profiles = normalize_execution_profiles(
        execution_profiles,
        family=family,
        runtime_strategy=runtime_strategy,
        reference_backend=reference_backend,
    )
    profile = profiles["reference"]
    if profile == DEFAULT_PROFILE:
        return COMMON_REFERENCE_PROFILE
    return profile


def _binding_profiles(
    binding: Binding,
    *,
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    if not binding.runnable:
        raise ValidationError(f"model {binding.model} has no reference-consistency workload")
    model = task_models[binding.model]
    profile = _declared_profile(
        family=str(model.get("family", "") or ""),
        runtime_strategy=str(model.get("runtime_strategy", "") or ""),
        reference_backend=str(model.get("reference_backend", "") or ""),
        execution_profiles=model.get("execution_profiles"),
    )
    profiles = [COMMON_REFERENCE_PROFILE]
    if profile != COMMON_REFERENCE_PROFILE:
        profiles.append(profile)
    suite = (suites or {}).get(binding.workload, {})
    scoring = suite.get("scoring", {}) if isinstance(suite, Mapping) else {}
    scoring_profile = (
        str(scoring.get("python_profile", "") or "") if isinstance(scoring, Mapping) else ""
    )
    if scoring_profile and scoring_profile not in profiles:
        profiles.append(scoring_profile)
    return tuple(profiles)


def ensure_environments(
    profile_names: Iterable[str],
    base_python: str,
) -> EnvironmentSelection:
    names_and_paths = []
    overrides = {}
    selected_base = base_python

    def announce_create(name: str) -> None:
        print(f"Creating reference environment: {name}", flush=True)

    for name in profile_names:
        path = resolve_profile_python(
            name,
            base_python,
            on_create=announce_create,
        )
        print(f"Using reference environment: {path}", flush=True)
        names_and_paths.append((name, path))
        if name == COMMON_REFERENCE_PROFILE:
            selected_base = path
        elif name != DEFAULT_PROFILE:
            overrides[profile_env_var(name)] = path
    return EnvironmentSelection(
        base_python=selected_base,
        names_and_paths=tuple(names_and_paths),
        overrides=overrides,
    )


def _ensure_reference_source(source: ReferenceSource, cache_root: Path) -> Path:
    checkout = cache_root / source.relative_checkout
    entrypoint = checkout / source.entrypoint
    if entrypoint.exists():
        print(f"Using reference source: {checkout}", flush=True)
        return checkout
    if checkout.exists():
        raise ValidationError(f"Incomplete cached {source.name} reference: {checkout}")

    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating reference source: {source.name}", flush=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{checkout.name}-",
            dir=checkout.parent,
        ) as temporary:
            staged = Path(temporary) / "checkout"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    source.repository,
                    str(staged),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(staged),
                    "checkout",
                    "--detach",
                    source.revision,
                ],
                check=True,
            )
            if not (staged / source.entrypoint).exists():
                raise ValidationError(
                    f"Pinned {source.name} checkout is missing {source.entrypoint}"
                )
            staged.rename(checkout)
    except subprocess.CalledProcessError as exc:
        raise ValidationError(
            f"Could not prepare pinned {source.name} reference "
            f"{source.revision}: git exited with code {exc.returncode}"
        ) from exc
    print(f"Using reference source: {checkout}", flush=True)
    return checkout


def ensure_reference_sources(
    family: str,
    cache_root: Path,
    model_reference_cache: Mapping[str, Any] | None = None,
    *,
    source_cache_root: Path | None = None,
) -> ReferenceSourceSelection:
    environment = {"TRTMC_STORAGE_ROOT": str(cache_root)}
    checkout_root = source_cache_root or cache_root
    declared_source = None
    if model_reference_cache:
        required = ("repository", "revision", "relative_path", "entrypoint")
        missing = [field for field in required if not model_reference_cache.get(field)]
        if missing:
            raise ValidationError(
                f"{family} model reference source is missing: " + ", ".join(missing)
            )
        declared_source = ReferenceSource(
            name=family,
            repository=str(model_reference_cache["repository"]),
            revision=str(model_reference_cache["revision"]),
            relative_checkout=Path(str(model_reference_cache["relative_path"])),
            entrypoint=Path(str(model_reference_cache["entrypoint"])),
        )
        checkout = _ensure_reference_source(declared_source, checkout_root)
        environment_variable = str(
            model_reference_cache.get("environment_variable", "") or ""
        ).strip()
        if environment_variable:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_variable) is None:
                raise ValidationError(
                    f"{family} model reference environment_variable is invalid: "
                    f"{environment_variable!r}"
                )
            environment[environment_variable] = str(checkout)

    if family == "elf_flow":
        checkout = _ensure_reference_source(ELF_SOURCE, checkout_root)
        return ReferenceSourceSelection(
            environment=environment,
            elf_reference_repo=checkout,
        )
    if family == "sana_wm":
        if declared_source is None:
            declared_source = SANA_WM_SOURCE
            checkout = _ensure_reference_source(declared_source, checkout_root)
        environment["SANA_WM_SCRIPT"] = str(checkout / declared_source.entrypoint)
    return ReferenceSourceSelection(environment=environment)


def _dataset_path(suite: Mapping[str, Any], dataset_root: Path | None) -> Path:
    raw = str(suite.get("dataset", {}).get("default_path", "") or "")
    if not raw:
        raise ValidationError(f"workload {suite.get('id')} has no default dataset path")
    path = Path(raw)
    if not path.is_absolute():
        return REPO_ROOT / path
    if dataset_root is None:
        return path
    try:
        relative = path.relative_to("/mnt/data")
    except ValueError:
        relative = Path(path.name)
    return dataset_root / relative


def _run_subprocess(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        output.write(f"$ {shlex.join(command)}\n")
        output.flush()
        completed = subprocess.run(
            list(command),
            check=False,
            text=True,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(env),
        )
    return completed.returncode


class WorkerTimeoutError(RuntimeError):
    """A supervised model worker exceeded its configured wall-clock limit."""


def _run_supervised_subprocess(
    command: Sequence[str],
    log_path: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> int:
    if timeout_seconds <= 0:
        return _run_subprocess(command, log_path, env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as output:
        output.write(f"$ {shlex.join(command)}\n")
        output.flush()
        process = subprocess.Popen(
            list(command),
            text=True,
            cwd=REPO_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(env),
            start_new_session=os.name == "posix",
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            output.write(
                "\n[trtmc-validate] model worker timed out after "
                f"{timeout_seconds:g} seconds; terminating process group\n"
            )
            output.flush()
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
            raise WorkerTimeoutError(
                f"model worker exceeded {timeout_seconds:g} seconds"
            ) from exc


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    roots = [str(PYTHON_ROOT), str(REPO_ROOT)]
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _comparison_command(
    binding: Binding,
    *,
    case_dir: Path,
    dataset: Path,
    arguments: argparse.Namespace,
    reference_python: str,
    reference_sources: ReferenceSourceSelection | None = None,
) -> list[str]:
    work_root = case_dir / "validation"
    workload = _required_workload(binding)
    command = [
        reference_python,
        str(REPO_ROOT / "tools" / "trtmc_compare.py"),
        "--suite",
        workload,
        "--dataset",
        str(dataset),
        "--model",
        binding.model,
        "--models-dir",
        str(arguments.models_dir),
        "--work-root",
        str(work_root),
        "--engine-dir",
        str(arguments.engine_dir),
        "--trtmc-binary",
        str(arguments.trtmc_binary),
        "--benchmark-binary",
        str(arguments.benchmark_binary),
        "--hf-python",
        reference_python,
        "--reference-cache-dir",
        str(arguments.reference_cache_dir),
        "--replace-bundle-on-build",
        "--single-device-only",
        "--include-waived",
        "--fail-fast",
    ]
    if arguments.limit:
        command.extend(["--limit", str(arguments.limit)])
    if binding.reference_cache_identity:
        command.extend(
            [
                "--reference-cache-identity",
                binding.reference_cache_identity,
            ]
        )
    if arguments.force_hf:
        command.append("--force-hf")
    if arguments.force_build:
        command.append("--force-build")
    if arguments.no_build:
        command.append("--require-prebuilt-bundles")
    if arguments.local_files_only:
        command.append("--local-files-only")
    if arguments.backend_dir:
        command.extend(["--backend-dir", str(arguments.backend_dir)])
    if arguments.model_plugin_dir:
        command.extend(["--model-plugin-dir", str(arguments.model_plugin_dir)])
    if arguments.cuda_visible_devices:
        command.extend(["--cuda-visible-devices", arguments.cuda_visible_devices])
    command.extend(["--hf-device", getattr(arguments, "hf_device", "cuda")])
    if getattr(arguments, "hf_device_map", ""):
        command.extend(["--hf-device-map", arguments.hf_device_map])
    if reference_sources and reference_sources.elf_reference_repo:
        command.extend(
            [
                "--elf-reference-repo",
                str(reference_sources.elf_reference_repo),
            ]
        )
    return command


MAX_REPRO_COMMANDS_PER_BACKEND = 3
_REPRO_COMMAND_LOG_NAMES = {
    "hf_native_run.log",
    "hf_run.log",
    "bundle_run.log",
}
_FAILED_SAMPLE_STATUSES = {"disagreement", "fail", "failed", "mismatch"}
_FAILED_SAMPLE_FIELDS = (
    "agreement_match",
    "exact_match",
    "passed",
    "top1_agreement",
    "transcript_exact",
)
_SAMPLE_ID_FIELDS = ("sample_id", "case_id", "id", "name")


def _command_record_from_log_line(line: str) -> tuple[str, str]:
    if line.startswith("$ "):
        return line[2:].strip(), ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    command = data.get("command")
    sample_id = next(
        (str(data[name]) for name in _SAMPLE_ID_FIELDS if data.get(name) is not None),
        "",
    )
    if isinstance(command, list) and command:
        return shlex.join(str(token) for token in command), sample_id
    if isinstance(command, str):
        return command.strip(), sample_id
    return "", ""


def _command_from_log_line(line: str) -> str:
    return _command_record_from_log_line(line)[0]


def _sample_id(record: Mapping[str, Any]) -> str:
    return next(
        (str(record[name]) for name in _SAMPLE_ID_FIELDS if record.get(name) is not None),
        "",
    )


def _explicit_disagreement_id(data: Mapping[str, Any]) -> str:
    disagreements = data.get("disagreements", [])
    if not isinstance(disagreements, list):
        return ""
    for item in disagreements:
        if isinstance(item, dict) and _sample_id(item):
            return _sample_id(item)
    return ""


def _record_is_disagreement(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "") or "").lower()
    if status in _FAILED_SAMPLE_STATUSES:
        return True
    return any(record.get(name) is False for name in _FAILED_SAMPLE_FIELDS)


def _failed_sample_id(data: Any) -> str:
    if isinstance(data, list):
        for item in data:
            failed = _failed_sample_id(item)
            if failed:
                return failed
        return ""
    if not isinstance(data, dict):
        return ""
    if _record_is_disagreement(data) and _sample_id(data):
        return _sample_id(data)
    for value in data.values():
        failed = _failed_sample_id(value)
        if failed:
            return failed
    return ""


def _first_disagreement_id(work_dir: Path) -> str:
    for name in ("summary.json", "eval_result.json"):
        path = work_dir / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("disagreements"), list):
            return _explicit_disagreement_id(data)
        failed = _failed_sample_id(data)
        if failed:
            return failed
    return ""


def _prepared_sample_ids(work_dir: Path) -> list[str]:
    prompts = work_dir / "prompts.jsonl"
    if not prompts.is_file():
        return []
    sample_ids = []
    with prompts.open(encoding="utf-8") as prompt_file:
        for index, line in enumerate(prompt_file):
            record = json.loads(line)
            if isinstance(record, dict):
                sample_ids.append(_sample_id(record) or f"sample-{index}")
    return sample_ids


def _sample_ids_match(candidate: str, target: str) -> bool:
    return bool(
        candidate
        and target
        and (
            candidate == target
            or candidate.startswith(f"{target}:")
            or target.startswith(f"{candidate}:")
        )
    )


def _summarize_command_log(
    path: Path,
    *,
    sample_ids: Sequence[str],
    target_sample_id: str,
) -> tuple[int, str]:
    count = 0
    first = ""
    selected = ""
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            command, logged_sample_id = _command_record_from_log_line(line)
            if not command:
                continue
            indexed_sample_id = sample_ids[count] if count < len(sample_ids) else ""
            command_sample_id = logged_sample_id or indexed_sample_id
            count += 1
            first = first or command
            if _sample_ids_match(command_sample_id, target_sample_id):
                selected = command
    return count, selected or first


def _command_log_kind(
    path: Path,
    *,
    has_native_reference: bool,
    has_native_reference_commands: bool,
    has_native_trtmc: bool,
) -> str | None:
    if has_native_reference and path.name == "hf_run.log":
        return None
    if has_native_reference_commands and path.name == "hf_native_run.log":
        return None
    if has_native_trtmc and path.name == "bundle_run.log":
        return None
    return "hf" if "hf" in path.name.lower() else "trtmc"


def _relocate_cached_reference_command(command: str, work_dir: Path) -> str:
    """Point a cached native reference command at the current materialized run."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    original_work_dir = None
    for flag in ("--manifest", "--prompts", "--answers"):
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                candidate = Path(tokens[index + 1])
                if candidate.is_absolute():
                    original_work_dir = candidate.parent
                    break
    if original_work_dir is None:
        return command
    original_prefix = str(original_work_dir)
    current_prefix = str(work_dir.resolve())
    relocated = [
        (
            current_prefix + token[len(original_prefix) :]
            if token == original_prefix or token.startswith(original_prefix + os.sep)
            else token
        )
        for token in tokens
    ]
    return shlex.join(relocated)


def _collect_command_logs(
    root: Path,
    *,
    log_paths: Sequence[Path],
    sample_ids: Sequence[str],
    representative_id: str,
) -> tuple[dict[str, list[str]], dict[str, int], dict[str, list[str]]]:
    commands: dict[str, list[str]] = {"hf": [], "trtmc": []}
    counts = {"hf": 0, "trtmc": 0}
    logs: dict[str, list[str]] = {"hf": [], "trtmc": []}
    has_native_reference = any(
        path.name in {"hf_native_run.log", "hf_native_commands.jsonl"} for path in log_paths
    )
    has_native_reference_commands = any(
        path.name == "hf_native_commands.jsonl" for path in log_paths
    )
    has_native_trtmc = any(path.name == "bundle_native_commands.jsonl" for path in log_paths)
    for path in log_paths:
        kind = _command_log_kind(
            path,
            has_native_reference=has_native_reference,
            has_native_reference_commands=has_native_reference_commands,
            has_native_trtmc=has_native_trtmc,
        )
        if kind is None:
            continue
        indexed_sample_ids = sample_ids if path.name == "bundle_run.log" else ()
        count, representative = _summarize_command_log(
            path,
            sample_ids=indexed_sample_ids,
            target_sample_id=representative_id,
        )
        if path.name == "hf_native_run.log":
            representative = _relocate_cached_reference_command(
                representative,
                root,
            )
        counts[kind] += count
        if count:
            logs[kind].append(str(path.relative_to(root)))
        _append_unique(commands, kind, representative)
    return commands, counts, logs


def _commands_from_logs(root: Path) -> dict[str, Any]:
    sample_ids = _prepared_sample_ids(root)
    disagreement_id = _first_disagreement_id(root)
    representative_id = disagreement_id or (sample_ids[0] if sample_ids else "")
    log_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.name in _REPRO_COMMAND_LOG_NAMES or path.name.endswith("_native_commands.jsonl"))
    )
    commands, counts, logs = _collect_command_logs(
        root,
        log_paths=log_paths,
        sample_ids=sample_ids,
        representative_id=representative_id,
    )
    for kind in commands:
        commands[kind] = commands[kind][:MAX_REPRO_COMMANDS_PER_BACKEND]
    return {
        **commands,
        "command_count": counts,
        "commands_shown": {kind: len(values) for kind, values in commands.items()},
        "command_logs": logs,
        "representative": {
            "sample_id": representative_id,
            "reason": "first_disagreement" if disagreement_id else "first_input",
        },
        "prepared_input_count": len(sample_ids),
    }


def _append_unique(commands: dict[str, list[str]], kind: str, command: str) -> None:
    if command and command not in commands[kind]:
        commands[kind].append(command)


_PRIMARY_COMPARISON_METRICS = (
    "metric_gate_pass_rate",
    "sample_pass_rate",
    "sample_agreement_rate",
    "prediction_agreement_rate",
    "vector_pass_rate",
    "top1_agreement",
    "backend_pixel_agreement",
    "mean_pairwise_ordering_agreement",
    "token_prefix_agreement",
    "token_agreement_rate",
    "exact_match_rate",
)
_PRIMARY_METRIC_BY_MODE = {
    "asr_transcript": "prediction_agreement_rate",
    "continuation": "token_prefix_agreement",
    "diffusion_image_clip_parity": "overall_pass_rate",
    "diffusion_text_parity": "token_agreement_rate",
    "encoder_embedding_parity": "vector_pass_rate",
    "full_duplex_bench_behavior_parity": "metric_gate_pass_rate",
    "image_classification_parity": "top1_agreement",
    "model_plugin_parity": "sample_pass_rate",
    "ocrbench_v2": "prediction_agreement_rate",
    "reranking_parity": "mean_pairwise_ordering_agreement",
    "semantic_segmentation_parity": "backend_pixel_agreement",
    "time_series_parity": "sample_agreement_rate",
}
_COMPARISON_METRICS = (
    *_PRIMARY_COMPARISON_METRICS,
    "overall_pass_rate",
    "passed_count",
    "valid_count",
    "skipped_count",
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "divergent_count",
    "hf_accuracy",
    "bundle_accuracy",
    "accuracy_delta_bundle_minus_hf",
    "tie_adjusted_accuracy_delta_bundle_minus_hf",
    "tie_adjusted_exact_match_rate",
    "accuracy_drop_from_hf",
    "raw_accuracy_drop_from_hf",
    "reference_tie_equivalent_count",
    "hf_top1_accuracy",
    "bundle_top1_accuracy",
    "top1_accuracy_drop_from_hf",
    "hf_mean_iou",
    "bundle_mean_iou",
    "backend_mean_iou",
    "worst_backend_mask_iou",
    "mean_iou_drop_from_hf",
    "tor_abs_delta",
    "backchannel_frequency_abs_delta",
    "backchannel_jsd_abs_delta",
    "mean_vector_cosine",
    "min_vector_cosine",
    "mean_pair_cosine_abs_delta",
    "max_pair_cosine_abs_delta",
    "mean_relative_l2",
    "max_relative_l2",
    "max_absolute_error",
)
_EXECUTION_ERROR_FIELDS = ("error", "exception", "traceback", "failure_class")


def _is_comparison_gate_failure(raw_result: Mapping[str, Any]) -> bool:
    failures = raw_result.get("gate_failures")
    return (
        raw_result.get("error_type") == "BenchmarkGateError"
        and isinstance(failures, list)
        and bool(failures)
    )


def _raw_comparison(result: Mapping[str, Any]) -> dict[str, Any]:
    raw_result = result.get("raw_result")
    if isinstance(raw_result, dict) and raw_result:
        return dict(raw_result)
    status = str(result.get("status", "") or "")
    return {"status": status} if status else {}


def _execution_details(
    result: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_gate_failure = _is_comparison_gate_failure(raw_result)
    has_error = any(
        raw_result.get(name)
        for name in _EXECUTION_ERROR_FIELDS
        if name != "error" or not comparison_gate_failure
    )
    completed = bool(raw_result) and not has_error
    return {
        "status": "completed" if completed else "error",
        "exit_code": result.get("returncode"),
    }


def _comparison_metrics(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        name: raw_result[name] for name in _COMPARISON_METRICS if raw_result.get(name) is not None
    }
    nested = raw_result.get("metrics", {})
    if isinstance(nested, Mapping):
        for name, summary in nested.items():
            if not isinstance(summary, Mapping):
                continue
            mean = summary.get("mean")
            if isinstance(mean, (int, float)) and not isinstance(mean, bool):
                metrics[str(name)] = mean
    return metrics


def _primary_metric(
    mode: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    preferred = _PRIMARY_METRIC_BY_MODE.get(mode)
    if preferred in metrics:
        return {"name": preferred, "value": metrics[preferred]}
    for name in _PRIMARY_COMPARISON_METRICS:
        if name in metrics:
            return {"name": name, "value": metrics[name]}
    return None


def _comparison_details(
    raw_result: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    raw_status = str(raw_result.get("status", "") or "")
    status_by_raw = {
        "pass": "agreement",
        "passed": "agreement",
        "fail": "disagreement",
        "failed": "disagreement",
        "skip": "not_run",
        "skipped": "not_run",
    }
    status = (
        status_by_raw.get(raw_status, "not_run")
        if execution.get("status") == "completed"
        else "not_run"
    )
    metrics = _comparison_metrics(raw_result)
    failures = raw_result.get("gate_failures", [])
    mode = str(raw_result.get("mode", "") or "")
    return {
        "status": status,
        "mode": mode,
        "primary_metric": _primary_metric(mode, metrics),
        "metrics": metrics,
        "failures": failures if isinstance(failures, list) else [],
    }


def _validation_details(
    execution: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    if execution.get("status") != "completed":
        return {"status": "failed"}
    status_by_comparison = {
        "agreement": "passed",
        "disagreement": "failed",
        "not_run": "skipped",
    }
    return {"status": status_by_comparison[str(comparison["status"])]}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalized_command_count(
    reproduce: Mapping[str, Any],
    kind: str,
    commands: Sequence[str],
) -> int:
    counts = reproduce.get("command_count", {})
    configured = counts.get(kind) if isinstance(counts, dict) else None
    try:
        return max(int(configured), len(commands))
    except (TypeError, ValueError):
        return len(commands)


def _normalized_command_logs(reproduce: Mapping[str, Any], kind: str) -> list[str]:
    logs = reproduce.get("command_logs", {})
    return _string_list(logs.get(kind, [])) if isinstance(logs, dict) else []


def _normalize_reproduction(value: Any) -> dict[str, Any]:
    reproduce = value if isinstance(value, dict) else {}
    all_commands = {kind: _string_list(reproduce.get(kind, [])) for kind in ("hf", "trtmc")}
    commands = {
        kind: values[:MAX_REPRO_COMMANDS_PER_BACKEND] for kind, values in all_commands.items()
    }
    dataset = reproduce.get("dataset", {})
    if not isinstance(dataset, dict):
        dataset = {"command": str(dataset)} if str(dataset).strip() else {}
    representative = reproduce.get("representative", {})
    if not isinstance(representative, dict):
        representative = {}
    return {
        "dataset": dataset,
        **commands,
        "command_count": {
            kind: _normalized_command_count(reproduce, kind, all_commands[kind])
            for kind in commands
        },
        "commands_shown": {kind: len(values) for kind, values in commands.items()},
        "command_logs": {kind: _normalized_command_logs(reproduce, kind) for kind in commands},
        "representative": representative,
    }


def _add_dataset_reproduction(
    reproduce: Mapping[str, Any],
    command: str,
    sample_limit: int = 0,
) -> dict[str, Any]:
    result = dict(reproduce)
    prepared_input_count = int(result.pop("prepared_input_count", 0) or 0)
    result["dataset"] = {
        "command": command,
        "sample_limit": sample_limit,
        "prepared_input_count": prepared_input_count,
    }
    return result


def _normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    if normalized.get("executor") == "e2e" or isinstance(
        normalized.get("raw_results"),
        list,
    ):
        normalized.update(
            {
                "workload": None,
                "execution": {"status": "not_run", "exit_code": None},
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "not_compared"},
                "not_compared_reason": LEGACY_E2E_REASON,
            }
        )
    raw_result = _raw_comparison(normalized)
    execution = normalized.get("execution")
    if not isinstance(execution, dict):
        execution = _execution_details(normalized, raw_result)
    comparison = normalized.get("comparison")
    if not isinstance(comparison, dict):
        comparison = _comparison_details(raw_result, execution)
    validation = normalized.get("validation")
    if not isinstance(validation, dict):
        validation = _validation_details(execution, comparison)
    precision_contract = normalized.get("precision_contract")
    if not isinstance(precision_contract, dict):
        candidate = raw_result.get("precision_contract")
        precision_contract = dict(candidate) if isinstance(candidate, dict) else {}
    task_type, user_contract = _result_task_metadata(normalized)
    family, operation, task_strategy = _result_model_report_metadata(normalized)
    normalized.update(
        {
            "schema_version": "trtmc.validation-result/v2",
            "family": family,
            "operation": operation,
            "task_strategy": task_strategy,
            "task_type": task_type,
            "user_contract": user_contract,
            "execution": execution,
            "comparison": comparison,
            "validation": validation,
            "reproduce": _normalize_reproduction(normalized.get("reproduce")),
        }
    )
    if precision_contract:
        normalized["precision_contract"] = precision_contract
    else:
        normalized.pop("precision_contract", None)
    normalized.pop("returncode", None)
    normalized.pop("status", None)
    return normalized


def _not_compared_result(binding: Binding) -> dict[str, Any]:
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": None,
            "executor": "not_compared",
            "execution": {"status": "not_run", "exit_code": None},
            "comparison": {
                "status": "not_run",
                "mode": "",
                "primary_metric": None,
                "metrics": {},
                "failures": [],
            },
            "validation": {"status": "not_compared"},
            "not_compared_reason": binding.not_compared_reason,
            "reference_environment": [],
            "reproduce": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _write_not_compared_case(binding: Binding, output: Path) -> tuple[dict[str, Any], Path]:
    case_dir = _case_directory(output, binding)
    case_dir.mkdir(parents=True, exist_ok=True)
    result = _not_compared_result(binding)
    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result, comparison


def _comparison_result(
    binding: Binding,
    *,
    case_dir: Path,
    returncode: int,
    reference_environment: EnvironmentSelection,
    dataset_command: str,
    sample_limit: int = 0,
    family: str = "",
    operation: str = "",
    task_strategy: str = "",
    task_type: str = "",
    user_contract: str = "",
) -> dict[str, Any]:
    workload = _required_workload(binding)
    summary_path = case_dir / "validation" / workload / "eval_summary.json"
    raw_result: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for candidate in summary.get("results", []):
            if candidate.get("model") == binding.model:
                raw_result = candidate
                break
        if not raw_result and summary.get("results"):
            raw_result = summary["results"][0]
    if not raw_result:
        raw_result = {
            "status": "failed",
            "error_type": "ComparisonProcessError",
            "error": (
                f"comparison exited with code {returncode} without writing "
                f"a model result to {summary_path}"
            ),
        }
    status = str(raw_result.get("status", "") or "")
    if status not in {"passed", "failed", "skipped"}:
        status = "passed" if returncode == 0 else "failed"
    work_dir = case_dir / "validation" / workload / binding.model
    disagreements = trtmc_disagreements.build_disagreement_artifact(
        work_dir=work_dir,
        case_dir=case_dir,
    )
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "family": family,
            "operation": operation,
            "task_strategy": task_strategy,
            "task_type": task_type,
            "user_contract": user_contract,
            "executor": "trtmc_compare",
            "status": status,
            "returncode": returncode,
            "reference_environment": [
                {"name": name, "python": path}
                for name, path in reference_environment.names_and_paths
            ],
            "reproduce": _add_dataset_reproduction(
                _commands_from_logs(work_dir),
                dataset_command,
                sample_limit,
            ),
            "raw_result": raw_result,
            "raw_result_path": str(summary_path),
            "disagreements": disagreements,
            "execution_log": str(case_dir / "execution.log"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _should_revalidate_reused_bundle(
    result: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> bool:
    """Return whether one accuracy failure needs a fresh-bundle confirmation.

    This is intentionally separate from the worker execution retry policy:
    comparison disagreements are normally never retried.  The sole exception
    is a completed comparison that explicitly reports it reused a bundle.  A
    single forced rebuild distinguishes a stale-cache false positive from a
    reproducible model-family accuracy failure.
    """
    if bool(getattr(arguments, "force_build", False)) or bool(
        getattr(arguments, "no_build", False)
    ):
        return False
    execution = result.get("execution", {})
    comparison = result.get("comparison", {})
    validation = result.get("validation", {})
    raw_result = result.get("raw_result", {})
    return (
        isinstance(execution, Mapping)
        and execution.get("status") == "completed"
        and isinstance(comparison, Mapping)
        and comparison.get("status") == "disagreement"
        and isinstance(validation, Mapping)
        and validation.get("status") == "failed"
        and isinstance(raw_result, Mapping)
        and raw_result.get("bundle_built") is False
    )


@dataclass
class _ReusedBundleRevalidationBudget:
    """Bound forced fresh-bundle confirmations across one validation run."""

    limit: int
    attempts_used: int = 0

    def __post_init__(self) -> None:
        self.limit = max(0, int(self.limit))
        self.attempts_used = min(
            self.limit,
            max(0, int(self.attempts_used)),
        )

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.attempts_used)

    def reserve(self) -> bool:
        if self.remaining == 0:
            return False
        self.attempts_used += 1
        return True

    def record_worker_result(self, result: Mapping[str, Any]) -> None:
        receipt = result.get("bundle_revalidation", {})
        if not isinstance(receipt, Mapping) or receipt.get("attempted") is not True:
            return
        try:
            attempt_count = int(receipt.get("attempt_count", 0) or 0)
        except (TypeError, ValueError):
            attempt_count = 0
        self.attempts_used = min(
            self.limit,
            self.attempts_used + max(0, attempt_count),
        )


def _reused_bundle_revalidation_budget(
    arguments: argparse.Namespace,
) -> _ReusedBundleRevalidationBudget:
    budget = getattr(arguments, "_reused_bundle_revalidation_budget", None)
    if isinstance(budget, _ReusedBundleRevalidationBudget):
        return budget
    budget = _ReusedBundleRevalidationBudget(
        limit=int(
            getattr(
                arguments,
                "reused_bundle_revalidation_limit",
                DEFAULT_REUSED_BUNDLE_REVALIDATION_LIMIT,
            )
        ),
        attempts_used=int(
            getattr(
                arguments,
                "reused_bundle_revalidation_attempts_used",
                0,
            )
        ),
    )
    setattr(arguments, "_reused_bundle_revalidation_budget", budget)
    return budget


def _archive_reused_bundle_failure(
    *,
    case_dir: Path,
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Preserve the pre-rebuild comparison receipt and its small text logs."""
    archived: dict[str, str] = {}
    comparison = case_dir / "comparison.reused-bundle.json"
    comparison.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    archived["comparison_result"] = str(comparison)
    workload = str(result.get("workload", "") or "")
    summary = case_dir / "validation" / workload / "eval_summary.json"
    if summary.is_file():
        target = summary.with_name("eval_summary.reused-bundle.json")
        summary.replace(target)
        archived["eval_summary.json"] = str(target)
    for name in ("execution.log", "disagreements.jsonl"):
        source = case_dir / name
        if not source.is_file():
            continue
        target = case_dir / f"{source.stem}.reused-bundle{source.suffix}"
        shutil.copy2(source, target)
        archived[name] = str(target)
    return archived


def _reused_bundle_failure_receipt(
    result: Mapping[str, Any],
    archived: Mapping[str, str],
) -> dict[str, Any]:
    """Keep the first disagreement self-contained in the published result."""
    execution = result.get("execution", {})
    validation = result.get("validation", {})
    comparison = result.get("comparison", {})
    raw_result = result.get("raw_result", {})
    return {
        "execution_status": (
            str(execution.get("status", ""))
            if isinstance(execution, Mapping)
            else ""
        ),
        "validation_status": (
            str(validation.get("status", ""))
            if isinstance(validation, Mapping)
            else ""
        ),
        "comparison_status": (
            str(comparison.get("status", ""))
            if isinstance(comparison, Mapping)
            else ""
        ),
        "bundle_built": (
            raw_result.get("bundle_built")
            if isinstance(raw_result, Mapping)
            else None
        ),
        "error_type": (
            str(raw_result.get("error_type", "") or "")
            if isinstance(raw_result, Mapping)
            else ""
        ),
        "error": (
            str(raw_result.get("error", "") or "")
            if isinstance(raw_result, Mapping)
            else ""
        ),
        "metrics": (
            dict(comparison.get("metrics", {}))
            if isinstance(comparison, Mapping)
            and isinstance(comparison.get("metrics"), Mapping)
            else {}
        ),
        "artifacts": dict(archived),
    }


def _bundle_revalidation_outcome(result: Mapping[str, Any]) -> str:
    raw_result = result.get("raw_result", {})
    rebuilt = (
        raw_result.get("bundle_built")
        if isinstance(raw_result, Mapping)
        else None
    )
    execution = result.get("execution", {})
    validation = result.get("validation", {})
    if not isinstance(execution, Mapping) or execution.get("status") != "completed":
        return "rebuild_execution_error"
    if rebuilt is not True:
        return "rebuild_not_confirmed"
    if isinstance(validation, Mapping) and validation.get("status") == "passed":
        return "recovered_after_rebuild"
    return "confirmed_after_rebuild"


def run_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    workload = _required_workload(binding)
    case_dir = _case_directory(Path(arguments.output), binding)
    case_dir.mkdir(parents=True, exist_ok=True)
    dataset_command = shlex.join([sys.executable, *sys.argv])
    suite = suites[workload]
    task_type, user_contract = _suite_task_metadata(suite)
    model = task_models[binding.model]
    task_strategy = str(model.get("task_strategy", "") or "")
    dataset = (
        Path(arguments.dataset)
        if arguments.dataset
        else _dataset_path(suite, arguments.dataset_root)
    )
    if arguments.dataset is None and not dataset.is_file():
        message = f"dataset does not exist: {dataset}"
        execution_log = case_dir / "execution.log"
        execution_log.write_text(
            f"Accuracy preflight failed\n{message}\n",
            encoding="utf-8",
        )
        result = _normalize_result(
            {
                "model": binding.model,
                "workload": workload,
                "family": str(model.get("family", "") or ""),
                "operation": _operation_for_task_strategy(task_strategy),
                "task_strategy": task_strategy,
                "task_type": task_type,
                "user_contract": user_contract,
                "executor": "dataset_preflight",
                "execution": {
                    "status": "error",
                    "exit_code": 1,
                    "retryable": False,
                },
                "comparison": {
                    "status": "not_run",
                    "mode": "",
                    "primary_metric": None,
                    "metrics": {},
                    "failures": [],
                },
                "validation": {"status": "failed"},
                "failure_stage": "preflight",
                "failure_domain": "data-artifact",
                "failure_code": "dataset_missing",
                "reference_environment": [],
                "reproduce": {
                    "dataset": {
                        "command": dataset_command,
                        "sample_limit": int(arguments.limit or 0),
                        "prepared_input_count": 0,
                    },
                    "hf": [],
                    "trtmc": [],
                },
                "raw_result": {
                    "status": "failed",
                    "error_type": "DatasetNotFoundError",
                    "error": message,
                },
                "raw_result_path": "",
                "disagreements": {
                    "count": 0,
                    "path": "disagreements.jsonl",
                    "inline_limit": trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
                    "reference_vanilla_available": False,
                    "trtmc_vanilla_available": False,
                },
                "execution_log": str(execution_log),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        (case_dir / "disagreements.jsonl").write_text("", encoding="utf-8")
        (case_dir / "comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result
    profiles = _binding_profiles(
        binding,
        task_models=task_models,
        suites=suites,
    )
    environment = ensure_environments(profiles, str(arguments.hf_python))
    reference_sources = ensure_reference_sources(
        str(task_models[binding.model].get("family", "") or ""),
        Path(arguments.reference_cache_dir),
        task_models[binding.model].get("model_reference_cache"),
        source_cache_root=(
            Path(arguments.reference_source_cache_dir)
            if arguments.reference_source_cache_dir is not None
            else None
        ),
    )
    process_env = _source_environment()
    process_env.update(environment.overrides)
    process_env.update(reference_sources.environment)
    command = _comparison_command(
        binding,
        case_dir=case_dir,
        dataset=dataset,
        arguments=arguments,
        reference_python=environment.base_python,
        reference_sources=reference_sources,
    )
    returncode = _run_subprocess(command, case_dir / "execution.log", process_env)
    result = _comparison_result(
        binding,
        case_dir=case_dir,
        returncode=returncode,
        reference_environment=environment,
        dataset_command=dataset_command,
        sample_limit=int(arguments.limit or 0),
        family=str(model.get("family", "") or ""),
        operation=_operation_for_task_strategy(task_strategy),
        task_strategy=task_strategy,
        task_type=task_type,
        user_contract=user_contract,
    )
    if _should_revalidate_reused_bundle(result, arguments):
        budget = _reused_bundle_revalidation_budget(arguments)
        initial_result = result
        if budget.reserve():
            archived = _archive_reused_bundle_failure(
                case_dir=case_dir,
                result=initial_result,
            )
            rebuild_command = [*command, "--force-build"]
            rebuild_returncode = _run_subprocess(
                rebuild_command,
                case_dir / "execution.log",
                process_env,
            )
            result = _comparison_result(
                binding,
                case_dir=case_dir,
                returncode=rebuild_returncode,
                reference_environment=environment,
                dataset_command=dataset_command,
                sample_limit=int(arguments.limit or 0),
            )
            result["bundle_revalidation"] = {
                "attempted": True,
                "trigger": "accuracy_failure_with_reused_bundle",
                "attempt_count": 1,
                "run_attempt_limit": budget.limit,
                "run_attempts_used": budget.attempts_used,
                "outcome": _bundle_revalidation_outcome(result),
                "initial": _reused_bundle_failure_receipt(
                    initial_result,
                    archived,
                ),
                "rebuild_command": shlex.join(rebuild_command),
            }
        else:
            result["bundle_revalidation"] = {
                "attempted": False,
                "trigger": "accuracy_failure_with_reused_bundle",
                "attempt_count": 0,
                "run_attempt_limit": budget.limit,
                "run_attempts_used": budget.attempts_used,
                "outcome": "not_attempted_run_limit_reached",
                "initial": _reused_bundle_failure_receipt(
                    initial_result,
                    {},
                ),
            }

    comparison = case_dir / "comparison.json"
    comparison.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _source_revision() -> str:
    configured = os.environ.get("TRTMC_VALIDATION_SOURCE_REVISION", "").strip()
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_build_identity(arguments: argparse.Namespace) -> dict[str, Any]:
    """Fail before GPU work when source and native build provenance differ."""
    expected_revision = _source_revision()
    if not expected_revision:
        raise ValidationError(
            "cannot determine the validation source revision for build identity preflight"
        )

    benchmark_binary = arguments.benchmark_binary.expanduser().resolve()
    build_root = benchmark_binary.parent
    trtmc_binary = arguments.trtmc_binary.expanduser().resolve()
    backend_dir = (
        arguments.backend_dir.expanduser().resolve()
        if arguments.backend_dir
        else build_root
    )
    model_plugin_dir = (
        arguments.model_plugin_dir.expanduser().resolve()
        if arguments.model_plugin_dir
        else build_root / "models"
    )
    worker = build_root / "trtmc_benchmark_worker"

    expected_paths = {
        "trtmc binary": build_root / "trtmc",
        "dataset benchmark": build_root / "trtmc_dataset_benchmark",
        "backend directory": build_root,
        "model plugin directory": build_root / "models",
    }
    actual_paths = {
        "trtmc binary": trtmc_binary,
        "dataset benchmark": benchmark_binary,
        "backend directory": backend_dir,
        "model plugin directory": model_plugin_dir,
    }
    for label, expected in expected_paths.items():
        actual = actual_paths[label]
        if actual != expected.resolve():
            raise ValidationError(
                f"{label} resolves to {actual}, outside the benchmark worker "
                f"build root {build_root}"
            )

    required_files = {
        "trtmc binary": trtmc_binary,
        "dataset benchmark": benchmark_binary,
        "benchmark worker": worker,
        "TensorRT backend": build_root / "libtrtmc_backend_trt.so",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ValidationError(
                f"{label} is missing for build identity preflight: {path}"
            )
    if not model_plugin_dir.is_dir():
        raise ValidationError(
            "model plugin directory is missing for build identity preflight: "
            f"{model_plugin_dir}"
        )

    try:
        metadata = worker_metadata(worker)
    except BenchmarkError as exc:
        raise ValidationError(
            f"cannot verify benchmark worker build identity: {exc}"
        ) from exc
    build = metadata.get("build", {})
    embedded_revision = str(build.get("source_revision", "") or "").strip()
    configuration = str(build.get("configuration", "") or "").strip()
    if not embedded_revision or embedded_revision == "unknown":
        raise ValidationError(
            "benchmark worker metadata is missing an embedded source revision"
        )
    if embedded_revision != expected_revision:
        raise ValidationError(
            "benchmark worker source revision mismatch: "
            f"embedded {embedded_revision}, expected {expected_revision}"
        )
    if not configuration or configuration == "unknown":
        raise ValidationError(
            "benchmark worker metadata is missing its build configuration"
        )

    artifacts = {
        label: {
            "path": str(path),
            "sha256": _sha256(path),
        }
        for label, path in required_files.items()
    }
    return {
        "schema_version": "trtmc.validation-build-identity/v1",
        "source_revision": expected_revision,
        "embedded_source_revision": embedded_revision,
        "build_configuration": configuration,
        "build_root": str(build_root),
        "backend_dir": str(backend_dir),
        "model_plugin_dir": str(model_plugin_dir),
        "artifacts": artifacts,
    }


def _write_build_identity(
    output: Path,
    identity: Mapping[str, Any],
) -> Path:
    """Persist the validated native-build receipt before model execution."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "build-identity.json"
    path.write_text(
        json.dumps(identity, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _campaign_started_at(output: Path, fallback: str) -> str:
    path = output / "run.json"
    if not path.is_file() or next(output.glob("*/*/comparison.json"), None) is None:
        return fallback
    try:
        started_at = json.loads(path.read_text(encoding="utf-8")).get("started_at")
        if not isinstance(started_at, str) or not started_at:
            return fallback
        datetime.fromisoformat(started_at)
    except (OSError, ValueError, json.JSONDecodeError):
        return fallback
    return started_at


def _query_nvidia_smi_gpus() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(f"could not query GPU identity: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise ValidationError(f"could not query GPU identity: {detail}")

    inventory: list[dict[str, Any]] = []
    for row in csv.reader(result.stdout.splitlines()):
        values = [value.strip() for value in row]
        if len(values) != 4:
            raise ValidationError("nvidia-smi returned malformed GPU identity data")
        try:
            index = int(values[0])
        except ValueError as exc:
            raise ValidationError("nvidia-smi returned a non-numeric GPU index") from exc
        uuid, name, pci_bus_id = values[1:]
        if not uuid or not name or not pci_bus_id:
            raise ValidationError("nvidia-smi returned incomplete GPU identity data")
        inventory.append(
            {
                "nvidia_smi_index": index,
                "uuid": uuid,
                "name": name,
                "pci_bus_id": pci_bus_id,
            }
        )
    return inventory


def _query_cuda_runtime_gpus() -> list[dict[str, Any]]:
    """Query identity through CUDA on platforms that do not ship nvidia-smi."""
    try:
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        raise ValidationError(f"could not import CUDA runtime bindings: {exc}") from exc

    try:
        status, count = cudart.cudaGetDeviceCount()
        if int(status) != 0:
            raise ValidationError(f"cudaGetDeviceCount failed with status {status}")
        inventory: list[dict[str, Any]] = []
        for index in range(int(count)):
            status, properties = cudart.cudaGetDeviceProperties(index)
            if int(status) != 0:
                raise ValidationError(
                    f"cudaGetDeviceProperties({index}) failed with status {status}"
                )
            status, pci_bus_id = cudart.cudaDeviceGetPCIBusId(32, index)
            if int(status) != 0:
                raise ValidationError(
                    f"cudaDeviceGetPCIBusId({index}) failed with status {status}"
                )
            name = properties.name
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace").rstrip("\x00")
            raw_uuid = bytes(properties.uuid.bytes)
            try:
                gpu_uuid = f"GPU-{uuidlib.UUID(bytes=raw_uuid)}"
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"CUDA runtime returned an invalid UUID for device {index}"
                ) from exc
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("utf-8", errors="replace")
            pci_bus_id = str(pci_bus_id).rstrip("\x00 ").strip()
            if not str(name).strip() or not pci_bus_id:
                raise ValidationError(
                    f"CUDA runtime returned incomplete GPU identity for device {index}"
                )
            inventory.append(
                {
                    "cuda_runtime_index": index,
                    "uuid": gpu_uuid,
                    "name": str(name).strip(),
                    "pci_bus_id": pci_bus_id,
                }
            )
        return inventory
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"could not query CUDA runtime GPU identity: {exc}") from exc


def _resolve_cuda_devices(
    cuda_visible_devices: str,
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    visible = cuda_visible_devices.strip()
    if visible.lower() in {"-1", "none", "void"}:
        return []
    selectors = (
        [str(device["nvidia_smi_index"]) for device in inventory]
        if not visible or visible.lower() == "all"
        else [selector.strip() for selector in visible.split(",")]
    )
    if not selectors or any(not selector for selector in selectors):
        raise ValidationError("CUDA_VISIBLE_DEVICES contains an empty selector")

    resolved: list[dict[str, Any]] = []
    for logical_index, selector in enumerate(selectors):
        if selector.isdigit():
            matches = [
                device for device in inventory if int(device["nvidia_smi_index"]) == int(selector)
            ]
        else:
            matches = [
                device
                for device in inventory
                if str(device["uuid"]) == selector or str(device["uuid"]).startswith(selector)
            ]
        if len(matches) != 1:
            raise ValidationError(
                "CUDA_VISIBLE_DEVICES selector does not resolve to exactly one "
                f"GPU identity: {selector}"
            )
        device = dict(matches[0])
        device["cuda_logical_index"] = logical_index
        resolved.append(device)
    return resolved


def _resolve_cuda_runtime_devices(
    cuda_visible_devices: str,
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    visible = cuda_visible_devices.strip()
    if visible.lower() in {"-1", "none", "void"}:
        return []
    selectors = (
        [str(device["cuda_runtime_index"]) for device in inventory]
        if not visible or visible.lower() == "all"
        else [selector.strip() for selector in visible.split(",")]
    )
    if not selectors or any(not selector for selector in selectors):
        raise ValidationError("CUDA_VISIBLE_DEVICES contains an empty selector")

    resolved: list[dict[str, Any]] = []
    for logical_index, selector in enumerate(selectors):
        if selector.isdigit():
            matches = [
                device
                for device in inventory
                if int(device["cuda_runtime_index"]) == int(selector)
            ]
        else:
            matches = [
                device
                for device in inventory
                if str(device["uuid"]) == selector or str(device["uuid"]).startswith(selector)
            ]
        if len(matches) != 1:
            raise ValidationError(
                "CUDA_VISIBLE_DEVICES selector does not resolve to exactly one "
                f"CUDA runtime GPU identity: {selector}"
            )
        device = dict(matches[0])
        device["cuda_logical_index"] = logical_index
        resolved.append(device)
    return resolved


def _runtime_gpu_devices(cuda_visible_devices: str) -> list[dict[str, Any]]:
    try:
        return _resolve_cuda_devices(
            cuda_visible_devices,
            _query_nvidia_smi_gpus(),
        )
    except ValidationError as nvidia_smi_error:
        try:
            return _resolve_cuda_runtime_devices(
                cuda_visible_devices,
                _query_cuda_runtime_gpus(),
            )
        except ValidationError as cuda_runtime_error:
            raise ValidationError(
                "could not query GPU identity with nvidia-smi or CUDA runtime: "
                f"nvidia-smi: {nvidia_smi_error}; CUDA runtime: {cuda_runtime_error}"
            ) from cuda_runtime_error


def write_run_metadata(
    output: Path,
    *,
    cuda_visible_devices: str | None = None,
) -> Path:
    started_at = _campaign_started_at(output, _utc_now().isoformat())
    effective_cuda_visible_devices = (
        cuda_visible_devices.strip()
        if cuda_visible_devices is not None and cuda_visible_devices.strip()
        else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    )
    gpu_devices = _runtime_gpu_devices(effective_cuda_visible_devices)
    if not gpu_devices:
        raise ValidationError(
            "GPU identity is unavailable; refusing to write ambiguous run metadata"
        )
    metadata = {
        "schema_version": "trtmc.validation-run/v1",
        "source_revision": _source_revision(),
        "hostname": platform.node(),
        "cuda_visible_devices": effective_cuda_visible_devices,
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
        "gpu_identity_source": (
            "nvidia-smi"
            if all("nvidia_smi_index" in device for device in gpu_devices)
            else "cuda-runtime"
        ),
        "gpu_devices": gpu_devices,
        "command": shlex.join(sys.argv),
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
    }
    path = output / "run.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def finalize_run_metadata(output: Path) -> Path:
    path = output / "run.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    finished_at = _utc_now()
    metadata["finished_at"] = finished_at.isoformat()
    metadata["duration_seconds"] = _elapsed_seconds(
        metadata.get("started_at"),
        finished_at,
    )
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _report_provenance(run: Mapping[str, Any]) -> str:
    fields = [
        ("source", run.get("source_revision")),
        ("host", run.get("hostname")),
    ]
    gpu_devices = run.get("gpu_devices", [])
    if isinstance(gpu_devices, Sequence) and not isinstance(gpu_devices, (str, bytes)):
        for device in gpu_devices:
            if not isinstance(device, Mapping):
                continue
            logical_index = device.get("cuda_logical_index")
            identity = [str(device.get("name", "") or "unknown GPU")]
            for label, key in (
                ("uuid", "uuid"),
                ("pci", "pci_bus_id"),
                ("runtime-nvidia-smi-index", "nvidia_smi_index"),
            ):
                value = device.get(key)
                if value not in (None, ""):
                    identity.append(f"{label}={value}")
            details = "; ".join(identity[1:])
            display = identity[0] + (f" ({details})" if details else "")
            fields.append(
                (
                    f"GPU logical {logical_index}",
                    display,
                )
            )
    if run.get("cuda_visible_devices") and not any(
        name.startswith("GPU logical ") for name, _ in fields
    ):
        fields.append(("GPU identity", "not recorded"))
    fields.append(
        (
            "CUDA_VISIBLE_DEVICES(process-local)",
            run.get("cuda_visible_devices"),
        )
    )
    return " · ".join(f"{name}={value}" for name, value in fields if value)


def _elapsed_seconds(
    started_at: Any,
    finished_at: datetime,
) -> float | None:
    if not isinstance(started_at, str) or not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return round(max(0.0, (finished_at - started).total_seconds()), 3)


def _format_duration(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return ""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def _merge_commands_from_result_logs(result: dict[str, Any]) -> None:
    raw_result = result.get("raw_result", {})
    work_dir = raw_result.get("work_dir") if isinstance(raw_result, dict) else None
    if not work_dir:
        return
    root = Path(str(work_dir))
    if not root.is_dir():
        return
    discovered = _commands_from_logs(root)
    reproduce = result.get("reproduce", {})
    if not isinstance(reproduce, dict):
        reproduce = {}
    for kind in ("hf", "trtmc"):
        existing = _string_list(reproduce.get(kind, []))
        extra = [command for command in existing if command not in discovered[kind]]
        discovered[kind] = (extra + discovered[kind])[:MAX_REPRO_COMMANDS_PER_BACKEND]
        discovered["command_count"][kind] += len(extra)
    discovered["dataset"] = reproduce.get("dataset", {})
    result["reproduce"] = _normalize_reproduction(discovered)


def _refresh_disagreement_artifact(
    result: dict[str, Any],
    case_dir: Path,
) -> None:
    raw_result = result.get("raw_result", {})
    work_dir = raw_result.get("work_dir") if isinstance(raw_result, dict) else None
    if not work_dir or not Path(str(work_dir)).is_dir():
        return
    result["disagreements"] = trtmc_disagreements.build_disagreement_artifact(
        work_dir=Path(str(work_dir)),
        case_dir=case_dir,
    )


def _result_commands(result: Mapping[str, Any], kind: str) -> list[str]:
    reproduce = result.get("reproduce", {})
    if not isinstance(reproduce, dict):
        return []
    commands = reproduce.get(kind, [])
    if not isinstance(commands, list):
        return []
    return [str(command) for command in commands if str(command).strip()]


def _render_command_group(
    label: str,
    commands: Sequence[str],
    *,
    total: int | None = None,
    logs: Sequence[str] = (),
) -> str:
    if not commands:
        body = '<span class="unavailable">Not reached; see comparison.json.</span>'
    else:
        shell = "\n".join(f"$ {command}" for command in commands)
        body = f"<pre><code>{html.escape(shell)}</code></pre>"
    command_total = len(commands) if total is None else total
    if command_total > len(commands):
        locations = ", ".join(logs) or "comparison.json"
        body += (
            f'<div class="detail">Showing {len(commands)} of {command_total} commands. '
            f"Full command log: {html.escape(locations)}.</div>"
        )
    return f"<h4>{html.escape(label)}</h4>{body}"


def _reproduction_count(result: Mapping[str, Any], kind: str) -> int:
    reproduce = result.get("reproduce", {})
    counts = reproduce.get("command_count", {}) if isinstance(reproduce, dict) else {}
    commands = _result_commands(result, kind)
    try:
        return (
            max(int(counts.get(kind)), len(commands)) if isinstance(counts, dict) else len(commands)
        )
    except (TypeError, ValueError):
        return len(commands)


def _reproduction_logs(result: Mapping[str, Any], kind: str) -> list[str]:
    reproduce = result.get("reproduce", {})
    logs = reproduce.get("command_logs", {}) if isinstance(reproduce, dict) else {}
    return _string_list(logs.get(kind, [])) if isinstance(logs, dict) else []


def _dataset_reproduction(result: Mapping[str, Any]) -> tuple[str, int, int]:
    reproduce = result.get("reproduce", {})
    dataset = reproduce.get("dataset", {}) if isinstance(reproduce, dict) else {}
    if not isinstance(dataset, dict):
        return "", 0, 0
    command = str(dataset.get("command", "") or "")
    try:
        sample_limit = int(dataset.get("sample_limit", 0) or 0)
        prepared = int(dataset.get("prepared_input_count", 0) or 0)
    except (TypeError, ValueError):
        sample_limit = 0
        prepared = 0
    return command, sample_limit, prepared


def _selected_sample_count(result: Mapping[str, Any]) -> int | None:
    reproduce = result.get("reproduce", {})
    dataset = reproduce.get("dataset", {}) if isinstance(reproduce, dict) else {}
    if not isinstance(dataset, dict):
        return None
    _command, sample_limit, prepared = _dataset_reproduction(result)
    if "prepared_input_count" in dataset:
        return min(sample_limit, prepared) if sample_limit > 0 else prepared
    return sample_limit if sample_limit > 0 else None


def _representative_note(result: Mapping[str, Any]) -> str:
    reproduce = result.get("reproduce", {})
    representative = reproduce.get("representative", {}) if isinstance(reproduce, dict) else {}
    if not isinstance(representative, dict):
        return ""
    sample_id = str(representative.get("sample_id", "") or "")
    if not sample_id:
        return ""
    reason = str(representative.get("reason", "") or "").replace("_", " ")
    return (
        '<div class="detail">Representative sample: '
        f"{html.escape(sample_id)}"
        f" ({html.escape(reason)}).</div>"
    )


def _json_preview(value: Any, *, max_characters: int = 2000) -> str:
    rendered = json.dumps(value, indent=2, ensure_ascii=False)
    if len(rendered) > max_characters:
        rendered = rendered[:max_characters] + "\n... see disagreements.jsonl"
    return f"<pre><code>{html.escape(rendered)}</code></pre>"


def _render_vanilla_command(label: str, command: str) -> str:
    if command:
        body = f"<pre><code>{html.escape('$ ' + command)}</code></pre>"
    else:
        body = (
            '<span class="unavailable">Native single-sample command unavailable '
            "for this backend.</span>"
        )
    return f"<h5>{html.escape(label)}</h5>{body}"


def _render_failure_media(
    record: Mapping[str, Any],
    *,
    asset_base: Path,
) -> str:
    artifacts = record.get("artifacts", {})
    media = artifacts.get("media", []) if isinstance(artifacts, dict) else []
    if not isinstance(media, list):
        return ""
    rendered = []
    for item in media:
        if not isinstance(item, dict):
            continue
        label = html.escape(str(item.get("label", "artifact")))
        relative_path = str(item.get("path", "") or "")
        if not relative_path:
            continue
        href = html.escape(str(asset_base / relative_path))
        body = _failure_media_tag(str(item.get("kind", "")), href, label)
        if not body:
            continue
        rendered.append(f"<figure><figcaption>{label}</figcaption>{body}</figure>")
    if not rendered:
        return ""
    return '<h5>Failure media</h5><div class="failure-media">' + "".join(rendered) + "</div>"


def _failure_media_tag(kind: str, href: str, label: str) -> str:
    if kind == "image":
        return f'<a href="{href}"><img src="{href}" alt="{label}" loading="lazy"></a>'
    if kind == "audio":
        return f'<audio controls preload="metadata" src="{href}"></audio>'
    if kind == "video":
        return f'<video controls preload="metadata" src="{href}"></video>'
    return ""


def _render_disagreement_record(
    record: Mapping[str, Any],
    *,
    asset_base: Path,
) -> str:
    sample_id = str(record.get("sample_id", "") or "unknown sample")
    reason = str(record.get("reason", "") or "comparison mismatch").replace("_", " ")
    reproduce = record.get("reproduce", {})
    reproduce = reproduce if isinstance(reproduce, dict) else {}
    return (
        '<details class="sample-difference">'
        f"<summary>{html.escape(sample_id)} · {html.escape(reason)}</summary>"
        '<div class="difference-grid">'
        f"<section><h5>Input</h5>{_json_preview(record.get('input', {}))}</section>"
        f"<section><h5>Reference result</h5>{_json_preview(record.get('reference_result', {}))}</section>"
        f"<section><h5>TRTMC result</h5>{_json_preview(record.get('trtmc_result', {}))}</section>"
        f"<section><h5>Comparison</h5>{_json_preview(record.get('comparison', {}))}</section>"
        "</div>"
        f"{_render_failure_media(record, asset_base=asset_base)}"
        f"{_render_vanilla_command('Reference vanilla command', str(reproduce.get('reference', '') or ''))}"
        f"{_render_vanilla_command('TRTMC vanilla command', str(reproduce.get('trtmc', '') or ''))}"
        "</details>"
    )


def _render_disagreements(
    result: Mapping[str, Any],
    *,
    case_dir: Path,
    artifact_href: str,
) -> str:
    metadata = result.get("disagreements", {})
    if not isinstance(metadata, dict):
        return ""
    try:
        count = int(metadata.get("count", 0) or 0)
        limit = int(
            metadata.get(
                "inline_limit",
                trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
            )
        )
    except (TypeError, ValueError):
        return ""
    if count <= 0:
        return ""
    artifact_name = str(metadata.get("path", "disagreements.jsonl"))
    preview = trtmc_disagreements.load_disagreement_preview(
        case_dir / artifact_name,
        limit=limit,
    )
    comparison = result.get("comparison", {})
    failed = isinstance(comparison, dict) and comparison.get("status") == "disagreement"
    noun = "failed samples" if failed else "sample differences"
    asset_base = Path(artifact_href).parent
    records = "".join(
        _render_disagreement_record(record, asset_base=asset_base) for record in preview
    )
    more = ""
    if count > len(preview):
        more = (
            f'<div class="detail">Showing {len(preview)} of {count}. '
            f'<a href="{html.escape(artifact_href)}">View all in disagreements.jsonl</a>.</div>'
        )
    return (
        f'<details class="failure-details"><summary>{count} {noun} · '
        "results and vanilla commands</summary>"
        f"{records}{more}</details>"
    )


def _render_reproduction(
    result: Mapping[str, Any],
    *,
    case_dir: Path,
    artifact_href: str,
) -> str:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        return f'<span class="unavailable">{html.escape(not_compared_reason)}</span>'
    reference_commands = _result_commands(result, "hf")
    trtmc_commands = _result_commands(result, "trtmc")
    dataset_command, _sample_limit, _prepared = _dataset_reproduction(result)
    reference_total = _reproduction_count(result, "hf")
    trtmc_total = _reproduction_count(result, "trtmc")
    selected_samples = _selected_sample_count(result)
    if selected_samples is not None:
        sample_label = "sample" if selected_samples == 1 else "samples"
        dataset_label = f"Dataset slice ({selected_samples} {sample_label})"
    else:
        dataset_label = "Full dataset"
    summary = (
        f"Dataset · Reference {len(reference_commands)}/{reference_total} · "
        f"TRTMC {len(trtmc_commands)}/{trtmc_total}"
    )
    return (
        f"{_render_disagreements(result, case_dir=case_dir, artifact_href=artifact_href)}"
        f"<details><summary>{summary}</summary>"
        '<div class="commands">'
        f"{_render_command_group(dataset_label, [dataset_command] if dataset_command else [])}"
        f"{_render_command_group('Reference sample', reference_commands, total=reference_total, logs=_reproduction_logs(result, 'hf'))}"
        f"{_render_command_group('TRTMC sample', trtmc_commands, total=trtmc_total, logs=_reproduction_logs(result, 'trtmc'))}"
        f"{_representative_note(result)}"
        "</div></details>"
    )


def _reference_result_status(result: Mapping[str, Any]) -> str:
    raw_result = result.get("raw_result", {})
    if not isinstance(raw_result, dict):
        return ""
    return str(raw_result.get("hf_cache_status") or raw_result.get("hf_reference_status") or "")


def _signal(status: str, labels: Mapping[str, str]) -> str:
    label = labels.get(status, status.replace("_", " ").title())
    safe_status = status if status.replace("_", "").isalnum() else "unknown"
    return (
        f'<span class="signal signal-{safe_status}">'
        '<span class="signal-light"></span>'
        f"{html.escape(label)}</span>"
    )


def _render_execution(result: Mapping[str, Any]) -> str:
    execution = result.get("execution", {})
    status = str(execution.get("status", "error")) if isinstance(execution, dict) else "error"
    rendered = _signal(
        status,
        {"completed": "Completed", "error": "Error", "not_run": "Not run"},
    )
    attempts = int(execution.get("attempt_count", 1)) if isinstance(execution, dict) else 1
    if attempts > 1:
        outcome = "recovered" if status == "completed" else "failed"
        rendered += f'<div class="detail">{outcome} after {attempts} attempts</div>'
    return rendered


def _render_reference(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return _signal("not_run", {"not_run": "Not configured"})
    status = _reference_result_status(result)
    display_status = {
        "reused": "cached",
        "adopted": "cached",
        "generated": "completed",
        "ran": "completed",
    }.get(status, "not_run")
    label = {
        "reused": "Reused",
        "adopted": "Adopted",
        "generated": "Generated",
        "ran": "Generated",
    }.get(status, "Not recorded")
    environments = ", ".join(
        str(item.get("name", ""))
        for item in result.get("reference_environment", [])
        if isinstance(item, dict)
    )
    detail = f'<div class="detail">{html.escape(environments)}</div>' if environments else ""
    return _signal(display_status, {display_status: label}) + detail


def _render_comparison(result: Mapping[str, Any]) -> str:
    comparison = result.get("comparison", {})
    if not isinstance(comparison, dict):
        return _signal("not_run", {"not_run": "Not compared"})
    status = str(comparison.get("status", "not_run"))
    signal = _signal(
        status,
        {
            "agreement": "Agreement",
            "disagreement": "Disagreement",
            "not_run": "Not compared",
        },
    )
    mode = str(comparison.get("mode", "") or "")
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    details = [value for value in (mode, not_compared_reason) if value]
    exclusion = result.get("platform_exclusion", {})
    if isinstance(exclusion, Mapping) and exclusion:
        reason = str(exclusion.get("reason", "") or "")
        detail = ": ".join(value for value in ("Platform excluded", reason) if value)
        if detail and detail not in details:
            details.append(detail)
    contract = result.get("precision_contract", {})
    if isinstance(contract, Mapping) and contract:
        base = str(contract.get("trtmc_base_precision", "") or "").upper()
        quantization = str(contract.get("trtmc_quantization", "") or "").upper()
        reference = str(contract.get("reference_precision", "") or "").upper()
        candidate = (
            f"{quantization} ({base} base)" if quantization and quantization != "NONE" else base
        )
        if candidate and reference:
            details.append(f"TRTMC {candidate} vs HF {reference}")
        comparison_kind = str(contract.get("comparison", "") or "")
        if comparison_kind == "quantized_vs_unquantized_reference":
            details.append("Quantized candidate vs unquantized reference")
        elif comparison_kind == "aligned":
            details.append("Aligned precision")
        elif comparison_kind == "reference_defined":
            details.append("Reference-defined precision")
    return signal + "".join(
        f'<div class="detail">{html.escape(detail)}</div>' for detail in details
    )


def _format_metric_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    is_ratio = any(
        token in name
        for token in ("accuracy", "agreement", "pass_rate", "exact_match", "divergence_rate")
    )
    if is_ratio:
        return f"{value * 100:.2f}%"
    if value and abs(value) < 0.001:
        return f"{value:.3e}"
    return f"{value:.6f}"


def _render_metrics(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return '<span class="unavailable">Not compared</span>'
    comparison = result.get("comparison", {})
    metrics = comparison.get("metrics", {}) if isinstance(comparison, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        return '<span class="unavailable">No metrics</span>'
    primary = comparison.get("primary_metric")
    primary_name = primary.get("name") if isinstance(primary, dict) else None
    ordered = ([primary_name] if primary_name in metrics else []) + [
        name for name in metrics if name != primary_name
    ]
    visible = ordered[:5]
    rows = [
        (
            f'<div class="metric{" primary" if name == primary_name else ""}">'
            f"<span>{html.escape(str(name))}</span>"
            f"<strong>{html.escape(_format_metric_value(str(name), metrics[name]))}</strong>"
            "</div>"
        )
        for name in visible
    ]
    remaining = len(ordered) - len(visible)
    if remaining:
        rows.append(f'<div class="detail">+{remaining} more in comparison.json</div>')
    return "".join(rows)


def _render_validation(result: Mapping[str, Any]) -> str:
    validation = result.get("validation", {})
    status = str(validation.get("status", "failed")) if isinstance(validation, dict) else "failed"
    return _signal(
        status,
        {
            "passed": "Pass",
            "failed": "Fail",
            "skipped": "Skipped",
            "not_compared": "Not compared",
        },
    )


def _render_samples(result: Mapping[str, Any]) -> str:
    if result.get("not_compared_reason"):
        return "—"
    selected_samples = _selected_sample_count(result)
    return str(selected_samples) if selected_samples is not None else "Full"


def _normalize_result_files(
    result_paths: Sequence[Path],
) -> list[dict[str, Any]]:
    results = []
    for path in result_paths:
        result = _normalize_result(json.loads(path.read_text(encoding="utf-8")))
        _merge_commands_from_result_logs(result)
        _refresh_disagreement_artifact(result, path.parent)
        path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results.append(result)
    return results


def _deduplicate_results(
    result_paths: Sequence[Path],
    results: Sequence[dict[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    selected: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    for path, result in zip(result_paths, results, strict=True):
        key = (
            str(result.get("model", "")),
            str(result.get("workload") or ""),
        )
        current = selected.get(key)
        if current is None or path.parent.name == NOT_COMPARED_DIRECTORY:
            selected[key] = (path, result)
    ordered = sorted(selected.values(), key=lambda item: str(item[0]))
    return (
        [path for path, _result in ordered],
        [result for _path, result in ordered],
    )


def _report_counts(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int], int]:
    validation_counts = {
        name: sum(result["validation"]["status"] == name for result in results)
        for name in ("passed", "failed", "skipped", "not_compared")
    }
    comparison_counts = {
        name: sum(result["comparison"]["status"] == name for result in results)
        for name in ("agreement", "disagreement", "not_run")
    }
    execution_errors = sum(result["execution"]["status"] == "error" for result in results)
    return validation_counts, comparison_counts, execution_errors


def _traffic_light_status(result: Mapping[str, Any]) -> str:
    execution_status = str(result["execution"]["status"])
    validation_status = str(result["validation"]["status"])
    comparison_status = str(result["comparison"]["status"])
    precision = _accuracy_precision(result)
    if execution_status != "completed":
        return "white"
    if "Not recorded" in precision.values():
        return "white"
    if comparison_status not in {"agreement", "disagreement"}:
        return "white"
    if validation_status not in {"passed", "failed"}:
        return "white"
    if comparison_status == "agreement" and validation_status == "passed":
        return "green"
    if comparison_status == "disagreement" or validation_status == "failed":
        return "red"
    return "white"


def _accuracy_precision(result: Mapping[str, Any]) -> dict[str, str]:
    contract = result.get("precision_contract", {})
    contract = contract if isinstance(contract, Mapping) else {}
    raw_result = result.get("raw_result", {})
    raw_result = raw_result if isinstance(raw_result, Mapping) else {}
    reference = (
        contract.get("reference_precision")
        or contract.get("reference_dtype")
        or raw_result.get("reference_precision")
        or raw_result.get("reference_dtype")
    )
    base = contract.get("trtmc_base_precision") or raw_result.get("precision")
    quantization = contract.get("trtmc_quantization")
    if quantization and str(quantization).lower() not in {"none", "false"}:
        candidate = (
            f"{str(quantization).lower()} ({str(base).lower()} base)"
            if base
            else str(quantization).lower()
        )
    else:
        candidate = str(base).lower() if base else ""
    return {
        "reference": str(reference).lower() if reference else "Not recorded",
        "candidate": candidate or "Not recorded",
    }


def _accuracy_issue(result: Mapping[str, Any]) -> dict[str, str] | None:
    if _traffic_light_status(result) != "white":
        return None
    execution = result.get("execution", {})
    execution = execution if isinstance(execution, Mapping) else {}
    if execution.get("status") == "error":
        raw_result = result.get("raw_result", {})
        raw_result = raw_result if isinstance(raw_result, Mapping) else {}
        if raw_result.get("error_type") == "SampleEvidenceError":
            acceptance = raw_result.get("sample_acceptance", {})
            issues = (
                acceptance.get("issues", [])
                if isinstance(acceptance, Mapping)
                else []
            )
            code = str(issues[0].get("code", "invalid_sample_evidence")) if issues else (
                "invalid_sample_evidence"
            )
            return {
                "priority": "P1",
                "stage": "compare",
                "domain": (
                    "data-artifact"
                    if code in {"incomplete_samples", "invalid_sample_counts"}
                    else "policy-config"
                ),
                "code": code,
                "message": str(raw_result.get("error") or code),
            }
        worker_failure = result.get("executor") == "model_worker"
        return {
            "priority": "P1",
            "stage": _accuracy_failure_stage(result),
            "domain": str(result.get("failure_domain") or "harness/unknown"),
            "code": str(
                result.get("failure_code")
                or ("worker_no_result" if worker_failure else "execution_error")
            ),
            "message": str(
                execution.get("last_error")
                or raw_result.get("error")
                or result.get("not_compared_reason")
                or "candidate execution did not produce a comparison"
            ),
        }
    precision = _accuracy_precision(result)
    if "Not recorded" in precision.values():
        return {
            "priority": "P1",
            "stage": "preflight",
            "domain": "policy-config",
            "code": "comparison_contract",
            "message": "Reference and TRTMC compute precision were not both recorded",
        }
    comparison = result.get("comparison", {})
    comparison = comparison if isinstance(comparison, Mapping) else {}
    return {
        "priority": "P1",
        "stage": "compare",
        "domain": "harness/unknown",
        "code": "no_valid_comparison",
        "message": str(
            result.get("not_compared_reason")
            or comparison.get("reason")
            or "comparison evidence is incomplete"
        ),
    }


def _accuracy_failure_stage(result: Mapping[str, Any]) -> str:
    explicit = str(result.get("failure_stage", "") or "").strip()
    if explicit:
        return explicit
    raw_result = result.get("raw_result", {})
    raw_result = raw_result if isinstance(raw_result, Mapping) else {}
    error_type = str(raw_result.get("error_type", "") or "")
    error = str(raw_result.get("error", "") or "")
    if error_type == "ReferenceExecutionError" or "HF reference subprocess failed" in error:
        return "reference"
    if result.get("executor") == "dataset_preflight":
        return "preflight"
    if result.get("executor") == "model_worker":
        return "compare"
    return "candidate"


def _accuracy_result_stage(result: Mapping[str, Any]) -> str:
    execution = result.get("execution", {})
    if isinstance(execution, Mapping) and execution.get("status") == "error":
        return _accuracy_failure_stage(result)
    return "compare"


def _materialize_accuracy_report_artifact(
    output: Path,
    case_dir: Path,
    path: Path,
    *,
    category: str,
) -> str:
    relative_case = case_dir.relative_to(output)
    relative_artifact = path.relative_to(case_dir)
    destination = (
        output
        / "artifacts"
        / "cases"
        / relative_case
        / category
        / relative_artifact
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    return destination.relative_to(output).as_posix()


def _report_log_links(output: Path, case_dir: Path) -> list[dict[str, str]]:
    links = []
    for path in sorted(case_dir.rglob("*.log")):
        if not path.is_file():
            continue
        links.append(
            {
                "label": path.relative_to(case_dir).as_posix(),
                "href": _materialize_accuracy_report_artifact(
                    output,
                    case_dir,
                    path,
                    category="logs",
                ),
            }
        )
    return links


def _report_command_artifacts(
    output: Path,
    case_dir: Path,
    result: Mapping[str, Any],
) -> list[dict[str, str]]:
    reproduce = result.get("reproduce", {})
    reproduce = reproduce if isinstance(reproduce, Mapping) else {}
    command_logs = reproduce.get("command_logs", {})
    command_logs = command_logs if isinstance(command_logs, Mapping) else {}
    requested = {
        str(name)
        for side in ("hf", "trtmc")
        for name in command_logs.get(side, [])
        if str(name).strip()
    }
    artifacts = []
    for name in sorted(requested):
        matches = sorted(path for path in case_dir.rglob(Path(name).name) if path.is_file())
        if not matches:
            continue
        path = matches[0]
        artifacts.append(
            {
                "label": name,
                "href": _materialize_accuracy_report_artifact(
                    output,
                    case_dir,
                    path,
                    category="commands",
                ),
            }
        )
    return artifacts


def _nonnegative_sample_count(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _accuracy_samples(result: Mapping[str, Any]) -> dict[str, int | None]:
    reproduce = result.get("reproduce", {})
    reproduce = reproduce if isinstance(reproduce, Mapping) else {}
    dataset = reproduce.get("dataset", {})
    dataset = dataset if isinstance(dataset, Mapping) else {}
    comparison = result.get("comparison", {})
    comparison = comparison if isinstance(comparison, Mapping) else {}
    metrics = comparison.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}

    planned = _nonnegative_sample_count(dataset.get("sample_limit"))
    if planned == 0:
        planned = None
    evaluated = _nonnegative_sample_count(dataset.get("prepared_input_count"))
    if evaluated is None:
        evaluated = _nonnegative_sample_count(metrics.get("valid_count"))
    if evaluated is None:
        evaluated = _nonnegative_sample_count(metrics.get("sample_count"))
    return {"planned": planned, "evaluated": evaluated}


def _accuracy_gate_sample_count(
    comparison: Mapping[str, Any],
    samples: Mapping[str, int | None],
) -> int | None:
    metrics = comparison.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    valid_count = _nonnegative_sample_count(metrics.get("valid_count"))
    return valid_count if valid_count is not None else samples.get("evaluated")


def _shadow_gate_metrics(
    comparison: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    value = comparison.get("metrics", {})
    metrics = dict(value) if isinstance(value, Mapping) else {}
    metrics.update(
        {
            str(name): metric
            for name, metric in raw_result.items()
            if isinstance(metric, (int, float)) and not isinstance(metric, bool)
        }
    )
    nested = raw_result.get("metrics", {})
    if isinstance(nested, Mapping):
        for name, summary in nested.items():
            if not isinstance(summary, Mapping):
                continue
            for statistic in ("mean", "min", "max"):
                metric = summary.get(statistic)
                if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                    key = str(name) if statistic == "mean" else f"{statistic}_{name}"
                    metrics[key] = metric
    return metrics


_SAMPLE_DIFFERENCE_RAW_KEYS = {
    "generated_token_ids",
    "generated_token_max_score_ids",
    "input_token_ids",
}


def _compact_sample_difference(value: Any) -> Any:
    if isinstance(value, Mapping):
        compact = {
            str(name): _compact_sample_difference(item)
            for name, item in value.items()
            if name not in _SAMPLE_DIFFERENCE_RAW_KEYS and name != "artifacts"
        }
        artifacts = value.get("artifacts")
        if isinstance(artifacts, Mapping) and artifacts.get("media"):
            compact["artifacts"] = {
                "media": _compact_sample_difference(artifacts["media"])
            }
        return compact
    if isinstance(value, list):
        return [_compact_sample_difference(item) for item in value]
    return value


def _public_sample_differences(
    output: Path,
    case_dir: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = result.get("disagreements", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    try:
        count = max(0, int(metadata.get("count", 0) or 0))
        limit = max(
            0,
            int(
                metadata.get(
                    "inline_limit",
                    trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
                )
            ),
        )
    except (TypeError, ValueError):
        count = 0
        limit = 0
    comparison = result.get("comparison", {})
    failed = isinstance(comparison, Mapping) and comparison.get("status") == "disagreement"
    public = {
        "count": count,
        "classification": "failed_samples" if failed else "sample_differences",
        "href": None,
        "preview": [],
    }
    if count <= 0:
        return public
    artifact_name = str(metadata.get("path", "disagreements.jsonl"))
    artifact = case_dir / artifact_name
    if not artifact.is_file():
        return public
    public["href"] = _materialize_accuracy_report_artifact(
        output,
        case_dir,
        artifact,
        category="differences",
    )
    public["preview"] = [
        _compact_sample_difference(record)
        for record in trtmc_disagreements.load_disagreement_preview(
            artifact,
            limit=limit,
        )
    ]
    return public


def _public_accuracy_result(
    output: Path,
    path: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    public = dict(result)
    samples = _accuracy_samples(result)
    comparison = result.get("comparison", {})
    comparison = dict(comparison) if isinstance(comparison, Mapping) else {}
    raw_result = result.get("raw_result", {})
    raw_result = raw_result if isinstance(raw_result, Mapping) else {}
    configured_gates = raw_result.get("configured_gates")
    sample_acceptance = raw_result.get("sample_acceptance")
    policy_mode = str(raw_result.get("gate_policy", "") or "")
    if configured_gates or (policy_mode == "observation_only" and not sample_acceptance):
        comparison["gate_evaluation"] = evaluate_shadow_gates(
            metrics=_shadow_gate_metrics(comparison, raw_result),
            configured_gates=(
                configured_gates if isinstance(configured_gates, Mapping) else {}
            ),
            sample_count=_accuracy_gate_sample_count(comparison, samples),
            policy_mode=policy_mode or "blocking",
            metric_kinds=(
                raw_result.get("gate_metric_kinds")
                if isinstance(raw_result.get("gate_metric_kinds"), Mapping)
                else {}
            ),
        )
    if isinstance(sample_acceptance, Mapping):
        comparison["sample_acceptance"] = dict(sample_acceptance)
    public.update(
        {
            "id": f"{result.get('model', '')}::{result.get('workload') or 'not-compared'}",
            "state": "terminal",
            "result": _traffic_light_status(result),
            "precision": _accuracy_precision(result),
            "samples": samples,
            "comparison": comparison,
            "sample_differences": _public_sample_differences(
                output,
                path.parent,
                result,
            ),
            "issue": _accuracy_issue(result),
            "debug": {
                "result": path.relative_to(output).as_posix(),
                "logs": _report_log_links(output, path.parent),
                "command_artifacts": _report_command_artifacts(
                    output,
                    path.parent,
                    result,
                ),
            },
        }
    )
    return public


def _report_rows(
    output: Path,
    results: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
) -> str:
    rows = []
    for result, path in zip(results, result_paths, strict=True):
        family, operation, task_strategy = _result_model_report_metadata(result)
        task_type, user_contract = _result_task_metadata(result)
        status = _traffic_light_status(result)
        search_filter = " ".join(
            (
                family,
                operation,
                str(result.get("model", "")),
                task_strategy,
                task_type,
                user_contract,
                str(result.get("workload") or ""),
            )
        ).lower()
        relative = path.relative_to(output)
        metadata = result.get("disagreements", {})
        artifact_name = (
            str(metadata.get("path", "disagreements.jsonl"))
            if isinstance(metadata, dict)
            else "disagreements.jsonl"
        )
        artifact_relative = relative.parent / artifact_name
        rows.append(
            f'<tr data-filter-search="{html.escape(search_filter, quote=True)}" '
            f'data-filter-model-type="{html.escape(family, quote=True)}" '
            f'data-filter-operation="{html.escape(operation, quote=True)}" '
            f'data-filter-task-type="{html.escape(task_type, quote=True)}" '
            f'data-filter-status="{status}">'
            f"<td><code>{html.escape(family or '—')}</code></td>"
            f"<td><code>{html.escape(operation or '—')}</code></td>"
            f"<td><code>{html.escape(str(result.get('model', '')))}</code></td>"
            f"<td>{_render_task_type(result)}</td>"
            f"<td>{html.escape(str(result.get('workload') or '—'))}</td>"
            f"<td>{_render_samples(result)}</td>"
            f"<td>{_render_execution(result)}</td>"
            f"<td>{_render_reference(result)}</td>"
            f"<td>{_render_comparison(result)}</td>"
            f"<td>{_render_metrics(result)}</td>"
            f"<td>{_render_validation(result)}</td>"
            f"<td>{_render_reproduction(result, case_dir=path.parent, artifact_href=str(artifact_relative))}</td>"
            f'<td><a href="{html.escape(str(relative))}">comparison.json</a></td>'
            "</tr>"
        )
    return "".join(rows)


def _render_task_type(result: Mapping[str, Any]) -> str:
    task_type, user_contract = _result_task_metadata(result)
    if not task_type:
        return "—"
    contract = f'<div class="detail">{html.escape(user_contract)}</div>' if user_contract else ""
    return f"<strong>{html.escape(task_type)}</strong>{contract}"


def _report_document(
    report: Mapping[str, Any],
    *,
    rows: str,
    comparison_counts: Mapping[str, int],
    execution_errors: int,
    traffic_light_counts: Mapping[str, int],
) -> str:
    results = list(report.get("results", []))
    provenance = _report_provenance(report.get("run", {}))
    duration = _format_duration(report["summary"].get("duration_seconds"))
    duration_summary = f" · {html.escape(duration)} total duration" if duration else ""
    platform_summary = ""
    excluded = int(report["summary"].get("platform_excluded", 0))
    if excluded:
        platform_summary = f" · {excluded} platform excluded"
    filters = render_report_filters(
        row_count=len(results),
        search_placeholder="Model type, operation, model, task type, or workload",
        filters=(
            ReportFilter(
                "model-type",
                "Model type",
                sorted_filter_values(
                    _result_model_report_metadata(result)[0] for result in results
                ),
            ),
            ReportFilter(
                "operation",
                "Operation",
                sorted_filter_values(
                    _result_model_report_metadata(result)[1] for result in results
                ),
            ),
            ReportFilter(
                "task-type",
                "Task type",
                sorted_filter_values(_result_task_metadata(result)[0] for result in results),
            ),
            ReportFilter(
                "status",
                "Status",
                tuple(
                    (status, label)
                    for status, label in (
                        ("green", "Green"),
                        ("yellow", "Yellow"),
                        ("red", "Red"),
                        ("white", "White"),
                    )
                    if any(_traffic_light_status(result) == status for result in results)
                ),
            ),
        ),
    )
    filter_script = render_report_filter_script()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRTMC Reference Consistency Report</title>
<style>
{COMMON_REPORT_STYLES}
.table-wrap table {{ min-width: 1900px; }}
details {{ min-width: 210px; }}
.commands {{ min-width: min(760px, 70vw); padding: 4px 0; }}
.commands h4 {{ margin: 12px 0 4px; }}
.failure-details {{ min-width: min(900px, 75vw); margin-bottom: 10px; }}
.sample-difference {{ margin: 8px 0; padding: 8px; border: 1px solid #dadce0;
                      border-radius: 4px; }}
.sample-difference h5 {{ margin: 10px 0 4px; }}
.difference-grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr));
                    gap: 10px; }}
.difference-grid pre {{ max-height: 260px; overflow: auto; }}
.failure-media {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
.failure-media figure {{ margin: 0; max-width: 360px; }}
.failure-media figcaption {{ color: #5f6368; margin-bottom: 4px; }}
.failure-media img, .failure-media video {{ display: block; max-width: 360px;
                                           max-height: 280px; }}
.failure-media audio {{ width: min(360px, 70vw); }}
.signal {{ display: inline-flex; align-items: center; gap: 7px; font-weight: 650;
           white-space: nowrap; }}
.signal-light {{ width: 10px; height: 10px; border-radius: 50%;
                 background: #80868b; box-shadow: 0 0 0 3px #eef0f1; }}
.signal-completed, .signal-agreement, .signal-passed {{ color: #137333; }}
.signal-completed .signal-light, .signal-agreement .signal-light,
.signal-passed .signal-light {{ background: #1e8e3e; box-shadow: 0 0 0 3px #e6f4ea; }}
.signal-error, .signal-disagreement, .signal-failed {{ color: #b3261e; }}
.signal-error .signal-light, .signal-disagreement .signal-light,
.signal-failed .signal-light {{ background: #d93025; box-shadow: 0 0 0 3px #fce8e6; }}
.signal-skipped {{ color: #8a4f00; }}
.signal-skipped .signal-light {{ background: #f9ab00; box-shadow: 0 0 0 3px #fef7e0; }}
.signal-cached {{ color: #185abc; }}
.signal-cached .signal-light {{ background: #1a73e8; box-shadow: 0 0 0 3px #e8f0fe; }}
.signal-not_run, .signal-not_compared {{ color: #5f6368; }}
.metric {{ display: flex; justify-content: space-between; gap: 14px;
           font-variant-numeric: tabular-nums; font-size: 12px; }}
.metric span {{ color: #5f6368; }}
.metric.primary {{ font-size: 13px; }}
.metric.primary span, .metric.primary strong {{ color: #202124; }}
</style></head><body>
<header class="report-header">
<p class="report-eyebrow">Validation report</p>
<h1>TRTMC Reference Consistency Report</h1>
<p class="purpose">Accuracy and output agreement against the model reference.</p>
</header>
<div class="traffic-summary" title="Agreement · Skipped · Disagreement · Not compared">
🟢 {traffic_light_counts["green"]} &nbsp; 🟡 {traffic_light_counts["yellow"]} &nbsp;
🔴 {traffic_light_counts["red"]} &nbsp; ⚪ {traffic_light_counts["white"]}
</div>
<div class="summary">{report["summary"]["cases"]} cases ·
{comparison_counts["agreement"]} agreements ·
{comparison_counts["disagreement"]} disagreements ·
{comparison_counts["not_run"]} not compared ·
{execution_errors} execution errors ·
{report["summary"]["selected_samples"]} samples{platform_summary}{duration_summary}<br>
{html.escape(provenance)}</div>
{filters}
<div class="table-wrap"><table><thead><tr><th>Model type</th><th>Operation</th><th>Model</th><th>Task type</th><th>Workload</th><th>Samples</th><th>Execution</th>
<th>Reference</th><th>Comparison</th><th>Agreement metrics</th>
<th>Validation</th><th>Vanilla reproduction</th><th>Result</th></tr></thead>
<tbody>{rows}</tbody></table></div>
{filter_script}
</body></html>
"""


def _accuracy_ledger_report_rows(
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger = ExecutionLedger.load(output, task_kind="accuracy")
    terminal_results: list[dict[str, Any]] = []
    public_results: list[dict[str, Any]] = []
    for entry in ledger.snapshot():
        case = entry["case"]
        receipt = entry["receipt"]
        report_fields = case.get("report", {})
        if not isinstance(report_fields, Mapping):
            raise ValidationError(f"invalid Accuracy ledger descriptor for {case['id']!r}")
        if receipt["state"] != "terminal":
            active = receipt["attempts"][-1] if receipt["attempts"] else {}
            evidence = active.get("evidence", {})
            evidence = evidence if isinstance(evidence, Mapping) else {}
            public_results.append(
                {
                    **dict(report_fields),
                    "id": case["id"],
                    "state": receipt["state"],
                    "result": None,
                    "progress": {
                        "stage": receipt["stage"],
                        "attempt": receipt["active_attempt"]
                        or len(receipt["attempts"]),
                    },
                    "precision": {
                        "reference": "Not recorded",
                        "candidate": "Not recorded",
                    },
                    "samples": {
                        "planned": report_fields.get("sample_limit"),
                        "evaluated": None,
                    },
                    "commands": copy.deepcopy(dict(evidence.get("commands", {}))),
                    "debug": {
                        "logs": copy.deepcopy(list(evidence.get("logs", []))),
                        "command_artifacts": [],
                    },
                }
            )
            continue
        result = _normalize_result(receipt["payload"])
        derived = _traffic_light_status(result)
        if receipt["result"] != derived:
            raise ValidationError(
                f"Accuracy ledger result mismatch for {case['id']!r}: "
                f"receipt={receipt['result']}, comparison={derived}"
            )
        result_path = output / str(case["result_path"])
        encoded = json.dumps(result, indent=2, ensure_ascii=False)
        if not result_path.is_file() or result_path.read_text(encoding="utf-8") != encoded:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(result_path)
        terminal_results.append(result)
        public = _public_accuracy_result(output, result_path, result)
        public["progress"] = {
            "stage": receipt["stage"],
            "attempt": len(receipt["attempts"]),
        }
        public_results.append(public)
    return terminal_results, public_results


def write_report(output: Path) -> tuple[Path, Path, dict[str, Any]]:
    if (output / "ledger" / "campaign.json").is_file():
        results, public_results = _accuracy_ledger_report_rows(output)
    else:
        result_paths = sorted(output.glob("*/*/comparison.json"))
        results = _normalize_result_files(result_paths)
        result_paths, results = _deduplicate_results(result_paths, results)
        selected = [
            (path, result)
            for path, result in zip(result_paths, results, strict=True)
            if not isinstance(result.get("platform_exclusion"), Mapping)
        ]
        result_paths = [path for path, _result in selected]
        results = [result for _path, result in selected]
        public_results = [
            _public_accuracy_result(output, path, result)
            for path, result in zip(result_paths, results, strict=True)
        ]
    validation_counts, comparison_counts, execution_errors = _report_counts(results)
    sample_counts = [
        count for result in results if (count := _selected_sample_count(result)) is not None
    ]
    generated_at = _utc_now()
    validation_status = (
        "failed"
        if not results or validation_counts["failed"]
        else "incomplete"
        if (
            validation_counts["not_compared"]
            or validation_counts["skipped"]
            or execution_errors
            or any(_traffic_light_status(result) == "white" for result in results)
        )
        else "passed"
    )
    summary = {
        "cases": len(public_results),
        "execution_completed": sum(
            result["execution"]["status"] == "completed" for result in results
        ),
        "execution_errors": execution_errors,
        "agreements": comparison_counts["agreement"],
        "disagreements": comparison_counts["disagreement"],
        "not_compared": comparison_counts["not_run"],
        "validation_passed": validation_counts["passed"],
        "validation_failed": validation_counts["failed"],
        "validation_skipped": validation_counts["skipped"],
        "selected_samples": sum(sample_counts),
    }
    run_path = output / "run.json"
    run: dict[str, Any] = {}
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        duration_seconds = run.get("duration_seconds")
        if duration_seconds is None:
            duration_seconds = _elapsed_seconds(
                run.get("started_at"),
                generated_at,
            )
        if duration_seconds is not None:
            summary["duration_seconds"] = duration_seconds
    if any(row["state"] != "terminal" for row in public_results):
        validation_status = "running"
    return qualification_report.materialize_report(
        output,
        report_kind="accuracy",
        title="TRTMC Accuracy & Fidelity Qualification",
        identity={
            "run_id": output.name,
            "disposition": validation_status,
            "source_revision": run.get("source_revision"),
        },
        run=run,
        results=public_results,
        metadata={
            "validation_status": validation_status,
            "summary": summary,
        },
    )


def _print_result(
    result: Mapping[str, Any],
    comparison: Path,
    report: Path,
    verbose: bool = False,
) -> None:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        print()
        print("Status: NOT COMPARED")
        print(f"Reason: {not_compared_reason}")
        print(f"Compare result: {comparison}")
        print(f"Report data:   {report.with_name('report.json')}")
        print(f"Report:         {report}")
        return
    execution = result.get("execution", {})
    validation = result.get("validation", {})
    raw_result = result.get("raw_result", {})
    validation_status = (
        str(validation.get("status", "")) if isinstance(validation, Mapping) else ""
    )
    status = {
        "passed": "PASSED",
        "failed": "FAILED",
        "skipped": "SKIPPED",
    }.get(validation_status, validation_status.upper() or "UNKNOWN")
    print()
    print(f"Status: {status}")
    if isinstance(execution, Mapping) and execution.get("status") == "error":
        error = (
            str(raw_result.get("error", ""))
            if isinstance(raw_result, Mapping)
            else ""
        )
        if error:
            print(f"Error: {error}")
        worker_log = str(result.get("worker_log", "") or "")
        if worker_log:
            print(f"Worker log: {worker_log}")
    if not verbose:
        print(f"Compare result: {comparison}")
        print(f"Report data: {report.with_name('report.json')}")
        print(f"Report: {report}")
        return
    reproduce = result.get("reproduce", {})
    hf_commands = reproduce.get("hf", []) if isinstance(reproduce, dict) else []
    trtmc_commands = reproduce.get("trtmc", []) if isinstance(reproduce, dict) else []
    dataset_command, _, _ = _dataset_reproduction(result)
    print()
    print("Reproduce dataset run:")
    print(f"  {dataset_command}" if dataset_command else "  unavailable; see comparison result")
    print()
    print("Reproduce representative HF:")
    if hf_commands:
        for command in hf_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print("Reproduce representative TRTMC:")
    if trtmc_commands:
        for command in trtmc_commands:
            print(f"  {command}")
    else:
        print("  unavailable; see comparison result")
    print()
    print(f"Compare result: {comparison}")
    print(f"Report data:   {report.with_name('report.json')}")
    print(f"Report:         {report}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TRTMC against model reference implementations."
    )
    parser.add_argument("model", nargs="?")
    parser.add_argument("workload", nargs="?")
    parser.add_argument("--all", action="store_true", help="run every ready model")
    parser.add_argument(
        "--model",
        dest="selected_models",
        action="append",
        default=[],
        help="canonical model name; repeatable",
    )
    parser.add_argument(
        "--model-selection",
        type=Path,
        help=(
            "JSON owner/family selection emitted by tools/model_ci.py; "
            "selects matching model profiles"
        ),
    )
    parser.add_argument(
        "--binding",
        dest="selected_bindings",
        action="append",
        default=[],
        metavar="MODEL=WORKLOAD",
        help="exact Accuracy binding; repeatable",
    )
    parser.add_argument(
        "--workload",
        dest="selected_workloads",
        action="append",
        default=[],
        help="Accuracy workload to run for every selected model; repeatable",
    )
    parser.add_argument(
        "--on-model-failure",
        choices=("continue", "stop"),
        default="continue",
        help="continue after a failed model or stop after recording it",
    )
    parser.add_argument(
        "--model-attempts",
        type=_positive_int,
        default=2,
        help="maximum worker attempts for execution errors; comparisons are not retried",
    )
    parser.add_argument(
        "--model-retry-delay-seconds",
        type=_nonnegative_float,
        default=5.0,
        help="delay before retrying a model worker after an execution error",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=_nonnegative_float,
        default=0.0,
        help=(
            "wall-clock limit for each model-worker attempt; terminate its process "
            "group and record an execution error on timeout (0 disables the limit)"
        ),
    )
    parser.add_argument(
        "--reused-bundle-revalidation-limit",
        type=_nonnegative_int,
        default=DEFAULT_REUSED_BUNDLE_REVALIDATION_LIMIT,
        help=(
            "maximum failed reused bundles to confirm with a forced fresh build "
            "across the complete run (default: 1)"
        ),
    )
    parser.add_argument(
        "--reused-bundle-revalidation-attempts-used",
        type=_nonnegative_int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "keep terminal results for exact bindings in an existing output "
            "from the same source revision"
        ),
    )
    parser.add_argument("--list", action="store_true", help="list model-first workloads")
    parser.add_argument(
        "--gate-census",
        action="store_true",
        help="print the resolved, non-blocking gate-policy census as JSON",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suites", type=Path, default=DEFAULT_SUITES)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=DEFAULT_REFERENCE_CACHE,
    )
    parser.add_argument(
        "--reference-source-cache-dir",
        type=Path,
        help=(
            "shared cache for pinned model-owned source repositories; defaults "
            "to --reference-cache-dir"
        ),
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        help="require mutable validation paths to stay below this root",
    )
    parser.add_argument(
        "--model-work-dir",
        type=Path,
        help=(
            "isolate engines per model/workload binding and optionally isolate "
            "the Hugging Face cache per model"
        ),
    )
    parser.add_argument(
        "--engine-retention",
        choices=RETENTION_POLICIES,
        default="retain",
        help="retain binding engines, delete them on pass, or always delete them",
    )
    parser.add_argument(
        "--hf-cache-mode",
        choices=("shared", "per_model"),
        default="shared",
        help="use the inherited Hugging Face cache or isolate it per model",
    )
    parser.add_argument(
        "--hf-cache-retention",
        choices=RETENTION_POLICIES,
        default="retain",
        help="retention for a per-model Hugging Face cache",
    )
    parser.add_argument(
        "--hf-cache-seed-dir",
        type=Path,
        help=(
            "existing HF_HOME tree to hard-link into each empty per-model cache; "
            "the seed is never deleted"
        ),
    )
    parser.add_argument(
        "--prepare-hf-on-demand",
        action="store_true",
        help=(
            "prepare the selected model and its family-owned Hugging Face "
            "dependencies immediately before each model worker"
        ),
    )
    parser.add_argument(
        "--minimum-free-space-gib",
        type=_nonnegative_float,
        default=0.0,
        help="refuse to start the next Accuracy binding below this free space",
    )
    parser.add_argument("--trtmc-binary", type=Path, default=REPO_ROOT / "build" / "trtmc")
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=REPO_ROOT / "build" / "trtmc_dataset_benchmark",
    )
    parser.add_argument("--hf-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--hf-device",
        default="cuda",
        help="device used by the Hugging Face Accuracy reference",
    )
    parser.add_argument(
        "--hf-device-map",
        default="",
        help="optional Hugging Face device map for the Accuracy reference",
    )
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--model-plugin-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "override the workload sample limit; use -1 for the complete dataset; "
            "0 remains accepted for compatibility"
        ),
    )
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print full reproduction commands and detailed worker progress",
    )
    return parser


def _load_validation_inputs(
    arguments: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    tuple[str, ...],
    dict[str, dict[str, Any]],
]:
    catalog = load_catalog(arguments.catalog)
    suites_list = validation_catalog.load_suites(arguments.suites)
    suites = {suite["id"]: suite for suite in suites_list}
    ready = ready_model_names(arguments.models_dir)
    task_models = _validation_models(arguments.models_dir)
    audit_catalog(catalog, ready_models=ready, suite_names=suites)
    audit_workload_compatibility(
        catalog,
        suites=suites,
        task_models=task_models,
    )
    return catalog, suites, ready, task_models


def _select_bindings(
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    ready_models: Iterable[str],
    task_models: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Binding]:
    selection_modes = sum(
        bool(value)
        for value in (
            arguments.all,
            arguments.model,
            arguments.selected_models,
            arguments.model_selection,
            arguments.selected_bindings,
        )
    )
    if selection_modes > 1:
        raise ValidationError(
            "choose exactly one model selection: MODEL, --model, "
            "--model-selection, --binding, or --all"
        )
    if not selection_modes:
        raise ValidationError(
            "provide MODEL [WORKLOAD], --model, --model-selection, --binding, --all, or --list"
        )
    if arguments.selected_bindings and (arguments.workload or arguments.selected_workloads):
        raise ValidationError(
            "--binding cannot be combined with positional WORKLOAD, or --workload"
        )
    if arguments.workload and arguments.selected_workloads:
        raise ValidationError("positional WORKLOAD cannot be combined with --workload")

    if arguments.selected_bindings:
        bindings: list[Binding] = []
        seen: set[tuple[str, str]] = set()
        for raw_binding in arguments.selected_bindings:
            model, separator, workload = raw_binding.partition("=")
            model = model.strip()
            workload = workload.strip()
            if not separator or not model or not workload:
                raise ValidationError(f"invalid --binding {raw_binding!r}; expected MODEL=WORKLOAD")
            identity = (model, workload)
            if identity not in seen:
                bindings.append(resolve_binding(catalog, model, workload))
                seen.add(identity)
        if arguments.dataset and len(bindings) != 1:
            raise ValidationError("--dataset requires exactly one model/workload binding")
        return bindings
    if arguments.all:
        models = tuple(ready_models)
    elif arguments.model:
        models = (arguments.model,)
    elif arguments.model_selection:
        if task_models is None:
            raise ValidationError("task model metadata is required for --model-selection")
        models = model_profiles_for_families(
            task_models,
            ready_models,
            model_selection.load_model_selection(arguments.model_selection),
        )
    else:
        models = model_selection.normalize_models(arguments.selected_models)

    workloads = (arguments.workload,) if arguments.workload else tuple(arguments.selected_workloads)
    bindings = resolve_bindings(
        catalog,
        models,
        workloads=workloads,
    )
    if arguments.dataset and len(bindings) != 1:
        raise ValidationError("--dataset requires exactly one model/workload binding")
    return bindings


def _print_bindings(
    bindings: Iterable[Binding],
    *,
    catalog: Mapping[str, Any],
    explicit_limit: int | None,
) -> None:
    print(
        json.dumps(
            [
                (
                    {
                        "model": binding.model,
                        "workload": binding.workload,
                        "sample_limit": resolve_sample_limit(
                            catalog,
                            binding,
                            explicit_limit,
                        ),
                    }
                    if binding.runnable
                    else {
                        "model": binding.model,
                        "workload": None,
                        "sample_limit": 0,
                        "status": "not_compared",
                        "reason": binding.not_compared_reason,
                    }
                )
                for binding in bindings
            ],
            indent=2,
        )
    )


def _prepare_run_directories(arguments: argparse.Namespace) -> None:
    if arguments.engine_retention != "retain" and arguments.model_work_dir is None:
        raise ValidationError("non-retained engines require --model-work-dir isolation")
    if arguments.hf_cache_mode == "per_model" and arguments.model_work_dir is None:
        raise ValidationError("--hf-cache-mode per_model requires --model-work-dir")
    if arguments.hf_cache_mode == "shared" and arguments.hf_cache_retention != "retain":
        raise ValidationError(
            "a shared Hugging Face cache only supports --hf-cache-retention retain"
        )
    if arguments.hf_cache_seed_dir is not None and arguments.hf_cache_mode != "per_model":
        raise ValidationError("--hf-cache-seed-dir requires --hf-cache-mode per_model")
    if arguments.storage_root is not None:
        storage_root = arguments.storage_root.expanduser().resolve()
        if not storage_root.is_dir():
            raise ValidationError(f"storage root does not exist: {storage_root}")
        arguments.storage_root = storage_root
        for label, path in (
            ("output", arguments.output),
            ("engine directory", arguments.engine_dir),
            ("reference cache directory", arguments.reference_cache_dir),
            ("reference source cache directory", arguments.reference_source_cache_dir),
            ("model work directory", arguments.model_work_dir),
            ("Hugging Face cache seed directory", arguments.hf_cache_seed_dir),
        ):
            if path is None:
                continue
            resolved = path.expanduser().resolve()
            try:
                resolved.relative_to(storage_root)
            except ValueError as exc:
                raise ValidationError(
                    f"{label} must stay below storage root {storage_root}: {resolved}"
                ) from exc
    if arguments.hf_cache_seed_dir is not None:
        seed = arguments.hf_cache_seed_dir.expanduser().resolve()
        if not seed.is_dir():
            raise ValidationError(f"Hugging Face cache seed directory does not exist: {seed}")
        arguments.hf_cache_seed_dir = seed
    if arguments.model_work_dir is not None:
        output = arguments.output.expanduser().resolve()
        model_work = arguments.model_work_dir.expanduser().resolve()
        if (
            output == model_work
            or output.is_relative_to(model_work)
            or model_work.is_relative_to(output)
        ):
            raise ValidationError("output and model work directory must be disjoint")
        arguments.output = output
        arguments.model_work_dir = model_work
        if arguments.hf_cache_seed_dir is not None and (
            arguments.hf_cache_seed_dir == model_work
            or arguments.hf_cache_seed_dir.is_relative_to(model_work)
            or model_work.is_relative_to(arguments.hf_cache_seed_dir)
        ):
            raise ValidationError(
                "Hugging Face cache seed and model work directory must be disjoint"
            )
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.model_work_dir is None:
        arguments.engine_dir.mkdir(parents=True, exist_ok=True)
    else:
        arguments.model_work_dir.mkdir(parents=True, exist_ok=True)
    arguments.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    if arguments.reference_source_cache_dir is not None:
        arguments.reference_source_cache_dir.mkdir(parents=True, exist_ok=True)


def _binding_resource_arguments(
    arguments: argparse.Namespace,
    binding: Binding,
) -> tuple[argparse.Namespace, Path | None, Path | None]:
    selected = copy.copy(arguments)
    if arguments.model_work_dir is None:
        return selected, None, None
    if Path(binding.model).name != binding.model:
        raise ValidationError(f"unsafe model name for model work: {binding.model!r}")
    workload = _required_workload(binding)
    if Path(workload).name != workload:
        raise ValidationError(f"unsafe workload name for binding work: {workload!r}")
    model_work = arguments.model_work_dir / binding.model
    # Dataset preparation can change the resolved build shape/profile. Keep the
    # engine below the exact Accuracy binding; only the HF cache is model-scoped.
    binding_work = model_work / "bindings" / workload
    selected.engine_dir = binding_work / "engines"
    selected.engine_dir.mkdir(parents=True, exist_ok=True)
    selected.hf_cache_dir = None
    if arguments.hf_cache_mode == "per_model":
        selected.hf_cache_dir = model_work / "hf-cache"
        selected.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        if arguments.hf_cache_seed_dir is not None and not any(
            selected.hf_cache_dir.iterdir()
        ):
            seed_source = arguments.hf_cache_seed_dir
            seed_destination = selected.hf_cache_dir
            if any(
                child.name.startswith(("models--", "datasets--", "spaces--"))
                for child in seed_source.iterdir()
            ):
                # Accept the common Hugging Face hub cache layout in addition
                # to a complete HF_HOME tree. HF_HOME expects these entries
                # below its hub/ directory.
                seed_destination = selected.hf_cache_dir / "hub"
            if (
                seed_source.stat().st_dev
                != selected.hf_cache_dir.stat().st_dev
            ):
                raise ValidationError(
                    "Hugging Face cache seed and per-model cache must use the same filesystem"
                )
            try:
                shutil.copytree(
                    seed_source,
                    seed_destination,
                    dirs_exist_ok=True,
                    symlinks=True,
                    copy_function=os.link,
                )
            except OSError as exc:
                shutil.rmtree(selected.hf_cache_dir, ignore_errors=True)
                raise ValidationError(
                    f"could not hard-link Hugging Face cache seed "
                    f"{seed_source}: {exc}"
                ) from exc
    return selected, binding_work, model_work


def _delete_for_retention(policy: str, *, passed: bool) -> bool:
    return policy == "delete_always" or (policy == "delete_on_pass" and passed)


def _cleanup_resource(path: Path, policy: str, *, passed: bool) -> dict[str, str]:
    evidence = {"path": str(path), "policy": policy, "status": "retained"}
    if not _delete_for_retention(policy, passed=passed):
        return evidence
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        evidence["status"] = "already_absent"
    except OSError as exc:
        evidence.update({"status": "failed", "error": str(exc)})
    else:
        evidence["status"] = "deleted"
    return evidence


def _cleanup_binding_engine(
    arguments: argparse.Namespace,
    binding_work: Path | None,
    *,
    passed: bool,
) -> dict[str, str]:
    if binding_work is None:
        return {"policy": "retain", "status": "shared_retained"}
    engine = _cleanup_resource(
        binding_work / "engines",
        arguments.engine_retention,
        passed=passed,
    )
    try:
        binding_work.rmdir()
    except OSError:
        pass
    return engine


def _cleanup_model_hf_cache(
    arguments: argparse.Namespace,
    model_work: Path | None,
    *,
    passed: bool,
    model_complete: bool,
) -> dict[str, str]:
    if model_work is None:
        return {"policy": "retain", "status": "shared_retained"}
    if arguments.hf_cache_mode == "shared":
        if model_complete:
            for directory in (model_work / "bindings", model_work):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        return {"policy": "retain", "status": "shared_retained"}
    if not model_complete:
        return {
            "path": str(model_work / "hf-cache"),
            "policy": arguments.hf_cache_retention,
            "status": "retained_until_model_complete",
        }
    hf_cache = _cleanup_resource(
        model_work / "hf-cache",
        arguments.hf_cache_retention,
        passed=passed,
    )
    for directory in (model_work / "bindings", model_work):
        try:
            directory.rmdir()
        except OSError:
            pass
    return hf_cache


def _worker_environment(arguments: argparse.Namespace) -> dict[str, str]:
    environment = _source_environment()
    hf_cache_dir = getattr(arguments, "hf_cache_dir", None)
    if hf_cache_dir is None:
        return environment
    environment["HF_HOME"] = str(Path(hf_cache_dir).resolve())
    for name in HF_CACHE_ENVIRONMENT_NAMES[1:]:
        environment.pop(name, None)
    return environment


def _accuracy_worker_attempt_evidence(
    binding: Binding,
    arguments: argparse.Namespace,
    *,
    worker_attempt: int,
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    worker_log = _worker_log_path(
        case_dir,
        case_attempt=int(getattr(arguments, "case_attempt", 1)),
        worker_attempt=worker_attempt,
    )
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    worker_log.touch(exist_ok=True)
    worker_command = _worker_command(binding, arguments)
    worker_environment = _worker_environment(arguments)
    return {
        "commands": {
            "worker": {
                "argv": worker_command,
                "rendered": shlex.join(worker_command),
                "cwd": str(REPO_ROOT),
            }
        },
        "logs": [
            {
                "label": (
                    f"Accuracy worker case attempt "
                    f"{getattr(arguments, 'case_attempt', 1)}, "
                    f"worker attempt {worker_attempt}"
                ),
                "href": worker_log.relative_to(arguments.output).as_posix(),
            }
        ],
        "environment": {
            name: worker_environment[name]
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "LD_LIBRARY_PATH",
                "PYTHONPATH",
                "PYTORCH_CUDA_ALLOC_CONF",
                "TRTMC_REFERENCE_PYTORCH_CUDA_ALLOC_CONF",
            )
            if worker_environment.get(name)
        },
    }


def _prepare_hf_on_demand(
    binding: Binding,
    arguments: argparse.Namespace,
) -> None:
    if not arguments.prepare_hf_on_demand:
        return
    case_dir = _case_directory(arguments.output, binding)
    case_dir.mkdir(parents=True, exist_ok=True)
    selected_models = case_dir / "hf_prepare.models.txt"
    selected_models.write_text(f"{binding.model}\n", encoding="utf-8")
    log_path = case_dir / "hf_prepare.log"
    command = [
        str(arguments.hf_python),
        str(HF_WARM_SCRIPT),
        "--models-file",
        str(selected_models),
        "--strict",
        "--fail-fast",
    ]
    if arguments.local_files_only:
        command.append("--local-only")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            env=_worker_environment(arguments),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise ValidationError(
            f"Hugging Face preparation failed for {binding.model} "
            f"with rc={completed.returncode}; see {log_path}"
        )


def _check_free_space(arguments: argparse.Namespace, binding: Binding) -> float:
    root = arguments.storage_root or arguments.model_work_dir or arguments.output
    free_gib = shutil.disk_usage(root).free / 1024**3
    if free_gib < arguments.minimum_free_space_gib:
        raise ValidationError(
            f"refusing to start {binding.model}/{binding.workload}: "
            f"only {free_gib:.1f} GiB free; "
            f"minimum is {arguments.minimum_free_space_gib:.1f} GiB"
        )
    return free_gib


def _worker_command(
    binding: Binding,
    arguments: argparse.Namespace,
) -> list[str]:
    workload = _required_workload(binding)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        binding.model,
        workload,
        "--model-worker",
    ]
    for option, value in (
        ("--catalog", arguments.catalog),
        ("--suites", arguments.suites),
        ("--models-dir", arguments.models_dir),
        ("--output", arguments.output),
        ("--engine-dir", arguments.engine_dir),
        ("--reference-cache-dir", arguments.reference_cache_dir),
        ("--trtmc-binary", arguments.trtmc_binary),
        ("--benchmark-binary", arguments.benchmark_binary),
        ("--hf-python", arguments.hf_python),
        ("--hf-device", arguments.hf_device),
        ("--model-attempts", arguments.model_attempts),
        ("--model-retry-delay-seconds", arguments.model_retry_delay_seconds),
        (
            "--reused-bundle-revalidation-limit",
            arguments.reused_bundle_revalidation_limit,
        ),
        (
            "--reused-bundle-revalidation-attempts-used",
            arguments.reused_bundle_revalidation_attempts_used,
        ),
    ):
        command.extend([option, str(value)])
    for option, value in (
        ("--reference-source-cache-dir", arguments.reference_source_cache_dir),
        ("--hf-device-map", arguments.hf_device_map),
        ("--dataset-root", arguments.dataset_root),
        ("--backend-dir", arguments.backend_dir),
        ("--model-plugin-dir", arguments.model_plugin_dir),
        ("--cuda-visible-devices", arguments.cuda_visible_devices),
    ):
        if value:
            command.extend([option, str(value)])
    if arguments.limit is not None:
        command.extend(["--limit", str(arguments.limit)])
    for option, enabled in (
        ("--force-hf", arguments.force_hf),
        ("--force-build", arguments.force_build),
        ("--no-build", arguments.no_build),
        ("--local-files-only", arguments.local_files_only),
        ("--prepare-hf-on-demand", arguments.prepare_hf_on_demand),
    ):
        if enabled:
            command.append(option)
    return command


def _worker_error_result(
    binding: Binding,
    *,
    command: Sequence[str],
    returncode: int,
    worker_log: Path,
    sample_limit: int,
    error: str,
    error_type: str = "WorkerProcessError",
) -> dict[str, Any]:
    return _normalize_result(
        {
            "schema_version": "trtmc.validation-result/v2",
            "model": binding.model,
            "workload": binding.workload,
            "executor": "model_worker",
            "status": "failed",
            "returncode": returncode,
            "reference_environment": [],
            "reproduce": {
                "dataset": {
                    "command": shlex.join(command),
                    "sample_limit": sample_limit,
                    "prepared_input_count": 0,
                },
                "hf": [],
                "trtmc": [],
            },
            "raw_result": {
                "status": "failed",
                "error_type": error_type,
                "error": error,
            },
            "raw_result_path": "",
            "disagreements": {
                "count": 0,
                "path": "disagreements.jsonl",
                "inline_limit": trtmc_disagreements.INLINE_DISAGREEMENT_LIMIT,
                "reference_vanilla_available": False,
                "trtmc_vanilla_available": False,
            },
            "execution_log": str(worker_log),
            "worker_log": str(worker_log),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


_WORKER_EXCEPTION_LINE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|[^:]+: error):\s+.+$"
)


def _worker_log_error(worker_log: Path) -> str:
    try:
        lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines[-200:]):
        rendered = line.strip()
        if _WORKER_EXCEPTION_LINE.fullmatch(rendered):
            return rendered
    return ""


def _run_supervised_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    attempt: int = 1,
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    case_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = case_dir / "comparison.json"
    comparison_path.unlink(missing_ok=True)
    worker_log = _worker_log_path(
        case_dir,
        case_attempt=int(getattr(arguments, "case_attempt", 1)),
        worker_attempt=attempt,
    )
    command = _worker_command(binding, arguments)
    launch_error = ""
    error_type = "WorkerProcessError"
    try:
        returncode = _run_supervised_subprocess(
            command,
            worker_log,
            _worker_environment(arguments),
            arguments.model_timeout_seconds,
        )
    except WorkerTimeoutError as exc:
        returncode = 124
        launch_error = str(exc)
        error_type = "WorkerTimeoutError"
    except OSError as exc:
        returncode = 127
        launch_error = f"could not start model worker: {exc}"
    try:
        if launch_error:
            raise ValidationError(launch_error)
        if not comparison_path.is_file():
            raise ValidationError(
                f"worker exited with code {returncode} without writing comparison.json"
            )
        loaded = json.loads(comparison_path.read_text(encoding="utf-8"))
        result = _normalize_result(loaded)
        if result.get("model") != binding.model or result.get("workload") != binding.workload:
            raise ValidationError("worker wrote comparison.json for a different binding")
    except (OSError, ValueError, ValidationError) as exc:
        worker_error = _worker_log_error(worker_log)
        result = _worker_error_result(
            binding,
            command=command,
            returncode=returncode,
            worker_log=worker_log,
            sample_limit=resolve_sample_limit(catalog, binding, arguments.limit),
            error=worker_error or str(exc),
            error_type=error_type,
        )
        (case_dir / "disagreements.jsonl").write_text("", encoding="utf-8")
    else:
        result["worker_log"] = str(worker_log)
        dataset = result.get("reproduce", {}).get("dataset", {})
        if isinstance(dataset, dict):
            dataset["command"] = shlex.join(token for token in command if token != "--model-worker")
    comparison_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _worker_log_path(
    case_dir: Path,
    *,
    case_attempt: int,
    worker_attempt: int,
) -> Path:
    if case_attempt == 1:
        name = "worker.log" if worker_attempt == 1 else f"worker.attempt-{worker_attempt}.log"
    else:
        suffix = "" if worker_attempt == 1 else f".worker-attempt-{worker_attempt}"
        name = f"worker.case-attempt-{case_attempt}{suffix}.log"
    return case_dir / name


def _archive_failed_attempt(case_dir: Path, attempt: int) -> dict[str, str]:
    archived = {}
    for name in ("comparison.json", "execution.log", "disagreements.jsonl"):
        source = case_dir / name
        if not source.is_file():
            continue
        path = case_dir / f"{source.stem}.attempt-{attempt}{source.suffix}"
        shutil.copy2(source, path)
        archived[name] = str(path)
    return archived


def _attempt_record(
    result: Mapping[str, Any],
    *,
    attempt: int,
    archived: Mapping[str, str],
) -> dict[str, Any]:
    execution = result.get("execution", {})
    validation = result.get("validation", {})
    raw_result = result.get("raw_result", {})
    execution_log = archived.get(
        "execution.log",
        str(result.get("execution_log", "") or ""),
    )
    return {
        "attempt": attempt,
        "execution_status": (
            str(execution.get("status", "")) if isinstance(execution, Mapping) else ""
        ),
        "validation_status": (
            str(validation.get("status", "")) if isinstance(validation, Mapping) else ""
        ),
        "worker_log": str(result.get("worker_log", "") or ""),
        "execution_log": execution_log,
        "comparison_result": archived.get("comparison.json", ""),
        "error_type": (
            str(raw_result.get("error_type", "")) if isinstance(raw_result, Mapping) else ""
        ),
        "error": (str(raw_result.get("error", "")) if isinstance(raw_result, Mapping) else ""),
    }


def _run_supervised_binding_with_retries(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    on_retry: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    revalidation_budget = _reused_bundle_revalidation_budget(arguments)
    attempts = []
    result: dict[str, Any] = {}
    for attempt in range(1, arguments.model_attempts + 1):
        attempt_arguments = copy.copy(arguments)
        attempt_arguments.reused_bundle_revalidation_attempts_used = (
            revalidation_budget.attempts_used
        )
        result = _run_supervised_binding(
            binding,
            arguments=attempt_arguments,
            catalog=catalog,
            attempt=attempt,
        )
        revalidation_budget.record_worker_result(result)
        execution = result.get("execution", {})
        execution_error = isinstance(execution, Mapping) and execution.get("status") == "error"
        retryable = not (
            isinstance(execution, Mapping) and execution.get("retryable") is False
        )
        archived = (
            _archive_failed_attempt(case_dir, attempt)
            if execution_error and retryable and attempt < arguments.model_attempts
            else {}
        )
        attempt_result = _attempt_record(
            result,
            attempt=attempt,
            archived=archived,
        )
        attempts.append(attempt_result)
        will_retry = execution_error and retryable and attempt < arguments.model_attempts
        if not will_retry:
            break
        if on_retry is not None:
            on_retry(attempt, result)
        print(
            f"  Attempt {attempt}/{arguments.model_attempts}: FAILED",
            flush=True,
        )
        if attempt_result["error"]:
            print(f"  Error: {attempt_result['error']}", flush=True)
        if attempt_result["worker_log"]:
            print(f"  Worker log: {attempt_result['worker_log']}", flush=True)
        print(
            f"  Retrying {binding.model} "
            f"(attempt {attempt + 1}/{arguments.model_attempts})",
            flush=True,
        )
        if arguments.model_retry_delay_seconds:
            time.sleep(arguments.model_retry_delay_seconds)
    execution = dict(result.get("execution", {}))
    execution.update(
        {
            "attempt_count": len(attempts),
            "max_attempts": arguments.model_attempts,
            "retry_count": max(0, len(attempts) - 1),
            "attempts": attempts,
        }
    )
    result["execution"] = execution
    comparison_path = case_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _resumable_binding_result(
    output: Path,
    binding: Binding,
) -> dict[str, Any] | None:
    comparison = _case_directory(output, binding) / "comparison.json"
    try:
        loaded = json.loads(comparison.read_text(encoding="utf-8"))
        result = _normalize_result(loaded)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if result.get("model") != binding.model or result.get("workload") != binding.workload:
        return None
    execution = result.get("execution")
    validation = result.get("validation")
    if not isinstance(execution, Mapping) or execution.get("status") not in {
        "completed",
        "error",
    }:
        return None
    if not isinstance(validation, Mapping) or validation.get("status") not in {
        "passed",
        "failed",
        "skipped",
    }:
        return None
    return result


def _resume_command(command: str) -> list[str]:
    try:
        arguments = shlex.split(command)
    except ValueError as exc:
        raise ValidationError(f"cannot parse recorded Accuracy command: {exc}") from exc
    presentation_only = {"--resume-existing", "--verbose"}
    return [argument for argument in arguments if argument not in presentation_only]


def _validate_resume_request(output: Path) -> None:
    run_path = output / "run.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"cannot resume without run metadata: {run_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read resume metadata {run_path}: {exc}") from exc
    if not isinstance(run, Mapping):
        raise ValidationError(f"resume metadata must contain a JSON object: {run_path}")
    previous = str(run.get("source_revision", "") or "")
    current = _source_revision()
    if not previous or previous != current:
        raise ValidationError(
            "cannot resume Accuracy results from a different source revision: "
            f"recorded={previous or '<missing>'}, current={current or '<missing>'}"
        )
    recorded_command = str(run.get("command", "") or "")
    current_command = shlex.join(sys.argv)
    if not recorded_command or _resume_command(recorded_command) != _resume_command(
        current_command
    ):
        raise ValidationError("cannot resume Accuracy results with a different resolved command")


def _accuracy_case_id(binding: Binding) -> str:
    return f"{binding.model}::{binding.workload or 'not-compared'}"


def _open_accuracy_ledger(
    bindings: Sequence[Binding],
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> ExecutionLedger:
    fingerprint_input = {
        "source_revision": _source_revision(),
        "command": _resume_command(shlex.join(sys.argv)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cases = []
    for binding in bindings:
        result_path = _case_directory(arguments.output, binding) / "comparison.json"
        sample_limit = (
            resolve_sample_limit(catalog, binding, arguments.limit)
            if binding.runnable
            else None
        )
        cases.append(
            {
                "id": _accuracy_case_id(binding),
                "result_path": result_path.relative_to(arguments.output).as_posix(),
                "report": {
                    "model": binding.model,
                    "workload": binding.workload,
                    "sample_limit": sample_limit,
                },
            }
        )
    try:
        return ExecutionLedger.open(
            arguments.output,
            campaign_id=arguments.output.name,
            task_kind="accuracy",
            fingerprint=fingerprint,
            cases=cases,
        )
    except ExecutionLedgerError as error:
        raise ValidationError(str(error)) from error


def _run_all_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> int:
    bindings = tuple(bindings)
    _prepare_run_directories(arguments)
    _reused_bundle_revalidation_budget(arguments)
    if arguments.resume_existing:
        _validate_resume_request(arguments.output)
    write_run_metadata(
        arguments.output,
        cuda_visible_devices=arguments.cuda_visible_devices,
    )
    ledger = _open_accuracy_ledger(bindings, arguments, catalog)
    if arguments.resume_existing:
        ledger.recover_interrupted()
    write_report(arguments.output)
    failed = False
    not_compared = False
    remaining = Counter(binding.model for binding in bindings if binding.runnable)
    model_passed = {model: True for model in remaining}
    for binding in bindings:
        case_id = _accuracy_case_id(binding)
        receipt = ledger.receipt(case_id)
        if not binding.runnable:
            if receipt["state"] == "terminal":
                result = _normalize_result(receipt["payload"])
                comparison = _case_directory(arguments.output, binding) / "comparison.json"
            else:
                ledger.begin(case_id, stage="preflight")
                write_report(arguments.output)
                print(
                    f"\nNot compared: {binding.model} / {binding.not_compared_reason}",
                    flush=True,
                )
                result, comparison = _write_not_compared_case(
                    binding,
                    arguments.output,
                )
                normalized = _normalize_result(result)
                ledger.finish(
                    case_id,
                    result=_traffic_light_status(normalized),
                    payload=normalized,
                )
            not_compared = True
            _, report_path, _ = write_report(arguments.output)
            _print_result(result, comparison, report_path, arguments.verbose)
            continue
        sample_limit = resolve_sample_limit(catalog, binding, arguments.limit)
        sample_note = "full dataset" if sample_limit == 0 else f"{sample_limit} samples"
        binding_arguments, binding_work, model_work = _binding_resource_arguments(
            arguments,
            binding,
        )
        if receipt["state"] == "terminal":
            result = _normalize_result(receipt["payload"])
            model_failed = result["validation"]["status"] == "failed"
            model_passed[binding.model] = model_passed[binding.model] and not model_failed
            remaining[binding.model] -= 1
            print(
                f"\nResume: keeping terminal result for {binding.model} / {binding.workload}",
                flush=True,
            )
            _, report_path, _ = write_report(arguments.output)
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            _print_result(result, comparison, report_path, arguments.verbose)
            failed = failed or model_failed
            if model_failed and arguments.on_model_failure == "stop":
                print(f"Stopping after failed model: {binding.model}", flush=True)
                break
            continue
        case_attempt = len(receipt["attempts"]) + 1
        binding_arguments.case_attempt = case_attempt
        ledger.begin(
            case_id,
            stage="preflight",
            evidence=_accuracy_worker_attempt_evidence(
                binding,
                binding_arguments,
                worker_attempt=1,
            ),
        )
        write_report(arguments.output)
        result = (
            _resumable_binding_result(arguments.output, binding)
            if arguments.resume_existing
            else None
        )
        if result is None:
            free_gib = _check_free_space(arguments, binding)
            ledger.update_stage(case_id, "compare")
            write_report(arguments.output)
            print(
                f"\nStarting worker: {binding.model} / {binding.workload} / "
                f"{sample_note} / {free_gib:.1f} GiB free",
                flush=True,
            )

            def record_retry(
                worker_attempt: int,
                attempt_result: Mapping[str, Any],
            ) -> None:
                stage = _accuracy_result_stage(attempt_result)
                execution = attempt_result.get("execution", {})
                execution = execution if isinstance(execution, Mapping) else {}
                raw_result = attempt_result.get("raw_result", {})
                raw_result = raw_result if isinstance(raw_result, Mapping) else {}
                timed_out = raw_result.get("error_type") == "WorkerTimeoutError"
                ledger.update_stage(case_id, stage)
                ledger.retry(
                    case_id,
                    attempt_outcome="timed_out" if timed_out else "failed",
                    evidence={
                        "return_code": execution.get("exit_code"),
                        "retryable": True,
                    },
                )
                ledger.begin(
                    case_id,
                    stage="compare",
                    evidence=_accuracy_worker_attempt_evidence(
                        binding,
                        binding_arguments,
                        worker_attempt=worker_attempt + 1,
                    ),
                )
                write_report(arguments.output)

            result = _run_supervised_binding_with_retries(
                binding,
                arguments=binding_arguments,
                catalog=catalog,
                on_retry=record_retry,
            )
        else:
            print(
                f"\nResume: importing terminal result for {binding.model} / {binding.workload}",
                flush=True,
            )
        model_failed = result["validation"]["status"] == "failed"
        model_passed[binding.model] = model_passed[binding.model] and not model_failed
        remaining[binding.model] -= 1
        stop_model = model_failed and arguments.on_model_failure == "stop"
        model_complete = remaining[binding.model] == 0 or stop_model
        cleanup = {
            "scope": "binding",
            "passed": not model_failed,
            "engine": _cleanup_binding_engine(
                arguments,
                binding_work,
                passed=not model_failed,
            ),
            "hf_cache": _cleanup_model_hf_cache(
                arguments,
                model_work,
                passed=(model_passed[binding.model] and remaining[binding.model] == 0),
                model_complete=model_complete,
            ),
        }
        result["resource_cleanup"] = cleanup
        cleanup_failed = any(
            resource.get("status") == "failed"
            for resource in (cleanup["engine"], cleanup["hf_cache"])
        )
        failed = failed or cleanup_failed
        comparison = _case_directory(arguments.output, binding) / "comparison.json"
        comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ledger.update_stage(case_id, _accuracy_result_stage(result))
        normalized = _normalize_result(result)
        light = _traffic_light_status(normalized)
        raw_result = normalized.get("raw_result", {})
        timed_out = (
            isinstance(raw_result, Mapping)
            and raw_result.get("error_type") == "WorkerTimeoutError"
        )
        ledger.finish(
            case_id,
            result=light,
            payload=normalized,
            attempt_outcome=(
                "timed_out"
                if timed_out
                else "failed"
                if normalized["execution"]["status"] == "error"
                else "completed"
            ),
            evidence={
                "return_code": normalized["execution"].get("exit_code"),
                "retryable": normalized["execution"].get("retryable"),
            },
        )
        _, report_path, _ = write_report(arguments.output)
        comparison = _case_directory(arguments.output, binding) / "comparison.json"
        _print_result(result, comparison, report_path, arguments.verbose)
        failed = failed or model_failed
        if stop_model:
            print(
                f"Stopping after failed model: {binding.model}",
                flush=True,
            )
            break
    finalize_run_metadata(arguments.output)
    write_report(arguments.output)
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _run_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> int:
    _prepare_run_directories(arguments)
    revalidation_budget = _reused_bundle_revalidation_budget(arguments)
    if not arguments.model_worker:
        write_run_metadata(
            arguments.output,
            cuda_visible_devices=arguments.cuda_visible_devices,
        )
    failed = False
    not_compared = False
    for binding in bindings:
        if not binding.runnable:
            print(
                f"\nNot compared: {binding.model} / {binding.not_compared_reason}",
                flush=True,
            )
            result, comparison = _write_not_compared_case(
                binding,
                arguments.output,
            )
            not_compared = True
            if not arguments.model_worker:
                _, report_path, _ = write_report(arguments.output)
                _print_result(result, comparison, report_path, arguments.verbose)
            continue
        binding_arguments = copy.copy(arguments)
        binding_arguments._reused_bundle_revalidation_budget = (
            revalidation_budget
        )
        binding_arguments.limit = resolve_sample_limit(
            catalog,
            binding,
            arguments.limit,
        )
        sample_note = (
            "full dataset" if binding_arguments.limit == 0 else f"{binding_arguments.limit} samples"
        )
        print(
            f"\n{binding.model} / {binding.workload} / {sample_note}",
            flush=True,
        )
        _prepare_hf_on_demand(binding, binding_arguments)
        result = run_binding(
            binding,
            arguments=binding_arguments,
            task_models=task_models,
            suites=suites,
        )
        if not arguments.model_worker:
            _, report_path, _ = write_report(arguments.output)
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            _print_result(result, comparison, report_path, arguments.verbose)
        failed = failed or result["validation"]["status"] == "failed"
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _main(arguments: argparse.Namespace) -> int:
    catalog, suites, ready, task_models = _load_validation_inputs(arguments)
    if arguments.gate_census:
        if any(
            (
                arguments.list,
                arguments.all,
                arguments.model,
                arguments.workload,
                arguments.selected_models,
                arguments.model_selection,
                arguments.selected_bindings,
                arguments.selected_workloads,
            )
        ):
            raise ValidationError(
                "--gate-census is a global inventory and cannot select models or workloads"
            )
        print(
            json.dumps(
                build_gate_census(
                    catalog=catalog,
                    suites=suites,
                    task_models=task_models,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.list:
        for name, spec in catalog["models"].items():
            not_compared_reason = str(spec.get("not_compared_reason", "") or "")
            if not_compared_reason:
                print(f"{name}: not compared ({not_compared_reason})")
                continue
            workloads = []
            for workload in spec["workloads"]:
                limit = catalog["sample_limits"][workload]
                limit_label = "all samples" if limit == -1 else f"{limit} samples"
                workloads.append(f"{workload} ({limit_label})")
            print(f"{name}: {', '.join(workloads)}")
        return 0
    bindings = _select_bindings(arguments, catalog, ready, task_models)
    audit_binding_compatibility(
        bindings,
        suites=suites,
        task_models=task_models,
    )
    if arguments.dry_run:
        _print_bindings(
            bindings,
            catalog=catalog,
            explicit_limit=arguments.limit,
        )
        return 0
    if not arguments.model_worker:
        build_identity = _validate_build_identity(arguments)
        _write_build_identity(arguments.output, build_identity)
        return _run_all_bindings(
            bindings,
            arguments=arguments,
            catalog=catalog,
        )
    return _run_bindings(
        bindings,
        arguments=arguments,
        catalog=catalog,
        task_models=task_models,
        suites=suites,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return _main(parser.parse_args(argv))
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

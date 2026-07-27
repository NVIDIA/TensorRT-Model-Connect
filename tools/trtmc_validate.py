#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run model-first TRTMC reference validation for Dev and QA."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

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
from tools import task_eval, trtmc_disagreements  # noqa: E402


DEFAULT_CATALOG = REPO_ROOT / "tests" / "validation" / "model_workloads.yaml"
DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "trtmc-validate"
DEFAULT_ENGINE_DIR = DEFAULT_OUTPUT / "engines"
DEFAULT_REFERENCE_CACHE = DEFAULT_OUTPUT / "references"
COMMON_REFERENCE_PROFILE = "reference_common"
NOT_COMPARED_DIRECTORY = "not-compared"
LEGACY_E2E_REASON = (
    "E2E execution does not compare aligned reference and TRTMC outputs."
)


class ValidationError(RuntimeError):
    """The requested validation cannot be resolved or executed."""


@dataclass(frozen=True)
class Binding:
    model: str
    workload: str | None
    not_compared_reason: str = ""

    @property
    def runnable(self) -> bool:
        return self.workload is not None


def _required_workload(binding: Binding) -> str:
    if binding.workload is None:
        raise ValidationError(
            f"model {binding.model} has no reference-consistency workload"
        )
    return binding.workload


def _case_directory(output: Path, binding: Binding) -> Path:
    return output / binding.model / (
        binding.workload if binding.workload is not None else NOT_COMPARED_DIRECTORY
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
    not_compared_reason = spec.get("not_compared_reason")
    if not_compared_reason is not None:
        if not isinstance(not_compared_reason, str) or not not_compared_reason.strip():
            raise ValidationError(
                f"{path}: {name}.not_compared_reason must be a non-empty string"
            )
        if "default" in spec or "workloads" in spec:
            raise ValidationError(
                f"{path}: {name} cannot declare workloads while marked not compared"
            )
        return
    workloads = spec.get("workloads")
    default = spec.get("default")
    valid_workloads = (
        isinstance(workloads, list)
        and bool(workloads)
        and all(isinstance(item, str) and item for item in workloads)
    )
    if not valid_workloads:
        raise ValidationError(f"{path}: {name}.workloads must contain names")
    if "e2e" in workloads:
        raise ValidationError(
            f"{path}: {name}.workloads cannot use e2e; reference consistency "
            "requires aligned reference and TRTMC outputs"
        )
    if default not in workloads:
        raise ValidationError(f"{path}: {name}.default must be one of {name}.workloads")


def _validate_sample_limits(path: Path, raw: Mapping[str, Any]) -> None:
    sample_limits = raw.get("sample_limits")
    if not isinstance(sample_limits, dict) or not sample_limits:
        raise ValidationError(f"{path}: sample_limits must be a non-empty mapping")
    for workload, limit in sample_limits.items():
        if not isinstance(workload, str) or not workload:
            raise ValidationError(f"{path}: invalid sample-limit workload {workload!r}")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValidationError(
                f"{path}: sample_limits.{workload} must be a positive integer"
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


def ready_model_names(models_root: Path = DEFAULT_MODELS) -> tuple[str, ...]:
    models = task_eval.load_manifest_records(models_root)
    return tuple(
        sorted(
            str(model["name"])
            for model in models
            if not model["requires_multi_device"]
            and not model.get("skip")
            and model.get("ci_tier") != "l0_only"
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
            for workload in spec.get("workloads", [])
            if workload not in known_workloads
        }
    )
    if unknown:
        raise ValidationError(f"unknown workloads: {', '.join(unknown)}")

    declared_sampled = {
        workload
        for spec in models.values()
        for workload in spec.get("workloads", [])
    }
    configured_sampled = set(catalog["sample_limits"])
    missing_limits = sorted(declared_sampled - configured_sampled)
    stale_limits = sorted(configured_sampled - declared_sampled)
    if missing_limits or stale_limits:
        details = []
        if missing_limits:
            details.append(f"missing sample limits: {', '.join(missing_limits)}")
        if stale_limits:
            details.append(f"unused sample limits: {', '.join(stale_limits)}")
        raise ValidationError("; ".join(details))


def audit_workload_compatibility(
    catalog: Mapping[str, Any],
    *,
    suites: Mapping[str, dict[str, Any]],
    task_models: Mapping[str, dict[str, Any]],
) -> None:
    incompatible = []
    for model_name, spec in catalog["models"].items():
        for workload in spec.get("workloads", []):
            matched, reason = task_eval.suite_match_reason(
                suites[workload],
                task_models[model_name],
            )
            if not matched:
                incompatible.append(f"{model_name}/{workload}: {reason}")
    if incompatible:
        raise ValidationError("incompatible model/workload bindings: " + "; ".join(incompatible))


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
                f"model {model} has no reference-consistency workloads: "
                f"{not_compared_reason}"
            )
        return Binding(
            model=model,
            workload=None,
            not_compared_reason=not_compared_reason,
        )
    selected = workload or spec["default"]
    if selected not in spec["workloads"]:
        available = ", ".join(spec["workloads"])
        raise ValidationError(
            f"model {model} does not declare workload {selected}; available: {available}"
        )
    return Binding(model=model, workload=selected)


def resolve_sample_limit(
    catalog: Mapping[str, Any],
    binding: Binding,
    explicit_limit: int | None,
) -> int:
    if explicit_limit is not None and explicit_limit < 0:
        raise ValidationError("--limit must be zero or greater")
    if not binding.runnable:
        return 0
    if explicit_limit is not None:
        return explicit_limit
    assert binding.workload is not None
    return int(catalog["sample_limits"][binding.workload])


def _task_eval_models(models_root: Path) -> dict[str, dict[str, Any]]:
    return {str(model["name"]): model for model in task_eval.load_manifest_records(models_root)}


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
) -> tuple[str, ...]:
    if not binding.runnable:
        raise ValidationError(
            f"model {binding.model} has no reference-consistency workload"
        )
    model = task_models[binding.model]
    profile = _declared_profile(
        family=str(model.get("family", "") or ""),
        runtime_strategy=str(model.get("runtime_strategy", "") or ""),
        reference_backend=str(model.get("reference_backend", "") or ""),
        execution_profiles=model.get("execution_profiles"),
    )
    return (
        (COMMON_REFERENCE_PROFILE,)
        if profile == COMMON_REFERENCE_PROFILE
        else (COMMON_REFERENCE_PROFILE, profile)
    )


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
                    f"Pinned {source.name} checkout is missing "
                    f"{source.entrypoint}"
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
) -> ReferenceSourceSelection:
    environment = {"TRTMC_STORAGE_ROOT": str(cache_root)}
    if family == "elf_flow":
        checkout = _ensure_reference_source(ELF_SOURCE, cache_root)
        return ReferenceSourceSelection(
            environment=environment,
            elf_reference_repo=checkout,
        )
    if family == "sana_wm":
        checkout = _ensure_reference_source(SANA_WM_SOURCE, cache_root)
        environment["SANA_WM_SCRIPT"] = str(
            checkout / SANA_WM_SOURCE.entrypoint
        )
    return ReferenceSourceSelection(environment=environment)


def _dataset_path(suite: Mapping[str, Any], dataset_root: Path | None) -> Path:
    raw = str(suite.get("dataset", {}).get("default_path", "") or "")
    if not raw:
        raise ValidationError(f"workload {suite.get('id')} has no default dataset path")
    path = Path(raw)
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
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(env),
        )
    return completed.returncode


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
    if reference_sources and reference_sources.elf_reference_repo:
        command.extend(
            [
                "--elf-reference-repo",
                str(reference_sources.elf_reference_repo),
            ]
        )
    return command


MAX_REPRO_COMMANDS_PER_BACKEND = 3
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


def _command_log_kind(path: Path, *, has_native_reference: bool) -> str | None:
    if has_native_reference and path.name == "hf_run.log":
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
    has_native_reference = any(path.name == "hf_native_run.log" for path in log_paths)
    for path in log_paths:
        kind = _command_log_kind(path, has_native_reference=has_native_reference)
        if kind is None:
            continue
        indexed_sample_ids = sample_ids if path.name == "trtfb_run.log" else ()
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
    log_paths = sorted(root.rglob("*.log"))
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
    "diffusion_text_parity": "token_agreement_rate",
    "encoder_embedding_parity": "vector_pass_rate",
    "image_classification_parity": "top1_agreement",
    "ocrbench_v2": "prediction_agreement_rate",
    "reranking_parity": "mean_pairwise_ordering_agreement",
    "semantic_segmentation_parity": "backend_pixel_agreement",
    "time_series_parity": "sample_agreement_rate",
}
_COMPARISON_METRICS = (
    *_PRIMARY_COMPARISON_METRICS,
    "token_id_prefix_agreement",
    "normalized_transcript_exact_agreement_rate",
    "correctness_agreement_rate",
    "divergence_rate",
    "divergent_count",
    "hf_accuracy",
    "trtfb_accuracy",
    "accuracy_delta_trtfb_minus_hf",
    "accuracy_drop_from_hf",
    "hf_top1_accuracy",
    "trtfb_top1_accuracy",
    "top1_accuracy_drop_from_hf",
    "hf_mean_iou",
    "trtfb_mean_iou",
    "backend_mean_iou",
    "mean_iou_drop_from_hf",
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
    return {
        name: raw_result[name]
        for name in _COMPARISON_METRICS
        if raw_result.get(name) is not None
    }


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
    all_commands = {
        kind: _string_list(reproduce.get(kind, []))
        for kind in ("hf", "trtmc")
    }
    commands = {
        kind: values[:MAX_REPRO_COMMANDS_PER_BACKEND]
        for kind, values in all_commands.items()
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
        "command_logs": {
            kind: _normalized_command_logs(reproduce, kind) for kind in commands
        },
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
    normalized.update(
        {
            "schema_version": "trtmc.validation-result/v2",
            "execution": execution,
            "comparison": comparison,
            "validation": validation,
            "reproduce": _normalize_reproduction(normalized.get("reproduce")),
        }
    )
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
    return _normalize_result({
        "schema_version": "trtmc.validation-result/v2",
        "model": binding.model,
        "workload": binding.workload,
        "executor": "trtmc_compare",
        "status": status,
        "returncode": returncode,
        "reference_environment": [
            {"name": name, "python": path} for name, path in reference_environment.names_and_paths
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
    })


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
    profiles = _binding_profiles(
        binding,
        task_models=task_models,
    )
    environment = ensure_environments(profiles, str(arguments.hf_python))
    reference_sources = ensure_reference_sources(
        str(task_models[binding.model].get("family", "") or ""),
        Path(arguments.reference_cache_dir),
    )
    process_env = _source_environment()
    process_env.update(environment.overrides)
    process_env.update(reference_sources.environment)
    dataset_command = shlex.join([sys.executable, *sys.argv])

    suite = suites[workload]
    dataset = (
        Path(arguments.dataset)
        if arguments.dataset
        else _dataset_path(suite, arguments.dataset_root)
    )
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
    )

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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def write_run_metadata(output: Path) -> Path:
    metadata = {
        "schema_version": "trtmc.validation-run/v1",
        "source_revision": _source_revision(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": shlex.join(sys.argv),
        "started_at": _utc_now().isoformat(),
    }
    path = output / "run.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _report_provenance(run: Mapping[str, Any]) -> str:
    fields = (
        ("source", run.get("source_revision")),
        ("host", run.get("hostname")),
        ("CUDA_VISIBLE_DEVICES", run.get("cuda_visible_devices")),
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
        return max(int(counts.get(kind)), len(commands)) if isinstance(counts, dict) else len(commands)
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


def _representative_note(result: Mapping[str, Any]) -> str:
    reproduce = result.get("reproduce", {})
    representative = (
        reproduce.get("representative", {}) if isinstance(reproduce, dict) else {}
    )
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
        rendered.append(
            f'<figure><figcaption>{label}</figcaption>{body}</figure>'
        )
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
    failed = (
        isinstance(comparison, dict)
        and comparison.get("status") == "disagreement"
    )
    noun = "failed samples" if failed else "sample differences"
    asset_base = Path(artifact_href).parent
    records = "".join(
        _render_disagreement_record(record, asset_base=asset_base)
        for record in preview
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
        return (
            '<span class="unavailable">'
            f"{html.escape(not_compared_reason)}"
            "</span>"
        )
    reference_commands = _result_commands(result, "hf")
    trtmc_commands = _result_commands(result, "trtmc")
    dataset_command, sample_limit, _ = _dataset_reproduction(result)
    reference_total = _reproduction_count(result, "hf")
    trtmc_total = _reproduction_count(result, "trtmc")
    if sample_limit:
        sample_label = "sample" if sample_limit == 1 else "samples"
        dataset_label = f"Dataset slice ({sample_limit} {sample_label})"
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
    return str(
        raw_result.get("hf_cache_status")
        or raw_result.get("hf_reference_status")
        or ""
    )


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
    return _signal(
        status,
        {"completed": "Completed", "error": "Error", "not_run": "Not run"},
    )


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
    detail_text = mode or not_compared_reason
    detail = (
        f'<div class="detail">{html.escape(detail_text)}</div>'
        if detail_text
        else ""
    )
    return signal + detail


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
    status = (
        str(validation.get("status", "failed"))
        if isinstance(validation, dict)
        else "failed"
    )
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
    _command, sample_limit, _ = _dataset_reproduction(result)
    if sample_limit:
        return str(sample_limit)
    return "Full"


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
    execution_errors = sum(
        result["execution"]["status"] == "error" for result in results
    )
    return validation_counts, comparison_counts, execution_errors


def _traffic_light_counts(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {"green": 0, "yellow": 0, "red": 0, "white": 0}
    for result in results:
        validation_status = str(result["validation"]["status"])
        comparison_status = str(result["comparison"]["status"])
        if validation_status == "skipped":
            counts["yellow"] += 1
        elif comparison_status == "agreement":
            counts["green"] += 1
        elif comparison_status == "disagreement":
            counts["red"] += 1
        else:
            counts["white"] += 1
    return counts


def _report_rows(
    output: Path,
    results: Sequence[Mapping[str, Any]],
    result_paths: Sequence[Path],
) -> str:
    rows = []
    for result, path in zip(results, result_paths, strict=True):
        relative = path.relative_to(output)
        metadata = result.get("disagreements", {})
        artifact_name = (
            str(metadata.get("path", "disagreements.jsonl"))
            if isinstance(metadata, dict)
            else "disagreements.jsonl"
        )
        artifact_relative = relative.parent / artifact_name
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('model', '')))}</td>"
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


def _report_document(
    report: Mapping[str, Any],
    *,
    rows: str,
    comparison_counts: Mapping[str, int],
    execution_errors: int,
    traffic_light_counts: Mapping[str, int],
) -> str:
    provenance = _report_provenance(report.get("run", {}))
    duration = _format_duration(report["summary"].get("duration_seconds"))
    duration_summary = f" · {html.escape(duration)} total duration" if duration else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TRTMC Reference Consistency Report</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 32px; color: #202124; }}
h1 {{ margin-bottom: 4px; }}
.purpose {{ color: #5f6368; margin-bottom: 8px; }}
.traffic-summary {{ font-size: 20px; font-weight: 650; margin: 14px 0 8px; }}
.summary {{ color: #5f6368; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dadce0; padding: 8px 10px; text-align: left; }}
th {{ background: #f8f9fa; }}
details {{ min-width: 210px; }}
summary {{ cursor: pointer; color: #185abc; }}
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
pre {{ margin: 0; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere;
       background: #f8f9fa; border: 1px solid #dadce0; border-radius: 4px; }}
.unavailable {{ color: #5f6368; }}
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
.detail {{ color: #5f6368; font-size: 12px; margin-top: 4px; }}
.metric {{ display: flex; justify-content: space-between; gap: 14px;
           font-variant-numeric: tabular-nums; font-size: 12px; }}
.metric span {{ color: #5f6368; }}
.metric.primary {{ font-size: 13px; }}
.metric.primary span, .metric.primary strong {{ color: #202124; }}
</style></head><body>
<h1>TRTMC Reference Consistency Report</h1>
<div class="purpose">Accuracy and output agreement against the model reference.</div>
<div class="traffic-summary" title="Agreement · Skipped · Disagreement · Not compared">
🟢 {traffic_light_counts["green"]} &nbsp; 🟡 {traffic_light_counts["yellow"]} &nbsp;
🔴 {traffic_light_counts["red"]} &nbsp; ⚪ {traffic_light_counts["white"]}
</div>
<div class="summary">{report["summary"]["cases"]} cases ·
{comparison_counts["agreement"]} agreements ·
{comparison_counts["disagreement"]} disagreements ·
{comparison_counts["not_run"]} not compared ·
{execution_errors} execution errors ·
{report["summary"]["selected_samples"]} samples{duration_summary}<br>
{html.escape(provenance)}</div>
<table><thead><tr><th>Model</th><th>Workload</th><th>Samples</th><th>Execution</th>
<th>Reference</th><th>Comparison</th><th>Agreement metrics</th>
<th>Validation</th><th>Vanilla reproduction</th><th>Result</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""


def write_report(output: Path) -> tuple[Path, Path, dict[str, Any]]:
    result_paths = sorted(output.glob("*/*/comparison.json"))
    results = _normalize_result_files(result_paths)
    result_paths, results = _deduplicate_results(result_paths, results)
    validation_counts, comparison_counts, execution_errors = _report_counts(results)
    traffic_light_counts = _traffic_light_counts(results)
    sample_limits = [_dataset_reproduction(result)[1] for result in results]
    generated_at = _utc_now()
    report = {
        "schema_version": "trtmc.validation-report/v2",
        "generated_at": generated_at.isoformat(),
        "validation_status": (
            "failed"
            if not results or validation_counts["failed"]
            else "incomplete"
            if validation_counts["not_compared"]
            else "passed"
        ),
        "summary": {
            "cases": len(results),
            "execution_completed": sum(
                result["execution"]["status"] == "completed"
                for result in results
            ),
            "execution_errors": execution_errors,
            "agreements": comparison_counts["agreement"],
            "disagreements": comparison_counts["disagreement"],
            "not_compared": comparison_counts["not_run"],
            "validation_passed": validation_counts["passed"],
            "validation_failed": validation_counts["failed"],
            "validation_skipped": validation_counts["skipped"],
            "selected_samples": sum(sample_limits),
        },
        "results": results,
    }
    run_path = output / "run.json"
    if run_path.is_file():
        report["run"] = json.loads(run_path.read_text(encoding="utf-8"))
        duration_seconds = _elapsed_seconds(
            report["run"].get("started_at"),
            generated_at,
        )
        if duration_seconds is not None:
            report["summary"]["duration_seconds"] = duration_seconds
    json_path = output / "report.json"
    html_path = output / "report.html"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    document = _report_document(
        report,
        rows=_report_rows(output, results, result_paths),
        comparison_counts=comparison_counts,
        execution_errors=execution_errors,
        traffic_light_counts=traffic_light_counts,
    )
    html_path.write_text(document, encoding="utf-8")
    return json_path, html_path, report


def _print_result(result: Mapping[str, Any], comparison: Path, report: Path) -> None:
    not_compared_reason = str(result.get("not_compared_reason", "") or "")
    if not_compared_reason:
        print()
        print(f"Compare result: {comparison}")
        print(f"Report:         {report}")
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
    print(f"Report:         {report}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate TRTMC against model reference implementations."
    )
    parser.add_argument("model", nargs="?")
    parser.add_argument("workload", nargs="?")
    parser.add_argument("--all", action="store_true", help="run every ready model")
    parser.add_argument(
        "--on-model-failure",
        choices=("continue", "stop"),
        default="continue",
        help=("with --all, continue after a failed model or stop after recording it"),
    )
    parser.add_argument(
        "--model-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--list", action="store_true", help="list model-first workloads")
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
    parser.add_argument("--trtmc-binary", type=Path, default=REPO_ROOT / "build" / "trtmc")
    parser.add_argument(
        "--benchmark-binary",
        type=Path,
        default=REPO_ROOT / "build" / "trtmc_dataset_benchmark",
    )
    parser.add_argument("--hf-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--model-plugin-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "override the workload sample limit; use 0 for the complete dataset"
        ),
    )
    parser.add_argument("--force-hf", action="store_true")
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    suites_list = task_eval.load_suites(arguments.suites)
    suites = {suite["id"]: suite for suite in suites_list}
    ready = ready_model_names(arguments.models_dir)
    task_models = _task_eval_models(arguments.models_dir)
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
) -> list[Binding]:
    if arguments.all:
        if arguments.model or arguments.workload or arguments.dataset:
            raise ValidationError("--all cannot be combined with MODEL, WORKLOAD, or --dataset")
        return [resolve_binding(catalog, model) for model in ready_models]
    if not arguments.model:
        raise ValidationError("provide MODEL [WORKLOAD], --all, or --list")
    return [resolve_binding(catalog, arguments.model, arguments.workload)]


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
    arguments.output.mkdir(parents=True, exist_ok=True)
    arguments.engine_dir.mkdir(parents=True, exist_ok=True)
    arguments.reference_cache_dir.mkdir(parents=True, exist_ok=True)


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
    ):
        command.extend([option, str(value)])
    for option, value in (
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
                "error_type": "WorkerProcessError",
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


def _run_supervised_binding(
    binding: Binding,
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    case_dir = _case_directory(arguments.output, binding)
    case_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = case_dir / "comparison.json"
    comparison_path.unlink(missing_ok=True)
    worker_log = case_dir / "worker.log"
    command = _worker_command(binding, arguments)
    launch_error = ""
    try:
        returncode = _run_subprocess(command, worker_log, _source_environment())
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
        result = _worker_error_result(
            binding,
            command=command,
            returncode=returncode,
            worker_log=worker_log,
            sample_limit=resolve_sample_limit(catalog, binding, arguments.limit),
            error=str(exc),
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


def _run_all_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
) -> int:
    _prepare_run_directories(arguments)
    write_run_metadata(arguments.output)
    failed = False
    for binding in bindings:
        if not binding.runnable:
            print(
                f"\nNot compared: {binding.model} / "
                f"{binding.not_compared_reason}",
                flush=True,
            )
            result, comparison = _write_not_compared_case(
                binding,
                arguments.output,
            )
            _, report_path, _ = write_report(arguments.output)
            _print_result(result, comparison, report_path)
            continue
        sample_limit = resolve_sample_limit(catalog, binding, arguments.limit)
        sample_note = (
            "full dataset"
            if sample_limit == 0
            else f"{sample_limit} samples"
        )
        print(
            f"\nStarting worker: {binding.model} / {binding.workload} / {sample_note}",
            flush=True,
        )
        result = _run_supervised_binding(
            binding,
            arguments=arguments,
            catalog=catalog,
        )
        _, report_path, _ = write_report(arguments.output)
        comparison = _case_directory(arguments.output, binding) / "comparison.json"
        _print_result(result, comparison, report_path)
        model_failed = result["validation"]["status"] == "failed"
        failed = failed or model_failed
        if model_failed and arguments.on_model_failure == "stop":
            print(
                f"Stopping after failed model: {binding.model}",
                flush=True,
            )
            break
    return 1 if failed else 0


def _run_bindings(
    bindings: Iterable[Binding],
    *,
    arguments: argparse.Namespace,
    catalog: Mapping[str, Any],
    task_models: Mapping[str, dict[str, Any]],
    suites: Mapping[str, dict[str, Any]],
) -> int:
    _prepare_run_directories(arguments)
    if not arguments.model_worker:
        write_run_metadata(arguments.output)
    failed = False
    not_compared = False
    for binding in bindings:
        if not binding.runnable:
            print(
                f"\nNot compared: {binding.model} / "
                f"{binding.not_compared_reason}",
                flush=True,
            )
            result, comparison = _write_not_compared_case(
                binding,
                arguments.output,
            )
            not_compared = True
            if not arguments.model_worker:
                _, report_path, _ = write_report(arguments.output)
                _print_result(result, comparison, report_path)
            continue
        binding_arguments = copy.copy(arguments)
        binding_arguments.limit = resolve_sample_limit(
            catalog,
            binding,
            arguments.limit,
        )
        sample_note = (
            "full dataset"
            if binding_arguments.limit == 0
            else f"{binding_arguments.limit} samples"
        )
        print(
            f"\n{binding.model} / {binding.workload} / {sample_note}",
            flush=True,
        )
        result = run_binding(
            binding,
            arguments=binding_arguments,
            task_models=task_models,
            suites=suites,
        )
        if not arguments.model_worker:
            _, report_path, _ = write_report(arguments.output)
            comparison = _case_directory(arguments.output, binding) / "comparison.json"
            _print_result(result, comparison, report_path)
        failed = failed or result["validation"]["status"] == "failed"
    if failed:
        return 1
    return 2 if not_compared and not arguments.all else 0


def _main(arguments: argparse.Namespace) -> int:
    catalog, suites, ready, task_models = _load_validation_inputs(arguments)
    if arguments.list:
        for name, spec in catalog["models"].items():
            not_compared_reason = str(spec.get("not_compared_reason", "") or "")
            if not_compared_reason:
                print(f"{name}: not compared ({not_compared_reason})")
                continue
            workloads = []
            for workload in spec["workloads"]:
                limit = catalog["sample_limits"][workload]
                workloads.append(f"{workload} ({limit} samples)")
            print(f"{name}: {', '.join(workloads)}")
        return 0
    bindings = _select_bindings(arguments, catalog, ready)
    if arguments.dry_run:
        _print_bindings(
            bindings,
            catalog=catalog,
            explicit_limit=arguments.limit,
        )
        return 0
    if arguments.all:
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

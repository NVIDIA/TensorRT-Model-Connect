#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import hashlib
import html
import json
import math
import os
import random
import re
import shlex
import shutil
import struct
import traceback
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e_harness.manifest_loader import load_manifest  # noqa: E402
from tests.e2e_harness.python_profiles import (  # noqa: E402
    normalize_execution_profiles,
    resolve_profile_python,
)
from tests.e2e_harness.registry import (  # noqa: E402
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
)
from tools.validation import artifacts as validation_artifacts  # noqa: E402
from tools.validation import catalog as validation_catalog  # noqa: E402
from tools.validation.model_plugin_contract import (  # noqa: E402
    deserialize_stage_output,
    manifest_path_from_work_manifest,
    response_from_output,
    select_case,
    serialize_stage_output,
)


# Shared catalog and artifact Interfaces used by the engine.
_generated_token_ids = validation_artifacts.generated_token_ids
predictions_file_valid = validation_artifacts.predictions_file_valid
DEFAULT_MODELS_DIR = validation_catalog.DEFAULT_MODELS_DIR
DEFAULT_SUITES = validation_catalog.DEFAULT_SUITES
_selector_values = validation_catalog._selector_values
infer_reference_family = validation_catalog.infer_reference_family
infer_user_contract = validation_catalog.infer_user_contract
load_manifest_records = validation_catalog.load_manifest_records
load_structured_file = validation_catalog.load_structured_file
load_suites = validation_catalog.load_suites
manifest_record = validation_catalog.manifest_record
resolve_suite_for_model = validation_catalog.resolve_suite_for_model
suite_by_id = validation_catalog.suite_by_id
suite_match_reason = validation_catalog.suite_match_reason


DEFAULT_WAIVES = REPO_ROOT / "tests" / "e2e" / "waives.txt"
REFERENCE_RUNNER = REPO_ROOT / "tools" / "trtmc_reference.py"
ERROR_OUTPUT_TEXT = "TensorRT Edge LLM cannot handle this request. Fails."
CHOICE_LETTERS = set("ABCDEFGHIJ")
GPT_OSS_MMLU_SYSTEM_PROMPT = "You are a helpful assistant. Answer with only the option letter."
_DIFFUSION_SAMPLE_INPUT_FIELDS = frozenset(
    {
        "action",
        "camera_intrinsics",
        "camera_intrinsics_file",
        "image",
        "image_path",
        "prompt_file",
        "rotation_speed_deg",
        "translation_speed",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ci_suite(suite: dict[str, Any], lane: str) -> dict[str, Any]:
    ci = suite.get("ci", {})
    if not isinstance(ci, dict) or ci.get("eligible") is not True:
        raise ValueError(f"Validation suite {suite['id']!r} is not CI-eligible")
    if str(ci.get("lane", "")) != lane:
        raise ValueError(
            f"Validation suite {suite['id']!r} belongs to lane "
            f"{ci.get('lane')!r}, not {lane!r}"
        )
    if int(ci.get("limit", 0) or 0) <= 0:
        raise ValueError(f"Validation suite {suite['id']!r} must define a positive CI limit")
    if not isinstance(ci.get("sample_seed"), int):
        raise ValueError(
            f"Validation suite {suite['id']!r} must define an integer CI sample seed"
        )
    models = suite.get("default_model_names", [])
    if not isinstance(models, list) or not models or not all(isinstance(item, str) for item in models):
        raise ValueError(f"Validation suite {suite['id']!r} has no default CI models")
    return ci


def _verified_ci_dataset(path: Path, expected_sha256: str) -> Path | None:
    if path.is_file() and _sha256_file(path) == expected_sha256:
        return path
    return None


def ensure_ci_dataset(
    suite: dict[str, Any], *, explicit_path: Path | None, cache_root: Path
) -> Path:
    dataset = suite.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("CI validation dataset configuration must be a mapping")
    expected_sha256 = str(dataset.get("sha256", ""))
    if len(expected_sha256) != 64:
        raise ValueError("CI validation dataset must define a SHA-256 digest")

    candidates = [explicit_path] if explicit_path is not None else []
    if dataset.get("default_path"):
        candidates.append(Path(str(dataset["default_path"])))
    for candidate in candidates:
        verified = _verified_ci_dataset(candidate, expected_sha256)
        if verified is not None:
            return verified
    if explicit_path is not None:
        raise ValueError("Explicit CI validation dataset is missing or has the wrong checksum")

    source = str(dataset.get("source", ""))
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme != "https":
        raise ValueError("CI validation dataset source must use HTTPS")
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError("CI validation dataset source has no filename")
    destination = cache_root / str(suite["id"]) / filename
    verified = _verified_ci_dataset(destination, expected_sha256)
    if verified is not None:
        return verified

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(source, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if _sha256_file(temporary) != expected_sha256:
            raise ValueError("Downloaded CI validation dataset has the wrong checksum")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def validate_eval_summary(
    summary: dict[str, Any], expected_models: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    raw_results = summary.get("results", [])
    if not isinstance(raw_results, list):
        return False, []
    results = [result for result in raw_results if isinstance(result, dict)]
    actual_models = [str(result.get("model", "")) for result in results]
    complete = len(results) == len(expected_models) and sorted(actual_models) == sorted(expected_models)
    return complete and all(result.get("status") == "passed" for result in results), results


def _public_ci_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "suite",
        "model",
        "hf_id",
        "status",
        "mode",
        "valid_count",
        "passed_count",
        "sample_agreement_rate",
        "prediction_agreement_rate",
        "mean_relative_l2",
        "max_relative_l2",
        "max_absolute_error",
        "error_type",
    )
    return {key: result[key] for key in keys if key in result}


def _public_time_series_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary_keys = (
        "status",
        "sample_count",
        "valid_count",
        "passed_count",
        "sample_agreement_rate",
        "mean_relative_l2",
        "max_relative_l2",
        "max_absolute_error",
    )
    case_keys = (
        "sample_id",
        "output_numel",
        "hf_output_shape",
        "trtfb_output_shape",
        "relative_l2",
        "max_absolute_error",
        "passed",
    )
    gate_keys = ("max_relative_l2", "max_absolute_error", "min_sample_agreement_rate")
    public = {key: summary[key] for key in summary_keys if key in summary}
    gates = summary.get("gates", {})
    if isinstance(gates, dict):
        public["gates"] = {key: gates[key] for key in gate_keys if key in gates}
    cases = summary.get("cases", [])
    if isinstance(cases, list):
        public["cases"] = [
            {key: case[key] for key in case_keys if key in case}
            for case in cases
            if isinstance(case, dict)
        ]
    return public


def _ci_metric(value: Any) -> str:
    return f"{float(value):.6e}" if isinstance(value, (int, float)) else "n/a"


def write_public_ci_artifacts(
    *,
    suite: dict[str, Any],
    expected_models: list[str],
    results: list[dict[str, Any]],
    work_root: Path,
    artifact_dir: Path,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    passed, _ = validate_eval_summary({"results": results}, expected_models)
    payload = {
        "suite": suite["id"],
        "ci": suite["ci"],
        "models": [_public_ci_result(result) for result in results],
        "passed": passed,
        "counts": {
            "expected": len(expected_models),
            "reported": len(results),
            "passed": sum(result.get("status") == "passed" for result in results),
            "failed": sum(result.get("status") == "failed" for result in results),
            "skipped": sum(result.get("status") == "skipped" for result in results),
        },
    }
    (artifact_dir / "eval_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {suite['id']} validation CI",
        "",
        f"- Passed: `{str(passed).lower()}`",
        f"- Models: `{len(results)}/{len(expected_models)}`",
        "",
        "| Model | Status | Agreement | Max relative-L2 | Max absolute error |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {model} | {status} | {agreement} | {relative} | {absolute} |".format(
                model=result.get("model", "unknown"),
                status=result.get("status", "unknown"),
                agreement=_ci_metric(result.get("sample_agreement_rate")),
                relative=_ci_metric(result.get("max_relative_l2")),
                absolute=_ci_metric(result.get("max_absolute_error")),
            )
        )
        model_name = str(result.get("model", ""))
        numeric_summary = work_root / str(suite["id"]) / model_name / "summary.json"
        if numeric_summary.is_file():
            destination = artifact_dir / "models" / model_name / "summary.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            raw_summary = json.loads(numeric_summary.read_text(encoding="utf-8"))
            if not isinstance(raw_summary, dict):
                raise ValueError(f"Invalid time-series summary for {model_name!r}")
            destination.write_text(
                json.dumps(_public_time_series_summary(raw_summary), indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
    report = "\n".join(lines) + "\n"
    (artifact_dir / "summary.md").write_text(report, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(report)


def configure_ci_eval(args: argparse.Namespace, suite: dict[str, Any]) -> list[str]:
    ci = validate_ci_suite(suite, args.ci_lane)
    default_models = list(suite["default_model_names"])
    requested_models = list(dict.fromkeys(args.model))
    unknown_models = sorted(set(requested_models) - set(default_models))
    if unknown_models:
        raise ValueError(f"CI validation model selection is not allowlisted: {unknown_models}")
    expected_models = requested_models or default_models
    if args.limit not in {0, int(ci["limit"])}:
        raise ValueError("CI validation limit must match the suite CI profile")
    if args.sample_seed not in {None, int(ci["sample_seed"])}:
        raise ValueError("CI validation sample seed must match the suite CI profile")
    args.model = expected_models
    args.limit = int(ci["limit"])
    args.sample_seed = int(ci["sample_seed"])
    args.single_device_only = True
    args.local_files_only = True
    if not args.waive_platform:
        args.waive_platform = "GB300"
    if not args.engine_dir:
        args.engine_dir = os.environ.get("ENGINE_DIR", "")
    if not args.engine_dir:
        raise ValueError("CI validation requires --engine-dir")
    explicit_dataset = Path(args.dataset) if args.dataset else None
    args.dataset = str(
        ensure_ci_dataset(
            suite,
            explicit_path=explicit_dataset,
            cache_root=Path(args.dataset_cache_root),
        )
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return expected_models


def cmd_prepare_ci_dataset(args: argparse.Namespace) -> int:
    suite = suite_by_id(load_suites(Path(args.suites)), args.suite)
    validate_ci_suite(suite, args.ci_lane)
    dataset = ensure_ci_dataset(
        suite,
        explicit_path=Path(args.dataset) if args.dataset else None,
        cache_root=Path(args.dataset_cache_root),
    )
    print(dataset.resolve())
    return 0


def _deep_merge_mappings(*values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(merged.get(key), dict):
                merged[key] = _deep_merge_mappings(merged[key], item)
            else:
                merged[key] = copy.deepcopy(item)
    return merged


def effective_validation_config(
    suite: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Resolve suite family/model overrides, then manifest-specific settings."""
    overrides = suite.get("model_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Suite {suite.get('id', '<unknown>')} model_overrides must be a mapping")
    by_family = overrides.get("by_family", {})
    by_model = overrides.get("by_model", {})
    if not isinstance(by_family, dict) or not isinstance(by_model, dict):
        raise ValueError(
            f"Suite {suite.get('id', '<unknown>')} model_overrides.by_family/by_model "
            "must be mappings"
        )
    return _deep_merge_mappings(
        by_family.get(str(model.get("family", "")), {}),
        by_model.get(str(model.get("name", "")), {}),
        model.get("task_eval", {}),
    )


def load_waives(path: Path = DEFAULT_WAIVES, platform: str = "") -> dict[str, tuple[str, str]]:
    waives: dict[str, tuple[str, str]] = {}
    if not path.is_file():
        return waives
    platform = platform.strip()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        name_part = parts[0]
        action = parts[1].upper()
        reason = parts[2] if len(parts) > 2 else ""
        if action not in {"SKIP", "XFAIL"}:
            continue
        if "/" in name_part:
            waive_platform, model_name = name_part.split("/", 1)
            if waive_platform != platform:
                continue
        else:
            model_name = name_part
        waives[model_name] = (action, reason)
    return waives


def build_plan(
    suites: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    suite_id: str | None = None,
    single_device_only: bool = False,
    include_non_matching: bool = False,
    use_default_models: bool = True,
    waives: dict[str, tuple[str, str]] | None = None,
    include_waived: bool = False,
) -> list[dict[str, Any]]:
    selected_suites = [suite_by_id(suites, suite_id)] if suite_id else suites
    rows: list[dict[str, Any]] = []
    waives = waives or {}
    for suite in selected_suites:
        default_model_names = (
            _selector_values(suite, "default_model_names") if use_default_models else set()
        )
        if use_default_models and not default_model_names:
            default_model_names = _selector_values(
                suite.get("selectors", {}), "default_model_names"
            )
        for model in models:
            matched, reason = suite_match_reason(suite, model)
            if matched and default_model_names and model["name"] not in default_model_names:
                matched = False
                reason = f"model={model['name']} is compatible but not selected by default"
            if single_device_only and model["requires_multi_device"]:
                matched = False
                reason = "requires multi-device runtime"
            waive = waives.get(str(model["name"]))
            if matched and waive and not include_waived:
                action, waive_reason = waive
                matched = False
                reason = f"waived {action}: {waive_reason}".strip()
            if matched or include_non_matching:
                rows.append(
                    {
                        "suite": suite["id"],
                        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
                        "model": model["name"],
                        "hf_id": model["hf_id"],
                        "bundle": model["bundle"],
                        "runtime_strategy": model["runtime_strategy"],
                        "task_strategy": model["task_strategy"],
                        "reference_family": model["reference_family"],
                        "user_contract": model["user_contract"],
                        "ci_tier": model["ci_tier"],
                        "requires_multi_device": model["requires_multi_device"],
                        "selected": matched,
                        "reason": reason,
                        "manifest": model["manifest"],
                    }
                )
    return rows


def print_plan_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        ("suite", 20),
        ("model", 28),
        ("runtime_strategy", 24),
        ("user_contract", 22),
        ("ci_tier", 12),
        ("scope", 8),
        ("reason", 32),
    ]
    header = "  ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        data = dict(row)
        data["scope"] = "multi" if row["requires_multi_device"] else "single"
        print("  ".join(str(data.get(name, ""))[:width].ljust(width) for name, width in columns))


def _request_prompt(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        parts = [str(msg.get("content", "")) for msg in messages if msg.get("content")]
        if parts:
            return "\n".join(parts)
    prompt = request.get("prompt")
    if isinstance(prompt, str):
        return prompt
    raise ValueError("MMLU request has neither messages content nor prompt")


def render_mmlu_prompt(prompt: str, validation_config: dict[str, Any] | None) -> str:
    config = validation_config if isinstance(validation_config, dict) else {}
    renderer = str(config.get("prompt_renderer", "") or "")
    if not renderer:
        return prompt
    if renderer != "gpt_oss_harmony_mcq":
        raise ValueError(f"Unsupported MMLU prompt renderer {renderer!r}")
    system_prompt = str(
        config.get("system_prompt", GPT_OSS_MMLU_SYSTEM_PROMPT)
        or GPT_OSS_MMLU_SYSTEM_PROMPT
    )
    return (
        f"<|start|>system<|message|>{system_prompt}<|end|>"
        f"<|start|>user<|message|>{prompt}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )


def _copy_dataset_header(data: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: v for k, v in data.items() if k not in {"requests", "samples"}}
    out["requests"] = requests
    return out


def load_mmlu_requests(
    dataset_path: Path,
    *,
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed = [
        (idx, req)
        for idx, req in enumerate(requests)
        if not subject or str(req.get("subject", "")) == subject
    ]
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def prepare_mmlu_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data, indexed = load_mmlu_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_prefix = str(suite.get("dataset", {}).get("sample_prefix", "mmlu"))
    requests = []
    for dataset_index, request in indexed:
        prepared_request = dict(request)
        prepared_request["sample_id"] = str(
            request.get("id")
            or request.get("sample_id")
            or f"{sample_prefix}_{dataset_index:06d}"
        )
        requests.append(prepared_request)
    answers = _copy_dataset_header(data, requests)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"

    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    with prompts_path.open("w", encoding="utf-8") as f:
        for out_idx, ((dataset_index, request), prepared_request) in enumerate(
            zip(indexed, requests, strict=True)
        ):
            prompt = render_mmlu_prompt(_request_prompt(request), validation_config)
            sample = {
                "sample_id": prepared_request["sample_id"],
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": request.get("subject", ""),
                "answer": request["answer"],
                "prompt": prompt,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    generation = _deep_merge_mappings(
        suite.get("generation", {}),
        (validation_config or {}).get("generation", {}),
    )

    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": generation,
        "files": {
            "answers": str(answers_path),
            "prompts": str(prompts_path),
        },
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_conditional_text_jsonl_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_config = suite.get("dataset", {})
    source_field = str(dataset_config.get("source_field", "input"))
    answer_field = str(dataset_config.get("answer_field", "output"))
    subject_field = str(dataset_config.get("subject_field", "subset"))
    sample_prefix = str(dataset_config.get("sample_prefix", "conditional_text"))
    indexed: list[tuple[int, dict[str, Any]]] = []
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for dataset_index, raw_line in enumerate(dataset_file):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{dataset_path}: row {dataset_index + 1} must be an object")
            source_text = str(row.get(source_field, "")).strip()
            answer = str(row.get(answer_field, "")).strip()
            row_subject = str(row.get(subject_field, "")).strip()
            if not source_text or not answer:
                raise ValueError(
                    f"{dataset_path}: row {dataset_index + 1} needs non-empty "
                    f"{source_field!r} and {answer_field!r}"
                )
            if subject and row_subject != subject:
                continue
            indexed.append(
                (
                    dataset_index,
                    {
                        "sample_id": str(row.get("id") or f"{sample_prefix}_{dataset_index:06d}"),
                        "dataset_index": dataset_index,
                        "source_text": source_text,
                        "answer": answer,
                        "subject": row_subject,
                    },
                )
            )
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    answers_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_config.get("name", dataset_path.stem)),
                "requests": [row for _idx, row in indexed],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (dataset_index, row) in enumerate(indexed):
            prompts_file.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "dataset_index": dataset_index,
                        "eval_index": eval_index,
                        "prompt": row["source_text"],
                        "source_text": row["source_text"],
                        "subject": row["subject"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    generation = _deep_merge_mappings(
        suite.get("generation", {}),
        (validation_config or {}).get("generation", {}),
    )
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": dataset_config.get("kind", ""),
        "reference_mode": suite.get("reference", {}).get("mode", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": generation,
        "scoring": suite.get("scoring", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_unconditional_text_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed: list[tuple[int, dict[str, Any]]] = []
    for dataset_index, raw_request in enumerate(requests):
        if not isinstance(raw_request, dict):
            raise ValueError(f"{dataset_path}: request {dataset_index} must be an object")
        row_subject = str(raw_request.get("subject", "unconditional")).strip()
        if subject and row_subject != subject:
            continue
        indexed.append(
            (
                dataset_index,
                {
                    "sample_id": str(raw_request.get("id") or f"unconditional_{dataset_index:06d}"),
                    "dataset_index": dataset_index,
                    "subject": row_subject,
                    "seed": int(raw_request.get("seed", dataset_index)),
                },
            )
        )
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    selected = [row for _idx, row in indexed]
    answers_path.write_text(
        json.dumps(
            {"dataset": data.get("dataset", dataset_path.stem), "requests": selected},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (dataset_index, row) in enumerate(indexed):
            prompts_file.write(
                json.dumps(
                    {
                        **row,
                        "dataset_index": dataset_index,
                        "eval_index": eval_index,
                        "prompt": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    generation = _deep_merge_mappings(
        suite.get("generation", {}),
        (validation_config or {}).get("generation", {}),
    )
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "reference_mode": suite.get("reference", {}).get("mode", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": generation,
        "scoring": suite.get("scoring", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_diffusion_prompt_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file, delimiter="\t")
        required = {"Prompt", "Category", "Challenge", "Note"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{dataset_path}: missing PartiPrompts columns {sorted(missing)}"
            )
        indexed = []
        for dataset_index, row in enumerate(reader):
            prompt = str(row.get("Prompt", "")).strip()
            if not prompt:
                raise ValueError(f"{dataset_path}: empty prompt at row {dataset_index + 2}")
            category = str(row.get("Category", "")).strip()
            if subject and category != subject:
                continue
            indexed.append(
                (
                    dataset_index,
                    {
                        "sample_id": f"partiprompts_{dataset_index:06d}",
                        "dataset_index": dataset_index,
                        "prompt": prompt,
                        "category": category,
                        "challenge": str(row.get("Challenge", "")).strip(),
                        "note": str(row.get("Note", "")).strip(),
                    },
                )
            )

    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    answers = {
        "dataset": "PartiPrompts",
        "requests": [request for _dataset_index, request in indexed],
    }
    answers_path.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (dataset_index, request) in enumerate(indexed):
            prompt_row = {
                "sample_id": request["sample_id"],
                "dataset_index": dataset_index,
                "eval_index": eval_index,
                "prompt": request["prompt"],
                "category": request["category"],
                "challenge": request["challenge"],
            }
            prompts_file.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")

    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": suite.get("generation", {}),
        "files": {
            "answers": str(answers_path),
            "prompts": str(prompts_path),
        },
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_diffusion_prompt_json_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_config = suite.get("dataset", {})
    asset_fields = dataset_config.get("asset_fields", [])
    if not isinstance(asset_fields, list) or not all(
        isinstance(field, str) for field in asset_fields
    ):
        raise ValueError(f"Suite {suite['id']} dataset.asset_fields must be a list of strings")
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed: list[tuple[int, dict[str, Any]]] = []
    for source_position, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"{dataset_path}: request {source_position} must be an object")
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{dataset_path}: request {source_position} has no prompt")
        category = str(request.get("category", "")).strip()
        if subject and subject not in {value.strip() for value in category.split(",")}:
            continue
        prepared = dict(request)
        prepared["prompt"] = prompt
        prepared.setdefault("sample_id", f"diffusion_prompt_{source_position:06d}")
        prepared.setdefault("dataset_index", source_position)
        prepared.setdefault("category", category)
        prepared.setdefault("challenge", "")
        for field in asset_fields:
            value = prepared.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{dataset_path}: request {source_position} has no {field!r} asset"
                )
            asset_path = Path(value)
            if not asset_path.is_absolute():
                asset_path = dataset_path.parent / asset_path
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"{dataset_path}: request {source_position} {field!r} asset "
                    f"does not exist: {asset_path}"
                )
            prepared[field] = str(asset_path.resolve())
        indexed.append((source_position, prepared))
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    selected = [request for _source_position, request in indexed]
    answers = _copy_dataset_header(data, selected)
    answers_path.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (_source_position, request) in enumerate(indexed):
            prompt_row = dict(request)
            prompt_row["eval_index"] = eval_index
            prompts_file.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_name": data.get("dataset", ""),
        "dataset_version": data.get("version", ""),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": suite.get("generation", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_model_plugin_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Prepare fixed rows consumed by model-owned reference/runner plugins."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    dataset_config = suite.get("dataset", {})
    asset_fields = dataset_config.get("input_asset_fields", [])
    if not isinstance(asset_fields, list) or not all(
        isinstance(field, str) and field for field in asset_fields
    ):
        raise ValueError(
            f"Suite {suite['id']} dataset.input_asset_fields must be a list of strings"
        )

    indexed: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for source_position, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"{dataset_path}: request {source_position} must be an object")
        prepared = copy.deepcopy(request)
        sample_id = str(
            prepared.get("sample_id", f"model_plugin_{source_position:06d}")
        ).strip()
        if not sample_id:
            raise ValueError(f"{dataset_path}: request {source_position} has no sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"{dataset_path}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        prepared["sample_id"] = sample_id
        prepared.setdefault("dataset_index", source_position)
        category = str(prepared.get("category", "") or "")
        if subject and subject not in {
            value.strip() for value in category.split(",")
        }:
            continue
        inputs = prepared.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(
                f"{dataset_path}: request {source_position} inputs must be an object"
            )
        for field in asset_fields:
            value = inputs.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{dataset_path}: request {source_position} has no inputs.{field}"
                )
            asset_path = Path(value)
            if not asset_path.is_absolute():
                asset_path = dataset_path.parent / asset_path
            if not asset_path.is_file():
                raise FileNotFoundError(
                    f"{dataset_path}: request {source_position} inputs.{field} "
                    f"does not exist: {asset_path}"
                )
            inputs[field] = str(asset_path.resolve())
            if field in {"audio", "image", "video", "video_path"}:
                prepared[field] = inputs[field]
        prepared["inputs"] = inputs
        indexed.append((source_position, prepared))
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    if not indexed:
        raise ValueError(
            f"{dataset_path}: no model-plugin requests selected"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    selected = [request for _source_position, request in indexed]
    answers_path.write_text(
        json.dumps(
            _copy_dataset_header(data, selected),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (_source_position, request) in enumerate(indexed):
            prompt_row = dict(request)
            prompt_row["eval_index"] = eval_index
            prompts_file.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_name": data.get("dataset", ""),
        "dataset_version": data.get("version", ""),
        "dataset_kind": "model_plugin_json",
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def load_seedtts_requests(
    dataset_path: Path,
    *,
    limit: int = 0,
    sample_seed: int | None = None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed = list(enumerate(requests))
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def prepare_seedtts_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if subject:
        raise ValueError(
            "SeedTTS reference-consistency validation does not support --subject; select a language suite instead"
        )
    data, indexed = load_seedtts_requests(
        dataset_path,
        limit=limit,
        sample_seed=sample_seed,
    )
    dataset_config = suite.get("dataset", {})
    language = str(dataset_config.get("language", "") or "")
    prepared_requests: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for out_idx, (dataset_index, request) in enumerate(indexed):
        if not isinstance(request, dict):
            raise ValueError(f"{dataset_path}: request {dataset_index} must be an object")
        reference = str(request.get("reference", "") or "").strip()
        if not reference:
            raise ValueError(f"{dataset_path}: request {dataset_index} has no reference text")
        audio_ref = str(request.get("reference_wav", "") or "")
        if not audio_ref:
            raise ValueError(f"{dataset_path}: request {dataset_index} has no reference_wav")
        reference_wav = resolve_dataset_asset_path(dataset_path, audio_ref)
        sample_id = str(request.get("id", f"seedtts_{dataset_index:06d}"))
        prepared = dict(request)
        prepared["answer"] = reference
        prepared["reference_wav"] = str(reference_wav)
        prepared["subject"] = language
        prepared_requests.append(prepared)
        prompt_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": language,
                "answer": reference,
                "prompt": reference,
                "reference_wav": str(reference_wav),
                "language": language,
            }
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    answers = _copy_dataset_header(data, prepared_requests)
    answers["scoring"] = suite.get("scoring", {})
    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    with prompts_path.open("w", encoding="utf-8") as f:
        for row in prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": dataset_config.get("kind", ""),
        "request_count": len(indexed),
        "language": language,
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": suite.get("generation", {}),
        "scoring": suite.get("scoring", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def _message_text_parts(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return parts


def _message_image_refs(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    refs: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        value = item.get("image") or item.get("path")
        if isinstance(value, str):
            refs.append(value)
    return refs


def _message_audio_refs(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    refs: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "audio":
            continue
        value = item.get("audio") or item.get("path")
        if isinstance(value, str):
            refs.append(value)
    return refs


def _unified_image_ref(item: dict[str, Any]) -> str:
    value = item.get("image") or item.get("path")
    return value if isinstance(value, str) else ""


def _normalize_unified_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: value for key, value in message.items() if key != "content"}
    content = message.get("content")
    if not isinstance(content, list):
        normalized["content"] = content
        return normalized

    normalized_content: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image":
            image_ref = _unified_image_ref(item)
            if image_ref:
                image_item = {key: value for key, value in item.items() if key != "path"}
                image_item["image"] = image_ref
                normalized_content.append(image_item)
            continue
        normalized_content.append(item)
    normalized["content"] = normalized_content
    return normalized


def _unified_sample_media_refs(sample: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    media = sample.get("media")
    if isinstance(media, list):
        for item in media:
            if isinstance(item, dict) and item.get("type") == "image":
                image_ref = _unified_image_ref(item)
                if image_ref:
                    refs.append(image_ref)
    return refs


def _unified_answer(sample: dict[str, Any]) -> tuple[str, list[str]]:
    raw_answer = sample.get("answer")
    if isinstance(raw_answer, dict):
        primary = str(raw_answer.get("primary", ""))
        raw_aliases = raw_answer.get("aliases", [])
        aliases = [str(alias) for alias in raw_aliases] if isinstance(raw_aliases, list) else []
    else:
        primary = str(raw_answer or "")
        aliases = []
    if primary and primary not in aliases:
        aliases.insert(0, primary)
    return primary, aliases


def _unified_answer_eval(sample: dict[str, Any]) -> Any:
    raw_answer = sample.get("answer")
    if isinstance(raw_answer, dict):
        return raw_answer.get("eval")
    return None


def unified_sample_to_vlm_request(sample: dict[str, Any]) -> dict[str, Any]:
    primary_answer, aliases = _unified_answer(sample)
    eval_method = _unified_answer_eval(sample)
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    messages = sample.get("messages")
    normalized_messages = (
        [_normalize_unified_message(message) for message in messages if isinstance(message, dict)]
        if isinstance(messages, list)
        else []
    )
    media_refs = _unified_sample_media_refs(sample)
    if not normalized_messages and media_refs:
        normalized_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": media_refs[0]},
                    {"type": "text", "text": str(sample.get("question", ""))},
                ],
            }
        ]

    request: dict[str, Any] = {
        "id": str(sample.get("id") or f"ocrbench_{sample.get('source_index', 0):06d}"),
        "messages": normalized_messages,
        "answer": primary_answer,
        "question": str(sample.get("question", "")),
        "subject": str(
            sample.get("category") or sample.get("type") or sample.get("dataset_name") or ""
        ),
    }
    if aliases:
        request["answer_aliases"] = aliases
    if media_refs:
        request["image"] = media_refs[0]
    for key in ("dataset_name", "type", "source_index", "metadata"):
        if key in sample:
            request[key] = sample[key]
    if str(
        sample.get("dataset", "") or sample.get("source", "") or ""
    ).lower() == "ocrbench_v2" or str(sample.get("id", "")).startswith("ocrbench_v2_"):
        request["ocrbench_type"] = sample.get("type")
        request["ocrbench_eval"] = eval_method
        request["ocrbench_answers"] = aliases
        request["ocrbench_bbox"] = metadata.get("bbox")
        request["ocrbench_bbox_list"] = metadata.get("bbox_list")
        request["ocrbench_content"] = metadata.get("content")
    return request


def vlm_request_prompt(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            if role and role not in {"system", "user"}:
                continue
            parts.extend(_message_text_parts(msg))
        if parts:
            return "\n\n".join(part.strip() for part in parts if part.strip())
    prompt = request.get("prompt")
    if isinstance(prompt, str):
        return prompt
    raise ValueError("VLM request has neither text messages nor prompt")


def _candidate_dataset_asset_paths(dataset_path: Path, asset_ref: str) -> list[Path]:
    asset = Path(asset_ref)
    if asset.is_absolute():
        return [asset]
    dataset_dir = dataset_path.parent
    candidates = [dataset_dir / asset]
    parts = asset.parts
    if parts and parts[0].lower() == dataset_dir.name.lower():
        candidates.append(dataset_dir.joinpath(*parts[1:]))
    if parts:
        parent = dataset_dir.parent
        for child in parent.iterdir() if parent.is_dir() else []:
            if child.name.lower() == parts[0].lower():
                candidates.append(child.joinpath(*parts[1:]))
                break
    candidates.append(Path("/mnt/data") / asset)
    return candidates


def resolve_dataset_asset_path(dataset_path: Path, asset_ref: str) -> Path:
    for candidate in _candidate_dataset_asset_paths(dataset_path, asset_ref):
        if candidate.is_file():
            return candidate.resolve()
    candidates = ", ".join(
        str(path) for path in _candidate_dataset_asset_paths(dataset_path, asset_ref)
    )
    raise FileNotFoundError(f"Could not resolve dataset asset {asset_ref!r}; tried: {candidates}")


def _load_indexed_json_requests(
    dataset_path: Path,
    *,
    subject: str,
    subject_field: str,
    limit: int,
    sample_seed: int | None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed: list[tuple[int, dict[str, Any]]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"{dataset_path}: request {index} must be an object")
        if subject and str(request.get(subject_field, "")) != subject:
            continue
        indexed.append((index, request))
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def _write_indexed_json_task_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    data: dict[str, Any],
    indexed: list[tuple[int, dict[str, Any]]],
    prompt_rows: list[dict[str, Any]],
    prepared_requests: list[dict[str, Any]],
    limit: int,
    subject: str,
    sample_seed: int | None,
    validation_config: dict[str, Any] | None,
) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    answers = _copy_dataset_header(data, prepared_requests)
    answers["scoring"] = suite.get("scoring", {})
    answers_path.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for row in prompt_rows:
            prompts_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "scoring": suite.get("scoring", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_image_classification_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_config = suite.get("dataset", {})
    subject_field = str(dataset_config.get("subject_field", "synset"))
    data, indexed = _load_indexed_json_requests(
        dataset_path,
        subject=subject,
        subject_field=subject_field,
        limit=limit,
        sample_seed=sample_seed,
    )
    prepared_requests: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for eval_index, (dataset_index, request) in enumerate(indexed):
        image_ref = str(request.get("image", "") or "")
        if not image_ref:
            raise ValueError(f"{dataset_path}: request {dataset_index} has no image")
        try:
            label = int(request["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} has invalid label"
            ) from exc
        image_path = resolve_dataset_asset_path(dataset_path, image_ref)
        sample_id = str(request.get("id", f"imagenette_{dataset_index:06d}"))
        row = {
            "sample_id": sample_id,
            "dataset_index": dataset_index,
            "eval_index": eval_index,
            "subject": str(request.get(subject_field, "")),
            "image": str(image_path),
            "label": label,
            "label_name": str(request.get("label_name", "")),
        }
        prompt_rows.append(row)
        prepared_requests.append({**request, **row})
    return _write_indexed_json_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        prompt_rows=prompt_rows,
        prepared_requests=prepared_requests,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def prepare_semantic_segmentation_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_config = suite.get("dataset", {})
    subject_field = str(dataset_config.get("subject_field", "subset"))
    data, indexed = _load_indexed_json_requests(
        dataset_path,
        subject=subject,
        subject_field=subject_field,
        limit=limit,
        sample_seed=sample_seed,
    )
    prepared_requests: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for eval_index, (dataset_index, request) in enumerate(indexed):
        image_ref = str(request.get("image", "") or "")
        mask_ref = str(request.get("mask", "") or "")
        if not image_ref or not mask_ref:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} requires image and mask"
            )
        image_path = resolve_dataset_asset_path(dataset_path, image_ref)
        mask_path = resolve_dataset_asset_path(dataset_path, mask_ref)
        sample_id = str(request.get("id", f"ade20k_{dataset_index:06d}"))
        row = {
            "sample_id": sample_id,
            "dataset_index": dataset_index,
            "eval_index": eval_index,
            "subject": str(request.get(subject_field, "")),
            "image": str(image_path),
            "mask": str(mask_path),
        }
        prompt_rows.append(row)
        prepared_requests.append({**request, **row})
    return _write_indexed_json_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        prompt_rows=prompt_rows,
        prepared_requests=prepared_requests,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def prepare_prompted_segmentation_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_config = suite.get("dataset", {})
    subject_field = str(dataset_config.get("subject_field", "category"))
    data, indexed = _load_indexed_json_requests(
        dataset_path,
        subject=subject,
        subject_field=subject_field,
        limit=limit,
        sample_seed=sample_seed,
    )
    prepared_requests: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for eval_index, (dataset_index, request) in enumerate(indexed):
        required = ("image", "instance_mask", "category_mask", "text_prompt")
        missing = [field for field in required if not request.get(field)]
        if missing:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} is missing {missing}"
            )
        try:
            point_x = float(request["point_x"])
            point_y = float(request["point_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} has invalid point prompt"
            ) from exc
        if not (0.0 <= point_x <= 1.0 and 0.0 <= point_y <= 1.0):
            raise ValueError(
                f"{dataset_path}: request {dataset_index} point must be normalized"
            )
        image_path = resolve_dataset_asset_path(dataset_path, str(request["image"]))
        instance_mask = resolve_dataset_asset_path(
            dataset_path, str(request["instance_mask"])
        )
        category_mask = resolve_dataset_asset_path(
            dataset_path, str(request["category_mask"])
        )
        sample_id = str(request.get("id", f"coco_prompt_{dataset_index:06d}"))
        row = {
            "sample_id": sample_id,
            "dataset_index": dataset_index,
            "eval_index": eval_index,
            "subject": str(request.get(subject_field, "")),
            "image": str(image_path),
            "instance_mask": str(instance_mask),
            "category_mask": str(category_mask),
            "point_x": point_x,
            "point_y": point_y,
            "text_prompt": str(request["text_prompt"]),
        }
        prompt_rows.append(row)
        prepared_requests.append({**request, **row})
    return _write_indexed_json_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        prompt_rows=prompt_rows,
        prepared_requests=prepared_requests,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def prepare_reranking_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_config = suite.get("dataset", {})
    subject_field = str(dataset_config.get("subject_field", "subset"))
    data, indexed = _load_indexed_json_requests(
        dataset_path,
        subject=subject,
        subject_field=subject_field,
        limit=limit,
        sample_seed=sample_seed,
    )
    prepared_requests: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for eval_index, (dataset_index, request) in enumerate(indexed):
        query = str(request.get("query", "") or "")
        documents = request.get("documents")
        if not query:
            raise ValueError(f"{dataset_path}: request {dataset_index} has no query")
        if not isinstance(documents, list) or len(documents) < 2:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} requires at least two documents"
            )
        documents = [str(document) for document in documents]
        if any(not document for document in documents):
            raise ValueError(
                f"{dataset_path}: request {dataset_index} contains an empty document"
            )
        relevant = request.get("relevant_document_indices", [])
        if not isinstance(relevant, list):
            raise ValueError(
                f"{dataset_path}: request {dataset_index} relevant indices must be a list"
            )
        try:
            relevant_indices = [int(index) for index in relevant]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_path}: request {dataset_index} has invalid relevant indices"
            ) from exc
        if any(index < 0 or index >= len(documents) for index in relevant_indices):
            raise ValueError(
                f"{dataset_path}: request {dataset_index} relevant index is out of range"
            )
        sample_id = str(request.get("id", f"reranking_{dataset_index:06d}"))
        row = {
            "sample_id": sample_id,
            "dataset_index": dataset_index,
            "eval_index": eval_index,
            "subject": str(request.get(subject_field, "")),
            "prompt": query,
            "query": query,
            "documents": documents,
            "relevant_document_indices": relevant_indices,
        }
        prompt_rows.append(row)
        prepared_requests.append({**request, **row})
    return _write_indexed_json_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        prompt_rows=prompt_rows,
        prepared_requests=prepared_requests,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def vlm_request_image_refs(request: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                refs.extend(_message_image_refs(msg))
    if not refs and isinstance(request.get("image"), str):
        refs.append(str(request["image"]))
    return refs


def vlm_request_images(dataset_path: Path, request: dict[str, Any]) -> list[Path]:
    refs = vlm_request_image_refs(request)
    return [resolve_dataset_asset_path(dataset_path, ref) for ref in refs]


def asr_request_audio_refs(request: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                refs.extend(_message_audio_refs(msg))
    if not refs and isinstance(request.get("audio"), str):
        refs.append(str(request["audio"]))
    return refs


def asr_request_audio(dataset_path: Path, request: dict[str, Any]) -> Path:
    refs = asr_request_audio_refs(request)
    if len(refs) != 1:
        raise ValueError(
            f"ASR reference-consistency validation expects exactly one audio asset per sample; found {len(refs)}"
        )
    return resolve_dataset_asset_path(dataset_path, refs[0])


def asr_request_prompt(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg, dict):
                parts.extend(_message_text_parts(msg))
        if parts:
            return "\n\n".join(part.strip() for part in parts if part.strip())
    prompt = request.get("prompt")
    if isinstance(prompt, str):
        return prompt
    return "Please transcribe the following audio."


def validate_asr_dataset_assets(
    dataset_path: Path,
    indexed: list[tuple[int, dict[str, Any]]],
) -> None:
    missing: list[tuple[int, str, str]] = []
    for dataset_index, request in indexed:
        sample_id = str(request.get("id") or f"asr_{dataset_index:06d}")
        for ref in asr_request_audio_refs(request):
            if not any(
                candidate.is_file()
                for candidate in _candidate_dataset_asset_paths(dataset_path, ref)
            ):
                missing.append((dataset_index, sample_id, ref))
    if missing:
        examples = "; ".join(
            f"dataset_index={idx} sample_id={sample_id} ref={ref!r}"
            for idx, sample_id, ref in missing[:5]
        )
        raise FileNotFoundError(
            f"ASR dataset has {len(missing)} missing audio asset(s) for "
            f"{dataset_path}; first missing: {examples}"
        )


def validate_vlm_dataset_assets(
    dataset_path: Path,
    indexed: list[tuple[int, dict[str, Any]]],
) -> None:
    missing: list[tuple[int, str, str]] = []
    for dataset_index, request in indexed:
        sample_id = str(request.get("id") or f"vlm_{dataset_index:06d}")
        for ref in vlm_request_image_refs(request):
            if not any(
                candidate.is_file()
                for candidate in _candidate_dataset_asset_paths(dataset_path, ref)
            ):
                missing.append((dataset_index, sample_id, ref))
    if missing:
        examples = "; ".join(
            f"dataset_index={idx} sample_id={sample_id} ref={ref!r}"
            for idx, sample_id, ref in missing[:5]
        )
        raise FileNotFoundError(
            f"VLM dataset has {len(missing)} missing image asset(s) for "
            f"{dataset_path}; first missing: {examples}"
        )


def _safe_sample_filename(sample_id: str, suffix: str = ".png") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return f"{stem or 'sample'}{suffix}"


def _resize_image_to_square(src: Path, dst: Path, image_size: int) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Fixed VLM reference-consistency validation normalization requires Pillow") from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
        rgb.resize((image_size, image_size), resampling).save(dst)


def _convert_audio_to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        shutil.copyfile(src, dst)
        return
    try:
        import soundfile as sf

        data, sample_rate = sf.read(src, always_2d=False)
        sf.write(dst, data, sample_rate, subtype="PCM_16")
        return
    except Exception:
        pass
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(src))
        torchaudio.save(str(dst), waveform, sample_rate)
        return
    except Exception:
        pass
    converters = [
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(dst)],
        ["sox", str(src), str(dst)],
        ["flac", "-d", "-f", "-o", str(dst), str(src)],
    ]
    for cmd in converters:
        if not shutil.which(cmd[0]):
            continue
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            return
    raise RuntimeError(
        f"Could not convert audio asset {src} to WAV. Install soundfile, torchaudio, "
        "ffmpeg, sox, or flac in the environment running reference-consistency validation prepare."
    )


def _normalized_single_user_vlm_request(
    request: dict[str, Any],
    *,
    prompt: str,
    image_path: Path,
) -> dict[str, Any]:
    normalized = {key: value for key, value in request.items() if key != "messages"}
    normalized["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return normalized


def load_vlm_chat_requests(
    dataset_path: Path,
    *,
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed = [
        (idx, req)
        for idx, req in enumerate(requests)
        if isinstance(req, dict) and (not subject or str(req.get("subject", "")) == subject)
    ]
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def load_vlm_unified_requests(
    dataset_path: Path,
    *,
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{dataset_path}: expected top-level 'samples' list")
    indexed = [
        (idx, unified_sample_to_vlm_request(sample))
        for idx, sample in enumerate(samples)
        if isinstance(sample, dict)
        and (not subject or str(sample.get("category", sample.get("type", ""))) == subject)
    ]
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def _request_answer_from_field(request: dict[str, Any], answer_field: str) -> str:
    value: Any = request
    for part in answer_field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"ASR request is missing answer field {answer_field!r}")
        value = value[part]
    return str(value)


def load_asr_chat_requests(
    dataset_path: Path,
    *,
    limit: int = 0,
    subject: str = "",
    subject_field: str = "subject",
    sample_seed: int | None = None,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    requests = data.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"{dataset_path}: expected top-level 'requests' list")
    indexed = [
        (idx, req)
        for idx, req in enumerate(requests)
        if isinstance(req, dict)
        and (not subject or str(req.get(subject_field, req.get("subject", ""))) == subject)
    ]
    if sample_seed is not None:
        rng = random.Random(sample_seed)
        rng.shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]
    return data, indexed


def prepare_asr_chat_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_cfg = suite.get("dataset", {})
    answer_field = str(dataset_cfg.get("answer_field", "reference"))
    subject_field = str(dataset_cfg.get("subject_field", "subject"))
    data, indexed = load_asr_chat_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        subject_field=subject_field,
        sample_seed=sample_seed,
    )
    validate_asr_dataset_assets(dataset_path, indexed)
    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"

    output_requests: list[dict[str, Any]] = []
    audio_count = 0
    with prompts_path.open("w", encoding="utf-8") as f:
        for out_idx, (dataset_index, request) in enumerate(indexed):
            sample_id = str(request.get("id") or f"asr_{dataset_index:06d}")
            answer = _request_answer_from_field(request, answer_field)
            subject_value = str(request.get(subject_field, request.get("subject", "")))
            source_audio = asr_request_audio(dataset_path, request)
            output_audio = work_dir / "audio" / _safe_sample_filename(sample_id, ".wav")
            _convert_audio_to_wav(source_audio, output_audio)
            audio_count += 1
            output_request = dict(request)
            output_request["answer"] = answer
            output_request["subject"] = subject_value
            output_request["audio"] = str(output_audio)
            output_requests.append(output_request)
            sample = {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": subject_value,
                "answer": answer,
                "prompt": asr_request_prompt(request),
                "audio": str(output_audio),
            }
            language = str(
                request.get("language")
                or suite.get("generation", {}).get("language", "")
                or ""
            )
            if language:
                sample["language"] = language
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    answers = _copy_dataset_header(data, output_requests)
    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "audio_count": audio_count,
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": suite.get("generation", {}),
        "files": {
            "answers": str(answers_path),
            "prompts": str(prompts_path),
        },
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def _prepare_vlm_requests(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    data: dict[str, Any],
    indexed: list[tuple[int, dict[str, Any]]],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    validate_vlm_dataset_assets(dataset_path, indexed)
    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"

    image_count = 0
    output_requests: list[dict[str, Any]] = []
    dataset_cfg = suite.get("dataset", {})
    normalization = dataset_cfg.get("normalization", {})
    if not isinstance(normalization, dict):
        normalization = {}
    fixed_image_size = int(normalization.get("image_size", 0) or 0)
    prompt_contract = str(normalization.get("prompt_contract", "native") or "native")
    with prompts_path.open("w", encoding="utf-8") as f:
        for out_idx, (dataset_index, request) in enumerate(indexed):
            images = vlm_request_images(dataset_path, request)
            if len(images) != 1:
                raise ValueError(
                    f"VLM reference-consistency validation currently supports exactly one image per sample; "
                    f"dataset_index={dataset_index} has {len(images)}"
                )
            image_count += len(images)
            sample_id = str(request.get("id") or f"vlm_{dataset_index:06d}")
            prompt = vlm_request_prompt(request)
            output_image = images[0]
            output_request = request
            if prompt_contract == "single_user_image_first":
                if fixed_image_size <= 0:
                    raise ValueError(
                        "single_user_image_first VLM normalization requires dataset.normalization.image_size"
                    )
                output_image = work_dir / "images" / _safe_sample_filename(sample_id)
                _resize_image_to_square(images[0], output_image, fixed_image_size)
                output_request = _normalized_single_user_vlm_request(
                    request,
                    prompt=prompt,
                    image_path=output_image,
                )
            elif prompt_contract != "native":
                raise ValueError(f"Unsupported VLM prompt normalization {prompt_contract!r}")
            output_requests.append(output_request)
            sample = {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": request.get("subject", ""),
                "answer": request["answer"],
                "prompt": prompt,
                "images": [str(output_image)],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    answers = _copy_dataset_header(data, output_requests)
    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(indexed),
        "image_count": image_count,
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "generation": suite.get("generation", {}),
        "files": {
            "answers": str(answers_path),
            "prompts": str(prompts_path),
        },
    }
    if normalization:
        manifest["normalization"] = normalization
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_vlm_chat_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data, indexed = load_vlm_chat_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
    )
    return _prepare_vlm_requests(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def prepare_vlm_unified_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data, indexed = load_vlm_unified_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
    )
    return _prepare_vlm_requests(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        data=data,
        indexed=indexed,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
        validation_config=validation_config,
    )


def prepare_sts_pair_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Prepare STS sentence pairs as byte-shared HF/TRTMC text inputs."""
    indexed: list[tuple[int, dict[str, Any]]] = []
    for dataset_index, row in enumerate(load_jsonl(dataset_path)):
        genre = str(row.get("genre", ""))
        if subject and genre != subject:
            continue
        sentence1 = str(row.get("sentence1", "")).strip()
        sentence2 = str(row.get("sentence2", "")).strip()
        if not sentence1 or not sentence2:
            raise ValueError(
                f"{dataset_path}: STS row {dataset_index} must contain sentence1 and sentence2"
            )
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_path}: STS row {dataset_index} has invalid score"
            ) from exc
        indexed.append((dataset_index, {**row, "score": score}))
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(indexed)
    if limit > 0:
        indexed = indexed[:limit]

    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    requests: list[dict[str, Any]] = []
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, (dataset_index, row) in enumerate(indexed):
            pair_id = f"stsbenchmark_{dataset_index:06d}"
            for pair_side in ("sentence1", "sentence2"):
                sample_id = f"{pair_id}_{'a' if pair_side == 'sentence1' else 'b'}"
                request = {
                    "sample_id": sample_id,
                    "pair_id": pair_id,
                    "pair_side": pair_side,
                    "score": row["score"],
                    "subject": str(row.get("genre", "")),
                    "dataset": str(row.get("dataset", "")),
                    "prompt": str(row[pair_side]),
                }
                requests.append(request)
                prompts_file.write(
                    json.dumps(
                        {
                            **request,
                            "dataset_index": dataset_index,
                            "eval_index": eval_index,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    answers = {
        "dataset": "MTEB STSBenchmark test",
        "scoring": suite.get("scoring", {}),
        "requests": requests,
    }
    answers_path.write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "pair_count": len(indexed),
        "request_count": len(requests),
        "subject": subject or "all",
        "limit": limit,
        "sample_seed": sample_seed,
        "scoring": suite.get("scoring", {}),
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def _time_series_columns(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        columns = [value]
    elif isinstance(value, list):
        columns = [str(column) for column in value]
    else:
        columns = []
    if not columns or any(not column for column in columns):
        raise ValueError(f"time_series.{field} must contain at least one column name")
    return columns


def prepare_time_series_csv_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Prepare model-shaped numeric windows from a shared time-series CSV."""
    if subject:
        raise ValueError("time_series_csv does not support --subject filtering")
    config = validation_config if isinstance(validation_config, dict) else {}
    time_series = config.get("time_series", {})
    if not isinstance(time_series, dict):
        raise ValueError("task_eval.time_series must be a mapping")

    input_columns = _time_series_columns(time_series.get("input_columns"), field="input_columns")
    target_columns = _time_series_columns(
        time_series.get("target_columns", input_columns), field="target_columns"
    )
    input_key = str(time_series.get("input_key", "field_input") or "field_input")
    if input_key not in {"field_input", "branch_input"}:
        raise ValueError("time_series.input_key must be either 'field_input' or 'branch_input'")
    context_length = int(time_series.get("context_length", 0) or 0)
    prediction_length = int(time_series.get("prediction_length", 0) or 0)
    stride = int(time_series.get("stride", prediction_length or 1) or 0)
    dataset_config = suite.get("dataset", {})
    test_fraction = float(time_series.get("test_fraction", 0.2))
    if context_length <= 0 or prediction_length <= 0 or stride <= 0:
        raise ValueError(
            "time_series context_length, prediction_length, and stride must be positive"
        )
    if not 0.0 < test_fraction <= 1.0:
        raise ValueError("time_series.test_fraction must be in (0, 1]")

    with dataset_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        required_columns = set(input_columns) | set(target_columns)
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise ValueError(f"{dataset_path}: missing time-series columns {missing_columns}")
        rows = list(reader)
    minimum_rows = context_length + prediction_length
    if len(rows) < minimum_rows:
        raise ValueError(f"{dataset_path}: needs at least {minimum_rows} rows, found {len(rows)}")

    test_target_start = int(
        time_series.get(
            "test_target_start",
            dataset_config.get("test_target_start", int(len(rows) * (1.0 - test_fraction))),
        )
    )
    test_end = int(time_series.get("test_end", dataset_config.get("test_end", len(rows))))
    if not context_length <= test_target_start < test_end <= len(rows):
        raise ValueError(
            "time_series test_target_start/test_end do not define a valid test window"
        )
    first_start = test_target_start - context_length
    last_start = test_end - context_length - prediction_length
    if last_start < first_start:
        raise ValueError("time_series test window is shorter than prediction_length")
    starts = list(range(first_start, last_start + 1, stride))
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(starts)
    if limit > 0:
        starts = starts[:limit]

    timestamp_column = str(time_series.get("timestamp_column", "date") or "")
    sample_prefix = str(dataset_config.get("sample_prefix", "time_series") or "time_series")
    frequency = time_series.get("frequency")
    work_dir.mkdir(parents=True, exist_ok=True)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"
    requests: list[dict[str, Any]] = []
    with prompts_path.open("w", encoding="utf-8") as prompts_file:
        for eval_index, start in enumerate(starts):
            context_rows = rows[start : start + context_length]
            target_rows = rows[start + context_length : start + context_length + prediction_length]
            try:
                input_values = [
                    float(row[column]) for row in context_rows for column in input_columns
                ]
                target_values = [
                    float(row[column]) for row in target_rows for column in target_columns
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{dataset_path}: non-numeric value in window starting at row {start}"
                ) from exc
            target_index = start + context_length
            sample_id = f"{sample_prefix}_{target_index:06d}"
            inputs: dict[str, Any] = {input_key: input_values}
            if frequency is not None:
                inputs["trunk_input"] = [int(frequency)]
            request = {
                "sample_id": sample_id,
                "dataset_index": target_index,
                "context_index": start,
                "input_columns": input_columns,
                "target_columns": target_columns,
                "target_values": target_values,
            }
            if timestamp_column and timestamp_column in fieldnames:
                request["context_start"] = str(context_rows[0][timestamp_column])
                request["context_end"] = str(context_rows[-1][timestamp_column])
                request["target_start"] = str(target_rows[0][timestamp_column])
                request["target_end"] = str(target_rows[-1][timestamp_column])
            requests.append(request)
            prompts_file.write(
                json.dumps(
                    {
                        **request,
                        "eval_index": eval_index,
                        "inputs": inputs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    answers = {
        "dataset": str(suite.get("dataset", {}).get("name", dataset_path.stem)),
        "scoring": suite.get("scoring", {}),
        "requests": requests,
    }
    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "suite": suite["id"],
        "dataset": str(dataset_path),
        "dataset_kind": suite.get("dataset", {}).get("kind", ""),
        "request_count": len(requests),
        "limit": limit,
        "sample_seed": sample_seed,
        "scoring": suite.get("scoring", {}),
        "time_series": {
            "input_columns": input_columns,
            "target_columns": target_columns,
            "input_key": input_key,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "stride": stride,
            "test_fraction": test_fraction,
            "test_target_start": test_target_start,
            "test_end": test_end,
            **({"frequency": int(frequency)} if frequency is not None else {}),
        },
        "files": {"answers": str(answers_path), "prompts": str(prompts_path)},
    }
    if validation_config:
        manifest["task_eval"] = validation_config
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_task_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    validation_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_kind = suite.get("dataset", {}).get("kind", "")
    if dataset_kind in {"mmlu_five_shot_json", "text_generation_json"}:
        return prepare_mmlu_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "seedtts_json":
        return prepare_seedtts_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "image_classification_json":
        return prepare_image_classification_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "semantic_segmentation_json":
        return prepare_semantic_segmentation_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "prompted_segmentation_json":
        return prepare_prompted_segmentation_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "reranking_json":
        return prepare_reranking_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind in {"vlm_chat_json", "vlm_grounding_json"}:
        return prepare_vlm_chat_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "vlm_unified_json":
        return prepare_vlm_unified_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "asr_chat_json":
        return prepare_asr_chat_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "diffusion_prompt_tsv":
        return prepare_diffusion_prompt_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "diffusion_prompt_json":
        return prepare_diffusion_prompt_json_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "model_plugin_json":
        return prepare_model_plugin_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "sts_pair_jsonl":
        return prepare_sts_pair_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "time_series_csv":
        return prepare_time_series_csv_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "conditional_text_jsonl":
        return prepare_conditional_text_jsonl_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    if dataset_kind == "unconditional_text_json":
        return prepare_unconditional_text_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            validation_config=validation_config,
        )
    raise ValueError(f"Unsupported validation dataset kind {dataset_kind!r}")


def load_jsonl(path: Path, *, errors: str = "strict") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors=errors) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_prompt_tokenizer(
    *,
    model_id: str,
    model_revision: str = "",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Prompt tokenization requires transformers") from exc

    tokenizer_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if model_revision:
        tokenizer_kwargs["revision"] = model_revision
    try:
        return AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    except (KeyError, TypeError, ValueError, OSError) as auto_error:
        try:
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer

            model_path = Path(model_id)
            if not model_path.is_dir():
                snapshot_kwargs: dict[str, Any] = {
                    "local_files_only": local_files_only,
                    "allow_patterns": ["tokenizer.json"],
                }
                if model_revision:
                    snapshot_kwargs["revision"] = model_revision
                model_path = Path(snapshot_download(model_id, **snapshot_kwargs))
            raw_tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        except Exception as fallback_error:
            raise RuntimeError(
                f"Tokenizer for {model_id!r} could not be loaded by AutoTokenizer "
                "or tokenizer.json"
            ) from fallback_error

        class RawTokenizerWrapper:
            def __call__(self, text: str, *, add_special_tokens: bool = False) -> Any:
                encoding = raw_tokenizer.encode(
                    text, add_special_tokens=add_special_tokens
                )

                class Encoded:
                    input_ids = encoding.ids

                return Encoded()

            def decode(self, token_ids: Sequence[int], **_: Any) -> str:
                return raw_tokenizer.decode(list(token_ids), skip_special_tokens=False)

        print(
            f"warning: AutoTokenizer failed for {model_id!r} ({auto_error}); "
            "using tokenizer.json for prompt accounting",
            file=sys.stderr,
        )
        return RawTokenizerWrapper()


def truncate_prompt_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    token_limit: int,
    truncation_side: str = "left",
) -> dict[str, Any]:
    """Truncate shared text prompts before either HF or TRTFB consumes them."""
    if token_limit <= 0:
        raise ValueError(f"prompt token limit must be positive, got {token_limit}")
    if truncation_side not in {"left", "right"}:
        raise ValueError(
            f"prompt truncation side must be 'left' or 'right', got {truncation_side!r}"
        )

    truncated_count = 0
    max_before = 0
    max_after = 0
    for row in rows:
        prompt = str(row.get("prompt", ""))
        token_ids = list(tokenizer(prompt, add_special_tokens=False).input_ids)
        before = len(token_ids)
        max_before = max(max_before, before)
        truncated = before > token_limit
        if truncated:
            token_ids = (
                token_ids[-token_limit:]
                if truncation_side == "left"
                else token_ids[:token_limit]
            )
            row["prompt"] = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            truncated_count += 1
        after = len(token_ids)
        max_after = max(max_after, after)
        row["prompt_tokens_before"] = before
        row["prompt_tokens_after"] = after
        row["prompt_truncated"] = truncated

    return {
        "token_limit": token_limit,
        "truncation_side": truncation_side,
        "prompt_count": len(rows),
        "truncated_count": truncated_count,
        "max_tokens_before": max_before,
        "max_tokens_after": max_after,
    }


def apply_work_prompt_token_limit(
    *,
    work_dir: Path,
    model_id: str,
    model_revision: str = "",
    token_limit: int,
    truncation_side: str = "left",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    tokenizer = _load_prompt_tokenizer(
        model_id=model_id,
        model_revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    prompts_path = work_dir / "prompts.jsonl"
    rows = load_jsonl(prompts_path)
    summary = truncate_prompt_rows(
        rows,
        tokenizer=tokenizer,
        token_limit=token_limit,
        truncation_side=truncation_side,
    )
    with prompts_path.open("w", encoding="utf-8") as prompt_file:
        for row in rows:
            prompt_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = work_dir / "manifest.json"
    manifest = work_manifest(work_dir)
    manifest["prompt_normalization"] = summary
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def max_prompt_token_length(
    *,
    model_id: str,
    model_revision: str = "",
    prompts_path: Path,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> int:
    tokenizer = _load_prompt_tokenizer(
        model_id=model_id,
        model_revision=model_revision,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    max_len = 0
    for row in load_jsonl(prompts_path):
        documents = row.get("documents")
        if isinstance(documents, list) and documents:
            query = str(row.get("query", row.get("prompt", "")))
            prompts = [
                f"question:{query}   passage:{document}" for document in documents
            ]
        else:
            prompts = [str(row.get("prompt", ""))]
        for prompt in prompts:
            length = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            max_len = max(max_len, length)
    return max_len


def validate_prompt_lengths_for_cache(
    *,
    model: dict[str, Any],
    work_dir: Path,
    max_cache_length: int,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> int:
    max_prompt_len = max_prompt_token_length(
        model_id=str(model["hf_id"]),
        model_revision=str(model.get("hf_revision", "") or ""),
        prompts_path=work_dir / "prompts.jsonl",
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code or bool(model.get("trust_remote_code", False)),
    )
    if max_prompt_len > max_cache_length:
        raise RuntimeError(
            f"Dataset prompt length exceeds bundle cache for {model['name']}: "
            f"max_prompt_tokens={max_prompt_len}, build_max_cache_length={max_cache_length}. "
            "Use a smaller dataset slice/subject or set --build-max-cache-length high enough "
            "for this model and TensorRT target."
        )
    return max_prompt_len


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = {"responses": rows}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def reference_cache_metadata(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "hf_cache.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_generated_token_ids(text: str) -> list[int] | None:
    for line in str(text or "").splitlines():
        if not line.strip().startswith("tokens:"):
            continue
        try:
            return [int(token) for token in line.split(":", 1)[1].strip().split()]
        except (ValueError, IndexError):
            return None
    return None


def _run_captured_utf8_subprocess(
    command: Sequence[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        **kwargs,
    )


def _parse_transcribe_stdout(text: str) -> str:
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"<\|[^|]+\|>", "", line).strip()
        if cleaned:
            return cleaned
    return ""


def convert_trtfb_jsonl_to_predictions(raw_path: Path, predictions_path: Path) -> None:
    rows = []
    for row in load_jsonl(raw_path, errors="replace"):
        rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "output_text": row.get("text", ""),
                "generated_tokens": row.get("generated_tokens"),
                "generated_token_ids": _generated_token_ids(row),
                "wall_ms": row.get("wall_ms"),
                "source": "trtfb",
            }
        )
    write_predictions(predictions_path, rows)


def _trtmc_binary_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "trtmc_binary", "") or "build/trtmc")


def _write_dataset_benchmark_reproduction(
    work_dir: Path,
    command: list[str],
) -> None:
    template = list(command)
    template[2] = "{input_jsonl}"
    template[3] = "{trtmc_raw_jsonl}"
    base_seed = None
    if "--seed" in template:
        seed_index = template.index("--seed") + 1
        base_seed = int(template[seed_index])
        template[seed_index] = "{sample_seed}"
    payload = {
        "schema_version": "trtmc.native-trtmc-reproduction/v1",
        "backend": "trtmc_dataset_benchmark",
        "command": template,
    }
    if base_seed is not None:
        payload["base_seed"] = base_seed
    (work_dir / "trtfb_repro.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


_NATIVE_TRTMC_COMMANDS = "trtfb_native_commands.jsonl"


def _reset_native_trtmc_commands(work_dir: Path) -> None:
    (work_dir / _NATIVE_TRTMC_COMMANDS).write_text("", encoding="utf-8")


def _append_native_trtmc_command(
    work_dir: Path,
    sample_id: str,
    command: Sequence[Any],
) -> None:
    tokens = [str(token) for token in command]
    if not tokens:
        return
    with (work_dir / _NATIVE_TRTMC_COMMANDS).open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {"sample_id": sample_id, "command": tokens},
                ensure_ascii=False,
            )
            + "\n"
        )


def _command_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(token, (str, int, float, Path)) for token in value
    ):
        return [str(token) for token in value]
    if isinstance(value, str):
        tokens = shlex.split(value)
        if len(tokens) > 1:
            return tokens
    return []


def _native_command_from_metadata(metadata: Any) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    command = _command_tokens(metadata.get("command"))
    if command:
        return command
    for value in metadata.values():
        nested = _native_command_from_metadata(value)
        if nested:
            return nested
    return []


def _record_output_native_command(
    work_dir: Path,
    sample_id: str,
    output: Any,
) -> None:
    raw_metadata = getattr(output, "metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    command = _native_command_from_metadata(metadata)
    if command:
        _append_native_trtmc_command(work_dir, sample_id, command)


def run_vlm_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (getattr(args, "log", "") or "trtfb_run.log")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 8))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(defaults.get("temperature", 1.0))
    )
    top_k = args.top_k if args.top_k is not None else int(defaults.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(defaults.get("top_p", 1.0))
    min_p = args.min_p if args.min_p is not None else float(defaults.get("min_p", 0.0))
    arg_seed = getattr(args, "seed", None)
    seed = arg_seed if arg_seed is not None else int(defaults.get("seed", -1))

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, Any]] = []
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_output.open("w", encoding="utf-8") as raw_f,
        log_path.open("w", encoding="utf-8") as log_f,
    ):
        for idx, prompt_row in enumerate(prompt_rows):
            images = prompt_row.get("images", [])
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"VLM TRTFB run expects exactly one image for sample {idx}")
            cmd = [
                _trtmc_binary_from_args(args),
                "run",
                args.bundle,
                "--prompt",
                str(prompt_row.get("prompt", "")),
                "--image",
                str(images[0]),
                "--max-new-tokens",
                str(max_new_tokens),
                "--temperature",
                str(temperature),
                "--top-k",
                str(top_k),
                "--top-p",
                str(top_p),
                "--min-p",
                str(min_p),
            ]
            if seed >= 0:
                cmd.extend(["--seed", str(seed + idx)])
            if args.hf_python:
                cmd.extend(["--hf-python", args.hf_python])
            if args.backend_dir:
                cmd.extend(["--backend-dir", args.backend_dir])
            if args.kv_cache_size:
                cmd.extend(["--kv-cache-size", args.kv_cache_size])
            if args.config:
                cmd.extend(["--config", args.config])
            for token in args.set or []:
                cmd.extend(["--set", token])
            if args.chat_template or bool(defaults.get("apply_chat_template", False)):
                cmd.append("--chat-template")
            if not bool(defaults.get("enable_thinking", True)):
                cmd.append("--no-thinking")

            _append_native_trtmc_command(
                work_dir,
                str(prompt_row.get("sample_id", f"vlm_{idx:06d}")),
                cmd,
            )
            log_f.write(f"$ {' '.join(cmd)}\n")
            start = time.perf_counter()
            proc = _run_captured_utf8_subprocess(
                cmd,
                env=env,
            )
            wall_ms = (time.perf_counter() - start) * 1000.0
            if proc.stderr:
                log_f.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log_f.write("\n")
            if proc.returncode != 0:
                if proc.stdout:
                    log_f.write(proc.stdout)
                    if not proc.stdout.endswith("\n"):
                        log_f.write("\n")
                raise RuntimeError(
                    f"VLM TRTFB reference-consistency validation failed for sample {idx} rc={proc.returncode}; see {log_path}"
                )
            output_text = _strip_generated_text_prefix(
                proc.stdout,
                str(prompt_row.get("prompt", "")),
            )
            row = {
                "sample_id": prompt_row.get("sample_id", f"vlm_{idx:06d}"),
                "output_text": output_text,
                "generated_tokens": None,
                "generated_token_ids": None,
                "wall_ms": wall_ms,
                "source": "trtfb",
            }
            rows.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.vlm_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    write_predictions(predictions, rows)


def run_asr_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 100))
    )

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, Any]] = []
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_output.open("w", encoding="utf-8") as raw_f,
        log_path.open("w", encoding="utf-8") as log_f,
    ):
        for idx, prompt_row in enumerate(prompt_rows):
            audio_path = str(prompt_row.get("audio", ""))
            if not audio_path:
                raise ValueError(f"ASR TRTFB run expects an audio path for sample {idx}")
            cmd = [
                _trtmc_binary_from_args(args),
                "transcribe",
                args.bundle,
                "--audio",
                audio_path,
                "--max-new-tokens",
                str(max_new_tokens),
            ]
            if args.hf_python:
                cmd.extend(["--hf-python", args.hf_python])
            cmd.extend(_asr_runtime_flags(prompt_row, defaults))

            _append_native_trtmc_command(
                work_dir,
                str(prompt_row.get("sample_id", f"asr_{idx:06d}")),
                cmd,
            )
            log_f.write(f"$ {' '.join(cmd)}\n")
            start = time.perf_counter()
            proc = _run_captured_utf8_subprocess(
                cmd,
                env=env,
            )
            wall_ms = (time.perf_counter() - start) * 1000.0
            if proc.stdout:
                log_f.write(proc.stdout)
                if not proc.stdout.endswith("\n"):
                    log_f.write("\n")
            if proc.stderr:
                log_f.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log_f.write("\n")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"ASR TRTFB reference-consistency validation failed for sample {idx} rc={proc.returncode}; see {log_path}"
                )
            generated_token_ids = _parse_generated_token_ids(proc.stderr)
            row = {
                "sample_id": prompt_row.get("sample_id", f"asr_{idx:06d}"),
                "output_text": _parse_transcribe_stdout(proc.stdout),
                "generated_tokens": len(generated_token_ids)
                if generated_token_ids is not None
                else None,
                "generated_token_ids": generated_token_ids,
                "wall_ms": wall_ms,
                "source": "trtfb",
            }
            rows.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.asr_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    write_predictions(predictions, rows)


def _asr_runtime_flags(
    prompt_row: dict[str, Any], defaults: dict[str, Any]
) -> list[str]:
    flags: list[str] = []
    language = str(prompt_row.get("language") or defaults.get("language", "") or "")
    if language:
        flags.extend(["--language", language])
    streaming = defaults.get("streaming", {})
    if not isinstance(streaming, dict) or not streaming.get("enabled"):
        return flags
    flags.append("--stream")
    if streaming.get("chunk_ms") is not None:
        flags.extend(["--chunk-ms", str(int(streaming["chunk_ms"]))])
    attention_context = streaming.get("att_context_size")
    if isinstance(attention_context, (list, tuple)) and len(attention_context) == 2:
        flags.extend(
            [
                "--att-context-size",
                f"{int(attention_context[0])},{int(attention_context[1])}",
            ]
        )
    return flags


def clean_text(text: str) -> str:
    text = re.sub(r"(?:<think>)?.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip().strip("().,")


def _strip_markdown(text: str) -> str:
    return text.replace("**", "").replace("__", "").replace("`", "")


def parse_multi_choice_response(text: str) -> str:
    text = _strip_markdown(text.strip())
    match = re.search(r"\banswer\s*(?:is|:)?\s*\(?([A-J])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if len(text) == 1 and text.upper() in CHOICE_LETTERS:
        return text.upper()
    match = re.match(r"^[\(]?([A-J])[\.\):\s]", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return text


def parse_model_prediction(text: str, *, answer_parser: str = "") -> str:
    if not answer_parser:
        return parse_multi_choice_response(clean_text(text))
    if answer_parser != "gpt_oss_harmony_final_mcq":
        raise ValueError(f"Unsupported validation answer parser {answer_parser!r}")

    final_marker = "<|channel|>final<|message|>"
    if final_marker in text:
        text = text.rsplit(final_marker, 1)[1]
    elif "<|channel|>" in text or "<|message|>" in text:
        return ""
    parsed = parse_multi_choice_response(clean_text(text))
    return parsed if parsed in CHOICE_LETTERS else ""


def is_correct(prediction: str, reference: str, *, answer_parser: str = "") -> bool:
    pred_clean = parse_model_prediction(prediction, answer_parser=answer_parser)
    ref_clean = clean_text(reference)
    if ref_clean in CHOICE_LETTERS and not answer_parser:
        pred_clean = parse_multi_choice_response(pred_clean)
    return pred_clean == ref_clean


def request_answer_values(request: dict[str, Any]) -> list[str]:
    values = [str(request["answer"])]
    aliases = request.get("answer_aliases", [])
    if isinstance(aliases, list):
        for alias in aliases:
            alias_text = str(alias)
            if alias_text not in values:
                values.append(alias_text)
    return values


def is_correct_for_request(
    prediction: str, request: dict[str, Any], *, answer_parser: str = ""
) -> bool:
    return any(
        is_correct(prediction, answer, answer_parser=answer_parser)
        for answer in request_answer_values(request)
    )


def normalize_asr_transcript(text: str) -> str:
    # Prompt-conditioned Nemotron ASR can append its detected/selected locale.
    # The model card treats this as metadata and strips it for clean transcript
    # scoring, matching HF decoding with skip_special_tokens=True.
    text = re.sub(r"<[a-z]{2,3}(?:-[a-z]{2})?>", " ", str(text or ""), flags=re.IGNORECASE)
    text = clean_text(str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    return " ".join(text.split())


def _edit_breakdown(ref_items: list[Any], hyp_items: list[Any]) -> dict[str, int]:
    rows = len(ref_items) + 1
    cols = len(hyp_items) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            sub_cost = 0 if ref_items[i - 1] == hyp_items[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    i = len(ref_items)
    j = len(hyp_items)
    matches = substitutions = insertions = deletions = 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref_items[i - 1] == hyp_items[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            matches += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return {
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


def _edit_error_count(breakdown: dict[str, int]) -> int:
    return int(breakdown["substitutions"] + breakdown["insertions"] + breakdown["deletions"])


def _word_error_rate(ref_words: list[str], hyp_words: list[str]) -> tuple[float, dict[str, int]]:
    breakdown = _edit_breakdown(ref_words, hyp_words)
    if not ref_words:
        return (0.0 if not hyp_words else 1.0), breakdown
    return _edit_error_count(breakdown) / len(ref_words), breakdown


def _character_error_rate(ref_text: str, hyp_text: str) -> tuple[float, dict[str, int]]:
    breakdown = _edit_breakdown(list(ref_text), list(hyp_text))
    if not ref_text:
        return (0.0 if not hyp_text else 1.0), breakdown
    return _edit_error_count(breakdown) / len(ref_text), breakdown


def _normalized_edit_distance(prediction: str, reference: str) -> float:
    if not prediction and not reference:
        return 0.0
    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 0.0
    breakdown = _edit_breakdown(list(reference), list(prediction))
    return _edit_error_count(breakdown) / max_len


def score_asr_transcript_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    max_wer: float = 0.1,
    max_cer: float = 0.05,
    max_ned: float = 0.1,
) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            f"Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )

    correct = 0
    exact = 0
    skipped = 0
    subject_stats: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    sample_wers: list[float] = []
    sample_cers: list[float] = []
    sample_neds: list[float] = []
    for idx, (response, request) in enumerate(zip(responses, requests, strict=True)):
        output_text = str(response.get("output_text", ""))
        subject = str(request.get("subject", ""))
        answer = str(request["answer"])
        normalized_answer = normalize_asr_transcript(answer)
        if output_text == ERROR_OUTPUT_TEXT:
            skipped += 1
            samples.append(
                {
                    "index": idx,
                    "sample_id": response.get("sample_id", f"sample_{idx}"),
                    "subject": subject,
                    "answer": answer,
                    "normalized_answer": normalized_answer,
                    "prediction": output_text,
                    "normalized_prediction": "",
                    "skipped": True,
                    "correct": False,
                    "score": 0.0,
                }
            )
            continue

        normalized_prediction = normalize_asr_transcript(output_text)
        ref_words = normalized_answer.split()
        hyp_words = normalized_prediction.split()
        wer, wer_breakdown = _word_error_rate(ref_words, hyp_words)
        cer, cer_breakdown = _character_error_rate(normalized_answer, normalized_prediction)
        ned = _normalized_edit_distance(normalized_prediction, normalized_answer)
        exact_match = normalized_prediction == normalized_answer
        ok = bool(exact_match or (wer <= max_wer and cer <= max_cer and ned <= max_ned))
        correct += int(ok)
        exact += int(exact_match)
        sample_wers.append(wer)
        sample_cers.append(cer)
        sample_neds.append(ned)
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(ok)
        samples.append(
            {
                "index": idx,
                "sample_id": response.get("sample_id", f"sample_{idx}"),
                "subject": subject,
                "answer": answer,
                "normalized_answer": normalized_answer,
                "prediction": output_text,
                "normalized_prediction": normalized_prediction,
                "word_error_rate": wer,
                "character_error_rate": cer,
                "normalized_edit_distance": ned,
                "wer_breakdown": wer_breakdown,
                "cer_breakdown": cer_breakdown,
                "exact_match": exact_match,
                "skipped": False,
                "correct": ok,
                "score": 1.0 if ok else 0.0,
            }
        )

    valid = len(requests) - skipped
    subject_accuracy = {}
    for subject, stats in sorted(subject_stats.items()):
        total = int(stats["total"])
        subject_accuracy[subject] = {
            "accuracy": (int(stats["correct"]) / total) if total else 0.0,
            "correct": int(stats["correct"]),
            "total": total,
        }
    return {
        "overall_accuracy": (correct / valid) if valid else 0.0,
        "exact_match_rate": (exact / valid) if valid else 0.0,
        "word_error_rate": _mean(sample_wers),
        "character_error_rate": _mean(sample_cers),
        "normalized_edit_distance": _mean(sample_neds),
        "correct": correct,
        "valid_count": valid,
        "skipped_count": skipped,
        "total_count": len(requests),
        "subject_accuracy": subject_accuracy,
        "samples": samples,
    }


OCRBENCH_EN_GROUPS = {
    "text_recognition": {
        "text recognition en",
        "fine-grained text recognition en",
        "full-page OCR en",
    },
    "text_detection": {"text grounding en", "VQA with position en"},
    "text_spotting": {"text spotting en"},
    "relationship_extraction": {
        "key information extraction en",
        "key information mapping en",
    },
    "element_parsing": {
        "document parsing en",
        "chart parsing en",
        "table parsing en",
        "formula recognition en",
    },
    "mathematical_calculation": {"math QA en", "text counting en"},
    "visual_text_understanding": {
        "document classification en",
        "cognition VQA en",
        "diagram QA en",
    },
    "knowledge_reasoning": {
        "reasoning VQA en",
        "science QA en",
        "APP agent en",
        "ASCII art classification en",
    },
}

OCRBENCH_CN_GROUPS = {
    "text_recognition": {"full-page OCR cn"},
    "relationship_extraction": {
        "key information extraction cn",
        "handwritten answer extraction cn",
    },
    "element_parsing": {
        "document parsing cn",
        "table parsing cn",
        "formula recognition cn",
    },
    "visual_text_understanding": {"cognition VQA cn"},
    "knowledge_reasoning": {"reasoning VQA cn", "text translation cn"},
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clip_score(score: float) -> float:
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _levenshtein_distance(s1: Any, s2: Any) -> int:
    if not isinstance(s1, str):
        s1 = list(s1)
    if not isinstance(s2, str):
        s2 = list(s2)
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        current = [i2 + 1]
        for i1, c1 in enumerate(s1):
            current.append(
                previous[i1] if c1 == c2 else 1 + min(previous[i1], previous[i1 + 1], current[-1])
            )
        previous = current
    return int(previous[-1])


def _normalized_edit_similarity(prediction: str, reference: str) -> float:
    length = max(len(prediction), len(reference))
    if length == 0:
        return 1.0
    return _clip_score(1.0 - (_levenshtein_distance(prediction, reference) / length))


def _ocrbench_eval_method(request: dict[str, Any]) -> str:
    raw = request.get("ocrbench_eval")
    if raw is None:
        return ""
    text = str(raw)
    return "" if text.lower() == "none" else text


def _ocrbench_answers(request: dict[str, Any]) -> list[str]:
    raw = request.get("ocrbench_answers")
    if isinstance(raw, list):
        values = [str(value) for value in raw]
    elif raw is None:
        values = request_answer_values(request)
    else:
        values = [str(raw)]
    return values


def _ocrbench_task_type(request: dict[str, Any]) -> str:
    return str(request.get("ocrbench_type") or request.get("type") or request.get("subject", ""))


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _ocrbench_tokens(text: str) -> list[str]:
    if _contains_chinese(text):
        return [char for char in text if not char.isspace()]
    return text.split()


def _ocrbench_anls(predict: str, answer: str) -> float:
    value = _normalized_edit_similarity(predict, answer)
    return value if value >= 0.5 else 0.0


def _ocrbench_vqa_score(
    predict: str,
    answers: list[str],
    *,
    case_sensitive: bool = False,
    chinese: bool = False,
) -> float:
    score = 0.0
    for raw_answer in answers:
        answer = str(raw_answer)
        current_predict = str(predict)
        if not case_sensitive:
            answer = answer.lower()
            current_predict = current_predict.lower()
        answer = answer.strip().replace("\n", " ")
        current_predict = current_predict.strip().replace("\n", " ")
        if chinese:
            answer = answer.replace(" ", "")
            current_predict = current_predict.replace(" ", "")
            short_answer = len(answer.split(",")) < 4
        else:
            short_answer = len(answer.split()) < 5
        if short_answer:
            if answer in current_predict:
                score = 1.0
        else:
            score = max(score, _ocrbench_anls(current_predict, answer))
    return _clip_score(score)


def _ocrbench_multiple_choice_score(predict: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    parsed = "".join(char for char in str(predict) if char.isalpha())
    return 1.0 if parsed == str(answers[0]) else 0.0


def _extract_first_number(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _ocrbench_counting_score(predict: str, answers: list[str], eval_method: str) -> float:
    score = 0.0
    predict_text = str(predict).lower().strip().replace("\n", " ")
    for raw_answer in answers:
        answer_text = str(raw_answer).lower().strip().replace("\n", " ")
        if eval_method == "exact match":
            score = max(score, 1.0 if answer_text in predict_text else 0.0)
            continue
        if eval_method != "regression":
            continue
        predict_number = _extract_first_number(predict_text)
        if predict_number is None:
            continue
        try:
            answer_number = int(answer_text)
        except ValueError:
            continue
        if answer_number <= 0 or predict_number <= 0 or predict_number >= 2 * answer_number:
            continue
        value = 1 - abs(predict_number - answer_number) / answer_number
        if value > 0.5:
            score = max(score, value)
    return _clip_score(score)


def _remove_latex_text_tags(latex: str) -> str:
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", latex)


def _ocrbench_formula_score(predict: str, answers: list[str], *, chinese: bool = False) -> float:
    predict_text = str(predict).strip().replace("\n", " ").replace(" ", "")
    for raw_answer in answers:
        answer = str(raw_answer)
        if chinese:
            answer = _remove_latex_text_tags(answer)
        answer = answer.strip().replace("\n", " ").replace(" ", "")
        if answer and answer in predict_text:
            return 1.0
    return 0.0


def _safe_f_measure(reference_tokens: set[str], hypothesis_tokens: set[str]) -> float:
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    overlap = len(reference_tokens & hypothesis_tokens)
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _bleu_score(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    precisions: list[float] = []
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        matched = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        precisions.append(matched / total)
    if any(precision == 0.0 for precision in precisions):
        return 0.0
    brevity = (
        1.0 if len(hypothesis) > len(reference) else math.exp(1 - len(reference) / len(hypothesis))
    )
    return _clip_score(brevity * math.exp(sum(math.log(precision) for precision in precisions) / 4))


def _meteor_like_score(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return 0.0
    ref_counter = Counter(reference)
    hyp_counter = Counter(hypothesis)
    matches = sum(min(count, hyp_counter[token]) for token, count in ref_counter.items())
    if matches == 0:
        return 0.0
    precision = matches / len(hypothesis)
    recall = matches / len(reference)
    return _clip_score((10 * precision * recall) / (recall + 9 * precision))


def _ocrbench_long_reading_score(predict: str, answer: str) -> float:
    prediction = str(predict)
    reference_text = str(answer)
    if not prediction or not reference_text:
        return 0.0
    reference = _ocrbench_tokens(reference_text)
    hypothesis = _ocrbench_tokens(prediction)
    bleu = _bleu_score(reference, hypothesis)
    meteor = _meteor_like_score(reference, hypothesis)
    f_measure = _safe_f_measure(set(reference), set(hypothesis))
    edit_similarity = _normalized_edit_similarity(prediction, reference_text)
    return _clip_score((bleu + meteor + f_measure + edit_similarity) / 4)


def _literal_or_json_dict(text: str) -> dict[str, Any]:
    content = str(text).strip()
    match = re.search(r"```(?:python|json)?\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(content)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    data: dict[str, Any] = {}
    pattern = r'["\']?([\w\s%-]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
    for key, value in re.findall(pattern, content):
        data[key.strip()] = value.strip()
    return data


def _generate_value_combinations(raw_answer: Any) -> list[dict[str, Any]]:
    if isinstance(raw_answer, dict):
        answer_dict = raw_answer
    else:
        answer_dict = _literal_or_json_dict(str(raw_answer).strip('"'))
    if not isinstance(answer_dict, dict) or not answer_dict:
        return []
    keys = list(answer_dict)
    value_lists = [
        value if isinstance(value, list) else [value]
        for value in (answer_dict[key] for key in keys)
    ]
    return [dict(zip(keys, values, strict=True)) for values in product(*value_lists)]


def _ocrbench_kie_f1(preds: dict[str, Any], gts: dict[str, Any]) -> float:
    keys = set(preds) | set(gts)
    if not keys:
        return 0.0
    scores: list[float] = []
    for key in keys:
        pred_value = preds.get(key)
        gt_value = gts.get(key)
        if pred_value is None or gt_value is None:
            scores.append(0.0)
            continue
        pred_text = str(pred_value).lower().strip().replace("\n", " ").replace(" ", "")
        gt_text = str(gt_value).lower().strip().replace("\n", " ").replace(" ", "")
        scores.append(1.0 if pred_text == gt_text else 0.0)
    return _mean(scores)


def _ocrbench_kie_score(predict: str, answers: list[str]) -> float:
    pred_dict = _literal_or_json_dict(predict)
    max_score = 0.0
    for answer in answers[:1]:
        for answer_dict in _generate_value_combinations(answer):
            max_score = max(max_score, _ocrbench_kie_f1(pred_dict, answer_dict))
    return _clip_score(max_score)


def _coerce_bbox(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return [int(coord) for coord in value]
        except Exception:
            return None
    if isinstance(value, str):
        matches = re.findall(r"-?\d+", value)
        if len(matches) >= 4:
            return [int(coord) for coord in matches[:4]]
    return None


def _extract_coordinates(text: str) -> list[int] | None:
    pattern = r"[\(\[]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]"
    coords_list: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for match in re.finditer(pattern, str(text)):
        coords = tuple(int(part) for part in match.groups())
        if not all(0 <= coord <= 1000 for coord in coords):
            continue
        if coords in seen:
            coords_list = [item for item in coords_list if tuple(item) != coords]
        coords_list.append(list(coords))
        seen.add(coords)
    return coords_list[-1] if coords_list else None


def _calculate_iou(box1: Any, box2: Any) -> float:
    lhs = _coerce_bbox(box1)
    rhs = _coerce_bbox(box2)
    if lhs is None or rhs is None:
        return 0.0
    x1_inter = max(lhs[0], rhs[0])
    y1_inter = max(lhs[1], rhs[1])
    x2_inter = min(lhs[2], rhs[2])
    y2_inter = min(lhs[3], rhs[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    lhs_area = max(0, lhs[2] - lhs[0]) * max(0, lhs[3] - lhs[1])
    rhs_area = max(0, rhs[2] - rhs[0]) * max(0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - inter_area
    return _clip_score(inter_area / union) if union else 0.0


def _extract_bounding_boxes_robust(predict: str) -> list[list[Any]]:
    results: list[list[Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    try:
        value = ast.literal_eval(str(predict))
    except Exception:
        value = None
    items = value if isinstance(value, (list, tuple)) else []
    if not items:
        items = re.findall(r"[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]", str(predict))
    for item in items:
        if isinstance(item, str):
            parts = item.split(",", 4)
        elif isinstance(item, (list, tuple)):
            parts = list(item[:5])
        else:
            continue
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2 = [int(str(part).strip()) for part in parts[:4]]
        except ValueError:
            continue
        if not all(0 <= coord <= 1000 for coord in (x1, y1, x2, y2)):
            continue
        text = str(parts[4]).replace("\n", "").strip().strip('"').strip("'")
        key = (x1, y1, x2, y2, text)
        if key in seen:
            continue
        seen.add(key)
        results.append([x1, y1, x2, y2, text])
    return results


def _quadrilateral_to_bbox(points: Any) -> list[int] | None:
    if not isinstance(points, (list, tuple)) or len(points) < 8:
        return _coerce_bbox(points)
    try:
        values = [int(point) for point in points]
    except Exception:
        return None
    xs = values[0::2]
    ys = values[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _ocrbench_spotting_score(predict: str, request: dict[str, Any]) -> tuple[float, str]:
    predictions = _extract_bounding_boxes_robust(predict)
    gt_boxes = request.get("ocrbench_bbox_list") or []
    gt_content = request.get("ocrbench_content") or []
    if not predictions or not isinstance(gt_boxes, list) or not isinstance(gt_content, list):
        return 0.0, ""
    matched_gt: set[int] = set()
    matches = 0
    for pred in predictions:
        pred_box = pred[:4]
        pred_text = str(pred[4])
        best_idx = -1
        best_iou = 0.0
        for idx, (gt_box_raw, gt_text) in enumerate(zip(gt_boxes, gt_content, strict=False)):
            if idx in matched_gt or pred_text != str(gt_text):
                continue
            gt_box = _quadrilateral_to_bbox(gt_box_raw)
            iou = _calculate_iou(pred_box, gt_box)
            if iou >= 0.5 and iou > best_iou:
                best_idx = idx
                best_iou = iou
        if best_idx >= 0:
            matched_gt.add(best_idx)
            matches += 1
    precision = matches / len(predictions) if predictions else 0.0
    recall = matches / len(gt_content) if gt_content else 0.0
    hmean = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return _clip_score(hmean), "builtin_hmean_approximation"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:html|markdown)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _wrap_html_table(table: str) -> str:
    table = table.replace("\n", "")
    if "<table" in table and "</table>" not in table:
        table = f"{table}</table>"
    elif "<table" not in table and "</table>" in table:
        table = f"<table>{table}"
    elif "<table" not in table and "</table>" not in table:
        table = f"<table>{table}</table>"
    if "<body" not in table:
        table = f"<body>{table}</body>"
    if "<html" not in table:
        table = f"<html>{table}</html>"
    return table


def _markdown_table_to_html(markdown_table: str) -> str:
    rows = [row for row in _strip_code_fence(markdown_table).splitlines() if row.strip()]
    if len(rows) >= 2 and set(rows[1].replace("|", "").replace(":", "").strip()) <= {"-", " "}:
        rows = [rows[0], *rows[2:]]
    html_rows = []
    for row in rows:
        cells = [html.escape(cell.strip() or " ") for cell in row.strip().split("|")[1:-1]]
        if cells:
            html_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return "<html><body><table>" + "".join(html_rows) + "</table></body></html>"


def _canonical_structured_text(text: str) -> str:
    text = html.unescape(_strip_code_fence(str(text))).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _structured_similarity(prediction: str, reference: str) -> tuple[float, str]:
    pred_clean = _canonical_structured_text(prediction)
    ref_clean = _canonical_structured_text(reference)
    return _normalized_edit_similarity(pred_clean, ref_clean), "builtin_structural_approximation"


def _dict_to_html(data: Any) -> str:
    if not isinstance(data, dict):
        data = _literal_or_json_dict(str(data))
    rows = []
    if isinstance(data, dict):
        for key, value in data.items():
            rows.append(
                f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
            )
    return "<html><body><table>" + "".join(rows) + "</table></body></html>"


def _ocrbench_table_score(
    predict: str, answers: list[str], question: str, task_type: str
) -> tuple[float, str]:
    if not answers or not isinstance(predict, str) or not predict:
        return 0.0, ""
    prediction = predict.replace("\n", "")
    answer = str(answers[0])
    if task_type == "table parsing cn" or "html" in question.lower():
        if "<body" in prediction:
            prediction = prediction[prediction.find("<body") :]
        elif "<table" in prediction:
            prediction = prediction[prediction.find("<table") :]
        else:
            return 0.0, ""
        return _structured_similarity(_wrap_html_table(prediction), _wrap_html_table(answer))
    if "markdown" in question.lower():
        return _structured_similarity(
            _markdown_table_to_html(predict),
            _markdown_table_to_html(answer),
        )
    return _structured_similarity(predict, answer)


def _ocrbench_chart_score(predict: str, answers: list[str]) -> tuple[float, str]:
    if not answers or not predict:
        return 0.0, ""
    pred_dict = _literal_or_json_dict(predict)
    answer_dict = _literal_or_json_dict(answers[0])
    if not pred_dict:
        return 0.0, ""
    return _structured_similarity(_dict_to_html(pred_dict), _dict_to_html(answer_dict))


def _ocrbench_document_score(predict: str, answers: list[str]) -> tuple[float, str]:
    if not answers or not isinstance(predict, str):
        return 0.0, ""
    return _structured_similarity(predict, answers[0])


def _ocrbench_group(task_type: str) -> tuple[str, str] | None:
    for group, types in OCRBENCH_EN_GROUPS.items():
        if task_type in types:
            return "en", group
    for group, types in OCRBENCH_CN_GROUPS.items():
        if task_type in types:
            return "cn", group
    return None


def _ocrbench_score_sample(predict: str, request: dict[str, Any]) -> tuple[float, str, str]:
    task_type = _ocrbench_task_type(request)
    answers = _ocrbench_answers(request)
    eval_method = _ocrbench_eval_method(request)
    question = str(request.get("question", ""))
    warning = ""

    basic_vqa_en = (
        OCRBENCH_EN_GROUPS["knowledge_reasoning"]
        | OCRBENCH_EN_GROUPS["mathematical_calculation"]
        | OCRBENCH_EN_GROUPS["visual_text_understanding"]
        | {"text recognition en"}
    )
    basic_vqa_en.discard("text counting en")
    basic_vqa_en.discard("formula recognition en")

    if task_type in basic_vqa_en:
        if eval_method == "multiple choice":
            return (
                _ocrbench_multiple_choice_score(predict, answers),
                "multiple_choice_exact",
                warning,
            )
        if eval_method == "case sensitive":
            return (
                _ocrbench_vqa_score(predict, answers, case_sensitive=True),
                "vqa_case_sensitive",
                warning,
            )
        return _ocrbench_vqa_score(predict, answers), "vqa", warning

    if task_type in {"cognition VQA cn", "reasoning VQA cn"}:
        if eval_method == "multiple choice":
            return (
                _ocrbench_multiple_choice_score(predict, answers),
                "multiple_choice_exact",
                warning,
            )
        if eval_method == "case sensitive":
            return (
                _ocrbench_vqa_score(predict, answers, case_sensitive=True),
                "vqa_case_sensitive",
                warning,
            )
        return _ocrbench_vqa_score(predict, answers, chinese=True), "cn_vqa", warning

    if task_type == "handwritten answer extraction cn":
        if "简答" in question:
            return (
                _ocrbench_long_reading_score(predict, answers[0] if answers else ""),
                "long_reading",
                warning,
            )
        if not answers:
            return 0.0, "contains", warning
        answer = answers[0]
        if len(answer) > 1:
            chars = list(answer)
            candidates = [
                "".join(chars),
                ".".join(chars),
                ". ".join(chars),
                ",".join(chars),
                ", ".join(chars),
                "、".join(chars),
                ";".join(chars),
                "; ".join(chars),
                " ".join(chars),
                "和".join(chars),
            ]
            return (
                (1.0 if any(candidate in predict for candidate in candidates) else 0.0),
                "contains",
                warning,
            )
        return (1.0 if answer in predict else 0.0), "contains", warning

    if task_type == "formula recognition cn":
        return _ocrbench_formula_score(predict, answers, chinese=True), "formula_contains", warning
    if task_type == "formula recognition en":
        return _ocrbench_formula_score(predict, answers), "formula_contains", warning
    if task_type == "text counting en":
        return _ocrbench_counting_score(predict, answers, eval_method), "counting", warning

    if task_type in {
        "fine-grained text recognition en",
        "full-page OCR en",
        "full-page OCR cn",
        "text translation cn",
    }:
        return (
            _ocrbench_long_reading_score(predict, answers[0] if answers else ""),
            "long_reading",
            warning,
        )

    if task_type in {"table parsing en", "table parsing cn"}:
        score, warning = _ocrbench_table_score(predict, answers, question, task_type)
        return score, "table_similarity", warning
    if task_type == "chart parsing en":
        score, warning = _ocrbench_chart_score(predict, answers)
        return score, "chart_similarity", warning
    if task_type in {"document parsing en", "document parsing cn"}:
        score, warning = _ocrbench_document_score(predict, answers)
        return score, "document_similarity", warning

    if task_type in {
        "key information extraction en",
        "key information mapping en",
        "key information extraction cn",
    }:
        return _ocrbench_kie_score(predict, answers), "key_value_f1", warning

    if task_type == "VQA with position en":
        pred_dict = _literal_or_json_dict(predict)
        content_score = _ocrbench_vqa_score(str(pred_dict.get("answer", "")), answers)
        bbox_score = _calculate_iou(pred_dict.get("bbox"), request.get("ocrbench_bbox"))
        return _clip_score(0.5 * content_score + 0.5 * bbox_score), "vqa_with_position", warning

    if task_type == "text grounding en":
        pred_bbox = _extract_coordinates(predict)
        gt_bbox = request.get("ocrbench_bbox") or answers
        return _calculate_iou(pred_bbox, gt_bbox), "bbox_iou", warning

    if task_type == "text spotting en":
        score, warning = _ocrbench_spotting_score(predict, request)
        return score, "text_spotting_hmean", warning

    return (
        1.0 if is_correct_for_request(predict, request) else 0.0,
        "exact_or_alias",
        "unknown_ocrbench_type",
    )


def score_ocrbench_v2_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            f"Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )

    samples: list[dict[str, Any]] = []
    subject_stats: dict[str, Counter[str]] = defaultdict(Counter)
    type_scores: dict[str, list[float]] = defaultdict(list)
    capability_scores: dict[str, dict[str, list[float]]] = {
        "en": defaultdict(list),
        "cn": defaultdict(list),
    }
    skipped = 0
    score_sum = 0.0
    perfect = 0
    for idx, (response, request) in enumerate(zip(responses, requests, strict=True)):
        output_text = str(response.get("output_text", ""))
        subject = str(request.get("subject", ""))
        task_type = _ocrbench_task_type(request)
        is_error_output = output_text == ERROR_OUTPUT_TEXT
        if is_error_output:
            skipped += 1
            sample_score, metric, warning = 0.0, "runtime_error", ""
        else:
            sample_score, metric, warning = _ocrbench_score_sample(output_text, request)
        sample_score = _clip_score(sample_score)
        score_sum += sample_score
        perfect += int(sample_score >= 1.0)
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(sample_score >= 1.0)
        subject_stats[subject]["score_sum"] += sample_score
        type_scores[task_type].append(sample_score)
        group = _ocrbench_group(task_type)
        if group:
            language, capability = group
            capability_scores[language][capability].append(sample_score)
        sample = {
            "index": idx,
            "sample_id": response.get("sample_id", f"sample_{idx}"),
            "subject": subject,
            "type": task_type,
            "answer": str(request.get("answer", "")),
            "answer_aliases": request_answer_values(request)[1:],
            "prediction": output_text,
            "skipped": is_error_output,
            "score": sample_score,
            "correct": sample_score >= 1.0,
            "metric": metric,
        }
        if warning:
            sample["metric_warning"] = warning
        samples.append(sample)

    subject_accuracy = {}
    for subject, stats in sorted(subject_stats.items()):
        total = int(stats["total"])
        subject_accuracy[subject] = {
            "accuracy": (float(stats["score_sum"]) / total) if total else 0.0,
            "correct": int(stats["correct"]),
            "score_sum": float(stats["score_sum"]),
            "total": total,
        }

    type_accuracy = {
        task_type: {"accuracy": _mean(scores), "total": len(scores)}
        for task_type, scores in sorted(type_scores.items())
    }
    language_scores: dict[str, dict[str, Any]] = {}
    all_capability_averages: list[float] = []
    for language, groups in capability_scores.items():
        capability_accuracy = {
            group: {"accuracy": _mean(scores), "total": len(scores)}
            for group, scores in sorted(groups.items())
            if scores
        }
        group_averages = [entry["accuracy"] for entry in capability_accuracy.values()]
        all_capability_averages.extend(group_averages)
        language_scores[language] = {
            "capability_accuracy": capability_accuracy,
            "overall_accuracy": _mean(group_averages),
        }

    valid_count = len(requests)
    return {
        "overall_accuracy": _mean(all_capability_averages)
        if all_capability_averages
        else (score_sum / valid_count if valid_count else 0.0),
        "sample_average_accuracy": score_sum / valid_count if valid_count else 0.0,
        "correct": perfect,
        "score_sum": score_sum,
        "valid_count": valid_count,
        "skipped_count": skipped,
        "total_count": len(requests),
        "subject_accuracy": subject_accuracy,
        "ocrbench_v2": {
            "language_scores": language_scores,
            "type_accuracy": type_accuracy,
        },
        "samples": samples,
    }


def _read_wav_metrics(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        if f.read(4) != b"RIFF":
            raise ValueError(f"{path}: not a RIFF WAV file")
        f.read(4)
        if f.read(4) != b"WAVE":
            raise ValueError(f"{path}: not a WAVE file")
        audio_format = 1
        channels = 1
        sample_rate = 0
        bits_per_sample = 16
        data = b""
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", size_bytes)[0]
            payload = f.read(chunk_size)
            if chunk_size % 2:
                f.read(1)
            if chunk_id == b"fmt " and len(payload) >= 16:
                audio_format, channels, sample_rate = struct.unpack("<HHI", payload[:8])
                bits_per_sample = struct.unpack("<H", payload[14:16])[0]
            elif chunk_id == b"data":
                data = payload
    bytes_per_sample = max(bits_per_sample // 8, 1)
    frame_size = bytes_per_sample * max(channels, 1)
    frame_count = len(data) // frame_size
    duration_s = frame_count / sample_rate if sample_rate else 0.0
    if not data or bits_per_sample not in {16, 32}:
        rms = 0.0
    elif audio_format == 3 and bits_per_sample == 32:
        count = len(data) // 4
        samples = struct.unpack(f"<{count}f", data[: count * 4])
        rms = math.sqrt(sum(float(value) ** 2 for value in samples) / count) if count else 0.0
    elif audio_format == 1 and bits_per_sample == 16:
        count = len(data) // 2
        samples = struct.unpack(f"<{count}h", data[: count * 2])
        rms = (
            math.sqrt(sum((float(value) / 32768.0) ** 2 for value in samples) / count)
            if count
            else 0.0
        )
    else:
        rms = 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_s": duration_s,
        "rms": rms,
        "frame_count": frame_count,
    }


def _tts_normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _tts_word_error_rate(prediction: str, reference: str) -> float:
    predicted_words = _tts_normalize_text(prediction).split()
    reference_words = _tts_normalize_text(reference).split()
    if not reference_words:
        return 0.0 if not predicted_words else 1.0
    return _levenshtein_distance(predicted_words, reference_words) / len(reference_words)


def score_tts_intelligibility_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            f"Predictions and answers must have the same length: {len(responses)} != {len(requests)}"
        )
    config = answers_data.get("scoring", {})
    config = config if isinstance(config, dict) else {}
    max_wer = float(config.get("max_wer", 0.25))
    max_ned = float(config.get("max_ned", 0.20))
    min_rms = float(config.get("min_rms", 0.001))
    min_duration_ratio = float(config.get("min_duration_ratio", 0.5))
    max_duration_ratio = float(config.get("max_duration_ratio", 2.0))
    correct = 0
    samples: list[dict[str, Any]] = []
    wers: list[float] = []
    neds: list[float] = []
    for idx, (response, request) in enumerate(zip(responses, requests, strict=True)):
        reference = str(request.get("reference", request.get("answer", "")))
        transcript = str(response.get("output_text", response.get("asr_transcript", "")) or "")
        wav_path = Path(str(response.get("wav_path", "")))
        wav_exists = wav_path.is_file()
        metrics: dict[str, Any] = {}
        error = str(response.get("error", "") or "")
        if wav_exists:
            try:
                metrics = _read_wav_metrics(wav_path)
            except (OSError, ValueError, struct.error) as exc:
                error = str(exc)
        rms = float(response.get("rms", metrics.get("rms", 0.0)) or 0.0)
        duration_s = float(response.get("duration_s", metrics.get("duration_s", 0.0)) or 0.0)
        reference_duration_s = 0.0
        reference_wav = Path(str(request.get("reference_wav", "")))
        if reference_wav.is_file():
            try:
                reference_duration_s = float(_read_wav_metrics(reference_wav)["duration_s"])
            except (OSError, ValueError, struct.error):
                reference_duration_s = 0.0
        duration_ratio = duration_s / reference_duration_s if reference_duration_s > 0 else 0.0
        wer = _tts_word_error_rate(transcript, reference)
        normalized_prediction = _tts_normalize_text(transcript)
        normalized_reference = _tts_normalize_text(reference)
        ned = 1.0 - _normalized_edit_similarity(normalized_prediction, normalized_reference)
        wers.append(wer)
        neds.append(ned)
        ok = (
            not error
            and wav_exists
            and rms >= min_rms
            and min_duration_ratio <= duration_ratio <= max_duration_ratio
            and bool(normalized_prediction)
            and wer <= max_wer
            and ned <= max_ned
        )
        correct += int(ok)
        samples.append(
            {
                "index": idx,
                "sample_id": response.get("sample_id", f"sample_{idx}"),
                "subject": request.get("subject", ""),
                "answer": reference,
                "prediction": transcript,
                "wav_path": str(wav_path) if str(wav_path) else "",
                "wav_exists": wav_exists,
                "rms": rms,
                "duration_s": duration_s,
                "reference_duration_s": reference_duration_s,
                "duration_ratio": duration_ratio,
                "wer": wer,
                "ned": ned,
                "score": 1.0 if ok else 0.0,
                "correct": ok,
                "skipped": False,
                "error": error,
            }
        )
    total = len(requests)
    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct": correct,
        "valid_count": total,
        "skipped_count": 0,
        "total_count": total,
        "mean_wer": _mean(wers),
        "mean_ned": _mean(neds),
        "thresholds": {
            "max_wer": max_wer,
            "max_ned": max_ned,
            "min_rms": min_rms,
            "min_duration_ratio": min_duration_ratio,
            "max_duration_ratio": max_duration_ratio,
        },
        "samples": samples,
    }


def _aligned_generated_texts(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            "Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )
    predictions: list[str] = []
    references: list[str] = []
    samples: list[dict[str, Any]] = []
    for index, (response, request) in enumerate(zip(responses, requests, strict=True)):
        expected_id = str(request.get("sample_id", f"conditional_text_{index:06d}"))
        actual_id = str(response.get("sample_id", expected_id))
        if actual_id != expected_id:
            raise ValueError(
                f"Prediction sample id mismatch at index {index}: {actual_id!r} != {expected_id!r}"
            )
        prediction = str(response.get("output_text", "")).strip()
        reference = str(request.get("answer", "")).strip()
        predictions.append(prediction)
        references.append(reference)
        samples.append(
            {
                "index": index,
                "sample_id": expected_id,
                "subject": str(request.get("subject", "")),
                "source_text": str(request.get("source_text", "")),
                "prediction": prediction,
                "reference": reference,
                "non_empty": bool(prediction),
            }
        )
    return predictions, references, samples


def score_sacrebleu_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
) -> dict[str, Any]:
    try:
        import sacrebleu
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "sacreBLEU scoring requires the validation optional dependency: "
            "pip install 'sacrebleu>=2.4'"
        ) from exc
    predictions, references, samples = _aligned_generated_texts(predictions_data, answers_data)
    non_empty = sum(bool(text) for text in predictions)
    bleu = sacrebleu.corpus_bleu(predictions, [references]) if predictions else None
    corpus_bleu = float(bleu.score) if bleu is not None else 0.0
    return {
        "mode": "sacrebleu",
        "corpus_bleu": corpus_bleu,
        "non_empty_rate": non_empty / len(predictions) if predictions else 0.0,
        "valid_count": len(predictions),
        "skipped_count": 0,
        "bleu": {
            "brevity_penalty": float(bleu.bp) if bleu is not None else 0.0,
            "system_length": int(bleu.sys_len) if bleu is not None else 0,
            "reference_length": int(bleu.ref_len) if bleu is not None else 0,
            "precisions": [float(value) for value in bleu.precisions] if bleu is not None else [],
        },
        "samples": samples,
    }


def _rouge_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def score_rouge_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
) -> dict[str, Any]:
    try:
        from rouge_score import rouge_scorer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "ROUGE scoring requires the validation optional dependency: "
            "pip install 'rouge-score>=0.1.2'"
        ) from exc
    predictions, references, samples = _aligned_generated_texts(predictions_data, answers_data)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_values: list[float] = []
    rouge2_values: list[float] = []
    rouge_l_values: list[float] = []
    for prediction, reference, sample in zip(predictions, references, samples, strict=True):
        scores = scorer.score(reference, prediction)
        sample_scores = {
            "rouge1": float(scores["rouge1"].fmeasure),
            "rouge2": float(scores["rouge2"].fmeasure),
            "rouge_l": float(scores["rougeL"].fmeasure),
        }
        rouge1_values.append(sample_scores["rouge1"])
        rouge2_values.append(sample_scores["rouge2"])
        rouge_l_values.append(sample_scores["rouge_l"])
        sample.update(sample_scores)
    non_empty = sum(bool(text) for text in predictions)
    return {
        "mode": "rouge",
        "rouge1": _mean(rouge1_values),
        "rouge2": _mean(rouge2_values),
        "rouge_l": _mean(rouge_l_values),
        "non_empty_rate": non_empty / len(predictions) if predictions else 0.0,
        "valid_count": len(predictions),
        "skipped_count": 0,
        "samples": samples,
    }


def _generated_token_sequence(response: dict[str, Any], text: str) -> list[Any]:
    token_ids = response.get("generated_token_ids")
    if isinstance(token_ids, list) and token_ids:
        return [int(token_id) for token_id in token_ids]
    return _rouge_tokens(text)


def score_unconditional_text_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    generation_ppl: float | None = None,
    unigram_entropy: float | None = None,
) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            "Predictions and requests must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )
    corpus_tokens: list[Any] = []
    bigrams: list[tuple[Any, Any]] = []
    samples: list[dict[str, Any]] = []
    non_empty = 0
    repeated_tokens = 0
    total_tokens = 0
    for index, (response, request) in enumerate(zip(responses, requests, strict=True)):
        expected_id = str(request.get("sample_id", f"unconditional_{index:06d}"))
        actual_id = str(response.get("sample_id", expected_id))
        if actual_id != expected_id:
            raise ValueError(
                f"Prediction sample id mismatch at index {index}: {actual_id!r} != {expected_id!r}"
            )
        text = str(response.get("output_text", "")).strip()
        tokens = _generated_token_sequence(response, text)
        non_empty += int(bool(text))
        total_tokens += len(tokens)
        repeated_tokens += len(tokens) - len(set(tokens))
        corpus_tokens.extend(tokens)
        bigrams.extend(zip(tokens, tokens[1:]))
        samples.append(
            {
                "index": index,
                "sample_id": expected_id,
                "prediction": text,
                "token_count": len(tokens),
                "non_empty": bool(text),
            }
        )
    counts = Counter(corpus_tokens)
    fallback_entropy = 0.0
    if corpus_tokens:
        for count in counts.values():
            probability = count / len(corpus_tokens)
            fallback_entropy -= probability * math.log(probability)
    result = {
        "mode": "unconditional_text_quality",
        "generation_ppl": generation_ppl,
        "unigram_entropy": fallback_entropy if unigram_entropy is None else unigram_entropy,
        "distinct_1": len(counts) / len(corpus_tokens) if corpus_tokens else 0.0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
        "token_repetition_rate": repeated_tokens / total_tokens if total_tokens else 1.0,
        "mean_token_count": total_tokens / len(responses) if responses else 0.0,
        "non_empty_rate": non_empty / len(responses) if responses else 0.0,
        "valid_count": len(responses),
        "skipped_count": 0,
        "samples": samples,
    }
    return result


def _diffusion_text_sample_token_ids(sample: Any) -> list[int]:
    if not isinstance(sample, dict):
        return []
    value = sample.get("token_ids", sample.get("generated_token_ids", []))
    if not isinstance(value, list):
        return []
    return [int(token_id) for token_id in value]


def _first_token_divergence(reference: list[int], actual: list[int]) -> int:
    for index, (reference_id, actual_id) in enumerate(zip(reference, actual, strict=False)):
        if reference_id != actual_id:
            return index
    return min(len(reference), len(actual))


def _diffusion_text_shared_sampling_inputs(row: dict[str, Any]) -> dict[str, str]:
    explicit = row.get("shared_sampling_inputs", {})
    if isinstance(explicit, dict) and explicit:
        return {
            str(key): str(Path(str(value)).resolve())
            for key, value in explicit.items()
        }
    shared_dir = str(row.get("shared_inputs_dir", "") or "")
    if not shared_dir:
        return {}
    root = Path(shared_dir)
    candidates = {
        "initial_latents": root / "initial_latents.f32",
        "sampling_steps": root / "sampling_steps.f32",
        "sde_noises": root / "sde_noises.f32",
    }
    return {key: str(path) for key, path in candidates.items() if path.is_file()}


def compare_diffusion_text_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
) -> dict[str, Any]:
    hf_rows = hf_data.get("responses", [])
    trtfb_rows = trtfb_data.get("responses", [])
    if len(hf_rows) != len(trtfb_rows):
        raise ValueError(
            f"ELF HF/TRTMC prediction count mismatch: hf={len(hf_rows)} trtfb={len(trtfb_rows)}"
        )
    token_matches = 0
    token_positions = 0
    shared_input_matches = 0
    samples: list[dict[str, Any]] = []
    for index, (hf_row, trtfb_row) in enumerate(zip(hf_rows, trtfb_rows, strict=True)):
        hf_id = str(hf_row.get("sample_id", hf_row.get("id", index)))
        trtfb_id = str(trtfb_row.get("sample_id", trtfb_row.get("id", index)))
        if hf_id != trtfb_id:
            raise ValueError(
                f"ELF HF/TRTMC sample id mismatch at index {index}: hf={hf_id!r} trtfb={trtfb_id!r}"
            )
        for backend, row in (("HF", hf_row), ("TRTMC", trtfb_row)):
            if not isinstance(row.get("token_ids", row.get("generated_token_ids")), list):
                raise ValueError(
                    f"ELF {backend} prediction {hf_id!r} must contain token_ids or "
                    "generated_token_ids for token agreement"
                )
        hf_tokens = _diffusion_text_sample_token_ids(hf_row)
        trtfb_tokens = _diffusion_text_sample_token_ids(trtfb_row)
        positions = max(len(hf_tokens), len(trtfb_tokens))
        matches = sum(left == right for left, right in zip(hf_tokens, trtfb_tokens, strict=False))
        hf_shared_inputs = _diffusion_text_shared_sampling_inputs(hf_row)
        trtfb_shared_inputs = _diffusion_text_shared_sampling_inputs(trtfb_row)
        shared_inputs_match = bool(hf_shared_inputs) and hf_shared_inputs == trtfb_shared_inputs
        token_matches += matches
        token_positions += positions
        shared_input_matches += int(shared_inputs_match)
        samples.append(
            {
                "sample_id": hf_id,
                "token_agreement_rate": matches / positions if positions else 0.0,
                "first_token_divergence": _first_token_divergence(hf_tokens, trtfb_tokens),
                "shared_sampling_inputs_match": shared_inputs_match,
            }
        )
    count = len(hf_rows)
    return {
        "valid_count": count,
        "token_agreement_rate": token_matches / token_positions if token_positions else 0.0,
        "shared_sampling_inputs_match_rate": shared_input_matches / count if count else 0.0,
        "samples": samples,
    }


def diffusion_text_task_metric_deltas(
    task_metric: str,
    diagnostics: dict[str, float],
) -> dict[str, float]:
    """Return absolute HF/TRTMC deltas for the task-level ELF metrics."""
    metric_pairs: dict[str, tuple[tuple[str, str, str], ...]] = {
        "sacrebleu": (("hf_corpus_bleu", "trtfb_corpus_bleu", "corpus_bleu_abs_delta"),),
        "rouge": (
            ("hf_rouge1", "trtfb_rouge1", "rouge1_abs_delta"),
            ("hf_rouge2", "trtfb_rouge2", "rouge2_abs_delta"),
            ("hf_rouge_l", "trtfb_rouge_l", "rouge_l_abs_delta"),
        ),
        "unconditional_text_quality": (
            ("hf_generation_ppl", "trtfb_generation_ppl", "generation_ppl_abs_delta"),
            ("hf_unigram_entropy", "trtfb_unigram_entropy", "unigram_entropy_abs_delta"),
        ),
    }
    pairs = metric_pairs.get(task_metric)
    if pairs is None:
        raise ValueError(f"Unsupported diffusion-text task metric {task_metric!r}")
    return {
        output_key: abs(float(diagnostics[hf_key]) - float(diagnostics[trtfb_key]))
        for hf_key, trtfb_key, output_key in pairs
    }


def compute_gpt2_generation_metrics(
    texts: list[str],
    *,
    model_id: str,
    device: str = "cuda",
    local_files_only: bool = False,
) -> dict[str, float]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("GPT-2 generation perplexity requires torch and transformers") from exc
    resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if resolved_device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()
    max_length = int(
        getattr(model.config, "n_positions", 0)
        or getattr(model.config, "max_position_embeddings", 1024)
        or 1024
    )
    total_nll = 0.0
    total_tokens = 0
    sample_entropies: list[float] = []
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )["input_ids"].to(resolved_device)
            token_counts = Counter(int(token_id) for token_id in encoded[0].tolist())
            token_total = sum(token_counts.values())
            if token_total:
                sample_entropies.append(
                    -sum(
                        (count / token_total) * math.log(count / token_total + 1e-10)
                        for count in token_counts.values()
                    )
                )
            predicted_tokens = max(0, int(encoded.shape[1]) - 1)
            if predicted_tokens == 0:
                continue
            output = model(input_ids=encoded, labels=encoded)
            total_nll += float(output.loss.detach().float().cpu()) * predicted_tokens
            total_tokens += predicted_tokens
    del model
    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()
    if total_tokens == 0:
        generation_ppl = math.inf
    else:
        generation_ppl = math.exp(total_nll / total_tokens)
    return {
        "generation_ppl": generation_ppl,
        "unigram_entropy": _mean(sample_entropies),
    }


def apply_metric_gates(result: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for gate, required_raw in gates.items():
        if gate.startswith("min_"):
            metric = gate[len("min_") :]
            operator = ">="
        elif gate.startswith("max_"):
            metric = gate[len("max_") :]
            operator = "<="
        else:
            continue
        actual_raw = result.get(metric)
        required = float(required_raw)
        if actual_raw is None:
            failures.append(
                {
                    "gate": gate,
                    "metric": metric,
                    "actual": None,
                    "required": required,
                    "reason": "metric unavailable",
                }
            )
            continue
        actual = float(actual_raw)
        passed = actual >= required if operator == ">=" else actual <= required
        if not passed:
            failures.append(
                {
                    "gate": gate,
                    "metric": metric,
                    "actual": actual,
                    "required": required,
                }
            )
    result["gate_failures"] = failures
    result["status"] = "failed" if failures else "passed"
    if failures:
        result["error_type"] = "BenchmarkGateError"
        result["error"] = "; ".join(
            f"{failure['gate']} actual={failure['actual']} required={failure['required']}"
            for failure in failures
        )
    return result


def continuation_task_quality_diagnostics(
    task_metric: str,
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    if task_metric != "sacrebleu":
        return {}
    hf_quality = score_sacrebleu_predictions(hf_predictions, answers)
    trtfb_quality = score_sacrebleu_predictions(trtfb_predictions, answers)
    return {
        "hf": hf_quality,
        "trtfb": trtfb_quality,
        "hf_corpus_bleu": hf_quality["corpus_bleu"],
        "trtfb_corpus_bleu": trtfb_quality["corpus_bleu"],
        "corpus_bleu_abs_delta": abs(
            hf_quality["corpus_bleu"] - trtfb_quality["corpus_bleu"]
        ),
    }


def _grounding_float_list(value: Any, *, count: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        output = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if count is not None and len(output) != count:
        return None
    return output


def _normalize_grounding_xyxy_1000(
    values: list[float],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    pixel_coordinates: bool = False,
) -> list[float] | None:
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if pixel_coordinates:
        if not image_width or not image_height:
            return None
        x1, x2 = x1 * 1000.0 / image_width, x2 * 1000.0 / image_width
        y1, y2 = y1 * 1000.0 / image_height, y2 * 1000.0 / image_height
    elif max(abs(value) for value in values) <= 1.0:
        x1, y1, x2, y2 = (value * 1000.0 for value in values)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    box = [
        max(0.0, min(1000.0, left)),
        max(0.0, min(1000.0, top)),
        max(0.0, min(1000.0, right)),
        max(0.0, min(1000.0, bottom)),
    ]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _grounding_request_metadata(request: dict[str, Any]) -> dict[str, Any]:
    metadata = request.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _grounding_request_image_size(request: dict[str, Any]) -> tuple[int | None, int | None]:
    metadata = _grounding_request_metadata(request)
    width = metadata.get("image_width", request.get("image_width"))
    height = metadata.get("image_height", request.get("image_height"))
    try:
        return int(width) if width else None, int(height) if height else None
    except (TypeError, ValueError):
        return None, None


def parse_grounding_bbox(
    text: str,
    *,
    request: dict[str, Any] | None = None,
) -> list[float] | None:
    structured = re.search(
        r"<box>\s*<([0-9]{1,4})>\s*<([0-9]{1,4})>\s*"
        r"<([0-9]{1,4})>\s*<([0-9]{1,4})>\s*</box>",
        str(text),
    )
    if structured:
        return _normalize_grounding_xyxy_1000(
            [float(value) for value in structured.groups()]
        )
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    values = [float(match) for match in re.findall(number, clean_text(str(text)))]
    if len(values) < 4:
        return None
    request = request or {}
    image_width, image_height = _grounding_request_image_size(request)
    return _normalize_grounding_xyxy_1000(
        values[:4],
        image_width=image_width,
        image_height=image_height,
        pixel_coordinates=bool(re.search(r"\b(?:px|pixel|pixels)\b", text, re.IGNORECASE)),
    )


def reference_grounding_bbox(request: dict[str, Any]) -> list[float] | None:
    metadata = _grounding_request_metadata(request)
    for key in ("bbox_1000", "bbox_1000_xyxy", "target_bbox_1000"):
        box = _grounding_float_list(metadata.get(key, request.get(key)), count=4)
        if box is not None:
            return _normalize_grounding_xyxy_1000(box)
    for key in ("bbox", "bbox_xyxy", "target_bbox"):
        box = _grounding_float_list(metadata.get(key, request.get(key)), count=4)
        if box is not None:
            return _normalize_grounding_xyxy_1000(box)
    answer = request.get("answer")
    if isinstance(answer, list):
        box = _grounding_float_list(answer, count=4)
        return _normalize_grounding_xyxy_1000(box) if box is not None else None
    return parse_grounding_bbox(str(answer), request=request) if answer is not None else None


def grounding_bbox_iou(
    left: list[float] | None,
    right: list[float] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return _clip_score(intersection / union) if union > 0.0 else 0.0


def score_grounding_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            "Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("grounding IoU threshold must be between 0 and 1")

    correct = 0
    skipped = 0
    subject_stats: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for index, (response, request) in enumerate(zip(responses, requests, strict=True)):
        output_text = str(response.get("output_text", ""))
        subject = str(request.get("subject", ""))
        target = reference_grounding_bbox(request)
        prediction = parse_grounding_bbox(output_text, request=request)
        is_skipped = output_text == ERROR_OUTPUT_TEXT
        iou = 0.0 if is_skipped else grounding_bbox_iou(prediction, target)
        is_correct = not is_skipped and iou >= iou_threshold
        skipped += int(is_skipped)
        correct += int(is_correct)
        if not is_skipped:
            subject_stats[subject]["total"] += 1
            subject_stats[subject]["correct"] += int(is_correct)
        samples.append(
            {
                "index": index,
                "sample_id": response.get("sample_id", f"sample_{index}"),
                "subject": subject,
                "answer": str(request.get("answer", "")),
                "target_bbox": target,
                "prediction": output_text,
                "parsed_prediction": prediction,
                "iou": iou,
                "iou_threshold": iou_threshold,
                "score": iou,
                "skipped": is_skipped,
                "correct": is_correct,
            }
        )
    valid = len(requests) - skipped
    subject_accuracy = {
        subject: {
            "accuracy": int(stats["correct"]) / int(stats["total"]),
            "correct": int(stats["correct"]),
            "total": int(stats["total"]),
        }
        for subject, stats in sorted(subject_stats.items())
        if int(stats["total"])
    }
    return {
        "overall_accuracy": correct / valid if valid else 0.0,
        "correct": correct,
        "valid_count": valid,
        "skipped_count": skipped,
        "total_count": len(requests),
        "iou_threshold": iou_threshold,
        "subject_accuracy": subject_accuracy,
        "samples": samples,
    }


def score_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    scorer: str = "exact_or_alias",
    answer_parser: str = "",
    require_valid_prediction: bool = False,
    scorer_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scorer == "ocrbench_v2":
        return score_ocrbench_v2_predictions(predictions_data, answers_data)
    if scorer == "asr_transcript":
        return score_asr_transcript_predictions(predictions_data, answers_data)
    if scorer == "tts_intelligibility":
        return score_tts_intelligibility_predictions(predictions_data, answers_data)
    if scorer == "sacrebleu":
        return score_sacrebleu_predictions(predictions_data, answers_data)
    if scorer == "rouge":
        return score_rouge_predictions(predictions_data, answers_data)
    if scorer == "unconditional_text_quality":
        return score_unconditional_text_predictions(predictions_data, answers_data)
    if scorer == "grounding_iou":
        scorer_options = scorer_options or {}
        return score_grounding_predictions(
            predictions_data,
            answers_data,
            iou_threshold=float(scorer_options.get("iou_threshold", 0.5)),
        )

    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            f"Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )

    correct = 0
    skipped = 0
    valid_predictions = 0
    subject_stats: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for idx, (response, request) in enumerate(zip(responses, requests, strict=True)):
        output_text = str(response.get("output_text", ""))
        subject = str(request.get("subject", ""))
        answer = str(request["answer"])
        answer_values = request_answer_values(request)
        if output_text == ERROR_OUTPUT_TEXT:
            skipped += 1
            samples.append(
                {
                    "index": idx,
                    "sample_id": response.get("sample_id", f"sample_{idx}"),
                    "subject": subject,
                    "answer": answer,
                    "answer_aliases": answer_values[1:],
                    "prediction": output_text,
                    "skipped": True,
                    "correct": False,
                }
            )
            continue
        parsed_prediction = parse_model_prediction(output_text, answer_parser=answer_parser)
        prediction_valid = bool(parsed_prediction)
        if require_valid_prediction and clean_text(answer) in CHOICE_LETTERS:
            prediction_valid = parsed_prediction in CHOICE_LETTERS
        valid_predictions += int(prediction_valid)
        ok = is_correct_for_request(output_text, request, answer_parser=answer_parser)
        correct += int(ok)
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(ok)
        samples.append(
            {
                "index": idx,
                "sample_id": response.get("sample_id", f"sample_{idx}"),
                "subject": subject,
                "answer": answer,
                "answer_aliases": answer_values[1:],
                "prediction": output_text,
                "parsed_prediction": parsed_prediction,
                "valid_prediction": prediction_valid,
                "skipped": False,
                "correct": ok,
            }
        )

    valid = len(requests) - skipped
    subject_accuracy = {}
    for subject, stats in sorted(subject_stats.items()):
        total = int(stats["total"])
        subject_accuracy[subject] = {
            "accuracy": (int(stats["correct"]) / total) if total else 0.0,
            "correct": int(stats["correct"]),
            "total": total,
        }
    return {
        "overall_accuracy": (correct / valid) if valid else 0.0,
        "correct": correct,
        "valid_count": valid,
        "valid_prediction_count": valid_predictions,
        "invalid_prediction_count": valid - valid_predictions,
        "valid_prediction_rate": (valid_predictions / valid) if valid else 0.0,
        "skipped_count": skipped,
        "total_count": len(requests),
        "subject_accuracy": subject_accuracy,
        "samples": samples,
    }


def compare_prediction_sets(
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    answers: dict[str, Any],
    *,
    scorer: str = "exact_or_alias",
    answer_parser: str = "",
    require_valid_prediction: bool = False,
    scorer_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hf_score = score_predictions(
        hf_predictions,
        answers,
        scorer=scorer,
        answer_parser=answer_parser,
        require_valid_prediction=require_valid_prediction,
        scorer_options=scorer_options,
    )
    trtfb_score = score_predictions(
        trtfb_predictions,
        answers,
        scorer=scorer,
        answer_parser=answer_parser,
        require_valid_prediction=require_valid_prediction,
        scorer_options=scorer_options,
    )
    hf_responses = hf_predictions["responses"]
    trt_responses = trtfb_predictions["responses"]
    requests = answers["requests"]
    if len(hf_responses) != len(trt_responses):
        raise ValueError("HF and TRTFB predictions must have the same length")

    agreement_score = 0.0
    correctness_agreement = 0
    buckets = Counter()
    disagreements: list[dict[str, Any]] = []
    asr_parity_samples: list[dict[str, Any]] = []
    reference_tie_equivalent_samples: list[dict[str, Any]] = []
    tie_accuracy_adjustment = 0
    for idx, (hf_row, trt_row, req) in enumerate(
        zip(hf_responses, trt_responses, requests, strict=True)
    ):
        answer = str(req["answer"])
        if scorer == "grounding_iou":
            hf_pred = hf_score["samples"][idx].get("parsed_prediction")
            trt_pred = trtfb_score["samples"][idx].get("parsed_prediction")
        elif scorer == "asr_transcript":
            hf_pred = normalize_asr_transcript(str(hf_row.get("output_text", "")))
            trt_pred = normalize_asr_transcript(str(trt_row.get("output_text", "")))
        elif scorer == "tts_intelligibility":
            hf_pred = _tts_normalize_text(str(hf_row.get("output_text", "")))
            trt_pred = _tts_normalize_text(str(trt_row.get("output_text", "")))
        else:
            hf_pred = parse_model_prediction(
                str(hf_row.get("output_text", "")), answer_parser=answer_parser
            )
            trt_pred = parse_model_prediction(
                str(trt_row.get("output_text", "")), answer_parser=answer_parser
            )
        hf_sample = hf_score["samples"][idx]
        trtfb_sample = trtfb_score["samples"][idx]
        hf_ok = bool(hf_sample.get("correct", False))
        trt_ok = bool(trtfb_sample.get("correct", False))
        correctness_match = hf_ok == trt_ok
        agreement_match = (
            correctness_match
            if scorer in {"grounding_iou", "ocrbench_v2", "tts_intelligibility"}
            else hf_pred == trt_pred
        )
        if scorer == "mcq" and not agreement_match:
            max_score_steps = hf_row.get("generated_token_max_score_ids", [])
            hf_token_ids = hf_row.get("generated_token_ids", [])
            trt_token_ids = trt_row.get("generated_token_ids", [])
            if (
                isinstance(max_score_steps, list)
                and len(max_score_steps) == 1
                and isinstance(max_score_steps[0], list)
                and len(max_score_steps[0]) > 1
                and isinstance(hf_token_ids, list)
                and len(hf_token_ids) == 1
                and int(hf_token_ids[0]) in max_score_steps[0]
                and isinstance(trt_token_ids, list)
                and len(trt_token_ids) == 1
                and int(trt_token_ids[0]) in max_score_steps[0]
            ):
                agreement_match = True
                tie_accuracy_adjustment += int(hf_ok) - int(trt_ok)
                reference_tie_equivalent_samples.append(
                    {
                        "index": idx,
                        "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                        "hf_prediction": hf_pred,
                        "trtfb_prediction": trt_pred,
                        "max_score_token_ids": max_score_steps[0],
                    }
                )
        prediction_score = float(agreement_match)
        if scorer == "asr_transcript":
            transcript_similarity = max(
                0.0, 1.0 - _normalized_edit_distance(hf_pred, trt_pred)
            )
            prediction_score = transcript_similarity
            asr_parity_samples.append(
                {
                    "index": idx,
                    "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                    "hf_prediction": hf_pred,
                    "trtfb_prediction": trt_pred,
                    "transcript_exact": agreement_match,
                    "transcript_similarity": transcript_similarity,
                    "correctness_agreement": correctness_match,
                }
            )
        if require_valid_prediction and (
            not bool(hf_sample.get("valid_prediction", False))
            or not bool(trtfb_sample.get("valid_prediction", False))
        ):
            agreement_match = False
            prediction_score = 0.0
        agreement_score += prediction_score
        correctness_agreement += int(correctness_match)
        if hf_ok and trt_ok:
            buckets["both_correct"] += 1
        elif hf_ok and not trt_ok:
            buckets["hf_correct_trtfb_wrong"] += 1
        elif not hf_ok and trt_ok:
            buckets["hf_wrong_trtfb_correct"] += 1
        else:
            buckets["both_wrong"] += 1
        if not agreement_match:
            disagreements.append(
                {
                    "index": idx,
                    "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                    "subject": req.get("subject", ""),
                    "answer": answer,
                    "hf_prediction": hf_pred,
                    "trtfb_prediction": trt_pred,
                    "hf_correct": hf_ok,
                    "trtfb_correct": trt_ok,
                    "hf_score": hf_sample.get("score", int(hf_ok)),
                    "trtfb_score": trtfb_sample.get("score", int(trt_ok)),
                }
            )

    total = len(requests)
    raw_accuracy_delta = (
        trtfb_score["overall_accuracy"] - hf_score["overall_accuracy"]
    )
    tie_adjusted_accuracy_delta = raw_accuracy_delta + (
        tie_accuracy_adjustment / total if total else 0.0
    )
    if abs(tie_adjusted_accuracy_delta) < 1e-12:
        tie_adjusted_accuracy_delta = 0.0
    summary = {
        "hf": hf_score,
        "trtfb": trtfb_score,
        "accuracy_delta_trtfb_minus_hf": raw_accuracy_delta,
        "tie_adjusted_accuracy_delta_trtfb_minus_hf": tie_adjusted_accuracy_delta,
        "prediction_agreement_rate": (agreement_score / total) if total else 0.0,
        "agreement_count": correctness_agreement,
        "correctness_agreement_rate": (
            correctness_agreement / total if total else 0.0
        ),
        "total_count": total,
        "buckets": dict(buckets),
        "disagreements": disagreements,
        "reference_tie_equivalent_count": len(reference_tie_equivalent_samples),
        "reference_tie_equivalent_samples": reference_tie_equivalent_samples,
    }
    if scorer == "asr_transcript":
        exact_count = sum(
            bool(sample["transcript_exact"]) for sample in asr_parity_samples
        )
        summary["normalized_transcript_exact_agreement_rate"] = (
            exact_count / total if total else 0.0
        )
        summary["asr_parity_samples"] = asr_parity_samples
    return summary


def prediction_agreement_gate_result(
    summary: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate task-accuracy and direct prediction-agreement gates."""
    max_accuracy_drop = float(
        gates.get("max_accuracy_drop_from_hf", 0.05)
    )
    min_agreement = float(
        gates.get("min_prediction_agreement", 0.90)
    )
    raw_accuracy_drop = (
        float(summary["hf"]["overall_accuracy"])
        - float(summary["trtfb"]["overall_accuracy"])
    )
    accuracy_drop = -float(
        summary.get(
            "tie_adjusted_accuracy_delta_trtfb_minus_hf",
            -raw_accuracy_drop,
        )
    )
    if abs(accuracy_drop) < 1e-12:
        accuracy_drop = 0.0
    prediction_agreement = float(summary["prediction_agreement_rate"])
    failures: list[dict[str, Any]] = []
    if accuracy_drop > max_accuracy_drop:
        failures.append(
            {
                "gate": "max_accuracy_drop_from_hf",
                "metric": "accuracy_drop_from_hf",
                "actual": accuracy_drop,
                "required": max_accuracy_drop,
            }
        )
    if prediction_agreement < min_agreement:
        failures.append(
            {
                "gate": "min_prediction_agreement",
                "metric": "prediction_agreement_rate",
                "actual": prediction_agreement,
                "required": min_agreement,
            }
        )
    result: dict[str, Any] = {
        "status": "failed" if failures else "passed",
        "accuracy_drop_from_hf": accuracy_drop,
        "raw_accuracy_drop_from_hf": raw_accuracy_drop,
        "gates": {
            "max_accuracy_drop_from_hf": max_accuracy_drop,
            "min_prediction_agreement": min_agreement,
        },
        "gate_failures": failures,
    }
    if failures:
        result["error_type"] = "BenchmarkGateError"
        result["error"] = "; ".join(
            f"{failure['gate']} actual={failure['actual']} "
            f"required={failure['required']}"
            for failure in failures
        )
    return result


def tts_intelligibility_gate_result(
    summary: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    max_pass_rate_drop = float(
        gates.get("max_pass_rate_drop_from_hf", 0.05)
    )
    min_correctness_agreement = float(
        gates.get("min_correctness_agreement", 0.95)
    )
    pass_rate_drop = (
        float(summary["hf"]["overall_accuracy"])
        - float(summary["trtfb"]["overall_accuracy"])
    )
    correctness_agreement = float(summary["correctness_agreement_rate"])
    return {
        "status": (
            "passed"
            if pass_rate_drop <= max_pass_rate_drop
            and correctness_agreement >= min_correctness_agreement
            else "failed"
        ),
        "pass_rate_drop_from_hf": pass_rate_drop,
        "correctness_agreement_rate": correctness_agreement,
        "gates": {
            "max_pass_rate_drop_from_hf": max_pass_rate_drop,
            "min_correctness_agreement": min_correctness_agreement,
        },
    }


def _load_diffusion_validation_comparator(work_dir: Path) -> Any:
    case, _reference, _runner = _load_diffusion_validation_plugins(work_dir)
    comparator = get_comparator(case.task_strategy)
    if comparator is None:
        raise RuntimeError(
            f"No comparator plugin {case.task_strategy!r} for {case.family}"
        )
    return comparator


def _model_owned_diffusion_native_acceptance(work_dir: Path) -> Any:
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        return None
    model_manifest = str(validation_config.get("model_manifest", "") or "")
    if not model_manifest:
        return None
    owner_path = Path(model_manifest)
    if not owner_path.is_absolute():
        owner_path = REPO_ROOT / owner_path
    case = load_manifest(owner_path)
    return copy.deepcopy(case.metadata.get("native_acceptance"))


def _compute_validation_clip_metrics(
    trt_frames_dir: str, hf_frames_dir: str, prompt: str
) -> Any:
    from tests.e2e.models.flux.e2e_plugins.comparators.clip_metrics import (
        compute_clip_metrics,
    )

    return compute_clip_metrics(trt_frames_dir, hf_frames_dir, prompt)


def _first_generated_image(frames_dir: str) -> Path | None:
    root = Path(frames_dir)
    if not root.is_dir():
        return None
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images = sorted(root.rglob(suffix))
        if images:
            return images[0]
    return None


def _generated_frame_paths(frames_dir: object) -> list[str]:
    root = Path(str(frames_dir or ""))
    if not root.is_dir():
        return []
    return [str(path) for path in sorted(root.glob("frame_*.png")) if path.is_file()]


def write_diffusion_visual_review(
    work_dir: Path,
    samples: list[dict[str, Any]],
) -> Path:
    report_path = work_dir / "visual_review.html"

    def image_tag(path_value: str, label: str) -> str:
        path = Path(path_value) if path_value else None
        if path is None or not path.is_file():
            return f"<div class='missing'>{html.escape(label)} image missing</div>"
        relative = os.path.relpath(path, report_path.parent)
        return (
            f"<figure><figcaption>{html.escape(label)}</figcaption>"
            f"<a href='{html.escape(relative)}'><img loading='lazy' "
            f"src='{html.escape(relative)}' alt='{html.escape(label)}'></a></figure>"
        )

    cards = []
    for sample in samples:
        questions = sample.get("questions", [])
        question_items = (
            "".join(
                f"<li>{html.escape(str(question.get('question', question)))}</li>"
                for question in questions
            )
            or "<li>No proposition questions provided.</li>"
        )
        metrics = sample.get("metrics", {})
        metric_rows = "".join(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{float(metric.get('value', 0.0)):.6f}</td>"
            f"<td>{html.escape(str(metric.get('operator', 'info')))}</td>"
            f"<td>{html.escape(str(metric.get('threshold', 'diagnostic')))}</td>"
            f"<td>{'yes' if metric.get('passed', False) else 'no'}</td>"
            "</tr>"
            for name, metric in sorted(metrics.items())
        )
        cards.append(
            f"<section class='case {html.escape(str(sample.get('status', '')))}'>"
            f"<h2>{sample.get('index', 0) + 1}. "
            f"{html.escape(str(sample.get('sample_id', '')))} — "
            f"{html.escape(str(sample.get('status', '')))}</h2>"
            f"<p class='prompt'>{html.escape(str(sample.get('prompt', '')))}</p>"
            "<div class='images'>"
            f"{image_tag(str(sample.get('hf_image', '')), 'HF')}"
            f"{image_tag(str(sample.get('trtfb_image', '')), 'TRTMC')}"
            "</div>"
            "<details open><summary>DPG proposition checklist</summary>"
            f"<ol>{question_items}</ol></details>"
            "<details><summary>Parity metrics</summary>"
            "<table><thead><tr><th>Metric</th><th>Value</th><th>Gate</th>"
            f"<th>Threshold</th><th>Passed</th></tr></thead><tbody>{metric_rows}</tbody></table>"
            "</details></section>"
        )
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Diffusion HF/TRTMC visual review</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;"
        "background:#f4f5f7;color:#171717}.case{background:white;padding:18px;margin:20px 0;"
        "border-radius:10px;border-left:8px solid #888}.case.passed{border-color:#258a45}"
        ".case.failed,.case.error{border-color:#c43b32}.prompt{font-size:1.05rem;line-height:1.5}"
        ".images{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}"
        "figcaption{font-weight:700;margin-bottom:6px}img{width:100%;height:auto;border:1px solid #bbb}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;text-align:left}"
        "li{margin:4px 0}.missing{padding:40px;background:#fee;color:#900}"
        "</style></head><body><h1>Diffusion HF/TRTMC visual review</h1>"
        "<p>Parity metrics answer whether both implementations agree. The DPG checklist is for "
        "manual prompt-adherence review and must not be inferred from CLIPScore alone.</p>"
        + "".join(cards) + "</body></html>\n",
        encoding="utf-8",
    )
    return report_path


def compare_diffusion_image_predictions(
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    work_dir: Path,
    gates: dict[str, Any],
) -> dict[str, Any]:
    from tests.e2e_harness.contracts import (
        StageOutput,
        StageSpec,
        StageStatus,
        ThresholdProfile,
    )

    hf_rows = hf_predictions.get("responses", [])
    trt_rows = trtfb_predictions.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(hf_rows) != len(trt_rows) or len(hf_rows) != len(requests):
        raise ValueError(
            "HF predictions, TRT predictions, and diffusion requests must have "
            f"the same length: {len(hf_rows)}, {len(trt_rows)}, {len(requests)}"
        )
    comparator = _load_diffusion_validation_comparator(work_dir)
    model_native_acceptance = _model_owned_diffusion_native_acceptance(work_dir)
    threshold = ThresholdProfile(
        task_strategy="diffusion_media_generation",
        profile_name="task_eval",
        metrics={str(key): float(value) for key, value in gates.items()},
    )
    stage = StageSpec(name="end_to_end", required=True)
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_passed: Counter[str] = Counter()
    metric_gated: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    passed_count = 0
    skipped_count = 0

    for index, (hf_row, trt_row, request) in enumerate(
        zip(hf_rows, trt_rows, requests, strict=True)
    ):
        expected_id = str(request.get("sample_id", f"partiprompts_{index:06d}"))
        hf_id = str(hf_row.get("sample_id", ""))
        trt_id = str(trt_row.get("sample_id", ""))
        if hf_id != expected_id or trt_id != expected_id:
            raise ValueError(
                f"Diffusion sample id mismatch at index {index}: "
                f"expected={expected_id!r} hf={hf_id!r} trtfb={trt_id!r}"
            )
        expected_prompt = str(request.get("prompt", ""))
        hf_prompt = str(hf_row.get("prompt", ""))
        trt_prompt = str(trt_row.get("prompt", ""))
        if expected_prompt and (
            hf_prompt != expected_prompt or trt_prompt != expected_prompt
        ):
            raise ValueError(
                f"Diffusion prompt mismatch at index {index}: "
                f"expected={expected_prompt!r} hf={hf_prompt!r} trtfb={trt_prompt!r}"
            )
        shared_inputs: dict[str, Any] = {}
        for field in ("seed", "condition_image_sha256", "action"):
            hf_value = hf_row.get(field)
            trt_value = trt_row.get(field)
            if hf_value in (None, "") and trt_value in (None, ""):
                continue
            if hf_value != trt_value:
                raise ValueError(
                    f"Diffusion shared input mismatch at index {index} for {field}: "
                    f"hf={hf_value!r} trtfb={trt_value!r}"
                )
            expected_value = request.get(field)
            if field != "seed" and expected_value not in (None, "") and hf_value != expected_value:
                raise ValueError(
                    f"Diffusion dataset input mismatch at index {index} for {field}: "
                    f"expected={expected_value!r} actual={hf_value!r}"
                )
            shared_inputs[field] = hf_value
        hf_latent_hash = str(hf_row.get("initial_latents_sha256", ""))
        trt_latent_hash = str(trt_row.get("initial_latents_sha256", ""))
        require_matching_latents = float(gates.get("require_matching_initial_latents", 0)) > 0
        if require_matching_latents and (not hf_latent_hash or not trt_latent_hash):
            raise ValueError(
                f"Diffusion parity requires matching initial latents at index {index}: "
                f"hf={hf_latent_hash!r} trtfb={trt_latent_hash!r}"
            )
        if hf_latent_hash or trt_latent_hash:
            if not hf_latent_hash or not trt_latent_hash or hf_latent_hash != trt_latent_hash:
                raise ValueError(
                    f"Diffusion initial latent mismatch at index {index}: "
                    f"hf={hf_latent_hash!r} trtfb={trt_latent_hash!r}"
                )
        invalid = (
            int(hf_row.get("returncode", 1)) != 0
            or int(trt_row.get("returncode", 1)) != 0
            or int(hf_row.get("num_frames", 0)) < 1
            or int(trt_row.get("num_frames", 0)) < 1
        )
        if invalid:
            skipped_count += 1
            samples.append({
                "index": index,
                "sample_id": expected_id,
                "category": request.get("category", ""),
                "challenge": request.get("challenge", ""),
                "prompt": request.get("prompt", ""),
                "questions": request.get("questions", []),
                "hf_image": str(_first_generated_image(str(hf_row.get("frames_dir", ""))) or ""),
                "trtfb_image": str(_first_generated_image(str(trt_row.get("frames_dir", ""))) or ""),
                "status": StageStatus.ERROR.value,
                "metrics": {},
                "shared_inputs": shared_inputs,
            })
            continue

        trt_frames_dir = trt_row.get("frames_dir", "")
        hf_frames_dir = hf_row.get("frames_dir", "")
        trt_output = StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": trt_row.get("returncode", 0),
                "num_frames": trt_row.get("num_frames", 0),
                "frames_dir": trt_frames_dir,
                "frame_paths": _generated_frame_paths(trt_frames_dir),
                "frame_stats": trt_row.get("frame_stats", {}),
                "prompt": trt_row.get("prompt", request.get("prompt", "")),
            },
        )
        native_acceptance = hf_row.get("native_acceptance")
        if native_acceptance is None:
            native_acceptance = model_native_acceptance
        hf_output = StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": hf_row.get("returncode", 0),
                "num_frames": hf_row.get("num_frames", 0),
                "frames_dir": hf_frames_dir,
                "frame_paths": _generated_frame_paths(hf_frames_dir),
                "frame_stats": hf_row.get("frame_stats", {}),
                "prompt": hf_row.get("prompt", request.get("prompt", "")),
                "native_acceptance": native_acceptance,
            },
        )
        result = comparator.compare(trt_output, hf_output, threshold, stage)
        required_clip_metrics = {
            "prompt_clipscore_delta",
            "hf_prompt_clipscore",
            "trt_prompt_clipscore",
            "trt_hf_image_clip_cosine",
        }
        missing_clip_metrics = required_clip_metrics.difference(result.metrics)
        if missing_clip_metrics:
            from tests.e2e_harness.contracts import MetricResult

            clip = _compute_validation_clip_metrics(
                str(trt_output.data["frames_dir"]),
                str(hf_output.data["frames_dir"]),
                str(trt_output.data["prompt"]),
            )
            if clip is not None:
                max_drop = (
                    float(gates["max_prompt_clipscore_drop"])
                    if "max_prompt_clipscore_drop" in gates
                    else None
                )
                hf_floor = (
                    float(gates["min_hf_prompt_clipscore"])
                    if "min_hf_prompt_clipscore" in gates
                    else None
                )
                image_floor = float(gates.get("min_trt_hf_image_clip_cosine", 0.0))
                clip_metrics = {
                    "prompt_clipscore_delta": MetricResult(
                        value=float(clip.prompt_clipscore_delta),
                        threshold=-max_drop if max_drop is not None else None,
                        operator=">=" if max_drop is not None else "info",
                        passed=(
                            max_drop is None
                            or float(clip.prompt_clipscore_delta) >= -max_drop
                        ),
                        note=(
                            f"trt={clip.trt_prompt_clipscore:.2f}, "
                            f"hf={clip.hf_prompt_clipscore:.2f}"
                            + (" [prompt truncated]" if clip.prompt_truncated else "")
                        ),
                    ),
                    "hf_prompt_clipscore": MetricResult(
                        value=float(clip.hf_prompt_clipscore),
                        threshold=hf_floor,
                        operator=">=" if hf_floor is not None else "info",
                        passed=(
                            hf_floor is None
                            or float(clip.hf_prompt_clipscore) >= hf_floor
                        ),
                    ),
                    "trt_prompt_clipscore": MetricResult(
                        value=float(clip.trt_prompt_clipscore),
                        threshold=None,
                        operator="info",
                        passed=True,
                    ),
                    "trt_hf_image_clip_cosine": MetricResult(
                        value=float(clip.trt_hf_image_clip_cosine),
                        threshold=image_floor if image_floor > 0.0 else None,
                        operator=">=",
                        passed=(
                            image_floor <= 0.0
                            or float(clip.trt_hf_image_clip_cosine) >= image_floor
                        ),
                    ),
                }
                result.metrics.update(clip_metrics)
                if any(
                    metric.threshold is not None and not metric.passed
                    for metric in clip_metrics.values()
                ):
                    result.status = StageStatus.FAILED.value
                missing_clip_metrics = required_clip_metrics.difference(result.metrics)
        if missing_clip_metrics:
            raise RuntimeError(
                "Diffusion scorecard did not produce required CLIP metrics "
                f"for {expected_id}: {sorted(missing_clip_metrics)}. "
                "Install open_clip and verify both HF and TRT image artifacts."
            )
        passed_count += int(result.status == StageStatus.PASSED.value)
        sample_metrics: dict[str, Any] = {}
        for metric_name, metric in result.metrics.items():
            value = float(metric.value)
            metric_values[metric_name].append(value)
            if metric.threshold is not None:
                metric_gated[metric_name] += 1
                metric_passed[metric_name] += int(metric.passed)
            sample_metrics[metric_name] = {
                "value": value,
                "threshold": metric.threshold,
                "operator": metric.operator,
                "passed": metric.passed,
                "note": metric.note,
            }
        gated_metrics = [
            metric for metric in result.metrics.values() if metric.threshold is not None
        ]
        gated_passed = sum(int(metric.passed) for metric in gated_metrics)
        result.message = (
            f"{'PASS' if result.status == StageStatus.PASSED.value else 'FAIL'}: "
            f"{gated_passed}/{len(gated_metrics)} gated metrics passed"
        )
        samples.append({
            "index": index,
            "sample_id": expected_id,
            "category": request.get("category", ""),
            "challenge": request.get("challenge", ""),
            "prompt": request.get("prompt", ""),
            "questions": request.get("questions", []),
            "hf_image": str(_first_generated_image(str(hf_output.data["frames_dir"])) or ""),
            "trtfb_image": str(_first_generated_image(str(trt_output.data["frames_dir"])) or ""),
            "initial_latents_sha256": hf_latent_hash,
            "shared_inputs": shared_inputs,
            "status": result.status,
            "metrics": sample_metrics,
            "message": result.message,
        })

    metrics = {}
    for metric_name, values in sorted(metric_values.items()):
        metrics[metric_name] = {
            "mean": _mean(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "gated_count": metric_gated[metric_name],
            "passed_count": metric_passed[metric_name],
        }
    valid_count = len(requests) - skipped_count
    visual_review = write_diffusion_visual_review(work_dir, samples)
    return {
        "mode": "diffusion_image_clip_parity",
        "overall_pass_rate": passed_count / valid_count if valid_count else 0.0,
        "passed_count": passed_count,
        "valid_count": valid_count,
        "skipped_count": skipped_count,
        "total_count": len(requests),
        "metrics": metrics,
        "samples": samples,
        "visual_review": str(visual_review),
    }


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Task Evaluation Summary",
        "",
        "| Backend | Accuracy | Correct | Valid | Skipped |",
        "|---|---:|---:|---:|---:|",
        (
            f"| HF ref | {summary['hf']['overall_accuracy']:.4f} | "
            f"{summary['hf']['correct']} | {summary['hf']['valid_count']} | "
            f"{summary['hf']['skipped_count']} |"
        ),
        (
            f"| TRTFB | {summary['trtfb']['overall_accuracy']:.4f} | "
            f"{summary['trtfb']['correct']} | {summary['trtfb']['valid_count']} | "
            f"{summary['trtfb']['skipped_count']} |"
        ),
        "",
        f"- accuracy_delta_trtfb_minus_hf: {summary['accuracy_delta_trtfb_minus_hf']:.4f}",
        f"- prediction_agreement_rate: {summary['prediction_agreement_rate']:.4f}",
    ]
    if "normalized_transcript_exact_agreement_rate" in summary:
        lines.extend(
            [
                f"- normalized_transcript_exact_agreement_rate: "
                f"{summary['normalized_transcript_exact_agreement_rate']:.4f}",
                f"- correctness_agreement_rate: "
                f"{summary['correctness_agreement_rate']:.4f}",
            ]
        )
    lines.extend([
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ])
    for key in ("both_correct", "hf_correct_trtfb_wrong", "hf_wrong_trtfb_correct", "both_wrong"):
        lines.append(f"| {key} | {summary['buckets'].get(key, 0)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generation_defaults(work_dir: Path) -> dict[str, Any]:
    manifest = work_manifest(work_dir)
    return manifest.get("generation", {})


def work_manifest(work_dir: Path) -> dict[str, Any]:
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def hf_generation_overrides(work_dir: Path) -> dict[str, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        return {}
    overrides: dict[str, Any] = {}
    if "hf_use_cache" in validation_config:
        overrides["use_cache"] = bool(validation_config["hf_use_cache"])
    return overrides


def _work_dataset_kind(work_dir: Path) -> str:
    return str(work_manifest(work_dir).get("dataset_kind", ""))


def _is_vlm_dataset_kind(kind: str) -> bool:
    return kind in {"vlm_chat_json", "vlm_grounding_json", "vlm_unified_json"}


def _is_asr_dataset_kind(kind: str) -> bool:
    return kind in {"asr_chat_json"}


def _is_diffusion_media_dataset_kind(kind: str) -> bool:
    return kind in {"diffusion_prompt_tsv", "diffusion_prompt_json"}


def _is_diffusion_text_dataset_kind(kind: str) -> bool:
    return kind in {"conditional_text_jsonl", "unconditional_text_json"}


def _is_tts_dataset_kind(kind: str) -> bool:
    return kind == "seedtts_json"


def _is_encoder_embedding_dataset_kind(kind: str) -> bool:
    return kind == "sts_pair_jsonl"


def _is_time_series_dataset_kind(kind: str) -> bool:
    return kind == "time_series_csv"


def _is_vision_task_dataset_kind(kind: str) -> bool:
    return kind in {
        "image_classification_json",
        "semantic_segmentation_json",
        "prompted_segmentation_json",
    }


def _is_reranking_dataset_kind(kind: str) -> bool:
    return kind == "reranking_json"


def _is_model_plugin_dataset_kind(kind: str) -> bool:
    return kind == "model_plugin_json"


def work_scoring(work_dir: Path) -> dict[str, Any]:
    scoring = work_manifest(work_dir).get("scoring", {})
    return scoring if isinstance(scoring, dict) else {}


def _write_pcm16_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    pcm = (samples * 32767.0).clip(-32768, 32767).astype(np.int16)
    data = pcm.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def _transcribe_audio_files(
    wav_paths: list[Path],
    *,
    python: str,
    model_id: str,
    local_files_only: bool = False,
) -> list[str]:
    if not wav_paths:
        return []
    script = """
import json
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample
from transformers import pipeline

paths = %(paths)r
model_id = %(model_id)r
local_files_only = %(local_files_only)r
device = 0 if torch.cuda.is_available() else -1
model_kwargs = {"local_files_only": True} if local_files_only else {}
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device,
    model_kwargs=model_kwargs,
)
target_sample_rate = int(transcriber.feature_extractor.sampling_rate)
waveforms = []
for path in paths:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    if sample_rate != target_sample_rate:
        target_length = int(round(len(audio) * target_sample_rate / sample_rate))
        audio = resample(audio, target_length).astype(np.float32)
    waveforms.append(audio)
outputs = transcriber(waveforms, batch_size=min(8, len(waveforms)))
if isinstance(outputs, dict):
    outputs = [outputs]
print(json.dumps([str(item.get("text", "")).strip() for item in outputs]))
""" % {
        "paths": [str(path) for path in wav_paths],
        "model_id": model_id,
        "local_files_only": local_files_only,
    }
    proc = subprocess.run(
        [python, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        timeout=max(600, 120 * len(wav_paths)),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TTS ASR transcription failed rc={proc.returncode}: {proc.stderr[-2000:]}"
        )
    for line in reversed(proc.stdout.splitlines()):
        try:
            transcripts = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(transcripts, list) and len(transcripts) == len(wav_paths):
            return [str(value) for value in transcripts]
    raise RuntimeError("TTS ASR transcription produced no parseable transcript list")


def _tts_response_row(
    sample_id: str, wav_path: Path, wall_ms: float, source: str
) -> dict[str, Any]:
    metrics = _read_wav_metrics(wav_path)
    return {
        "sample_id": sample_id,
        "output_text": "",
        "wav_path": str(wav_path),
        "wav_exists": True,
        "rms": metrics["rms"],
        "duration_s": metrics["duration_s"],
        "sample_rate": metrics["sample_rate"],
        "wall_ms": wall_ms,
        "source": source,
    }


def _is_canary_asr_reference(args: argparse.Namespace) -> bool:
    reference_family = str(getattr(args, "reference_family", "") or "").lower()
    family = str(getattr(args, "family", "") or "").lower()
    model = str(getattr(args, "model", "") or "").lower()
    return reference_family == "asr_canary" or family == "canary" or "canary" in model


def _is_nemo_asr_reference(args: argparse.Namespace) -> bool:
    reference_family = str(getattr(args, "reference_family", "") or "").lower()
    family = str(getattr(args, "family", "") or "").lower()
    model = str(getattr(args, "model", "") or "").lower()
    return (
        _is_canary_asr_reference(args)
        or family == "nemotron_speech_streaming"
        or "nemotron-speech-streaming" in model
        or reference_family == "asr_nemo"
    )


def _model_dtype(torch_mod: Any, dtype_name: str) -> Any:
    if dtype_name == "float16":
        return torch_mod.float16
    if dtype_name == "bfloat16":
        return torch_mod.bfloat16
    return "auto"


def _load_pil_images(image_paths: list[str]) -> list[Any]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("VLM reference-consistency validation requires Pillow") from exc
    images: list[Any] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def _read_wav_float32(path: str) -> tuple[Any, int]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ASR HF reference requires numpy to read WAV audio") from exc
    import wave

    with wave.open(path, "rb") as wav_f:
        channels = wav_f.getnchannels()
        sample_width = wav_f.getsampwidth()
        sample_rate = wav_f.getframerate()
        frames = wav_f.readframes(wav_f.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width {sample_width} bytes for {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def _resample_audio(audio: Any, source_sr: int, target_sr: int) -> Any:
    if source_sr == target_sr:
        return audio
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ASR HF reference requires numpy to resample audio") from exc
    if len(audio) == 0:
        return audio
    target_len = max(1, int(len(audio) * target_sr / source_sr))
    source_x = np.arange(len(audio), dtype=np.float32)
    target_x = np.linspace(0, len(audio) - 1, target_len, dtype=np.float32)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def _write_wav_pcm16(path: Path, audio: Any, sample_rate: int) -> None:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ASR HF reference requires numpy to write WAV audio") from exc
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    pcm = np.clip(audio_array * 32768.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_f:
        wav_f.setnchannels(1)
        wav_f.setsampwidth(2)
        wav_f.setframerate(sample_rate)
        wav_f.writeframes(pcm.tobytes())


def _transcription_text(transcriptions: Any) -> str:
    value = transcriptions
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value)


def _vlm_model_classes(transformers_mod: Any) -> list[Any]:
    classes = []
    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ):
        cls = getattr(transformers_mod, name, None)
        if cls is not None:
            classes.append(cls)
    if not classes:
        raise RuntimeError(
            "Transformers installation does not expose a VLM-capable AutoModel class"
        )
    return classes


def _load_vlm_model(transformers_mod: Any, model_id: str, model_kwargs: dict[str, Any]) -> Any:
    errors: list[str] = []
    for model_cls in _vlm_model_classes(transformers_mod):
        try:
            return model_cls.from_pretrained(model_id, **model_kwargs).eval()
        except ValueError as exc:
            errors.append(f"{model_cls.__name__}: {exc}")
    raise RuntimeError(
        f"Could not load VLM HF reference model {model_id!r} with available AutoModel classes: "
        + " | ".join(errors)
    )


def _vlm_prompt_has_image_placeholder(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<|image_pad|>",
            "<|vision_start|>",
            "<image>",
            "<IMG_CONTEXT>",
        )
    )


def _vlm_fallback_prompt(prompt: str, template: str = "") -> str:
    if not template:
        return prompt
    return template.replace("{prompt}", prompt)


def _apply_vlm_chat_template(obj: Any, messages: list[Any]) -> str:
    if not hasattr(obj, "apply_chat_template"):
        return ""
    try:
        return str(
            obj.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except ValueError as exc:
        if "chat_template" in str(exc):
            return ""
        raise


def _vlm_chat_text(
    processor: Any,
    request: dict[str, Any],
    fallback_prompt: str,
    model_id: str = "",
    fallback_prompt_template: str = "",
) -> str:
    if not fallback_prompt_template and (
        "{prompt}" in model_id or _vlm_prompt_has_image_placeholder(model_id)
    ):
        fallback_prompt_template = model_id
        model_id = ""
    messages = request.get("messages")
    rendered = ""
    if hasattr(processor, "apply_chat_template") and isinstance(messages, list):
        rendered = _apply_vlm_chat_template(processor, messages)
    tokenizer = getattr(processor, "tokenizer", None)
    if (
        not rendered
        and tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and isinstance(messages, list)
    ):
        rendered = _apply_vlm_chat_template(tokenizer, messages)
    if rendered:
        return rendered
    if _vlm_prompt_has_image_placeholder(fallback_prompt):
        return fallback_prompt
    return _vlm_fallback_prompt(fallback_prompt, fallback_prompt_template)


def _strip_generated_text_prefix(text: str, prompt: str) -> str:
    generated = text.strip()
    for marker in ("assistant\n", "assistant:", "ASSISTANT:"):
        if marker in generated:
            generated = generated.split(marker, 1)[-1].strip()
            break
    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt) :].strip()
    return generated


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _is_deepseek_ocr_hf_model(model_id: str, model: Any) -> bool:
    return "deepseek-ocr" in model_id.lower() and hasattr(model, "infer")


def _deepseek_ocr_prompt(prompt: str) -> str:
    if "<image>" in prompt:
        return prompt
    return f"<image>\n{prompt}"


def _run_deepseek_ocr_hf_reference(
    *,
    model: Any,
    tokenizer: Any,
    answers: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    work_dir: Path,
) -> None:
    raw_path = work_dir / "hf_raw.jsonl"
    pred_path = work_dir / "hf_predictions.json"
    output_root = work_dir / "hf_deepseek_ocr_outputs"
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, _request in enumerate(answers["requests"]):
            prompt_row = prompt_rows[idx]
            image_paths = [str(path) for path in prompt_row.get("images", [])]
            if len(image_paths) != 1:
                raise ValueError(
                    f"DeepSeek-OCR HF reference expects exactly one image for sample {idx}"
                )
            sample_id = str(prompt_row.get("sample_id", f"vlm_{idx:06d}"))
            output_path = output_root / sample_id
            start = time.perf_counter()
            output_text = model.infer(
                tokenizer,
                prompt=_deepseek_ocr_prompt(str(prompt_row.get("prompt", ""))),
                image_file=image_paths[0],
                output_path=str(output_path),
                save_results=False,
                eval_mode=True,
            )
            wall_ms = (time.perf_counter() - start) * 1000.0
            output_text = "" if output_text is None else str(output_text)
            generated_token_ids = []
            try:
                generated_token_ids = [
                    int(token_id)
                    for token_id in tokenizer(output_text, add_special_tokens=False).input_ids
                ]
            except Exception:
                generated_token_ids = []
            row = {
                "sample_id": sample_id,
                "output_text": output_text,
                "generated_tokens": len(generated_token_ids),
                "generated_token_ids": generated_token_ids,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(
                f"[validation.vlm_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
            )
    write_predictions(pred_path, responses)


def run_vlm_hf_reference(args: argparse.Namespace) -> None:
    try:
        import torch
        import transformers
        from transformers import AutoProcessor, logging
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("VLM run-hf requires torch and transformers") from exc

    work_dir = Path(args.work_dir)
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    if len(prompt_rows) != len(answers["requests"]):
        raise ValueError("answers.json and prompts.jsonl must contain the same number of samples")
    defaults = generation_defaults(work_dir)
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    vlm_fallback_prompt_template = str(
        validation_config.get("vlm_fallback_prompt_template", "") or ""
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 8))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(defaults.get("temperature", 1.0))
    )
    top_k = args.top_k if args.top_k is not None else int(defaults.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(defaults.get("top_p", 1.0))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", -1))
    do_sample = args.do_sample or bool(defaults.get("do_sample", False))

    logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_kwargs = {
        "torch_dtype": _model_dtype(torch, args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = _load_vlm_model(transformers, args.model, model_kwargs)
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    else:
        device = model.device

    if _is_deepseek_ocr_hf_model(args.model, model):
        _run_deepseek_ocr_hf_reference(
            model=model,
            tokenizer=processor,
            answers=answers,
            prompt_rows=prompt_rows,
            work_dir=work_dir,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if tokenizer is not None and pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, request in enumerate(answers["requests"]):
            prompt_row = prompt_rows[idx]
            image_paths = [str(path) for path in prompt_row.get("images", [])]
            if len(image_paths) != 1:
                raise ValueError(f"VLM HF reference expects exactly one image for sample {idx}")
            images = _load_pil_images(image_paths)
            prompt = _vlm_chat_text(
                processor,
                request,
                str(prompt_row.get("prompt", "")),
                args.model,
                vlm_fallback_prompt_template,
            )
            inputs = processor(
                text=[prompt],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = _to_device(inputs, device)
            if seed >= 0:
                torch.manual_seed(seed + idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + idx)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_beams": 1,
            }
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generate_kwargs)
            wall_ms = (time.perf_counter() - start) * 1000.0
            input_len = int(inputs["input_ids"].shape[1])
            generated = output_ids[0, input_len:]
            if hasattr(processor, "batch_decode"):
                output_text = processor.batch_decode(
                    generated.unsqueeze(0), skip_special_tokens=False
                )[0]
            elif tokenizer is not None:
                output_text = tokenizer.decode(generated, skip_special_tokens=False)
            else:
                output_text = str(generated.tolist())
            row = {
                "sample_id": prompt_row.get("sample_id", f"vlm_{idx:06d}"),
                "output_text": output_text,
                "generated_tokens": int(generated.shape[0]),
                "generated_token_ids": [int(token_id) for token_id in generated.tolist()],
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(
                f"[validation.vlm_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
            )
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_asr_hf_reference(args: argparse.Namespace) -> None:
    if _is_nemo_asr_reference(args):
        _run_nemo_asr_hf_reference(args)
        return

    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, logging
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ASR run-hf requires torch and transformers") from exc

    work_dir = Path(args.work_dir)
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    if len(prompt_rows) != len(answers["requests"]):
        raise ValueError("answers.json and prompts.jsonl must contain the same number of samples")
    defaults = generation_defaults(work_dir)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 100))
    )
    seed = args.seed if args.seed is not None else int(defaults.get("seed", -1))

    logging.set_verbosity_error()
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model_kwargs = {
        "torch_dtype": _model_dtype(torch, args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, **model_kwargs).eval()
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    else:
        device = model.device
    model_dtype = next(model.parameters()).dtype
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, _request in enumerate(answers["requests"]):
            prompt_row = prompt_rows[idx]
            audio_path = str(prompt_row.get("audio", ""))
            if not audio_path:
                raise ValueError(f"ASR HF reference expects an audio path for sample {idx}")
            audio, sample_rate = _read_wav_float32(audio_path)
            audio = _resample_audio(audio, sample_rate, target_sr)
            inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt")
            inputs = {
                key: (
                    value.to(device=device, dtype=model_dtype)
                    if hasattr(value, "is_floating_point") and value.is_floating_point()
                    else value.to(device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }
            if seed >= 0:
                torch.manual_seed(seed + idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + idx)
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            wall_ms = (time.perf_counter() - start) * 1000.0
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            generated_token_ids = [int(token_id) for token_id in output_ids[0].tolist()]
            row = {
                "sample_id": prompt_row.get("sample_id", f"asr_{idx:06d}"),
                "output_text": output_text,
                "generated_tokens": len(generated_token_ids),
                "generated_token_ids": generated_token_ids,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(
                f"[validation.asr_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
            )
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_prompt_conditioned_nemo_asr(model_id: str) -> bool:
    return "nemotron-3.5-asr-streaming" in model_id.lower()


def _run_nemo_asr_hf_reference(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    if len(prompt_rows) != len(answers["requests"]):
        raise ValueError("answers.json and prompts.jsonl must contain the same number of samples")
    defaults = generation_defaults(work_dir)
    target_sr = int(defaults.get("sample_rate", 16000) or 16000)

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    canary_audio_dir = work_dir / "hf_canary_audio"

    # The model card requires Transformers >=5.13 for Nemotron 3.5. The
    # standard NeMo transcribe path in older environments cannot keep the
    # prompt feature aligned with cache-aware encoder frames.
    if _is_prompt_conditioned_nemo_asr(args.model):
        _run_nemotron35_transformers_reference(
            args=args,
            prompt_rows=prompt_rows,
            raw_path=raw_path,
            pred_path=pred_path,
            target_sr=target_sr,
            canary_audio_dir=canary_audio_dir,
        )
        return

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        _run_nemo_asr_hf_pipeline_reference(
            args=args,
            prompt_rows=prompt_rows,
            raw_path=raw_path,
            pred_path=pred_path,
            target_sr=target_sr,
            canary_audio_dir=canary_audio_dir,
        )
        return

    map_location = str(getattr(args, "device", "") or "cpu")
    model = nemo_asr.models.ASRModel.from_pretrained(
        args.model, map_location=map_location
    )
    try:
        if map_location and map_location != "cpu" and hasattr(model, "to"):
            model = model.to(map_location)
    except Exception:
        pass
    model.eval()

    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, prompt_row in enumerate(prompt_rows):
            sample_id = str(prompt_row.get("sample_id", f"asr_{idx:06d}"))
            audio_path = str(prompt_row.get("audio", ""))
            if not audio_path:
                raise ValueError(f"NeMo ASR HF reference expects an audio path for sample {idx}")
            audio, sample_rate = _read_wav_float32(audio_path)
            audio = _resample_audio(audio, sample_rate, target_sr)
            mono_path = canary_audio_dir / _safe_sample_filename(sample_id, ".wav")
            _write_wav_pcm16(mono_path, audio, target_sr)
            start = time.perf_counter()
            transcriptions = model.transcribe([str(mono_path)], batch_size=1)
            wall_ms = (time.perf_counter() - start) * 1000.0
            row = {
                "sample_id": sample_id,
                "output_text": _transcription_text(transcriptions),
                "generated_tokens": None,
                "generated_token_ids": None,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.nemo_asr_hf] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    write_predictions(pred_path, responses)
    del model
    gc.collect()


def _run_nemotron35_transformers_reference(
    *,
    args: argparse.Namespace,
    prompt_rows: list[dict[str, Any]],
    raw_path: Path,
    pred_path: Path,
    target_sr: int,
    canary_audio_dir: Path,
) -> None:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Nemotron 3.5 ASR reference requires transformers>=5.13"
        ) from exc

    device = torch.device(str(getattr(args, "device", "") or "cpu"))
    common_kwargs = {
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        "local_files_only": bool(getattr(args, "local_files_only", False)),
    }
    processor = AutoProcessor.from_pretrained(args.model, **common_kwargs)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=_model_dtype(torch, getattr(args, "dtype", "auto")),
        **common_kwargs,
    ).eval()
    model.to(device)
    defaults = generation_defaults(Path(args.work_dir))
    max_new_tokens = int(defaults.get("max_new_tokens", 256) or 256)

    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, prompt_row in enumerate(prompt_rows):
            sample_id = str(prompt_row.get("sample_id", f"asr_{idx:06d}"))
            audio_path = str(prompt_row.get("audio", ""))
            if not audio_path:
                raise ValueError(
                    f"Nemotron 3.5 ASR HF reference expects an audio path for sample {idx}"
                )
            audio, sample_rate = _read_wav_float32(audio_path)
            audio = _resample_audio(audio, sample_rate, target_sr)
            mono_path = canary_audio_dir / _safe_sample_filename(sample_id, ".wav")
            _write_wav_pcm16(mono_path, audio, target_sr)
            language = str(
                prompt_row.get("language")
                or defaults.get("language", "en-US")
                or "en-US"
            )
            inputs = processor(
                audio,
                sampling_rate=target_sr,
                language=language,
                return_tensors="pt",
            )
            inputs = _to_device(inputs, device)
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
            wall_ms = (time.perf_counter() - start) * 1000.0
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            token_ids = [int(token_id) for token_id in sequences[0].detach().cpu().tolist()]
            row = {
                "sample_id": sample_id,
                "output_text": processor.batch_decode(
                    sequences, skip_special_tokens=True
                )[0],
                "generated_tokens": len(token_ids),
                "generated_token_ids": token_ids,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(
                f"[validation.nemotron35_hf] sample={idx + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_nemo_asr_hf_pipeline_reference(
    *,
    args: argparse.Namespace,
    prompt_rows: list[dict[str, Any]],
    raw_path: Path,
    pred_path: Path,
    target_sr: int,
    canary_audio_dir: Path,
) -> None:
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("NeMo ASR reference requires NeMo or transformers pipeline") from exc

    device = (
        0
        if str(getattr(args, "device", "")).startswith("cuda") and torch.cuda.is_available()
        else -1
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        torch_dtype=_model_dtype(torch, getattr(args, "dtype", "auto")),
        device=device,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
        local_files_only=bool(getattr(args, "local_files_only", False)),
    )
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, prompt_row in enumerate(prompt_rows):
            sample_id = str(prompt_row.get("sample_id", f"asr_{idx:06d}"))
            audio_path = str(prompt_row.get("audio", ""))
            if not audio_path:
                raise ValueError(
                    f"NeMo ASR HF pipeline reference expects an audio path for sample {idx}"
                )
            audio, sample_rate = _read_wav_float32(audio_path)
            audio = _resample_audio(audio, sample_rate, target_sr)
            mono_path = canary_audio_dir / _safe_sample_filename(sample_id, ".wav")
            _write_wav_pcm16(mono_path, audio, target_sr)
            start = time.perf_counter()
            result = pipe({"raw": audio, "sampling_rate": target_sr})
            wall_ms = (time.perf_counter() - start) * 1000.0
            row = {
                "sample_id": sample_id,
                "output_text": _transcription_text(result),
                "generated_tokens": None,
                "generated_token_ids": None,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(
                f"[validation.nemo_asr_hf_pipeline] sample={idx + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def _load_vision_validation_plugins(work_dir: Path) -> tuple[Any, Any, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    manifest_ref = str(validation_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("vision reference-consistency validation requires task_eval.model_manifest")
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    case = load_manifest(manifest_path)
    model_test_dir = str(case.metadata.get("model_test_dir", "") or "")
    activate_model_plugins(model_test_dir)
    reference = get_reference(case.reference_backend)
    runner = get_runner(case.task_strategy)
    if reference is None:
        raise RuntimeError(
            f"No reference plugin {case.reference_backend!r} for {case.family}"
        )
    if runner is None:
        raise RuntimeError(f"No runner plugin {case.task_strategy!r} for {case.family}")
    return case, reference, runner


def _load_time_series_validation_plugins(work_dir: Path) -> tuple[Any, Any, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    manifest_ref = str(validation_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("time-series reference-consistency validation requires task_eval.model_manifest")
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    case = load_manifest(manifest_path)
    activate_model_plugins(str(case.metadata.get("model_test_dir", "") or ""))
    reference = get_reference(case.reference_backend)
    runner = get_runner(case.task_strategy)
    if reference is None:
        raise RuntimeError(f"No reference plugin {case.reference_backend!r} for {case.family}")
    if runner is None:
        raise RuntimeError(f"No runner plugin {case.task_strategy!r} for {case.family}")
    return case, reference, runner


def _time_series_case_for_request(template: Any, prompt_row: dict[str, Any], index: int) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"time_series_{index:06d}"))
    inputs = prompt_row.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"Time-series sample {case.name!r} has no numeric inputs")
    case.inputs = copy.deepcopy(inputs)
    return case


def _time_series_full_inference_stage(case: Any) -> Any:
    from tests.e2e_harness.contracts import StageSpec

    for stage in case.stages:
        if stage.name == "full_inference":
            return stage
    return StageSpec(name="full_inference", required=True)


def _time_series_response(*, case: Any, source: str, output: Any) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    error = str(data.get("error", "") or "")
    values = data.get("output_field", data.get("field"))
    if error:
        raise RuntimeError(f"{source} time-series inference failed for {case.name}: {error}")
    if not isinstance(values, list) or not values:
        raise RuntimeError(
            f"{source} time-series inference produced no output tensor for {case.name}"
        )
    try:
        output_values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{source} time-series inference produced non-numeric output for {case.name}"
        ) from exc
    output_shape = data.get("output_shape")
    if not isinstance(output_shape, list):
        output_shape = [int(data.get("output_dim", len(output_values)))]
    return {
        "sample_id": case.name,
        "source": source,
        "output_values": output_values,
        "output_shape": [int(dim) for dim in output_shape],
        "output_name": str(data.get("reference_output_name", "") or ""),
        "returncode": int(metadata.get("returncode", 0) or 0),
        "wall_ms": float(output.timing_s) * 1000.0,
    }


def run_time_series_hf_reference(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, reference, _runner = _load_time_series_validation_plugins(work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "hf_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            case = _time_series_case_for_request(template, prompt_row, index)
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                hf_python=sys.executable,
                reference_python=sys.executable,
            )
            output = reference.run_stage(case, _time_series_full_inference_stage(case), context)
            response = _time_series_response(case=case, source="hf", output=output)
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.time_series_hf] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_time_series_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, _reference, runner = _load_time_series_validation_plugins(work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            case = _time_series_case_for_request(template, prompt_row, index)
            case.bundle = bundle_path.name
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            output = runner.run_stage(case, _time_series_full_inference_stage(case), context)
            _record_output_native_command(work_dir, case.name, output)
            log_file.write(
                json.dumps(
                    {
                        "sample_id": case.name,
                        **(output.metadata if isinstance(output.metadata, dict) else {}),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_file.flush()
            response = _time_series_response(case=case, source="trtfb", output=output)
            if response["returncode"] != 0:
                raise RuntimeError(
                    f"TRT time-series run failed for {case.name}: "
                    f"returncode={response['returncode']}"
                )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.time_series_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def _vision_case_for_request(
    template: Any,
    prompt_row: dict[str, Any],
    validation_config: dict[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"vision_{index:06d}"))
    case.inputs["image"] = str(prompt_row["image"])
    prompt_mode = str(validation_config.get("prompt_mode", "") or "")
    if prompt_mode == "point":
        case.inputs["point_x"] = float(prompt_row["point_x"])
        case.inputs["point_y"] = float(prompt_row["point_y"])
        case.inputs["is_foreground"] = True
    elif prompt_mode == "text":
        text_prompt = str(prompt_row["text_prompt"])
        case.inputs["prompt"] = text_prompt
        case.inputs["text_prompt"] = text_prompt
    return case


def _vision_full_inference_stage(case: Any) -> Any:
    from tests.e2e_harness.contracts import StageSpec

    for stage in case.stages:
        if stage.name == "full_inference":
            return stage
    return StageSpec(name="full_inference", required=True)


def _persist_numpy_output(value: Any, path: Path) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))
    return str(path)


def _vision_response(
    *,
    case: Any,
    source: str,
    output: Any,
    dataset_kind: str,
    prompt_row: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    response: dict[str, Any] = {
        "sample_id": case.name,
        "source": source,
        "returncode": int(data.get("returncode", metadata.get("returncode", 0))),
        "image": str(prompt_row["image"]),
        "wall_ms": float(output.timing_s) * 1000.0,
    }
    if dataset_kind == "image_classification_json":
        response.update(
            {
                "top_class": int(data["top_class"]),
                "top_score": float(data.get("top_score", 0.0)),
                "num_classes": int(data.get("num_classes", 0)),
            }
        )
        return response
    if dataset_kind == "semantic_segmentation_json":
        class_map_path = str(
            data.get("class_map_path")
            or data.get("segmentation_map_path")
            or data.get("output_path")
            or ""
        )
        if not class_map_path and data.get("class_map") is not None:
            class_map_path = _persist_numpy_output(
                data["class_map"], artifact_dir / case.name / f"{source}_class_map.npy"
            )
        if not class_map_path or not Path(class_map_path).is_file():
            raise RuntimeError(f"{source} semantic segmentation produced no class map")
        response["class_map_path"] = class_map_path
        raw_class_map_path = str(data.get("raw_class_map_path") or "")
        if raw_class_map_path:
            if not Path(raw_class_map_path).is_file():
                raise RuntimeError(
                    f"{source} semantic segmentation raw class map does not exist: "
                    f"{raw_class_map_path}"
                )
            response["raw_class_map_path"] = raw_class_map_path
        response["visualization_path"] = str(
            data.get("viz_path") or data.get("output_path") or ""
        )
        return response
    if dataset_kind == "prompted_segmentation_json":
        masks_path = str(data.get("masks_path", "") or "")
        masks = data.get("masks")
        if not masks_path and masks is not None:
            masks_path = _persist_numpy_output(
                masks, artifact_dir / case.name / f"{source}_masks.npy"
            )
        num_masks = int(data.get("num_masks", len(masks) if masks is not None else 0))
        stderr = str(metadata.get("stderr", "") or "")
        empty_prediction = num_masks == 0 and (
            masks is not None or "produced no masks" in stderr.lower()
        )
        if (not masks_path or not Path(masks_path).is_file()) and not empty_prediction:
            raise RuntimeError(f"{source} prompted segmentation produced no masks")
        scores = data.get("mask_scores") or data.get("iou_scores") or data.get("scores") or []
        response.update(
            {
                "masks_path": masks_path,
                "mask_scores": [float(value) for value in scores],
                "num_masks": num_masks,
                "empty_prediction": empty_prediction,
                "point_x": prompt_row.get("point_x"),
                "point_y": prompt_row.get("point_y"),
                "text_prompt": str(prompt_row.get("text_prompt", "")),
                "segmented_image_path": str(data.get("segmented_image_path", "") or ""),
            }
        )
        return response
    raise ValueError(f"Unsupported vision task dataset kind {dataset_kind!r}")


def run_vision_hf_reference(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, reference, _runner = _load_vision_validation_plugins(work_dir)
    manifest = work_manifest(work_dir)
    dataset_kind = str(manifest.get("dataset_kind", ""))
    validation_config = manifest.get("task_eval", {})
    validation_config = validation_config if isinstance(validation_config, dict) else {}
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "hf_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            case = _vision_case_for_request(template, prompt_row, validation_config, index)
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                hf_python=sys.executable,
                reference_python=sys.executable,
            )
            output = reference.run_stage(
                case, _vision_full_inference_stage(case), context
            )
            response = _vision_response(
                case=case,
                source="hf",
                output=output,
                dataset_kind=dataset_kind,
                prompt_row=prompt_row,
                artifact_dir=artifacts_dir,
            )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.vision_hf] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_vision_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, _reference, runner = _load_vision_validation_plugins(work_dir)
    manifest = work_manifest(work_dir)
    dataset_kind = str(manifest.get("dataset_kind", ""))
    validation_config = manifest.get("task_eval", {})
    validation_config = validation_config if isinstance(validation_config, dict) else {}
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            case = _vision_case_for_request(template, prompt_row, validation_config, index)
            case.bundle = bundle_path.name
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            output = runner.run_stage(
                case, _vision_full_inference_stage(case), context
            )
            _record_output_native_command(work_dir, case.name, output)
            response = _vision_response(
                case=case,
                source="trtfb",
                output=output,
                dataset_kind=dataset_kind,
                prompt_row=prompt_row,
                artifact_dir=artifacts_dir,
            )
            if response["returncode"] != 0 and not response.get("empty_prediction"):
                raise RuntimeError(
                    f"TRT vision run failed for {case.name}: "
                    f"returncode={response['returncode']}"
                )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.vision_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def _load_reranking_validation_plugins(work_dir: Path) -> tuple[Any, Any, Any, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    manifest_ref = str(validation_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("reranking reference-consistency validation requires task_eval.model_manifest")
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    case = load_manifest(manifest_path)
    model_test_dir = str(case.metadata.get("model_test_dir", "") or "")
    activate_model_plugins(model_test_dir)
    reference = get_reference(case.reference_backend)
    runner = get_runner(case.task_strategy)
    comparator = get_comparator(case.task_strategy)
    if reference is None:
        raise RuntimeError(
            f"No reference plugin {case.reference_backend!r} for {case.family}"
        )
    if runner is None:
        raise RuntimeError(f"No runner plugin {case.task_strategy!r} for {case.family}")
    if comparator is None:
        raise RuntimeError(
            f"No comparator plugin {case.task_strategy!r} for {case.family}"
        )
    return case, reference, runner, comparator


def _reranking_case_for_request(
    template: Any, prompt_row: dict[str, Any], index: int
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"reranking_{index:06d}"))
    case.inputs["prompt"] = str(prompt_row["query"])
    case.inputs["documents"] = [str(document) for document in prompt_row["documents"]]
    return case


def _reranking_full_inference_stage(case: Any) -> Any:
    from tests.e2e_harness.contracts import StageSpec

    for stage in case.stages:
        if stage.name == "full_inference":
            return stage
    return StageSpec(name="full_inference", required=True)


def _reranking_response(
    *, case: Any, source: str, output: Any, prompt_row: dict[str, Any]
) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    scores = data.get("scores")
    if not isinstance(scores, list) or len(scores) != len(prompt_row["documents"]):
        raise RuntimeError(
            f"{source} reranking produced {len(scores) if isinstance(scores, list) else 0} "
            f"scores for {len(prompt_row['documents'])} documents"
        )
    return {
        "sample_id": case.name,
        "source": source,
        "query": str(prompt_row["query"]),
        "documents": [str(document) for document in prompt_row["documents"]],
        "scores": [float(score) for score in scores],
        "wall_ms": float(output.timing_s) * 1000.0,
    }


def _write_reranking_run_metadata(log_file: Any, sample_id: str, output: Any) -> None:
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    log_file.write(
        json.dumps({"sample_id": sample_id, "metadata": metadata}, ensure_ascii=False)
        + "\n"
    )
    log_file.flush()


def run_reranking_hf_reference(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, reference, _runner, _comparator = _load_reranking_validation_plugins(
        work_dir
    )
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "hf_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    metadata_path = work_dir / "hf_reference_metadata.jsonl"
    responses: list[dict[str, Any]] = []
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            case = _reranking_case_for_request(template, prompt_row, index)
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                hf_python=sys.executable,
                reference_python=sys.executable,
            )
            output = reference.run_stage(
                case, _reranking_full_inference_stage(case), context
            )
            response = _reranking_response(
                case=case, source="hf", output=output, prompt_row=prompt_row
            )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            _write_reranking_run_metadata(metadata_file, case.name, output)
            print(
                f"[validation.reranking_hf] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_reranking_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, _reference, runner, _comparator = _load_reranking_validation_plugins(
        work_dir
    )
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    metadata_path = work_dir / (args.log or "trtfb_run.log")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            case = _reranking_case_for_request(template, prompt_row, index)
            case.bundle = bundle_path.name
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            output = runner.run_stage(
                case, _reranking_full_inference_stage(case), context
            )
            _record_output_native_command(work_dir, case.name, output)
            response = _reranking_response(
                case=case, source="trtfb", output=output, prompt_row=prompt_row
            )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            _write_reranking_run_metadata(metadata_file, case.name, output)
            print(
                f"[validation.reranking_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def _load_diffusion_validation_plugins(work_dir: Path) -> tuple[Any, Any, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    manifest_ref = str(validation_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError(
            "diffusion reference-consistency validation requires task_eval.model_manifest in the work manifest"
        )
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    case = load_manifest(manifest_path)
    model_test_dir = str(case.metadata.get("model_test_dir", "") or "")
    activate_model_plugins(model_test_dir)
    reference = get_reference(case.reference_backend)
    runner = get_runner(case.task_strategy)
    comparator = get_comparator(case.task_strategy)
    if reference is None:
        raise RuntimeError(
            f"No reference plugin {case.reference_backend!r} for {case.family}"
        )
    if runner is None:
        raise RuntimeError(f"No runner plugin {case.task_strategy!r} for {case.family}")
    if comparator is None:
        raise RuntimeError(
            f"No comparator plugin {case.task_strategy!r} for {case.family}"
        )
    return case, reference, runner


def _diffusion_case_for_prompt(
    template: Any,
    prompt_row: dict[str, Any],
    generation: dict[str, Any],
    index: int,
) -> Any:
    case = copy.deepcopy(template)
    case.name = str(prompt_row.get("sample_id", f"diffusion_{index:06d}"))
    case.inputs.update(generation)
    case.inputs["prompt"] = str(prompt_row["prompt"])
    for field in _DIFFUSION_SAMPLE_INPUT_FIELDS:
        if field in prompt_row:
            case.inputs[field] = copy.deepcopy(prompt_row[field])
    seed = int(generation.get("seed", case.determinism.get("seed", 42)))
    case.inputs["seed"] = seed + index
    return case


def _diffusion_end_to_end_stage(case: Any) -> Any:
    from tests.e2e_harness.contracts import StageSpec

    for stage in case.stages:
        if stage.name == "end_to_end":
            return stage
    return StageSpec(name="end_to_end", required=True)


def _diffusion_response(
    sample_id: str,
    source: str,
    output: Any,
    *,
    case: Any | None = None,
) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    response = {
        "sample_id": sample_id,
        "source": source,
        "returncode": int(data.get("returncode", 1)),
        "num_frames": int(data.get("num_frames", 0)),
        "frames_dir": str(data.get("frames_dir", "")),
        "frame_stats": data.get("frame_stats", {}),
        "prompt": str(data.get("prompt", "")),
        "initial_latents_sha256": str(data.get("initial_latents_sha256", "")),
        "wall_ms": float(output.timing_s) * 1000.0,
    }
    if data.get("native_acceptance") is not None:
        response["native_acceptance"] = copy.deepcopy(data["native_acceptance"])
    if case is not None:
        response["seed"] = int(case.inputs.get("seed", case.determinism.get("seed", 42)))
        response["action"] = str(case.inputs.get("action", ""))
        image_value = str(
            case.inputs.get("image") or case.inputs.get("image_path") or ""
        )
        response["condition_image"] = image_value
        image_path = Path(image_value) if image_value else None
        if image_path is not None and image_path.is_file():
            response["condition_image_sha256"] = _sha256_file(image_path)
    return response


def run_diffusion_hf_reference(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, reference, _runner = _load_diffusion_validation_plugins(work_dir)
    requested_precision = _REFERENCE_DTYPE_TO_PRECISION.get(
        str(getattr(args, "dtype", "auto"))
    )
    if requested_precision:
        template.metadata["reference_precision"] = requested_precision
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    generation = generation_defaults(work_dir)
    artifacts_dir = work_dir / "hf_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            case = _diffusion_case_for_prompt(template, prompt_row, generation, index)
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                hf_python=sys.executable,
                reference_python=sys.executable,
            )
            output = reference.run_stage(
                case, _diffusion_end_to_end_stage(case), context
            )
            response = _diffusion_response(case.name, "hf", output, case=case)
            response["prompt"] = str(prompt_row["prompt"])
            if response["returncode"] != 0 or response["num_frames"] < 1:
                raise RuntimeError(
                    f"HF diffusion reference failed for {case.name}: "
                    f"returncode={response['returncode']} frames={response['num_frames']}"
                )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.diffusion_hf] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_tts_hf_reference(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("TTS run-hf requires numpy and torch") from exc

    work_dir = Path(args.work_dir)
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    if len(prompt_rows) != len(answers["requests"]):
        raise ValueError("answers.json and prompts.jsonl must contain the same number of samples")
    defaults = generation_defaults(work_dir)
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 42))
    device = torch.device(args.device)
    output_dir = work_dir / "hf_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_magpie = "magpie" in args.model.lower()

    if is_magpie:
        try:
            from nemo.collections.tts.models import MagpieTTSModel
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Magpie TTS reference requires NeMo MagpieTTSModel") from exc
        model = MagpieTTSModel.from_pretrained(args.model).eval().to(device)
        processor = None
    else:
        try:
            from transformers import AutoProcessor, BarkModel, logging
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Bark TTS reference requires transformers BarkModel") from exc
        logging.set_verbosity_error()
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        dtype = _model_dtype(torch, args.dtype)
        model = (
            BarkModel.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                torch_dtype=dtype,
            )
            .eval()
            .to(device)
        )

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, prompt_row in enumerate(prompt_rows):
            prompt = str(prompt_row.get("prompt", ""))
            sample_id = str(prompt_row.get("sample_id", f"seedtts_{idx:06d}"))
            sample_seed = seed + idx
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
            start = time.perf_counter()
            with torch.inference_mode():
                if is_magpie:
                    audio_tensor, audio_len = model.do_tts(
                        transcript=prompt,
                        language=str(prompt_row.get("language", "en") or "en"),
                        use_cfg=True,
                    )
                    audio = audio_tensor.detach().cpu().numpy().reshape(-1)
                    actual_len = int(audio_len.item()) if audio_len.numel() else len(audio)
                    audio = audio[:actual_len]
                    sample_rate = 22050
                else:
                    inputs = processor(prompt, return_tensors="pt")
                    inputs = _to_device(inputs, device)
                    audio_values = model.generate(**inputs)
                    audio = audio_values.detach().cpu().numpy().reshape(-1)
                    sample_rate = int(model.generation_config.sample_rate)
            wall_ms = (time.perf_counter() - start) * 1000.0
            wav_path = output_dir / f"{_safe_sample_filename(sample_id, '.wav')}"
            _write_pcm16_wav(wav_path, np.asarray(audio), sample_rate)
            row = _tts_response_row(sample_id, wav_path, wall_ms, "hf")
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.tts_hf] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scoring = work_scoring(work_dir)
    transcripts = _transcribe_audio_files(
        [Path(row["wav_path"]) for row in responses],
        python=sys.executable,
        model_id=str(scoring.get("asr_model", "openai/whisper-large-v3-turbo")),
        local_files_only=args.local_files_only,
    )
    for row, transcript in zip(responses, transcripts, strict=True):
        row["output_text"] = transcript
        row["asr_transcript"] = transcript
    write_predictions(pred_path, responses)
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for row in responses:
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def encoder_reference_class_names(reference_family: str) -> tuple[str, str]:
    if reference_family == "dpr_context_embed":
        return "DPRContextEncoder", "DPRContextEncoderTokenizerFast"
    return "AutoModel", "AutoTokenizer"


def load_hf_text_generation_model(
    transformers_module: Any,
    model_id: str,
    *,
    model_kwargs: dict[str, Any],
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, bool]:
    """Load the matching causal or encoder-decoder HF generation class."""
    config = transformers_module.AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    model_class = (
        transformers_module.AutoModelForSeq2SeqLM
        if is_encoder_decoder
        else transformers_module.AutoModelForCausalLM
    )
    return model_class.from_pretrained(model_id, **model_kwargs).eval(), is_encoder_decoder


def run_encoder_embedding_hf_reference(args: argparse.Namespace) -> None:
    try:
        import torch
        import transformers
        from transformers import logging
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("encoder/embedding HF reference requires torch and transformers") from exc

    work_dir = Path(args.work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    task_config = work_manifest(work_dir).get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    vector_mode = "embedding" if task_config.get("task_strategy") == "embedding" else "cls"
    model_class_name, tokenizer_class_name = encoder_reference_class_names(
        str(args.reference_family or "")
    )
    model_class = getattr(transformers, model_class_name)
    tokenizer_class = getattr(transformers, tokenizer_class_name)

    logging.set_verbosity_error()
    tokenizer = tokenizer_class.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    dtype: str | Any = "auto"
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = model_class.from_pretrained(args.model, **model_kwargs).eval()
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    else:
        device = model.device

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            encoded = tokenizer(
                str(prompt_row["prompt"]),
                return_tensors="pt",
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**encoded, output_hidden_states=True)
            wall_ms = (time.perf_counter() - start) * 1000.0
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    hidden = hidden_states[-1]
                elif isinstance(outputs, (tuple, list)) and outputs:
                    hidden = outputs[0]
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError(
                    f"HF encoder output for {prompt_row['sample_id']} has no rank-3 hidden state"
                )
            if vector_mode == "embedding":
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                vector_tensor = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vector_tensor = torch.nn.functional.normalize(vector_tensor, p=2, dim=-1)[0]
            else:
                vector_tensor = hidden[0, 0]
            row = {
                "sample_id": str(prompt_row["sample_id"]),
                "pair_id": str(prompt_row["pair_id"]),
                "pair_side": str(prompt_row["pair_side"]),
                "score": float(prompt_row["score"]),
                "vector_mode": vector_mode,
                "vector": vector_tensor.float().cpu().numpy().tolist(),
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.encoder_hf] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hf_reference(args: argparse.Namespace) -> None:
    dataset_kind = _work_dataset_kind(Path(args.work_dir))
    if _is_reranking_dataset_kind(dataset_kind):
        run_reranking_hf_reference(args)
        return
    if _is_time_series_dataset_kind(dataset_kind):
        run_time_series_hf_reference(args)
        return
    if _is_vision_task_dataset_kind(dataset_kind):
        run_vision_hf_reference(args)
        return
    if _is_encoder_embedding_dataset_kind(dataset_kind):
        run_encoder_embedding_hf_reference(args)
        return
    if _is_diffusion_text_dataset_kind(dataset_kind):
        run_diffusion_text_hf_reference(args)
        return
    if _is_tts_dataset_kind(dataset_kind):
        run_tts_hf_reference(args)
        return
    if _is_vlm_dataset_kind(dataset_kind):
        run_vlm_hf_reference(args)
        return
    if _is_asr_dataset_kind(dataset_kind):
        run_asr_hf_reference(args)
        return
    if _is_diffusion_media_dataset_kind(dataset_kind):
        run_diffusion_hf_reference(args)
        return
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("run-hf requires torch and transformers") from exc

    work_dir = Path(args.work_dir)
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    if len(prompt_rows) != len(answers["requests"]):
        raise ValueError("answers.json and prompts.jsonl must contain the same number of samples")
    defaults = generation_defaults(work_dir)
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 1))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(defaults.get("temperature", 1.0))
    )
    top_k = args.top_k if args.top_k is not None else int(defaults.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(defaults.get("top_p", 1.0))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", -1))
    do_sample = args.do_sample or bool(defaults.get("do_sample", False))
    apply_chat_template = args.apply_chat_template or bool(
        defaults.get("apply_chat_template", answers.get("apply_chat_template", False))
    )
    generation_overrides = hf_generation_overrides(work_dir)

    transformers.logging.set_verbosity_error()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype: str | Any = "auto"
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model, is_encoder_decoder = load_hf_text_generation_model(
        transformers,
        args.model,
        model_kwargs=model_kwargs,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    else:
        device = model.device

    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    pred_path = work_dir / (args.predictions or "hf_predictions.json")
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, request in enumerate(answers["requests"]):
            prompt = str(prompt_rows[idx].get("prompt") or _request_prompt(request))
            if apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    request["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if seed >= 0:
                torch.manual_seed(seed + idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + idx)
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_overrides,
                )
            wall_ms = (time.perf_counter() - start) * 1000.0
            if is_encoder_decoder:
                generated = output_ids[0]
                decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
                if (
                    decoder_start_token_id is not None
                    and generated.numel() > 0
                    and int(generated[0]) == int(decoder_start_token_id)
                ):
                    generated = generated[1:]
                output_text = tokenizer.decode(generated, skip_special_tokens=True)
            else:
                generated = output_ids[0, encoded["input_ids"].shape[1] :]
                output_text = tokenizer.decode(generated, skip_special_tokens=False)
            generated_token_ids = [int(token_id) for token_id in generated.tolist()]
            row = {
                "sample_id": prompt_rows[idx].get("sample_id", f"mmlu_{idx:06d}"),
                "output_text": output_text,
                "generated_tokens": int(generated.shape[0]),
                "generated_token_ids": generated_token_ids,
                "wall_ms": wall_ms,
                "source": "hf",
            }
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr)
    write_predictions(pred_path, responses)
    # Release the HF torch model and its GPU allocations so any in-process TRT
    # build/run that follows does not contend with a resident reference model.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_diffusion_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, _reference, runner = _load_diffusion_validation_plugins(work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    generation = generation_defaults(work_dir)
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (getattr(args, "log", "") or "trtfb_run.log")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            case = _diffusion_case_for_prompt(template, prompt_row, generation, index)
            case.bundle = bundle_path.name
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            output = runner.run_stage(
                case, _diffusion_end_to_end_stage(case), context
            )
            _record_output_native_command(work_dir, case.name, output)
            command = (
                output.metadata.get("command")
                if isinstance(output.metadata, dict)
                else None
            )
            if isinstance(command, list) and command:
                log_file.write(
                    f"$ {shlex.join(str(token) for token in command)}\n"
                )
            elif isinstance(command, str) and command:
                log_file.write(f"$ {command}\n")
            log_file.flush()
            response = _diffusion_response(case.name, "trtfb", output, case=case)
            response["prompt"] = str(prompt_row["prompt"])
            if response["returncode"] != 0 or response["num_frames"] < 1:
                raise RuntimeError(
                    f"TRT diffusion run failed for {case.name}: "
                    f"returncode={response['returncode']} frames={response['num_frames']}"
                )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.diffusion_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


_ELF_REPLAY_INPUT_KEYS = {
    "elf_replay_artifact",
    "upstream_replay_artifact",
    "initial_latents_raw",
    "initial_latents_path",
    "condition_latents_raw",
    "condition_latents_path",
    "condition_mask_raw",
    "condition_mask_path",
    "sampling_steps_raw",
    "sampling_steps_path",
    "sde_noise_raw",
    "sde_noise_path",
    "expected_generated_jsonl_path",
    "expected_jsonl_path",
    "expected_generated_samples",
    "replay_samples",
}


def _load_diffusion_text_validation_runner(work_dir: Path) -> tuple[Any, Any]:
    validation_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    manifest_ref = str(validation_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError("diffusion text reference-consistency validation requires task_eval.model_manifest")
    manifest_path = Path(manifest_ref)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    case = load_manifest(manifest_path)
    activate_model_plugins(str(case.metadata.get("model_test_dir", "") or ""))
    runner = get_runner(case.task_strategy)
    if runner is None:
        raise RuntimeError(f"No runner plugin {case.task_strategy!r} for {case.family}")
    return case, runner


def run_diffusion_text_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext, StageSpec

    work_dir = Path(args.work_dir)
    template, runner = _load_diffusion_text_validation_runner(work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    generation = generation_defaults(work_dir)
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    bundle_path = Path(args.bundle).resolve()
    base_seed = int(generation.get("seed", 42))
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for index, prompt_row in enumerate(prompt_rows):
            case = copy.deepcopy(template)
            for key in _ELF_REPLAY_INPUT_KEYS:
                case.inputs.pop(key, None)
            sample_id = str(prompt_row.get("sample_id", f"diffusion_text_{index:06d}"))
            source_text = str(prompt_row.get("source_text", prompt_row.get("prompt", "")))
            dataset_index = int(prompt_row.get("dataset_index", index))
            case.name = sample_id
            case.bundle = bundle_path.name
            case.inputs.update(generation)
            case.inputs.update(
                {
                    "prompt": source_text,
                    "source_text": source_text,
                    "num_samples": 1,
                    "seed": int(prompt_row.get("seed", base_seed + dataset_index)),
                }
            )
            shared_dir = work_dir / "hf_shared_inputs" / sample_id
            initial_latents = shared_dir / "initial_latents.f32"
            sampling_steps = shared_dir / "sampling_steps.f32"
            sde_noises = shared_dir / "sde_noises.f32"
            missing = [
                str(path) for path in (initial_latents, sampling_steps) if not path.is_file()
            ]
            if str(generation.get("sampling_method", "ode")) == "sde" and not sde_noises.is_file():
                missing.append(str(sde_noises))
            if missing:
                raise RuntimeError(
                    "ELF TRTMC parity requires HF-exported shared sampling inputs; missing "
                    + ", ".join(missing)
                )
            case.inputs["initial_latents_raw"] = str(initial_latents)
            case.inputs["sampling_steps_raw"] = str(sampling_steps)
            shared_sampling_inputs = {
                "initial_latents": str(initial_latents),
                "sampling_steps": str(sampling_steps),
            }
            if sde_noises.is_file():
                case.inputs["sde_noise_raw"] = str(sde_noises)
                shared_sampling_inputs["sde_noises"] = str(sde_noises)
            context = RunContext(
                case=case,
                artifacts_dir=str(artifacts_dir),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(getattr(args, "model_plugin_dir", "") or ""),
            )
            output = runner.run_stage(case, StageSpec(name="decoded_text", required=True), context)
            _record_output_native_command(work_dir, sample_id, output)
            generated_samples = output.data.get("generated_samples", [])
            generated = generated_samples[0] if generated_samples else {}
            output_text = str(
                generated.get("generated", "") if isinstance(generated, dict) else output.text or ""
            ).strip()
            token_ids = (
                generated.get("token_ids")
                if isinstance(generated, dict) and isinstance(generated.get("token_ids"), list)
                else []
            )
            response = {
                "sample_id": sample_id,
                "output_text": output_text,
                "generated_token_ids": [int(token_id) for token_id in token_ids],
                "wall_ms": float(output.timing_s) * 1000.0,
                "source": "trtfb",
                "seed": case.inputs["seed"],
                "shared_sampling_inputs": shared_sampling_inputs,
            }
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.diffusion_text_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_diffusion_text_hf_reference(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    manifest = work_manifest(work_dir)
    task_config = manifest.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    reference = task_config.get("reference", {})
    reference = reference if isinstance(reference, dict) else {}
    reference_repo = str(getattr(args, "elf_reference_repo", "") or "")
    if not reference_repo:
        raise ValueError("ELF HF reference requires --elf-reference-repo")
    config_value = str(reference.get("config", "") or "")
    checkpoint = str(reference.get("checkpoint", args.model) or args.model)
    if not config_value:
        raise ValueError("ELF HF reference requires reference.config in the suite")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = Path(reference_repo) / config_path

    prompts = load_jsonl(work_dir / "prompts.jsonl")
    answers = json.loads((work_dir / "answers.json").read_text(encoding="utf-8"))
    requests = answers.get("requests", [])
    if len(prompts) != len(requests):
        raise ValueError("ELF HF reference prompt/answer count mismatch")
    reference_dataset = work_dir / "hf_reference_dataset.jsonl"
    with reference_dataset.open("w", encoding="utf-8") as output:
        for prompt, request in zip(prompts, requests, strict=True):
            output.write(
                json.dumps(
                    {
                        "id": str(prompt.get("sample_id", "")),
                        "input": str(prompt.get("source_text", prompt.get("prompt", ""))),
                        "output": str(request.get("answer", "")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    generation = generation_defaults(work_dir)
    output_path = work_dir / (args.predictions or "hf_predictions.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "elf_hf_reference.py"),
        "--reference-repo",
        reference_repo,
        "--config",
        str(config_path),
        "--checkpoint",
        checkpoint,
        "--dataset",
        str(reference_dataset),
        "--output",
        str(output_path),
        "--shared-inputs-dir",
        str(work_dir / "hf_shared_inputs"),
        "--generation-mode",
        str(generation.get("generation_mode", "conditional")),
        "--sampling-method",
        str(generation.get("sampling_method", "ode")),
        "--num-steps",
        str(generation.get("num_sampling_steps", 64)),
        "--cfg-scale",
        str(generation.get("cfg_scale", 1.0)),
        "--self-cond-cfg-scale",
        str(generation.get("self_cond_cfg_scale", 1.0)),
        "--sde-gamma",
        str(generation.get("sde_gamma", 0.0)),
        "--seed",
        str(generation.get("seed", 42)),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        cwd=reference_repo,
    )
    (work_dir / "hf_reference_stdout.log").write_text(result.stdout, encoding="utf-8")
    (work_dir / "hf_reference_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"ELF HF reference failed rc={result.returncode}; "
            f"see {work_dir / 'hf_reference_stderr.log'}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    raw_path = work_dir / (args.raw_output or "hf_raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for row in payload.get("responses", []):
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _runtime_config_tokens(config: dict[str, Any], prefix: str = "") -> list[str]:
    tokens: list[str] = []
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            tokens.extend(_runtime_config_tokens(value, name))
        elif isinstance(value, bool):
            tokens.append(f"{name}={'true' if value else 'false'}")
        elif value is not None:
            tokens.append(f"{name}={value}")
    return tokens


def run_tts_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    defaults = generation_defaults(work_dir)
    task_config = work_manifest(work_dir).get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    output_dir = work_dir / "trtfb_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
    model_max_new_tokens = task_config.get("model_max_new_tokens")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(
            model_max_new_tokens
            if model_max_new_tokens is not None
            else defaults.get("max_new_tokens", 0)
        )
    )
    runtime_config = task_config.get("runtime_config", {})
    runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
    set_tokens = _runtime_config_tokens(runtime_config) + list(args.set or [])
    family = str(task_config.get("family", "") or "")
    arg_seed = getattr(args, "seed", None)
    seed = arg_seed if arg_seed is not None else int(defaults.get("seed", -1))
    has_explicit_bark_seed = any(
        token.split("=", 1)[0] == "audio_bark.seed" for token in set_tokens
    )
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_output.open("w", encoding="utf-8") as raw_f,
        log_path.open("w", encoding="utf-8") as log_f,
    ):
        for idx, prompt_row in enumerate(prompt_rows):
            sample_id = str(prompt_row.get("sample_id", f"seedtts_{idx:06d}"))
            wav_path = output_dir / _safe_sample_filename(sample_id, ".wav")
            cmd = [
                _trtmc_binary_from_args(args),
                "generate-audio",
                args.bundle,
                "--prompt",
                str(prompt_row.get("prompt", "")),
                "--output",
                str(wav_path),
            ]
            if max_new_tokens > 0:
                cmd.extend(["--max-new-tokens", str(max_new_tokens)])
            if args.hf_python:
                cmd.extend(["--hf-python", args.hf_python])
            if args.backend_dir:
                cmd.extend(["--backend-dir", args.backend_dir])
            if args.config:
                cmd.extend(["--config", args.config])
            sample_set_tokens = list(set_tokens)
            if family == "bark" and seed >= 0 and not has_explicit_bark_seed:
                sample_set_tokens.append(f"audio_bark.seed={seed + idx}")
            for token in sample_set_tokens:
                cmd.extend(["--set", token])
            _append_native_trtmc_command(work_dir, sample_id, cmd)
            log_f.write(f"$ {' '.join(cmd)}\n")
            start = time.perf_counter()
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
            wall_ms = (time.perf_counter() - start) * 1000.0
            if proc.stdout:
                log_f.write(proc.stdout)
                if not proc.stdout.endswith("\n"):
                    log_f.write("\n")
            if proc.stderr:
                log_f.write(proc.stderr)
                if not proc.stderr.endswith("\n"):
                    log_f.write("\n")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"TRTFB TTS reference-consistency validation failed for sample {idx} rc={proc.returncode}; see {log_path}"
                )
            if not wav_path.is_file():
                raise RuntimeError(
                    f"TRTFB TTS reference-consistency validation produced no WAV for sample {idx}: {wav_path}"
                )
            row = _tts_response_row(sample_id, wav_path, wall_ms, "trtfb")
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[validation.tts_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    scoring = work_scoring(work_dir)
    transcripts = _transcribe_audio_files(
        [Path(row["wav_path"]) for row in responses],
        python=str(args.hf_python or sys.executable),
        model_id=str(scoring.get("asr_model", "openai/whisper-large-v3-turbo")),
    )
    for row, transcript in zip(responses, transcripts, strict=True):
        row["output_text"] = transcript
        row["asr_transcript"] = transcript
    write_predictions(predictions, responses)
    with raw_output.open("w", encoding="utf-8") as raw_f:
        for row in responses:
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_encoder_vector_stdout(stdout: str, key: str) -> list[float]:
    try:
        data = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"TRTMC encoder output is not JSON: {stdout[:500]}") from exc
    vector = data.get(key) if isinstance(data, dict) else None
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"TRTMC encoder output does not contain non-empty {key!r}")
    return [float(value) for value in vector]


def run_encoder_embedding_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    task_config = work_manifest(work_dir).get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    vector_mode = "embedding" if task_config.get("task_strategy") == "embedding" else "cls"
    command_name = "embed" if vector_mode == "embedding" else "encode"
    output_key = "embedding" if vector_mode == "embedding" else "cls_embedding"
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        log_path.open("w", encoding="utf-8") as log_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            cmd = [
                _trtmc_binary_from_args(args),
                command_name,
                args.bundle,
                "--prompt",
                str(prompt_row["prompt"]),
            ]
            if args.hf_python:
                cmd.extend(["--hf-python", args.hf_python])
            if args.backend_dir:
                cmd.extend(["--backend-dir", args.backend_dir])
            if args.model_plugin_dir:
                cmd.extend(["--model-plugin-dir", args.model_plugin_dir])
            if args.config:
                cmd.extend(["--config", args.config])
            for token in args.set or []:
                cmd.extend(["--set", token])
            _append_native_trtmc_command(
                work_dir,
                str(prompt_row["sample_id"]),
                cmd,
            )
            start = time.perf_counter()
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
            wall_ms = (time.perf_counter() - start) * 1000.0
            log_file.write(f"$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}")
            log_file.flush()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"TRTMC {command_name} failed for {prompt_row['sample_id']} "
                    f"rc={proc.returncode}; see {log_path}"
                )
            row = {
                "sample_id": str(prompt_row["sample_id"]),
                "pair_id": str(prompt_row["pair_id"]),
                "pair_side": str(prompt_row["pair_side"]),
                "score": float(prompt_row["score"]),
                "vector_mode": vector_mode,
                "vector": _parse_encoder_vector_stdout(proc.stdout, output_key),
                "wall_ms": wall_ms,
                "source": "trtfb",
            }
            responses.append(row)
            raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_file.flush()
            print(
                f"[validation.encoder_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_model_plugin_trtfb(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    manifest = work_manifest(work_dir)
    manifest_path = manifest_path_from_work_manifest(
        manifest,
        repo_root=REPO_ROOT,
    )
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    metadata_path = work_dir / (args.log or "trtfb_run.log")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    _reset_native_trtmc_commands(work_dir)
    with (
        raw_path.open("w", encoding="utf-8") as raw_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        for index, prompt_row in enumerate(prompt_rows):
            case, stage = select_case(
                manifest_path,
                prompt_row,
                source_index=index,
            )
            case.bundle = bundle_path.name
            activate_model_plugins(
                str(case.metadata.get("model_test_dir", "") or "")
            )
            runner = get_runner(case.task_strategy)
            if runner is None:
                raise RuntimeError(
                    f"No runner plugin {case.task_strategy!r} for {case.family}"
                )
            sample_id = str(
                prompt_row.get("sample_id", f"model_plugin_{index:06d}")
            )
            sample_artifacts = artifacts_dir / sample_id
            context = RunContext(
                case=case,
                artifacts_dir=str(sample_artifacts),
                binary_path=str(args.trtmc_binary),
                hf_python=str(getattr(args, "hf_python", "") or ""),
                runtime_python=str(getattr(args, "hf_python", "") or ""),
                reference_python=str(getattr(args, "hf_python", "") or ""),
                engine_dir=str(bundle_path.parent),
                model_plugin_dir=str(
                    getattr(args, "model_plugin_dir", "") or ""
                ),
            )
            output = runner.run_stage(case, stage, context)
            _record_output_native_command(work_dir, sample_id, output)
            serialized = serialize_stage_output(
                output,
                artifact_dir=sample_artifacts / "serialized",
                sample_id=sample_id,
            )
            response = response_from_output(
                sample_id=sample_id,
                source="trtfb",
                testcase=case.metadata["validation_manifest_case_name"],
                output=output,
                serialized_output=serialized,
            )
            responses.append(response)
            raw_file.write(json.dumps(response, ensure_ascii=False) + "\n")
            raw_file.flush()
            metadata_file.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "testcase": case.metadata[
                            "validation_manifest_case_name"
                        ],
                        "stage": stage.name,
                        "timing_s": float(output.timing_s),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            metadata_file.flush()
            print(
                f"[validation.model_plugin_trtfb] "
                f"sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def run_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    dataset_kind = _work_dataset_kind(work_dir)
    if _is_model_plugin_dataset_kind(dataset_kind):
        run_model_plugin_trtfb(args)
        return
    if _is_reranking_dataset_kind(dataset_kind):
        run_reranking_trtfb(args)
        return
    if _is_time_series_dataset_kind(dataset_kind):
        run_time_series_trtfb(args)
        return
    if _is_vision_task_dataset_kind(dataset_kind):
        run_vision_trtfb(args)
        return
    if _is_encoder_embedding_dataset_kind(dataset_kind):
        run_encoder_embedding_trtfb(args)
        return
    if _is_tts_dataset_kind(dataset_kind):
        run_tts_trtfb(args)
        return
    if _is_vlm_dataset_kind(dataset_kind):
        run_vlm_trtfb(args)
        return
    if _is_asr_dataset_kind(dataset_kind):
        run_asr_trtfb(args)
        return
    if _is_diffusion_media_dataset_kind(dataset_kind):
        run_diffusion_trtfb(args)
        return
    if _is_diffusion_text_dataset_kind(dataset_kind):
        run_diffusion_text_trtfb(args)
        return
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(defaults.get("max_new_tokens", 1))
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else float(defaults.get("temperature", 1.0))
    )
    top_k = args.top_k if args.top_k is not None else int(defaults.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(defaults.get("top_p", 1.0))
    min_p = args.min_p if args.min_p is not None else float(defaults.get("min_p", 0.0))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", -1))

    cmd = [
        args.benchmark_binary,
        args.bundle,
        str(work_dir / "prompts.jsonl"),
        str(raw_output),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--min-p",
        str(min_p),
    ]
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    if args.hf_python:
        cmd.extend(["--hf-python", args.hf_python])
    if args.backend_dir:
        cmd.extend(["--backend-dir", args.backend_dir])
    if getattr(args, "model_plugin_dir", ""):
        cmd.extend(["--model-plugin-dir", args.model_plugin_dir])
    if args.kv_cache_size:
        cmd.extend(["--kv-cache-size", args.kv_cache_size])
    if args.config:
        cmd.extend(["--config", args.config])
    for token in args.set or []:
        cmd.extend(["--set", token])
    if args.chat_template or bool(defaults.get("apply_chat_template", False)):
        cmd.append("--chat-template")
    if not bool(defaults.get("enable_thinking", True)):
        cmd.append("--no-thinking")

    _write_dataset_benchmark_reproduction(work_dir, cmd)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    log_path = work_dir / (args.log or "trtfb_run.log")
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {shlex.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(
            cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(f"TRTFB reference-consistency validation failed with rc={proc.returncode}; see {log_path}")
    convert_trtfb_jsonl_to_predictions(raw_output, predictions)


def _manifest_build_method(build_args: dict[str, Any]) -> str | None:
    backend = str(build_args.get("backend", build_args.get("method", "")) or "").lower()
    if backend in {"torchtrt", "torch_trt"} or build_args.get("torch_trt", False):
        return "torchtrt"
    if backend == "trt":
        return "trt"
    return None


def _manifest_tensor_parallel_size(build_args: dict[str, Any]) -> int | None:
    parallel = build_args.get("parallel", {})
    if isinstance(parallel, dict):
        value = parallel.get("tp_size", parallel.get("tensor_parallel_size"))
        mode = str(parallel.get("mode", "") or "").lower()
    else:
        value = build_args.get("tp_size", build_args.get("tensor_parallel_size"))
        mode = str(build_args.get("parallel_mode", "") or "").lower()
    if value is None:
        return None
    try:
        tp_size = int(value)
    except (TypeError, ValueError):
        return None
    if tp_size > 1 or mode == "tensor_parallel":
        return tp_size
    return None


def _model_asset_path(model: dict[str, Any], value: str) -> Path:
    asset = Path(value)
    if asset.is_absolute():
        return asset
    manifest = Path(str(model.get("manifest", "") or ""))
    if manifest:
        if not manifest.is_absolute():
            manifest = REPO_ROOT / manifest
        model_dir = (
            manifest.parent.parent
            if manifest.parent.name == "manifests"
            else manifest.parent
        )
        if asset.parts[:3] == ("tests", "e2e", "data"):
            candidate = model_dir / "data" / asset.name
        elif asset.parts and asset.parts[0] == "data":
            candidate = model_dir / asset
        else:
            candidate = model_dir / "data" / asset
        if candidate.is_file():
            return candidate
    return REPO_ROOT / "tests" / "e2e" / "data" / asset


def build_bundle_command(
    model: dict[str, Any],
    *,
    trtmc_binary: str,
    bundle_path: Path,
    max_cache_length: int | None = None,
    extra_build_args: list[str] | None = None,
) -> list[str]:
    cache_length = int(max_cache_length or model.get("max_cache_length", 256))
    cmd = [
        trtmc_binary,
        "build",
        str(model["hf_id"]),
    ]
    hf_revision = str(model.get("hf_revision", "") or "")
    if hf_revision:
        cmd.extend(["--model-revision", hf_revision])
    cmd.extend([
        "-o",
        str(bundle_path),
        "--max-cache-length",
        str(cache_length),
    ])
    build_args = model.get("build_args", {})
    method = _manifest_build_method(build_args)
    if method:
        cmd.extend(["--method", method])
    decoder_engine_layout = build_args.get("decoder_engine_layout")
    if decoder_engine_layout is not None:
        cmd.extend(["--decoder-engine-layout", str(decoder_engine_layout)])
    tp_size = _manifest_tensor_parallel_size(build_args)
    if tp_size is not None and tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])
    precision = str(model.get("precision", "fp32") or "fp32")
    if precision != "fp32":
        cmd.extend(["--precision", precision])
    fp32_layers = model.get("fp32_layers") or []
    if fp32_layers:
        cmd.extend(
            ["--fp32-layers", ",".join(str(layer) for layer in fp32_layers)])
    quantization = model.get("quantization", {})
    if isinstance(quantization, dict):
        quant_format = quantization.get("format")
        if quant_format and quant_format != "none":
            cmd.extend(["--quantize", str(quant_format)])
            scale_artifact = quantization.get("scale_artifact")
            if scale_artifact:
                cmd.extend(["--quant-scales", str(scale_artifact)])
            calibration_samples = quantization.get("calibration_samples")
            if calibration_samples is not None:
                cmd.extend(["--quant-calibration-samples", str(calibration_samples)])
    if model.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    for key, flag in (
        ("image_height", "--image-height"),
        ("image_width", "--image-width"),
        ("video_num_frames", "--video-num-frames"),
        ("video_height", "--video-height"),
        ("video_width", "--video-width"),
        ("num_inference_steps", "--num-inference-steps"),
    ):
        value = model.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    fp8_scales = model.get("fp8_scales")
    if fp8_scales:
        scales_path = _model_asset_path(model, str(fp8_scales))
        if scales_path.is_file():
            cmd.extend(["--fp8-scales", str(scales_path)])
    if extra_build_args:
        cmd.extend(extra_build_args)
    return cmd


def requested_build_max_cache_length(
    suite: dict[str, Any],
    model: dict[str, Any],
    build_max_cache_length: int | None = None,
    prompt_max_tokens: int | None = None,
) -> int:
    if build_max_cache_length is not None:
        return int(build_max_cache_length)
    model_cache = int(model.get("max_cache_length", 256) or 256)
    suite_cache = int(suite.get("build", {}).get("min_max_cache_length", 0) or 0)
    prompt_cache = int(prompt_max_tokens or 0)
    return max(model_cache, suite_cache, prompt_cache)


def generation_cache_headroom(
    *,
    scorer: str,
    validation_config: dict[str, Any],
    generation: dict[str, Any],
    max_new_tokens: int | None,
) -> int:
    # A prompt that exactly fills the measured cache still needs room for the
    # first generated token. Reserve the declared generation budget unless a
    # non-continuation workload explicitly proves that it does not generate.
    reserve_headroom = scorer == "continuation" or bool(
        validation_config.get("build_generation_headroom", True)
    )
    if not reserve_headroom:
        return 0
    return int(
        max_new_tokens
        if max_new_tokens is not None
        else generation.get("max_new_tokens", 0)
    )


def bundle_inspection(
    bundle_path: Path,
    trtmc_binary: str,
) -> dict[str, str]:
    try:
        proc = subprocess.run(
            [trtmc_binary, "inspect", str(bundle_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    values = {}
    for label, value in re.findall(r"^([^:\n]+):\s+(.+?)\s*$", proc.stdout, re.MULTILINE):
        values[label.strip()] = value.strip()
    return values


def bundle_max_cache_length(bundle_path: Path, trtmc_binary: str) -> int | None:
    value = bundle_inspection(bundle_path, trtmc_binary).get("Max cache length")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def runtime_tensorrt_abi() -> str:
    try:
        import tensorrt
    except Exception:
        return ""
    parts = str(tensorrt.__version__).split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else ""


def _bundle_can_be_reused(
    inspection: Mapping[str, str],
    *,
    max_cache_length: int | None,
    expected_precision: str,
    allow_unknown: bool,
) -> bool:
    if not inspection:
        return allow_unknown
    raw_cache = inspection.get("Max cache length")
    if max_cache_length is not None and raw_cache is not None:
        try:
            if int(raw_cache) < max_cache_length:
                return False
        except ValueError:
            return False
    raw_precision = inspection.get("Precision")
    if expected_precision:
        if raw_precision is None:
            if not allow_unknown:
                return False
        else:
            try:
                bundle_precision = _canonical_reference_precision(
                    raw_precision,
                    field="bundle precision",
                )
            except ValueError:
                return False
            if bundle_precision != expected_precision:
                return False
    bundle_abi = inspection.get("TRT ABI", "")
    runtime_abi = runtime_tensorrt_abi()
    return not bundle_abi or not runtime_abi or bundle_abi == runtime_abi


def _remove_bundle_before_or_after_replacement(
    bundle_path: Path,
    replace_existing: bool,
) -> None:
    if replace_existing and (bundle_path.exists() or bundle_path.is_symlink()):
        bundle_path.unlink()


def ensure_bundle(
    model: dict[str, Any],
    *,
    bundle_path: Path,
    trtmc_binary: str,
    max_cache_length: int | None = None,
    force_build: bool = False,
    replace_existing: bool = False,
    extra_build_args: list[str] | None = None,
    log_path: Path | None = None,
    cuda_visible_devices: str = "",
) -> tuple[Path, bool]:
    expected_precision = _canonical_reference_precision(
        model.get("precision", "fp32"),
        field="TRTMC base precision",
    )
    if bundle_path.is_file() and not force_build:
        inspection = bundle_inspection(bundle_path, trtmc_binary)
        if _bundle_can_be_reused(
            inspection,
            max_cache_length=max_cache_length,
            expected_precision=expected_precision,
            allow_unknown=not replace_existing,
        ):
            return bundle_path, False
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_bundle_before_or_after_replacement(
        bundle_path,
        replace_existing,
    )
    cmd = build_bundle_command(
        model,
        trtmc_binary=trtmc_binary,
        bundle_path=bundle_path,
        max_cache_length=max_cache_length,
        extra_build_args=extra_build_args,
    )
    log_path = log_path or bundle_path.with_suffix(".build.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    build_env = None
    if cuda_visible_devices:
        build_env = os.environ.copy()
        build_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {shlex.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=build_env,
        )
    if proc.returncode != 0:
        _remove_bundle_before_or_after_replacement(
            bundle_path,
            replace_existing,
        )
        raise RuntimeError(
            f"Bundle build failed for {model['name']} rc={proc.returncode}; see {log_path}"
        )
    return bundle_path, True


def model_matches_selector(model: dict[str, Any], selector: str) -> bool:
    return selector in {
        str(model.get("name", "")),
        str(model.get("hf_id", "")),
        str(model.get("bundle", "")),
    }


def selected_models_for_suite(
    suite: dict[str, Any],
    models: list[dict[str, Any]],
    *,
    selectors: list[str] | None = None,
    single_device_only: bool = False,
    waives: dict[str, tuple[str, str]] | None = None,
    include_waived: bool = False,
) -> list[dict[str, Any]]:
    rows = build_plan(
        [suite],
        models,
        suite_id=suite["id"],
        single_device_only=single_device_only,
        use_default_models=not bool(selectors),
        waives=waives,
        include_waived=include_waived or bool(selectors),
    )
    selected_names = {row["model"] for row in rows if row["selected"]}
    selected = [model for model in models if model["name"] in selected_names]
    if selectors:
        filtered: list[dict[str, Any]] = []
        missing: list[str] = []
        for selector in selectors:
            matches = [model for model in selected if model_matches_selector(model, selector)]
            if not matches:
                missing.append(selector)
            filtered.extend(matches)
        if missing:
            raise ValueError(
                f"Model selector(s) not found in suite {suite['id']}: {', '.join(missing)}"
            )
        # Preserve manifest order and drop duplicates from overlapping selectors.
        wanted = {model["name"] for model in filtered}
        selected = [model for model in selected if model["name"] in wanted]
    return selected


def _namespace_for_run_hf(
    args: argparse.Namespace, model: dict[str, Any], work_dir: Path
) -> argparse.Namespace:
    task_config = model.get("task_eval", {})
    task_config = task_config if isinstance(task_config, dict) else {}
    if (work_dir / "manifest.json").is_file():
        work_config = work_manifest(work_dir).get("task_eval", {})
        if isinstance(work_config, dict):
            task_config = work_config
    return argparse.Namespace(
        model=model["hf_id"],
        model_revision=str(model.get("hf_revision", "") or ""),
        family=model.get("family", ""),
        reference_family=model.get("reference_family", ""),
        work_dir=str(work_dir),
        predictions="hf_predictions.json",
        raw_output="hf_raw.jsonl",
        dtype=resolve_hf_reference_dtype(args, model, work_dir),
        device=args.hf_device,
        device_map=args.hf_device_map,
        attn_impl=args.hf_attn_impl,
        experts_implementation=str(
            task_config.get("hf_experts_implementation", "") or ""
        ),
        trust_remote_code=args.trust_remote_code or bool(model.get("trust_remote_code", False)),
        local_files_only=args.local_files_only,
        do_sample=args.do_sample,
        apply_chat_template=args.apply_chat_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        seed=args.seed,
        elf_reference_repo=getattr(args, "elf_reference_repo", ""),
    )


_REFERENCE_PRECISION_TO_DTYPE = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}
_REFERENCE_DTYPE_TO_PRECISION = {
    dtype: precision
    for precision, dtype in (
        ("fp16", "float16"),
        ("bf16", "bfloat16"),
        ("fp32", "float32"),
    )
}
_NATIVE_PRECISION_DATASET_KINDS = {
    "asr_chat_json",
    "mmlu_five_shot_json",
    "model_plugin_json",
    "seedtts_json",
    "text_generation_json",
    "sts_pair_jsonl",
    "vlm_chat_json",
    "vlm_grounding_json",
    "vlm_unified_json",
}


def _canonical_reference_precision(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    dtype = _REFERENCE_PRECISION_TO_DTYPE.get(normalized)
    if dtype is None:
        supported = ", ".join(("fp16", "bf16", "fp32"))
        raise ValueError(f"{field} must be one of {supported}; got {value!r}")
    return _REFERENCE_DTYPE_TO_PRECISION[dtype]


def _model_quantization_format(model: Mapping[str, Any]) -> str:
    quantization = model.get("quantization", {})
    if isinstance(quantization, Mapping):
        quant_format = str(quantization.get("format", "") or "").strip().lower()
        if quant_format and quant_format != "none":
            return quant_format
    if model.get("fp8_scales"):
        return "fp8"
    return ""


def apply_comparison_precision(
    model: Mapping[str, Any],
    validation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply validation-owned precision settings without changing CI manifests."""

    configured = validation_config.get("comparison_precision")
    updated = copy.deepcopy(dict(model))
    fp32_layers = validation_config.get("trtmc_fp32_layers")
    if configured not in (None, ""):
        quantization = _model_quantization_format(updated)
        if quantization:
            model_name = str(updated.get("name", "") or "quantized model")
            raise ValueError(
                f"{model_name} uses {quantization.upper()} quantization; "
                "comparison_precision may only override unquantized base precision"
            )
        updated["precision"] = _canonical_reference_precision(
            configured,
            field="validation.comparison_precision",
        )
    if fp32_layers is not None:
        if not isinstance(fp32_layers, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fp32_layers
        ):
            raise ValueError("validation.trtmc_fp32_layers must be non-negative integers")
        updated["fp32_layers"] = list(fp32_layers)
    return updated


def _configured_reference_precision(
    model: Mapping[str, Any],
    work_dir: Path,
) -> str:
    task_config = model.get("task_eval", {})
    configured = (
        task_config.get("reference_precision")
        if isinstance(task_config, Mapping)
        else None
    )
    manifest_path = work_dir / "manifest.json"
    if manifest_path.is_file():
        work_config = work_manifest(work_dir).get("task_eval", {})
        if isinstance(work_config, Mapping):
            configured = work_config.get("reference_precision", configured)
    if configured in (None, ""):
        return ""
    return _canonical_reference_precision(
        configured,
        field="task_eval.reference_precision",
    )


def _allows_declared_reference_precision_mismatch(
    model: Mapping[str, Any],
    work_dir: Path,
) -> bool:
    task_config = model.get("task_eval", {})
    allowed = (
        bool(task_config.get("allow_reference_precision_mismatch", False))
        if isinstance(task_config, Mapping)
        else False
    )
    manifest_path = work_dir / "manifest.json"
    if manifest_path.is_file():
        work_config = work_manifest(work_dir).get("task_eval", {})
        if isinstance(work_config, Mapping):
            allowed = bool(
                work_config.get(
                    "allow_reference_precision_mismatch",
                    allowed,
                )
            )
    return allowed


def resolve_reference_precision_contract(
    args: argparse.Namespace,
    model: Mapping[str, Any],
    work_dir: Path,
) -> dict[str, str]:
    """Resolve the declared TRTMC/reference precision relationship."""

    base_precision = _canonical_reference_precision(
        model.get("precision", "fp32"),
        field="TRTMC base precision",
    )
    quantization = _model_quantization_format(model)
    configured = _configured_reference_precision(model, work_dir)
    requested_dtype = str(getattr(args, "hf_dtype", "auto") or "auto")
    requested = (
        ""
        if requested_dtype == "auto"
        else _canonical_reference_precision(
            requested_dtype,
            field="--hf-dtype",
        )
    )

    if quantization and not configured:
        model_name = str(model.get("name", "") or "quantized model")
        raise ValueError(
            f"{model_name} uses {quantization.upper()} quantization and requires "
            "task_eval.reference_precision"
        )
    if requested and configured and requested != configured:
        raise ValueError(
            f"--hf-dtype {requested} conflicts with "
            f"task_eval.reference_precision {configured}"
        )

    reference_precision = requested or configured
    if not reference_precision:
        reference_precision = (
            base_precision
            if _work_dataset_kind(work_dir) in _NATIVE_PRECISION_DATASET_KINDS
            else "auto"
        )

    dataset_kind = _work_dataset_kind(work_dir)
    declared_mismatch_allowed = bool(configured) and (
        _allows_declared_reference_precision_mismatch(model, work_dir)
    )
    if (
        not quantization
        and dataset_kind in _NATIVE_PRECISION_DATASET_KINDS
        and reference_precision not in {"auto", base_precision}
        and not declared_mismatch_allowed
    ):
        raise ValueError(
            f"reference precision {reference_precision} does not match "
            f"TRTMC base precision {base_precision}"
        )

    comparison = (
        "quantized_vs_unquantized_reference"
        if quantization
        else "aligned"
        if reference_precision == base_precision
        else "reference_defined"
    )
    return {
        "trtmc_base_precision": base_precision,
        "trtmc_quantization": quantization or "none",
        "reference_precision": reference_precision,
        "reference_dtype": (
            _REFERENCE_PRECISION_TO_DTYPE[reference_precision]
            if reference_precision != "auto"
            else "auto"
        ),
        "comparison": comparison,
    }


def resolve_hf_reference_dtype(
    args: argparse.Namespace,
    model: Mapping[str, Any],
    work_dir: Path,
) -> str:
    """Resolve the reference dtype for native Transformers parity workloads."""

    return resolve_reference_precision_contract(
        args,
        model,
        work_dir,
    )["reference_dtype"]


def _namespace_for_run_trtfb(
    args: argparse.Namespace, bundle_path: Path, work_dir: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        bundle=str(bundle_path),
        work_dir=str(work_dir),
        trtmc_binary=args.trtmc_binary,
        benchmark_binary=args.benchmark_binary,
        hf_python=args.hf_python,
        backend_dir=args.backend_dir,
        model_plugin_dir=getattr(args, "model_plugin_dir", ""),
        kv_cache_size=args.kv_cache_size,
        config=args.config,
        set=args.set or [],
        cuda_visible_devices=args.cuda_visible_devices,
        chat_template=args.chat_template,
        predictions="trtfb_predictions.json",
        raw_output="trtfb_raw.jsonl",
        log="trtfb_run.log",
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        seed=args.seed,
    )


def model_reference_python(model: dict[str, Any], base_python: str) -> str:
    profiles = normalize_execution_profiles(
        model.get("execution_profiles"),
        family=str(model.get("family", "") or ""),
        runtime_strategy=str(model.get("runtime_strategy", "") or ""),
        reference_backend=str(model.get("reference_backend", "") or ""),
    )
    return resolve_profile_python(profiles["reference"], base_python)


def _read_bundle_section(bundle_path: Path, section_name: str) -> bytes:
    max_header_size = 100 * 1024 * 1024
    with bundle_path.open("rb") as bundle:
        if bundle.read(8) != b"TRTFB\x00\x01\x00":
            raise ValueError(f"{bundle_path} is not a TRTMC bundle")
        raw_header_size = bundle.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"{bundle_path} has a truncated header size")
        header_size = struct.unpack("<Q", raw_header_size)[0]
        if header_size > max_header_size:
            raise ValueError(
                f"{bundle_path} header exceeds {max_header_size} bytes"
            )
        raw_header = bundle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"{bundle_path} has a truncated JSON header")
        header = json.loads(raw_header)
        sections = header.get("sections", {})
        section = sections.get(section_name) if isinstance(sections, dict) else None
        if not isinstance(section, dict):
            raise ValueError(f"{bundle_path} has no {section_name!r} section")
        offset = int(section.get("offset", -1))
        size = int(section.get("size", -1))
        data_start = 16 + header_size
        if offset < 0 or size < 0 or data_start + offset + size > bundle_path.stat().st_size:
            raise ValueError(
                f"{bundle_path} has an invalid {section_name!r} section range"
            )
        bundle.seek(data_start + offset)
        data = bundle.read(size)
        if len(data) != size:
            raise ValueError(f"{bundle_path} has a truncated {section_name!r} section")
        return data


def _load_text_input_contract(
    *,
    model: Mapping[str, Any],
    bundle_path: Path,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    revision = str(model.get("hf_revision", "") or "")
    if revision:
        tokenizer_kwargs["revision"] = revision
    hf_tokenizer = AutoTokenizer.from_pretrained(
        str(model["hf_id"]),
        **tokenizer_kwargs,
    )
    bundle_tokenizer = Tokenizer.from_str(
        _read_bundle_section(bundle_path, "tokenizer.json").decode("utf-8")
    )
    bundle_config = json.loads(
        _read_bundle_section(bundle_path, "config.json").decode("utf-8")
    )
    if not isinstance(bundle_config, dict):
        raise ValueError(f"{bundle_path} config.json must contain an object")
    return hf_tokenizer, bundle_tokenizer, bundle_config


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        [int(token_id) for token_id in token_ids],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_token_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if int(left_id) != int(right_id):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def validate_text_input_token_contract(
    *,
    model: Mapping[str, Any],
    work_dir: Path,
    bundle_path: Path,
    local_files_only: bool,
    trust_remote_code: bool,
) -> None:
    """Fail before inference when HF and the bundle would consume different IDs."""

    hf_tokenizer, bundle_tokenizer, bundle_config = _load_text_input_contract(
        model=model,
        bundle_path=bundle_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    has_exact_frame = (
        "tokenizer_special_prefix_ids" in bundle_config
        or "tokenizer_special_suffix_ids" in bundle_config
    )
    prefix = [
        int(token_id)
        for token_id in bundle_config.get("tokenizer_special_prefix_ids", [])
    ]
    suffix = [
        int(token_id)
        for token_id in bundle_config.get("tokenizer_special_suffix_ids", [])
    ]
    bundle_add_special_tokens = bool(
        bundle_config.get("tokenizer_add_special_tokens", False)
    )

    samples: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    for index, row in enumerate(load_jsonl(work_dir / "prompts.jsonl")):
        sample_id = str(row.get("sample_id", f"sample-{index}"))
        prompt = str(row.get("prompt", ""))
        hf_ids = [int(token_id) for token_id in hf_tokenizer(prompt).input_ids]
        bundle_encoding = bundle_tokenizer.encode(
            prompt,
            add_special_tokens=False if has_exact_frame else bundle_add_special_tokens,
        )
        bundle_ids = [int(token_id) for token_id in bundle_encoding.ids]
        if has_exact_frame:
            bundle_ids = prefix + bundle_ids + suffix
        first_difference = _first_token_difference(hf_ids, bundle_ids)
        sample = {
            "sample_id": sample_id,
            "hf_token_count": len(hf_ids),
            "trtmc_token_count": len(bundle_ids),
            "hf_token_sha256": _token_ids_sha256(hf_ids),
            "trtmc_token_sha256": _token_ids_sha256(bundle_ids),
        }
        if first_difference is not None:
            window_start = max(0, first_difference - 4)
            window_end = first_difference + 12
            sample.update(
                {
                    "first_difference": first_difference,
                    "hf_token_window": hf_ids[window_start:window_end],
                    "trtmc_token_window": bundle_ids[window_start:window_end],
                }
            )
            if first_mismatch is None:
                first_mismatch = sample
        samples.append(sample)

    artifact = {
        "schema_version": "trtmc.input-token-contract/v1",
        "model": str(model.get("name", "")),
        "hf_id": str(model.get("hf_id", "")),
        "bundle": str(bundle_path),
        "status": "mismatch" if first_mismatch else "aligned",
        "samples": samples,
    }
    (work_dir / "input_token_contract.json").write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )
    if first_mismatch is not None:
        raise RuntimeError(
            "HF/TRTMC input token contract mismatch: "
            f"model={model.get('name', '')}, "
            f"sample={first_mismatch['sample_id']}, "
            f"first_difference={first_mismatch['first_difference']}, "
            f"HF={first_mismatch['hf_token_window']}, "
            f"TRTMC={first_mismatch['trtmc_token_window']}; "
            f"see {work_dir / 'input_token_contract.json'}"
        )


def run_hf_reference_subprocess(
    args: argparse.Namespace, model: dict[str, Any], work_dir: Path
) -> None:
    """Run the HF reference in a dedicated subprocess.

    Keeping the torch/transformers reference model in its own process guarantees
    its GPU memory is fully reclaimed when the process exits, before the TRT
    bundle build and TRTFB inference run for the same model. This prevents the
    resident reference model from contending with the TRT engine + KV cache.
    """
    hf_args = _namespace_for_run_hf(args, model, work_dir)
    base_python = str(getattr(args, "hf_python", "") or sys.executable)
    hf_python = model_reference_python(model, base_python)
    cmd = [
        hf_python,
        str(REFERENCE_RUNNER),
        "run",
        "--model",
        str(hf_args.model),
        "--family",
        str(hf_args.family),
        "--reference-family",
        str(hf_args.reference_family),
        "--work-dir",
        str(hf_args.work_dir),
        "--predictions",
        str(hf_args.predictions),
        "--raw-output",
        str(hf_args.raw_output),
        "--dtype",
        str(hf_args.dtype),
        "--device",
        str(hf_args.device),
    ]
    if hf_args.model_revision:
        cmd.extend(["--model-revision", str(hf_args.model_revision)])
    reference_cache_dir = str(
        getattr(args, "reference_cache_dir", "") or ""
    )
    if reference_cache_dir:
        cmd.extend(["--cache-dir", reference_cache_dir])
    reference_cache_identity = str(
        getattr(args, "reference_cache_identity", "") or ""
    )
    if reference_cache_identity:
        cmd.extend(
            ["--reference-cache-identity", reference_cache_identity]
        )
    if hf_args.device_map:
        cmd.extend(["--device-map", str(hf_args.device_map)])
    if hf_args.attn_impl:
        cmd.extend(["--attn-impl", str(hf_args.attn_impl)])
    if hf_args.experts_implementation:
        cmd.extend(
            [
                "--experts-implementation",
                str(hf_args.experts_implementation),
            ]
        )
    if hf_args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if hf_args.local_files_only:
        cmd.append("--local-files-only")
    if hf_args.do_sample:
        cmd.append("--do-sample")
    if hf_args.apply_chat_template:
        cmd.append("--apply-chat-template")
    if hf_args.elf_reference_repo:
        cmd.extend(["--elf-reference-repo", str(hf_args.elf_reference_repo)])
    if bool(getattr(args, "force_hf", False)):
        cmd.append("--force")
    for flag, value in (
        ("--max-new-tokens", hf_args.max_new_tokens),
        ("--temperature", hf_args.temperature),
        ("--top-k", hf_args.top_k),
        ("--top-p", hf_args.top_p),
        ("--min-p", hf_args.min_p),
        ("--seed", hf_args.seed),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    log_path = work_dir / "hf_run.log"
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"$ {shlex.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(
            cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"HF reference subprocess failed for {model['name']} rc={proc.returncode}; see {log_path}"
        )


def _vector_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError(f"Vector dimension mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("Cannot compare a zero-norm encoder vector")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = average_rank
        start = end
    return ranks


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator <= 0.0:
        return None
    return sum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _spearman_correlation(left: list[float], right: list[float]) -> float | None:
    return _pearson_correlation(_rank_values(left), _rank_values(right))


def compare_encoder_embedding_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    hf_rows = hf_data.get("responses", [])
    trtfb_rows = trtfb_data.get("responses", [])
    if not isinstance(hf_rows, list) or not isinstance(trtfb_rows, list):
        raise ValueError("Encoder predictions must contain response lists")
    if len(hf_rows) != len(trtfb_rows):
        raise ValueError(
            f"Encoder HF/TRTMC prediction count mismatch: {len(hf_rows)} != {len(trtfb_rows)}"
        )
    min_vector_cosine_gate = float(gates.get("min_vector_cosine", 0.99))
    min_vector_pass_rate_gate = float(gates.get("min_vector_pass_rate", 1.0))
    max_pair_delta_gate = float(gates.get("max_pair_cosine_abs_delta", 0.02))
    vector_cosines: list[float] = []
    samples: list[dict[str, Any]] = []
    hf_pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    trtfb_pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, (hf_row, trtfb_row) in enumerate(zip(hf_rows, trtfb_rows, strict=True)):
        hf_id = str(hf_row.get("sample_id", index))
        trtfb_id = str(trtfb_row.get("sample_id", index))
        if hf_id != trtfb_id:
            raise ValueError(
                f"Encoder HF/TRTMC sample id mismatch at {index}: {hf_id!r} != {trtfb_id!r}"
            )
        hf_vector = hf_row.get("vector")
        trtfb_vector = trtfb_row.get("vector")
        if not isinstance(hf_vector, list) or not isinstance(trtfb_vector, list):
            raise ValueError(f"Encoder prediction {hf_id!r} is missing vector data")
        cosine = _vector_cosine(
            [float(value) for value in hf_vector],
            [float(value) for value in trtfb_vector],
        )
        vector_cosines.append(cosine)
        pair_id = str(hf_row.get("pair_id", ""))
        pair_side = str(hf_row.get("pair_side", ""))
        if not pair_id or pair_side not in {"sentence1", "sentence2"}:
            raise ValueError(f"Encoder prediction {hf_id!r} has invalid pair metadata")
        hf_pairs[pair_id][pair_side] = hf_row
        trtfb_pairs[pair_id][pair_side] = trtfb_row
        samples.append(
            {
                "sample_id": hf_id,
                "pair_id": pair_id,
                "pair_side": pair_side,
                "vector_dim": len(hf_vector),
                "vector_cosine": cosine,
                "passed": cosine >= min_vector_cosine_gate,
            }
        )

    pair_rows: list[dict[str, Any]] = []
    pair_deltas: list[float] = []
    gold_scores: list[float] = []
    hf_similarities: list[float] = []
    trtfb_similarities: list[float] = []
    for pair_id, hf_pair in hf_pairs.items():
        trtfb_pair = trtfb_pairs.get(pair_id, {})
        if set(hf_pair) != {"sentence1", "sentence2"} or set(trtfb_pair) != {
            "sentence1",
            "sentence2",
        }:
            raise ValueError(f"Encoder pair {pair_id!r} is incomplete")
        hf_similarity = _vector_cosine(
            [float(value) for value in hf_pair["sentence1"]["vector"]],
            [float(value) for value in hf_pair["sentence2"]["vector"]],
        )
        trtfb_similarity = _vector_cosine(
            [float(value) for value in trtfb_pair["sentence1"]["vector"]],
            [float(value) for value in trtfb_pair["sentence2"]["vector"]],
        )
        delta = abs(trtfb_similarity - hf_similarity)
        score = float(hf_pair["sentence1"].get("score", 0.0))
        pair_deltas.append(delta)
        gold_scores.append(score)
        hf_similarities.append(hf_similarity)
        trtfb_similarities.append(trtfb_similarity)
        pair_rows.append(
            {
                "pair_id": pair_id,
                "score": score,
                "hf_cosine": hf_similarity,
                "trtfb_cosine": trtfb_similarity,
                "cosine_abs_delta": delta,
                "passed": delta <= max_pair_delta_gate,
            }
        )

    vector_passed = sum(value >= min_vector_cosine_gate for value in vector_cosines)
    vector_pass_rate = vector_passed / len(vector_cosines) if vector_cosines else 0.0
    max_pair_delta = max(pair_deltas, default=float("inf"))
    status = (
        "passed"
        if vector_cosines
        and pair_deltas
        and vector_pass_rate >= min_vector_pass_rate_gate
        and max_pair_delta <= max_pair_delta_gate
        else "failed"
    )
    return {
        "status": status,
        "valid_count": len(vector_cosines),
        "pair_count": len(pair_rows),
        "vector_passed_count": vector_passed,
        "vector_pass_rate": vector_pass_rate,
        "mean_vector_cosine": sum(vector_cosines) / len(vector_cosines)
        if vector_cosines
        else 0.0,
        "min_vector_cosine": min(vector_cosines, default=0.0),
        "mean_pair_cosine_abs_delta": sum(pair_deltas) / len(pair_deltas)
        if pair_deltas
        else float("inf"),
        "max_pair_cosine_abs_delta": max_pair_delta,
        "hf_sts_spearman": _spearman_correlation(gold_scores, hf_similarities),
        "trtfb_sts_spearman": _spearman_correlation(gold_scores, trtfb_similarities),
        "gates": {
            "min_vector_cosine": min_vector_cosine_gate,
            "min_vector_pass_rate": min_vector_pass_rate_gate,
            "max_pair_cosine_abs_delta": max_pair_delta_gate,
        },
        "samples": samples,
        "pairs": pair_rows,
    }


def _validation_response_rows(data: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = data.get("responses", [])
    if not isinstance(rows, list):
        raise ValueError(f"{label} predictions must contain a response list")
    return rows


def compare_time_series_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Gate every numeric sample with the model-owned E2E parity metrics."""
    hf_rows = _validation_response_rows(hf_data, "HF")
    trtfb_rows = _validation_response_rows(trtfb_data, "TRTMC")
    if len(hf_rows) != len(trtfb_rows):
        raise ValueError(
            f"Time-series HF/TRTMC prediction count mismatch: {len(hf_rows)} != {len(trtfb_rows)}"
        )
    max_relative_l2 = float(gates.get("max_relative_l2", 0.01))
    max_absolute_error = float(gates.get("max_absolute_error", 0.1))
    min_sample_agreement_rate = float(gates.get("min_sample_agreement_rate", 1.0))
    cases: list[dict[str, Any]] = []
    for index, (hf_row, trtfb_row) in enumerate(zip(hf_rows, trtfb_rows, strict=True)):
        hf_id = str(hf_row.get("sample_id", index))
        trtfb_id = str(trtfb_row.get("sample_id", index))
        if hf_id != trtfb_id:
            raise ValueError(
                f"Time-series sample id mismatch at {index}: {hf_id!r} != {trtfb_id!r}"
            )
        hf_values = hf_row.get("output_values")
        trtfb_values = trtfb_row.get("output_values")
        if not isinstance(hf_values, list) or not isinstance(trtfb_values, list):
            raise ValueError(f"Time-series prediction {hf_id!r} is missing output_values")
        if len(hf_values) != len(trtfb_values) or not hf_values:
            cases.append(
                {
                    "sample_id": hf_id,
                    "passed": False,
                    "error": (
                        "output element count mismatch: "
                        f"HF={len(hf_values)} TRTMC={len(trtfb_values)}"
                    ),
                }
            )
            continue
        hf_vector = [float(value) for value in hf_values]
        trtfb_vector = [float(value) for value in trtfb_values]
        if not all(math.isfinite(value) for value in hf_vector + trtfb_vector):
            cases.append(
                {
                    "sample_id": hf_id,
                    "passed": False,
                    "error": "non-finite output value",
                }
            )
            continue
        squared_error = sum(
            (trtfb - hf) ** 2 for hf, trtfb in zip(hf_vector, trtfb_vector, strict=True)
        )
        reference_squared_norm = sum(value * value for value in hf_vector)
        relative_l2 = (
            math.sqrt(squared_error / reference_squared_norm)
            if reference_squared_norm >= 1e-24
            else math.sqrt(squared_error)
        )
        absolute_error = max(
            abs(trtfb - hf) for hf, trtfb in zip(hf_vector, trtfb_vector, strict=True)
        )
        cases.append(
            {
                "sample_id": hf_id,
                "output_numel": len(hf_vector),
                "hf_output_shape": hf_row.get("output_shape", []),
                "trtfb_output_shape": trtfb_row.get("output_shape", []),
                "relative_l2": relative_l2,
                "max_absolute_error": absolute_error,
                "passed": (relative_l2 <= max_relative_l2 and absolute_error <= max_absolute_error),
            }
        )

    valid_cases = [case for case in cases if "relative_l2" in case]
    passed_count = sum(bool(case.get("passed")) for case in cases)
    agreement_rate = passed_count / len(cases) if cases else 0.0
    status = (
        "passed"
        if cases and len(valid_cases) == len(cases) and agreement_rate >= min_sample_agreement_rate
        else "failed"
    )
    return {
        "status": status,
        "sample_count": len(cases),
        "valid_count": len(valid_cases),
        "passed_count": passed_count,
        "sample_agreement_rate": agreement_rate,
        "mean_relative_l2": (
            sum(float(case["relative_l2"]) for case in valid_cases) / len(valid_cases)
            if valid_cases
            else float("inf")
        ),
        "max_relative_l2": max(
            (float(case["relative_l2"]) for case in valid_cases),
            default=float("inf"),
        ),
        "max_absolute_error": max(
            (float(case["max_absolute_error"]) for case in valid_cases),
            default=float("inf"),
        ),
        "gates": {
            "max_relative_l2": max_relative_l2,
            "max_absolute_error": max_absolute_error,
            "min_sample_agreement_rate": min_sample_agreement_rate,
        },
        "cases": cases,
    }


def write_time_series_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Time-series HF/TRTMC Parity Summary",
        "",
        f"- status: {summary['status']}",
        f"- sample_agreement_rate: {summary['sample_agreement_rate']:.4f}",
        f"- max_relative_l2: {summary['max_relative_l2']:.6e}",
        f"- max_absolute_error: {summary['max_absolute_error']:.6e}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_image_classification_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    answers: dict[str, Any],
    *,
    gates: dict[str, Any],
) -> dict[str, Any]:
    hf_rows = _validation_response_rows(hf_data, "HF")
    trtfb_rows = _validation_response_rows(trtfb_data, "TRTMC")
    requests = answers.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError("Classification answers must contain requests")
    hf_by_id = {str(row["sample_id"]): row for row in hf_rows}
    trtfb_by_id = {str(row["sample_id"]): row for row in trtfb_rows}
    cases: list[dict[str, Any]] = []
    for request in requests:
        sample_id = str(request.get("sample_id") or request.get("id") or "")
        hf_row = hf_by_id.get(sample_id)
        trtfb_row = trtfb_by_id.get(sample_id)
        if hf_row is None or trtfb_row is None:
            cases.append({"sample_id": sample_id, "passed": False, "error": "missing prediction"})
            continue
        label = int(request["label"])
        hf_top = int(hf_row["top_class"])
        trtfb_top = int(trtfb_row["top_class"])
        cases.append(
            {
                "sample_id": sample_id,
                "label": label,
                "label_name": str(request.get("label_name", "")),
                "hf_top_class": hf_top,
                "trtfb_top_class": trtfb_top,
                "hf_correct": hf_top == label,
                "trtfb_correct": trtfb_top == label,
                "top1_agreement": hf_top == trtfb_top,
            }
        )
    valid = [case for case in cases if "error" not in case]
    hf_accuracy = _mean([float(case["hf_correct"]) for case in valid])
    trtfb_accuracy = _mean([float(case["trtfb_correct"]) for case in valid])
    agreement = _mean([float(case["top1_agreement"]) for case in valid])
    max_drop = float(gates.get("max_top1_accuracy_drop_from_hf", 0.01))
    min_agreement = float(gates.get("min_top1_agreement", 0.98))
    accuracy_drop = hf_accuracy - trtfb_accuracy
    status = (
        "passed"
        if valid
        and len(valid) == len(cases)
        and accuracy_drop <= max_drop
        and agreement >= min_agreement
        else "failed"
    )
    return {
        "status": status,
        "valid_count": len(valid),
        "sample_count": len(cases),
        "hf_top1_accuracy": hf_accuracy,
        "trtfb_top1_accuracy": trtfb_accuracy,
        "top1_accuracy_drop_from_hf": accuracy_drop,
        "top1_agreement": agreement,
        "gates": {
            "max_top1_accuracy_drop_from_hf": max_drop,
            "min_top1_agreement": min_agreement,
        },
        "cases": cases,
    }


def _load_segmentation_array(path: str) -> Any:
    import numpy as np

    artifact = Path(path)
    if artifact.suffix.lower() == ".npy":
        return np.load(artifact, allow_pickle=False)
    from PIL import Image

    return np.asarray(Image.open(artifact).convert("L"))


def _resize_label_array(values: Any, shape: tuple[int, int]) -> Any:
    import numpy as np

    array = np.asarray(values)
    if array.shape == shape:
        return array
    from PIL import Image

    resized = Image.fromarray(array.astype(np.uint8)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST
    )
    return np.asarray(resized)


def _segmentation_confusion(
    reference: Any,
    prediction: Any,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> Any:
    import numpy as np

    ref = np.asarray(reference, dtype=np.int64)
    pred = _resize_label_array(prediction, ref.shape).astype(np.int64)
    valid = (ref >= 0) & (ref < num_classes) & (pred >= 0) & (pred < num_classes)
    if ignore_index is not None:
        valid &= ref != ignore_index
    encoded = num_classes * ref[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _segmentation_metrics(confusion: Any) -> dict[str, float]:
    import numpy as np

    matrix = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(matrix)
    gt_total = matrix.sum(axis=1)
    pred_total = matrix.sum(axis=0)
    union = gt_total + pred_total - true_positive
    present = union > 0
    mean_iou = float(np.mean(true_positive[present] / union[present])) if present.any() else 0.0
    total = matrix.sum()
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
    return {"mean_iou": mean_iou, "pixel_accuracy": pixel_accuracy}


def compare_semantic_segmentation_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    answers: dict[str, Any],
    *,
    gates: dict[str, Any],
    num_classes: int,
    ignore_index: int,
) -> dict[str, Any]:
    import numpy as np

    hf_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(hf_data, "HF")
    }
    trtfb_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(trtfb_data, "TRTMC")
    }
    requests = answers.get("requests", [])
    hf_ground = np.zeros((num_classes, num_classes), dtype=np.int64)
    trtfb_ground = np.zeros_like(hf_ground)
    backend = np.zeros_like(hf_ground)
    cases: list[dict[str, Any]] = []
    for request in requests:
        sample_id = str(request.get("sample_id") or request.get("id") or "")
        if sample_id not in hf_by_id or sample_id not in trtfb_by_id:
            cases.append({"sample_id": sample_id, "passed": False, "error": "missing prediction"})
            continue
        ground_truth = _load_segmentation_array(str(request["mask"]))
        hf_map = _load_segmentation_array(str(hf_by_id[sample_id]["class_map_path"]))
        hf_backend_map = _load_segmentation_array(
            str(
                hf_by_id[sample_id].get("raw_class_map_path")
                or hf_by_id[sample_id]["class_map_path"]
            )
        )
        trtfb_map = _load_segmentation_array(
            str(trtfb_by_id[sample_id]["class_map_path"])
        )
        hf_conf = _segmentation_confusion(
            ground_truth,
            hf_map,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        trtfb_conf = _segmentation_confusion(
            ground_truth,
            trtfb_map,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        resized_hf = _resize_label_array(hf_backend_map, trtfb_map.shape)
        backend_conf = _segmentation_confusion(
            resized_hf,
            trtfb_map,
            num_classes=num_classes,
        )
        hf_ground += hf_conf
        trtfb_ground += trtfb_conf
        backend += backend_conf
        case_backend = _segmentation_metrics(backend_conf)
        cases.append(
            {
                "sample_id": sample_id,
                "backend_pixel_agreement": case_backend["pixel_accuracy"],
                "backend_mean_iou": case_backend["mean_iou"],
                "hf_ground_truth_mean_iou": _segmentation_metrics(hf_conf)["mean_iou"],
                "trtfb_ground_truth_mean_iou": _segmentation_metrics(trtfb_conf)["mean_iou"],
            }
        )
    hf_metrics = _segmentation_metrics(hf_ground)
    trtfb_metrics = _segmentation_metrics(trtfb_ground)
    backend_metrics = _segmentation_metrics(backend)
    min_pixel = float(gates.get("min_backend_pixel_agreement", 0.98))
    min_backend_iou = float(gates.get("min_backend_mean_iou", 0.95))
    max_drop = float(gates.get("max_mean_iou_drop_from_hf", 0.01))
    mean_iou_drop = hf_metrics["mean_iou"] - trtfb_metrics["mean_iou"]
    valid_count = sum("error" not in case for case in cases)
    status = (
        "passed"
        if valid_count > 0
        and valid_count == len(cases)
        and backend_metrics["pixel_accuracy"] >= min_pixel
        and backend_metrics["mean_iou"] >= min_backend_iou
        and mean_iou_drop <= max_drop
        else "failed"
    )
    return {
        "status": status,
        "sample_count": len(cases),
        "valid_count": valid_count,
        "hf_mean_iou": hf_metrics["mean_iou"],
        "trtfb_mean_iou": trtfb_metrics["mean_iou"],
        "mean_iou_drop_from_hf": mean_iou_drop,
        "hf_pixel_accuracy": hf_metrics["pixel_accuracy"],
        "trtfb_pixel_accuracy": trtfb_metrics["pixel_accuracy"],
        "backend_mean_iou": backend_metrics["mean_iou"],
        "backend_pixel_agreement": backend_metrics["pixel_accuracy"],
        "gates": {
            "min_backend_pixel_agreement": min_pixel,
            "min_backend_mean_iou": min_backend_iou,
            "max_mean_iou_drop_from_hf": max_drop,
        },
        "cases": cases,
    }


def _mask_stack(path: str) -> Any:
    import numpy as np

    masks = np.asarray(np.load(path, allow_pickle=False))
    while masks.ndim > 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected [mask,height,width] at {path}, got {masks.shape}")
    return masks.astype(bool)


def _selected_prompt_mask(masks: Any, scores: list[Any], prompt_mode: str) -> Any:
    import numpy as np

    if prompt_mode == "text":
        return np.any(masks, axis=0)
    index = int(np.argmax(np.asarray(scores, dtype=np.float64))) if scores else 0
    return masks[min(index, masks.shape[0] - 1)]


def _selected_prompt_prediction_mask(
    row: dict[str, Any], prompt_mode: str, target_shape: tuple[int, ...]
) -> Any:
    import numpy as np

    if bool(row.get("empty_prediction")) or int(row.get("num_masks", -1)) == 0:
        return np.zeros(target_shape, dtype=bool)
    return _selected_prompt_mask(
        _mask_stack(str(row["masks_path"])),
        list(row.get("mask_scores", [])),
        prompt_mode,
    )


def _binary_mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    left_mask = np.asarray(left, dtype=bool)
    right_mask = _resize_label_array(np.asarray(right, dtype=np.uint8), left_mask.shape) > 0
    intersection = np.logical_and(left_mask, right_mask).sum()
    union = np.logical_or(left_mask, right_mask).sum()
    return float(intersection / union) if union else 1.0


def compare_prompted_segmentation_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    answers: dict[str, Any],
    *,
    gates: dict[str, Any],
    prompt_mode: str,
    ground_truth_mask_field: str,
) -> dict[str, Any]:
    hf_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(hf_data, "HF")
    }
    trtfb_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(trtfb_data, "TRTMC")
    }
    cases: list[dict[str, Any]] = []
    for request in answers.get("requests", []):
        sample_id = str(request.get("sample_id") or request.get("id") or "")
        hf_row = hf_by_id.get(sample_id)
        trtfb_row = trtfb_by_id.get(sample_id)
        if hf_row is None or trtfb_row is None:
            cases.append({"sample_id": sample_id, "passed": False, "error": "missing prediction"})
            continue
        ground_truth = _load_segmentation_array(str(request[ground_truth_mask_field])) > 0
        hf_mask = _selected_prompt_prediction_mask(
            hf_row,
            prompt_mode,
            ground_truth.shape,
        )
        trtfb_mask = _selected_prompt_prediction_mask(
            trtfb_row,
            prompt_mode,
            ground_truth.shape,
        )
        backend_iou = _binary_mask_iou(hf_mask, trtfb_mask)
        hf_gt_iou = _binary_mask_iou(ground_truth, hf_mask)
        trtfb_gt_iou = _binary_mask_iou(ground_truth, trtfb_mask)
        cases.append(
            {
                "sample_id": sample_id,
                "prompt_mode": prompt_mode,
                "text_prompt": str(request.get("text_prompt", "")),
                "backend_mask_iou": backend_iou,
                "hf_ground_truth_iou": hf_gt_iou,
                "trtfb_ground_truth_iou": trtfb_gt_iou,
                "ground_truth_iou_drop_from_hf": hf_gt_iou - trtfb_gt_iou,
                "hf_empty_prediction": bool(hf_row.get("empty_prediction")),
                "trtfb_empty_prediction": bool(trtfb_row.get("empty_prediction")),
                "hf_segmented_image_path": str(hf_row.get("segmented_image_path", "")),
                "trtfb_segmented_image_path": str(
                    trtfb_row.get("segmented_image_path", "")
                ),
            }
        )
    valid = [case for case in cases if "error" not in case]
    mean_backend_iou = _mean([float(case["backend_mask_iou"]) for case in valid])
    hf_gt_iou = _mean([float(case["hf_ground_truth_iou"]) for case in valid])
    trtfb_gt_iou = _mean([float(case["trtfb_ground_truth_iou"]) for case in valid])
    min_backend_iou = float(gates.get("min_backend_mask_iou", 0.90))
    max_gt_drop = float(gates.get("max_ground_truth_iou_drop_from_hf", 0.05))
    gt_drop = hf_gt_iou - trtfb_gt_iou
    status = (
        "passed"
        if valid
        and len(valid) == len(cases)
        and mean_backend_iou >= min_backend_iou
        and gt_drop <= max_gt_drop
        else "failed"
    )
    return {
        "status": status,
        "sample_count": len(cases),
        "valid_count": len(valid),
        "prompt_mode": prompt_mode,
        "mean_backend_mask_iou": mean_backend_iou,
        "hf_mean_ground_truth_iou": hf_gt_iou,
        "trtfb_mean_ground_truth_iou": trtfb_gt_iou,
        "ground_truth_iou_drop_from_hf": gt_drop,
        "gates": {
            "min_backend_mask_iou": min_backend_iou,
            "max_ground_truth_iou_drop_from_hf": max_gt_drop,
        },
        "cases": cases,
    }


def compare_reranking_prediction_sets(
    hf_data: dict[str, Any],
    trtfb_data: dict[str, Any],
    answers: dict[str, Any],
    *,
    gates: dict[str, Any],
    comparator: Any,
) -> dict[str, Any]:
    from tests.e2e_harness.contracts import (
        StageOutput,
        StageSpec,
        StageStatus,
        ThresholdProfile,
    )

    hf_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(hf_data, "HF")
    }
    trtfb_by_id = {
        str(row["sample_id"]): row
        for row in _validation_response_rows(trtfb_data, "TRTMC")
    }
    metric_thresholds = {
        "pairwise_ordering_agreement": float(
            gates.get("pairwise_ordering_agreement", 0.9)
        ),
        "kendall_tau": float(gates.get("kendall_tau", 0.8)),
        "spearman_rho": float(gates.get("spearman_rho", 0.8)),
        "score_correlation": float(gates.get("score_correlation", 0.9)),
    }
    threshold = ThresholdProfile(
        task_strategy="reranking", metrics=metric_thresholds
    )
    stage = StageSpec(name="full_inference", required=True)
    cases: list[dict[str, Any]] = []
    for request in answers.get("requests", []):
        sample_id = str(request.get("sample_id") or request.get("id") or "")
        hf_row = hf_by_id.get(sample_id)
        trtfb_row = trtfb_by_id.get(sample_id)
        if hf_row is None or trtfb_row is None:
            cases.append(
                {"sample_id": sample_id, "passed": False, "error": "missing prediction"}
            )
            continue
        hf_scores = hf_row.get("scores")
        trtfb_scores = trtfb_row.get("scores")
        if not isinstance(hf_scores, list) or not isinstance(trtfb_scores, list):
            cases.append(
                {"sample_id": sample_id, "passed": False, "error": "missing scores"}
            )
            continue
        if not hf_scores or len(hf_scores) != len(trtfb_scores):
            cases.append(
                {
                    "sample_id": sample_id,
                    "passed": False,
                    "error": (
                        f"score count mismatch: HF={len(hf_scores)}, "
                        f"TRTMC={len(trtfb_scores)}"
                    ),
                }
            )
            continue
        comparison = comparator.compare(
            StageOutput(stage_name=stage.name, data={"scores": trtfb_scores}),
            StageOutput(stage_name=stage.name, data={"scores": hf_scores}),
            threshold,
            stage,
        )
        metrics = {
            name: {
                "value": float(metric.value),
                "threshold": metric.threshold,
                "operator": metric.operator,
                "passed": bool(metric.passed),
                "note": metric.note,
            }
            for name, metric in comparison.metrics.items()
        }
        relevant_indices = {
            int(index) for index in request.get("relevant_document_indices", [])
        }
        hf_top_index = max(range(len(hf_scores)), key=lambda index: hf_scores[index])
        trtfb_top_index = max(
            range(len(trtfb_scores)), key=lambda index: trtfb_scores[index]
        )
        cases.append(
            {
                "sample_id": sample_id,
                "passed": comparison.status == StageStatus.PASSED.value,
                "status": comparison.status,
                "message": comparison.message,
                "metrics": metrics,
                "hf_scores": [float(score) for score in hf_scores],
                "trtfb_scores": [float(score) for score in trtfb_scores],
                "hf_top_document_index": hf_top_index,
                "trtfb_top_document_index": trtfb_top_index,
                "relevant_document_indices": sorted(relevant_indices),
                "hf_top1_correct": hf_top_index in relevant_indices,
                "trtfb_top1_correct": trtfb_top_index in relevant_indices,
            }
        )
    valid = [case for case in cases if "error" not in case]
    passed_count = sum(bool(case["passed"]) for case in valid)
    sample_pass_rate = passed_count / len(valid) if valid else 0.0
    min_sample_pass_rate = float(gates.get("min_sample_pass_rate", 1.0))
    metric_summaries: dict[str, dict[str, float]] = {}
    for name in metric_thresholds:
        values = [float(case["metrics"][name]["value"]) for case in valid]
        metric_summaries[name] = {
            "mean": _mean(values),
            "min": min(values) if values else 0.0,
        }
    gold_cases = [case for case in valid if case["relevant_document_indices"]]
    hf_top1_accuracy = _mean(
        [1.0 if case["hf_top1_correct"] else 0.0 for case in gold_cases]
    )
    trtfb_top1_accuracy = _mean(
        [1.0 if case["trtfb_top1_correct"] else 0.0 for case in gold_cases]
    )
    status = (
        "passed"
        if valid
        and len(valid) == len(cases)
        and sample_pass_rate >= min_sample_pass_rate
        else "failed"
    )
    return {
        "status": status,
        "sample_count": len(cases),
        "valid_count": len(valid),
        "passed_count": passed_count,
        "sample_pass_rate": sample_pass_rate,
        "hf_top1_accuracy": hf_top1_accuracy,
        "trtfb_top1_accuracy": trtfb_top1_accuracy,
        "metrics": metric_summaries,
        "gates": {**metric_thresholds, "min_sample_pass_rate": min_sample_pass_rate},
        "cases": cases,
    }


def write_encoder_embedding_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Encoder / Embedding Parity Summary",
        "",
        f"- status: {summary['status']}",
        f"- vector_pass_rate: {summary['vector_pass_rate']:.4f}",
        f"- mean_vector_cosine: {summary['mean_vector_cosine']:.6f}",
        f"- min_vector_cosine: {summary['min_vector_cosine']:.6f}",
        f"- max_pair_cosine_abs_delta: {summary['max_pair_cosine_abs_delta']:.6f}",
        f"- hf_sts_spearman: {_format_optional_float(summary.get('hf_sts_spearman'))}",
        f"- trtfb_sts_spearman: {_format_optional_float(summary.get('trtfb_sts_spearman'))}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_model_plugin_prediction_sets(
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    work_dir: Path,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    from tests.e2e_harness.contracts import StageStatus, ThresholdProfile

    hf_rows = hf_predictions.get("responses", [])
    trt_rows = trtfb_predictions.get("responses", [])
    requests = answers_data.get("requests", [])
    if not all(isinstance(rows, list) for rows in (hf_rows, trt_rows, requests)):
        raise ValueError(
            "model-plugin predictions and answers must contain response/request lists"
        )
    if len(hf_rows) != len(trt_rows) or len(hf_rows) != len(requests):
        raise ValueError(
            "HF predictions, TRTMC predictions, and model-plugin requests must "
            f"have the same length: {len(hf_rows)}, {len(trt_rows)}, "
            f"{len(requests)}"
        )

    manifest = work_manifest(work_dir)
    manifest_path = manifest_path_from_work_manifest(
        manifest,
        repo_root=REPO_ROOT,
    )
    min_sample_pass_rate = float(gates.get("min_sample_pass_rate", 1.0))
    threshold_overrides = {
        str(name): float(value)
        for name, value in gates.items()
        if name != "min_sample_pass_rate"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    cases: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = defaultdict(list)
    valid_count = 0
    passed_count = 0
    skipped_count = 0
    execution_errors: list[dict[str, Any]] = []

    for index, (request, hf_row, trt_row) in enumerate(
        zip(requests, hf_rows, trt_rows, strict=True)
    ):
        if not all(
            isinstance(row, dict) for row in (request, hf_row, trt_row)
        ):
            raise ValueError(f"model-plugin row {index} must contain objects")
        sample_id = str(
            request.get("sample_id", f"model_plugin_{index:06d}")
        )
        if (
            str(hf_row.get("sample_id", "")) != sample_id
            or str(trt_row.get("sample_id", "")) != sample_id
        ):
            raise ValueError(
                f"model-plugin sample id mismatch at index {index}: "
                f"expected={sample_id!r} hf={hf_row.get('sample_id')!r} "
                f"trtfb={trt_row.get('sample_id')!r}"
            )
        case, stage = select_case(
            manifest_path,
            request,
            source_index=index,
        )
        expected_case = str(
            case.metadata["validation_manifest_case_name"]
        )
        if (
            str(hf_row.get("testcase", "")) != expected_case
            or str(trt_row.get("testcase", "")) != expected_case
            or str(hf_row.get("stage", "")) != stage.name
            or str(trt_row.get("stage", "")) != stage.name
        ):
            raise ValueError(
                f"model-plugin contract mismatch for {sample_id}: "
                f"expected testcase={expected_case!r} stage={stage.name!r}"
            )
        activate_model_plugins(
            str(case.metadata.get("model_test_dir", "") or "")
        )
        comparator = get_comparator(case.task_strategy)
        if comparator is None:
            raise RuntimeError(
                f"No comparator plugin {case.task_strategy!r} for {case.family}"
            )
        hf_payload = hf_row.get("stage_output")
        trt_payload = trt_row.get("stage_output")
        if not isinstance(hf_payload, Mapping) or not isinstance(
            trt_payload, Mapping
        ):
            raise ValueError(
                f"model-plugin predictions for {sample_id} have no stage_output"
            )
        hf_output = deserialize_stage_output(hf_payload)
        trt_output = deserialize_stage_output(trt_payload)
        backend_failures = []
        for backend, output in (("hf", hf_output), ("trtmc", trt_output)):
            metadata = output.metadata if isinstance(output.metadata, Mapping) else {}
            data = output.data if isinstance(output.data, Mapping) else {}
            returncode = int(metadata.get("returncode", data.get("returncode", 0)) or 0)
            if returncode:
                backend_failures.append(
                    {
                        "backend": backend,
                        "returncode": returncode,
                        "stderr": str(
                            metadata.get("stderr", data.get("stderr", ""))
                        )[-1000:],
                    }
                )
        if backend_failures:
            execution_errors.append(
                {
                    "sample_id": sample_id,
                    "testcase": expected_case,
                    "stage": stage.name,
                    "failures": backend_failures,
                }
            )
            skipped_count += 1
            cases.append(
                {
                    "sample_id": sample_id,
                    "testcase": expected_case,
                    "stage": stage.name,
                    "passed": False,
                    "status": StageStatus.ERROR.value,
                    "message": "; ".join(
                        f"{failure['backend']} exited with "
                        f"returncode {failure['returncode']}"
                        for failure in backend_failures
                    ),
                    "composite_rule": "",
                    "metrics": {},
                }
            )
            continue
        threshold = ThresholdProfile(
            task_strategy=case.task_strategy,
            profile_name=case.comparison_profile,
            metrics={
                **{
                    str(name): float(value)
                    for name, value in case.threshold_overrides.items()
                },
                **threshold_overrides,
            },
        )
        comparison = comparator.compare(
            trt_output,
            hf_output,
            threshold,
            stage,
        )
        metrics = {
            name: {
                "value": float(metric.value),
                "threshold": metric.threshold,
                "operator": metric.operator,
                "passed": bool(metric.passed),
                "note": metric.note,
            }
            for name, metric in comparison.metrics.items()
        }
        for name, metric in comparison.metrics.items():
            metric_values[name].append(float(metric.value))
        is_valid = comparison.status in {
            StageStatus.PASSED.value,
            StageStatus.FAILED.value,
        }
        is_passed = comparison.status == StageStatus.PASSED.value
        valid_count += int(is_valid)
        passed_count += int(is_passed)
        skipped_count += int(not is_valid)
        cases.append(
            {
                "sample_id": sample_id,
                "testcase": expected_case,
                "stage": stage.name,
                "passed": is_passed,
                "status": comparison.status,
                "message": comparison.message,
                "composite_rule": comparison.composite_rule,
                "metrics": metrics,
            }
        )

    sample_pass_rate = passed_count / valid_count if valid_count else 0.0
    status = (
        "passed"
        if valid_count == len(cases)
        and sample_pass_rate >= min_sample_pass_rate
        else "failed"
    )
    metrics_summary = {
        name: {
            "mean": _mean(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
        for name, values in sorted(metric_values.items())
        if values
    }
    return {
        "status": status,
        "sample_count": len(cases),
        "valid_count": valid_count,
        "passed_count": passed_count,
        "skipped_count": skipped_count,
        "sample_pass_rate": sample_pass_rate,
        "metrics": metrics_summary,
        "gates": {"min_sample_pass_rate": min_sample_pass_rate},
        "cases": cases,
        "execution_errors": execution_errors,
    }


def run_full_duplex_bench_comparison(
    *,
    python: str,
    hf_predictions: Path,
    trtfb_predictions: Path,
    answers: Path,
    work_dir: Path,
    gates: Mapping[str, Any],
    local_files_only: bool,
) -> dict[str, Any]:
    """Run the dependency-heavy paper metric scorer in the reference env."""
    required_gates = (
        "max_tor_abs_delta",
        "max_backchannel_frequency_abs_delta",
        "max_backchannel_jsd_abs_delta",
    )
    missing = [name for name in required_gates if name not in gates]
    if missing:
        raise ValueError(
            "Full-Duplex-Bench scoring requires gates: " + ", ".join(missing)
        )
    output_path = work_dir / "summary.json"
    command = [
        python,
        str(REPO_ROOT / "tools" / "full_duplex_bench_score.py"),
        "--hf-predictions",
        str(hf_predictions),
        "--trtmc-predictions",
        str(trtfb_predictions),
        "--requests",
        str(answers),
        "--cache-root",
        str(work_dir / "full_duplex_bench_score_cache"),
        "--output",
        str(output_path),
        "--max-tor-abs-delta",
        str(float(gates["max_tor_abs_delta"])),
        "--max-backchannel-frequency-abs-delta",
        str(float(gates["max_backchannel_frequency_abs_delta"])),
        "--max-backchannel-jsd-abs-delta",
        str(float(gates["max_backchannel_jsd_abs_delta"])),
    ]
    if local_files_only:
        command.append("--local-files-only")
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    (work_dir / "full_duplex_bench_score.log").write_text(
        f"$ {' '.join(shlex.quote(token) for token in command)}\n"
        f"{completed.stdout}{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            "Full-Duplex-Bench scorer failed "
            f"(rc={completed.returncode}); see "
            f"{work_dir / 'full_duplex_bench_score.log'}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            "Full-Duplex-Bench scorer failed without producing summary.json "
            f"(rc={completed.returncode}); see "
            f"{work_dir / 'full_duplex_bench_score.log'}"
        )
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    expected_status = "passed" if completed.returncode == 0 else "failed"
    if summary.get("status") != expected_status:
        raise RuntimeError(
            "Full-Duplex-Bench scorer exit status does not match summary; "
            f"see {work_dir / 'full_duplex_bench_score.log'}"
        )
    return summary


def eval_one_model(
    *,
    suite: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    suite = resolve_suite_for_model(suite, model)
    work_root = Path(args.work_root)
    work_dir = work_root / suite["id"] / str(model["name"])
    dataset_path = Path(args.dataset or suite.get("dataset", {}).get("default_path", ""))
    if not dataset_path:
        raise ValueError(f"Suite {suite['id']} has no dataset path; pass --dataset")
    scorer = str(suite.get("scoring", {}).get("scorer", "mcq"))
    dataset_kind = str(suite.get("dataset", {}).get("kind", ""))
    reference_mode = str(suite.get("reference", {}).get("mode", "") or "")
    no_hf_reference = reference_mode in {"gold_only", "metric_only"}
    reference_backend = reference_mode or str(
        model.get("reference_backend", "hf_transformers") or "hf_transformers"
    )
    validation_config = effective_validation_config(suite, model)
    model = apply_comparison_precision(model, validation_config)
    suite_reference = suite.get("reference", {})
    if isinstance(suite_reference, dict) and suite_reference:
        validation_config["reference"] = suite_reference
    if model.get("manifest"):
        validation_config["model_manifest"] = str(model["manifest"])
    if model.get("family"):
        validation_config["family"] = str(model["family"])
    if model.get("task_strategy"):
        validation_config["task_strategy"] = str(model["task_strategy"])
    runtime_config = model.get("runtime_config", {})
    if isinstance(runtime_config, dict) and runtime_config:
        validation_config["runtime_config"] = runtime_config
    if model.get("max_new_tokens") is not None:
        validation_config["model_max_new_tokens"] = model["max_new_tokens"]
    prepare_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        limit=args.limit,
        subject=args.subject,
        sample_seed=args.sample_seed,
        validation_config=validation_config,
    )
    precision_contract = (
        None
        if no_hf_reference
        else resolve_reference_precision_contract(args, model, work_dir)
    )

    prompt_token_limit = int(validation_config.get("prompt_token_limit", 0) or 0)
    prompt_normalization: dict[str, Any] | None = None
    if prompt_token_limit:
        if bool(suite.get("generation", {}).get("apply_chat_template", False)):
            raise ValueError(
                "prompt_token_limit requires apply_chat_template=false so HF and TRTFB "
                "consume the same normalized prompt"
            )
        prompt_normalization = apply_work_prompt_token_limit(
            work_dir=work_dir,
            model_id=str(model["hf_id"]),
            model_revision=str(model.get("hf_revision", "") or ""),
            token_limit=prompt_token_limit,
            truncation_side=str(validation_config.get("prompt_truncation_side", "left")),
            local_files_only=args.local_files_only,
            trust_remote_code=(
                args.trust_remote_code or bool(model.get("trust_remote_code", False))
            ),
        )

    answers_path = work_dir / "answers.json"
    hf_reused = False
    if not no_hf_reference:
        # Run HF in its own process so its GPU memory is fully reclaimed before
        # the TRT bundle build and TRTFB inference for this model.
        run_hf_reference_subprocess(args, model, work_dir)
    hf_cache = reference_cache_metadata(work_dir)
    if hf_cache.get("status") in {"reused", "adopted"}:
        hf_reused = True

    if args.bundle:
        if len(args.model or []) != 1:
            raise ValueError("--bundle can only be used when exactly one --model is selected")
        bundle_path = Path(args.bundle)
    else:
        engine_dir = Path(args.engine_dir or (work_root / "_bundles"))
        bundle_path = engine_dir / str(model["bundle"])
    if getattr(args, "require_prebuilt_bundles", False) and not bundle_path.is_file():
        raise FileNotFoundError(f"Required prebuilt validation bundle is missing: {bundle_path}")

    max_prompt_len = None
    if (
        not args.skip_prompt_length_check
        and not _is_asr_dataset_kind(dataset_kind)
        and not _is_diffusion_media_dataset_kind(dataset_kind)
        and not _is_diffusion_text_dataset_kind(dataset_kind)
        and not _is_tts_dataset_kind(dataset_kind)
        and not _is_time_series_dataset_kind(dataset_kind)
        and not _is_vision_task_dataset_kind(dataset_kind)
        and not _is_model_plugin_dataset_kind(dataset_kind)
    ):
        prompt_rows_path = work_dir / "prompts.jsonl"
        max_prompt_len = max_prompt_token_length(
            model_id=str(model["hf_id"]),
            model_revision=str(model.get("hf_revision", "") or ""),
            prompts_path=prompt_rows_path,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code or bool(model.get("trust_remote_code", False)),
        )
    generation = generation_defaults(work_dir)
    generation_headroom = generation_cache_headroom(
        scorer=scorer,
        validation_config=validation_config,
        generation=generation,
        max_new_tokens=args.max_new_tokens,
    )
    required_prompt_cache = (
        int(max_prompt_len or 0) + generation_headroom if max_prompt_len is not None else None
    )
    max_cache_length = requested_build_max_cache_length(
        suite,
        model,
        args.build_max_cache_length,
        prompt_max_tokens=required_prompt_cache,
    )
    if required_prompt_cache is not None and required_prompt_cache > max_cache_length:
        raise RuntimeError(
            f"Dataset prompt and generation exceed bundle cache for {model['name']}: "
            f"max_prompt_tokens={max_prompt_len}, "
            f"generation_cache_headroom={generation_headroom}, "
            f"required_cache_length={required_prompt_cache}, "
            f"build_max_cache_length={max_cache_length}. "
            "Use a smaller dataset slice/subject or set --build-max-cache-length high enough "
            "for this model and TensorRT target."
        )
    bundle_path, built = ensure_bundle(
        model,
        bundle_path=bundle_path,
        trtmc_binary=args.trtmc_binary,
        max_cache_length=max_cache_length,
        force_build=args.force_build,
        replace_existing=bool(
            getattr(args, "replace_bundle_on_build", False)
        ),
        extra_build_args=args.extra_build_arg,
        log_path=work_dir / "build.log",
        cuda_visible_devices=args.cuda_visible_devices,
    )

    if (
        scorer == "continuation"
        and dataset_kind in {"mmlu_five_shot_json", "text_generation_json"}
        and not args.apply_chat_template
        and not bool(generation.get("apply_chat_template", False))
    ):
        validate_text_input_token_contract(
            model=model,
            work_dir=work_dir,
            bundle_path=bundle_path,
            local_files_only=args.local_files_only,
            trust_remote_code=(
                args.trust_remote_code
                or bool(model.get("trust_remote_code", False))
            ),
        )

    # TRT inference intentionally runs every eval invocation so runtime changes
    # are never hidden behind stale predictions.
    run_trtfb(_namespace_for_run_trtfb(args, bundle_path, work_dir))

    base_result = {
        "suite": suite["id"],
        "model": model["name"],
        "hf_id": model["hf_id"],
        "work_dir": str(work_dir),
        "bundle": str(bundle_path),
        "build_max_cache_length": max_cache_length,
        "max_prompt_tokens": max_prompt_len,
        "generation_cache_headroom": generation_headroom,
        "reference_backend": reference_backend,
        "hf_reference_status": reference_mode
        if no_hf_reference
        else "reused"
        if hf_reused
        else "ran",
        "hf_reused": hf_reused,
        "hf_cache_status": str(hf_cache.get("status", "") or ""),
        "hf_cache_key": str(hf_cache.get("key", "") or ""),
        "bundle_built": built,
        "model_plugin_dir": str(getattr(args, "model_plugin_dir", "") or ""),
    }
    if precision_contract is not None:
        base_result["reference_dtype"] = precision_contract["reference_dtype"]
        base_result["precision_contract"] = precision_contract
    if prompt_normalization is not None:
        base_result["prompt_normalization"] = prompt_normalization

    if scorer == "full_duplex_bench_behavior_parity":
        scoring = suite.get("scoring", {})
        scorer_profile = str(scoring.get("python_profile", "") or "")
        base_python = str(getattr(args, "hf_python", "") or sys.executable)
        scorer_python = (
            resolve_profile_python(scorer_profile, base_python)
            if scorer_profile
            else model_reference_python(model, base_python)
        )
        summary = run_full_duplex_bench_comparison(
            python=scorer_python,
            hf_predictions=work_dir / "hf_predictions.json",
            trtfb_predictions=work_dir / "trtfb_predictions.json",
            answers=answers_path,
            work_dir=work_dir,
            gates=suite.get("gates", {}),
            local_files_only=bool(args.local_files_only),
        )
        report_metrics: dict[str, dict[str, float]] = {}
        for metric_name, metric in summary["metrics"].items():
            for value_name in (
                "hf",
                "trtmc",
                "abs_delta",
                "threshold",
                "paired_changed_count",
                "paired_mean_abs_delta",
                "paired_max_abs_delta",
            ):
                report_metrics[f"{metric_name}.{value_name}"] = {
                    "mean": float(metric[value_name])
                }
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "sample_count": summary["sample_count"],
            "valid_count": summary["valid_count"],
            "passed_count": summary["passed_count"],
            "metric_gate_count": summary["metric_gate_count"],
            "metric_gate_pass_rate": summary["metric_gate_pass_rate"],
            "metrics": report_metrics,
            "gates": summary["gates"],
            "gate_failures": summary["gate_failures"],
            "benchmark_provenance": {
                "dataset": summary.get("dataset", ""),
                "dataset_source_revision": summary.get(
                    "dataset_source_revision", ""
                ),
                "dataset_selection_seed": summary.get(
                    "dataset_selection_seed", ""
                ),
                "samples_per_category": summary.get(
                    "samples_per_category", 0
                ),
                "evaluator_revision": summary.get("evaluator_revision", ""),
                "asr_model": summary.get("asr_model", ""),
                "asr_revision": summary.get("asr_revision", ""),
            },
        }
        if summary["gate_failures"]:
            result.update(
                {
                    "error_type": "BenchmarkGateError",
                    "error": (
                        f"{len(summary['gate_failures'])} Full-Duplex-Bench "
                        "HF/TRTMC metric delta gate(s) failed"
                    ),
                }
            )
    elif scorer == "model_plugin_parity":
        hf_data = json.loads(
            (work_dir / "hf_predictions.json").read_text(encoding="utf-8")
        )
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        summary = compare_model_plugin_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            work_dir=work_dir,
            gates=suite.get("gates", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "passed_count": summary["passed_count"],
            "skipped_count": summary["skipped_count"],
            "sample_pass_rate": summary["sample_pass_rate"],
            "metrics": summary["metrics"],
        }
        if summary["execution_errors"]:
            result["error_type"] = "ModelPluginExecutionError"
            result["error"] = (
                f"{len(summary['execution_errors'])} model-plugin sample(s) "
                "failed during native backend execution"
            )
            result["execution_errors"] = summary["execution_errors"]
    elif scorer == "time_series_parity":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_time_series_prediction_sets(
            hf_data,
            trtfb_data,
            gates=suite.get("gates", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_time_series_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "passed_count": summary["passed_count"],
            "sample_agreement_rate": summary["sample_agreement_rate"],
            "prediction_agreement_rate": summary["sample_agreement_rate"],
            "mean_relative_l2": summary["mean_relative_l2"],
            "max_relative_l2": summary["max_relative_l2"],
            "max_absolute_error": summary["max_absolute_error"],
        }
    elif scorer == "image_classification_parity":
        hf_data = json.loads(
            (work_dir / "hf_predictions.json").read_text(encoding="utf-8")
        )
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        summary = compare_image_classification_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            gates=suite.get("gates", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "hf_top1_accuracy": summary["hf_top1_accuracy"],
            "trtfb_top1_accuracy": summary["trtfb_top1_accuracy"],
            "top1_accuracy_drop_from_hf": summary["top1_accuracy_drop_from_hf"],
            "top1_agreement": summary["top1_agreement"],
        }
    elif scorer == "semantic_segmentation_parity":
        hf_data = json.loads(
            (work_dir / "hf_predictions.json").read_text(encoding="utf-8")
        )
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        dataset_config = suite.get("dataset", {})
        summary = compare_semantic_segmentation_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            gates=suite.get("gates", {}),
            num_classes=int(dataset_config.get("num_classes", 150)),
            ignore_index=int(dataset_config.get("ignore_index", 255)),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "hf_mean_iou": summary["hf_mean_iou"],
            "trtfb_mean_iou": summary["trtfb_mean_iou"],
            "mean_iou_drop_from_hf": summary["mean_iou_drop_from_hf"],
            "backend_mean_iou": summary["backend_mean_iou"],
            "backend_pixel_agreement": summary["backend_pixel_agreement"],
        }
    elif scorer == "prompted_segmentation_parity":
        hf_data = json.loads(
            (work_dir / "hf_predictions.json").read_text(encoding="utf-8")
        )
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        summary = compare_prompted_segmentation_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            gates=suite.get("gates", {}),
            prompt_mode=str(validation_config.get("prompt_mode", "")),
            ground_truth_mask_field=str(
                validation_config.get("ground_truth_mask_field", "instance_mask")
            ),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "prompt_mode": summary["prompt_mode"],
            "mean_backend_mask_iou": summary["mean_backend_mask_iou"],
            "hf_mean_ground_truth_iou": summary["hf_mean_ground_truth_iou"],
            "trtfb_mean_ground_truth_iou": summary["trtfb_mean_ground_truth_iou"],
            "ground_truth_iou_drop_from_hf": summary[
                "ground_truth_iou_drop_from_hf"
            ],
        }
    elif scorer == "reranking_parity":
        hf_data = json.loads(
            (work_dir / "hf_predictions.json").read_text(encoding="utf-8")
        )
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        _template, _reference, _runner, comparator = (
            _load_reranking_validation_plugins(work_dir)
        )
        summary = compare_reranking_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            gates=suite.get("gates", {}),
            comparator=comparator,
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "passed_count": summary["passed_count"],
            "sample_pass_rate": summary["sample_pass_rate"],
            "hf_top1_accuracy": summary["hf_top1_accuracy"],
            "trtfb_top1_accuracy": summary["trtfb_top1_accuracy"],
            "mean_pairwise_ordering_agreement": summary["metrics"][
                "pairwise_ordering_agreement"
            ]["mean"],
            "min_pairwise_ordering_agreement": summary["metrics"][
                "pairwise_ordering_agreement"
            ]["min"],
        }
    elif scorer == "encoder_embedding_parity":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads(
            (work_dir / "trtfb_predictions.json").read_text(encoding="utf-8")
        )
        summary = compare_encoder_embedding_prediction_sets(
            hf_data,
            trtfb_data,
            gates=suite.get("gates", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_encoder_embedding_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": scorer,
            "status": summary["status"],
            "valid_count": summary["valid_count"],
            "pair_count": summary["pair_count"],
            "vector_pass_rate": summary["vector_pass_rate"],
            "mean_vector_cosine": summary["mean_vector_cosine"],
            "min_vector_cosine": summary["min_vector_cosine"],
            "mean_pair_cosine_abs_delta": summary["mean_pair_cosine_abs_delta"],
            "max_pair_cosine_abs_delta": summary["max_pair_cosine_abs_delta"],
            "hf_sts_spearman": summary["hf_sts_spearman"],
            "trtfb_sts_spearman": summary["trtfb_sts_spearman"],
        }
    elif scorer == "diffusion_text_parity":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_diffusion_text_prediction_sets(hf_data, trtfb_data)
        task_metric = str(suite.get("scoring", {}).get("task_metric", "") or "")
        answers_data = json.loads(answers_path.read_text(encoding="utf-8"))
        diagnostics: dict[str, Any] = {}
        if task_metric == "sacrebleu":
            for label, data in (("hf", hf_data), ("trtfb", trtfb_data)):
                score = score_sacrebleu_predictions(data, answers_data)
                diagnostics[f"{label}_corpus_bleu"] = score["corpus_bleu"]
        elif task_metric == "rouge":
            for label, data in (("hf", hf_data), ("trtfb", trtfb_data)):
                score = score_rouge_predictions(data, answers_data)
                diagnostics[f"{label}_rouge1"] = score["rouge1"]
                diagnostics[f"{label}_rouge2"] = score["rouge2"]
                diagnostics[f"{label}_rouge_l"] = score["rouge_l"]
        elif task_metric == "unconditional_text_quality":
            scoring = suite.get("scoring", {})
            perplexity_model = str(scoring.get("perplexity_model", "") or "")
            for label, data in (("hf", hf_data), ("trtfb", trtfb_data)):
                texts = [str(row.get("output_text", "")) for row in data.get("responses", [])]
                metrics = compute_gpt2_generation_metrics(
                    texts,
                    model_id=perplexity_model,
                    device=str(scoring.get("perplexity_device", "cuda") or "cuda"),
                    local_files_only=args.local_files_only,
                )
                diagnostics[f"{label}_generation_ppl"] = metrics["generation_ppl"]
                diagnostics[f"{label}_unigram_entropy"] = metrics["unigram_entropy"]
        diagnostics.update(diffusion_text_task_metric_deltas(task_metric, diagnostics))
        summary["task_quality_diagnostics"] = diagnostics
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        result = {
            **base_result,
            "mode": scorer,
            "valid_count": summary["valid_count"],
            "token_agreement_rate": summary["token_agreement_rate"],
            "shared_sampling_inputs_match_rate": summary["shared_sampling_inputs_match_rate"],
            **diagnostics,
        }
        apply_metric_gates(result, suite.get("gates", {}))
    elif scorer in {"sacrebleu", "rouge", "unconditional_text_quality"} and no_hf_reference:
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        answers_data = json.loads(answers_path.read_text(encoding="utf-8"))
        if scorer == "sacrebleu":
            summary = score_sacrebleu_predictions(trtfb_data, answers_data)
            metric_names = ("corpus_bleu", "non_empty_rate", "valid_count", "skipped_count")
        elif scorer == "rouge":
            summary = score_rouge_predictions(trtfb_data, answers_data)
            metric_names = (
                "rouge1",
                "rouge2",
                "rouge_l",
                "non_empty_rate",
                "valid_count",
                "skipped_count",
            )
        else:
            scoring = suite.get("scoring", {})
            perplexity_model = str(scoring.get("perplexity_model", "") or "")
            generation_metrics: dict[str, float] = {}
            if perplexity_model:
                generation_metrics = compute_gpt2_generation_metrics(
                    [str(row.get("output_text", "")) for row in trtfb_data.get("responses", [])],
                    model_id=perplexity_model,
                    device=str(scoring.get("perplexity_device", "cuda") or "cuda"),
                    local_files_only=args.local_files_only,
                )
            summary = score_unconditional_text_predictions(
                trtfb_data,
                answers_data,
                generation_ppl=generation_metrics.get("generation_ppl"),
                unigram_entropy=generation_metrics.get("unigram_entropy"),
            )
            metric_names = (
                "generation_ppl",
                "unigram_entropy",
                "distinct_1",
                "distinct_2",
                "token_repetition_rate",
                "mean_token_count",
                "non_empty_rate",
                "valid_count",
                "skipped_count",
            )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_diffusion_text_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": scorer,
            **{name: summary.get(name) for name in metric_names},
        }
        apply_metric_gates(result, suite.get("gates", {}))
    elif scorer == "continuation":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_continuation_sets(
            hf_data,
            trtfb_data,
            require_token_ids=True,
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_continuation_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": "continuation",
            "evaluation_policy": (
                "threshold_gated" if suite.get("gates", {}) else "diagnostic_only"
            ),
            "comparison_granularity": summary.get("comparison_granularity", ""),
            "exact_match_rate": summary["exact_match_rate"],
            "tie_adjusted_exact_match_rate": summary[
                "tie_adjusted_exact_match_rate"
            ],
            "reference_tie_equivalent_count": summary[
                "reference_tie_equivalent_count"
            ],
            "token_prefix_agreement": summary["token_prefix_agreement"],
            "mean_first_divergence": summary["mean_first_divergence"],
            "divergent_count": summary["divergent_count"],
            "divergence_rate": summary["divergence_rate"],
            "mean_divergent_first_divergence": summary[
                "mean_divergent_first_divergence"
            ],
            "mean_divergent_prefix_ratio": summary["mean_divergent_prefix_ratio"],
            "min_divergent_prefix_ratio": summary["min_divergent_prefix_ratio"],
            "mean_divergent_severity": summary["mean_divergent_severity"],
            "max_divergent_severity": summary["max_divergent_severity"],
            # Preserve the historical generic agreement field for downstream
            # consumers; the divergence fields above are the primary output.
            "prediction_agreement_rate": summary["token_prefix_agreement"],
        }
        diagnostics = continuation_task_quality_diagnostics(
            str(suite.get("scoring", {}).get("task_metric", "")),
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
        )
        if diagnostics:
            result.update(
                {
                    "hf_corpus_bleu": diagnostics["hf_corpus_bleu"],
                    "trtfb_corpus_bleu": diagnostics["trtfb_corpus_bleu"],
                    "corpus_bleu_abs_delta": diagnostics["corpus_bleu_abs_delta"],
                }
            )
            summary["task_quality_diagnostics"] = {
                "hf": diagnostics["hf"],
                "trtfb": diagnostics["trtfb"],
                "corpus_bleu_abs_delta": diagnostics["corpus_bleu_abs_delta"],
            }
            (work_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        apply_metric_gates(result, suite.get("gates", {}))
    elif scorer == "diffusion_image_clip_parity":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_diffusion_image_predictions(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            work_dir=work_dir,
            gates=suite.get("gates", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_diffusion_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": scorer,
            "overall_pass_rate": summary["overall_pass_rate"],
            "passed_count": summary["passed_count"],
            "valid_count": summary["valid_count"],
            "skipped_count": summary["skipped_count"],
            "metrics": summary["metrics"],
            "status": (
                "passed"
                if summary["valid_count"] > 0
                and summary["passed_count"] == summary["valid_count"]
                and summary["skipped_count"] == 0
                else "failed"
            ),
        }
    else:
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_prediction_sets(
            hf_data,
            trtfb_data,
            json.loads(answers_path.read_text(encoding="utf-8")),
            scorer=scorer,
            answer_parser=str(validation_config.get("answer_parser", "") or ""),
            require_valid_prediction=bool(
                validation_config.get("require_valid_prediction", False)
            ),
            scorer_options=suite.get("scoring", {}),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": scorer,
            "hf_accuracy": summary["hf"]["overall_accuracy"],
            "trtfb_accuracy": summary["trtfb"]["overall_accuracy"],
            "accuracy_delta_trtfb_minus_hf": summary["accuracy_delta_trtfb_minus_hf"],
            "tie_adjusted_accuracy_delta_trtfb_minus_hf": summary[
                "tie_adjusted_accuracy_delta_trtfb_minus_hf"
            ],
            "reference_tie_equivalent_count": summary[
                "reference_tie_equivalent_count"
            ],
            "prediction_agreement_rate": summary["prediction_agreement_rate"],
            "hf_valid_prediction_rate": summary["hf"].get("valid_prediction_rate"),
            "trtfb_valid_prediction_rate": summary["trtfb"].get("valid_prediction_rate"),
        }
        if scorer in {"grounding_iou", "mcq", "asr_transcript"}:
            result.update(
                prediction_agreement_gate_result(
                    summary,
                    suite.get("gates", {}),
                )
            )
            if scorer == "asr_transcript":
                result.update(
                    {
                    "normalized_transcript_exact_agreement_rate": summary[
                        "normalized_transcript_exact_agreement_rate"
                    ],
                    "correctness_agreement_rate": summary[
                        "correctness_agreement_rate"
                    ],
                    }
                )
        elif scorer == "tts_intelligibility":
            result.update(
                tts_intelligibility_gate_result(
                    summary,
                    suite.get("gates", {}),
                )
            )
    (work_dir / "eval_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# Continuation divergence diagnostics for base / completion models
# (generation-only, no logits).
# HF reference and TRTFB both greedily generate from the same plain-text prompt;
# we compare the two continuations. No gold answer or logprobs are needed. The
# primary metrics describe how often they diverge and how much of each divergent
# continuation remains after its first differing token. Legacy agreement fields
# remain in the result for compatibility.
# ---------------------------------------------------------------------------


def _first_divergence(a: list[Any], b: list[Any]) -> int:
    """Index of the first differing element; min length if one is a prefix."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def compare_continuation_sets(
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    tokenize: Any = None,
    require_token_ids: bool = False,
) -> dict[str, Any]:
    """Compare HF vs TRTFB greedy continuations.

    Prefer generated token ids emitted by the runners. ``tokenize`` is only a
    compatibility fallback for older prediction files that do not contain ids.
    """
    hf_rows = hf_predictions.get("responses", [])
    trt_rows = trtfb_predictions.get("responses", [])
    if not isinstance(hf_rows, list) or not isinstance(trt_rows, list):
        raise ValueError("HF and TRTFB predictions must contain response lists")
    if len(hf_rows) != len(trt_rows):
        raise ValueError(
            f"HF and TRTFB predictions must have the same length: {len(hf_rows)} != {len(trt_rows)}"
        )
    if not all(isinstance(row, dict) for row in hf_rows + trt_rows):
        raise ValueError("HF and TRTFB predictions must contain object rows")

    hf_id_rows = [_generated_token_ids(row) for row in hf_rows]
    trt_id_rows = [_generated_token_ids(row) for row in trt_rows]
    has_all_token_ids = all(ids is not None for ids in hf_id_rows + trt_id_rows)
    if require_token_ids and not has_all_token_ids:
        missing = []
        for label, rows in (("hf", hf_id_rows), ("trtfb", trt_id_rows)):
            missing.extend(f"{label}[{idx}]" for idx, ids in enumerate(rows) if ids is None)
        raise ValueError(
            "Continuation token-id metric requires generated_token_ids in every prediction row; "
            f"missing: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}"
        )
    if has_all_token_ids:
        comparison_granularity = "generated_token_ids"
    elif tokenize is not None:
        comparison_granularity = "retokenized_output_text"
    else:
        comparison_granularity = "characters"

        def tokenize(s: str) -> list[str]:
            return list(s)

    exact = 0
    tie_adjusted_exact = 0
    text_exact = 0
    total_matched = 0
    total_reference = 0
    div_positions: list[int] = []
    divergent_positions: list[int] = []
    divergent_prefix_ratios: list[float] = []
    divergent_severities: list[float] = []
    samples: list[dict[str, Any]] = []
    reference_tie_equivalent_samples: list[dict[str, Any]] = []
    for idx, (hf_row, trt_row) in enumerate(zip(hf_rows, trt_rows, strict=True)):
        hf_text = str(hf_row.get("output_text", ""))
        trt_text = str(trt_row.get("output_text", ""))
        text_is_exact = hf_text == trt_text
        text_exact += int(text_is_exact)
        if has_all_token_ids:
            hf_tokens = hf_id_rows[idx] or []
            trt_tokens = trt_id_rows[idx] or []
        else:
            hf_tokens = tokenize(hf_text)
            trt_tokens = tokenize(trt_text)
        is_exact = hf_tokens == trt_tokens
        exact += int(is_exact)
        divergence = _first_divergence(hf_tokens, trt_tokens)
        reference_tie_equivalent = False
        max_score_token_ids: list[int] = []
        if has_all_token_ids and not is_exact:
            max_score_steps = hf_row.get("generated_token_max_score_ids", [])
            if (
                isinstance(max_score_steps, list)
                and divergence < len(max_score_steps)
                and isinstance(max_score_steps[divergence], list)
                and len(max_score_steps[divergence]) > 1
                and divergence < len(hf_tokens)
                and divergence < len(trt_tokens)
            ):
                try:
                    max_score_token_ids = [
                        int(token_id) for token_id in max_score_steps[divergence]
                    ]
                    reference_tie_equivalent = (
                        int(hf_tokens[divergence]) in max_score_token_ids
                        and int(trt_tokens[divergence]) in max_score_token_ids
                    )
                except (TypeError, ValueError):
                    max_score_token_ids = []
        tie_adjusted_exact += int(is_exact or reference_tie_equivalent)
        if reference_tie_equivalent:
            reference_tie_equivalent_samples.append(
                {
                    "index": idx,
                    "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                    "first_divergence": divergence,
                    "hf_token_id": int(hf_tokens[divergence]),
                    "trtfb_token_id": int(trt_tokens[divergence]),
                    "max_score_token_ids": max_score_token_ids,
                }
            )
        reference_len = max(1, len(hf_tokens), len(trt_tokens))
        normalized_divergence = 1.0 if is_exact else divergence / reference_len
        divergence_severity = (
            0.0 if is_exact else (reference_len - divergence) / reference_len
        )
        total_matched += min(divergence, reference_len)
        total_reference += reference_len
        div_positions.append(divergence)
        if not is_exact:
            divergent_positions.append(divergence)
            divergent_prefix_ratios.append(normalized_divergence)
            divergent_severities.append(divergence_severity)
        hf_token_at_divergence = hf_tokens[divergence] if divergence < len(hf_tokens) else None
        trt_token_at_divergence = trt_tokens[divergence] if divergence < len(trt_tokens) else None
        samples.append(
            {
                "index": idx,
                "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                "exact": is_exact,
                "text_exact": text_is_exact,
                "diverged": not is_exact,
                "first_divergence": divergence,
                "normalized_first_divergence": normalized_divergence,
                "divergence_severity": divergence_severity,
                "hf_len": len(hf_tokens),
                "trtfb_len": len(trt_tokens),
                "hf_token_at_divergence": hf_token_at_divergence,
                "trtfb_token_at_divergence": trt_token_at_divergence,
                "reference_tie_equivalent": reference_tie_equivalent,
            }
        )

    count = len(hf_rows)
    exact_rate = (exact / count) if count else 0.0
    tie_adjusted_exact_rate = (tie_adjusted_exact / count) if count else 0.0
    prefix_agreement = (total_matched / total_reference) if total_reference else 0.0
    mean_divergence = (sum(div_positions) / count) if count else 0.0
    divergent_count = len(divergent_positions)
    return {
        "comparison_granularity": comparison_granularity,
        "divergence_metric_scope": "divergent_samples_only",
        "normalization_denominator": "max_hf_trtfb_generated_length",
        "exact_match_rate": exact_rate,
        "tie_adjusted_exact_match_rate": tie_adjusted_exact_rate,
        "token_id_exact_match_rate": exact_rate if has_all_token_ids else None,
        "text_exact_match_rate": (text_exact / count) if count else 0.0,
        "token_prefix_agreement": prefix_agreement,
        "token_id_prefix_agreement": prefix_agreement if has_all_token_ids else None,
        "mean_first_divergence": mean_divergence,
        "mean_first_token_id_divergence": mean_divergence if has_all_token_ids else None,
        "divergent_count": divergent_count,
        "divergence_rate": divergent_count / count if count else 0.0,
        "mean_divergent_first_divergence": (
            sum(divergent_positions) / divergent_count if divergent_count else None
        ),
        "mean_divergent_prefix_ratio": (
            sum(divergent_prefix_ratios) / divergent_count if divergent_count else None
        ),
        "min_divergent_prefix_ratio": (
            min(divergent_prefix_ratios) if divergent_prefix_ratios else None
        ),
        "mean_divergent_severity": (
            sum(divergent_severities) / divergent_count if divergent_count else 0.0
        ),
        "max_divergent_severity": (
            max(divergent_severities) if divergent_severities else 0.0
        ),
        "count": count,
        "exact_count": exact,
        "tie_adjusted_exact_count": tie_adjusted_exact,
        "text_exact_count": text_exact,
        "reference_tie_equivalent_count": len(reference_tie_equivalent_samples),
        "reference_tie_equivalent_samples": reference_tie_equivalent_samples,
        "samples": samples,
    }


def cmd_compare_continuation(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    hf_path = Path(args.hf_predictions) if args.hf_predictions else work_dir / "hf_predictions.json"
    trtfb_path = (
        Path(args.trtfb_predictions)
        if args.trtfb_predictions
        else work_dir / "trtfb_predictions.json"
    )
    tokenize = None
    if args.model:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        tokenize = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    summary = compare_continuation_sets(
        json.loads(hf_path.read_text(encoding="utf-8")),
        json.loads(trtfb_path.read_text(encoding="utf-8")),
        tokenize=tokenize,
    )
    output_path = Path(args.output) if args.output else work_dir / "continuation_parity.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"divergent_count={summary['divergent_count']} "
        f"divergence_rate={summary['divergence_rate']:.4f} "
        f"mean_divergent_first_divergence="
        f"{_format_optional_float(summary['mean_divergent_first_divergence'], precision=2)} "
        f"mean_divergent_prefix_ratio="
        f"{_format_optional_float(summary['mean_divergent_prefix_ratio'])} "
        f"min_divergent_prefix_ratio="
        f"{_format_optional_float(summary['min_divergent_prefix_ratio'])} "
        f"mean_divergent_severity={summary['mean_divergent_severity']:.4f} "
        f"max_divergent_severity={summary['max_divergent_severity']:.4f} "
        f"output={output_path}"
    )
    return 0


def _model_tokenizer(model: dict[str, Any], args: argparse.Namespace) -> Any:
    """Return a tokenize(str)->list[int] using the model tokenizer, or None."""
    try:
        from transformers import AutoTokenizer

        tokenizer_kwargs = {
            "trust_remote_code": getattr(args, "trust_remote_code", False)
            or bool(model.get("trust_remote_code", False)),
            "local_files_only": getattr(args, "local_files_only", False),
        }
        model_revision = str(model.get("hf_revision", "") or "")
        if model_revision:
            tokenizer_kwargs["revision"] = model_revision
        tok = AutoTokenizer.from_pretrained(str(model["hf_id"]), **tokenizer_kwargs)
        return lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    except Exception:
        return None


def _format_optional_float(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def write_continuation_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Continuation Divergence Summary",
        "",
        "`first_divergence` is the number of generated tokens that match before the "
        "first difference. Prefix and severity aggregates include divergent samples "
        "only; exact samples do not dilute divergence severity.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| comparison_granularity | {summary.get('comparison_granularity', '')} |",
        f"| divergence_metric_scope | {summary.get('divergence_metric_scope', '')} |",
        f"| normalization_denominator | {summary.get('normalization_denominator', '')} |",
        f"| sample_count | {summary['count']} |",
        f"| divergent_count | {summary['divergent_count']} |",
        f"| divergence_rate | {summary['divergence_rate']:.4f} |",
        f"| mean_divergent_first_divergence | {_format_optional_float(summary.get('mean_divergent_first_divergence'), precision=2)} |",
        f"| mean_divergent_prefix_ratio | {_format_optional_float(summary.get('mean_divergent_prefix_ratio'))} |",
        f"| min_divergent_prefix_ratio | {_format_optional_float(summary.get('min_divergent_prefix_ratio'))} |",
        f"| mean_divergent_severity | {summary['mean_divergent_severity']:.4f} |",
        f"| max_divergent_severity | {summary['max_divergent_severity']:.4f} |",
        "",
        "## Compatibility Diagnostics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| exact_match_rate | {summary['exact_match_rate']:.4f} |",
        f"| tie_adjusted_exact_match_rate | {summary['tie_adjusted_exact_match_rate']:.4f} |",
        f"| reference_tie_equivalent_count | {summary['reference_tie_equivalent_count']} |",
        f"| token_id_exact_match_rate | {_format_optional_float(summary.get('token_id_exact_match_rate'))} |",
        f"| text_exact_match_rate | {summary['text_exact_match_rate']:.4f} |",
        f"| token_prefix_agreement | {summary['token_prefix_agreement']:.4f} |",
        f"| token_id_prefix_agreement | {_format_optional_float(summary.get('token_id_prefix_agreement'))} |",
        f"| mean_first_divergence | {summary['mean_first_divergence']:.2f} |",
        f"| mean_first_token_id_divergence | {_format_optional_float(summary.get('mean_first_token_id_divergence'), precision=2)} |",
    ]
    divergent_samples = [sample for sample in summary.get("samples", []) if sample["diverged"]]
    lines.extend(["", "## Divergent Samples", ""])
    if not divergent_samples:
        lines.append("No divergent samples.")
    else:
        lines.extend(
            [
                "| Sample | First divergence | HF length | TRTFB length | Prefix ratio | Severity | HF token | TRTFB token |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for sample in divergent_samples:
            lines.append(
                f"| {sample['sample_id']} | {sample['first_divergence']} | "
                f"{sample['hf_len']} | {sample['trtfb_len']} | "
                f"{sample['normalized_first_divergence']:.4f} | "
                f"{sample['divergence_severity']:.4f} | "
                f"{sample['hf_token_at_divergence']} | "
                f"{sample['trtfb_token_at_divergence']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diffusion_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diffusion Image CLIP Parity Summary",
        "",
        f"- overall_pass_rate: {summary['overall_pass_rate']:.4f}",
        f"- passed: {summary['passed_count']}/{summary['valid_count']}",
        f"- skipped: {summary['skipped_count']}",
        "",
        "| Metric | Mean | Min | Max | Gated passed |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric_name, metric in sorted(summary.get("metrics", {}).items()):
        lines.append(
            f"| {metric_name} | {metric['mean']:.4f} | {metric['min']:.4f} | "
            f"{metric['max']:.4f} | {metric['passed_count']}/{metric['gated_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diffusion_text_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Diffusion Text Task Evaluation Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric in (
        "corpus_bleu",
        "rouge1",
        "rouge2",
        "rouge_l",
        "generation_ppl",
        "unigram_entropy",
        "distinct_1",
        "distinct_2",
        "token_repetition_rate",
        "mean_token_count",
        "non_empty_rate",
        "valid_count",
        "skipped_count",
    ):
        if metric not in summary or summary[metric] is None:
            continue
        value = summary[metric]
        rendered = str(value) if isinstance(value, int) else f"{float(value):.4f}"
        lines.append(f"| {metric} | {rendered} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_result_line(model: dict[str, Any], result: dict[str, Any]) -> str:
    common = f"hf_reused={result['hf_reused']} bundle_built={result['bundle_built']}"
    if result.get("mode") == "full_duplex_bench_behavior_parity":
        return (
            f"model={model['name']} metric_gate_pass_rate="
            f"{result['metric_gate_pass_rate']:.4f} "
            f"passed={result['passed_count']}/{result['metric_gate_count']} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "reranking_parity":
        return (
            f"model={model['name']} sample_pass_rate={result['sample_pass_rate']:.4f} "
            f"mean_pairwise={result['mean_pairwise_ordering_agreement']:.4f} "
            f"min_pairwise={result['min_pairwise_ordering_agreement']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "model_plugin_parity":
        return (
            f"model={model['name']} sample_pass_rate={result['sample_pass_rate']:.4f} "
            f"passed={result['passed_count']}/{result['valid_count']} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "time_series_parity":
        return (
            f"model={model['name']} agreement={result['sample_agreement_rate']:.4f} "
            f"max_rel_l2={result['max_relative_l2']:.6e} "
            f"max_abs_error={result['max_absolute_error']:.6e} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "encoder_embedding_parity":
        return (
            f"model={model['name']} vector_pass_rate={result['vector_pass_rate']:.4f} "
            f"min_vector_cosine={result['min_vector_cosine']:.6f} "
            f"max_pair_delta={result['max_pair_cosine_abs_delta']:.6f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "continuation":
        return (
            f"model={model['name']} divergent_count={result['divergent_count']} "
            f"divergence_rate={result['divergence_rate']:.4f} "
            f"mean_divergent_first_divergence="
            f"{_format_optional_float(result['mean_divergent_first_divergence'], precision=2)} "
            f"mean_divergent_prefix_ratio="
            f"{_format_optional_float(result['mean_divergent_prefix_ratio'])} "
            f"min_divergent_prefix_ratio="
            f"{_format_optional_float(result['min_divergent_prefix_ratio'])} "
            f"mean_divergent_severity={result['mean_divergent_severity']:.4f} "
            f"max_divergent_severity={result['max_divergent_severity']:.4f} "
            f"granularity={result.get('comparison_granularity', '')} {common}"
        )
    if result.get("mode") == "diffusion_image_clip_parity":
        return (
            f"model={model['name']} pass_rate={result['overall_pass_rate']:.4f} "
            f"passed={result['passed_count']}/{result['valid_count']} {common}"
        )
    if result.get("mode") == "diffusion_text_parity":
        delta_fields = (
            "corpus_bleu_abs_delta",
            "rouge1_abs_delta",
            "generation_ppl_abs_delta",
        )
        delta = next(
            (f"{name}={float(result[name]):.4f}" for name in delta_fields if name in result),
            "task_metric_delta=unavailable",
        )
        return (
            f"model={model['name']} {delta} "
            f"token_agreement={result['token_agreement_rate']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "sacrebleu":
        return (
            f"model={model['name']} bleu={result['corpus_bleu']:.4f} "
            f"non_empty={result['non_empty_rate']:.4f} status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "rouge":
        return (
            f"model={model['name']} rouge1={result['rouge1']:.4f} "
            f"rouge2={result['rouge2']:.4f} rougeL={result['rouge_l']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "unconditional_text_quality":
        ppl = _format_optional_float(result.get("generation_ppl"), precision=2)
        return (
            f"model={model['name']} gen_ppl={ppl} "
            f"entropy={result['unigram_entropy']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "image_classification_parity":
        return (
            f"model={model['name']} hf_top1={result['hf_top1_accuracy']:.4f} "
            f"trtfb_top1={result['trtfb_top1_accuracy']:.4f} "
            f"top1_agreement={result['top1_agreement']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "semantic_segmentation_parity":
        return (
            f"model={model['name']} hf_miou={result['hf_mean_iou']:.4f} "
            f"trtfb_miou={result['trtfb_mean_iou']:.4f} "
            f"backend_miou={result['backend_mean_iou']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    if result.get("mode") == "prompted_segmentation_parity":
        return (
            f"model={model['name']} backend_mask_iou="
            f"{result['mean_backend_mask_iou']:.4f} "
            f"hf_gt_iou={result['hf_mean_ground_truth_iou']:.4f} "
            f"trtfb_gt_iou={result['trtfb_mean_ground_truth_iou']:.4f} "
            f"status={result.get('status', '')} {common}"
        )
    return (
        f"model={model['name']} hf={result['hf_accuracy']:.4f} "
        f"trtfb={result['trtfb_accuracy']:.4f} "
        f"agreement={result['prediction_agreement_rate']:.4f} {common}"
    )


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--seed", type=int)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local dataset reference-consistency validation for TRTMC bundles.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list-suites")
    p.add_argument("--suites", default=str(DEFAULT_SUITES))
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("plan")
    p.add_argument("--suites", default=str(DEFAULT_SUITES))
    p.add_argument("--suite")
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    p.add_argument(
        "--waives",
        default=str(DEFAULT_WAIVES),
        help="E2E waives file used to skip known-bad models.",
    )
    p.add_argument(
        "--waive-platform", default="", help="Platform prefix to honor in the waives file."
    )
    p.add_argument(
        "--include-waived", action="store_true", help="Include models listed in the waives file."
    )
    p.add_argument(
        "--single-device-only",
        dest="single_device_only",
        action="store_true",
        help="Exclude manifests that require multi-device or distributed runtime.",
    )
    p.add_argument("--include-non-matching", action="store_true")
    p.add_argument("--format", choices=["table", "json"], default="table")
    p.add_argument("--output")

    p = sub.add_parser("prepare")
    p.add_argument("--suites", default=str(DEFAULT_SUITES))
    p.add_argument("--suite", default="mmlu_five_shot_mcq")
    p.add_argument("--dataset")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--subject", default="")
    p.add_argument("--sample-seed", type=int)

    p = sub.add_parser("run-trtfb")
    p.add_argument("--bundle", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--trtmc-binary", default="build/trtmc")
    p.add_argument("--benchmark-binary", default="build/trtmc_dataset_benchmark")
    p.add_argument("--hf-python", default="")
    p.add_argument("--backend-dir", default="")
    p.add_argument("--model-plugin-dir", default="")
    p.add_argument("--kv-cache-size", default="")
    p.add_argument("--config", default="")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--cuda-visible-devices", default="")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--predictions")
    p.add_argument("--raw-output")
    p.add_argument("--log")
    add_generation_args(p)

    p = sub.add_parser("convert-trtfb")
    p.add_argument("--raw", required=True)
    p.add_argument("--predictions", required=True)

    p = sub.add_parser(
        "compare-continuation",
        help="Measure HF/TRTFB continuation divergence frequency and severity.",
    )
    p.add_argument("--work-dir", required=True)
    p.add_argument("--hf-predictions")
    p.add_argument("--trtfb-predictions")
    p.add_argument("--model", default="", help="Optional model id; enables token-level parity.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output")

    p = sub.add_parser("prepare-media")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--vbench-info", type=Path)
    p.add_argument("--gedit-source", default="")
    p.add_argument("--sana-wm-root", type=Path)
    p.add_argument("--limit", type=int, default=10)

    p = sub.add_parser("prepare-ci-dataset")
    p.add_argument("--suites", default=str(DEFAULT_SUITES))
    p.add_argument("--suite", required=True)
    p.add_argument("--ci-lane", required=True)
    p.add_argument("--dataset")
    p.add_argument("--dataset-cache-root", required=True)

    p = sub.add_parser("score")
    p.add_argument("--answers")
    p.add_argument("--predictions")
    p.add_argument("--output")
    p.add_argument("--work-dir")
    p.add_argument("--label", default="")
    p.add_argument("--scorer", default="exact_or_alias")

    p = sub.add_parser("compare")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--answers")
    p.add_argument("--hf-predictions")
    p.add_argument("--trtfb-predictions")
    p.add_argument("--output", default="")
    p.add_argument("--markdown", default="")
    p.add_argument("--scorer", default="exact_or_alias")

    p = sub.add_parser("eval-worker", help=argparse.SUPPRESS)
    p.add_argument("--request", required=True)

    p = sub.add_parser("eval")
    p.add_argument("--suites", default=str(DEFAULT_SUITES))
    p.add_argument("--suite", default="mmlu_five_shot_mcq")
    p.add_argument("--dataset")
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Manifest name, HF id, or bundle filename to evaluate. Repeatable.",
    )
    p.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    p.add_argument(
        "--waives",
        default=str(DEFAULT_WAIVES),
        help="E2E waives file used to skip known-bad models.",
    )
    p.add_argument(
        "--waive-platform", default="", help="Platform prefix to honor in the waives file."
    )
    p.add_argument(
        "--include-waived", action="store_true", help="Include models listed in the waives file."
    )
    p.add_argument("--work-root", default="/tmp/trtmc-validation")
    p.add_argument("--engine-dir", default="")
    p.add_argument(
        "--ci-lane",
        default="",
        help="Run the suite's fail-closed CI profile for this lane.",
    )
    p.add_argument("--dataset-cache-root", default=".ci/validation-data")
    p.add_argument("--artifact-dir", default="")
    p.add_argument(
        "--bundle", default="", help="Prebuilt bundle path; only valid with one --model."
    )
    p.add_argument(
        "--single-device-only",
        dest="single_device_only",
        action="store_true",
        help="Exclude manifests that require multi-device or distributed runtime.",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--subject", default="")
    p.add_argument("--sample-seed", type=int)
    p.add_argument("--force-hf", action="store_true", help="Regenerate HF reference outputs.")
    p.add_argument(
        "--reference-cache-dir",
        default="",
        help="Shared setting-keyed cache managed by tools/trtmc_reference.py.",
    )
    p.add_argument(
        "--reference-cache-identity",
        default="",
        help=(
            "Explicit identity for TRTMC variants that share one reference "
            "contract and may reuse the same cached reference result."
        ),
    )
    p.add_argument("--force-build", action="store_true", help="Rebuild the .trtfb bundle.")
    p.add_argument(
        "--replace-bundle-on-build",
        action="store_true",
        help="Remove the existing bundle before rebuilding it at the same path.",
    )
    p.add_argument(
        "--require-prebuilt-bundles",
        action="store_true",
        help="Fail instead of building when a selected bundle is missing.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first model failure. By default eval records the failure and continues.",
    )
    p.add_argument(
        "--build-max-cache-length",
        type=int,
        help="Override the build max-cache-length. Defaults to max(manifest, suite minimum).",
    )
    p.add_argument(
        "--skip-prompt-length-check",
        action="store_true",
        help="Skip tokenizer-based prompt length validation before TRTFB build/run.",
    )
    p.add_argument("--trtmc-binary", default="build/trtmc")
    p.add_argument("--benchmark-binary", default="build/trtmc_dataset_benchmark")
    p.add_argument("--hf-python", default="")
    p.add_argument("--elf-reference-repo", default="")
    p.add_argument("--backend-dir", default="")
    p.add_argument("--model-plugin-dir", default="")
    p.add_argument("--kv-cache-size", default="")
    p.add_argument("--config", default="")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--extra-build-arg", action="append", default=[])
    p.add_argument("--cuda-visible-devices", default="")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--apply-chat-template", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument(
        "--hf-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    p.add_argument("--hf-device", default="cuda")
    p.add_argument("--hf-device-map", default="")
    p.add_argument("--hf-attn-impl", default="")
    add_generation_args(p)
    return parser


def cmd_list_suites(args: argparse.Namespace) -> int:
    suites = load_suites(Path(args.suites))
    if args.json:
        print(json.dumps({"suites": suites}, indent=2))
    else:
        for suite in suites:
            print(
                f"{suite['id']}\t{suite.get('dataset', {}).get('kind', '')}\t{suite.get('description', '')}"
            )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    suites = load_suites(Path(args.suites))
    models = load_manifest_records(Path(args.models_dir))
    waives = load_waives(Path(args.waives), args.waive_platform) if args.waives else {}
    rows = build_plan(
        suites,
        models,
        suite_id=args.suite,
        single_device_only=args.single_device_only,
        include_non_matching=args.include_non_matching,
        waives=waives,
        include_waived=args.include_waived,
    )
    payload = {"count": len(rows), "rows": rows}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print_plan_table(rows)
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    suites = load_suites(Path(args.suites))
    suite = suite_by_id(suites, args.suite)
    dataset_path = Path(args.dataset or suite.get("dataset", {}).get("default_path", ""))
    if not dataset_path:
        raise ValueError(f"Suite {args.suite} has no dataset path; pass --dataset")
    outputs = prepare_task_dataset(
        dataset_path=dataset_path,
        work_dir=Path(args.work_dir),
        suite=suite,
        limit=args.limit,
        subject=args.subject,
        sample_seed=args.sample_seed,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))
    return 0


def cmd_convert_trtfb(args: argparse.Namespace) -> int:
    convert_trtfb_jsonl_to_predictions(Path(args.raw), Path(args.predictions))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    if args.work_dir:
        work_dir = Path(args.work_dir)
        answers_path = Path(args.answers) if args.answers else work_dir / "answers.json"
        predictions_path = (
            Path(args.predictions)
            if args.predictions
            else work_dir / f"{args.label}_predictions.json"
            if args.label
            else work_dir / "trtfb_predictions.json"
        )
        label = args.label or predictions_path.stem.removesuffix("_predictions")
        output_path = Path(args.output) if args.output else work_dir / f"{label}_score.json"
    else:
        if not args.answers or not args.predictions:
            raise ValueError("score requires --answers and --predictions without --work-dir")
        answers_path = Path(args.answers)
        predictions_path = Path(args.predictions)
        output_path = (
            Path(args.output) if args.output else predictions_path.with_suffix(".score.json")
        )
    score = score_predictions(
        json.loads(predictions_path.read_text(encoding="utf-8")),
        json.loads(answers_path.read_text(encoding="utf-8")),
        scorer=args.scorer,
    )
    output_path.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"accuracy={score['overall_accuracy']:.4f} "
        f"correct={score['correct']}/{score['valid_count']} "
        f"skipped={score['skipped_count']} output={output_path}"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir)
    answers_path = Path(args.answers) if args.answers else work_dir / "answers.json"
    hf_path = Path(args.hf_predictions) if args.hf_predictions else work_dir / "hf_predictions.json"
    trtfb_path = (
        Path(args.trtfb_predictions)
        if args.trtfb_predictions
        else work_dir / "trtfb_predictions.json"
    )
    summary = compare_prediction_sets(
        json.loads(hf_path.read_text(encoding="utf-8")),
        json.loads(trtfb_path.read_text(encoding="utf-8")),
        json.loads(answers_path.read_text(encoding="utf-8")),
        scorer=args.scorer,
    )
    output = Path(args.output) if args.output else work_dir / "summary.json"
    markdown = Path(args.markdown) if args.markdown else work_dir / "summary.md"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_markdown(summary, markdown)
    print(
        f"hf_accuracy={summary['hf']['overall_accuracy']:.4f} "
        f"trtfb_accuracy={summary['trtfb']['overall_accuracy']:.4f} "
        f"agreement={summary['prediction_agreement_rate']:.4f} "
        f"output={output}"
    )
    return 0


def model_work_dir(args: argparse.Namespace, suite: dict[str, Any], model: dict[str, Any]) -> Path:
    return Path(args.work_root) / suite["id"] / str(model["name"])


def model_bundle_path(args: argparse.Namespace, model: dict[str, Any]) -> Path:
    if args.bundle:
        return Path(args.bundle)
    work_root = Path(args.work_root)
    engine_dir = Path(args.engine_dir or (work_root / "_bundles"))
    return engine_dir / str(model["bundle"])


def model_failure_result(
    *,
    suite: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
    exc: BaseException | None = None,
    error_type: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "suite": suite["id"],
        "model": model["name"],
        "hf_id": model["hf_id"],
        "work_dir": str(model_work_dir(args, suite, model)),
        "bundle": str(model_bundle_path(args, model)),
        "status": "failed",
        "error_type": error_type or (type(exc).__name__ if exc else "RuntimeError"),
        "error": error or (str(exc) if exc else ""),
    }


def model_skipped_result(
    *,
    suite: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
    reason: str,
) -> dict[str, Any]:
    return {
        "suite": suite["id"],
        "model": model["name"],
        "hf_id": model["hf_id"],
        "work_dir": str(model_work_dir(args, suite, model)),
        "bundle": str(model_bundle_path(args, model)),
        "status": "skipped",
        "reason": reason,
    }


def gpu_memory_used_mib() -> list[int]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    used: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            used.append(int(line.split()[0]))
        except (ValueError, IndexError):
            return []
    return used


def gpu_memory_back_to_baseline(
    *,
    before_mib: list[int],
    timeout_s: float = 30.0,
    poll_s: float = 2.0,
    margin_mib: int = 512,
) -> tuple[bool | None, list[int]]:
    if not before_mib:
        return None, []
    deadline = time.monotonic() + timeout_s
    current: list[int] = []
    while time.monotonic() <= deadline:
        current = gpu_memory_used_mib()
        if not current or len(current) != len(before_mib):
            return None, current
        if all(
            after <= before + margin_mib for before, after in zip(before_mib, current, strict=True)
        ):
            return True, current
        time.sleep(poll_s)
    return False, current


def is_oom_failure(result: dict[str, Any], returncode: int = 0) -> bool:
    text = " ".join(
        str(result.get(key, "")) for key in ("error", "error_type", "worker_log_tail")
    ).lower()
    return (
        "out of memory" in text
        or "cuda oom" in text
        or "cublas_status_alloc_failed" in text
        or "cudnn_status_alloc_failed" in text
        or "std::bad_alloc" in text
        or returncode == -9
    )


def should_use_model_workers(args: argparse.Namespace, selected: list[dict[str, Any]]) -> bool:
    return len(selected) > 1 and not bool(getattr(args, "disable_model_process_isolation", False))


def _worker_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "cmd" and isinstance(value, (str, int, float, bool, list, type(None)))
    }


def _read_log_tail(path: Path, max_chars: int = 4096) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def run_eval_model_worker(
    *,
    suite: dict[str, Any],
    model: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    work_dir = model_work_dir(args, suite, model)
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "eval_worker_request.json"
    result_path = work_dir / "eval_worker_result.json"
    log_path = work_dir / "eval_worker.log"
    before_mib = gpu_memory_used_mib()
    request = {
        "suite": suite,
        "model": model,
        "args": _worker_args_payload(args),
        "result_path": str(result_path),
    }
    request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "eval-worker",
        "--request",
        str(request_path),
    ]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )

    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = model_failure_result(
            suite=suite,
            model=model,
            args=args,
            error_type="WorkerProcessError",
            error=f"worker exited with rc={proc.returncode} before writing {result_path}",
        )
    result["worker_returncode"] = proc.returncode
    result["worker_log"] = str(log_path)
    if proc.returncode != 0 and result.get("status") != "failed":
        result["status"] = "failed"
        result["error_type"] = "WorkerProcessError"
        result["error"] = f"worker exited with rc={proc.returncode}; see {log_path}"
    if result.get("status") == "failed":
        result["worker_log_tail"] = _read_log_tail(log_path)

    # Verify GPU memory returned near this model's pre-run baseline for EVERY
    # model (pass or fail), not only on OOM, so a leak or a still-running child
    # from one model cannot silently corrupt the next model's run.
    cleanup_confirmed, after_mib = gpu_memory_back_to_baseline(before_mib=before_mib)
    result["gpu_memory_before_mib"] = before_mib
    result["gpu_memory_after_cleanup_mib"] = after_mib
    result["gpu_cleanup_confirmed"] = cleanup_confirmed
    if cleanup_confirmed is False:
        result["gpu_cleanup_error"] = (
            "GPU memory did not return near the pre-model baseline after the worker exited"
        )
    return result


def cmd_eval_worker(args: argparse.Namespace) -> int:
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    suite = request["suite"]
    model = request["model"]
    worker_args = argparse.Namespace(**request["args"])
    result_path = Path(request["result_path"])
    try:
        result = eval_one_model(suite=suite, model=model, args=worker_args)
        result.setdefault("status", "passed")
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as exc:
        traceback.print_exc()
        result = model_failure_result(suite=suite, model=model, args=worker_args, exc=exc)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1


def cmd_prepare_media(args: argparse.Namespace) -> int:
    from tools.prepare_media_validation_datasets import prepare_media_datasets

    outputs = prepare_media_datasets(
        output_root=args.output_root,
        vbench_info=args.vbench_info,
        gedit_source=args.gedit_source,
        sana_wm_root=args.sana_wm_root,
        limit=args.limit,
    )
    for output in outputs:
        payload = json.loads(output.read_text(encoding="utf-8"))
        print(f"{output}: {payload['request_count']} requests")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    suites = load_suites(Path(args.suites))
    suite = suite_by_id(suites, args.suite)
    ci_lane = str(getattr(args, "ci_lane", ""))
    expected_models = (
        configure_ci_eval(args, suite)
        if ci_lane
        else list(suite.get("default_model_names", []))
    )
    dataset_kind = suite.get("dataset", {}).get("kind", "")
    if dataset_kind not in {
        "mmlu_five_shot_json",
        "text_generation_json",
        "vlm_chat_json",
        "vlm_grounding_json",
        "vlm_unified_json",
        "asr_chat_json",
        "diffusion_prompt_tsv",
        "diffusion_prompt_json",
        "model_plugin_json",
        "conditional_text_jsonl",
        "unconditional_text_json",
        "seedtts_json",
        "sts_pair_jsonl",
        "image_classification_json",
        "semantic_segmentation_json",
        "prompted_segmentation_json",
        "time_series_csv",
        "reranking_json",
    }:
        raise ValueError(f"eval does not support dataset kind {dataset_kind!r}")
    models = load_manifest_records(Path(args.models_dir))
    waives = load_waives(Path(args.waives), args.waive_platform) if args.waives else {}
    selected = selected_models_for_suite(
        suite,
        models,
        selectors=args.model,
        single_device_only=args.single_device_only,
        waives=waives,
        include_waived=args.include_waived,
    )
    if not selected:
        raise ValueError(f"No models selected for suite {suite['id']}")
    if args.bundle and len(selected) != 1:
        raise ValueError("--bundle can only be used when exactly one model is selected")

    results = []
    use_workers = should_use_model_workers(args, selected)
    for idx, model in enumerate(selected, start=1):
        print(f"[validation] ({idx}/{len(selected)}) suite={suite['id']} model={model['name']}")
        if use_workers:
            result = run_eval_model_worker(suite=suite, model=model, args=args)
            if result.get("status") == "failed":
                results.append(result)
                if result.get("mode"):
                    print(f"[validation] {_format_result_line(model, result)}")
                else:
                    print(
                        f"[validation] model={model['name']} status=failed "
                        f"error_type={result.get('error_type', '')} "
                        f"error={result.get('error', '')} log={result.get('worker_log', '')}"
                    )
                if args.fail_fast:
                    raise RuntimeError(
                        f"Model {model['name']} failed in worker; see {result.get('worker_log', '')}"
                    )
                if result.get("gpu_cleanup_confirmed") is False:
                    reason = (
                        f"Skipped because GPU cleanup after {model['name']} OOM was not confirmed"
                    )
                    print(f"[validation] {reason}")
                    for skipped_model in selected[idx:]:
                        results.append(
                            model_skipped_result(
                                suite=suite,
                                model=skipped_model,
                                args=args,
                                reason=reason,
                            )
                        )
                    break
                continue
        else:
            try:
                result = eval_one_model(suite=suite, model=model, args=args)
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = model_failure_result(suite=suite, model=model, args=args, exc=exc)
                results.append(result)
                print(
                    f"[validation] model={model['name']} status=failed "
                    f"error_type={type(exc).__name__} error={exc}"
                )
                continue
        result.setdefault("status", "passed")
        results.append(result)
        print(f"[validation] {_format_result_line(model, result)}")
        if result.get("gpu_cleanup_confirmed") is False:
            reason = f"Skipped because GPU cleanup after {model['name']} was not confirmed"
            print(f"[validation] {reason}")
            for skipped_model in selected[idx:]:
                results.append(
                    model_skipped_result(
                        suite=suite,
                        model=skipped_model,
                        args=args,
                        reason=reason,
                    )
                )
            break

    failed_count = sum(1 for result in results if result.get("status") == "failed")
    skipped_count = sum(1 for result in results if result.get("status") == "skipped")
    out = {
        "suite": suite["id"],
        "count": len(results),
        "passed_count": len(results) - failed_count - skipped_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "model_process_isolation": use_workers,
        "results": results,
    }
    summary_path = Path(args.work_root) / suite["id"] / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[validation] summary={summary_path}")
    artifact_dir = str(getattr(args, "artifact_dir", ""))
    if artifact_dir:
        write_public_ci_artifacts(
            suite=suite,
            expected_models=expected_models,
            results=results,
            work_root=Path(args.work_root),
            artifact_dir=Path(artifact_dir),
        )
    if ci_lane:
        passed, _ = validate_eval_summary(out, expected_models)
        return 0 if passed else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "list-suites":
        return cmd_list_suites(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "prepare-media":
        return cmd_prepare_media(args)
    if args.cmd == "prepare-ci-dataset":
        return cmd_prepare_ci_dataset(args)
    if args.cmd == "run-trtfb":
        run_trtfb(args)
        return 0
    if args.cmd == "convert-trtfb":
        return cmd_convert_trtfb(args)
    if args.cmd == "compare-continuation":
        return cmd_compare_continuation(args)
    if args.cmd == "score":
        return cmd_score(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "eval-worker":
        return cmd_eval_worker(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    raise AssertionError(f"Unhandled command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

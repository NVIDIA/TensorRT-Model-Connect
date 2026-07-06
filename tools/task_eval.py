#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import html
import json
import math
import os
import random
import re
import shutil
import struct
import traceback
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e_harness.manifest_loader import iter_manifest_paths, load_manifest  # noqa: E402
from tests.e2e_harness.registry import (  # noqa: E402
    activate_model_plugins,
    get_comparator,
    get_reference,
    get_runner,
)


DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_WAIVES = REPO_ROOT / "tests" / "e2e" / "waives.txt"
ERROR_OUTPUT_TEXT = "TensorRT Edge LLM cannot handle this request. Fails."
CHOICE_LETTERS = set("ABCDEFGHIJ")
GPT_OSS_MMLU_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer with only the option letter."
)


def load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        raise RuntimeError("PyYAML is required for task-eval YAML files") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def load_suites(path: Path = DEFAULT_SUITES) -> list[dict[str, Any]]:
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
    for suite in suites:
        if suite["id"] == suite_id:
            return suite
    known = ", ".join(sorted(s["id"] for s in suites))
    raise ValueError(f"Unknown suite {suite_id!r}. Known suites: {known}")


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


def effective_task_eval_config(
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


def infer_reference_family(raw: dict[str, Any]) -> str:
    return str(raw.get("reference_family", "") or "")


def infer_user_contract(raw: dict[str, Any], reference_family: str) -> str:
    return str(raw.get("user_contract", "") or "")


def manifest_record(path: Path) -> dict[str, Any]:
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
    runtime_strategy = str(raw.get("runtime_strategy") or "")
    task_strategy = str(raw.get("task_strategy") or runtime_strategy)
    reference_family = infer_reference_family(raw)
    reference_backend = str(raw.get("reference_backend", "") or "")
    if not reference_backend:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                reference_backend = load_manifest(path).reference_backend
        except Exception:
            reference_backend = "hf_transformers"
    user_contract = infer_user_contract(raw, reference_family)
    distributed = raw.get("distributed_runtime", {})
    requires_multi_device = bool(distributed.get("enabled")) or (
        str(raw.get("ci_tier", "")) == "multi_device"
    )
    return {
        "name": model_name,
        "manifest": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "hf_id": raw.get("hf_id") or raw.get("model_id", ""),
        "bundle": raw.get("bundle", f"{path.stem}.trtfb"),
        "family": raw.get("family", ""),
        "runtime_strategy": runtime_strategy,
        "task_strategy": task_strategy,
        "reference_family": reference_family,
        "reference_backend": reference_backend,
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
            build_args.get("max_cache_length", 256) if isinstance(build_args, dict) else 256,
        ),
        "precision": raw.get("precision", "fp32"),
        "quantization": raw.get("quantization", {}),
        "build_args": build_args if isinstance(build_args, dict) else {},
        "fp8_scales": raw.get("fp8_scales", ""),
        "image_height": raw.get("image_height"),
        "image_width": raw.get("image_width"),
        "video_num_frames": raw.get("video_num_frames"),
        "video_height": raw.get("video_height"),
        "video_width": raw.get("video_width"),
        "num_inference_steps": raw.get("num_inference_steps"),
        "task_eval": task_eval_config if isinstance(task_eval_config, dict) else {},
        "runtime_config": raw.get("runtime_config", {})
        if isinstance(raw.get("runtime_config", {}), dict)
        else {},
        "max_new_tokens": raw.get("max_new_tokens"),
    }


def load_manifest_records(models_dir: Path = DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    return [manifest_record(path) for path in iter_manifest_paths(models_dir)]


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


def _selector_values(selectors: dict[str, Any], key: str) -> set[str]:
    values = selectors.get(key, [])
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    return {str(v) for v in values}


def suite_match_reason(suite: dict[str, Any], model: dict[str, Any]) -> tuple[bool, str]:
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
        return False, f"runtime_strategy={model['runtime_strategy']} not selected"
    if user_contracts and model["user_contract"] not in user_contracts:
        return False, f"user_contract={model['user_contract'] or '<empty>'} not selected"
    if exclude_user_contracts and model["user_contract"] in exclude_user_contracts:
        return False, f"user_contract={model['user_contract']} excluded"
    if families and model["family"] not in families:
        return False, f"family={model['family']} not selected"
    if exclude_families and model["family"] in exclude_families:
        return False, f"family={model['family']} excluded"
    if model.get("skip"):
        return False, f"manifest skip: {model['skip']}"
    return True, "selected"


def resolve_suite_for_model(
    suite: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Resolve manifest generation settings and per-model gates for one suite run."""
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
            f"Suite {suite['id']} family profile for {model.get('family')} "
            "must be a mapping"
        )

    profiles = resolved.get("model_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Suite {suite['id']} model_profiles must be a mapping")
    profile = profiles.get(str(model.get("name", "")), {})
    if not isinstance(profile, dict):
        raise ValueError(
            f"Suite {suite['id']} profile for {model.get('name')} must be a mapping"
        )
    for owner, source in ((model.get("family"), family_profile), (model.get("name"), profile)):
        profile_generation = source.get("generation", {})
        if not isinstance(profile_generation, dict):
            raise ValueError(
                f"Suite {suite['id']} generation profile for {owner} must be a mapping"
            )
        generation.update(profile_generation)
    resolved["generation"] = generation

    gates = dict(resolved.get("gates", {}))
    for owner, source in ((model.get("family"), family_profile), (model.get("name"), profile)):
        profile_gates = source.get("gates", {})
        if not isinstance(profile_gates, dict):
            raise ValueError(
                f"Suite {suite['id']} gate profile for {owner} must be a mapping"
            )
        gates.update(profile_gates)
    resolved["gates"] = gates
    return resolved


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


def render_mmlu_prompt(prompt: str, task_eval_config: dict[str, Any] | None) -> str:
    config = task_eval_config if isinstance(task_eval_config, dict) else {}
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
    task_eval_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data, indexed = load_mmlu_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    requests = [req for _idx, req in indexed]
    answers = _copy_dataset_header(data, requests)
    answers_path = work_dir / "answers.json"
    prompts_path = work_dir / "prompts.jsonl"
    manifest_path = work_dir / "manifest.json"

    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    with prompts_path.open("w", encoding="utf-8") as f:
        for out_idx, (dataset_index, request) in enumerate(indexed):
            prompt = render_mmlu_prompt(_request_prompt(request), task_eval_config)
            sample = {
                "sample_id": f"mmlu_{dataset_index:06d}",
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": request.get("subject", ""),
                "answer": request["answer"],
                "prompt": prompt,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    generation = _deep_merge_mappings(
        suite.get("generation", {}),
        (task_eval_config or {}).get("generation", {}),
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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
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
    task_eval_config: dict[str, Any] | None = None,
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
            indexed.append((dataset_index, {
                "sample_id": f"partiprompts_{dataset_index:06d}",
                "dataset_index": dataset_index,
                "prompt": prompt,
                "category": category,
                "challenge": str(row.get("Challenge", "")).strip(),
                "note": str(row.get("Note", "")).strip(),
            }))

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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
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
    task_eval_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
    task_eval_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if subject:
        raise ValueError(
            "SeedTTS task eval does not support --subject; select a language suite instead"
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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
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
            f"ASR task eval expects exactly one audio asset per sample; found {len(refs)}"
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
        raise RuntimeError("Fixed VLM task eval normalization requires Pillow") from exc
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
        "ffmpeg, sox, or flac in the environment running task eval prepare."
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
    task_eval_config: dict[str, Any] | None = None,
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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
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
    task_eval_config: dict[str, Any] | None = None,
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
                    f"VLM task eval currently supports exactly one image per sample; "
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
    if task_eval_config:
        manifest["task_eval"] = task_eval_config
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
    task_eval_config: dict[str, Any] | None = None,
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
        task_eval_config=task_eval_config,
    )


def prepare_vlm_unified_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    task_eval_config: dict[str, Any] | None = None,
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
        task_eval_config=task_eval_config,
    )


def prepare_task_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
    task_eval_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    dataset_kind = suite.get("dataset", {}).get("kind", "")
    if dataset_kind == "mmlu_five_shot_json":
        return prepare_mmlu_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "seedtts_json":
        return prepare_seedtts_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "vlm_chat_json":
        return prepare_vlm_chat_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "vlm_unified_json":
        return prepare_vlm_unified_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "asr_chat_json":
        return prepare_asr_chat_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "diffusion_prompt_tsv":
        return prepare_diffusion_prompt_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    if dataset_kind == "diffusion_prompt_json":
        return prepare_diffusion_prompt_json_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
            task_eval_config=task_eval_config,
        )
    raise ValueError(f"Unsupported task-eval dataset kind {dataset_kind!r}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def max_prompt_token_length(
    *,
    model_id: str,
    prompts_path: Path,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> int:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Prompt length check requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    max_len = 0
    for row in load_jsonl(prompts_path):
        prompt = str(row.get("prompt", ""))
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


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        return None


def _generated_token_ids(row: dict[str, Any]) -> list[int] | None:
    generated = _int_list(row.get("generated_token_ids"))
    if generated is not None:
        return generated
    return _int_list(row.get("token_ids"))


def predictions_file_valid(
    predictions_path: Path,
    answers_path: Path,
    *,
    require_token_ids: bool = False,
) -> bool:
    if not predictions_path.is_file() or not answers_path.is_file():
        return False
    try:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    responses = predictions.get("responses")
    requests = answers.get("requests")
    if (
        not isinstance(responses, list)
        or not isinstance(requests, list)
        or len(responses) != len(requests)
    ):
        return False
    if require_token_ids:
        return all(
            isinstance(row, dict) and _generated_token_ids(row) is not None for row in responses
        )
    return True


def _parse_generated_token_ids(text: str) -> list[int] | None:
    for line in str(text or "").splitlines():
        if not line.strip().startswith("tokens:"):
            continue
        try:
            return [int(token) for token in line.split(":", 1)[1].strip().split()]
        except (ValueError, IndexError):
            return None
    return None


def _parse_transcribe_stdout(text: str) -> str:
    for line in str(text or "").splitlines():
        cleaned = re.sub(r"<\|[^|]+\|>", "", line).strip()
        if cleaned:
            return cleaned
    return ""


def convert_trtfb_jsonl_to_predictions(raw_path: Path, predictions_path: Path) -> None:
    rows = []
    for row in load_jsonl(raw_path):
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


def run_vlm_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
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

            log_f.write(f"$ {' '.join(cmd)}\n")
            start = time.perf_counter()
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
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
                    f"VLM TRTFB task eval failed for sample {idx} rc={proc.returncode}; see {log_path}"
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
            print(f"[task_eval.vlm_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
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

            log_f.write(f"$ {' '.join(cmd)}\n")
            start = time.perf_counter()
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
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
                    f"ASR TRTFB task eval failed for sample {idx} rc={proc.returncode}; see {log_path}"
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
            print(f"[task_eval.asr_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    write_predictions(predictions, rows)


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
        raise ValueError(f"Unsupported task-eval answer parser {answer_parser!r}")

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


def score_predictions(
    predictions_data: dict[str, Any],
    answers_data: dict[str, Any],
    *,
    scorer: str = "exact_or_alias",
    answer_parser: str = "",
    require_valid_prediction: bool = False,
) -> dict[str, Any]:
    if scorer == "ocrbench_v2":
        return score_ocrbench_v2_predictions(predictions_data, answers_data)
    if scorer == "asr_transcript":
        return score_asr_transcript_predictions(predictions_data, answers_data)
    if scorer == "tts_intelligibility":
        return score_tts_intelligibility_predictions(predictions_data, answers_data)

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
) -> dict[str, Any]:
    hf_score = score_predictions(
        hf_predictions,
        answers,
        scorer=scorer,
        answer_parser=answer_parser,
        require_valid_prediction=require_valid_prediction,
    )
    trtfb_score = score_predictions(
        trtfb_predictions,
        answers,
        scorer=scorer,
        answer_parser=answer_parser,
        require_valid_prediction=require_valid_prediction,
    )
    hf_responses = hf_predictions["responses"]
    trt_responses = trtfb_predictions["responses"]
    requests = answers["requests"]
    if len(hf_responses) != len(trt_responses):
        raise ValueError("HF and TRTFB predictions must have the same length")

    agreement = 0
    buckets = Counter()
    disagreements: list[dict[str, Any]] = []
    for idx, (hf_row, trt_row, req) in enumerate(
        zip(hf_responses, trt_responses, requests, strict=True)
    ):
        answer = str(req["answer"])
        if scorer == "asr_transcript":
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
        agreement_match = (
            hf_ok == trt_ok
            if scorer in {"ocrbench_v2", "asr_transcript", "tts_intelligibility"}
            else hf_pred == trt_pred
        )
        if require_valid_prediction and (
            not bool(hf_sample.get("valid_prediction", False))
            or not bool(trtfb_sample.get("valid_prediction", False))
        ):
            agreement_match = False
        agreement += int(agreement_match)
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
    return {
        "hf": hf_score,
        "trtfb": trtfb_score,
        "accuracy_delta_trtfb_minus_hf": (
            trtfb_score["overall_accuracy"] - hf_score["overall_accuracy"]
        ),
        "prediction_agreement_rate": (agreement / total) if total else 0.0,
        "agreement_count": agreement,
        "total_count": total,
        "buckets": dict(buckets),
        "disagreements": disagreements,
    }


def _load_diffusion_task_eval_comparator(work_dir: Path) -> Any:
    case, _reference, _runner = _load_diffusion_task_eval_plugins(work_dir)
    comparator = get_comparator(case.task_strategy)
    if comparator is None:
        raise RuntimeError(
            f"No comparator plugin {case.task_strategy!r} for {case.family}"
        )
    return comparator


def _compute_task_eval_clip_metrics(
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
        question_items = "".join(
            f"<li>{html.escape(str(question.get('question', question)))}</li>"
            for question in questions
        ) or "<li>No proposition questions provided.</li>"
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
    comparator = _load_diffusion_task_eval_comparator(work_dir)
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
            })
            continue

        trt_output = StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": trt_row.get("returncode", 0),
                "num_frames": trt_row.get("num_frames", 0),
                "frames_dir": trt_row.get("frames_dir", ""),
                "frame_stats": trt_row.get("frame_stats", {}),
                "prompt": trt_row.get("prompt", request.get("prompt", "")),
            },
        )
        hf_output = StageOutput(
            stage_name="end_to_end",
            data={
                "returncode": hf_row.get("returncode", 0),
                "num_frames": hf_row.get("num_frames", 0),
                "frames_dir": hf_row.get("frames_dir", ""),
                "frame_stats": hf_row.get("frame_stats", {}),
                "prompt": hf_row.get("prompt", request.get("prompt", "")),
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

            clip = _compute_task_eval_clip_metrics(
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
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
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


def _work_dataset_kind(work_dir: Path) -> str:
    return str(work_manifest(work_dir).get("dataset_kind", ""))


def _is_vlm_dataset_kind(kind: str) -> bool:
    return kind in {"vlm_chat_json", "vlm_unified_json"}


def _is_asr_dataset_kind(kind: str) -> bool:
    return kind in {"asr_chat_json"}


def _is_diffusion_media_dataset_kind(kind: str) -> bool:
    return kind in {"diffusion_prompt_tsv", "diffusion_prompt_json"}


def _is_tts_dataset_kind(kind: str) -> bool:
    return kind == "seedtts_json"


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
        raise RuntimeError("VLM task eval requires Pillow") from exc
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
                f"[task_eval.vlm_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
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
    task_eval_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(task_eval_config, dict):
        task_eval_config = {}
    vlm_fallback_prompt_template = str(
        task_eval_config.get("vlm_fallback_prompt_template", "") or ""
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
                f"[task_eval.vlm_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
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
                f"[task_eval.asr_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr
            )
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    model = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location=map_location)
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
            print(f"[task_eval.nemo_asr_hf] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
    write_predictions(pred_path, responses)
    del model
    gc.collect()


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
        model_kwargs={
            "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
            "local_files_only": bool(getattr(args, "local_files_only", False)),
        },
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
            result = pipe(str(mono_path))
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
                f"[task_eval.nemo_asr_hf_pipeline] sample={idx + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


def _load_diffusion_task_eval_plugins(work_dir: Path) -> tuple[Any, Any, Any]:
    task_eval_config = work_manifest(work_dir).get("task_eval", {})
    if not isinstance(task_eval_config, dict):
        task_eval_config = {}
    manifest_ref = str(task_eval_config.get("model_manifest", "") or "")
    if not manifest_ref:
        raise ValueError(
            "diffusion task eval requires task_eval.model_manifest in the work manifest"
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
    seed = int(generation.get("seed", case.determinism.get("seed", 42)))
    case.inputs["seed"] = seed + index
    return case


def _diffusion_end_to_end_stage(case: Any) -> Any:
    from tests.e2e_harness.contracts import StageSpec

    for stage in case.stages:
        if stage.name == "end_to_end":
            return stage
    return StageSpec(name="end_to_end", required=True)


def _diffusion_response(sample_id: str, source: str, output: Any) -> dict[str, Any]:
    data = output.data if isinstance(output.data, dict) else {}
    return {
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


def run_diffusion_hf_reference(args: argparse.Namespace) -> None:
    from tests.e2e_harness.contracts import RunContext

    work_dir = Path(args.work_dir)
    template, reference, _runner = _load_diffusion_task_eval_plugins(work_dir)
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
            response = _diffusion_response(case.name, "hf", output)
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
                f"[task_eval.diffusion_hf] sample={index + 1}/{len(prompt_rows)}",
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
            print(f"[task_eval.tts_hf] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)

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


def run_hf_reference(args: argparse.Namespace) -> None:
    dataset_kind = _work_dataset_kind(Path(args.work_dir))
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, logging
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

    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(
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
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
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
                )
            wall_ms = (time.perf_counter() - start) * 1000.0
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
            print(f"[task_eval.hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr)
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
    template, _reference, runner = _load_diffusion_task_eval_plugins(work_dir)
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    generation = generation_defaults(work_dir)
    artifacts_dir = work_dir / "trtfb_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    pred_path = work_dir / (args.predictions or "trtfb_predictions.json")
    bundle_path = Path(args.bundle).resolve()
    responses: list[dict[str, Any]] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
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
            response = _diffusion_response(case.name, "trtfb", output)
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
                f"[task_eval.diffusion_trtfb] sample={index + 1}/{len(prompt_rows)}",
                file=sys.stderr,
            )
    write_predictions(pred_path, responses)


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
                    f"TRTFB TTS task eval failed for sample {idx} rc={proc.returncode}; see {log_path}"
                )
            if not wav_path.is_file():
                raise RuntimeError(
                    f"TRTFB TTS task eval produced no WAV for sample {idx}: {wav_path}"
                )
            row = _tts_response_row(sample_id, wav_path, wall_ms, "trtfb")
            responses.append(row)
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            print(f"[task_eval.tts_trtfb] sample={idx + 1}/{len(prompt_rows)}", file=sys.stderr)
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


def run_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    dataset_kind = _work_dataset_kind(work_dir)
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

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    log_path = work_dir / (args.log or "trtfb_run.log")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(f"TRTFB task eval failed with rc={proc.returncode}; see {log_path}")
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
        "-o",
        str(bundle_path),
        "--max-cache-length",
        str(cache_length),
    ]
    build_args = model.get("build_args", {})
    method = _manifest_build_method(build_args)
    if method:
        cmd.extend(["--method", method])
    tp_size = _manifest_tensor_parallel_size(build_args)
    if tp_size is not None and tp_size > 1:
        cmd.extend(["--tp-size", str(tp_size)])
    precision = str(model.get("precision", "fp32") or "fp32")
    if precision != "fp32":
        cmd.extend(["--precision", precision])
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
        scales_path = REPO_ROOT / "tests" / "e2e" / "data" / str(fp8_scales)
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


def bundle_max_cache_length(bundle_path: Path, trtmc_binary: str) -> int | None:
    try:
        proc = subprocess.run(
            [trtmc_binary, "inspect", str(bundle_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"^Max cache length:\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def ensure_bundle(
    model: dict[str, Any],
    *,
    bundle_path: Path,
    trtmc_binary: str,
    max_cache_length: int | None = None,
    force_build: bool = False,
    extra_build_args: list[str] | None = None,
    log_path: Path | None = None,
) -> tuple[Path, bool]:
    if bundle_path.is_file() and not force_build:
        if max_cache_length is None:
            return bundle_path, False
        existing_cache = bundle_max_cache_length(bundle_path, trtmc_binary)
        if existing_cache is None or existing_cache >= max_cache_length:
            return bundle_path, False
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_bundle_command(
        model,
        trtmc_binary=trtmc_binary,
        bundle_path=bundle_path,
        max_cache_length=max_cache_length,
        extra_build_args=extra_build_args,
    )
    log_path = log_path or bundle_path.with_suffix(".build.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
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
    return argparse.Namespace(
        model=model["hf_id"],
        family=model.get("family", ""),
        reference_family=model.get("reference_family", ""),
        work_dir=str(work_dir),
        predictions="hf_predictions.json",
        raw_output="hf_raw.jsonl",
        dtype=args.hf_dtype,
        device=args.hf_device,
        device_map=args.hf_device_map,
        attn_impl=args.hf_attn_impl,
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
    )


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
    hf_python = str(getattr(args, "hf_python", "") or sys.executable)
    cmd = [
        hf_python,
        str(Path(__file__).resolve()),
        "run-hf",
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
    if hf_args.device_map:
        cmd.extend(["--device-map", str(hf_args.device_map)])
    if hf_args.attn_impl:
        cmd.extend(["--attn-impl", str(hf_args.attn_impl)])
    if hf_args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if hf_args.local_files_only:
        cmd.append("--local-files-only")
    if hf_args.do_sample:
        cmd.append("--do-sample")
    if hf_args.apply_chat_template:
        cmd.append("--apply-chat-template")
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
        proc = subprocess.run(
            cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"HF reference subprocess failed for {model['name']} rc={proc.returncode}; see {log_path}"
        )


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
    reference_backend = str(model.get("reference_backend", "hf_transformers") or "hf_transformers")
    task_eval_config = effective_task_eval_config(suite, model)
    if model.get("manifest"):
        task_eval_config["model_manifest"] = str(model["manifest"])
    if model.get("family"):
        task_eval_config["family"] = str(model["family"])
    runtime_config = model.get("runtime_config", {})
    if isinstance(runtime_config, dict) and runtime_config:
        task_eval_config["runtime_config"] = runtime_config
    if model.get("max_new_tokens") is not None:
        task_eval_config["model_max_new_tokens"] = model["max_new_tokens"]
    prepare_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        limit=args.limit,
        subject=args.subject,
        sample_seed=args.sample_seed,
        task_eval_config=task_eval_config,
    )

    answers_path = work_dir / "answers.json"
    hf_predictions = work_dir / "hf_predictions.json"
    hf_reused = predictions_file_valid(hf_predictions, answers_path) and not args.force_hf
    if not hf_reused:
        # Run HF in its own process so its GPU memory is fully reclaimed before
        # the TRT bundle build and TRTFB inference for this model.
        run_hf_reference_subprocess(args, model, work_dir)

    if args.bundle:
        if len(args.model or []) != 1:
            raise ValueError("--bundle can only be used when exactly one --model is selected")
        bundle_path = Path(args.bundle)
    else:
        engine_dir = Path(args.engine_dir or (work_root / "_bundles"))
        bundle_path = engine_dir / str(model["bundle"])

    max_prompt_len = None
    if (
        not args.skip_prompt_length_check
        and not _is_asr_dataset_kind(dataset_kind)
        and not _is_diffusion_media_dataset_kind(dataset_kind)
        and not _is_tts_dataset_kind(dataset_kind)
    ):
        prompt_rows_path = work_dir / "prompts.jsonl"
        max_prompt_len = max_prompt_token_length(
            model_id=str(model["hf_id"]),
            prompts_path=prompt_rows_path,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code or bool(model.get("trust_remote_code", False)),
        )
    generation = generation_defaults(work_dir)
    generation_headroom = 0
    if bool(task_eval_config.get("build_generation_headroom", False)):
        generation_headroom = int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else generation.get("max_new_tokens", 0)
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
    if max_prompt_len is not None and max_prompt_len > max_cache_length:
        raise RuntimeError(
            f"Dataset prompt length exceeds bundle cache for {model['name']}: "
            f"max_prompt_tokens={max_prompt_len}, build_max_cache_length={max_cache_length}. "
            "Use a smaller dataset slice/subject or set --build-max-cache-length high enough "
            "for this model and TensorRT target."
        )
    bundle_path, built = ensure_bundle(
        model,
        bundle_path=bundle_path,
        trtmc_binary=args.trtmc_binary,
        max_cache_length=max_cache_length,
        force_build=args.force_build,
        extra_build_args=args.extra_build_arg,
        log_path=work_dir / "build.log",
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
        "reference_backend": reference_backend,
        "hf_reference_status": "reused" if hf_reused else "ran",
        "hf_reused": hf_reused,
        "bundle_built": built,
        "model_plugin_dir": str(getattr(args, "model_plugin_dir", "") or ""),
    }

    if scorer == "continuation":
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_continuation_sets(
            hf_data,
            trtfb_data,
            tokenize=_model_tokenizer(model, args),
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_continuation_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": "continuation",
            "comparison_granularity": summary.get("comparison_granularity", ""),
            "exact_match_rate": summary["exact_match_rate"],
            "token_prefix_agreement": summary["token_prefix_agreement"],
            "mean_first_divergence": summary["mean_first_divergence"],
            "prediction_agreement_rate": summary["token_prefix_agreement"],
        }
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
            answer_parser=str(task_eval_config.get("answer_parser", "") or ""),
            require_valid_prediction=bool(
                task_eval_config.get("require_valid_prediction", False)
            ),
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
            "prediction_agreement_rate": summary["prediction_agreement_rate"],
            "hf_valid_prediction_rate": summary["hf"].get("valid_prediction_rate"),
            "trtfb_valid_prediction_rate": summary["trtfb"].get("valid_prediction_rate"),
        }
    (work_dir / "eval_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


# ---------------------------------------------------------------------------
# Continuation parity for base / completion models (generation-only, no logits).
# HF reference and TRTFB both greedily generate from the same plain-text prompt;
# we compare the two continuations. No gold answer or logprobs are needed — the
# metric is HF<->TRTFB output agreement (conversion fidelity).
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
    text_exact = 0
    total_matched = 0
    total_reference = 0
    div_positions: list[int] = []
    samples: list[dict[str, Any]] = []
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
        reference_len = max(1, len(hf_tokens), len(trt_tokens))
        total_matched += min(divergence, reference_len)
        total_reference += reference_len
        div_positions.append(divergence)
        hf_token_at_divergence = hf_tokens[divergence] if divergence < len(hf_tokens) else None
        trt_token_at_divergence = trt_tokens[divergence] if divergence < len(trt_tokens) else None
        samples.append(
            {
                "index": idx,
                "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                "exact": is_exact,
                "text_exact": text_is_exact,
                "first_divergence": divergence,
                "hf_len": len(hf_tokens),
                "trtfb_len": len(trt_tokens),
                "hf_token_at_divergence": hf_token_at_divergence,
                "trtfb_token_at_divergence": trt_token_at_divergence,
            }
        )

    count = len(hf_rows)
    exact_rate = (exact / count) if count else 0.0
    prefix_agreement = (total_matched / total_reference) if total_reference else 0.0
    mean_divergence = (sum(div_positions) / count) if count else 0.0
    return {
        "comparison_granularity": comparison_granularity,
        "exact_match_rate": exact_rate,
        "token_id_exact_match_rate": exact_rate if has_all_token_ids else None,
        "text_exact_match_rate": (text_exact / count) if count else 0.0,
        "token_prefix_agreement": prefix_agreement,
        "token_id_prefix_agreement": prefix_agreement if has_all_token_ids else None,
        "mean_first_divergence": mean_divergence,
        "mean_first_token_id_divergence": mean_divergence if has_all_token_ids else None,
        "count": count,
        "exact_count": exact,
        "text_exact_count": text_exact,
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
        f"exact_match={summary['exact_match_rate']:.4f} "
        f"token_agreement={summary['token_prefix_agreement']:.4f} "
        f"mean_first_divergence={summary['mean_first_divergence']:.2f} "
        f"output={output_path}"
    )
    return 0


def _model_tokenizer(model: dict[str, Any], args: argparse.Namespace) -> Any:
    """Return a tokenize(str)->list[int] using the model tokenizer, or None."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            str(model["hf_id"]),
            trust_remote_code=getattr(args, "trust_remote_code", False)
            or bool(model.get("trust_remote_code", False)),
            local_files_only=getattr(args, "local_files_only", False),
        )
        return lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    except Exception:
        return None


def _format_optional_float(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def write_continuation_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Continuation Parity Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| comparison_granularity | {summary.get('comparison_granularity', '')} |",
        f"| exact_match_rate | {summary['exact_match_rate']:.4f} |",
        f"| token_id_exact_match_rate | {_format_optional_float(summary.get('token_id_exact_match_rate'))} |",
        f"| text_exact_match_rate | {summary['text_exact_match_rate']:.4f} |",
        f"| token_prefix_agreement | {summary['token_prefix_agreement']:.4f} |",
        f"| token_id_prefix_agreement | {_format_optional_float(summary.get('token_id_prefix_agreement'))} |",
        f"| mean_first_divergence | {summary['mean_first_divergence']:.2f} |",
        f"| mean_first_token_id_divergence | {_format_optional_float(summary.get('mean_first_token_id_divergence'), precision=2)} |",
        f"| count | {summary['count']} |",
    ]
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


def _format_result_line(model: dict[str, Any], result: dict[str, Any]) -> str:
    common = f"hf_reused={result['hf_reused']} bundle_built={result['bundle_built']}"
    if result.get("mode") == "continuation":
        return (
            f"model={model['name']} exact={result['exact_match_rate']:.4f} "
            f"token_agreement={result['token_prefix_agreement']:.4f} "
            f"mean_first_divergence={result['mean_first_divergence']:.2f} "
            f"granularity={result.get('comparison_granularity', '')} {common}"
        )
    if result.get("mode") == "diffusion_image_clip_parity":
        return (
            f"model={model['name']} pass_rate={result['overall_pass_rate']:.4f} "
            f"passed={result['passed_count']}/{result['valid_count']} {common}"
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
    parser = argparse.ArgumentParser(description="Local dataset task evaluation for TRTMC bundles.")
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

    p = sub.add_parser("run-hf")
    p.add_argument("--model", required=True)
    p.add_argument("--family", default="")
    p.add_argument("--reference-family", default="")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--predictions")
    p.add_argument("--raw-output")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default="")
    p.add_argument("--attn-impl", default="")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--apply-chat-template", action="store_true")
    add_generation_args(p)

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

    p = sub.add_parser("compare-continuation")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--hf-predictions")
    p.add_argument("--trtfb-predictions")
    p.add_argument("--model", default="", help="Optional model id; enables token-level parity.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output")

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
    p.add_argument("--work-root", default="/tmp/trtmc-task-eval")
    p.add_argument("--engine-dir", default="")
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
    p.add_argument("--force-build", action="store_true", help="Rebuild the .trtfb bundle.")
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
    p.add_argument("--hf-dtype", choices=["auto", "float16", "bfloat16"], default="auto")
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


def cmd_eval(args: argparse.Namespace) -> int:
    suites = load_suites(Path(args.suites))
    suite = suite_by_id(suites, args.suite)
    dataset_kind = suite.get("dataset", {}).get("kind", "")
    if dataset_kind not in {
        "mmlu_five_shot_json",
        "vlm_chat_json",
        "vlm_unified_json",
        "asr_chat_json",
        "diffusion_prompt_tsv",
        "diffusion_prompt_json",
        "seedtts_json",
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
        print(f"[task_eval] ({idx}/{len(selected)}) suite={suite['id']} model={model['name']}")
        if use_workers:
            result = run_eval_model_worker(suite=suite, model=model, args=args)
            if result.get("status") == "failed":
                results.append(result)
                print(
                    f"[task_eval] model={model['name']} status=failed "
                    f"error_type={result.get('error_type', '')} error={result.get('error', '')} "
                    f"log={result.get('worker_log', '')}"
                )
                if args.fail_fast:
                    raise RuntimeError(
                        f"Model {model['name']} failed in worker; see {result.get('worker_log', '')}"
                    )
                if result.get("gpu_cleanup_confirmed") is False:
                    reason = (
                        f"Skipped because GPU cleanup after {model['name']} OOM was not confirmed"
                    )
                    print(f"[task_eval] {reason}")
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
                    f"[task_eval] model={model['name']} status=failed "
                    f"error_type={type(exc).__name__} error={exc}"
                )
                continue
        result.setdefault("status", "passed")
        results.append(result)
        print(f"[task_eval] {_format_result_line(model, result)}")
        if result.get("gpu_cleanup_confirmed") is False:
            reason = f"Skipped because GPU cleanup after {model['name']} was not confirmed"
            print(f"[task_eval] {reason}")
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
    print(f"[task_eval] summary={summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cmd == "list-suites":
        return cmd_list_suites(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "prepare":
        return cmd_prepare(args)
    if args.cmd == "run-hf":
        run_hf_reference(args)
        return 0
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

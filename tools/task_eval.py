#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import traceback
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e_harness.contracts import (  # noqa: E402
    MODEL_REFERENCE_FAMILY,
    REFERENCE_FAMILY_TO_USER_CONTRACT,
    RUNTIME_TO_TASK_STRATEGY,
)


DEFAULT_SUITES = REPO_ROOT / "tests" / "task_eval" / "validation_suites.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "tests" / "e2e" / "models"
DEFAULT_WAIVES = REPO_ROOT / "tests" / "e2e" / "waives.txt"
ERROR_OUTPUT_TEXT = "TensorRT Edge LLM cannot handle this request. Fails."
CHOICE_LETTERS = set("ABCDEFGHIJ")


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
    data = load_structured_file(path)
    suites = data.get("suites", [])
    if not isinstance(suites, list):
        raise ValueError(f"{path}: 'suites' must be a list")
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


def _base_case_name(name: str) -> str:
    return re.sub(r"-tp\d+$", "", name)


def infer_reference_family(raw: dict[str, Any]) -> str:
    if raw.get("reference_family"):
        return str(raw["reference_family"])
    name = str(raw.get("name", ""))
    return MODEL_REFERENCE_FAMILY.get(name) or MODEL_REFERENCE_FAMILY.get(_base_case_name(name), "")


def infer_user_contract(raw: dict[str, Any], reference_family: str) -> str:
    if raw.get("user_contract"):
        return str(raw["user_contract"])
    return REFERENCE_FAMILY_TO_USER_CONTRACT.get(reference_family, "")


def manifest_record(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    build_args = raw.get("build_args", {})
    runtime_strategy = str(raw.get("runtime_strategy", "decoder_kv_cache"))
    task_strategy = str(
        raw.get("task_strategy") or RUNTIME_TO_TASK_STRATEGY.get(runtime_strategy, "")
    )
    reference_family = infer_reference_family(raw)
    user_contract = infer_user_contract(raw, reference_family)
    distributed = raw.get("distributed_runtime", {})
    requires_multi_device = bool(distributed.get("enabled")) or (
        str(raw.get("ci_tier", "")) == "multi_device"
    )
    return {
        "name": raw.get("name", path.stem),
        "manifest": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "hf_id": raw.get("hf_id") or raw.get("model_id", ""),
        "bundle": raw.get("bundle", f"{path.stem}.trtfb"),
        "family": raw.get("family", ""),
        "runtime_strategy": runtime_strategy,
        "task_strategy": task_strategy,
        "reference_family": reference_family,
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
    }


def load_manifest_records(models_dir: Path = DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    return [manifest_record(path) for path in sorted(models_dir.glob("*.json"))]


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
    task_strategies = _selector_values(selectors, "task_strategies")
    runtime_strategies = _selector_values(selectors, "runtime_strategies")
    user_contracts = _selector_values(selectors, "user_contracts")
    exclude_user_contracts = _selector_values(selectors, "exclude_user_contracts")
    families = _selector_values(selectors, "families")
    exclude_families = _selector_values(selectors, "exclude_families")

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


def build_plan(
    suites: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    suite_id: str | None = None,
    single_device_only: bool = False,
    include_non_matching: bool = False,
    waives: dict[str, tuple[str, str]] | None = None,
    include_waived: bool = False,
) -> list[dict[str, Any]]:
    selected_suites = [suite_by_id(suites, suite_id)] if suite_id else suites
    rows: list[dict[str, Any]] = []
    waives = waives or {}
    for suite in selected_suites:
        for model in models:
            matched, reason = suite_match_reason(suite, model)
            if single_device_only and model["requires_multi_device"]:
                matched = False
                reason = "requires multi-device runtime"
            waive = waives.get(str(model["name"]))
            if matched and waive and not include_waived:
                action, waive_reason = waive
                matched = False
                reason = f"waived {action}: {waive_reason}".strip()
            if matched or include_non_matching:
                rows.append({
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
                })
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


def _copy_dataset_header(data: dict[str, Any], requests: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: v for k, v in data.items() if k != "requests"}
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
            sample = {
                "sample_id": f"mmlu_{dataset_index:06d}",
                "dataset_index": dataset_index,
                "eval_index": out_idx,
                "subject": request.get("subject", ""),
                "answer": request["answer"],
                "prompt": _request_prompt(request),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

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
        value = item.get("image")
        if isinstance(value, str):
            refs.append(value)
    return refs


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
    candidates = ", ".join(str(path) for path in _candidate_dataset_asset_paths(dataset_path, asset_ref))
    raise FileNotFoundError(f"Could not resolve dataset asset {asset_ref!r}; tried: {candidates}")


def vlm_request_images(dataset_path: Path, request: dict[str, Any]) -> list[Path]:
    refs: list[str] = []
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                refs.extend(_message_image_refs(msg))
    if not refs and isinstance(request.get("image"), str):
        refs.append(str(request["image"]))
    return [resolve_dataset_asset_path(dataset_path, ref) for ref in refs]


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


def _normalized_single_user_vlm_request(
    request: dict[str, Any],
    *,
    prompt: str,
    image_path: Path,
) -> dict[str, Any]:
    normalized = {key: value for key, value in request.items() if key != "messages"}
    normalized["messages"] = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
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


def prepare_vlm_chat_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
) -> dict[str, Path]:
    data, indexed = load_vlm_chat_requests(
        dataset_path,
        limit=limit,
        subject=subject,
        sample_seed=sample_seed,
    )
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"answers": answers_path, "prompts": prompts_path, "manifest": manifest_path}


def prepare_task_dataset(
    *,
    dataset_path: Path,
    work_dir: Path,
    suite: dict[str, Any],
    limit: int = 0,
    subject: str = "",
    sample_seed: int | None = None,
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
        )
    if dataset_kind == "vlm_chat_json":
        return prepare_vlm_chat_dataset(
            dataset_path=dataset_path,
            work_dir=work_dir,
            suite=suite,
            limit=limit,
            subject=subject,
            sample_seed=sample_seed,
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
    if not isinstance(responses, list) or not isinstance(requests, list) or len(responses) != len(requests):
        return False
    if require_token_ids:
        return all(isinstance(row, dict) and _generated_token_ids(row) is not None for row in responses)
    return True


def convert_trtfb_jsonl_to_predictions(raw_path: Path, predictions_path: Path) -> None:
    rows = []
    for row in load_jsonl(raw_path):
        rows.append({
            "sample_id": row.get("sample_id", ""),
            "output_text": row.get("text", ""),
            "generated_tokens": row.get("generated_tokens"),
            "generated_token_ids": _generated_token_ids(row),
            "wall_ms": row.get("wall_ms"),
            "source": "trtfb",
        })
    write_predictions(predictions_path, rows)


def _trtmc_binary_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "trtmc_binary", "") or "build/trtmc")


def run_vlm_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    log_path = work_dir / (args.log or "trtfb_run.log")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(
        defaults.get("max_new_tokens", 8)
    )
    temperature = args.temperature if args.temperature is not None else float(
        defaults.get("temperature", 1.0)
    )
    top_k = args.top_k if args.top_k is not None else int(defaults.get("top_k", 1))
    top_p = args.top_p if args.top_p is not None else float(defaults.get("top_p", 1.0))
    min_p = args.min_p if args.min_p is not None else float(defaults.get("min_p", 0.0))
    seed = args.seed if args.seed is not None else int(defaults.get("seed", -1))

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    rows: list[dict[str, Any]] = []
    prompt_rows = load_jsonl(work_dir / "prompts.jsonl")
    with raw_output.open("w", encoding="utf-8") as raw_f, log_path.open("w", encoding="utf-8") as log_f:
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


def is_correct(prediction: str, reference: str) -> bool:
    pred_clean = clean_text(prediction)
    ref_clean = clean_text(reference)
    if ref_clean in CHOICE_LETTERS:
        pred_clean = parse_multi_choice_response(pred_clean)
    return pred_clean == ref_clean


def score_predictions(predictions_data: dict[str, Any], answers_data: dict[str, Any]) -> dict[str, Any]:
    responses = predictions_data.get("responses", [])
    requests = answers_data.get("requests", [])
    if len(responses) != len(requests):
        raise ValueError(
            f"Predictions and answers must have the same length: "
            f"{len(responses)} != {len(requests)}"
        )

    correct = 0
    skipped = 0
    subject_stats: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[dict[str, Any]] = []
    for idx, (response, request) in enumerate(zip(responses, requests, strict=True)):
        output_text = str(response.get("output_text", ""))
        subject = str(request.get("subject", ""))
        answer = str(request["answer"])
        if output_text == ERROR_OUTPUT_TEXT:
            skipped += 1
            samples.append({
                "index": idx,
                "sample_id": response.get("sample_id", f"sample_{idx}"),
                "subject": subject,
                "answer": answer,
                "prediction": output_text,
                "skipped": True,
                "correct": False,
            })
            continue
        ok = is_correct(output_text, answer)
        correct += int(ok)
        subject_stats[subject]["total"] += 1
        subject_stats[subject]["correct"] += int(ok)
        samples.append({
            "index": idx,
            "sample_id": response.get("sample_id", f"sample_{idx}"),
            "subject": subject,
            "answer": answer,
            "prediction": output_text,
            "parsed_prediction": parse_multi_choice_response(clean_text(output_text)),
            "skipped": False,
            "correct": ok,
        })

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
        "skipped_count": skipped,
        "total_count": len(requests),
        "subject_accuracy": subject_accuracy,
        "samples": samples,
    }


def compare_prediction_sets(
    hf_predictions: dict[str, Any],
    trtfb_predictions: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    hf_score = score_predictions(hf_predictions, answers)
    trtfb_score = score_predictions(trtfb_predictions, answers)
    hf_responses = hf_predictions["responses"]
    trt_responses = trtfb_predictions["responses"]
    requests = answers["requests"]
    if len(hf_responses) != len(trt_responses):
        raise ValueError("HF and TRTFB predictions must have the same length")

    agreement = 0
    buckets = Counter()
    disagreements: list[dict[str, Any]] = []
    for idx, (hf_row, trt_row, req) in enumerate(zip(hf_responses, trt_responses, requests, strict=True)):
        answer = str(req["answer"])
        hf_pred = parse_multi_choice_response(clean_text(str(hf_row.get("output_text", ""))))
        trt_pred = parse_multi_choice_response(clean_text(str(trt_row.get("output_text", ""))))
        hf_ok = is_correct(str(hf_row.get("output_text", "")), answer)
        trt_ok = is_correct(str(trt_row.get("output_text", "")), answer)
        agreement += int(hf_pred == trt_pred)
        if hf_ok and trt_ok:
            buckets["both_correct"] += 1
        elif hf_ok and not trt_ok:
            buckets["hf_correct_trtfb_wrong"] += 1
        elif not hf_ok and trt_ok:
            buckets["hf_wrong_trtfb_correct"] += 1
        else:
            buckets["both_wrong"] += 1
        if hf_pred != trt_pred:
            disagreements.append({
                "index": idx,
                "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
                "subject": req.get("subject", ""),
                "answer": answer,
                "hf_prediction": hf_pred,
                "trtfb_prediction": trt_pred,
            })

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


def _vlm_model_class(transformers_mod: Any) -> Any:
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers_mod, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("Transformers installation does not expose a VLM-capable AutoModel class")


def _vlm_prompt_has_image_placeholder(text: str) -> bool:
    return any(marker in text for marker in (
        "<|image_pad|>",
        "<|vision_start|>",
        "<image>",
        "<IMG_CONTEXT>",
    ))


def _vlm_fallback_prompt(model_id: str, prompt: str) -> str:
    lower_id = model_id.lower()
    if "qwen" in lower_id and "vl" in lower_id:
        return f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
    if "internvl" in lower_id:
        return f"<IMG_CONTEXT>\n{prompt}"
    return prompt


def _vlm_chat_text(
    processor: Any,
    request: dict[str, Any],
    fallback_prompt: str,
    model_id: str,
) -> str:
    messages = request.get("messages")
    rendered = ""
    if hasattr(processor, "apply_chat_template") and isinstance(messages, list):
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if (
        not rendered
        and tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and isinstance(messages, list)
    ):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if rendered:
        return rendered
    if _vlm_prompt_has_image_placeholder(fallback_prompt):
        return fallback_prompt
    return _vlm_fallback_prompt(model_id, fallback_prompt)


def _strip_generated_text_prefix(text: str, prompt: str) -> str:
    generated = text.strip()
    for marker in ("assistant\n", "assistant:", "ASSISTANT:"):
        if marker in generated:
            generated = generated.split(marker, 1)[-1].strip()
            break
    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def _to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


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
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(
        defaults.get("max_new_tokens", 8)
    )
    temperature = args.temperature if args.temperature is not None else float(
        defaults.get("temperature", 1.0)
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
    model_cls = _vlm_model_class(transformers)
    model_kwargs = {
        "torch_dtype": _model_dtype(torch, args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = model_cls.from_pretrained(args.model, **model_kwargs).eval()
    if not args.device_map:
        device = torch.device(args.device)
        model.to(device)
    else:
        device = model.device

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
            print(f"[task_eval.vlm_hf] sample={idx + 1}/{len(answers['requests'])}", file=sys.stderr)
    write_predictions(pred_path, responses)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_hf_reference(args: argparse.Namespace) -> None:
    if _work_dataset_kind(Path(args.work_dir)) == "vlm_chat_json":
        run_vlm_hf_reference(args)
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
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(
        defaults.get("max_new_tokens", 1)
    )
    temperature = args.temperature if args.temperature is not None else float(
        defaults.get("temperature", 1.0)
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
            generated = output_ids[0, encoded["input_ids"].shape[1]:]
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


def run_trtfb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    if _work_dataset_kind(work_dir) == "vlm_chat_json":
        run_vlm_trtfb(args)
        return
    defaults = generation_defaults(work_dir)
    raw_output = work_dir / (args.raw_output or "trtfb_raw.jsonl")
    predictions = work_dir / (args.predictions or "trtfb_predictions.json")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(
        defaults.get("max_new_tokens", 1)
    )
    temperature = args.temperature if args.temperature is not None else float(
        defaults.get("temperature", 1.0)
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
        proc = subprocess.run(cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env)
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
        raise RuntimeError(f"Bundle build failed for {model['name']} rc={proc.returncode}; see {log_path}")
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
            raise ValueError(f"Model selector(s) not found in suite {suite['id']}: {', '.join(missing)}")
        # Preserve manifest order and drop duplicates from overlapping selectors.
        wanted = {model["name"] for model in filtered}
        selected = [model for model in selected if model["name"] in wanted]
    return selected


def _namespace_for_run_hf(args: argparse.Namespace, model: dict[str, Any], work_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=model["hf_id"],
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


def _namespace_for_run_trtfb(args: argparse.Namespace, bundle_path: Path, work_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        bundle=str(bundle_path),
        work_dir=str(work_dir),
        trtmc_binary=args.trtmc_binary,
        benchmark_binary=args.benchmark_binary,
        hf_python=args.hf_python,
        backend_dir=args.backend_dir,
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
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-hf",
        "--model", str(hf_args.model),
        "--work-dir", str(hf_args.work_dir),
        "--predictions", str(hf_args.predictions),
        "--raw-output", str(hf_args.raw_output),
        "--dtype", str(hf_args.dtype),
        "--device", str(hf_args.device),
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
    work_root = Path(args.work_root)
    work_dir = work_root / suite["id"] / str(model["name"])
    dataset_path = Path(args.dataset or suite.get("dataset", {}).get("default_path", ""))
    if not dataset_path:
        raise ValueError(f"Suite {suite['id']} has no dataset path; pass --dataset")
    scorer = str(suite.get("scoring", {}).get("scorer", "mcq"))
    prepare_task_dataset(
        dataset_path=dataset_path,
        work_dir=work_dir,
        suite=suite,
        limit=args.limit,
        subject=args.subject,
        sample_seed=args.sample_seed,
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
    if not args.skip_prompt_length_check:
        prompt_rows_path = work_dir / "prompts.jsonl"
        max_prompt_len = max_prompt_token_length(
            model_id=str(model["hf_id"]),
            prompts_path=prompt_rows_path,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code or bool(model.get("trust_remote_code", False)),
        )
    max_cache_length = requested_build_max_cache_length(
        suite,
        model,
        args.build_max_cache_length,
        prompt_max_tokens=max_prompt_len,
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
        "hf_reused": hf_reused,
        "bundle_built": built,
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
    else:
        hf_data = json.loads((work_dir / "hf_predictions.json").read_text(encoding="utf-8"))
        trtfb_data = json.loads((work_dir / "trtfb_predictions.json").read_text(encoding="utf-8"))
        summary = compare_prediction_sets(
            hf_data, trtfb_data, json.loads(answers_path.read_text(encoding="utf-8"))
        )
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_summary_markdown(summary, work_dir / "summary.md")
        result = {
            **base_result,
            "mode": "mcq",
            "hf_accuracy": summary["hf"]["overall_accuracy"],
            "trtfb_accuracy": summary["trtfb"]["overall_accuracy"],
            "accuracy_delta_trtfb_minus_hf": summary["accuracy_delta_trtfb_minus_hf"],
            "prediction_agreement_rate": summary["prediction_agreement_rate"],
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
            f"HF and TRTFB predictions must have the same length: "
            f"{len(hf_rows)} != {len(trt_rows)}"
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
        tokenize = lambda s: list(s)

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
        samples.append({
            "index": idx,
            "sample_id": hf_row.get("sample_id", f"sample_{idx}"),
            "exact": is_exact,
            "text_exact": text_is_exact,
            "first_divergence": divergence,
            "hf_len": len(hf_tokens),
            "trtfb_len": len(trt_tokens),
            "hf_token_at_divergence": hf_token_at_divergence,
            "trtfb_token_at_divergence": trt_token_at_divergence,
        })

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
        Path(args.trtfb_predictions) if args.trtfb_predictions else work_dir / "trtfb_predictions.json"
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


def _format_result_line(model: dict[str, Any], result: dict[str, Any]) -> str:
    common = f"hf_reused={result['hf_reused']} bundle_built={result['bundle_built']}"
    if result.get("mode") == "continuation":
        return (
            f"model={model['name']} exact={result['exact_match_rate']:.4f} "
            f"token_agreement={result['token_prefix_agreement']:.4f} "
            f"mean_first_divergence={result['mean_first_divergence']:.2f} "
            f"granularity={result.get('comparison_granularity', '')} {common}"
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
    p.add_argument("--waives", default=str(DEFAULT_WAIVES), help="E2E waives file used to skip known-bad models.")
    p.add_argument("--waive-platform", default="", help="Platform prefix to honor in the waives file.")
    p.add_argument("--include-waived", action="store_true", help="Include models listed in the waives file.")
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

    p = sub.add_parser("compare")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--answers")
    p.add_argument("--hf-predictions")
    p.add_argument("--trtfb-predictions")
    p.add_argument("--output", default="")
    p.add_argument("--markdown", default="")

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
    p.add_argument("--waives", default=str(DEFAULT_WAIVES), help="E2E waives file used to skip known-bad models.")
    p.add_argument("--waive-platform", default="", help="Platform prefix to honor in the waives file.")
    p.add_argument("--include-waived", action="store_true", help="Include models listed in the waives file.")
    p.add_argument("--work-root", default="/tmp/trtmc-task-eval")
    p.add_argument("--engine-dir", default="")
    p.add_argument("--bundle", default="", help="Prebuilt bundle path; only valid with one --model.")
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
            print(f"{suite['id']}\t{suite.get('dataset', {}).get('kind', '')}\t{suite.get('description', '')}")
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
        output_path = Path(args.output) if args.output else predictions_path.with_suffix(".score.json")
    score = score_predictions(
        json.loads(predictions_path.read_text(encoding="utf-8")),
        json.loads(answers_path.read_text(encoding="utf-8")),
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
        Path(args.trtfb_predictions) if args.trtfb_predictions else work_dir / "trtfb_predictions.json"
    )
    summary = compare_prediction_sets(
        json.loads(hf_path.read_text(encoding="utf-8")),
        json.loads(trtfb_path.read_text(encoding="utf-8")),
        json.loads(answers_path.read_text(encoding="utf-8")),
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
        if all(after <= before + margin_mib for before, after in zip(before_mib, current, strict=True)):
            return True, current
        time.sleep(poll_s)
    return False, current


def is_oom_failure(result: dict[str, Any], returncode: int = 0) -> bool:
    text = " ".join(
        str(result.get(key, ""))
        for key in ("error", "error_type", "worker_log_tail")
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
    cmd = [sys.executable, str(Path(__file__).resolve()), "eval-worker", "--request", str(request_path)]
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, check=False, text=True, stdout=log_f, stderr=subprocess.STDOUT, env=env)

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
        result["status"] = "passed"
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
    if dataset_kind not in {"mmlu_five_shot_json", "vlm_chat_json"}:
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
        result["status"] = "passed"
        results.append(result)
        print(f"[task_eval] {_format_result_line(model, result)}")
        if result.get("gpu_cleanup_confirmed") is False:
            reason = (
                f"Skipped because GPU cleanup after {model['name']} was not confirmed"
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

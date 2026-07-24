#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-sample disagreement evidence for TRTMC validation reports."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trtmc.validation-disagreement/v1"
INLINE_DISAGREEMENT_LIMIT = 20
_FORBIDDEN_REPRO_ENTRYPOINTS = (
    "task_eval.py",
    "trtmc_compare.py",
    "trtmc_reference.py",
    "trtmc_validate.py",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _sample_id(record: Mapping[str, Any], fallback: str = "") -> str:
    for name in ("sample_id", "case_id", "id", "name"):
        value = record.get(name)
        if value is not None and str(value):
            return str(value)
    return fallback


def _record_is_disagreement(record: Mapping[str, Any]) -> bool:
    if record.get("diverged") is True:
        return True
    for name in (
        "agreement_match",
        "exact",
        "exact_match",
        "passed",
        "top1_agreement",
        "transcript_exact",
    ):
        if record.get(name) is False:
            return True
    return str(record.get("status", "") or "").lower() in {
        "disagreement",
        "failed",
        "mismatch",
    }


def _summary_disagreements(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = summary.get("disagreements")
    if isinstance(explicit, list):
        return [item for item in explicit if isinstance(item, dict)]
    for collection_name in ("samples", "cases"):
        collection = summary.get(collection_name)
        if isinstance(collection, list):
            return [
                item
                for item in collection
                if isinstance(item, dict) and _record_is_disagreement(item)
            ]
    return []


def _indexed_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        _sample_id(row, f"sample-{index}"): dict(row)
        for index, row in enumerate(rows)
    }


def _prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    responses = data.get("responses", [])
    return _indexed_rows(responses) if isinstance(responses, list) else {}


def _safe_sample_name(sample_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return value or "sample"


def _replace_placeholders(value: str, replacements: Mapping[str, str]) -> str:
    result = value
    for name, replacement in replacements.items():
        result = result.replace(f"{{{name}}}", replacement)
    return result


def _command_from_template(
    metadata: Mapping[str, Any],
    replacements: Mapping[str, str],
) -> str:
    command = metadata.get("command", [])
    if not isinstance(command, list) or not command:
        return ""
    tokens = [
        _replace_placeholders(str(token), replacements)
        for token in command
    ]
    rendered = shlex.join(tokens)
    if any(name in rendered for name in _FORBIDDEN_REPRO_ENTRYPOINTS):
        return ""
    return rendered


def _write_trtmc_input(path: Path, prompt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(prompt), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _reproduction_commands(
    *,
    sample_id: str,
    prompt: Mapping[str, Any],
    work_dir: Path,
    case_dir: Path,
    reference_metadata: Mapping[str, Any],
    trtmc_metadata: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    sample_dir = case_dir / "repro" / _safe_sample_name(sample_id)
    input_path = sample_dir / "input.jsonl"
    reference_predictions = sample_dir / "reference_predictions.json"
    reference_raw = sample_dir / "reference_raw.jsonl"
    trtmc_raw = sample_dir / "trtmc_raw.jsonl"
    replacements = {
        "sample_id": sample_id,
        "work_dir": str(work_dir),
        "input_jsonl": str(input_path),
        "reference_predictions_json": str(reference_predictions),
        "reference_raw_jsonl": str(reference_raw),
        "trtmc_raw_jsonl": str(trtmc_raw),
    }
    reference_command = _command_from_template(reference_metadata, replacements)
    trtmc_command = _command_from_template(trtmc_metadata, replacements)
    artifacts = {}
    if trtmc_command and prompt:
        _write_trtmc_input(input_path, prompt)
        artifacts["trtmc_input"] = str(input_path.relative_to(case_dir))
    return (
        {"reference": reference_command, "trtmc": trtmc_command},
        artifacts,
    )


def _reason(record: Mapping[str, Any]) -> str:
    configured = str(record.get("reason", "") or "")
    if configured:
        return configured
    if record.get("diverged") is True or record.get("exact") is False:
        return "output_divergence"
    if record.get("top1_agreement") is False:
        return "top1_mismatch"
    if record.get("transcript_exact") is False:
        return "transcript_mismatch"
    return "comparison_threshold"


def build_disagreement_artifact(
    *,
    work_dir: Path,
    case_dir: Path,
) -> dict[str, Any]:
    summary = _load_json(work_dir / "summary.json")
    comparison_rows = _summary_disagreements(summary)
    prompts = _indexed_rows(_load_jsonl(work_dir / "prompts.jsonl"))
    reference_rows = _prediction_rows(work_dir / "hf_predictions.json")
    trtmc_rows = _prediction_rows(work_dir / "trtfb_predictions.json")
    reference_metadata = _load_json(work_dir / "hf_native_repro.json")
    trtmc_metadata = _load_json(work_dir / "trtfb_repro.json")
    records = []
    for index, comparison in enumerate(comparison_rows):
        sample_id = _sample_id(comparison, f"sample-{index}")
        prompt = prompts.get(sample_id, {})
        commands, artifacts = _reproduction_commands(
            sample_id=sample_id,
            prompt=prompt,
            work_dir=work_dir,
            case_dir=case_dir,
            reference_metadata=reference_metadata,
            trtmc_metadata=trtmc_metadata,
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "reason": _reason(comparison),
                "input": prompt,
                "reference_result": reference_rows.get(sample_id, {}),
                "trtmc_result": trtmc_rows.get(sample_id, {}),
                "comparison": dict(comparison),
                "reproduce": commands,
                "artifacts": artifacts,
            }
        )

    artifact_path = case_dir / "disagreements.jsonl"
    with artifact_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "count": len(records),
        "path": artifact_path.name,
        "inline_limit": INLINE_DISAGREEMENT_LIMIT,
        "reference_vanilla_available": bool(reference_metadata),
        "trtmc_vanilla_available": bool(trtmc_metadata),
    }


def load_disagreement_preview(
    path: Path,
    *,
    limit: int = INLINE_DISAGREEMENT_LIMIT,
) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if len(rows) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows

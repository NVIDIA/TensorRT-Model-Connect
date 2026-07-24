#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-sample disagreement evidence for TRTMC validation reports."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trtmc.validation-disagreement/v1"
INLINE_DISAGREEMENT_LIMIT = 20
_FORBIDDEN_REPRO_ENTRYPOINTS = (
    "task_eval.py",
    "trtmc_compare.py",
    "trtmc_reference.py",
    "trtmc_validate.py",
)
_IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
_AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}
_VIDEO_SUFFIXES = {".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_MEDIA_FILE_LIMIT = 128 * 1024 * 1024


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
    for collection_name in ("samples", "cases", "pairs"):
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


def _expand_pair_disagreements(
    rows: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expanded = []
    for index, row in enumerate(rows):
        sample_id = _sample_id(row, f"sample-{index}")
        pair_id = str(row.get("pair_id", "") or "")
        if sample_id in prompts or not pair_id:
            expanded.append(dict(row))
            continue
        pair_samples = [
            prompt_id
            for prompt_id, prompt in prompts.items()
            if str(prompt.get("pair_id", "") or "") == pair_id
        ]
        if not pair_samples:
            expanded.append(dict(row))
            continue
        for prompt_id in pair_samples:
            expanded.append({**dict(row), "sample_id": prompt_id})
    return expanded


def _prediction_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    responses = data.get("responses", [])
    return _indexed_rows(responses) if isinstance(responses, list) else {}


def _answer_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    requests = data.get("requests", [])
    return _indexed_rows(requests) if isinstance(requests, list) else {}


def _native_trtmc_commands(path: Path) -> dict[str, list[str]]:
    commands = {}
    for row in _load_jsonl(path):
        sample_id = _sample_id(row)
        command = row.get("command")
        if sample_id and isinstance(command, list) and command:
            commands[sample_id] = [str(token) for token in command]
    return commands


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


def _media_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    return ""


def _copy_media(
    *,
    source: Path,
    media_dir: Path,
    case_dir: Path,
    label: str,
    ordinal: int,
) -> dict[str, str] | None:
    kind = _media_kind(source)
    if (
        not kind
        or not source.is_file()
        or source.stat().st_size > _MEDIA_FILE_LIMIT
    ):
        return None
    stem = _safe_sample_name(label).lower()
    target = media_dir / f"{ordinal:02d}-{stem}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "label": label,
        "kind": kind,
        "path": str(target.relative_to(case_dir)),
    }


def _frame_paths(root: Path) -> list[Path]:
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(images) <= 3:
        return images
    return [images[0], images[len(images) // 2], images[-1]]


def _media_candidates(
    label: str,
    record: Mapping[str, Any],
) -> list[tuple[str, Path]]:
    direct_fields = (
        "audio",
        "condition_image",
        "hf_image",
        "image",
        "output_video",
        "segmented_image_path",
        "trtfb_image",
        "video",
        "video_path",
        "wav_path",
    )
    candidates = [
        (f"{label} {field}", Path(value))
        for field in direct_fields
        if isinstance((value := record.get(field)), str) and value
    ]
    images = record.get("images", [])
    if isinstance(images, list):
        candidates.extend(
            (f"{label} input image {index + 1}", Path(str(value)))
            for index, value in enumerate(images)
            if str(value)
        )
    candidates.extend(_frame_candidates(label, record.get("frames_dir")))
    return candidates


def _frame_candidates(label: str, value: Any) -> list[tuple[str, Path]]:
    if not isinstance(value, str) or not Path(value).is_dir():
        return []
    root = Path(value)
    images = [
        (f"{label} frame {index + 1}", path)
        for index, path in enumerate(_frame_paths(root))
    ]
    videos = [
        (f"{label} video", path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
    ]
    return images + videos


def _collect_media(
    *,
    sample_dir: Path,
    case_dir: Path,
    prompt: Mapping[str, Any],
    reference_result: Mapping[str, Any],
    trtmc_result: Mapping[str, Any],
) -> list[dict[str, str]]:
    media_dir = sample_dir / "media"
    seen = set()
    copied = []
    sources = (
        ("Input", prompt),
        ("Reference", reference_result),
        ("TRTMC", trtmc_result),
    )
    for label, record in sources:
        for candidate_label, source in _media_candidates(label, record):
            try:
                resolved = source.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            item = _copy_media(
                source=resolved,
                media_dir=media_dir,
                case_dir=case_dir,
                label=candidate_label,
                ordinal=len(copied) + 1,
            )
            if item is not None:
                seen.add(resolved)
                copied.append(item)
    return copied


def _reproduction_commands(
    *,
    sample_id: str,
    prompt: Mapping[str, Any],
    work_dir: Path,
    case_dir: Path,
    reference_metadata: Mapping[str, Any],
    trtmc_metadata: Mapping[str, Any],
    native_reference_command: Sequence[str] = (),
    native_trtmc_command: Sequence[str] = (),
) -> tuple[dict[str, str], dict[str, Any]]:
    sample_dir = case_dir / "repro" / _safe_sample_name(sample_id)
    input_path = sample_dir / "input.jsonl"
    reference_predictions = sample_dir / "reference_predictions.json"
    reference_raw = sample_dir / "reference_raw.jsonl"
    reference_input = sample_dir / "reference_input.jsonl"
    reference_artifacts = sample_dir / "reference_artifacts"
    trtmc_raw = sample_dir / "trtmc_raw.jsonl"
    replacements = {
        "sample_id": sample_id,
        "work_dir": str(work_dir),
        "input_jsonl": str(input_path),
        "reference_predictions_json": str(reference_predictions),
        "reference_raw_jsonl": str(reference_raw),
        "reference_input_jsonl": str(reference_input),
        "reference_artifacts_dir": str(reference_artifacts),
        "trtmc_raw_jsonl": str(trtmc_raw),
        "sample_seed": _sample_seed(trtmc_metadata, prompt),
        "reference_sample_seed": _sample_seed(reference_metadata, prompt),
    }
    reference_command = _resolved_command(
        reference_metadata,
        replacements,
        native_reference_command,
    )
    trtmc_command = _resolved_command(
        trtmc_metadata,
        replacements,
        native_trtmc_command,
    )
    artifacts = {}
    if _write_elf_reference_input(
        reference_input,
        sample_id=sample_id,
        prompt=prompt,
        enabled=bool(
            reference_command
            and reference_metadata.get("input_format") == "elf_reference_jsonl"
        ),
    ):
        artifacts["reference_input"] = str(reference_input.relative_to(case_dir))
    if trtmc_command and prompt:
        _write_trtmc_input(input_path, prompt)
        artifacts["trtmc_input"] = str(input_path.relative_to(case_dir))
    return (
        {"reference": reference_command, "trtmc": trtmc_command},
        artifacts,
    )


def _sample_seed(
    metadata: Mapping[str, Any],
    prompt: Mapping[str, Any],
) -> str:
    value = metadata.get("base_seed")
    if value is None:
        return ""
    index = prompt.get("seed_index", prompt.get("eval_index", 0))
    return str(int(value) + int(index))


def _resolved_command(
    metadata: Mapping[str, Any],
    replacements: Mapping[str, str],
    native_command: Sequence[str],
) -> str:
    command = _command_from_template(metadata, replacements)
    if command or not native_command:
        return command
    rendered = shlex.join(str(token) for token in native_command)
    if any(name in rendered for name in _FORBIDDEN_REPRO_ENTRYPOINTS):
        return ""
    return rendered


def _write_elf_reference_input(
    path: Path,
    *,
    sample_id: str,
    prompt: Mapping[str, Any],
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": sample_id,
                "input": str(
                    prompt.get("source_text", prompt.get("prompt", ""))
                ),
                "output": str(prompt.get("answer", "")),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


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
    prompts = _indexed_rows(_load_jsonl(work_dir / "prompts.jsonl"))
    comparison_rows = _expand_pair_disagreements(
        _summary_disagreements(summary),
        prompts,
    )
    answers = _answer_rows(work_dir / "answers.json")
    reference_rows = _prediction_rows(work_dir / "hf_predictions.json")
    trtmc_rows = _prediction_rows(work_dir / "trtfb_predictions.json")
    reference_metadata = _load_json(work_dir / "hf_native_repro.json")
    native_reference_commands = _native_trtmc_commands(
        work_dir / "hf_native_commands.jsonl"
    )
    trtmc_metadata = _load_json(work_dir / "trtfb_repro.json")
    native_trtmc_commands = _native_trtmc_commands(
        work_dir / "trtfb_native_commands.jsonl"
    )
    records = []
    for index, comparison in enumerate(comparison_rows):
        sample_id = _sample_id(comparison, f"sample-{index}")
        prompt = {
            **answers.get(sample_id, {}),
            **prompts.get(sample_id, {}),
        }
        commands, artifacts = _reproduction_commands(
            sample_id=sample_id,
            prompt=prompt,
            work_dir=work_dir,
            case_dir=case_dir,
            reference_metadata=reference_metadata,
            trtmc_metadata=trtmc_metadata,
            native_reference_command=native_reference_commands.get(
                sample_id,
                (),
            ),
            native_trtmc_command=native_trtmc_commands.get(sample_id, ()),
        )
        media = _collect_media(
            sample_dir=case_dir / "repro" / _safe_sample_name(sample_id),
            case_dir=case_dir,
            prompt=prompt,
            reference_result=reference_rows.get(sample_id, {}),
            trtmc_result=trtmc_rows.get(sample_id, {}),
        )
        if media:
            artifacts["media"] = media
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
        "reference_vanilla_available": bool(
            reference_metadata.get("command") or native_reference_commands
        ),
        "trtmc_vanilla_available": bool(
            trtmc_metadata.get("command") or native_trtmc_commands
        ),
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

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build per-sample disagreement evidence for TRTMC validation reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
from typing import Any, Callable, Mapping, Sequence


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
MAX_MEDIA_FILE_BYTES = 128 * 1024 * 1024
MAX_MEDIA_FILES_PER_SAMPLE = 16
MAX_MEDIA_BYTES_PER_SAMPLE = 256 * 1024 * 1024
MAX_MEDIA_SCAN_ENTRIES = 4096
MAX_MEDIA_SCAN_DEPTH = 64
MAX_DISAGREEMENT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_DISAGREEMENT_RECORDS = 1000
MAX_CASE_MEDIA_FILES = 1600
MAX_CASE_MEDIA_BYTES = 1024 * 1024 * 1024


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _validate_json_depth(
    value: Any,
    *,
    path: Path,
    maximum_depth: int = 256,
) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > maximum_depth:
            raise ValueError(
                f"JSON nesting exceeds {maximum_depth} in {path}"
            )
        if isinstance(current, float) and not math.isfinite(current):
            raise ValueError(f"non-finite JSON number in {path}")
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"JSON string is not valid UTF-8 in {path}"
                ) from exc
        if isinstance(current, Mapping):
            for key, item in current.items():
                if isinstance(key, str):
                    try:
                        key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise ValueError(
                            f"JSON object key is not valid UTF-8 in {path}"
                        ) from exc
                pending.append((item, depth + 1))
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth + 1) for item in current)


def _artifact_text(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None,
) -> str | None:
    if read_artifact is not None:
        return read_artifact(path)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _parse_json(text: str, *, path: Path) -> Any:
    data = json.loads(
        text,
        parse_constant=_reject_nonstandard_json_constant,
    )
    _validate_json_depth(data, path=path)
    return data


def _load_json(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None = None,
) -> dict[str, Any]:
    text = _artifact_text(path, read_artifact)
    if text is None:
        return {}
    data = _parse_json(text, path=path)
    return data if isinstance(data, dict) else {}


def _load_jsonl(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None = None,
) -> list[dict[str, Any]]:
    text = _artifact_text(path, read_artifact)
    if text is None:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _parse_json(line, path=path)
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


def _gate_metric(gate: str) -> tuple[str, str] | None:
    if gate.startswith("min_"):
        return gate.removeprefix("min_"), "min"
    if gate.startswith("max_"):
        return gate.removeprefix("max_"), "max"
    return None


def _numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _failed_summary_gates(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(summary.get("status", "") or "").lower() not in {"fail", "failed"}:
        return []
    gates = summary.get("gates")
    if not isinstance(gates, Mapping):
        return []
    failures = []
    for gate, threshold in gates.items():
        metric_and_direction = _gate_metric(str(gate))
        if metric_and_direction is None:
            continue
        metric, direction = metric_and_direction
        actual_number = _numeric(summary.get(metric))
        threshold_number = _numeric(threshold)
        if actual_number is None or threshold_number is None:
            continue
        failed = (
            actual_number < threshold_number
            if direction == "min"
            else actual_number > threshold_number
        )
        if failed:
            failures.append(
                {
                    "gate": str(gate),
                    "metric": metric,
                    "actual": summary.get(metric),
                    "threshold": threshold,
                    "direction": direction,
                }
            )
    return failures


def _summary_case_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    for collection_name in ("samples", "cases", "pairs"):
        collection = summary.get(collection_name)
        if isinstance(collection, list):
            return [dict(item) for item in collection if isinstance(item, dict)]
    return []


def _worst_gate_case(
    cases: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
) -> dict[str, Any] | None:
    metric = str(failure["metric"])
    candidates = [
        (case, _numeric(case.get(metric)))
        for case in cases
        if _numeric(case.get(metric)) is not None
    ]
    if not candidates:
        return dict(cases[0]) if cases else None
    selector = min if failure["direction"] == "min" else max
    return dict(selector(candidates, key=lambda item: item[1])[0])


def _summary_gate_disagreements(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases = _summary_case_rows(summary)
    selected: dict[str, dict[str, Any]] = {}
    for failure in _failed_summary_gates(summary):
        case = _worst_gate_case(cases, failure)
        if case is None:
            continue
        sample_id = _sample_id(case, f"sample-{len(selected)}")
        public_failure = {
            name: value
            for name, value in failure.items()
            if name != "direction"
        }
        if sample_id in selected:
            selected[sample_id]["failed_gates"].append(public_failure)
            continue
        selected[sample_id] = {
            **case,
            "sample_id": sample_id,
            "status": "failed",
            "reason": "summary_gate_failure",
            "failed_gates": [public_failure],
        }
    return list(selected.values())


def _explicit_disagreements(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = summary.get("disagreements")
    if not isinstance(explicit, list):
        return []
    selected = []
    for item in explicit:
        if not isinstance(item, dict):
            continue
        selected.append(item)
        if len(selected) > MAX_DISAGREEMENT_RECORDS:
            raise ValueError(
                "disagreement record count exceeds "
                f"{MAX_DISAGREEMENT_RECORDS}"
            )
    return selected


def _recorded_disagreements(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    for collection_name in ("samples", "cases", "pairs"):
        collection = summary.get(collection_name)
        if isinstance(collection, list):
            disagreements = []
            for item in collection:
                if not isinstance(item, dict) or not _record_is_disagreement(
                    item
                ):
                    continue
                disagreements.append(item)
                if len(disagreements) > MAX_DISAGREEMENT_RECORDS:
                    raise ValueError(
                        "disagreement record count exceeds "
                        f"{MAX_DISAGREEMENT_RECORDS}"
                    )
            if disagreements:
                return disagreements
    return []


def _summary_disagreements(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = _explicit_disagreements(summary)
    if explicit:
        return explicit
    recorded = _recorded_disagreements(summary)
    if recorded:
        return recorded
    return _summary_gate_disagreements(summary)


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

    def append(record: dict[str, Any]) -> None:
        expanded.append(record)
        if len(expanded) > MAX_DISAGREEMENT_RECORDS:
            raise ValueError(
                "disagreement record count exceeds "
                f"{MAX_DISAGREEMENT_RECORDS}"
            )

    for index, row in enumerate(rows):
        sample_id = _sample_id(row, f"sample-{index}")
        pair_id = str(row.get("pair_id", "") or "")
        if sample_id in prompts or not pair_id:
            append(dict(row))
            continue
        pair_samples = [
            prompt_id
            for prompt_id, prompt in prompts.items()
            if str(prompt.get("pair_id", "") or "") == pair_id
        ]
        if not pair_samples:
            append(dict(row))
            continue
        for prompt_id in pair_samples:
            append({**dict(row), "sample_id": prompt_id})
    return expanded


def _prediction_rows(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    data = _load_json(path, read_artifact)
    responses = data.get("responses", [])
    return _indexed_rows(responses) if isinstance(responses, list) else {}


def _answer_rows(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    data = _load_json(path, read_artifact)
    requests = data.get("requests", [])
    return _indexed_rows(requests) if isinstance(requests, list) else {}


def _native_trtmc_commands(
    path: Path,
    read_artifact: Callable[[Path], str | None] | None = None,
) -> dict[str, list[str]]:
    commands = {}
    text = _artifact_text(path, read_artifact)
    if text is None:
        return commands
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _parse_json(line, path=path)
        if not isinstance(row, Mapping):
            raise ValueError(
                f"native command rows must be objects: {path}"
            )
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or "\x00" in sample_id
        ):
            raise ValueError(
                "native command rows require a non-empty, NUL-free "
                f"sample_id: {path}"
            )
        command = row.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(
                not isinstance(token, str) or "\x00" in token
                for token in command
            )
            or not command[0].strip()
        ):
            raise ValueError(
                "native command rows require a non-empty list of "
                f"NUL-free string tokens: {path}"
            )
        commands[sample_id] = list(command)
    return commands


def _safe_sample_name(sample_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return value or "sample"


def _sample_directory_name(sample_id: str) -> str:
    readable = _safe_sample_name(sample_id)[:80]
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


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
    if command == []:
        return ""
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(token, str) or "\x00" in token
            for token in command
        )
        or not command[0].strip()
    ):
        raise ValueError(
            "native reproduction command must be a non-empty list of "
            "NUL-free string tokens"
        )
    tokens = [
        _replace_placeholders(token, replacements)
        for token in command
    ]
    if any("\x00" in token for token in tokens):
        raise ValueError(
            "native reproduction command contains a NUL character"
        )
    rendered = shlex.join(tokens)
    if any(name in rendered for name in _FORBIDDEN_REPRO_ENTRYPOINTS):
        return ""
    return rendered


def _write_trtmc_input(
    path: Path,
    prompt: Mapping[str, Any],
    write_artifact: Callable[[Path, str], None] | None = None,
) -> None:
    rendered = json.dumps(dict(prompt), ensure_ascii=False) + "\n"
    if write_artifact is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        write_artifact(path, rendered)


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
    require_single_link: bool,
    remaining_bytes: int,
    copy_artifact: Callable[[Path, Path, bool, int], int] | None = None,
) -> tuple[dict[str, str], int] | None:
    kind = _media_kind(source)
    maximum_bytes = min(MAX_MEDIA_FILE_BYTES, remaining_bytes)
    try:
        source_size = source.stat().st_size
    except OSError:
        return None
    if (
        not kind
        or not source.is_file()
        or source_size > maximum_bytes
    ):
        return None
    stem = _safe_sample_name(label).lower()
    target = media_dir / f"{ordinal:02d}-{stem}{source.suffix.lower()}"
    if copy_artifact is None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_bytes = target.stat().st_size
        if copied_bytes > maximum_bytes:
            target.unlink(missing_ok=True)
            return None
    else:
        copied_bytes = copy_artifact(
            source,
            target,
            require_single_link,
            maximum_bytes,
        )
    return (
        {
            "label": label,
            "kind": kind,
            "path": str(target.relative_to(case_dir)),
        },
        copied_bytes,
    )


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _media_candidates(
    label: str,
    record: Mapping[str, Any],
    *,
    trusted_output_root: Path | None = None,
    scan_artifacts: Callable[[Path, int], Sequence[Path]] | None = None,
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
        "visualization_path",
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
            for index, value in enumerate(
                images[:MAX_MEDIA_FILES_PER_SAMPLE]
            )
            if str(value)
        )
    candidates.extend(
        _frame_candidates(
            label,
            record.get("frames_dir"),
            trusted_output_root=trusted_output_root,
            scan_artifacts=scan_artifacts,
        )
    )
    return candidates


def _frame_candidates(
    label: str,
    value: Any,
    *,
    trusted_output_root: Path | None = None,
    scan_artifacts: Callable[[Path, int], Sequence[Path]] | None = None,
) -> list[tuple[str, Path]]:
    if not isinstance(value, str):
        return []
    try:
        root = Path(value).resolve(strict=True)
    except OSError:
        return []
    if (
        trusted_output_root is not None
        and not _is_below(root, trusted_output_root)
    ):
        return []
    if not root.is_dir():
        return []
    if scan_artifacts is None:
        paths = _scan_frame_artifacts(root, MAX_MEDIA_SCAN_ENTRIES)
    else:
        paths = scan_artifacts(root, MAX_MEDIA_SCAN_ENTRIES)
    image_paths = []
    video_paths = []
    for path in paths:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            image_paths.append(path)
        elif suffix in _VIDEO_SUFFIXES:
            video_paths.append(path)
    image_paths.sort()
    video_paths.sort()
    if len(image_paths) > 3:
        image_paths = [
            image_paths[0],
            image_paths[len(image_paths) // 2],
            image_paths[-1],
        ]
    images = [
        (f"{label} frame {index + 1}", path)
        for index, path in enumerate(image_paths)
    ]
    videos = [(f"{label} video", path) for path in video_paths]
    return images + videos


def _scan_frame_artifacts_from_fd(
    root: Path,
    root_fd: int,
    maximum_entries: int,
) -> list[Path]:
    """Stream a bounded directory tree through held, non-symlink descriptors."""
    discovered: list[Path] = []
    visited = 0

    def visit(directory_fd: int, relative: Path, depth: int) -> bool:
        nonlocal visited
        if depth > MAX_MEDIA_SCAN_DEPTH:
            return False
        with os.scandir(directory_fd) as entries:
            while visited < maximum_entries:
                try:
                    entry = next(entries)
                except StopIteration:
                    return True
                visited += 1
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth == MAX_MEDIA_SCAN_DEPTH:
                            continue
                        child_fd = os.open(
                            entry.name,
                            (
                                os.O_RDONLY
                                | os.O_DIRECTORY
                                | os.O_NOFOLLOW
                                | getattr(os, "O_CLOEXEC", 0)
                            ),
                            dir_fd=directory_fd,
                        )
                        try:
                            if not visit(
                                child_fd,
                                relative / entry.name,
                                depth + 1,
                            ):
                                return False
                        finally:
                            os.close(child_fd)
                    elif entry.is_file(follow_symlinks=False):
                        discovered.append(root / relative / entry.name)
                except FileNotFoundError:
                    continue
        return visited < maximum_entries

    visit(root_fd, Path(), 0)
    return discovered


def _scan_frame_artifacts(
    root: Path,
    maximum_entries: int,
) -> list[Path]:
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_fd = os.open(root, flags)
    except OSError:
        return []
    try:
        return _scan_frame_artifacts_from_fd(
            root,
            root_fd,
            maximum_entries,
        )
    except OSError:
        return []
    finally:
        os.close(root_fd)


def _collect_media(
    *,
    sample_dir: Path,
    case_dir: Path,
    prompt: Mapping[str, Any],
    reference_result: Mapping[str, Any],
    trtmc_result: Mapping[str, Any],
    copy_artifact: Callable[[Path, Path, bool, int], int] | None = None,
    trusted_output_root: Path | None = None,
    scan_artifacts: Callable[[Path, int], Sequence[Path]] | None = None,
    remaining_case_files: int = MAX_CASE_MEDIA_FILES,
    remaining_case_bytes: int = MAX_CASE_MEDIA_BYTES,
) -> tuple[list[dict[str, str]], int]:
    media_dir = sample_dir / "media"
    seen = set()
    copied = []
    copied_bytes = 0
    sources = (
        ("Input", prompt, None),
        ("Reference", reference_result, trusted_output_root),
        ("TRTMC", trtmc_result, trusted_output_root),
    )
    for label, record, allowed_root in sources:
        for candidate_label, source in _media_candidates(
            label,
            record,
            trusted_output_root=allowed_root,
            scan_artifacts=scan_artifacts,
        ):
            if len(copied) >= MAX_MEDIA_FILES_PER_SAMPLE:
                return copied, copied_bytes
            try:
                resolved = source.resolve(strict=True)
            except OSError:
                continue
            if allowed_root is not None and not _is_below(
                resolved,
                allowed_root,
            ):
                continue
            if resolved in seen:
                continue
            try:
                candidate_size = resolved.stat().st_size
            except OSError:
                continue
            if len(copied) >= remaining_case_files:
                raise ValueError(
                    "disagreement case media exceeds "
                    f"{MAX_CASE_MEDIA_FILES} files"
                )
            if candidate_size > remaining_case_bytes - copied_bytes:
                raise ValueError(
                    "disagreement case media exceeds "
                    f"{MAX_CASE_MEDIA_BYTES} bytes"
                )
            copied_item = _copy_media(
                source=resolved,
                media_dir=media_dir,
                case_dir=case_dir,
                label=candidate_label,
                ordinal=len(copied) + 1,
                require_single_link=allowed_root is not None,
                remaining_bytes=MAX_MEDIA_BYTES_PER_SAMPLE - copied_bytes,
                copy_artifact=copy_artifact,
            )
            if copied_item is not None:
                item, item_bytes = copied_item
                seen.add(resolved)
                copied.append(item)
                copied_bytes += item_bytes
    return copied, copied_bytes


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
    write_artifact: Callable[[Path, str], None] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    sample_dir = case_dir / "repro" / _sample_directory_name(sample_id)
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
        write_artifact=write_artifact,
    ):
        artifacts["reference_input"] = str(reference_input.relative_to(case_dir))
    if trtmc_command and prompt:
        _write_trtmc_input(input_path, prompt, write_artifact)
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
    if any(
        not isinstance(token, str) or "\x00" in token
        for token in native_command
    ) or not native_command[0].strip():
        raise ValueError(
            "native reproduction command must contain only NUL-free "
            "string tokens"
        )
    rendered = shlex.join(native_command)
    if any(name in rendered for name in _FORBIDDEN_REPRO_ENTRYPOINTS):
        return ""
    return rendered


def _write_elf_reference_input(
    path: Path,
    *,
    sample_id: str,
    prompt: Mapping[str, Any],
    enabled: bool,
    write_artifact: Callable[[Path, str], None] | None = None,
) -> bool:
    if not enabled:
        return False
    rendered = (
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
        + "\n"
    )
    if write_artifact is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    else:
        write_artifact(path, rendered)
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
    write_artifact: Callable[[Path, str], None] | None = None,
    copy_artifact: Callable[[Path, Path, bool, int], int] | None = None,
    read_artifact: Callable[[Path], str | None] | None = None,
    scan_artifacts: Callable[[Path, int], Sequence[Path]] | None = None,
    media_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    try:
        trusted_output_root = work_dir.resolve(strict=True)
    except OSError:
        trusted_output_root = work_dir.absolute()
    summary = _load_json(work_dir / "summary.json", read_artifact)
    prompts = _indexed_rows(
        _load_jsonl(work_dir / "prompts.jsonl", read_artifact)
    )
    comparison_rows = _expand_pair_disagreements(
        _summary_disagreements(summary),
        prompts,
    )
    if len(comparison_rows) > MAX_DISAGREEMENT_RECORDS:
        raise ValueError(
            "disagreement record count exceeds "
            f"{MAX_DISAGREEMENT_RECORDS}"
        )
    answers = _answer_rows(work_dir / "answers.json", read_artifact)
    reference_rows = _prediction_rows(
        work_dir / "hf_predictions.json",
        read_artifact,
    )
    trtmc_rows = _prediction_rows(
        work_dir / "trtfb_predictions.json",
        read_artifact,
    )
    reference_metadata = _load_json(
        work_dir / "hf_native_repro.json",
        read_artifact,
    )
    native_reference_commands = _native_trtmc_commands(
        work_dir / "hf_native_commands.jsonl",
        read_artifact,
    )
    trtmc_metadata = _load_json(
        work_dir / "trtfb_repro.json",
        read_artifact,
    )
    native_trtmc_commands = _native_trtmc_commands(
        work_dir / "trtfb_native_commands.jsonl",
        read_artifact,
    )
    rendered_records: list[str] = []
    artifact_bytes = 0
    artifact_path = case_dir / "disagreements.jsonl"
    case_media_files = 0
    case_media_bytes = 0
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
            write_artifact=write_artifact,
        )
        media, media_bytes = _collect_media(
            sample_dir=case_dir / "repro" / _sample_directory_name(sample_id),
            case_dir=case_dir,
            prompt=prompt,
            reference_result=reference_rows.get(sample_id, {}),
            trtmc_result=trtmc_rows.get(sample_id, {}),
            copy_artifact=copy_artifact,
            trusted_output_root=trusted_output_root,
            scan_artifacts=scan_artifacts,
            remaining_case_files=(
                min(
                    MAX_CASE_MEDIA_FILES - case_media_files,
                    media_budget.get("files", MAX_CASE_MEDIA_FILES)
                    if media_budget is not None
                    else MAX_CASE_MEDIA_FILES,
                )
            ),
            remaining_case_bytes=(
                min(
                    MAX_CASE_MEDIA_BYTES - case_media_bytes,
                    media_budget.get("bytes", MAX_CASE_MEDIA_BYTES)
                    if media_budget is not None
                    else MAX_CASE_MEDIA_BYTES,
                )
            ),
        )
        case_media_files += len(media)
        case_media_bytes += media_bytes
        if media_budget is not None:
            media_budget["files"] -= len(media)
            media_budget["bytes"] -= media_bytes
        if media:
            artifacts["media"] = media
        record = {
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
        _validate_json_depth(record, path=artifact_path)
        rendered_record = (
            json.dumps(record, ensure_ascii=False) + "\n"
        )
        artifact_bytes += len(rendered_record.encode("utf-8"))
        if artifact_bytes > MAX_DISAGREEMENT_ARTIFACT_BYTES:
            raise ValueError(
                "disagreement artifact exceeds "
                f"{MAX_DISAGREEMENT_ARTIFACT_BYTES} bytes: "
                f"{artifact_path}"
            )
        rendered_records.append(rendered_record)
    artifact_text = "".join(rendered_records)
    if write_artifact is None:
        artifact_path.write_text(artifact_text, encoding="utf-8")
    else:
        write_artifact(artifact_path, artifact_text)
    return {
        "count": len(rendered_records),
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
    read_artifact: Callable[[Path], str | None] | None = None,
    expected_count: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    record_count = 0
    text = _artifact_text(path, read_artifact)
    if text is None:
        if expected_count not in (None, 0):
            raise ValueError(
                f"disagreement artifact count is 0, expected {expected_count}: "
                f"{path}"
            )
        return rows
    for line in text.splitlines():
        if not line.strip():
            continue
        row = _parse_json(line, path=path)
        if not isinstance(row, dict):
            raise ValueError(
                f"disagreement artifact rows must be objects: {path}"
            )
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"disagreement artifact row has invalid schema_version: "
                f"{path}"
            )
        sample_id = row.get("sample_id")
        if (
            not isinstance(sample_id, str)
            or not sample_id.strip()
            or "\x00" in sample_id
        ):
            raise ValueError(
                f"disagreement artifact row requires sample_id: {path}"
            )
        reason = row.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or "\x00" in reason
        ):
            raise ValueError(
                f"disagreement artifact row requires reason: {path}"
            )
        for field in (
            "input",
            "reference_result",
            "trtmc_result",
            "comparison",
            "reproduce",
            "artifacts",
        ):
            if not isinstance(row.get(field), Mapping):
                raise ValueError(
                    "disagreement artifact row field "
                    f"{field} must be an object: {path}"
                )
        reproduce = row["reproduce"]
        for field in ("reference", "trtmc"):
            command = reproduce.get(field, "")
            if (
                not isinstance(command, str)
                or "\x00" in command
                or (command != "" and not command.strip())
            ):
                raise ValueError(
                    "disagreement artifact row reproduce."
                    f"{field} must be empty or a non-whitespace, "
                    f"NUL-free string: {path}"
                )
        record_count += 1
        if record_count > MAX_DISAGREEMENT_RECORDS:
            raise ValueError(
                "disagreement artifact record count exceeds "
                f"{MAX_DISAGREEMENT_RECORDS}: {path}"
            )
        if len(rows) < limit:
            rows.append(row)
    if expected_count is not None and record_count != expected_count:
        raise ValueError(
            f"disagreement artifact count is {record_count}, "
            f"expected {expected_count}: {path}"
        )
    return rows

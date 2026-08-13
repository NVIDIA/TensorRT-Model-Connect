#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a self-contained HTML report from E2E test artifacts.

Reads result.json files produced by the unified E2E harness and assembles
a single HTML page with summary dashboard, per-model details, embedded
media (images, audio, video frames), and reproduction commands.

Usage:
    python scripts/generate_e2e_report.py \\
      --artifacts-dir /tmp/e2e_artifacts/artifacts \\
      -o /tmp/e2e_artifacts/e2e_report.html \\
      [--manifest-dir tests/e2e/models] \\
      [--project-dir .] \\
      [--title "E2E Report"]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from reporting.vlm_assessment import (
    render_diffusion_vlm_assessment as _render_diffusion_vlm_assessment,
)

# Maximum file size to embed inline (10 MB).  A value of zero disables the
# limit.  Model-proof CI raises this to a bounded 32 MiB per evidence file so
# its portable report can retain normal audio/video artifacts without allowing
# an unbounded, PR-controlled HTML artifact.
_MAX_EMBED_BYTES = 10 * 1024 * 1024

# Number of evenly-spaced frames to embed for diffusion models.
_MAX_DIFFUSION_FRAMES = 6

_TRTMC_TIMING_RE = re.compile(
    r"^\[trtmc\.timing\]\s+"
    r"prefill_ms=(?P<prefill_ms>[-+0-9.eE]+)\s+"
    r"decode_ms=(?P<decode_ms>[-+0-9.eE]+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+)\s*$",
    re.MULTILINE,
)
_TRTMC_LOAD_TIMING_RE = re.compile(
    r"^\[trtmc\.load_timing\]\s+.*?label=\"(?P<label>[^\"]+)\".*?"
    r"load_deserialize_ms=(?P<ms>[-+0-9.eE]+)"
    r"(?:\s+plan_bytes=(?P<plan_bytes>[0-9]+))?",
    re.MULTILINE,
)
_TRTMC_ENGINE_TIMING_RE = re.compile(
    r"^\[trtmc\.engine_timing\]\s+.*?label=\"(?P<label>[^\"]+)\".*?"
    r"execute_ms=(?P<ms>[-+0-9.eE]+)",
    re.MULTILINE,
)
_TEST_CASE_RE = re.compile(r"(?:test_e2e|test_model_e2e)\[([^\]]+)\]")
_CONSOLE_OUTCOME_RE = re.compile(
    r"(?:tests/test_e2e\.py::test_e2e|"
    r"tests/e2e/models/[^\s:]+::test_model_e2e)\[([^\]]+)\]\s+"
    r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b(.*)"
)
_PYTEST_TO_RESULT_STATUS = {
    "PASSED": "pass",
    "XPASS": "pass",
    "FAILED": "fail",
    "ERROR": "error",
    "SKIPPED": "skip",
    "XFAIL": "skip",
}

# ---------------------------------------------------------------------------
# Modality classification
# ---------------------------------------------------------------------------

_TASK_STRATEGY_TO_MODALITY = {
    "text_generation_causal": "text",
    "diffusion_text_generation": "diffusion_text",
    "vision_language_generation": "vl",
    "diffusion_media_generation": "diffusion",
    "text_to_audio": "audio",
    "speech_to_text": "audio",
    "speech_to_speech": "audio",
    "segmentation": "segmentation",
    "prompted_segmentation": "segmentation",
    "image_classification": "classification",
    "encoder_only_nlp": "numeric",
    "embedding": "numeric",
    "reranking": "reranking",
    "neural_operator": "neural_operator",
    "omni_multimodal": "omni",
    # Kept explicit for forward-compatible manifests.  Unknown strategies
    # still receive the structured TRT/reference fallback renderer.
    "object_detection": "detection",
}


def classify_modality(result: Dict[str, Any]) -> str:
    """Return a modality string for the given result dict."""
    ts = (result.get("case_config") or {}).get("task_strategy", "")
    return _TASK_STRATEGY_TO_MODALITY.get(ts, "generic")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Walk *artifacts_dir* and parse every ``result.json`` found."""
    results: List[Dict[str, Any]] = []
    if not artifacts_dir.is_dir():
        return results
    for p in sorted(artifacts_dir.iterdir()):
        rj = p / "result.json"
        if rj.is_file():
            try:
                data = json.loads(rj.read_text(encoding="utf-8"))
                # Stash the directory so renderers can find artifacts.
                data["_artifact_dir"] = str(p)
                results.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"WARNING: skipping {rj}: {exc}", file=sys.stderr)
    results = _merge_pytest_outcomes(
        results,
        _load_pytest_outcomes(_e2e_root_from_artifacts_dir(artifacts_dir)),
    )
    _attach_diffusion_vlm_assessments(results, artifacts_dir)
    return results


def _indexed_manifest_paths(models_dir: Path) -> List[Path]:
    """Return JSON manifests declared by per-family MODEL.toml indexes."""
    paths: List[Path] = []
    for index_path in sorted(models_dir.glob("*/MODEL.toml")):
        try:
            with index_path.open("rb") as index_file:
                index = tomllib.load(index_file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"WARNING: skipping {index_path}: {exc}", file=sys.stderr)
            continue
        entries = index.get("test_manifests", [])
        if not isinstance(entries, list):
            print(
                f"WARNING: skipping {index_path}: test_manifests is not a list",
                file=sys.stderr,
            )
            continue
        for entry in entries:
            if not isinstance(entry, str):
                continue
            manifest_path = index_path.parent / entry
            if manifest_path.is_file():
                paths.append(manifest_path)
            else:
                print(
                    f"WARNING: {index_path} references missing manifest {entry!r}",
                    file=sys.stderr,
                )
    return paths


def load_model_manifests(models_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Load every model and its declared testcase inventory."""
    if models_dir is None or not models_dir.is_dir():
        return []

    models: List[Dict[str, Any]] = []
    for manifest_path in _indexed_manifest_paths(models_dir):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: skipping {manifest_path}: {exc}", file=sys.stderr)
            continue
        testcases = raw.get("testcases", []) if isinstance(raw, dict) else []
        if not isinstance(testcases, list) or not testcases:
            continue

        model_name = str(raw.get("name") or manifest_path.stem)
        family = str(raw.get("family") or "")
        task_strategy = str(raw.get("task_strategy") or "")
        cases: List[Dict[str, str]] = []
        for testcase in testcases:
            if not isinstance(testcase, dict) or not testcase.get("name"):
                continue
            cases.append(
                {
                    "name": str(testcase["name"]),
                    "ci_tier": str(
                        testcase.get("ci_tier") or raw.get("ci_tier") or "default"
                    ),
                    "task_strategy": str(
                        testcase.get("task_strategy") or task_strategy
                    ),
                }
            )
        if not cases:
            continue
        models.append(
            {
                "name": model_name,
                "family": family,
                "bundle": str(raw.get("bundle") or ""),
                "testcases": cases,
            }
        )
    return sorted(models, key=lambda model: str(model["name"]))


def _e2e_root_from_artifacts_dir(artifacts_dir: Path) -> Path:
    if artifacts_dir.name == "artifacts":
        return artifacts_dir.parent
    return artifacts_dir


def _extract_case_name(text: str) -> str:
    match = _TEST_CASE_RE.search(text)
    return match.group(1) if match else ""


def _record_pytest_outcome(
    outcomes: Dict[str, Dict[str, str]],
    case_name: str,
    outcome: Dict[str, str],
) -> None:
    outcomes[case_name] = outcome


def _result_model_name(result: Dict[str, Any]) -> str:
    case_config = result.get("case_config", {}) or {}
    metadata = case_config.get("metadata", {}) or {}
    return str(metadata.get("model_name") or result.get("case_name") or "")


def _clean_pytest_reason(reason: str) -> str:
    text = reason.strip()
    while text.startswith("(") and text.endswith(")") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _junit_files(e2e_root: Path) -> List[Path]:
    worker_files = sorted(e2e_root.glob("junit-gpu*.xml"))
    if worker_files:
        return worker_files
    merged = e2e_root / "junit.xml"
    return [merged] if merged.is_file() else []


def _load_pytest_outcomes(e2e_root: Path) -> Dict[str, Dict[str, str]]:
    outcomes: Dict[str, Dict[str, str]] = {}

    for xml_path in _junit_files(e2e_root):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"WARNING: skipping {xml_path}: {exc}", file=sys.stderr)
            continue
        for testcase in root.iter("testcase"):
            case_name = _extract_case_name(
                " ".join(str(testcase.attrib.get(key, "")) for key in ("classname", "name"))
            )
            if not case_name:
                continue
            status = "PASSED"
            reason = ""
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")
            if error is not None:
                status = "ERROR"
                reason = error.attrib.get("message", "") or (error.text or "")
            elif failure is not None:
                status = "FAILED"
                reason = failure.attrib.get("message", "") or (failure.text or "")
            elif skipped is not None:
                skip_type = skipped.attrib.get("type", "")
                status = "XFAIL" if skip_type == "pytest.xfail" else "SKIPPED"
                reason = skipped.attrib.get("message", "") or (skipped.text or "")
            _record_pytest_outcome(
                outcomes,
                case_name,
                {
                    "pytest_status": status,
                    "reason": _clean_pytest_reason(reason),
                    "source": xml_path.name,
                },
            )

    for log_path in sorted(e2e_root.glob("console-*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"WARNING: skipping {log_path}: {exc}", file=sys.stderr)
            continue
        for line in lines:
            match = _CONSOLE_OUTCOME_RE.search(line)
            if not match:
                continue
            case_name, status, rest = match.groups()
            if status not in {"XFAIL", "XPASS"} and case_name not in outcomes:
                continue
            _record_pytest_outcome(
                outcomes,
                case_name,
                {
                    "pytest_status": status,
                    "reason": _clean_pytest_reason(rest.split("[", 1)[0]),
                    "source": log_path.name,
                },
            )

    return outcomes


def _merge_pytest_outcomes(
    results: List[Dict[str, Any]],
    outcomes: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        item = dict(result)
        case_name = str(result.get("case_name") or "")
        if case_name:
            seen.add(case_name)
        outcome = outcomes.get(case_name)
        if outcome:
            item["_pytest_outcome"] = outcome
        status = outcome.get("pytest_status", "") if outcome else ""
        if status in _PYTEST_TO_RESULT_STATUS:
            item["_raw_status"] = item.get("status")
            item["status"] = _PYTEST_TO_RESULT_STATUS[status]
        merged.append(item)

    # Some model-owned pytest entrypoints execute a manifest whose individual
    # testcases each emit result.json.  The JUnit node is named after the
    # parent model, not a fifth testcase.  Attach that model-level outcome to
    # its children instead of synthesizing a schema-less passing result that
    # would fail strict evidence validation as an "unknown strategy".
    grouped_model_names = {_result_model_name(item) for item in merged}
    for case_name, outcome in sorted(outcomes.items()):
        if case_name in seen:
            continue
        if case_name in grouped_model_names:
            for item in merged:
                if _result_model_name(item) == case_name:
                    item["_pytest_model_outcome"] = outcome
            continue
        status = _PYTEST_TO_RESULT_STATUS.get(outcome.get("pytest_status", ""), "error")
        merged.append(
            {
                "case_name": case_name,
                "status": status,
                "failure_type": "pytest_failed" if status in {"fail", "error"} else None,
                "case_config": {},
                "stages": {
                    "pytest": {
                        "status": status,
                        "message": outcome.get("reason", ""),
                        "metrics": {},
                    }
                },
                "timing": {},
                "_summary_only": True,
                "_pytest_outcome": outcome,
            }
        )
    return merged


def _load_diffusion_vlm_assessments(artifacts_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load optional paired-VLM diffusion assessments keyed by case name."""
    candidates = [
        artifacts_dir.parent / "diffusion_vlm_assessment.json",
        artifacts_dir / "diffusion_vlm_assessment.json",
        artifacts_dir.parent / "diffusion_vlm_similarity.json",
        artifacts_dir / "diffusion_vlm_similarity.json",
    ]
    assessment_path = next((p for p in candidates if p.is_file()), None)
    if assessment_path is None:
        return {}

    try:
        payload = json.loads(assessment_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: skipping {assessment_path}: {exc}", file=sys.stderr)
        return {}

    model_id = payload.get("model_id", "")
    provenance = {
        "source_revision": payload.get("source_revision"),
        "workflow_run_id": payload.get("workflow_run_id"),
        "workflow_run_attempt": payload.get("workflow_run_attempt"),
        "coverage_complete": payload.get("coverage_complete"),
        "expected_case_names": payload.get("expected_case_names"),
        "assessed_case_names": payload.get("assessed_case_names"),
    }
    by_case: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        case_name = str(item.get("case_name") or "")
        if not case_name:
            continue
        enriched = dict(item)
        enriched.setdefault("model_id", model_id)
        enriched.setdefault("_assessment_path", str(assessment_path))
        enriched["_assessment_provenance"] = provenance
        by_case[case_name] = enriched
    return by_case


def _attach_diffusion_vlm_assessments(
    results: List[Dict[str, Any]],
    artifacts_dir: Path,
) -> None:
    assessments = _load_diffusion_vlm_assessments(artifacts_dir)
    if not assessments:
        return
    for result in results:
        case_name = str(result.get("case_name") or "")
        if case_name in assessments:
            result["vlm_assessment"] = assessments[case_name]


# ---------------------------------------------------------------------------
# File embedding helpers
# ---------------------------------------------------------------------------


def encode_file_base64(path: Path, mime: str) -> Optional[str]:
    """Return a ``data:`` URI for *path*, or ``None`` if too large / missing."""
    if not _valid_media_file(path):
        return None
    if _MAX_EMBED_BYTES > 0 and path.stat().st_size > _MAX_EMBED_BYTES:
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _valid_media_file(path: Path) -> bool:
    """Reject empty/corrupt media before claiming it is embedded evidence."""
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
        if size <= 0:
            return False
        with path.open("rb") as stream:
            header = stream.read(16)
            trailer = b""
            if size >= 2:
                stream.seek(-2, 2)
                trailer = stream.read(2)
    except OSError:
        return False
    ext = path.suffix.lower()
    validators = {
        ".png": header.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": header.startswith(b"\xff\xd8") and trailer == b"\xff\xd9",
        ".jpeg": header.startswith(b"\xff\xd8") and trailer == b"\xff\xd9",
        ".gif": header.startswith((b"GIF87a", b"GIF89a")),
        ".webp": header.startswith(b"RIFF") and header[8:12] == b"WEBP",
        ".wav": header.startswith(b"RIFF") and header[8:12] == b"WAVE",
        ".flac": header.startswith(b"fLaC"),
        ".ogg": header.startswith(b"OggS"),
        ".mp3": header.startswith(b"ID3") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        ),
        ".mp4": len(header) >= 12 and header[4:8] == b"ftyp",
        ".webm": header.startswith(b"\x1aE\xdf\xa3"),
    }
    return validators.get(ext, False)


def _mime_for_ext(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".gif": "image/gif",
    }.get(ext.lower(), "application/octet-stream")


def _path_within(path: Path, root: Path) -> Optional[Path]:
    """Resolve *path* and return it only when it is a regular file below *root*.

    E2E result files are produced by code from the pull request.  Treat their
    path fields as untrusted: without this boundary a crafted result could
    cause the report to embed an HF cache file or another host-mounted file.
    Resolving both paths also rejects symlink escapes.
    """
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _resolve_input_media(ref: Any, project_dir: Optional[Path]) -> Optional[Path]:
    """Resolve a manifest-owned input file below the checked-out source tree.

    Isolated model proofs run with their positive source projection mounted at
    ``/src``. Their result JSON therefore records manifest inputs as absolute
    ``/src/...`` paths. A later combined-report job checks out that revision at
    another path. Rebase only that exact isolated-source prefix; all other
    absolute paths remain invalid. ``_path_within`` rejects both traversal and
    symlink escapes after rebasing.
    """
    if not isinstance(ref, (str, Path)) or not str(ref) or project_dir is None:
        return None
    raw = Path(str(ref))
    if not raw.is_absolute():
        return _path_within(project_dir / raw, project_dir)

    direct = _path_within(raw, project_dir)
    if direct is not None:
        return direct

    try:
        isolated_relative = raw.relative_to(Path("/src"))
    except ValueError:
        return None
    return _path_within(project_dir / isolated_relative, project_dir)


def _resolve_artifact_media(ref: Any, art_dir: Path) -> Optional[Path]:
    """Resolve a generated artifact below the E2E artifact root."""
    if not isinstance(ref, (str, Path)) or not str(ref):
        return None
    raw = Path(str(ref))
    candidate = raw if raw.is_absolute() else art_dir / raw
    return _path_within(candidate, art_dir)


def _first_existing_media(
    refs: Any,
    art_dir: Path,
) -> Optional[Path]:
    values = refs if isinstance(refs, list) else [refs]
    for ref in values:
        path = _resolve_artifact_media(ref, art_dir)
        if path is not None:
            return path
    return None


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "pass": "#22c55e",
    "fail": "#ef4444",
    "skip": "#eab308",
    "error": "#f97316",
    "passed": "#22c55e",
    "failed": "#ef4444",
    "skipped": "#eab308",
}


def _status_label(status: str) -> str:
    return status.replace("_", " ").upper()


def _badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#6b7280")
    label = _status_label(status)
    return f'<span class="badge" style="background:{color}">{html.escape(label)}</span>'


def _esc(text: Any) -> str:
    return html.escape(str(text)) if text is not None else ""


def _code_block(text: str, block_id: str) -> str:
    """Render a dark code block with a copy button."""
    return (
        f'<div class="code-wrap">'
        f'<button class="copy-btn" onclick="copyCmd(\'{block_id}\')">Copy</button>'
        f'<pre id="{block_id}"><code>{_esc(text)}</code></pre>'
        f"</div>"
    )


def _select_frames(frame_paths: List[Path], max_frames: int) -> List[Path]:
    """Pick *max_frames* evenly-spaced frames from *frame_paths*."""
    n = len(frame_paths)
    if n <= max_frames:
        return frame_paths
    indices = [int(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frame_paths[i] for i in indices]


def _list_frames_in_dir(dir_path: Path) -> List[Path]:
    """Return sorted frame/image files from *dir_path*."""
    frame_paths = sorted(dir_path.glob("frame_*.png"))
    if frame_paths:
        return frame_paths
    # Backward compatibility for runs that save a single non-frame_*.png image.
    ext_globs = ("*.png", "*.jpg", "*.jpeg")
    files: List[Path] = []
    for pattern in ext_globs:
        files.extend(sorted(dir_path.glob(pattern)))
    return files


def _directory_within(path: Path, root: Path) -> Optional[Path]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_dir() else None


def _resolve_frame_paths(
    frame_refs: Any,
    art_dir: Path,
    fallback_dir_name: str,
) -> List[Path]:
    """Resolve frame refs (file(s) or directory path(s)) into concrete files."""
    refs: List[str] = []
    if isinstance(frame_refs, str):
        refs = [frame_refs]
    elif isinstance(frame_refs, list):
        refs = [str(ref) for ref in frame_refs if ref]

    frame_paths: List[Path] = []
    artifact_root = art_dir
    for ref in refs:
        raw = Path(ref)
        candidate = raw if raw.is_absolute() else art_dir / raw
        safe_dir = _directory_within(candidate, artifact_root)
        if safe_dir is not None:
            for child in _list_frames_in_dir(safe_dir):
                safe_child = _path_within(child, artifact_root)
                if safe_child is not None and safe_child.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".webp", ".gif"
                }:
                    frame_paths.append(safe_child)
            continue
        safe_file = _path_within(candidate, artifact_root)
        if safe_file is not None and safe_file.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif"
        }:
            frame_paths.append(safe_file)

    if not frame_paths:
        fallback_dir = _directory_within(art_dir / fallback_dir_name, artifact_root)
        if fallback_dir is not None:
            for child in _list_frames_in_dir(fallback_dir):
                safe_child = _path_within(child, artifact_root)
                if safe_child is not None and safe_child.suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".webp", ".gif"
                }:
                    frame_paths.append(safe_child)

    # De-duplicate while preserving order.
    deduped: List[Path] = []
    seen: set[str] = set()
    for fp in frame_paths:
        key = str(fp)
        if key not in seen:
            seen.add(key)
            deduped.append(fp)
    return deduped


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------


def _render_metrics_table(stages: Dict[str, Any]) -> str:
    """Render a table of all metrics across all stages."""
    rows: List[str] = []
    for stage_name, stage_data in stages.items():
        metrics = stage_data.get("metrics", {})
        for metric_name, m in metrics.items():
            if isinstance(m, dict):
                value = m.get("value", "")
                threshold = m.get("threshold")
                operator = m.get("operator", "")
                passed = m.get("passed", True)
                note = m.get("note", "")
            else:
                value = m
                threshold = None
                operator = ""
                passed = True
                note = ""
            icon = "&#10003;" if passed else "&#10007;"
            icon_cls = "pass-icon" if passed else "fail-icon"
            row_cls = "metric-pass" if passed else "metric-fail"
            thr_str = f"{threshold}" if threshold is not None else "&mdash;"
            rows.append(
                f"<tr class='{row_cls}'>"
                f"<td>{_esc(stage_name)}</td>"
                f"<td>{_esc(metric_name)}</td>"
                f"<td>{_format_value(value)}</td>"
                f"<td>{thr_str}</td>"
                f"<td>{_esc(operator)}</td>"
                f"<td class='{icon_cls}'>{icon}</td>"
                f"<td>{_esc(note)}</td>"
                f"</tr>"
            )
    if not rows:
        return "<p><em>No metrics available.</em></p>"
    return (
        '<table class="metrics-table">'
        "<thead><tr>"
        "<th>Stage</th><th>Metric</th><th>Value</th>"
        "<th>Threshold</th><th>Op</th><th>Pass</th><th>Note</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) < 0.001 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return _esc(v)


# ---------------------------------------------------------------------------
# Timing table
# ---------------------------------------------------------------------------


def _render_timing_table(timing: Dict[str, float]) -> str:
    if not timing:
        return ""
    rows = []
    total = 0.0
    for phase, secs in timing.items():
        s = float(secs) if secs is not None else 0.0
        total += s
        rows.append(f"<tr><td>{_esc(phase)}</td><td>{s:.2f}s</td></tr>")
    rows.append(f"<tr class='total-row'><td>Total</td><td>{total:.2f}s</td></tr>")
    return (
        '<table class="timing-table">'
        "<thead><tr><th>Phase</th><th>Time</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _sum_timing_prefix(
    timing: Dict[str, Any],
    prefixes: Tuple[str, ...],
    exclude: Tuple[str, ...] = (),
    exclude_prefixes: Tuple[str, ...] = (),
) -> float:
    total = 0.0
    for key, value in timing.items():
        if key in exclude or not any(key.startswith(prefix) for prefix in prefixes):
            continue
        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if value is None:
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _timing_label_key(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return cleaned or "engine"


def _read_stage_log(ref: Any, art_dir: Path) -> str:
    if not isinstance(ref, str) or not ref:
        return ""
    path = _resolve_artifact_media(ref, art_dir)
    if path is not None:
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""
    return ""


def _stage_text_blobs(result: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    art_dir = Path(result.get("_artifact_dir") or ".")
    stage_blobs: List[Tuple[str, List[str]]] = []
    for stage_key, stage in (result.get("stage_outputs") or {}).items():
        if not str(stage_key).startswith("trt_") or not isinstance(stage, dict):
            continue
        stage_name = str(stage.get("stage_name") or str(stage_key).removeprefix("trt_"))
        blobs: List[str] = []
        seen_blobs: set[str] = set()

        def append_blob(text: str) -> None:
            if not text:
                return
            key = text if len(text) < 10000 else f"{len(text)}:{text[:2000]}:{text[-2000:]}"
            if key in seen_blobs:
                return
            seen_blobs.add(key)
            blobs.append(text)

        def visit(value: Any) -> None:
            if isinstance(value, str):
                append_blob(value)
            elif isinstance(value, dict):
                log_bases: set[str] = set()
                for key, child in value.items():
                    if key.endswith("_log") or key == "stderr_log":
                        log_text = _read_stage_log(child, art_dir)
                        if log_text:
                            append_blob(log_text)
                        log_bases.add("stderr" if key == "stderr_log" else key[:-4])
                for key, child in value.items():
                    if key.endswith("_log") or key == "stderr_log":
                        continue
                    if key in log_bases or (key == "stderr_truncated" and "stderr" in log_bases):
                        continue
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(stage.get("metadata") or {})
        visit(stage.get("data") or {})
        if stage.get("text"):
            blobs.append(str(stage["text"]))
        stage_blobs.append((stage_name, blobs))
    return stage_blobs


def _extract_labeled_timing(
    pattern: re.Pattern[str],
    blobs: List[str],
) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    for text in blobs:
        for match in pattern.finditer(text or ""):
            try:
                seconds = float(match.group("ms")) / 1000.0
            except (TypeError, ValueError):
                continue
            key = _timing_label_key(match.group("label"))
            timings[key] = timings.get(key, 0.0) + seconds
    return timings


def _extract_labeled_load_stats(blobs: List[str]) -> Dict[str, Tuple[int, int]]:
    stats: Dict[str, Tuple[int, int]] = {}
    for text in blobs:
        for match in _TRTMC_LOAD_TIMING_RE.finditer(text or ""):
            key = _timing_label_key(match.group("label"))
            try:
                plan_bytes = int(match.group("plan_bytes") or 0)
            except (TypeError, ValueError):
                plan_bytes = 0
            count, total_bytes = stats.get(key, (0, 0))
            stats[key] = (count + 1, total_bytes + plan_bytes)
    return stats


def _collect_load_component_stats(result: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    grouped: Dict[str, Tuple[int, int]] = {}
    for stage_name, blobs in _stage_text_blobs(result):
        for label, (count, plan_bytes) in _extract_labeled_load_stats(blobs).items():
            key = f"trt_component_load_deserialize_{stage_name}_{label}_s"
            component = _format_component_only_label("trt_component_load_deserialize_", key)
            old_count, old_bytes = grouped.get(component, (0, 0))
            grouped[component] = (old_count + count, old_bytes + plan_bytes)
    return grouped


def _extract_cli_generation_timing(blobs: List[str]) -> float | None:
    for text in blobs:
        match = _TRTMC_TIMING_RE.search(text or "")
        if match is None:
            continue
        try:
            return float(match.group("total_ms")) / 1000.0
        except (TypeError, ValueError):
            continue
    return None


def _augment_timing_from_stage_outputs(result: Dict[str, Any], timing: Dict[str, Any]) -> None:
    for stage_name, blobs in _stage_text_blobs(result):
        engine_components = _extract_labeled_timing(_TRTMC_ENGINE_TIMING_RE, blobs)
        if engine_components:
            timing[f"trt_engine_{stage_name}_s"] = sum(engine_components.values())
            for label, value in engine_components.items():
                timing[f"trt_component_engine_{stage_name}_{label}_s"] = value
        else:
            cli_engine = _extract_cli_generation_timing(blobs)
            if cli_engine is not None:
                timing[f"trt_engine_{stage_name}_s"] = cli_engine

        load_components = _extract_labeled_timing(_TRTMC_LOAD_TIMING_RE, blobs)
        if load_components:
            timing[f"trt_load_deserialize_{stage_name}_s"] = sum(load_components.values())
            for label, value in load_components.items():
                timing[f"trt_component_load_deserialize_{stage_name}_{label}_s"] = value


def _normalize_detailed_timing(result: Dict[str, Any]) -> Dict[str, float]:
    """Return normalized timing categories for the per-model timing table."""
    timing = dict(result.get("timing", {}) or {})
    _augment_timing_from_stage_outputs(result, timing)
    details: Dict[str, float] = {}
    persisted_inference: float | None = None

    persisted = result.get("detailed_timing", {}) or {}
    for key, value in persisted.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if key == "inference_s":
            persisted_inference = numeric
        else:
            details[key] = numeric

    inference = _sum_timing_prefix(timing, ("trt_engine_",))
    trt_validation = _sum_timing_prefix(
        timing,
        ("trt_",),
        exclude=("trt_compile_s", "trt_build_s"),
        exclude_prefixes=("trt_engine_", "trt_load_deserialize_", "trt_component_"),
    )

    if inference:
        details["inference_s"] = inference
    elif persisted_inference is not None and not trt_validation:
        details["inference_s"] = persisted_inference

    load_deserialize = _sum_timing_prefix(timing, ("trt_load_deserialize_",))
    if load_deserialize:
        details["trt_load_deserialization_s"] = load_deserialize

    for key, value in timing.items():
        if not key.startswith(("trt_component_engine_", "trt_component_load_deserialize_")):
            continue
        if value is None:
            continue
        try:
            details[key] = float(value)
        except (TypeError, ValueError):
            continue

    if trt_validation:
        details.setdefault("trt_validation_s", trt_validation)

    reference = _sum_timing_prefix(timing, ("ref_",))
    if reference:
        details.setdefault("reference_s", reference)

    comparison = _sum_timing_prefix(timing, ("compare_", "contract_"))
    if comparison:
        details.setdefault("comparison_s", comparison)

    preflight = timing.get("preflight_s")
    if preflight is not None:
        try:
            details.setdefault("preflight_s", float(preflight))
        except (TypeError, ValueError):
            pass

    return details


def _format_seconds(value: Any) -> str:
    if value is None:
        return "&mdash;"
    try:
        seconds = float(value)
        if 0.0 < abs(seconds) < 0.01:
            return f"{seconds * 1000.0:.2f}ms"
        return f"{seconds:.2f}s"
    except (TypeError, ValueError):
        return "&mdash;"


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _format_component_weight_label(key: str) -> str:
    component = key.removeprefix("weights_loading_").removesuffix("_s")
    return component.replace("_", " ")


def _format_component_compile_label(key: str) -> str:
    component = key.removeprefix("trt_compile_").removesuffix("_s")
    component = component.removeprefix("extra_")
    return component.replace("_", " ")


def _split_stage_component_label(prefix: str, key: str) -> Tuple[str, str]:
    stem = key.removeprefix(prefix).removesuffix("_s")
    parts = stem.split("_")
    stage = "unknown"
    label_parts: list[str] = []
    for idx in range(len(parts), 0, -1):
        candidate = "_".join(parts[:idx])
        if candidate in {
            "audio_encode",
            "debug_pipeline",
            "full_generation",
            "full_inference",
            "end_to_end",
            "end_to_end_video",
            "frame_quality",
            "generate",
            "denoising_step",
            "text_encode",
            "talker_decode",
            "thinker_decode",
            "vae_decode",
            "vision_encode",
            "prefill",
            "decode",
            "generate_audio",
            "speech_to_text",
            "speech_to_speech",
        }:
            stage = candidate
            label_parts = parts[idx:]
            break
    if not label_parts:
        label_parts = parts
    component = " ".join(label_parts).replace(" plan", "")
    stage_label = stage.replace("_", " ")
    return stage_label, component


def _format_stage_component_label(prefix: str, key: str) -> str:
    stage_label, component = _split_stage_component_label(prefix, key)
    return f"{component} ({stage_label})" if component else stage_label


def _format_component_only_label(prefix: str, key: str) -> str:
    _, component = _split_stage_component_label(prefix, key)
    return component or "engine"


def _aggregate_component_timings(
    details: Dict[str, float],
    prefix: str,
) -> List[Tuple[str, float]]:
    grouped: Dict[str, float] = {}
    for key in sorted(details):
        if not key.startswith(prefix):
            continue
        value = details.get(key)
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        label = _format_component_only_label(prefix, key)
        grouped[label] = grouped.get(label, 0.0) + seconds
    return sorted(grouped.items())


def _sum_child_values(children: List[Tuple[str, Any]]) -> float | None:
    if not children:
        return None
    total = 0.0
    found = False
    for _, value in children:
        if value is None:
            continue
        try:
            total += float(value)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def _append_overhead_child(
    children: List[Tuple[str, Any]],
    parent_value: Any,
    label: str,
) -> None:
    try:
        parent = float(parent_value)
    except (TypeError, ValueError):
        return
    child_sum = _sum_child_values(children) or 0.0
    remainder = parent - child_sum
    if remainder > 0.005:
        children.append((label, remainder))


def _format_load_component_label(component: str, stats: Dict[str, Tuple[int, int]]) -> str:
    count, plan_bytes = stats.get(component, (0, 0))
    if count <= 0:
        return component
    if plan_bytes <= 0:
        suffix = f"{count} loads" if count > 1 else "1 load"
    elif count == 1:
        suffix = f"{_format_bytes(plan_bytes)} plan"
    else:
        suffix = f"{count} loads, {_format_bytes(plan_bytes)} total plan bytes"
    return f"{component} ({suffix})"


def _has_extra_compile_breakdown(details: Dict[str, float]) -> bool:
    return any(
        key.startswith("trt_compile_extra_") and key != "trt_compile_extra_engines_s"
        for key in details
    )


def _is_compile_child_key(key: str, details: Dict[str, float]) -> bool:
    if not key.startswith("trt_compile_"):
        return False
    if key in {"trt_compile_s", "trt_compile_diffusion_components_s"}:
        return False
    if key == "trt_compile_extra_engines_s" and _has_extra_compile_breakdown(details):
        return False
    return True


def _render_timing_breakdown(
    label: str,
    children: List[Tuple[str, Any]],
) -> str:
    if not children:
        return _esc(label)

    rows = []
    for child_label, child_value in children:
        rows.append(
            '<div class="timing-breakdown-row">'
            f"<span>{_esc(child_label)}</span>"
            f"<span>{_format_seconds(child_value)}</span>"
            "</div>"
        )
    return (
        '<details class="timing-expand">'
        f"<summary>{_esc(label)}</summary>"
        '<div class="timing-breakdown">' + "\n".join(rows) + "</div></details>"
    )


def _render_detailed_timing_row(
    label: str,
    value: Any,
    children: List[Tuple[str, Any]] | None = None,
) -> str:
    label_html = _render_timing_breakdown(label, children or [])
    return f"<tr><td>{label_html}</td><td>{_format_seconds(value)}</td></tr>"


def _render_detailed_timing_table(result: Dict[str, Any]) -> str:
    details = _normalize_detailed_timing(result)
    load_component_stats = _collect_load_component_stats(result)
    optional_rows = [
        ("bundle_write_s", "Bundle write"),
        ("quantization_context_s", "Quantization context"),
        ("fp8_calibration_s", "FP8 calibration"),
        ("fp8_scales_write_s", "FP8 scales write"),
        ("reference_s", "Reference"),
        ("preflight_s", "Preflight"),
    ]

    weight_children = [
        (_format_component_weight_label(key), details.get(key))
        for key in sorted(details)
        if key.startswith("weights_loading_") and key != "weights_loading_s"
    ]
    _append_overhead_child(weight_children, details.get("weights_loading_s"), "unattributed")
    compile_children = [
        (_format_component_compile_label(key), details.get(key))
        for key in sorted(details)
        if _is_compile_child_key(key, details)
    ]
    _append_overhead_child(compile_children, details.get("trt_compile_s"), "unattributed")

    engine_component_children = [
        (
            "engine execution: " + label,
            value,
        )
        for label, value in _aggregate_component_timings(details, "trt_component_engine_")
    ]
    if not engine_component_children and "inference_s" in details:
        engine_component_children = [("TRT engine execution", details.get("inference_s"))]

    load_component_children = [
        (
            "load/deserialization: " + _format_load_component_label(label, load_component_stats),
            value,
        )
        for label, value in _aggregate_component_timings(details, "trt_component_load_deserialize_")
    ]
    if not load_component_children and "trt_load_deserialization_s" in details:
        load_component_children = [
            ("TRT engine load/deserialization", details.get("trt_load_deserialization_s"))
        ]

    inference_children = engine_component_children + load_component_children
    inference_total = _sum_child_values(inference_children)

    rows: List[str] = [
        _render_detailed_timing_row(
            "Weights loading",
            details.get("weights_loading_s"),
            weight_children,
        ),
        _render_detailed_timing_row(
            "TRT compile",
            details.get("trt_compile_s"),
            compile_children,
        ),
        _render_detailed_timing_row(
            "Inference",
            inference_total,
            inference_children,
        ),
        _render_detailed_timing_row("Comparison", details.get("comparison_s")),
    ]
    for key, label in optional_rows:
        if key in details:
            rows.append(_render_detailed_timing_row(label, details.get(key)))

    return (
        '<table class="timing-table detailed-timing-table">'
        "<thead><tr><th>Phase</th><th>Time</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _render_timing_sections(result: Dict[str, Any]) -> str:
    parts = ["<h4>Detailed Timing</h4>", _render_detailed_timing_table(result)]
    raw_timing = _render_timing_table(result.get("timing", {}) or {})
    if raw_timing:
        parts.append(
            '<details class="raw-timing-section">'
            "<summary>Raw Timing Phases</summary>"
            f"{raw_timing}"
            "</details>"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Repro commands
# ---------------------------------------------------------------------------

_CMD_COUNTER = 0


def _next_cmd_id() -> str:
    global _CMD_COUNTER  # noqa: PLW0603
    _CMD_COUNTER += 1
    return f"cmd_{_CMD_COUNTER}"


def _render_repro_commands(repro: Dict[str, str]) -> str:
    if not repro:
        return ""
    parts = ['<div class="repro-section"><h4>Reproduction Commands</h4>']
    for label, cmd in repro.items():
        cid = _next_cmd_id()
        parts.append(f"<p><strong>{_esc(label)}</strong></p>")
        parts.append(_code_block(cmd, cid))
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Text comparison
# ---------------------------------------------------------------------------


def _render_text_comparison(trt_text: Optional[str], ref_text: Optional[str]) -> str:
    if trt_text is None and ref_text is None:
        return ""
    return (
        '<div class="text-compare">'
        '<div class="text-col">'
        "<h4>TRT Output</h4>"
        f"<pre>{_esc(trt_text or '(none)')}</pre>"
        "</div>"
        '<div class="text-col">'
        "<h4>Reference Output</h4>"
        f"<pre>{_esc(ref_text or '(none)')}</pre>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Modality renderers
# ---------------------------------------------------------------------------


def _get_stage_text(stage_outputs: Dict[str, Any], prefix: str) -> Optional[str]:
    """Extract .text from the first stage output matching *prefix*."""
    for key, val in stage_outputs.items():
        if key.startswith(prefix) and isinstance(val, dict):
            t = val.get("text")
            if t is not None:
                return str(t)
    return None


def _stage_pairs(result: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group serialized stage outputs into TRT/reference pairs by stage name."""
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in (result.get("stage_outputs") or {}).items():
        if not isinstance(value, dict):
            continue
        for prefix in ("trt", "ref"):
            marker = f"{prefix}_"
            if str(key).startswith(marker):
                pairs.setdefault(str(key)[len(marker):], {})[prefix] = value
                break
    return pairs


def _walk_named_values(value: Any, names: set[str]) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names:
                found.append(child)
            if isinstance(child, (dict, list)):
                found.extend(_walk_named_values(child, names))
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                found.extend(_walk_named_values(child, names))
    return found


def _find_output_media(
    result: Dict[str, Any],
    prefix: str,
    artifact_names: Tuple[str, ...],
    stage_names: Tuple[str, ...],
) -> Optional[Path]:
    """Find the first safe media path registered or persisted by a stage."""
    paths = _find_output_media_all(
        result, prefix, artifact_names, stage_names)
    return paths[0] if paths else None


def _find_output_media_all(
    result: Dict[str, Any],
    prefix: str,
    artifact_names: Tuple[str, ...],
    stage_names: Tuple[str, ...],
    suffixes: Optional[set[str]] = None,
) -> List[Path]:
    """Find all safe, de-duplicated media paths for one output side."""
    art_dir = Path(result.get("_artifact_dir") or ".")
    artifacts = result.get("artifacts") or {}
    found: List[Path] = []

    def append(path: Optional[Path]) -> None:
        if path is None:
            return
        if suffixes is not None and path.suffix.lower() not in suffixes:
            return
        if path not in found:
            found.append(path)

    for name in artifact_names:
        refs = artifacts.get(f"{prefix}_{name}")
        values = refs if isinstance(refs, list) else [refs]
        for ref in values:
            append(_resolve_artifact_media(ref, art_dir))

    for key, stage in (result.get("stage_outputs") or {}).items():
        if not str(key).startswith(f"{prefix}_") or not isinstance(stage, dict):
            continue
        for ref in _walk_named_values(stage, set(stage_names)):
            values = ref if isinstance(ref, list) else [ref]
            for value in values:
                append(_resolve_artifact_media(value, art_dir))
    return found


def _input_ref(inputs: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    for name in names:
        value = inputs.get(name)
        if value:
            return value
    return None


def _render_audio_player(title: str, path: Optional[Path]) -> str:
    if path is None:
        return (
            '<div class="media-card missing-media">'
            f"<h4>{_esc(title)}</h4>"
            '<p class="missing">Required audio file is unavailable.</p></div>'
        )
    uri = encode_file_base64(path, _mime_for_ext(path.suffix))
    if uri is None:
        return (
            '<div class="media-card missing-media">'
            f"<h4>{_esc(title)}</h4>"
            f'<p class="missing">Audio could not be embedded: {_esc(path.name)}</p></div>'
        )
    return (
        '<div class="media-card">'
        f"<h4>{_esc(title)}</h4>"
        f'<audio controls preload="metadata" src="{uri}"></audio>'
        f'<p class="media-filename">{_esc(path.name)}</p></div>'
    )


def _render_image_card(title: str, path: Optional[Path]) -> str:
    if path is None:
        return (
            '<div class="media-card missing-media">'
            f"<h4>{_esc(title)}</h4>"
            '<p class="missing">Required image is unavailable.</p></div>'
        )
    uri = encode_file_base64(path, _mime_for_ext(path.suffix))
    if uri is None:
        return (
            '<div class="media-card missing-media">'
            f"<h4>{_esc(title)}</h4>"
            f'<p class="missing">Image could not be embedded: {_esc(path.name)}</p></div>'
        )
    return (
        '<div class="media-card">'
        f"<h4>{_esc(title)}</h4>"
        f'<img src="{uri}" class="preview-img" alt="{_esc(title)}" />'
        f'<p class="media-filename">{_esc(path.name)}</p></div>'
    )


def _compact_data(value: Any, list_limit: int = 24, depth: int = 0) -> Any:
    """Return a deterministic, bounded representation for structured output."""
    if depth >= 5:
        return "<nested value omitted>"
    if isinstance(value, dict):
        return {
            str(key): _compact_data(child, list_limit, depth + 1)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if key not in {"stderr", "stdout", "audio_samples"}
        }
    if isinstance(value, list):
        compact = [_compact_data(child, list_limit, depth + 1) for child in value[:list_limit]]
        if len(value) > list_limit:
            compact.append(f"<... {len(value) - list_limit} more values>")
        return compact
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + f"\n<... {len(value) - 1000} more characters>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _structured_text(stage: Optional[Dict[str, Any]]) -> str:
    if not stage:
        return "(unavailable)"
    payload: Dict[str, Any] = {}
    if stage.get("text") not in (None, ""):
        payload["text"] = stage.get("text")
    data = stage.get("data")
    if data not in (None, {}, []):
        payload["data"] = data
    if not payload:
        payload["metadata"] = stage.get("metadata") or {}
    return json.dumps(_compact_data(payload), indent=2, ensure_ascii=False)


def _render_structured_stage_comparison(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for stage_name, pair in _stage_pairs(result).items():
        parts.append(f"<h4>Stage: {_esc(stage_name)}</h4>")
        parts.append(_render_text_comparison(
            _structured_text(pair.get("trt")),
            _structured_text(pair.get("ref")),
        ))
    if not parts:
        return '<p class="missing">No serialized stage outputs were recorded.</p>'
    return "\n".join(parts)


def _flatten_numeric_preview(value: Any, limit: int = 16) -> Tuple[List[float], int, float]:
    """Return the first numeric leaves, total numeric leaf count, and L2 sum."""
    preview: List[float] = []
    total = 0
    sumsq = 0.0
    stack = [value]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (int, float)):
            total += 1
            sumsq += float(cur) * float(cur)
            if len(preview) < limit:
                preview.append(float(cur))
        elif isinstance(cur, list):
            stack.extend(reversed(cur))
    return preview, total, sumsq


def _get_stage_feature_output(
    stage_outputs: Dict[str, Any], prefix: str
) -> Optional[Tuple[str, Any]]:
    """Extract a structured numeric feature output from a matching stage."""
    feature_keys = ("cls_embedding", "embedding")
    for key, val in stage_outputs.items():
        if not key.startswith(prefix) or not isinstance(val, dict):
            continue
        data = val.get("data")
        if not isinstance(data, dict):
            continue
        for feature_key in feature_keys:
            if feature_key in data:
                return feature_key, data[feature_key]
    return None


def _format_feature_output(feature: Optional[Tuple[str, Any]]) -> Optional[str]:
    """Format an embedding/feature vector so generic reports show references."""
    if feature is None:
        return None
    name, value = feature
    preview, total, sumsq = _flatten_numeric_preview(value)
    if total <= 0:
        return f"{name}: (no numeric values)"

    norm = sumsq**0.5
    suffix = " ..." if total > len(preview) else ""
    preview_text = ", ".join(f"{x:.6g}" for x in preview)
    return (
        f"{name} ({total} values)\n"
        f"preview[0:{len(preview)}]: [{preview_text}{suffix}]\n"
        f"l2_norm: {norm:.6g}"
    )


def render_text_model(result: Dict[str, Any]) -> str:
    """Render detail section for a text-generation model."""
    cc = result.get("case_config", {})
    prompt = (cc.get("inputs") or {}).get("prompt", "")
    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")

    parts = []
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    if trt_text is None or ref_text is None:
        parts.append(_render_structured_stage_comparison(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_diffusion_text_model(result: Dict[str, Any]) -> str:
    """Render non-autoregressive text samples and their configured reference."""
    cc = result.get("case_config") or {}
    inputs = cc.get("inputs") or {}
    stage_outputs = result.get("stage_outputs") or {}
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")
    parts = []
    for label, names in (
        ("Prompt", ("prompt",)),
        ("Source text", ("source_text", "condition_text")),
        ("Generation mode", ("generation_mode", "sampling_method")),
        ("Samples", ("num_samples",)),
        ("Diffusion steps", ("num_steps", "num_sampling_steps", "steps")),
        ("Seed", ("seed",)),
    ):
        value = _input_ref(inputs, names)
        if value not in (None, ""):
            parts.append(f"<p><strong>{_esc(label)}:</strong> {_esc(value)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append("<h4>Generated and Expected Samples</h4>")
    parts.append(_render_structured_stage_comparison(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_vl_model(result: Dict[str, Any], project_dir: Optional[Path]) -> str:
    """Render detail section for a vision-language model."""
    cc = result.get("case_config", {})
    inputs = cc.get("inputs") or {}
    prompt = inputs.get("prompt", "")
    image_rel = inputs.get("image", "")

    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")

    parts = []

    # Embed input image
    if image_rel:
        parts.append('<div class="media-compare">')
        parts.append(
            _render_image_card(
                "Input Image", _resolve_input_media(image_rel, project_dir)
            )
        )
        parts.append("</div>")

    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_diffusion_model(
    result: Dict[str, Any], project_dir: Optional[Path] = None
) -> str:
    """Render detail section for a diffusion model."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})
    inputs = (result.get("case_config") or {}).get("inputs") or {}

    parts = []

    prompt = inputs.get("prompt") or ""
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    condition_image = _input_ref(
        inputs, ("image", "image_path", "input_image", "conditioning_image"))
    if condition_image:
        parts.append('<div class="media-compare">')
        parts.append(_render_image_card(
            "Conditioning Image",
            _resolve_input_media(condition_image, project_dir),
        ))
        parts.append("</div>")

    # TRT frames gallery
    trt_frame_paths = _resolve_frame_paths(
        artifacts.get("trt_frames", []),
        art_dir=art_dir,
        fallback_dir_name="frames",
    )

    # Reference frames (side-by-side if available)
    ref_frame_paths = _resolve_frame_paths(
        artifacts.get("ref_frames", []),
        art_dir=art_dir,
        fallback_dir_name="ref_frames",
    )

    if trt_frame_paths and ref_frame_paths:
        selected_trt = _select_frames(trt_frame_paths, _MAX_DIFFUSION_FRAMES)
        selected_ref = _select_frames(ref_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>Visual Review: TRT vs Reference</h4>")
        parts.append(
            f"<p>TRT / Base frames: {len(trt_frame_paths)}; "
            f"Reference frames: {len(ref_frame_paths)}; "
            f"showing up to {_MAX_DIFFUSION_FRAMES} evenly spaced frames per side.</p>"
        )
        parts.append('<div class="frame-compare">')
        for idx in range(max(len(selected_trt), len(selected_ref))):
            trt_fp = selected_trt[idx] if idx < len(selected_trt) else None
            ref_fp = selected_ref[idx] if idx < len(selected_ref) else None
            trt_uri = (
                encode_file_base64(trt_fp, _mime_for_ext(trt_fp.suffix))
                if trt_fp is not None else None
            )
            ref_uri = (
                encode_file_base64(ref_fp, _mime_for_ext(ref_fp.suffix))
                if ref_fp is not None else None
            )
            parts.append('<div class="frame-pair">')
            labels = " / ".join(
                path.name for path in (trt_fp, ref_fp) if path is not None)
            parts.append(
                f'<div class="frame-pair-title">Sample {idx + 1}: {_esc(labels)}</div>')
            parts.append('<div class="frame-pair-images">')
            if trt_uri:
                parts.append(
                    '<figure><figcaption>TRT / Base</figcaption>'
                    f'<img src="{trt_uri}" class="frame-img" alt="TRT frame" /></figure>'
                )
            else:
                parts.append("<span class='missing'>TRT frame unavailable or too large</span>")
            if ref_uri:
                parts.append(
                    '<figure><figcaption>Reference</figcaption>'
                    f'<img src="{ref_uri}" class="frame-img" alt="Reference frame" /></figure>'
                )
            else:
                parts.append("<span class='missing'>Reference frame unavailable or too large</span>")
            parts.append("</div></div>")
        parts.append("</div>")
    elif trt_frame_paths:
        selected = _select_frames(trt_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>TRT Generated Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected:
            uri = encode_file_base64(fp, _mime_for_ext(fp.suffix))
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" alt="TRT frame" />')
            else:
                parts.append("<span class='missing'>Frame too large</span>")
        parts.append("</div>")
    elif ref_frame_paths:
        selected_ref = _select_frames(ref_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>Reference Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected_ref:
            uri = encode_file_base64(fp, _mime_for_ext(fp.suffix))
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" alt="Reference frame" />')
        parts.append("</div>")

    parts.append(_render_diffusion_vlm_assessment(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def _stage_output_returncode(stage: Dict[str, Any]) -> Any:
    for key in ("data", "metadata"):
        value = stage.get(key, {})
        if isinstance(value, dict) and "returncode" in value:
            return value.get("returncode")
    return stage.get("returncode")


def _stage_output_error_excerpt(stage: Dict[str, Any]) -> str:
    data = stage.get("data", {})
    metadata = stage.get("metadata", {})
    candidates: List[Any] = []
    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("error"),
                data.get("parse_error"),
                data.get("stderr_truncated"),
                data.get("stderr"),
            ]
        )
    for candidate in candidates:
        if candidate:
            raw = str(candidate)
            for marker in (
                "LocalEntryNotFoundError",
                "OSError",
                "Cannot find",
                "couldn't connect",
                "Error in call",
            ):
                for line in raw.splitlines():
                    if marker in line:
                        text = " ".join(line.split())
                        return text[:297] + "..." if len(text) > 300 else text
            text = " ".join(raw.split())
            return text[:297] + "..." if len(text) > 300 else text
    if isinstance(data, dict) and data.get("stderr_log"):
        return f"see {data.get('stderr_log')}"
    if isinstance(metadata, dict) and metadata.get("command"):
        return f"command: {metadata.get('command')}"
    return ""


def _render_missing_reference_audio_notice(stage_outputs: Dict[str, Any]) -> str:
    for stage_key, stage in stage_outputs.items():
        if not str(stage_key).startswith("ref_") or not isinstance(stage, dict):
            continue
        returncode = _stage_output_returncode(stage)
        error_excerpt = _stage_output_error_excerpt(stage)
        failed = returncode not in (None, 0, "0")
        if not failed and not error_excerpt:
            continue
        details = [f"reference stage {_esc(stage_key)}"]
        if returncode is not None:
            details.append(f"returned {_esc(returncode)}")
        if error_excerpt:
            details.append(_esc(error_excerpt))
        return (
            '<p class="failure-info"><strong>Reference Audio unavailable:</strong> '
            + "; ".join(details)
            + "</p>"
        )
    return ""


def render_audio_model(
    result: Dict[str, Any], project_dir: Optional[Path] = None
) -> str:
    """Render detail section for an audio model."""
    stage_outputs = result.get("stage_outputs", {})
    cc = result.get("case_config", {})
    task_strategy = cc.get("task_strategy", "")
    inputs = cc.get("inputs") or {}

    parts = []

    prompt = inputs.get("prompt") or ""
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")

    input_audio_ref = _input_ref(
        inputs, ("audio", "audio_path", "input_audio", "input_audio_path"))
    if input_audio_ref:
        input_audio = _resolve_input_media(input_audio_ref, project_dir)
        parts.append('<div class="media-compare">')
        parts.append(_render_audio_player("Input / Source Audio", input_audio))
        parts.append("</div>")

    # For speech_to_text, show transcript comparison
    if task_strategy == "speech_to_text":
        trt_text = _get_stage_text(stage_outputs, "trt_")
        ref_text = _get_stage_text(stage_outputs, "ref_")
        if trt_text is not None or ref_text is not None:
            parts.append("<h4>Transcript Comparison</h4>")
            parts.append(_render_text_comparison(trt_text, ref_text))

    if task_strategy in {"text_to_audio", "speech_to_speech"}:
        trt_wav = _find_output_media(
            result, "trt", ("wav", "audio"),
            ("wav_path", "audio_output_path"))
        ref_wav = _find_output_media(
            result, "ref", ("wav", "audio"),
            ("wav_path", "audio_output_path"))
        parts.append("<h4>Audio Comparison</h4>")
        parts.append('<div class="media-compare">')
        parts.append(_render_audio_player("TRT / Base Audio", trt_wav))
        parts.append(_render_audio_player("Reference Audio", ref_wav))
        parts.append("</div>")
        if ref_wav is None:
            notice = _render_missing_reference_audio_notice(stage_outputs)
            if notice:
                parts.append(notice)
            parts.append("<h4>Configured Reference Evidence</h4>")
            parts.append(_render_structured_stage_comparison(result))

        trt_data = _first_stage_data(result, "trt")
        ref_data = _first_stage_data(result, "ref")
        metadata_rows = []
        for label, key in (
            ("Duration (seconds)", "duration_s"),
            ("Sample rate", "sample_rate"),
            ("Samples", "num_samples"),
            ("RMS", "rms"),
            ("Reference frames", "num_frames"),
            ("Reference token shape", "token_shape"),
        ):
            if key not in trt_data and key not in ref_data:
                continue
            metadata_rows.append(
                f"<tr><td>{_esc(label)}</td><td>{_format_value(trt_data.get(key))}</td>"
                f"<td>{_format_value(ref_data.get(key))}</td></tr>"
            )
        if metadata_rows:
            parts.append("<h4>Audio Metadata</h4>")
            parts.append(
                '<table class="metrics-table"><thead><tr><th>Field</th>'
                '<th>TRT / Base</th><th>Reference</th></tr></thead><tbody>'
                + "".join(metadata_rows) + "</tbody></table>"
            )

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_segmentation_model(result: Dict[str, Any], project_dir: Optional[Path]) -> str:
    """Render detail section for a segmentation model."""
    cc = result.get("case_config", {})
    inputs = cc.get("inputs") or {}
    image_rel = inputs.get("image", "")
    prompt = inputs.get("prompt", "")

    parts = []

    # Input image
    if image_rel:
        parts.append('<div class="media-compare">')
        parts.append(
            _render_image_card(
                "Input Image", _resolve_input_media(image_rel, project_dir)
            )
        )
        parts.append("</div>")

    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    if "point_x" in inputs or "point_y" in inputs:
        parts.append(
            "<p><strong>Point prompt:</strong> "
            f"x={_esc(inputs.get('point_x', ''))}, "
            f"y={_esc(inputs.get('point_y', ''))}, "
            f"foreground={_esc(inputs.get('is_foreground', True))}</p>"
        )

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    trt_visuals = _find_output_media_all(
        result, "trt", ("segmented_image", "segmentation_map", "output"),
        ("segmented_image_path", "segmentation_map_path", "viz_path", "output_path"),
        image_suffixes)
    ref_visuals = _find_output_media_all(
        result, "ref", ("segmented_image", "segmentation_map", "output"),
        ("segmented_image_path", "segmentation_map_path", "viz_path", "output_path"),
        image_suffixes)
    parts.append("<h4>Segmentation Comparison</h4>")
    parts.append('<div class="media-compare">')
    if not trt_visuals:
        parts.append(_render_image_card("TRT / Base Segmentation", None))
    for index, path in enumerate(trt_visuals, 1):
        suffix = f" {index}" if len(trt_visuals) > 1 else ""
        parts.append(_render_image_card(f"TRT / Base Segmentation{suffix}", path))
    if not ref_visuals:
        parts.append(_render_image_card("Reference Segmentation", None))
    for index, path in enumerate(ref_visuals, 1):
        suffix = f" {index}" if len(ref_visuals) > 1 else ""
        parts.append(_render_image_card(f"Reference Segmentation{suffix}", path))
    parts.append("</div>")

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def _first_stage_data(
    result: Dict[str, Any], prefix: str
) -> Dict[str, Any]:
    for key, value in (result.get("stage_outputs") or {}).items():
        if str(key).startswith(f"{prefix}_") and isinstance(value, dict):
            data = value.get("data")
            if isinstance(data, dict):
                return data
    return {}


def _first_stage_value(
    result: Dict[str, Any], prefix: str, names: Tuple[str, ...]
) -> Any:
    for pair in _stage_pairs(result).values():
        stage = pair.get(prefix)
        if not stage:
            continue
        data = stage.get("data") or {}
        for name in names:
            if name in data:
                return data[name]
    return None


def render_classification_model(
    result: Dict[str, Any], project_dir: Optional[Path]
) -> str:
    cc = result.get("case_config") or {}
    inputs = cc.get("inputs") or {}
    image_ref = _input_ref(inputs, ("image", "image_path", "input_image"))
    trt = _first_stage_data(result, "trt")
    ref = _first_stage_data(result, "ref")
    parts = ['<div class="media-compare">']
    parts.append(_render_image_card(
        "Classification Input", _resolve_input_media(image_ref, project_dir)))
    parts.append("</div>")
    parts.append("<h4>Prediction Comparison</h4>")
    rows = []
    for label, key in (
        ("Top class", "top_class"),
        ("Top score", "top_score"),
        ("Number of classes", "num_classes"),
    ):
        rows.append(
            f"<tr><td>{_esc(label)}</td><td>{_format_value(trt.get(key))}</td>"
            f"<td>{_format_value(ref.get(key))}</td></tr>"
        )
    parts.append(
        '<table class="metrics-table"><thead><tr><th>Field</th>'
        '<th>TRT / Base</th><th>Reference</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_detection_model(
    result: Dict[str, Any], project_dir: Optional[Path]
) -> str:
    cc = result.get("case_config") or {}
    inputs = cc.get("inputs") or {}
    image_ref = _input_ref(inputs, ("image", "image_path", "input_image"))
    parts = ['<div class="media-compare">']
    parts.append(_render_image_card(
        "Detection Input", _resolve_input_media(image_ref, project_dir)))
    parts.append("</div>")

    def detections(prefix: str) -> List[Dict[str, Any]]:
        value = _first_stage_value(result, prefix, ("detections",))
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        data = _first_stage_data(result, prefix)
        boxes = data.get("boxes") or []
        scores = data.get("scores") or []
        labels = data.get("labels") or data.get("class_ids") or []
        return [
            {
                "box": box,
                "score": scores[index] if index < len(scores) else None,
                "label": labels[index] if index < len(labels) else None,
            }
            for index, box in enumerate(boxes)
        ]

    def table(title: str, values: List[Dict[str, Any]]) -> str:
        rows = []
        for index, item in enumerate(values, 1):
            box = item.get("box", item.get("bbox", item.get("boxes", "")))
            rows.append(
                f"<tr><td>{index}</td><td>{_esc(item.get('label', ''))}</td>"
                f"<td>{_format_value(item.get('score'))}</td>"
                f"<td>{_esc(box)}</td></tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="4" class="missing">No detections recorded.</td></tr>')
        return (
            f"<h4>{_esc(title)}</h4>"
            '<table class="metrics-table"><thead><tr><th>#</th><th>Label</th>'
            '<th>Score</th><th>Box</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>"
        )

    parts.append('<div class="detection-compare">')
    parts.append(f'<div>{table("TRT / Base Detections", detections("trt"))}</div>')
    parts.append(f'<div>{table("Reference Detections", detections("ref"))}</div>')
    parts.append("</div>")
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_reranking_model(result: Dict[str, Any]) -> str:
    cc = result.get("case_config") or {}
    inputs = cc.get("inputs") or {}
    query = inputs.get("query") or inputs.get("prompt") or ""
    documents = inputs.get("documents") or _first_stage_value(
        result, "trt", ("documents",)) or []
    trt_scores = _first_stage_value(result, "trt", ("scores",)) or []
    ref_scores = _first_stage_value(result, "ref", ("scores",)) or []

    def rank_positions(scores: List[Any]) -> Dict[int, int]:
        return {
            index: rank
            for rank, (index, _score) in enumerate(
                sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True),
                1,
            )
        }

    trt_ranks = rank_positions(trt_scores)
    ref_ranks = rank_positions(ref_scores)
    parts = []
    if query:
        parts.append(f"<p><strong>Query:</strong> {_esc(query)}</p>")
    rows = []
    count = max(len(documents), len(trt_scores), len(ref_scores))
    for index in range(count):
        document = documents[index] if index < len(documents) else ""
        trt_score = trt_scores[index] if index < len(trt_scores) else None
        ref_score = ref_scores[index] if index < len(ref_scores) else None
        rows.append(
            f"<tr><td>{index + 1}</td><td>{_esc(document)}</td>"
            f"<td>{_format_value(trt_score)}</td><td>{_esc(trt_ranks.get(index, ''))}</td>"
            f"<td>{_format_value(ref_score)}</td><td>{_esc(ref_ranks.get(index, ''))}</td></tr>"
        )
    parts.append(
        '<table class="metrics-table"><thead><tr><th>#</th><th>Document</th>'
        '<th>TRT / Base score</th><th>TRT rank</th>'
        '<th>Reference score</th><th>Reference rank</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table>"
    )
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def _numeric_series(value: Any) -> List[float]:
    values: List[float] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
        elif isinstance(current, list):
            stack.extend(reversed(current))
    return values


def _render_series_plot(trt_values: List[float], ref_values: List[float]) -> str:
    if not trt_values and not ref_values:
        return ""
    max_points = 240

    def sample(values: List[float]) -> List[float]:
        if len(values) <= max_points:
            return values
        return [
            values[round(i * (len(values) - 1) / (max_points - 1))]
            for i in range(max_points)
        ]

    trt_values = sample(trt_values)
    ref_values = sample(ref_values)
    combined = trt_values + ref_values
    low, high = min(combined), max(combined)
    span = high - low or 1.0
    width, height, pad = 800.0, 260.0, 28.0

    def points(values: List[float]) -> str:
        if not values:
            return ""
        denom = max(1, len(values) - 1)
        return " ".join(
            f"{pad + i * (width - 2 * pad) / denom:.1f},"
            f"{height - pad - (value - low) * (height - 2 * pad) / span:.1f}"
            for i, value in enumerate(values)
        )

    return (
        '<div class="series-plot"><svg viewBox="0 0 800 260" '
        'role="img" aria-label="TRT and reference numeric output plot">'
        '<rect x="0" y="0" width="800" height="260" fill="#f8fafc" />'
        f'<polyline points="{points(ref_values)}" fill="none" '
        'stroke="#2563eb" stroke-width="2" />'
        f'<polyline points="{points(trt_values)}" fill="none" '
        'stroke="#dc2626" stroke-width="2" />'
        '</svg><p class="plot-legend"><span class="legend-trt">TRT / Base</span> '
        '<span class="legend-ref">Reference</span></p></div>'
    )


def render_neural_operator_model(result: Dict[str, Any]) -> str:
    inputs = ((result.get("case_config") or {}).get("inputs") or {})
    trt_value = _first_stage_value(
        result, "trt", ("output_field", "field", "prediction", "forecast"))
    ref_value = _first_stage_value(
        result, "ref", ("output_field", "field", "prediction", "forecast"))
    trt_values = _numeric_series(trt_value)
    ref_values = _numeric_series(ref_value)
    parts = []
    relevant_inputs = {
        key: value for key, value in inputs.items()
        if key in {
            "branch_input", "trunk_input", "field_input", "input_field",
            "context", "prediction_length", "horizon", "frequency",
        }
    }
    if relevant_inputs:
        input_json = json.dumps(
            _compact_data(relevant_inputs), indent=2, ensure_ascii=False)
        parts.append("<h4>Model Inputs</h4>")
        parts.append(f'<pre class="structured-input">{_esc(input_json)}</pre>')
    plot = _render_series_plot(trt_values, ref_values)
    if plot:
        parts.extend(["<h4>Output Series Comparison</h4>", plot])
    parts.append(_render_structured_stage_comparison(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_omni_model(
    result: Dict[str, Any], project_dir: Optional[Path]
) -> str:
    cc = result.get("case_config") or {}
    inputs = cc.get("inputs") or {}
    parts = []
    prompt = inputs.get("prompt") or ""
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    image_ref = _input_ref(inputs, ("image", "image_path", "input_image"))
    audio_ref = _input_ref(
        inputs, ("audio", "audio_path", "input_audio", "input_audio_path"))
    if image_ref or audio_ref:
        parts.append('<div class="media-compare">')
        if image_ref:
            parts.append(_render_image_card(
                "Input Image", _resolve_input_media(image_ref, project_dir)))
        if audio_ref:
            parts.append(_render_audio_player(
                "Input Audio", _resolve_input_media(audio_ref, project_dir)))
        parts.append("</div>")

    trt_audio = _find_output_media(
        result, "trt", ("wav", "audio"), ("wav_path", "audio_output_path"))
    ref_audio = _find_output_media(
        result, "ref", ("wav", "audio"), ("wav_path", "audio_output_path"))
    if trt_audio is not None or ref_audio is not None:
        parts.append("<h4>Audio Comparison</h4>")
        parts.append('<div class="media-compare">')
        parts.append(_render_audio_player("TRT / Base Audio", trt_audio))
        parts.append(_render_audio_player("Reference Audio", ref_audio))
        parts.append("</div>")

    parts.append(_render_structured_stage_comparison(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_generic_model(result: Dict[str, Any]) -> str:
    """Render numeric/embedding output with a structured fallback."""
    cc = result.get("case_config", {})
    prompt = (cc.get("inputs") or {}).get("prompt", "")
    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")
    trt_feature = _format_feature_output(_get_stage_feature_output(stage_outputs, "trt_"))
    ref_feature = _format_feature_output(_get_stage_feature_output(stage_outputs, "ref_"))

    parts = []
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    if trt_text or ref_text:
        parts.append(_render_text_comparison(trt_text, ref_text))
    elif trt_feature or ref_feature:
        parts.append(_render_text_comparison(trt_feature, ref_feature))
    else:
        parts.append(_render_structured_stage_comparison(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-model collapsible section
# ---------------------------------------------------------------------------


def _uses_external_reference(result: Dict[str, Any]) -> bool:
    cc = result.get("case_config") or {}
    backend = str(cc.get("reference_backend") or "")
    oracle_level = str(result.get("oracle_level") or cc.get("oracle_level") or "")
    return backend != "invariant_only" and not oracle_level.startswith("L4")


def _native_visual_acceptance(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cc = result.get("case_config") or {}
    metadata = cc.get("metadata") or {}
    policy = metadata.get("native_acceptance")
    if not isinstance(policy, dict):
        return None
    expected = {
        "kind": "native_visual_semantic_acceptance",
        "reference_role": "diagnostic",
        "requires_nightly_vlm": True,
        "vlm_frame_samples": 6,
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        return None
    if not isinstance(policy.get("rationale"), str) or not policy["rationale"].strip():
        return None
    return policy


def _render_oracle_context(result: Dict[str, Any]) -> str:
    cc = result.get("case_config") or {}
    backend = str(cc.get("reference_backend") or "unspecified")
    oracle_level = str(result.get("oracle_level") or cc.get("oracle_level") or "unspecified")
    css_class = "oracle-external" if _uses_external_reference(result) else "oracle-invariant"
    native_acceptance = _native_visual_acceptance(result)
    if native_acceptance is not None:
        rationale = str(native_acceptance["rationale"])
        return (
            f'<div class="oracle-context {css_class}">'
            "<strong>Native visual semantic acceptance policy</strong><br>"
            f"Oracle: {_esc(oracle_level)} via {_esc(backend)}<br>"
            "Official reference pixel parity: diagnostic, not claimed.<br>"
            "All-frame temporal activity/cadence alignment: required.<br>"
            "Six-frame Nightly VLM semantic gate: required.<br>"
            f"Rationale: {_esc(rationale)}</div>"
        )
    note = "External/reference output comparison is configured."
    if not _uses_external_reference(result):
        ref_audio = _find_output_media(
            result, "ref", ("wav", "audio"),
            ("wav_path", "audio_output_path"))
        if ref_audio is not None:
            note = (
                "The automated gate uses model-owned invariants. The embedded "
                "external reference audio is human-review evidence, not a "
                "waveform-equality gate."
            )
        else:
            note = (
                "No external reference output is configured; this case validates "
                "model-owned invariants only."
            )
    return (
        f'<div class="oracle-context {css_class}"><strong>Oracle:</strong> '
        f"{_esc(oracle_level)} via {_esc(backend)} &mdash; {_esc(note)}</div>"
    )


def render_model_section(
    result: Dict[str, Any],
    project_dir: Optional[Path],
) -> str:
    """Render a single collapsible ``<details>`` for one model."""
    name = result.get("case_name", "unknown")
    status = result.get("status", "error")
    cc = result.get("case_config", {})
    family = cc.get("family", "")
    task_strategy = cc.get("task_strategy", "")
    hf_id = cc.get("hf_id", "")

    modality = classify_modality(result)
    badge = _badge(status)

    header = f'<details id="model-{_esc(name)}"><summary>{badge} <strong>{_esc(name)}</strong>'
    if family or task_strategy:
        header += f" &mdash; {_esc(family)} / {_esc(task_strategy)}"
    elif result.get("_summary_only"):
        header += " &mdash; pytest-only grouped member"
    if hf_id:
        header += f" <small>({_esc(hf_id)})</small>"
    header += "</summary>"

    # Failure info
    body_parts = [_render_oracle_context(result)]
    if result.get("_summary_only"):
        body_parts.append(
            '<p class="summary-only-info">'
            "This testcase was recovered from pytest/JUnit output; no "
            "per-case result.json artifact was found for it.</p>"
        )
    pytest_outcome = result.get("_pytest_outcome")
    pytest_status = ""
    if isinstance(pytest_outcome, dict):
        pytest_status = str(pytest_outcome.get("pytest_status") or "")
        pytest_reason = str(pytest_outcome.get("reason") or "")
        note = f"Pytest outcome: <strong>{_esc(pytest_status)}</strong>"
        if pytest_reason:
            note += f" &mdash; {_esc(pytest_reason)}"
        body_parts.append(f'<p class="waive-info">{note}</p>')

    failure_type = result.get("failure_type")
    if failure_type and pytest_status != "XFAIL":
        body_parts.append(
            f'<p class="failure-info">Failure type: <strong>{_esc(failure_type)}</strong></p>'
        )

    # Dispatch to modality renderer
    if modality == "text":
        body_parts.append(render_text_model(result))
    elif modality == "diffusion_text":
        body_parts.append(render_diffusion_text_model(result))
    elif modality == "vl":
        body_parts.append(render_vl_model(result, project_dir))
    elif modality == "diffusion":
        body_parts.append(render_diffusion_model(result, project_dir))
    elif modality == "audio":
        body_parts.append(render_audio_model(result, project_dir))
    elif modality == "segmentation":
        body_parts.append(render_segmentation_model(result, project_dir))
    elif modality == "classification":
        body_parts.append(render_classification_model(result, project_dir))
    elif modality == "detection":
        body_parts.append(render_detection_model(result, project_dir))
    elif modality == "reranking":
        body_parts.append(render_reranking_model(result))
    elif modality == "neural_operator":
        body_parts.append(render_neural_operator_model(result))
    elif modality == "omni":
        body_parts.append(render_omni_model(result, project_dir))
    elif modality == "structured":
        body_parts.append(_render_structured_stage_comparison(result))
    else:
        body_parts.append(render_generic_model(result))

    body = "\n".join(body_parts)
    return f'{header}\n<div class="model-body">{body}</div>\n</details>'


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------


def _key_metric(result: Dict[str, Any]) -> str:
    """Extract the single most representative metric for the summary row."""
    assessment = result.get("vlm_assessment")
    if isinstance(assessment, dict):
        judgment = assessment.get("vlm_judgment", {})
        if isinstance(judgment, dict) and "semantic_similarity_0_to_5" in judgment:
            return (
                "vlm_semantic_similarity="
                f"{_format_value(judgment.get('semantic_similarity_0_to_5'))}"
            )

    stages = result.get("stages", {})
    # Priority: logit_cosine_p5, token_agreement_rate, miou, psnr, mel_distance
    priority = [
        "logit_cosine_p5",
        "token_agreement_rate",
        "miou",
        "psnr",
        "ssim",
        "mel_distance",
        "pixel_accuracy",
        "normalized_text_edit_distance",
    ]
    for stage_data in stages.values():
        metrics = stage_data.get("metrics", {})
        for key in priority:
            if key in metrics:
                m = metrics[key]
                val = m.get("value", m) if isinstance(m, dict) else m
                return f"{key}={_format_value(val)}"
    return ""


def _total_time_seconds(result: Dict[str, Any]) -> Optional[float]:
    timing = result.get("timing", {})
    if not timing:
        return None
    total = 0.0
    for value in timing.values():
        if value is None:
            continue
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return total


def _total_time(result: Dict[str, Any]) -> str:
    total = _total_time_seconds(result)
    if total is None:
        return ""
    return f"{total:.1f}s"


def _total_time_sort_key(result: Dict[str, Any]) -> float:
    total = _total_time_seconds(result)
    return total if total is not None else -1.0


def _model_group_key(result: Dict[str, Any]) -> str:
    return _result_model_name(result)


def _model_group_label(group_key: str) -> str:
    return group_key


def _model_group_sort_key(group_key: str, result: Dict[str, Any]) -> Tuple[int, str]:
    name = str(result.get("case_name", ""))
    return (0 if name == group_key else 1, name)


def _grouped_model_results(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        group_key = _model_group_key(result)
        if not group_key:
            continue
        groups.setdefault(group_key, []).append(result)
    return {
        key: sorted(items, key=lambda item: _model_group_sort_key(key, item))
        for key, items in sorted(groups.items())
    }


def _result_by_case_name(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(result.get("case_name")): result
        for result in results
        if result.get("case_name")
    }


def _with_declared_testcases(
    results: List[Dict[str, Any]],
    model_manifests: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add manifest-only rows and parent metadata without changing result totals."""
    by_name = _result_by_case_name(results)
    declared_names = {
        str(case["name"])
        for model in model_manifests
        for case in model["testcases"]
    }
    display_results = [
        result for result in results if str(result.get("case_name") or "") not in declared_names
    ]

    for model in model_manifests:
        model_name = str(model["name"])
        family = str(model.get("family") or "")
        for case in model["testcases"]:
            case_name = str(case["name"])
            existing = by_name.get(case_name)
            item = (
                dict(existing)
                if existing is not None
                else {
                    "case_name": case_name,
                    "status": "not_run",
                    "stages": {},
                    "timing": {},
                    "_manifest_only": True,
                }
            )
            case_config = dict(item.get("case_config") or {})
            metadata = dict(case_config.get("metadata") or {})
            metadata.setdefault("model_name", model_name)
            metadata.setdefault("ci_tier", str(case.get("ci_tier") or "default"))
            case_config["metadata"] = metadata
            case_config.setdefault("family", family)
            case_config.setdefault("task_strategy", str(case.get("task_strategy") or ""))
            item["case_config"] = case_config
            display_results.append(item)
    return display_results


def _status_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "error"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _status_summary(results: List[Dict[str, Any]]) -> str:
    counts = _status_counts(results)
    if len(counts) == 1:
        status = next(iter(counts))
        return _badge(status)
    return " ".join(
        f'<span class="status-count">{_esc(_status_label(status))}: {count}</span>'
        for status, count in sorted(counts.items())
    )


def _model_representative(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in items:
        if not item.get("_summary_only") and not item.get("_manifest_only"):
            return item
    for item in items:
        if not item.get("_manifest_only"):
            return item
    return items[0]


def _summary_sort_key(item: Tuple[Dict[str, Any], List[Dict[str, Any]]]) -> float:
    representative, _members = item
    return _total_time_sort_key(representative)


def _summary_dashboard_items(
    results: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    groups = _grouped_model_results(results)
    dashboard_items = [(_model_representative(items), items) for items in groups.values()]
    return sorted(dashboard_items, key=_summary_sort_key, reverse=True)


def _render_summary_model_details(
    group_key: str,
    representative: Dict[str, Any],
    items: List[Dict[str, Any]],
    family: str,
    task_strategy: str,
    key_metric: str,
    total_time: str,
) -> str:
    title = _model_group_label(group_key)
    testcase_label = "testcase" if len(items) == 1 else "testcases"
    rows = []
    for item in items:
        name = str(item.get("case_name", "unknown"))
        case_config = item.get("case_config", {}) or {}
        child_family = case_config.get("family", "")
        child_task_strategy = case_config.get("task_strategy", "")
        status = str(item.get("status", "error"))
        summary_only = " pytest-only" if item.get("_summary_only") else ""
        manifest_only = " manifest-only" if item.get("_manifest_only") else ""
        metadata = case_config.get("metadata", {}) or {}
        ci_tier = str(metadata.get("ci_tier") or "default")
        if item.get("_manifest_only"):
            testcase_cell = _esc(name)
        else:
            testcase_cell = f'<a href="#model-{_esc(name)}">{_esc(name)}</a>'
        rows.append(
            f'<tr class="summary-testcase-row{summary_only}{manifest_only}">'
            f"<td>{testcase_cell}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td>{_esc(ci_tier)}</td>"
            f"<td>{_esc(child_family) or '&mdash;'}</td>"
            f"<td>{_esc(child_task_strategy) or '&mdash;'}</td>"
            f"<td>{_esc(_key_metric(item)) or '&mdash;'}</td>"
            f"<td>{_esc(_total_time(item)) or '&mdash;'}</td>"
            f"</tr>"
        )
    return (
        '<details class="summary-model-details">'
        '<summary class="summary-model-summary">'
        '<span class="summary-model-main">'
        f'<span class="summary-model-title">{_esc(title)}</span>'
        f'<span class="testcase-count">{len(items)} {testcase_label}</span>'
        "</span>"
        f"<span>{_esc(family) or '&mdash;'}</span>"
        f"<span>{_esc(task_strategy) or '&mdash;'}</span>"
        f"<span>{_status_summary(items)}</span>"
        f"<span>{key_metric or '&mdash;'}</span>"
        f"<span>{total_time or '&mdash;'}</span>"
        "</summary>"
        '<div class="summary-subtest-wrap">'
        '<table class="summary-subtest-table">'
        "<thead><tr>"
        "<th>Testcase</th><th>Status</th><th>CI Tier</th><th>Family</th>"
        "<th>Task Strategy</th>"
        "<th>Key Metric</th><th>Time</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div></details>"
    )


def _summary_data_status(members: List[Dict[str, Any]]) -> str:
    return " ".join(sorted(_status_counts(members)))


def _summary_data_name(members: List[Dict[str, Any]]) -> str:
    return " ".join(
        str(member.get("case_name", "")).lower() for member in members if member.get("case_name")
    )


def render_summary_dashboard(
    results: List[Dict[str, Any]],
    model_manifests: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the top-of-page summary table with counters and filters."""
    model_manifests = model_manifests or []
    display_results = _with_declared_testcases(results, model_manifests)
    grouped_results = _grouped_model_results(display_results)
    counts: Dict[str, int] = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for r in results:
        s = r.get("status", "error")
        counts[s] = counts.get(s, 0) + 1

    inventory_counters = ""
    if grouped_results:
        testcase_count = sum(len(testcases) for testcases in grouped_results.values())
        model_label = "Model" if len(grouped_results) == 1 else "Models"
        testcase_label = "Testcase" if testcase_count == 1 else "Testcases"
        inventory_counters = (
            f'<span class="counter model-counter">{len(grouped_results)} {model_label}</span>'
            f'<span class="counter testcase-counter">{testcase_count} {testcase_label}</span>'
        )
    counters = (
        f'<div class="counters">'
        f'<span class="counter pass-counter">{counts["pass"]} Passed</span>'
        f'<span class="counter fail-counter">{counts["fail"]} Failed</span>'
        f'<span class="counter skip-counter">{counts["skip"]} Skipped</span>'
        f'<span class="counter error-counter">{counts["error"]} Error</span>'
        f'<span class="counter total-counter">{len(results)} Results</span>'
        f"{inventory_counters}</div>"
    )

    filters = (
        '<div class="filters">'
        '<input type="text" id="search-box" placeholder="Search models..." '
        'oninput="filterModels()" />'
        '<select id="status-filter" onchange="filterModels()">'
        '<option value="">All</option>'
        '<option value="pass">Pass</option>'
        '<option value="fail">Fail</option>'
        '<option value="skip">Skip</option>'
        '<option value="error">Error</option>'
        "</select>"
        "</div>"
    )

    rows: List[str] = []
    for r, members in _summary_dashboard_items(display_results):
        cc = r.get("case_config", {})
        family = cc.get("family", "")
        task_strategy = cc.get("task_strategy", "")
        km = _key_metric(r)
        tt = _total_time(r)
        row_attrs = (
            f'class="summary-row" data-status="{_esc(_summary_data_status(members))}" '
            f'data-name="{_esc(_summary_data_name(members))}"'
        )
        rows.append(
            f"<tr {row_attrs}>"
        '<td class="summary-model-cell" colspan="6">'
            + _render_summary_model_details(
                _model_group_key(r),
                r,
                members,
                str(family),
                str(task_strategy),
                km,
                tt,
            )
            + "</td></tr>"
        )

    table = (
        '<table class="summary-table" id="summary-table">'
        "<thead><tr>"
        "<th>Model</th><th>Family</th><th>Task Strategy</th>"
        "<th>Status</th><th>Key Metric</th><th>Time</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>"
    )

    return f'<section class="dashboard">{counters}\n{filters}\n{table}</section>'


# ---------------------------------------------------------------------------
# Environment section
# ---------------------------------------------------------------------------


def render_env_section(results: List[Dict[str, Any]]) -> str:
    """Render environment info from the first result that has it."""
    for r in results:
        env = r.get("env_fingerprint", {})
        if env:
            items = []
            for k, v in env.items():
                if k == "timestamp":
                    continue
                items.append(f"<li><strong>{_esc(k)}:</strong> {_esc(v)}</li>")
            return (
                '<section class="env-section">'
                "<h2>Environment</h2>"
                f'<ul class="env-list">{"".join(items)}</ul>'
                "</section>"
            )
    return ""


# ---------------------------------------------------------------------------
# Model-proof provenance and evidence contract
# ---------------------------------------------------------------------------


def _load_optional_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {"load_error": f"{path.name}: {exc}"}
    return payload if isinstance(payload, dict) else {}


def _proof_context(
    status: Dict[str, Any],
    proof: Dict[str, Any],
    selection: Dict[str, Any],
) -> Dict[str, Any]:
    context = dict(status)
    for key in (
        "model", "source_revision", "suite", "runtime_model",
        "runtime_library", "runtime_library_sha256", "sibling_model_count",
        "model_dso_count", "staged_runtime_library_sha256",
        "staged_model_dso_count", "engine_builds_per_model", "engine_build_count",
        "engine_build_verification", "gpu_id", "gpu_resource_class",
        "gpu_slot_ids", "gpu_slots_per_device", "gpu_lease_evidence",
        "min_free_gpu_memory_mib", "gpu_memory_admission",
        "network", "plugin_search", "passed", "e2e_proof_kind",
        "e2e_proof_kinds",
    ):
        if key in proof:
            context[key] = proof[key]
    if selection:
        context["selection"] = selection
    return context


def validate_proof_context(
    status: Dict[str, Any],
    proof: Dict[str, Any],
    selection: Dict[str, Any],
) -> List[str]:
    """Validate provenance required by a successful isolated model proof."""
    issues: List[str] = []
    if not status or status.get("load_error"):
        return [f"Model-proof status is missing or invalid: {status.get('load_error', 'missing')}"]

    validation_rc = status.get("validation_exit_code")
    successful_validation = validation_rc == 0 or proof.get("passed") is True
    for key in ("model", "source_revision", "suite", "steps"):
        if not status.get(key):
            issues.append(f"Model-proof status is missing {key}")

    if not successful_validation:
        return issues

    if not proof or proof.get("load_error"):
        issues.append(
            f"Final proof JSON is missing or invalid: {proof.get('load_error', 'missing')}"
        )
    if not selection or selection.get("load_error"):
        issues.append(
            "Test selection JSON is missing or invalid: "
            f"{selection.get('load_error', 'missing')}"
        )
    if issues:
        return issues

    raw_e2e_cases = selection.get("e2e_cases")
    if not isinstance(raw_e2e_cases, list):
        issues.append("Test selection e2e_cases must be a list")
        e2e_cases: List[Dict[str, Any]] = []
    else:
        e2e_cases = raw_e2e_cases
    raw_steps = status.get("steps")
    if not isinstance(raw_steps, dict):
        issues.append("Model-proof status steps must be an object")
        steps: Dict[str, Any] = {}
    else:
        steps = raw_steps

    if proof.get("passed") is not True:
        issues.append("Final proof JSON does not declare passed=true")
    for key in ("model", "source_revision", "runtime_model", "runtime_library"):
        if not proof.get(key):
            issues.append(f"Final proof JSON is missing {key}")
    digest = str(proof.get("runtime_library_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        issues.append("Final proof JSON has no valid runtime library SHA-256")
    if proof.get("sibling_model_count") != 0:
        issues.append("Final proof JSON does not prove zero sibling models")
    if proof.get("model_dso_count") != 1:
        issues.append("Final proof JSON does not prove exactly one model DSO")
    staged_digest = str(proof.get("staged_runtime_library_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", staged_digest):
        issues.append("Final proof JSON has no valid staged runtime library SHA-256")
    elif staged_digest != digest:
        issues.append("Staged runtime library SHA-256 does not match the built library")
    if proof.get("staged_model_dso_count") != 1:
        issues.append("Final proof JSON does not prove exactly one staged model DSO")
    if proof.get("engine_builds_per_model") != 1:
        issues.append("Final proof JSON does not prove one full bundle build per model")
    selected_build_models = {
        str(case.get("model") or case.get("name") or "")
        for case in e2e_cases
        if isinstance(case, dict) and (case.get("model") or case.get("name"))
    }
    if proof.get("engine_build_count") != len(selected_build_models):
        issues.append(
            "Final proof JSON engine build count does not match selected configurations"
        )
    if proof.get("engine_build_verification") != "engine-build-verification.json":
        issues.append("Final proof JSON does not identify engine build verification evidence")
    gpu_id = str(proof.get("gpu_id") or "")
    if not re.fullmatch(r"[0-9]+", gpu_id):
        issues.append("Final proof JSON has no valid host GPU ID")
    if gpu_id != str(status.get("gpu_id") or ""):
        issues.append("Proof GPU ID does not match model-proof status")
    if gpu_id != str(selection.get("gpu_id") or ""):
        issues.append("Proof GPU ID does not match test selection")
    resource_class = proof.get("gpu_resource_class")
    slot_ids = proof.get("gpu_slot_ids")
    slots_per_device = proof.get("gpu_slots_per_device")
    lease_evidence = proof.get("gpu_lease_evidence")
    if (
        not isinstance(resource_class, str)
        or resource_class not in {"shared", "exclusive_gpu"}
    ):
        issues.append("Final proof JSON has no valid GPU resource class")
    if not isinstance(slots_per_device, int) or isinstance(slots_per_device, bool) \
            or slots_per_device < 1:
        issues.append("Final proof JSON has no valid GPU slots-per-device value")
    valid_slot_ids = (
        isinstance(slot_ids, list)
        and bool(slot_ids)
        and all(
            isinstance(slot, int) and not isinstance(slot, bool)
            for slot in slot_ids
        )
        and len(slot_ids) == len(set(slot_ids))
        and isinstance(slots_per_device, int)
        and not isinstance(slots_per_device, bool)
        and all(0 <= slot < slots_per_device for slot in slot_ids)
    )
    if not valid_slot_ids:
        issues.append("Final proof JSON has no valid unique GPU slot IDs")
    elif resource_class == "shared" and len(slot_ids) != 1:
        issues.append("Shared GPU proof must hold exactly one GPU slot")
    elif resource_class == "exclusive_gpu" and sorted(slot_ids) != list(
        range(slots_per_device)
    ):
        issues.append("Exclusive GPU proof must hold every slot on its GPU")
    if lease_evidence != "gpu-lease.json":
        issues.append("Final proof JSON does not identify GPU lease evidence")
    for label, payload in (
        ("Model-proof status", status),
        ("Test selection", selection),
    ):
        payload_slot_ids = payload.get("gpu_slot_ids")
        payload_slots_per_device = payload.get("gpu_slots_per_device")
        if (
            not isinstance(payload_slots_per_device, int)
            or isinstance(payload_slots_per_device, bool)
            or payload_slots_per_device < 1
        ):
            issues.append(f"{label} has no valid GPU slots-per-device value")
        if (
            not isinstance(payload_slot_ids, list)
            or not payload_slot_ids
            or any(
                not isinstance(slot, int) or isinstance(slot, bool)
                for slot in payload_slot_ids
            )
        ):
            issues.append(f"{label} has no valid GPU slot IDs")
    for field in (
        "gpu_resource_class", "gpu_slot_ids", "gpu_slots_per_device",
        "gpu_lease_evidence",
    ):
        if proof.get(field) != status.get(field):
            issues.append(f"Proof {field} does not match model-proof status")
        if proof.get(field) != selection.get(field):
            issues.append(f"Proof {field} does not match test selection")
    case_requirements: List[int] = []
    case_resource_classes: List[str] = []
    for index, case in enumerate(e2e_cases):
        if not isinstance(case, dict):
            issues.append(f"Selected E2E case {index} must be an object")
            continue
        case_resource = case.get("resource_class")
        case_gpu_resource = case.get("gpu_resource_class")
        for field, value in (
            ("resource_class", case_resource),
            ("gpu_resource_class", case_gpu_resource),
        ):
            if (
                not isinstance(value, str)
                or value not in {"shared", "exclusive_gpu"}
            ):
                issues.append(f"Selected E2E case has an invalid {field}")
        if (
            isinstance(case_resource, str)
            and case_resource in {"shared", "exclusive_gpu"}
            and isinstance(case_gpu_resource, str)
            and case_gpu_resource in {"shared", "exclusive_gpu"}
        ):
            if case_resource != case_gpu_resource:
                issues.append("Selected E2E case GPU resource classes do not match")
            else:
                case_resource_classes.append(case_resource)
        if "min_free_gpu_memory_mib" not in case:
            issues.append("Selected E2E case is missing minimum free GPU memory")
        requirement = case.get("min_free_gpu_memory_mib", 0)
        if (
            not isinstance(requirement, int)
            or isinstance(requirement, bool)
            or requirement < 0
        ):
            issues.append("Selected E2E case has an invalid minimum free GPU memory value")
        else:
            case_requirements.append(requirement)
            if requirement and case_gpu_resource != "exclusive_gpu":
                issues.append(
                    "Selected E2E case requires free GPU memory without exclusive GPU access"
                )
    selection_resource = selection.get("resource_class")
    selection_gpu_resource = selection.get("gpu_resource_class")
    for field, value in (
        ("resource_class", selection_resource),
        ("gpu_resource_class", selection_gpu_resource),
    ):
        if (
            not isinstance(value, str)
            or value not in {"shared", "exclusive_gpu"}
        ):
            issues.append(f"Test selection has an invalid {field}")
    if (
        isinstance(selection_resource, str)
        and selection_resource in {"shared", "exclusive_gpu"}
        and isinstance(selection_gpu_resource, str)
        and selection_gpu_resource in {"shared", "exclusive_gpu"}
    ):
        if selection_resource != selection_gpu_resource:
            issues.append("Test selection GPU resource classes do not match")
        if case_resource_classes:
            required_resource = (
                "exclusive_gpu"
                if "exclusive_gpu" in case_resource_classes
                else "shared"
            )
            if selection_resource != required_resource:
                issues.append(
                    "Test selection GPU resource class does not match selected cases"
                )
    if "min_free_gpu_memory_mib" not in selection:
        issues.append("Test selection is missing minimum free GPU memory")
    min_free_gpu_memory_mib = selection.get("min_free_gpu_memory_mib", 0)
    if (
        not isinstance(min_free_gpu_memory_mib, int)
        or isinstance(min_free_gpu_memory_mib, bool)
        or min_free_gpu_memory_mib < 0
    ):
        issues.append("Test selection has no valid minimum free GPU memory value")
    elif min_free_gpu_memory_mib:
        if min_free_gpu_memory_mib != max(case_requirements, default=0):
            issues.append(
                "Test selection minimum free GPU memory does not match its selected cases"
            )
        if resource_class != "exclusive_gpu":
            issues.append("Minimum free GPU memory requires an exclusive GPU proof")
        for label, payload in (("Proof", proof), ("Model-proof status", status)):
            payload_minimum = payload.get("min_free_gpu_memory_mib")
            if (
                not isinstance(payload_minimum, int)
                or isinstance(payload_minimum, bool)
                or payload_minimum != min_free_gpu_memory_mib
            ):
                issues.append(
                    f"{label} minimum free GPU memory does not match test selection"
                )
        admission_fields = {
            "source",
            "required_free_mib",
            "observed_total_mib",
            "observed_used_mib",
            "observed_free_mib",
        }
        admission = proof.get("gpu_memory_admission")
        if not isinstance(admission, dict):
            issues.append("Proof has no GPU memory admission evidence")
        else:
            if set(admission) != admission_fields:
                issues.append("Proof GPU memory admission has unexpected or missing fields")
            if admission.get("source") not in {
                "nvidia-smi",
                "linux-numa-meminfo",
            }:
                issues.append("Proof GPU memory admission has an invalid source")
            required_free_mib = admission.get("required_free_mib")
            if (
                not isinstance(required_free_mib, int)
                or isinstance(required_free_mib, bool)
                or required_free_mib != min_free_gpu_memory_mib
            ):
                issues.append(
                    "Proof GPU memory admission requirement does not match test selection"
                )
            observations = {
                field: admission.get(field)
                for field in (
                    "observed_total_mib",
                    "observed_used_mib",
                    "observed_free_mib",
                )
            }
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in observations.values()
            ):
                issues.append("Proof GPU memory admission has invalid observed memory values")
            else:
                if (
                    observations["observed_used_mib"]
                    > observations["observed_total_mib"]
                    or observations["observed_free_mib"]
                    > observations["observed_total_mib"]
                ):
                    issues.append("Proof GPU memory admission values are inconsistent")
                if (
                    admission.get("source") == "linux-numa-meminfo"
                    and observations["observed_used_mib"]
                    + observations["observed_free_mib"]
                    != observations["observed_total_mib"]
                ):
                    issues.append("Proof GPU NUMA memory admission values do not reconcile")
                if observations["observed_free_mib"] < min_free_gpu_memory_mib:
                    issues.append(
                        "Proof GPU memory admission is below the selected requirement"
                    )
        for label, payload in (
            ("Model-proof status", status),
            ("Test selection", selection),
        ):
            payload_admission = payload.get("gpu_memory_admission")
            if not isinstance(payload_admission, dict):
                issues.append(f"{label} has no GPU memory admission evidence")
            else:
                if set(payload_admission) != admission_fields:
                    issues.append(
                        f"{label} GPU memory admission has unexpected or missing fields"
                    )
                if payload_admission.get("source") not in {
                    "nvidia-smi",
                    "linux-numa-meminfo",
                }:
                    issues.append(
                        f"{label} GPU memory admission has an invalid source"
                    )
                payload_requirement = payload_admission.get("required_free_mib")
                if (
                    not isinstance(payload_requirement, int)
                    or isinstance(payload_requirement, bool)
                    or payload_requirement != min_free_gpu_memory_mib
                ):
                    issues.append(
                        f"{label} GPU memory admission requirement does not match "
                        "test selection"
                    )
                payload_observations = {
                    field: payload_admission.get(field)
                    for field in (
                        "observed_total_mib",
                        "observed_used_mib",
                        "observed_free_mib",
                    )
                }
                if any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in payload_observations.values()
                ):
                    issues.append(
                        f"{label} GPU memory admission has invalid observed memory values"
                    )
                elif (
                    payload_observations["observed_used_mib"]
                    > payload_observations["observed_total_mib"]
                    or payload_observations["observed_free_mib"]
                    > payload_observations["observed_total_mib"]
                    or payload_observations["observed_free_mib"]
                    < min_free_gpu_memory_mib
                ):
                    issues.append(
                        f"{label} GPU memory admission has inconsistent memory values"
                    )
                elif (
                    payload_admission.get("source") == "linux-numa-meminfo"
                    and payload_observations["observed_used_mib"]
                    + payload_observations["observed_free_mib"]
                    != payload_observations["observed_total_mib"]
                ):
                    issues.append(
                        f"{label} GPU NUMA memory admission values do not reconcile"
                    )
            if payload_admission != admission:
                issues.append(f"{label} GPU memory admission does not match final proof")
    else:
        if max(case_requirements, default=0) != 0:
            issues.append(
                "Test selection minimum free GPU memory does not match its selected cases"
            )
        for label, payload in (("Proof", proof), ("Model-proof status", status)):
            if "min_free_gpu_memory_mib" not in payload:
                issues.append(f"{label} is missing minimum free GPU memory")
            payload_minimum = payload.get("min_free_gpu_memory_mib")
            if (
                not isinstance(payload_minimum, int)
                or isinstance(payload_minimum, bool)
                or payload_minimum < 0
            ):
                issues.append(f"{label} has an invalid minimum free GPU memory value")
            elif payload_minimum != 0:
                issues.append(f"{label} has an unexpected minimum free GPU memory value")
            if payload.get("gpu_memory_admission") is not None:
                issues.append(f"{label} has unexpected GPU memory admission evidence")
        if selection.get("gpu_memory_admission") is not None:
            issues.append("Test selection has unexpected GPU memory admission evidence")
    if proof.get("network") != "disabled" or proof.get("plugin_search") != "strict":
        issues.append("Final proof JSON is missing hermetic network/plugin guarantees")
    if proof.get("model") != status.get("model"):
        issues.append("Proof model does not match model-proof status")
    if proof.get("source_revision") != status.get("source_revision"):
        issues.append("Proof revision does not match the pinned model-proof revision")
    if selection.get("requested_model") != status.get("model"):
        issues.append("Selected model does not match model-proof status")
    if not selection.get("e2e_test") or not e2e_cases:
        issues.append("Test selection does not identify an E2E test and case")

    supported_e2e_proof_kinds = {
        "reference",
        "snapshot_regression",
        "functional_invariant",
    }
    e2e_proof_kind = proof.get("e2e_proof_kind")
    if (
        not isinstance(e2e_proof_kind, str)
        or e2e_proof_kind
        not in supported_e2e_proof_kinds | {"mixed"}
    ):
        issues.append("Final proof JSON has no valid E2E proof-kind classification")
    raw_e2e_proof_kinds = proof.get("e2e_proof_kinds")
    if (
        raw_e2e_proof_kinds is None
        and isinstance(e2e_proof_kind, str)
        and e2e_proof_kind in supported_e2e_proof_kinds
    ):
        e2e_proof_kinds = [e2e_proof_kind]
    elif (
        isinstance(raw_e2e_proof_kinds, list)
        and raw_e2e_proof_kinds
        and all(isinstance(kind, str) for kind in raw_e2e_proof_kinds)
        and raw_e2e_proof_kinds == sorted(set(raw_e2e_proof_kinds))
        and set(raw_e2e_proof_kinds) <= supported_e2e_proof_kinds
    ):
        e2e_proof_kinds = raw_e2e_proof_kinds
    else:
        e2e_proof_kinds = []
        issues.append("Final proof JSON has invalid per-case E2E proof kinds")
    expected_e2e_proof_kind = (
        e2e_proof_kinds[0]
        if len(e2e_proof_kinds) == 1
        else "mixed"
    )
    if e2e_proof_kinds and e2e_proof_kind != expected_e2e_proof_kind:
        issues.append("Aggregate E2E proof kind does not match per-case proof kinds")
    if status.get("e2e_proof_kind") != e2e_proof_kind:
        issues.append("E2E proof kind does not match model-proof status")
    if raw_e2e_proof_kinds is not None and status.get("e2e_proof_kinds") != e2e_proof_kinds:
        issues.append("Per-case E2E proof kinds do not match model-proof status")
    e2e_reference = steps.get("e2e_reference")
    expected_reference_status = (
        "passed" if "reference" in e2e_proof_kinds else "skipped"
    )
    if (
        not isinstance(e2e_reference, dict)
        or e2e_reference.get("status") != expected_reference_status
    ):
        issues.append(f"Validation step e2e_reference must be {expected_reference_status}")

    for name, step in steps.items():
        if name == "html_report" or not isinstance(step, dict):
            continue
        step_status = step.get("status")
        if (
            not isinstance(step_status, str)
            or step_status not in {"passed", "skipped"}
        ):
            issues.append(f"Validation step {name} is not complete")
    return issues


def _load_proof_diagnostics(status_path: Optional[Path]) -> Dict[str, str]:
    """Load bounded, escaped-at-render-time failure excerpts for standalone HTML."""
    if status_path is None or not status_path.is_file():
        return {}
    root = status_path.parent
    diagnostics: Dict[str, str] = {}
    for filename in (
        "configure.log", "build.log", "cpp-tests.log",
        "python-model-tests.log", "e2e.log", "console.log",
    ):
        path = _path_within(root / filename, root)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if text.strip():
            diagnostics[filename] = text[-16000:]

    junit_messages = []
    for relpath in ("python-model-tests.xml", "e2e/junit.xml"):
        path = _path_within(root / relpath, root)
        if path is None:
            continue
        try:
            xml_root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for testcase in xml_root.iter("testcase"):
            for tag in ("failure", "error", "skipped"):
                node = testcase.find(tag)
                if node is None:
                    continue
                name = testcase.attrib.get("name", "unknown")
                message = node.attrib.get("message", "") or (node.text or "")
                junit_messages.append(f"{relpath}: {name}: {tag}: {message}"[:4000])
    if junit_messages:
        diagnostics["JUnit outcomes"] = "\n\n".join(junit_messages[:50])
    return diagnostics


def render_proof_section(context: Dict[str, Any]) -> str:
    if not context:
        return ""
    selection = context.get("selection") or {}
    admission = context.get("gpu_memory_admission")
    admission_summary = None
    if isinstance(admission, dict):
        admission_summary = (
            f"{admission.get('observed_free_mib')} MiB free / "
            f"{admission.get('required_free_mib')} MiB required"
        )
    rows = []
    e2e_proof_kinds = context.get("e2e_proof_kinds")
    proof_kinds_summary = (
        ", ".join(e2e_proof_kinds)
        if isinstance(e2e_proof_kinds, list)
        else None
    )
    reference_parity_claimed = (
        "reference" in e2e_proof_kinds
        if isinstance(e2e_proof_kinds, list)
        else context.get("e2e_proof_kind") == "reference"
    )
    fields = (
        ("Model ownership ID", context.get("model")),
        ("Pinned source revision", context.get("source_revision")),
        ("Suite", context.get("suite")),
        ("Final outcome", context.get("outcome") or (
            "passed" if context.get("passed") else "incomplete")),
        ("Runtime model", context.get("runtime_model")),
        ("Runtime library", context.get("runtime_library")),
        ("Runtime library SHA-256", context.get("runtime_library_sha256")),
        ("Staged runtime library SHA-256", context.get("staged_runtime_library_sha256")),
        ("Sibling models in projection", context.get("sibling_model_count")),
        ("Model DSOs produced", context.get("model_dso_count")),
        ("Model DSOs staged", context.get("staged_model_dso_count")),
        ("Full bundle builds per model", context.get("engine_builds_per_model")),
        ("Full bundle builds recorded", context.get("engine_build_count")),
        ("Engine build verification", context.get("engine_build_verification")),
        ("Host GPU ID", context.get("gpu_id")),
        ("GPU resource class", context.get("gpu_resource_class")),
        ("GPU slot IDs", context.get("gpu_slot_ids")),
        ("GPU slots per device", context.get("gpu_slots_per_device")),
        ("Minimum free GPU memory", context.get("min_free_gpu_memory_mib")),
        ("GPU memory admission", admission_summary),
        ("GPU lease evidence", context.get("gpu_lease_evidence")),
        ("Container network", context.get("network")),
        ("Plugin search", context.get("plugin_search")),
        ("E2E proof kind", context.get("e2e_proof_kind")),
        ("E2E per-case proof kinds", proof_kinds_summary),
        ("E2E reference parity claimed", reference_parity_claimed),
    )
    for label, value in fields:
        if value is None or value == "":
            continue
        rows.append(f"<tr><td>{_esc(label)}</td><td>{_esc(value)}</td></tr>")

    parts = ['<section class="proof-section"><h2>Isolation Proof</h2>']
    if context.get("load_error"):
        parts.append(
            f'<p class="failure-info"><strong>Proof metadata error:</strong> '
            f"{_esc(context['load_error'])}</p>"
        )
    parts.append(
        '<table class="proof-table"><tbody>' + "".join(rows) + "</tbody></table>"
    )
    steps = context.get("steps") or {}
    if isinstance(steps, dict) and steps:
        step_rows = []
        for name, details in steps.items():
            if isinstance(details, dict):
                status = str(details.get("status") or "pending")
                evidence = details.get("evidence") or ""
            else:
                status = str(details)
                evidence = ""
            step_rows.append(
                f"<tr><td>{_esc(str(name).replace('_', ' ').title())}</td>"
                f"<td>{_badge(status)}</td><td>{_esc(evidence)}</td></tr>"
            )
        parts.append("<h4>Validation Steps</h4>")
        parts.append(
            '<table class="metrics-table"><thead><tr><th>Step</th><th>Status</th>'
            '<th>Evidence</th></tr></thead><tbody>' + "".join(step_rows)
            + "</tbody></table>"
        )

    cpp_tests = selection.get("runtime_tests") or []
    python_tests = selection.get("python_tests") or []
    e2e_cases = selection.get("e2e_cases") or []
    if cpp_tests or python_tests or e2e_cases or selection.get("e2e_test"):
        parts.append("<h4>Selected Tests</h4><ul class=\"proof-list\">")
        for test in cpp_tests:
            parts.append(f"<li>C++: <code>{_esc(test)}</code></li>")
        for test in python_tests:
            parts.append(f"<li>Python: <code>{_esc(test)}</code></li>")
        for case in e2e_cases:
            name = case.get("name") if isinstance(case, dict) else case
            parts.append(f"<li>E2E: <code>{_esc(name)}</code></li>")
        if selection.get("e2e_test"):
            parts.append(
                f"<li>E2E test file: <code>{_esc(selection['e2e_test'])}</code></li>"
            )
        parts.append("</ul>")

    evidence_files = context.get("evidence_files") or []
    if evidence_files:
        parts.append("<h4>Raw Evidence Files</h4><ul class=\"proof-list\">")
        for filename in evidence_files:
            parts.append(f"<li><code>{_esc(filename)}</code></li>")
        parts.append("</ul>")
    diagnostics = context.get("diagnostics") or {}
    if diagnostics:
        parts.append("<h4>Failure and Log Excerpts</h4>")
        for label, text in diagnostics.items():
            parts.append(
                f"<details><summary>{_esc(label)}</summary>"
                f"<pre class=\"diagnostic-log\">{_esc(text)}</pre></details>"
            )
    parts.append("</section>")
    return "\n".join(parts)


def render_proof_batch_section(contexts: List[Dict[str, Any]]) -> str:
    """Render a compact batch summary followed by each model's proof.

    A premerge matrix produces one hermetic proof artifact per ownership
    model.  Keeping those proof contexts separate preserves the per-model
    isolation evidence while the rest of this module renders one familiar
    E2E dashboard across all of their testcase results.
    """
    if not contexts:
        return ""

    summary_rows: List[str] = []
    details: List[str] = []
    for context in contexts:
        model = str(context.get("model") or "unknown")
        outcome = str(
            context.get("outcome")
            or ("passed" if context.get("passed") is True else "incomplete")
        )
        selection = context.get("selection") or {}
        selected_cases = selection.get("e2e_cases") or []
        case_count = len(selected_cases) if isinstance(selected_cases, list) else 0
        summary_rows.append(
            "<tr>"
            f"<td><code>{_esc(model)}</code></td>"
            f"<td>{_badge(outcome)}</td>"
            f"<td><code>{_esc(context.get('gpu_id') or '')}</code></td>"
            f"<td>{case_count}</td>"
            f"<td>{_esc(context.get('runtime_library') or '')}</td>"
            "</tr>"
        )
        details.append(
            '<details class="proof-model-details" open>'
            f"<summary><strong>{_esc(model)}</strong> &mdash; "
            f"{_badge(outcome)}</summary>"
            f"{render_proof_section(context)}"
            "</details>"
        )

    return (
        '<section class="proof-batch-section"><h2>Isolated Model Proofs</h2>'
        '<table class="proof-table"><thead><tr>'
        "<th>Model</th><th>Outcome</th><th>GPU</th><th>E2E cases</th>"
        "<th>Runtime library</th></tr></thead><tbody>"
        + "".join(summary_rows)
        + "</tbody></table></section>"
        + "\n".join(details)
    )


def _embeddable(path: Optional[Path]) -> bool:
    if path is None or not _valid_media_file(path):
        return False
    return _MAX_EMBED_BYTES <= 0 or path.stat().st_size <= _MAX_EMBED_BYTES


def _paired_outputs_present(result: Dict[str, Any]) -> bool:
    return any(pair.get("trt") and pair.get("ref") for pair in _stage_pairs(result).values())


def validate_evidence(
    results: List[Dict[str, Any]], project_dir: Optional[Path]
) -> List[str]:
    """Return fail-closed omissions in successful model evidence.

    Failed E2E cases are exempt from modality-specific requirements because
    later artifacts may not exist. Their rendered report is still marked as
    partial evidence. A passing case may not silently omit the user-facing
    TRT/reference evidence appropriate for its task strategy.
    """
    issues: List[str] = []
    for result in results:
        if result.get("status") != "pass":
            continue
        name = str(result.get("case_name") or "unknown")
        cc = result.get("case_config") or {}
        inputs = cc.get("inputs") or {}
        strategy = str(cc.get("task_strategy") or "")
        external_reference = _uses_external_reference(result)
        prefix = f"{name} ({strategy or 'unknown strategy'})"
        if strategy not in _TASK_STRATEGY_TO_MODALITY:
            issues.append(f"{prefix}: task strategy has no explicit report renderer")
            continue

        def require(condition: bool, message: str) -> None:
            if not condition:
                issues.append(f"{prefix}: {message}")

        if strategy in {"text_generation_causal", "diffusion_text_generation"}:
            require(_get_stage_text(result.get("stage_outputs") or {}, "trt_") is not None,
                    "missing TRT/base text output")
            if external_reference:
                require(_get_stage_text(
                    result.get("stage_outputs") or {}, "ref_") is not None,
                    "missing reference text output")
            else:
                require(_paired_outputs_present(result),
                        "missing invariant reference stage evidence")
        elif strategy == "vision_language_generation":
            image = _resolve_input_media(
                _input_ref(inputs, ("image", "image_path", "input_image")), project_dir)
            require(_embeddable(image), "input image is missing or cannot be embedded")
            require(_get_stage_text(result.get("stage_outputs") or {}, "trt_") is not None,
                    "missing TRT/base text output")
            if external_reference:
                require(_get_stage_text(
                    result.get("stage_outputs") or {}, "ref_") is not None,
                    "missing reference text output")
        elif strategy == "diffusion_media_generation":
            art_dir = Path(result.get("_artifact_dir") or ".")
            artifacts = result.get("artifacts") or {}
            trt_frames = _resolve_frame_paths(
                artifacts.get("trt_frames", []), art_dir, "frames")
            ref_frames = _resolve_frame_paths(
                artifacts.get("ref_frames", []), art_dir, "ref_frames")
            require(bool(trt_frames), "missing TRT/base image or video frames")
            if external_reference or _native_visual_acceptance(result) is not None:
                require(bool(ref_frames), "missing reference image or video frames")
            require(all(_embeddable(path) for path in _select_frames(
                trt_frames, _MAX_DIFFUSION_FRAMES)),
                "one or more selected TRT/base frames cannot be embedded")
            if ref_frames:
                require(all(_embeddable(path) for path in _select_frames(
                    ref_frames, _MAX_DIFFUSION_FRAMES)),
                    "one or more selected reference frames cannot be embedded")
            condition_ref = _input_ref(
                inputs, ("image", "image_path", "input_image", "conditioning_image"))
            if condition_ref:
                require(_embeddable(_resolve_input_media(condition_ref, project_dir)),
                        "declared conditioning image is missing or cannot be embedded")
        elif strategy in {"text_to_audio", "speech_to_speech"}:
            trt_audio = _find_output_media(
                result, "trt", ("wav", "audio"),
                ("wav_path", "audio_output_path"))
            ref_audio = _find_output_media(
                result, "ref", ("wav", "audio"),
                ("wav_path", "audio_output_path"))
            require(_embeddable(trt_audio), "missing or unembeddable TRT/base audio")
            if strategy == "speech_to_speech":
                input_audio = _resolve_input_media(_input_ref(
                    inputs, ("audio", "audio_path", "input_audio", "input_audio_path")),
                    project_dir)
                require(_embeddable(input_audio),
                        "missing or unembeddable input/source audio")
                require(_embeddable(ref_audio),
                        "missing or unembeddable reference audio")
            else:
                require(_embeddable(ref_audio),
                        "missing or unembeddable reference audio")
        elif strategy == "speech_to_text":
            input_audio = _resolve_input_media(_input_ref(
                inputs, ("audio", "audio_path", "input_audio", "input_audio_path")),
                project_dir)
            require(_embeddable(input_audio), "missing or unembeddable input/source audio")
            require(_get_stage_text(result.get("stage_outputs") or {}, "trt_") is not None,
                    "missing TRT/base transcript")
            if external_reference:
                require(_get_stage_text(
                    result.get("stage_outputs") or {}, "ref_") is not None,
                    "missing reference transcript")
        elif strategy in {"segmentation", "prompted_segmentation"}:
            input_image = _resolve_input_media(_input_ref(
                inputs, ("image", "image_path", "input_image")), project_dir)
            image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            trt_visuals = _find_output_media_all(
                result, "trt", ("segmented_image", "segmentation_map", "output"),
                ("segmented_image_path", "segmentation_map_path", "viz_path", "output_path"),
                image_suffixes)
            ref_visuals = _find_output_media_all(
                result, "ref", ("segmented_image", "segmentation_map", "output"),
                ("segmented_image_path", "segmentation_map_path", "viz_path", "output_path"),
                image_suffixes)
            require(_embeddable(input_image), "missing or unembeddable input image")
            require(bool(trt_visuals) and all(_embeddable(path) for path in trt_visuals),
                    "missing or unembeddable TRT/base segmentation")
            if external_reference:
                require(bool(ref_visuals) and all(_embeddable(path) for path in ref_visuals),
                        "missing or unembeddable reference segmentation")
        elif strategy == "image_classification":
            image = _resolve_input_media(
                _input_ref(inputs, ("image", "image_path", "input_image")), project_dir)
            require(_embeddable(image), "missing or unembeddable classification input")
            require(_first_stage_value(result, "trt", ("top_class",)) is not None,
                    "missing TRT/base class prediction")
            require(_first_stage_value(result, "trt", ("top_score",)) is not None,
                    "missing TRT/base class score")
            if external_reference:
                require(_first_stage_value(result, "ref", ("top_class",)) is not None,
                        "missing reference class prediction")
                require(_first_stage_value(result, "ref", ("top_score",)) is not None,
                        "missing reference class score")
        elif strategy == "object_detection":
            image = _resolve_input_media(
                _input_ref(inputs, ("image", "image_path", "input_image")), project_dir)
            require(_embeddable(image), "missing or unembeddable detection input")
            require(_first_stage_value(result, "trt", ("detections", "boxes")) is not None,
                    "missing TRT/base detections")
            if external_reference:
                require(_first_stage_value(
                    result, "ref", ("detections", "boxes")) is not None,
                    "missing reference detections")
        elif strategy == "reranking":
            require(bool(_first_stage_value(result, "trt", ("scores",))),
                    "missing TRT/base reranking scores")
            if external_reference:
                require(bool(_first_stage_value(result, "ref", ("scores",))),
                        "missing reference reranking scores")
        elif strategy == "neural_operator":
            trt_field = _first_stage_value(
                result, "trt", ("output_field", "field", "prediction", "forecast",
                                "output_field_preview"))
            ref_field = _first_stage_value(
                result, "ref", ("output_field", "field", "prediction", "forecast",
                                "output_field_preview"))
            require(bool(_numeric_series(trt_field)),
                    "missing numeric TRT/base field output")
            if external_reference:
                require(bool(_numeric_series(ref_field)),
                        "missing numeric reference field output")
        elif strategy == "omni_multimodal":
            require(_paired_outputs_present(result),
                    "missing paired TRT/base and reference stage outputs")
            trt_audio = _find_output_media(
                result, "trt", ("wav", "audio"),
                ("wav_path", "audio_output_path"))
            ref_audio = _find_output_media(
                result, "ref", ("wav", "audio"),
                ("wav_path", "audio_output_path"))
            requires_audio = "audio" in str(cc.get("user_contract") or "").lower()
            if requires_audio or trt_audio is not None or ref_audio is not None:
                require(_embeddable(trt_audio), "missing or unembeddable TRT/base audio")
                require(_embeddable(ref_audio), "missing or unembeddable reference audio")
        elif strategy in {"encoder_only_nlp", "embedding"}:
            trt_feature = _get_stage_feature_output(
                result.get("stage_outputs") or {}, "trt_")
            ref_feature = _get_stage_feature_output(
                result.get("stage_outputs") or {}, "ref_")
            require(
                trt_feature is not None
                and _flatten_numeric_preview(trt_feature[1])[1] > 0,
                "missing TRT/base numeric feature output",
            )
            if external_reference:
                require(
                    ref_feature is not None
                    and _flatten_numeric_preview(ref_feature[1])[1] > 0,
                    "missing reference numeric feature output",
                )
        else:
            require(_paired_outputs_present(result),
                    "missing paired TRT/base and reference structured outputs")
    return issues


def render_evidence_issues(issues: List[str]) -> str:
    if not issues:
        return (
            '<section class="evidence-ok"><h2>Evidence Completeness</h2>'
            '<p>All required user-facing evidence is embedded in this report.</p></section>'
        )
    items = "".join(f"<li>{_esc(issue)}</li>" for issue in issues)
    return (
        '<section class="evidence-fail"><h2>Evidence Completeness</h2>'
        '<p><strong>The report is incomplete.</strong></p>'
        f"<ul>{items}</ul></section>"
    )


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

_REPORT_ASSETS_DIR = Path(__file__).resolve().with_name("generate_e2e_report_assets")
_REPORT_CSS_FILENAME = "e2e_report.css"
_REPORT_JS_FILENAME = "e2e_report.js"


def _load_report_asset(filename: str) -> str:
    """Load a report asset that will be embedded into the generated HTML."""
    asset_path = _REPORT_ASSETS_DIR / filename
    try:
        return asset_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing E2E report asset: {asset_path}") from exc


def _load_report_css() -> str:
    return _load_report_asset(_REPORT_CSS_FILENAME)


def _load_report_js() -> str:
    return _load_report_asset(_REPORT_JS_FILENAME)


def render_report(
    results: List[Dict[str, Any]],
    title: str = "E2E Test Report",
    project_dir: Optional[Path] = None,
    proof_context: Optional[Dict[str, Any]] = None,
    evidence_issues: Optional[List[str]] = None,
    model_manifests: Optional[List[Dict[str, Any]]] = None,
    proof_contexts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the full self-contained HTML report."""
    # Reset command counter for deterministic output.
    global _CMD_COUNTER  # noqa: PLW0603
    _CMD_COUNTER = 0

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8" />')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1" />')
    parts.append(f"<title>{_esc(title)}</title>")
    parts.append(f"<style>{_load_report_css()}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_esc(title)}</h1>")

    if proof_context:
        parts.append(render_proof_section(proof_context))
    if proof_contexts:
        parts.append(render_proof_batch_section(proof_contexts))

    # Timestamp
    if results:
        ts = results[0].get("timestamp", "")
        if ts:
            parts.append(f'<p class="subtitle">Generated from run at {_esc(ts)}</p>')

    # Environment
    parts.append(render_env_section(results))

    if evidence_issues is not None:
        display_issues = list(evidence_issues)
        for result in results:
            if result.get("status") == "pass":
                continue
            cc = result.get("case_config") or {}
            display_issues.append(
                f"{result.get('case_name') or 'unknown'} "
                f"({cc.get('task_strategy') or 'unknown strategy'}): "
                f"E2E status is {result.get('status') or 'unknown'}; "
                "user-facing evidence may be partial"
            )
        parts.append(render_evidence_issues(display_issues))

    # Summary dashboard
    parts.append("<h2>Summary</h2>")
    parts.append(render_summary_dashboard(results, model_manifests))

    # Per-model details
    parts.append("<h2>Model Details</h2>")
    for r in results:
        parts.append(render_model_section(r, project_dir))

    parts.append(f"<script>{_load_report_js()}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from E2E artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Root directory containing per-model result directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Indexed E2E model manifests used to show declared testcases.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project root for resolving relative image paths.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="E2E Test Report",
        help="Report title (shown in header and <title>).",
    )
    parser.add_argument(
        "--proof-status",
        type=Path,
        default=None,
        help="Optional progressive model-proof status JSON.",
    )
    parser.add_argument(
        "--proof-json",
        type=Path,
        default=None,
        help="Optional final model isolation proof JSON.",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Optional selected model tests JSON.",
    )
    parser.add_argument(
        "--strict-evidence",
        action="store_true",
        help="Return non-zero after writing the report if passing cases omit required evidence.",
    )
    parser.add_argument(
        "--max-embed-bytes",
        type=int,
        default=_MAX_EMBED_BYTES,
        help="Maximum bytes per embedded media file; zero means unlimited.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.max_embed_bytes < 0:
        print("ERROR: --max-embed-bytes must be non-negative", file=sys.stderr)
        return 2
    global _MAX_EMBED_BYTES  # noqa: PLW0603
    _MAX_EMBED_BYTES = args.max_embed_bytes

    results = load_all_results(args.artifacts_dir)
    model_manifests = load_model_manifests(args.manifest_dir)
    if not results:
        print(
            f"WARNING: No result.json files found in {args.artifacts_dir}",
            file=sys.stderr,
        )

    issues = validate_evidence(results, args.project_dir)
    status = _load_optional_json(args.proof_status)
    proof = _load_optional_json(args.proof_json)
    selection = _load_optional_json(args.selection_json)
    if args.strict_evidence and any(
        path is not None
        for path in (args.proof_status, args.proof_json, args.selection_json)
    ):
        issues.extend(validate_proof_context(status, proof, selection))
    context = _proof_context(status, proof, selection)
    diagnostics = _load_proof_diagnostics(args.proof_status)
    if diagnostics:
        context["diagnostics"] = diagnostics
    if not results:
        issues.append("No per-model result.json was produced; E2E evidence is unavailable")
    if args.strict_evidence and context:
        report_step = context.setdefault("steps", {}).setdefault("html_report", {})
        report_step["status"] = "failed" if issues else "passed"
        report_step["evidence"] = "model-proof-report.html"
        if issues:
            context["outcome"] = "failed"
            context["exit_code"] = 2
        elif context.get("outcome") == "report-validation":
            context["outcome"] = "passed"
            context["exit_code"] = 0
    html_content = render_report(
        results,
        title=args.title,
        project_dir=args.project_dir,
        proof_context=context,
        evidence_issues=issues,
        model_manifests=model_manifests,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(
        f"Report written to {args.output} ({size_kb:.0f} KB, {len(results)} results)",
        file=sys.stderr,
    )
    if args.strict_evidence and issues:
        for issue in issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

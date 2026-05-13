#!/usr/bin/env python3
"""Generate a self-contained HTML report from E2E test artifacts.

Reads result.json files produced by the unified E2E harness and assembles
a single HTML page with summary dashboard, per-model details, embedded
media (images, audio, video frames), and reproduction commands.

Usage:
    python scripts/generate_e2e_report.py \\
      --artifacts-dir /tmp/e2e_artifacts/artifacts \\
      -o /tmp/e2e_artifacts/e2e_report.html \\
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

# Maximum file size to embed inline (10 MB).
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
_TEST_CASE_RE = re.compile(r"test_e2e\[([^\]]+)\]")
_CONSOLE_OUTCOME_RE = re.compile(
    r"tests/test_e2e\.py::test_e2e\[([^\]]+)\]\s+"
    r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b(.*)"
)
_PYTEST_TO_RESULT_STATUS = {
    "XFAIL": "skip",
    "XPASS": "pass",
}

# ---------------------------------------------------------------------------
# Modality classification
# ---------------------------------------------------------------------------

_TASK_STRATEGY_TO_MODALITY = {
    "text_generation_causal": "text",
    "vision_language_generation": "vl",
    "diffusion_media_generation": "diffusion",
    "text_to_audio": "audio",
    "speech_to_text": "audio",
    "speech_to_speech": "audio",
    "segmentation": "segmentation",
    "prompted_segmentation": "segmentation",
    "encoder_only_nlp": "generic",
    "embedding": "generic",
    "reranking": "generic",
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
    _apply_pytest_waive_outcomes(results, artifacts_dir)
    _attach_diffusion_vlm_assessments(results, artifacts_dir)
    return results


def _e2e_root_from_artifacts_dir(artifacts_dir: Path) -> Path:
    if artifacts_dir.name == "artifacts":
        return artifacts_dir.parent
    return artifacts_dir


def _extract_case_name(text: str) -> str:
    match = _TEST_CASE_RE.search(text)
    return match.group(1) if match else ""


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


def _load_pytest_waive_outcomes(e2e_root: Path) -> Dict[str, Dict[str, str]]:
    outcomes: Dict[str, Dict[str, str]] = {}

    for xml_path in _junit_files(e2e_root):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"WARNING: skipping {xml_path}: {exc}", file=sys.stderr)
            continue
        for testcase in root.iter("testcase"):
            case_name = _extract_case_name(
                " ".join(str(testcase.attrib.get(key, ""))
                         for key in ("classname", "name"))
            )
            if not case_name:
                continue
            skipped = testcase.find("skipped")
            if skipped is None or skipped.attrib.get("type", "") != "pytest.xfail":
                continue
            outcomes[case_name] = {
                "pytest_status": "XFAIL",
                "reason": _clean_pytest_reason(
                    skipped.attrib.get("message", "") or (skipped.text or "")),
                "source": xml_path.name,
            }

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
            if status not in {"XFAIL", "XPASS"}:
                continue
            outcomes[case_name] = {
                "pytest_status": status,
                "reason": _clean_pytest_reason(rest.split("[", 1)[0]),
                "source": log_path.name,
            }

    return outcomes


def _apply_pytest_waive_outcomes(
    results: List[Dict[str, Any]], artifacts_dir: Path
) -> None:
    outcomes = _load_pytest_waive_outcomes(_e2e_root_from_artifacts_dir(artifacts_dir))
    if not outcomes:
        return
    for result in results:
        case_name = str(result.get("case_name") or "")
        outcome = outcomes.get(case_name)
        if not outcome:
            continue
        result["_pytest_outcome"] = outcome
        status = outcome.get("pytest_status", "")
        if status in _PYTEST_TO_RESULT_STATUS:
            result["_raw_status"] = result.get("status")
            result["status"] = _PYTEST_TO_RESULT_STATUS[status]


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
    if not path.is_file():
        return None
    if path.stat().st_size > _MAX_EMBED_BYTES:
        return None
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _mime_for_ext(ext: str) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".wav": "audio/wav",
        ".gif": "image/gif",
    }.get(ext.lower(), "application/octet-stream")


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


def _badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#6b7280")
    return (
        f'<span class="badge" style="background:{color}">'
        f"{html.escape(status.upper())}</span>"
    )


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
    for ref in refs:
        p = Path(ref)
        if not p.is_absolute():
            p = art_dir / ref
        if p.is_dir():
            frame_paths.extend(_list_frames_in_dir(p))
        elif p.is_file():
            frame_paths.append(p)

    if not frame_paths:
        fallback_dir = art_dir / fallback_dir_name
        if fallback_dir.is_dir():
            frame_paths = _list_frames_in_dir(fallback_dir)

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
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
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
    raw_path = Path(ref)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([
            art_dir / raw_path,
            art_dir / raw_path.name,
            art_dir.parent / raw_path.name,
        ])
    for path in candidates:
        if path.is_file():
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
            "crossover_ref_t5_trt_dit",
            "crossover_trt_t5_ref_dit",
            "debug_pipeline",
            "dit_step",
            "full_generation",
            "full_inference",
            "end_to_end",
            "end_to_end_video",
            "frame_quality",
            "generate",
            "t5_encode",
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
        key.startswith("trt_compile_extra_")
        and key != "trt_compile_extra_engines_s"
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
        '<div class="timing-breakdown">'
        + "\n".join(rows)
        + "</div></details>"
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
        for label, value in _aggregate_component_timings(
            details, "trt_component_engine_")
    ]
    if not engine_component_children and "inference_s" in details:
        engine_component_children = [("TRT engine execution", details.get("inference_s"))]

    load_component_children = [
        (
            "load/deserialization: " + _format_load_component_label(label, load_component_stats),
            value,
        )
        for label, value in _aggregate_component_timings(
            details, "trt_component_load_deserialize_")
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


def _render_text_comparison(
    trt_text: Optional[str], ref_text: Optional[str]
) -> str:
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

    norm = sumsq ** 0.5
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
    if image_rel and project_dir:
        img_path = project_dir / image_rel
        uri = encode_file_base64(img_path, _mime_for_ext(img_path.suffix))
        if uri:
            parts.append(
                f'<p><strong>Input Image:</strong></p>'
                f'<img src="{uri}" class="preview-img" />'
            )
        else:
            parts.append(f"<p><em>Image not found: {_esc(image_rel)}</em></p>")

    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    parts.append(_render_text_comparison(trt_text, ref_text))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_diffusion_model(result: Dict[str, Any]) -> str:
    """Render detail section for a diffusion model."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})

    parts = []

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
        parts.append('<div class="frame-compare">')
        for idx, (trt_fp, ref_fp) in enumerate(zip(selected_trt, selected_ref)):
            trt_uri = encode_file_base64(trt_fp, "image/png")
            ref_uri = encode_file_base64(ref_fp, "image/png")
            parts.append('<div class="frame-pair">')
            parts.append(f'<div class="frame-pair-title">Frame {idx + 1}</div>')
            parts.append('<div class="frame-pair-images">')
            if trt_uri:
                parts.append(
                    '<figure><figcaption>TRT</figcaption>'
                    f'<img src="{trt_uri}" class="frame-img" /></figure>')
            else:
                parts.append("<span class='missing'>TRT frame too large</span>")
            if ref_uri:
                parts.append(
                    '<figure><figcaption>Reference</figcaption>'
                    f'<img src="{ref_uri}" class="frame-img" /></figure>')
            else:
                parts.append("<span class='missing'>Reference frame too large</span>")
            parts.append("</div></div>")
        parts.append("</div>")
    elif trt_frame_paths:
        selected = _select_frames(trt_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>TRT Generated Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected:
            uri = encode_file_base64(fp, "image/png")
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" />')
            else:
                parts.append("<span class='missing'>Frame too large</span>")
        parts.append("</div>")
    elif ref_frame_paths:
        selected_ref = _select_frames(ref_frame_paths, _MAX_DIFFUSION_FRAMES)
        parts.append("<h4>Reference Frames</h4>")
        parts.append('<div class="frame-gallery">')
        for fp in selected_ref:
            uri = encode_file_base64(fp, "image/png")
            if uri:
                parts.append(f'<img src="{uri}" class="frame-img" />')
        parts.append("</div>")

    parts.append(_render_diffusion_vlm_assessment(result))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def _render_diffusion_vlm_assessment(result: Dict[str, Any]) -> str:
    assessment = result.get("vlm_assessment")
    if not isinstance(assessment, dict):
        return (
            "<h4>VLM Semantic Assessment</h4>"
            "<p><em>No VLM assessment artifact was found for this model.</em></p>"
        )

    judgment = assessment.get("vlm_judgment", {})
    if not isinstance(judgment, dict):
        judgment = {}
    gate = judgment.get("vlm_gate", {})
    if not isinstance(gate, dict):
        gate = {}

    gate_failed = bool(gate.get("failed", False))
    if gate_failed:
        gate_label = "FAIL"
        gate_cls = "vlm-fail"
    else:
        gate_label = "PASS"
        gate_cls = "vlm-pass"
    reasons = gate.get("reasons") or []
    if isinstance(reasons, list):
        reason_text = (
            "; ".join(str(r) for r in reasons)
            or str(judgment.get("reason", ""))
        )
    else:
        reason_text = str(reasons)

    rows = [
        ("Judge model", assessment.get("model_id", "")),
        ("Semantic similarity", judgment.get("semantic_similarity_0_to_5", "")),
        ("TRT prompt alignment", judgment.get("trt_prompt_alignment_0_to_5", "")),
        ("HF prompt alignment", judgment.get("hf_prompt_alignment_0_to_5", "")),
        ("TRT visual quality", judgment.get("trt_visual_quality_0_to_5", "")),
        ("HF visual quality", judgment.get("hf_visual_quality_0_to_5", "")),
        ("TRT relative to HF", judgment.get("trt_relative_to_hf", "")),
        ("Regression", judgment.get("is_regression", "")),
        ("Gate", gate_label),
        ("Reason", reason_text or judgment.get("reason", "")),
    ]
    table_rows = "\n".join(
        f"<tr><td>{_esc(name)}</td><td>{_format_value(value)}</td></tr>"
        for name, value in rows
        if value not in ("", None)
    )

    descriptions = []
    trt_description = judgment.get("trt_description")
    hf_description = judgment.get("hf_description")
    if trt_description:
        descriptions.append(
            f"<p><strong>TRT description:</strong> {_esc(trt_description)}</p>")
    if hf_description:
        descriptions.append(
            f"<p><strong>HF description:</strong> {_esc(hf_description)}</p>")

    return (
        '<section class="vlm-assessment">'
        '<h4>VLM Semantic Assessment</h4>'
        f'<p class="{gate_cls}"><strong>Gate:</strong> {gate_label}</p>'
        '<table class="vlm-table"><tbody>'
        f"{table_rows}"
        "</tbody></table>"
        + "".join(descriptions)
        + "</section>"
    )


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
        candidates.extend([
            data.get("error"),
            data.get("parse_error"),
            data.get("stderr_truncated"),
            data.get("stderr"),
        ])
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


def render_audio_model(result: Dict[str, Any]) -> str:
    """Render detail section for an audio model (Whisper, Bark, etc.)."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})
    stage_outputs = result.get("stage_outputs", {})
    cc = result.get("case_config", {})
    task_strategy = cc.get("task_strategy", "")

    parts = []

    # For speech_to_text, show transcript comparison
    if task_strategy == "speech_to_text":
        trt_text = _get_stage_text(stage_outputs, "trt_")
        ref_text = _get_stage_text(stage_outputs, "ref_")
        if trt_text or ref_text:
            parts.append("<h4>Transcript Comparison</h4>")
            parts.append(_render_text_comparison(trt_text, ref_text))

    # Embed TRT audio
    trt_wav = artifacts.get("trt_wav", "")
    if trt_wav:
        wav_path = art_dir / trt_wav
        uri = encode_file_base64(wav_path, "audio/wav")
        if uri:
            parts.append("<h4>TRT Audio</h4>")
            parts.append(f'<audio controls src="{uri}"></audio>')
        else:
            parts.append("<p><em>TRT WAV not found or too large.</em></p>")

    # Embed reference audio
    ref_wav = artifacts.get("ref_wav", "")
    if ref_wav:
        wav_path = art_dir / ref_wav
        uri = encode_file_base64(wav_path, "audio/wav")
        if uri:
            parts.append("<h4>Reference Audio</h4>")
            parts.append(f'<audio controls src="{uri}"></audio>')
        else:
            parts.append("<p><em>Reference WAV not found or too large.</em></p>")
    elif task_strategy == "text_to_audio":
        notice = _render_missing_reference_audio_notice(stage_outputs)
        if notice:
            parts.append(notice)

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_segmentation_model(
    result: Dict[str, Any], project_dir: Optional[Path]
) -> str:
    """Render detail section for a segmentation model."""
    art_dir = Path(result.get("_artifact_dir", ""))
    artifacts = result.get("artifacts", {})
    cc = result.get("case_config", {})
    inputs = cc.get("inputs") or {}
    image_rel = inputs.get("image", "")

    parts = []

    # Input image
    if image_rel and project_dir:
        img_path = project_dir / image_rel
        uri = encode_file_base64(img_path, _mime_for_ext(img_path.suffix))
        if uri:
            parts.append(
                f'<p><strong>Input Image:</strong></p>'
                f'<img src="{uri}" class="preview-img" />'
            )

    segmented_image = artifacts.get("trt_segmented_image", "")
    if segmented_image:
        seg_path = art_dir / segmented_image
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>TRT Segmented Image</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    ref_segmented_image = artifacts.get("ref_segmented_image", "")
    if ref_segmented_image:
        seg_path = art_dir / ref_segmented_image
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>Reference Segmented Image</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    # Segmentation map
    seg_map = artifacts.get("trt_segmentation_map", "") or artifacts.get("trt_output", "")
    if seg_map:
        seg_path = art_dir / seg_map
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>TRT Segmentation Map</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    ref_seg_map = artifacts.get("ref_segmentation_map", "")
    if ref_seg_map:
        seg_path = art_dir / ref_seg_map
        uri = encode_file_base64(seg_path, "image/png")
        if uri:
            parts.append("<h4>Reference Segmentation Map</h4>")
            parts.append(f'<img src="{uri}" class="preview-img" />')

    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


def render_generic_model(result: Dict[str, Any]) -> str:
    """Render detail section for generic models (BERT, embedding, etc.)."""
    cc = result.get("case_config", {})
    prompt = (cc.get("inputs") or {}).get("prompt", "")
    stage_outputs = result.get("stage_outputs", {})
    trt_text = _get_stage_text(stage_outputs, "trt_")
    ref_text = _get_stage_text(stage_outputs, "ref_")
    trt_feature = _format_feature_output(
        _get_stage_feature_output(stage_outputs, "trt_"))
    ref_feature = _format_feature_output(
        _get_stage_feature_output(stage_outputs, "ref_"))

    parts = []
    if prompt:
        parts.append(f"<p><strong>Prompt:</strong> {_esc(prompt)}</p>")
    if trt_text or ref_text:
        parts.append(_render_text_comparison(trt_text, ref_text))
    elif trt_feature or ref_feature:
        parts.append(_render_text_comparison(trt_feature, ref_feature))
    parts.append(_render_metrics_table(result.get("stages", {})))
    parts.append(_render_repro_commands(result.get("repro_commands", {})))
    parts.append(_render_timing_sections(result))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-model collapsible section
# ---------------------------------------------------------------------------


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

    header = (
        f'<details id="model-{_esc(name)}">'
        f"<summary>{badge} <strong>{_esc(name)}</strong>"
        f" &mdash; {_esc(family)} / {_esc(task_strategy)}"
    )
    if hf_id:
        header += f" <small>({_esc(hf_id)})</small>"
    header += "</summary>"

    # Failure info
    body_parts = []
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
            f'<p class="failure-info">Failure type: '
            f"<strong>{_esc(failure_type)}</strong></p>"
        )

    # Dispatch to modality renderer
    if modality == "text":
        body_parts.append(render_text_model(result))
    elif modality == "vl":
        body_parts.append(render_vl_model(result, project_dir))
    elif modality == "diffusion":
        body_parts.append(render_diffusion_model(result))
    elif modality == "audio":
        body_parts.append(render_audio_model(result))
    elif modality == "segmentation":
        body_parts.append(render_segmentation_model(result, project_dir))
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


def render_summary_dashboard(results: List[Dict[str, Any]]) -> str:
    """Render the top-of-page summary table with counters and filters."""
    counts: Dict[str, int] = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for r in results:
        s = r.get("status", "error")
        counts[s] = counts.get(s, 0) + 1

    counters = (
        f'<div class="counters">'
        f'<span class="counter pass-counter">{counts["pass"]} Passed</span>'
        f'<span class="counter fail-counter">{counts["fail"]} Failed</span>'
        f'<span class="counter skip-counter">{counts["skip"]} Skipped</span>'
        f'<span class="counter error-counter">{counts["error"]} Error</span>'
        f'<span class="counter total-counter">{len(results)} Total</span>'
        f"</div>"
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
    sorted_results = sorted(results, key=_total_time_sort_key, reverse=True)
    for r in sorted_results:
        name = r.get("case_name", "unknown")
        status = r.get("status", "error")
        cc = r.get("case_config", {})
        family = cc.get("family", "")
        task_strategy = cc.get("task_strategy", "")
        km = _key_metric(r)
        tt = _total_time(r)
        rows.append(
            f'<tr class="summary-row" data-status="{_esc(status)}" '
            f'data-name="{_esc(name.lower())}">'
            f'<td><a href="#model-{_esc(name)}">{_esc(name)}</a></td>'
            f"<td>{_esc(family)}</td>"
            f"<td>{_esc(task_strategy)}</td>"
            f"<td>{_badge(status)}</td>"
            f"<td>{km}</td>"
            f"<td>{tt}</td>"
            f"</tr>"
        )

    table = (
        '<table class="summary-table" id="summary-table">'
        "<thead><tr>"
        "<th>Model</th><th>Family</th><th>Task Strategy</th>"
        "<th>Status</th><th>Key Metric</th><th>Time</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
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
# Full report assembly
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  Helvetica, Arial, sans-serif; background: #f8f9fa; color: #1a1a2e;
  max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 12px; }
h4 { margin: 12px 0 6px; }
.subtitle { color: #6b7280; margin-bottom: 20px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
  color: #fff; font-size: 0.8em; font-weight: 600; }
.counters { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.counter { padding: 6px 16px; border-radius: 8px; font-weight: 600;
  font-size: 0.95em; }
.pass-counter { background: #dcfce7; color: #166534; }
.fail-counter { background: #fee2e2; color: #991b1b; }
.skip-counter { background: #fef9c3; color: #854d0e; }
.error-counter { background: #ffedd5; color: #9a3412; }
.total-counter { background: #e0e7ff; color: #3730a3; }
.filters { display: flex; gap: 8px; margin-bottom: 12px; }
#search-box { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px;
  flex: 1; max-width: 300px; }
#status-filter { padding: 6px 12px; border: 1px solid #d1d5db; border-radius: 6px; }
.summary-table, .metrics-table, .timing-table { width: 100%;
  border-collapse: collapse; margin: 8px 0; font-size: 0.9em; }
.summary-table th, .metrics-table th, .timing-table th { background: #1e293b;
  color: #fff; padding: 8px 12px; text-align: left; }
.summary-table td, .metrics-table td, .timing-table td { padding: 6px 12px;
  border-bottom: 1px solid #e2e8f0; }
.summary-table tbody tr:hover { background: #f1f5f9; }
.metric-pass { background: #f0fdf4; }
.metric-fail { background: #fef2f2; }
.pass-icon { color: #16a34a; font-weight: bold; }
.fail-icon { color: #dc2626; font-weight: bold; }
.total-row td { font-weight: 700; border-top: 2px solid #1e293b; }
details { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  margin: 8px 0; }
details[open] { border-color: #94a3b8; }
summary { padding: 12px 16px; cursor: pointer; font-size: 1em; }
summary:hover { background: #f8fafc; }
.timing-expand, .timing-expand[open] { background: transparent; border: 0;
  border-radius: 0; margin: 0; }
.timing-expand summary { padding: 0; font-size: inherit; font-weight: 600;
  line-height: 1.35; }
.timing-expand summary:hover { background: transparent; }
.timing-breakdown { margin-top: 6px; display: grid; gap: 3px; color: #475569;
  font-size: 0.92em; }
.timing-breakdown-row { display: grid; grid-template-columns: minmax(0, 1fr) auto;
  gap: 16px; align-items: baseline; }
.timing-breakdown-row span:first-child { min-width: 0; overflow-wrap: anywhere; }
.timing-breakdown-row span:last-child { white-space: nowrap;
  font-variant-numeric: tabular-nums; }
.model-body { padding: 12px 16px; }
.failure-info { color: #dc2626; margin-bottom: 8px; }
.waive-info { color: #92400e; margin-bottom: 8px; }
.text-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
  margin: 8px 0; }
.text-col pre { background: #f1f5f9; padding: 12px; border-radius: 6px;
  white-space: pre-wrap; word-break: break-word; font-size: 0.85em;
  max-height: 300px; overflow-y: auto; }
.code-wrap { position: relative; margin: 6px 0; }
.code-wrap pre { background: #1e293b; color: #e2e8f0; padding: 12px;
  border-radius: 6px; overflow-x: auto; font-size: 0.85em; }
.copy-btn { position: absolute; top: 6px; right: 6px; background: #475569;
  color: #fff; border: none; border-radius: 4px; padding: 2px 8px;
  cursor: pointer; font-size: 0.75em; }
.copy-btn:hover { background: #64748b; }
.frame-gallery { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.frame-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px; margin: 8px 0; }
.frame-pair { border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px;
  background: #fff; }
.frame-pair-title { font-weight: 600; font-size: 0.85em; margin-bottom: 6px; }
.frame-pair-images { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.frame-pair figure { margin: 0; }
.frame-pair figcaption { font-size: 0.75em; color: #64748b; margin-bottom: 3px; }
.frame-pair .frame-img { width: 100%; height: auto; max-height: 170px;
  object-fit: contain; background: #f8fafc; }
.frame-img { max-width: 220px; max-height: 180px; border-radius: 4px;
  border: 1px solid #e2e8f0; }
.preview-img { max-width: 400px; max-height: 300px; border-radius: 6px;
  margin: 6px 0; border: 1px solid #e2e8f0; }
.vlm-assessment { margin: 12px 0; padding: 10px; border: 1px solid #e2e8f0;
  border-radius: 6px; background: #f8fafc; }
.vlm-pass { color: #166534; }
.vlm-fail { color: #991b1b; }
.vlm-table { width: 100%; border-collapse: collapse; margin: 6px 0;
  font-size: 0.9em; }
.vlm-table td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; }
.vlm-table td:first-child { width: 220px; color: #475569; font-weight: 600; }
audio { margin: 6px 0; }
.missing { color: #9ca3af; font-style: italic; }
.env-section ul { list-style: none; columns: 2; }
.env-section li { padding: 2px 0; font-size: 0.9em; }
.repro-section { margin-top: 12px; }
@media (max-width: 768px) {
  .text-compare { grid-template-columns: 1fr; }
  .env-section ul { columns: 1; }
}
"""

_JS = """\
function copyCmd(id) {
  var el = document.getElementById(id);
  if (!el) return;
  var text = el.textContent || el.innerText;
  navigator.clipboard.writeText(text).then(function() {
    var btn = el.parentElement.querySelector('.copy-btn');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(function() {
      btn.textContent = 'Copy'; }, 1500); }
  });
}
function filterModels() {
  var q = (document.getElementById('search-box').value || '').toLowerCase();
  var s = document.getElementById('status-filter').value;
  var rows = document.querySelectorAll('.summary-row');
  for (var i = 0; i < rows.length; i++) {
    var name = rows[i].getAttribute('data-name') || '';
    var status = rows[i].getAttribute('data-status') || '';
    var show = (!q || name.indexOf(q) >= 0) && (!s || status === s);
    rows[i].style.display = show ? '' : 'none';
  }
}
"""


def render_report(
    results: List[Dict[str, Any]],
    title: str = "E2E Test Report",
    project_dir: Optional[Path] = None,
) -> str:
    """Assemble the full self-contained HTML report."""
    # Reset command counter for deterministic output.
    global _CMD_COUNTER  # noqa: PLW0603
    _CMD_COUNTER = 0

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en"><head>')
    parts.append('<meta charset="utf-8" />')
    parts.append(
        '<meta name="viewport" content="width=device-width, initial-scale=1" />'
    )
    parts.append(f"<title>{_esc(title)}</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")
    parts.append(f"<h1>{_esc(title)}</h1>")

    # Timestamp
    if results:
        ts = results[0].get("timestamp", "")
        if ts:
            parts.append(f'<p class="subtitle">Generated from run at {_esc(ts)}</p>')

    # Environment
    parts.append(render_env_section(results))

    # Summary dashboard
    parts.append("<h2>Summary</h2>")
    parts.append(render_summary_dashboard(results))

    # Per-model details
    parts.append("<h2>Model Details</h2>")
    for r in results:
        parts.append(render_model_section(r, project_dir))

    parts.append(f"<script>{_JS}</script>")
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    results = load_all_results(args.artifacts_dir)
    if not results:
        print(
            f"WARNING: No result.json files found in {args.artifacts_dir}",
            file=sys.stderr,
        )

    html_content = render_report(
        results,
        title=args.title,
        project_dir=args.project_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(
        f"Report written to {args.output} ({size_kb:.0f} KB, "
        f"{len(results)} models)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

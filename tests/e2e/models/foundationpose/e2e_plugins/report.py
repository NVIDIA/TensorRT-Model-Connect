# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned report evidence for FoundationPose refinement and scoring."""

from __future__ import annotations

import html
import math
from pathlib import Path
import struct
from typing import Any

_MAX_HYPOTHESES = 252

_STYLE = """
<style>
.fp-report{border:1px solid #cbd5e1;border-radius:12px;padding:16px;background:#f8fafc}.fp-heading{display:flex;justify-content:space-between;gap:16px;align-items:start}.fp-heading h4{margin:2px 0 6px}.fp-heading p{margin:0;color:#475569;max-width:820px}.fp-status{padding:7px 12px;border-radius:999px;font-weight:700;color:#fff;background:#b91c1c}.fp-status.pass{background:#15803d}.fp-summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:14px 0}.fp-card{padding:9px;border:1px solid #cbd5e1;border-radius:8px;background:#fff}.fp-card span{display:block;color:#64748b;font-size:.75em}.fp-card strong{font:700 .9em monospace}.fp-table{width:100%;border-collapse:collapse;background:#fff}.fp-table th,.fp-table td{padding:7px;border:1px solid #cbd5e1;text-align:right;font-family:monospace}.fp-table th:first-child,.fp-table td:first-child{text-align:center}.fp-table tr.best{background:#dcfce7}.fp-help{margin-top:12px;color:#475569;font-size:.82em}.fp-metric-fail{color:#991b1b}.fp-metric-pass{color:#166534}@media(max-width:760px){.fp-summary{grid-template-columns:repeat(2,minmax(0,1fr))}.fp-heading{display:block}.fp-status{display:inline-block;margin-top:8px}.fp-table{font-size:.78em}}
</style>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _artifact_file(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the artifact directory") from error
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file")
    return path


def _f32(path: Path, count: int) -> tuple[float, ...]:
    payload = path.read_bytes()
    expected = count * 4
    if len(payload) != expected:
        raise ValueError(f"{path.name} must contain exactly {count} float32 values")
    values = struct.unpack(f"<{count}f", payload)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path.name} contains non-finite values")
    return values


def _metric(result: dict[str, Any], name: str) -> tuple[str, str]:
    metric = (
        ((result.get("stages") or {}).get("synthetic_crop_pose_refinement") or {})
        .get("metrics", {})
        .get(name, {})
    )
    if not isinstance(metric, dict) or not isinstance(metric.get("value"), (int, float)):
        return "unavailable", ""
    css = "fp-metric-pass" if metric.get("passed") is True else "fp-metric-fail"
    return f"{float(metric['value']):.6g}", css


def _card(label: str, value: str, css: str = "") -> str:
    return (
        '<div class="fp-card">'
        f'<span>{_esc(label)}</span><strong class="{_esc(css)}">{_esc(value)}</strong></div>'
    )


def render(result: dict[str, Any], *, project_dir: Path) -> str:
    """Render bounded pose/ranking evidence from the native output artifacts."""
    del project_dir
    try:
        artifact_ref = result.get("_artifact_dir")
        if not isinstance(artifact_ref, str) or not artifact_ref:
            raise ValueError("artifact directory is missing")
        artifact_dir = Path(artifact_ref).resolve(strict=True)
        native_dir = (artifact_dir / "native").resolve(strict=True)
        native_dir.relative_to(artifact_dir)
        if not native_dir.is_dir():
            raise ValueError("native artifact directory is not a directory")
        inputs = (result.get("case_config") or {}).get("inputs") or {}
        count = inputs.get("num_hypotheses")
        if type(count) is not int or not 1 <= count <= _MAX_HYPOTHESES:
            raise ValueError("num_hypotheses must be an integer in [1, 252]")
        candidates = _f32(_artifact_file(native_dir, "candidate_poses.f32"), count * 16)
        refined = _f32(_artifact_file(native_dir, "trt_refined_poses.f32"), count * 16)
        scores = _f32(_artifact_file(native_dir, "trt_scores.f32"), count)
        best = max(range(count), key=scores.__getitem__)
        reference = (
            ((result.get("stage_outputs") or {}).get("ref_synthetic_crop_pose_refinement") or {})
            .get("data", {})
            .get("best_index")
        )
        if type(reference) is not int or not 0 <= reference < count:
            raise ValueError("reference best_index is missing or invalid")
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return f'<p class="missing">FoundationPose evidence unavailable: {_esc(error)}</p>'

    rows = []
    for index, score in enumerate(scores):
        offset = index * 16
        source_xyz = candidates[offset + 3], candidates[offset + 7], candidates[offset + 11]
        refined_xyz = refined[offset + 3], refined[offset + 7], refined[offset + 11]
        shift = math.sqrt(sum((right - left) ** 2 for left, right in zip(source_xyz, refined_xyz)))
        row_class = ' class="best"' if index == best else ""
        rows.append(
            f"<tr{row_class}><td>{index}{' ★' if index == best else ''}</td>"
            f"<td>{score:.6f}</td>"
            f"<td>{source_xyz[0]:.5f}, {source_xyz[1]:.5f}, {source_xyz[2]:.5f}</td>"
            f"<td>{refined_xyz[0]:.5f}, {refined_xyz[1]:.5f}, {refined_xyz[2]:.5f}</td>"
            f"<td>{shift:.6f}</td></tr>"
        )

    pose_error, pose_css = _metric(result, "pose_max_abs_error")
    score_error, score_css = _metric(result, "score_max_abs_error")
    throughput, throughput_css = _metric(result, "tracking_throughput_hz")
    latency, latency_css = _metric(result, "tracking_latency_p95_ms")
    status = str(result.get("status") or "error")
    status_class = " pass" if status == "pass" else ""
    agreement = "yes" if best == reference else "no"
    return (
        _STYLE + '<section class="fp-report"><header class="fp-heading"><div>'
        "<h4>FoundationPose refinement and ranking</h4><p>Deterministic preprocessed "
        "RGB+XYZ crop pairs exercise iterative pose refinement and joint scoring. "
        "Highlighted output is the TensorRT best hypothesis.</p></div>"
        f'<span class="fp-status{status_class}">{_esc(status.upper())}</span></header>'
        '<div class="fp-summary">'
        + _card("Pose max |error|", pose_error, pose_css)
        + _card("Score max |error|", score_error, score_css)
        + _card("Tracking throughput (Hz)", throughput, throughput_css)
        + _card("Tracking p95 (ms)", latency, latency_css)
        + '</div><table class="fp-table"><thead><tr><th>Hypothesis</th><th>Score</th>'
        "<th>Candidate translation (m)</th><th>Refined translation (m)</th>"
        "<th>Translation update (m)</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table><p class="fp-help">TensorRT best index: '
        f"<strong>{best}</strong>; ONNX Runtime best index: <strong>{reference}</strong>; "
        f"best-hypothesis agreement: <strong>{agreement}</strong>. Numerical gates also require "
        "every emitted transform to remain rigid.</p></section>"
    )

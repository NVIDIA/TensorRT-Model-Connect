# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained LFM2-MoE continuation and expert-routing report fragment."""

from __future__ import annotations

import html
from typing import Any

_MAX_TEXT = 4000
_MAX_EXPERT_ROWS = 64

_STYLE = """
<style>
.lfm2moe-report{border:1px solid #cbd5e1;border-radius:12px;padding:16px;background:#f8fafc}
.lfm2moe-report h4{margin:2px 0 6px}
.lfm2moe-report p{color:#475569}
.lfm2moe-table{width:100%;border-collapse:collapse;margin:10px 0;font-size:.86em}
.lfm2moe-table th,.lfm2moe-table td{border:1px solid #cbd5e1;padding:6px 9px;text-align:left;vertical-align:top}
.lfm2moe-table th{background:#eff6ff;white-space:nowrap}
.lfm2moe-table td{background:#fff;word-break:break-word}
.lfm2moe-case{margin-top:14px;padding:12px;border:1px solid #cbd5e1;border-radius:9px;background:#fff}
.lfm2moe-case h5{margin:0 0 6px}
.lfm2moe-routing{margin-top:8px}
.lfm2moe-bars{display:grid;grid-template-columns:auto 1fr auto;gap:3px 8px;align-items:center;font-size:.78em}
.lfm2moe-bar{height:11px;border-radius:3px;background:linear-gradient(90deg,#2563eb,#60a5fa);min-width:2px}
.lfm2moe-note{padding:8px 11px;border-left:4px solid #2563eb;background:#eff6ff;font-size:.84em}
.missing{color:#64748b;font-style:italic}
</style>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _clip(value: Any) -> str:
    text = str(value if value is not None else "")
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT] + " …"
    return text


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _stage_payload(stage_outputs: dict, prefix: str) -> dict:
    """Return the first stage payload whose key starts with ``prefix``."""
    preferred = stage_outputs.get(f"{prefix}full_generation")
    if isinstance(preferred, dict):
        return preferred
    for key in sorted(stage_outputs):
        if str(key).startswith(prefix) and isinstance(stage_outputs[key], dict):
            return stage_outputs[key]
    return {}


def _output_summary(payload: dict) -> tuple[str, str]:
    data = _as_dict(payload.get("data"))
    text = data.get("text")
    if not isinstance(text, str) or not text:
        text = data.get("cpp_text")
    if not isinstance(text, str):
        text = payload.get("text") if isinstance(payload.get("text"), str) else ""
    tokens = data.get("token_ids")
    token_note = ""
    if isinstance(tokens, list) and tokens:
        shown = ", ".join(str(token) for token in tokens[:48])
        if len(tokens) > 48:
            shown += ", …"
        token_note = f"{len(tokens)} tokens: [{shown}]"
    return _clip(text), token_note


def _cases(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    raw_cases = result.get("cases")
    if isinstance(raw_cases, list):
        return [case for case in raw_cases if isinstance(case, dict)]
    return [result]


def _case_prompt(case: dict) -> str:
    inputs = _as_dict(_as_dict(case.get("case_config")).get("inputs"))
    prompt = inputs.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    for prefix in ("trt_", "ref_"):
        data = _as_dict(_stage_payload(_as_dict(case.get("stage_outputs")), prefix).get("data"))
        candidate = data.get("prompt")
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _expert_counts(case: dict) -> list[tuple[str, float]]:
    """Normalize ``moe_expert_counts`` metadata into (label, count) rows."""
    candidates: list[Any] = [case.get("moe_expert_counts")]
    candidates.append(_as_dict(case.get("metadata")).get("moe_expert_counts"))
    stage_outputs = _as_dict(case.get("stage_outputs"))
    for prefix in ("trt_", "ref_"):
        payload = _stage_payload(stage_outputs, prefix)
        candidates.append(_as_dict(payload.get("data")).get("moe_expert_counts"))
        candidates.append(_as_dict(payload.get("metadata")).get("moe_expert_counts"))
    for candidate in candidates:
        rows: list[tuple[str, float]] = []
        if isinstance(candidate, dict):
            for key, value in candidate.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    rows = []
                    break
                if float(value) < 0:
                    rows = []
                    break
                rows.append((f"expert {key}", float(value)))
        elif isinstance(candidate, list):
            for index, value in enumerate(candidate):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    rows = []
                    break
                if float(value) < 0:
                    rows = []
                    break
                rows.append((f"expert {index}", float(value)))
        if rows:
            return rows[:_MAX_EXPERT_ROWS]
    return []


def _routing_section(case: dict) -> str:
    rows = _expert_counts(case)
    if not rows:
        return (
            '<div class="lfm2moe-note">No <code>moe_expert_counts</code> metadata in this '
            "result. Expert-routing capture is available through the family debug runner, "
            "which replays the same bundle step by step.</div>"
        )
    peak = max(count for _, count in rows) or 1.0
    total = sum(count for _, count in rows)
    bars = []
    for label, count in rows:
        width = max(1.0, 100.0 * count / peak)
        shown = f"{count:g}"
        bars.append(
            f"<span>{_esc(label)}</span>"
            f'<span class="lfm2moe-bar" style="width:{width:.2f}%"></span>'
            f"<span>{_esc(shown)}</span>"
        )
    return (
        '<div class="lfm2moe-routing"><strong>Routing summary</strong> '
        f"({len(rows)} experts, {total:g} routed activations)"
        f'<div class="lfm2moe-bars">{"".join(bars)}</div></div>'
    )


def _case_section(case: dict, index: int) -> str:
    name = case.get("case") or case.get("name") or _as_dict(case.get("case_config")).get("name")
    title = _esc(name) if name else f"case {index + 1}"
    stage_outputs = _as_dict(case.get("stage_outputs"))
    trt_text, trt_tokens = _output_summary(_stage_payload(stage_outputs, "trt_"))
    ref_text, ref_tokens = _output_summary(_stage_payload(stage_outputs, "ref_"))
    rows = (
        ("Prompt", _case_prompt(case), ""),
        ("TensorRT output", trt_text, trt_tokens),
        ("Reference output", ref_text, ref_tokens),
    )
    cells = []
    for label, text, note in rows:
        body = _esc(text) if text else '<span class="missing">unavailable</span>'
        if note:
            body += f'<br /><span class="missing">{_esc(note)}</span>'
        cells.append(f"<tr><th>{_esc(label)}</th><td>{body}</td></tr>")
    return (
        f'<article class="lfm2moe-case"><h5>{title}</h5>'
        f'<table class="lfm2moe-table">{"".join(cells)}</table>'
        f"{_routing_section(case)}</article>"
    )


def render(result: Any, *, project_dir: Any) -> str:
    """Render an HTML fragment for LFM2-MoE results; degrade, never raise."""
    del project_dir
    try:
        cases = _cases(result)
        if not cases:
            return (
                _STYLE + '<section class="lfm2moe-report"><p class="missing">'
                "LFM2-MoE report: no result payload available.</p></section>"
            )
        sections = "".join(_case_section(case, index) for index, case in enumerate(cases))
        return (
            _STYLE + '<section class="lfm2moe-report"><h4>LFM2-MoE continuation report</h4>'
            "<p>TensorRT and pinned Transformers continuations for each testcase, with "
            "per-case expert-routing usage when the result carries "
            "<code>moe_expert_counts</code>.</p>" + sections + "</section>"
        )
    except Exception as error:  # noqa: BLE001 - report hooks must never raise
        return f'<p class="missing">LFM2-MoE report unavailable: {_esc(error)}</p>'

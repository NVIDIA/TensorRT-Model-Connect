# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion VLM semantic assessment report component."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import html
from typing import Any


@dataclass(frozen=True)
class VlmGate:
    """Normalized VLM gate state."""

    failed: bool
    label: str
    css_class: str
    reason_text: str


@dataclass(frozen=True)
class VlmAssessment:
    """Normalized fields rendered in the VLM assessment block."""

    model_id: Any
    judgment: Mapping[str, Any]
    gate: VlmGate


def _esc(text: Any) -> str:
    return html.escape(str(text)) if text is not None else ""


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return _esc(value)


def _reason_text(reasons: Any, judgment: Mapping[str, Any]) -> str:
    if isinstance(reasons, list):
        return "; ".join(str(reason) for reason in reasons) or str(
            judgment.get("reason", "")
        )
    if reasons:
        return str(reasons)
    return str(judgment.get("reason", ""))


def normalize_assessment(raw: Any) -> VlmAssessment | None:
    """Normalize a raw result-level VLM assessment payload for rendering."""
    if not isinstance(raw, Mapping):
        return None

    raw_judgment = raw.get("vlm_judgment", {})
    judgment: Mapping[str, Any] = (
        raw_judgment if isinstance(raw_judgment, Mapping) else {}
    )

    raw_gate = judgment.get("vlm_gate", {})
    gate_data: Mapping[str, Any] = raw_gate if isinstance(raw_gate, Mapping) else {}
    failed = bool(gate_data.get("failed", False))
    label = "FAIL" if failed else "PASS"
    css_class = "vlm-fail" if failed else "vlm-pass"

    return VlmAssessment(
        model_id=raw.get("model_id", ""),
        judgment=judgment,
        gate=VlmGate(
            failed=failed,
            label=label,
            css_class=css_class,
            reason_text=_reason_text(gate_data.get("reasons"), judgment),
        ),
    )


def render_diffusion_vlm_assessment(result: Mapping[str, Any]) -> str:
    """Render the diffusion VLM semantic assessment section for a model result."""
    assessment = normalize_assessment(result.get("vlm_assessment"))
    if assessment is None:
        return (
            "<h4>VLM Semantic Assessment</h4>"
            "<p><em>No VLM assessment artifact was found for this model.</em></p>"
        )

    judgment = assessment.judgment
    rows = [
        ("Judge model", assessment.model_id),
        ("Semantic similarity", judgment.get("semantic_similarity_0_to_5", "")),
        ("TRT prompt alignment", judgment.get("trt_prompt_alignment_0_to_5", "")),
        ("HF prompt alignment", judgment.get("hf_prompt_alignment_0_to_5", "")),
        ("TRT visual quality", judgment.get("trt_visual_quality_0_to_5", "")),
        ("HF visual quality", judgment.get("hf_visual_quality_0_to_5", "")),
        ("TRT relative to HF", judgment.get("trt_relative_to_hf", "")),
        ("Regression", judgment.get("is_regression", "")),
        ("Gate", assessment.gate.label),
        ("Reason", assessment.gate.reason_text or judgment.get("reason", "")),
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
            f"<p><strong>TRT description:</strong> {_esc(trt_description)}</p>"
        )
    if hf_description:
        descriptions.append(
            f"<p><strong>HF description:</strong> {_esc(hf_description)}</p>"
        )

    return (
        '<section class="vlm-assessment">'
        "<h4>VLM Semantic Assessment</h4>"
        f'<p class="{assessment.gate.css_class}"><strong>Gate:</strong> '
        f"{assessment.gate.label}</p>"
        '<table class="vlm-table"><tbody>'
        f"{table_rows}"
        "</tbody></table>"
        + "".join(descriptions)
        + "</section>"
    )

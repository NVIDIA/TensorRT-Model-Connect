#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, dependency-free HTML primitives for standalone TRTMC reports."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Iterable, Mapping, Sequence


COMMON_REPORT_STYLES = """
:root {
  --bg: #f3f5f2;
  --panel: #ffffff;
  --panel-raised: #f8faf7;
  --line: rgba(24, 39, 26, 0.10);
  --line-strong: rgba(24, 39, 26, 0.18);
  --text: #18201a;
  --muted: #6c756e;
  --green: #5c9600;
  --green-bright: #76b900;
  --green-dark: #315500;
  --amber: #a96f00;
  --danger: #b93434;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { background: var(--bg); }
body {
  min-width: 320px;
  margin: 0;
  padding: 28px;
  color: var(--text);
  background:
    linear-gradient(rgba(24, 39, 26, 0.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(24, 39, 26, 0.018) 1px, transparent 1px),
    radial-gradient(circle at 50% -20%, #e3eddd 0, var(--bg) 43%);
  background-size: 48px 48px, 48px 48px, auto;
  font-size: 14px;
  line-height: 1.45;
}
button, input, select { font: inherit; }
button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible,
summary:focus-visible {
  outline: 2px solid var(--green-bright);
  outline-offset: 2px;
}
.report-header { margin-bottom: 18px; }
.report-eyebrow {
  margin: 0 0 7px;
  color: var(--green);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}
h1 { margin: 0 0 5px; font-size: clamp(25px, 3vw, 34px); line-height: 1.15; }
.purpose, .meta, .summary { color: var(--muted); }
.purpose { margin: 0; font-size: 15px; }
.meta { margin: 5px 0 14px; }
.summary { margin: 0 0 20px; }
.campaign-duration { margin: 12px 0 18px; font-size: 18px; }
.traffic-summary {
  display: inline-flex;
  gap: 16px;
  align-items: center;
  margin: 0 0 12px;
  padding: 9px 13px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 8px 26px rgba(35, 48, 37, 0.07);
  font-size: 16px;
  font-weight: 700;
}
.filters {
  position: sticky;
  z-index: 10;
  top: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: end;
  margin: 16px 0;
  padding: 12px;
  border: 1px solid var(--line-strong);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 10px 30px rgba(35, 48, 37, 0.09);
  backdrop-filter: blur(14px);
}
.filters label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 750;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.filters input, .filters select, .filters button {
  min-height: 36px;
  padding: 7px 10px;
  border: 1px solid var(--line-strong);
  border-radius: 9px;
  color: var(--text);
  background: var(--panel);
  font-size: 13px;
  letter-spacing: normal;
  text-transform: none;
}
.filters input { min-width: 250px; }
.filters select { min-width: 145px; }
.filters button { cursor: pointer; font-weight: 700; }
.filters button:hover { border-color: rgba(92, 150, 0, 0.45); background: #f2f8ea; }
.filter-count { margin: 0 0 8px auto; color: var(--muted); font-size: 13px; }
.table-wrap {
  width: 100%;
  overflow: auto;
  border: 1px solid var(--line-strong);
  border-radius: 14px;
  background: var(--panel);
  box-shadow: 0 14px 40px rgba(35, 48, 37, 0.08);
}
table { width: 100%; border-collapse: separate; border-spacing: 0; }
th, td {
  padding: 8px 10px;
  border: 0;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th:last-child, td:last-child { border-right: 0; }
tbody tr:last-child td { border-bottom: 0; }
th {
  color: #3f4a42;
  background: #edf1eb;
  font-size: 12px;
  font-weight: 750;
  white-space: nowrap;
}
tbody tr:nth-child(even) { background: var(--panel-raised); }
tbody tr:hover { background: #f2f8ea; }
a, summary { color: var(--green-dark); }
summary { cursor: pointer; font-weight: 650; }
pre {
  margin: 4px 0 10px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #f4f6f3;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
code { font-size: 12px; }
.detail, .timing-meta { margin-top: 3px; color: var(--muted); font-size: 11px; }
.unavailable { color: var(--muted); }
@media (max-width: 720px) {
  body { padding: 18px 12px; }
  .filters { position: static; }
  .filters label, .filters input, .filters select { width: 100%; min-width: 0; }
  .filter-count { width: 100%; margin-left: 0; }
}
"""


TASK_TYPE_BY_USER_CONTRACT = {
    "multiple_choice_qa": "Text → Choice",
    "continuation_parity": "Text → Text",
    "code_completion": "Text → Code",
    "translation": "Text → Text",
    "seq2seq_output": "Text → Text",
    "any-to-any": "Image + Text → Text",
    "speech_response": "Audio → Audio",
    "vl_answer": "Image + Text → Text",
    "ocr_text": "Image → Text",
    "exact_transcript": "Audio → Text",
    "tts_audio": "Text → Audio",
    "diffusion_image": "Text → Image",
    "diffusion_video": "Text → Video",
    "image-to-image": "Image + Text → Image",
    "image_classification": "Image → Class",
    "segmentation_mask": "Image → Mask",
    "prompted_mask": "Image + Prompt → Mask",
    "time_series_prediction_parity": "Time Series → Time Series",
    "encoder_embedding_parity": "Text → Embedding",
    "embedding_vector": "Text → Embedding",
    "ranking_order": "Query + Documents → Ranking",
    "diffusion_text_generation": "Text → Text",
    "stereo_disparity": "Stereo Images → Disparity",
}

TASK_TYPE_BY_TASK_STRATEGY = {
    "text_generation_causal": "Text → Text",
    "diffusion_text_generation": "Text → Text",
    "vision_language_generation": "Image + Text → Text",
    "text_to_audio": "Text → Audio",
    "speech_to_speech": "Audio → Audio",
    "speech_to_text": "Audio → Text",
    "image_classification": "Image → Class",
    "image_feature_extraction": "Image → Features",
    "segmentation": "Image → Mask",
    "prompted_segmentation": "Image + Prompt → Mask",
    "neural_operator": "Time Series → Time Series",
    "encoder_only_nlp": "Text → Embedding",
    "embedding": "Text → Embedding",
    "reranking": "Query + Documents → Ranking",
    "stereo_disparity": "Stereo Images → Disparity",
}


def task_type_label(
    *,
    user_contract: object = "",
    task_strategy: object = "",
    operation: object = "",
    request: object = None,
) -> str:
    """Resolve the common user-facing task taxonomy for report rows."""
    contract = str(user_contract or "")
    if contract in TASK_TYPE_BY_USER_CONTRACT:
        return TASK_TYPE_BY_USER_CONTRACT[contract]

    strategy = str(task_strategy or "")
    if strategy == "diffusion_media_generation":
        request_fields = request if isinstance(request, Mapping) else {}
        if request_fields.get("media_type") == "video" or request_fields.get(
            "video_num_frames"
        ):
            return "Text → Video"
        if request_fields.get("image_path"):
            return "Image + Text → Image"
        return "Text → Image"
    if strategy == "omni_multimodal":
        return (
            "Text → Audio" if str(operation or "") == "generate_audio" else "Any → Any"
        )
    if strategy in TASK_TYPE_BY_TASK_STRATEGY:
        return TASK_TYPE_BY_TASK_STRATEGY[strategy]

    return {
        "classify": "Image → Class",
        "extract_features": "Image → Features",
        "embed": "Text → Embedding",
        "encode": "Text → Embedding",
        "generate_audio": "Text → Audio",
        "generate_image": "Text → Image",
        "rerank": "Query + Documents → Ranking",
        "segment": "Image → Mask",
        "segment_prompted": "Image + Prompt → Mask",
        "solve": "Time Series → Time Series",
        "speak": "Audio → Audio",
        "transcribe": "Audio → Text",
    }.get(str(operation or ""), "")


@dataclass(frozen=True)
class ReportFilter:
    """One exact-match select in the shared report filter bar."""

    key: str
    label: str
    options: Sequence[str | tuple[str, str]]
    all_label: str = "All"
    token_values: bool = False


def sorted_filter_values(values: Iterable[object]) -> tuple[str, ...]:
    """Return stable, non-empty display values for a report select."""
    return tuple(
        sorted({str(value) for value in values if value is not None and str(value)})
    )


def _filter_id(key: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]*", key):
        raise ValueError(f"invalid report filter key: {key!r}")
    return f"report-filter-{key}"


def render_report_filters(
    *,
    row_count: int,
    search_placeholder: str,
    filters: Sequence[ReportFilter],
) -> str:
    """Render the common search/select/reset bar used by report tables."""
    controls = [
        "<label>Search\n"
        f'<input id="report-filter-search" data-report-filter data-filter-key="search" '
        f'type="search" placeholder="{html.escape(search_placeholder, quote=True)}">\n'
        "</label>"
    ]
    for report_filter in filters:
        control_id = _filter_id(report_filter.key)
        mode = "tokens" if report_filter.token_values else "exact"
        options = [f'<option value="">{html.escape(report_filter.all_label)}</option>']
        for option in report_filter.options:
            value, label = option if isinstance(option, tuple) else (option, option)
            options.append(
                f'<option value="{html.escape(str(value), quote=True)}">'
                f"{html.escape(str(label))}</option>"
            )
        controls.append(
            f"<label>{html.escape(report_filter.label)}\n"
            f'<select id="{control_id}" data-report-filter '
            f'data-filter-key="{html.escape(report_filter.key, quote=True)}" '
            f'data-filter-mode="{mode}">{"".join(options)}</select>\n'
            "</label>"
        )
    controls.extend(
        [
            '<button id="report-filter-reset" type="button">Reset</button>',
            '<span class="filter-count" id="report-filter-count">'
            f"Showing {row_count} of {row_count} rows</span>",
        ]
    )
    return (
        '<div class="filters" aria-label="Report filters">\n'
        + "\n".join(controls)
        + "\n</div>"
    )


def render_report_filter_script() -> str:
    """Render shared dependency-free filtering behavior for annotated rows."""
    return """<script>
(() => {
  const controls = Array.from(document.querySelectorAll("[data-report-filter]"));
  const reset = document.getElementById("report-filter-reset");
  const count = document.getElementById("report-filter-count");
  const rows = Array.from(document.querySelectorAll("tbody tr[data-filter-search]"));
  const applyFilters = () => {
    let visible = 0;
    for (const row of rows) {
      const matches = controls.every((control) => {
        const value = control.value.trim().toLowerCase();
        if (!value) return true;
        const key = control.dataset.filterKey;
        const candidate = (row.getAttribute("data-filter-" + key) || "").toLowerCase();
        if (key === "search") return candidate.includes(value);
        if (control.dataset.filterMode === "tokens") {
          return candidate.split(/\\s+/).includes(value);
        }
        return candidate === value;
      });
      row.hidden = !matches;
      if (matches) visible += 1;
    }
    count.textContent = "Showing " + visible + " of " + rows.length + " rows";
  };
  for (const control of controls) {
    control.addEventListener(control.type === "search" ? "input" : "change", applyFilters);
  }
  reset.addEventListener("click", () => {
    for (const control of controls) control.value = "";
    applyFilters();
    document.getElementById("report-filter-search").focus();
  });
})();
</script>"""

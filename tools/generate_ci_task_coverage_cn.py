#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the Chinese TRTMC CI task coverage report."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - used by Python 3.10 CI images
    import tomli as tomllib  # type: ignore[import-not-found]

from tests.e2e_harness.manifest_loader import iter_manifest_paths  # noqa: E402


DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"
DEFAULT_OUTPUT = REPO_ROOT / "trtmc_ci_task_coverage_cn.html"
MULTI_DEVICE_TIER = "multi_device"
L0_ONLY_TIER = "l0_only"

TASK_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "diffusion_media_generation",
            "diffusion_text_generation",
            "embedding",
            "encoder_only_nlp",
            "image_classification",
            "image_feature_extraction",
            "neural_operator",
            "omni_multimodal",
            "prompted_segmentation",
            "reranking",
            "segmentation",
            "speech_to_speech",
            "speech_to_text",
            "text_generation_causal",
            "text_to_audio",
            "vision_language_generation",
        )
    )
}


@dataclass(frozen=True)
class CaseRecord:
    name: str
    hf_id: str
    family: str
    runtime_strategy: str
    task_strategy: str
    reference_backend: str
    oracle_level: str
    user_contract: str
    ci_tier: str
    skip_reason: str
    xfail_reason: str
    l0_replacement: str
    l0_replacement_reason: str
    manifest_path: str
    raw: dict[str, Any]


def _html(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _model_test_dir_from_manifest_path(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent
    return manifest_path.parent


def _load_model_defaults(manifest_path: Path, task_strategy: str) -> dict[str, Any]:
    index_path = _model_test_dir_from_manifest_path(manifest_path) / "MODEL.toml"
    if not index_path.is_file():
        return {}
    with index_path.open("rb") as handle:
        data = tomllib.load(handle)
    defaults = data.get("e2e_defaults", {})
    if not isinstance(defaults, dict):
        return {}
    task_defaults = defaults.get(task_strategy, {})
    return task_defaults if isinstance(task_defaults, dict) else {}


def _default_reference_backend(task_strategy: str) -> str:
    if task_strategy == "diffusion_media_generation":
        return "hf_diffusers"
    if task_strategy == "diffusion_text_generation":
        return "invariant_only"
    if task_strategy == "neural_operator":
        return "torch_reference"
    return "hf_transformers"


def _default_oracle_level(reference_backend: str) -> str:
    if reference_backend == "invariant_only":
        return "L4_invariants"
    if reference_backend == "golden_snapshot":
        return "L3_snapshot_regression"
    return "L1_external_reference"


def _load_record(manifest_path: Path) -> CaseRecord:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_strategy = str(raw.get("task_strategy") or raw.get("runtime_strategy") or "")
    defaults = _load_model_defaults(manifest_path, task_strategy)
    reference_backend = str(
        raw.get("reference_backend")
        or defaults.get("reference_backend")
        or _default_reference_backend(task_strategy)
    )
    if raw.get("oracle_level"):
        oracle_level = str(raw["oracle_level"])
    elif raw.get("reference_backend"):
        oracle_level = _default_oracle_level(reference_backend)
    else:
        oracle_level = str(
            defaults.get("oracle_level") or _default_oracle_level(reference_backend)
        )
    try:
        manifest_display = manifest_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        manifest_display = manifest_path.as_posix()

    return CaseRecord(
        name=str(raw.get("name") or manifest_path.stem),
        hf_id=str(raw.get("hf_id") or raw.get("model_id") or ""),
        family=str(raw.get("family") or ""),
        runtime_strategy=str(raw.get("runtime_strategy") or ""),
        task_strategy=task_strategy,
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        user_contract=str(raw.get("user_contract") or ""),
        ci_tier=str(raw.get("ci_tier") or "default"),
        skip_reason=str(raw.get("skip_reason") or raw.get("skip") or ""),
        xfail_reason=str(raw.get("xfail_reason") or ""),
        l0_replacement=str(raw.get("l0_replacement") or ""),
        l0_replacement_reason=str(raw.get("l0_replacement_reason") or ""),
        manifest_path=manifest_display,
        raw=raw,
    )


def _load_records(models_dir: Path) -> list[CaseRecord]:
    return [_load_record(path) for path in iter_manifest_paths(models_dir)]


def _bucket(record: CaseRecord) -> str:
    task = record.task_strategy
    contract = record.user_contract
    runtime = record.runtime_strategy
    name = record.name

    if task == "speech_to_text":
        return "Audio / Automatic Speech Recognition"
    if task == "text_to_audio":
        return "Audio / Text-to-Speech"
    if task == "speech_to_speech":
        return "Audio / Audio-to-Audio"
    if task == "omni_multimodal":
        return "Multimodal / Omni Audio Generation"
    if task == "diffusion_media_generation":
        if contract == "image-to-image":
            return "Computer Vision / Image-to-Image"
        if (
            contract == "diffusion_video"
            or runtime in {"diffusion_wan", "diffusion_ltx"}
            or "-t2v" in name.lower()
        ):
            return "Computer Vision / Text-to-Video"
        return "Computer Vision / Text-to-Image"
    if task == "diffusion_text_generation":
        return "NLP / Text Generation (diffusion-text, invariant)"
    if task == "embedding":
        return "NLP or Multimodal / Feature Extraction or Sentence Similarity"
    if task == "reranking":
        return "NLP / Text Ranking or Multimodal Retrieval"
    if task == "image_classification":
        return "Computer Vision / Image Classification"
    if task == "image_feature_extraction":
        return "Computer Vision / Image Feature Extraction"
    if task == "segmentation":
        return "Computer Vision / Image Segmentation"
    if task == "prompted_segmentation":
        return "Computer Vision / Mask Generation"
    if task == "neural_operator":
        if contract == "time_series_quantile_forecast":
            return "Other / Time-Series Forecasting (quantile)"
        if contract == "time_series_regression":
            return "Other / Time-Series Regression"
        return "Other / Time-Series Forecasting"
    if task == "vision_language_generation":
        if contract == "ocr_text":
            return "Multimodal / Image-Text-to-Text (OCR)"
        return "Multimodal / Image-Text-to-Text / VQA"
    if task == "encoder_only_nlp":
        if contract == "embedding_vector":
            return "NLP / Sentence Similarity or Feature Extraction"
        return "NLP / Feature Extraction"
    if task == "text_generation_causal":
        if contract == "chat_response":
            return "NLP / Text Generation (chat/instruct)"
        if contract == "code_completion":
            return "NLP / Text Generation (code completion)"
        if contract == "model_card_generation_parity":
            return "NLP / Text Generation (diffusion-text, invariant)"
        if contract == "sampling":
            return "NLP / Text Generation (sampling/invariant)"
        if contract == "seq2seq_output":
            return "NLP / Text2Text Generation"
        if contract == "translation":
            return "NLP / Translation"
        if name.startswith("t5-"):
            return "NLP / Text2Text Generation"
        if "translation" in runtime:
            return "NLP / Translation"
        return "NLP / Text Generation"
    return f"Other / {task or 'Unclassified'}"


def _area(bucket: str) -> str:
    if bucket.startswith("NLP or Multimodal"):
        return "NLP / Multimodal"
    return bucket.split(" / ", 1)[0]


def _waive(record: CaseRecord) -> tuple[str, str]:
    if record.xfail_reason:
        return "XFAIL", record.xfail_reason
    if record.skip_reason:
        return "SKIP", record.skip_reason
    return "通过门禁", ""


def _row_sort_key(record: CaseRecord) -> tuple[int, str, str]:
    return (TASK_ORDER.get(record.task_strategy, 999), _bucket(record), record.name)


def _nightly_records(records: list[CaseRecord]) -> list[CaseRecord]:
    return sorted(
        (
            record
            for record in records
            if record.ci_tier not in {MULTI_DEVICE_TIER, L0_ONLY_TIER}
        ),
        key=_row_sort_key,
    )


def _premerge_records(
    records: list[CaseRecord],
) -> tuple[list[CaseRecord], list[tuple[CaseRecord, CaseRecord | None]]]:
    by_name = {record.name: record for record in records}
    selected_names: set[str] = set()
    replacements: list[tuple[CaseRecord, CaseRecord | None]] = []
    for record in sorted(records, key=lambda item: item.name):
        if record.ci_tier == MULTI_DEVICE_TIER:
            continue
        if record.l0_replacement:
            selected_names.add(record.l0_replacement)
            replacements.append((record, by_name.get(record.l0_replacement)))
        else:
            selected_names.add(record.name)

    selected = [by_name[name] for name in selected_names if name in by_name]
    return sorted(selected, key=_row_sort_key), replacements


def _counter_table(counter: Counter[str]) -> str:
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(f"<tr><td>{_html(key)}</td><td class=\"num\">{count}</td></tr>")
    return "\n".join(rows)


def _summary_table(title: str, note: str, counter: Counter[str]) -> str:
    return (
        f"<div class=\"card\"><h2>{_html(title)}</h2>"
        f"<p class=\"note\">{_html(note)}</p>"
        "<table class=\"summary-table\"><thead><tr><th>分类</th>"
        "<th class=\"num\">case / manifest 数</th></tr></thead><tbody>"
        f"{_counter_table(counter)}</tbody></table></div>"
    )


def _detail_rows(view: str, records: list[CaseRecord]) -> str:
    rows = []
    for record in records:
        bucket = _bucket(record)
        waive, reason = _waive(record)
        pill = "ok" if waive == "通过门禁" else "warn" if waive == "SKIP" else "bad"
        contract_cell = (
            "<td class=\"missing-cell\"><span class=\"missing-label\">空</span></td>"
            if not record.user_contract
            else f"<td>{_html(record.user_contract)}</td>"
        )
        reference_cell = (
            "<td class=\"missing-cell\"><span class=\"missing-label\">空</span></td>"
            if not record.reference_backend
            else f"<td>{_html(record.reference_backend)}</td>"
        )
        rows.append(
            "<tr "
            f"data-ci=\"{_html(view)}\" "
            f"data-task=\"{_html(record.task_strategy)}\" "
            f"data-bucket=\"{_html(bucket)}\" "
            f"data-waive=\"{_html(waive)}\">"
            "<td class=\"row-index\"></td>"
            f"<td><code>{_html(record.name)}</code></td>"
            f"<td>{_html(record.hf_id)}</td>"
            f"<td>{_html(record.task_strategy)}</td>"
            f"<td>{_html(bucket)}</td>"
            f"<td>{_html(record.runtime_strategy)}</td>"
            f"<td>{_html(record.ci_tier)}</td>"
            f"{reference_cell}"
            f"{contract_cell}"
            f"<td>{_html(record.oracle_level)}</td>"
            f"<td><span class=\"pill {pill}\">{_html(waive)}</span></td>"
            f"<td>{_html(reason)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _select_options(values: list[str], selected: str = "") -> str:
    options = [f"<option value=\"\"{selected == '' and ' selected' or ''}>全部</option>"]
    for value in values:
        options.append(
            f"<option value=\"{_html(value)}\"{selected == value and ' selected' or ''}>"
            f"{_html(value)}</option>"
        )
    return "\n".join(options)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _raw_summary(
    nightly: list[CaseRecord],
    premerge: list[CaseRecord],
    all_records: list[CaseRecord],
    replacements: list[tuple[CaseRecord, CaseRecord | None]],
) -> str:
    def block(title: str, rows: list[CaseRecord]) -> list[str]:
        lines = [f"## {title}: {len(rows)} models", "internal_task_strategy"]
        for key, count in sorted(Counter(row.task_strategy for row in rows).items()):
            lines.append(f"  {key}: {count}")
        lines.append("")
        lines.append("hf_task_bucket")
        for key, count in sorted(Counter(_bucket(row) for row in rows).items()):
            lines.append(f"  {key}: {count}")
        lines.append("")
        return lines

    lines: list[str] = []
    lines.extend(block("nightly_full_e2e", nightly))
    lines.extend(block("premerge_l0_fallback", premerge))
    lines.extend(block("all_manifests", all_records))
    if replacements:
        lines.append("premerge_l0_replacements")
        for source, replacement in replacements:
            replacement_name = replacement.name if replacement else source.l0_replacement
            suffix = (
                f"  # {source.l0_replacement_reason}"
                if source.l0_replacement_reason
                else ""
            )
            lines.append(f"  {source.name} -> {replacement_name}{suffix}")
    return "\n".join(lines).rstrip()


def _metric(value: str, label: str) -> str:
    return f"<div class=\"card metric\"><strong>{_html(value)}</strong><span>{_html(label)}</span></div>"


def _render_html(records: list[CaseRecord]) -> str:
    nightly = _nightly_records(records)
    premerge, replacements = _premerge_records(records)
    table_records = nightly + premerge

    unique_models = len({record.name for record in table_records})
    unique_hf = len({record.hf_id for record in table_records if record.hf_id})
    nightly_tasks = Counter(record.task_strategy for record in nightly)
    nightly_buckets = Counter(_bucket(record) for record in nightly)
    nightly_areas = Counter(_area(_bucket(record)) for record in nightly)
    nightly_oracles = Counter(record.oracle_level for record in nightly)
    nightly_references = Counter(record.reference_backend for record in nightly)
    nightly_waives = Counter(_waive(record)[0] for record in nightly if _waive(record)[0] != "通过门禁")
    missing_contracts = sum(1 for record in nightly if not record.user_contract)

    task_options = _select_options(sorted({record.task_strategy for record in table_records}))
    bucket_options = _select_options(sorted({_bucket(record) for record in table_records}))
    waive_options = _select_options(["通过门禁", "SKIP", "XFAIL"])
    detail_rows = _detail_rows("nightly", nightly) + "\n" + _detail_rows("premerge_l0", premerge)
    raw_summary = _raw_summary(nightly, premerge, records, replacements)
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    skip_count = nightly_waives.get("SKIP", 0)
    xfail_count = nightly_waives.get("XFAIL", 0)

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TRTMC CI 模型 / 任务覆盖报告（中文）</title>
  <style>
    :root {{ --bg:#f7f8fa; --fg:#1f2933; --muted:#5b6775; --line:#d7dde5; --panel:#fff; --accent:#147a72; --warn:#9a6200; --bad:#a33a3a; --ok:#2f6f44; --code:#eef2f6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--fg); background:var(--bg); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:#17212b; color:#fff; padding:24px 28px; }}
    header h1 {{ margin:0 0 6px; font-size:26px; font-weight:720; }}
    header p {{ margin:0; color:#c9d3df; }}
    .source {{ margin-top:8px; color:#aebccc; font-size:12px; }}
    main {{ max-width:1440px; margin:0 auto; padding:24px; }}
    section {{ margin:0 0 24px; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    h3 {{ margin:0 0 10px; font-size:15px; }}
    .grid {{ display:grid; gap:14px; }}
    .stats {{ grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); }}
    .cols {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .views {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .metric {{ display:flex; flex-direction:column; gap:4px; }}
    .metric strong {{ font-size:26px; line-height:1.1; }}
    .metric span,.note {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; }}
    th {{ position:sticky; top:0; z-index:1; background:#eef2f6; font-weight:680; }}
    tbody tr:hover {{ background:#f4f8fb; }}
    .num,.row-index {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .row-index {{ color:var(--muted); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); max-height:760px; }}
    .summary-table th {{ position:static; }}
    code {{ background:var(--code); border-radius:4px; padding:1px 4px; }}
    .pill {{ display:inline-block; min-width:74px; padding:2px 7px; border-radius:999px; font-size:12px; text-align:center; border:1px solid transparent; }}
    .pill.ok {{ color:var(--ok); background:#edf7f0; border-color:#cde8d4; }}
    .pill.warn {{ color:var(--warn); background:#fff7e8; border-color:#f1d79d; }}
    .pill.bad {{ color:var(--bad); background:#fff0f0; border-color:#e7c2c2; }}
    .missing-cell {{ background:#fff2cc; color:#7a4b00; font-weight:680; border-left:3px solid #d89b00; }}
    .missing-label {{ display:inline-block; padding:2px 6px; border-radius:4px; background:#ffe4a6; color:#6b4200; font-size:12px; font-weight:700; }}
    .filters {{ display:grid; grid-template-columns:1.4fr repeat(4,minmax(150px,1fr)); gap:10px; align-items:end; }}
    .view-stats {{ display:flex; flex-wrap:wrap; gap:10px 18px; margin:10px 0; }}
    .view-stats strong {{ font-variant-numeric:tabular-nums; }}
    .all-view-note {{ display:none; margin:8px 0 0; padding:9px 11px; border-left:3px solid var(--warn); background:#fff8e8; color:#6f4a00; }}
    label {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
    input,select {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; color:var(--fg); }}
    pre {{ white-space:pre-wrap; background:#111827; color:#e5e7eb; padding:16px; border-radius:8px; overflow:auto; }}
    @media (max-width:900px) {{ .stats,.cols,.views,.filters {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>TRTMC CI 模型 / 任务覆盖报告（中文）</h1>
  <p>面向 QA 的 nightly E2E 覆盖视图：看当前 CI 覆盖了哪些模型、哪些任务类型，以及验证强度如何。</p>
  <div class="source">Repo: {_html(REPO_ROOT)} · commit: {_html(_git_commit())} · generated: {_html(generated)} · 数据来源：当前 checkout 的 python/tensorrt_model_connect/models manifests</div>
</header>
<main>
  <section class="grid stats">
    {_metric(str(len(nightly)), "Nightly full E2E case / manifest 数")}
    {_metric(str(len(premerge)), "Premerge L0 case / manifest 数")}
    {_metric(str(unique_models), "Unique model name（全部视图去重）")}
    {_metric(str(unique_hf), "Unique HF repo（全部视图去重）")}
    {_metric(str(len(nightly_tasks)), "Nightly 内部 task strategy 数")}
    {_metric(f"{skip_count} SKIP / {xfail_count} XFAIL", "Nightly 中被 waive 的 case")}
    {_metric(str(missing_contracts), "Nightly 缺失 user_contract 元数据")}
  </section>

  <section class="card">
    <p><strong>判断真实 release 覆盖率时，优先看 Nightly full E2E。</strong> Nightly 跑的是更接近发布门禁的真实模型集合，适合用来判断模型覆盖、任务覆盖、waive 情况，以及验证强度。</p>
    <div class="grid views">
      <div><h3>Nightly</h3><p>完整覆盖视图。它包含默认 E2E、nightly_only 和 contract_only case，排除 multi_device 与 l0_only。</p></div>
      <div><h3>Premerge L0</h3><p>PR 阶段的快速保护。它会用 L0 小模型代表替代大模型或 probe case，所以不能算完整模型覆盖。</p></div>
      <div><h3>全部</h3><p>把 Nightly 和 Premerge L0 合并展示。同一模型如果两边都覆盖，会出现两行；这是 CI 视图合并导致的展示重复。</p></div>
    </div>
  </section>

  <section class="grid cols">
    {_summary_table("Nightly Internal Task Strategy（TRTMC 内部怎么测）", "TRTMC E2E harness 自己的 runner/comparator 分类。", nightly_tasks)}
    {_summary_table("Nightly HF Task Bucket（用户视角的任务类型）", "把 TRTMC manifest 映射成 Hugging Face 风格任务桶；这是 QA mapping，不是实时查询 HF pipeline_tag。", nightly_buckets)}
  </section>

  <section class="grid cols">
    {_summary_table("Nightly HF Top-Level Area（更粗的大类分布）", "", nightly_areas)}
    <div class="card"><h2>Nightly Oracle / Reference（验证强度和对照来源）</h2><div class="grid cols">
      <table class="summary-table"><thead><tr><th>Oracle level</th><th class="num">case / manifest 数</th></tr></thead><tbody>{_counter_table(nightly_oracles)}</tbody></table>
      <table class="summary-table"><thead><tr><th>Reference backend</th><th class="num">case / manifest 数</th></tr></thead><tbody>{_counter_table(nightly_references)}</tbody></table>
    </div></div>
  </section>

  <section>
    <h2>模型明细</h2>
    <div class="filters card">
      <div><label for="q">搜索</label><input id="q" type="search" placeholder="模型名、HF ID、contract、runtime..."></div>
      <div><label for="ci">CI 视图</label><select id="ci"><option value="">全部</option><option value="nightly" selected>Nightly</option><option value="premerge_l0">Premerge L0</option></select></div>
      <div><label for="task">内部 task</label><select id="task">{task_options}</select></div>
      <div><label for="bucket">HF task bucket</label><select id="bucket">{bucket_options}</select></div>
      <div><label for="waive">Waive</label><select id="waive">{waive_options}</select></div>
    </div>
    <div class="view-stats note">
      <span>当前显示 <strong id="shown">0</strong> 行</span>
      <span>Unique model name <strong id="uniqueModels">0</strong></span>
      <span>Unique HF repo <strong id="uniqueHf">0</strong></span>
    </div>
    <p class="note">默认显示 Nightly，因为它是判断 QA release 覆盖率的主要视图。空 Reference / User contract 会高亮。</p>
    <p id="allViewNote" class="all-view-note">当前为“全部”视图：表格同时展示 Nightly 和 Premerge L0。同一个 model name 如果两边都覆盖，会出现两行；这是 CI 视图合并导致的展示重复，不代表 Nightly 内部重复。</p>
    <div class="table-wrap">
      <table id="details">
        <thead><tr><th>序号</th><th>模型</th><th>HF ID</th><th>内部 task</th><th>HF 任务桶</th><th>Runtime</th><th>CI tier</th><th>Reference</th><th>User contract</th><th>Oracle</th><th>Waive</th><th>原因</th></tr></thead>
        <tbody>{detail_rows}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>原始汇总（英文自动输出）</h2>
    <p class="note">用于和自动化结果对账。</p>
    <pre>{_html(raw_summary)}</pre>
  </section>
</main>
<script>
(function() {{
  const q = document.getElementById('q');
  const ci = document.getElementById('ci');
  const task = document.getElementById('task');
  const bucket = document.getElementById('bucket');
  const waive = document.getElementById('waive');
  const shown = document.getElementById('shown');
  const uniqueModels = document.getElementById('uniqueModels');
  const uniqueHf = document.getElementById('uniqueHf');
  const allViewNote = document.getElementById('allViewNote');
  const rows = Array.from(document.querySelectorAll('#details tbody tr'));
  function apply() {{
    const text = q.value.trim().toLowerCase();
    let n = 0;
    const models = new Set();
    const hfRepos = new Set();
    for (const row of rows) {{
      const okText = !text || row.innerText.toLowerCase().includes(text);
      const okCi = !ci.value || row.dataset.ci === ci.value;
      const okTask = !task.value || row.dataset.task === task.value;
      const okBucket = !bucket.value || row.dataset.bucket === bucket.value;
      const okWaive = !waive.value || row.dataset.waive === waive.value;
      const show = okText && okCi && okTask && okBucket && okWaive;
      row.style.display = show ? '' : 'none';
      const indexCell = row.querySelector('.row-index');
      if (show) {{
        n++;
        if (indexCell) indexCell.textContent = n;
        const cells = row.querySelectorAll('td');
        if (cells[1]) models.add(cells[1].innerText.trim());
        if (cells[2]) hfRepos.add(cells[2].innerText.trim());
      }} else if (indexCell) {{
        indexCell.textContent = '';
      }}
    }}
    shown.textContent = n;
    uniqueModels.textContent = models.size;
    uniqueHf.textContent = hfRepos.size;
    allViewNote.style.display = ci.value ? 'none' : 'block';
  }}
  [q, ci, task, bucket, waive].forEach(el => el.addEventListener('input', apply));
  apply();
}})();
</script>
</body>
</html>
"""
    return html_doc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="E2E model manifest directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    models_dir = args.models_dir if args.models_dir.is_absolute() else REPO_ROOT / args.models_dir
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    records = _load_records(models_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_html(records), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

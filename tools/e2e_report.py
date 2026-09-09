# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render recorded pytest observations without supplying model pass criteria."""

from __future__ import annotations

import argparse
import ast
import base64
import html
import io
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

NPY_MAX_BYTES = 64 * 1024
NPY_MAX_ELEMENTS = 4096
_FORMATS = {
    "f2": "e",
    "f4": "f",
    "f8": "d",
    "i1": "b",
    "i2": "h",
    "i4": "i",
    "i8": "q",
    "u1": "B",
    "u2": "H",
    "u4": "I",
    "u8": "Q",
}


def decode_npy(data: bytes) -> dict[str, Any] | None:
    """Return complete numeric values, or None for unsupported/untrusted bytes.

    Only a single C-order numeric array is accepted. No pickle, imports, object
    dtypes, structured dtypes, trailing arrays, or executable expressions run.
    """
    if not isinstance(data, bytes) or len(data) > NPY_MAX_BYTES or len(data) < 10:
        return None
    if data[:6] != b"\x93NUMPY":
        return None
    version = data[6:8]
    if version == b"\x01\x00":
        offset, length_size = 10, 2
    elif version in (b"\x02\x00", b"\x03\x00"):
        offset, length_size = 12, 4
    else:
        return None
    header_size = int.from_bytes(data[8 : 8 + length_size], "little")
    if not 1 <= header_size <= 4096 or len(data) < offset + header_size:
        return None
    header = data[offset : offset + header_size]
    if not header.endswith(b"\n"):
        return None
    try:
        metadata = ast.literal_eval(header.decode("utf-8" if version[0] == 3 else "latin1").strip())
    except (SyntaxError, ValueError, UnicodeError, RecursionError, MemoryError):
        return None
    if not isinstance(metadata, dict) or set(metadata) != {"descr", "fortran_order", "shape"}:
        return None
    if metadata["fortran_order"] is not False:
        return None
    shape, dtype = metadata["shape"], metadata["descr"]
    if (
        not isinstance(shape, tuple)
        or len(shape) > 8
        or not all(type(size) is int and 0 <= size <= NPY_MAX_ELEMENTS for size in shape)
    ):
        return None
    count = math.prod(shape)
    if count > NPY_MAX_ELEMENTS or not isinstance(dtype, str):
        return None
    match = re.fullmatch(r"([<>=|])([fiu][1248])", dtype)
    if match is None or match[2] not in _FORMATS:
        return None
    item_size = int(match[2][1:])
    if match[1] == "|" and item_size != 1:
        return None
    start = offset + header_size
    if len(data) != start + count * item_size:
        return None
    endian = "<" if sys.byteorder == "little" else ">"
    if match[1] in "<>":
        endian = match[1]
    try:
        values = list(struct.unpack(f"{endian}{count}{_FORMATS[match[2]]}", data[start:]))
    except (struct.error, OverflowError):
        return None
    if not all(math.isfinite(number) for number in values):
        return None
    return {"shape": list(shape), "dtype": dtype, "values": values}


def classification_index(logits: Any) -> tuple[int, int | float] | None:
    """Find a display-only class index from complete 1D or single-batch logits."""
    shape = None
    if isinstance(logits, dict):
        shape = logits.get("shape")
        logits = logits.get("values")
    if shape is not None and (
        not isinstance(shape, list) or not all(type(size) is int for size in shape)
    ):
        return None
    if not isinstance(logits, list):
        return None
    if len(logits) == 1 and isinstance(logits[0], list):
        logits = logits[0]
    if not logits or len(logits) > NPY_MAX_ELEMENTS:
        return None
    if shape is not None and shape not in ([len(logits)], [1, len(logits)]):
        return None
    if not all(type(number) in (int, float) for number in logits):
        return None
    try:
        if not all(math.isfinite(number) for number in logits):
            return None
    except OverflowError:
        return None
    index = max(range(len(logits)), key=logits.__getitem__)
    return index, logits[index]


_LIMIT = 32 * 1024 * 1024
_INLINE_BUDGET = 256 * 1024 * 1024
_CSS = """
:root{font:15px/1.55 system-ui,sans-serif;color:#172b42;background:#f5f7fa;color-scheme:light}
*{box-sizing:border-box}body{max-width:1440px;margin:32px auto;padding:0 24px}h1{font-size:30px;letter-spacing:-.03em;margin:0}h2{font-size:21px;margin:0}h3{font-size:15px;margin:0 0 10px}p{margin:8px 0}a{color:#086e80;text-underline-offset:3px}
.case,.overview{border:1px solid #dce3eb;border-radius:14px;background:#fff;margin:22px 0;padding:24px}.case{scroll-margin-top:86px}.case-head,.chips,.counts{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.case-head{justify-content:space-between;align-items:start}.meta,.note,.key{color:#596b7d}.key{display:block;font:11px ui-monospace,monospace;margin-top:2px}.meta{font-size:13px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#596b7d;margin:0 0 3px}.badge,.chip{display:inline-block;border-radius:6px;padding:3px 9px;font-size:12px;background:#eef3f7}.badge{font-weight:750;text-transform:uppercase}.passed{color:#12644b;background:#e7f5ef}.failed,.error{color:#a12630;background:#fff0f1}.skipped,.partial,.running{color:#815309;background:#fff5df}.chips{margin:12px 0 20px}.chip{background:#f1f5f9}.result-line{border-top:1px solid #e2e8ee;padding-top:14px;margin-top:16px}.io-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.io-panel{min-width:0;border:1px solid #e2e8ee;border-radius:10px;padding:16px;background:#fbfcfd}.readable{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;max-height:210px;overflow:auto;margin:0}.facts{margin:8px 0}.facts div{padding:4px 0;overflow-wrap:anywhere}.facts dt{color:#596b7d;font-size:12px}.facts dd{margin:0;font-weight:550}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:14px 0 0}figcaption{font-weight:600;font-size:13px;margin:8px 0}img,video{display:block;max-width:100%;width:100%;max-height:240px;object-fit:contain;border-radius:6px;background:#eef2f6}audio{width:100%;max-width:100%}figure .note{font-size:12px}svg{display:block;width:100%;max-height:180px}.numeric-values{padding-left:24px;overflow-wrap:anywhere;font-size:13px}.numeric-preview figcaption{font-size:12px}details{margin:12px 0;border-top:1px solid #e2e8ee;padding-top:12px}summary{cursor:pointer;font-weight:600;min-height:24px}summary:hover{color:#086e80}details details{margin-left:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px;border-radius:7px;font:12px/1.5 ui-monospace,monospace;max-height:440px;overflow:auto}code{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e2e8ee;overflow-wrap:anywhere}th{color:#596b7d;font-weight:600}.table-scroll{overflow-x:auto}.filters{display:flex;gap:12px;position:sticky;top:0;z-index:1;padding:14px 0;background:#f5f7fa}input,select{min-width:0;font:inherit;padding:10px 12px;border:1px solid #bac7d3;border-radius:7px;background:white}input{flex:1}.counts{margin:12px 0}.count{padding-right:20px}.count strong{font-size:25px;display:block}.empty{padding:14px;background:#fff5df;border-radius:7px}.notice{padding:10px 12px;border-radius:7px}.failure-summary{color:#a12630}.index td:first-child{min-width:170px}[hidden]{display:none!important}
@media(max-width:900px){.io-grid{grid-template-columns:1fr}.io-panel{display:block}.pair{grid-template-columns:1fr 1fr}}
@media(max-width:600px){body{padding:0 12px}.case,.overview{padding:16px}.pair{grid-template-columns:1fr}.filters{position:static;flex-wrap:wrap}.filters input{flex-basis:100%}h1{font-size:26px}.case{scroll-margin-top:12px}}
"""
_JS = """
function filterCases(){const q=document.getElementById('search').value.toLowerCase();const s=document.getElementById('status').value;let n=0;document.querySelectorAll('.case').forEach(e=>{e.hidden=!(e.dataset.name.includes(q)&&(!s||e.dataset.status===s));if(!e.hidden)n++;});document.querySelectorAll('.index-row').forEach(e=>{e.hidden=!(e.dataset.name.includes(q)&&(!s||e.dataset.status===s));});document.getElementById('visible-count').textContent=n+' cases shown';document.getElementById('no-results').hidden=n!==0;}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _json(value: Any) -> str:
    return _escape(json.dumps(value, ensure_ascii=False, indent=2))


def _media(path: str, root: Path, media_type: str, budget: list[int]) -> tuple[str, str | None]:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        return "", "Artifact path must remain within its testcase"
    source = root / relative
    if any(part.is_symlink() for part in (source, *source.parents) if part != root.parent):
        return "", "Symlink artifacts are not embedded"
    if not source.resolve().is_relative_to(root.resolve()) or not source.is_file():
        return "", "Artifact file is unavailable"
    if source.stat().st_size > _LIMIT:
        return "", "Artifact exceeds the inline size bound"
    allowed = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    mime = allowed.get(source.suffix.lower())
    if mime is None and source.suffix.lower() != ".ppm":
        return "", None
    if source.stat().st_size > budget[0]:
        return (
            "",
            "Aggregate inline media bound reached; original retained in the evidence directory",
        )
    data = source.read_bytes()
    if source.suffix.lower() == ".ppm":
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                converted = io.BytesIO()
                image.save(converted, format="PNG")
                data = converted.getvalue()
                mime = "image/png"
        except (ImportError, OSError, ValueError):
            return "", "PPM preview needs Pillow; the original remains in the evidence directory"
    if mime is None:
        return "", None
    if len(data) > min(_LIMIT, budget[0]):
        return "", "Converted preview exceeds the remaining inline bound"
    budget[0] -= len(data)
    uri = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    if mime.startswith("image/"):
        return f'<img loading="lazy" alt="Recorded output" src="{uri}">', None
    element = "audio" if mime.startswith("audio/") else "video"
    return f'<{element} controls preload="none" src="{uri}"></{element}>', None


def _checks(data: dict[str, Any]) -> str:
    checks = data.get("checks", [])
    if not checks:
        return '<p class="note">No assertion measurements were recorded. Consult the execution outcome and failure stage.</p>'
    rows = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = _escape(check.get("status", "unknown"))
        rows.append(
            f'<tr><td class="{status}">{status}</td><td><code>{_escape(check.get("expression", ""))}</code>'
            f"<details><summary>Evaluated operands / threshold</summary><pre>{_escape(check.get('explanation', ''))}</pre></details></td></tr>"
        )
    return (
        "<table><thead><tr><th>Result</th><th>Original assertion and recorded values</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


_LABELS = {
    "name": "Recipe",
    "hf_id": "Checkpoint",
    "hf_revision": "Checkpoint revision",
    "task": "Task",
    "precision": "Build precision",
    "reference_precision": "Reference precision",
    "tensor_parallel_size": "Tensor parallel devices",
    "context_parallel_size": "Context parallel devices",
    "max_sequence_length": "Maximum sequence length",
    "max_batch_size": "Maximum batch size",
    "image_height": "Image height",
    "image_width": "Image width",
    "video_num_frames": "Video frames",
    "quantization": "Quantization",
    "fp32_layers": "Layers kept in FP32",
    "dynamic_kv": "Dynamic KV cache",
    "max_new_tokens": "Maximum new tokens",
    "seed": "Random seed",
    "temperature": "Temperature",
    "top_k": "Top K",
    "top_p": "Top P",
    "min_p": "Minimum P",
    "repetition_penalty": "Repetition penalty",
    "use_chat_template": "Chat template",
    "enable_thinking": "Thinking",
    "num_inference_steps": "Inference steps",
    "guidance_scale": "Guidance scale",
    "cfg_scale": "CFG scale",
    "num_steps": "Inference steps",
    "top_class": "Class ID",
    "top_score": "Raw score",
    "num_masks": "Generated masks",
    "frames_count": "Frames",
    "sample_rate": "Sample rate (Hz)",
    "num_samples": "Audio samples",
    "dim": "Dimensions",
    "height": "Height",
    "width": "Width",
    "num_frames": "Frames",
    "num_tokens": "Tokens",
}
_TEXT_KEYS = (
    "text",
    "reference_text",
    "generated_text",
    "transcript",
    "transcription",
    "decoded",
    "caption",
    "answer",
)
_PAYLOAD_KEYS = {
    "manifest",
    "case",
    "testcases",
    "inputs",
    "prompt",
    "test_prompt",
    "prompt_repeat",
    "negative_prompt",
}
_QUIET_KEYS = {
    "stdout",
    "stderr",
    "runtime_command",
    "argv",
    "command",
    "path",
    "artifact",
    "media_type",
    "size_bytes",
    "preview",
    "values",
    "token_ids",
    "reference_ids",
    "actual_decoded",
}
_OUTPUT_NAMES = {
    "embedding": "Embedding vector",
    "encoding": "Encoded features",
    "classification": "Class scores",
    "image_classification": "Class scores",
    "forecasting": "Forecast",
    "time_series_forecasting": "Forecast",
    "reranking": "Ranking scores",
    "segmentation": "Segmentation masks",
    "video_segmentation": "Segmentation masks",
}


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " ").capitalize())


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _context(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = _mapping(data.get("inputs"))
    return inputs, _mapping(inputs.get("manifest")), _mapping(inputs.get("case"))


def _value(value: Any) -> str:
    if value is None:
        return "Not specified"
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, (str, int, float)):
        return str(value)
    if (
        isinstance(value, list)
        and len(value) <= 12
        and all(isinstance(item, (str, int, float)) for item in value)
    ):
        return ", ".join(str(item) for item in value) or "None"
    return f"{len(value)} recorded {'fields' if isinstance(value, dict) else 'items'}"


def _recipe(data: dict[str, Any]) -> tuple[str, str, str, str]:
    _, manifest, _ = _context(data)
    checkpoint = _mapping(data.get("checkpoint"))
    recipe = str(manifest.get("name", "Recipe not recorded"))
    model = str(checkpoint.get("hf_id") or manifest.get("hf_id") or "Prepared local checkpoint")
    task = str(manifest.get("task", "Task not recorded")).replace("_", " ")
    build = []
    if manifest.get("precision"):
        build.append(str(manifest["precision"]).upper())
    if manifest.get("tensor_parallel_size") == manifest.get("context_parallel_size") == 1:
        build.append("Single device")
    else:
        for key, prefix in (("tensor_parallel_size", "TP"), ("context_parallel_size", "CP")):
            if key in manifest:
                build.append(f"{prefix}{manifest[key]}")
    return recipe, model, task, " · ".join(build) or "Build configuration not recorded"


def _settings_table(values: dict[str, Any], *, excluded: set[str]) -> str:
    rows = []
    for key, value in values.items():
        if key in excluded:
            continue
        display = _escape(_value(value))
        if isinstance(value, (dict, list)) and (isinstance(value, dict) or len(value) > 12):
            display += f"<details><summary>Exact value</summary><pre>{_json(value)}</pre></details>"
        rows.append(
            f"<tr><th>{_escape(_label(key))}<span class='key'>{_escape(key)}</span></th><td>{display}</td></tr>"
        )
    return (
        "<table><tbody>" + "".join(rows) + "</tbody></table>"
        if rows
        else "<p class='note'>No settings recorded.</p>"
    )


def _settings(data: dict[str, Any]) -> str:
    inputs, manifest, case = _context(data)
    return (
        "<details><summary>Run settings — recipe, build and request</summary>"
        "<p class='note'>Recorded recipe and testcase settings. Missing fields have no assumed defaults. Exact runtime commands remain in technical details.</p>"
        "<h3>Recipe and build</h3>"
        + _settings_table(manifest, excluded=_PAYLOAD_KEYS | {"family"})
        + "<h3>Request and reference</h3>"
        + _settings_table(case, excluded=_PAYLOAD_KEYS | {"name"})
        + "<h3>Request input options</h3>"
        + _settings_table(
            _mapping(case.get("inputs")),
            excluded=_PAYLOAD_KEYS
            | {
                "text",
                "query",
                "source_text",
                "documents",
                "past_values",
                "input_values",
                "initial_latents",
                "state",
                "image",
                "left_image",
                "right_image",
                "audio",
                "video",
            },
        )
        + "<details><summary>All recorded input fields</summary><pre>"
        + _json(inputs)
        + "</pre></details></details>"
    )


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        dimensions = value.get("shape")
        if isinstance(dimensions, list) and all(isinstance(n, int) for n in dimensions):
            return " × ".join(map(str, dimensions)) or "scalar"
        return ""
    if not isinstance(value, list):
        return ""
    dimensions, current = [], value
    while isinstance(current, list):
        dimensions.append(len(current))
        if not current or not isinstance(current[0], list):
            break
        if not all(isinstance(item, list) and len(item) == len(current[0]) for item in current):
            return f"{len(value)} items, variable dimensions"
        current = current[0]
    return " × ".join(map(str, dimensions))


def _numeric_data(value: Any) -> tuple[list[int | float], str, int] | None:
    shape = _shape(value)
    if isinstance(value, dict):
        raw = value.get("values", value.get("preview"))
        if isinstance(raw, dict):
            return _numeric_data(raw)
    else:
        raw = value
    if not isinstance(raw, list):
        return None
    shape = shape or _shape(raw)
    dimensions = shape.split(" × ")
    if not all(part.isdigit() for part in dimensions):
        return None
    total = math.prod(int(part) for part in dimensions)
    numbers: list[int | float] = []

    def collect(items: list[Any]) -> bool:
        for item in items:
            if isinstance(item, list):
                if not collect(item):
                    return False
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                try:
                    finite = math.isfinite(item)
                except OverflowError:
                    finite = False
                if not finite:
                    return False
                numbers.append(item)
            else:
                return False
            if len(numbers) >= 64:
                break
        return True

    if not collect(raw) or len(numbers) > total:
        return None
    return numbers, shape, total


def _numeric_preview(value: Any, label: str, *, peer: Any = None) -> str:
    recorded = _numeric_data(value)
    if recorded is None:
        return ""
    numbers, shape, total = recorded
    result = _facts([(label + " shape", shape)])
    if not numbers:
        return result + "<p class='note'>No numeric values recorded.</p>"
    shown = len(numbers)
    caption = f"First {shown} of {total} values" if shown < total else f"All {total} values"
    if " × " in shape:
        caption += " · flattened order"
    if total <= 8:
        return (
            result
            + f"<p class='note'>{_escape(caption)}</p><ol class='numeric-values'>"
            + "".join(f"<li>{_escape(number)}</li>" for number in numbers)
            + "</ol>"
        )
    other = _numeric_data(peer)
    bounds_values = numbers + (other[0] if other else [])
    low, high = min(bounds_values), max(bounds_values)
    magnitude = max(abs(low), abs(high), 1)
    normalized_low, normalized_high = low / magnitude, high / magnitude
    span = normalized_high - normalized_low
    points = []
    for index, number in enumerate(numbers):
        x = 48 + 260 * index / max(1, shown - 1)
        y = 72 if span == 0 else 118 - 94 * ((number / magnitude - normalized_low) / span)
        points.append(f"{x:.2f},{y:.2f}")
    scale = "Shared native/reference scale" if other else "Scale of this preview"
    result += f"<figure class='numeric-preview'><figcaption>{_escape(caption)}</figcaption><svg viewBox='0 0 320 148' role='img' aria-label='{_escape(label + ': ' + caption + '; ' + scale)}'>"
    result += "<path d='M48 20V118H310' fill='none' stroke='#bac7d3'/>"
    result += f"<text x='44' y='27' text-anchor='end' font-size='10' fill='#596b7d'>{_escape(format(high, '.5g'))}</text>"
    result += f"<text x='44' y='120' text-anchor='end' font-size='10' fill='#596b7d'>{_escape(format(low, '.5g'))}</text>"
    result += (
        f"<polyline points='{' '.join(points)}' fill='none' stroke='#087f8c' stroke-width='2'/>"
    )
    result += f"<text x='48' y='138' font-size='10' fill='#596b7d'>1</text><text x='310' y='138' text-anchor='end' font-size='10' fill='#596b7d'>{shown}</text></svg>"
    result += f"<p class='note'>{scale} · horizontal axis: recorded value index. Display only; the original assertions determine the result.</p></figure>"
    return result


def _numeric_preview_notice(value: Any, label: str) -> str:
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("values"), dict):
        return _numeric_preview_notice(value["values"], label)
    preview = value.get("preview")
    if not isinstance(preview, list) or not preview:
        return ""
    nonfinite = sum(
        (isinstance(number, float) and not math.isfinite(number))
        or (
            isinstance(number, str)
            and number.lower()
            in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", "+nan", "-nan"}
        )
        for number in preview
    )
    if not nonfinite:
        return ""
    scope = (
        f"all {len(preview)} values in this recorded preview are non-finite"
        if nonfinite == len(preview)
        else f"this recorded preview contains {nonfinite} non-finite {'value' if nonfinite == 1 else 'values'} out of {len(preview)}"
    )
    return f"<p class='note'>{_escape(label)} plot unavailable: {_escape(scope)}. This describes the preview only, not the full tensor; the test outcome is unchanged.</p>"


def _text_preview(text: str) -> str:
    short = text[:600]
    preview = '<p class="readable">' + _escape(short) + ("…" if len(text) > 600 else "") + "</p>"
    if len(text) > 600:
        preview += (
            '<details><summary>Full text</summary><p class="readable">'
            + _escape(text)
            + "</p></details>"
        )
    return preview


def _prompt_preview(text: str) -> str:
    try:
        structured = json.loads(text)
    except (ValueError, TypeError):
        structured = None
    if isinstance(structured, dict) and isinstance(structured.get("description"), str):
        result = _text_preview(structured["description"])
        result += _facts(
            [
                (_label(key), structured[key])
                for key in ("camera", "lighting", "duration")
                if isinstance(structured.get(key), (str, int, float))
            ]
        )
        return (
            result
            + "<details><summary>Original structured prompt</summary><pre>"
            + _escape(text)
            + "</pre></details>"
        )
    return _text_preview(text)


def _facts(items: list[tuple[str, Any]]) -> str:
    return (
        '<dl class="facts">'
        + "".join(
            f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"
            for label, value in items
        )
        + "</dl>"
        if items
        else ""
    )


def _input_summary(data: dict[str, Any]) -> str:
    inputs, _, case = _context(data)
    request = {**_mapping(case.get("inputs")), **case, **inputs}
    native = _mapping(data.get("native"))
    if "input_values" in native:
        request["input_values"] = native["input_values"]
        request.pop("past_values", None)
    text = next(
        (
            request[key]
            for key in ("prompt", "test_prompt", "text", "query", "source_text")
            if isinstance(request.get(key), str) and request[key]
        ),
        None,
    )
    parts = [_prompt_preview(text)] if text else []
    documents = request.get("documents")
    if isinstance(documents, list):
        for index, document in enumerate(documents):
            if isinstance(document, str):
                content = f"<h4>Document {index + 1}</h4>" + _text_preview(document)
                parts.append(
                    content
                    if index < 2
                    else "<details><summary>Additional document</summary>" + content + "</details>"
                )
    points = [("Point X (recorded)", request["point_x"])] if "point_x" in request else []
    if "point_y" in request:
        points.append(("Point Y (recorded)", request["point_y"]))
    if "window_index" in request:
        points.append(("Recorded window index", request["window_index"]))
    parts.append(_facts(points))
    repeat = _mapping(request.get("prompt_repeat"))
    if not text and repeat:
        parts.append(_text_preview(str(repeat.get("text", ""))))
        parts.append(
            f"<p class='note'>Repeated {_escape(repeat.get('count', 'unknown'))} times; full construction is in run settings.</p>"
        )
    for key in (
        "image",
        "left_image",
        "right_image",
        "audio",
        "video",
        "asset",
        "raw_file",
        "dataset",
        "dataset_id",
        "input_values",
        "past_values",
        "state",
        "initial_latents",
        "expected_output_kind",
        "num_hypotheses",
        "mesh_diameter",
        "refinement_iterations",
    ):
        value = request.get(key)
        if value is None:
            continue
        numeric = _numeric_preview(value, _label(key))
        if numeric:
            parts.append(numeric)
            continue
        if isinstance(value, str):
            label = value.rsplit("/", 1)[-1] if "/" in value else value
        elif isinstance(value, dict) and value.get("artifact"):
            label = "Recorded file" + (f" · shape {_shape(value)}" if _shape(value) else "")
        else:
            label = ("Shape " + _shape(value)) if _shape(value) else _value(value)
        parts.append(_facts([(_label(key), label)]))
    if not any(parts):
        parts.append(
            "<p class='note'>Input preview not recorded. See run settings for the testcase input contract.</p>"
        )
    return "".join(parts)


def _diagnostic_key(key: str) -> bool:
    return key.endswith(("_ms", "_mib", "_bytes", "_ordinal", "_exact")) or key.startswith(
        "device_"
    )


def _nested_scalars(
    value: dict[str, Any], prefix: str = "", depth: int = 0
) -> list[tuple[str, Any]]:
    result = []
    for key, item in value.items():
        if key in _QUIET_KEYS or _diagnostic_key(key):
            continue
        label = prefix + _label(key)
        if isinstance(item, (bool, int, float)) or (
            isinstance(item, str) and len(item) < 120 and "/" not in item
        ):
            result.append((label, _value(item)))
        elif isinstance(item, dict) and depth < 2:
            result.extend(_nested_scalars(item, label + " · ", depth + 1))
    return result


def _classification_summary(value: dict[str, Any]) -> str:
    if "top_class" in value:
        return ""
    result = classification_index(value.get("logits"))
    if result is None:
        return "<p class='note'>Class index not recorded; complete saved logits are unavailable for a display summary.</p>"
    index, score = result
    return (
        _facts([("Class index (from saved logits)", index), ("Raw score", score)])
        + "<p class='note'>Zero-based index of the highest saved logit. Display only; no class name or probability was recorded.</p>"
    )


def _output_summary(value: Any, *, role: str, task: str, peer: Any = None) -> str:
    if value is None:
        return "<p class='note'>No output recorded. Check the result and failure stage below.</p>"
    if isinstance(value, dict) and value.get("mode") == "contract_only":
        return (
            "<p><strong>Contract checks only</strong></p><p class='note'>No reference output was generated.</p>"
            + (_facts([("Checked contract", value["oracle"])]) if value.get("oracle") else "")
        )
    if isinstance(value, str):
        return _text_preview(value)
    if isinstance(value, (int, float)):
        label = "Class ID" if "classification" in task else "Recorded value"
        return _facts([(label, value)])
    if isinstance(value, list):
        return _numeric_preview(
            value, _OUTPUT_NAMES.get(task, "Numeric output"), peer=peer
        ) or _facts([("Output shape", _shape(value))])
    if not isinstance(value, dict):
        return "<p class='note'>No readable output preview recorded.</p>"
    parts, facts = [], []
    text_keys = ("reference_text", *_TEXT_KEYS) if role == "reference" else _TEXT_KEYS
    text = next(
        (value[key] for key in text_keys if isinstance(value.get(key), str) and value[key]), None
    )
    if text:
        parts.append(_text_preview(text))
    elif role == "native" and task == "text_generation":
        message = "Decoded text not recorded."
        if isinstance(value.get("token_ids"), list):
            message += " Recorded token IDs remain in technical details."
        elif "logits" in value:
            message += " Saved token scores are summarized below."
        parts.append(f"<p class='note'>{message}</p>")
    if "classification" in task:
        parts.append(_classification_summary(value))
    numeric = _numeric_preview(value, _OUTPUT_NAMES.get(task, "Numeric output"), peer=peer)
    if numeric:
        parts.append(numeric)
    else:
        parts.append(_numeric_preview_notice(value, _OUTPUT_NAMES.get(task, "Numeric output")))
    for key, item in value.items():
        if (
            key in _TEXT_KEYS
            or _diagnostic_key(key)
            or key in _QUIET_KEYS
            or key.endswith("_shape")
            or key in {"shape", "input_values", "input_mask"}
        ):
            continue
        if isinstance(item, (bool, int, float)) or (
            isinstance(item, str) and len(item) < 120 and "/" not in item
        ):
            facts.append((_label(key), _value(item)))
        elif isinstance(item, (list, dict)) and _shape(item):
            values = _numeric_data(item)
            if values and (values[2] <= 8 or "score" in key):
                parts.append(_numeric_preview(item, _label(key), peer=_mapping(peer).get(key)))
            else:
                facts.append((_label(key) + " shape", _shape(item)))
                if values and values[0]:
                    sample = ", ".join(format(number, ".5g") for number in values[0][:4])
                    facts.append((_label(key) + " sample (first values)", sample))
            if values is None:
                parts.append(_numeric_preview_notice(item, _label(key)))
        elif isinstance(item, dict) and key in {
            "summary",
            "output",
            "outputs",
            "result",
            "results",
        }:
            facts.extend(_nested_scalars(item, _label(key) + " · "))
    if not numeric:
        shape = _shape(value) or _shape(value.get("values"))
        if shape:
            facts.insert(0, (_OUTPUT_NAMES.get(task, "Numeric output") + " shape", shape))
    for key in ("token_ids", "reference_ids"):
        if isinstance(value.get(key), list):
            facts.append(("Recorded tokens", len(value[key])))
            break
    parts.append(_facts(facts[:8]))
    if not any(parts):
        message = (
            "Recorded media shown below."
            if value.get("artifact")
            else "Detailed output recorded; expand technical details to inspect the exact values."
        )
        if set(value).issubset({"stdout", "stderr"}):
            message = "Execution logs recorded; no model output preview was captured."
        parts.append(f"<p class='note'>{message}</p>")
    return "".join(parts)


def _artifact_references(value: Any) -> set[str]:
    if isinstance(value, dict):
        paths = {str(value["artifact"])} if value.get("artifact") else set()
        for child in value.values():
            paths.update(_artifact_references(child))
        return paths
    if isinstance(value, list):
        paths = set()
        for child in value:
            paths.update(_artifact_references(child))
        return paths
    return set()


def _media_role(artifact: dict[str, Any], title: str, references: dict[str, set[str]]) -> str:
    for role, paths in references.items():
        if str(artifact.get("path", "")) in paths:
            return role
    recorded = str(artifact.get("role", "")).lower()
    if recorded in ("inputs", "native", "reference"):
        return recorded
    words = title.lower().replace("/", " ").replace("_", " ").split()
    for role in ("reference", "native", "input"):
        if role in words:
            return "inputs" if role == "input" else role
    return "additional"


def _artifacts(data: dict[str, Any], root: Path, budget: list[int]) -> tuple[dict[str, str], str]:
    titles, captions, notes = {}, {}, []
    for view in data.get("views", []) if isinstance(data.get("views"), list) else []:
        if not isinstance(view, dict):
            continue
        referenced = False
        for kind in ("image", "audio", "video"):
            media = view.get(kind)
            if isinstance(media, dict) and media.get("artifact"):
                titles[media["artifact"]] = view.get("title", "Recorded view")
                captions[media["artifact"]] = view.get("caption", "")
                referenced = True
        if not referenced and view.get("caption"):
            notes.append(
                f"<p class='note'><strong>{_escape(view.get('title', 'Evidence note'))}</strong>: {_escape(view['caption'])}</p>"
            )
    references = {
        role: _artifact_references(data.get(role)) for role in ("inputs", "native", "reference")
    }
    preview, extra, files = {}, [], []
    artifacts = data.get("artifacts", [])
    current = set().union(*references.values())
    artifacts = sorted(
        artifacts,
        key=lambda item: (
            str(item.get("path", "")) not in current if isinstance(item, dict) else True,
            "overlay" not in str(titles.get(str(item.get("path", "")), "")).lower()
            if isinstance(item, dict)
            else True,
        ),
    )
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path", ""))
        label = (
            titles.get(path) or f"{artifact.get('role', 'output')} / {artifact.get('label', path)}"
        )
        rendered, issue = _media(path, root, str(artifact.get("media_type", "")), budget)
        if rendered:
            caption = f"<p class='note'>{_escape(captions[path])}</p>" if captions.get(path) else ""
            figure = (
                f"<figure><figcaption>{_escape(label)}</figcaption>{rendered}{caption}</figure>"
            )
            role = _media_role(artifact, str(label), references)
            if str(label).casefold() in {
                "class color key",
                "class colour key",
                "legend",
                "color legend",
                "colour legend",
            }:
                preview["legend"] = preview.get("legend", "") + figure
            elif role != "additional" and role not in preview:
                preview[role] = figure
            else:
                extra.append(figure)
        else:
            suffix = f" — {_escape(issue)}" if issue else " — retained in the evidence directory"
            files.append(f"<li><code>{_escape(path)}</code>{suffix}</li>")
    more = "".join(notes)
    if extra:
        more += (
            "<details><summary>More recorded media ("
            + str(len(extra))
            + ")</summary><div class='pair'>"
            + "".join(extra)
            + "</div></details>"
        )
    if files:
        more += (
            "<details><summary>Raw evidence files</summary><ul>"
            + "".join(files)
            + "</ul></details>"
        )
    return preview, more


def _result_summary(data: dict[str, Any]) -> str:
    checks = [item for item in data.get("checks", []) if isinstance(item, dict)]
    passed = sum(item.get("status") == "passed" for item in checks)
    failed = sum(item.get("status") == "failed" for item in checks)
    summary = (
        f"Recorded assertions: {passed} passed · {failed} failed."
        if checks
        else "No assertion measurements were recorded. Consult the execution outcome and failure stage."
    )
    if data.get("failure_stage"):
        summary = f"Stopped during {str(data['failure_stage']).replace('_', ' ')}. " + summary
    return summary


def _numeric_display_copy(value: Any, root: Path) -> Any:
    if isinstance(value, list):
        return [_numeric_display_copy(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    copied = {key: _numeric_display_copy(item, root) for key, item in value.items()}
    artifact = value.get("artifact")
    if not isinstance(artifact, str):
        return copied
    path = Path(artifact)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".npy":
        return copied
    source = root / path
    if any(part.is_symlink() for part in (source, *source.parents) if part != root.parent):
        return copied
    try:
        if (
            not source.resolve().is_relative_to(root.resolve())
            or not source.is_file()
            or source.stat().st_size > NPY_MAX_BYTES
        ):
            return copied
        with source.open("rb") as stream:
            decoded = decode_npy(stream.read(NPY_MAX_BYTES + 1))
    except OSError:
        return copied
    if decoded is not None and ("shape" not in value or value["shape"] == decoded["shape"]):
        copied.update({"values": decoded["values"], "shape": decoded["shape"]})
    return copied


def _content(data: dict[str, Any], root: Path, budget: list[int], index: int = 0) -> str:
    family, case = str(data.get("family", "unknown")), str(data.get("case", "unknown"))
    status = str(data.get("status", "unknown"))
    recipe, checkpoint, task, build = _recipe(data)
    _, manifest, _ = _context(data)
    recipe_url = (
        "https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/model-recipes/families/"
        + quote(family.replace("_", "-"), safe="")
    )
    parts = [
        f'<section class="case" id="case-{index}" data-name="{_escape((family + " " + case + " " + recipe + " " + checkpoint).lower())}" data-status="{_escape(status)}">',
        f'<div class="case-head"><div><p class="eyebrow">{_escape(family)}</p><h2>{_escape(case)}</h2></div><span class="badge {_escape(status)}">{_escape(status)}</span></div>',
        f'<p class="meta">Recipe: <strong>{_escape(recipe)}</strong> · <a href="{recipe_url}">Website recipe</a></p>',
        f'<p class="meta">Checkpoint: {_escape(checkpoint)}</p><div class="chips"><span class="chip">{_escape(task)}</span><span class="chip">{_escape(build)}</span></div>',
    ]
    if data.get("issues") or data.get("evidence_status") == "partial":
        parts.append(
            '<p class="notice partial">Partial evidence — some details were unavailable or exceeded the recording limit. The test outcome is unchanged.</p>'
        )
    if "forecast" in task:
        parts.append(
            '<p class="note">Preview of the last recorded window. All recorded windows remain in technical details.</p>'
        )
    previews, more = _artifacts(data, root, budget)
    display = dict(data)
    display.update(
        {
            role: _numeric_display_copy(data.get(role), root)
            for role in ("inputs", "native", "reference")
        }
    )
    parts.append('<div class="io-grid">')
    for role, title in (
        ("inputs", "Input"),
        ("native", "Native output"),
        ("reference", "Reference output"),
    ):
        summary = (
            _input_summary(display)
            if role == "inputs"
            else _output_summary(
                display.get(role),
                role=role,
                task=str(manifest.get("task", "")),
                peer=display.get("reference" if role == "native" else "native"),
            )
        )
        parts.append(
            f'<div class="io-panel"><h3>{title}</h3>{summary}{previews.get(role, "")}</div>'
        )
    parts.append(
        "</div>"
        + previews.get("legend", "")
        + '<p class="result-line">'
        + _escape(_result_summary(data))
        + "</p>"
    )
    parts.append(more + _settings(data))
    parts.append(
        '<details><summary>How this result was checked</summary><p class="note">Original family assertions and evaluated operands. The report does not add thresholds or recompute the verdict.</p>'
        + _checks(data)
        + "</details>"
    )
    if data.get("failure"):
        parts.append(
            '<details><summary class="failure-summary">Failure details</summary><pre>'
            + _json(data["failure"])
            + "</pre></details>"
        )
    technical = {
        key: value for key, value in data.items() if key not in {"inputs", "checks", "failure"}
    }
    parts.append(
        "<details><summary>Technical details — raw outputs, logs and provenance</summary><pre>"
        + _json(technical)
        + "</pre></details></section>"
    )
    return "".join(parts)


def render_report(
    cases: list[tuple[dict[str, Any], Path]], title: str = "Model correctness evidence"
) -> str:
    budget = [_INLINE_BUDGET]
    counts = {
        status: sum(data.get("status") == status for data, _ in cases)
        for status in ("passed", "failed", "error", "skipped")
    }
    summary = "".join(
        f'<div class="count"><strong>{count}</strong>{status}</div>'
        for status, count in counts.items()
    )
    rows = []
    for index, (data, _) in enumerate(cases):
        recipe, checkpoint, task, build = _recipe(data)
        family, case, status = (
            str(data.get("family", "unknown")),
            str(data.get("case", "unknown")),
            str(data.get("status", "unknown")),
        )
        search = _escape((family + " " + case + " " + recipe + " " + checkpoint).lower())
        rows.append(
            f'<tr class="index-row" data-name="{search}" data-status="{_escape(status)}"><td><a href="#case-{index}">{_escape(family)}</a><div class="meta">{_escape(case)}</div></td><td><span class="badge {_escape(status)}">{_escape(status)}</span></td><td>{_escape(recipe)}</td><td>{_escape(task)}</td><td>{_escape(build)}</td></tr>'
        )
    index_html = (
        "<details><summary>All models and recipes ("
        + str(len(cases))
        + ' cases)</summary><div class="table-scroll"><table class="index"><thead><tr><th>Model / case</th><th>Result</th><th>Recipe</th><th>Task</th><th>Build</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div></details>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)}</title><style>{_CSS}</style></head><body>
<p class="eyebrow">Model validation</p><h1>{_escape(title)}</h1><p class="note">See the input, native result and reference for each model. Expand a card's details for settings and exact measurements.</p><div class="overview"><div class="counts">{summary}</div><p class="note">Status is the recorded testcase outcome. Missing evidence is not a passing model comparison.</p>{index_html}</div>
<div class="filters"><input id="search" aria-label="Search model, recipe or case" placeholder="Search model, recipe or case" oninput="filterCases()"><select id="status" aria-label="Filter status" onchange="filterCases()"><option value="">All results</option><option>passed</option><option>failed</option><option>error</option><option>skipped</option></select></div><p id="visible-count" class="meta" aria-live="polite">{len(cases)} cases shown</p><p id="no-results" class="empty" hidden>No matching cases. Clear the search or change the result filter.</p>
{"".join(_content(data, root, budget, index) for index, (data, root) in enumerate(cases))}<script>{_JS}</script></body></html>"""


def render_case(data: dict[str, Any], root: Path) -> str:
    return render_report([(data, root)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts_root", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cases = []
    for path in sorted(arguments.artifacts_root.rglob("evidence.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(
            arguments.artifacts_root.resolve()
        ):
            raise ValueError("evidence file must remain within its root")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError(f"unsupported evidence schema: {path}")
        cases.append((data, path.parent))
    if not cases:
        parser.error("no testcase evidence.json was found")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_report(cases), encoding="utf-8")
    print(f"Wrote {len(cases)} testcase reports to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

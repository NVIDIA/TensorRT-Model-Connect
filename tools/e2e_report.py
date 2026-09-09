# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render recorded pytest observations without supplying model pass criteria."""

from __future__ import annotations

import argparse
import ast
import base64
import html
import hashlib
import io
import json
import math
import re
import struct
import sys
from collections import Counter
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
:root{font:15px/1.45 system-ui,sans-serif;color:#172b42;background:#f5f7fa;color-scheme:light}
*{box-sizing:border-box}body{max-width:1280px;margin:24px auto;padding:0 24px}h1{font-size:28px;letter-spacing:-.03em;margin:0}h2{font-size:20px;margin:0;overflow-wrap:anywhere}h3{font-size:13px;margin:0 0 8px;color:#52657a}h4{font-size:12px;margin:0 0 6px}p{margin:6px 0}a{color:#086e80;text-underline-offset:3px}.case{border:1px solid #dce3eb;border-radius:12px;background:#fff;margin:16px 0;padding:18px;scroll-margin-top:86px}.case-head{display:flex;gap:12px;align-items:start;justify-content:space-between}.badge{display:inline-block;border-radius:6px;padding:4px 8px;font-size:12px;font-weight:650;flex-shrink:0}.reference,.passed{color:#12644b;background:#e7f5ef}.failed,.error{color:#a12630;background:#fff0f1}.limited,.unverified,.skipped,.partial,.running{color:#815309;background:#fff5df}.meta,.note,.key{color:#596b7d;font-size:12px}.key{display:block;font:11px ui-monospace,monospace;margin-top:2px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:#596b7d;margin:0 0 3px}.result-basis{font-size:13px;margin:5px 0}.recipe-line{margin:0 0 14px}.io-grid,.demo-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:14px}.io-panel{min-width:0;border:1px solid #e2e8ee;border-radius:8px;padding:12px;background:#fbfcfd}.readable{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px;line-height:1.45;max-height:132px;overflow:auto;margin:0}.facts{margin:4px 0}.facts div{padding:2px 0;overflow-wrap:anywhere}.facts dt{display:inline;color:#596b7d;font-size:12px}.facts dt:after{content:': '}.facts dd{display:inline;margin:0;font-weight:550}.pair,.output-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:4px 0 0;display:flex;flex-direction:column}figure>img,figure>audio,figure>video,figure>svg{order:0}figcaption{font-weight:500;font-size:11px;margin:4px 0;order:1}figure>.note{font-size:11px;margin:2px 0;order:2}img,video{display:block;max-width:100%;width:100%;height:170px;object-fit:contain;border-radius:5px;background:#eef2f6}audio{width:100%;max-width:100%;height:42px}svg{display:block;width:100%;height:120px}svg text{fill:#596b7d}.native-key{color:#0b7687}.reference-key{color:#b45e1c}.numeric-preview table{font-size:12px}.numeric-preview td,.numeric-preview th{padding:3px 7px}.readable.text-excerpt{max-height:none}.excerpt-gap{display:block;color:#596b7d;font-size:11px;margin:4px 0}.class-result{font-size:16px}.class-result strong{display:block;font-size:34px;font-weight:650;line-height:1.2}.case-details{margin:12px 0 0;border-top:1px solid #e2e8ee;padding-top:9px}.case-details>summary{font-size:13px}.case-details h3{margin-top:15px}details{margin:10px 0}summary{cursor:pointer;font-weight:600;min-height:22px}summary:hover{color:#086e80}details details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:12px;border-radius:6px;font:12px/1.5 ui-monospace,monospace;max-height:400px;overflow:auto}code{overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #e2e8ee;overflow-wrap:anywhere}th{color:#596b7d;font-weight:600}.table-scroll{overflow-x:auto}.filters{display:flex;gap:10px;position:sticky;top:0;z-index:1;padding:10px 0;background:#f5f7fa}input,select{min-width:0;font:inherit;padding:8px 10px;border:1px solid #bac7d3;border-radius:6px;background:white}input{flex:1}.counts{display:flex;gap:16px;margin:8px 0;font-size:13px}.count strong{margin-right:4px}.empty{padding:12px;background:#fff5df;border-radius:6px}.failure-summary{color:#a12630}.partial-note{color:#815309}.index td:first-child{min-width:170px}[hidden]{display:none!important}
@media(max-width:650px){body{padding:0 12px;margin:16px auto}.case{padding:14px;scroll-margin-top:12px}.case-head{flex-wrap:wrap;gap:5px}.io-grid,.demo-grid{grid-template-columns:1fr;gap:10px}.io-panel{padding:10px}.filters{position:static;flex-wrap:wrap}.filters input{flex-basis:100%}h1{font-size:25px}h2{font-size:18px}.readable{max-height:116px}img,video{height:150px}svg{height:105px}.pair{grid-template-columns:1fr}.recipe-line{margin-bottom:10px}}
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
    for key, value in sorted(values.items()):
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
    _, manifest, case = _context(data)
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
        + "</details>"
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
    for key, item in sorted(value.items()):
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
    parts, facts, summary_facts = [], [], []
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
    for key, item in sorted(value.items()):
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
            nested = _nested_scalars(item, _label(key) + " · ")
            (summary_facts if key == "summary" else facts).extend(nested)
    if not numeric:
        shape = _shape(value) or _shape(value.get("values"))
        if shape:
            facts.insert(0, (_OUTPUT_NAMES.get(task, "Numeric output") + " shape", shape))
    for key in ("token_ids", "reference_ids"):
        if isinstance(value.get(key), list):
            facts.append(("Recorded tokens", len(value[key])))
            break
    parts.append(_facts(facts[:8] + summary_facts[: max(0, 8 - len(facts))]))
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
    historical_inputs = set()
    for observation in data.get("observations", []):
        if isinstance(observation, dict) and observation.get("name") == "inputs":
            historical_inputs.update(_artifact_references(observation.get("value")))
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
    seen_media: dict[str, str] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = str(artifact.get("path", ""))
        label = (
            titles.get(path) or f"{artifact.get('role', 'output')} / {artifact.get('label', path)}"
        )
        before = budget[0]
        rendered, issue = _media(path, root, str(artifact.get("media_type", "")), budget)
        if rendered:
            role = _media_role(artifact, str(label), references)
            if role == "additional" and path in historical_inputs:
                role = "inputs"
            fingerprint = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if fingerprint in seen_media:
                budget[0] = before
                original_role = seen_media[fingerprint]
                if role in {"inputs", "native", "reference"} and role not in preview:
                    original = {
                        "inputs": "Input",
                        "native": "Output",
                        "reference": "Reference",
                    }.get(original_role, "recorded media in Details")
                    preview[role] = (
                        f'<p class="note">Same recorded media as {_escape(original)}.</p>'
                    )
                    if role == "reference":
                        preview["reference_same"] = original_role
                continue
            seen_media[fingerprint] = role
            caption = f"<p class='note'>{_escape(captions[path])}</p>" if captions.get(path) else ""
            figure = (
                f"<figure><figcaption>{_escape(label)}</figcaption>{rendered}{caption}</figure>"
            )
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


def _demo_text(text: str, limit: int = 420) -> str:
    """Keep a long request's final question visible beside a bounded beginning."""
    if len(text) <= limit:
        return '<p class="readable">' + _escape(text) + "</p>"
    head, tail = limit * 3 // 5, limit * 2 // 5
    return (
        '<p class="readable text-excerpt">'
        + _escape(text[:head])
        + '<span class="excerpt-gap">[… middle omitted; full text in Details …]</span>'
        + _escape(text[-tail:])
        + "</p>"
    )


def _demo_text_value(value: Any, role: str = "native") -> str:
    if isinstance(value, str):
        return value
    keys = ("reference_text", *_TEXT_KEYS) if role == "reference" else _TEXT_KEYS
    return next(
        (
            value[key]
            for key in keys
            if isinstance(value, dict) and isinstance(value.get(key), str) and value[key]
        ),
        "",
    )


def _demo_identity(data: dict[str, Any]) -> tuple[str, str]:
    """Use the website's recipe vocabulary without repeating the testcase title."""
    recipe, checkpoint, task, build = _recipe(data)
    family = str(data.get("family", "unknown"))
    title = checkpoint if checkpoint != "Prepared local checkpoint" else family.replace("_", " ")
    url = (
        "https://nvidia.github.io/TensorRT-Model-Connect/models-recipes/model-recipes/families/"
        + quote(family.replace("_", "-"), safe="")
    )
    recipe_link = (
        f'<a href="{url}">Recipe: {_escape(recipe)}</a>' if recipe != "Recipe not recorded" else ""
    )
    items = [recipe_link]
    if task != "Task not recorded":
        items.append(_escape(task))
    if build != "Build configuration not recorded":
        items.append(_escape(build))
    return title, " · ".join(item for item in items if item)


def _demo_numeric_candidate(value: Any) -> Any:
    """Retain only a contiguous finite beginning when a saved preview has gaps."""
    if _numeric_data(value) is not None:
        return value
    if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
        return None
    preview = value.get("preview")
    if not isinstance(preview, list) or "values" in value:
        return None
    prefix = []
    for number in preview[:64]:
        try:
            finite = type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite:
            break
        prefix.append(number)
    candidate = {**value, "preview": prefix} if prefix else None
    return candidate if _numeric_data(candidate) is not None else None


def _demo_variant(data: dict[str, Any]) -> str:
    recipe = _recipe(data)[0]
    case = str(data.get("case", ""))
    variant = case[len(recipe) :] if case.casefold().startswith(recipe.casefold()) else case
    variant = variant.strip("-_ ").replace("_", " ").replace("-", " ")
    return f'<span class="meta">Variant: {_escape(variant)}</span>' if variant else ""


def _demo_numeric_value(value: Any, task: str = "") -> tuple[str, Any] | None:
    """Choose one primary recorded tensor for the default demo."""
    label = _OUTPUT_NAMES.get(task, "Numeric output")
    candidate = _demo_numeric_candidate(value)
    if candidate is not None:
        return label, candidate
    if not isinstance(value, dict):
        return None
    for key in (
        "scores",
        "actions",
        "forecast",
        "predictions",
        "embedding",
        "embeddings",
        "features",
        "encoding",
        "values",
        "depth",
        "points",
        "iou_scores",
        "masks",
        "logits",
    ):
        candidate = _demo_numeric_candidate(value.get(key))
        if candidate is not None:
            return _label(key), candidate
    return None


def _demo_numeric_comparison(native: Any, reference: Any = None, task: str = "") -> str:
    """Display one shared-axis sample, never use the preview to set a verdict."""
    chosen = _demo_numeric_value(native, task)
    if chosen is None:
        return ""
    label, value = chosen
    first = _numeric_data(value)
    assert first is not None
    other = _demo_numeric_value(reference, task)
    second = _numeric_data(other[1]) if other and other[0] == label else None
    numbers, shape, total = first
    if not numbers:
        return f'<p class="note">{_escape(label)}: no numeric values recorded.</p>'
    paired = second is not None and second[1:] == first[1:]
    series = [("Native", numbers, "#0b7687")]
    if paired:
        series.append(("Reference", second[0], "#b45e1c"))
    shown = len(numbers)
    caption = f"First {shown} of {total} values" if shown < total else f"All {total} values"
    caption = f"{label} · {shape} · {caption}"
    if " × " in shape:
        caption += " · flattened order"
    if paired and len(second[0]) != shown:
        caption += f"; reference: first {len(second[0])} of {total} values"
    if total <= 8:
        header = (
            "<tr><th>Value</th>" + "".join(f"<th>{name}</th>" for name, _, _ in series) + "</tr>"
        )
        rows = []
        for index in range(max(len(values) for _, values, _ in series)):
            cells = "".join(
                f"<td>{_escape(format(values[index], '.6g')) if index < len(values) else 'Not recorded'}</td>"
                for _, values, _ in series
            )
            rows.append(f"<tr><th>{index + 1}</th>{cells}</tr>")
        return f'<figure class="numeric-preview"><figcaption>{_escape(caption)} · rounded display</figcaption><table>{header}{"".join(rows)}</table></figure>'
    bounds = [number for _, values, _ in series for number in values]
    low, high = min(bounds), max(bounds)
    magnitude = max(abs(low), abs(high), 1)
    span = high / magnitude - low / magnitude
    max_length = max(len(values) for _, values, _ in series)
    lines = []
    for name, values, color in series:
        points = []
        for index, number in enumerate(values):
            x = 48 + 300 * index / max(1, max_length - 1)
            y = 63 if span == 0 else 108 - 88 * ((number / magnitude - low / magnitude) / span)
            points.append(f"{x:.2f},{y:.2f}")
        dash = ' stroke-dasharray="5 4"' if name == "Reference" else ""
        lines.append(
            f'<polyline aria-label="{name}" points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
        if len(points) == 1:
            x, y = points[0].split(",")
            lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    legend = (
        "<span class='native-key'>Native</span> / <span class='reference-key'>Reference</span> · shared scale"
        if paired
        else "Native"
    )
    chart = (
        f'<svg viewBox="0 0 360 134" role="img" aria-label="{_escape(caption)}">'
        '<path d="M48 16V108H350" fill="none" stroke="#cad3de"/>'
        f'<text x="43" y="24" text-anchor="end" font-size="10">{_escape(format(high, ".5g"))}</text>'
        f'<text x="43" y="110" text-anchor="end" font-size="10">{_escape(format(low, ".5g"))}</text>'
        + "".join(lines)
        + f'<text x="48" y="129" font-size="10">1</text><text x="350" y="129" text-anchor="end" font-size="10">{max_length}</text></svg>'
    )
    return f'<figure class="numeric-preview"><figcaption>{_escape(caption)}</figcaption>{chart}<p class="note">{legend} · recorded value index</p></figure>'


def _demo_input(data: dict[str, Any], media: str = "") -> str:
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
        "",
    )
    parts = []
    if text:
        try:
            structured = json.loads(text)
        except (ValueError, TypeError):
            structured = None
        if isinstance(structured, dict) and isinstance(structured.get("description"), str):
            parts.append(_demo_text(structured["description"]))
            parts.append(
                '<p class="note">'
                + _escape(
                    " · ".join(
                        str(structured[key])
                        for key in ("camera", "lighting", "duration")
                        if key in structured
                    )
                )
                + "</p>"
            )
        else:
            parts.append(_demo_text(text))
    repeat = _mapping(request.get("prompt_repeat"))
    if not text and repeat:
        parts.append(_demo_text(str(repeat.get("text", ""))))
        parts.append(
            f'<p class="note">Repeated {_escape(repeat.get("count", "unknown"))} times</p>'
        )
    documents = request.get("documents")
    if isinstance(documents, list):
        for index, document in enumerate(documents[:2]):
            if isinstance(document, str):
                parts.append(
                    f'<p class="note">Document {index + 1}</p>' + _demo_text(document, 220)
                )
        if len(documents) > 2:
            parts.append(f'<p class="note">{len(documents) - 2} more documents in Details</p>')
    points = [("Point X", request["point_x"])] if "point_x" in request else []
    if "point_y" in request:
        points.append(("Point Y", request["point_y"]))
    if points:
        parts.append(
            '<p class="note">'
            + _escape(" · ".join(f"{name}: {value}" for name, value in points))
            + " (recorded coordinates)</p>"
        )
    if media:
        parts.append(media)
    else:
        for key in ("input_values", "past_values", "state", "initial_latents"):
            if request.get(key) is not None:
                numeric = _demo_numeric_comparison(request[key], task="input")
                if numeric:
                    parts.append(
                        numeric.replace("Numeric output", _label(key)).replace("Native", "Input")
                    )
                    break
        if not any(parts):
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
            ):
                value = request.get(key)
                if isinstance(value, str):
                    parts.append(
                        f'<p class="readable">{_escape(_label(key))}: {_escape(value.rsplit("/", 1)[-1])}</p>'
                    )
    if "window_index" in request:
        parts.append(f'<p class="note">Recorded window {_escape(request["window_index"])}</p>')
    return (
        "".join(parts)
        or '<p class="note">Input preview unavailable; recorded input settings are in Details.</p>'
    )


def _demo_nonfinite(value: Any) -> str:
    """Summarize non-finite saved prefixes without judging the whole tensor."""
    fields = []

    def collect(item: Any, label: str) -> None:
        if not isinstance(item, dict):
            return
        preview = item.get("preview")
        if isinstance(preview, list) and preview:
            count = sum(
                (isinstance(number, float) and not math.isfinite(number))
                or (
                    isinstance(number, str)
                    and number.lower()
                    in {
                        "inf",
                        "+inf",
                        "-inf",
                        "infinity",
                        "+infinity",
                        "-infinity",
                        "nan",
                        "+nan",
                        "-nan",
                    }
                )
                for number in preview
            )
            if count:
                fields.append(f"{label} ({count}/{len(preview)})")
        for key, child in item.items():
            if isinstance(child, dict):
                collect(child, label if key == "values" else _label(key))

    collect(value, "Output")
    if not fields:
        return ""
    names = ", ".join(dict.fromkeys(fields))
    return f'<p class="note">{_escape(names)}: non-finite saved preview values; preview only, not the full tensor.</p>'


def _demo_output(value: Any, *, role: str, task: str, media: str = "", peer: Any = None) -> str:
    """Show one useful output representation instead of generic tensor metadata."""
    if value is None or _mapping(value).get("mode") == "contract_only":
        return ""
    if "classification" in task:
        chosen = value.get("top_class") if isinstance(value, dict) else value
        if isinstance(chosen, (int, float)):
            return f'<p class="class-result">Class ID <strong>{_escape(chosen)}</strong></p>'
        saved = classification_index(_mapping(value).get("logits", value))
        if saved is not None:
            return f'<p class="class-result">Class ID <strong>{saved[0]}</strong></p><p class="note">From complete saved logits</p>'
        return (
            '<p class="note">Class preview unavailable; complete saved logits are unavailable.</p>'
        )
    notices = _demo_nonfinite(value)
    text = _demo_text_value(value, role)
    if media:
        return (_demo_text(text) if text else "") + media + notices
    if text:
        return _demo_text(text) + notices
    if task == "text_generation":
        keys = (
            ("reference_ids", "token_ids")
            if role == "reference"
            else ("token_ids", "reference_ids")
        )
        tokens = next(
            (_mapping(value)[key] for key in keys if isinstance(_mapping(value).get(key), list)),
            None,
        )
        if tokens is not None:
            return f'<p class="note">{len(tokens)} recorded tokens; decoded text unavailable. Token IDs are in Details.</p>'
        detail = "logits" if _demo_numeric_value(value, task) else "values"
        return f'<p class="note">Decoded text unavailable; recorded {detail} are in Details.</p>'
    numeric = _demo_numeric_comparison(value, peer, task)
    if numeric:
        return numeric + notices
    if notices:
        return notices
    if isinstance(value, (int, float)):
        return f'<p class="readable">{_escape(value)}</p>'
    if isinstance(value, dict) and any(key in value for key in ("probe_returncode", "receipt")):
        return '<p class="note">Runtime checks only; no generated response was recorded.</p>'
    facts = _nested_scalars(_mapping(value))
    if facts:
        return _facts(facts[:2]) + notices
    return (
        notices or '<p class="note">Output preview unavailable; recorded values are in Details.</p>'
    )


def _demo_raw_data(data: dict[str, Any]) -> dict[str, Any]:
    """Link identical observation snapshots instead of printing every copy."""
    result = dict(data)
    observations = data.get("observations")
    if not isinstance(observations, list):
        return result
    seen = {
        json.dumps(value, sort_keys=True): key
        for key, value in data.items()
        if key != "observations" and isinstance(value, (dict, list)) and value
    }
    displayed = []
    for index, item in enumerate(observations):
        if not isinstance(item, dict) or "value" not in item:
            displayed.append(item)
            continue
        fingerprint = json.dumps(item["value"], sort_keys=True)
        if fingerprint in seen:
            displayed.append({**item, "value": {"same_recorded_value_as": seen[fingerprint]}})
        else:
            displayed.append(item)
            seen[fingerprint] = f"observations[{index}].value"
    result["observations"] = displayed
    return result


_ASSESS_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_ASSESS_METRIC_KEY = re.compile(
    r"cosine|relative_l2|rel_l2|frobenius|(?:abs|pointwise|score)_error|"
    r"action_(?:max_abs_error|mean_abs_error|rmse)|iou|agreement|match_rate|"
    r"psnr|ssim|pixel_accuracy",
    re.I,
)

_ASSESS_METRIC_VALUE = re.compile(
    r"cosine|relative_l2|relative_frobenius|absolute_error|\bdelta\b|\bious?\b|"
    r"class_ious|box_iou|score_error|\bagreement\b|\w+_agreement|\branking\b|"
    r"\bpsnr\b|\bssim\b|left\s*-\s*right|poses\s*-\s*reference_poses|"
    r"scores\s*-\s*reference_scores|left\s*==\s*right",
    re.I,
)

_ASSESS_NATIVE_WORD = re.compile(
    r"\b(?:actual\w*|native\w*|hypothesis|candidate|canonical_actual)\b"
)

_ASSESS_REFERENCE_WORD = re.compile(r"\b(?:expected\w*|reference\w*|ref_\w*|canonical_expected)\b")

_ASSESS_TEXT_DISTANCE = re.compile(r"(?:edit_distance|text_distance|\bned\b)")

_ASSESS_HEALTH = re.compile(
    r"pixel|pixels|\bstats\b|\brms\b|sample_rate|samples|num_frames|"
    r"all_finite|isfinite|_std|\bmean\b|\bduration\b|\breceipt\b|probe_returncode"
)


def _assess_result(kind: str, label: str, summary: str) -> dict:
    return {"kind": kind, "label": label, "summary": summary}


def _assess_records(data: dict, name: str) -> list:
    values = []
    if name in data and data[name] is not None:
        values.append(data[name])
    observations = data.get("observations")
    if isinstance(observations, list):
        values.extend(
            item["value"]
            for item in observations
            if isinstance(item, dict) and item.get("name") == name and item.get("value") is not None
        )
    return values


def _assess_field(values: list, name: str):
    for value in values:
        if isinstance(value, dict) and value.get(name) is not None:
            return value[name]
    return None


def _assess_normalize_text(value) -> str | None:
    return " ".join(value.casefold().split()) if isinstance(value, str) else None


def _assess_numeric_comparison(check: dict) -> str | None:
    """Quote a scalar evaluation, never derive one from a pass or array preview."""
    raw = check.get("explanation")
    if not isinstance(raw, str) or not raw:
        return None
    first = raw.splitlines()[0][:500]
    first = re.sub(r"np\.float(?:16|32|64)\((" + _ASSESS_NUMBER + r")\)", r"\1", first)
    match = re.match(
        r"\s*\(*\s*("
        + _ASSESS_NUMBER
        + r")\s*(<=|>=|==|<|>)\s*("
        + _ASSESS_NUMBER
        + r")\s*\)*\s*$",
        first,
    )
    if not match:
        return None
    left, op, right = float(match[1]), match[2], float(match[3])
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    ok = {
        "<=": left <= right,
        ">=": left >= right,
        "==": left == right,
        "<": left < right,
        ">": left > right,
    }[op]
    if not ok:
        return None
    return f"{left:.6g} {op} {right:.6g}"


def _assess_assertion_names(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return ""
    return " ".join(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))


def _assess_self_comparison(expression: str) -> bool:
    """A native/native metric must not become parity because a ref exists."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and len(node.args) >= 2:
            function = ast.unparse(node.func)
            if re.search(r"cosine|relative_l2|edit_distance|allclose|array_equal", function):
                if ast.dump(node.args[0]) == ast.dump(node.args[1]):
                    return True
    return False


def _assessment(data, status=None) -> dict:
    """Return kind in reference/limited/failed/unverified, label, and one why.

    status is the individual case execution status, never family certification.
    A 'reference' result requires a successful recognized comparison assertion
    and native/reference output context. All categories describe this test only.
    """
    data = data if isinstance(data, dict) else {}
    checks = (
        [item for item in data.get("checks", []) if isinstance(item, dict)]
        if isinstance(data.get("checks"), list)
        else []
    )
    stage = str(data.get("failure_stage") or "").lower()
    state = str(status if status is not None else data.get("status") or "").lower()
    failed_checks = [item for item in checks if item.get("status") in {"failed", "error"}]
    if state in {"failed", "error"} or failed_checks or data.get("status") in {"failed", "error"}:
        if stage == "reference":
            return _assess_result(
                "failed",
                "Reference run failed",
                "Reference execution failed before output comparison; correctness is undetermined.",
            )
        if stage in {"compare", "comparison"}:
            return _assess_result(
                "failed",
                "Check failed",
                "A recorded comparison-stage check failed; see its assertion or error.",
            )
        return _assess_result(
            "failed",
            "Execution failed" if not failed_checks else "Check failed",
            "Execution or a recorded check failed; this does not establish a model-output mismatch.",
        )
    if state in {"not-run", "not_run", "skipped", "missing", "unknown", ""}:
        return _assess_result(
            "unverified",
            "Not verified",
            "Completed execution evidence is unavailable; correctness is undetermined.",
        )
    if state != "passed":
        return _assess_result(
            "unverified",
            "Not verified",
            "The recorded status does not establish a completed successful check.",
        )
    passed = [
        item
        for item in checks
        if item.get("status") == "passed" and isinstance(item.get("expression"), str)
    ]
    native, reference = _assess_records(data, "native"), _assess_records(data, "reference")
    inputs = _assess_records(data, "inputs")
    case_context = [value.get("case", {}) for value in inputs if isinstance(value, dict)]
    context = inputs + case_context
    thresholds = {}
    for value in reversed(_assess_records(data, "thresholds")):
        if isinstance(value, dict):
            thresholds.update(value)
    expressions = [str(item["expression"])[:5000] for item in passed]
    joined = "\n".join(expressions)

    if not passed or not native:
        return _assess_result(
            "unverified",
            "Not verified",
            "Successful output checks or native output evidence are missing.",
        )

    # Explicit oracle declarations and declared expected responses are stronger
    # provenance evidence than arbitrary appearances of "reference" in logs.
    if any(
        isinstance(value, dict) and value.get("mode") in {"contract_only", "invariant_only"}
        for value in reference
    ):
        return _assess_result(
            "limited",
            "Contract checks passed",
            "The declared oracle checks runtime or output contracts, without an independent reference comparison.",
        )
    fixture = _assess_field(context, "expected_response_text")
    reference_text = _assess_field(reference, "text")
    if isinstance(fixture, str) and fixture and fixture == reference_text:
        return _assess_result(
            "limited",
            "Contract checks passed",
            "Expected response and runtime checks passed; no upstream output comparison.",
        )

    # A successful second-place exception is explicitly different from top-1
    # equality. Require the exception assertions, not just the two class values.
    actual_class, expected_class = (
        _assess_field(native, "top_class"),
        _assess_field(reference, "top_class"),
    )
    if actual_class is not None and expected_class is not None and actual_class != expected_class:
        if any(
            "second_class" in expression and "==" in expression for expression in expressions
        ) and any("top1_margin" in expression and "<=" in expression for expression in expressions):
            return _assess_result(
                "reference",
                "Reference checks passed",
                f"Top classes differ ({actual_class} vs {expected_class}); the runner-up and allowed reference-margin checks passed.",
            )

    # An artifact alone is never comparison evidence; it only supplies context
    # for the successful assertions recognized below.
    if native and reference:
        for check in passed:
            expression = str(check["expression"])[:5000]
            scalar = _assess_numeric_comparison(check)
            names = _assess_assertion_names(expression)
            native_side = bool(_ASSESS_NATIVE_WORD.search(names))
            reference_side = bool(_ASSESS_REFERENCE_WORD.search(names))
            if _assess_self_comparison(expression):
                continue
            if (
                "==" in expression
                and native_side
                and reference_side
                and ("top_class" in expression or "argmax" in expression)
            ):
                return _assess_result(
                    "reference",
                    "Reference checks passed",
                    "Top class matched the reference; full-logit equality is not asserted.",
                )
            # Equality must compare output values; shape, counts, paths and
            # metadata equality do not qualify as model-reference comparison.
            if (
                "==" in expression
                and native_side
                and reference_side
                and not re.search(
                    r"\.shape|\.size|\.ndim|num_|sample_rate|channels|path|\.is_", expression
                )
            ):
                if any(
                    word in expression
                    for word in (
                        "token",
                        "_ids",
                        '"text"',
                        "'text'",
                        "reference_text",
                        "expected_text",
                    )
                ):
                    return _assess_result(
                        "reference",
                        "Reference checks passed",
                        "Recorded output text or token equality with the reference passed.",
                    )
            if (
                _ASSESS_TEXT_DISTANCE.search(expression)
                and native_side
                and reference_side
                and "prompt" not in names
            ):
                # OR clauses may pass solely on an expected answer. A scalar
                # true evaluation or identical retained text establishes the
                # actual reference branch; otherwise leave it limited.
                actual_text = _assess_field(native, "text") or _assess_field(
                    reference, "actual_decoded"
                )
                expected_text = _assess_field(reference, "reference_text") or _assess_field(
                    reference, "text"
                )
                equal_text = bool(_assess_normalize_text(actual_text)) and _assess_normalize_text(
                    actual_text
                ) == _assess_normalize_text(expected_text)
                if " or " not in expression or scalar or equal_text:
                    detail = f" ({scalar})" if scalar else ""
                    return _assess_result(
                        "reference",
                        "Reference checks passed",
                        f"Text comparison passed{detail}.",
                    )
            if re.search(r"\bned\s*<=|\bned_ok\s+or\s+token_ok\b", expression):
                actual_text = _assess_field(native, "text") or _assess_field(
                    reference, "actual_decoded"
                )
                expected_text = _assess_field(reference, "reference_text") or _assess_field(
                    reference, "text"
                )
                equal_text = bool(_assess_normalize_text(actual_text)) and _assess_normalize_text(
                    actual_text
                ) == _assess_normalize_text(expected_text)
                if scalar or equal_text:
                    detail = f" ({scalar})" if scalar else " (the retained texts agree)"
                    return _assess_result(
                        "reference",
                        "Reference checks passed",
                        f"Reference-text comparison passed{detail}.",
                    )
            if (
                "==" in expression
                and any(
                    pair in expression
                    for pair in ("canonical_left == canonical_right", "left == right")
                )
                and _assess_field(reference, "token_ids") is not None
            ):
                return _assess_result(
                    "reference",
                    "Reference checks passed",
                    "Token comparison passed after the test's canonicalization.",
                )
            # Named pairwise metrics (not output-health statistics) are emitted
            # either directly or through local aliases such as delta/ious.
            metric_keys = _ASSESS_METRIC_KEY.search(expression)
            metric_value = _ASSESS_METRIC_VALUE.search(expression)
            comparison = any(operator in expression for operator in (">=", "<=", "==", ">", "<"))
            if metric_keys and metric_value and comparison and (scalar or " or " not in expression):
                if "cosine" in expression:
                    surface = "Reference cosine"
                elif "psnr" in expression or "ssim" in expression:
                    surface = "Reference image"
                elif "iou" in expression or "pixel_accuracy" in expression:
                    surface = "Reference mask or localization"
                elif "ranking" in expression:
                    surface = "Reference ranking"
                elif "agreement" in expression or "match_rate" in expression:
                    surface = "Reference token or ranking"
                else:
                    surface = "Reference numeric"
                detail = f" ({scalar})" if scalar else ""
                return _assess_result(
                    "reference",
                    "Reference checks passed",
                    f"{surface} comparison passed its recorded limit{detail}.",
                )

        # Generic metric loops report `value <= threshold`; pair them only with
        # multiple named comparison limits, paired structured outputs, and
        # actual recorded scalar evaluations. Their metric names are not
        # recoverable from this schema, so do not invent individual values.
        numeric_limits = [
            key
            for key, value in thresholds.items()
            if _ASSESS_METRIC_KEY.search(str(key))
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        common_output_keys = set()
        for left in native:
            for right in reference:
                if isinstance(left, dict) and isinstance(right, dict):
                    common_output_keys.update(
                        set(left)
                        & set(right)
                        - {"shape", "dtype", "preview", "artifact", "path", "sample_rate"}
                    )
        generic = [
            check
            for check in passed
            if re.fullmatch(r"\s*value\s*(?:<=|>=)\s*threshold\s*", check["expression"])
            and _assess_numeric_comparison(check)
        ]
        if len(numeric_limits) >= 2 and len(common_output_keys) >= 2 and len(generic) >= 2:
            return _assess_result(
                "reference",
                "Reference checks passed",
                "Recorded reference metrics passed their limits; individual metric evaluations are not identified.",
            )

    if any(
        _ASSESS_TEXT_DISTANCE.search(expression)
        and re.search(r"\bprompt\b|_case_text\(", expression)
        for expression in expressions
    ):
        return _assess_result(
            "limited",
            "Contract checks passed",
            "Audio/text checks passed against the prompt; no native/reference output comparison.",
        )
    if "expected_answer_matches" in joined and " or " in joined:
        return _assess_result(
            "limited",
            "Contract checks passed",
            "An expected-answer alternative can satisfy this check; reference agreement is not established.",
        )
    if _ASSESS_HEALTH.search(joined) and native:
        return _assess_result(
            "limited",
            "Contract checks passed",
            "Output health or runtime checks passed; reference agreement is not established.",
        )
    return _assess_result(
        "unverified",
        "Not verified",
        "Recorded assertions do not establish an output comparison or contract check.",
    )


def _content(
    data: dict[str, Any], root: Path, budget: list[int], index: int = 0, show_variant: bool = False
) -> str:
    family, case = str(data.get("family", "unknown")), str(data.get("case", "unknown"))
    status = str(data.get("status", "unknown"))
    recipe, checkpoint, task, _ = _recipe(data)
    title, config = _demo_identity(data)
    if show_variant and (variant := _demo_variant(data)):
        config += " · " + variant
    assessment = _assessment(data, status)
    previews, more = _artifacts(data, root, budget)
    display = dict(data)
    display.update(
        {
            role: _numeric_display_copy(data.get(role), root)
            for role in ("inputs", "native", "reference")
        }
    )
    task = str(_context(data)[1].get("task", ""))
    native, reference = display.get("native"), display.get("reference")
    search = _escape((family + " " + case + " " + recipe + " " + checkpoint).lower())
    parts = [
        f'<section class="case" id="case-{index}" data-name="{search}" data-status="{_escape(assessment["kind"])}" data-execution-status="{_escape(status)}">',
        f'<div class="case-head"><h2>{_escape(title)}</h2><span class="badge {_escape(assessment["kind"])}">{_escape(assessment["label"])}</span></div>',
        f'<p class="result-basis">{_escape(assessment["summary"])}</p>',
        f'<p class="meta recipe-line">{config}</p>' if config else "",
    ]
    if data.get("issues") or data.get("evidence_status") == "partial":
        parts.append(
            '<p class="note partial-note">Partial evidence; some details are unavailable.</p>'
        )
    reference_detail = ""
    if native is not None or previews.get("native"):
        primary = _demo_output(native, role="native", task=task, media=previews.get("native", ""))
        other = _demo_output(
            reference, role="reference", task=task, media=previews.get("reference", "")
        )
        numeric = _demo_numeric_value(native, task)
        paired = _demo_numeric_value(reference, task)
        combined = ""
        if (
            not previews.get("native")
            and not previews.get("reference")
            and numeric
            and paired
            and numeric[0] == paired[0]
            and "classification" not in task
            and task != "text_generation"
            and not _demo_text_value(native, "native")
            and not _demo_text_value(reference, "reference")
        ):
            first, second = _numeric_data(numeric[1]), _numeric_data(paired[1])
            if first is not None and second is not None and first[1:] == second[1:]:
                combined = _demo_numeric_comparison(native, reference, task)
        if combined:
            notices = dict.fromkeys((_demo_nonfinite(native), _demo_nonfinite(reference)))
            primary = combined + "".join(notices)
        elif other:
            reference_detail = "<h3>Reference output</h3>" + other
        if "forecast" in task:
            primary += '<p class="note">Last recorded window; all windows are in Details.</p>'
        parts.append(
            '<div class="io-grid"><div class="io-panel"><h3>Input</h3>'
            + _demo_input(display, previews.get("inputs", ""))
            + '</div><div class="io-panel"><h3>Output</h3>'
            + primary
            + previews.get("legend", "")
            + "</div></div>"
        )
    elif data.get("inputs"):
        parts.append(
            '<div class="io-grid"><div class="io-panel"><h3>Input</h3>'
            + _demo_input(display, previews.get("inputs", ""))
            + '</div><div class="io-panel"><h3>Output</h3><p class="note">No output was recorded.</p></div></div>'
        )
    parts.append(
        '<details class="case-details"><summary>Details</summary>' + reference_detail + more
    )
    parts.append(_settings(data))
    parts.append("<h3>Checks</h3>" + _checks(data))
    if data.get("failure"):
        parts.append(
            '<h3 class="failure-summary">Failure</h3><pre>' + _json(data["failure"]) + "</pre>"
        )
    parts.append(
        '<details><summary>Raw recorded fields, full text and logs</summary><p class="note">Identical observation snapshots point to the full value already shown. Original evidence files are unchanged.</p><pre>'
        + _json(_demo_raw_data(data))
        + "</pre></details></details></section>"
    )
    return "".join(parts)


def render_report(
    cases: list[tuple[dict[str, Any], Path]], title: str = "Recorded model results"
) -> str:
    budget = [_INLINE_BUDGET]
    recipe_counts = Counter((data.get("family"), _recipe(data)[0]) for data, _ in cases)
    assessments = [_assessment(data) for data, _ in cases]
    summary = " · ".join(
        f"<span><strong>{sum(item['kind'] == kind for item in assessments)}</strong> {label}</span>"
        for kind, label in (
            ("reference", "reference checks passed"),
            ("limited", "contract checks passed"),
            ("failed", "validation failed"),
            ("unverified", "not verified"),
        )
        if any(item["kind"] == kind for item in assessments)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)}</title><style>{_CSS}</style></head><body>
<h1>{_escape(title)}</h1><p class="note">Recorded examples. Reference checks validate implementation agreement, not factual answer accuracy.</p><div class="counts">{summary}</div>
<div class="filters"><input id="search" aria-label="Search model, recipe or case" placeholder="Search model, recipe or case" oninput="filterCases()"><select id="status" aria-label="Filter status" onchange="filterCases()"><option value="">All results</option><option value="reference">Reference checks passed</option><option value="limited">Contract checks passed</option><option value="failed">Validation failed</option><option value="unverified">Not verified</option></select></div><p id="visible-count" class="meta" aria-live="polite">{len(cases)} cases shown</p><p id="no-results" class="empty" hidden>No matching cases. Clear the search or change the result filter.</p>
{"".join(_content(data, root, budget, index, recipe_counts[(data.get("family"), _recipe(data)[0])] > 1) for index, (data, root) in enumerate(cases))}<script>{_JS}</script></body></html>"""


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

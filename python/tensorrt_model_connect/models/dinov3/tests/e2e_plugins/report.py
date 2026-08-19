# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained DINOv3 dense-feature report visualization."""

from __future__ import annotations

import base64
import html
import math
import struct
import zlib
from pathlib import Path
from typing import Any

_DELTA_LIMIT = 0.01
_MAX_PATCHES = 1024
_MAX_VALUES = 4 * 1024 * 1024

_VIRIDIS = (
    (0.000, (68, 1, 84)),
    (0.125, (71, 44, 122)),
    (0.250, (59, 82, 139)),
    (0.375, (44, 114, 142)),
    (0.500, (33, 145, 140)),
    (0.625, (40, 174, 128)),
    (0.750, (94, 201, 98)),
    (0.875, (173, 220, 48)),
    (1.000, (253, 231, 37)),
)
_MAGMA = (
    (0.00, (0, 0, 4)),
    (0.25, (81, 18, 124)),
    (0.50, (183, 55, 121)),
    (0.75, (252, 137, 97)),
    (1.00, (252, 253, 191)),
)

_STYLE = """
<style>
.dinov3-feature-report{border:1px solid #cbd5e1;border-radius:12px;padding:16px;background:#f8fafc}
.dinov3-heading{display:flex;justify-content:space-between;gap:16px}.dinov3-heading h4{margin:2px 0 6px}
.dinov3-heading p{max-width:850px;color:#475569}.dinov3-summary{padding:9px 12px;border:1px solid #bfdbfe;border-radius:8px;background:#eff6ff;display:grid;gap:2px;font-size:.82em;white-space:nowrap}
.dinov3-intro{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:18px;align-items:center;margin:16px 0 20px}.dinov3-input{margin:0}.dinov3-frame{position:relative;aspect-ratio:1;background:#0f172a}.dinov3-frame img{display:block;width:100%;height:100%;object-fit:fill}.dinov3-input figcaption{margin-top:7px;color:#475569;font-size:.82em}
.dinov3-help{padding:13px 16px;border-left:4px solid #2563eb;background:#eff6ff}.dinov3-help h5{margin:0}.dinov3-help p,.dinov3-help li{font-size:.86em}.dinov3-help ol{margin:6px 0 8px 20px}
.dinov3-marker{position:absolute;left:var(--x);top:var(--y);width:calc(100%/var(--grid));aspect-ratio:1;transform:translate(-50%,-50%);border:2px solid #ef4444;box-sizing:border-box;filter:drop-shadow(0 0 2px #fff);z-index:2}.dinov3-marker:before,.dinov3-marker:after{content:"";position:absolute;background:#ef4444}.dinov3-marker:before{left:-30%;right:-30%;top:calc(50% - 1px);height:2px}.dinov3-marker:after{top:-30%;bottom:-30%;left:calc(50% - 1px);width:2px}.dinov3-marker span{position:absolute;left:100%;top:-15px;padding:1px 3px;background:#b91c1c;color:#fff;font:700 10px monospace}.dinov3-marker.compact span{display:none}
.dinov3-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.dinov3-card{padding:12px;border:1px solid #cbd5e1;border-radius:9px;background:#fff;min-width:0}.dinov3-card header{display:flex;justify-content:space-between;gap:8px}.dinov3-card h5{margin:0}.dinov3-card header p{margin:0;color:#64748b;font-size:.74em;text-align:right}.dinov3-maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.dinov3-map{margin:0;min-width:0}.dinov3-map h6{text-align:center;min-height:2.2em;margin:6px 0 3px;font-size:.73em}.dinov3-map img{width:100%;height:100%;image-rendering:auto}
.dinov3-legends{display:grid;grid-template-columns:2fr 1fr;gap:8px;margin-top:8px}.dinov3-legend{display:grid;grid-template-columns:repeat(3,1fr);font-size:.68em;color:#64748b}.dinov3-legend span:nth-child(3){text-align:center}.dinov3-legend span:last-child{text-align:right}.dinov3-gradient{grid-column:1/-1;height:9px;border:1px solid #94a3b8;border-radius:9px}.dinov3-sim{background:linear-gradient(90deg,#440154 0%,#472c7a 12.5%,#3b528b 25%,#2c728e 37.5%,#21918c 50%,#28ae80 62.5%,#5ec962 75%,#addc30 87.5%,#fde725 100%)}.dinov3-delta{background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf)}.dinov3-stats{border-top:1px solid #e2e8f0;margin:8px 0 0;padding-top:7px;color:#475569;font-size:.74em}.model-owned-diagnostics{margin-top:16px;border-style:dashed}.model-owned-diagnostics-body{padding:0 12px 12px}
@media(max-width:768px){.dinov3-heading{display:block}.dinov3-summary{margin-top:9px}.dinov3-intro,.dinov3-cards{grid-template-columns:1fr}}
@media(max-width:560px){.dinov3-maps{grid-template-columns:repeat(2,minmax(0,1fr))}.dinov3-map.delta{grid-column:1/-1;width:calc(50% - 4px);justify-self:center}.dinov3-legends{grid-template-columns:1fr}.dinov3-card header{display:block}.dinov3-card header p{text-align:left}.dinov3-map h6,.dinov3-stats{font-size:.79em}}
</style>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _unit(values: list[float]) -> list[float]:
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0 or not math.isfinite(scale):
        raise ValueError("patch feature is zero or non-finite")
    scaled = [value / scale for value in values]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("patch feature norm is invalid")
    return [value / norm for value in scaled]


def _patches(result: dict[str, Any]) -> tuple[list[list[float]], list[list[float]], int]:
    outputs = result.get("stage_outputs") or {}
    trt = (outputs.get("trt_full_inference") or {}).get("data") or {}
    ref = (outputs.get("ref_full_inference") or {}).get("data") or {}
    trt_tensor = trt.get("last_hidden_state") or {}
    ref_tensor = ref.get("last_hidden_state") or {}
    shape = ref_tensor.get("shape")
    if shape != trt_tensor.get("shape") or not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("TRT and Reference last_hidden_state shapes must match [B,T,D]")
    if shape[0] != 1 or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape
    ):
        raise ValueError("query-patch maps require one valid batch")
    registers = ref.get("num_register_tokens")
    if (
        registers != trt.get("num_register_tokens")
        or not isinstance(registers, int)
        or isinstance(registers, bool)
        or registers < 0
    ):
        raise ValueError("TRT and Reference register counts must match")
    patch_count = shape[1] - 1 - registers
    grid = math.isqrt(patch_count)
    if patch_count <= 0 or grid * grid != patch_count or patch_count > _MAX_PATCHES:
        raise ValueError("spatial patch count must form a bounded square grid")
    if patch_count * shape[2] > _MAX_VALUES:
        raise ValueError("spatial feature tensor exceeds the report bound")

    def extract(payload: dict[str, Any]) -> list[list[float]]:
        values = payload.get("data")
        if not isinstance(values, list) or len(values) != math.prod(shape):
            raise ValueError("last_hidden_state data does not match its shape")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("last_hidden_state must contain only numbers")
        width = shape[2]
        start = (1 + registers) * width
        return [
            _unit([float(value) for value in values[offset : offset + width]])
            for offset in range(start, len(values), width)
        ]

    return extract(trt_tensor), extract(ref_tensor), grid


def _queries(grid: int) -> list[int]:
    if grid < 3:
        return list(range(min(grid * grid, 8)))
    anchors = [min(grid - 1, grid * quarter // 4) for quarter in (1, 2, 3)]
    return [
        row * grid + column
        for row in anchors
        for column in anchors
        if (row, column) != (anchors[1], anchors[1])
    ]


def _dot(left: list[float], right: list[float]) -> float:
    return min(1.0, max(-1.0, math.fsum(a * b for a, b in zip(left, right))))


def _color(value: float, stops: tuple[tuple[float, tuple[int, int, int]], ...]):
    value = min(1.0, max(0.0, value))
    low_at, low = stops[0]
    high_at, high = stops[-1]
    for position, candidate in stops[1:]:
        if value <= position:
            high_at, high = position, candidate
            break
        low_at, low = position, candidate
    ratio = (value - low_at) / (high_at - low_at or 1.0)
    return tuple(round(a + ratio * (b - a)) for a, b in zip(low, high))


def _png(grid: int, pixels: list[tuple[int, int, int]]) -> str:
    rows = b"".join(
        b"\0" + bytes(channel for pixel in pixels[row * grid : (row + 1) * grid] for channel in pixel)
        for row in range(grid)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", grid, grid, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    data += chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _marker(label: str, row: int, column: int, grid: int, compact: bool = False) -> str:
    x = 100.0 * (column + 0.5) / grid
    y = 100.0 * (row + 0.5) / grid
    suffix = " compact" if compact else ""
    return (
        f'<span class="dinov3-marker{suffix}" style="--x:{x:.6f}%;--y:{y:.6f}%" '
        f'aria-label="{_esc(label or "query patch")}, row {row + 1}, column {column + 1}">'
        f'<span aria-hidden="true">{_esc(label)}</span></span>'
    )


def _input(result: dict[str, Any], project_dir: Path, queries: list[int], grid: int) -> str:
    source = str((((result.get("case_config") or {}).get("inputs") or {}).get("image") or ""))
    raw = Path(source)
    try:
        root = project_dir.resolve(strict=True)
        if raw.is_absolute() and raw.is_relative_to("/src"):
            raw = root / raw.relative_to("/src")
        path = (raw if raw.is_absolute() else root / raw).resolve(strict=True)
        path.relative_to(root)
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("input image exceeds the 10 MiB report limit")
        payload = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        uri = f"data:{mime};base64," + base64.b64encode(payload).decode()
        image = f'<img src="{uri}" alt="Model input with eight query patches" />'
        filename = _esc(path.name)
    except (FileNotFoundError, OSError, ValueError):
        image, filename = '<p class="missing">Input image unavailable.</p>', ""
    markers = "".join(
        _marker(f"Q{number}", *divmod(index, grid), grid)
        for number, index in enumerate(queries, 1)
    )
    return (
        '<figure class="dinov3-input"><div class="dinov3-frame" '
        f'style="--grid:{grid}">{image}{markers}</div><figcaption>'
        f'<strong>Model-space input.</strong> {grid} × {grid} patch grid; {filename}'
        "</figcaption></figure>"
    )


def _map(title: str, uri: str, kind: str, query: int, grid: int, low: float, high: float):
    row, column = divmod(query, grid)
    return (
        f'<figure class="dinov3-map {kind}" data-kind="{kind}" data-query-index="{query}" '
        f'data-scale-min="{low:.9f}" data-scale-max="{high:.9f}"><h6>{title}</h6>'
        f'<div class="dinov3-frame" style="--grid:{grid}"><img src="{uri}" '
        f'alt="{title}, query row {row + 1}, column {column + 1}" />'
        f'{_marker("", row, column, grid, True)}</div></figure>'
    )


def render(result: dict[str, Any], *, project_dir: Path) -> str:
    """Render query-patch cosine maps from complete paired feature tensors."""
    try:
        trt, ref, grid = _patches(result)
    except (AttributeError, OverflowError, TypeError, ValueError) as error:
        return f'<p class="missing">DINOv3 feature visualization unavailable: {_esc(error)}</p>'
    queries = _queries(grid)
    cards: list[str] = []
    shown_deltas: list[float] = []
    for number, query_index in enumerate(queries, 1):
        query = ref[query_index]
        ref_map = [_dot(query, patch) for patch in ref]
        trt_map = [_dot(query, patch) for patch in trt]
        deltas = [abs(a - b) for a, b in zip(trt_map, ref_map)]
        shown_deltas.extend(deltas)
        low = min(ref_map)
        if 1.0 - low < 0.05:
            low = -1.0
        clipped = sum(value < low or value > 1.0 for value in trt_map)
        clipped_note = f" · {clipped} TRT cells clipped" if clipped else ""

        def sim(value: float):
            return _color((value - low) / (1.0 - low), _VIRIDIS)

        ref_uri = _png(grid, [sim(value) for value in ref_map])
        trt_uri = _png(grid, [sim(value) for value in trt_map])
        delta_uri = _png(grid, [_color(value / _DELTA_LIMIT, _MAGMA) for value in deltas])
        row, column = divmod(query_index, grid)
        cards.append(
            f'<article class="dinov3-card" data-query-index="{query_index}"><header>'
            f'<h5>Q{number} · row {row + 1}, column {column + 1}</h5>'
            '<p>Shared Reference query embedding</p></header><div class="dinov3-maps">'
            f'{_map("Reference similarity", ref_uri, "reference", query_index, grid, low, 1.0)}'
            f'{_map("TensorRT similarity", trt_uri, "trt", query_index, grid, low, 1.0)}'
            f'{_map("Absolute map delta", delta_uri, "delta", query_index, grid, 0.0, _DELTA_LIMIT)}'
            '</div><div class="dinov3-legends"><div class="dinov3-legend">'
            f'<span class="dinov3-gradient dinov3-sim"></span><span>{low:.4f}</span>'
            '<span>cosine</span><span>1.0000</span></div><div class="dinov3-legend">'
            '<span class="dinov3-gradient dinov3-delta"></span><span>0</span>'
            f'<span>|Δ|</span><span>≥ {_DELTA_LIMIT:.3f}</span></div></div>'
            '<p class="dinov3-stats">Query cosine '
            f'<strong>{_dot(query, trt[query_index]):.6f}</strong> · mean map |Δ| '
            f'<strong>{math.fsum(deltas) / len(deltas):.6f}</strong> · max map |Δ| '
            f'<strong>{max(deltas):.6f}</strong>'
            f'{clipped_note}</p></article>'
        )
    maximum = max(shown_deltas, default=0.0)
    return (
        _STYLE
        + '<section class="dinov3-feature-report"><header class="dinov3-heading"><div>'
        '<h4>DINOv3 query-patch similarity maps</h4><p>Each red query patch uses one '
        'Reference embedding against both spatial feature grids. Matching colors are '
        'directly comparable; CLS and register tokens are excluded.</p></div>'
        f'<div class="dinov3-summary"><strong>{len(queries)} queries</strong>'
        f'<span>{grid} × {grid} patches</span><span>Max shown |Δ| {maximum:.6f}</span>'
        '</div></header><div class="dinov3-intro">'
        f'{_input(result, project_dir, queries, grid)}<aside class="dinov3-help">'
        '<h5>How to compare</h5><ol><li>Find Q1–Q8 on the input.</li>'
        '<li>Compare Reference and TensorRT color patterns.</li>'
        '<li>Use the black-to-yellow delta map to locate differences.</li></ol>'
        f'<p>Similarity scales are shared per row. Delta uses fixed [0, {_DELTA_LIMIT:.2f}], '
        'not an acceptance threshold. These probes are diagnostic; full-tensor metrics '
        'below are authoritative.</p></aside></div><div class="dinov3-cards">'
        + "".join(cards)
        + "</div></section>"
    )

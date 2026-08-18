# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained visual comparison for stereo disparity results."""

from __future__ import annotations

from array import array
import ast
import base64
import html
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable
import zlib

_WIDTH = 700
_HEIGHT = 700
_PIXELS = _WIDTH * _HEIGHT
_RAW_BYTES = _PIXELS * 4
_ERROR_LIMIT_PX = 2.0
_MAX_INPUT_BYTES = 10 * 1024 * 1024

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
_ERROR = (
    (0.00, (0, 0, 4)),
    (0.25, (81, 18, 124)),
    (0.50, (183, 55, 121)),
    (0.75, (252, 137, 97)),
    (1.00, (252, 253, 191)),
)

_STYLE = """
<style>
.ffs-stereo-report{border:1px solid #cbd5e1;border-radius:12px;padding:16px;background:#f8fafc}
.ffs-stereo-heading{display:flex;justify-content:space-between;gap:16px;align-items:start}.ffs-stereo-heading h4{margin:2px 0 6px}.ffs-stereo-heading p{margin:0;color:#475569;max-width:820px}.ffs-stereo-status{padding:7px 12px;border-radius:999px;font-weight:700;color:#fff;background:#b91c1c;white-space:nowrap}.ffs-stereo-status.pass{background:#15803d}
.ffs-stereo-help{margin:14px 0;padding:10px 14px;border-left:4px solid #2563eb;background:#eff6ff;color:#334155}.ffs-stereo-help strong{color:#0f172a}
.ffs-stereo-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.ffs-stereo-grid.outputs{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:12px}.ffs-stereo-card{margin:0;padding:10px;border:1px solid #cbd5e1;border-radius:9px;background:#fff;min-width:0}.ffs-stereo-card h5{margin:0 0 7px;text-align:center}.ffs-stereo-card img{display:block;width:100%;height:auto;image-rendering:auto;background:#0f172a}.ffs-stereo-card figcaption{margin-top:7px;color:#475569;font-size:.78em}
.ffs-stereo-legend{height:10px;border:1px solid #94a3b8;border-radius:10px;margin-top:8px;background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}.ffs-stereo-legend.error{background:linear-gradient(90deg,#000004,#51127c,#b73779,#fc8961,#fcfdbf)}.ffs-stereo-scale{display:flex;justify-content:space-between;color:#64748b;font-size:.72em}
.ffs-stereo-metrics{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 0;padding:0;list-style:none}.ffs-stereo-metrics li{padding:5px 8px;border:1px solid #cbd5e1;border-radius:7px;background:#fff;font:700 .76em monospace}.ffs-stereo-metrics li.fail{border-color:#ef4444;background:#fef2f2;color:#991b1b}
@media(max-width:850px){.ffs-stereo-grid.outputs{grid-template-columns:1fr}.ffs-stereo-heading{display:block}.ffs-stereo-status{display:inline-block;margin-top:10px}}
@media(max-width:560px){.ffs-stereo-grid{grid-template-columns:1fr}}
</style>
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _color(
    value: float,
    stops: tuple[tuple[float, tuple[int, int, int]], ...],
) -> tuple[int, int, int]:
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


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def _png_uri(pixels: bytes) -> str:
    if len(pixels) != _PIXELS * 3:
        raise ValueError("visualization pixel count is invalid")
    stride = _WIDTH * 3
    rows = b"".join(
        b"\0" + pixels[offset : offset + stride]
        for offset in range(0, len(pixels), stride)
    )
    header = struct.pack(">IIBBBBB", _WIDTH, _HEIGHT, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header)
    png += _chunk(b"IDAT", zlib.compress(rows, 9)) + _chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _heatmap(
    values: Iterable[float],
    *,
    high: float,
    stops: tuple[tuple[float, tuple[int, int, int]], ...],
) -> str:
    pixels = bytearray()
    for value in values:
        if not math.isfinite(value) or value < 0.0:
            pixels.extend((255, 0, 255))
        else:
            pixels.extend(_color(value / high, stops))
    return _png_uri(bytes(pixels))


def _raw_f32(path: Path) -> array[float]:
    payload = path.read_bytes()
    if len(payload) != _RAW_BYTES:
        raise ValueError(f"{path.name} must contain exactly {_PIXELS} float32 values")
    values = array("f")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _npy_f32(path: Path) -> array[float]:
    if not _RAW_BYTES < path.stat().st_size <= _RAW_BYTES + 4096:
        raise ValueError(f"{path.name} has an invalid NumPy file size")
    payload = path.read_bytes()
    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise ValueError(f"{path.name} is not a NumPy array")
    major = payload[6]
    if major == 1:
        header_size = struct.unpack("<H", payload[8:10])[0]
        data_offset = 10 + header_size
    elif major in (2, 3):
        if len(payload) < 12:
            raise ValueError(f"{path.name} has a truncated NumPy header")
        header_size = struct.unpack("<I", payload[8:12])[0]
        data_offset = 12 + header_size
    else:
        raise ValueError(f"{path.name} uses unsupported NumPy format {major}")
    header = ast.literal_eval(payload[data_offset - header_size : data_offset].decode("latin1"))
    if (
        not isinstance(header, dict)
        or header.get("descr") not in ("<f4", "=f4")
        or header.get("fortran_order") is not False
        or tuple(header.get("shape") or ()) != (_HEIGHT, _WIDTH)
        or len(payload) - data_offset != _RAW_BYTES
    ):
        raise ValueError(f"{path.name} must be a C-contiguous 700x700 little-endian float32 array")
    values = array("f")
    values.frombytes(payload[data_offset:])
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _artifact_path(artifact_dir: Path, name: str) -> Path:
    path = (artifact_dir / name).resolve(strict=True)
    try:
        path.relative_to(artifact_dir)
    except ValueError as error:
        raise ValueError(f"{name} escapes the artifact directory") from error
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file")
    return path


def _input_uri(path: Path) -> str:
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{path.name} exceeds the report size limit")
    payload = path.read_bytes()
    if (
        len(payload) < 24
        or not payload.startswith(b"\x89PNG\r\n\x1a\n")
        or payload[12:16] != b"IHDR"
        or struct.unpack(">II", payload[16:24]) != (_WIDTH, _HEIGHT)
    ):
        raise ValueError(f"{path.name} must be a {_WIDTH}x{_HEIGHT} PNG")
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _percentile(values: Iterable[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value) and value >= 0.0)
    if not finite:
        raise ValueError("reference disparity has no finite non-negative values")
    index = round((len(finite) - 1) * quantile)
    return finite[index]


def _metric_summary(result: dict[str, Any]) -> str:
    metrics = ((result.get("stages") or {}).get("full_inference") or {}).get("metrics") or {}
    items = []
    for name, label in (
        ("global_cosine", "cosine"),
        ("mean_abs_error", "MAE"),
        ("bad_2px_fraction", "bad-2px"),
    ):
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            continue
        passed = metric.get("passed") is True
        css = "" if passed else ' class="fail"'
        value = metric.get("value")
        threshold = metric.get("threshold")
        operator = metric.get("operator") or ""
        items.append(
            f"<li{css}>{_esc(label)} {_esc(value)} {_esc(operator)} {_esc(threshold)}</li>"
        )
    return '<ul class="ffs-stereo-metrics">' + "".join(items) + "</ul>" if items else ""


def _card(title: str, uri: str, caption: str, *, legend: str = "") -> str:
    return (
        '<figure class="ffs-stereo-card">'
        f"<h5>{_esc(title)}</h5>"
        f'<img src="{uri}" alt="{_esc(title)}" />'
        f"{legend}<figcaption>{caption}</figcaption></figure>"
    )


def render(result: dict[str, Any], *, project_dir: Path) -> str:
    """Render paired inputs, shared-scale disparities, and absolute error."""
    del project_dir
    try:
        artifact_ref = result.get("_artifact_dir")
        if not isinstance(artifact_ref, str) or not artifact_ref:
            raise ValueError("artifact directory is missing")
        artifact_dir = Path(artifact_ref).resolve(strict=True)
        if not artifact_dir.is_dir():
            raise ValueError("artifact directory is not a directory")
        left_uri = _input_uri(_artifact_path(artifact_dir, "left.png"))
        right_uri = _input_uri(_artifact_path(artifact_dir, "right.png"))
        actual = _raw_f32(_artifact_path(artifact_dir, "disparity.f32"))
        reference = _npy_f32(_artifact_path(artifact_dir, "torch_disparity.npy"))
        high = max(1.0, _percentile(reference, 0.995))
        trt_uri = _heatmap(actual, high=high, stops=_VIRIDIS)
        reference_uri = _heatmap(reference, high=high, stops=_VIRIDIS)
        error_uri = _heatmap(
            (abs(left - right) for left, right in zip(actual, reference)),
            high=_ERROR_LIMIT_PX,
            stops=_ERROR,
        )
    except (FileNotFoundError, OSError, SyntaxError, TypeError, ValueError) as error:
        return f'<p class="missing">Stereo visualization unavailable: {_esc(error)}</p>'

    status = str(result.get("status") or "error")
    status_class = " pass" if status == "pass" else ""
    disparity_legend = (
        '<div class="ffs-stereo-legend"></div><div class="ffs-stereo-scale">'
        f"<span>0 px</span><span>{high:.3f} px (reference p99.5)</span></div>"
    )
    error_legend = (
        '<div class="ffs-stereo-legend error"></div><div class="ffs-stereo-scale">'
        f"<span>0 px</span><span>≥ {_ERROR_LIMIT_PX:g} px</span></div>"
    )
    return (
        _STYLE
        + '<section class="ffs-stereo-report"><header class="ffs-stereo-heading"><div>'
        '<h4>Fast Foundation Stereo visual review</h4><p>TensorRT and Reference use '
        'one reference-derived disparity scale. The error map uses the numeric bad-2px '
        'boundary, so bright regions immediately identify material disagreement.</p></div>'
        f'<span class="ffs-stereo-status{status_class}">{_esc(status.upper())}</span>'
        '</header><div class="ffs-stereo-help"><strong>How to review:</strong> confirm the '
        'right input is a horizontal shift of the left, compare the two disparity color '
        'patterns, then look for bright error regions. Images are diagnostic; the numeric '
        'certification below remains authoritative.</div><div class="ffs-stereo-grid">'
        + _card("Left input", left_uri, "Synthetic rectified left image.")
        + _card("Right input", right_uri, "Same scene shifted horizontally.")
        + '</div><div class="ffs-stereo-grid outputs">'
        + _card(
            "Reference disparity",
            reference_uri,
            "Official PyTorch output.",
            legend=disparity_legend,
        )
        + _card(
            "TensorRT disparity",
            trt_uri,
            "Native runtime output on the identical pair.",
            legend=disparity_legend,
        )
        + _card(
            "Absolute error",
            error_uri,
            "Black is agreement; yellow reaches 2 px, and values above it count as bad pixels.",
            legend=error_legend,
        )
        + "</div>"
        + _metric_summary(result)
        + "</section>"
    )

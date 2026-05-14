#!/usr/bin/env python3
"""Generate a self-contained HTML performance dashboard from perf.db.

Reads the PerfDB SQLite database and produces a single HTML file with:
  - Summary dashboard: all tracked models, latest metrics, regression status
  - Per-model history: recent runs table with inline sparkline trend
  - Environment section: GPU / TRT / CUDA versions

Usage:
    python scripts/generate_perf_report.py \\
      --perf-db /path/to/perf.db \\
      -o perf_report.html \\
      [--title "Perf Report - PR !123"] \\
      [--history-limit 10] \\
      [--regression-threshold 0.10]
"""

from __future__ import annotations

import argparse
import html
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_HISTORY_LIMIT = 10
_DEFAULT_REGRESSION_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Database helpers (read-only)
# ---------------------------------------------------------------------------


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_model_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT DISTINCT model_name FROM perf_runs ORDER BY model_name"
    )
    return [row[0] for row in cur.fetchall()]


def _load_latest_run(
    conn: sqlite3.Connection, model_name: str
) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM perf_runs WHERE model_name = ? "
        "ORDER BY timestamp DESC LIMIT 1",
        (model_name,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _load_history(
    conn: sqlite3.Connection, model_name: str, limit: int = _DEFAULT_HISTORY_LIMIT
) -> List[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM perf_runs WHERE model_name = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (model_name, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def _load_baseline(
    conn: sqlite3.Connection, model_name: str
) -> Optional[Dict[str, Any]]:
    """Return the explicit or best-throughput baseline for a model."""
    # Explicit baseline table (may not exist in older DBs)
    try:
        cur = conn.execute(
            "SELECT p.* FROM baselines b "
            "JOIN perf_runs p ON b.run_id = p.run_id "
            "WHERE b.model_name = ? "
            "ORDER BY b.updated_at DESC LIMIT 1",
            (model_name,),
        )
        row = cur.fetchone()
        if row:
            return dict(row)
    except Exception:
        pass

    # Fallback: highest throughput
    cur = conn.execute(
        "SELECT * FROM perf_runs "
        "WHERE model_name = ? AND throughput_tps IS NOT NULL "
        "ORDER BY throughput_tps DESC LIMIT 1",
        (model_name,),
    )
    row = cur.fetchone()
    if row:
        return dict(row)

    # Fallback: fastest trt_run_s
    cur = conn.execute(
        "SELECT * FROM perf_runs "
        "WHERE model_name = ? AND trt_run_s IS NOT NULL "
        "ORDER BY trt_run_s ASC LIMIT 1",
        (model_name,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _load_environments(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    try:
        cur = conn.execute(
            "SELECT * FROM environments ORDER BY first_seen DESC LIMIT 3"
        )
        return [dict(row) for row in cur.fetchall()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------


def regression_status(
    latest: Dict[str, Any],
    baseline: Optional[Dict[str, Any]],
    threshold: float = _DEFAULT_REGRESSION_THRESHOLD,
) -> Tuple[str, str]:
    """Return (status, detail_string).

    status values: 'regression' | 'ok' | 'no_baseline' | 'no_data'
    """
    if baseline is None:
        return "no_baseline", "No baseline"

    base_tps = baseline.get("throughput_tps")
    curr_tps = latest.get("throughput_tps")
    if base_tps and curr_tps and base_tps > 0:
        ratio = curr_tps / base_tps
        if ratio < (1.0 - threshold):
            return (
                "regression",
                f"TPS {curr_tps:.1f} vs {base_tps:.1f} ({ratio:.0%})",
            )
        return "ok", f"TPS {curr_tps:.1f} vs {base_tps:.1f} ({ratio:.0%})"

    base_trt = baseline.get("trt_run_s")
    curr_trt = latest.get("trt_run_s")
    if base_trt and curr_trt and base_trt > 0:
        ratio = curr_trt / base_trt
        if ratio > (1.0 + threshold):
            return (
                "regression",
                f"Run {curr_trt:.2f}s vs {base_trt:.2f}s ({ratio:.0%})",
            )
        return "ok", f"Run {curr_trt:.2f}s vs {base_trt:.2f}s ({ratio:.0%})"

    return "no_data", "No comparable metrics"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _esc(v: Any) -> str:
    return html.escape(str(v)) if v is not None else ""


def fmt_metric(v: Any, unit: str = "", precision: int = 1) -> str:
    """Format a numeric metric for display, return '—' for None/missing."""
    if v is None:
        return "&mdash;"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _esc(v)
    fmt = f"{fv:.{precision}f}"
    return f"{fmt}{unit}" if unit else fmt


def fmt_commit(commit: str) -> str:
    """Shorten a git commit hash to 8 chars."""
    return (commit or "")[:8] or "&mdash;"


def fmt_ts(ts: str) -> str:
    """Shorten an ISO timestamp to date + time (drop seconds + tz)."""
    if not ts:
        return "&mdash;"
    # e.g. "2026-03-17T14:23:45.123456+00:00" -> "2026-03-17 14:23"
    t = ts.replace("T", " ")[:16]
    return _esc(t)


# ---------------------------------------------------------------------------
# SVG sparkline
# ---------------------------------------------------------------------------


def sparkline_svg(
    values: List[float],
    width: int = 80,
    height: int = 24,
    color: str = "#3b82f6",
) -> str:
    """Return an inline SVG sparkline for *values* (newest-last order).

    Returns empty string if fewer than 2 data points.
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    mn = min(vals)
    mx = max(vals)
    rng = mx - mn if mx != mn else 1.0
    pts = []
    for i, v in enumerate(vals):
        x = round(i * (width - 2) / (len(vals) - 1)) + 1
        y = round((1.0 - (v - mn) / rng) * (height - 4)) + 2
        pts.append(f"{x},{y}")
    polyline = " ".join(pts)
    return (
        f'<svg width="{width}" height="{height}" class="sparkline" '
        f'title="{mn:.1f} – {mx:.1f}">'
        f'<polyline points="{polyline}" '
        f'fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "regression": "#ef4444",
    "ok": "#22c55e",
    "no_baseline": "#6b7280",
    "no_data": "#9ca3af",
}
_STATUS_LABELS = {
    "regression": "REGRESSION",
    "ok": "OK",
    "no_baseline": "NO BASELINE",
    "no_data": "NO DATA",
}


def _status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#6b7280")
    label = _STATUS_LABELS.get(status, status.upper())
    return (
        f'<span class="badge" style="background:{color}">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------


def render_summary_dashboard(
    model_rows: List[Dict[str, Any]],
) -> str:
    """Render top-level summary table + counters + filter controls.

    Each entry in *model_rows* should have keys:
      model_name, status, detail, latest (run dict), tps_history (list[float])
    """
    counts: Dict[str, int] = {
        "regression": 0,
        "ok": 0,
        "no_baseline": 0,
        "no_data": 0,
    }
    for row in model_rows:
        s = row.get("status", "no_data")
        counts[s] = counts.get(s, 0) + 1

    counters = (
        '<div class="counters">'
        f'<span class="counter regression-counter">'
        f'{counts["regression"]} Regression</span>'
        f'<span class="counter ok-counter">{counts["ok"]} OK</span>'
        f'<span class="counter nobase-counter">'
        f'{counts["no_baseline"]} No Baseline</span>'
        f'<span class="counter total-counter">'
        f'{len(model_rows)} Total</span>'
        "</div>"
    )

    filters = (
        '<div class="filters">'
        '<input type="text" id="search-box" placeholder="Search models…" '
        'oninput="filterRows()" />'
        '<select id="status-filter" onchange="filterRows()">'
        '<option value="">All</option>'
        '<option value="regression">Regression</option>'
        '<option value="ok">OK</option>'
        '<option value="no_baseline">No Baseline</option>'
        "</select>"
        "</div>"
    )

    table_rows: List[str] = []
    for row in model_rows:
        name = row["model_name"]
        status = row.get("status", "no_data")
        detail = row.get("detail", "")
        latest = row.get("latest") or {}
        history = row.get("tps_history") or []

        tps = fmt_metric(latest.get("throughput_tps"), precision=1)
        decode = fmt_metric(latest.get("decode_ms_mean"), "ms")
        build = fmt_metric(latest.get("build_s"), "s")
        speedup = fmt_metric(latest.get("speedup"), "x", precision=2)
        last_ts = fmt_ts(latest.get("timestamp", ""))
        spark = sparkline_svg(list(reversed(history)))  # oldest→newest for chart

        table_rows.append(
            f'<tr class="perf-row" data-status="{_esc(status)}" '
            f'data-name="{_esc(name.lower())}">'
            f'<td><a href="#model-{_esc(name)}">{_esc(name)}</a></td>'
            f"<td>{tps}</td>"
            f"<td>{decode}</td>"
            f"<td>{build}</td>"
            f"<td>{speedup}</td>"
            f"<td>{_status_badge(status)}<br>"
            f'<small class="detail-text">{_esc(detail)}</small></td>'
            f"<td>{spark}</td>"
            f"<td>{last_ts}</td>"
            f"</tr>"
        )

    table = (
        '<table class="summary-table" id="summary-table">'
        "<thead><tr>"
        "<th>Model</th>"
        "<th>TPS</th>"
        "<th>Decode(ms)</th>"
        "<th>Build(s)</th>"
        "<th>Speedup</th>"
        "<th>vs Baseline</th>"
        "<th>Trend (TPS)</th>"
        "<th>Last Run</th>"
        "</tr></thead><tbody>"
        + "\n".join(table_rows)
        + "</tbody></table>"
    )

    return f'<section class="dashboard">{counters}\n{filters}\n{table}</section>'


# ---------------------------------------------------------------------------
# Per-model history section
# ---------------------------------------------------------------------------


def render_model_section(
    model_name: str,
    latest: Optional[Dict[str, Any]],
    history: List[Dict[str, Any]],
    status: str,
    detail: str,
) -> str:
    """Render a collapsible <details> section for one model."""
    badge = _status_badge(status)
    hf_id = (latest or {}).get("hf_id", "")
    branch = (latest or {}).get("git_branch", "")
    commit = fmt_commit((latest or {}).get("git_commit", ""))

    header = (
        f'<details id="model-{_esc(model_name)}">'
        f"<summary>{badge} <strong>{_esc(model_name)}</strong>"
    )
    if hf_id and hf_id != model_name:
        header += f" <small>({_esc(hf_id)})</small>"
    header += "</summary>"

    body_parts: List[str] = []

    # Latest run summary
    if latest:
        body_parts.append(
            f'<p class="run-meta">Latest: commit <code>{commit}</code>'
            f" on <em>{_esc(branch)}</em>"
            f" at {fmt_ts(latest.get('timestamp', ''))}"
            f"</p>"
        )
        body_parts.append(
            f'<p class="regression-detail">{_esc(detail)}</p>'
        )

    # History table
    if history:
        rows_html: List[str] = []
        for r in history:
            c = fmt_commit(r.get("git_commit", ""))
            ts = fmt_ts(r.get("timestamp", ""))
            tps = fmt_metric(r.get("throughput_tps"), precision=1)
            decode = fmt_metric(r.get("decode_ms_mean"), "ms")
            build = fmt_metric(r.get("build_s"), "s")
            speedup = fmt_metric(r.get("speedup"), "x", precision=2)
            src = _esc(r.get("source", ""))
            e2e = _esc(r.get("e2e_status", ""))
            rows_html.append(
                f"<tr>"
                f"<td><code>{c}</code></td>"
                f"<td>{ts}</td>"
                f"<td>{tps}</td>"
                f"<td>{decode}</td>"
                f"<td>{speedup}</td>"
                f"<td>{build}</td>"
                f"<td>{src}</td>"
                f"<td>{e2e}</td>"
                f"</tr>"
            )
        body_parts.append(
            '<table class="history-table">'
            "<thead><tr>"
            "<th>Commit</th><th>Timestamp</th><th>TPS</th>"
            "<th>Decode(ms)</th><th>Speedup</th><th>Build(s)</th>"
            "<th>Source</th><th>E2E Status</th>"
            "</tr></thead><tbody>"
            + "\n".join(rows_html)
            + "</tbody></table>"
        )

        # Sparkline (larger, labeled)
        tps_vals = [
            r["throughput_tps"] for r in reversed(history)
            if r.get("throughput_tps") is not None
        ]
        if len(tps_vals) >= 2:
            svg = sparkline_svg(tps_vals, width=200, height=48)
            if svg:
                body_parts.append(
                    f'<div class="spark-wrap">'
                    f"<span class='spark-label'>TPS trend (oldest→newest)</span>"
                    f"{svg}</div>"
                )
    else:
        body_parts.append("<p><em>No history available.</em></p>")

    body = "\n".join(body_parts)
    return f'{header}\n<div class="model-body">{body}</div>\n</details>'


# ---------------------------------------------------------------------------
# Environment section
# ---------------------------------------------------------------------------


def render_env_section(envs: List[Dict[str, Any]]) -> str:
    if not envs:
        return ""
    env = envs[0]
    items = []
    for k in ("gpu_name", "driver", "trt_version", "cuda_version", "hostname"):
        v = env.get(k)
        if v:
            items.append(f"<li><strong>{_esc(k)}:</strong> {_esc(v)}</li>")
    return (
        '<section class="env-section">'
        "<h2>Environment</h2>"
        f'<ul class="env-list">{"".join(items)}</ul>'
        "</section>"
    )


# ---------------------------------------------------------------------------
# CSS / JS
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
  Helvetica, Arial, sans-serif; background: #f8f9fa; color: #1a1a2e;
  max-width: 1400px; margin: 0 auto; padding: 20px; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 12px; }
.subtitle { color: #6b7280; margin-bottom: 20px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
  color: #fff; font-size: 0.75em; font-weight: 700; }
.counters { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.counter { padding: 6px 16px; border-radius: 8px; font-weight: 600;
  font-size: 0.9em; }
.regression-counter { background: #fee2e2; color: #991b1b; }
.ok-counter { background: #dcfce7; color: #166534; }
.nobase-counter { background: #f1f5f9; color: #475569; }
.total-counter { background: #e0e7ff; color: #3730a3; }
.filters { display: flex; gap: 8px; margin-bottom: 12px; }
#search-box { padding: 6px 12px; border: 1px solid #d1d5db;
  border-radius: 6px; flex: 1; max-width: 300px; }
#status-filter { padding: 6px 12px; border: 1px solid #d1d5db;
  border-radius: 6px; }
.summary-table, .history-table { width: 100%; border-collapse: collapse;
  margin: 8px 0; font-size: 0.88em; }
.summary-table th, .history-table th { background: #1e293b; color: #fff;
  padding: 8px 12px; text-align: left; white-space: nowrap; }
.summary-table td, .history-table td { padding: 6px 10px;
  border-bottom: 1px solid #e2e8f0; vertical-align: middle; }
.summary-table tbody tr:hover { background: #f1f5f9; }
.detail-text { color: #6b7280; font-size: 0.8em; }
details { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
  margin: 8px 0; }
details[open] { border-color: #94a3b8; }
summary { padding: 12px 16px; cursor: pointer; font-size: 1em;
  user-select: none; }
summary:hover { background: #f8fafc; }
.model-body { padding: 12px 16px; }
.run-meta { font-size: 0.85em; color: #475569; margin-bottom: 4px; }
.run-meta code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
.regression-detail { font-size: 0.85em; color: #374151; margin-bottom: 10px; }
.spark-wrap { margin: 10px 0; }
.spark-label { font-size: 0.75em; color: #6b7280; display: block;
  margin-bottom: 2px; }
svg.sparkline { display: inline-block; vertical-align: middle; }
.env-section ul { list-style: none; columns: 2; }
.env-section li { padding: 2px 0; font-size: 0.9em; }
@media (max-width: 768px) { .env-section ul { columns: 1; } }
"""

_JS = """\
function filterRows() {
  var q = (document.getElementById('search-box').value || '').toLowerCase();
  var s = document.getElementById('status-filter').value;
  var rows = document.querySelectorAll('.perf-row');
  for (var i = 0; i < rows.length; i++) {
    var name = rows[i].getAttribute('data-name') || '';
    var status = rows[i].getAttribute('data-status') || '';
    var show = (!q || name.indexOf(q) >= 0) && (!s || status === s);
    rows[i].style.display = show ? '' : 'none';
  }
}
"""


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


def render_report(
    model_rows: List[Dict[str, Any]],
    envs: List[Dict[str, Any]],
    title: str = "Performance Dashboard",
    generated_at: str = "",
) -> str:
    """Assemble and return a self-contained HTML report string."""
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h1>{_esc(title)}</h1>",
    ]

    if generated_at:
        parts.append(
            f'<p class="subtitle">Generated at {_esc(generated_at)}</p>'
        )

    parts.append(render_env_section(envs))
    parts.append("<h2>Performance Summary</h2>")
    parts.append(render_summary_dashboard(model_rows))

    parts.append("<h2>Per-Model Details</h2>")
    for row in model_rows:
        parts.append(
            render_model_section(
                model_name=row["model_name"],
                latest=row.get("latest"),
                history=row.get("history") or [],
                status=row.get("status", "no_data"),
                detail=row.get("detail", ""),
            )
        )

    parts.append(f"<script>{_JS}</script>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Top-level data loading
# ---------------------------------------------------------------------------


def load_report_data(
    db_path: str,
    history_limit: int = _DEFAULT_HISTORY_LIMIT,
    threshold: float = _DEFAULT_REGRESSION_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all data needed for the report from *db_path*.

    Returns:
        (model_rows, envs) where model_rows is a list of per-model dicts
        and envs is a list of environment dicts.
    """
    conn = _open_db(db_path)
    try:
        model_names = _load_model_names(conn)
        envs = _load_environments(conn)

        model_rows: List[Dict[str, Any]] = []
        for name in model_names:
            latest = _load_latest_run(conn, name)
            if latest is None:
                continue
            history = _load_history(conn, name, limit=history_limit)
            baseline = _load_baseline(conn, name)
            status, detail = regression_status(latest, baseline, threshold)

            # TPS history (newest first from DB; keep that order for the table,
            # caller reverses for sparkline if needed)
            tps_history = [
                r["throughput_tps"]
                for r in history
                if r.get("throughput_tps") is not None
            ]

            model_rows.append(
                {
                    "model_name": name,
                    "latest": latest,
                    "history": history,
                    "baseline": baseline,
                    "status": status,
                    "detail": detail,
                    "tps_history": tps_history,
                }
            )
    finally:
        conn.close()

    return model_rows, envs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML performance report from perf.db."
    )
    parser.add_argument(
        "--perf-db",
        required=True,
        metavar="PATH",
        help="Path to the PerfDB SQLite file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--title",
        default="Performance Dashboard",
        help="Report title (default: 'Performance Dashboard').",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=_DEFAULT_HISTORY_LIMIT,
        metavar="N",
        help=f"Max history rows per model (default: {_DEFAULT_HISTORY_LIMIT}).",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=_DEFAULT_REGRESSION_THRESHOLD,
        metavar="F",
        help=(
            "Fractional regression threshold (default: "
            f"{_DEFAULT_REGRESSION_THRESHOLD:.0%})."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    from datetime import datetime, timezone  # noqa: PLC0415

    args = parse_args(argv)

    if not Path(args.perf_db).exists():
        print(f"ERROR: perf-db not found: {args.perf_db}", file=sys.stderr)
        return 1

    model_rows, envs = load_report_data(
        args.perf_db,
        history_limit=args.history_limit,
        threshold=args.regression_threshold,
    )

    if not model_rows:
        print("WARNING: No perf runs found in database.", file=sys.stderr)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_content = render_report(
        model_rows,
        envs,
        title=args.title,
        generated_at=generated_at,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_content, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    regressions = sum(1 for r in model_rows if r["status"] == "regression")
    print(
        f"Report written to {args.output} "
        f"({size_kb:.0f} KB, {len(model_rows)} models, "
        f"{regressions} regressions)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

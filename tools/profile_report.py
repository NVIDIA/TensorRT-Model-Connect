#!/usr/bin/env python3
"""Auto-generated HTML profiling report.

Combines per-layer TRT timing, three-way perf comparison, CPU phase breakdown,
and Nsight kernel data into a self-contained HTML report with Chart.js charts.

All inputs are optional — the report adapts to whichever JSON files are provided.

Usage:
    # All inputs
    python tools/profile_report.py \\
      --layer-profile layer_profile.json \\
      --perf-compare perf_compare.json \\
      --cpu-profile cpu_profile.json \\
      --nsight-trt nsight_trt.json \\
      --nsight-hf nsight_hf.json \\
      -o report.html

    # Minimal: just perf comparison
    python tools/profile_report.py \\
      --perf-compare perf_compare.json \\
      -o report.html

    # One-shot from profile.py JSON artifacts
    python tools/profile_report.py \\
      --output-dir /tmp/qwen_profile \\
      -o report.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Layer op-type classification (for color coding)
# ---------------------------------------------------------------------------

_OP_COLORS = {
    "MatMul": "#4e79a7",        # blue — linear / attention matmuls
    "Gemm": "#4e79a7",
    "Softmax": "#f28e2b",       # orange — attention softmax
    "ElementWise": "#59a14f",   # green — add / mul / etc.
    "Shuffle": "#e15759",       # red — reshape / transpose
    "Norm": "#76b7b2",          # teal — RMSNorm / LayerNorm
    "RMS": "#76b7b2",
    "Layer": "#76b7b2",
    "Convolution": "#edc948",   # yellow
    "Conv": "#edc948",
    "Activation": "#b07aa1",    # purple — GELU / SiLU
    "Gelu": "#b07aa1",
    "Silu": "#b07aa1",
    "Reduce": "#ff9da7",        # pink — sum / mean
    "Scale": "#9c755f",         # brown
    "Slice": "#bab0ac",         # gray
    "Gather": "#bab0ac",
    "Concat": "#bab0ac",
}
_DEFAULT_COLOR = "#d4d4d4"


def _layer_color(name: str) -> str:
    for prefix, color in _OP_COLORS.items():
        if prefix.lower() in name.lower():
            return color
    return _DEFAULT_COLOR


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[profile_report] Warning: {path} not found, skipping.",
              file=sys.stderr)
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f5f5f5; color: #222; margin: 0; padding: 16px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; margin: 24px 0 8px; border-bottom: 2px solid #ddd;
        padding-bottom: 4px; color: #444; }}
  .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 16px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 16px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,.12); margin-bottom: 16px; }}
  .speedup-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .speedup-badge {{ border-radius: 8px; padding: 12px 20px; text-align: center;
                    min-width: 120px; }}
  .speedup-badge .val {{ font-size: 2rem; font-weight: 700; }}
  .speedup-badge .lbl {{ font-size: 0.75rem; color: #555; margin-top: 4px; }}
  .fast {{ background: #d4edda; color: #155724; }}
  .slow {{ background: #f8d7da; color: #721c24; }}
  .neutral {{ background: #e2e3e5; color: #383d41; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th {{ background: #f0f0f0; text-align: left; padding: 6px 10px;
        border-bottom: 2px solid #ccc; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #fafafa; }}
  .chart-wrap {{ position: relative; }}
  canvas {{ max-width: 100%; }}
  .no-data {{ color: #999; font-style: italic; padding: 8px 0; }}
  .section-note {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">{meta_line}</div>

{body_html}

<script>
const PROFILE_DATA = {profile_data_json};
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
(function() {{
  const D = PROFILE_DATA;

  // --- Three-way latency grouped bar chart ---
  if (D.perf_compare && document.getElementById("chart-latency")) {{
    const pc = D.perf_compare;
    const trt = pc.trt || {{}};
    const hf = pc.hf || {{}};
    const cp = pc.hf_compiled || null;
    const cpp = pc.trt_cpp || null;

    const labels = ["Prefill", "Decode", "Total"];
    const trtVals = [
      (trt.prefill_ms || {{}}).mean || 0,
      (trt.decode_ms || {{}}).mean || 0,
      (trt.total_ms || {{}}).mean || 0,
    ];
    const hfVals = [
      (hf.prefill_ms || {{}}).mean || 0,
      (hf.decode_ms || {{}}).mean || 0,
      (hf.total_ms || {{}}).mean || 0,
    ];
    const datasets = [];
    if (cpp) {{
      datasets.push({{
        label: "TRT (C++)",
        data: [
          (cpp.prefill_ms || {{}}).mean || 0,
          (cpp.decode_ms || {{}}).mean || 0,
          (cpp.total_ms || {{}}).mean || 0,
        ],
        backgroundColor: "#1a6faf",
      }});
    }}
    datasets.push({{ label: "TRT (Python)", data: trtVals, backgroundColor: "#4e79a7" }});
    datasets.push({{ label: "HF (eager)", data: hfVals, backgroundColor: "#f28e2b" }});
    if (cp) {{
      datasets.push({{
        label: "HF (compile/" + (cp.compile_mode || "?") + ")",
        data: [
          (cp.prefill_ms || {{}}).mean || 0,
          (cp.decode_ms || {{}}).mean || 0,
          (cp.total_ms || {{}}).mean || 0,
        ],
        backgroundColor: "#59a14f",
      }});
    }}
    new Chart(document.getElementById("chart-latency"), {{
      type: "bar",
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: "top" }} }},
        scales: {{
          y: {{ title: {{ display: true, text: "Latency (ms)" }} }},
        }},
      }},
    }});
  }}

  // --- Per-layer horizontal bar ---
  if (D.layer_profile && document.getElementById("chart-layers")) {{
    const layers = (D.layer_profile.layers || []).slice(0, 30);
    const colors = D._layer_colors || [];
    new Chart(document.getElementById("chart-layers"), {{
      type: "bar",
      data: {{
        labels: layers.map(l => l.name.length > 50 ? l.name.slice(0,47)+"..." : l.name),
        datasets: [{{
          label: "Mean (ms)",
          data: layers.map(l => l.mean_ms),
          backgroundColor: colors.length ? colors : "#4e79a7",
        }}],
      }},
      options: {{
        indexAxis: "y",
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ title: {{ display: true, text: "Mean (ms)" }} }},
        }},
      }},
    }});
  }}

  // --- CPU phase breakdown + C++ overhead comparison ---
  if (D.cpu_profile && document.getElementById("chart-cpu")) {{
    const phases = D.cpu_profile.phases || [];
    const cpp = (D.perf_compare || {{}}).trt_cpp;
    const pyTot = D.cpu_profile.total_ms || 0;
    const nSteps = D.cpu_profile.metadata && D.cpu_profile.metadata.max_new_tokens || 1;

    const phaseColors = {{
      mask_build: "#bab0ac", h2d: "#4e79a7", tensor_bind: "#76b7b2",
      execute: "#59a14f", d2d_cache: "#edc948", d2d_state: "#edc948",
      d2h: "#e15759", argmax: "#b07aa1",
    }};
    new Chart(document.getElementById("chart-cpu"), {{
      type: "bar",
      data: {{
        labels: phases.map(p => p.phase),
        datasets: [{{
          label: "Mean (ms)",
          data: phases.map(p => p.mean_ms),
          backgroundColor: phases.map(p => phaseColors[p.phase] || "#76b7b2"),
        }}],
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ title: {{ display: true, text: "Mean (ms)" }} }},
        }},
      }},
    }});
  }}

  // --- Nsight kernel pie charts ---
  ["trt", "hf"].forEach(function(backend) {{
    const canvas = document.getElementById("chart-nsys-" + backend);
    if (!canvas || !D["nsight_" + backend]) return;
    const kernels = (D["nsight_" + backend].top_kernels || []).slice(0, 10);
    if (!kernels.length) return;
    const COLORS = [
      "#4e79a7","#f28e2b","#59a14f","#e15759","#76b7b2",
      "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"
    ];
    new Chart(canvas, {{
      type: "pie",
      data: {{
        labels: kernels.map(k => k.name.length > 40 ? k.name.slice(0,37)+"..." : k.name),
        datasets: [{{
          data: kernels.map(k => k.pct || k.total_ms),
          backgroundColor: COLORS.slice(0, kernels.length),
        }}],
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ position: "right", labels: {{ font: {{ size: 10 }} }} }},
          title: {{ display: true, text: "Top GPU Kernels — " + backend.toUpperCase() }},
        }},
      }},
    }});
  }});
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _speedup_badge(label: str, value: float | None) -> str:
    if value is None:
        cls, disp = "neutral", "N/A"
    elif value >= 1.0:
        cls = "fast"
        disp = f"{value:.2f}x"
    else:
        cls = "slow"
        disp = f"{value:.2f}x"
    return (f'<div class="speedup-badge {cls}">'
            f'<div class="val">{disp}</div>'
            f'<div class="lbl">{label}</div></div>')


def _build_speedup_section(pc: dict) -> str:
    sp = pc.get("speedup", {})
    badges = []
    if "cpp_vs_hf_decode" in sp:
        badges.append(_speedup_badge("C++ TRT vs HF decode", sp.get("cpp_vs_hf_decode")))
    if "cpp_vs_trt_python_decode" in sp:
        badges.append(_speedup_badge("C++ vs Python TRT decode", sp.get("cpp_vs_trt_python_decode")))
    if "decode" in sp:
        badges.append(_speedup_badge("TRT (Python) vs HF decode", sp.get("decode")))
    if "prefill" in sp:
        badges.append(_speedup_badge("TRT vs HF prefill", sp.get("prefill")))
    if "trt_vs_compile_decode" in sp:
        badges.append(_speedup_badge(
            "TRT vs compile decode", sp.get("trt_vs_compile_decode")))
    if "trt_vs_compile_prefill" in sp:
        badges.append(_speedup_badge(
            "TRT vs compile prefill", sp.get("trt_vs_compile_prefill")))

    token_match = pc.get("token_match")
    match_badge = ""
    if token_match is not None:
        match_cls = "fast" if token_match else "slow"
        match_lbl = "Token match" if token_match else "Token MISMATCH"
        match_badge = (f'<div class="speedup-badge {match_cls}">'
                       f'<div class="val">{"✓" if token_match else "✗"}</div>'
                       f'<div class="lbl">{match_lbl}</div></div>')

    return ('<div class="card"><h2>Speedup Summary</h2>'
            f'<div class="speedup-grid">{"".join(badges)}{match_badge}</div>'
            "</div>")


def _latency_row(label: str, *cells) -> str:
    tds = "".join(f"<td>{c}</td>" for c in cells)
    return f"<tr><td><b>{label}</b></td>{tds}</tr>"


def _ms(d: dict | None, key: str = "mean") -> str:
    if not d:
        return "—"
    return f"{d.get(key, 0):.2f}"


def _build_latency_table(pc: dict) -> str:
    trt = pc.get("trt", {})
    hf = pc.get("hf", {})
    cp = pc.get("hf_compiled")
    cpp = pc.get("trt_cpp")
    sp = pc.get("speedup", {})

    has_compile = cp is not None
    has_cpp = cpp is not None

    def _sp(key: str) -> str:
        v = sp.get(key)
        return f"{v:.2f}x" if v else "—"

    # Build headers and rows dynamically based on available backends
    col_headers = []
    if has_cpp:
        col_headers.append("TRT (C++)")
    col_headers.append("TRT (Python)")
    col_headers.append("HF (eager)")
    if has_compile:
        col_headers.append(f"HF (compile/{cp['compile_mode']})")
        col_headers.append("TRT/compile")

    def _row(label: str, *vals: str) -> tuple:
        return (label, *vals)

    def _speedup_col(baseline_key: str, target_key: str) -> str:
        bv = sp.get(baseline_key)
        tv = sp.get(target_key)
        if bv and tv:
            return f"{bv / tv:.2f}x" if tv != 0 else "—"
        v = sp.get(baseline_key)
        return f"{v:.2f}x" if v else "—"

    rows_data = []
    for metric, field in [
        ("Prefill (ms)", "prefill_ms"),
        ("Decode (ms)", "decode_ms"),
        ("Per-token (ms)", "per_token_ms"),
        ("Throughput (t/s)", "throughput_tps"),
        ("Total (ms)", "total_ms"),
    ]:
        cols = []
        if has_cpp:
            cols.append(_ms(cpp.get(field)))
        cols.append(_ms(trt.get(field)))
        cols.append(_ms(hf.get(field)))
        if has_compile:
            cols.append(_ms(cp.get(field)))
            if field == "prefill_ms":
                cols.append(_sp("trt_vs_compile_prefill"))
            elif field == "decode_ms":
                cols.append(_sp("trt_vs_compile_decode"))
            elif field == "total_ms":
                cols.append(_sp("trt_vs_compile_total"))
            else:
                cols.append("—")
        rows_data.append((metric, *cols))

    # Speedup row
    if has_cpp:
        sp_cols = ["—"]  # C++ is the baseline
        sp_cols.append(_sp("cpp_vs_trt_python_decode"))
        sp_cols.append(_sp("cpp_vs_hf_decode"))
        if has_compile:
            sp_cols.append("—")
            sp_cols.append("—")
        rows_data.append(("Speedup (decode)", *sp_cols))
    else:
        sp_cols = ["—"]  # TRT Python is baseline
        sp_cols.append(_sp("decode"))
        if has_compile:
            sp_cols.append("—")
            sp_cols.append("—")
        rows_data.append(("Speedup (decode)", *sp_cols))

    ths = "".join(f"<th>{h}</th>" for h in [""] + col_headers)
    trs = "\n".join(_latency_row(*row) for row in rows_data)
    meta = pc.get("metadata", {})
    model = meta.get("model", "")
    gpu = meta.get("gpu", "")
    iters = meta.get("iterations", "?")
    warmup = meta.get("warmup", "?")
    dtype = meta.get("hf_dtype", "?")

    title = f"{'Four' if has_cpp and has_compile else 'Three' if (has_cpp or has_compile) else 'Two'}-Way Latency Comparison"

    return (
        '<div class="card">'
        f"<h2>{title}</h2>"
        f'<p class="section-note">Model: {model} | GPU: {gpu} | '
        f"dtype: {dtype} | {iters} iters / {warmup} warmup</p>"
        '<canvas id="chart-latency" height="80"></canvas>'
        f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"
        "</div>"
    )


def _build_cpp_vs_python_analysis(pc: dict, cpu: dict | None) -> str:
    """Explain why C++ is faster than Python TrtRunner using CPU phase data."""
    cpp = pc.get("trt_cpp")
    trt = pc.get("trt", {})
    sp  = pc.get("speedup", {})
    if not cpp:
        return ""

    meta = pc.get("metadata", {})
    n_steps = meta.get("max_new_tokens", 20)

    cpp_step_ms  = (cpp.get("decode_ms") or {}).get("mean", 0) / n_steps
    py_step_ms   = (trt.get("decode_ms") or {}).get("mean", 0) / n_steps
    overhead_ms  = py_step_ms - cpp_step_ms

    # Pull CPU phase data if available
    phase_rows = ""
    cpu_total_ms = 0.0
    if cpu:
        phases = cpu.get("phases", [])
        cpu_total_ms = cpu.get("total_ms", 0.0)
        unmeasured_ms = max(0.0, py_step_ms - cpu_total_ms)

        phase_colors = {
            "execute":     "#4e79a7",
            "d2d_cache":   "#f28e2b",
            "d2h":         "#e15759",
            "tensor_bind": "#76b7b2",
            "h2d":         "#59a14f",
            "mask_build":  "#b07aa1",
            "argmax":      "#ff9da7",
        }
        for p in phases:
            color = phase_colors.get(p["phase"], "#aaa")
            phase_rows += (
                f"<tr><td><span style='display:inline-block;width:12px;height:12px;"
                f"background:{color};border-radius:2px;margin-right:6px'></span>"
                f"{p['phase']}</td>"
                f"<td>{p['mean_ms']:.4f}</td>"
                f"<td>{p['pct']:.1f}%</td>"
                f"<td>{'✓ shared with C++' if p['phase']=='execute' else '✗ Python only'}</td></tr>"
            )
        if unmeasured_ms > 0.01:
            phase_rows += (
                f"<tr style='color:#e15759'><td><span style='display:inline-block;width:12px;"
                f"height:12px;background:rgba(225,87,89,0.4);border-radius:2px;margin-right:6px'>"
                f"</span>Python interpreter overhead (unmeasured)</td>"
                f"<td>{unmeasured_ms:.4f}</td>"
                f"<td>{100.0*unmeasured_ms/py_step_ms:.1f}%</td>"
                f"<td>✗ Python only (loop, numpy, dict alloc…)</td></tr>"
            )

    cpp_vs_py = sp.get("cpp_vs_trt_python_decode")

    summary = (
        f"<p>The C++ runtime is <strong>{cpp_vs_py:.2f}× faster</strong> than the Python "
        f"TrtRunner on decode ({cpp_step_ms:.2f} ms vs {py_step_ms:.2f} ms per step), "
        f"saving <strong>{overhead_ms:.2f} ms/step</strong> "
        f"({overhead_ms * n_steps:.1f} ms total over {n_steps} decode steps).</p>"
        if cpp_vs_py else ""
    )

    table = ""
    if phase_rows:
        table = (
            "<h3>Python TRT per-step phase breakdown</h3>"
            "<table><thead><tr><th>Phase</th><th>Mean (ms)</th>"
            "<th>% of Python step</th><th>C++ equivalent</th></tr></thead>"
            f"<tbody>{phase_rows}</tbody></table>"
            f"<p style='font-size:0.85rem;color:#666'>"
            f"Python measured total: {cpu_total_ms:.3f} ms/step &nbsp;|&nbsp; "
            f"C++ total: {cpp_step_ms:.2f} ms/step &nbsp;|&nbsp; "
            f"Python overhead eliminated by C++: "
            f"{max(0.0, py_step_ms - cpp_step_ms):.2f} ms/step</p>"
        )

    explanation = (
        "<h3>What C++ eliminates</h3>"
        "<ul>"
        "<li><strong>Python interpreter loop</strong> — each decode step crosses the "
        "Python/C++ boundary multiple times (TensorMap dict, step() call, argmax); "
        "C++ runs a tight native loop with zero interpreter overhead.</li>"
        "<li><strong>Per-step numpy allocation</strong> — Python TrtRunner allocates a "
        "new NumPy array for logits every step and calls <code>np.argmax()</code>; "
        "C++ reuses a <code>std::vector&lt;float&gt;</code> and calls "
        "<code>std::max_element</code>.</li>"
        "<li><strong>Forced CPU/GPU sync per step</strong> — reading logits back to Python "
        "forces a <code>cudaStreamSynchronize</code> every step; C++ batches transfers "
        "more efficiently.</li>"
        "<li><strong>TensorMap construction</strong> — Python rebuilds the input dict "
        "(<code>tensor_bind</code> phase) on every step; C++ reuses stack-allocated "
        "<code>Tensor</code> structs.</li>"
        "</ul>"
    )

    return (
        '<div class="card">'
        "<h2>C++ vs Python TRT Runtime — Overhead Analysis</h2>"
        + summary + table + explanation
        + "</div>"
    )


def _build_layer_section(lp: dict) -> str:
    layers = lp.get("layers", [])
    total_ms = lp.get("total_ms", 0.0)
    top30 = layers[:30]

    colors_js = json.dumps([_layer_color(l["name"]) for l in top30])

    rows = []
    cum = 0.0
    for i, l in enumerate(top30, 1):
        cum += l["pct"]
        rows.append(
            f"<tr><td>{i}</td>"
            f'<td style="max-width:360px;overflow:hidden;text-overflow:ellipsis;'
            f'white-space:nowrap;font-family:monospace;font-size:0.8rem">'
            f'{l["name"]}</td>'
            f'<td>{l["mean_ms"]:.4f}</td>'
            f'<td>{l["std_ms"]:.4f}</td>'
            f'<td>{l["pct"]:.1f}%</td>'
            f'<td>{cum:.1f}%</td>'
            f'<td>{l["calls"]}</td></tr>'
        )

    meta = lp.get("metadata", {})
    note = (f"Model: {meta.get('model','')} | GPU: {meta.get('gpu','')} | "
            f"Total: {total_ms:.3f} ms/step | Top {len(top30)} of {len(layers)} layers")

    return (
        '<div class="card">'
        "<h2>Per-Layer TRT Kernel Timing (IProfiler)</h2>"
        f'<p class="section-note">{note}</p>'
        '<canvas id="chart-layers" height="120"></canvas>'
        f'<script>PROFILE_DATA._layer_colors = {colors_js};</script>'
        "<table><thead><tr>"
        "<th>#</th><th>Layer</th><th>Mean (ms)</th><th>Std</th>"
        "<th>%</th><th>Cum %</th><th>Calls</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</div>"
    )


def _build_cpu_phase_section(cp: dict) -> str:
    phases = cp.get("phases", [])
    total_ms = cp.get("total_ms", 0.0)
    bottleneck = cp.get("bottleneck", "")
    meta = cp.get("metadata", {})

    rows = "".join(
        f"<tr><td>{p['phase']}</td>"
        f"<td>{p['mean_ms']:.4f}</td>"
        f"<td>{p['std_ms']:.4f}</td>"
        f"<td>{p['pct']:.1f}%</td>"
        f"<td>{p['samples']}</td></tr>"
        for p in phases
    )
    note = (f"Model: {meta.get('model','')} | Layers: {meta.get('num_layers','')} | "
            f"Total: {total_ms:.3f} ms/step | Bottleneck: {bottleneck}")

    return (
        '<div class="card">'
        "<h2>CPU Phase Breakdown (per decode step)</h2>"
        f'<p class="section-note">{note}</p>'
        '<canvas id="chart-cpu" height="60"></canvas>'
        "<table><thead><tr><th>Phase</th><th>Mean (ms)</th>"
        "<th>Std</th><th>%</th><th>Samples</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "</div>"
    )


def _build_nsight_section(nsight_trt: dict | None,
                          nsight_hf: dict | None,
                          nsight_cpp: dict | None = None) -> str:
    sections = []
    for label, backend, data in [
        ("TRT (C++)", "cpp", nsight_cpp),
        ("TRT (Python)", "trt", nsight_trt),
        ("HF", "hf", nsight_hf),
    ]:
        if not data:
            continue
        kernels = data.get("top_kernels", [])
        util = data.get("gpu_utilization_pct")
        meta = data.get("metadata", {})
        note = (f"Model: {meta.get('model','')} | GPU: {meta.get('gpu','')} | "
                f"Tool: {data.get('tool','nsys')}")
        if util is not None:
            note += f" | GPU util: {util:.1f}%"

        rows = "".join(
            f"<tr><td>{i}</td>"
            f"<td style='font-family:monospace;font-size:0.78rem'>"
            f"{k['name'][:60]}</td>"
            f"<td>{k['total_ms']:.4f}</td>"
            f"<td>{k['calls']}</td>"
            f"<td>{k['avg_us']:.3f}</td>"
            f"<td>{k.get('pct',0):.1f}%</td></tr>"
            for i, k in enumerate(kernels[:20], 1)
        )
        sections.append(
            f"<h2>Nsight Kernel Summary — {label}</h2>"
            f'<p class="section-note">{note}</p>'
            f'<canvas id="chart-nsys-{backend}" height="120"></canvas>'
            "<table><thead><tr><th>#</th><th>Kernel</th>"
            "<th>Total (ms)</th><th>Calls</th><th>Avg (us)</th><th>%</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )

    if not sections:
        return ""
    return '<div class="card">' + "\n".join(sections) + "</div>"


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def build_report(
    perf_compare: dict | None,
    layer_profile: dict | None,
    cpu_profile: dict | None,
    nsight_trt: dict | None,
    nsight_hf: dict | None,
    nsight_cpp: dict | None = None,
) -> str:
    body_parts = []

    # Title and metadata
    meta = {}
    if perf_compare:
        meta = perf_compare.get("metadata", {})
    elif layer_profile:
        meta = layer_profile.get("metadata", {})
    elif cpu_profile:
        meta = cpu_profile.get("metadata", {})

    model = meta.get("model", "Unknown model")
    gpu = meta.get("gpu", "")
    ts = meta.get("timestamp", "")
    title = f"TRT Profile — {model}"
    meta_line = " | ".join(filter(None, [gpu, ts]))

    # Speedup summary
    if perf_compare:
        body_parts.append(_build_speedup_section(perf_compare))
        body_parts.append(_build_latency_table(perf_compare))

    # C++ vs Python overhead analysis (needs both perf_compare with trt_cpp and cpu_profile)
    if perf_compare and perf_compare.get("trt_cpp"):
        body_parts.append(_build_cpp_vs_python_analysis(perf_compare, cpu_profile))

    # Per-layer timing
    if layer_profile:
        body_parts.append(_build_layer_section(layer_profile))

    # CPU phase breakdown
    if cpu_profile:
        body_parts.append(_build_cpu_phase_section(cpu_profile))

    # Nsight
    nsight_section = _build_nsight_section(nsight_trt, nsight_hf, nsight_cpp)
    if nsight_section:
        body_parts.append(nsight_section)

    if not body_parts:
        body_parts.append(
            '<div class="card"><p class="no-data">'
            "No profiling data provided. Pass at least one of: "
            "--perf-compare, --layer-profile, --cpu-profile, --nsight-*"
            "</p></div>")

    # Embed all data as JSON
    profile_data = {
        "perf_compare": perf_compare,
        "layer_profile": layer_profile,
        "cpu_profile": cpu_profile,
        "nsight_trt": nsight_trt,
        "nsight_hf": nsight_hf,
        "nsight_cpp": nsight_cpp,
    }

    return _HTML_TEMPLATE.format(
        title=title,
        meta_line=meta_line,
        body_html="\n".join(body_parts),
        profile_data_json=json.dumps(profile_data, indent=2),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-contained HTML profiling report")
    parser.add_argument("--perf-compare", metavar="JSON",
                        help="perf_compare.json from profile.py or perf_compare.py")
    parser.add_argument("--layer-profile", metavar="JSON",
                        help="layer_profile.json from profile.py")
    parser.add_argument("--cpu-profile", metavar="JSON",
                        help="cpu_profile.json from cpu_profile.py")
    parser.add_argument("--nsight-trt", metavar="JSON",
                        help="Nsight JSON for TRT (Python) backend")
    parser.add_argument("--nsight-hf", metavar="JSON",
                        help="Nsight JSON for HF backend")
    parser.add_argument("--nsight-cpp", metavar="JSON",
                        help="Nsight JSON for C++ binary backend")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="Auto-discover *.json artifacts from this directory")
    parser.add_argument("-o", "--output", default="report.html",
                        help="Output HTML file (default: report.html)")
    args = parser.parse_args()

    # Auto-discover from --output-dir if provided
    auto_dir = Path(args.output_dir) if args.output_dir else None
    if auto_dir:
        def _auto(name: str, explicit: str | None) -> str | None:
            if explicit:
                return explicit
            p = auto_dir / name
            return str(p) if p.exists() else None

        args.perf_compare = _auto("perf_compare.json", args.perf_compare)
        args.layer_profile = _auto("layer_profile.json", args.layer_profile)
        args.cpu_profile = _auto("cpu_profile.json", args.cpu_profile)
        args.nsight_trt = _auto("nsight_nsys_trt.json", args.nsight_trt)
        args.nsight_hf = _auto("nsight_nsys_hf.json", args.nsight_hf)
        args.nsight_cpp = _auto("nsight_nsys_cpp.json", args.nsight_cpp)

    html = build_report(
        perf_compare=_load_json(args.perf_compare),
        layer_profile=_load_json(args.layer_profile),
        cpu_profile=_load_json(args.cpu_profile),
        nsight_trt=_load_json(args.nsight_trt),
        nsight_hf=_load_json(args.nsight_hf),
        nsight_cpp=_load_json(args.nsight_cpp),
    )

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"[profile_report] Report saved to {out}", file=sys.stderr)
    print(f"[profile_report] Open with: open {out}  (or serve over HTTP)",
          file=sys.stderr)


if __name__ == "__main__":
    main()

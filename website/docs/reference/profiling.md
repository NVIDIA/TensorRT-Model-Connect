# Profiling Guide

Tools for measuring TRT vs HF performance, finding per-layer bottlenecks,
breaking down CPU-side overhead, and collecting GPU kernel traces.

---

## Quick start — one command for everything

```bash
python tools/trtmc_profile.py \
  --model Qwen/Qwen3-0.6B \
  --bundle ./engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --output-dir /tmp/qwen_profile \
  --json
```

This runs three passes in order (serial, no GPU memory contention):

1. **TRT + IProfiler** — e2e latency and per-layer kernel timing
2. **HF eager** — baseline latency
3. **HF torch.compile** — compiled latency

Prints a combined console report and saves `perf_compare.json` +
`layer_profile.json` to `--output-dir`.

---

## Tools overview

| Tool | What it does |
|------|-------------|
| `tools/trtmc_profile.py` | Unified entry point: 3-way benchmark + per-layer timing |
| `tools/perf_compare.py` | E2E latency: TRT vs HF eager vs torch.compile |
| `tools/layer_profiler.py` | TRT IProfiler wrapper (library, used by trtmc_profile.py) |
| `tools/cpu_profile.py` | CPU-phase timing breakdown for TRT decode steps |
| `tools/trtmc_profile.py --nsight` | Optional Nsight Systems collection through the unified profiler |
| `tools/profile_report.py` | HTML report from JSON artifacts |

---

## trtmc_profile.py — 3-way benchmark + per-layer timing

Runs all profiling passes in a single process so GPU memory is not split.

```bash
python tools/trtmc_profile.py \
  --model Qwen/Qwen3-0.6B \
  [--bundle /path/to/model.trtfb]   # skip engine build if bundle exists
  [--prompt "Hello"]
  [--max-new-tokens 20]
  [--max-cache-length 256]          # only used when building engine on the fly
  [--warmup 2]
  [--iterations 5]
  [--dtype float16|float32|bfloat16]
  [--trust-remote-code]
  [--no-compile]                    # skip torch.compile pass
  [--compile-mode reduce-overhead|default|max-autotune]
  [--no-layer-profile]              # skip IProfiler (faster e2e-only run)
  [--trtmc-binary ./build/trtmc]      # add C++ binary pass (requires --bundle)
  [--hf-python /opt/venv/bin/python]# python for C++ binary tokenizer
  [--output-dir /tmp/out]
  [--json]                          # save JSON artifacts to --output-dir
  [--verbose]
```

**Example output:**

```
════════════════════════════════════════════════════════════
  Qwen/Qwen3-0.6B  ·  NVIDIA H100  ·  TRT 10.0  ·  5 iters
════════════════════════════════════════════════════════════

  E2E Latency Comparison
────────────────────────────────────────────────────────────────────────────────
                        TRT (C++)  TRT (Python)  HF (eager)  HF (reduce-overhead)
  ──────────────       ──────────  ──────────────  ──────────────  ──────────────
  Prefill (ms)          9.5 +/- 0.2  10.1 +/- 0.3    5.2 +/- 0.2     6.1 +/- 0.3
  Decode (ms)          38.5 +/- 0.8  41.0 +/- 0.9   82.5 +/- 1.2    66.0 +/- 1.0
  Throughput (t/s)              519             488             242             303
  Speedup vs TRT (C++)            —           0.94x           0.47x           0.59x

  Token match (TRT Python vs HF eager): True

  Per-Layer TRT Kernel Timing  (total: 5.23 ms/step)  — top 15 slowest
────────────────────────────────────────────────────────────────────────────────
  Layer                                        Mean (ms)  Std      %
  ─────────────────────────────────────────    ─────────  ───────  ──────
  MatMul_attention_qkv_layer0                    1.2140   0.0220  23.2%
  Softmax_attention_layer0                       0.7830   0.0120  15.0%
  ...
  Bottleneck: 'MatMul_attention_qkv_layer0'  (1.2140 ms, 23.2%)
```

**JSON artifacts saved when `--json` is used:**

- `perf_compare.json` — 3-way latency comparison with speedup ratios
- `layer_profile.json` — per-layer timing, sorted by slowest, with percentages

---

## perf_compare.py — standalone e2e benchmark

Use this when you only need latency numbers without per-layer breakdown.

```bash
python tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  [--no-compile]
  [--compile-mode reduce-overhead|default|max-autotune]
  [--json results.json]
  [--warmup 2] [--iterations 5]
```

Output JSON schema (`results.json`):

```json
{
  "metadata": { "model": "...", "gpu": "...", "timestamp": "..." },
  "trt":         { "prefill_ms": {"mean": ..., "std": ...}, "decode_ms": {...}, ... },
  "hf":          { "prefill_ms": {...}, ... },
  "hf_compiled": { "compile_mode": "reduce-overhead", "prefill_ms": {...}, ... },
  "speedup": {
    "decode": 2.01,
    "trt_vs_compile_decode": 1.61
  },
  "token_match": true
}
```

---

## cpu_profile.py — CPU-phase breakdown

Shows where host-side time goes during a single TRT decode step. Useful for
understanding whether the bottleneck is CPU overhead (mask building, H2D
transfers, tensor binding) or actual GPU execution.

Phases instrumented:

| Phase | What it covers |
|-------|---------------|
| `mask_build` | Attention mask construction + host buffer preparation |
| `h2d` | H2D memcpy: token id, position id, mask |
| `tensor_bind` | `context.set_tensor_address()` calls (scales with num_layers) |
| `execute` | `execute_async_v3` dispatch + GPU kernel completion |
| `d2d_cache` | D2D KV cache update memcpy |
| `d2h` | Logits readback + stream sync |
| `argmax` | `np.argmax` on host logits |

> **Note:** Each phase ends with an explicit `cudaStreamSynchronize`, so the run
> is ~2–3× slower than production. Times are accurate per-phase attributions,
> not production latencies.

```bash
# Standard decoder (decoder_kv_cache, decoder_moe)
python tools/cpu_profile.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --max-new-tokens 10 \
  --warmup 3 --iterations 20 \
  --json cpu_profile.json

# Mamba/SSM model
python tools/cpu_profile.py \
  --model state-spaces/mamba-370m \
  --runner mamba \
  --max-new-tokens 10 \
  --json cpu_profile_mamba.json
```

**Example output:**

```
════════════════════════════════════════════════════════════
  CPU Phase Breakdown — Qwen/Qwen3-0.6B  (20 samples)
════════════════════════════════════════════════════════════
  Phase          Mean(ms)  Std     %
  ────────────   ────────  ──────  ──────
  mask_build        0.050   0.010    0.8%
  h2d               0.100   0.020    1.6%
  tensor_bind       0.300   0.050    4.9%
  execute           5.000   0.200   82.0%  ← BOTTLENECK
  d2d_cache         0.400   0.050    6.6%
  d2h               0.200   0.030    3.3%
  argmax            0.050   0.010    0.8%
```

If `execute` dominates (> ~75%), the bottleneck is GPU compute — focus on
TRT engine optimization or look at the per-layer profile. If `tensor_bind` or
`h2d` are large, the bottleneck is CPU overhead.

---

## nsight_collect.py — GPU kernel traces

Wraps `nsys` (Nsight Systems) and `ncu` (Nsight Compute) to automate profiling
and parse results into structured JSON.

**Prerequisites:**

- `nsys` — included in CUDA toolkit; works with standard container privileges.
- `ncu` — requires `--privileged` Docker flag (needs `CAP_SYS_ADMIN`).

```bash
# nsys: TRT backend (kernel timelines + CUDA API summary)
python tools/nsight_collect.py \
  --mode nsys \
  --backend trt \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --max-new-tokens 5 \
  --output-dir /tmp/nsight_out \
  --json nsight_trt.json

# nsys: HuggingFace eager backend
python tools/nsight_collect.py \
  --mode nsys \
  --backend hf \
  --model Qwen/Qwen3-0.6B \
  --output-dir /tmp/nsight_out \
  --json nsight_hf.json

# nsys: both backends in one run
python tools/nsight_collect.py \
  --mode nsys \
  --backend all \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --output-dir /tmp/nsight_out

# ncu: per-kernel hardware metrics (requires --privileged)
python tools/nsight_collect.py \
  --mode ncu \
  --backend trt \
  --model Qwen/Qwen3-0.6B \
  --bundle /path/to/qwen3.trtfb \
  --output-dir /tmp/nsight_out
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `nsys` | `nsys` or `ncu` |
| `--backend` | `trt` | `trt`, `hf`, `hf_compile`, or `all` |
| `--model` | required | HF repo ID or local model dir |
| `--bundle` | — | Pre-built `.trtfb` (TRT backend only; skips engine build) |
| `--max-new-tokens` | 10 | Decode steps per profiling run |
| `--dtype` | `float16` | HF model dtype |
| `--compile-mode` | `reduce-overhead` | torch.compile mode for `hf_compile` backend |
| `--top-n` | 20 | Number of top kernels to report |
| `--json` | — | Path to save parsed JSON output |
| `--verbose` | — | Show nsys/ncu stdout |

**Output JSON schema (nsys):**

```json
{
  "tool": "nsys",
  "backend": "trt",
  "top_kernels": [
    {"name": "volta_fp16_s884gemm", "total_ms": 3.2, "calls": 48,
     "avg_us": 66.7, "pct": 55.0}
  ],
  "total_kernel_ms": 5.8,
  "cuda_api_summary": [
    {"name": "cudaMemcpyAsync", "total_ms": 0.3, "calls": 96, "avg_us": 3.1}
  ],
  "gpu_utilization_pct": 87.3
}
```

**Output JSON schema (ncu):**

```json
{
  "tool": "ncu",
  "backend": "trt",
  "kernel_metrics": [
    {"kernel": "volta_fp16_s884gemm",
     "sm_throughput_pct": 84.2,
     "dram_bw_pct": 61.5,
     "achieved_occupancy_pct": 78.0}
  ]
}
```

---

## profile_report.py — HTML report

Combines all JSON artifacts into a self-contained HTML report with Chart.js
charts. All inputs are optional — the report adapts to whatever is provided.

```bash
# From JSON files
python tools/profile_report.py \
  --perf-compare perf_compare.json \
  --layer-profile layer_profile.json \
  --cpu-profile cpu_profile.json \
  --nsight-trt nsight_trt.json \
  --nsight-hf nsight_hf.json \
  -o report.html

# Auto-discover JSONs from a directory (trtmc_profile.py output)
python tools/profile_report.py \
  --output-dir /tmp/qwen_profile \
  -o report.html
```

The report includes:
- **Speedup badges** — green (TRT faster) / red (TRT slower) per phase
- **3-way latency table** — TRT / HF eager / HF compiled
- **Grouped bar chart** — prefill vs decode latency across backends
- **Per-layer horizontal bar chart** — top N slowest TRT layers with color coding by op type
- **CPU phase bar chart** — ms per phase with bottleneck highlighted
- **Nsight kernel pie charts** — GPU time share per kernel (TRT and HF side-by-side)

---

## Full workflow for a new model

```bash
MODEL="Qwen/Qwen3-0.6B"
BUNDLE="./engines/qwen3-0.6b.trtfb"
OUT="/tmp/profile_${MODEL//\//-}"

# 1. 4-way benchmark (C++ + TRT Python + HF eager + HF compile) + per-layer timing
python tools/trtmc_profile.py \
  --model "$MODEL" --bundle "$BUNDLE" \
  --max-new-tokens 20 --warmup 3 --iterations 10 \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --output-dir "$OUT" --json

# 2. CPU-phase breakdown
python tools/cpu_profile.py \
  --model "$MODEL" --bundle "$BUNDLE" \
  --max-new-tokens 10 --warmup 3 --iterations 20 \
  --json "$OUT/cpu_profile.json"

# 3. Nsight Systems (kernel timeline)
python tools/nsight_collect.py \
  --mode nsys --backend all \
  --model "$MODEL" --bundle "$BUNDLE" \
  --max-new-tokens 5 --output-dir "$OUT/nsight" \
  --json "$OUT/nsight_trt.json"

# 4. HTML report
python tools/profile_report.py \
  --output-dir "$OUT" \
  --nsight-trt "$OUT/nsight_trt.json" \
  -o "$OUT/report.html"

echo "Report: $OUT/report.html"
```

---

## Interpreting results

| Observation | Likely cause | Action |
|-------------|-------------|--------|
| `execute` > 75% in cpu_profile | GPU compute bound | Check per-layer profile for slowest ops |
| `tensor_bind` > 10% | Too many tensor address calls per step | Check num_layers, consider bind caching |
| `h2d` > 5% | H2D transfer overhead | Check mask/input sizes |
| MatMul dominates per-layer | Expected for transformer models | Check occupancy via ncu |
| TRT decode slower than HF | Inefficient engine, graph issues | Check layer profile + ncu sm_throughput_pct |
| Low `sm_throughput_pct` (< 50%) in ncu | Kernel not filling the GPU | Batch size / seq len / precision issue |
| Low `dram_bw_pct` + low `sm_throughput_pct` | Memory-bound small kernel | Fusion opportunity |

---

## Running as a diff test check

The `layer_profile` check is registered in the diff framework and runs automatically
when you use `diff_logits.py` or `diff_framework`:

```bash
python tools/diff_logits.py \
  --model Qwen/Qwen3-0.6B \
  --check layer_profile \
  --battery
```

The check attaches `LayerProfiler` to a `TrtRunner`, runs inference, and reports
the top slowest layers as diff test metrics.

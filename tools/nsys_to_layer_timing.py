#!/usr/bin/env python3
"""Convert Nsight Systems profile data into layer_timing.json for sol_estimate.py.

Reads an nsys .nsys-rep (or pre-exported .sqlite) file, extracts per-kernel
timing from the CUPTI_ACTIVITY_KIND_KERNEL table, groups kernels into decode
steps (via CUDA Graph graphId), identifies the repeating per-layer kernel
pattern, and outputs per-layer timing JSON compatible with:

    python3 tools/sol_estimate.py --layer-timing-json layer_timing.json

Usage:
    # From .nsys-rep (auto-exports to .sqlite via nsys stats)
    python3 tools/nsys_to_layer_timing.py \\
        --input profile.nsys-rep --output layer_timing.json

    # From pre-exported .sqlite
    python3 tools/nsys_to_layer_timing.py \\
        --sqlite profile.sqlite --output layer_timing.json

    # With explicit layer count (skips auto-detection)
    python3 tools/nsys_to_layer_timing.py \\
        --input profile.nsys-rep --output layer_timing.json --model-layers 28

    # Skip more warmup steps
    python3 tools/nsys_to_layer_timing.py \\
        --input profile.nsys-rep --output layer_timing.json --warmup-steps 10

    # Verbose diagnostics
    python3 tools/nsys_to_layer_timing.py \\
        --input profile.nsys-rep --output layer_timing.json --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KernelRecord:
    """A single GPU kernel execution."""
    start_ns: int
    end_ns: int
    name: str
    graph_id: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1e6


@dataclass
class LayerTiming:
    """Aggregated timing for one transformer layer."""
    name: str
    time_ms: float
    kernel_count: int


# ---------------------------------------------------------------------------
# SQLite query
# ---------------------------------------------------------------------------

def export_sqlite(nsys_rep_path: str, verbose: bool = False) -> str:
    """Run `nsys stats --force-export` to produce .sqlite alongside .nsys-rep.

    Returns the path to the .sqlite file.
    """
    sqlite_path = str(nsys_rep_path).replace(".nsys-rep", ".sqlite")

    if os.path.exists(sqlite_path):
        if verbose:
            print(f"  [info] SQLite already exists: {sqlite_path}",
                  file=sys.stderr)
        return sqlite_path

    if verbose:
        print(f"  [info] Exporting SQLite from: {nsys_rep_path}",
              file=sys.stderr)

    cmd = [
        "nsys", "stats",
        "--force-export", "true",
        "--format", "sqlite",
        "--output", ".",
        str(nsys_rep_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        print("ERROR: 'nsys' not found in PATH. Install Nsight Systems or "
              "provide --sqlite directly.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: nsys stats timed out after 120s.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        # nsys stats sometimes prints to stdout even on error
        print(f"ERROR: nsys stats failed (rc={result.returncode}):",
              file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(sqlite_path):
        print(f"ERROR: Expected SQLite output not found at: {sqlite_path}",
              file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  [info] Exported: {sqlite_path}", file=sys.stderr)

    return sqlite_path


def load_kernels(sqlite_path: str, verbose: bool = False) -> list[KernelRecord]:
    """Load kernel records from nsys SQLite database.

    Queries CUPTI_ACTIVITY_KIND_KERNEL joined with StringIds to resolve
    kernel names. Returns records sorted by start time.
    """
    if not os.path.exists(sqlite_path):
        print(f"ERROR: SQLite file not found: {sqlite_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Check that the required tables exist
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if kernel_table not in tables:
        print(f"ERROR: Table {kernel_table} not found in {sqlite_path}. "
              f"Available tables: {sorted(tables)}", file=sys.stderr)
        sys.exit(1)

    if "StringIds" not in tables:
        print(f"ERROR: Table StringIds not found in {sqlite_path}.",
              file=sys.stderr)
        sys.exit(1)

    # Check available columns to handle schema variations
    cols = {row[1] for row in conn.execute(
        f"PRAGMA table_info({kernel_table})").fetchall()}

    has_graph_id = "graphId" in cols

    if has_graph_id:
        query = f"""
            SELECT k.start, k.end, s.value AS kernelName, k.graphId
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            ORDER BY k.start
        """
    else:
        query = f"""
            SELECT k.start, k.end, s.value AS kernelName, 0 AS graphId
            FROM {kernel_table} k
            JOIN StringIds s ON k.shortName = s.id
            ORDER BY k.start
        """

    rows = conn.execute(query).fetchall()
    conn.close()

    kernels = [
        KernelRecord(
            start_ns=row["start"],
            end_ns=row["end"],
            name=row["kernelName"],
            graph_id=row["graphId"],
        )
        for row in rows
    ]

    if verbose:
        print(f"  [info] Loaded {len(kernels)} kernel records from SQLite",
              file=sys.stderr)
        if kernels:
            graph_ids = sorted(set(k.graph_id for k in kernels))
            print(f"  [info] Graph IDs present: {graph_ids}", file=sys.stderr)

    return kernels


# ---------------------------------------------------------------------------
# Decode step segmentation
# ---------------------------------------------------------------------------

def find_cuda_graph_id(
    kernels: list[KernelRecord],
    engine_section: str = "primary",
    verbose: bool = False,
) -> int:
    """Identify the CUDA Graph ID for the target engine section.

    engine_section:
        "primary"   — graph with most kernel executions (default, backward compat)
        "secondary" — second most-executed graph (e.g., vision encoder in VL)
        "<number>"  — specific CUDA graph ID
    """
    from collections import Counter
    graph_counts: Counter[int] = Counter()
    for k in kernels:
        if k.graph_id > 0:
            graph_counts[k.graph_id] += 1

    if not graph_counts:
        print("ERROR: No CUDA Graph kernels found (all graphId=0). "
              "Is CUDA Graphs enabled for this profile?", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("  [info] CUDA Graph distribution:", file=sys.stderr)
        for gid, cnt in graph_counts.most_common(10):
            print(f"         graphId={gid}: {cnt} kernels", file=sys.stderr)

    # Resolve engine_section to a graph ID
    if engine_section.isdigit():
        graph_id = int(engine_section)
        if graph_id not in graph_counts:
            print(f"ERROR: Graph ID {graph_id} not found. Available: "
                  f"{sorted(graph_counts.keys())}", file=sys.stderr)
            sys.exit(1)
    elif engine_section == "primary":
        graph_id = graph_counts.most_common(1)[0][0]
    elif engine_section == "secondary":
        if len(graph_counts) < 2:
            print("ERROR: Only 1 CUDA Graph found — no secondary engine. "
                  f"Available: {sorted(graph_counts.keys())}", file=sys.stderr)
            sys.exit(1)
        graph_id = graph_counts.most_common(2)[1][0]
    else:
        print(f"ERROR: Unknown --engine-section value: '{engine_section}'. "
              f"Use 'primary', 'secondary', 'all', or a graph ID number.",
              file=sys.stderr)
        sys.exit(1)

    count = graph_counts[graph_id]
    if verbose:
        print(f"  [info] Selected graphId={graph_id} "
              f"({count} kernels, section={engine_section})", file=sys.stderr)

    return graph_id


def segment_decode_steps(
    kernels: list[KernelRecord],
    graph_id: int,
    verbose: bool = False,
) -> list[list[KernelRecord]]:
    """Segment kernel list into per-step groups.

    All kernels with the target graph_id are extracted and split into
    steps. Steps are detected by large time gaps between consecutive
    kernels (inter-step gap >> intra-step gap).
    """
    graph_kernels = [k for k in kernels if k.graph_id == graph_id]

    if len(graph_kernels) < 2:
        print(f"ERROR: Only {len(graph_kernels)} kernels for graphId={graph_id}. "
              "Need at least 2 to form steps.", file=sys.stderr)
        sys.exit(1)

    # Compute gaps between consecutive kernels
    gaps = []
    for i in range(1, len(graph_kernels)):
        gap = graph_kernels[i].start_ns - graph_kernels[i - 1].end_ns
        gaps.append(gap)

    # The inter-step gap is much larger than intra-step gaps.
    # Use the median gap as baseline and look for gaps > 10x median.
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]

    # Threshold: 10x the median intra-kernel gap, with a minimum of 1us
    # to avoid false splits on very tight kernels
    threshold = max(median_gap * 10, 1000)  # ns

    if verbose:
        print(f"  [info] Gap statistics (ns): min={sorted_gaps[0]}, "
              f"median={median_gap}, max={sorted_gaps[-1]}, "
              f"threshold={threshold}", file=sys.stderr)

    # Split into steps at large gaps
    steps: list[list[KernelRecord]] = []
    current_step: list[KernelRecord] = [graph_kernels[0]]

    for i in range(1, len(graph_kernels)):
        gap = graph_kernels[i].start_ns - graph_kernels[i - 1].end_ns
        if gap > threshold:
            steps.append(current_step)
            current_step = [graph_kernels[i]]
        else:
            current_step.append(graph_kernels[i])

    if current_step:
        steps.append(current_step)

    if verbose:
        step_sizes = [len(s) for s in steps]
        print(f"  [info] Found {len(steps)} decode steps, "
              f"kernels/step: {sorted(set(step_sizes))}", file=sys.stderr)

    return steps


# ---------------------------------------------------------------------------
# Pattern detection: auto-detect kernels_per_layer
# ---------------------------------------------------------------------------

def detect_kernels_per_layer(
    step: list[KernelRecord],
    model_layers: int | None = None,
    engine_section: str = "primary",
    verbose: bool = False,
) -> tuple[int, int]:
    """Auto-detect the number of kernels per transformer layer.

    Supports multiple architectures:
    - Standard decoder: [emb_norm] [layer_0..N] [lm_head]
    - ViT encoder:      [patch_embed] [block_0..N] [norm] [head]
    - Encoder-only:     [embed] [layer_0..N] [pooler]

    The engine_section hint helps choose between decoder (primary) and
    encoder (secondary) heuristics. For 'secondary', tries more prefix/suffix
    combinations since vision encoders have different kernel patterns.

    We detect the repeating pattern by:
    1. If model_layers is known: kernels_per_layer = (total - prefix - suffix) / model_layers
    2. If model_layers is unknown: find the period by checking when the
       kernel name sequence repeats.

    Returns:
        (kernels_per_layer, detected_model_layers)
    """
    total = len(step)
    names = [k.name for k in step]

    # Vision encoders and other non-decoder architectures may have larger
    # prefix (patch_embed: 2-5 kernels) and suffix (norm+head: 2-4 kernels).
    # Expand search range for secondary engines.
    max_prefix_suffix = 4 if engine_section == "primary" else 8

    if model_layers is not None:
        # Known layer count: try prefix/suffix combos to find matching body
        for prefix in range(0, min(max_prefix_suffix, total)):
            for suffix in range(0, min(max_prefix_suffix, total - prefix)):
                body = total - prefix - suffix
                if body > 0 and body % model_layers == 0:
                    kpl = body // model_layers
                    if verbose:
                        print(f"  [info] With prefix={prefix}, suffix={suffix}: "
                              f"kpl={kpl}", file=sys.stderr)
                    return kpl, model_layers
        print(f"ERROR: Cannot divide {total} kernels into {model_layers} "
              f"layers evenly (tried prefix/suffix 0-{max_prefix_suffix - 1}).",
              file=sys.stderr)
        sys.exit(1)

    # Auto-detect: find the repeating period in the kernel name sequence.
    # Try multiple prefix/suffix combinations to handle different architectures:
    # - Decoder: prefix=1 (emb_norm), suffix=1 (lm_head)
    # - ViT:     prefix=2-5 (patch_embed+pos), suffix=2-4 (norm+head)
    # - Encoder: prefix=1 (embed), suffix=1 (pooler)
    for prefix in range(0, min(max_prefix_suffix, total)):
        for suffix in range(0, min(max_prefix_suffix, total - prefix)):
            body = names[prefix:total - suffix] if suffix > 0 else names[prefix:]
            blen = len(body)
            if blen < 2:
                continue
            for period in range(1, blen // 2 + 1):
                if blen % period != 0:
                    continue
                match = True
                for i in range(period, blen):
                    if body[i] != body[i % period]:
                        match = False
                        break
                if match:
                    n_layers = blen // period
                    if n_layers >= 2:  # at least 2 layers for a real model
                        if verbose:
                            print(f"  [info] Auto-detect: prefix={prefix}, "
                                  f"suffix={suffix}, period={period}, "
                                  f"layers={n_layers}", file=sys.stderr)
                        return period, n_layers

    print("ERROR: Could not detect repeating kernel pattern. "
          "Use --model-layers to specify explicitly.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Layer timing extraction
# ---------------------------------------------------------------------------

def extract_layer_timings(
    step: list[KernelRecord],
    kernels_per_layer: int,
    model_layers: int,
    verbose: bool = False,
) -> tuple[list[LayerTiming], float, float]:
    """Extract per-layer timing from a single decode step.

    Returns:
        (layer_timings, lm_head_ms, emb_norm_ms)
    """
    total = len(step)
    expected_body = model_layers * kernels_per_layer

    # Determine prefix and suffix sizes
    # Default: 1 kernel prefix (emb_norm), 1 kernel suffix (lm_head)
    # But verify by checking what makes the body match expected_body
    prefix = 0
    suffix = 0
    for p in range(0, min(4, total)):
        for s in range(0, min(4, total - p)):
            body = total - p - s
            if body == expected_body:
                prefix = p
                suffix = s
                break
        else:
            continue
        break

    if total - prefix - suffix != expected_body:
        # Could not find matching prefix/suffix
        if verbose:
            print(f"  [warn] Cannot match prefix/suffix: total={total}, "
                  f"expected_body={expected_body}", file=sys.stderr)
        # Force: assume 1 prefix, 1 suffix
        prefix = 1
        suffix = 1

    # Embedding norm timing (prefix kernels)
    emb_norm_ms = 0.0
    for i in range(prefix):
        emb_norm_ms += step[i].duration_ms

    # LM head timing (suffix kernels)
    lm_head_ms = 0.0
    body_end = total - suffix
    for i in range(body_end, total):
        lm_head_ms += step[i].duration_ms

    # Per-layer timing
    layers: list[LayerTiming] = []
    for layer_idx in range(model_layers):
        start = prefix + layer_idx * kernels_per_layer
        end = start + kernels_per_layer
        layer_kernels = step[start:end]
        layer_ms = sum(k.duration_ms for k in layer_kernels)
        layers.append(LayerTiming(
            name=f"layer_{layer_idx}",
            time_ms=layer_ms,
            kernel_count=len(layer_kernels),
        ))

    return layers, lm_head_ms, emb_norm_ms


def average_layer_timings(
    all_steps_timings: list[tuple[list[LayerTiming], float, float]],
) -> tuple[list[LayerTiming], float]:
    """Average layer timings across multiple decode steps.

    Returns:
        (averaged_layer_timings, averaged_lm_head_ms)
    """
    n_steps = len(all_steps_timings)
    if n_steps == 0:
        return [], 0.0

    n_layers = len(all_steps_timings[0][0])

    avg_layers: list[LayerTiming] = []
    for layer_idx in range(n_layers):
        times = [step[0][layer_idx].time_ms for step in all_steps_timings]
        avg_ms = sum(times) / len(times)
        avg_layers.append(LayerTiming(
            name=f"layer_{layer_idx}",
            time_ms=avg_ms,
            kernel_count=all_steps_timings[0][0][layer_idx].kernel_count,
        ))

    avg_lm_head = sum(step[1] for step in all_steps_timings) / n_steps

    return avg_layers, avg_lm_head


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_layer_timing_json(
    layers: list[LayerTiming],
    lm_head_ms: float,
) -> dict:
    """Build the output JSON structure for sol_estimate.py --layer-timing-json."""
    total_ms = sum(lt.time_ms for lt in layers) + lm_head_ms
    return {
        "layers": [
            {"name": lt.name, "time_ms": round(lt.time_ms, 4)}
            for lt in layers
        ],
        "lm_head_ms": round(lm_head_ms, 4),
        "total_ms": round(total_ms, 4),
    }


def process_profile(
    sqlite_path: str,
    output_path: str | None = None,
    warmup_steps: int = 5,
    model_layers: int | None = None,
    engine_section: str = "primary",
    verbose: bool = False,
) -> dict:
    """End-to-end: load kernels, detect pattern, compute layer timing.

    Returns the layer_timing dict. If output_path is set, also writes JSON.
    """
    # 1. Load kernels
    kernels = load_kernels(sqlite_path, verbose=verbose)
    if not kernels:
        print("ERROR: No kernel records found in SQLite.", file=sys.stderr)
        sys.exit(1)

    # 2. Find the target CUDA Graph ID
    if engine_section == "all":
        # "all" mode: use all non-zero graph IDs (skip non-graph kernels)
        from collections import Counter
        graph_counts: Counter[int] = Counter()
        for k in kernels:
            if k.graph_id > 0:
                graph_counts[k.graph_id] += 1
        if not graph_counts:
            print("ERROR: No CUDA Graph kernels found.", file=sys.stderr)
            sys.exit(1)
        graph_id = graph_counts.most_common(1)[0][0]
        if verbose:
            print(f"  [info] engine_section=all: using all graph IDs, "
                  f"primary={graph_id} for step segmentation", file=sys.stderr)
    else:
        graph_id = find_cuda_graph_id(
            kernels, engine_section=engine_section, verbose=verbose)

    # 3. Segment into decode steps
    steps = segment_decode_steps(kernels, graph_id, verbose=verbose)

    # Validate step consistency: all steps should have the same kernel count
    step_sizes = set(len(s) for s in steps)
    if len(step_sizes) > 1:
        # Filter to the most common step size (handles partial first/last steps)
        from collections import Counter
        size_counts = Counter(len(s) for s in steps)
        common_size, _ = size_counts.most_common(1)[0]
        if verbose:
            print(f"  [info] Step sizes vary: {dict(size_counts)}. "
                  f"Keeping only size={common_size}", file=sys.stderr)
        steps = [s for s in steps if len(s) == common_size]

    if not steps:
        print("ERROR: No valid decode steps after filtering.", file=sys.stderr)
        sys.exit(1)

    kernels_per_step = len(steps[0])
    if verbose:
        print(f"  [info] {len(steps)} decode steps, "
              f"{kernels_per_step} kernels each", file=sys.stderr)

    # 4. Auto-detect kernels_per_layer
    kpl, detected_layers = detect_kernels_per_layer(
        steps[0], model_layers=model_layers,
        engine_section=engine_section, verbose=verbose,
    )

    if verbose:
        print(f"  [info] Pattern: {kpl} kernels/layer, "
              f"{detected_layers} layers", file=sys.stderr)

    # 5. Skip warmup steps
    if warmup_steps >= len(steps):
        warmup_steps = max(0, len(steps) - 1)
        if verbose:
            print(f"  [warn] Reduced warmup to {warmup_steps} "
                  f"(only {len(steps)} steps total)", file=sys.stderr)

    active_steps = steps[warmup_steps:]
    if verbose:
        print(f"  [info] Using {len(active_steps)} steps "
              f"(skipped {warmup_steps} warmup)", file=sys.stderr)

    # 6. Extract per-layer timing from each step
    all_timings = []
    for step in active_steps:
        timings = extract_layer_timings(
            step, kpl, detected_layers, verbose=verbose,
        )
        all_timings.append(timings)

    # 7. Average across steps
    avg_layers, avg_lm_head = average_layer_timings(all_timings)

    # 8. Build output JSON
    result = build_layer_timing_json(avg_layers, avg_lm_head)

    # Add metadata
    result["metadata"] = {
        "source": sqlite_path,
        "engine_section": engine_section,
        "total_steps": len(steps),
        "warmup_steps": warmup_steps,
        "active_steps": len(active_steps),
        "kernels_per_step": kernels_per_step,
        "kernels_per_layer": kpl,
        "model_layers": detected_layers,
        "graph_id": graph_id,
    }

    # Print summary
    print("\nLayer Timing Summary")
    print(f"{'=' * 55}")
    print(f"  Source:            {Path(sqlite_path).name}")
    print(f"  Decode steps:      {len(steps)} total, "
          f"{len(active_steps)} used (skip {warmup_steps} warmup)")
    print(f"  Kernels/step:      {kernels_per_step}")
    print(f"  Model layers:      {detected_layers}")
    print(f"  Kernels/layer:     {kpl}")
    print(f"  Total step time:   {result['total_ms']:.3f} ms")
    print(f"  Avg layer time:    "
          f"{sum(l.time_ms for l in avg_layers) / len(avg_layers):.4f} ms")
    print(f"  LM head time:      {avg_lm_head:.4f} ms")

    # Show per-layer breakdown
    print(f"\n  {'Layer':<12} {'Time (ms)':>10} {'Kernels':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8}")
    for lt in avg_layers:
        print(f"  {lt.name:<12} {lt.time_ms:>10.4f} {lt.kernel_count:>8}")
    print(f"  {'lm_head':<12} {avg_lm_head:>10.4f}")
    print(f"  {'-'*12} {'-'*10}")
    print(f"  {'total':<12} {result['total_ms']:>10.4f}")

    # Check layer time variance
    times = [lt.time_ms for lt in avg_layers]
    if times:
        mean_t = sum(times) / len(times)
        var = sum((t - mean_t) ** 2 for t in times) / len(times)
        std = math.sqrt(var)
        cv = (std / mean_t * 100) if mean_t > 0 else 0
        if cv > 10:
            print(f"\n  [note] Layer time CV={cv:.1f}% (>10%). "
                  f"Some layers may be outliers.")

    print(f"{'=' * 55}")

    # 9. Write output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWritten to: {output_path}")
        print(f"Use with:   python3 tools/sol_estimate.py "
              f"--layer-timing-json {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert nsys profile data to layer_timing.json "
                    "for sol_estimate.py --layer-timing-json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From .nsys-rep (auto-exports .sqlite)
  python3 tools/nsys_to_layer_timing.py \\
      --input profile.nsys-rep --output layer_timing.json

  # From pre-exported .sqlite
  python3 tools/nsys_to_layer_timing.py \\
      --sqlite profile.sqlite --output layer_timing.json

  # With known layer count
  python3 tools/nsys_to_layer_timing.py \\
      --input profile.nsys-rep --output layer_timing.json --model-layers 28

  # Feed into sol_estimate.py
  python3 tools/sol_estimate.py --model example-org/example-decoder --gpu B200 --dtype fp32 \\
      --cache-length 256 --layer-timing-json layer_timing.json
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", dest="nsys_rep",
        help="Path to .nsys-rep file (will auto-export .sqlite)",
    )
    input_group.add_argument(
        "--sqlite",
        help="Path to pre-exported .sqlite file",
    )

    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path for layer_timing.json (default: print to stdout)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5,
        help="Number of initial decode steps to skip as warmup (default: 5)",
    )
    parser.add_argument(
        "--model-layers", type=int, default=None,
        help="Number of transformer layers (auto-detected if omitted)",
    )
    parser.add_argument(
        "--engine-section", default="primary",
        help="Which engine to analyze: 'primary' (default, most-executed graph), "
             "'secondary' (second graph, e.g., vision encoder), 'all' (combined), "
             "or a specific CUDA graph ID number",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed diagnostics to stderr",
    )

    args = parser.parse_args()

    # Resolve SQLite path
    if args.nsys_rep:
        if not os.path.exists(args.nsys_rep):
            print(f"ERROR: File not found: {args.nsys_rep}", file=sys.stderr)
            sys.exit(1)
        sqlite_path = export_sqlite(args.nsys_rep, verbose=args.verbose)
    else:
        sqlite_path = args.sqlite
        if not os.path.exists(sqlite_path):
            print(f"ERROR: File not found: {sqlite_path}", file=sys.stderr)
            sys.exit(1)

    # Process
    result = process_profile(
        sqlite_path=sqlite_path,
        output_path=args.output,
        warmup_steps=args.warmup_steps,
        model_layers=args.model_layers,
        engine_section=args.engine_section,
        verbose=args.verbose,
    )

    # If no output path, dump JSON to stdout
    if not args.output:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

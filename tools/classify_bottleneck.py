#!/usr/bin/env python3
"""Automatic bottleneck classification from profiling data.

Takes nsys SQLite and/or L1 CPU profile JSON, applies a decision tree
based on empirical thresholds from B100/B200 experiments (2026-04-03),
and outputs the bottleneck classification + recommended techniques.

Usage:
    # From nsys SQLite (most detailed)
    python3 tools/classify_bottleneck.py --nsys-sqlite profile.sqlite

    # From L1 CPU profile JSON
    python3 tools/classify_bottleneck.py --l1-json profile_l1.json

    # Both (nsys takes precedence for kernel analysis)
    python3 tools/classify_bottleneck.py --nsys-sqlite profile.sqlite --l1-json profile_l1.json

    # JSON output
    python3 tools/classify_bottleneck.py --nsys-sqlite profile.sqlite --json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e_harness.runtime_strategy_metadata import runtime_strategy_performance_mode  # noqa: E402


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class BottleneckResult:
    classification: str  # sync | bandwidth | compute | latency | mixed
    confidence: str      # high | medium | low
    evidence: list[str] = field(default_factory=list)
    techniques: list[dict] = field(default_factory=list)  # [{name, impact, reason}]


# ---------------------------------------------------------------------------
# Technique knowledge base (empirical, 2026-04-03)
# ---------------------------------------------------------------------------

TECHNIQUES = {
    "gpu_argmax": {
        "name": "GPU-side argmax",
        "cli": "--set runtime.prefer_gpu_greedy=true",
        "impact_small_model": "+30%",
        "impact_large_model": "+7%",
    },
    "fp16": {
        "name": "FP16 precision",
        "cli": "--precision fp16",
        "impact": "+25-80% (combined with GPU argmax)",
    },
    "cuda_graphs": {
        "name": "CUDA Graphs",
        "cli": "enabled by default",
        "impact": "+10-15%",
    },
    "bf16": {
        "name": "BF16 precision",
        "cli": "--precision bf16",
        "impact": "similar to FP16, better numerical stability",
    },
}

# Per-mode technique priorities (mode → bottleneck → technique order)
TECHNIQUE_PRIORITY_BY_MODE: dict[str, dict[str, list[str]]] = {
    "decode": {
        "sync": ["gpu_argmax", "fp16", "cuda_graphs"],
        "bandwidth": ["fp16", "gpu_argmax", "cuda_graphs"],
        "compute": ["fp16", "bf16", "gpu_argmax"],
        "latency": ["cuda_graphs", "gpu_argmax", "fp16"],
        "mixed": ["gpu_argmax", "fp16", "cuda_graphs"],
    },
    "diffusion": {
        "compute": ["fp16", "bf16", "cuda_graphs"],
        "bandwidth": ["fp16", "bf16"],
        "latency": ["cuda_graphs", "fp16"],
        "mixed": ["fp16", "cuda_graphs"],
        "sync": ["fp16", "cuda_graphs"],
    },
    "enc_dec": {
        "sync": ["gpu_argmax", "fp16", "cuda_graphs"],
        "bandwidth": ["fp16", "gpu_argmax", "cuda_graphs"],
        "compute": ["fp16", "bf16"],
        "latency": ["cuda_graphs", "fp16"],
        "mixed": ["fp16", "gpu_argmax", "cuda_graphs"],
    },
    "single_pass": {
        "compute": ["fp16", "bf16"],
        "bandwidth": ["fp16", "bf16"],
        "latency": ["fp16"],
        "mixed": ["fp16", "bf16"],
        "sync": ["fp16"],
    },
    "multi_stage": {
        "sync": ["gpu_argmax", "fp16", "cuda_graphs"],
        "bandwidth": ["fp16", "cuda_graphs"],
        "compute": ["fp16", "bf16", "cuda_graphs"],
        "latency": ["cuda_graphs", "fp16"],
        "mixed": ["fp16", "cuda_graphs"],
    },
}

# Backward-compatible default
TECHNIQUE_PRIORITY = TECHNIQUE_PRIORITY_BY_MODE["decode"]

# ---------------------------------------------------------------------------
# Nsys-based classification
# ---------------------------------------------------------------------------

def _resolve_graph_id_filter(db: sqlite3.Connection, engine_section: str) -> int | None:
    """Resolve --engine-section to a graph ID for filtering, or None for 'all'."""
    if engine_section == "all":
        return None

    if engine_section.isdigit():
        return int(engine_section)

    # Query graph distribution
    try:
        rows = db.execute("""
            SELECT graphId, COUNT(*) as cnt
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE graphId > 0
            GROUP BY graphId
            ORDER BY cnt DESC
        """).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    if engine_section == "primary":
        return rows[0][0]
    elif engine_section == "secondary":
        if len(rows) < 2:
            return None
        return rows[1][0]

    return None


def classify_from_nsys(
    sqlite_path: str,
    pipeline_type: str = "decoder_kv_cache",
    engine_section: str = "all",
) -> BottleneckResult:
    """Classify bottleneck from nsys SQLite export.

    engine_section: 'all' (default), 'primary', 'secondary', or a graph ID.
    When not 'all', kernel analysis is filtered to the specified CUDA graph.
    """
    db = sqlite3.connect(sqlite_path)

    graph_id = _resolve_graph_id_filter(db, engine_section)
    graph_filter = f"AND k.graphId = {graph_id}" if graph_id is not None else ""

    # Get CUDA API summary
    api_rows = db.execute("""
        SELECT s.value as name, SUM(end - start) as total_ns, COUNT(*) as calls
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON r.nameId = s.id
        GROUP BY s.value
        ORDER BY total_ns DESC
    """).fetchall()

    # Fallback: try the simpler query if RUNTIME table doesn't exist
    if not api_rows:
        try:
            api_rows = db.execute(f"""
                SELECT 'unknown' as name, SUM(end - start) as total_ns, COUNT(*) as calls
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                WHERE 1=1 {graph_filter}
            """).fetchall()
        except Exception:
            pass

    # Get kernel summary (filtered by engine section)
    kernel_rows = db.execute(f"""
        SELECT s.value as name, SUM(k.end - k.start) as total_ns, COUNT(*) as calls,
               AVG(k.end - k.start) as avg_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE 1=1 {graph_filter}
        GROUP BY s.value
        ORDER BY total_ns DESC
    """).fetchall()

    # Get memory operation summary
    memcpy_rows = db.execute("""
        SELECT copyKind, SUM(bytes) as total_bytes, COUNT(*) as calls,
               SUM(end - start) as total_ns
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
    """).fetchall()

    # Analyze
    evidence = []
    if graph_id is not None:
        evidence.append(f"Filtered to engine_section={engine_section} (graphId={graph_id})")
    scores = {"sync": 0.0, "bandwidth": 0.0, "compute": 0.0, "latency": 0.0}

    # --- Kernel analysis ---
    total_kernel_ns = sum(r[1] for r in kernel_rows)
    total_kernel_calls = sum(r[2] for r in kernel_rows)
    avg_kernel_ns = total_kernel_ns / max(total_kernel_calls, 1)

    if total_kernel_calls > 0:
        evidence.append(f"GPU kernels: {total_kernel_calls} calls, "
                        f"{total_kernel_ns/1e6:.1f}ms total, "
                        f"{avg_kernel_ns:.0f}ns avg")

    # Small kernels (<10us avg) → latency-bound
    if avg_kernel_ns < 10000:
        scores["latency"] += 3
        evidence.append(f"Avg kernel {avg_kernel_ns:.0f}ns < 10us → latency-bound")

    # Check GEMV vs GEMM ratio
    gemv_ns = sum(r[1] for r in kernel_rows if "gemvx" in r[0].lower() or "gemv" in r[0].lower())
    gemm_ns = sum(r[1] for r in kernel_rows if "gemm" in r[0].lower() and "gemv" not in r[0].lower())

    if total_kernel_ns > 0:
        gemv_pct = gemv_ns / total_kernel_ns * 100
        gemm_pct = gemm_ns / total_kernel_ns * 100
        evidence.append(f"GEMV: {gemv_pct:.0f}%, GEMM: {gemm_pct:.0f}% of kernel time")

        if gemv_pct > 40:
            scores["bandwidth"] += 2
            evidence.append("GEMV dominant → bandwidth-bound (batch=1, weight reads)")
        if gemm_pct > 40:
            scores["compute"] += 2
            evidence.append("GEMM dominant → compute-bound (large matrices, tensor cores)")

    # --- Memory transfer analysis ---
    d2h_bytes = 0
    d2h_calls = 0
    for kind, total_bytes, calls, total_ns in memcpy_rows:
        if kind == 2:  # D2H
            d2h_bytes = total_bytes
            d2h_calls = calls

    if d2h_calls > 100 and d2h_bytes > 10 * 1024 * 1024:  # >100 calls, >10MB
        scores["sync"] += 3
        evidence.append(f"D2H: {d2h_calls} calls, {d2h_bytes/1e6:.1f}MB → sync-bound "
                        "(likely logit copies for CPU argmax)")
    elif d2h_calls > 0:
        evidence.append(f"D2H: {d2h_calls} calls, {d2h_bytes/1e6:.1f}MB (low)")

    # --- Determine classification ---
    best = max(scores, key=scores.get)
    best_score = scores[best]
    second_score = sorted(scores.values(), reverse=True)[1]

    if best_score == 0:
        classification = "mixed"
        confidence = "low"
        evidence.append("No clear dominant bottleneck → mixed")
    elif best_score - second_score < 1:
        classification = "mixed"
        confidence = "medium"
        evidence.append(f"Close scores: {scores} → mixed")
    else:
        classification = best
        confidence = "high" if best_score >= 3 else "medium"

    # Build technique recommendations (mode-aware)
    mode = runtime_strategy_performance_mode(pipeline_type, default="decode")
    mode_priorities = TECHNIQUE_PRIORITY_BY_MODE.get(mode, TECHNIQUE_PRIORITY)
    techniques = []
    for tech_key in mode_priorities.get(classification, ["fp16"]):
        tech = TECHNIQUES[tech_key]
        techniques.append({
            "name": tech["name"],
            "how": tech.get("cli", ""),
            "impact": tech.get("impact", tech.get("impact_small_model", "")),
        })

    db.close()
    return BottleneckResult(
        classification=classification,
        confidence=confidence,
        evidence=evidence,
        techniques=techniques,
    )


# ---------------------------------------------------------------------------
# L1 profile-based classification
# ---------------------------------------------------------------------------

def classify_from_l1(
    l1_path: str,
    pipeline_type: str = "decoder_kv_cache",
) -> BottleneckResult:
    """Classify bottleneck from L1 CPU profile JSON."""
    with open(l1_path) as f:
        data = json.load(f)

    evidence = []
    scores = {"sync": 0.0, "bandwidth": 0.0, "compute": 0.0, "latency": 0.0}

    # Extract phase timings
    execute_ms = data.get("execute_ms", data.get("execute", {}).get("mean", 0))
    d2h_ms = data.get("d2h_ms", data.get("d2h", {}).get("mean", 0))
    h2d_ms = data.get("h2d_ms", data.get("h2d", {}).get("mean", 0))
    argmax_ms = data.get("argmax_ms", data.get("argmax", {}).get("mean", 0))
    total_ms = data.get("total_ms", data.get("step_ms", {}).get("mean", 0))

    if total_ms <= 0:
        total_ms = execute_ms + d2h_ms + h2d_ms + argmax_ms

    if total_ms > 0:
        d2h_pct = (d2h_ms + argmax_ms) / total_ms * 100
        execute_pct = execute_ms / total_ms * 100
        transfer_pct = (d2h_ms + h2d_ms) / total_ms * 100

        evidence.append(f"execute: {execute_pct:.0f}%, D2H+argmax: {d2h_pct:.0f}%, "
                        f"transfers: {transfer_pct:.0f}% of step")

        if d2h_pct > 15:
            scores["sync"] += 3
            evidence.append(f"D2H+argmax = {d2h_pct:.0f}% > 15% → sync-bound")
        if execute_pct > 80:
            scores["compute"] += 2
            evidence.append(f"Execute = {execute_pct:.0f}% > 80% → compute-bound")
        if transfer_pct > 10:
            scores["sync"] += 1
            evidence.append(f"Transfers = {transfer_pct:.0f}% > 10%")

    # Determine classification
    best = max(scores, key=scores.get)
    best_score = scores[best]

    if best_score == 0:
        classification = "mixed"
        confidence = "low"
    else:
        classification = best
        confidence = "high" if best_score >= 3 else "medium"

    mode = runtime_strategy_performance_mode(pipeline_type, default="decode")
    mode_priorities = TECHNIQUE_PRIORITY_BY_MODE.get(mode, TECHNIQUE_PRIORITY)
    techniques = []
    for tech_key in mode_priorities.get(classification, ["fp16"]):
        tech = TECHNIQUES[tech_key]
        techniques.append({
            "name": tech["name"],
            "how": tech.get("cli", ""),
            "impact": tech.get("impact", tech.get("impact_small_model", "")),
        })

    return BottleneckResult(
        classification=classification,
        confidence=confidence,
        evidence=evidence,
        techniques=techniques,
    )


# ---------------------------------------------------------------------------
# Suggest next technique from results history
# ---------------------------------------------------------------------------

def suggest_next_technique(
    results_path: str,
    classification: str,
) -> dict | None:
    """Given past results, suggest the next technique to try.

    Reads evolve_results.jsonl, checks which techniques were already tried,
    and returns the next untried technique for the given bottleneck class.

    Returns dict {name, how, reason} or None if all exhausted.
    """
    tried = set()
    try:
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tried.add(r.get("technique", "").lower())
    except FileNotFoundError:
        pass

    priority = TECHNIQUE_PRIORITY.get(classification, ["gpu_argmax", "fp16"])
    for tech_key in priority:
        tech = TECHNIQUES[tech_key]
        if tech["name"].lower() not in tried and tech_key not in tried:
            return {
                "name": tech["name"],
                "how": tech.get("cli", ""),
                "reason": f"Next priority for {classification}-bound workload",
            }

    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_report(result: BottleneckResult) -> None:
    print()
    print("=" * 60)
    print("  Bottleneck Classification")
    print("=" * 60)
    print(f"  Class:      {result.classification}-bound")
    print(f"  Confidence: {result.confidence}")
    print()
    print("  Evidence:")
    for e in result.evidence:
        print(f"    - {e}")
    print()
    print("  Recommended techniques (priority order):")
    for i, t in enumerate(result.techniques, 1):
        print(f"    {i}. {t['name']}")
        print(f"       How: {t['how']}")
        print(f"       Impact: {t['impact']}")
    print("=" * 60)


def to_json(result: BottleneckResult) -> dict:
    return {
        "classification": result.classification,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "techniques": result.techniques,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify performance bottleneck from profiling data.")
    parser.add_argument("--nsys-sqlite",
                        help="Path to nsys SQLite export")
    parser.add_argument("--l1-json",
                        help="Path to L1 CPU profile JSON")
    parser.add_argument("--pipeline-type", default="decoder_kv_cache",
                        help="Runtime strategy declared in tests/runtime_strategy_matrix.yaml")
    parser.add_argument("--engine-section", default="all",
                        help="Which engine to analyze: 'all' (default), 'primary', "
                             "'secondary', or a specific CUDA graph ID number")
    parser.add_argument("--results-jsonl",
                        help="Path to evolve_results.jsonl (for next technique suggestion)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON")

    args = parser.parse_args()

    if not args.nsys_sqlite and not args.l1_json:
        parser.error("At least one of --nsys-sqlite or --l1-json required")

    # Classify (nsys preferred over L1)
    if args.nsys_sqlite:
        result = classify_from_nsys(args.nsys_sqlite, args.pipeline_type,
                                    engine_section=args.engine_section)
    else:
        result = classify_from_l1(args.l1_json, args.pipeline_type)

    # Suggest next technique if results history provided
    suggestion = None
    if args.results_jsonl:
        suggestion = suggest_next_technique(args.results_jsonl, result.classification)

    if args.json:
        out = to_json(result)
        if suggestion:
            out["next_technique"] = suggestion
        print(json.dumps(out, indent=2))
    else:
        print_report(result)
        if suggestion:
            print(f"\n  Next technique to try: {suggestion['name']}")
            print(f"  How: {suggestion['how']}")
            print(f"  Reason: {suggestion['reason']}")
        elif args.results_jsonl:
            print("\n  All techniques for this bottleneck class exhausted.")


if __name__ == "__main__":
    main()

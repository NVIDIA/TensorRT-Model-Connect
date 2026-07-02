#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance database for tracking inference benchmarks over time.

Stores E2E harness results and perf_compare.py benchmarks in a local SQLite
database. Provides a CLI for querying history, baselines, and exporting data.

Usage:
    python3 tools/perfdb.py history <model_name> [--limit N] [--db PATH]
    python3 tools/perfdb.py baseline <model_name> [--db PATH]
    python3 tools/perfdb.py export [--format csv|json] [--db PATH]

Programmatic:
    from perfdb import PerfDB
    db = PerfDB("/path/to/perf.db")
    db.record_e2e(result, case, env, git_info)
    db.query_history("example-decoder")
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DB_PATH = "perf_results.db"

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS environments (
    env_id          TEXT PRIMARY KEY,
    gpu_name        TEXT NOT NULL,
    driver          TEXT NOT NULL DEFAULT '',
    trt_version     TEXT NOT NULL DEFAULT '',
    cuda_version    TEXT NOT NULL DEFAULT '',
    hostname        TEXT NOT NULL DEFAULT '',
    first_seen      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perf_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    env_id              TEXT NOT NULL REFERENCES environments(env_id),
    git_commit          TEXT NOT NULL DEFAULT '',
    git_branch          TEXT NOT NULL DEFAULT '',
    source              TEXT NOT NULL DEFAULT 'manual',
    model_name          TEXT NOT NULL,
    hf_id               TEXT NOT NULL DEFAULT '',
    runtime_strategy    TEXT NOT NULL DEFAULT '',
    task_strategy       TEXT NOT NULL DEFAULT '',
    max_cache_length    INTEGER DEFAULT NULL,
    builder_workspace_mb REAL DEFAULT NULL,
    precision           TEXT NOT NULL DEFAULT '',
    build_s             REAL DEFAULT NULL,
    trt_run_s           REAL DEFAULT NULL,
    ref_run_s           REAL DEFAULT NULL,
    prefill_ms_mean     REAL DEFAULT NULL,
    prefill_ms_std      REAL DEFAULT NULL,
    decode_ms_mean      REAL DEFAULT NULL,
    decode_ms_std       REAL DEFAULT NULL,
    per_token_ms        REAL DEFAULT NULL,
    throughput_tps      REAL DEFAULT NULL,
    speedup             REAL DEFAULT NULL,
    token_match         INTEGER DEFAULT NULL,
    e2e_status          TEXT NOT NULL DEFAULT '',
    extra_json          TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_perf_runs_model_ts
    ON perf_runs (model_name, timestamp);

CREATE INDEX IF NOT EXISTS idx_perf_runs_env_model
    ON perf_runs (env_id, model_name);

"""


def _compute_env_id(fingerprint: Dict[str, str]) -> str:
    """Compute a deterministic env_id from environment fingerprint fields."""
    key_parts = [
        fingerprint.get("gpu_name", ""),
        fingerprint.get("driver", ""),
        fingerprint.get("trt_version", ""),
        fingerprint.get("cuda_version", ""),
        fingerprint.get("hostname", ""),
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def detect_env_fingerprint() -> Dict[str, str]:
    """Detect the current hardware/software environment.

    Returns a dict with keys: gpu_name, driver, trt_version, cuda_version,
    hostname — suitable for passing to PerfDB.ensure_env().
    """
    import socket

    env: Dict[str, str] = {
        "gpu_name": "",
        "driver": "",
        "trt_version": "",
        "cuda_version": "",
        "hostname": "",
    }

    try:
        env["hostname"] = socket.gethostname()
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",", 1)
            env["gpu_name"] = parts[0].strip()
            if len(parts) > 1:
                env["driver"] = parts[1].strip()
    except Exception:
        pass

    try:
        import tensorrt as trt
        env["trt_version"] = trt.__version__
    except Exception:
        pass

    try:
        r = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.6, V12.6.77"
                    idx = line.lower().find("release")
                    env["cuda_version"] = line[idx:].split(",")[0].replace(
                        "release ", "").strip()
                    break
    except Exception:
        pass

    return env


def _get_git_info() -> Dict[str, str]:
    """Retrieve current git commit and branch."""
    info: Dict[str, str] = {"commit": "", "branch": ""}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except Exception:
        pass
    return info


class PerfDB:
    """SQLite-backed performance tracking database.

    Thread-safe for single-writer usage. The database file is created
    automatically if it does not exist.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        is_memory = db_path == ":memory:"
        if not is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Environment management
    # ------------------------------------------------------------------

    def ensure_env(self, fingerprint: Dict[str, str]) -> str:
        """Upsert an environment record and return its env_id.

        Args:
            fingerprint: Dict with keys gpu_name, driver, trt_version,
                cuda_version, hostname.

        Returns:
            The hex env_id (first 16 chars of sha256).
        """
        env_id = _compute_env_id(fingerprint)
        now = datetime.now(timezone.utc).isoformat()

        # INSERT OR IGNORE so we never overwrite first_seen
        self._conn.execute(
            "INSERT OR IGNORE INTO environments "
            "(env_id, gpu_name, driver, trt_version, cuda_version, hostname, first_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                env_id,
                fingerprint.get("gpu_name", ""),
                fingerprint.get("driver", ""),
                fingerprint.get("trt_version", ""),
                fingerprint.get("cuda_version", ""),
                fingerprint.get("hostname", ""),
                now,
            ),
        )
        self._conn.commit()
        return env_id

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_e2e(
        self,
        result: Any,
        case: Any,
        env_fingerprint: Dict[str, str],
        git_info: Optional[Dict[str, str]] = None,
    ) -> int:
        """Record an E2E harness result.

        Args:
            result: An E2EResult dataclass instance (from contracts.py).
            case: An E2ECase dataclass instance.
            env_fingerprint: Environment fingerprint dict.
            git_info: Optional dict with 'commit' and 'branch' keys.
                If None, auto-detected from git.

        Returns:
            The run_id of the inserted row.
        """
        if git_info is None:
            git_info = _get_git_info()

        env_id = self.ensure_env(env_fingerprint)
        now = datetime.now(timezone.utc).isoformat()

        timing = getattr(result, "timing", {}) or {}
        build_s = timing.get("build_s") or timing.get("bundle_build_s")
        # E2E harness uses trt_full_generation_s; perf_compare uses trt_run_s
        trt_run_s = (timing.get("trt_full_generation_s")
                     or timing.get("trt_generate_s")
                     or timing.get("trt_run_s"))
        ref_run_s = (timing.get("ref_full_generation_s")
                     or timing.get("ref_generate_s")
                     or timing.get("ref_run_s"))

        # Extract perf metrics from stage outputs if available
        extra: Dict[str, Any] = {}
        stages = getattr(result, "stages", {}) or {}
        stage_summary = {}
        for sname, cr in stages.items():
            metrics_dict = {}
            cr_metrics = getattr(cr, "metrics", {}) or {}
            for mname, metric_result in cr_metrics.items():
                if hasattr(metric_result, "value"):
                    metrics_dict[mname] = metric_result.value
                else:
                    metrics_dict[mname] = metric_result
            stage_summary[sname] = {
                "status": getattr(cr, "status", ""),
                "metrics": metrics_dict,
            }
        if stage_summary:
            extra["stages"] = stage_summary

        # Determine token_match from stage metrics
        token_match = None
        for cr in stages.values():
            cr_metrics = getattr(cr, "metrics", {}) or {}
            if "token_agreement_rate" in cr_metrics:
                metric_result = cr_metrics["token_agreement_rate"]
                val = (
                    metric_result.value
                    if hasattr(metric_result, "value")
                    else metric_result
                )
                token_match = 1 if val >= 1.0 else 0
                break

        cursor = self._conn.execute(
            "INSERT INTO perf_runs "
            "(timestamp, env_id, git_commit, git_branch, source, "
            " model_name, hf_id, runtime_strategy, task_strategy, "
            " max_cache_length, precision, "
            " build_s, trt_run_s, ref_run_s, "
            " token_match, e2e_status, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                env_id,
                git_info.get("commit", ""),
                git_info.get("branch", ""),
                "e2e_harness",
                getattr(case, "name", ""),
                getattr(case, "hf_id", ""),
                getattr(case, "runtime_strategy", ""),
                getattr(case, "task_strategy", ""),
                case.inputs.get("max_cache_length") if hasattr(case, "inputs") else None,
                "",  # precision not tracked in E2E
                build_s,
                trt_run_s,
                ref_run_s,
                token_match,
                getattr(result, "status", ""),
                json.dumps(extra),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def record_perf_compare(
        self,
        json_data: Dict[str, Any],
        git_info: Optional[Dict[str, str]] = None,
    ) -> int:
        """Record a perf_compare.py benchmark result.

        Args:
            json_data: The structured JSON output from build_json_output().
            git_info: Optional dict with 'commit' and 'branch' keys.

        Returns:
            The run_id of the inserted row.
        """
        if git_info is None:
            git_info = _get_git_info()

        metadata = json_data.get("metadata", {})
        trt = json_data.get("trt", {})
        hf = json_data.get("hf", {})
        speedup_data = json_data.get("speedup", {})

        env_fp = {
            "gpu_name": metadata.get("gpu", ""),
            "driver": "",
            "trt_version": metadata.get("trt_version", ""),
            "cuda_version": "",
            "hostname": "",
        }
        env_id = self.ensure_env(env_fp)
        now = datetime.now(timezone.utc).isoformat()

        prefill_ms = trt.get("prefill_ms", {})
        decode_ms = trt.get("decode_ms", {})
        per_token = trt.get("per_token_ms", {})
        throughput = trt.get("throughput_tps", {})

        token_match = json_data.get("token_match")
        if token_match is not None:
            token_match = 1 if token_match else 0

        extra: Dict[str, Any] = {
            "hf": hf,
            "prompt": metadata.get("prompt", ""),
            "num_input_tokens": metadata.get("num_input_tokens", 0),
            "max_new_tokens": metadata.get("max_new_tokens", 0),
            "warmup": metadata.get("warmup", 0),
            "iterations": metadata.get("iterations", 0),
        }

        cursor = self._conn.execute(
            "INSERT INTO perf_runs "
            "(timestamp, env_id, git_commit, git_branch, source, "
            " model_name, hf_id, precision, "
            " prefill_ms_mean, prefill_ms_std, "
            " decode_ms_mean, decode_ms_std, "
            " per_token_ms, throughput_tps, "
            " speedup, token_match, e2e_status, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                env_id,
                git_info.get("commit", ""),
                git_info.get("branch", ""),
                "perf_compare",
                metadata.get("model", ""),
                metadata.get("model", ""),
                metadata.get("hf_dtype", ""),
                prefill_ms.get("mean"),
                prefill_ms.get("std"),
                decode_ms.get("mean"),
                decode_ms.get("std"),
                per_token.get("mean") if isinstance(per_token, dict) else None,
                throughput.get("mean") if isinstance(throughput, dict) else None,
                speedup_data.get("decode"),
                token_match,
                "pass",
                json.dumps(extra),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_history(
        self,
        model_name: str,
        limit: int = 20,
        env_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent perf runs for a model, newest first.

        Args:
            model_name: Model name to query.
            limit: Maximum number of rows to return.
            env_id: Optional environment filter.

        Returns:
            List of row dicts.
        """
        if env_id:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? AND env_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (model_name, env_id, limit),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (model_name, limit),
            )
        return [dict(row) for row in cursor.fetchall()]

    def query_baseline(
        self,
        model_name: str,
        env_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the best known performance for a model.

        'Best' is defined as the run with the highest throughput_tps.
        For E2E harness runs (which may not have throughput_tps), falls
        back to the fastest trt_run_s.

        Args:
            model_name: Model name to query.
            env_id: Optional environment filter.

        Returns:
            Row dict or None if no runs exist.
        """
        # Try throughput first (perf_compare runs)
        if env_id:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? AND env_id = ? "
                "  AND throughput_tps IS NOT NULL "
                "ORDER BY throughput_tps DESC LIMIT 1",
                (model_name, env_id),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? "
                "  AND throughput_tps IS NOT NULL "
                "ORDER BY throughput_tps DESC LIMIT 1",
                (model_name,),
            )
        row = cursor.fetchone()
        if row:
            return dict(row)

        # Fallback: fastest TRT run time (E2E harness runs)
        if env_id:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? AND env_id = ? "
                "  AND trt_run_s IS NOT NULL "
                "ORDER BY trt_run_s ASC LIMIT 1",
                (model_name, env_id),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM perf_runs "
                "WHERE model_name = ? "
                "  AND trt_run_s IS NOT NULL "
                "ORDER BY trt_run_s ASC LIMIT 1",
                (model_name,),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    def compare_to_baseline(
        self,
        model_name: str,
        env_id: Optional[str],
        current: Dict[str, Any],
        threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """Compare a current run against the baseline.

        Args:
            model_name: Model name.
            env_id: Environment ID for baseline lookup.
            current: Dict with perf metrics (throughput_tps, decode_ms_mean,
                per_token_ms, trt_run_s).
            threshold: Fractional regression threshold. 0.10 means a 10%
                slowdown triggers a regression warning.

        Returns:
            Dict with keys: has_baseline, regression, details.
        """
        baseline = self.query_baseline(model_name, env_id)
        if baseline is None:
            return {
                "has_baseline": False,
                "regression": False,
                "details": "No baseline found",
            }

        regressions: List[str] = []

        # Compare throughput (higher is better)
        base_tps = baseline.get("throughput_tps")
        curr_tps = current.get("throughput_tps")
        if base_tps and curr_tps and base_tps > 0:
            ratio = curr_tps / base_tps
            if ratio < (1.0 - threshold):
                regressions.append(
                    f"throughput_tps: {curr_tps:.1f} vs baseline {base_tps:.1f} "
                    f"({ratio:.2%} of baseline, threshold {1.0 - threshold:.0%})"
                )

        # Compare decode latency (lower is better)
        base_decode = baseline.get("decode_ms_mean")
        curr_decode = current.get("decode_ms_mean")
        if base_decode and curr_decode and base_decode > 0:
            ratio = curr_decode / base_decode
            if ratio > (1.0 + threshold):
                regressions.append(
                    f"decode_ms_mean: {curr_decode:.1f}ms vs baseline {base_decode:.1f}ms "
                    f"({ratio:.2%} of baseline, threshold {1.0 + threshold:.0%})"
                )

        # Compare per-token latency (lower is better)
        base_pt = baseline.get("per_token_ms")
        curr_pt = current.get("per_token_ms")
        if base_pt and curr_pt and base_pt > 0:
            ratio = curr_pt / base_pt
            if ratio > (1.0 + threshold):
                regressions.append(
                    f"per_token_ms: {curr_pt:.2f}ms vs baseline {base_pt:.2f}ms "
                    f"({ratio:.2%} of baseline, threshold {1.0 + threshold:.0%})"
                )

        return {
            "has_baseline": True,
            "regression": len(regressions) > 0,
            "details": "; ".join(regressions) if regressions else "Within threshold",
            "baseline_run_id": baseline.get("run_id"),
            "baseline_timestamp": baseline.get("timestamp"),
        }

    # ------------------------------------------------------------------
    # Rolling baseline
    # ------------------------------------------------------------------

    def query_rolling_baseline(
        self,
        model_name: str,
        env_id: Optional[str] = None,
        last_n: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Compute a rolling baseline from the last N successful runs.

        Returns mean and std for throughput_tps, decode_ms_mean, and
        per_token_ms across the most recent *last_n* runs that have
        throughput_tps populated.  Falls back to :meth:`query_baseline`
        if fewer than 2 runs exist.

        Args:
            model_name: Model name to query.
            env_id: Optional environment filter.
            last_n: Number of recent runs to average over.

        Returns:
            Dict with keys: throughput_tps_mean, throughput_tps_std,
            decode_ms_mean_mean, decode_ms_mean_std, per_token_ms_mean,
            per_token_ms_std, run_count, oldest_run_id, newest_run_id,
            newest_timestamp.  Returns None if no qualifying runs exist.
        """
        params: list = [model_name]
        env_clause = ""
        if env_id:
            env_clause = " AND env_id = ? "
            params.append(env_id)
        params.append(last_n)

        cursor = self._conn.execute(
            "SELECT run_id, timestamp, throughput_tps, decode_ms_mean, "
            "       per_token_ms "
            "FROM perf_runs "
            "WHERE model_name = ? " + env_clause +
            "  AND throughput_tps IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        tps_vals = [r["throughput_tps"] for r in rows]
        decode_vals = [r["decode_ms_mean"] for r in rows
                       if r["decode_ms_mean"] is not None]
        pt_vals = [r["per_token_ms"] for r in rows
                   if r["per_token_ms"] is not None]

        import statistics as _stats_mod

        def _mean_std(vals: List[float]):
            if not vals:
                return None, None
            m = _stats_mod.mean(vals)
            s = _stats_mod.stdev(vals) if len(vals) > 1 else 0.0
            return m, s

        tps_mean, tps_std = _mean_std(tps_vals)
        dec_mean, dec_std = _mean_std(decode_vals)
        pt_mean, pt_std = _mean_std(pt_vals)

        return {
            "throughput_tps_mean": tps_mean,
            "throughput_tps_std": tps_std,
            "decode_ms_mean_mean": dec_mean,
            "decode_ms_mean_std": dec_std,
            "per_token_ms_mean": pt_mean,
            "per_token_ms_std": pt_std,
            "run_count": len(rows),
            "oldest_run_id": rows[-1]["run_id"],
            "newest_run_id": rows[0]["run_id"],
            "newest_timestamp": rows[0]["timestamp"],
        }

    def update_baseline(
        self,
        model_name: str,
        env_id: str,
        run_id: int,
    ) -> None:
        """Mark a specific run as the explicit baseline.

        Uses a lightweight ``baselines`` table (auto-created) that stores
        one row per (model_name, env_id) pair.  The gate tool reads this
        first; if no explicit baseline is set, it falls back to
        :meth:`query_baseline` (best-throughput heuristic).

        Args:
            model_name: Model name.
            env_id: Environment ID.
            run_id: The run_id from perf_runs to mark as baseline.
        """
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS baselines ("
            "  model_name TEXT NOT NULL, "
            "  env_id     TEXT NOT NULL, "
            "  run_id     INTEGER NOT NULL REFERENCES perf_runs(run_id), "
            "  updated_at TEXT NOT NULL, "
            "  PRIMARY KEY (model_name, env_id)"
            ")"
        )
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO baselines "
            "(model_name, env_id, run_id, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (model_name, env_id, run_id, now),
        )
        self._conn.commit()

    def query_explicit_baseline(
        self,
        model_name: str,
        env_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the explicitly-set baseline run, if any.

        Args:
            model_name: Model name.
            env_id: Optional environment filter.

        Returns:
            Row dict from perf_runs, or None if no explicit baseline is set.
        """
        try:
            self._conn.execute(
                "SELECT 1 FROM baselines LIMIT 0")
        except Exception:
            return None

        if env_id:
            cursor = self._conn.execute(
                "SELECT p.* FROM baselines b "
                "JOIN perf_runs p ON b.run_id = p.run_id "
                "WHERE b.model_name = ? AND b.env_id = ?",
                (model_name, env_id),
            )
        else:
            cursor = self._conn.execute(
                "SELECT p.* FROM baselines b "
                "JOIN perf_runs p ON b.run_id = p.run_id "
                "WHERE b.model_name = ? "
                "ORDER BY b.updated_at DESC LIMIT 1",
                (model_name,),
            )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all(self, fmt: str = "json") -> str:
        """Export all perf_runs as JSON or CSV string.

        Args:
            fmt: 'json' or 'csv'.

        Returns:
            Formatted string.
        """
        cursor = self._conn.execute(
            "SELECT * FROM perf_runs ORDER BY timestamp DESC"
        )
        rows = [dict(r) for r in cursor.fetchall()]

        if fmt == "csv":
            if not rows:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            return output.getvalue()
        else:
            return json.dumps(rows, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_history(args: argparse.Namespace) -> None:
    db = PerfDB(args.db)
    rows = db.query_history(args.model_name, limit=args.limit)
    db.close()

    if not rows:
        print(f"No runs found for model: {args.model_name}")
        return

    print(f"Recent runs for {args.model_name} (newest first):")
    print(f"{'Run':>5}  {'Timestamp':>25}  {'Source':>12}  {'Status':>6}  "
          f"{'Decode(ms)':>12}  {'Tput(t/s)':>10}  {'Speedup':>8}  {'Branch'}")
    print("-" * 110)
    for row in rows:
        decode = f"{row['decode_ms_mean']:.1f}" if row["decode_ms_mean"] else "-"
        tput = f"{row['throughput_tps']:.1f}" if row["throughput_tps"] else "-"
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] else "-"
        print(f"{row['run_id']:>5}  {row['timestamp']:>25}  {row['source']:>12}  "
              f"{row['e2e_status']:>6}  {decode:>12}  {tput:>10}  {speedup:>8}  "
              f"{row['git_branch']}")


def _cli_baseline(args: argparse.Namespace) -> None:
    db = PerfDB(args.db)
    baseline = db.query_baseline(args.model_name)
    db.close()

    if baseline is None:
        print(f"No baseline found for model: {args.model_name}")
        return

    print(f"Best known performance for {args.model_name}:")
    for key in ("run_id", "timestamp", "source", "git_commit", "git_branch",
                "decode_ms_mean", "per_token_ms", "throughput_tps",
                "speedup", "token_match", "e2e_status"):
        val = baseline.get(key)
        if val is not None:
            print(f"  {key}: {val}")


def _cli_export(args: argparse.Namespace) -> None:
    db = PerfDB(args.db)
    output = db.export_all(fmt=args.format)
    db.close()
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Exported to {args.output}", file=sys.stderr)
    else:
        print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Performance database CLI")
    parser.add_argument(
        "--db", default=_DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {_DEFAULT_DB_PATH})")

    subparsers = parser.add_subparsers(dest="command")

    # history
    p_history = subparsers.add_parser(
        "history", help="Show recent runs for a model")
    p_history.add_argument("model_name", help="Model name to query")
    p_history.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of rows (default: 20)")

    # baseline
    p_baseline = subparsers.add_parser(
        "baseline", help="Show best known performance for a model")
    p_baseline.add_argument("model_name", help="Model name to query")

    # export
    p_export = subparsers.add_parser(
        "export", help="Export all runs")
    p_export.add_argument(
        "--format", choices=["json", "csv"], default="json",
        help="Export format (default: json)")
    p_export.add_argument(
        "--output", "-o", metavar="PATH",
        help="Output file path (default: stdout)")

    args = parser.parse_args()

    if args.command == "history":
        _cli_history(args)
    elif args.command == "baseline":
        _cli_baseline(args)
    elif args.command == "export":
        _cli_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

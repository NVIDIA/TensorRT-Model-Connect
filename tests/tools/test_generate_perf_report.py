"""Unit tests for scripts/generate_perf_report.py — HTML performance dashboard.

Tests cover:
  - regression_status() logic (regression / ok / no_baseline / no_data)
  - sparkline_svg() output validity
  - fmt_metric() / fmt_commit() / fmt_ts() formatting helpers
  - render_summary_dashboard() HTML structure
  - render_model_section() collapsible detail section
  - render_report() full integration (HTML validity, key content)
  - load_report_data() from an in-memory SQLite database

All tests are pure-Python; no GPU, TRT, or perf.db on disk required.

Trace: ARCH-REPORT-001, UD-REPORT-PERF
Intent: Validate performance report generator regression detection, formatting, and HTML rendering
Preconditions: scripts/ is on sys.path; in-memory SQLite database with synthetic perf data is available
Postconditions: Regression status is correctly computed and rendered HTML contains expected structure and content
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional



# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------


def _import_report():
    """Import generate_perf_report from scripts/."""
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("generate_perf_report")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    model_name: str = "decoder-small",
    throughput_tps: Optional[float] = 150.0,
    decode_ms_mean: Optional[float] = 6.5,
    trt_run_s: Optional[float] = None,
    build_s: Optional[float] = 5.0,
    speedup: Optional[float] = 2.5,
    git_commit: str = "abc12345def67890",
    git_branch: str = "main",
    timestamp: str = "2026-03-17T10:00:00Z",
    source: str = "e2e_harness",
    e2e_status: str = "pass",
    hf_id: str = "example-org/decoder-small",
) -> Dict[str, Any]:
    """Return a synthetic perf_run row dict."""
    return {
        "run_id": 1,
        "model_name": model_name,
        "throughput_tps": throughput_tps,
        "decode_ms_mean": decode_ms_mean,
        "trt_run_s": trt_run_s,
        "build_s": build_s,
        "speedup": speedup,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "timestamp": timestamp,
        "source": source,
        "e2e_status": e2e_status,
        "hf_id": hf_id,
        "per_token_ms": None,
        "ref_run_s": None,
        "prefill_ms_mean": None,
        "prefill_ms_std": None,
        "decode_ms_std": None,
    }


# ---------------------------------------------------------------------------
# regression_status tests
# ---------------------------------------------------------------------------


class TestRegressionStatus:
    """UT-PERF-REPORT-001..006 — regression_status()."""

    def test_no_baseline_returns_no_baseline(self):
        mod = _import_report()
        status, detail = mod.regression_status(_make_run(), None)
        assert status == "no_baseline"
        assert "No baseline" in detail

    def test_tps_regression_detected(self):
        """UT-PERF-REPORT-001: >10% TPS drop triggers regression."""
        mod = _import_report()
        latest = _make_run(throughput_tps=80.0)
        baseline = _make_run(throughput_tps=100.0)
        status, detail = mod.regression_status(latest, baseline, threshold=0.10)
        assert status == "regression"
        assert "80" in detail
        assert "100" in detail

    def test_tps_ok_within_threshold(self):
        """UT-PERF-REPORT-002: 5% TPS drop below 10% threshold is OK."""
        mod = _import_report()
        latest = _make_run(throughput_tps=96.0)
        baseline = _make_run(throughput_tps=100.0)
        status, _ = mod.regression_status(latest, baseline, threshold=0.10)
        assert status == "ok"

    def test_tps_improvement_is_ok(self):
        """UT-PERF-REPORT-003: TPS improvement is OK, not regression."""
        mod = _import_report()
        latest = _make_run(throughput_tps=120.0)
        baseline = _make_run(throughput_tps=100.0)
        status, _ = mod.regression_status(latest, baseline)
        assert status == "ok"

    def test_trt_run_fallback_regression(self):
        """UT-PERF-REPORT-004: Falls back to trt_run_s when no TPS available."""
        mod = _import_report()
        latest = _make_run(throughput_tps=None, trt_run_s=12.0)
        baseline = _make_run(throughput_tps=None, trt_run_s=10.0)
        status, detail = mod.regression_status(latest, baseline, threshold=0.10)
        assert status == "regression"
        assert "12" in detail

    def test_no_data_when_no_metrics(self):
        """UT-PERF-REPORT-005: Returns no_data when neither TPS nor trt_run_s available."""
        mod = _import_report()
        latest = _make_run(throughput_tps=None, trt_run_s=None)
        baseline = _make_run(throughput_tps=None, trt_run_s=None)
        status, _ = mod.regression_status(latest, baseline)
        assert status == "no_data"

    def test_custom_threshold(self):
        """UT-PERF-REPORT-006: Custom threshold (5%) triggers on smaller drops."""
        mod = _import_report()
        latest = _make_run(throughput_tps=93.0)
        baseline = _make_run(throughput_tps=100.0)
        status_strict, _ = mod.regression_status(latest, baseline, threshold=0.05)
        status_loose, _ = mod.regression_status(latest, baseline, threshold=0.10)
        assert status_strict == "regression"
        assert status_loose == "ok"


# ---------------------------------------------------------------------------
# sparkline_svg tests
# ---------------------------------------------------------------------------


class TestSparklineSvg:
    """UT-PERF-REPORT-007..010 — sparkline_svg()."""

    def test_empty_returns_empty_string(self):
        """UT-PERF-REPORT-007: Empty or single-value list returns ''."""
        mod = _import_report()
        assert mod.sparkline_svg([]) == ""
        assert mod.sparkline_svg([42.0]) == ""

    def test_two_values_produces_svg(self):
        """UT-PERF-REPORT-008: Two values produce valid SVG."""
        mod = _import_report()
        svg = mod.sparkline_svg([10.0, 20.0])
        assert svg.startswith("<svg")
        assert "<polyline" in svg
        assert "points=" in svg

    def test_all_same_values(self):
        """UT-PERF-REPORT-009: All-identical values do not cause division by zero."""
        mod = _import_report()
        svg = mod.sparkline_svg([5.0, 5.0, 5.0, 5.0])
        assert "<svg" in svg

    def test_none_values_skipped(self):
        """UT-PERF-REPORT-010: None values are filtered out."""
        mod = _import_report()
        svg = mod.sparkline_svg([10.0, None, 20.0, None, 30.0])
        assert "<svg" in svg
        assert "polyline" in svg

    def test_custom_dimensions(self):
        """Width and height appear in the SVG tag."""
        mod = _import_report()
        svg = mod.sparkline_svg([1.0, 2.0, 3.0], width=120, height=36)
        assert 'width="120"' in svg
        assert 'height="36"' in svg


# ---------------------------------------------------------------------------
# Formatting helper tests
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    """UT-PERF-REPORT-011..013 — fmt_metric / fmt_commit / fmt_ts."""

    def test_fmt_metric_none(self):
        """UT-PERF-REPORT-011: None returns em-dash."""
        mod = _import_report()
        assert mod.fmt_metric(None) == "&mdash;"

    def test_fmt_metric_float(self):
        mod = _import_report()
        assert mod.fmt_metric(12.3456) == "12.3"
        assert mod.fmt_metric(12.3456, "ms") == "12.3ms"
        assert mod.fmt_metric(2.567, "x", precision=2) == "2.57x"

    def test_fmt_commit_shortens(self):
        """UT-PERF-REPORT-012: Commit hash is shortened to 8 chars."""
        mod = _import_report()
        assert mod.fmt_commit("abc12345def67890") == "abc12345"
        assert mod.fmt_commit("") == "&mdash;"
        assert mod.fmt_commit(None) == "&mdash;"

    def test_fmt_ts_formats(self):
        """UT-PERF-REPORT-013: ISO timestamp is trimmed."""
        mod = _import_report()
        result = mod.fmt_ts("2026-03-17T14:23:45.123456+00:00")
        assert "2026-03-17" in result
        assert "14:23" in result

    def test_fmt_ts_empty(self):
        mod = _import_report()
        assert mod.fmt_ts("") == "&mdash;"


# ---------------------------------------------------------------------------
# Dashboard rendering tests
# ---------------------------------------------------------------------------


class TestRenderSummaryDashboard:
    """UT-PERF-REPORT-014..015 — render_summary_dashboard()."""

    def _make_row(self, name: str, status: str) -> Dict[str, Any]:
        return {
            "model_name": name,
            "status": status,
            "detail": f"detail for {name}",
            "latest": _make_run(model_name=name),
            "tps_history": [100.0, 105.0, 110.0],
        }

    def test_counters_present(self):
        """UT-PERF-REPORT-014: Counter spans appear for each status."""
        mod = _import_report()
        rows = [
            self._make_row("decoder-small", "ok"),
            self._make_row("decoder-regression", "regression"),
            self._make_row("decoder-no-baseline", "no_baseline"),
        ]
        html_str = mod.render_summary_dashboard(rows)
        assert "1 Regression" in html_str
        assert "1 OK" in html_str
        assert "3 Total" in html_str

    def test_model_names_in_table(self):
        """UT-PERF-REPORT-015: Model names appear as links in table."""
        mod = _import_report()
        rows = [self._make_row("decoder-small", "ok"), self._make_row("decoder-alt", "ok")]
        html_str = mod.render_summary_dashboard(rows)
        assert "decoder-small" in html_str
        assert "decoder-alt" in html_str
        assert 'href="#model-decoder-small"' in html_str

    def test_sparkline_in_row(self):
        """Trend sparkline SVG is embedded in each row with enough history."""
        mod = _import_report()
        rows = [self._make_row("decoder-small", "ok")]
        html_str = mod.render_summary_dashboard(rows)
        assert "<svg" in html_str


# ---------------------------------------------------------------------------
# Per-model section rendering tests
# ---------------------------------------------------------------------------


class TestRenderModelSection:
    """UT-PERF-REPORT-016..017 — render_model_section()."""

    def test_details_tag_with_id(self):
        """UT-PERF-REPORT-016: Section has <details id='model-NAME'>."""
        mod = _import_report()
        html_str = mod.render_model_section(
            model_name="decoder-small",
            latest=_make_run(),
            history=[_make_run(), _make_run(timestamp="2026-03-16T10:00:00Z")],
            status="ok",
            detail="TPS 150 vs 140 (107%)",
        )
        assert 'id="model-decoder-small"' in html_str
        assert "decoder-small" in html_str

    def test_history_table_rendered(self):
        """UT-PERF-REPORT-017: History table is present when runs exist."""
        mod = _import_report()
        history = [_make_run(timestamp=f"2026-03-{17 - i:02d}T10:00:00Z") for i in range(3)]
        html_str = mod.render_model_section(
            model_name="decoder-small",
            latest=history[0],
            history=history,
            status="ok",
            detail="ok",
        )
        assert "history-table" in html_str
        assert "<th>Commit</th>" in html_str

    def test_no_history_shows_message(self):
        """Empty history renders a fallback message."""
        mod = _import_report()
        html_str = mod.render_model_section(
            model_name="decoder-small",
            latest=None,
            history=[],
            status="no_data",
            detail="",
        )
        assert "No history available" in html_str


# ---------------------------------------------------------------------------
# Full report integration test
# ---------------------------------------------------------------------------


class TestRenderReport:
    """UT-PERF-REPORT-018 — render_report() integration."""

    def test_full_report_is_valid_html(self):
        """UT-PERF-REPORT-018: Full report starts with <!DOCTYPE html>."""
        mod = _import_report()
        rows = [
            {
                "model_name": "decoder-small",
                "latest": _make_run("decoder-small"),
                "history": [_make_run("decoder-small")],
                "status": "ok",
                "detail": "TPS 150 vs 140",
                "tps_history": [140.0, 145.0, 150.0],
            },
            {
                "model_name": "decoder-alt",
                "latest": _make_run("decoder-alt", throughput_tps=80.0),
                "history": [],
                "status": "regression",
                "detail": "TPS 80 vs 100 (80%)",
                "tps_history": [],
            },
        ]
        envs = [
            {
                "gpu_name": "NVIDIA B200",
                "driver": "560.28.03",
                "trt_version": "10.15",
                "cuda_version": "12.6",
                "hostname": "gb300-host",
            }
        ]
        html_str = mod.render_report(
            rows, envs, title="Test Report", generated_at="2026-03-17 10:00 UTC"
        )
        assert html_str.startswith("<!DOCTYPE html>")
        assert "<html" in html_str
        assert "Test Report" in html_str
        assert "NVIDIA B200" in html_str
        assert "decoder-small" in html_str
        assert "decoder-alt" in html_str
        assert "REGRESSION" in html_str
        assert "<script>" in html_str
        # Self-contained: no external <link> or <script src>
        assert "<link" not in html_str
        assert 'src="http' not in html_str


# ---------------------------------------------------------------------------
# In-memory DB integration test
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE environments (
    env_id TEXT PRIMARY KEY,
    gpu_name TEXT NOT NULL,
    driver TEXT NOT NULL DEFAULT '',
    trt_version TEXT NOT NULL DEFAULT '',
    cuda_version TEXT NOT NULL DEFAULT '',
    hostname TEXT NOT NULL DEFAULT '',
    first_seen TEXT NOT NULL
);
CREATE TABLE perf_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    env_id TEXT NOT NULL,
    git_commit TEXT NOT NULL DEFAULT '',
    git_branch TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'manual',
    model_name TEXT NOT NULL,
    hf_id TEXT NOT NULL DEFAULT '',
    runtime_strategy TEXT NOT NULL DEFAULT '',
    task_strategy TEXT NOT NULL DEFAULT '',
    max_cache_length INTEGER DEFAULT NULL,
    builder_workspace_mb REAL DEFAULT NULL,
    precision TEXT NOT NULL DEFAULT '',
    build_s REAL DEFAULT NULL,
    trt_run_s REAL DEFAULT NULL,
    ref_run_s REAL DEFAULT NULL,
    prefill_ms_mean REAL DEFAULT NULL,
    prefill_ms_std REAL DEFAULT NULL,
    decode_ms_mean REAL DEFAULT NULL,
    decode_ms_std REAL DEFAULT NULL,
    per_token_ms REAL DEFAULT NULL,
    throughput_tps REAL DEFAULT NULL,
    speedup REAL DEFAULT NULL,
    token_match INTEGER DEFAULT NULL,
    e2e_status TEXT NOT NULL DEFAULT '',
    extra_json TEXT NOT NULL DEFAULT '{}'
);
"""


def _make_inmemory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite DB with schema and test data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)

    conn.execute(
        "INSERT INTO environments VALUES (?,?,?,?,?,?,?)",
        ("env1", "NVIDIA B200", "560.28", "10.15", "12.6", "host1",
         "2026-03-01T00:00:00Z"),
    )

    # decoder-small: 3 runs (tps improving)
    for i, tps in enumerate([120.0, 135.0, 150.0]):
        conn.execute(
            "INSERT INTO perf_runs "
            "(timestamp,env_id,git_commit,git_branch,source,model_name,hf_id,"
            " build_s,decode_ms_mean,throughput_tps,speedup,e2e_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"2026-03-{15 + i:02d}T10:00:00Z",
                "env1",
                f"commit{i:04d}",
                "main",
                "e2e_harness",
                "decoder-small",
                "example-org/decoder-small",
                5.0 + i,
                7.0 - i * 0.5,
                tps,
                2.0 + i * 0.1,
                "pass",
            ),
        )

    # decoder-alt: 1 run, no tps (only trt_run_s)
    conn.execute(
        "INSERT INTO perf_runs "
        "(timestamp,env_id,git_commit,git_branch,source,model_name,hf_id,"
        " build_s,trt_run_s,e2e_status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "2026-03-17T10:00:00Z", "env1", "commita", "main",
            "e2e_harness", "decoder-alt", "example-org/decoder-alt",
            4.0, 8.5, "pass",
        ),
    )

    conn.commit()
    return conn


class TestLoadReportData:
    """UT-PERF-REPORT-019..020 — load_report_data() with in-memory DB."""

    def _write_db_to_tempfile(self, tmp_path: Path) -> str:
        """Write the in-memory DB to a temp file and return its path."""
        src_conn = _make_inmemory_db()
        db_path = str(tmp_path / "test.db")
        dst_conn = sqlite3.connect(db_path)
        src_conn.backup(dst_conn)
        src_conn.close()
        dst_conn.close()
        return db_path

    def test_model_names_loaded(self, tmp_path: Path):
        """UT-PERF-REPORT-019: Both models appear in model_rows."""
        mod = _import_report()
        db_path = self._write_db_to_tempfile(tmp_path)
        rows, envs = mod.load_report_data(db_path)
        names = [r["model_name"] for r in rows]
        assert "decoder-small" in names
        assert "decoder-alt" in names

    def test_env_loaded(self, tmp_path: Path):
        """Environment info is returned."""
        mod = _import_report()
        db_path = self._write_db_to_tempfile(tmp_path)
        _, envs = mod.load_report_data(db_path)
        assert len(envs) >= 1
        assert envs[0]["gpu_name"] == "NVIDIA B200"

    def test_model_history_newest_first(self, tmp_path: Path):
        """UT-PERF-REPORT-020: History is ordered newest-first."""
        mod = _import_report()
        db_path = self._write_db_to_tempfile(tmp_path)
        rows, _ = mod.load_report_data(db_path, history_limit=10)
        decoder_small = next(r for r in rows if r["model_name"] == "decoder-small")
        history = decoder_small["history"]
        assert len(history) == 3
        # Newest first: 2026-03-17 > 2026-03-16 > 2026-03-15
        assert history[0]["timestamp"] > history[1]["timestamp"]

    def test_model_status_ok(self, tmp_path: Path):
        """Latest run (highest TPS) equals baseline → no regression."""
        mod = _import_report()
        db_path = self._write_db_to_tempfile(tmp_path)
        rows, _ = mod.load_report_data(db_path)
        decoder_small = next(r for r in rows if r["model_name"] == "decoder-small")
        # Latest TPS=150 is also the best, so no regression
        assert decoder_small["status"] == "ok"

    def test_report_end_to_end(self, tmp_path: Path):
        """Full report generation produces valid HTML with expected content."""
        mod = _import_report()
        db_path = self._write_db_to_tempfile(tmp_path)
        rows, envs = mod.load_report_data(db_path)
        html_str = mod.render_report(rows, envs, title="CI Perf Report")
        assert "<!DOCTYPE html>" in html_str
        assert "decoder-small" in html_str
        assert "decoder-alt" in html_str
        assert "NVIDIA B200" in html_str
        assert "CI Perf Report" in html_str

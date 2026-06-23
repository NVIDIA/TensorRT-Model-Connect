"""Unit tests for tools/perfdb.py — performance database.

All tests use in-memory SQLite (:memory:), no GPU or filesystem needed.

Trace: ARCH-PERF-001, UD-PERF-DB
Intent: Validate PerfDB schema creation, record ingestion, environment fingerprinting, and query correctness
Preconditions: In-memory SQLite database is used; no GPU or filesystem needed
Postconditions: Records are correctly stored, queried, and environment IDs are deterministically computed
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


# Ensure tools/ is importable (conftest.py adds it, but be explicit)
_TOOLS_DIR = str(Path(__file__).resolve().parents[2] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from perfdb import PerfDB, _compute_env_id  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(**overrides):
    fp = {
        "gpu_name": "NVIDIA H100",
        "driver": "535.129.03",
        "trt_version": "10.1.0",
        "cuda_version": "12.2",
        "hostname": "gpu-server-01",
    }
    fp.update(overrides)
    return fp


def _make_perf_compare_json(**overrides):
    data = {
        "metadata": {
            "model": "example-org/example-decoder",
            "gpu": "NVIDIA H100",
            "trt_version": "10.1.0",
            "prompt": "The capital of France is",
            "num_input_tokens": 6,
            "max_new_tokens": 20,
            "warmup": 2,
            "iterations": 5,
            "hf_dtype": "float16",
            "timestamp": "2025-01-15T00:00:00+00:00",
        },
        "trt": {
            "prefill_ms": {"mean": 5.0, "std": 0.5, "values": [4.5, 5.0, 5.5]},
            "decode_ms": {"mean": 100.0, "std": 2.0, "values": [98, 100, 102]},
            "per_token_ms": {"mean": 5.0, "std": 0.1},
            "throughput_tps": {"mean": 200.0, "std": 4.0},
            "total_ms": {"mean": 105.0, "std": 2.5},
            "num_decode_tokens": 20,
        },
        "hf": {
            "prefill_ms": {"mean": 15.0, "std": 1.0},
            "decode_ms": {"mean": 300.0, "std": 5.0},
            "per_token_ms": {"mean": 15.0, "std": 0.3},
            "throughput_tps": {"mean": 66.7, "std": 1.0},
            "total_ms": {"mean": 315.0, "std": 6.0},
            "num_decode_tokens": 20,
        },
        "speedup": {
            "prefill": 3.0,
            "decode": 3.0,
            "per_token": 3.0,
            "throughput": 3.0,
            "total": 3.0,
        },
        "token_match": True,
    }
    data.update(overrides)
    return data


class _FakeMetricResult:
    """Minimal stand-in for contracts.MetricResult."""
    def __init__(self, value, passed=True):
        self.value = value
        self.passed = passed


class _FakeCompareResult:
    """Minimal stand-in for contracts.CompareResult."""
    def __init__(self, status="passed", metrics=None, message=""):
        self.status = status
        self.metrics = metrics or {}
        self.message = message


class _FakeE2EResult:
    """Minimal stand-in for contracts.E2EResult."""
    def __init__(self, case_name="example-decoder", status="pass",
                 timing=None, stages=None):
        self.case_name = case_name
        self.status = status
        self.timing = timing or {}
        self.stages = stages or {}


class _FakeE2ECase:
    """Minimal stand-in for contracts.E2ECase."""
    def __init__(self, name="example-decoder", hf_id="example-org/example-decoder",
                 runtime_strategy="decoder_kv_cache",
                 task_strategy="text_generation_causal"):
        self.name = name
        self.hf_id = hf_id
        self.runtime_strategy = runtime_strategy
        self.task_strategy = task_strategy
        self.inputs = {"max_cache_length": 256}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchema:
    """Test database schema creation."""

    def test_create_tables(self):
        db = PerfDB(":memory:")
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "environments" in tables
        assert "perf_runs" in tables
        db.close()

    def test_create_indexes(self):
        db = PerfDB(":memory:")
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_perf_runs_model_ts" in indexes
        assert "idx_perf_runs_env_model" in indexes
        db.close()

    def test_idempotent_schema(self):
        """Creating PerfDB twice on the same connection is safe."""
        db = PerfDB(":memory:")
        # Re-run schema (simulates second open)
        db._ensure_schema()
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "environments" in tables
        assert "perf_runs" in tables
        db.close()


class TestEnsureEnv:
    """Test environment upsert."""

    def test_insert_new_env(self):
        db = PerfDB(":memory:")
        fp = _make_fingerprint()
        env_id = db.ensure_env(fp)
        assert isinstance(env_id, str)
        assert len(env_id) == 16

        # Verify row exists
        cursor = db._conn.execute(
            "SELECT * FROM environments WHERE env_id = ?", (env_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row["gpu_name"] == "NVIDIA H100"
        assert row["hostname"] == "gpu-server-01"
        db.close()

    def test_dedup_same_env(self):
        """Same fingerprint returns same env_id, no duplicate rows."""
        db = PerfDB(":memory:")
        fp = _make_fingerprint()
        id1 = db.ensure_env(fp)
        id2 = db.ensure_env(fp)
        assert id1 == id2

        cursor = db._conn.execute("SELECT COUNT(*) FROM environments")
        assert cursor.fetchone()[0] == 1
        db.close()

    def test_different_envs_different_ids(self):
        db = PerfDB(":memory:")
        id1 = db.ensure_env(_make_fingerprint(gpu_name="NVIDIA H100"))
        id2 = db.ensure_env(_make_fingerprint(gpu_name="NVIDIA A100"))
        assert id1 != id2

        cursor = db._conn.execute("SELECT COUNT(*) FROM environments")
        assert cursor.fetchone()[0] == 2
        db.close()

    def test_env_id_deterministic(self):
        """Same inputs always produce the same env_id."""
        fp = _make_fingerprint()
        id1 = _compute_env_id(fp)
        id2 = _compute_env_id(fp)
        assert id1 == id2


class TestRecordAndQueryHistory:
    """Test recording runs and querying history."""

    def test_record_e2e_and_query(self):
        db = PerfDB(":memory:")
        fp = _make_fingerprint()
        case = _FakeE2ECase()
        result = _FakeE2EResult(
            timing={"build_s": 10.5, "trt_generate_s": 1.2, "ref_generate_s": 3.4},
            stages={
                "generate": _FakeCompareResult(
                    status="passed",
                    metrics={"token_agreement_rate": _FakeMetricResult(1.0)},
                ),
            },
        )

        git_info = {"commit": "abc123", "branch": "main"}
        run_id = db.record_e2e(result, case, fp, git_info)
        assert isinstance(run_id, int)
        assert run_id > 0

        history = db.query_history("example-decoder")
        assert len(history) == 1
        row = history[0]
        assert row["model_name"] == "example-decoder"
        assert row["source"] == "e2e_harness"
        assert row["git_commit"] == "abc123"
        assert row["build_s"] == 10.5
        assert row["trt_run_s"] == 1.2
        assert row["token_match"] == 1
        assert row["e2e_status"] == "pass"
        db.close()

    def test_record_perf_compare_and_query(self):
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        git_info = {"commit": "def456", "branch": "feat/perf"}

        run_id = db.record_perf_compare(json_data, git_info)
        assert run_id > 0

        history = db.query_history("example-org/example-decoder")
        assert len(history) == 1
        row = history[0]
        assert row["source"] == "perf_compare"
        assert row["prefill_ms_mean"] == 5.0
        assert row["decode_ms_mean"] == 100.0
        assert row["throughput_tps"] == 200.0
        assert row["speedup"] == 3.0
        assert row["token_match"] == 1
        db.close()

    def test_query_history_limit(self):
        db = PerfDB(":memory:")
        fp = _make_fingerprint()
        case = _FakeE2ECase()

        # Insert 5 runs
        for i in range(5):
            result = _FakeE2EResult(timing={"build_s": float(i)})
            db.record_e2e(result, case, fp, {"commit": f"c{i}", "branch": "main"})

        # Limit to 3
        history = db.query_history("example-decoder", limit=3)
        assert len(history) == 3
        db.close()

    def test_query_history_env_filter(self):
        db = PerfDB(":memory:")
        fp1 = _make_fingerprint(gpu_name="H100")
        fp2 = _make_fingerprint(gpu_name="A100")
        case = _FakeE2ECase()

        env1 = db.ensure_env(fp1)
        env2 = db.ensure_env(fp2)

        db.record_e2e(
            _FakeE2EResult(timing={"build_s": 1.0}),
            case, fp1, {"commit": "c1", "branch": "main"})
        db.record_e2e(
            _FakeE2EResult(timing={"build_s": 2.0}),
            case, fp2, {"commit": "c2", "branch": "main"})

        # Filter by env
        h1 = db.query_history("example-decoder", env_id=env1)
        assert len(h1) == 1
        assert h1[0]["env_id"] == env1

        h2 = db.query_history("example-decoder", env_id=env2)
        assert len(h2) == 1
        assert h2[0]["env_id"] == env2
        db.close()

    def test_query_empty_history(self):
        db = PerfDB(":memory:")
        history = db.query_history("nonexistent-model")
        assert history == []
        db.close()


class TestQueryBaseline:
    """Test baseline (best-perf) queries."""

    def test_baseline_by_throughput(self):
        db = PerfDB(":memory:")
        # Insert two perf_compare runs with different throughputs
        json1 = _make_perf_compare_json()
        json1["trt"]["throughput_tps"]["mean"] = 150.0
        json2 = _make_perf_compare_json()
        json2["trt"]["throughput_tps"]["mean"] = 250.0

        db.record_perf_compare(json1, {"commit": "c1", "branch": "main"})
        db.record_perf_compare(json2, {"commit": "c2", "branch": "main"})

        baseline = db.query_baseline("example-org/example-decoder")
        assert baseline is not None
        assert baseline["throughput_tps"] == 250.0
        db.close()

    def test_baseline_fallback_to_trt_run_s(self):
        """E2E runs without throughput_tps use trt_run_s as fallback."""
        db = PerfDB(":memory:")
        fp = _make_fingerprint()
        case = _FakeE2ECase()

        # Run with trt_run_s but no throughput
        r1 = _FakeE2EResult(timing={"trt_generate_s": 5.0})
        r2 = _FakeE2EResult(timing={"trt_generate_s": 2.0})

        db.record_e2e(r1, case, fp, {"commit": "c1", "branch": "main"})
        db.record_e2e(r2, case, fp, {"commit": "c2", "branch": "main"})

        baseline = db.query_baseline("example-decoder")
        assert baseline is not None
        assert baseline["trt_run_s"] == 2.0  # fastest
        db.close()

    def test_baseline_none_when_empty(self):
        db = PerfDB(":memory:")
        assert db.query_baseline("nonexistent") is None
        db.close()


class TestCompareToBaseline:
    """Test regression detection."""

    def test_no_baseline(self):
        db = PerfDB(":memory:")
        result = db.compare_to_baseline("model", None, {"throughput_tps": 100})
        assert result["has_baseline"] is False
        assert result["regression"] is False
        db.close()

    def test_within_threshold(self):
        db = PerfDB(":memory:")
        # Insert a baseline
        json_data = _make_perf_compare_json()
        json_data["trt"]["throughput_tps"]["mean"] = 200.0
        json_data["trt"]["decode_ms"]["mean"] = 100.0
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        # Current run is 5% slower (within 10% threshold)
        current = {"throughput_tps": 192.0, "decode_ms_mean": 104.0}
        result = db.compare_to_baseline("example-org/example-decoder", None, current)
        assert result["has_baseline"] is True
        assert result["regression"] is False
        assert "Within threshold" in result["details"]
        db.close()

    def test_regression_detected(self):
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        json_data["trt"]["throughput_tps"]["mean"] = 200.0
        json_data["trt"]["decode_ms"]["mean"] = 100.0
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        # Current run is 25% slower (beyond 10% threshold)
        current = {"throughput_tps": 140.0, "decode_ms_mean": 130.0}
        result = db.compare_to_baseline("example-org/example-decoder", None, current)
        assert result["has_baseline"] is True
        assert result["regression"] is True
        assert "throughput_tps" in result["details"]
        assert "decode_ms_mean" in result["details"]
        db.close()

    def test_custom_threshold(self):
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        json_data["trt"]["throughput_tps"]["mean"] = 200.0
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        # 8% regression with 5% threshold
        current = {"throughput_tps": 184.0}
        result = db.compare_to_baseline(
            "example-org/example-decoder", None, current, threshold=0.05)
        assert result["regression"] is True

        # Same 8% regression with 10% threshold
        result = db.compare_to_baseline(
            "example-org/example-decoder", None, current, threshold=0.10)
        assert result["regression"] is False
        db.close()


class TestExport:
    """Test export functionality."""

    def test_export_json(self):
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        output = db.export_all(fmt="json")
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["model_name"] == "example-org/example-decoder"
        db.close()

    def test_export_csv(self):
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        output = db.export_all(fmt="csv")
        lines = output.strip().split("\n")
        assert len(lines) == 2  # header + 1 row
        assert "model_name" in lines[0]
        assert "example-org/example-decoder" in lines[1]
        db.close()

    def test_export_empty_json(self):
        db = PerfDB(":memory:")
        output = db.export_all(fmt="json")
        assert json.loads(output) == []
        db.close()

    def test_export_empty_csv(self):
        db = PerfDB(":memory:")
        output = db.export_all(fmt="csv")
        assert output == ""
        db.close()


class TestCLI:
    """Test CLI subcommands via main()."""

    def test_cli_history(self, capsys):
        """CLI history subcommand prints recent runs."""
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json()
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})
        db.close()

        # --db must come before the subcommand (parent parser argument)
        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb", "--db", ":memory:", "history", "nonexistent"]):
            perfdb_main()
        captured = capsys.readouterr()
        assert "No runs found" in captured.out

    def test_cli_baseline(self, capsys):
        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb", "--db", ":memory:", "baseline", "nonexistent"]):
            perfdb_main()
        captured = capsys.readouterr()
        assert "No baseline found" in captured.out

    def test_cli_export_json(self, capsys):
        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb", "--db", ":memory:", "export", "--format", "json"]):
            perfdb_main()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == []

    def test_cli_export_csv(self, capsys):
        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb", "--db", ":memory:", "export", "--format", "csv"]):
            perfdb_main()
        captured = capsys.readouterr()
        assert captured.out == "\n"  # empty CSV

    def test_cli_no_command(self, capsys):
        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb"]):
            perfdb_main()
        # Should print help (no crash)
        captured = capsys.readouterr()
        assert "performance" in captured.out.lower() or "usage" in captured.out.lower()


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_record_e2e_no_timing(self):
        """E2E result with empty timing dict."""
        db = PerfDB(":memory:")
        result = _FakeE2EResult(timing={})
        case = _FakeE2ECase()
        fp = _make_fingerprint()
        run_id = db.record_e2e(result, case, fp, {"commit": "", "branch": ""})
        assert run_id > 0

        history = db.query_history("example-decoder")
        assert len(history) == 1
        assert history[0]["build_s"] is None
        db.close()

    def test_record_perf_compare_token_mismatch(self):
        """perf_compare result with token_match=False."""
        db = PerfDB(":memory:")
        json_data = _make_perf_compare_json(token_match=False)
        db.record_perf_compare(json_data, {"commit": "c1", "branch": "main"})

        history = db.query_history("example-org/example-decoder")
        assert history[0]["token_match"] == 0
        db.close()

    def test_multiple_models(self):
        """Different models are kept separate."""
        db = PerfDB(":memory:")
        fp = _make_fingerprint()

        case1 = _FakeE2ECase(name="model-a", hf_id="org/model-a")
        case2 = _FakeE2ECase(name="model-b", hf_id="org/model-b")

        db.record_e2e(_FakeE2EResult(), case1, fp, {"commit": "c1", "branch": "main"})
        db.record_e2e(_FakeE2EResult(), case2, fp, {"commit": "c2", "branch": "main"})

        assert len(db.query_history("model-a")) == 1
        assert len(db.query_history("model-b")) == 1
        assert len(db.query_history("model-c")) == 0
        db.close()

    def test_e2e_token_match_false(self):
        """Token agreement < 1.0 sets token_match=0."""
        db = PerfDB(":memory:")
        result = _FakeE2EResult(
            stages={
                "generate": _FakeCompareResult(
                    metrics={"token_agreement_rate": _FakeMetricResult(0.8)},
                ),
            },
        )
        case = _FakeE2ECase()
        fp = _make_fingerprint()
        db.record_e2e(result, case, fp, {"commit": "c1", "branch": "main"})

        history = db.query_history("example-decoder")
        assert history[0]["token_match"] == 0
        db.close()

    def test_extra_json_stored(self):
        """Extra data is stored as JSON in extra_json column."""
        db = PerfDB(":memory:")
        result = _FakeE2EResult(
            stages={
                "generate": _FakeCompareResult(
                    status="passed",
                    metrics={"logit_cosine_p5": _FakeMetricResult(0.999)},
                ),
            },
        )
        case = _FakeE2ECase()
        fp = _make_fingerprint()
        db.record_e2e(result, case, fp, {"commit": "c1", "branch": "main"})

        history = db.query_history("example-decoder")
        extra = json.loads(history[0]["extra_json"])
        assert "stages" in extra
        assert "generate" in extra["stages"]
        assert extra["stages"]["generate"]["metrics"]["logit_cosine_p5"] == 0.999
        db.close()


# ---------------------------------------------------------------------------
# detect_env_fingerprint tests
# ---------------------------------------------------------------------------

class TestDetectEnvFingerprint:
    """Tests for detect_env_fingerprint() — hardware/software detection."""

    def test_returns_all_five_keys(self):
        from perfdb import detect_env_fingerprint
        with patch("subprocess.run") as mock_run, \
             patch("socket.gethostname", return_value="test-host"):
            mock_run.return_value = type("R", (), {
                "returncode": 1, "stdout": ""})()
            env = detect_env_fingerprint()
        assert set(env.keys()) == {
            "gpu_name", "driver", "trt_version", "cuda_version", "hostname"}

    def test_parses_nvidia_smi_output(self):
        from perfdb import detect_env_fingerprint
        fake_smi = type("R", (), {
            "returncode": 0,
            "stdout": "NVIDIA H100, 535.129.03\n"})()
        fake_nvcc = type("R", (), {
            "returncode": 1, "stdout": ""})()

        def side_effect(cmd, **kw):
            if "nvidia-smi" in cmd:
                return fake_smi
            return fake_nvcc

        with patch("subprocess.run", side_effect=side_effect), \
             patch("socket.gethostname", return_value="gpu-01"):
            env = detect_env_fingerprint()
        assert env["gpu_name"] == "NVIDIA H100"
        assert env["driver"] == "535.129.03"
        assert env["hostname"] == "gpu-01"

    def test_parses_nvcc_output(self):
        from perfdb import detect_env_fingerprint
        fake_smi = type("R", (), {"returncode": 1, "stdout": ""})()
        fake_nvcc = type("R", (), {
            "returncode": 0,
            "stdout": "nvcc: NVIDIA (R) Cuda compiler\n"
                      "Cuda compilation tools, release 12.6, V12.6.77\n"})()

        def side_effect(cmd, **kw):
            if "nvcc" in cmd:
                return fake_nvcc
            return fake_smi

        with patch("subprocess.run", side_effect=side_effect), \
             patch("socket.gethostname", return_value="h"):
            env = detect_env_fingerprint()
        assert env["cuda_version"] == "12.6"

    def test_graceful_fallback_on_failure(self):
        from perfdb import detect_env_fingerprint
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ImportError("no tensorrt")
            return real_import(name, *args, **kwargs)

        with patch("subprocess.run", side_effect=OSError("no nvidia-smi")), \
             patch("socket.gethostname", side_effect=OSError("no host")), \
             patch("builtins.__import__", side_effect=fake_import):
            env = detect_env_fingerprint()
        # All fields should be empty strings, no crash
        for v in env.values():
            assert v == ""


# ---------------------------------------------------------------------------
# busy_timeout / WAL tests
# ---------------------------------------------------------------------------

class TestSQLitePragmas:
    """Tests for SQLite WAL mode and busy_timeout configuration."""

    def test_busy_timeout_set(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = PerfDB(db_path)
        result = db._conn.execute("PRAGMA busy_timeout").fetchone()
        assert result[0] == 30000
        db.close()

    def test_wal_mode(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = PerfDB(db_path)
        result = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"
        db.close()


# ---------------------------------------------------------------------------
# export --output tests
# ---------------------------------------------------------------------------

class TestExportToFile:
    """Tests for perfdb.py export --output file writing."""

    def test_export_to_file(self, tmp_path):
        import os
        db_path = str(tmp_path / "test.db")
        out_path = str(tmp_path / "out.json")
        db = PerfDB(db_path)
        db.record_perf_compare(_make_perf_compare_json())
        db.close()

        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb.py", "--db", db_path,
                                 "export", "--format", "json",
                                 "--output", out_path]):
            perfdb_main()

        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_csv_to_file(self, tmp_path):
        import os
        db_path = str(tmp_path / "test.db")
        out_path = str(tmp_path / "out.csv")
        db = PerfDB(db_path)
        db.record_perf_compare(_make_perf_compare_json())
        db.close()

        from perfdb import main as perfdb_main
        with patch("sys.argv", ["perfdb.py", "--db", db_path,
                                 "export", "--format", "csv",
                                 "--output", out_path]):
            perfdb_main()

        assert os.path.exists(out_path)
        content = open(out_path).read()
        assert "run_id" in content  # CSV header

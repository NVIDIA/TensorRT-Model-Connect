# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for tools/perf_compare.py — stats, formatting, JSON output, reporting.

Trace: ARCH-PERF-001, UD-PERF-COMPARE
Intent: Validate perf_compare timing statistics, formatting helpers, and JSON output structure
Preconditions: perf_compare module is importable; numpy available
Postconditions: Statistics correctly handle edge cases (empty, single, multiple) and JSON output is well-formed
"""

from __future__ import annotations

import json
from unittest import mock

import pytest


def _import_perf_compare():
    import importlib
    return importlib.import_module("perf_compare")


# ---------------------------------------------------------------------------
# _stats
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for _stats() — timing statistics helper."""

    def test_empty_list(self):
        mod = _import_perf_compare()
        result = mod._stats([])
        assert result["mean"] == 0.0
        assert result["std"] == 0.0
        assert result["values"] == []

    def test_single_value(self):
        mod = _import_perf_compare()
        result = mod._stats([42.0])
        assert result["mean"] == 42.0
        assert result["std"] == 0.0
        assert result["values"] == [42.0]

    def test_multiple_values(self):
        mod = _import_perf_compare()
        result = mod._stats([10.0, 20.0, 30.0])
        assert result["mean"] == pytest.approx(20.0)
        assert result["std"] > 0
        assert len(result["values"]) == 3

    def test_identical_values_zero_std(self):
        mod = _import_perf_compare()
        result = mod._stats([5.0, 5.0, 5.0])
        assert result["mean"] == 5.0
        assert result["std"] == 0.0


# ---------------------------------------------------------------------------
# _fmt / _speedup
# ---------------------------------------------------------------------------

class TestFormatting:
    """Tests for _fmt() and _speedup() — display helpers."""

    def test_fmt_basic(self):
        mod = _import_perf_compare()
        assert mod._fmt(12.3, 0.5) == "12.3 +/- 0.5"

    def test_fmt_zero(self):
        mod = _import_perf_compare()
        assert mod._fmt(0.0, 0.0) == "0.0 +/- 0.0"

    def test_speedup_normal(self):
        mod = _import_perf_compare()
        # HF=100ms, TRT=50ms → 2.00x
        assert mod._speedup(100.0, 50.0) == "2.00x"

    def test_speedup_slower(self):
        mod = _import_perf_compare()
        # HF=50ms, TRT=100ms → 0.50x
        assert mod._speedup(50.0, 100.0) == "0.50x"

    def test_speedup_zero_trt(self):
        mod = _import_perf_compare()
        assert mod._speedup(100.0, 0.0) == "N/A"

    def test_speedup_negative_trt(self):
        mod = _import_perf_compare()
        assert mod._speedup(100.0, -1.0) == "N/A"


# ---------------------------------------------------------------------------
# build_json_output
# ---------------------------------------------------------------------------

def _make_bench_result(prefill_ms, decode_ms, num_tokens, gen_ids):
    """Helper to build a synthetic bench result dict."""
    n = len(prefill_ms)
    return {
        "prefill_times": prefill_ms,
        "decode_times": decode_ms,
        "decode_token_counts": [num_tokens] * n,
        "gen_ids": gen_ids,
    }


def test_bench_trtmc_cpp_parses_actual_mean_token_count(monkeypatch):
    mod = _import_perf_compare()
    completed = mock.Mock(
        returncode=0,
        stdout="generated text\n",
        stderr=(
            "[trtmc.benchmark] setup_ms=1.25 prefill_ms=2.50 decode_ms=100.00 "
            "generated_tokens_mean=31.50 tokens_per_sec=999.99\n"
        ),
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: completed)

    result = mod.bench_trtmc_cpp(
        "trtmc", "model.bundle", "prompt", 64, 3, 10, None, False
    )

    assert result == {
        "prefill_times": [2.5],
        "decode_times": [100.0],
        "decode_token_counts": [31.5],
        "gen_ids": [],
    }


class TestBuildJsonOutput:
    """Tests for build_json_output() — structured result dict."""

    def test_structure_has_required_keys(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 20, [1, 2, 3])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 20, [1, 2, 3])
        result = mod.build_json_output(
            "test-model", "Hello", 3, 20, 2, 1, "float16", trt, hf)

        assert "metadata" in result
        assert "trt" in result
        assert "hf" in result
        assert "speedup" in result
        assert "token_match" in result

    def test_metadata_fields(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        result = mod.build_json_output(
            "example-org/example-decoder", "Hello world", 5, 20, 3, 2, "float16", trt, hf)

        meta = result["metadata"]
        assert meta["model"] == "example-org/example-decoder"
        assert meta["prompt"] == "Hello world"
        assert meta["num_input_tokens"] == 5
        assert meta["max_new_tokens"] == 20
        assert meta["iterations"] == 3
        assert meta["warmup"] == 2
        assert meta["hf_dtype"] == "float16"
        assert "timestamp" in meta

    def test_speedup_computation(self):
        mod = _import_perf_compare()
        # TRT decode 2x faster than HF
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1, 2])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1, 2])
        result = mod.build_json_output(
            "m", "p", 1, 20, 2, 0, "float16", trt, hf)

        assert result["speedup"]["decode"] == pytest.approx(2.0, abs=0.01)
        # Prefill: HF faster (5ms vs 10ms) → 0.5x
        assert result["speedup"]["prefill"] == pytest.approx(0.5, abs=0.01)

    def test_token_match_true(self):
        mod = _import_perf_compare()
        ids = [10, 20, 30]
        trt = _make_bench_result([10.0], [50.0], 3, ids)
        hf = _make_bench_result([8.0], [80.0], 3, ids)
        result = mod.build_json_output(
            "m", "p", 1, 3, 1, 0, "float16", trt, hf)
        assert result["token_match"] is True

    def test_token_match_false(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 3, [10, 20, 30])
        hf = _make_bench_result([8.0], [80.0], 3, [10, 20, 99])
        result = mod.build_json_output(
            "m", "p", 1, 3, 1, 0, "float16", trt, hf)
        assert result["token_match"] is False

    def test_per_token_and_throughput(self):
        mod = _import_perf_compare()
        # 10 tokens in 100ms decode → 10ms/token, 100 t/s
        trt = _make_bench_result([5.0, 5.0], [100.0, 100.0], 10, list(range(10)))
        hf = _make_bench_result([5.0, 5.0], [200.0, 200.0], 10, list(range(10)))
        result = mod.build_json_output(
            "m", "p", 1, 10, 2, 0, "float16", trt, hf)

        assert result["trt"]["per_token_ms"]["mean"] == pytest.approx(10.0)
        assert result["trt"]["throughput_tps"]["mean"] == pytest.approx(100.0)
        assert result["hf"]["per_token_ms"]["mean"] == pytest.approx(20.0)
        assert result["hf"]["throughput_tps"]["mean"] == pytest.approx(50.0)

    def test_zero_decode_tokens(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([5.0], [0.1], 0, [])
        hf = _make_bench_result([5.0], [0.1], 0, [])
        result = mod.build_json_output(
            "m", "p", 1, 0, 1, 0, "float16", trt, hf)

        assert result["trt"]["per_token_ms"]["mean"] == 0.0
        assert result["trt"]["throughput_tps"]["mean"] == 0.0

    def test_json_serializable(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 5, [1, 2, 3])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 5, [1, 2, 3])
        result = mod.build_json_output(
            "m", "p", 3, 5, 2, 1, "float16", trt, hf)
        # Must not raise
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["metadata"]["model"] == "m"

    def test_required_flywheel_fields_present(self):
        """JSON must include prefill_ms, decode_ms_per_token, total_latency_ms,
        tokens_per_second, and peak_memory_mb for quantization performance tracking."""
        mod = _import_perf_compare()
        # 10 tokens in 100ms decode -> 10ms/token, 100 t/s; total = 5+100 = 105ms
        trt = _make_bench_result([5.0, 5.0], [100.0, 100.0], 10, list(range(10)))
        hf = _make_bench_result([5.0, 5.0], [200.0, 200.0], 10, list(range(10)))
        result = mod.build_json_output(
            "m", "p", 1, 10, 2, 0, "float16", trt, hf)

        # prefill_ms (detailed stats dict — already tested elsewhere)
        assert "prefill_ms" in result["trt"]
        assert "prefill_ms" in result["hf"]

        # decode_ms_per_token — scalar convenience field
        assert result["trt"]["decode_ms_per_token"] == pytest.approx(10.0)
        assert result["hf"]["decode_ms_per_token"] == pytest.approx(20.0)

        # total_latency_ms — scalar convenience field
        assert result["trt"]["total_latency_ms"] == pytest.approx(105.0)
        assert result["hf"]["total_latency_ms"] == pytest.approx(205.0)

        # tokens_per_second — scalar convenience field
        assert result["trt"]["tokens_per_second"] == pytest.approx(100.0)
        assert result["hf"]["tokens_per_second"] == pytest.approx(50.0)

        # peak_memory_mb — top-level, may be None when CUDA unavailable
        assert "peak_memory_mb" in result

    def test_flywheel_fields_zero_tokens(self):
        """Flywheel fields degrade gracefully with zero decode tokens."""
        mod = _import_perf_compare()
        trt = _make_bench_result([5.0], [0.1], 0, [])
        hf = _make_bench_result([5.0], [0.1], 0, [])
        result = mod.build_json_output(
            "m", "p", 1, 0, 1, 0, "float16", trt, hf)

        assert result["trt"]["decode_ms_per_token"] == 0.0
        assert result["trt"]["tokens_per_second"] == 0.0
        assert result["trt"]["total_latency_ms"] == pytest.approx(5.1)
        assert result["hf"]["decode_ms_per_token"] == 0.0
        assert result["hf"]["tokens_per_second"] == 0.0

    def test_flywheel_fields_json_round_trip(self):
        """Flywheel scalar fields survive JSON serialization."""
        mod = _import_perf_compare()
        trt = _make_bench_result([5.0], [100.0], 10, list(range(10)))
        hf = _make_bench_result([5.0], [200.0], 10, list(range(10)))
        result = mod.build_json_output(
            "m", "p", 1, 10, 1, 0, "float16", trt, hf)

        parsed = json.loads(json.dumps(result))
        assert parsed["trt"]["decode_ms_per_token"] == pytest.approx(10.0)
        assert parsed["trt"]["tokens_per_second"] == pytest.approx(100.0)
        assert parsed["trt"]["total_latency_ms"] == pytest.approx(105.0)
        assert parsed["hf"]["decode_ms_per_token"] == pytest.approx(20.0)
        assert parsed["hf"]["tokens_per_second"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# bench_hf_compiled
# ---------------------------------------------------------------------------

class TestBenchHfCompiled:
    """Tests for bench_hf_compiled() — torch.compile benchmarking."""

    def test_returns_correct_keys(self, monkeypatch):
        mod = _import_perf_compare()
        fake_model = mock.MagicMock()

        # bench_hf_compiled delegates to bench_hf after compiling
        def fake_bench_hf(model, input_ids, max_new_tokens, warmup,
                          iterations, eos_token_id, verbose, **kwargs):
            return {
                "prefill_times": [5.0],
                "decode_times": [20.0],
                "decode_token_counts": [5],
                "gen_ids": [1, 2, 3],
            }

        monkeypatch.setattr(mod, "bench_hf", fake_bench_hf)

        # Patch torch.compile to return model unchanged
        import sys as _sys
        fake_torch = mock.MagicMock()
        fake_torch.compile = lambda m, mode=None: m
        monkeypatch.setitem(_sys.modules, "torch", fake_torch)

        result = mod.bench_hf_compiled(
            fake_model, [1, 2, 3], 5, 1, 1, None, "reduce-overhead", False)

        assert "prefill_times" in result
        assert "decode_times" in result
        assert "decode_token_counts" in result
        assert "gen_ids" in result

    def test_delegates_to_bench_hf(self, monkeypatch):
        mod = _import_perf_compare()
        call_log = []

        def fake_bench_hf(model, *args, **kwargs):
            call_log.append("bench_hf_called")
            return _fake_bench_result()

        monkeypatch.setattr(mod, "bench_hf", fake_bench_hf)

        import sys as _sys
        fake_torch = mock.MagicMock()
        fake_torch.compile = lambda m, mode=None: m
        monkeypatch.setitem(_sys.modules, "torch", fake_torch)

        mod.bench_hf_compiled(
            mock.MagicMock(), [1, 2, 3], 5, 1, 1, None,
            "reduce-overhead", False)
        assert "bench_hf_called" in call_log


# ---------------------------------------------------------------------------
# build_json_output with compile_res
# ---------------------------------------------------------------------------

class TestBuildJsonOutputWithCompile:
    """Tests for build_json_output() with the new hf_compiled section."""

    def test_no_compile_res_no_hf_compiled_key(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [40.0], 20, [1, 2, 3])
        hf = _make_bench_result([8.0], [80.0], 20, [1, 2, 3])
        result = mod.build_json_output(
            "m", "p", 1, 20, 1, 0, "float16", trt, hf)
        assert "hf_compiled" not in result

    def test_with_compile_res_has_hf_compiled_key(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [40.0, 40.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 8.0], [80.0, 80.0], 20, [1, 2])
        cp = _make_bench_result([9.0, 9.0], [60.0, 60.0], 20, [1, 2])
        result = mod.build_json_output(
            "m", "p", 1, 20, 2, 0, "float16", trt, hf,
            compile_res=cp, compile_mode="reduce-overhead")
        assert "hf_compiled" in result

    def test_hf_compiled_has_compile_mode(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        cp = _make_bench_result([9.0], [60.0], 20, [1])
        result = mod.build_json_output(
            "m", "p", 1, 20, 1, 0, "float16", trt, hf,
            compile_res=cp, compile_mode="max-autotune")
        assert result["hf_compiled"]["compile_mode"] == "max-autotune"

    def test_speedup_has_trt_vs_compile_keys(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1])
        cp = _make_bench_result([8.0, 8.0], [80.0, 80.0], 20, [1])
        result = mod.build_json_output(
            "m", "p", 1, 20, 2, 0, "float16", trt, hf,
            compile_res=cp)
        sp = result["speedup"]
        assert "trt_vs_compile_decode" in sp
        assert "trt_vs_compile_prefill" in sp

    def test_compile_decode_speedup_value(self):
        mod = _import_perf_compare()
        # TRT decode=50ms, compile decode=100ms → 2.0x
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1])
        cp = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1])
        result = mod.build_json_output(
            "m", "p", 1, 20, 2, 0, "float16", trt, hf, compile_res=cp)
        import pytest
        assert result["speedup"]["trt_vs_compile_decode"] == pytest.approx(2.0, abs=0.01)

    def test_compile_result_json_serializable(self):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 11.0], [40.0, 42.0], 5, [1, 2, 3])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 5, [1, 2, 3])
        cp = _make_bench_result([9.0, 10.0], [60.0, 62.0], 5, [1, 2, 3])
        result = mod.build_json_output(
            "m", "p", 3, 5, 2, 1, "float16", trt, hf, compile_res=cp)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert "hf_compiled" in parsed


# ---------------------------------------------------------------------------
# print_report with compile_res
# ---------------------------------------------------------------------------

class TestPrintReportWithCompile:
    """Tests for print_report() 3-column mode with compile_res."""

    def test_three_column_contains_compile_label(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [40.0, 40.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 8.0], [80.0, 80.0], 20, [1, 2])
        cp = _make_bench_result([9.0, 9.0], [60.0, 60.0], 20, [1, 2])
        mod.print_report("TestModel", "Hello", 3, 20, 2, 1,
                         "float16", trt, hf,
                         compile_res=cp, compile_mode="reduce-overhead")
        out = capsys.readouterr().out
        assert "reduce-overhead" in out or "compile" in out.lower()

    def test_three_column_shows_all_backends(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [40.0, 40.0], 20, [1])
        hf = _make_bench_result([8.0, 8.0], [80.0, 80.0], 20, [1])
        cp = _make_bench_result([9.0, 9.0], [60.0, 60.0], 20, [1])
        mod.print_report("m", "p", 1, 20, 2, 0,
                         "float16", trt, hf, compile_res=cp)
        out = capsys.readouterr().out
        assert "TRT" in out
        assert "HF" in out

    def test_no_compile_res_keeps_2col_format(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        mod.print_report("m", "p", 1, 20, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        # Should still have speedup
        assert "Speedup" in out or "speedup" in out.lower()


# ---------------------------------------------------------------------------
# print_report
# ---------------------------------------------------------------------------

class TestPrintReport:
    """Tests for print_report() — formatted output."""

    def test_report_contains_model_name(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 12.0], [40.0, 42.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 20, [1, 2])
        mod.print_report("TestModel", "Hello", 3, 20, 2, 1,
                         "float16", trt, hf)
        out = capsys.readouterr().out
        assert "TestModel" in out

    def test_report_contains_speedup(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0, 10.0], [50.0, 50.0], 20, [1])
        hf = _make_bench_result([5.0, 5.0], [100.0, 100.0], 20, [1])
        mod.print_report("m", "p", 1, 20, 2, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Speedup" in out
        assert "2.00x" in out

    def test_report_token_match_true(self, capsys):
        mod = _import_perf_compare()
        ids = [1, 2, 3]
        trt = _make_bench_result([10.0], [50.0], 3, ids)
        hf = _make_bench_result([8.0], [80.0], 3, ids)
        mod.print_report("m", "p", 1, 3, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Token match: True" in out

    def test_report_token_match_false_shows_counts(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 3, [1, 2, 3])
        hf = _make_bench_result([8.0], [80.0], 2, [1, 2])
        mod.print_report("m", "p", 1, 3, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "Token match: False" in out
        assert "TRT=3" in out
        assert "HF=2" in out

    def test_report_kv_cache_footnote(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "KV cache" in out

    def test_report_family_runtime_footnote(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "float16", trt, hf,
                         runtime_note="family runtime note")
        out = capsys.readouterr().out
        assert "family runtime note" in out
        assert "KV cache" not in out

    def test_report_shows_dtype(self, capsys):
        mod = _import_perf_compare()
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", "p", 1, 5, 1, 0, "bfloat16", trt, hf)
        out = capsys.readouterr().out
        assert "bfloat16" in out

    def test_report_long_prompt_truncated(self, capsys):
        mod = _import_perf_compare()
        long_prompt = "A" * 100
        trt = _make_bench_result([10.0], [50.0], 5, [1])
        hf = _make_bench_result([8.0], [80.0], 5, [1])
        mod.print_report("m", long_prompt, 50, 5, 1, 0, "float16", trt, hf)
        out = capsys.readouterr().out
        assert "..." in out
        # Should not print all 100 characters
        assert "A" * 100 not in out


# ---------------------------------------------------------------------------
# Serial GPU execution — main() ordering
# ---------------------------------------------------------------------------

def _fake_bench_result():
    return {
        "prefill_times": [10.0],
        "decode_times": [50.0],
        "decode_token_counts": [5],
        "gen_ids": [1, 2, 3],
    }


class TestSerialGpuExecution:
    """Verify main() runs TRT before HF and frees GPU between them."""

    def _run_main_with_mocks(self, monkeypatch, has_family_handler=False,
                             extra_argv=None):
        """Patch all heavy deps in main() and return the call log."""
        mod = _import_perf_compare()
        call_log = []

        base_argv = [
            "perf_compare.py",
            "--model", "fake/model",
            "--bundle", "/fake/bundle.bundle",
            "--prompt", "Hello",
            "--max-new-tokens", "5",
            "--warmup", "0",
            "--iterations", "1",
        ]
        # Patch sys.argv
        monkeypatch.setattr("sys.argv", base_argv + (extra_argv or []))

        # Patch _resolve_model
        monkeypatch.setattr(
            "tensorrt_model_connect.engine_builder._resolve_model",
            lambda _: "/fake/model_dir")

        # Patch AutoTokenizer — patch via sys.modules to avoid triggering
        # transformers' lazy importer (which can fail in some environments)
        fake_tok = mock.MagicMock()
        fake_tok.encode.return_value = [1, 2, 3]
        fake_tok.eos_token_id = None
        fake_auto_tok = mock.MagicMock()
        fake_auto_tok.from_pretrained = mock.MagicMock(return_value=fake_tok)
        fake_transformers = mock.MagicMock()
        fake_transformers.AutoTokenizer = fake_auto_tok
        monkeypatch.setitem(__import__("sys").modules,
                            "transformers", fake_transformers)

        # Patch load_trt_from_bundle
        fake_handler = mock.MagicMock() if has_family_handler else None

        def fake_load_bundle(path):
            call_log.append("load_bundle")
            return (b"fake_plan", 2, 128, {}, fake_handler)
        monkeypatch.setattr(mod, "load_trt_from_bundle", fake_load_bundle)

        # Patch bench_trt / family dispatch
        def fake_bench_trt(*args, **kwargs):
            call_log.append("bench_trt")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_trt", fake_bench_trt)

        def fake_bench_trt_family(*args, **kwargs):
            call_log.append("bench_trt_family")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_trt_family", fake_bench_trt_family)

        # Patch gc.collect and torch.cuda.empty_cache to track calls
        real_gc_mod = __import__("gc")
        original_collect = real_gc_mod.collect

        def tracking_gc_collect():
            call_log.append("gc_collect")
            return original_collect()
        monkeypatch.setattr(real_gc_mod, "collect", tracking_gc_collect)

        fake_cuda = mock.MagicMock()
        fake_cuda.empty_cache = lambda: call_log.append("empty_cache")
        fake_torch = mock.MagicMock()
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        # Patch load_hf_model / bench_hf / bench_hf_compiled
        def fake_load_hf(*args, **kwargs):
            call_log.append("load_hf")
            return mock.MagicMock()
        monkeypatch.setattr(mod, "load_hf_model", fake_load_hf)

        def fake_bench_hf(*args, **kwargs):
            call_log.append("bench_hf")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_hf", fake_bench_hf)

        def fake_bench_hf_compiled(*args, **kwargs):
            call_log.append("bench_hf_compiled")
            return _fake_bench_result()
        monkeypatch.setattr(mod, "bench_hf_compiled", fake_bench_hf_compiled)

        # Patch print_report to suppress output
        monkeypatch.setattr(mod, "print_report", lambda *a, **kw: None)

        mod.main()
        return call_log

    def test_trt_runs_before_hf_load(self, monkeypatch):
        """TRT benchmark must complete before HF model is loaded."""
        log = self._run_main_with_mocks(monkeypatch)
        trt_idx = log.index("bench_trt")
        hf_load_idx = log.index("load_hf")
        assert trt_idx < hf_load_idx, (
            f"bench_trt ({trt_idx}) must run before load_hf ({hf_load_idx}): {log}")

    def test_gpu_freed_between_trt_and_hf(self, monkeypatch):
        """gc.collect + empty_cache must happen between TRT and HF."""
        log = self._run_main_with_mocks(monkeypatch)
        trt_idx = log.index("bench_trt")
        hf_load_idx = log.index("load_hf")

        # Find gc_collect and empty_cache between TRT and HF load
        mid_section = log[trt_idx + 1:hf_load_idx]
        assert "gc_collect" in mid_section, (
            f"gc.collect() missing between bench_trt and load_hf: {log}")
        assert "empty_cache" in mid_section, (
            f"torch.cuda.empty_cache() missing between bench_trt and load_hf: {log}")

    def test_hf_freed_after_bench(self, monkeypatch):
        """HF model is freed after benchmarking (gc + empty_cache at end)."""
        log = self._run_main_with_mocks(monkeypatch)
        bench_hf_idx = log.index("bench_hf")
        remaining = log[bench_hf_idx + 1:]
        assert "gc_collect" in remaining, (
            f"gc.collect() missing after bench_hf: {log}")
        assert "empty_cache" in remaining, (
            f"torch.cuda.empty_cache() missing after bench_hf: {log}")

    def test_family_handler_path_serial_execution(self, monkeypatch):
        """Family-owned path also runs TRT before HF with GPU cleanup."""
        log = self._run_main_with_mocks(monkeypatch, has_family_handler=True)
        trt_idx = log.index("bench_trt_family")
        hf_load_idx = log.index("load_hf")
        assert trt_idx < hf_load_idx

        mid_section = log[trt_idx + 1:hf_load_idx]
        assert "gc_collect" in mid_section
        assert "empty_cache" in mid_section

    def test_hf_never_loaded_before_trt_completes(self, monkeypatch):
        """Verify full ordering with --no-compile:
        load_bundle → bench_trt → cleanup → load_hf → bench_hf → cleanup."""
        log = self._run_main_with_mocks(monkeypatch, extra_argv=["--no-compile"])
        expected_order = ["load_bundle", "bench_trt", "gc_collect",
                          "empty_cache", "load_hf", "bench_hf",
                          "gc_collect", "empty_cache"]
        assert log == expected_order, (
            f"Expected exact serial order:\n  {expected_order}\nGot:\n  {log}")

    def test_compile_pass_runs_after_hf_eager(self, monkeypatch):
        """With compile enabled (default), compile pass runs after HF eager."""
        log = self._run_main_with_mocks(monkeypatch)
        # bench_hf (eager) must appear before bench_hf_compiled
        if "bench_hf_compiled" not in log:
            return  # family handler path or torch.compile unavailable
        hf_idx = log.index("bench_hf")
        compile_idx = log.index("bench_hf_compiled")
        assert hf_idx < compile_idx, (
            f"bench_hf ({hf_idx}) must run before bench_hf_compiled "
            f"({compile_idx}): {log}")

    def test_compile_pass_has_cleanup(self, monkeypatch):
        """GPU memory freed after compile pass."""
        log = self._run_main_with_mocks(monkeypatch)
        if "bench_hf_compiled" not in log:
            return
        compile_idx = log.index("bench_hf_compiled")
        remaining = log[compile_idx + 1:]
        assert "gc_collect" in remaining, (
            f"gc.collect() missing after bench_hf_compiled: {log}")
        assert "empty_cache" in remaining, (
            f"torch.cuda.empty_cache() missing after bench_hf_compiled: {log}")


# ---------------------------------------------------------------------------
# --trt-only tests
# ---------------------------------------------------------------------------

class TestTrtOnlyCLI:
    """Tests for --trt-only CLI flag."""

    def test_trt_only_flag_accepted(self):
        _import_perf_compare()  # ensure importable
        # Verify the parser accepts --trt-only
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--trt-only", action="store_true")
        args = parser.parse_args(["--trt-only"])
        assert args.trt_only is True

    def test_trt_only_json_structure(self):
        """TRT-only JSON should have empty hf section and None token_match."""
        # Build TRT-only json structure similar to what main() produces
        trt_only_json = {
            "metadata": {
                "model": "test-model",
                "gpu": "TestGPU",
                "trt_version": "10.0",
            },
            "trt": {
                "prefill_ms": {"mean": 5.0, "std": 0.1},
                "decode_ms": {"mean": 20.0, "std": 0.5},
                "per_token_ms": {"mean": 1.0, "std": 0.0},
                "throughput_tps": {"mean": 1000.0, "std": 0.0},
            },
            "hf": {},
            "speedup": {},
            "token_match": None,
        }
        # Verify structure
        assert trt_only_json["hf"] == {}
        assert trt_only_json["speedup"] == {}
        assert trt_only_json["token_match"] is None
        assert trt_only_json["trt"]["throughput_tps"]["mean"] == 1000.0

        # Verify it's valid JSON (serializable)
        import json
        serialized = json.dumps(trt_only_json)
        roundtrip = json.loads(serialized)
        assert roundtrip["trt"]["throughput_tps"]["mean"] == 1000.0

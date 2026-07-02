# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/profile.py — unified profiling entry point."""

from __future__ import annotations

import importlib


def _import():
    return importlib.import_module("trtmc_profile")


def _make_bench_result(prefill_ms, decode_ms, num_tokens, gen_ids):
    n = len(prefill_ms)
    return {
        "prefill_times": prefill_ms,
        "decode_times": decode_ms,
        "decode_token_counts": [num_tokens] * n,
        "gen_ids": gen_ids,
    }


# ---------------------------------------------------------------------------
# _stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty(self):
        mod = _import()
        r = mod._stats([])
        assert r["mean"] == 0.0
        assert r["std"] == 0.0

    def test_single_value(self):
        mod = _import()
        r = mod._stats([7.0])
        assert r["mean"] == 7.0
        assert r["std"] == 0.0

    def test_multiple_values(self):
        mod = _import()
        r = mod._stats([10.0, 20.0, 30.0])
        assert abs(r["mean"] - 20.0) < 0.01
        assert r["std"] > 0

    def test_identical_zero_std(self):
        mod = _import()
        r = mod._stats([5.0, 5.0, 5.0])
        assert r["std"] == 0.0


# ---------------------------------------------------------------------------
# _fmt()
# ---------------------------------------------------------------------------

class TestFmt:
    def test_basic(self):
        mod = _import()
        assert mod._fmt(1.5, 0.3) == "1.5 +/- 0.3"

    def test_zero(self):
        mod = _import()
        assert mod._fmt(0.0, 0.0) == "0.0 +/- 0.0"


# ---------------------------------------------------------------------------
# _speedup()
# ---------------------------------------------------------------------------

class TestSpeedup:
    def test_faster(self):
        mod = _import()
        # baseline=100, target=50 → 2.00x
        assert mod._speedup(100.0, 50.0) == "2.00x"

    def test_slower(self):
        mod = _import()
        # baseline=50, target=100 → 0.50x
        assert mod._speedup(50.0, 100.0) == "0.50x"

    def test_zero_target(self):
        mod = _import()
        assert mod._speedup(100.0, 0.0) == "N/A"


# ---------------------------------------------------------------------------
# print_combined_report()
# ---------------------------------------------------------------------------

class TestPrintCombinedReport:
    def _make_args(self):
        return dict(
            model_name="example-org/example-decoder",
            prompt="The capital of France is",
            num_input_tokens=6,
            max_new_tokens=20,
            warmup=2,
            iterations=5,
            hf_dtype="float16",
            gpu="H100",
            trt_ver="10.0",
            compile_mode="reduce-overhead",
        )

    def test_model_name_in_output(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0, 11.0], [40.0, 42.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 9.0], [80.0, 82.0], 20, [1, 2])
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert "example-org/example-decoder" in out

    def test_gpu_info_in_output(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert "H100" in out
        assert "10.0" in out

    def test_three_backends_shown_when_compile_present(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0, 10.0], [40.0, 40.0], 20, [1, 2])
        hf = _make_bench_result([8.0, 8.0], [80.0, 80.0], 20, [1, 2])
        cp = _make_bench_result([9.0, 9.0], [60.0, 60.0], 20, [1, 2])
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=cp, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert "TRT" in out
        assert "HF" in out
        # compile label should appear
        assert "reduce-overhead" in out or "compile" in out.lower()

    def test_token_match_reported(self, capsys):
        mod = _import()
        ids = [10, 20, 30]
        trt = _make_bench_result([10.0], [40.0], 3, ids)
        hf = _make_bench_result([8.0], [80.0], 3, ids)
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert "True" in out or "match" in out.lower()

    def test_layer_data_shows_bottleneck(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        layer_data = {
            "layers": [
                {"name": "MatMul_attention", "mean_ms": 3.0, "std_ms": 0.1,
                 "pct": 60.0, "calls": 5},
                {"name": "Softmax_0", "mean_ms": 1.0, "std_ms": 0.05,
                 "pct": 20.0, "calls": 5},
                {"name": "ElementWise_add", "mean_ms": 1.0, "std_ms": 0.05,
                 "pct": 20.0, "calls": 5},
            ],
            "total_ms": 5.0,
        }
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=layer_data,
            **self._make_args())
        out = capsys.readouterr().out
        assert "MatMul_attention" in out
        assert "Bottleneck" in out

    def test_no_layer_data_skips_layer_section(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert "Per-Layer" not in out

    def test_output_is_string(self, capsys):
        mod = _import()
        trt = _make_bench_result([10.0], [40.0], 20, [1])
        hf = _make_bench_result([8.0], [80.0], 20, [1])
        mod.print_combined_report(
            trt_res=trt, hf_res=hf, compile_res=None, layer_data=None,
            **self._make_args())
        out = capsys.readouterr().out
        assert isinstance(out, str)
        assert len(out) > 0

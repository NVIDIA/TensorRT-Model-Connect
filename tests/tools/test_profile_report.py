# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/profile_report.py — HTML report generator."""

from __future__ import annotations

import importlib
import json


def _import():
    return importlib.import_module("profile_report")


def _make_perf_compare(with_compile: bool = False) -> dict:
    base = {
        "metadata": {
            "model": "example-org/example-decoder",
            "gpu": "NVIDIA H100",
            "hf_dtype": "float16",
            "iterations": 5,
            "warmup": 2,
            "timestamp": "2026-03-20T00:00:00Z",
        },
        "trt": {
            "prefill_ms": {"mean": 10.0, "std": 0.5},
            "decode_ms": {"mean": 40.0, "std": 1.0},
            "per_token_ms": {"mean": 2.0, "std": 0.1},
            "throughput_tps": {"mean": 500.0, "std": 10.0},
            "total_ms": {"mean": 50.0, "std": 1.5},
        },
        "hf": {
            "prefill_ms": {"mean": 5.0, "std": 0.3},
            "decode_ms": {"mean": 80.0, "std": 2.0},
            "per_token_ms": {"mean": 4.0, "std": 0.2},
            "throughput_tps": {"mean": 250.0, "std": 5.0},
            "total_ms": {"mean": 85.0, "std": 2.5},
        },
        "speedup": {
            "prefill": 0.5,
            "decode": 2.0,
            "total": 1.7,
        },
        "token_match": True,
    }
    if with_compile:
        base["hf_compiled"] = {
            "compile_mode": "reduce-overhead",
            "prefill_ms": {"mean": 6.0, "std": 0.4},
            "decode_ms": {"mean": 65.0, "std": 1.5},
            "per_token_ms": {"mean": 3.25, "std": 0.15},
            "throughput_tps": {"mean": 307.0, "std": 7.0},
            "total_ms": {"mean": 71.0, "std": 2.0},
        }
        base["speedup"]["trt_vs_compile_decode"] = 1.625
        base["speedup"]["trt_vs_compile_prefill"] = 0.6
        base["speedup"]["trt_vs_compile_total"] = 1.42
    return base


def _make_layer_profile() -> dict:
    return {
        "metadata": {
            "model": "example-org/example-decoder",
            "gpu": "NVIDIA H100",
            "timestamp": "2026-03-20T00:00:00Z",
        },
        "total_ms": 5.5,
        "layers": [
            {"name": "MatMul_qkv_0", "mean_ms": 1.2, "std_ms": 0.05,
             "pct": 21.8, "calls": 50},
            {"name": "Softmax_attn_0", "mean_ms": 0.8, "std_ms": 0.03,
             "pct": 14.5, "calls": 50},
            {"name": "ElementWise_ffn_0", "mean_ms": 0.7, "std_ms": 0.04,
             "pct": 12.7, "calls": 50},
            {"name": "RMSNorm_0", "mean_ms": 0.4, "std_ms": 0.02,
             "pct": 7.3, "calls": 50},
        ],
    }


def _make_cpu_profile() -> dict:
    return {
        "metadata": {
            "model": "example-org/example-decoder",
            "gpu": "NVIDIA H100",
            "num_layers": 28,
            "timestamp": "2026-03-20T00:00:00Z",
        },
        "total_ms": 6.1,
        "bottleneck": "execute",
        "phases": [
            {"phase": "mask_build", "mean_ms": 0.05, "std_ms": 0.01, "pct": 0.8, "samples": 20},
            {"phase": "h2d", "mean_ms": 0.1, "std_ms": 0.02, "pct": 1.6, "samples": 20},
            {"phase": "tensor_bind", "mean_ms": 0.3, "std_ms": 0.05, "pct": 4.9, "samples": 20},
            {"phase": "execute", "mean_ms": 5.0, "std_ms": 0.2, "pct": 82.0, "samples": 20},
            {"phase": "d2d_cache", "mean_ms": 0.4, "std_ms": 0.05, "pct": 6.6, "samples": 20},
            {"phase": "d2h", "mean_ms": 0.2, "std_ms": 0.03, "pct": 3.3, "samples": 20},
            {"phase": "argmax", "mean_ms": 0.05, "std_ms": 0.01, "pct": 0.8, "samples": 20},
        ],
    }


def _make_nsight(backend: str = "trt") -> dict:
    return {
        "tool": "nsys",
        "backend": backend,
        "metadata": {
            "model": "example-org/example-decoder",
            "gpu": "NVIDIA H100",
            "tool": "nsys",
            "timestamp": "2026-03-20T00:00:00Z",
        },
        "top_kernels": [
            {"name": "volta_fp16_s884gemm", "total_ms": 3.2, "calls": 48,
             "avg_us": 66.7, "pct": 55.0},
            {"name": "softmax_fwd_kernel", "total_ms": 1.1, "calls": 48,
             "avg_us": 22.9, "pct": 18.9},
        ],
        "total_kernel_ms": 5.8,
        "cuda_api_summary": [],
        "gpu_utilization_pct": 87.3,
    }


# ---------------------------------------------------------------------------
# _layer_color
# ---------------------------------------------------------------------------

class TestLayerColor:
    def test_matmul_returns_blue(self):
        mod = _import()
        color = mod._layer_color("MatMul_qkv_0")
        assert color != mod._DEFAULT_COLOR

    def test_softmax_returns_non_default(self):
        mod = _import()
        color = mod._layer_color("Softmax_attn")
        assert color != mod._DEFAULT_COLOR

    def test_unknown_returns_default(self):
        mod = _import()
        color = mod._layer_color("SomeUnknownOp_xyz")
        assert color == mod._DEFAULT_COLOR

    def test_case_insensitive(self):
        mod = _import()
        c1 = mod._layer_color("matmul_0")
        c2 = mod._layer_color("MATMUL_0")
        assert c1 == c2

    def test_returns_string(self):
        mod = _import()
        assert isinstance(mod._layer_color("MatMul_0"), str)


# ---------------------------------------------------------------------------
# build_report() — HTML generation
# ---------------------------------------------------------------------------

class TestBuildReport:
    def test_returns_string(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None,
            cpu_profile=None,
            nsight_trt=None,
            nsight_hf=None,
        )
        assert isinstance(html, str)

    def test_is_valid_html(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=_make_layer_profile(),
            cpu_profile=_make_cpu_profile(),
            nsight_trt=_make_nsight("trt"),
            nsight_hf=_make_nsight("hf"),
        )
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_model_name_in_title(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "example-org/example-decoder" in html

    def test_gpu_in_output(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "H100" in html

    def test_speedup_section_present(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "speedup" in html.lower() or "Speedup" in html

    def test_layer_section_present_when_data_given(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=None,
            layer_profile=_make_layer_profile(),
            cpu_profile=None, nsight_trt=None, nsight_hf=None,
        )
        assert "MatMul_qkv_0" in html
        assert "IProfiler" in html

    def test_cpu_phase_section_present(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=None, layer_profile=None,
            cpu_profile=_make_cpu_profile(),
            nsight_trt=None, nsight_hf=None,
        )
        assert "execute" in html
        assert "mask_build" in html

    def test_nsight_section_present(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=None, layer_profile=None, cpu_profile=None,
            nsight_trt=_make_nsight("trt"),
            nsight_hf=None,
        )
        assert "volta_fp16_s884gemm" in html

    def test_no_data_shows_message(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=None, layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        # Should show some "no data" message
        assert "No profiling data" in html or "no-data" in html

    def test_three_way_compile_shown(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(with_compile=True),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "reduce-overhead" in html or "compile" in html.lower()

    def test_token_match_badge(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "token" in html.lower() and "match" in html.lower()

    def test_embedded_json_parseable(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=_make_layer_profile(),
            cpu_profile=_make_cpu_profile(),
            nsight_trt=None, nsight_hf=None,
        )
        # Extract the embedded JSON
        start = html.find("const PROFILE_DATA = ") + len("const PROFILE_DATA = ")
        end = html.find(";\n</script>", start)
        raw_json = html[start:end]
        parsed = json.loads(raw_json)
        assert "perf_compare" in parsed
        assert "layer_profile" in parsed

    def test_chartjs_cdn_referenced(self):
        mod = _import()
        html = mod.build_report(
            perf_compare=_make_perf_compare(),
            layer_profile=None, cpu_profile=None,
            nsight_trt=None, nsight_hf=None,
        )
        assert "chart.js" in html.lower() or "cdn.jsdelivr" in html


# ---------------------------------------------------------------------------
# _load_json helper
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_returns_none_for_none_path(self):
        mod = _import()
        assert mod._load_json(None) is None

    def test_returns_none_for_missing_file(self):
        mod = _import()
        result = mod._load_json("/nonexistent/path/file.json")
        assert result is None

    def test_loads_valid_json(self):
        import tempfile
        import os
        mod = _import()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False) as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            result = mod._load_json(path)
            assert result == {"key": "value"}
        finally:
            os.unlink(path)

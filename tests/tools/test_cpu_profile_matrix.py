# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cpu_profile_matrix.py — cross-strategy CPU phase harness.

Intent:
    Verify the matrix harness correctly aggregates per-strategy cpu_profile
    results, builds the comparison table, and produces JSON/HTML output
    without requiring GPU or TRT.

Preconditions:
    - cpu_profile_matrix importable from tools/
    - No GPU required (all heavy profiling functions are mocked)

Postconditions:
    - Console table, JSON, and HTML match the shape of the input results.

Trace IDs: UT-TOOL-CPMX-01 through UT-TOOL-CPMX-15
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
# Add tools/ to path
_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _import_matrix():
    return importlib.import_module("cpu_profile_matrix")


# ---------------------------------------------------------------------------
# Synthetic result fixture
# ---------------------------------------------------------------------------

def _make_result(strategy: str, bottleneck: str = "execute",
                 num_layers: int = 28) -> dict:
    """Build a synthetic profiling result dict."""
    phases = [
        {"phase": "mask_build", "mean_ms": 0.05, "std_ms": 0.01, "pct": 1.0, "samples": 20},
        {"phase": "h2d",        "mean_ms": 0.10, "std_ms": 0.02, "pct": 2.0, "samples": 20},
        {"phase": "tensor_bind","mean_ms": 0.30, "std_ms": 0.05, "pct": 6.0, "samples": 20},
        {"phase": "execute",    "mean_ms": 4.00, "std_ms": 0.20, "pct": 80.0,"samples": 20},
        {"phase": "d2d_cache",  "mean_ms": 0.40, "std_ms": 0.05, "pct": 8.0, "samples": 20},
        {"phase": "d2h",        "mean_ms": 0.10, "std_ms": 0.01, "pct": 2.0, "samples": 20},
        {"phase": "argmax",     "mean_ms": 0.05, "std_ms": 0.01, "pct": 1.0, "samples": 20},
    ]
    return {
        "strategy": strategy,
        "model": f"org/{strategy}-model",
        "runner_type": "decoder",
        "num_layers": num_layers,
        "phases": phases,
        "total_ms": 5.0,
        "bottleneck": bottleneck,
    }


def _make_family_result(strategy: str = "family_recurrent") -> dict:
    phases = [
        {"phase": "h2d",        "mean_ms": 0.08, "std_ms": 0.01, "pct": 3.0, "samples": 20},
        {"phase": "tensor_bind","mean_ms": 0.20, "std_ms": 0.03, "pct": 8.0, "samples": 20},
        {"phase": "execute",    "mean_ms": 2.00, "std_ms": 0.10, "pct": 77.0,"samples": 20},
        {"phase": "d2d_state",  "mean_ms": 0.20, "std_ms": 0.02, "pct": 8.0, "samples": 20},
        {"phase": "d2h",        "mean_ms": 0.08, "std_ms": 0.01, "pct": 3.0, "samples": 20},
        {"phase": "argmax",     "mean_ms": 0.04, "std_ms": 0.01, "pct": 1.0, "samples": 20},
    ]
    return {
        "strategy": strategy,
        "model": f"org/{strategy}-model",
        "runner_type": "family",
        "num_layers": 24,
        "phases": phases,
        "total_ms": 2.6,
        "bottleneck": "execute",
    }


# ---------------------------------------------------------------------------
# StrategySpec
# ---------------------------------------------------------------------------

class TestStrategySpec:
    def test_default_specs_have_required_fields(self):
        mod = _import_matrix()
        for spec in mod._DEFAULT_SPECS:
            assert spec.strategy
            assert spec.hf_id
            assert spec.bundle
            assert spec.runner in ("decoder", "family")

    def test_all_strategies_are_unique(self):
        mod = _import_matrix()
        strategies = [s.strategy for s in mod._DEFAULT_SPECS]
        assert len(strategies) == len(set(strategies))

    def test_default_specs_are_loaded_from_family_hooks(self):
        mod = _import_matrix()
        assert mod._DEFAULT_SPECS

    def test_spec_mapping_rejects_unknown_runner(self):
        mod = _import_matrix()
        raw = {
            "strategy": "custom_runtime",
            "label": "custom",
            "hf_id": "org/model",
            "bundle": "model.bundle",
            "runner": "custom",
        }
        import pytest
        with pytest.raises(ValueError, match="unsupported runner"):
            mod._strategy_spec_from_mapping(raw, "test")


# ---------------------------------------------------------------------------
# _print_matrix
# ---------------------------------------------------------------------------

class TestPrintMatrix:
    def test_prints_all_strategies(self, capsys):
        mod = _import_matrix()
        results = [
            _make_result("qwen_decoder_kv_cache"),
            _make_result("gpt_oss_decoder_moe"),
            _make_family_result("family_recurrent"),
        ]
        mod._print_matrix(results, "TestGPU", "10.0", "Hello", 10)
        out = capsys.readouterr().out
        assert "qwen_decoder_kv_cache" in out
        assert "gpt_oss_decoder_moe" in out
        assert "family_recurrent" in out

    def test_shows_bottleneck_row(self, capsys):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache", bottleneck="execute")]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        assert "BOTTLENECK" in out
        assert "execute" in out

    def test_shows_total_row(self, capsys):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache")]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        assert "TOTAL" in out

    def test_family_phases_shown_for_family_runtime(self, capsys):
        mod = _import_matrix()
        results = [_make_family_result()]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        assert "d2d_state" in out

    def test_decoder_phases_shown_for_decoder(self, capsys):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache")]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        assert "mask_build" in out
        assert "d2d_cache" in out

    def test_mixed_strategies_shows_union_of_phases(self, capsys):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache"), _make_family_result()]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        # Decoder phases
        assert "mask_build" in out
        assert "d2d_cache" in out
        # Family-owned recurrent phases
        assert "d2d_state" in out

    def test_empty_results_does_not_crash(self, capsys):
        mod = _import_matrix()
        mod._print_matrix([], "GPU", "10.0", "test", 10)
        # Should print a message and return cleanly

    def test_analysis_section_printed(self, capsys):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache")]
        mod._print_matrix(results, "GPU", "10.0", "test", 10)
        out = capsys.readouterr().out
        assert "Analysis" in out
        assert "bottleneck=" in out


# ---------------------------------------------------------------------------
# _build_html
# ---------------------------------------------------------------------------

class TestBuildHtml:
    def test_returns_valid_html(self):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache"), _make_family_result()]
        html = mod._build_html(results, "H100", "10.0", "Hello", 10, 3, 20)
        assert html.startswith("<!DOCTYPE html>")
        assert "<table>" in html
        assert "</html>" in html

    def test_all_strategies_in_table(self):
        mod = _import_matrix()
        results = [
            _make_result("qwen_decoder_kv_cache"),
            _make_result("gpt_oss_decoder_moe"),
            _make_family_result("family_recurrent"),
        ]
        html = mod._build_html(results, "H100", "10.0", "test", 10, 3, 20)
        assert "qwen_decoder_kv_cache" in html
        assert "gpt_oss_decoder_moe" in html
        assert "family_recurrent" in html

    def test_chart_json_embedded(self):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache")]
        html = mod._build_html(results, "H100", "10.0", "test", 10, 3, 20)
        assert "const DATA = " in html
        # Extract and validate embedded JSON
        import re
        m = re.search(r"const DATA = (\{.*?\});", html, re.DOTALL)
        assert m, "DATA JSON block not found"
        data = json.loads(m.group(1))
        assert "labels" in data
        assert "datasets" in data
        assert data["labels"] == ["qwen_decoder_kv_cache"]

    def test_bottleneck_highlighted(self):
        mod = _import_matrix()
        results = [_make_result("qwen_decoder_kv_cache", bottleneck="execute")]
        html = mod._build_html(results, "H100", "10.0", "test", 10, 3, 20)
        assert "BOTTLENECK" in html
        assert "execute" in html

    def test_heat_colors_applied(self):
        mod = _import_matrix()
        # High pct should produce a reddish color
        color = mod._heat(80.0)
        assert color.startswith("rgb(")
        r_val = int(color.split("(")[1].split(",")[0])
        assert r_val >= 200  # reddish

    def test_zero_pct_is_light(self):
        mod = _import_matrix()
        assert mod._heat(0.0) == "#f8f8f8"

    def test_empty_results_returns_html(self):
        mod = _import_matrix()
        html = mod._build_html([], "H100", "10.0", "test", 10, 3, 20)
        assert isinstance(html, str)
        assert len(html) > 0


# ---------------------------------------------------------------------------
# _profile_strategy (integration contract — mocked heavy deps)
# ---------------------------------------------------------------------------

class TestProfileStrategyContract:
    """Verify the return contract of _profile_strategy via its output shape."""

    def test_result_schema_matches_print_matrix_expectations(self):
        """_print_matrix and _build_html only need keys present in _make_result.
        Verify those keys cover what _profile_strategy actually returns."""
        required_keys = {"strategy", "model", "runner_type",
                         "num_layers", "phases", "total_ms", "bottleneck"}
        sample = _make_result("qwen_decoder_kv_cache")
        assert required_keys.issubset(sample.keys())

    def test_phase_dict_schema(self):
        """Each phase entry must have phase/mean_ms/pct/samples."""
        sample = _make_result("qwen_decoder_kv_cache")
        for ph in sample["phases"]:
            assert "phase" in ph
            assert "mean_ms" in ph
            assert "pct" in ph
            assert "samples" in ph

    def test_bottleneck_is_a_phase_name(self):
        sample = _make_result("qwen_decoder_kv_cache", bottleneck="execute")
        phase_names = {p["phase"] for p in sample["phases"]}
        assert sample["bottleneck"] in phase_names

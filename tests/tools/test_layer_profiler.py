"""Unit tests for tools/layer_profiler.py — LayerProfiler (TRT IProfiler)."""

from __future__ import annotations

import importlib


def _import():
    return importlib.import_module("layer_profiler")


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------

class TestLayerProfilerAccumulation:
    def test_single_layer_single_call(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("MatMul_0", 1.5)
        data = p.to_dict()
        assert len(data["layers"]) == 1
        assert data["layers"][0]["name"] == "MatMul_0"
        assert data["layers"][0]["mean_ms"] == 1.5
        assert data["layers"][0]["calls"] == 1

    def test_single_layer_multiple_calls(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("MatMul_0", 1.0)
        p.report_layer_time("MatMul_0", 3.0)
        data = p.to_dict()
        assert len(data["layers"]) == 1
        layer = data["layers"][0]
        assert layer["mean_ms"] == 2.0
        assert layer["calls"] == 2
        assert layer["std_ms"] > 0

    def test_multiple_layers(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("LayerA", 2.0)
        p.report_layer_time("LayerB", 1.0)
        p.report_layer_time("LayerC", 3.0)
        data = p.to_dict()
        names = [l["name"] for l in data["layers"]]
        assert set(names) == {"LayerA", "LayerB", "LayerC"}

    def test_accumulates_across_calls(self):
        mod = _import()
        p = mod.LayerProfiler()
        for _ in range(5):
            p.report_layer_time("LayerX", 2.0)
        data = p.to_dict()
        assert data["layers"][0]["calls"] == 5
        assert data["layers"][0]["mean_ms"] == 2.0


# ---------------------------------------------------------------------------
# Sorting (slowest first)
# ---------------------------------------------------------------------------

class TestLayerProfilerSorting:
    def test_sorted_by_mean_ms_descending(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("Fast", 0.5)
        p.report_layer_time("Slow", 5.0)
        p.report_layer_time("Medium", 2.0)
        data = p.to_dict()
        means = [l["mean_ms"] for l in data["layers"]]
        assert means == sorted(means, reverse=True)

    def test_top_layer_is_bottleneck(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("Bottleneck", 10.0)
        p.report_layer_time("Other", 1.0)
        data = p.to_dict()
        assert data["layers"][0]["name"] == "Bottleneck"


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------

class TestLayerProfilerPercentages:
    def test_percentages_sum_to_100(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 1.0)
        p.report_layer_time("B", 2.0)
        p.report_layer_time("C", 7.0)
        data = p.to_dict()
        total_pct = sum(l["pct"] for l in data["layers"])
        assert abs(total_pct - 100.0) < 0.1

    def test_single_layer_100_pct(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("Only", 5.0)
        data = p.to_dict()
        assert data["layers"][0]["pct"] == 100.0

    def test_equal_layers_equal_pct(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 2.0)
        p.report_layer_time("B", 2.0)
        data = p.to_dict()
        pcts = [l["pct"] for l in data["layers"]]
        assert abs(pcts[0] - 50.0) < 0.1
        assert abs(pcts[1] - 50.0) < 0.1


# ---------------------------------------------------------------------------
# total_ms
# ---------------------------------------------------------------------------

class TestLayerProfilerTotalMs:
    def test_total_ms_is_sum_of_means(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 3.0)
        p.report_layer_time("B", 7.0)
        data = p.to_dict()
        assert abs(data["total_ms"] - 10.0) < 0.01

    def test_total_ms_with_multiple_calls(self):
        mod = _import()
        p = mod.LayerProfiler()
        # 2 calls: mean = 2.0
        p.report_layer_time("A", 1.0)
        p.report_layer_time("A", 3.0)
        # 3 calls: mean = 4.0
        p.report_layer_time("B", 3.0)
        p.report_layer_time("B", 4.0)
        p.report_layer_time("B", 5.0)
        data = p.to_dict()
        assert abs(data["total_ms"] - 6.0) < 0.01  # 2.0 + 4.0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestLayerProfilerReset:
    def test_reset_clears_all_records(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 1.0)
        p.report_layer_time("B", 2.0)
        p.reset()
        data = p.to_dict()
        assert data["layers"] == []
        assert data["total_ms"] == 0.0

    def test_accumulates_after_reset(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 1.0)
        p.reset()
        p.report_layer_time("B", 5.0)
        data = p.to_dict()
        assert len(data["layers"]) == 1
        assert data["layers"][0]["name"] == "B"


# ---------------------------------------------------------------------------
# Empty case
# ---------------------------------------------------------------------------

class TestLayerProfilerEmpty:
    def test_empty_to_dict(self):
        mod = _import()
        p = mod.LayerProfiler()
        data = p.to_dict()
        assert data["layers"] == []
        assert data["total_ms"] == 0.0

    def test_empty_with_metadata(self):
        mod = _import()
        p = mod.LayerProfiler()
        data = p.to_dict(metadata={"model": "test"})
        assert data["metadata"]["model"] == "test"
        assert data["layers"] == []


# ---------------------------------------------------------------------------
# metadata passthrough
# ---------------------------------------------------------------------------

class TestLayerProfilerMetadata:
    def test_metadata_passed_through(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 1.0)
        meta = {"model": "example-org/example-decoder", "gpu": "H100"}
        data = p.to_dict(metadata=meta)
        assert data["metadata"]["model"] == "example-org/example-decoder"
        assert data["metadata"]["gpu"] == "H100"

    def test_no_metadata_defaults_empty(self):
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("A", 1.0)
        data = p.to_dict()
        assert data["metadata"] == {}


# ---------------------------------------------------------------------------
# JSON serializable
# ---------------------------------------------------------------------------

class TestLayerProfilerSerializable:
    def test_to_dict_is_json_serializable(self):
        import json
        mod = _import()
        p = mod.LayerProfiler()
        p.report_layer_time("MatMul", 1.234)
        p.report_layer_time("Softmax", 0.5)
        data = p.to_dict(metadata={"model": "test"})
        # Should not raise
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert len(parsed["layers"]) == 2

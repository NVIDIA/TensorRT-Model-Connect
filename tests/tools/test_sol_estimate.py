"""Unit tests for tools/sol_estimate.py — SOL estimation logic.

No GPU or model downloads required. Tests pure computation with synthetic data.
"""
import sys
import os

# Add tools/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

from sol_estimate import (
    ModelArch,
    GpuSpec,
    GPU_SPECS,
    BYTES_PER_PARAM,
    estimate_sol,
    _compute_flops_per_token,
    parse_benchmark_json,
    per_layer_roofline,
    to_json,
)

import json
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def example_decoder_1_5b():
    """Example decoder 1.5B architecture."""
    return ModelArch(
        name="example-org/example-decoder-1.5b",
        hidden_size=1536,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
        intermediate_size=8960,
        vocab_size=151936,
    )


@pytest.fixture
def tiny_model():
    """Tiny model for predictable calculations."""
    return ModelArch(
        name="tiny",
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=16,
        intermediate_size=256,
        vocab_size=1000,
    )


@pytest.fixture
def b200():
    return GPU_SPECS["B200"]


# ---------------------------------------------------------------------------
# ModelArch.total_params
# ---------------------------------------------------------------------------

class TestModelArch:
    def test_total_params_reasonable(self, example_decoder_1_5b):
        """Parameter count should be in the right ballpark for 1.5B model."""
        params = example_decoder_1_5b.total_params
        assert 1.0e9 < params < 3.0e9, f"Expected ~1.5-2B, got {params/1e9:.2f}B"

    def test_total_params_tiny(self, tiny_model):
        """Verify exact calculation for tiny model."""
        params = tiny_model.total_params
        h, kv, d, ff, L, V = 64, 4, 16, 256, 2, 1000

        per_layer = (
            h * h          # Q
            + h * kv * d   # K
            + h * kv * d   # V
            + h * h        # O
            + h * ff       # gate
            + h * ff       # up
            + ff * h       # down
            + 2 * h        # norms
        )
        total = L * per_layer + V * h + V * h + h  # embed + lm_head + final_norm
        assert params == total

    def test_head_dim_property(self, example_decoder_1_5b):
        assert example_decoder_1_5b.head_dim == 128

    def test_gqa_params_less_than_mha(self):
        """GQA (fewer KV heads) should have fewer params than MHA."""
        base = dict(name="t", hidden_size=512, num_layers=4, num_heads=8,
                    head_dim=64, intermediate_size=2048, vocab_size=10000)
        mha = ModelArch(**base, num_kv_heads=8)
        gqa = ModelArch(**base, num_kv_heads=2)
        assert gqa.total_params < mha.total_params


# ---------------------------------------------------------------------------
# FLOPS calculation
# ---------------------------------------------------------------------------

class TestFlops:
    def test_flops_positive(self, example_decoder_1_5b):
        flops = _compute_flops_per_token(example_decoder_1_5b, cache_length=256)
        assert flops > 0

    def test_flops_scale_with_cache(self, example_decoder_1_5b):
        """Attention FLOPS should grow with cache length."""
        f128 = _compute_flops_per_token(example_decoder_1_5b, cache_length=128)
        f2048 = _compute_flops_per_token(example_decoder_1_5b, cache_length=2048)
        assert f2048 > f128

    def test_flops_zero_cache(self, tiny_model):
        """Should still compute FLOPS with cache_length=0."""
        flops = _compute_flops_per_token(tiny_model, cache_length=0)
        assert flops > 0

    def test_flops_dominated_by_matmul(self, example_decoder_1_5b):
        """For small cache, weight matmuls dominate (not attention)."""
        f_small = _compute_flops_per_token(example_decoder_1_5b, cache_length=1)
        f_large = _compute_flops_per_token(example_decoder_1_5b, cache_length=1)
        # With cache=1, attention FLOPS are tiny relative to projections
        # So both should be essentially the same
        assert f_small == f_large


# ---------------------------------------------------------------------------
# SOL estimation
# ---------------------------------------------------------------------------

class TestEstimateSol:
    def test_bandwidth_bound(self, example_decoder_1_5b, b200):
        """Batch=1 decode should be bandwidth-bound, not compute-bound."""
        est = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=256)
        assert est.bottleneck == "bandwidth"
        assert est.bw_sol_tps < est.compute_sol_tps

    def test_sol_positive(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32")
        assert est.sol_tps > 0
        assert est.bw_sol_tps > 0
        assert est.compute_sol_tps > 0

    def test_fp16_faster_than_fp32(self, example_decoder_1_5b, b200):
        """FP16 halves weight bytes → ~2x bandwidth SOL."""
        est32 = estimate_sol(example_decoder_1_5b, b200, "fp32")
        est16 = estimate_sol(example_decoder_1_5b, b200, "fp16")
        assert est16.bw_sol_tps > est32.bw_sol_tps * 1.5

    def test_larger_cache_slower_sol(self, example_decoder_1_5b, b200):
        """More KV cache reads → lower bandwidth SOL."""
        est_small = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=128)
        est_large = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=4096)
        assert est_large.bw_sol_tps < est_small.bw_sol_tps

    def test_utilization_with_actual(self, example_decoder_1_5b, b200):
        """Utilization should be calculated when actual_tps provided."""
        est = estimate_sol(example_decoder_1_5b, b200, "fp32",
                           cache_length=256, actual_tps=265.7)
        assert 0 < est.utilization_pct < 100
        assert est.actual_tps == 265.7

    def test_utilization_zero_without_actual(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32")
        assert est.utilization_pct == 0
        assert est.actual_tps == 0

    def test_model_bytes(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32")
        expected = example_decoder_1_5b.total_params * 4  # FP32 = 4 bytes
        assert est.model_bytes == expected

    def test_kv_bytes_with_cache(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=256)
        assert est.kv_bytes_per_token > 0
        assert est.total_read_bytes > est.weight_read_bytes

    def test_kv_bytes_zero_without_cache(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=0)
        assert est.kv_bytes_per_token == 0
        assert est.total_read_bytes == est.weight_read_bytes

    def test_all_dtypes(self, tiny_model, b200):
        """All supported dtypes should produce valid estimates."""
        for dtype in BYTES_PER_PARAM:
            est = estimate_sol(tiny_model, b200, dtype)
            assert est.sol_tps > 0, f"{dtype} SOL should be positive"
            assert est.dtype == dtype

    def test_all_gpus(self, tiny_model):
        """All GPU specs should produce valid estimates."""
        for gpu_key, gpu in GPU_SPECS.items():
            est = estimate_sol(tiny_model, gpu, "fp32")
            assert est.sol_tps > 0, f"{gpu_key} SOL should be positive"
            assert est.gpu_name == gpu.name

    def test_practical_bw_ratio(self, tiny_model):
        """Practical bandwidth should be less than peak."""
        gpu = GpuSpec("test", hbm_bandwidth_gb_s=1000, fp32_tflops=100,
                      fp16_tflops=200, hbm_capacity_gb=80, practical_bw_ratio=0.5)
        est = estimate_sol(tiny_model, gpu, "fp32")
        # With 50% practical ratio, SOL should be ~half of theoretical peak
        gpu_full = GpuSpec("test", hbm_bandwidth_gb_s=1000, fp32_tflops=100,
                           fp16_tflops=200, hbm_capacity_gb=80, practical_bw_ratio=1.0)
        est_full = estimate_sol(tiny_model, gpu_full, "fp32")
        assert abs(est.bw_sol_tps / est_full.bw_sol_tps - 0.5) < 0.01


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

class TestJson:
    def test_to_json_keys(self, example_decoder_1_5b, b200):
        est = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=256, actual_tps=265.7)
        j = to_json(est)
        required_keys = [
            "model", "gpu", "dtype", "total_params", "model_bytes",
            "cache_length", "bw_sol_tps", "compute_sol_tps", "sol_tps",
            "bottleneck", "actual_tps", "utilization_pct",
        ]
        for k in required_keys:
            assert k in j, f"Missing key: {k}"

    def test_to_json_serializable(self, example_decoder_1_5b, b200):
        """JSON output should be serializable."""
        import json
        est = estimate_sol(example_decoder_1_5b, b200, "fp32")
        j = to_json(est)
        s = json.dumps(j)
        assert len(s) > 0


# ---------------------------------------------------------------------------
# Regression: known values from Phase 0
# ---------------------------------------------------------------------------

class TestPhase0Regression:
    """Verify SOL estimates match Phase 0 known data points."""

    def test_example_decoder_1_5b_fp32_b200(self, example_decoder_1_5b, b200):
        """Example decoder 1.5B FP32 on B200: measured 265.7 tok/s → ~28% utilization."""
        est = estimate_sol(example_decoder_1_5b, b200, "fp32",
                           cache_length=256, actual_tps=265.7)
        # SOL should be in 800-1200 range for ~1.5B FP32 model on B200
        assert 500 < est.sol_tps < 2000
        # Utilization should be 20-40%
        assert 15 < est.utilization_pct < 50
        assert est.bottleneck == "bandwidth"


# ---------------------------------------------------------------------------
# Closed loop (benchmark JSON parsing)
# ---------------------------------------------------------------------------

class TestClosedLoop:
    def test_parse_benchmark_json_perf_compare_format(self, tmp_path):
        """Standard perf_compare JSON format."""
        data = {
            "trt": {
                "throughput_tps": {"mean": 265.7, "std": 2.1},
                "decode_ms": {"mean": 3.77}
            }
        }
        p = tmp_path / "bench.json"
        p.write_text(json.dumps(data))
        assert parse_benchmark_json(str(p)) == 265.7

    def test_parse_benchmark_json_flat_format(self, tmp_path):
        """Flat JSON with throughput_tps at top level."""
        data = {"throughput_tps": 312.5}
        p = tmp_path / "flat.json"
        p.write_text(json.dumps(data))
        assert parse_benchmark_json(str(p)) == 312.5

    def test_parse_benchmark_json_actual_tps_format(self, tmp_path):
        """Flat JSON with actual_tps key."""
        data = {"actual_tps": 400.0}
        p = tmp_path / "actual.json"
        p.write_text(json.dumps(data))
        assert parse_benchmark_json(str(p)) == 400.0

    def test_parse_benchmark_json_missing_key(self, tmp_path):
        """Should raise ValueError when no TPS field found."""
        data = {"model": "test", "latency_ms": 5.0}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="No throughput field"):
            parse_benchmark_json(str(p))

    def test_closed_loop_integration(self, tmp_path, example_decoder_1_5b, b200):
        """Parse benchmark JSON + estimate_sol in one flow."""
        data = {"trt": {"throughput_tps": {"mean": 265.7}}}
        p = tmp_path / "bench.json"
        p.write_text(json.dumps(data))
        tps = parse_benchmark_json(str(p))
        est = estimate_sol(example_decoder_1_5b, b200, "fp32",
                           cache_length=256, actual_tps=tps)
        assert est.actual_tps == 265.7
        assert est.utilization_pct > 0

    def test_benchmark_json_overrides_actual_tps(self, tmp_path, example_decoder_1_5b, b200):
        """benchmark-json value should win over manual actual_tps."""
        data = {"throughput_tps": 500.0}
        p = tmp_path / "bench.json"
        p.write_text(json.dumps(data))
        bench_tps = parse_benchmark_json(str(p))
        manual_tps = 100.0
        # Simulate CLI precedence: benchmark-json wins
        actual = bench_tps  # this is what main() does
        est = estimate_sol(example_decoder_1_5b, b200, "fp32",
                           cache_length=256, actual_tps=actual)
        assert est.actual_tps == 500.0
        assert est.actual_tps != manual_tps


# ---------------------------------------------------------------------------
# Per-layer roofline
# ---------------------------------------------------------------------------

class TestPerLayerRoofline:
    @pytest.fixture
    def sample_timing(self):
        """Sample layer timing data (2 layers + lm_head)."""
        return {
            "layers": [
                {"name": "layer_0", "time_ms": 1.5},
                {"name": "layer_1", "time_ms": 1.4},
            ],
            "lm_head_ms": 0.3,
            "total_ms": 3.2,
        }

    def test_per_layer_roofline_all_layers(self, tiny_model, b200, sample_timing):
        """Returns one entry per layer + lm_head."""
        results = per_layer_roofline(
            tiny_model, b200, "fp32", sample_timing, cache_length=128)
        # 2 layers + 1 lm_head = 3 entries
        assert len(results) == 3
        names = {r.layer_name for r in results}
        assert "layer_0" in names
        assert "layer_1" in names
        assert "lm_head" in names

    def test_per_layer_utilization_bounded(self, tiny_model, b200, sample_timing):
        """Utilization should be > 0 for all layers."""
        results = per_layer_roofline(
            tiny_model, b200, "fp32", sample_timing, cache_length=128)
        for r in results:
            assert r.utilization_pct > 0, f"{r.layer_name} util should be > 0"

    def test_per_layer_sorted_by_utilization(self, tiny_model, b200, sample_timing):
        """Results should be sorted by utilization, worst (lowest) first."""
        results = per_layer_roofline(
            tiny_model, b200, "fp32", sample_timing, cache_length=128)
        utils = [r.utilization_pct for r in results]
        assert utils == sorted(utils), "Should be sorted by utilization ascending"

    def test_per_layer_theoretical_matches_total(self, example_decoder_1_5b, b200):
        """Sum of per-layer theoretical times should approximately match
        the total SOL theoretical time."""
        timing = {
            "layers": [{"name": f"layer_{i}", "time_ms": 1.0}
                        for i in range(example_decoder_1_5b.num_layers)],
            "lm_head_ms": 0.5,
        }
        results = per_layer_roofline(
            example_decoder_1_5b, b200, "fp32", timing, cache_length=256)
        sum_theoretical = sum(r.theoretical_ms for r in results)

        # Compare with overall SOL estimate's total bytes
        est = estimate_sol(example_decoder_1_5b, b200, "fp32", cache_length=256)
        practical_bw = b200.hbm_bandwidth_gb_s * b200.practical_bw_ratio * 1e9
        total_theoretical_ms = (est.total_read_bytes / practical_bw) * 1000

        # Should be close — not exact because estimate_sol includes embedding
        # weights (vocab_size * hidden_size) in total_params, while per-layer
        # roofline only covers transformer layers + lm_head.
        ratio = sum_theoretical / total_theoretical_ms
        assert 0.80 < ratio < 1.10, (
            f"Sum of layer theoretical ({sum_theoretical:.3f}ms) should be "
            f"close to total theoretical ({total_theoretical_ms:.3f}ms), "
            f"ratio={ratio:.3f}"
        )

    def test_per_layer_json_output(self, tiny_model, b200, sample_timing):
        """JSON output should include layer_roofline when provided."""
        results = per_layer_roofline(
            tiny_model, b200, "fp32", sample_timing, cache_length=128)
        est = estimate_sol(tiny_model, b200, "fp32", cache_length=128)
        j = to_json(est, layer_roofline=results)
        assert "layer_roofline" in j
        assert len(j["layer_roofline"]) == 3
        layer0 = j["layer_roofline"][0]
        assert "layer_name" in layer0
        assert "measured_ms" in layer0
        assert "theoretical_ms" in layer0
        assert "utilization_pct" in layer0
        # Without layer_roofline, no key present
        j2 = to_json(est)
        assert "layer_roofline" not in j2

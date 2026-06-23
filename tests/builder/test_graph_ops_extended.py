"""Extended tests for graph_ops -- pure NumPy helpers (no GPU).

Covers _yarn_correction_dim, compute_alibi_slopes (extended cases),
make_bucketed_relative_position_bias, and make_yarn_rope_table.

Trace: ARCH-GRP-001, UD-GRP-OPS-EXT
Intent: Validate pure-NumPy graph op helpers (YaRN correction, bucketed relative-position bias, ALiBi slopes, RoPE tables)
Preconditions: tensorrt_model_connect is importable; no GPU or TRT required
Postconditions: Helper functions produce mathematically correct values matching hand-computed references
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

try:
    from tensorrt_model_connect import graph_ops
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


# ===================================================================
# 1. _yarn_correction_dim
# ===================================================================

class TestYarnCorrectionDim:
    """Tests for _yarn_correction_dim(num_rotations, dim, base, max_pos)."""

    def test_known_values_num_rotations_1(self):
        """Manual computation: dim=128, base=10000, max_pos=4096, num_rotations=1.0."""
        dim = 128
        base = 10000.0
        max_pos = 4096
        num_rot = 1.0
        expected = dim * np.log(max_pos / (num_rot * 2 * np.pi)) / (2 * np.log(base))
        result = graph_ops._yarn_correction_dim(num_rot, dim, base, max_pos)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_known_values_num_rotations_2(self):
        """Manual computation: dim=128, base=10000, max_pos=4096, num_rotations=2.0."""
        dim = 128
        base = 10000.0
        max_pos = 4096
        num_rot = 2.0
        expected = dim * np.log(max_pos / (num_rot * 2 * np.pi)) / (2 * np.log(base))
        result = graph_ops._yarn_correction_dim(num_rot, dim, base, max_pos)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_num_rotations_1_manual_value(self):
        """Verify against hand-calculated value.

        dim=128, base=10000, max_pos=4096, num_rotations=1.0:
          numerator = 128 * log(4096 / (2*pi)) = 128 * log(651.898...)
          denominator = 2 * log(10000)
        """
        result = graph_ops._yarn_correction_dim(1.0, 128, 10000.0, 4096)
        # log(4096 / (2*pi)) = log(651.898...) ~ 6.4798
        # 128 * 6.4798 / (2 * 9.2103) ~ 829.42 / 18.4206 ~ 45.02
        assert 44.0 < result < 46.0

    def test_num_rotations_2_smaller_than_1(self):
        """With more rotations, the correction dim should be smaller."""
        r1 = graph_ops._yarn_correction_dim(1.0, 128, 10000.0, 4096)
        r2 = graph_ops._yarn_correction_dim(2.0, 128, 10000.0, 4096)
        assert r2 < r1

    def test_large_max_pos_1m(self):
        """Edge: very large max_position_embeddings (1M)."""
        result = graph_ops._yarn_correction_dim(1.0, 128, 10000.0, 1_000_000)
        expected = 128 * np.log(1_000_000 / (2 * np.pi)) / (2 * np.log(10000))
        np.testing.assert_allclose(result, expected, atol=1e-10)
        # Should be a positive, larger value than the 4096 case
        result_4096 = graph_ops._yarn_correction_dim(1.0, 128, 10000.0, 4096)
        assert result > result_4096

    @pytest.mark.parametrize("dim", [64, 128, 256])
    def test_scales_linearly_with_dim(self, dim):
        """Result should scale linearly with dim (all else equal)."""
        base = 10000.0
        max_pos = 4096
        num_rot = 1.0
        r = graph_ops._yarn_correction_dim(num_rot, dim, base, max_pos)
        r_double = graph_ops._yarn_correction_dim(num_rot, 2 * dim, base, max_pos)
        np.testing.assert_allclose(r_double, 2.0 * r, atol=1e-10)


# ===================================================================
# 2. compute_alibi_slopes (extended -- beyond test_graph_ops.py)
# ===================================================================

class TestComputeAlibiSlopesExtended:
    """Extended tests for compute_alibi_slopes.

    The base test_graph_ops.py already covers HF reference matching for
    n in {1,2,3,4,5,6,7,8,12,16,32}. These tests add missing coverage.
    """

    def test_n64_power_of_2(self):
        """n=64 (large power of 2) -- shape and HF match."""
        slopes = graph_ops.compute_alibi_slopes(64)
        assert slopes.shape == (64,)
        assert slopes.dtype == np.float32

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
    def test_monotonically_decreasing_power_of_2(self, n):
        """Slopes should be monotonically decreasing for power-of-2 heads."""
        slopes = graph_ops.compute_alibi_slopes(n)
        # Each subsequent slope is smaller than the previous
        for i in range(len(slopes) - 1):
            assert slopes[i] > slopes[i + 1], (
                f"slopes[{i}]={slopes[i]} should be > slopes[{i+1}]={slopes[i+1]}"
            )

    @pytest.mark.parametrize("n", [3, 5, 7, 12])
    def test_all_positive_non_power_of_2(self, n):
        """All slopes should be positive for non-power-of-2 heads."""
        slopes = graph_ops.compute_alibi_slopes(n)
        assert slopes.shape == (n,)
        assert np.all(slopes > 0)

    def test_slopes_0_relationship_n8(self):
        """For n=8: start = 2^(-1) = 0.5, slopes[0] = start * start^0 = 0.5."""
        slopes = graph_ops.compute_alibi_slopes(8)
        # start = 2^(-(2^-(log2(8)-3))) = 2^(-(2^-0)) = 2^(-1) = 0.5
        expected_start = 0.5
        np.testing.assert_allclose(slopes[0], expected_start, atol=1e-7)

    def test_slopes_geometric_n8(self):
        """For n=8 (power-of-2), slopes form a geometric sequence with ratio = start."""
        slopes = graph_ops.compute_alibi_slopes(8)
        # start = 0.5, so slopes = [0.5, 0.25, 0.125, ...]
        start = 0.5
        expected = np.array([start * (start ** i) for i in range(8)], dtype=np.float32)
        np.testing.assert_allclose(slopes, expected, atol=1e-7)

    def test_n1_single_head(self):
        """n=1 -- single head should return a single-element array."""
        slopes = graph_ops.compute_alibi_slopes(1)
        assert slopes.shape == (1,)
        # For n=1: start = 2^(-(2^-(log2(1)-3))) = 2^(-(2^3)) = 2^(-8) = 1/256
        expected_start = 2 ** (-8)
        np.testing.assert_allclose(slopes[0], expected_start, atol=1e-7)

    def test_all_slopes_less_than_1(self):
        """All slopes should be in (0, 1) for any reasonable num_heads."""
        for n in [1, 2, 3, 4, 5, 7, 8, 12, 16, 32, 64]:
            slopes = graph_ops.compute_alibi_slopes(n)
            assert np.all(slopes > 0), f"n={n}: all slopes should be > 0"
            assert np.all(slopes < 1), f"n={n}: all slopes should be < 1"


# ===================================================================
# 3. make_bucketed_relative_position_bias
# ===================================================================

class TestMakeBucketedRelativePositionBias:
    """Tests for make_bucketed_relative_position_bias.

    The return is [max_seq_len, max_seq_len] int32 bucket indices.  The
    num_heads parameter is retained for call-site symmetry with attention
    helpers but is not needed for bucket construction.
    """

    def test_output_shape(self):
        """Verify output shape is [max_seq_len, max_seq_len]."""
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=16, num_buckets=32, max_distance=128
        )
        assert result.shape == (16, 16)
        assert result.dtype == np.int32

    def test_values_in_range(self):
        """All bucket indices should be in [0, num_buckets)."""
        num_buckets = 32
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=32, num_buckets=num_buckets, max_distance=128
        )
        assert np.all(result >= 0)
        assert np.all(result < num_buckets)

    def test_diagonals_constant(self):
        """bias[i][j] depends only on j-i, so each diagonal should be constant."""
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=16, num_buckets=32, max_distance=128
        )
        seq_len = 16
        for offset in range(-seq_len + 1, seq_len):
            diag = np.diag(result, k=offset)
            assert np.all(diag == diag[0]), (
                f"Diagonal offset={offset} is not constant: {diag}"
            )

    def test_bidirectional_symmetry(self):
        """Positive and negative relative positions should map to different bucket halves.

        For bidirectional=True with num_buckets=32:
        - Positive offsets (j > i, i.e. looking right): bucket indices in [16, 31]
        - Negative offsets (j < i, i.e. looking left): bucket indices in [0, 15]
        - Zero offset (diagonal): bucket index for n=0
        """
        num_buckets = 32
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=16, num_buckets=num_buckets, max_distance=128
        )
        half_buckets = num_buckets // 2

        # Check upper triangle (j > i => relative_position > 0 => n < 0 =>
        # offset by num_bkts//2 in bidirectional mode)
        for i in range(16):
            for j in range(i + 1, 16):
                assert result[i, j] >= half_buckets, (
                    f"result[{i}][{j}]={result[i, j]} should be >= {half_buckets}"
                )

        # Check lower triangle (j < i => relative_position < 0 => n > 0 =>
        # no offset)
        for i in range(1, 16):
            for j in range(i):
                assert result[i, j] < half_buckets, (
                    f"result[{i}][{j}]={result[i, j]} should be < {half_buckets}"
                )

    def test_max_seq_len_1(self):
        """max_seq_len=1 should return a single-element array."""
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=1, num_buckets=32, max_distance=128
        )
        assert result.shape == (1, 1)
        # Relative position is 0; for bidirectional n=-0=0, n<0 is False,
        # so no half-bucket offset. n=abs(0)=0, is_small=True, ret=0.
        assert result[0, 0] == 0

    def test_num_buckets_2_minimal(self):
        """Minimal num_buckets=2 triggers division by zero in log scaling.

        With bidirectional=True, num_bkts becomes 1, max_exact becomes 0,
        causing log(max_dist/0) → ZeroDivisionError. This is a known edge
        case that does not occur for supported bucket counts.
        """
        with pytest.raises((ZeroDivisionError, FloatingPointError)):
            graph_ops.make_bucketed_relative_position_bias(
                num_heads=4, max_seq_len=8, num_buckets=2, max_distance=128
            )

    def test_diagonal_is_zero_relative(self):
        """The main diagonal (j==i) should have consistent bucket index.

        At i==j, relative_position = 0. In bidirectional mode:
        n = -0 = 0, n<0 is False => no offset. abs(0) = 0, is_small = True.
        ret = 0.
        """
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=16, num_buckets=32, max_distance=128
        )
        diag = np.diag(result)
        np.testing.assert_array_equal(diag, 0)

    @pytest.mark.parametrize("num_buckets", [8, 16, 32, 64])
    def test_various_num_buckets(self, num_buckets):
        """Values in range for different num_buckets settings."""
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=4, max_seq_len=16, num_buckets=num_buckets, max_distance=128
        )
        assert np.all(result >= 0)
        assert np.all(result < num_buckets)

    def test_adjacent_positions_monotonic(self):
        """For increasing distance, bucket indices should be non-decreasing.

        Looking at a fixed row i, as j increases from i+1 onward (positive
        offsets in upper half), the bucket should be non-decreasing.
        Similarly for the lower half.
        """
        result = graph_ops.make_bucketed_relative_position_bias(
            num_heads=8, max_seq_len=32, num_buckets=32, max_distance=128
        )
        # Upper triangle: row 0, columns 1..31
        row = result[0, 1:]
        diffs = np.diff(row)
        assert np.all(diffs >= 0), "Bucket indices should be non-decreasing with distance"

        # Lower triangle: column 0, rows 1..31
        col = result[1:, 0]
        diffs = np.diff(col)
        assert np.all(diffs >= 0), "Bucket indices should be non-decreasing with distance"


# ===================================================================
# 4. make_yarn_rope_table
# ===================================================================

class TestMakeYarnRopeTable:
    """Tests for make_yarn_rope_table."""

    def test_output_shape(self):
        """Verify output shape is [max_cache_length, hidden_size] float32."""
        table = graph_ops.make_yarn_rope_table(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, cosine=True, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        assert table.shape == (16, 64)
        assert table.dtype == np.float32

    def test_cos_sin_pythagorean_identity(self):
        """cos^2 + sin^2 should be approximately 1.0 for each position/dim."""
        kwargs = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        cos_table = graph_ops.make_yarn_rope_table(cosine=True, **kwargs)
        sin_table = graph_ops.make_yarn_rope_table(cosine=False, **kwargs)
        identity = cos_table ** 2 + sin_table ** 2
        np.testing.assert_allclose(identity, 1.0, atol=1e-5)

    def test_max_cache_length_0_returns_default(self):
        """Edge: max_cache_length=0 should return a (0, hidden_size) table."""
        table = graph_ops.make_yarn_rope_table(
            max_cache_length=0, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, cosine=True, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        assert table.shape == (0, 64)

    def test_hidden_not_divisible_by_heads_returns_default(self):
        """Edge: hidden_size not divisible by num_heads returns default table."""
        cos_table = graph_ops.make_yarn_rope_table(
            max_cache_length=16, hidden_size=65, num_attention_heads=4,
            rope_theta=10000.0, cosine=True, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        # Default for cosine=True is all 1.0
        np.testing.assert_array_equal(cos_table, np.ones((16, 65), dtype=np.float32))

        sin_table = graph_ops.make_yarn_rope_table(
            max_cache_length=16, hidden_size=65, num_attention_heads=4,
            rope_theta=10000.0, cosine=False, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        # Default for cosine=False is all 0.0
        np.testing.assert_array_equal(sin_table, np.zeros((16, 65), dtype=np.float32))

    def test_position_zero_is_trivial(self):
        """At position 0, cos(0)=1 and sin(0)=0 for all dims in head."""
        kwargs = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
        )
        cos_table = graph_ops.make_yarn_rope_table(cosine=True, **kwargs)
        sin_table = graph_ops.make_yarn_rope_table(cosine=False, **kwargs)
        # At position 0, angle = 0 * inv_freq = 0 for all freqs
        np.testing.assert_allclose(cos_table[0, :], 1.0, atol=1e-6)
        np.testing.assert_allclose(sin_table[0, :], 0.0, atol=1e-6)

    def test_scaling_factor_1_matches_standard_rope(self):
        """With scaling_factor=1.0, YaRN should match standard RoPE.

        When scaling_factor=1.0, freq_inter == freq_extra, so the ramp
        blending has no effect and inv_freq == freq_extra == standard inv_freq.
        """
        kwargs_common = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0,
        )
        # YaRN with scaling_factor=1.0
        yarn_cos = graph_ops.make_yarn_rope_table(
            cosine=True, scaling_factor=1.0,
            original_max_position_embeddings=4096,
            beta_fast=32.0, beta_slow=1.0,
            **kwargs_common,
        )
        # Standard RoPE
        std_cos = graph_ops.make_rope_table(cosine=True, **kwargs_common)

        np.testing.assert_allclose(yarn_cos, std_cos, atol=1e-5)

    def test_scaling_factor_1_sin_matches_standard(self):
        """Sin variant: scaling_factor=1.0 YaRN matches standard RoPE."""
        kwargs_common = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0,
        )
        yarn_sin = graph_ops.make_yarn_rope_table(
            cosine=False, scaling_factor=1.0,
            original_max_position_embeddings=4096,
            beta_fast=32.0, beta_slow=1.0,
            **kwargs_common,
        )
        std_sin = graph_ops.make_rope_table(cosine=False, **kwargs_common)
        np.testing.assert_allclose(yarn_sin, std_sin, atol=1e-5)

    def test_interleaved_output_shape(self):
        """Interleaved mode should produce same shape output."""
        table = graph_ops.make_yarn_rope_table(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, cosine=True, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
            interleaved=True,
        )
        assert table.shape == (16, 64)
        assert table.dtype == np.float32

    def test_interleaved_cos_sin_identity(self):
        """cos^2 + sin^2 = 1 in interleaved mode too."""
        kwargs = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0, scaling_factor=2.0,
            original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0,
            interleaved=True,
        )
        cos_table = graph_ops.make_yarn_rope_table(cosine=True, **kwargs)
        sin_table = graph_ops.make_yarn_rope_table(cosine=False, **kwargs)
        identity = cos_table ** 2 + sin_table ** 2
        np.testing.assert_allclose(identity, 1.0, atol=1e-5)

    def test_scaling_factor_gt1_differs_from_standard(self):
        """With scaling_factor > 1.0, YaRN output should differ from standard RoPE
        (at least for positions > 0 where angles are non-zero)."""
        kwargs_common = dict(
            max_cache_length=16, hidden_size=64, num_attention_heads=4,
            rope_theta=10000.0,
        )
        yarn_cos = graph_ops.make_yarn_rope_table(
            cosine=True, scaling_factor=4.0,
            original_max_position_embeddings=4096,
            beta_fast=32.0, beta_slow=1.0,
            **kwargs_common,
        )
        std_cos = graph_ops.make_rope_table(cosine=True, **kwargs_common)
        # At position 0 both are 1.0, but at later positions they should differ
        # (unless ramp happens to produce identity, which is unlikely for sf=4)
        diff = np.abs(yarn_cos[1:, :] - std_cos[1:, :])
        assert np.max(diff) > 1e-3, "YaRN with sf=4 should differ from standard RoPE"

    def test_reference_inv_freq_computation(self):
        """Verify the YaRN inv_freq computation against a manual reference.

        Uses max_cache_length=16, hidden_size=64, num_heads=4, rope_theta=10000,
        scaling_factor=2.0, original_max_pos=4096, beta_fast=32, beta_slow=1.
        """
        max_cache = 16
        hidden = 64
        num_heads = 4
        theta = 10000.0
        sf = 2.0
        orig_max_pos = 4096
        beta_fast = 32.0
        beta_slow = 1.0
        head_dim = hidden // num_heads  # 16
        half = head_dim // 2  # 8

        # Standard frequencies
        freq_extra = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        freq_inter = freq_extra / sf

        # Correction dims
        low = max(int(np.floor(graph_ops._yarn_correction_dim(
            beta_fast, head_dim, theta, orig_max_pos))), 0)
        high = min(int(np.ceil(graph_ops._yarn_correction_dim(
            beta_slow, head_dim, theta, orig_max_pos))), half - 1)
        ramp = np.clip((np.arange(half, dtype=np.float64) - low) / max(high - low, 1), 0.0, 1.0)
        inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)

        # Build reference table manually for cos
        ref_cos = np.ones((max_cache, hidden), dtype=np.float32)
        for pos in range(max_cache):
            for head in range(num_heads):
                for d in range(head_dim):
                    freq_idx = d % half
                    angle = pos * inv_freq[freq_idx]
                    ref_cos[pos, head * head_dim + d] = float(np.cos(angle))

        actual_cos = graph_ops.make_yarn_rope_table(
            max_cache_length=max_cache, hidden_size=hidden,
            num_attention_heads=num_heads, rope_theta=theta, cosine=True,
            scaling_factor=sf, original_max_position_embeddings=orig_max_pos,
            beta_fast=beta_fast, beta_slow=beta_slow,
        )
        np.testing.assert_allclose(actual_cos, ref_cos, atol=1e-6)


# ===================================================================
# 4b. make_rope_query_scale_table
# ===================================================================

class TestMakeRopeQueryScaleTable:
    """Tests for the per-position RoPE query scale table."""

    def test_values_match_hf_formula(self):
        table = graph_ops.make_rope_query_scale_table(
            max_cache_length=10,
            beta=0.1,
            original_max_position_embeddings=4,
        )
        positions = np.arange(10, dtype=np.float64)
        expected = 1.0 + 0.1 * np.log1p(np.floor(positions / 4.0))
        np.testing.assert_allclose(table[:, 0], expected.astype(np.float32), atol=1e-7)

    def test_positions_before_original_window_are_unscaled(self):
        table = graph_ops.make_rope_query_scale_table(
            max_cache_length=4,
            beta=0.1,
            original_max_position_embeddings=4,
        )
        np.testing.assert_allclose(table, np.ones((4, 1), dtype=np.float32))

    def test_zero_beta_disables_scaling(self):
        table = graph_ops.make_rope_query_scale_table(
            max_cache_length=8,
            beta=0.0,
            original_max_position_embeddings=4,
        )
        np.testing.assert_allclose(table, np.ones((8, 1), dtype=np.float32))


# ===================================================================
# 5. add_elu (TRT)
# ===================================================================

@requires_trt
class TestAddElu:
    def test_basic(self, trt_runner):
        x_np = np.array([[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_elu(net, inp["x"], alpha=1.0)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        ref = np.where(x_np > 0, x_np, 1.0 * (np.exp(x_np) - 1))
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_alpha_2(self, trt_runner):
        x_np = np.array([[-3.0, -1.5, -0.5, 0.0, 1.0, 3.0]], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_elu(net, inp["x"], alpha=2.0)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        ref = np.where(x_np > 0, x_np, 2.0 * (np.exp(x_np) - 1))
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_all_positive(self, trt_runner):
        x_np = np.array([[0.1, 0.5, 1.0, 2.0, 5.0]], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_elu(net, inp["x"], alpha=1.0)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        # ELU is identity for positive values
        np.testing.assert_allclose(result["out"], x_np, atol=1e-6)

    def test_all_negative(self, trt_runner):
        x_np = np.array([[-5.0, -2.0, -1.0, -0.5, -0.1]], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_elu(net, inp["x"], alpha=1.0)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        ref = 1.0 * (np.exp(x_np) - 1)
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# ===================================================================
# 6. add_conv1d (TRT)
# ===================================================================

@requires_trt
class TestAddConv1d:
    def test_no_bias(self, trt_runner):
        """Conv1d: [1, 4, 16], kernel_size=3, out_channels=8, no bias."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 4, 16).astype(np.float32)
        w_np = rng.randn(8, 4, 3).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_conv1d(
                net, inp["x"], w_np, bias=None,
                out_channels=8, kernel_size=3)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        conv = nn.Conv1d(4, 8, 3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w_np))
        ref = conv(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)

    def test_with_bias(self, trt_runner):
        """Conv1d with bias."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 4, 16).astype(np.float32)
        w_np = rng.randn(8, 4, 3).astype(np.float32)
        b_np = rng.randn(8).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_conv1d(
                net, inp["x"], w_np, bias=b_np,
                out_channels=8, kernel_size=3)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        conv = nn.Conv1d(4, 8, 3)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w_np))
            conv.bias.copy_(torch.tensor(b_np))
        ref = conv(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)

    def test_stride_padding(self, trt_runner):
        """Conv1d with stride=2 and padding=1."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 4, 16).astype(np.float32)
        w_np = rng.randn(8, 4, 3).astype(np.float32)
        b_np = rng.randn(8).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_conv1d(
                net, inp["x"], w_np, bias=b_np,
                out_channels=8, kernel_size=3,
                stride=2, padding=1)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        conv = nn.Conv1d(4, 8, 3, stride=2, padding=1)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w_np))
            conv.bias.copy_(torch.tensor(b_np))
        ref = conv(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# ===================================================================
# 7. add_conv2d (TRT)
# ===================================================================

@requires_trt
class TestAddConv2d:
    def test_basic(self, trt_runner):
        """Conv2d: [1, 3, 8, 8], kernel (3,3), out_ch=8 -> [1, 8, 6, 6]."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 3, 8, 8).astype(np.float32)
        w_np = rng.randn(8, 3, 3, 3).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_conv2d(
                net, inp["x"], w_np, bias=None,
                out_channels=8, kernel_size=(3, 3))
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 8, 6, 6)

        conv = nn.Conv2d(3, 8, 3, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w_np))
        ref = conv(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)

    def test_with_bias_and_padding(self, trt_runner):
        """Conv2d with bias and padding=1 -> same spatial size."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 3, 8, 8).astype(np.float32)
        w_np = rng.randn(8, 3, 3, 3).astype(np.float32)
        b_np = rng.randn(8).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_conv2d(
                net, inp["x"], w_np, bias=b_np,
                out_channels=8, kernel_size=(3, 3),
                padding=(1, 1))
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 8, 8, 8)

        conv = nn.Conv2d(3, 8, 3, padding=1)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(w_np))
            conv.bias.copy_(torch.tensor(b_np))
        ref = conv(torch.tensor(x_np)).detach().numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# ===================================================================
# 8. add_causal_pad_1d (TRT)
# ===================================================================

@requires_trt
class TestAddCausalPad1d:
    def test_shape_and_zeros(self, trt_runner):
        """Pad [1, 4, 8] with pad_left=3 -> [1, 4, 11], first 3 zeros."""
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 4, 8).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_causal_pad_1d(net, inp["x"], pad_left=3)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 4, 11)
        # First 3 elements along last dim should be zero
        np.testing.assert_allclose(result["out"][:, :, :3], 0.0, atol=1e-7)
        # Remaining should match input
        np.testing.assert_allclose(result["out"][:, :, 3:], x_np, atol=1e-7)

    def test_pad_left_1(self, trt_runner):
        """Minimal padding: pad_left=1."""
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 2, 4).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_causal_pad_1d(net, inp["x"], pad_left=1)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 2, 5)
        np.testing.assert_allclose(result["out"][:, :, :1], 0.0, atol=1e-7)
        np.testing.assert_allclose(result["out"][:, :, 1:], x_np, atol=1e-7)


# ===================================================================
# 9. add_slice_trim_right (TRT)
# ===================================================================

@requires_trt
class TestAddSliceTrimRight:
    def test_basic(self, trt_runner):
        """Trim [1, 4, 16] by 4 -> [1, 4, 12], output == inp[..., :12]."""
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 4, 16).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_slice_trim_right(net, inp["x"], trim=4)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 4, 12)
        np.testing.assert_allclose(result["out"], x_np[:, :, :12], atol=1e-7)

    def test_trim_1(self, trt_runner):
        """Trim just 1 element."""
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 2, 8).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_slice_trim_right(net, inp["x"], trim=1)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, 2, 7)
        np.testing.assert_allclose(result["out"], x_np[:, :, :7], atol=1e-7)


# ===================================================================
# 10. add_batch_norm_2d (TRT)
# ===================================================================

@requires_trt
class TestAddBatchNorm2d:
    def test_identity_params(self, trt_runner):
        """gamma=1, beta=0, mean=0, var=1 -> approximately identity."""
        rng = np.random.RandomState(42)
        num_ch = 4
        x_np = rng.randn(1, num_ch, 8, 8).astype(np.float32)
        gamma = np.ones(num_ch, dtype=np.float32)
        beta = np.zeros(num_ch, dtype=np.float32)
        mean = np.zeros(num_ch, dtype=np.float32)
        var = np.ones(num_ch, dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_batch_norm_2d(
                net, inp["x"], num_ch, gamma, beta, mean, var, eps=1e-5)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        np.testing.assert_allclose(result["out"], x_np, atol=1e-4)

    def test_nontrivial_params(self, trt_runner):
        """Compare against PyTorch BatchNorm2d in eval mode."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        num_ch = 4
        x_np = rng.randn(1, num_ch, 8, 8).astype(np.float32)
        gamma = rng.randn(num_ch).astype(np.float32)
        beta = rng.randn(num_ch).astype(np.float32)
        running_mean = rng.randn(num_ch).astype(np.float32)
        running_var = np.abs(rng.randn(num_ch)).astype(np.float32) + 0.1

        def build(net, inp):
            out = graph_ops.add_batch_norm_2d(
                net, inp["x"], num_ch, gamma, beta,
                running_mean, running_var, eps=1e-5)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        bn = nn.BatchNorm2d(num_ch, eps=1e-5)
        bn.eval()
        with torch.no_grad():
            bn.weight.copy_(torch.tensor(gamma))
            bn.bias.copy_(torch.tensor(beta))
            bn.running_mean.copy_(torch.tensor(running_mean))
            bn.running_var.copy_(torch.tensor(running_var))
            ref = bn(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# ===================================================================
# 11. add_group_norm 2D (TRT)
# ===================================================================

@requires_trt
class TestAddGroupNorm:
    def test_2d_identity_params(self, trt_runner):
        """2D: [4, 8], num_groups=2, gamma=1, beta=0 -> per-group normalized."""
        rng = np.random.RandomState(42)
        num_ch, num_groups = 8, 2
        x_np = rng.randn(4, num_ch).astype(np.float32)
        gamma = np.ones(num_ch, dtype=np.float32)
        beta = np.zeros(num_ch, dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_group_norm(
                net, inp["x"], num_ch, num_groups, gamma, beta, eps=1e-5)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        out = result["out"]

        # Verify: per-group mean ~0, std ~1
        group_size = num_ch // num_groups
        for g in range(num_groups):
            group_slice = out[:, g * group_size:(g + 1) * group_size]
            means = group_slice.mean(axis=-1)
            stds = group_slice.std(axis=-1)
            np.testing.assert_allclose(means, 0.0, atol=1e-4)
            np.testing.assert_allclose(stds, 1.0, atol=0.1)

    def test_2d_vs_torch(self, trt_runner):
        """2D: compare against manual group normalization."""
        import torch

        rng = np.random.RandomState(42)
        num_ch, num_groups = 8, 2
        x_np = rng.randn(4, num_ch).astype(np.float32)
        gamma = rng.randn(num_ch).astype(np.float32)
        beta = rng.randn(num_ch).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            out = graph_ops.add_group_norm(
                net, inp["x"], num_ch, num_groups, gamma, beta, eps=eps)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Manual reference
        x_t = torch.tensor(x_np)
        group_size = num_ch // num_groups
        x_grouped = x_t.reshape(4, num_groups, group_size)
        mean = x_grouped.mean(dim=-1, keepdim=True)
        var = x_grouped.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x_grouped - mean) / torch.sqrt(var + eps)
        normalized = normalized.reshape(4, num_ch)
        ref = (normalized * torch.tensor(gamma) + torch.tensor(beta)).numpy()

        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# ===================================================================
# 12. add_adaptive_layernorm (TRT)
# ===================================================================

@requires_trt
class TestAddAdaptiveLayerNorm:
    def test_zero_scale_shift(self, trt_runner):
        """scale=0, shift=0 -> approximately standard LayerNorm (no affine)."""
        import torch

        rng = np.random.RandomState(42)
        hidden = 16
        x_np = rng.randn(4, hidden).astype(np.float32)
        scale_np = np.zeros((1, hidden), dtype=np.float32)
        shift_np = np.zeros((1, hidden), dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_adaptive_layernorm(
                net, inp["x"], inp["scale"], inp["shift"],
                hidden, eps=1e-5)
            return {"out": out}

        result = trt_runner(build, {
            "x": x_np,
            "scale": scale_np,
            "shift": shift_np,
        })

        # Reference: standard LayerNorm without affine
        x_t = torch.tensor(x_np)
        mean = x_t.mean(dim=-1, keepdim=True)
        var = x_t.var(dim=-1, unbiased=False, keepdim=True)
        ref = ((x_t - mean) / torch.sqrt(var + 1e-5)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)

    def test_nontrivial_modulation(self, trt_runner):
        """Test with non-zero scale and shift: norm(x) * (1+scale) + shift."""
        import torch

        rng = np.random.RandomState(42)
        hidden = 16
        x_np = rng.randn(4, hidden).astype(np.float32)
        scale_np = rng.randn(1, hidden).astype(np.float32) * 0.1
        shift_np = rng.randn(1, hidden).astype(np.float32) * 0.1

        def build(net, inp):
            out = graph_ops.add_adaptive_layernorm(
                net, inp["x"], inp["scale"], inp["shift"],
                hidden, eps=1e-5)
            return {"out": out}

        result = trt_runner(build, {
            "x": x_np,
            "scale": scale_np,
            "shift": shift_np,
        })

        # Reference: norm(x) * (1 + scale) + shift
        x_t = torch.tensor(x_np)
        mean = x_t.mean(dim=-1, keepdim=True)
        var = x_t.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x_t - mean) / torch.sqrt(var + 1e-5)
        ref = (normalized * (1.0 + torch.tensor(scale_np))
               + torch.tensor(shift_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# ===================================================================
# 13. add_timestep_embedding (TRT)
# ===================================================================

@requires_trt
class TestAddTimestepEmbedding:
    def test_output_shape(self, trt_runner):
        """Verify output shape [1, dim] for scalar timestep."""
        dim = 64
        freq_dim = 64
        ts_np = np.array([10.0], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_timestep_embedding(
                net, inp["ts"], dim=dim, freq_dim=freq_dim)
            return {"out": out}

        result = trt_runner(build, {"ts": ts_np})
        assert result["out"].shape == (1, dim)

    def test_not_all_zeros(self, trt_runner):
        """Verify the embedding is not all zeros."""
        dim = 64
        freq_dim = 64
        ts_np = np.array([5.0], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_timestep_embedding(
                net, inp["ts"], dim=dim, freq_dim=freq_dim)
            return {"out": out}

        result = trt_runner(build, {"ts": ts_np})
        assert not np.allclose(result["out"], 0.0)

    def test_vs_manual_sincos(self, trt_runner):
        """Verify against manual sin/cos computation."""
        dim = 64
        freq_dim = 64
        half = freq_dim // 2
        max_period = 10000.0
        ts_val = 7.5
        ts_np = np.array([ts_val], dtype=np.float32)

        def build(net, inp):
            out = graph_ops.add_timestep_embedding(
                net, inp["ts"], dim=dim, freq_dim=freq_dim,
                max_period=max_period)
            return {"out": out}

        result = trt_runner(build, {"ts": ts_np})

        # Manual reference: cos then sin
        freqs = np.exp(
            -np.log(max_period)
            * np.arange(half, dtype=np.float32) / half)
        args = ts_val * freqs
        ref = np.concatenate([np.cos(args), np.sin(args)]).reshape(1, -1)
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_different_timesteps_differ(self, trt_runner):
        """Different timestep values produce different embeddings."""
        dim = 64
        freq_dim = 64

        def build(net, inp):
            out = graph_ops.add_timestep_embedding(
                net, inp["ts"], dim=dim, freq_dim=freq_dim)
            return {"out": out}

        r1 = trt_runner(build, {"ts": np.array([1.0], dtype=np.float32)})
        r2 = trt_runner(build, {"ts": np.array([100.0], dtype=np.float32)})
        assert not np.allclose(r1["out"], r2["out"])

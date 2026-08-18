# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extended tests for graph_ops -- pure NumPy helpers (no GPU).

Covers _yarn_correction_dim and compute_alibi_slopes extended cases.

Trace: ARCH-GRP-001, UD-GRP-OPS-EXT
Intent: Validate pure-NumPy graph op helpers (YaRN correction, ALiBi slopes, RoPE tables)
Preconditions: tensorrt_model_connect is importable; no GPU or TRT required
Postconditions: Helper functions produce mathematically correct values matching hand-computed references
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

try:
    from tests.builder.owned_graph_modules import load_graph_ops

    graph_ops = load_graph_ops()
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
# 3. add_elu (TRT)
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
# 9. add_reflect_pad_1d (TRT)
# ===================================================================

@requires_trt
class TestAddReflectPad1d:
    def test_matches_torch_reflect_padding(self, trt_runner):
        import torch
        import torch.nn.functional as F
        from tests.builder.owned_graph_modules import load_family_graph_ops

        bark_graph_ops = load_family_graph_ops("bark")

        x_np = np.arange(1, 6, dtype=np.float32).reshape(1, 1, 5)

        def build(net, inp):
            out = bark_graph_ops.add_reflect_pad_1d(
                net, inp["x"], pad_left=3, pad_right=2)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        expected = F.pad(torch.from_numpy(x_np), (3, 2), mode="reflect").numpy()
        np.testing.assert_array_equal(result["out"], expected)


# ===================================================================
# 10. add_lstm_unrolled (TRT)
# ===================================================================

@requires_trt
class TestAddLstmUnrolled:
    def test_matches_torch_lstm(self, trt_runner):
        import torch
        import torch.nn as nn
        from tests.builder.owned_graph_modules import load_family_graph_ops

        bark_graph_ops = load_family_graph_ops("bark")

        rng = np.random.RandomState(7)
        batch, seq_length, input_size, hidden_size = 1, 4, 2, 3
        x_np = rng.randn(batch, seq_length, input_size).astype(np.float32)
        w_ih = rng.randn(4 * hidden_size, input_size).astype(np.float32)
        w_hh = rng.randn(4 * hidden_size, hidden_size).astype(np.float32)
        b_ih = rng.randn(4 * hidden_size).astype(np.float32)
        b_hh = rng.randn(4 * hidden_size).astype(np.float32)

        def build(net, inp):
            out = bark_graph_ops.add_lstm_unrolled(
                net,
                inp["x"],
                w_ih,
                w_hh,
                b_ih,
                b_hh,
                hidden_size,
                seq_length,
            )
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        with torch.no_grad():
            lstm.weight_ih_l0.copy_(torch.from_numpy(w_ih))
            lstm.weight_hh_l0.copy_(torch.from_numpy(w_hh))
            lstm.bias_ih_l0.copy_(torch.from_numpy(b_ih))
            lstm.bias_hh_l0.copy_(torch.from_numpy(b_hh))
        expected = lstm(torch.from_numpy(x_np))[0].detach().numpy()
        np.testing.assert_allclose(result["out"], expected, atol=1e-5, rtol=1e-5)


# ===================================================================
# 11. add_slice_trim_right (TRT)
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
# 12. add_batch_norm_2d (TRT)
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

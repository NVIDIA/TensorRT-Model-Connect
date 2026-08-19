# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT graph block tests requiring GPU.

Tests for composable building blocks in graph_blocks.py:
  - apply_norm (LayerNorm and RMSNorm dispatch)
  - add_swiglu_mlp (gate/up/down SwiGLU)
  - add_gelu_fc_mlp (fc1/activation/fc2)

All tests require TRT + CUDA GPU.

Trace: ARCH-GRP-001, UD-GRP-BLOCKS
Intent: Validate composable TRT graph blocks (norm, SwiGLU MLP, GELU MLP) against PyTorch references
Preconditions: TRT and CUDA GPU are available; synthetic weight arrays match expected dimensions
Postconditions: TRT graph block outputs match PyTorch reference within numerical tolerance
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorrt_model_connect.models.qwen.tests._trt_test_support import (
    requires_trt,
    run_trt_graph,
)

try:
    from tensorrt_model_connect.models.qwen import graph_blocks, graph_ops
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


@pytest.fixture
def trt_runner():
    return run_trt_graph


class TestInferKvAttentionSize:
    def test_returns_compact_width(self):
        weights = {"layer.0.w_k": np.zeros((16, 8), dtype=np.float32)}

        assert graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=2, head_dim=4) == 8

    def test_rejects_expanded_kv_width(self):
        weights = {"layer.0.w_k": np.zeros((16, 16), dtype=np.float32)}

        with pytest.raises(ValueError, match="compact K/V width 8"):
            graph_blocks.infer_kv_attention_size(
                weights, num_kv_heads=2, head_dim=4)

    def test_rejects_mismatched_metadata(self):
        weights = {"_kv_attention_size": 16}

        with pytest.raises(ValueError, match="_kv_attention_size=16"):
            graph_blocks.infer_kv_attention_size(
                weights, num_kv_heads=2, head_dim=4)


# ===================================================================
# 1. apply_norm (TRT)
# ===================================================================

@requires_trt
class TestApplyNorm:
    def test_layernorm(self, trt_runner):
        """apply_norm with norm_type='layernorm' vs PyTorch LayerNorm."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        hidden = 32
        x_np = rng.randn(4, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        beta_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_blocks.apply_norm(
                net, inp["x"], hidden, gamma_np, beta_np,
                eps_t, norm_type="layernorm")
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        ln = nn.LayerNorm(hidden, eps=eps)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(gamma_np))
            ln.bias.copy_(torch.tensor(beta_np))
            ref = ln(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_layernorm_no_beta(self, trt_runner):
        """apply_norm with norm_type='layernorm' and beta=None fills zeros."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        hidden = 32
        x_np = rng.randn(4, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_blocks.apply_norm(
                net, inp["x"], hidden, gamma_np, None,
                eps_t, norm_type="layernorm")
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        ln = nn.LayerNorm(hidden, eps=eps)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(gamma_np))
            ln.bias.zero_()
            ref = ln(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_rmsnorm(self, trt_runner):
        """apply_norm with norm_type='rmsnorm' vs manual RMSNorm."""
        import torch

        rng = np.random.RandomState(42)
        hidden = 32
        x_np = rng.randn(4, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_blocks.apply_norm(
                net, inp["x"], hidden, gamma_np, None,
                eps_t, norm_type="rmsnorm")
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        x_t = torch.tensor(x_np)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        ref = (x_t / rms * torch.tensor(gamma_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    @pytest.mark.parametrize("hidden", [16, 32, 64])
    def test_rmsnorm_various_sizes(self, trt_runner, hidden):
        """RMSNorm works across different hidden sizes."""
        import torch

        rng = np.random.RandomState(42)
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp, h=hidden):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_blocks.apply_norm(
                net, inp["x"], h, gamma_np, None,
                eps_t, norm_type="rmsnorm")
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        x_t = torch.tensor(x_np)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        ref = (x_t / rms * torch.tensor(gamma_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# ===================================================================
# 2. add_swiglu_mlp (TRT)
# ===================================================================

@requires_trt
class TestAddSwigluMlp:
    def test_output_shape(self, trt_runner):
        """SwiGLU MLP: verify output shape [seq, hidden_size]."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_gate = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_up = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_down = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_gate": w_gate,
            "mlp.w_up": w_up,
            "mlp.w_down": w_down,
        }

        def build(net, inp):
            out = graph_blocks.add_swiglu_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (seq, hidden_size)

    def test_vs_manual(self, trt_runner):
        """SwiGLU MLP: gate/up/down computation vs manual NumPy."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_gate = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_up = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_down = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_gate": w_gate,
            "mlp.w_up": w_up,
            "mlp.w_down": w_down,
        }

        def build(net, inp):
            out = graph_blocks.add_swiglu_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Manual reference: SiLU(gate) * up, then down
        gate_proj = x_np @ w_gate
        up_proj = x_np @ w_up
        # SiLU = x * sigmoid(x)
        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_proj))
        swish = gate_proj * sigmoid_gate
        gated = swish * up_proj
        ref = gated @ w_down
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

    def test_single_token(self, trt_runner):
        """SwiGLU MLP with single-token input (seq=1)."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size = 32, 64

        w_gate = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_up = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_down = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(1, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_gate": w_gate,
            "mlp.w_up": w_up,
            "mlp.w_down": w_down,
        }

        def build(net, inp):
            out = graph_blocks.add_swiglu_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, hidden_size)

        # Manual reference
        gate_proj = x_np @ w_gate
        up_proj = x_np @ w_up
        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_proj))
        swish = gate_proj * sigmoid_gate
        gated = swish * up_proj
        ref = gated @ w_down
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)


# ===================================================================
# 3. add_gelu_fc_mlp (TRT)
# ===================================================================

@requires_trt
class TestAddGeluFcMlp:
    def test_output_shape(self, trt_runner):
        """GELU FC MLP: verify output shape [seq, hidden_size]."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_fc1 = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_fc2 = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_fc1": w_fc1,
            "mlp.w_fc2": w_fc2,
        }

        def build(net, inp):
            out = graph_blocks.add_gelu_fc_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (seq, hidden_size)

    def test_vs_manual(self, trt_runner):
        """GELU FC MLP: fc1 -> gelu_new -> fc2 vs manual computation."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_fc1 = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_fc2 = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_fc1": w_fc1,
            "mlp.w_fc2": w_fc2,
        }

        def build(net, inp):
            out = graph_blocks.add_gelu_fc_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Manual: fc1 -> gelu_new -> fc2
        fc1_out = x_np @ w_fc1
        # gelu_new: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_out = 0.5 * fc1_out * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (fc1_out + 0.044715 * fc1_out ** 3)))
        ref = gelu_out @ w_fc2
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

    def test_with_biases(self, trt_runner):
        """GELU FC MLP with fc1_bias and fc2_bias."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_fc1 = rng.randn(hidden_size, mlp_size).astype(np.float32)
        fc1_bias = rng.randn(mlp_size).astype(np.float32)
        w_fc2 = rng.randn(mlp_size, hidden_size).astype(np.float32)
        fc2_bias = rng.randn(hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_fc1": w_fc1,
            "mlp.fc1_bias": fc1_bias,
            "mlp.w_fc2": w_fc2,
            "mlp.fc2_bias": fc2_bias,
        }

        def build(net, inp):
            out = graph_blocks.add_gelu_fc_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Manual: fc1 + bias -> gelu_new -> fc2 + bias
        fc1_out = x_np @ w_fc1 + fc1_bias
        gelu_out = 0.5 * fc1_out * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (fc1_out + 0.044715 * fc1_out ** 3)))
        ref = gelu_out @ w_fc2 + fc2_bias
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

    def test_silu_activation(self, trt_runner):
        """GELU FC MLP with activation='silu' instead of default gelu_new."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size, seq = 32, 64, 4

        w_fc1 = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_fc2 = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(seq, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_fc1": w_fc1,
            "mlp.w_fc2": w_fc2,
        }

        def build(net, inp):
            out = graph_blocks.add_gelu_fc_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size,
                activation="silu")
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Manual: fc1 -> silu -> fc2
        fc1_out = x_np @ w_fc1
        silu_out = fc1_out * (1.0 / (1.0 + np.exp(-fc1_out)))
        ref = silu_out @ w_fc2
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

    def test_single_token(self, trt_runner):
        """GELU FC MLP with single-token input (seq=1)."""
        rng = np.random.RandomState(42)
        hidden_size, mlp_size = 32, 64

        w_fc1 = rng.randn(hidden_size, mlp_size).astype(np.float32)
        w_fc2 = rng.randn(mlp_size, hidden_size).astype(np.float32)
        x_np = rng.randn(1, hidden_size).astype(np.float32)

        weights = {
            "mlp.w_fc1": w_fc1,
            "mlp.w_fc2": w_fc2,
        }

        def build(net, inp):
            out = graph_blocks.add_gelu_fc_mlp(
                net, inp["x"],
                weights=weights, prefix="mlp",
                hidden_size=hidden_size, mlp_size=mlp_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        assert result["out"].shape == (1, hidden_size)

        fc1_out = x_np @ w_fc1
        gelu_out = 0.5 * fc1_out * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (fc1_out + 0.044715 * fc1_out ** 3)))
        ref = gelu_out @ w_fc2
        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

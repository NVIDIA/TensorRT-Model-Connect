"""Tests for all 18 graph_ops.py functions.

Pure-numpy tests run everywhere. TRT graph tests require TensorRT + CUDA GPU.

Trace: ARCH-GRP-001, UD-GRP-OPS
Intent: Validate atomic TRT graph ops (RoPE, ALiBi, RMSNorm, attention, etc.) against NumPy/PyTorch references
Preconditions: tensorrt_model_connect is importable; TRT+GPU available for graph-level tests
Postconditions: Each graph op produces numerically correct output matching its reference implementation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect import graph_ops

from tests.builder.conftest import requires_trt


# ===================================================================
# 1. layer_tensor_name (pure string, no TRT)
# ===================================================================

class TestLayerTensorName:
    def test_basic(self):
        assert graph_ops.layer_tensor_name("cache_k", 0) == "cache_k_0"
        assert graph_ops.layer_tensor_name("cache_v", 5) == "cache_v_5"
        assert graph_ops.layer_tensor_name("present_k", 31) == "present_k_31"

    def test_various_stems(self):
        assert graph_ops.layer_tensor_name("foo", 99) == "foo_99"


# ===================================================================
# 2. compute_alibi_slopes (pure numpy, no TRT)
# ===================================================================

def _hf_alibi_slopes(num_heads: int) -> np.ndarray:
    """Reference: HF BloomModel.build_alibi_tensor slope computation."""
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = np.arange(1, 1 + closest_power_of_2, dtype=np.int32)
    slopes = np.power(base, powers)
    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        extra_powers = np.arange(
            1, 1 + 2 * (num_heads - closest_power_of_2), 2, dtype=np.int32)
        slopes = np.concatenate([slopes, np.power(extra_base, extra_powers)])
    return slopes.astype(np.float32)


class TestComputeAlibiSlopes:
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 6, 12, 3, 5, 7])
    def test_vs_hf_reference(self, n):
        ours = graph_ops.compute_alibi_slopes(n)
        ref = _hf_alibi_slopes(n)
        assert ours.shape == (n,)
        np.testing.assert_allclose(ours, ref, atol=1e-7)


# ===================================================================
# 3. make_rope_table (pure numpy, no TRT)
# ===================================================================

def _hf_rope_table_llama(max_len, hidden, num_heads, theta, cosine):
    """Standard LLaMA-style RoPE: pairs (d, d+half_rotary)."""
    head_dim = hidden // num_heads
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    table = np.full((max_len, hidden), 1.0 if cosine else 0.0, dtype=np.float32)
    for pos in range(max_len):
        for h in range(num_heads):
            for i in range(half):
                angle = pos * inv_freq[i]
                val = float(np.cos(angle) if cosine else np.sin(angle))
                table[pos, h * head_dim + i] = val
                table[pos, h * head_dim + half + i] = val
    return table


def _hf_rope_table_interleaved(max_len, hidden, num_heads, theta, cosine,
                                partial_factor=1.0):
    """CodeGen/GPT-J style: repeat_interleave, pairs (d, d+1)."""
    head_dim = hidden // num_heads
    rotary_ndims = int(head_dim * partial_factor)
    half = rotary_ndims // 2
    inv_freq = 1.0 / (
        theta ** (np.arange(0, rotary_ndims, 2, dtype=np.float64) / rotary_ndims))
    table = np.full((max_len, hidden), 1.0 if cosine else 0.0, dtype=np.float32)
    for pos in range(max_len):
        for h in range(num_heads):
            for i in range(half):
                angle = pos * inv_freq[i]
                val = float(np.cos(angle) if cosine else np.sin(angle))
                table[pos, h * head_dim + 2 * i] = val
                table[pos, h * head_dim + 2 * i + 1] = val
    return table


class TestMakeRopeTable:
    def test_rotated_half_cos(self):
        ours = graph_ops.make_rope_table(16, 64, 4, 10000.0, cosine=True)
        ref = _hf_rope_table_llama(16, 64, 4, 10000.0, True)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_rotated_half_sin(self):
        ours = graph_ops.make_rope_table(16, 64, 4, 10000.0, cosine=False)
        ref = _hf_rope_table_llama(16, 64, 4, 10000.0, False)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_interleaved_cos(self):
        ours = graph_ops.make_rope_table(
            16, 64, 4, 10000.0, cosine=True, interleaved=True)
        ref = _hf_rope_table_interleaved(16, 64, 4, 10000.0, True)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_interleaved_sin(self):
        ours = graph_ops.make_rope_table(
            16, 64, 4, 10000.0, cosine=False, interleaved=True)
        ref = _hf_rope_table_interleaved(16, 64, 4, 10000.0, False)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_partial_rotary_standard(self):
        ours = graph_ops.make_rope_table(
            16, 64, 4, 10000.0, cosine=True, partial_rotary_factor=0.5)
        head_dim = 16
        # Non-rotary dims (last 8 per head) should remain cos default (1.0)
        for h in range(4):
            np.testing.assert_array_equal(
                ours[:, h * head_dim + 8 : h * head_dim + 16], 1.0)

    def test_partial_rotary_interleaved(self):
        ours = graph_ops.make_rope_table(
            16, 64, 4, 10000.0, cosine=True,
            partial_rotary_factor=0.5, interleaved=True)
        ref = _hf_rope_table_interleaved(16, 64, 4, 10000.0, True, 0.5)
        np.testing.assert_allclose(ours, ref, atol=1e-6)

    def test_edge_empty(self):
        table = graph_ops.make_rope_table(0, 64, 4, 10000.0, cosine=True)
        assert table.shape == (0, 64)


# ===================================================================
# 4. RoPE reference helpers
# ===================================================================

def _hf_rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope_rows_reference(
    x: np.ndarray,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    heads: int,
) -> np.ndarray:
    head_dim = x.shape[-1] // heads
    out = np.empty_like(x)
    for head in range(heads):
        start = head * head_dim
        end = start + head_dim
        x_head = x[:, start:end]
        out[:, start:end] = (
            x_head * cos_table[:, start:end]
            + _hf_rotate_half(x_head) * sin_table[:, start:end]
        )
    return out


# ===================================================================
# TRT graph op tests (require TRT + GPU)
# ===================================================================

# 5. add_constant
@requires_trt
class TestAddConstant:
    def test_shape_and_value(self, trt_runner):
        values = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        def build(net, inp):
            c = graph_ops.add_constant(net, (1, 3), values)
            # Add to input to produce an output (can't output a constant alone easily)
            s = net.add_elementwise(
                inp["x"], c, __import__("tensorrt").ElementWiseOperation.SUM)
            return {"out": s.get_output(0)}

        x = np.zeros((1, 3), dtype=np.float32)
        result = trt_runner(build, {"x": x})
        np.testing.assert_allclose(result["out"], values, atol=1e-7)


# 6. add_matmul_rhs_constant
@requires_trt
class TestAddMatmulRhsConstant:
    def test_vs_numpy(self, trt_runner):
        rng = np.random.RandomState(42)
        lhs_np = rng.randn(1, 16).astype(np.float32)
        rhs_np = rng.randn(16, 32).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_matmul_rhs_constant(net, inp["x"], 16, 32, rhs_np)
            return {"out": out}

        result = trt_runner(build, {"x": lhs_np})
        ref = lhs_np @ rhs_np
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)

    def test_different_sizes(self, trt_runner):
        rng = np.random.RandomState(123)
        lhs_np = rng.randn(4, 8).astype(np.float32)
        rhs_np = rng.randn(8, 4).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_matmul_rhs_constant(net, inp["x"], 8, 4, rhs_np)
            return {"out": out}

        result = trt_runner(build, {"x": lhs_np})
        ref = lhs_np @ rhs_np
        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# 7. add_bias_sum
@requires_trt
class TestAddBiasSum:
    def test_vs_numpy(self, trt_runner):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 64).astype(np.float32)
        bias_np = rng.randn(64).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_bias_sum(net, inp["x"], 64, bias_np)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        ref = x_np + bias_np.reshape(1, 64)
        np.testing.assert_allclose(result["out"], ref, atol=1e-6)


# 8. add_rms_norm
@requires_trt
class TestAddRmsNorm:
    @pytest.mark.parametrize("hidden", [64, 768])
    def test_vs_torch(self, trt_runner, hidden):
        import torch
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_rms_norm(net, inp["x"], hidden, gamma_np, eps_t)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        x_t = torch.tensor(x_np)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        ref = (x_t / rms * torch.tensor(gamma_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# 9. add_rms_norm_per_head
@requires_trt
class TestAddRmsNormPerHead:
    @pytest.mark.parametrize("num_heads,head_dim", [(4, 16), (12, 64)])
    def test_vs_torch(self, trt_runner, num_heads, head_dim):
        import torch
        hidden = num_heads * head_dim
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp, nh=num_heads, hd=head_dim):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_rms_norm_per_head(
                net, inp["x"], nh, hd, gamma_np, eps_t)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        x_t = torch.tensor(x_np).reshape(num_heads, head_dim)
        rms = torch.sqrt((x_t ** 2).mean(dim=-1, keepdim=True) + eps)
        g = torch.tensor(gamma_np).reshape(num_heads, head_dim)
        ref = (x_t / rms * g).reshape(1, hidden).numpy()
        np.testing.assert_allclose(result["out"].flatten(), ref.flatten(), atol=1e-5)


# 10. add_layer_norm
@requires_trt
class TestAddLayerNorm:
    @pytest.mark.parametrize("hidden", [64, 768])
    def test_vs_torch(self, trt_runner, hidden):
        import torch
        import torch.nn as nn
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, hidden).astype(np.float32)
        gamma_np = rng.randn(hidden).astype(np.float32)
        beta_np = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_layer_norm(
                net, inp["x"], hidden, gamma_np, beta_np, eps_t)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        ln = nn.LayerNorm(hidden, eps=eps)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(gamma_np))
            ln.bias.copy_(torch.tensor(beta_np))
            ref = ln(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# 11. add_gelu_new
@requires_trt
class TestAddGeluNew:
    def test_vs_hf_reference(self, trt_runner):
        import torch
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 128).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_gelu_new(net, inp["x"])
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        x_t = torch.tensor(x_np)
        ref = (0.5 * x_t * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x_t + 0.044715 * x_t ** 3)))).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)


# 12. add_activation
@requires_trt
class TestAddActivation:
    @pytest.mark.parametrize("act_name", [
        "silu", "relu", "gelu_new", "gelu", "relu2", "squared_relu"])
    def test_vs_torch(self, trt_runner, act_name):
        import torch
        import torch.nn as nn
        rng = np.random.RandomState(42)
        x_np = rng.randn(1, 128).astype(np.float32)

        def build(net, inp, an=act_name):
            out = graph_ops.add_activation(net, inp["x"], an)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        x_t = torch.tensor(x_np)
        if act_name == "silu":
            ref = nn.SiLU()(x_t).numpy()
        elif act_name == "relu":
            ref = nn.ReLU()(x_t).numpy()
        elif act_name in ("relu2", "squared_relu"):
            relu_out = nn.ReLU()(x_t)
            ref = (relu_out * relu_out).numpy()
        else:  # gelu_new, gelu
            ref = (0.5 * x_t * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x_t + 0.044715 * x_t ** 3)))).numpy()
        np.testing.assert_allclose(result["out"], ref, atol=1e-5)

    def test_unsupported_raises(self):
        """ValueError for unknown activation type (no TRT needed for this)."""
        import importlib
        if not importlib.util.find_spec("tensorrt"):
            pytest.skip("tensorrt not available")
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        inp = network.add_input("x", trt.float32, (1, 4))
        with pytest.raises(ValueError, match="Unsupported activation"):
            graph_ops.add_activation(network, inp, "swish_unknown")


# ===================================================================
# 14. add_self_attention_block (TRT)
# ===================================================================

@requires_trt
class TestAddSelfAttentionBlock:
    def test_vs_manual_attention(self, trt_runner):
        """Test full self-attention block vs manual QKV->softmax->output."""
        rng = np.random.RandomState(42)
        seq, hidden, heads = 4, 16, 2
        head_dim = hidden // heads

        x_np = rng.randn(seq, hidden).astype(np.float32)
        w_q = rng.randn(hidden, hidden).astype(np.float32)
        w_k = rng.randn(hidden, hidden).astype(np.float32)
        w_v = rng.randn(hidden, hidden).astype(np.float32)
        w_o = rng.randn(hidden, hidden).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_self_attention_block(
                net, inp["x"], w_q, w_k, w_v, w_o,
                hidden, heads, seq)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        trt_out = result["out"]

        # Manual reference
        q = x_np @ w_q  # [seq, hidden]
        k = x_np @ w_k
        v = x_np @ w_v
        scale = 1.0 / np.sqrt(head_dim)

        q_h = q.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        k_h = k.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        v_h = v.reshape(seq, heads, head_dim).transpose(1, 0, 2)

        scores = (q_h @ k_h.transpose(0, 2, 1)) * scale
        # Stable softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        ctx = attn @ v_h  # [heads, seq, head_dim]
        ctx_flat = ctx.transpose(1, 0, 2).reshape(seq, hidden)
        ref = ctx_flat @ w_o

        np.testing.assert_allclose(trt_out, ref, atol=1e-3)

    def test_with_biases(self, trt_runner):
        rng = np.random.RandomState(42)
        seq, hidden, heads = 4, 16, 2

        x_np = rng.randn(seq, hidden).astype(np.float32)
        w_q = rng.randn(hidden, hidden).astype(np.float32)
        w_k = rng.randn(hidden, hidden).astype(np.float32)
        w_v = rng.randn(hidden, hidden).astype(np.float32)
        w_o = rng.randn(hidden, hidden).astype(np.float32)
        q_bias = rng.randn(hidden).astype(np.float32)
        o_bias = rng.randn(hidden).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_self_attention_block(
                net, inp["x"], w_q, w_k, w_v, w_o,
                hidden, heads, seq,
                q_bias=q_bias, o_bias=o_bias)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        trt_out = result["out"]

        # Manual reference with biases
        head_dim = hidden // heads
        q = x_np @ w_q + q_bias
        k = x_np @ w_k
        v = x_np @ w_v
        scale = 1.0 / np.sqrt(head_dim)

        q_h = q.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        k_h = k.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        v_h = v.reshape(seq, heads, head_dim).transpose(1, 0, 2)

        scores = (q_h @ k_h.transpose(0, 2, 1)) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        ctx = attn @ v_h
        ctx_flat = ctx.transpose(1, 0, 2).reshape(seq, hidden)
        ref = ctx_flat @ w_o + o_bias

        np.testing.assert_allclose(trt_out, ref, atol=1e-3)


# ===================================================================
# 15. add_self_attention_block_with_rope (TRT)
# ===================================================================

@requires_trt
class TestAddSelfAttentionBlockWithRope:
    def test_vs_manual(self, trt_runner):
        rng = np.random.RandomState(42)
        seq, hidden, heads = 4, 16, 2
        head_dim = hidden // heads

        x_np = rng.randn(seq, hidden).astype(np.float32)
        w_q = rng.randn(hidden, hidden).astype(np.float32)
        w_k = rng.randn(hidden, hidden).astype(np.float32)
        w_v = rng.randn(hidden, hidden).astype(np.float32)
        w_o = rng.randn(hidden, hidden).astype(np.float32)

        cos_table = graph_ops.make_rope_table(seq, hidden, heads, 10000.0, True)
        sin_table = graph_ops.make_rope_table(seq, hidden, heads, 10000.0, False)
        def build(net, inp):
            out = graph_ops.add_self_attention_block_with_rope(
                net, inp["x"], w_q, w_k, w_v, w_o,
                hidden, heads, seq,
                cos_table, sin_table)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})
        trt_out = result["out"]

        # Manual: project, apply RoPE, attention
        q = x_np @ w_q
        k = x_np @ w_k
        v = x_np @ w_v

        q_roped = _apply_rope_rows_reference(q, cos_table, sin_table, heads)
        k_roped = _apply_rope_rows_reference(k, cos_table, sin_table, heads)

        scale = 1.0 / np.sqrt(head_dim)
        q_h = q_roped.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        k_h = k_roped.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        v_h = v.reshape(seq, heads, head_dim).transpose(1, 0, 2)

        scores = (q_h @ k_h.transpose(0, 2, 1)) * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
        ctx = attn @ v_h
        ctx_flat = ctx.transpose(1, 0, 2).reshape(seq, hidden)
        ref = ctx_flat @ w_o

        np.testing.assert_allclose(trt_out, ref, atol=1e-3)


# ===================================================================
# 16. add_windowed_self_attention_with_rope (TRT)
# ===================================================================

@requires_trt
class TestAddWindowedSelfAttentionWithRope:
    def test_single_window_matches_full(self, trt_runner):
        """With 1 window, windowed attention should match full attention."""
        rng = np.random.RandomState(42)
        # 1 window => num_windows=1, so seq == window seq
        seq, hidden, heads = 4, 16, 2
        num_windows = 1

        x_np = rng.randn(seq, hidden).astype(np.float32)
        w_q = rng.randn(hidden, hidden).astype(np.float32)
        w_k = rng.randn(hidden, hidden).astype(np.float32)
        w_v = rng.randn(hidden, hidden).astype(np.float32)
        w_o = rng.randn(hidden, hidden).astype(np.float32)

        cos_table = graph_ops.make_rope_table(seq, hidden, heads, 10000.0, True)
        sin_table = graph_ops.make_rope_table(seq, hidden, heads, 10000.0, False)

        def build_windowed(net, inp):
            out = graph_ops.add_windowed_self_attention_with_rope(
                net, inp["x"], w_q, w_k, w_v, w_o,
                hidden, heads, seq, num_windows,
                cos_table, sin_table)
            return {"out": out}

        def build_full(net, inp):
            out = graph_ops.add_self_attention_block_with_rope(
                net, inp["x"], w_q, w_k, w_v, w_o,
                hidden, heads, seq,
                cos_table, sin_table)
            return {"out": out}

        result_win = trt_runner(build_windowed, {"x": x_np})
        result_full = trt_runner(build_full, {"x": x_np})

        np.testing.assert_allclose(
            result_win["out"], result_full["out"], atol=1e-4)


# ===================================================================
# 17. add_patch_embed_3d (TRT)
# ===================================================================

@requires_trt
class TestAddPatchEmbed3d:
    def test_output_shape(self, trt_runner):
        """Verify output shape matches expected num_patches x embed_dim."""
        rng = np.random.RandomState(42)
        T, C, H, W = 2, 3, 28, 28
        patch_size = 14
        embed_dim = 32
        in_channels = C
        tc = T * C

        weight = rng.randn(embed_dim, tc, patch_size, patch_size).astype(np.float32)
        bias = rng.randn(embed_dim).astype(np.float32)
        pixel_values = rng.randn(tc, H, W).astype(np.float32)

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        expected_patches = num_patches_h * num_patches_w  # 2*2 = 4

        def build(net, inp):
            out = graph_ops.add_patch_embed_3d(
                net, inp["pixels"], weight, bias,
                in_channels=in_channels, embed_dim=embed_dim,
                temporal_patch_size=T, patch_size=patch_size)
            return {"out": out}

        result = trt_runner(build, {"pixels": pixel_values})
        assert result["out"].shape == (expected_patches, embed_dim)

    def test_vs_conv2d_manual(self, trt_runner):
        """Verify patch_embed_3d matches manual Conv2D computation."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        T, C, H, W = 2, 3, 28, 28
        patch_size = 14
        embed_dim = 8
        tc = T * C

        weight = rng.randn(embed_dim, tc, patch_size, patch_size).astype(np.float32)
        bias = rng.randn(embed_dim).astype(np.float32)
        pixel_values = rng.randn(tc, H, W).astype(np.float32)

        def build(net, inp):
            out = graph_ops.add_patch_embed_3d(
                net, inp["pixels"], weight, bias,
                in_channels=C, embed_dim=embed_dim,
                temporal_patch_size=T, patch_size=patch_size)
            return {"out": out}

        result = trt_runner(build, {"pixels": pixel_values})

        # Reference: torch Conv2d
        conv = nn.Conv2d(tc, embed_dim, patch_size, stride=patch_size)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor(weight))
            conv.bias.copy_(torch.tensor(bias))
            inp_t = torch.tensor(pixel_values).unsqueeze(0)  # [1, tc, H, W]
            out_t = conv(inp_t)  # [1, embed_dim, H', W']
            ref = out_t.permute(0, 2, 3, 1).reshape(-1, embed_dim).numpy()

        np.testing.assert_allclose(result["out"], ref, atol=1e-4)


# ===================================================================
# 18. add_spatial_merge (TRT)
# ===================================================================

@requires_trt
class TestAddSpatialMerge:
    def test_vs_manual_mlp(self, trt_runner):
        """Test spatial_merge LN + 2-layer MLP against manual numpy."""
        import torch
        import torch.nn as nn

        rng = np.random.RandomState(42)
        seq, input_dim = 4, 16
        hidden_dim = 32
        output_dim = 16
        merge_size = 2
        eps = 1e-6

        x_np = rng.randn(seq, input_dim).astype(np.float32)
        w_fc1 = rng.randn(input_dim, hidden_dim).astype(np.float32)
        w_fc2 = rng.randn(hidden_dim, output_dim).astype(np.float32)
        b_fc1 = rng.randn(hidden_dim).astype(np.float32)
        b_fc2 = rng.randn(output_dim).astype(np.float32)
        norm_gamma = rng.randn(input_dim).astype(np.float32)

        def build(net, inp):
            eps_t = graph_ops.add_constant(
                net, (1, 1), np.array([eps], dtype=np.float32))
            out = graph_ops.add_spatial_merge(
                net, inp["x"], w_fc1, w_fc2, b_fc1, b_fc2,
                norm_gamma, input_dim, hidden_dim, output_dim,
                eps_t, seq, merge_size)
            return {"out": out}

        result = trt_runner(build, {"x": x_np})

        # Reference: LayerNorm + Linear + GELU + Linear
        x_t = torch.tensor(x_np)
        # LayerNorm (manual, with zero beta)
        ln = nn.LayerNorm(input_dim, eps=eps)
        with torch.no_grad():
            ln.weight.copy_(torch.tensor(norm_gamma))
            ln.bias.zero_()
            normed = ln(x_t)
        fc1_out = normed @ torch.tensor(w_fc1) + torch.tensor(b_fc1)
        # GELU (tanh approx)
        gelu_out = 0.5 * fc1_out * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (fc1_out + 0.044715 * fc1_out ** 3)))
        ref = (gelu_out @ torch.tensor(w_fc2) + torch.tensor(b_fc2)).numpy()

        np.testing.assert_allclose(result["out"], ref, atol=1e-3)

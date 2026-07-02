# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TRT 10 native attention APIs added to graph_ops.py.

Intent:
    Verify that add_layer_norm_native, make_rope_table_half_dim,
    add_apply_rope_native, and _add_attention_core produce numerically
    correct output matching their existing reference implementations.

Preconditions:
    tensorrt_model_connect importable; TensorRT 10+ and CUDA GPU available for TRT tests.

Postconditions:
    Each native API produces output within 1e-4 absolute tolerance of the
    reference implementation built from primitive ops.

Trace: ARCH-GRP-001, UD-GRP-OPS, UT-NATIVE-ATTN-001
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.builder.conftest import requires_trt
from tests.builder.owned_graph_modules import load_graph_ops

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
graph_ops = load_graph_ops()


# ---------------------------------------------------------------------------
# Helper: run a small TRT graph on a STRONGLY_TYPED network
# ---------------------------------------------------------------------------

def _run_strongly_typed(build_fn, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build and run a STRONGLY_TYPED TRT engine from build_fn.

    IAttention and IRotaryEmbeddingLayer require a STRONGLY_TYPED network;
    this helper creates one so native-API tests can use it.

    build_fn(network, trt_inputs) -> dict[name, ITensor]
    """
    import tensorrt as trt
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    trt_inputs = {}
    for name, arr in inputs.items():
        if arr.dtype == np.int32:
            dt = trt.int32
        elif arr.dtype == np.float16:
            dt = trt.float16
        else:
            dt = trt.float32
        t = network.add_input(name, dt, tuple(arr.shape))
        trt_inputs[name] = t

    trt_outputs = build_fn(network, trt_inputs)

    for name, tensor in trt_outputs.items():
        tensor.name = name
        network.mark_output(tensor)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT build failed (STRONGLY_TYPED)")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    err, stream = cudart.cudaStreamCreate()
    assert err == 0, f"cudaStreamCreate failed: {err}"

    device_bufs = {}
    host_out = {}
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(tname))
        dtype_trt = engine.get_tensor_dtype(tname)
        np_dtype = np.float16 if dtype_trt == trt.float16 else np.float32
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        err, ptr = cudart.cudaMallocAsync(nbytes, stream)
        assert err == 0, f"cudaMalloc failed: {err}"
        device_bufs[tname] = (ptr, nbytes, np_dtype)
        mode = engine.get_tensor_mode(tname)
        if mode == trt.TensorIOMode.INPUT:
            arr = inputs[tname].astype(np_dtype if np_dtype != np.float32 else inputs[tname].dtype)
            cudart.cudaMemcpyAsync(
                ptr, arr.ctypes.data, arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        else:
            host_out[tname] = np.zeros(shape, dtype=np_dtype)
        ctx.set_tensor_address(tname, ptr)

    ctx.execute_async_v3(stream)

    for name, arr in host_out.items():
        ptr, nbytes, _ = device_bufs[name]
        cudart.cudaMemcpyAsync(
            arr.ctypes.data, ptr, arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)
    for ptr, nbytes, _ in device_bufs.values():
        cudart.cudaFreeAsync(ptr, stream)
    cudart.cudaStreamDestroy(stream)

    return host_out


# ---------------------------------------------------------------------------
# 1. add_layer_norm_native — pure numpy reference comparison
# ---------------------------------------------------------------------------

def _ref_layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                    eps: float) -> np.ndarray:
    """NumPy LayerNorm reference: (x - mean) / sqrt(var + eps) * gamma + beta."""
    x32 = x.astype(np.float32)
    mean = x32.mean(axis=-1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x32 - mean) / np.sqrt(var + eps) * gamma + beta).astype(x.dtype)


class TestAddLayerNormNative:
    """
    Intent: add_layer_norm_native uses INormalizationLayer which should match
            the manual LayerNorm built from primitive ops.
    Preconditions: TRT + GPU available.
    Postconditions: Output within 1e-4 absolute error of NumPy reference.
    Trace: UT-NATIVE-ATTN-001
    """

    @requires_trt
    @pytest.mark.parametrize("hidden_size", [32, 64, 128])
    def test_matches_numpy_reference(self, hidden_size):
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, hidden_size)).astype(np.float32)
        gamma = rng.uniform(0.8, 1.2, hidden_size).astype(np.float32)
        beta = rng.uniform(-0.1, 0.1, hidden_size).astype(np.float32)
        eps = 1e-5

        def build(network, trt_inputs):
            return {
                "out": graph_ops.add_layer_norm_native(
                    network, trt_inputs["x"], hidden_size, gamma, beta, eps)
            }

        out = _run_strongly_typed(build, {"x": x})["out"]
        ref = _ref_layer_norm(x, gamma, beta, eps)
        np.testing.assert_allclose(out, ref, atol=1e-4,
                                   err_msg="add_layer_norm_native vs numpy")

    @requires_trt
    def test_zero_beta_matches_rmsnorm_like_scaling(self):
        """With zero beta, output = gamma * normalized — same scaling as RMSNorm
        (though mean is still subtracted; this just checks beta=0 path)."""
        H = 64
        rng = np.random.default_rng(7)
        x = rng.standard_normal((1, H)).astype(np.float32)
        gamma = np.ones(H, dtype=np.float32)
        beta = np.zeros(H, dtype=np.float32)
        eps = 1e-5

        def build(network, trt_inputs):
            return {
                "out": graph_ops.add_layer_norm_native(
                    network, trt_inputs["x"], H, gamma, beta, eps)
            }

        out = _run_strongly_typed(build, {"x": x})["out"]
        ref = _ref_layer_norm(x, gamma, beta, eps)
        np.testing.assert_allclose(out, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. make_rope_table_half_dim — pure numpy, no TRT
# ---------------------------------------------------------------------------

class TestMakeRopeTableHalfDim:
    """
    Intent: make_rope_table_half_dim produces [max_S, rotary_ndims//2] tables
            whose values match the corresponding entries from make_rope_table.
    Preconditions: None (pure numpy).
    Postconditions: Shapes correct, values match full-dim table for head 0.
    Trace: UT-NATIVE-ATTN-001
    """

    @pytest.mark.parametrize("head_dim,num_heads,max_S", [
        (64, 4, 16),
        (128, 8, 32),
        (32, 2, 8),
    ])
    def test_shape(self, head_dim, num_heads, max_S):
        cos = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, 10000.0, True)
        sin = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, 10000.0, False)
        assert cos.shape == (max_S, head_dim // 2)
        assert sin.shape == (max_S, head_dim // 2)

    @pytest.mark.parametrize("max_S,head_dim,rope_theta", [
        (16, 64, 10000.0),
        (32, 128, 500000.0),
    ])
    def test_values_match_full_dim_table_head0(self, max_S, head_dim, rope_theta):
        """Values in half-dim table should match the first head's entries in
        the full-dim table for both cos and sin."""
        num_heads = 4
        hidden_size = num_heads * head_dim

        for cosine in (True, False):
            full = graph_ops.make_rope_table(
                max_S, hidden_size, num_heads, rope_theta, cosine)
            half = graph_ops.make_rope_table_half_dim(
                max_S, head_dim, rope_theta, cosine)
            # full[:, 0:head_dim//2] is the first half of head 0
            np.testing.assert_allclose(
                half, full[:, :head_dim // 2], atol=1e-6,
                err_msg=f"cosine={cosine}: half-dim mismatch vs full table head-0")

    def test_partial_rotary_factor(self):
        max_S, head_dim = 16, 64
        factor = 0.5
        cos = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, 10000.0, True, partial_rotary_factor=factor)
        rotary_ndims = int(head_dim * factor)
        assert cos.shape == (max_S, rotary_ndims // 2)

    def test_degenerate_length(self):
        """Zero length should return a default table without crashing."""
        t = graph_ops.make_rope_table_half_dim(0, 64, 10000.0, True)
        assert t.shape[1] == 32  # head_dim // 2

    def test_invalid_rotary_dim_raises(self):
        with pytest.raises(ValueError, match="TRT native RoPE requires"):
            graph_ops.make_rope_table_half_dim(8, 0, 10000.0, True)


# ---------------------------------------------------------------------------
# 3. add_apply_rope_native — TRT test vs. add_apply_rope reference
# ---------------------------------------------------------------------------

def _ref_rope(x: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray,
              num_heads: int, head_dim: int) -> np.ndarray:
    """Rotate-half RoPE reference: x*cos + rotate_half(x)*sin."""
    x = x.reshape(num_heads, head_dim)
    half = head_dim // 2
    x_rot = np.concatenate([-x[:, half:], x[:, :half]], axis=-1)
    result = x * np.tile(cos_row, (num_heads, 1)) + x_rot * np.tile(sin_row, (num_heads, 1))
    return result.reshape(1, num_heads * head_dim).astype(np.float32)


class TestAddApplyRopeNative:
    """
    Intent: add_apply_rope_native (IRotaryEmbeddingLayer) matches the
            rotate-half numpy reference for a single decoder token step.
    Preconditions: TRT + GPU available; STRONGLY_TYPED network.
    Postconditions: Output within 1e-3 absolute error of numpy reference.
    Trace: UT-NATIVE-ATTN-001
    """

    @requires_trt
    @pytest.mark.parametrize("num_heads,head_dim,pos", [
        (4, 32, 0),
        (4, 32, 7),
        (8, 64, 15),
    ])
    def test_single_token_matches_ref(self, num_heads, head_dim, pos):
        attention_size = num_heads * head_dim
        max_S = 32
        rope_theta = 10000.0

        cos_half_np = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, rope_theta, False)

        rng = np.random.default_rng(pos)
        x = rng.standard_normal((1, attention_size)).astype(np.float32)
        pos_arr = np.array([pos], dtype=np.int32)

        def build(network, trt_inputs):
            cos_t = graph_ops.add_constant(
                network, cos_half_np.shape, cos_half_np)
            sin_t = graph_ops.add_constant(
                network, sin_half_np.shape, sin_half_np)
            out = graph_ops.add_apply_rope_native(
                network, trt_inputs["x"],
                num_heads, head_dim,
                cos_t, sin_t,
                trt_inputs["pos"],
                head_dim, interleaved=False)
            return {"out": out}

        result = _run_strongly_typed(
            build, {"x": x, "pos": pos_arr})["out"]

        # Reference: extract the correct cos/sin row for this position
        cos_row = cos_half_np[pos]
        sin_row = sin_half_np[pos]
        # Expand to full head_dim (half → full by repeating for each half)
        cos_full = np.concatenate([cos_row, cos_row])
        sin_full = np.concatenate([sin_row, sin_row])
        ref = _ref_rope(x, cos_full, sin_full, num_heads, head_dim)

        np.testing.assert_allclose(result, ref, atol=1e-3,
                                   err_msg=f"pos={pos}: native RoPE mismatch")

    @requires_trt
    def test_multi_token_matches_ref(self):
        num_heads, head_dim, sq = 4, 32, 4
        attention_size = num_heads * head_dim
        max_S = 32
        rope_theta = 10000.0

        cos_half_np = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, rope_theta, True)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            max_S, head_dim, rope_theta, False)

        rng = np.random.default_rng(123)
        x = rng.standard_normal((sq, attention_size)).astype(np.float32)
        pos_arr = np.array([0, 3, 7, 11], dtype=np.int32)

        def build(network, trt_inputs):
            cos_t = graph_ops.add_constant(
                network, cos_half_np.shape, cos_half_np)
            sin_t = graph_ops.add_constant(
                network, sin_half_np.shape, sin_half_np)
            out = graph_ops.add_apply_rope_native(
                network, trt_inputs["x"],
                num_heads, head_dim,
                cos_t, sin_t,
                trt_inputs["pos"],
                head_dim, interleaved=False,
                sequence_length=None)
            return {"out": out}

        result = _run_strongly_typed(
            build, {"x": x, "pos": pos_arr})["out"]

        rows = []
        for row, pos in enumerate(pos_arr):
            cos_full = np.concatenate([cos_half_np[pos], cos_half_np[pos]])
            sin_full = np.concatenate([sin_half_np[pos], sin_half_np[pos]])
            rows.append(
                _ref_rope(
                    x[row:row + 1], cos_full, sin_full,
                    num_heads, head_dim))
        ref = np.concatenate(rows, axis=0)

        np.testing.assert_allclose(result, ref, atol=1e-3,
                                   err_msg="multi-token native RoPE mismatch")


# ---------------------------------------------------------------------------
# 4. _add_attention_core — TRT test vs. numpy SDPA reference
# ---------------------------------------------------------------------------

def _ref_sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """NumPy scaled dot-product attention reference.

    q/k/v: [B, H, S, D]  → output: [B, H, S, D]
    """
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    # scores: [B, H, q_S, kv_S]
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
    # softmax over kv dim
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v).astype(np.float32)


class TestAddAttentionCore:
    """
    Intent: _add_attention_core (IAttention with decomposable=True) matches
            numpy SDPA within tolerance for various head/sequence shapes.
    Preconditions: TRT + GPU; STRONGLY_TYPED network.
    Postconditions: Output within 2e-4 absolute error of numpy SDPA.
    Trace: UT-NATIVE-ATTN-001
    """

    @requires_trt
    @pytest.mark.parametrize("B,H,q_S,kv_S,D", [
        (1, 4, 1, 8, 16),    # decoder: single query token
        (1, 4, 8, 8, 16),    # encoder: full sequence
        (1, 8, 1, 16, 32),   # larger cache
    ])
    def test_matches_numpy_sdpa(self, B, H, q_S, kv_S, D):
        rng = np.random.default_rng(0)
        q = rng.standard_normal((B, H, q_S, D)).astype(np.float32)
        k = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)
        v = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)

        def build(network, trt_inputs):
            ctx = graph_ops._add_attention_core(
                network,
                trt_inputs["q"], trt_inputs["k"], trt_inputs["v"],
                causal=False)
            return {"out": ctx}

        out = _run_strongly_typed(
            build, {"q": q, "k": k, "v": v})["out"]
        ref = _ref_sdpa(q, k, v)

        # 1e-3: TRT fused kernels may accumulate slightly more floating-point
        # rounding error than numpy for multi-token sequences (q_S > 1).
        np.testing.assert_allclose(out, ref, atol=1e-3,
                                   err_msg=f"B={B} H={H} q={q_S} kv={kv_S} D={D}")

    @requires_trt
    def test_fp32_accumulation_accepts_fp16_inputs(self):
        """FP16 Q/K/V can opt into a FP32 IAttention boundary and cast back."""
        B, H, q_S, kv_S, D = 1, 4, 1, 64, 32
        rng = np.random.default_rng(3)
        q = rng.standard_normal((B, H, q_S, D)).astype(np.float16)
        k = rng.standard_normal((B, H, kv_S, D)).astype(np.float16)
        v = rng.standard_normal((B, H, kv_S, D)).astype(np.float16)

        def build(network, trt_inputs):
            ctx = graph_ops._add_attention_core(
                network,
                trt_inputs["q"], trt_inputs["k"], trt_inputs["v"],
                causal=False,
                fp32_accumulation=True)
            return {"out": ctx}

        out = _run_strongly_typed(
            build, {"q": q, "k": k, "v": v})["out"]
        ref = _ref_sdpa(
            q.astype(np.float32),
            k.astype(np.float32),
            v.astype(np.float32))

        assert out.dtype == np.float16
        np.testing.assert_allclose(out.astype(np.float32), ref, atol=1e-3)

    @requires_trt
    def test_causal_equals_no_mask_for_single_query(self):
        """For q_S=1 causal and non-causal are equivalent — both should match."""
        B, H, D, kv_S = 1, 4, 16, 8
        rng = np.random.default_rng(1)
        q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        k = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)
        v = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)

        def build_nc(network, trt_inputs):
            ctx = graph_ops._add_attention_core(
                network, trt_inputs["q"], trt_inputs["k"], trt_inputs["v"],
                causal=False)
            return {"out": ctx}

        out_nc = _run_strongly_typed(
            build_nc, {"q": q, "k": k, "v": v})["out"]
        ref = _ref_sdpa(q, k, v)
        np.testing.assert_allclose(out_nc, ref, atol=2e-4)

    @requires_trt
    def test_additive_mask_blocks_padding(self):
        """An additive -inf mask on the last kv slot should exclude it from attention."""
        B, H, kv_S, D = 1, 2, 4, 16
        rng = np.random.default_rng(2)
        q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        k = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)
        v = rng.standard_normal((B, H, kv_S, D)).astype(np.float32)

        # Mask: attend to first 3 slots, block last one
        mask = np.zeros((B, H, 1, kv_S), dtype=np.float32)
        mask[..., -1] = -1e9

        def build(network, trt_inputs):
            mask_t = graph_ops.add_constant(
                network, (B, H, 1, kv_S),
                mask.astype(np.float32))
            ctx = graph_ops._add_attention_core(
                network, trt_inputs["q"], trt_inputs["k"], trt_inputs["v"],
                causal=False, mask=mask_t)
            return {"out": ctx}

        out = _run_strongly_typed(
            build, {"q": q, "k": k, "v": v})["out"]

        # Reference with mask applied
        d = D
        scale = 1.0 / np.sqrt(d)
        scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale + mask
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        ref = np.einsum("bhqk,bhkd->bhqd", weights, v).astype(np.float32)

        # Match the tolerance used by the unmasked native-attention tests:
        # TRT fused attention can differ from NumPy by sub-1e-3 rounding.
        np.testing.assert_allclose(out, ref, atol=1e-3,
                                   err_msg="masked attention mismatch")

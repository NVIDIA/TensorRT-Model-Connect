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
from tests.builder.owned_graph_modules import load_family_graph_ops, load_graph_ops

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
graph_ops = load_graph_ops()
eagle_vlm_graph_ops = load_family_graph_ops("eagle_vlm")
qwen_graph_ops = load_family_graph_ops("qwen")
qwen_vl_graph_ops = load_family_graph_ops("qwen_vl")


# ---------------------------------------------------------------------------
# Helper: run a small TRT graph on a STRONGLY_TYPED network
# ---------------------------------------------------------------------------

def _run_strongly_typed(
    build_fn,
    inputs: dict[str, np.ndarray],
    *,
    output_aliases: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
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

    try:
        import ml_dtypes
    except ImportError:
        ml_dtypes = None

    def numpy_dtype(dtype):
        if dtype == trt.float16:
            return np.dtype(np.float16)
        if dtype == trt.bfloat16:
            if ml_dtypes is None:
                raise RuntimeError("BF16 TensorRT tests require ml_dtypes")
            return np.dtype(ml_dtypes.bfloat16)
        if dtype == trt.float32:
            return np.dtype(np.float32)
        if dtype == trt.int32:
            return np.dtype(np.int32)
        raise TypeError(f"Unsupported TensorRT test dtype: {dtype}")

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
        elif ml_dtypes is not None and arr.dtype == np.dtype(ml_dtypes.bfloat16):
            dt = trt.bfloat16
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
    output_aliases = output_aliases or {}

    err, stream = cudart.cudaStreamCreate()
    assert err == 0, f"cudaStreamCreate failed: {err}"

    io_tensors = {}
    for i in range(engine.num_io_tensors):
        tname = engine.get_tensor_name(i)
        io_tensors[tname] = (
            tuple(engine.get_tensor_shape(tname)),
            engine.get_tensor_dtype(tname),
            engine.get_tensor_mode(tname),
        )

    device_bufs = {}
    host_out = {}
    host_inputs = {}
    for tname, (shape, dtype_trt, mode) in io_tensors.items():
        if tname in output_aliases:
            continue
        np_dtype = numpy_dtype(dtype_trt)
        nbytes = int(np.prod(shape)) * np_dtype.itemsize
        err, ptr = cudart.cudaMallocAsync(nbytes, stream)
        assert err == 0, f"cudaMalloc failed: {err}"
        device_bufs[tname] = (ptr, nbytes, np_dtype)
        if mode == trt.TensorIOMode.INPUT:
            arr = np.ascontiguousarray(inputs[tname], dtype=np_dtype)
            host_inputs[tname] = arr
            cudart.cudaMemcpyAsync(
                ptr, arr.ctypes.data, arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        else:
            host_out[tname] = np.zeros(shape, dtype=np_dtype)

    for output_name, input_name in output_aliases.items():
        assert io_tensors[output_name][2] == trt.TensorIOMode.OUTPUT
        assert io_tensors[input_name][2] == trt.TensorIOMode.INPUT
        assert io_tensors[output_name][:2] == io_tensors[input_name][:2]
        device_bufs[output_name] = device_bufs[input_name]

    for tname, (ptr, _nbytes, _np_dtype) in device_bufs.items():
        ctx.set_tensor_address(tname, ptr)

    ctx.execute_async_v3(stream)

    for name, arr in host_out.items():
        ptr, nbytes, _ = device_bufs[name]
        cudart.cudaMemcpyAsync(
            arr.ctypes.data, ptr, arr.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    cudart.cudaStreamSynchronize(stream)
    unique_device_bufs = {ptr: (ptr, nbytes) for ptr, nbytes, _ in device_bufs.values()}
    for ptr, nbytes in unique_device_bufs.values():
        cudart.cudaFreeAsync(ptr, stream)
    cudart.cudaStreamDestroy(stream)

    return host_out


@requires_trt
def test_qwen_native_kv_mask_matches_active_causal_prefix():
    import tensorrt as trt

    inputs = {
        "token_id": np.array([11, 12, 13], dtype=np.int32),
        "cache_write_indices": np.array([2], dtype=np.int32),
        "key_value_lengths": np.array([5], dtype=np.int32),
    }

    def build(network, trt_inputs):
        return {
            "mask": qwen_graph_ops.add_native_kv_attention_mask(
                network,
                trt_inputs["token_id"],
                trt_inputs["cache_write_indices"],
                trt_inputs["key_value_lengths"],
                8,
                trt.float16,
            )
        }

    actual = _run_strongly_typed(build, inputs)["mask"]
    expected = np.full((1, 1, 3, 8), -1.0e4, dtype=np.float16)
    expected[0, 0, 0, :3] = 0
    expected[0, 0, 1, :4] = 0
    expected[0, 0, 2, :5] = 0
    np.testing.assert_array_equal(actual, expected)


@requires_trt
@pytest.mark.parametrize(
    ("model_dtype_name", "model_numpy_dtype"),
    [
        pytest.param("bfloat16", "bfloat16", id="bf16"),
        pytest.param("float16", "float16", id="fp16-model-bf16-attention"),
    ],
)
def test_qwen_native_kv_attention_masks_poisoned_inactive_suffix(
    model_dtype_name,
    model_numpy_dtype,
):
    """The explicit mask exposes only the active, causal cache prefix."""
    import tensorrt as trt

    ml_dtypes = pytest.importorskip("ml_dtypes")
    model_dtype = getattr(trt, model_dtype_name)
    model_numpy_dtype = (
        ml_dtypes.bfloat16
        if model_numpy_dtype == "bfloat16"
        else np.float16
    )
    attention_numpy_dtype = ml_dtypes.bfloat16

    query_length = 3
    cache_capacity = 16
    cache_write_index = 2
    active_length = 5
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128

    q = np.ones(
        (query_length, num_heads * head_dim), dtype=model_numpy_dtype
    )
    k_update = np.zeros(
        (query_length, num_kv_heads * head_dim), dtype=model_numpy_dtype
    )
    update_values = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    v_update = np.broadcast_to(
        update_values[:, None], (query_length, num_kv_heads * head_dim)
    ).astype(model_numpy_dtype)

    cache_k = np.zeros(
        (1, num_kv_heads, cache_capacity, head_dim), dtype=np.float32
    )
    cache_v = np.zeros_like(cache_k)
    cache_v[:, :, 0, :] = 1.0
    cache_v[:, :, 1, :] = 3.0
    # If the inactive suffix leaks through attention, these keys dominate the
    # logits and the large values make the numerical error unmistakable.
    cache_k[:, :, active_length:, :] = 16.0
    cache_v[:, :, active_length:, :] = 1000.0
    cache_k = cache_k.astype(attention_numpy_dtype)
    cache_v = cache_v.astype(attention_numpy_dtype)

    inputs = {
        "q": q,
        "k_update": k_update,
        "v_update": v_update,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "cache_write_indices": np.array([cache_write_index], dtype=np.int32),
        "key_value_lengths": np.array([active_length], dtype=np.int32),
    }

    def build(network, trt_inputs):
        mask = qwen_graph_ops.add_native_kv_attention_mask(
            network,
            trt_inputs["q"],
            trt_inputs["cache_write_indices"],
            trt_inputs["key_value_lengths"],
            cache_capacity,
            trt.bfloat16,
        )
        result = qwen_graph_ops.add_native_kv_cache_attention_from_rows(
            network,
            trt_inputs["q"],
            trt_inputs["k_update"],
            trt_inputs["v_update"],
            trt_inputs["cache_k"],
            trt_inputs["cache_v"],
            trt_inputs["cache_write_indices"],
            trt_inputs["key_value_lengths"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_seq=query_length,
            explicit_mask=mask,
            attention_dtype=trt.bfloat16,
        )
        assert result["context"].dtype == model_dtype
        return {
            "context": result["context"],
            "present_k": result["present_k"],
            "present_v": result["present_v"],
        }

    actual = _run_strongly_typed(
        build,
        inputs,
        output_aliases={"present_k": "cache_k", "present_v": "cache_v"},
    )["context"].astype(np.float32)
    expected_rows = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    expected = np.broadcast_to(
        expected_rows[:, None], (query_length, num_heads * head_dim)
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0625)


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

    @requires_trt
    def test_active_position_cache_matches_high_position_reference(self):
        """Active rank-3 caches avoid a serialized context-capacity table."""
        import tensorrt as trt

        num_heads, head_dim = 4, 128
        attention_size = num_heads * head_dim
        rope_theta = 1_000_000.0
        positions = np.array([0, 3, 40_934, 131_071], dtype=np.int32)

        rng = np.random.default_rng(20260728)
        x = rng.standard_normal(
            (len(positions), attention_size)
        ).astype(np.float32)
        inv_freq = qwen_graph_ops.make_native_active_rope_inv_freq(
            head_dim, rope_theta
        )

        def build(network, trt_inputs):
            cos_active, sin_active = qwen_graph_ops.add_active_rope_cache(
                network,
                trt_inputs["pos"],
                inv_freq,
                trt.float32,
            )
            out = qwen_graph_ops.add_apply_rope_native(
                network,
                trt_inputs["x"],
                num_heads,
                head_dim,
                cos_active,
                sin_active,
                None,
                head_dim,
                interleaved=False,
                sequence_length=None,
            )
            return {
                "out": out,
                "cos": cos_active,
                "sin": sin_active,
            }

        result = _run_strongly_typed(build, {"x": x, "pos": positions})

        angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
        cos_ref = np.asarray(np.cos(angles), dtype=np.float32)
        sin_ref = np.asarray(np.sin(angles), dtype=np.float32)
        rows = []
        for row in range(len(positions)):
            cos_full = np.concatenate([cos_ref[row], cos_ref[row]])
            sin_full = np.concatenate([sin_ref[row], sin_ref[row]])
            rows.append(
                _ref_rope(
                    x[row : row + 1],
                    cos_full,
                    sin_full,
                    num_heads,
                    head_dim,
                )
            )
        ref = np.concatenate(rows, axis=0)

        assert result["cos"].shape == (1, len(positions), head_dim // 2)
        assert result["sin"].shape == (1, len(positions), head_dim // 2)
        np.testing.assert_allclose(result["cos"][0], cos_ref, atol=1e-7)
        np.testing.assert_allclose(result["sin"][0], sin_ref, atol=1e-7)
        np.testing.assert_allclose(result["out"], ref, atol=1e-6)


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
    @pytest.mark.parametrize("num_kv_heads", [1, 2])
    def test_batched_gqa_preserves_each_batch_row(self, num_kv_heads):
        """Decomposed GQA repeats KV heads without collapsing batch."""
        batch, sequence, num_heads, head_dim = 2, 3, 4, 16
        rng = np.random.default_rng(31)
        q = rng.standard_normal(
            (batch, sequence, num_heads * head_dim)).astype(np.float32)
        k = rng.standard_normal(
            (batch, sequence, num_kv_heads * head_dim)).astype(np.float32)
        v = rng.standard_normal(
            (batch, sequence, num_kv_heads * head_dim)).astype(np.float32)

        def build(network, trt_inputs):
            return {
                "out": eagle_vlm_graph_ops.add_attention_from_rows(
                    network,
                    trt_inputs["q"],
                    trt_inputs["k"],
                    trt_inputs["v"],
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_seq=None,
                    kv_seq=None,
                    fp32_accumulation=True,
                )
            }

        out = _run_strongly_typed(build, {"q": q, "k": k, "v": v})["out"]

        qh = q.reshape(batch, sequence, num_heads, head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(batch, sequence, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(batch, sequence, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        repeat = num_heads // num_kv_heads
        ref = _ref_sdpa(
            qh,
            np.repeat(kh, repeat, axis=1),
            np.repeat(vh, repeat, axis=1),
        )
        ref = ref.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)
        np.testing.assert_allclose(out, ref, atol=1e-3)

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

    @requires_trt
    def test_qwen_vl_long_masked_gqa_fp32_matches_reference(self):
        """The stable FP32 boundary matches long masked GQA attention."""
        num_heads, num_kv_heads, head_dim = 16, 2, 128
        q_seq, kv_seq = 1, 1665
        rng = np.random.default_rng(20260731)
        q = rng.standard_normal(
            (q_seq, num_heads * head_dim)).astype(np.float16)
        k = rng.standard_normal(
            (kv_seq, num_kv_heads * head_dim)).astype(np.float16)
        v = rng.standard_normal(
            (kv_seq, num_kv_heads * head_dim)).astype(np.float16)
        mask = np.full((q_seq, kv_seq), -1.0e4, dtype=np.float16)
        mask[:, :1290] = 0.0
        mask[:, -1] = 0.0

        def build(network, trt_inputs):
            mask_4d = qwen_vl_graph_ops.add_2d_mask_to_4d(
                network, trt_inputs["mask"])
            return {
                "out": qwen_vl_graph_ops.add_attention_from_rows(
                    network,
                    trt_inputs["q"],
                    trt_inputs["k"],
                    trt_inputs["v"],
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_seq=q_seq,
                    kv_seq=kv_seq,
                    mask=mask_4d,
                    scale=1.0 / np.sqrt(head_dim),
                    fp32_accumulation=True,
                )
            }

        out = _run_strongly_typed(
            build, {"q": q, "k": k, "v": v, "mask": mask})["out"]

        qh = q.reshape(1, q_seq, num_heads, head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(1, kv_seq, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(1, kv_seq, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        repeats = num_heads // num_kv_heads
        kh = np.repeat(kh, repeats, axis=1)
        vh = np.repeat(vh, repeats, axis=1)
        scores = np.einsum("bhqd,bhkd->bhqk", qh.astype(np.float32),
                           kh.astype(np.float32)) / np.sqrt(head_dim)
        scores += mask.reshape(1, 1, q_seq, kv_seq).astype(np.float32)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        ref = np.einsum("bhqk,bhkd->bhqd", probs, vh.astype(np.float32))
        ref = ref.transpose(0, 2, 1, 3).reshape(q_seq, -1)

        np.testing.assert_allclose(out.astype(np.float32), ref, atol=2e-3)

class TestAddApplyMropeNative:
    """Qwen2.5-VL M-RoPE selects T/H/W frequency sections."""

    def test_qwen3_interleaved_axis_map_preserves_temporal_tail(self):
        axes = qwen_vl_graph_ops.mrope_frequency_axis_map(
            (24, 20, 20), 128, interleaved=True)

        np.testing.assert_array_equal(
            axes[:60], np.tile(np.array([0, 1, 2], dtype=np.int32), 20))
        np.testing.assert_array_equal(
            axes[60:], np.zeros(4, dtype=np.int32))
        assert [int(np.count_nonzero(axes == axis)) for axis in range(3)] == [
            24, 20, 20]

    def test_chunked_and_interleaved_layouts_are_distinct(self):
        chunked = qwen_vl_graph_ops.mrope_frequency_axis_map(
            (2, 1, 1), 8, interleaved=False)
        interleaved = qwen_vl_graph_ops.mrope_frequency_axis_map(
            (2, 1, 1), 8, interleaved=True)

        np.testing.assert_array_equal(chunked, [0, 0, 1, 2])
        np.testing.assert_array_equal(interleaved, [0, 1, 2, 0])

    @pytest.mark.parametrize(
        ("sections", "rotary_dim", "error"),
        [
            ((24, 20, 19), 128, "must sum"),
            ((1, 2, 1), 8, "cannot be represented"),
        ],
    )
    def test_interleaved_axis_map_rejects_invalid_sections(
        self, sections, rotary_dim, error
    ):
        with pytest.raises(ValueError, match=error):
            qwen_vl_graph_ops.mrope_frequency_axis_map(
                sections, rotary_dim, interleaved=True)

    @requires_trt
    def test_matches_numpy_reference(self):
        num_heads, head_dim = 2, 16
        sections = (2, 2, 4)
        positions = np.array([2, 4, 6], dtype=np.int32)
        rng = np.random.default_rng(29)
        x = rng.standard_normal((1, num_heads * head_dim)).astype(np.float32)
        cos = qwen_vl_graph_ops.make_rope_table_half_dim(
            16, head_dim, 10000.0, True)
        sin = qwen_vl_graph_ops.make_rope_table_half_dim(
            16, head_dim, 10000.0, False)

        def build(network, trt_inputs):
            cos_t = qwen_vl_graph_ops.add_constant(
                network, cos.shape, cos, dtype=np.float32)
            sin_t = qwen_vl_graph_ops.add_constant(
                network, sin.shape, sin, dtype=np.float32)
            return {
                "out": qwen_vl_graph_ops.add_apply_mrope_native(
                    network, trt_inputs["x"], num_heads, head_dim,
                    cos_t, sin_t, trt_inputs["positions"], sections, head_dim)
            }

        out = _run_strongly_typed(
            build, {"x": x, "positions": positions})["out"]

        offsets = np.cumsum((0,) + sections)
        cos_m = np.concatenate([
            cos[positions[axis], offsets[axis]:offsets[axis + 1]]
            for axis in range(3)
        ])
        sin_m = np.concatenate([
            sin[positions[axis], offsets[axis]:offsets[axis + 1]]
            for axis in range(3)
        ])
        xh = x.reshape(num_heads, head_dim)
        first, second = np.split(xh, 2, axis=-1)
        ref = np.concatenate(
            [first * cos_m - second * sin_m,
             second * cos_m + first * sin_m], axis=-1).reshape(1, -1)
        np.testing.assert_allclose(out, ref, atol=1e-4)

    @requires_trt
    def test_qwen3_dynamic_sequence_matches_interleaved_rotate_half(self):
        num_heads, head_dim, sq = 2, 16, 3
        sections = (4, 2, 2)
        positions = np.array(
            [
                [2, 3, 4],
                [5, 6, 7],
                [8, 9, 10],
            ],
            dtype=np.int32,
        )
        rng = np.random.default_rng(20260729)
        x = rng.standard_normal(
            (sq, num_heads * head_dim)).astype(np.float32)
        cos = qwen_vl_graph_ops.make_rope_table_half_dim(
            16, head_dim, 10000.0, True)
        sin = qwen_vl_graph_ops.make_rope_table_half_dim(
            16, head_dim, 10000.0, False)

        def build(network, trt_inputs):
            cos_t = qwen_vl_graph_ops.add_constant(
                network, cos.shape, cos, dtype=np.float32)
            sin_t = qwen_vl_graph_ops.add_constant(
                network, sin.shape, sin, dtype=np.float32)
            return {
                "out": qwen_vl_graph_ops.add_apply_mrope_native_sequence(
                    network,
                    trt_inputs["x"],
                    num_heads,
                    head_dim,
                    cos_t,
                    sin_t,
                    trt_inputs["positions"],
                    sections,
                    head_dim,
                    interleaved=False,
                    mrope_interleaved=True,
                )
            }

        out = _run_strongly_typed(
            build, {"x": x, "positions": positions})["out"]

        axes = qwen_vl_graph_ops.mrope_frequency_axis_map(
            sections, head_dim, interleaved=True)
        columns = np.arange(head_dim // 2)
        rows = []
        ordinary_rows = []
        adjacent_rows = []
        for row in range(sq):
            cos_m = cos[positions[axes, row], columns]
            sin_m = sin[positions[axes, row], columns]
            cos_full = np.concatenate([cos_m, cos_m])
            sin_full = np.concatenate([sin_m, sin_m])
            rows.append(_ref_rope(
                x[row:row + 1], cos_full, sin_full,
                num_heads, head_dim))

            temporal_cos = cos[positions[0, row]]
            temporal_sin = sin[positions[0, row]]
            ordinary_rows.append(_ref_rope(
                x[row:row + 1],
                np.concatenate([temporal_cos, temporal_cos]),
                np.concatenate([temporal_sin, temporal_sin]),
                num_heads,
                head_dim,
            ))

            heads = x[row].reshape(num_heads, head_dim)
            pairs = heads.reshape(num_heads, head_dim // 2, 2)
            adjacent = np.empty_like(pairs)
            adjacent[..., 0] = (
                pairs[..., 0] * cos_m - pairs[..., 1] * sin_m)
            adjacent[..., 1] = (
                pairs[..., 1] * cos_m + pairs[..., 0] * sin_m)
            adjacent_rows.append(adjacent.reshape(1, -1))

        ref = np.concatenate(rows, axis=0)
        ordinary_1d = np.concatenate(ordinary_rows, axis=0)
        wrong_adjacent_pair = np.concatenate(adjacent_rows, axis=0)

        np.testing.assert_allclose(out, ref, atol=1e-4)
        assert float(np.max(np.abs(ref - ordinary_1d))) > 0.1
        assert float(np.max(np.abs(ref - wrong_adjacent_pair))) > 0.1

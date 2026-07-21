# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
# Graph Ops


trt = trt_compat.get_trt()


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


# Process-lifetime keepalive for bf16 numpy buffers passed to
# trt.Weights(trt.bfloat16, ptr, count). That constructor does NOT copy,
# so we need to keep the numpy array alive until the TRT engine is built.
# Keyed by id(network) so different builds don't conflict.
_bf16_weight_keepalive: dict[int, list] = {}


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32).

    Handles bfloat16 (ml_dtypes.bfloat16) explicitly: TRT's Python bindings
    can't auto-detect the TRT type from the V16 numpy extension dtype, so
    we route to the explicit (DataType, ptr, count) constructor — which
    requires us to keep the numpy buffer alive for as long as TRT may
    read from it (engine build time).
    """
    arr = np.ascontiguousarray(values, dtype=dtype)
    try:
        import ml_dtypes
        is_bf16 = arr.dtype == ml_dtypes.bfloat16
    except ImportError:
        is_bf16 = False
    if is_bf16:
        weights = trt.Weights(trt.bfloat16, arr.ctypes.data, arr.size)
    else:
        weights = trt.Weights(arr)
    layer = network.add_constant(shape, weights)
    # The bf16 trt.Weights constructor does not copy; the array must outlive
    # TRT's read of the buffer (engine build). Setting an attribute on the
    # pybind11-wrapped INetworkDefinition is unreliable; use a module-level
    # dict keyed by id(network).
    if is_bf16:
        _bf16_weight_keepalive.setdefault(id(network), []).append(arr)
    return layer.get_output(0)


def _add_scalar_constant(
    network: trt.INetworkDefinition,
    value: float,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    return add_constant(network, (), np.array(value, dtype=dtype), dtype=dtype)


def _add_fp8_qdq(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    scale: float,
    *,
    output_dtype: trt.DataType | None = None,
) -> trt.ITensor:
    """Explicit FP8 Q/DQ pair for static per-tensor quantization."""
    scale_t = _add_scalar_constant(network, scale, dtype=np.float32)
    q = network.add_quantize(tensor, scale_t, trt.DataType.FP8)
    dq = network.add_dequantize(
        q.get_output(0), scale_t, output_dtype or tensor.dtype)
    return dq.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (
        (lhs_width, rhs_width)
        if rank <= 2
        else (1,) * (rank - 2) + (lhs_width, rhs_width)
    )
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE,
        rhs, trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(
        network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm: gamma * (x / sqrt(mean(x^2) + eps)).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.

    TRT's native normalization API implements mean-centered LayerNorm, not
    RMSNorm, so this remains a manual shared implementation.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_rms_norm_per_head(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Per-head RMSNorm for [Sq, num_heads * head_dim] tensors.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    ``sequence_length=None`` means runtime-dynamic Sq.
    ``gamma`` may be [num_heads * head_dim] or [head_dim] broadcast to heads.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    seq_dim = -1 if sequence_length is None else sequence_length
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (seq_dim, num_heads, head_dim)

    reshaped = reshape_in.get_output(0)
    if need_cast:
        reshaped = network.add_cast(reshaped, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    eps_3d = network.add_shuffle(eps_tensor)
    eps_3d.reshape_dims = (1, 1, 1)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_3d.get_output(0), trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.size == head_dim:
        gamma_t = add_constant(
            network, (1, 1, head_dim), gamma_arr.reshape(1, 1, head_dim),
            dtype=np.float32)
    else:
        gamma_t = add_constant(
            network, (1, num_heads, head_dim),
            gamma_arr.reshape(num_heads, head_dim), dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def add_rms_norm_last_dim(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm over the final dimension for any-rank tensors.

    Generalises :func:`add_rms_norm` (rank 2) to inputs of rank >= 2 by
    reducing over the last axis. Diffusion batched builders use this for
    ``[B, S, D]`` tensors.
    """
    rank = len(tuple(inp.shape))
    if rank < 2:
        raise ValueError("add_rms_norm_last_dim expects rank >= 2")

    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << (rank - 1), keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_shape = (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, gamma_shape, np.asarray(gamma).reshape(gamma_shape),
        dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_rms_norm_per_head_batched(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Per-head RMSNorm for ``[B, S, num_heads * head_dim]`` tensors.

    Batched companion to :func:`add_rms_norm_per_head` used by diffusion
    builders whose leading dim is a dynamic batch (``-1``). ``gamma`` may be
    ``[num_heads * head_dim]`` or ``[head_dim]`` broadcast to heads.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype

    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (-1, seq_dim, num_heads, head_dim)
    reshaped = reshape_in.get_output(0)
    if need_cast:
        reshaped = network.add_cast(reshaped, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    eps_4d = network.add_shuffle(eps_tensor)
    eps_4d.reshape_dims = (1, 1, 1, 1)
    sq = network.add_elementwise(reshaped, reshaped, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 3, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_4d.get_output(0), trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD)

    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.size == head_dim:
        gamma_shape = (1, 1, 1, head_dim)
        gamma_arr = gamma_arr.reshape(gamma_shape)
    else:
        gamma_shape = (1, 1, num_heads, head_dim)
        gamma_arr = gamma_arr.reshape(gamma_shape)
    gamma_t = add_constant(network, gamma_shape, gamma_arr, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (-1, seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

    Constants are cast to ``inp.dtype`` so the elementwise ops are valid in
    a STRONGLY_TYPED network when ``inp`` is bf16 (storage np_dtype is
    fp16, runtime trt_dtype is bfloat16) or any other non-matching combo.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(name, value):
        c = add_constant(
            network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(
        x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(
        x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM)
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0),
        trt.ElementWiseOperation.PROD)
    # tanh(...)
    tanh_l = network.add_activation(
        tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(
        one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(
        half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0),
        trt.ElementWiseOperation.PROD)
    return result.get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu', 'relu2'/'squared_relu'."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    elif activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU)
        sq = network.add_elementwise(
            relu.get_output(0), relu.get_output(0),
            trt.ElementWiseOperation.PROD)
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(
            inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


def make_t5_relative_position_bias(
    num_heads: int,
    max_seq_len: int,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Compute T5-style relative position bias table.

    Returns: [num_heads, max_seq_len, max_seq_len] float32 bias table.
    This is baked as a constant into the TRT graph.
    """
    def _relative_position_bucket(
        relative_position: np.ndarray,
        bidirectional: bool = True,
        num_bkts: int = 32,
        max_dist: int = 128,
    ) -> np.ndarray:
        """Map relative position to bucket index (T5 algorithm)."""
        ret = np.zeros_like(relative_position, dtype=np.int32)
        n = -relative_position
        if bidirectional:
            num_bkts //= 2
            ret += (n < 0).astype(np.int32) * num_bkts
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)

        max_exact = num_bkts // 2
        is_small = n < max_exact

        # Clamp to avoid log(0)
        n_clamped = np.maximum(n.astype(np.float32), 1)
        val_if_large = max_exact + (
            np.log(n_clamped / max_exact)
            / np.log(max_dist / max_exact)
            * (num_bkts - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_bkts - 1)

        ret += np.where(is_small, n, val_if_large)
        return ret

    # Build relative position matrix
    context_position = np.arange(max_seq_len, dtype=np.int32)[:, None]
    memory_position = np.arange(max_seq_len, dtype=np.int32)[None, :]
    relative_position = memory_position - context_position

    buckets = _relative_position_bucket(
        relative_position,
        bidirectional=True,
        num_bkts=num_buckets,
        max_dist=max_distance,
    )

    return buckets.astype(np.int32)


def add_layer_norm_no_affine(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm without learnable affine: (x - mean) / sqrt(var + eps).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.

    Builders that have a scalar epsilon should prefer add_layer_norm_native
    with all-ones gamma and all-zeros beta. This helper is the shared fallback
    for call sites that only thread epsilon as an ITensor.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    mean = network.add_reduce(
        inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)
    result = normalized.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


# ---------------------------------------------------------------------------
# TRT 10 native attention APIs (TRT 10.x)
#
# Three primitives replace manual primitive chains:
#   add_layer_norm_native  → INormalizationLayer  (replaces add_layer_norm)
#   add_apply_rope_native  → IRotaryEmbeddingLayer
#   add_attention_core     → IAttention           (replaces score+softmax+V)
# ---------------------------------------------------------------------------

def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (
        (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    )
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype)
    beta_t = add_constant(
        network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)


def validate_native_rope_dim(
    rotary_embedding_dim: int,
    *,
    field_name: str = "rotary_embedding_dim",
) -> int:
    """Validate the dimension contract required by TRT native RoPE."""
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2 != 0:
        raise ValueError(
            f"TRT native RoPE requires {field_name} to be an even value >= 2; "
            f"got {rotary_embedding_dim}")
    return rotary_embedding_dim


def make_rope_table_half_dim(
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    partial_rotary_factor: float = 1.0,
    interleaved: bool = False,
) -> np.ndarray:
    """Build a RoPE cos/sin table of shape [max_cache_length, rotary_ndims // 2].

    IRotaryEmbeddingLayer expects the cos/sin cache with only the *half*
    rotary dimension (it internally handles both halves).  This is different
    from make_rope_table which produces [max_cache_length, hidden_size] by
    repeating the per-head values across all heads.

    Args:
        max_cache_length: Number of positions (rows in the table).
        head_dim:         Full head dimension (D).
        rope_theta:       Base frequency for inverse-frequency computation.
        cosine:           True → cos table, False → sin table.
        partial_rotary_factor: Fraction of head dims that rotate (default 1.0).
        interleaved:      If True, adjacent-pair frequencies (CodeGen/GPT-J).
                          If False, half-split frequencies (LLaMA/Qwen).

    Returns:
        Float32 array [max_cache_length, rotary_ndims // 2].
    """
    rotary_ndims = int(head_dim * partial_rotary_factor)
    rotary_ndims = validate_native_rope_dim(rotary_ndims)
    half = rotary_ndims // 2
    default = 1.0 if cosine else 0.0
    if max_cache_length <= 0 or rope_theta <= 0.0:
        return np.full((max(max_cache_length, 1), max(half, 1)),
                       default, dtype=np.float32)
    table = np.full((max_cache_length, half), default, dtype=np.float32)
    for pos in range(max_cache_length):
        for d in range(half):
            # For both interleaved and rotate-half the frequency index is d
            # (the distinction only affects which input pair is rotated; the
            # freq assignment per half-dim is the same).
            exponent = (2.0 * d) / rotary_ndims
            inv_freq = rope_theta ** (-exponent)
            angle = pos * inv_freq
            table[pos, d] = np.cos(angle) if cosine else np.sin(angle)
    return table


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Apply RoPE via TRT native IRotaryEmbeddingLayer.

    Handles both single-token decoder steps and dynamic-Sq prefill/decode
    graphs without a manual rotate-half matmul chain.

    Shape contract (IRotaryEmbeddingLayer with position_ids):
      input:           [1, num_heads, Sq, head_dim]  (reshaped internally)
      cos_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      sin_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      position_id:     [Sq] int32, reshaped to [1, Sq] internally
      interleaved:     False → rotate-half (LLaMA/Qwen)
                       True  → adjacent-pair (CodeGen/GPT-J)

    Args:
        inp:                  [Sq, num_heads * head_dim].
        num_heads:            Number of attention heads.
        head_dim:             Per-head dimension.
        cos_cache_2d:         Pre-built 2-D cos table constant.
        sin_cache_2d:         Pre-built 2-D sin table constant.
        position_id:          Runtime position indices, shape [Sq] int32.
        rotary_embedding_dim: Number of head dims that participate in RoPE.
        interleaved:          Frequency layout (see above).
        sequence_length:      Static Sq, or None for runtime-dynamic Sq.

    Returns:
        [Sq, num_heads * head_dim] with RoPE applied.
    """
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim

    inp_4d = reshape_rows_to_heads_4d(
        network, inp, num_heads, head_dim, sequence_length)

    # Reshape position_id [Sq] -> [1, Sq] (batch=1).
    seq_dim = -1 if sequence_length is None else sequence_length
    pos_2d = network.add_shuffle(position_id)
    pos_2d.reshape_dims = (1, seq_dim)

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        interleaved,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(
        network, rope.get_output(0), attention_size, sequence_length)


def add_apply_rope_native_sequence(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_3d: trt.ITensor,
    sin_cache_3d: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Apply native RoPE with per-position caches [1, Sq, rotary_dim / 2]."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim
    inp_4d = reshape_rows_to_heads_4d(
        network, inp, num_heads, head_dim, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_3d,
        sin_cache_3d,
        interleaved,
        rotary_embedding_dim,
    )
    return reshape_heads_4d_to_rows(
        network, rope.get_output(0), attention_size, sequence_length)


def add_apply_rope_native_from_full_cache(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_full: trt.ITensor,
    sin_cache_full: trt.ITensor,
    sequence_length: int,
    interleaved: bool = False,
    rotary_embedding_dim: int | None = None,
) -> trt.ITensor:
    """Apply native RoPE from per-head full-dim cos/sin caches [Sq, D]."""
    rope_dim = head_dim if rotary_embedding_dim is None else rotary_embedding_dim
    rope_dim = validate_native_rope_dim(rope_dim)
    half = rope_dim // 2
    stride = 2 if interleaved else 1
    cos_half = network.add_slice(
        cos_cache_full,
        start=(0, 0),
        shape=(sequence_length, half),
        stride=(1, stride))
    sin_half = network.add_slice(
        sin_cache_full,
        start=(0, 0),
        shape=(sequence_length, half),
        stride=(1, stride))
    cos_3d = network.add_shuffle(cos_half.get_output(0))
    cos_3d.reshape_dims = (1, sequence_length, half)
    sin_3d = network.add_shuffle(sin_half.get_output(0))
    sin_3d.reshape_dims = (1, sequence_length, half)
    return add_apply_rope_native_sequence(
        network,
        inp,
        num_heads,
        head_dim,
        cos_3d.get_output(0),
        sin_3d.get_output(0),
        rotary_embedding_dim=rope_dim,
        interleaved=interleaved,
        sequence_length=sequence_length)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    fp32_accumulation: bool = False,
    quant_scales: dict[str, float] | None = None,
) -> trt.ITensor:
    """Scaled dot-product attention via TRT native IAttention layer.

    Replaces the manual Q@K^T → scale → softmax → @V chain.  TRT 10 fuses
    this into a single kernel when a compatible implementation is available;
    decomposable=True ensures a correct fallback to primitives otherwise.

    NOTE: TRT IAttention computes raw BMM1 = Q @ K^T without any built-in
    1/sqrt(D) scaling.  We pre-scale Q by 1/sqrt(D) so that the fused kernel
    computes the standard scaled dot-product attention formula.

    Args:
        q_4d:    Query  [B, H, q_seq, D].
        k_4d:    Key    [B, H, kv_seq, D].
        v_4d:    Value  [B, H, kv_seq, D].
        causal:  Apply causal (autoregressive) mask.  Mutually exclusive
                 with ``mask``.
        mask:    Optional additive float mask [B, H, q_seq, kv_seq] added
                 to scaled logits before softmax.  Cannot be used with
                 causal=True.
        scale:   Optional Q pre-scale factor.  Defaults to 1/sqrt(D).
        fp32_accumulation:
                 Cast Q/K/V to FP32 before IAttention, then cast the context
                 back to the original Q dtype.  TRT may still select a
                 Half-input fused MHA tactic after optimizing the casts, while
                 keeping the IAttention accumulation/output boundary in FP32.
        quant_scales:
                 Optional FP8 static quantization scales for IAttention.
                 Expected keys are q_bmm_scale, k_bmm_scale, v_bmm_scale,
                 softmax_scale, and optionally bmm2_output_scale.

    Returns:
        Context tensor [B, H, q_seq, D].
    """
    output_dtype = q_4d.dtype
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    # Pre-scale inputs: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0

    def _scale_attention_input(tensor: trt.ITensor, factor: float) -> trt.ITensor:
        # Use FP16 weights directly for FP16; BF16 has no numpy native type so
        # create as FP32 and cast; FP32 falls through to the default.
        scale_np_dtype = np.float16 if tensor.dtype == trt.float16 else np.float32
        scale_t = add_constant(
            network, (1, 1, 1, 1), np.array([[[[factor]]]]),
            dtype=scale_np_dtype)
        if tensor.dtype == trt.bfloat16:
            scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
        return network.add_elementwise(
            tensor, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    if quant_scales:
        # Match ModelOpt's FP8 SDPA export pattern: Q and K are both scaled by
        # sqrt(sm_scale) before the FP8 Q/DQ that feeds BMM1.
        sqrt_scale = float(np.sqrt(scale))
        q_attn = _scale_attention_input(q_4d, sqrt_scale)
        k_attn = _scale_attention_input(k_4d, sqrt_scale)
    else:
        q_attn = _scale_attention_input(q_4d, scale)
        k_attn = k_4d
    v_attn = v_4d
    if quant_scales:
        required = ("q_bmm_scale", "k_bmm_scale", "v_bmm_scale", "softmax_scale")
        missing = [key for key in required if quant_scales.get(key) is None]
        if missing:
            raise ValueError(
                "quantized IAttention requires scales for "
                f"{', '.join(required)}; missing {', '.join(missing)}")
        q_attn = _add_fp8_qdq(
            network, q_attn, float(quant_scales["q_bmm_scale"]),
            output_dtype=output_dtype)
        k_attn = _add_fp8_qdq(
            network, k_attn, float(quant_scales["k_bmm_scale"]),
            output_dtype=output_dtype)
        v_attn = _add_fp8_qdq(
            network, v_attn, float(quant_scales["v_bmm_scale"]),
            output_dtype=output_dtype)

    attn = network.add_attention(
        q_attn, k_attn, v_attn,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    if quant_scales:
        attn.normalization_quantize_scale = _add_scalar_constant(
            network, float(quant_scales["softmax_scale"]), dtype=np.float32)
        attn.normalization_quantize_to_type = trt.DataType.FP8
    # Allow fallback for ordinary BF16/FP16 attention.  Quantized IAttention
    # should stay fused; otherwise TensorRT may lower a very large FP8 fallback
    # subgraph through Myelin instead of selecting the MHA kernel.
    attn.decomposable = not bool(quant_scales)
    if mask is not None and not causal:
        attn.mask = mask
    result = _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)
    if quant_scales and quant_scales.get("bmm2_output_scale") is not None:
        result = _add_fp8_qdq(
            network, result, float(quant_scales["bmm2_output_scale"]),
            output_dtype=output_dtype)
    return result


def _scalar_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: trt.DataType,
) -> trt.ITensor:
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    const = add_constant(
        network, shape, np.full(shape, value, dtype=np_dtype),
        dtype=np_dtype)
    if dtype == trt.bfloat16:
        const = network.add_cast(const, trt.bfloat16).get_output(0)
    return const


def add_tanh_softcap(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cap: float,
    *,
    scalar_shape: tuple[int, ...],
) -> trt.ITensor:
    """Apply ``tanh(tensor / cap) * cap`` using scalar broadcasting."""
    cap_t = _scalar_constant_for_trt_dtype(
        network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(
        tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(
        scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(
        capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


def _repeat_kv_heads_4d(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> trt.ITensor:
    if num_kv_heads == num_heads:
        return x_4d
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by "
            f"num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(
        network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0),
            shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1))
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_attention_core_with_logit_softcap(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: trt.ITensor | None,
    scale: float,
    logit_softcap: float,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim)
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim)

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(
        network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE,
        score_k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(
        network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE,
        v_4d, trt.MatrixOperation.NONE).get_output(0)
    return _cast_back_to_trt_dtype(network, context, output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_seq: int | None,
    kv_seq: int | None,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    logit_softcap: float | None = None,
    fp32_accumulation: bool = False,
    quant_scales: dict[str, float] | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Native IAttention for row-major [S, H * D] Q/K/V tensors.

    ``num_kv_heads`` can be smaller than ``num_heads`` for GQA/MQA. TRT
    native IAttention supports this directly, so callers should not expand K/V
    heads unless the model semantics require per-query-head K/V values.
    """
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network, q, num_heads, head_dim, sequence_length=q_seq,
        tag=None if tag is None else tag + ".q")
    k_4d = reshape_rows_to_heads_4d(
        network, k, kv_heads, head_dim, sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k")
    v_4d = reshape_rows_to_heads_4d(
        network, v, kv_heads, head_dim, sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v")
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if quant_scales:
            raise NotImplementedError(
                "quantized IAttention with logit_softcap is not supported")
        if causal:
            raise NotImplementedError(
                "logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network, q_4d, k_4d, v_4d,
            num_heads=num_heads, num_kv_heads=kv_heads, head_dim=head_dim,
            mask=mask, scale=scale, logit_softcap=float(logit_softcap))
    else:
        ctx_4d = add_attention_core(
            network, q_4d, k_4d, v_4d, causal=causal, mask=mask, scale=scale,
            fp32_accumulation=fp32_accumulation,
            quant_scales=quant_scales)
    return reshape_heads_4d_to_rows(
        network, ctx_4d, attention_size, sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx")


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


# ---------------------------------------------------------------------------
# TVM-FFI kernel bridge
# ---------------------------------------------------------------------------

def add_tvm_ffi_kernel(
    network: trt.INetworkDefinition,
    kernel_name: str,
    inputs: list[trt.ITensor],
    output_specs: list[dict],
    workspace_bytes: int = 0,
    extra_args: list[dict] | None = None,
) -> list[trt.ITensor]:
    """Add a TVM-FFI kernel call as a TRT plugin layer.

    Args:
        network: TRT network being built.
        kernel_name: TVM-FFI global function name (e.g. "my_ns.my_kernel").
        inputs: List of input ITensor objects.
        output_specs: List of dicts, one per output. Each dict has:
            - "dims": "same_as_input_N" or list of ints for fixed shape
            - "dtype": "float32" or "float16" (default "float32")
        workspace_bytes: Extra workspace bytes for the kernel (default 0).
        extra_args: Optional list of extra scalar/pointer args to pass after
            tensors. Each dict has "type" ("none"|"int"|"float"|"ptr") and
            optional "value".

    Returns:
        List of output ITensor objects.
    """
    import json

    registry = trt.get_plugin_registry()
    # TRT 11+ renamed get_plugin_creator → get_creator; keep both for back-compat.
    _get = getattr(registry, "get_creator", None) or registry.get_plugin_creator
    creator = _get("TvmFfiKernel", "1", "")
    if creator is None:
        raise RuntimeError(
            "TvmFfiKernel plugin not found in TRT registry. "
            "Ensure the C++ plugin is compiled with TRTMC_HAS_TVM_FFI=1."
        )

    spec_dict = {
        "num_inputs": len(inputs),
        "num_outputs": len(output_specs),
        "outputs": output_specs,
        "workspace_bytes": workspace_bytes,
    }
    if extra_args:
        spec_dict["extra_args"] = extra_args
    shape_spec = json.dumps(spec_dict)

    fields = [
        trt.PluginField("kernel_name", kernel_name.encode("utf-8"),
                         trt.PluginFieldType.CHAR),
        trt.PluginField("shape_spec", shape_spec.encode("utf-8"),
                         trt.PluginFieldType.CHAR),
    ]
    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin("tvm_ffi_kernel", fc)

    layer = network.add_plugin_v2(inputs, plugin)
    return [layer.get_output(i) for i in range(layer.num_outputs)]


# Graph Blocks


trt = trt_compat.get_trt()


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------

def make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul
    else:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
                dtype=dtype)
        return matmul


_make_matmul_fn = make_matmul_fn


# Utils


trt = trt_compat.get_trt()

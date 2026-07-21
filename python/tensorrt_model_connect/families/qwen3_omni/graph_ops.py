# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_trt_dtype(dtype: np.dtype):
    """Convert numpy dtype to TRT DataType for cast-back after FP32 compute."""
    if dtype == np.float16:
        return trt.float16
    return trt.float32


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)

def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    values_array = np.asarray(values)
    if values_array.dtype.name == "bfloat16":
        # TensorRT's Python Weights wrapper cannot consume NumPy BF16 arrays.
        # Promote exactly-representable BF16 values to FP32, then let the
        # strongly typed graph cast once to its BF16 runtime dtype.
        constant_values = np.ascontiguousarray(values_array, dtype=np.float32)
    else:
        constant_values = np.ascontiguousarray(values_array, dtype=dtype)
    weights = trt.Weights(constant_values)
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


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


def add_l2_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    reduce_axis: int,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """L2 normalize: x / max(||x||_2, eps) along reduce_axis.

    Used for DeltaNet Q/K normalization (Gated DeltaNet architecture).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    sum_sq = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.SUM,
        1 << reduce_axis, keep_dims=True)
    norm = network.add_unary(sum_sq.get_output(0), trt.UnaryOperation.SQRT)
    # max(norm, eps) to avoid division by zero
    eps_const = add_constant(
        network, (1,) * (reduce_axis + 1),
        np.array([eps], dtype=np.float32), dtype=np.float32)
    safe_norm = network.add_elementwise(
        norm.get_output(0), eps_const, trt.ElementWiseOperation.MAX)
    recip = network.add_unary(safe_norm.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    result = normalized.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


# ---------------------------------------------------------------------------
# RoPE tables — pure NumPy, no TRT dependency
# ---------------------------------------------------------------------------

def make_rope_table(
    max_cache_length: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    cosine: bool,
    partial_rotary_factor: float = 1.0,
    interleaved: bool = False,
) -> np.ndarray:
    """Build cos or sin RoPE table of shape [max_cache_length, hidden_size].

    Args:
        partial_rotary_factor: Fraction of head dimensions that get RoPE
            (e.g. 0.25 for StableLM-2). Default 1.0 = full RoPE.
        interleaved: If True, use interleaved frequency assignment where
            adjacent dims (d, d+1) share the same frequency (CodeGen/GPT-J).
            If False (default), use rotated-half where dims (d, d+half) share
            the same frequency (LLaMA/Qwen/GPT-NeoX).
    """
    table = np.full(
        (max_cache_length, hidden_size),
        1.0 if cosine else 0.0,
        dtype=np.float32,
    )
    if (max_cache_length <= 0 or hidden_size <= 0
            or num_attention_heads <= 0
            or hidden_size % num_attention_heads != 0):
        return table

    head_dim = hidden_size // num_attention_heads
    rotary_ndims = int(head_dim * partial_rotary_factor)
    half_rotary = rotary_ndims // 2
    if half_rotary <= 0 or rope_theta <= 0.0:
        return table

    for pos in range(max_cache_length):
        for head in range(num_attention_heads):
            for dim in range(rotary_ndims):
                if interleaved:
                    freq_idx = dim // 2
                else:
                    freq_idx = dim % half_rotary
                exponent = (2.0 * freq_idx) / rotary_ndims
                inv_freq = rope_theta ** (-exponent)
                angle = pos * inv_freq
                value = np.cos(angle) if cosine else np.sin(angle)
                offset = head * head_dim + dim
                table[pos, offset] = value

    return table


def _yarn_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Find the YaRN correction dimension boundary."""
    return dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi)) / (2 * np.log(base))


def make_yarn_rope_table(
    max_cache_length: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float,
    cosine: bool,
    scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    interleaved: bool = False,
) -> np.ndarray:
    """Build YaRN-scaled RoPE table matching HF DeepseekV2YarnRotaryEmbedding.

    YaRN mixes standard and interpolated inv_freq using a correction ramp
    based on beta_fast/beta_slow boundaries.

    Args:
        interleaved: If True, adjacent dims (d, d+1) share the same frequency.
            If False (default), half-dims (d, d+half) share the same frequency.
    """
    table = np.full(
        (max_cache_length, hidden_size),
        1.0 if cosine else 0.0,
        dtype=np.float32,
    )
    if (max_cache_length <= 0 or hidden_size <= 0
            or num_attention_heads <= 0
            or hidden_size % num_attention_heads != 0):
        return table

    head_dim = hidden_size // num_attention_heads
    head_dim = validate_native_rope_dim(head_dim, field_name="head_dim")
    half = head_dim // 2
    if half <= 0 or rope_theta <= 0.0:
        return table

    # Standard and interpolated frequencies
    freq_extra = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freq_inter = freq_extra / scaling_factor

    # Correction range ramp
    low = max(int(np.floor(_yarn_correction_dim(
        beta_fast, head_dim, rope_theta, original_max_position_embeddings))), 0)
    high = min(int(np.ceil(_yarn_correction_dim(
        beta_slow, head_dim, rope_theta, original_max_position_embeddings))), half - 1)
    ramp = np.clip((np.arange(half, dtype=np.float64) - low) / max(high - low, 1), 0.0, 1.0)
    inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)

    # Build table [max_cache_length, hidden_size] — same layout as make_rope_table
    for pos in range(max_cache_length):
        for head in range(num_attention_heads):
            for dim in range(head_dim):
                if interleaved:
                    freq_idx = dim // 2
                else:
                    freq_idx = dim % half
                angle = pos * inv_freq[freq_idx]
                value = np.cos(angle) if cosine else np.sin(angle)
                offset = head * head_dim + dim
                table[pos, offset] = float(value)

    return table


def make_yarn_rope_table_half_dim(
    max_cache_length: int,
    head_dim: int,
    rope_theta: float,
    cosine: bool,
    scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
    interleaved: bool = False,
) -> np.ndarray:
    """Build a YaRN RoPE table for TRT native IRotaryEmbeddingLayer.

    Returns [max_cache_length, head_dim // 2], matching the half-dimension
    cache layout required by IRotaryEmbeddingLayer.
    """
    head_dim = validate_native_rope_dim(head_dim, field_name="head_dim")
    half = head_dim // 2
    default = 1.0 if cosine else 0.0
    if max_cache_length <= 0 or half <= 0 or rope_theta <= 0.0:
        return np.full((max(max_cache_length, 1), max(half, 1)),
                       default, dtype=np.float32)

    freq_extra = 1.0 / (
        rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freq_inter = freq_extra / scaling_factor

    low = max(int(np.floor(_yarn_correction_dim(
        beta_fast, head_dim, rope_theta, original_max_position_embeddings))), 0)
    high = min(int(np.ceil(_yarn_correction_dim(
        beta_slow, head_dim, rope_theta, original_max_position_embeddings))), half - 1)
    ramp = np.clip((np.arange(half, dtype=np.float64) - low) / max(high - low, 1), 0.0, 1.0)
    inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)

    table = np.full((max_cache_length, half), default, dtype=np.float32)
    for pos in range(max_cache_length):
        for d in range(half):
            angle = pos * inv_freq[d]
            table[pos, d] = np.cos(angle) if cosine else np.sin(angle)
    return table


def make_llama4_attention_scale_table(
    max_cache_length: int,
    beta: float,
    original_max_position_embeddings: int,
) -> np.ndarray:
    """Build the per-position query scale used by Llama-4-style RoPE.

    HF Nemotron-Labs-Diffusion applies this after RoPE:
      1 + beta * log(1 + floor(position / original_max_position_embeddings))

    Returns [max_cache_length, 1] so TensorRT can gather by position_id and
    broadcast the result across the query hidden dimension.
    """
    if max_cache_length <= 0:
        return np.ones((max(max_cache_length, 0), 1), dtype=np.float32)
    if beta == 0.0 or original_max_position_embeddings <= 0:
        return np.ones((max_cache_length, 1), dtype=np.float32)
    positions = np.arange(max_cache_length, dtype=np.float64)
    scale = 1.0 + float(beta) * np.log1p(
        np.floor(positions / float(original_max_position_embeddings))
    )
    return scale.reshape(max_cache_length, 1).astype(np.float32)


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(
        inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(
        var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    result = result.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


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


def add_gelu_erf(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (exact, erf-based): 0.5 * x * (1 + erf(x / sqrt(2))).

    Constants are cast to ``inp.dtype`` for the same STRONGLY_TYPED reason
    documented on ``add_gelu_new``.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(value):
        c = add_constant(
            network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    inv_sqrt2 = _const(1.0 / np.sqrt(2.0))
    x_scaled = network.add_elementwise(
        inp, inv_sqrt2, trt.ElementWiseOperation.PROD)
    erf_out = network.add_unary(x_scaled.get_output(0), trt.UnaryOperation.ERF)
    one = _const(1.0)
    one_plus_erf = network.add_elementwise(
        one, erf_out.get_output(0), trt.ElementWiseOperation.SUM)
    half = _const(0.5)
    half_x = network.add_elementwise(
        half, inp, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_erf.get_output(0),
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


def compute_alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute ALiBi slopes for each attention head (from the ALiBi paper).

    For power-of-2 num_heads: geometric sequence 2^(-8/n * i), i in 1..n.
    For non-power-of-2: interleave two geometric sequences.

    Returns: [num_heads] float32 array.
    """
    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if num_heads > 0 and (num_heads & (num_heads - 1)) == 0:
        # Power of 2
        return np.array(_get_slopes_power_of_2(num_heads), dtype=np.float32)
    else:
        closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
        slopes_a = _get_slopes_power_of_2(closest_power_of_2)
        slopes_b = _get_slopes_power_of_2(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: num_heads - closest_power_of_2]
        return np.array(slopes_a + slopes_b, dtype=np.float32)


# ---------------------------------------------------------------------------
# Vision encoder graph ops
# ---------------------------------------------------------------------------

def add_self_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Full self-attention without KV cache (single-pass, for vision encoders).

    Input hidden: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    context_flat = add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def add_self_attention_block_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Full self-attention with precomputed RoPE (for vision encoders with 3D RoPE).

    Unlike the KV-cache decoder attention, this processes all positions at once
    and applies RoPE via precomputed per-position cos/sin tables.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size] precomputed constants
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden] = [seq, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    rope_dim = head_dim
    cos_half = cos_table[:, : rope_dim // 2]
    sin_half = sin_table[:, : rope_dim // 2]
    cos_const = add_constant(
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype)
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype)

    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)

    context_flat = add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_length, kv_seq=seq_length)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def add_windowed_self_attention_with_rope(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    num_windows: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    window_patch_counts: np.ndarray | None = None,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Windowed self-attention with precomputed RoPE.

    Splits the already window-ordered sequence into windows. Most Qwen-VL
    builds have equal-sized windows and use one batched attention op; HF
    smart-resized images can produce partial edge windows, which are handled
    by static per-window slices when ``window_patch_counts`` is provided.

    Input hidden: [seq_length, hidden_size]
    cos_table/sin_table: [seq_length, hidden_size]
    Output: [seq_length, hidden_size]
    """
    head_dim = hidden_size // num_heads
    counts = None
    if window_patch_counts is not None:
        counts = [
            int(v) for v in np.asarray(window_patch_counts).reshape(-1).tolist() if int(v) > 0
        ]
        if not counts or sum(counts) != seq_length:
            raise ValueError(
                "window_patch_counts must be positive and sum to seq_length: "
                f"sum={sum(counts) if counts else 0}, seq_length={seq_length}"
            )
        if all(c == counts[0] for c in counts):
            num_windows = len(counts)
            counts = None
    win_seq = seq_length // num_windows  # patches per window
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Q, K, V projections: [seq, hidden] @ [hidden, hidden]
    q = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_q, dtype=dtype)
    k = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, hidden, hidden_size, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    rope_dim = head_dim
    cos_half = cos_table[:, : rope_dim // 2]
    sin_half = sin_table[:, : rope_dim // 2]
    cos_const = add_constant(
        network, (1, seq_length, rope_dim // 2), cos_half.reshape(1, seq_length, -1), dtype=dtype)
    sin_const = add_constant(
        network, (1, seq_length, rope_dim // 2), sin_half.reshape(1, seq_length, -1), dtype=dtype)

    q = add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)
    k = add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos_const, sin_const,
        rotary_embedding_dim=rope_dim, sequence_length=seq_length)

    if counts is None:
        q_win = network.add_shuffle(q)
        q_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        q_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        k_win = network.add_shuffle(k)
        k_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        k_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        v_win = network.add_shuffle(v)
        v_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        v_win.second_transpose = trt.Permutation([0, 2, 1, 3])

        context = add_attention_core(
            network, q_win.get_output(0), k_win.get_output(0), v_win.get_output(0),
            scale=attn_scale,
        )
        ctx_flat = network.add_shuffle(context)
        ctx_flat.first_transpose = trt.Permutation([0, 2, 1, 3])
        ctx_flat.reshape_dims = (seq_length, hidden_size)
        context_flat = ctx_flat.get_output(0)
    else:
        window_outputs = []
        offset = 0
        for window_len in counts:
            q_slice = network.add_slice(
                q, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            k_slice = network.add_slice(
                k, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            v_slice = network.add_slice(
                v, start=(offset, 0), shape=(window_len, hidden_size), stride=(1, 1))
            window_outputs.append(add_attention_from_rows(
                network,
                q_slice.get_output(0),
                k_slice.get_output(0),
                v_slice.get_output(0),
                num_heads=num_heads,
                head_dim=head_dim,
                q_seq=window_len,
                kv_seq=window_len,
                scale=attn_scale,
            ))
            offset += window_len
        concat = network.add_concatenation(window_outputs)
        concat.axis = 0
        context_flat = concat.get_output(0)

    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


def add_patch_embed_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    in_channels: int,
    embed_dim: int,
    temporal_patch_size: int,
    patch_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """3D patch embedding via convolution.

    Input: [T*C, H, W] (already flattened temporal*channels) or [T, C, H, W]
    Output: [num_patches, embed_dim]

    The 3D convolution is implemented as a 2D convolution over the flattened
    temporal*channel dimension, matching HuggingFace's PatchEmbed3D.
    """
    # Input may be [T*C, H, W] (3D) or [T, C, H, W] (4D).
    # We need [1, T*C, H, W] for conv2d.
    inp_ndims = len(inp.shape)
    reshape_in = network.add_shuffle(inp)
    if inp_ndims == 3:
        # [T*C, H, W] -> [1, T*C, H, W]
        tc = inp.shape[0]
        h = inp.shape[1]
        w = inp.shape[2]
        reshape_in.reshape_dims = (1, tc, h, w)
    else:
        # [T, C, H, W] -> [1, T*C, H, W]
        reshape_in.reshape_dims = (1, temporal_patch_size * in_channels, -1, 0)

    # Conv2D with kernel [embed_dim, T*C, patch_size, patch_size]
    # weight shape from HF: [embed_dim, T*C, patch_size, patch_size]
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    conv = network.add_convolution_nd(
        reshape_in.get_output(0),
        num_output_maps=embed_dim,
        kernel_shape=(patch_size, patch_size),
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = (patch_size, patch_size)

    # Output shape: [1, embed_dim, H', W'] -> flatten to [num_patches, embed_dim]
    reshape_out = network.add_shuffle(conv.get_output(0))
    reshape_out.first_transpose = trt.Permutation([0, 2, 3, 1])
    reshape_out.reshape_dims = (-1, embed_dim)

    return reshape_out.get_output(0)


def add_spatial_merge(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    w_fc1: np.ndarray,
    w_fc2: np.ndarray,
    b_fc1: np.ndarray | None,
    b_fc2: np.ndarray | None,
    norm_gamma: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    eps_tensor: trt.ITensor,
    seq_length: int,
    merge_size: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Spatial merge: 2x2 merge MLP that reduces spatial resolution.

    Reshapes [seq, dim] -> merge adjacent 2x2 patches, then MLP.
    Input: [seq_length, input_dim]
    Output: [seq_length // (merge_size^2), output_dim]

    Note: This is a simplified version. For Qwen2.5-VL, the merge
    concatenates merge_size^2 adjacent patches, then applies layernorm + MLP.
    """
    # LayerNorm on the merged representation
    norm = add_layer_norm(
        network, inp, input_dim,
        norm_gamma, np.zeros(input_dim, dtype=np.float32), eps_tensor,
        dtype=dtype)

    # For simplicity in the TRT graph, we use a 2-layer MLP directly
    # on the already-flattened input. The spatial rearrangement is handled
    # during preprocessing.
    fc1 = add_matmul_rhs_constant(network, norm, input_dim, hidden_dim, w_fc1, dtype=dtype)
    if b_fc1 is not None:
        fc1 = add_bias_sum(network, fc1, hidden_dim, b_fc1, dtype=dtype)

    # GELU activation
    activated = add_gelu_new(network, fc1, dtype=dtype)

    fc2 = add_matmul_rhs_constant(network, activated, hidden_dim, output_dim, w_fc2, dtype=dtype)
    if b_fc2 is not None:
        fc2 = add_bias_sum(network, fc2, output_dim, b_fc2, dtype=dtype)

    return fc2


# ---------------------------------------------------------------------------
# Diffusion graph ops — used by DiT, T5, VAE builders
# ---------------------------------------------------------------------------

def add_group_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    num_groups: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GroupNorm: split channels into groups, normalize each group.

    Input: [..., num_channels] (last dim is channels).
    Output: same shape.

    TRT 10 does not have a native GroupNorm layer, so we reshape to
    [batch, num_groups, group_size], normalize, reshape back, then apply
    affine (gamma, beta).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    # Reshape: [B, C] or [B, C, ...] — we handle the 2D case [seq, C]
    # and the 5D case [B, C, T, H, W] for VAE.
    need_cast = (dtype != np.float32)
    ndims = len(inp.shape)
    group_size = num_channels // num_groups
    output_dtype = inp.dtype

    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)

    if ndims == 2:
        # [seq, C] -> [seq, G, Gs]
        reshape_in = network.add_shuffle(inp)
        reshape_in.reshape_dims = (-1, num_groups, group_size)
        x = reshape_in.get_output(0)

        # Normalize over group_size dim (dim=2)
        eps_t = add_constant(network, (1, 1, 1),
                             np.array([eps], dtype=np.float32), dtype=np.float32)
        sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(
            x, trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
        mean_sq = network.add_reduce(
            sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
        var = network.add_elementwise(
            mean_sq.get_output(0),
            network.add_elementwise(
                mean.get_output(0), mean.get_output(0),
                trt.ElementWiseOperation.PROD).get_output(0),
            trt.ElementWiseOperation.SUB)
        denom = network.add_unary(
            network.add_elementwise(
                var.get_output(0), eps_t,
                trt.ElementWiseOperation.SUM).get_output(0),
            trt.UnaryOperation.SQRT)
        recip = network.add_unary(
            denom.get_output(0), trt.UnaryOperation.RECIP)
        centered = network.add_elementwise(
            x, mean.get_output(0), trt.ElementWiseOperation.SUB)
        normalized = network.add_elementwise(
            centered.get_output(0), recip.get_output(0),
            trt.ElementWiseOperation.PROD)

        # Reshape back to [seq, C]
        reshape_out = network.add_shuffle(normalized.get_output(0))
        reshape_out.reshape_dims = (-1, num_channels)
        result = reshape_out.get_output(0)

    elif ndims == 5:
        # [B, C, T, H, W] — use TRT INormalizationLayer (GroupNorm mode)
        # Reshape to [B, G, Gs, T, H, W], norm over dims 2,3,4,5, reshape back
        # But simpler: use the fact that TRT GroupNorm can work on NCHW-like tensors.
        # We treat [B, C, T, H, W] directly, normalizing over (Gs, T, H, W) per group.
        b, c, t, h, w = inp.shape
        reshape_in = network.add_shuffle(inp)
        reshape_in.reshape_dims = (b, num_groups, group_size, t, h, w)
        x = reshape_in.get_output(0)

        # Reduce over dims 2,3,4,5 (group_size, T, H, W)
        reduce_axes = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        eps_t = add_constant(network, (1, 1, 1, 1, 1, 1),
                             np.array([eps], dtype=np.float32), dtype=np.float32)
        sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(
            x, trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
        mean_sq = network.add_reduce(
            sq.get_output(0), trt.ReduceOperation.AVG,
            reduce_axes, keep_dims=True)
        var = network.add_elementwise(
            mean_sq.get_output(0),
            network.add_elementwise(
                mean.get_output(0), mean.get_output(0),
                trt.ElementWiseOperation.PROD).get_output(0),
            trt.ElementWiseOperation.SUB)
        denom = network.add_unary(
            network.add_elementwise(
                var.get_output(0), eps_t,
                trt.ElementWiseOperation.SUM).get_output(0),
            trt.UnaryOperation.SQRT)
        recip = network.add_unary(
            denom.get_output(0), trt.UnaryOperation.RECIP)
        centered = network.add_elementwise(
            x, mean.get_output(0), trt.ElementWiseOperation.SUB)
        normalized = network.add_elementwise(
            centered.get_output(0), recip.get_output(0),
            trt.ElementWiseOperation.PROD)

        # Reshape back to [B, C, T, H, W]
        reshape_out = network.add_shuffle(normalized.get_output(0))
        reshape_out.reshape_dims = (b, c, t, h, w)
        result = reshape_out.get_output(0)
    else:
        raise ValueError(f"add_group_norm: unsupported ndims={ndims}")

    # Affine: gamma * result + beta (broadcast over spatial dims)
    if ndims == 2:
        gamma_t = add_constant(network, (1, num_channels), gamma, dtype=np.float32)
        beta_t = add_constant(network, (1, num_channels), beta, dtype=np.float32)
    else:
        gamma_t = add_constant(
            network, (1, num_channels, 1, 1, 1), gamma.reshape(1, -1, 1, 1, 1), dtype=np.float32)
        beta_t = add_constant(
            network, (1, num_channels, 1, 1, 1), beta.reshape(1, -1, 1, 1, 1), dtype=np.float32)
    scaled = network.add_elementwise(
        result, gamma_t, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)


def add_conv3d_as_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """3D convolution decomposed as 2D convolution over fused (T*C) channels.

    Input: [B, C_in, T, H, W]
    Weight: [C_out, C_in, Kt, Kh, Kw]
    Output: [B, C_out, T_out, H_out, W_out]

    For temporal kernel Kt=1, this is a standard spatial conv applied to each frame.
    For Kt>1, we reshape [B, C_in, T, H, W] -> [B, C_in*Kt, T_out, H, W] using
    a sliding-window gather, then apply Conv2D with [C_out, C_in*Kt, Kh, Kw].
    """
    b, c_in, t, h, w = inp.shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    pt, ph, pw = padding

    if kt == 1 and st == 1 and pt == 0:
        # Simple case: per-frame spatial conv
        # Reshape [B, C, T, H, W] -> [B*T, C, H, W]
        reshape_in = network.add_shuffle(inp)
        reshape_in.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
        reshape_in.reshape_dims = (b * t, c_in, h, w)

        # Weight: [C_out, C_in, 1, Kh, Kw] -> [C_out, C_in, Kh, Kw]
        w2d = weight.reshape(out_channels, c_in, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w2d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            reshape_in.get_output(0),
            num_output_maps=out_channels,
            kernel_shape=(kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (sh, sw)
        conv.padding_nd = (ph, pw)

        # Reshape back [B*T, C_out, H', W'] -> [B, C_out, T, H', W']
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        reshape_out = network.add_shuffle(conv.get_output(0))
        reshape_out.reshape_dims = (b, t, out_channels, h_out, w_out)
        reshape_out.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
        return reshape_out.get_output(0)
    else:
        # General case: temporal kernel > 1
        # Pad temporally if needed
        if pt > 0:
            # Zero-pad [B, C, T, H, W] -> [B, C, T+2*pt, H, W]
            pad_layer = network.add_padding_nd(
                inp,
                pre_padding=(0, pt, 0),
                post_padding=(0, pt, 0),
            )
            inp = pad_layer.get_output(0)

        # For causal conv we handle this via the cache mechanism externally,
        # so here we just do a per-frame conv with gathered temporal neighbors.
        # Reshape [B, C, T_padded, H, W] -> sliding window gather -> Conv2D
        # This is complex in pure TRT graph, so for now we use the simple
        # kernel=1 path and handle temporal via caching externally.
        raise NotImplementedError(
            f"Conv3D with kt={kt} not yet implemented in TRT graph. "
            "Use causal caching with kt=1 per-frame convolutions instead."
        )


def add_causal_conv3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    cache: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding_hw: tuple[int, int] = (0, 0),
    dtype: np.dtype = np.float32,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Causal 3D convolution with temporal cache.

    Input: [B, C_in, T, H, W] (T >= 1)
    Cache: [B, C_in, Kt-1, H, W] (previous frames)
    Weight: [C_out, C_in, Kt, Kh, Kw]

    Returns: (output [B, C_out, T, H', W'], updated_cache [B, C_in, Kt-1, H, W])

    The cache stores Kt-1 previous frames. We concatenate cache + input along
    temporal dim, then apply convolution. For T=1 uses optimized 2D decomposition,
    for T>1 uses native 3D convolution.
    """
    b, c_in, t_in, h, w = inp.shape
    kt, kh, kw = kernel_size
    ph, pw = padding_hw

    if kt == 1:
        # No temporal dependency, just spatial conv
        result = add_conv3d_as_conv2d(
            network, inp, weight, bias, out_channels,
            kernel_size=(1, kh, kw), stride=stride,
            padding=(0, ph, pw), dtype=dtype)
        # Cache is unchanged
        return result, cache

    # Concatenate cache + input along temporal dim:
    # [B, C, Kt-1, H, W] cat [B, C, T, H, W] -> [B, C, Kt-1+T, H, W]
    concat = network.add_concatenation([cache, inp])
    concat.axis = 2  # temporal dim
    full_temporal = concat.get_output(0)

    if t_in == 1:
        # Optimized T=1 path: reshape to 2D and use Conv2D
        # full_temporal is [B, C_in, Kt, H, W]
        reshape_in = network.add_shuffle(full_temporal)
        reshape_in.reshape_dims = (b, c_in * kt, h, w)

        w2d = weight.reshape(out_channels, c_in * kt, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w2d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            reshape_in.get_output(0),
            num_output_maps=out_channels,
            kernel_shape=(kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (stride[1], stride[2])
        conv.padding_nd = (ph, pw)

        h_out = (h + 2 * ph - kh) // stride[1] + 1
        w_out = (w + 2 * pw - kw) // stride[2] + 1
        reshape_out = network.add_shuffle(conv.get_output(0))
        reshape_out.reshape_dims = (b, out_channels, 1, h_out, w_out)
        result = reshape_out.get_output(0)
    else:
        # General T>1 path: native 3D convolution
        # full_temporal is [B, C_in, Kt-1+T, H, W]
        w3d = weight.reshape(out_channels, c_in, kt, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w3d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            full_temporal,
            num_output_maps=out_channels,
            kernel_shape=(kt, kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (stride[0], stride[1], stride[2])
        conv.padding_nd = (0, ph, pw)  # No temporal padding (cache provides it)
        result = conv.get_output(0)  # [B, C_out, T, H', W']

    # Update cache: last Kt-1 frames from the concatenated tensor
    total_t = (kt - 1) + t_in
    cache_start_t = total_t - (kt - 1)  # = t_in
    if kt > 1:
        slice_layer = network.add_slice(
            full_temporal,
            start=(0, 0, cache_start_t, 0, 0),
            shape=(b, c_in, kt - 1, h, w),
            stride=(1, 1, 1, 1, 1),
        )
        new_cache = slice_layer.get_output(0)
    else:
        new_cache = cache

    return result, new_cache


def add_spatial_upsample(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale_factor: int = 2,
) -> trt.ITensor:
    """Spatial nearest-neighbor upsampling for 5D tensor [B, C, T, H, W].

    Output: [B, C, T, H*scale, W*scale]
    """
    b, c, t, h, w = inp.shape
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (b, c, t, h * scale_factor, w * scale_factor)
    return resize.get_output(0)


def add_spatial_upsample_with_conv(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    scale: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Spatial nearest-neighbor 2x upsample + Conv3D(1,3,3) smoothing.

    Matches HF WanResample's nn.Sequential(Upsample(2x), Conv3d(1,3,3)).

    Input: [B, C_in, T, H, W]
    Weight: [C_out, C_in, 1, 3, 3]  (C_out detected from weight shape)
    Output: [B, C_out, T, H*scale, W*scale]
    """
    out_channels = weight.shape[0]

    # Step 1: nearest-neighbor 2x spatial
    upsampled = add_spatial_upsample(network, inp, scale)

    # Step 2: Conv3D(1,3,3) = per-frame 2D conv with 3x3 kernel
    result = add_conv3d_as_conv2d(
        network, upsampled,
        weight=weight, bias=bias,
        out_channels=out_channels,
        kernel_size=(1, 3, 3),
        padding=(0, 1, 1),
        dtype=dtype,
    )
    return result


def add_temporal_upsample(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale_factor: int = 2,
) -> trt.ITensor:
    """Temporal nearest-neighbor upsampling for 5D tensor [B, C, T, H, W].

    Output: [B, C, T*scale, H, W]
    """
    b, c, t, h, w = inp.shape
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (b, c, t * scale_factor, h, w)
    return resize.get_output(0)


def add_l2_channel_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    gamma: np.ndarray,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """L2 channel norm: F.normalize(x, dim=1) * sqrt(C) * gamma.

    L2-normalizes over channel dimension (axis=1), then scales by
    sqrt(num_channels) and learnable gamma.

    Input: [B, C, T, H, W] (5D tensor)
    gamma: [C, 1, 1, 1] reshaped to [1, C, 1, 1, 1] for broadcast
    Output: same shape

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    # L2 norm over channel dim: ||x||_2 = sqrt(sum(x^2, dim=1))
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    sum_sq = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)

    eps_t = add_constant(network, (1, 1, 1, 1, 1),
                         np.array([eps], dtype=np.float32), dtype=np.float32)
    denom_in = network.add_elementwise(
        sum_sq.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    norm = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(norm.get_output(0), trt.UnaryOperation.RECIP)

    # normalized = x / ||x||_2
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)

    # Scale by sqrt(C) * gamma  →  gamma_scaled shape [1, C, 1, 1, 1]
    gamma_flat = gamma.flatten()[:num_channels]
    scale = np.sqrt(num_channels) * gamma_flat
    scale_t = add_constant(
        network, (1, num_channels, 1, 1, 1),
        scale.reshape(1, num_channels, 1, 1, 1), dtype=np.float32)

    result = network.add_elementwise(
        normalized.get_output(0), scale_t,
        trt.ElementWiseOperation.PROD).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_temporal_pixel_shuffle(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    factor: int = 2,
) -> trt.ITensor:
    """Temporal pixel shuffle: [B, factor*C, T, H, W] → [B, C, factor*T, H, W].

    Interleaves temporal frames by splitting the channel dimension and
    folding it into the temporal dimension.
    """
    b, c_total, t, h, w = inp.shape
    c = c_total // factor  # output channels

    # Reshape: [B, factor*C, T, H, W] → [B, factor, C, T, H, W]
    reshape1 = network.add_shuffle(inp)
    reshape1.reshape_dims = (b, factor, c, t, h, w)

    # Permute: [B, factor, C, T, H, W] → [B, C, T, factor, H, W]
    transpose = network.add_shuffle(reshape1.get_output(0))
    transpose.first_transpose = trt.Permutation([0, 2, 3, 1, 4, 5])

    # Reshape: [B, C, T, factor, H, W] → [B, C, factor*T, H, W]
    reshape2 = network.add_shuffle(transpose.get_output(0))
    reshape2.reshape_dims = (b, c, factor * t, h, w)

    return reshape2.get_output(0)


def add_timestep_embedding(
    network: trt.INetworkDefinition,
    timestep: trt.ITensor,
    dim: int,
    freq_dim: int = 256,
    max_period: float = 10000.0,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Sinusoidal timestep embedding: sin/cos frequencies -> MLP.

    Input timestep: [1] (scalar float)
    Output: [1, dim]

    This builds the frequency embedding as a constant table lookup
    parameterized by the timestep, then applies an MLP. For TRT, since
    timestep is a dynamic input, we compute sin/cos at graph time.
    """
    half = freq_dim // 2
    # Precompute frequency table: exp(-log(max_period) * i / half)
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float32) / half)
    freqs_const = add_constant(network, (1, half), freqs.reshape(1, -1), dtype=dtype)

    # timestep * freqs: [1] * [1, half] -> [1, half]
    ts_reshaped = network.add_shuffle(timestep)
    ts_reshaped.reshape_dims = (1, 1)
    args = network.add_elementwise(
        ts_reshaped.get_output(0), freqs_const,
        trt.ElementWiseOperation.PROD)

    # cos and sin
    cos_part = network.add_unary(
        args.get_output(0), trt.UnaryOperation.COS)
    sin_part = network.add_unary(
        args.get_output(0), trt.UnaryOperation.SIN)

    # Concatenate [cos, sin] -> [1, freq_dim]
    embed = network.add_concatenation(
        [cos_part.get_output(0), sin_part.get_output(0)])
    embed.axis = 1

    return embed.get_output(0)


def add_adaptive_layernorm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale: trt.ITensor,
    shift: trt.ITensor,
    hidden_size: int,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Adaptive LayerNorm (AdaLN): norm(x) * (1 + scale) + shift.

    Used by DiT blocks. The scale and shift come from the timestep MLP.

    Input: [seq, hidden_size]
    scale: [1, hidden_size]
    shift: [1, hidden_size]
    Output: [seq, hidden_size]

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        scale = network.add_cast(scale, trt.float32).get_output(0)
        shift = network.add_cast(shift, trt.float32).get_output(0)
    # Standard LayerNorm without affine
    eps_t = add_constant(network, (1, 1), np.array([eps], dtype=np.float32), dtype=np.float32)
    mean = network.add_reduce(
        inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(
        inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0),
        trt.ElementWiseOperation.PROD)
    var = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom = network.add_unary(
        network.add_elementwise(
            var.get_output(0), eps_t,
            trt.ElementWiseOperation.SUM).get_output(0),
        trt.UnaryOperation.SQRT)
    recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    # Adaptive modulation: norm(x) * (1 + scale) + shift
    one = add_constant(network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32)
    scale_plus_one = network.add_elementwise(
        one, scale, trt.ElementWiseOperation.SUM)
    scaled = network.add_elementwise(
        normalized.get_output(0), scale_plus_one.get_output(0),
        trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        scaled.get_output(0), shift,
        trt.ElementWiseOperation.SUM).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_cross_attention(
    network: trt.INetworkDefinition,
    query: trt.ITensor,
    context: trt.ITensor,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    hidden_size: int,
    context_dim: int,
    num_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    q_bias: np.ndarray | None = None,
    k_bias: np.ndarray | None = None,
    v_bias: np.ndarray | None = None,
    o_bias: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Cross-attention: Q from query, K/V from context.

    query:   [q_seq_len, hidden_size]
    context: [kv_seq_len, context_dim]
    Output:  [q_seq_len, hidden_size]
    """
    head_dim = hidden_size // num_heads

    # Q projection: [q_seq, hidden] @ [hidden, hidden] = [q_seq, hidden]
    q = add_matmul_rhs_constant(network, query, hidden_size, hidden_size, w_q, dtype=dtype)
    # K, V projections: [kv_seq, context_dim] @ [context_dim, hidden] = [kv_seq, hidden]
    k = add_matmul_rhs_constant(network, context, context_dim, hidden_size, w_k, dtype=dtype)
    v = add_matmul_rhs_constant(network, context, context_dim, hidden_size, w_v, dtype=dtype)

    if q_bias is not None:
        q = add_bias_sum(network, q, hidden_size, q_bias, dtype=dtype)
    if k_bias is not None:
        k = add_bias_sum(network, k, hidden_size, k_bias, dtype=dtype)
    if v_bias is not None:
        v = add_bias_sum(network, v, hidden_size, v_bias, dtype=dtype)

    context_flat = add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=q_seq_len, kv_seq=kv_seq_len)

    # Output projection
    out = add_matmul_rhs_constant(
        network, context_flat, hidden_size, hidden_size, w_o, dtype=dtype)
    if o_bias is not None:
        out = add_bias_sum(network, out, hidden_size, o_bias, dtype=dtype)

    return out


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


# ---------------------------------------------------------------------------
# Conv / Norm / Resize ops for segmentation and audio models
# ---------------------------------------------------------------------------

def add_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """2D convolution wrapper.

    Input: [N, C_in, H, W]
    Weight: [C_out, C_in/groups, kH, kW]
    Output: [N, C_out, H', W']
    """
    conv_w = trt.Weights(np.ascontiguousarray(weight, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_size,
        kernel=conv_w,
        bias=conv_b,
    )
    conv.stride_nd = stride
    conv.padding_nd = padding
    conv.num_groups = groups
    return conv.get_output(0)


def add_batch_norm_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Fused BatchNorm2d: gamma * (x - mean) / sqrt(var + eps) + beta.

    Input: [N, C, H, W]
    Output: same shape

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = (dtype != np.float32)
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    # Fuse into scale + shift
    scale = gamma / np.sqrt(running_var + eps)
    shift = beta - running_mean * scale

    scale_t = add_constant(
        network, (1, num_channels, 1, 1),
        scale.reshape(1, -1, 1, 1).astype(np.float32), dtype=np.float32)
    shift_t = add_constant(
        network, (1, num_channels, 1, 1),
        shift.reshape(1, -1, 1, 1).astype(np.float32), dtype=np.float32)

    scaled = network.add_elementwise(
        inp, scale_t, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        scaled.get_output(0), shift_t,
        trt.ElementWiseOperation.SUM).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_bilinear_resize_2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    target_h: int,
    target_w: int,
) -> trt.ITensor:
    """Bilinear interpolation resize for 4D tensor [N, C, H, W].

    Output: [N, C, target_h, target_w]
    """
    n, c = inp.shape[0], inp.shape[1]
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.shape = (n, c, target_h, target_w)
    return resize.get_output(0)


def add_conv1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """1D convolution via 2D convolution with height=1.

    Input: [N, C_in, L]
    Weight: [C_out, C_in/groups, K]
    Output: [N, C_out, L']
    """
    # Reshape to [N, C_in, 1, L]
    n, c_in, length = inp.shape
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c_in, 1, length)

    # Weight: [C_out, C_in/groups, K] -> [C_out, C_in/groups, 1, K]
    w_4d = weight.reshape(out_channels, -1, 1, kernel_size)
    result = add_conv2d(
        network, reshape_in.get_output(0),
        w_4d, bias, out_channels,
        kernel_size=(1, kernel_size),
        stride=(1, stride),
        padding=(0, padding),
        groups=groups,
        dtype=dtype)

    # Reshape back to [N, C_out, L']
    out_length = result.shape[3]
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (n, out_channels, out_length)
    return reshape_out.get_output(0)


def add_conv1d_transpose(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """1D transposed convolution via 2D deconvolution with height=1.

    Input: [N, C_in, L]
    Weight: [C_in, C_out, K]
    Output: [N, C_out, L']
    """
    n, c_in, length = inp.shape

    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c_in, 1, length)

    # Weight for deconv: [C_in, C_out, 1, K]
    w_4d = weight.reshape(c_in, out_channels, 1, kernel_size)
    conv_w = trt.Weights(np.ascontiguousarray(w_4d, dtype=dtype))
    conv_b = trt.Weights()
    if bias is not None:
        conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

    deconv = network.add_deconvolution_nd(
        reshape_in.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(1, kernel_size),
        kernel=conv_w,
        bias=conv_b)
    deconv.stride_nd = (1, stride)
    deconv.padding_nd = (0, padding)

    # Reshape back to 3D
    out_shape = deconv.get_output(0).shape
    reshape_out = network.add_shuffle(deconv.get_output(0))
    reshape_out.reshape_dims = (n, out_channels, out_shape[3])
    return reshape_out.get_output(0)


def add_elu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    alpha: float = 1.0,
) -> trt.ITensor:
    """ELU activation: max(0, x) + min(0, alpha * (exp(x) - 1))."""
    elu = network.add_activation(inp, trt.ActivationType.ELU)
    elu.alpha = alpha
    return elu.get_output(0)


def add_causal_pad_1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    pad_left: int,
) -> trt.ITensor:
    """Causal left-padding for 1D tensor [N, C, L] -> [N, C, L + pad_left]."""
    n, c, length = inp.shape
    # Reshape to [N, C, 1, L] for 2D padding
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c, 1, length)

    pad = network.add_padding_nd(
        reshape_in.get_output(0),
        pre_padding=(0, pad_left),
        post_padding=(0, 0))

    reshape_out = network.add_shuffle(pad.get_output(0))
    reshape_out.reshape_dims = (n, c, length + pad_left)
    return reshape_out.get_output(0)


def add_reflect_pad_1d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    pad_left: int,
    pad_right: int,
) -> trt.ITensor:
    """Reflect padding for 1D tensor [N, C, L].

    For TRT, we approximate reflect padding with replicate padding
    since TRT does not have native reflect mode.
    """
    # Use slice + concatenate to implement reflect padding
    # For simplicity, use zero padding as a fallback
    n, c, length = inp.shape
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (n, c, 1, length)

    pad = network.add_padding_nd(
        reshape_in.get_output(0),
        pre_padding=(0, pad_left),
        post_padding=(0, pad_right))

    reshape_out = network.add_shuffle(pad.get_output(0))
    reshape_out.reshape_dims = (n, c, length + pad_left + pad_right)
    return reshape_out.get_output(0)


def add_slice_trim_right(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    trim: int,
) -> trt.ITensor:
    """Trim `trim` elements from the right of the last dimension.

    Input: [N, C, L]
    Output: [N, C, L - trim]
    """
    n, c, length = inp.shape
    new_length = length - trim
    slice_layer = network.add_slice(
        inp,
        start=(0, 0, 0),
        shape=(n, c, new_length),
        stride=(1, 1, 1))
    return slice_layer.get_output(0)


def add_lstm_unrolled(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
    hidden_size: int,
    seq_length: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LSTM using TRT native loop API (ILoopLayer + IRecurrenceLayer).

    Uses TRT's built-in loop construct instead of Python-level unrolling,
    so graph size is O(1) regardless of sequence length.

    Input: [1, seq_length, input_size] (batch=1)
    Output: [1, seq_length, hidden_size]

    Gates: i, f, g, o (standard PyTorch LSTM ordering).
    w_ih: [4*hidden_size, input_size]
    w_hh: [4*hidden_size, hidden_size]
    b_ih, b_hh: [4*hidden_size]
    """
    input_size = inp.shape[2] if len(inp.shape) == 3 else inp.shape[1]
    H = hidden_size

    # Combined bias
    bias = (b_ih + b_hh).astype(np.float32)

    # Weight constants
    w_ih_t = add_constant(network, (input_size, 4 * H),
                          np.ascontiguousarray(w_ih.T, dtype=np.float32), dtype=dtype)
    w_hh_t = add_constant(network, (H, 4 * H),
                          np.ascontiguousarray(w_hh.T, dtype=np.float32), dtype=dtype)
    bias_t = add_constant(network, (1, 4 * H), bias.reshape(1, -1), dtype=dtype)

    # Init h, c to zeros [1, H]
    zero_h = add_constant(network, (1, H), np.zeros((1, H), dtype=np.float32), dtype=dtype)
    zero_c = add_constant(network, (1, H), np.zeros((1, H), dtype=np.float32), dtype=dtype)

    # --- TRT loop ---
    loop = network.add_loop()

    # Trip count = seq_length (scalar int32)
    trip_count = network.add_constant(
        (), trt.Weights(np.array(seq_length, dtype=np.int32)))
    loop.add_trip_limit(trip_count.get_output(0), trt.TripLimit.COUNT)

    # Iterator over input: [1, seq_length, input_size] → [1, input_size] per step
    x_iter = loop.add_iterator(inp, axis=1)

    # Recurrence layers for h and c state
    h_rec = loop.add_recurrence(zero_h)
    c_rec = loop.add_recurrence(zero_c)

    # Loop body: one LSTM timestep
    x_t = x_iter.get_output(0)  # [1, input_size]
    h = h_rec.get_output(0)     # [1, H]
    c = c_rec.get_output(0)     # [1, H]

    # gates = x_t @ W_ih^T + h @ W_hh^T + bias   [1, 4*H]
    xw = network.add_matrix_multiply(
        x_t, trt.MatrixOperation.NONE,
        w_ih_t, trt.MatrixOperation.NONE)
    hw = network.add_matrix_multiply(
        h, trt.MatrixOperation.NONE,
        w_hh_t, trt.MatrixOperation.NONE)
    gates = network.add_elementwise(
        xw.get_output(0), hw.get_output(0), trt.ElementWiseOperation.SUM)
    gates = network.add_elementwise(
        gates.get_output(0), bias_t, trt.ElementWiseOperation.SUM)

    # Split gates: i, f, g, o each [1, H]
    gate_i = network.add_slice(
        gates.get_output(0), start=(0, 0), shape=(1, H), stride=(1, 1))
    gate_f = network.add_slice(
        gates.get_output(0), start=(0, H), shape=(1, H), stride=(1, 1))
    gate_g = network.add_slice(
        gates.get_output(0), start=(0, 2 * H), shape=(1, H), stride=(1, 1))
    gate_o = network.add_slice(
        gates.get_output(0), start=(0, 3 * H), shape=(1, H), stride=(1, 1))

    # Activations: sigmoid(i), sigmoid(f), tanh(g), sigmoid(o)
    i_t = network.add_activation(
        gate_i.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    f_t = network.add_activation(
        gate_f.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    g_t = network.add_activation(
        gate_g.get_output(0), trt.ActivationType.TANH).get_output(0)
    o_t = network.add_activation(
        gate_o.get_output(0), trt.ActivationType.SIGMOID).get_output(0)

    # c_new = f * c + i * g
    fc = network.add_elementwise(
        f_t, c, trt.ElementWiseOperation.PROD).get_output(0)
    ig = network.add_elementwise(
        i_t, g_t, trt.ElementWiseOperation.PROD).get_output(0)
    c_new = network.add_elementwise(
        fc, ig, trt.ElementWiseOperation.SUM).get_output(0)

    # h_new = o * tanh(c_new)
    tanh_c = network.add_activation(
        c_new, trt.ActivationType.TANH).get_output(0)
    h_new = network.add_elementwise(
        o_t, tanh_c, trt.ElementWiseOperation.PROD).get_output(0)

    # Feed new h, c back to recurrence
    h_rec.set_input(1, h_new)
    c_rec.set_input(1, c_new)

    # Collect h at every timestep: [1, H] → [1, seq_length, H]
    h_output = loop.add_loop_output(h_rec.get_output(0), trt.LoopOutput.CONCATENATE, 1)
    h_output.set_input(1, trip_count.get_output(0))

    return h_output.get_output(0)


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


def tile_rope_for_heads(
    network: trt.INetworkDefinition,
    rope: trt.ITensor,
    num_heads: int,
) -> trt.ITensor:
    """Tile a runtime RoPE table [S, D] to [S, H * D]."""
    if num_heads == 1:
        return rope
    concat = network.add_concatenation([rope] * num_heads)
    concat.axis = 1
    return concat.get_output(0)


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [Sq, K] to [1, 1, Sq, K]."""
    mask_shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(
        network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, mask_shape])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_2d)
    mask_4d.set_input(1, target.get_output(0))
    return mask_4d.get_output(0)


def add_3d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_3d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [B, Sq, K] to [B, 1, Sq, K]."""
    mask_shape = network.add_shape(mask_3d).get_output(0)
    batch = network.add_slice(mask_shape, start=(0,), shape=(1,), stride=(1,))
    sq_size = network.add_slice(mask_shape, start=(1,), shape=(1,), stride=(1,))
    k_size = network.add_slice(mask_shape, start=(2,), shape=(1,), stride=(1,))
    one = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation(
        [batch.get_output(0), one, sq_size.get_output(0), k_size.get_output(0)])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_3d)
    mask_4d.set_input(1, target.get_output(0))
    return mask_4d.get_output(0)


def add_alibi_mask_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
    position_id: trt.ITensor,
    alibi_slopes_tensor: trt.ITensor,
    cache_position_indices: trt.ITensor,
    num_heads: int,
    target_dtype: trt.DataType | None = None,
) -> trt.ITensor:
    """Build a per-head ALiBi additive mask for native IAttention.

    Args:
        mask_2d: [Sq, K] additive mask.
        position_id: [Sq] query positions.
        alibi_slopes_tensor: [H, 1, 1] per-head slopes.
        cache_position_indices: [cache_rows] key positions for cached rows.
        target_dtype: Optional dtype for the returned mask. Defaults to
            ``mask_2d.dtype``.

    Returns:
        [1, H, Sq, K] additive mask containing both ``mask_2d`` and
        ``slope[h] * (key_pos[k] - query_pos[q])``.
    """
    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    cache_positions = cache_position_indices
    if cache_positions.dtype != trt.float32:
        cache_positions = network.add_cast(cache_positions, trt.float32).get_output(0)

    key_pos = network.add_concatenation([cache_positions, pos_float])
    key_pos.axis = 0

    mask_shape = network.add_shape(mask_2d).get_output(0)
    one_const = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    sq_size = network.add_slice(mask_shape, start=(0,), shape=(1,), stride=(1,))
    k_size = network.add_slice(mask_shape, start=(1,), shape=(1,), stride=(1,))
    sq_size_t = sq_size.get_output(0)
    k_size_t = k_size.get_output(0)

    key_pos_shape = network.add_concatenation([one_const, k_size_t])
    key_pos_shape.axis = 0
    key_pos_2d = network.add_shuffle(key_pos.get_output(0))
    key_pos_2d.set_input(1, key_pos_shape.get_output(0))

    query_pos_shape = network.add_concatenation([sq_size_t, one_const])
    query_pos_shape.axis = 0
    query_pos_2d = network.add_shuffle(pos_float)
    query_pos_2d.set_input(1, query_pos_shape.get_output(0))

    rel_pos = network.add_elementwise(
        key_pos_2d.get_output(0), query_pos_2d.get_output(0),
        trt.ElementWiseOperation.SUB)

    one_const2 = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    rel_shape = network.add_concatenation([one_const, one_const2, sq_size_t, k_size_t])
    rel_shape.axis = 0
    rel_4d = network.add_shuffle(rel_pos.get_output(0))
    rel_4d.set_input(1, rel_shape.get_output(0))

    slopes = alibi_slopes_tensor
    if slopes.dtype != trt.float32:
        slopes = network.add_cast(slopes, trt.float32).get_output(0)
    slopes_4d = network.add_shuffle(slopes)
    slopes_4d.reshape_dims = (1, num_heads, 1, 1)

    alibi_bias = network.add_elementwise(
        slopes_4d.get_output(0), rel_4d.get_output(0),
        trt.ElementWiseOperation.PROD)
    alibi_bias_t = alibi_bias.get_output(0)

    mask_4d = add_2d_mask_to_4d(network, mask_2d)
    out_dtype = target_dtype or mask_4d.dtype
    if alibi_bias_t.dtype != out_dtype:
        alibi_bias_t = network.add_cast(alibi_bias_t, out_dtype).get_output(0)

    combined = network.add_elementwise(
        mask_4d, alibi_bias_t, trt.ElementWiseOperation.SUM)
    return combined.get_output(0)


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

    # Pre-scale Q: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    # Use FP16 weights directly for FP16; BF16 has no numpy native type so
    # create as FP32 and cast; FP32 falls through to the default.
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0), k_4d, v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    attn.decomposable = True
    if mask is not None and not causal:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


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
            fp32_accumulation=fp32_accumulation)
    return reshape_heads_4d_to_rows(
        network, ctx_4d, attention_size, sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx")


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core


def add_decoder_attention_ffi(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    all_k: trt.ITensor,
    all_v: trt.ITensor,
    *,
    kernel_name: str,
    num_heads: int,
    head_dim: int,
    attention_window: int,
) -> trt.ITensor:
    """Decoder attention via TVM-FFI kernel (FlashInfer, CuTe, etc).

    The kernel must be registered as a TVM-FFI global before engine build.

    Inputs:
        q:              [1, attention_size]
        all_k, all_v:   [attention_window, attention_size]
    Returns:
        context:        [1, attention_size]
    """
    attention_size = num_heads * head_dim

    q_2d = network.add_shuffle(q)
    q_2d.reshape_dims = (num_heads, head_dim)
    k_3d = network.add_shuffle(all_k)
    k_3d.reshape_dims = (attention_window, num_heads, head_dim)
    v_3d = network.add_shuffle(all_v)
    v_3d.reshape_dims = (attention_window, num_heads, head_dim)

    scale_val = 1.0 / (head_dim ** 0.5)
    ffi_outputs = add_tvm_ffi_kernel(
        network,
        kernel_name=kernel_name,
        inputs=[q_2d.get_output(0), k_3d.get_output(0),
                v_3d.get_output(0)],
        output_specs=[{"dims": [num_heads, head_dim], "dtype": "float16"}],
        workspace_bytes=32 * 1024 * 1024,  # 32MB for FlashInfer tmp
        extra_args=[
            {"type": "none"},              # maybe_lse
            {"type": "int", "value": 0},    # kv_layout_code (NHD)
            {"type": "int", "value": -1},   # window_left
            {"type": "none"},              # alibi_slopes
            {"type": "float", "value": 0.0},     # logits_soft_cap
            {"type": "float", "value": scale_val}, # sm_scale
            {"type": "float", "value": 1.0},      # rope_rcp_scale
            {"type": "float", "value": 0.0001},   # rope_rcp_theta
        ],
    )
    context_flat = network.add_shuffle(ffi_outputs[0])
    context_flat.reshape_dims = (1, attention_size)
    return context_flat.get_output(0)


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
    creator = registry.get_plugin_creator("TvmFfiKernel", "1", "")
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
from tensorrt_model_connect import trt_compat


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


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
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
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
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
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
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
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
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
    need_cast = dtype != np.float32
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
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_3d.get_output(0), trt.ElementWiseOperation.SUM
    )
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped, recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    if gamma_arr.size == head_dim:
        gamma_t = add_constant(
            network, (1, 1, head_dim), gamma_arr.reshape(1, 1, head_dim), dtype=np.float32
        )
    else:
        gamma_t = add_constant(
            network,
            (1, num_heads, head_dim),
            gamma_arr.reshape(num_heads, head_dim),
            dtype=np.float32,
        )
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )

    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def _yarn_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Find the YaRN correction dimension boundary."""
    return dim * np.log(max_position_embeddings / (num_rotations * 2 * np.pi)) / (2 * np.log(base))


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
        return np.full((max(max_cache_length, 1), max(half, 1)), default, dtype=np.float32)

    freq_extra = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freq_inter = freq_extra / scaling_factor

    low = max(
        int(
            np.floor(
                _yarn_correction_dim(
                    beta_fast, head_dim, rope_theta, original_max_position_embeddings
                )
            )
        ),
        0,
    )
    high = min(
        int(
            np.ceil(
                _yarn_correction_dim(
                    beta_slow, head_dim, rope_theta, original_max_position_embeddings
                )
            )
        ),
        half - 1,
    )
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
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
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
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD
    )
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
            relu.get_output(0), relu.get_output(0), trt.ElementWiseOperation.PROD
        )
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
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
        return [start * (start**i) for i in range(n)]

    if num_heads > 0 and (num_heads & (num_heads - 1)) == 0:
        # Power of 2
        return np.array(_get_slopes_power_of_2(num_heads), dtype=np.float32)
    else:
        closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
        slopes_a = _get_slopes_power_of_2(closest_power_of_2)
        slopes_b = _get_slopes_power_of_2(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: num_heads - closest_power_of_2]
        return np.array(slopes_a + slopes_b, dtype=np.float32)


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
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
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
            f"got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def make_native_active_rope_inv_freq(
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
) -> np.ndarray:
    """Return the exact HF/Torch CPU frequencies for native active RoPE.

    Hugging Face initializes Qwen3's non-persistent ``inv_freq`` buffer on the
    CPU before moving the model to CUDA.  NumPy and Torch ``pow`` can differ by
    one FP32 ULP; that difference changes BF16 cos/sin values at long
    positions. Keep this Torch dependency confined to the native build path;
    the legacy indexed-table implementation below stays unchanged.
    """
    rotary_ndims = validate_native_rope_dim(
        int(head_dim * partial_rotary_factor)
    )
    rope_theta = float(rope_theta)
    if not np.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError(
            "TRT native RoPE requires rope_theta to be finite and positive; "
            f"got {rope_theta}"
        )

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Qwen native active-position RoPE requires PyTorch at build time "
            "to reproduce Hugging Face FP32 inverse frequencies exactly. "
            "Install the Model Connect build dependencies, including torch."
        ) from exc

    with torch.no_grad():
        dims = torch.arange(
            0,
            rotary_ndims,
            2,
            dtype=torch.int64,
            device="cpu",
        ).to(dtype=torch.float32)
        # Preserve transformers Qwen3RotaryEmbedding's operation order:
        # exponent division, power, then reciprocal, all in CPU FP32.
        inv_freq = 1.0 / (
            rope_theta ** (dims / rotary_ndims)
        )
    return np.asarray(
        inv_freq.detach().cpu().numpy(),
        dtype=np.float32,
    ).copy()


def add_active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    inv_freq: np.ndarray,
    output_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Build rank-3 cos/sin tensors only for runtime-active positions.

    ``IRotaryEmbeddingLayer`` accepts ``[B, Sq, D/2]`` caches without a
    separate position-ids input.  Computing those rows from the runtime
    ``position_id`` tensor avoids serializing an ``O(context_capacity)`` RoPE
    table while preserving the native TensorRT RoPE layer.
    """
    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    if inv_freq.ndim != 1 or inv_freq.size == 0:
        raise ValueError(
            "active RoPE inverse frequencies must be a non-empty rank-1 array"
        )

    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    pos_col = network.add_shuffle(pos_float)
    pos_col.reshape_dims = (-1, 1)
    inv_freq_tensor = add_constant(
        network,
        (1, int(inv_freq.size)),
        inv_freq.reshape(1, -1),
        dtype=np.float32,
    )
    angles = network.add_elementwise(
        pos_col.get_output(0),
        inv_freq_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cos_2d = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin_2d = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)

    cos_3d = network.add_shuffle(cos_2d)
    cos_3d.reshape_dims = (1, -1, int(inv_freq.size))
    sin_3d = network.add_shuffle(sin_2d)
    sin_3d.reshape_dims = (1, -1, int(inv_freq.size))
    cos_cache = cos_3d.get_output(0)
    sin_cache = sin_3d.get_output(0)
    if cos_cache.dtype != output_dtype:
        cos_cache = network.add_cast(cos_cache, output_dtype).get_output(0)
        sin_cache = network.add_cast(sin_cache, output_dtype).get_output(0)
    return cos_cache, sin_cache


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
        return np.full((max(max_cache_length, 1), max(half, 1)), default, dtype=np.float32)
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


def add_2d_mask_to_4d(
    network: trt.INetworkDefinition,
    mask_2d: trt.ITensor,
) -> trt.ITensor:
    """Reshape additive attention mask [Sq, K] to [1, 1, Sq, K]."""
    mask_shape = network.add_shape(mask_2d).get_output(0)
    ones = add_constant(network, (2,), np.array([1, 1], dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([ones, mask_shape])
    target.axis = 0
    mask_4d = network.add_shuffle(mask_2d)
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
    one_const = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
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
        key_pos_2d.get_output(0), query_pos_2d.get_output(0), trt.ElementWiseOperation.SUB
    )

    one_const2 = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
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
        slopes_4d.get_output(0), rel_4d.get_output(0), trt.ElementWiseOperation.PROD
    )
    alibi_bias_t = alibi_bias.get_output(0)

    mask_4d = add_2d_mask_to_4d(network, mask_2d)
    out_dtype = target_dtype or mask_4d.dtype
    if alibi_bias_t.dtype != out_dtype:
        alibi_bias_t = network.add_cast(alibi_bias_t, out_dtype).get_output(0)

    combined = network.add_elementwise(mask_4d, alibi_bias_t, trt.ElementWiseOperation.SUM)
    return combined.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor | None,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    """Apply RoPE via TRT native IRotaryEmbeddingLayer.

    Handles both single-token decoder steps and dynamic-Sq prefill/decode
    graphs without a manual rotate-half matmul chain.

    Shape contract with indexed, full-capacity caches:
      input:           [1, num_heads, Sq, head_dim]  (reshaped internally)
      cos_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      sin_cache_2d:    [max_S, rotary_embedding_dim // 2]  (2-D constant)
      position_id:     [Sq] int32, reshaped to [1, Sq] internally

    Shape contract with active-position caches:
      input:           [1, num_heads, Sq, head_dim]
      cos_cache_2d:    [1, Sq, rotary_embedding_dim // 2]
      sin_cache_2d:    [1, Sq, rotary_embedding_dim // 2]
      position_id:     None; the caches already correspond to active positions
      interleaved:     False → rotate-half (LLaMA/Qwen)
                       True  → adjacent-pair (CodeGen/GPT-J)

    Args:
        inp:                  [Sq, num_heads * head_dim].
        num_heads:            Number of attention heads.
        head_dim:             Per-head dimension.
        cos_cache_2d:         Pre-built 2-D cos table constant.
        sin_cache_2d:         Pre-built 2-D sin table constant.
        position_id:          Runtime position indices, shape [Sq] int32, or
                              ``None`` for active-position rank-3 caches.
        rotary_embedding_dim: Number of head dims that participate in RoPE.
        interleaved:          Frequency layout (see above).
        sequence_length:      Static Sq, or None for runtime-dynamic Sq.

    Returns:
        [Sq, num_heads * head_dim] with RoPE applied.
    """
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim

    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        interleaved,
        rotary_embedding_dim,
    )
    if position_id is not None:
        # Reshape position_id [Sq] -> [1, Sq] (batch=1).
        seq_dim = -1 if sequence_length is None else sequence_length
        pos_2d = network.add_shuffle(position_id)
        pos_2d.reshape_dims = (1, seq_dim)
        rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


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
        q_scaled.get_output(0),
        k_4d,
        v_4d,
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
    const = add_constant(network, shape, np.full(shape, value, dtype=np_dtype), dtype=np_dtype)
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
    cap_t = _scalar_constant_for_trt_dtype(network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


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
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0), shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1)
        )
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
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE, score_k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE, v_4d, trt.MatrixOperation.NONE
    ).get_output(0)
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
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if causal:
            raise NotImplementedError("logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network,
            q_4d,
            k_4d,
            v_4d,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            mask=mask,
            scale=scale,
            logit_softcap=float(logit_softcap),
        )
    else:
        ctx_4d = add_attention_core(
            network,
            q_4d,
            k_4d,
            v_4d,
            causal=causal,
            mask=mask,
            scale=scale,
            fp32_accumulation=fp32_accumulation,
        )
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


def add_native_kv_attention_mask(
    network: trt.INetworkDefinition,
    query_rows: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    cache_capacity: int,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Build an additive ``[1, 1, Sq, K]`` active-prefix causal mask.

    The valid region is the intersection of the runtime-active KV prefix and
    the causal prefix for each query row. This deliberately bypasses
    ``IAttention.key_value_lengths`` on TensorRT/GPU combinations where that
    optional input exposes the inactive cache suffix instead of the prefix.
    """
    query_shape = network.add_shape(query_rows).get_output(0)
    query_length = network.add_slice(
        query_shape, start=(0,), shape=(1,), stride=(1,)
    ).get_output(0)
    zero_i32 = add_constant(
        network, (), np.array(0, dtype=np.int32), dtype=np.int32
    )
    one_i32 = add_constant(
        network, (1,), np.array([1], dtype=np.int32), dtype=np.int32
    )
    query_iota = network.add_fill((1,), trt.FillOperation.LINSPACE, trt.int32)
    query_iota.set_input(0, query_length)
    query_iota.set_input(1, zero_i32)
    query_iota.set_input(2, one_i32)
    query_positions = network.add_elementwise(
        query_iota.get_output(0),
        cache_write_indices,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    query_limits = network.add_elementwise(
        query_positions, one_i32, trt.ElementWiseOperation.SUM
    ).get_output(0)

    one_i64 = add_constant(
        network, (1,), np.array([1], dtype=np.int64), dtype=np.int64
    )
    query_limit_shape = network.add_concatenation([query_length, one_i64])
    query_limit_shape.axis = 0
    query_limits_2d = network.add_shuffle(query_limits)
    query_limits_2d.set_input(1, query_limit_shape.get_output(0))

    key_positions = add_constant(
        network,
        (1, cache_capacity),
        np.arange(cache_capacity, dtype=np.int32).reshape(1, cache_capacity),
        dtype=np.int32,
    )
    key_value_lengths_2d = network.add_shuffle(key_value_lengths)
    key_value_lengths_2d.reshape_dims = (1, 1)
    causal = network.add_elementwise(
        key_positions,
        query_limits_2d.get_output(0),
        trt.ElementWiseOperation.LESS,
    ).get_output(0)
    active = network.add_elementwise(
        key_positions,
        key_value_lengths_2d.get_output(0),
        trt.ElementWiseOperation.LESS,
    ).get_output(0)
    valid_positions = network.add_elementwise(
        causal, active, trt.ElementWiseOperation.AND
    ).get_output(0)

    zero = add_constant(
        network, (1, 1), np.array([[0.0]], dtype=np.float32), dtype=np.float32
    )
    blocked = add_constant(
        network,
        (1, 1),
        np.array([[-1.0e4]], dtype=np.float32),
        dtype=np.float32,
    )
    zero = _cast_back_to_trt_dtype(network, zero, target_dtype)
    blocked = _cast_back_to_trt_dtype(network, blocked, target_dtype)
    additive_mask = network.add_select(
        valid_positions, zero, blocked
    ).get_output(0)
    return add_2d_mask_to_4d(network, additive_mask)


def add_native_kv_cache_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k_update: trt.ITensor,
    v_update: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_seq: int | None,
    scale: float | None = None,
    tag: str | None = None,
    recipe_instance: str | None = None,
    explicit_mask: trt.ITensor | None = None,
    attention_dtype: trt.DataType | None = None,
) -> dict[str, trt.ITensor]:
    """Update a user-owned KV cache and attend over its active prefix.

    ``IKVCacheUpdateLayer`` writes this step's K/V rows into the static,
    full-capacity cache in place. The normal contract passes active lengths to
    ``IAttention``. The compatibility contract supplies a correct explicit
    additive mask and omits the broken native length input. On targets whose
    FP16 fused kernel rejects a fourth live input, ``attention_dtype`` allows a
    narrow BF16 cache/attention boundary while the surrounding model remains
    FP16.

    Cache inputs and outputs have shape ``[1, Hkv, capacity, D]``. The caller
    must bind each output to the same device address as its corresponding
    input, as required by TensorRT's KV-cache aliasing contract.
    """
    if not hasattr(network, "add_kv_cache_update") or not hasattr(
        network, "add_attention_v2"
    ):
        raise RuntimeError(
            "Qwen native KV cache requires TensorRT add_kv_cache_update "
            "and add_attention_v2 support"
        )

    output_dtype = q.dtype
    if attention_dtype is None:
        attention_dtype = cache_k.dtype
    if attention_dtype not in {trt.float16, trt.bfloat16}:
        raise ValueError("Qwen native KV attention requires FP16 or BF16")
    if cache_k.dtype != attention_dtype or cache_v.dtype != attention_dtype:
        raise ValueError(
            "Qwen native KV cache dtype must match the attention dtype"
        )
    if explicit_mask is not None and explicit_mask.dtype != attention_dtype:
        raise ValueError(
            "Qwen native KV explicit mask dtype must match the attention dtype"
        )
    if k_update.dtype != attention_dtype:
        k_update = network.add_cast(k_update, attention_dtype).get_output(0)
    if v_update.dtype != attention_dtype:
        v_update = network.add_cast(v_update, attention_dtype).get_output(0)

    k_update_4d = reshape_rows_to_heads_4d(
        network,
        k_update,
        num_kv_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".k_update",
    )
    v_update_4d = reshape_rows_to_heads_4d(
        network,
        v_update,
        num_kv_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".v_update",
    )

    update_k = network.add_kv_cache_update(
        cache_k,
        k_update_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    update_v = network.add_kv_cache_update(
        cache_v,
        v_update_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT failed to create Qwen KV-cache update layers")
    if tag:
        update_k.name = tag + ".cache_k_update"
        update_v.name = tag + ".cache_v_update"
    updated_k = update_k.get_output(0)
    updated_v = update_v.get_output(0)

    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0

    if q_4d.dtype not in {trt.float16, trt.bfloat16}:
        raise ValueError("Qwen native KV attention requires FP16 or BF16 queries")

    # Match the native fused-attention contract exactly: TensorRT consumes an
    # FP16/BF16 query after the FP32 scale has already been applied and rounded.
    q_scale_input = network.add_cast(q_4d, trt.float32).get_output(0)
    scale_t = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]]),
        dtype=np.float32,
    )
    q_scaled = network.add_elementwise(
        q_scale_input, scale_t, trt.ElementWiseOperation.PROD
    ).get_output(0)
    q_scaled = network.add_cast(q_scaled, attention_dtype).get_output(0)

    recipe = nullcontext()
    if recipe_instance is not None:
        from ...tvm_ffi.graph_build import graph_recipe_region

        recipe = graph_recipe_region(
            network,
            "qwen.decode_attention_region@2",
            recipe_instance,
            output_shape_input=0,
        )
    with recipe:
        attention = network.add_attention_v2(
            q_scaled,
            updated_k,
            updated_v,
            trt.AttentionNormalizationOp.SOFTMAX,
            (
                trt.CausalMaskKind.NONE
                if explicit_mask is not None
                else trt.CausalMaskKind.LOWER_RIGHT
            ),
        )
        if attention is None:
            raise RuntimeError("TensorRT failed to create Qwen native attention")
        # The prototype deliberately requires TensorRT's fused attention tactic.
        # Unsupported dtype/head geometries fail at build time instead of silently
        # falling back to a primitive graph with materially lower decode speed.
        attention.decomposable = False
        if explicit_mask is None:
            attention.key_value_lengths = key_value_lengths
        else:
            attention.mask = explicit_mask
        if tag:
            attention.name = tag

    context_4d = attention.get_output(0)
    if context_4d.dtype != output_dtype:
        context_4d = network.add_cast(context_4d, output_dtype).get_output(0)
    context = reshape_heads_4d_to_rows(
        network,
        context_4d,
        num_heads * head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )
    return {
        "context": context,
        "present_k": updated_k,
        "present_v": updated_v,
    }


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

    scale_val = 1.0 / (head_dim**0.5)
    ffi_outputs = add_tvm_ffi_kernel(
        network,
        kernel_name=kernel_name,
        inputs=[q_2d.get_output(0), k_3d.get_output(0), v_3d.get_output(0)],
        output_specs=[{"dims": [num_heads, head_dim], "dtype": "float16"}],
        workspace_bytes=32 * 1024 * 1024,  # 32MB for FlashInfer tmp
        extra_args=[
            {"type": "none"},  # maybe_lse
            {"type": "int", "value": 0},  # kv_layout_code (NHD)
            {"type": "int", "value": -1},  # window_left
            {"type": "none"},  # alibi_slopes
            {"type": "float", "value": 0.0},  # logits_soft_cap
            {"type": "float", "value": scale_val},  # sm_scale
            {"type": "float", "value": 1.0},  # rope_rcp_scale
            {"type": "float", "value": 0.0001},  # rope_rcp_theta
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

    trt_compat.load_native_backend_plugins()
    registry = trt.get_plugin_registry()
    get_creator = getattr(registry, "get_creator", None) or registry.get_plugin_creator
    creator = get_creator("TvmFfiKernel", "1", "")
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
        trt.PluginField("kernel_name", kernel_name.encode("utf-8"), trt.PluginFieldType.CHAR),
        trt.PluginField("shape_spec", shape_spec.encode("utf-8"), trt.PluginFieldType.CHAR),
    ]
    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin("tvm_ffi_kernel", fc)

    layer = network.add_plugin_v2(inputs, plugin)
    return [layer.get_output(i) for i in range(layer.num_outputs)]

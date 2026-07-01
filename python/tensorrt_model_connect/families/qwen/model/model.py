"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
from typing import TYPE_CHECKING
import sys
from ..config import ModelConfig
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

    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)

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
        trt.PluginField("kernel_name", kernel_name.encode("utf-8"), trt.PluginFieldType.CHAR),
        trt.PluginField("shape_spec", shape_spec.encode("utf-8"), trt.PluginFieldType.CHAR),
    ]
    fc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin("tvm_ffi_kernel", fc)

    layer = network.add_plugin_v2(inputs, plugin)
    return [layer.get_output(i) for i in range(layer.num_outputs)]


# Graph Blocks


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


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
            return add_matmul_rhs_constant(network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)

        return matmul
    else:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
            )

        return matmul


_make_matmul_fn = make_matmul_fn


def infer_kv_attention_size(
    weights: dict,
    *,
    prefix: str = "layer.0",
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate and return the compact K/V row width."""
    expected = int(num_kv_heads * head_dim)
    explicit = weights.get("_kv_attention_size")
    if explicit is not None and int(explicit) != expected:
        raise ValueError(
            f"Compact K/V cache width must be num_kv_heads * head_dim "
            f"({expected}), got _kv_attention_size={int(explicit)}"
        )
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(f"{prefix}.w_k must use compact K/V width {expected}, got {actual}")
    return expected


def apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden_size, dtype=np.float32)
        if eps is not None:
            return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)
        # Native INormalizationLayer requires a build-time scalar epsilon.
        # Some callers only pass epsilon as an ITensor, so keep the manual
        # shared fallback until those builders thread the scalar too.
        return add_layer_norm(network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype)
    else:
        return add_rms_norm(network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    eps_tensor: trt.ITensor,
    kv_attention_size: int | None = None,
    num_kv_heads: int | None = None,
    attention_scale: float | None = None,
    eps: float | None = None,
    norm_type: str = "rmsnorm",
    position_type: str = "rope",
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    # TRT 10 native API tensors.
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
) -> dict[str, trt.ITensor]:
    """Pre-norm -> QKV -> RoPE -> cache concat -> attention -> output proj.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.

    This function uses TRT 10 native APIs for the basic transformer primitives:
      - IRotaryEmbeddingLayer for RoPE
      - IAttention for scaled dot-product attention
    ALiBi is represented as a per-head additive attention mask and still uses
    native IAttention.
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if kv_attention_size is None:
        kv_attention_size = num_kv_heads * head_dim

    # Weight name for quant scale lookup — use layer_prefix if provided,
    # otherwise fall back to the weights-dict prefix.
    _lp = layer_prefix or prefix

    # Pre-attention norm
    normed = apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor,
        norm_type,
        dtype=dtype,
        eps=eps,
    )

    # QKV projections
    q = matmul(normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], f"{_lp}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], f"{_lp}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], f"{_lp}.w_v")

    # Optional QKV biases
    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = add_bias_sum(network, q, attention_size, q_bias, dtype=dtype)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = add_bias_sum(network, k, kv_attention_size, k_bias, dtype=dtype)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = add_bias_sum(network, v, kv_attention_size, v_bias, dtype=dtype)

    # Optional per-head q/k norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = add_rms_norm_per_head(network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype
        )

    # ------------------------------------------------------------------ #
    # RoPE via native IRotaryEmbeddingLayer                              #
    # ------------------------------------------------------------------ #
    use_native_attention = ffi_attention_kernel is None

    if position_type == "rope":
        if cos_half_tensor is None or sin_half_tensor is None:
            raise ValueError(
                "RoPE attention requires half-dimension cos/sin tensors for "
                "TRT native IRotaryEmbeddingLayer"
            )
        rope_dim = rotary_embedding_dim or head_dim
        rope_dim = validate_native_rope_dim(rope_dim)
        q = add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_half_tensor,
            sin_half_tensor,
            position_id,
            rope_dim,
            interleaved_rope,
        )
        k = add_apply_rope_native(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_half_tensor,
            sin_half_tensor,
            position_id,
            rope_dim,
            interleaved_rope,
        )

    # Save present K/V (before concatenation, this is the raw projection output)
    present_k = k
    present_v = v

    # Reshape current K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    # ------------------------------------------------------------------ #
    # Attention core — native IAttention or FFI kernel                    #
    # ------------------------------------------------------------------ #
    if use_native_attention:
        kv_seq = None if dynamic_kv_cache else attention_window
        if alibi_slopes_tensor is not None:
            if alibi_indices_tensor is None:
                raise ValueError("ALiBi attention requires cache position indices")
            if dynamic_kv_cache:
                raise ValueError("dynamic_kv_cache is not supported for ALiBi attention")
            mask_4d = add_alibi_mask_4d(
                network,
                attention_mask,
                position_id,
                alibi_slopes_tensor,
                alibi_indices_tensor,
                num_heads,
            )
        else:
            mask_4d = add_2d_mask_to_4d(network, attention_mask)

        context = add_attention_from_rows(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=1,
            kv_seq=kv_seq,
            causal=False,
            mask=mask_4d,
            scale=attention_scale,
        )
    elif ffi_attention_kernel is not None:
        if num_kv_heads != num_heads:
            raise ValueError(
                "FFI decoder attention requires num_kv_heads == num_heads; "
                "use TRT native attention for compact GQA/MQA KV cache"
            )
        # Fused attention kernel via TVM-FFI plugin
        context = add_decoder_attention_ffi(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            kernel_name=ffi_attention_kernel,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_window=attention_window,
        )

    # Output projection
    attn_out = matmul(context, attention_size, hidden_size, weights[f"{prefix}.w_o"], f"{_lp}.w_o")

    # Optional output projection bias
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = add_bias_sum(network, attn_out, hidden_size, o_bias, dtype=dtype)

    return {
        "normed": normed,
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    mlp_out = matmul(
        gated.get_output(0), mlp_size, hidden_size, weights[f"{prefix}.w_down"], f"{_lp}.w_down"
    )
    return mlp_out


def add_gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    activation: str = "gelu_new",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    fc1 = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)

    activated = add_activation(network, fc1, activation, dtype=dtype)

    fc2 = matmul(activated, mlp_size, hidden_size, weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)

    return fc2


# Utils


trt = trt_compat.get_trt()


def const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in storage dtype and cast it to runtime dtype."""
    const = add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


# Dual Profile Decoder Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


def _make_matmul_fn(
    network: trt.INetworkDefinition,
    dtype: np.dtype,
    quant_ctx: "QuantContext | None",
):
    """Mirror of ``_make_matmul_fn`` for the dual-profile path.

    Returns a callable ``(lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor``
    that routes through ``QuantContext.maybe_quantized_matmul`` when present
    and falls back to a plain ``add_matmul_rhs_constant`` otherwise. The
    ``weight_name`` is the dotted weight key (e.g. ``layer.0.w_q``) used by
    the quantization profile to look up scales and the per-layer exclude
    pattern.
    """
    if quant_ctx is None:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return add_matmul_rhs_constant(network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)

        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
        )

    return matmul


def _norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return add_rms_norm(network, inp, hidden, gamma, eps_tensor, dtype=dtype)


# ---------------------------------------------------------------------------
# MLP helpers.
# ---------------------------------------------------------------------------


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(
        gated.get_output(0), mlp_size, hidden, weights[f"{prefix}.w_down"], f"{prefix}.w_down"
    )
    return mlp_out


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden, weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = add_bias_sum(network, fc2, hidden, fc2_bias, dtype=work_np_dtype)
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the dual-profile builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}"
        )
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


def _yarn_rope_kwargs(config: "ModelConfig") -> dict | None:
    """Return YaRN RoPE parameters from HF config dicts when present."""
    raw = getattr(config, "raw", {}) or {}
    rope_cfg = raw.get("rope_parameters")
    if not isinstance(rope_cfg, dict):
        rope_cfg = raw.get("rope_scaling")
    if not isinstance(rope_cfg, dict):
        return None
    rope_type = str(rope_cfg.get("rope_type", rope_cfg.get("type", ""))).lower()
    if rope_type != "yarn":
        return None
    return {
        "scaling_factor": float(rope_cfg.get("factor", 1.0)),
        "original_max_position_embeddings": int(
            rope_cfg.get("original_max_position_embeddings", config.max_position_embeddings)
        ),
        "beta_fast": float(rope_cfg.get("beta_fast", 32.0)),
        "beta_slow": float(rope_cfg.get("beta_slow", 1.0)),
    }


def _llama4_attention_scale_kwargs(config: "ModelConfig") -> dict | None:
    """Return Llama-4-style query scale parameters when present."""
    raw = getattr(config, "raw", {}) or {}
    rope_cfg = raw.get("rope_parameters")
    if not isinstance(rope_cfg, dict):
        rope_cfg = raw.get("rope_scaling")
    if not isinstance(rope_cfg, dict):
        return None
    beta = rope_cfg.get("llama_4_scaling_beta")
    if beta is None:
        return None
    beta = float(beta)
    if beta == 0.0:
        return None
    return {
        "beta": beta,
        "original_max_position_embeddings": int(
            rope_cfg.get("original_max_position_embeddings", config.max_position_embeddings)
        ),
    }


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
    full_logits_output: bool = False,
) -> bytes:
    """Build a prefill/decode-capable dynamic-Sq decoder engine.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    ``quant_ctx`` (optional) routes every projection matmul through
    ``QuantContext.maybe_quantized_matmul`` for fp8 / int8 Q/DQ insertion;
    when ``None`` the matmuls are plain fp16 / bf16 / fp32.

    ``profile_mode`` controls which optimization profiles are emitted:

    * ``"dual_profile"``: one prefill profile followed by one or more decode
      profiles. When ``dynamic_kv_profile_rows`` is provided, the decode side
      gets one profile per bucket — letting TriAttention pick the smallest
      active KV cache bucket at runtime while still benefitting from batched
      prefill on the prompt.
    * ``"prefill"``: one prefill profile only. This is used by split-engine
      bundles, where decode is served by a separate fixed-Sq=1 engine.

    Dynamic-KV cache bucket profiles are only meaningful in ``dual_profile``
    mode. In either mode, cache_k/cache_v inputs are declared dynamic when
    bucket profiles are requested so each profile can constrain their row count.
    """
    _supports_config(config, weights)
    if profile_mode not in ("dual_profile", "prefill"):
        raise ValueError(f"profile_mode must be 'dual_profile' or 'prefill', got {profile_mode!r}")

    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    multi_bucket_decode = bool(dynamic_kv_profile_rows)
    if multi_bucket_decode:
        decode_buckets: list[int] = []
        seen = set()
        for raw in dynamic_kv_profile_rows or []:
            clamped = max(1, min(int(raw), max_cache_length))
            if clamped not in seen:
                seen.add(clamped)
                decode_buckets.append(clamped)
        decode_buckets.sort()
        if not decode_buckets:
            decode_buckets = [max_cache_length]
            multi_bucket_decode = False

    attention_size = weights.get("_attention_size", config.attention_size)
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

    cache_shape: tuple[int, int]
    if multi_bucket_decode:
        cache_shape = (-1, kv_attention_size)
    else:
        cache_shape = (max_cache_length, kv_attention_size)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(layer_tensor_name("cache_k", i), work_trt_dtype, cache_shape)
        cv = network.add_input(layer_tensor_name("cache_v", i), work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Cast mask to compute dtype for elementwise broadcast.
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    # Two (or 1+N) optimization profiles — same graph, different Sq / cache.
    def _add_profile(
        opt_sq: int,
        max_sq: int,
        *,
        fixed: bool = False,
        cache_rows_min: int | None = None,
        cache_rows_opt: int | None = None,
        cache_rows_max: int | None = None,
    ):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        if multi_bucket_decode:
            cmn = cache_rows_min if cache_rows_min is not None else 1
            cop = cache_rows_opt if cache_rows_opt is not None else max_cache_length
            cmx = cache_rows_max if cache_rows_max is not None else max_cache_length
            for i in range(num_layers):
                for name in (
                    layer_tensor_name("cache_k", i),
                    layer_tensor_name("cache_v", i),
                ):
                    prof.set_shape(
                        name,
                        (cmn, kv_attention_size),
                        (cop, kv_attention_size),
                        (cmx, kv_attention_size),
                    )
        trt_config.add_optimization_profile(prof)

    import os as _os_dbg

    if profile_mode == "prefill":
        _add_profile(
            opt_prefill_length,
            max_prefill_length,
            fixed=False,
            cache_rows_min=1,
            cache_rows_opt=max_cache_length,
            cache_rows_max=max_cache_length,
        )
    elif _os_dbg.environ.get("TRTMC_DECODE_ONLY_DEBUG") == "1":
        # Diagnostic: build a one-profile engine with dynamic-shape inputs
        # but Sq pinned to 1. Lets us isolate dynamic-shape enqueueV3
        # overhead from per-profile kernel specialisation.
        _add_profile(1, 1, fixed=True)
    else:
        _reverse = _os_dbg.environ.get("TRTMC_REVERSE_PROFILE_ORDER", "0") == "1"
        if _reverse:
            # Decode profile registered first so it commits its preferred
            # weight layout before the prefill profile compiles.
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(
                        1,
                        1,
                        fixed=True,
                        cache_rows_min=1,
                        cache_rows_opt=bucket,
                        cache_rows_max=bucket,
                    )
            else:
                _add_profile(1, 1, fixed=True)
            _add_profile(
                opt_prefill_length,
                max_prefill_length,
                fixed=False,
                cache_rows_min=1,
                cache_rows_opt=max_cache_length,
                cache_rows_max=max_cache_length,
            )
        else:
            _add_profile(
                opt_prefill_length,
                max_prefill_length,
                fixed=False,
                cache_rows_min=1,
                cache_rows_opt=max_cache_length,
                cache_rows_max=max_cache_length,
            )
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(
                        1,
                        1,
                        fixed=True,
                        cache_rows_min=1,
                        cache_rows_opt=bucket,
                        cache_rows_max=bucket,
                    )
            else:
                _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"], work_np_dtype, work_trt_dtype
    )

    # RoPE tables (only when position_type == "rope"). Built for the worst
    # case key length max_cache_length + max_prefill_length, since RoPE is
    # gathered by position_id at runtime. The half-dim tables feed TRT's
    # native IRotaryEmbeddingLayer.
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    q_position_scale_table: trt.ITensor | None = None
    if position_type == "rope":
        kmax = max_cache_length + max_prefill_length
        validate_native_rope_dim(rotary_embedding_dim)
        yarn_kwargs = _yarn_rope_kwargs(config)
        if yarn_kwargs is not None:
            cos_half_np = make_yarn_rope_table_half_dim(
                kmax, head_dim, config.rope_theta, True, interleaved=interleaved_rope, **yarn_kwargs
            )
            sin_half_np = make_yarn_rope_table_half_dim(
                kmax,
                head_dim,
                config.rope_theta,
                False,
                interleaved=interleaved_rope,
                **yarn_kwargs,
            )
        else:
            cos_half_np = make_rope_table_half_dim(
                kmax,
                head_dim,
                config.rope_theta,
                True,
                partial_rotary_factor,
                interleaved=interleaved_rope,
            )
            sin_half_np = make_rope_table_half_dim(
                kmax,
                head_dim,
                config.rope_theta,
                False,
                partial_rotary_factor,
                interleaved=interleaved_rope,
            )
        cos_half_table = const_in_work_dtype(
            network, cos_half_np.shape, cos_half_np, work_np_dtype, work_trt_dtype
        )
        sin_half_table = const_in_work_dtype(
            network, sin_half_np.shape, sin_half_np, work_np_dtype, work_trt_dtype
        )
        llama4_scale_kwargs = _llama4_attention_scale_kwargs(config)
        if llama4_scale_kwargs is not None:
            q_scale_np = make_llama4_attention_scale_table(kmax, **llama4_scale_kwargs)
            q_position_scale_table = const_in_work_dtype(
                network, q_scale_np.shape, q_scale_np, work_np_dtype, work_trt_dtype
            )

    # Learned position embedding (GPT-2 / OPT / GPT-Neo / XGLM).
    position_embed_table: trt.ITensor | None = None
    if position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = const_in_work_dtype(
            network, pos_embed_np.shape, pos_embed_np, work_np_dtype, work_trt_dtype
        )

    # ALiBi slopes + cache-slot positions for multi-row mask augmentation.
    alibi_slopes_tensor: trt.ITensor | None = None
    alibi_cache_positions_fp32: trt.ITensor | None = None
    if position_type == "alibi":
        alibi_slopes_np = compute_alibi_slopes(num_heads)
        # Slopes live as fp32 so the (key_pos - q_pos) math stays in fp32;
        # add_alibi_mask_4d casts the final bias to work_trt_dtype before adding
        # to the additive mask.
        alibi_slopes_tensor = add_constant(
            network, (num_heads, 1, 1), alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32
        )
        # Cache slot k (for k in [0, max_cache_length)) holds the K/V at
        # position k. The current step's K/V live in slots
        # [max_cache_length, max_cache_length + Sq) and their positions come
        # from position_id at runtime, so we only pre-build the cache half.
        alibi_cache_positions_fp32 = add_constant(
            network,
            (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32),
            dtype=np.float32,
        )

    eps_tensor = add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    eps_tensor_per_head = add_constant(
        network, (1, 1, 1), np.array([[[config.rms_norm_eps]]], dtype=np.float32), dtype=np.float32
    )

    # Attention scale.
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
        )
        hidden_state = pos_add.get_output(0)

    # Make sure the main hidden stream is in the requested runtime dtype
    # before entering the layer stack (BF16 mode stores fp16 constants).
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get("embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network,
            hidden_state,
            hidden,
            embed_norm,
            embed_norm_beta,
            eps_tensor,
            "layernorm",
            work_np_dtype,
        )

    # Build the 4D additive mask once — shared across layers. ALiBi
    # variants augment the mask with per-head linear bias.
    if position_type == "alibi":
        mask_4d = add_alibi_mask_4d(
            network,
            attention_mask_work,
            position_id,
            alibi_slopes_tensor,
            alibi_cache_positions_fp32,
            num_heads,
            target_dtype=work_trt_dtype,
        )
    else:
        mask_4d = add_2d_mask_to_4d(network, attention_mask_work)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention norm.
        normed = _norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor,
            norm_type,
            work_np_dtype,
        )

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")

        # Optional QKV biases (Qwen2 / GPT-2 / OPT / Bloom / Falcon / etc.).
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = add_bias_sum(network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = add_bias_sum(network, k, kv_attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = add_bias_sum(network, v, kv_attention_size, v_bias, dtype=work_np_dtype)

        # Optional per-head q/k norm (Qwen3).
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = add_rms_norm_per_head(
                network,
                q,
                num_heads,
                head_dim,
                q_norm,
                eps_tensor_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = add_rms_norm_per_head(
                network,
                k,
                num_kv_heads,
                head_dim,
                k_norm,
                eps_tensor_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )

        # Position embedding (RoPE only; learned was applied above and ALiBi
        # is added into the attention mask).
        if position_type == "rope":
            q = add_apply_rope_native(
                network,
                q,
                num_heads,
                head_dim,
                cos_half_table,
                sin_half_table,
                position_id,
                rotary_embedding_dim,
                interleaved_rope,
                sequence_length=None,
            )
            k = add_apply_rope_native(
                network,
                k,
                num_kv_heads,
                head_dim,
                cos_half_table,
                sin_half_table,
                position_id,
                rotary_embedding_dim,
                interleaved_rope,
                sequence_length=None,
            )
            if q_position_scale_table is not None:
                q_scale = network.add_gather(q_position_scale_table, position_id, 0).get_output(0)
                if q_scale.dtype != q.dtype:
                    q_scale = network.add_cast(q_scale, q.dtype).get_output(0)
                q = network.add_elementwise(q, q_scale, trt.ElementWiseOperation.PROD).get_output(0)

        # Present K / V (this step's raw K / V), shape (Sq, attn_size).
        present_k_outs.append(k)
        present_v_outs.append(v)

        # Concatenate cached + current K / V along the sequence dim.
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0

        context = add_attention_from_rows(
            network,
            q,
            all_k_cat.get_output(0),
            all_v_cat.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=None,
            kv_seq=None,
            causal=False,
            mask=mask_4d,
            scale=attn_scale,
            tag=f"{prefix}.attn",
        )

        attn_out = matmul(
            context, attention_size, hidden, weights[f"{prefix}.w_o"], f"{prefix}.w_o"
        )
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = add_bias_sum(network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network,
                    hidden_state,
                    hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor,
                    norm_type,
                    work_np_dtype,
                )
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM
            )
            norm2 = _norm_multi(
                network,
                residual1.get_output(0),
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor,
                norm_type,
                work_np_dtype,
            )

        # MLP — SwiGLU (Llama-style) or GeluFC (GPT-2-style).
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network,
                norm2,
                matmul=matmul,
                weights=weights,
                prefix=prefix,
                hidden=hidden,
                mlp_size=mlp_size,
                activation=activation,
                work_np_dtype=work_np_dtype,
            )
        else:
            mlp_out = _swiglu_mlp(
                network,
                norm2,
                matmul=matmul,
                weights=weights,
                prefix=prefix,
                hidden=hidden,
                mlp_size=mlp_size,
            )

        # Final residual.
        if parallel_residual:
            sum_attn = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
            )
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
            )
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network,
            hidden_state,
            hidden,
            final_norm,
            weights.get("final_norm_beta"),
            eps_tensor,
            norm_type,
            work_np_dtype,
        )

    lm_input = hidden_state
    if not full_logits_output:
        # Only the LAST prompt token's logits matter for the next-token sample,
        # so slice hidden_state from (Sq, hidden) to (1, hidden) before the LM
        # head. This keeps the output contract identical to the single-token
        # engine (logits shape = (1, vocab)) under both profiles and avoids
        # computing (Sq - 1) redundant vocab-sized matmul rows during prefill.
        shape_t = network.add_shape(hidden_state).get_output(0)  # [2] int64
        one_hidden = add_constant(
            network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64
        )
        start_sub = network.add_elementwise(shape_t, one_hidden, trt.ElementWiseOperation.SUB)
        start_t = start_sub.get_output(0)  # [Sq - 1, 0]
        size_t = add_constant(network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
        slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
        slicer.set_input(1, start_t)
        slicer.set_input(2, size_t)
        lm_input = slicer.get_output(0)

    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = add_matmul_rhs_constant(
        network, lm_input, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = add_bias_sum(network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outs[i]
        pv = present_v_outs[i]
        pk.name = layer_tensor_name("present_k", i)
        pv.name = layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        mode_label = "prefill-profile" if profile_mode == "prefill" else "dual-profile"
        print(
            f"[trtmc build] Building {mode_label} engine "
            f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
            f"kv={kv_attention_size}, "
            f"mlp={mlp_size}, cache={max_cache_length}, "
            f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
            f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
            f"precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)


# Standard Decoder Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    """Mark a tensor as a network output for debug inspection."""
    # Use an identity layer to avoid aliasing issues with existing outputs.
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def build_standard_decoder_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx: QuantContext | None = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = False,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    hidden_state_output: bool = False,
    full_logits_output: bool = False,
) -> bytes:
    """Build a TRT engine plan (serialized bytes) for a standard decoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from checkpoint_mapper.
        max_cache_length: KV cache length (engine is compiled for this value).
        precision: Compute precision ("fp32", "fp16", or "bf16").
        norm_type: "rmsnorm" or "layernorm".
        mlp_type: "swiglu" (3 projections: gate/up/down) or
                  "gelu_fc" (2 projections: fc1/fc2 with activation).
        position_type: "rope" (rotary), "learned" (absolute position embeddings),
            or "alibi" (attention with linear biases, no position embeddings).
        activation: Activation function for gelu_fc MLP ("gelu_new", "gelu", "relu", "relu2").
        partial_rotary_factor: Fraction of head dims that get RoPE (default 1.0).
        interleaved_rope: If True, use interleaved RoPE (CodeGen/GPT-J) where
            adjacent dims (d, d+1) share frequencies. Default False uses
            rotated-half (LLaMA/Qwen) where (d, d+half) share frequencies.
        scale_attn_weights: Whether to scale attention scores by 1/sqrt(head_dim).
            Most models use this (True, default). GPT-Neo does NOT scale (False).
        embed_input: If True, add input_embed [1, hidden] and use_input_embed [1]
            engine inputs. When use_input_embed==1, the decoder uses input_embed
            directly instead of the embedding lookup. Used for VL models where
            the vision encoder provides fused embeddings during prefill.
        verbose: Print TRT builder logs.
        debug_layer_outputs: If True, mark per-layer hidden states as network
            outputs for diff testing.

    Returns:
        Serialized engine plan bytes.
    """
    import os as _os

    # Mark the graph as honoring the internal decoder role contract. This is
    # embedded in the mutable config for family helpers that need to branch on
    # the active engine layout while building.
    config.raw["_decoder_engine_layout_supported"] = True
    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))

    # Dispatch to the dynamic-Sq builder for dual-profile and split-prefill
    # engines. Quantized builds (``quant_ctx``) thread Q/DQ insertion through
    # every projection matmul via
    # ``QuantContext.maybe_quantized_matmul``, so they share the dispatch.
    #
    # The legacy single-profile graph below stays in place for paths the
    # dual-profile builder does not yet cover:
    #
    #   - embed_input=True             (VL prefill replacement, Bark sub-engines)
    #   - debug_layer_outputs=True     (per-layer hidden-state dumps)
    #   - hidden_state_output=True     (speech / Bark hidden output)
    #   - config.raw.dynamic_kv_cache  (TriAttention multi-bucket decode)
    #
    # ``TRTMC_NO_DUAL_PROFILE=1`` is an internal escape hatch (perf A/B,
    # bisects against the legacy graph). It is *not* intended as a
    # supported user-facing flag.
    _dual_profile_disabled_for = (
        embed_input
        or debug_layer_outputs
        or hidden_state_output
        or bool(config.raw.get("dynamic_kv_cache", False))
        or _os.environ.get("TRTMC_NO_DUAL_PROFILE") == "1"
    )
    if decoder_engine_role == "prefill" and _dual_profile_disabled_for:
        raise NotImplementedError(
            "split prefill engine is not supported for this standard decoder configuration"
        )
    if not _dual_profile_disabled_for and decoder_engine_role in ("dual_profile", "prefill"):
        return build_dual_profile_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=interleaved_rope,
            parallel_residual=parallel_residual,
            scale_attn_weights=scale_attn_weights,
            verbose=verbose,
            profile_mode=("prefill" if decoder_engine_role == "prefill" else "dual_profile"),
            full_logits_output=full_logits_output,
        )

    if full_logits_output:
        raise NotImplementedError("full_logits_output requires the dual-profile decoder builder")

    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    attention_window = max_cache_length + 1
    dynamic_kv_cache = bool(config.raw.get("dynamic_kv_cache", False))
    dynamic_kv_opt_rows = int(config.raw.get("_dynamic_kv_opt_length", max_cache_length))
    dynamic_kv_opt_rows = max(1, min(dynamic_kv_opt_rows, max_cache_length))
    raw_profile_rows = config.raw.get("_dynamic_kv_profile_rows")
    if raw_profile_rows:
        dynamic_kv_profile_rows = []
        for row in raw_profile_rows:
            clamped = max(1, min(int(row), max_cache_length))
            if clamped not in dynamic_kv_profile_rows:
                dynamic_kv_profile_rows.append(clamped)
        dynamic_kv_profile_rows.sort()
    else:
        dynamic_kv_profile_rows = []

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Precision configuration
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16  # stored as float16, TRT uses bfloat16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    if dynamic_kv_cache and position_type == "alibi":
        raise ValueError("dynamic_kv_cache is not supported for ALiBi decoder builds")

    # ---------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, -1) if dynamic_kv_cache else (1, attention_window)
    )

    # Optional VL inputs: when embed_input=True, the decoder can accept
    # a pre-computed embedding vector instead of a token ID.
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input("input_embed", work_trt_dtype, (1, hidden))
        use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (1,))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            layer_tensor_name("cache_k", i),
            work_trt_dtype,
            (-1, kv_attention_size) if dynamic_kv_cache else (max_cache_length, kv_attention_size),
        )
        cv = network.add_input(
            layer_tensor_name("cache_v", i),
            work_trt_dtype,
            (-1, kv_attention_size) if dynamic_kv_cache else (max_cache_length, kv_attention_size),
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    if dynamic_kv_cache:
        if dynamic_kv_profile_rows:
            for profile_rows in dynamic_kv_profile_rows:
                profile = builder.create_optimization_profile()
                # Keep all profiles valid for short prompts / early decode steps.
                # The profile-specific value is the opt/max row budget, not a
                # lower bound on the live cache length.
                min_rows = 1
                profile.set_shape(
                    "attention_mask",
                    (1, min_rows + 1),
                    (1, profile_rows + 1),
                    (1, profile_rows + 1),
                )
                for i in range(num_layers):
                    min_cache_shape = (min_rows, kv_attention_size)
                    cache_shape = (profile_rows, kv_attention_size)
                    profile.set_shape(
                        layer_tensor_name("cache_k", i),
                        min_cache_shape,
                        cache_shape,
                        cache_shape,
                    )
                    profile.set_shape(
                        layer_tensor_name("cache_v", i),
                        min_cache_shape,
                        cache_shape,
                        cache_shape,
                    )
                trt_config.add_optimization_profile(profile)
        else:
            profile = builder.create_optimization_profile()
            profile.set_shape(
                "attention_mask", (1, 2), (1, dynamic_kv_opt_rows + 1), (1, attention_window)
            )
            for i in range(num_layers):
                profile.set_shape(
                    layer_tensor_name("cache_k", i),
                    (1, kv_attention_size),
                    (dynamic_kv_opt_rows, kv_attention_size),
                    (max_cache_length, kv_attention_size),
                )
                profile.set_shape(
                    layer_tensor_name("cache_v", i),
                    (1, kv_attention_size),
                    (dynamic_kv_opt_rows, kv_attention_size),
                    (max_cache_length, kv_attention_size),
                )
            trt_config.add_optimization_profile(profile)

    # Cast attention mask to work dtype for elementwise compatibility
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

    # ---------------------------------------------------------------
    # Shared constants
    # ---------------------------------------------------------------
    embedding_table = add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )

    # RoPE tables (only needed when position_type == "rope")
    position_embed_table = None
    alibi_slopes_tensor = None
    alibi_indices_tensor = None

    # Native RoPE tensors for IRotaryEmbeddingLayer (TRT 10+).
    # Shape: [attention_window, rotary_ndims // 2].
    cos_half_tensor = None
    sin_half_tensor = None
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    if position_type == "rope":
        validate_native_rope_dim(rotary_embedding_dim)
        cos_half_np = make_rope_table_half_dim(
            attention_window,
            head_dim,
            config.rope_theta,
            True,
            partial_rotary_factor,
            interleaved=interleaved_rope,
        )
        sin_half_np = make_rope_table_half_dim(
            attention_window,
            head_dim,
            config.rope_theta,
            False,
            partial_rotary_factor,
            interleaved=interleaved_rope,
        )
        cos_half_tensor = add_constant(network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
        cos_half_tensor = _cast_work_dtype(cos_half_tensor)
        sin_half_tensor = add_constant(network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
        sin_half_tensor = _cast_work_dtype(sin_half_tensor)
    elif position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = add_constant(
            network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype
        )
    elif position_type == "alibi":
        alibi_slopes_np = compute_alibi_slopes(num_heads)
        alibi_slopes_tensor = add_constant(
            network, (num_heads, 1, 1), alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32
        )
        # Cache position indices [0, 1, ..., max_cache_length-1].
        # The current token's position (position_id) is appended at runtime.
        alibi_indices_tensor = add_constant(
            network,
            (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32),
            dtype=np.float32,
        )

    eps_tensor = add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0
    # ---------------------------------------------------------------
    # Embedding lookup (with optional embed_input override for VL)
    # ---------------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)

    if embed_input and input_embed_tensor is not None and use_input_embed_tensor is not None:
        # Conditional embedding: (1 - flag) * token_embed + flag * input_embed
        # use_input_embed is [1] scalar (FP32), broadcast to [1, hidden]
        flag_broadcast = network.add_shuffle(use_input_embed_tensor)
        flag_broadcast.reshape_dims = (1, 1)
        # Cast flag to work dtype for elementwise compatibility
        flag_for_math = flag_broadcast.get_output(0)
        if work_trt_dtype != trt.float32:
            flag_for_math = network.add_cast(flag_for_math, work_trt_dtype).get_output(0)
        one_const = const_in_work_dtype(
            network, (1, 1), np.array([1.0], dtype=work_np_dtype), work_np_dtype, work_trt_dtype
        )
        token_embed = _cast_work_dtype(token_embed)
        inv_flag = network.add_elementwise(one_const, flag_for_math, trt.ElementWiseOperation.SUB)
        # (1 - flag) * token_embed
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed, trt.ElementWiseOperation.PROD
        )
        # flag * input_embed
        embed_part = network.add_elementwise(
            flag_for_math, input_embed_tensor, trt.ElementWiseOperation.PROD
        )
        # sum
        hidden_state_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0), trt.ElementWiseOperation.SUM
        )
        hidden_state = hidden_state_sum.get_output(0)
    else:
        hidden_state = token_embed

    # Add learned position embedding if applicable
    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
        )
        hidden_state = pos_add.get_output(0)

    # In BF16 mode many embedding/position constants are still materialized from
    # float16 storage. Normalize the decoder's main hidden stream back to the
    # requested runtime dtype before entering the layer stack.
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (e.g. BLOOM) — use native INormalizationLayer
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get("embedding_norm_beta")
        if embed_norm_beta is None:
            embed_norm_beta = np.zeros(hidden, dtype=work_np_dtype)
        hidden_state = add_layer_norm_native(
            network,
            hidden_state,
            hidden,
            embed_norm,
            embed_norm_beta,
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # FFI attention kernel: set by the perf agent on their branch.
    # Default: None (use native TRT attention).
    ffi_attention_kernel = None

    # ---------------------------------------------------------------
    # Decoder layers
    # ---------------------------------------------------------------
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        result = _add_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            attention_scale=attn_scale,
            eps_tensor=eps_tensor,
            eps=config.rms_norm_eps,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            norm_type=norm_type,
            mlp_type=mlp_type,
            position_type=position_type,
            activation=activation,
            parallel_residual=parallel_residual,
            alibi_slopes_tensor=alibi_slopes_tensor,
            alibi_indices_tensor=alibi_indices_tensor,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            rotary_embedding_dim=rotary_embedding_dim,
            interleaved_rope=interleaved_rope,
            ffi_attention_kernel=ffi_attention_kernel,
            dynamic_kv_cache=dynamic_kv_cache,
        )

        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # ---------------------------------------------------------------
    # Final norm
    # ---------------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            weights.get("final_norm_beta"),
            eps_tensor,
            norm_type,
            dtype=work_np_dtype,
            eps=config.rms_norm_eps,
        )

    # Optional: mark hidden state as extra output for speech pipelines
    if hidden_state_output:
        hs_out = network.add_identity(hidden_state).get_output(0)
        hs_out.name = "hidden_state"
        network.mark_output(hs_out)

    # ---------------------------------------------------------------
    # LM head (logits)
    # ---------------------------------------------------------------
    # Output vocab may differ from input vocab (e.g. Bark semantic: 129600 in, 10048 out).
    # Derive from w_out shape if available.
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    # LM head bias (if present, e.g. CodeGen) or zero bias for C++ parity
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        b_out = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = add_bias_sum(network, logits, out_vocab, b_out, dtype=work_np_dtype)

    # Logits output: always FP32 for accurate argmax/sampling
    if work_trt_dtype != trt.float32:
        logits_cast = network.add_cast(logits, trt.float32)
        logits = logits_cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    # ---------------------------------------------------------------
    # Present K/V outputs
    # ---------------------------------------------------------------
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = layer_tensor_name("present_k", i)
        pv.name = layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # ---------------------------------------------------------------
    # Build engine
    # ---------------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building TRT engine ({num_layers} layers, "
            f"hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, "
            f"mlp={mlp_size}, "
            f"cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    return apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type, dtype=dtype, eps=eps
    )


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    attention_scale: float | None,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    mlp_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    parallel_residual: bool = False,
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
    eps: float | None = None,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""

    # Attention block (pre-norm -> QKV -> RoPE -> cache -> attn -> out proj)
    attn = add_attention_block(
        network,
        hidden,
        cache_k,
        cache_v,
        attention_mask,
        position_id,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_length=max_cache_length,
        attention_scale=attention_scale,
        eps_tensor=eps_tensor,
        eps=eps,
        norm_type=norm_type,
        position_type=position_type,
        alibi_slopes_tensor=alibi_slopes_tensor,
        alibi_indices_tensor=alibi_indices_tensor,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=rotary_embedding_dim,
        interleaved_rope=interleaved_rope,
        ffi_attention_kernel=ffi_attention_kernel,
        dynamic_kv_cache=dynamic_kv_cache,
    )
    attn_out = attn["attn_out"]
    present_k = attn["present_k"]
    present_v = attn["present_v"]

    # --- Parallel vs sequential residual ---
    if parallel_residual:
        post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(
                network,
                hidden,
                hidden_size,
                post_attn_norm_w,
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor,
                norm_type,
                dtype=dtype,
                eps=eps,
            )
        else:
            norm2 = attn["normed"]
    else:
        residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _apply_norm(
            network,
            residual1.get_output(0),
            hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor,
            norm_type,
            dtype=dtype,
            eps=eps,
        )

    # MLP
    if mlp_type == "gelu_fc":
        mlp_out = add_gelu_fc_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            activation=activation,
            dtype=dtype,
            quant_ctx=quant_ctx,
            layer_prefix=prefix,
        )
    else:
        mlp_out = add_swiglu_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            dtype=dtype,
            quant_ctx=quant_ctx,
            layer_prefix=prefix,
        )

    # Final residual connection
    if parallel_residual:
        sum_attn = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
        residual2 = network.add_elementwise(
            sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
        )
        post_attn_tensor = sum_attn.get_output(0)
    else:
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
        )
        post_attn_tensor = residual1.get_output(0)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": post_attn_tensor,
        "present_k": present_k,
        "present_v": present_v,
    }

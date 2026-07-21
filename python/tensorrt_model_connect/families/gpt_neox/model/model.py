# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ..config import ModelConfig

if TYPE_CHECKING:
    from ....quantization.context import QuantContext
    from ..weights import WeightDict

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
    GPT-NeoX uses LayerNorm with learned scale and bias.

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


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
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
    Args:
        inp:                  [Sq, num_heads * head_dim].
        num_heads:            Number of attention heads.
        head_dim:             Per-head dimension.
        cos_cache_2d:         Pre-built 2-D cos table constant.
        sin_cache_2d:         Pre-built 2-D sin table constant.
        position_id:          Runtime position indices, shape [Sq] int32.
        rotary_embedding_dim: Number of head dims that participate in RoPE.
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
        False,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
) -> trt.ITensor:
    """Apply masked, scaled dot-product attention to GPT-NeoX Q/K/V."""
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
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
        False,
    )
    attn.decomposable = True
    if mask is not None:
        attn.mask = mask
    return attn.get_output(0)


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
    mask: trt.ITensor | None = None,
    scale: float | None = None,
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
    ctx_4d = add_attention_core(network, q_4d, k_4d, v_4d, mask=mask, scale=scale)
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


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
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply GPT-NeoX LayerNorm at a native TensorRT precision boundary."""
    return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


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
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    attention_scale: float,
    eps: float,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    rotary_embedding_dim: int,
    dynamic_kv_cache: bool = False,
) -> dict[str, trt.ITensor]:
    """Pre-norm -> QKV -> RoPE -> cache concat -> attention -> output proj.

    Returns the projected attention output and current K/V rows.
    Does NOT apply residual -- callers compose the residual pattern.

    This function uses TRT 10 native APIs for the basic transformer primitives:
      - IRotaryEmbeddingLayer for RoPE
      - IAttention for scaled dot-product attention
    """
    matmul = make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1
    normed = apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps,
        dtype=dtype,
    )
    q = matmul(normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
    q = add_bias_sum(network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    k = add_bias_sum(network, k, kv_attention_size, weights[f"{prefix}.k_bias"], dtype=dtype)
    v = add_bias_sum(network, v, kv_attention_size, weights[f"{prefix}.v_bias"], dtype=dtype)
    q = add_apply_rope_native(
        network,
        q,
        num_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
    )
    k = add_apply_rope_native(
        network,
        k,
        num_kv_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
    )
    present_k = k
    present_v = v
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0
    kv_seq = None if dynamic_kv_cache else attention_window
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
        mask=add_2d_mask_to_4d(network, attention_mask),
        scale=attention_scale,
    )
    attn_out = matmul(
        context, attention_size, hidden_size, weights[f"{prefix}.w_o"], f"{prefix}.w_o"
    )
    attn_out = add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype
    )
    return {"attn_out": attn_out, "present_k": present_k, "present_v": present_v}


def add_gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = make_matmul_fn(network, dtype, quant_ctx)
    fc1 = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1 = add_bias_sum(network, fc1, mlp_size, weights[f"{prefix}.fc1_bias"], dtype=dtype)
    activated = add_gelu_new(network, fc1, dtype=dtype)
    fc2 = matmul(activated, mlp_size, hidden_size, weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2 = add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], dtype=dtype)
    return fc2


@dataclass(frozen=True)
class BuilderContext:
    """TensorRT objects shared by engine builders."""

    logger: trt.Logger
    builder: trt.Builder
    network: trt.INetworkDefinition
    config: trt.IBuilderConfig


def create_builder_context(
    *,
    verbose: bool,
    workspace_bytes: int,
    strongly_typed: bool = True,
    disable_tf32: bool = False,
) -> BuilderContext:
    """Create a TensorRT builder, network, and config with common defaults."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if disable_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    return BuilderContext(
        logger=logger,
        builder=builder,
        network=network,
        config=config,
    )


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


def norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    """Apply dynamic-shape GPT-NeoX LayerNorm."""
    return add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)


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
    partial_rotary_factor: float = 1.0,
    parallel_residual: bool = False,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build a TRT engine plan (serialized bytes) for a standard decoder.

    Args:
        config: Model architecture from config.json.
        weights: Loaded weight dict from checkpoint_mapper.
        max_cache_length: KV cache length (engine is compiled for this value).
        precision: Compute precision ("fp32", "fp16", or "bf16").
        partial_rotary_factor: Fraction of head dims that get RoPE (default 1.0).
        parallel_residual: Apply attention and MLP to the same normalized input.
        verbose: Print TRT builder logs.
        debug_layer_outputs: If True, mark per-layer hidden states as network
            outputs for diff testing.

    Returns:
        Serialized engine plan bytes.
    """
    import os as _os

    config.raw["_decoder_engine_layout_supported"] = True
    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))
    requested_fp32_layers = tuple(config.raw.get("_fp32_layers", ()))
    _dual_profile_disabled_for = (
        debug_layer_outputs
        or bool(requested_fp32_layers)
        or bool(config.raw.get("dynamic_kv_cache", False))
        or (_os.environ.get("TRTMC_NO_DUAL_PROFILE") == "1")
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
            partial_rotary_factor=partial_rotary_factor,
            parallel_residual=parallel_residual,
            verbose=verbose,
            profile_mode="prefill" if decoder_engine_role == "prefill" else "dual_profile",
        )
    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    fp32_layers = frozenset((int(layer) for layer in requested_fp32_layers))
    invalid_fp32_layers = sorted(
        (layer for layer in fp32_layers if layer < 0 or layer >= num_layers)
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")
    if precision == "fp32":
        fp32_layers = frozenset()
    if fp32_layers and quant_ctx is not None:
        raise ValueError("fp32_layers is not supported with quantized builds")
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
    builder_context = create_builder_context(verbose=verbose, workspace_bytes=1 << 30)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, -1) if dynamic_kv_cache else (1, attention_window)
    )
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
                        layer_tensor_name("cache_k", i), min_cache_shape, cache_shape, cache_shape
                    )
                    profile.set_shape(
                        layer_tensor_name("cache_v", i), min_cache_shape, cache_shape, cache_shape
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
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

    embedding_table = add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True, partial_rotary_factor
    )
    sin_half_np = make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False, partial_rotary_factor
    )
    cos_half_tensor = add_constant(network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    cos_half_tensor = _cast_work_dtype(cos_half_tensor)
    sin_half_tensor = add_constant(network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    sin_half_tensor = _cast_work_dtype(sin_half_tensor)
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")
    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_is_fp32 = layer_idx in fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

        def _cast_layer_dtype(tensor: trt.ITensor | None) -> trt.ITensor | None:
            if tensor is None or tensor.dtype == layer_trt_dtype:
                return tensor
            return network.add_cast(tensor, layer_trt_dtype).get_output(0)

        result = _add_decoder_layer(
            network=network,
            hidden=_cast_layer_dtype(hidden_state),
            cache_k=_cast_layer_dtype(cache_k_inputs[layer_idx]),
            cache_v=_cast_layer_dtype(cache_v_inputs[layer_idx]),
            attention_mask=_cast_layer_dtype(attention_mask),
            position_id=position_id,
            attention_scale=attn_scale,
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
            parallel_residual=parallel_residual,
            dtype=layer_np_dtype,
            quant_ctx=quant_ctx,
            cos_half_tensor=_cast_layer_dtype(cos_half_tensor),
            sin_half_tensor=_cast_layer_dtype(sin_half_tensor),
            rotary_embedding_dim=rotary_embedding_dim,
            dynamic_kv_cache=dynamic_kv_cache,
        )
        hidden_state = _cast_work_dtype(result["hidden"])
        present_k_outputs.append(_cast_work_dtype(result["present_k"]))
        present_v_outputs.append(_cast_work_dtype(result["present_v"]))
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")
    hidden_state = apply_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        config.rms_norm_eps,
        dtype=work_np_dtype,
    )
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    if work_trt_dtype != trt.float32:
        logits_cast = network.add_cast(logits, trt.float32)
        logits = logits_cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = layer_tensor_name("present_k", i)
        pv.name = layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)
    if verbose:
        print(
            f"[trtmc build] Building TRT engine ({num_layers} layers, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    attention_scale: float,
    eps: float,
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
    parallel_residual: bool = False,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    rotary_embedding_dim: int,
    dynamic_kv_cache: bool = False,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""
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
        eps=eps,
        dtype=dtype,
        quant_ctx=quant_ctx,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=rotary_embedding_dim,
        dynamic_kv_cache=dynamic_kv_cache,
    )
    attn_out = attn["attn_out"]
    present_k = attn["present_k"]
    present_v = attn["present_v"]
    if parallel_residual:
        norm2 = apply_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps,
            dtype=dtype,
        )
    else:
        residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = apply_norm(
            network,
            residual1.get_output(0),
            hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps,
            dtype=dtype,
        )
    mlp_out = add_gelu_fc_mlp(
        network,
        norm2,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
        quant_ctx=quant_ctx,
    )
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
    partial_rotary_factor: float = 1.0,
    parallel_residual: bool = False,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
) -> bytes:
    """Build a prefill/decode-capable dynamic-Sq decoder engine.

    ``partial_rotary_factor`` and ``parallel_residual`` mirror the same
    parameters on ``build_standard_decoder_engine``.

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
    builder_context = create_builder_context(verbose=verbose, workspace_bytes=1 << 30)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.float16)
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.bfloat16)
    else:
        work_np_dtype, work_trt_dtype = (np.float32, trt.float32)
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
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

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
                for name in (layer_tensor_name("cache_k", i), layer_tensor_name("cache_v", i)):
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
        _add_profile(1, 1, fixed=True)
    else:
        _reverse = _os_dbg.environ.get("TRTMC_REVERSE_PROFILE_ORDER", "0") == "1"
        if _reverse:
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
    embedding_table = const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"], work_np_dtype, work_trt_dtype
    )
    kmax = max_cache_length + max_prefill_length
    validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = make_rope_table_half_dim(
        kmax, head_dim, config.rope_theta, True, partial_rotary_factor
    )
    sin_half_np = make_rope_table_half_dim(
        kmax, head_dim, config.rope_theta, False, partial_rotary_factor
    )
    cos_half_table = const_in_work_dtype(
        network, cos_half_np.shape, cos_half_np, work_np_dtype, work_trt_dtype
    )
    sin_half_table = const_in_work_dtype(
        network, sin_half_np.shape, sin_half_np, work_np_dtype, work_trt_dtype
    )
    eps_tensor = add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    matmul = make_matmul_fn(network, work_np_dtype, quant_ctx)
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    mask_4d = add_2d_mask_to_4d(network, attention_mask_work)
    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights[f"{prefix}.input_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        q = matmul(normed, hidden, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
        q = add_bias_sum(
            network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=work_np_dtype
        )
        k = add_bias_sum(
            network, k, kv_attention_size, weights[f"{prefix}.k_bias"], dtype=work_np_dtype
        )
        v = add_bias_sum(
            network, v, kv_attention_size, weights[f"{prefix}.v_bias"], dtype=work_np_dtype
        )
        q = add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_half_table,
            sin_half_table,
            position_id,
            rotary_embedding_dim,
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
            sequence_length=None,
        )
        present_k_outs.append(k)
        present_v_outs.append(v)
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
            mask=mask_4d,
            scale=attn_scale,
            tag=f"{prefix}.attn",
        )
        attn_out = matmul(
            context, attention_size, hidden, weights[f"{prefix}.w_o"], f"{prefix}.w_o"
        )
        attn_out = add_bias_sum(
            network, attn_out, hidden, weights[f"{prefix}.o_bias"], dtype=work_np_dtype
        )
        if parallel_residual:
            norm2 = norm_multi(
                network,
                hidden_state,
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights[f"{prefix}.post_attn_norm_beta"],
                eps_tensor,
                work_np_dtype,
            )
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM
            )
            norm2 = norm_multi(
                network,
                residual1.get_output(0),
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights[f"{prefix}.post_attn_norm_beta"],
                eps_tensor,
                work_np_dtype,
            )
        mlp_out = add_gelu_fc_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
        )
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
    hidden_state = norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        work_np_dtype,
    )
    shape_t = network.add_shape(hidden_state).get_output(0)
    one_hidden = add_constant(network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start_sub = network.add_elementwise(shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)
    size_t = add_constant(network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    slicer.set_input(1, start_t)
    slicer.set_input(2, size_t)
    last_hidden = slicer.get_output(0)
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
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
            f"[trtmc build] Building {mode_label} engine (layers={num_layers}, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, norm={'layernorm'}, mlp_type={'gelu_fc'}, pos={'rope'}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)

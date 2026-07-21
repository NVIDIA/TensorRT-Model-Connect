# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
from typing import TYPE_CHECKING
from dataclasses import dataclass
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


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    q_seq: int | None,
    kv_seq: int | None,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """GPT-Neo native attention for row-major Q/K/V tensors."""
    attention_size = num_heads * head_dim
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
        num_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        num_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    ctx_4d = add_attention_core(
        network,
        q_4d,
        k_4d,
        v_4d,
        causal=False,
        mask=mask,
        scale=scale,
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
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Apply GPT-Neo LayerNorm, using the native operation when possible."""
    if beta is None:
        beta = np.zeros(hidden_size, dtype=np.float32)
    if eps is not None:
        return add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)
    return add_layer_norm(network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype)


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    eps_tensor: trt.ITensor,
    kv_attention_size: int,
    attention_scale: float | None = None,
    eps: float | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    dynamic_kv_cache: bool = False,
) -> dict[str, trt.ITensor]:
    """Add GPT-Neo pre-norm, learned-position attention, and output projection.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.

    TensorRT native attention receives the family-owned additive mask. GPT-Neo
    deliberately uses an attention scale of 1.0.
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1
    _lp = layer_prefix or prefix
    normed = apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps_tensor,
        dtype=dtype,
        eps=eps,
    )
    q = matmul(normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], f"{_lp}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], f"{_lp}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], f"{_lp}.w_v")
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
        q_seq=1,
        kv_seq=kv_seq,
        mask=add_2d_mask_to_4d(network, attention_mask),
        scale=attention_scale,
    )
    attn_out = matmul(context, attention_size, hidden_size, weights[f"{prefix}.w_o"], f"{_lp}.w_o")
    attn_out = add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype
    )
    return {"normed": normed, "attn_out": attn_out, "present_k": present_k, "present_v": present_v}


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
    layer_prefix: str = "",
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix
    fc1 = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1 = add_bias_sum(network, fc1, mlp_size, weights[f"{prefix}.fc1_bias"], dtype=dtype)
    activated = add_gelu_new(network, fc1, dtype=dtype)
    fc2 = matmul(activated, mlp_size, hidden_size, weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2 = add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], dtype=dtype)
    return fc2


# Utils


trt = trt_compat.get_trt()


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
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    dtype: np.dtype,
) -> trt.ITensor:
    """Apply GPT-Neo LayerNorm from a dual-profile builder."""
    if beta is None:
        beta = np.zeros(hidden, dtype=np.float32)
    return add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)


# Default Decoder


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
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build a GPT-Neo TensorRT engine plan.

    Args:
        config: Model architecture from config.json.
        weights: Loaded family-owned weight dictionary.
        max_cache_length: KV cache length (engine is compiled for this value).
        precision: Compute precision ("fp32", "fp16", or "bf16").
        verbose: Print TRT builder logs.
        debug_layer_outputs: If True, mark per-layer hidden states as network
            outputs for diff testing.

    Returns:
        Serialized engine plan bytes.
    """
    import os as _os

    config.raw["_decoder_engine_layout_supported"] = True
    decoder_engine_role = str(config.raw.get("_decoder_engine_role", "dual_profile"))
    _dual_profile_disabled_for = (
        debug_layer_outputs
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
            verbose=verbose,
            profile_mode="prefill" if decoder_engine_role == "prefill" else "dual_profile",
        )
    attention_size: int = weights.get("_attention_size", config.attention_size)
    mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = attention_size // num_heads
    kv_attention_size = infer_kv_attention_size(weights, num_kv_heads=num_heads, head_dim=head_dim)
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
    if work_trt_dtype != trt.float32:
        mask_cast = network.add_cast(attention_mask, work_trt_dtype)
        attention_mask = mask_cast.get_output(0)

    embedding_table = add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    pos_embed_np = weights["position_embedding"]
    position_embed_table = add_constant(
        network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype
    )
    eps_tensor = add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    attn_scale = 1.0
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)
    pos_gather = network.add_gather(position_embed_table, position_id, 0)
    pos_add = network.add_elementwise(
        hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
    )
    hidden_state = pos_add.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")
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
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
            dynamic_kv_cache=dynamic_kv_cache,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")
    hidden_state = _apply_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        dtype=work_np_dtype,
        eps=config.rms_norm_eps,
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


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Apply GPT-Neo LayerNorm."""
    return apply_norm(network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype, eps=eps)


def _add_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    attention_scale: float | None,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    mlp_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    dynamic_kv_cache: bool = False,
    eps: float | None = None,
) -> dict[str, trt.ITensor]:
    """Add one standard decoder layer block. Returns hidden, present_k, present_v."""
    attn = add_attention_block(
        network,
        hidden,
        cache_k,
        cache_v,
        attention_mask,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_cache_length=max_cache_length,
        attention_scale=attention_scale,
        eps_tensor=eps_tensor,
        eps=eps,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
        dynamic_kv_cache=dynamic_kv_cache,
    )
    attn_out = attn["attn_out"]
    present_k = attn["present_k"]
    present_v = attn["present_v"]
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    norm2 = _apply_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps_tensor,
        dtype=dtype,
        eps=eps,
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
        layer_prefix=prefix,
    )
    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
    )
    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


# Default Dual Profile Decoder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


_make_matmul_fn = make_matmul_fn


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1 = add_bias_sum(
        network, fc1, mlp_size, weights[f"{prefix}.fc1_bias"], dtype=work_np_dtype
    )
    activated = add_gelu_new(network, fc1, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden, weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2 = add_bias_sum(
        network, fc2, hidden, weights[f"{prefix}.fc2_bias"], dtype=work_np_dtype
    )
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject non-GPT-Neo configs or incomplete family weights."""
    model_type = getattr(config, "model_type", "").lower()
    if model_type != "gpt_neo":
        raise NotImplementedError(f"GPT-Neo builder does not support model_type={model_type!r}")
    for name in ("embedding", "position_embedding", "final_norm", "w_out"):
        if name not in weights:
            raise NotImplementedError(f"missing GPT-Neo weight: {name}")


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
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
) -> bytes:
    """Build a prefill/decode-capable dynamic-Sq decoder engine.

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
    head_dim = attention_size // num_heads
    kv_attention_size = infer_kv_attention_size(weights, num_kv_heads=num_heads, head_dim=head_dim)
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
    pos_embed_np = weights["position_embedding"]
    position_embed_table = const_in_work_dtype(
        network, pos_embed_np.shape, pos_embed_np, work_np_dtype, work_trt_dtype
    )
    eps_tensor = add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    attn_scale = 1.0
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    pos_gather = network.add_gather(position_embed_table, position_id, 0)
    pos_add = network.add_elementwise(
        hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
    )
    hidden_state = pos_add.get_output(0)
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
        residual1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = norm_multi(
            network,
            residual1.get_output(0),
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        mlp_out = _gelu_fc_mlp(
            network,
            norm2,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
            work_np_dtype=work_np_dtype,
        )
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
            f"[trtmc build] Building {mode_label} engine (layers={num_layers}, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, norm={'layernorm'}, mlp_type={'gelu_fc'}, pos={'learned'}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)

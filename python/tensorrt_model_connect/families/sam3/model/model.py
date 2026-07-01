"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import math
import sys
from ..weights import WeightDict
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
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    inv_sqrt2 = _const(1.0 / np.sqrt(2.0))
    x_scaled = network.add_elementwise(inp, inv_sqrt2, trt.ElementWiseOperation.PROD)
    erf_out = network.add_unary(x_scaled.get_output(0), trt.UnaryOperation.ERF)
    one = _const(1.0)
    one_plus_erf = network.add_elementwise(one, erf_out.get_output(0), trt.ElementWiseOperation.SUM)
    half = _const(0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_erf.get_output(0), trt.ElementWiseOperation.PROD
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
    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_3d,
        sin_cache_3d,
        interleaved,
        rotary_embedding_dim,
    )
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


# Core Builder


def _trt():
    from tensorrt_model_connect import trt_compat

    return trt_compat.get_trt()


def _graph_ops():
    from . import model as graph_ops

    return graph_ops


def _const(network, shape: tuple[int, ...], values, dtype=np.float32):
    return _graph_ops().add_constant(network, shape, np.asarray(values).reshape(shape), dtype=dtype)


def _scalar(network, value: float, rank: int):
    return _const(network, (1,) * max(rank, 1), np.array([value], dtype=np.float32))


def _linear(network, inp, weights: WeightDict, prefix: str, in_size: int, out_size: int):
    graph_ops = _graph_ops()
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_size, out_size, weights[f"{prefix}.weight"]
    )
    return graph_ops.add_bias_sum(network, out, out_size, weights[f"{prefix}.bias"])


def _layer_norm(network, inp, weights: WeightDict, prefix: str, hidden_size: int, eps: float):
    return _graph_ops().add_layer_norm_native(
        network,
        inp,
        hidden_size,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        eps,
    )


def _sam3_mlp(
    network,
    inp,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
):
    graph_ops = _graph_ops()
    out = _linear(network, inp, weights, f"{prefix}.fc1", hidden_size, intermediate_size)
    out = graph_ops.add_activation(network, out, hidden_act)
    return _linear(network, out, weights, f"{prefix}.fc2", intermediate_size, hidden_size)


def _decoder_mlp(network, inp, weights: WeightDict, prefix: str, layer_dims: tuple[int, ...]):
    trt = _trt()
    out = inp
    for idx, (in_size, out_size) in enumerate(zip(layer_dims, layer_dims[1:]), start=1):
        out = _linear(network, out, weights, f"{prefix}.layer{idx}", in_size, out_size)
        if idx < len(layer_dims) - 1:
            out = network.add_activation(out, trt.ActivationType.RELU).get_output(0)
    return out


def _attention(
    network,
    query,
    key,
    value,
    weights: WeightDict,
    prefix: str,
    *,
    hidden_size: int,
    num_heads: int,
    q_seq: int,
    kv_seq: int,
    mask=None,
):
    graph_ops = _graph_ops()
    head_dim = hidden_size // num_heads
    q = _linear(network, query, weights, f"{prefix}.q_proj", hidden_size, hidden_size)
    k = _linear(network, key, weights, f"{prefix}.k_proj", hidden_size, hidden_size)
    v = _linear(network, value, weights, f"{prefix}.v_proj", hidden_size, hidden_size)
    ctx = graph_ops.add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=q_seq,
        kv_seq=kv_seq,
        mask=mask,
    )
    return _linear(network, ctx, weights, f"{prefix}.o_proj", hidden_size, hidden_size)


def _flatten_nchw(network, inp, channels: int, height: int, width: int):
    trt = _trt()
    sh = network.add_shuffle(inp)
    sh.first_transpose = trt.Permutation([0, 2, 3, 1])
    sh.reshape_dims = (height * width, channels)
    return sh.get_output(0)


def _rows_to_nchw(network, inp, channels: int, height: int, width: int):
    trt = _trt()
    sh = network.add_shuffle(inp)
    sh.reshape_dims = (1, height, width, channels)
    out = network.add_shuffle(sh.get_output(0))
    out.first_transpose = trt.Permutation([0, 3, 1, 2])
    return out.get_output(0)


def _text_rows(network, text_features, text_seq_len: int, hidden_size: int):
    sh = network.add_shuffle(text_features)
    sh.reshape_dims = (text_seq_len, hidden_size)
    return sh.get_output(0)


def _text_padding_mask(network, attention_mask, text_seq_len: int):
    trt = _trt()
    mask = network.add_cast(attention_mask, trt.float32).get_output(0)
    one = _const(network, (1, text_seq_len), np.ones((1, text_seq_len), dtype=np.float32))
    inv = network.add_elementwise(one, mask, trt.ElementWiseOperation.SUB).get_output(0)
    neg = _const(
        network,
        (1, text_seq_len),
        np.full((1, text_seq_len), -10000.0, dtype=np.float32),
    )
    additive = network.add_elementwise(inv, neg, trt.ElementWiseOperation.PROD).get_output(0)
    sh = network.add_shuffle(additive)
    sh.reshape_dims = (1, 1, 1, text_seq_len)
    return sh.get_output(0)


def _sigmoid(network, inp):
    return network.add_activation(inp, _trt().ActivationType.SIGMOID).get_output(0)


def _clamp(network, inp, min_value: float, max_value: float):
    trt = _trt()
    rank = len(tuple(inp.shape))
    lo = _scalar(network, min_value, rank)
    hi = _scalar(network, max_value, rank)
    x = network.add_elementwise(inp, lo, trt.ElementWiseOperation.MAX).get_output(0)
    return network.add_elementwise(x, hi, trt.ElementWiseOperation.MIN).get_output(0)


def _inverse_sigmoid(network, inp, eps: float = 1e-3):
    trt = _trt()
    x = _clamp(network, inp, eps, 1.0 - eps)
    one = _scalar(network, 1.0, len(tuple(inp.shape)))
    denom = network.add_elementwise(one, x, trt.ElementWiseOperation.SUB).get_output(0)
    ratio = network.add_elementwise(x, denom, trt.ElementWiseOperation.DIV).get_output(0)
    return network.add_unary(ratio, trt.UnaryOperation.LOG).get_output(0)


def _slice_cols(network, inp, start: int, size: int):
    return network.add_slice(inp, (0, start), (inp.shape[0], size), (1, 1)).get_output(0)


def _cxcywh_to_xyxy(network, boxes):
    trt = _trt()
    cx = _slice_cols(network, boxes, 0, 1)
    cy = _slice_cols(network, boxes, 1, 1)
    w = _slice_cols(network, boxes, 2, 1)
    h = _slice_cols(network, boxes, 3, 1)
    half = _scalar(network, 0.5, 2)
    half_w = network.add_elementwise(w, half, trt.ElementWiseOperation.PROD).get_output(0)
    half_h = network.add_elementwise(h, half, trt.ElementWiseOperation.PROD).get_output(0)
    x1 = network.add_elementwise(cx, half_w, trt.ElementWiseOperation.SUB).get_output(0)
    y1 = network.add_elementwise(cy, half_h, trt.ElementWiseOperation.SUB).get_output(0)
    x2 = network.add_elementwise(cx, half_w, trt.ElementWiseOperation.SUM).get_output(0)
    y2 = network.add_elementwise(cy, half_h, trt.ElementWiseOperation.SUM).get_output(0)
    concat = network.add_concatenation([x1, y1, x2, y2])
    concat.axis = 1
    return concat.get_output(0)


def _signed_log_scale(network, inp):
    trt = _trt()
    eight = _scalar(network, 8.0, len(tuple(inp.shape)))
    scaled = network.add_elementwise(inp, eight, trt.ElementWiseOperation.PROD).get_output(0)
    abs_scaled = network.add_unary(scaled, trt.UnaryOperation.ABS).get_output(0)
    eps = _scalar(network, 1e-6, len(tuple(inp.shape)))
    safe_abs = network.add_elementwise(abs_scaled, eps, trt.ElementWiseOperation.MAX).get_output(0)
    sign = network.add_elementwise(scaled, safe_abs, trt.ElementWiseOperation.DIV).get_output(0)
    one = _scalar(network, 1.0, len(tuple(inp.shape)))
    plus_one = network.add_elementwise(abs_scaled, one, trt.ElementWiseOperation.SUM).get_output(0)
    logged = network.add_unary(plus_one, trt.UnaryOperation.LOG).get_output(0)
    inv_log8 = _scalar(network, 1.0 / math.log(8.0), len(tuple(inp.shape)))
    logged = network.add_elementwise(logged, inv_log8, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(sign, logged, trt.ElementWiseOperation.PROD).get_output(0)


def _box_sine_position(network, boxes, hidden_size: int, num_queries: int):
    trt = _trt()
    num_pos = hidden_size // 2
    dim_t = 10000.0 ** (
        2 * (np.arange(num_pos, dtype=np.int32) // 2).astype(np.float32) / float(num_pos)
    )
    dim_t_t = _const(network, (1, num_pos), dim_t.reshape(1, num_pos))
    even_mask = (np.arange(num_pos) % 2 == 0).astype(np.float32).reshape(1, num_pos)
    odd_mask = 1.0 - even_mask
    even_t = _const(network, (1, num_pos), even_mask)
    odd_t = _const(network, (1, num_pos), odd_mask)
    scale = _scalar(network, 2.0 * math.pi, 2)

    pieces = []
    for col in (1, 0, 2, 3):
        component = _slice_cols(network, boxes, col, 1)
        scaled = network.add_elementwise(
            component, scale, trt.ElementWiseOperation.PROD
        ).get_output(0)
        values = network.add_elementwise(scaled, dim_t_t, trt.ElementWiseOperation.DIV).get_output(
            0
        )
        sin = network.add_unary(values, trt.UnaryOperation.SIN).get_output(0)
        cos = network.add_unary(values, trt.UnaryOperation.COS).get_output(0)
        sin_part = network.add_elementwise(sin, even_t, trt.ElementWiseOperation.PROD).get_output(0)
        cos_part = network.add_elementwise(cos, odd_t, trt.ElementWiseOperation.PROD).get_output(0)
        pieces.append(
            network.add_elementwise(sin_part, cos_part, trt.ElementWiseOperation.SUM).get_output(0)
        )

    concat = network.add_concatenation(pieces)
    concat.axis = 1
    return concat.get_output(0)


def _box_rpb(
    network,
    boxes,
    weights: WeightDict,
    *,
    height: int,
    width: int,
    hidden_size: int,
    num_heads: int,
    num_queries: int,
):
    trt = _trt()
    xyxy = _cxcywh_to_xyxy(network, boxes)
    x_edges = network.add_concatenation(
        [
            _slice_cols(network, xyxy, 0, 1),
            _slice_cols(network, xyxy, 2, 1),
        ]
    )
    x_edges.axis = 1
    y_edges = network.add_concatenation(
        [
            _slice_cols(network, xyxy, 1, 1),
            _slice_cols(network, xyxy, 3, 1),
        ]
    )
    y_edges.axis = 1

    x_edge_sh = network.add_shuffle(x_edges.get_output(0))
    x_edge_sh.reshape_dims = (num_queries, 1, 2)
    y_edge_sh = network.add_shuffle(y_edges.get_output(0))
    y_edge_sh.reshape_dims = (num_queries, 1, 2)

    coords_w = (np.arange(width, dtype=np.float32) / float(width)).reshape(1, width, 1)
    coords_h = (np.arange(height, dtype=np.float32) / float(height)).reshape(1, height, 1)
    x_deltas = network.add_elementwise(
        _const(network, (1, width, 1), coords_w),
        x_edge_sh.get_output(0),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    y_deltas = network.add_elementwise(
        _const(network, (1, height, 1), coords_h),
        y_edge_sh.get_output(0),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    x_log = _signed_log_scale(network, x_deltas)
    y_log = _signed_log_scale(network, y_deltas)
    x_embed = _decoder_mlp(network, x_log, weights, "box_rpb_embed_x", (2, hidden_size, num_heads))
    y_embed = _decoder_mlp(network, y_log, weights, "box_rpb_embed_y", (2, hidden_size, num_heads))

    x_sh = network.add_shuffle(x_embed)
    x_sh.reshape_dims = (num_queries, 1, width, num_heads)
    y_sh = network.add_shuffle(y_embed)
    y_sh.reshape_dims = (num_queries, height, 1, num_heads)
    rpb = network.add_elementwise(
        y_sh.get_output(0), x_sh.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    perm = network.add_shuffle(rpb)
    perm.first_transpose = trt.Permutation([3, 0, 1, 2])
    perm.reshape_dims = (num_heads, num_queries, height * width)
    zeros = _const(
        network,
        (num_heads, 1, height * width),
        np.zeros((num_heads, 1, height * width), dtype=np.float32),
    )
    padded = network.add_concatenation([zeros, perm.get_output(0)])
    padded.axis = 1
    out = network.add_shuffle(padded.get_output(0))
    out.reshape_dims = (1, num_heads, num_queries + 1, height * width)
    return out.get_output(0)


def _group_norm_4d(
    network, inp, weights: WeightDict, prefix: str, *, channels: int, groups: int, eps: float
):
    trt = _trt()
    n, c, h, w = inp.shape
    group_size = channels // groups
    sh = network.add_shuffle(inp)
    sh.reshape_dims = (n, groups, group_size, h, w)
    x = sh.get_output(0)
    reduce_axes = (1 << 2) | (1 << 3) | (1 << 4)
    sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(x, trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
    mean_sq = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, reduce_axes, keep_dims=True
    )
    var = network.add_elementwise(
        mean_sq.get_output(0),
        network.add_elementwise(
            mean.get_output(0), mean.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0),
        trt.ElementWiseOperation.SUB,
    )
    eps_t = _const(network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32))
    denom = network.add_unary(
        network.add_elementwise(var.get_output(0), eps_t, trt.ElementWiseOperation.SUM).get_output(
            0
        ),
        trt.UnaryOperation.SQRT,
    )
    recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
    centered = network.add_elementwise(x, mean.get_output(0), trt.ElementWiseOperation.SUB)
    normed = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    out = network.add_shuffle(normed.get_output(0))
    out.reshape_dims = (n, c, h, w)
    gamma = _const(
        network,
        (1, channels, 1, 1),
        weights[f"{prefix}.weight"].reshape(1, channels, 1, 1),
    )
    beta = _const(
        network,
        (1, channels, 1, 1),
        weights[f"{prefix}.bias"].reshape(1, channels, 1, 1),
    )
    scaled = network.add_elementwise(
        out.get_output(0), gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(scaled, beta, trt.ElementWiseOperation.SUM).get_output(0)


def _nearest_resize_2d(network, inp, target_h: int, target_w: int):
    trt = _trt()
    n, c = inp.shape[0], inp.shape[1]
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (n, c, target_h, target_w)
    return resize.get_output(0)


def _weighted_text_pool(
    network, text_features, attention_mask, *, text_seq_len: int, hidden_size: int
):
    trt = _trt()
    mask = network.add_cast(attention_mask, trt.float32).get_output(0)
    mask_rows = network.add_shuffle(mask)
    mask_rows.reshape_dims = (text_seq_len, 1)
    weighted = network.add_elementwise(
        text_features, mask_rows.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    total = network.add_reduce(
        weighted, trt.ReduceOperation.SUM, 1 << 0, keep_dims=True
    ).get_output(0)
    count = network.add_reduce(
        mask_rows.get_output(0), trt.ReduceOperation.SUM, 1 << 0, keep_dims=True
    ).get_output(0)
    one = _const(network, (1, 1), np.array([[1.0]], dtype=np.float32))
    count = network.add_elementwise(count, one, trt.ElementWiseOperation.MAX).get_output(0)
    return network.add_elementwise(total, count, trt.ElementWiseOperation.DIV).get_output(0)


def build_sam3_core_engine(
    weights: WeightDict,
    *,
    text_seq_len: int,
    hidden_size: int,
    fpn_hidden_size: int,
    fpn_shapes: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    num_queries: int,
    detr_encoder_layers: int,
    detr_encoder_heads: int,
    detr_encoder_intermediate_size: int,
    detr_decoder_layers: int,
    detr_decoder_heads: int,
    detr_decoder_intermediate_size: int,
    mask_num_heads: int,
    mask_num_upsampling_stages: int,
    layer_norm_eps: float,
    encoder_hidden_act: str = "relu",
    decoder_hidden_act: str = "relu",
    verbose: bool = False,
) -> bytes:
    """Build the SAM3 text-prompt core engine with TensorRT APIs."""
    del mask_num_upsampling_stages
    if hidden_size != fpn_hidden_size:
        raise ValueError(
            "SAM3 core builder expects DETR and FPN hidden sizes to match; "
            f"got hidden={hidden_size}, fpn={fpn_hidden_size}"
        )
    trt = _trt()
    graph_ops = _graph_ops()

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

    text_features_in = network.add_input(
        "sam3_text_features", trt.float32, (1, text_seq_len, hidden_size)
    )
    text_mask_in = network.add_input("sam3_text_attention_mask", trt.int32, (1, text_seq_len))

    fpn_hidden = []
    fpn_position = []
    for level, (height, width) in enumerate(fpn_shapes):
        fpn_hidden.append(
            network.add_input(
                f"sam3_fpn_hidden_{level}", trt.float32, (1, hidden_size, height, width)
            )
        )
        fpn_position.append(
            network.add_input(
                f"sam3_fpn_position_{level}", trt.float32, (1, hidden_size, height, width)
            )
        )

    text_features = _text_rows(network, text_features_in, text_seq_len, hidden_size)
    text_mask = _text_padding_mask(network, text_mask_in, text_seq_len)

    enc_h, enc_w = fpn_shapes[2]
    seq_len = enc_h * enc_w
    vision_features = _flatten_nchw(network, fpn_hidden[2], hidden_size, enc_h, enc_w)
    vision_pos = _flatten_nchw(network, fpn_position[2], hidden_size, enc_h, enc_w)

    encoder_hidden = vision_features
    for layer_idx in range(detr_encoder_layers):
        prefix = f"detr_encoder.layers.{layer_idx}"
        residual = encoder_hidden
        normed = _layer_norm(
            network, encoder_hidden, weights, f"{prefix}.layer_norm1", hidden_size, layer_norm_eps
        )
        with_pos = network.add_elementwise(
            normed, vision_pos, trt.ElementWiseOperation.SUM
        ).get_output(0)
        attn = _attention(
            network,
            with_pos,
            with_pos,
            normed,
            weights,
            f"{prefix}.self_attn",
            hidden_size=hidden_size,
            num_heads=detr_encoder_heads,
            q_seq=seq_len,
            kv_seq=seq_len,
        )
        encoder_hidden = network.add_elementwise(
            residual, attn, trt.ElementWiseOperation.SUM
        ).get_output(0)

        residual = encoder_hidden
        normed = _layer_norm(
            network, encoder_hidden, weights, f"{prefix}.layer_norm2", hidden_size, layer_norm_eps
        )
        cross = _attention(
            network,
            normed,
            text_features,
            text_features,
            weights,
            f"{prefix}.cross_attn",
            hidden_size=hidden_size,
            num_heads=detr_encoder_heads,
            q_seq=seq_len,
            kv_seq=text_seq_len,
            mask=text_mask,
        )
        encoder_hidden = network.add_elementwise(
            residual, cross, trt.ElementWiseOperation.SUM
        ).get_output(0)

        residual = encoder_hidden
        normed = _layer_norm(
            network, encoder_hidden, weights, f"{prefix}.layer_norm3", hidden_size, layer_norm_eps
        )
        mlp = _sam3_mlp(
            network,
            normed,
            weights,
            f"{prefix}.mlp",
            hidden_size,
            detr_encoder_intermediate_size,
            encoder_hidden_act,
        )
        encoder_hidden = network.add_elementwise(
            residual, mlp, trt.ElementWiseOperation.SUM
        ).get_output(0)

    query_embed = _const(network, (num_queries, hidden_size), weights["query_embed.weight"])
    reference_boxes = _sigmoid(
        network, _const(network, (num_queries, 4), weights["reference_points.weight"])
    )
    presence = _const(network, (1, hidden_size), weights["presence_token.weight"])
    hidden = network.add_concatenation([presence, query_embed])
    hidden.axis = 0
    decoder_hidden = hidden.get_output(0)
    last_presence_logits = None

    for layer_idx in range(detr_decoder_layers):
        prefix = f"detr_decoder.layers.{layer_idx}"
        query_sine = _box_sine_position(network, reference_boxes, hidden_size, num_queries)
        query_pos = _decoder_mlp(
            network,
            query_sine,
            weights,
            "ref_point_head",
            (hidden_size * 2, hidden_size, hidden_size),
        )
        zero_pos = _const(network, (1, hidden_size), np.zeros((1, hidden_size), dtype=np.float32))
        query_pos_padded = network.add_concatenation([zero_pos, query_pos])
        query_pos_padded.axis = 0
        query_pos_t = query_pos_padded.get_output(0)

        residual = decoder_hidden
        query_with_pos = network.add_elementwise(
            decoder_hidden, query_pos_t, trt.ElementWiseOperation.SUM
        ).get_output(0)
        attn = _attention(
            network,
            query_with_pos,
            query_with_pos,
            decoder_hidden,
            weights,
            f"{prefix}.self_attn",
            hidden_size=hidden_size,
            num_heads=detr_decoder_heads,
            q_seq=num_queries + 1,
            kv_seq=num_queries + 1,
        )
        decoder_hidden = network.add_elementwise(
            residual, attn, trt.ElementWiseOperation.SUM
        ).get_output(0)
        decoder_hidden = _layer_norm(
            network,
            decoder_hidden,
            weights,
            f"{prefix}.self_attn_layer_norm",
            hidden_size,
            layer_norm_eps,
        )

        residual = decoder_hidden
        query_with_pos = network.add_elementwise(
            decoder_hidden, query_pos_t, trt.ElementWiseOperation.SUM
        ).get_output(0)
        attn = _attention(
            network,
            query_with_pos,
            text_features,
            text_features,
            weights,
            f"{prefix}.text_cross_attn",
            hidden_size=hidden_size,
            num_heads=detr_decoder_heads,
            q_seq=num_queries + 1,
            kv_seq=text_seq_len,
            mask=text_mask,
        )
        decoder_hidden = network.add_elementwise(
            residual, attn, trt.ElementWiseOperation.SUM
        ).get_output(0)
        decoder_hidden = _layer_norm(
            network,
            decoder_hidden,
            weights,
            f"{prefix}.text_cross_attn_layer_norm",
            hidden_size,
            layer_norm_eps,
        )

        residual = decoder_hidden
        query_with_pos = network.add_elementwise(
            decoder_hidden, query_pos_t, trt.ElementWiseOperation.SUM
        ).get_output(0)
        key_with_pos = network.add_elementwise(
            encoder_hidden, vision_pos, trt.ElementWiseOperation.SUM
        ).get_output(0)
        rpb = _box_rpb(
            network,
            reference_boxes,
            weights,
            height=enc_h,
            width=enc_w,
            hidden_size=hidden_size,
            num_heads=detr_decoder_heads,
            num_queries=num_queries,
        )
        attn = _attention(
            network,
            query_with_pos,
            key_with_pos,
            encoder_hidden,
            weights,
            f"{prefix}.vision_cross_attn",
            hidden_size=hidden_size,
            num_heads=detr_decoder_heads,
            q_seq=num_queries + 1,
            kv_seq=seq_len,
            mask=rpb,
        )
        decoder_hidden = network.add_elementwise(
            residual, attn, trt.ElementWiseOperation.SUM
        ).get_output(0)
        decoder_hidden = _layer_norm(
            network,
            decoder_hidden,
            weights,
            f"{prefix}.vision_cross_attn_layer_norm",
            hidden_size,
            layer_norm_eps,
        )

        residual = decoder_hidden
        mlp = _sam3_mlp(
            network,
            decoder_hidden,
            weights,
            f"{prefix}.mlp",
            hidden_size,
            detr_decoder_intermediate_size,
            decoder_hidden_act,
        )
        decoder_hidden = network.add_elementwise(
            residual, mlp, trt.ElementWiseOperation.SUM
        ).get_output(0)
        decoder_hidden = _layer_norm(
            network,
            decoder_hidden,
            weights,
            f"{prefix}.mlp_layer_norm",
            hidden_size,
            layer_norm_eps,
        )

        query_hidden = network.add_slice(
            decoder_hidden, (1, 0), (num_queries, hidden_size), (1, 1)
        ).get_output(0)
        normalized_queries = _layer_norm(
            network,
            query_hidden,
            weights,
            "detr_decoder.output_layer_norm",
            hidden_size,
            layer_norm_eps,
        )
        delta_boxes = _decoder_mlp(
            network,
            normalized_queries,
            weights,
            "box_head",
            (hidden_size, hidden_size, hidden_size, 4),
        )
        reference_boxes = _sigmoid(
            network,
            network.add_elementwise(
                delta_boxes,
                _inverse_sigmoid(network, reference_boxes),
                trt.ElementWiseOperation.SUM,
            ).get_output(0),
        )

        presence_hidden = network.add_slice(
            decoder_hidden, (0, 0), (1, hidden_size), (1, 1)
        ).get_output(0)
        presence_hidden = _layer_norm(
            network, presence_hidden, weights, "presence_layer_norm", hidden_size, layer_norm_eps
        )
        last_presence_logits = _decoder_mlp(
            network,
            presence_hidden,
            weights,
            "presence_head",
            (hidden_size, hidden_size, hidden_size, 1),
        )
        last_presence_logits = _clamp(network, last_presence_logits, -10.0, 10.0)

    decoder_queries = normalized_queries
    pred_boxes = _cxcywh_to_xyxy(network, reference_boxes)
    pred_boxes_out = network.add_shuffle(pred_boxes)
    pred_boxes_out.reshape_dims = (1, num_queries, 4)
    pred_boxes_t = pred_boxes_out.get_output(0)
    pred_boxes_t.name = "pred_boxes"
    network.mark_output(pred_boxes_t)

    prompt_features = text_features
    text_residual = text_features
    text_features = _decoder_mlp(
        network,
        text_features,
        weights,
        "dot_product_scoring.text_mlp",
        (hidden_size, detr_decoder_intermediate_size, hidden_size),
    )
    text_features = network.add_elementwise(
        text_residual, text_features, trt.ElementWiseOperation.SUM
    ).get_output(0)
    text_features = _layer_norm(
        network,
        text_features,
        weights,
        "dot_product_scoring.text_mlp_out_norm",
        hidden_size,
        layer_norm_eps,
    )
    pooled_text = _weighted_text_pool(
        network, text_features, text_mask_in, text_seq_len=text_seq_len, hidden_size=hidden_size
    )
    proj_text = _linear(
        network, pooled_text, weights, "dot_product_scoring.text_proj", hidden_size, hidden_size
    )
    proj_queries = _linear(
        network,
        decoder_queries,
        weights,
        "dot_product_scoring.query_proj",
        hidden_size,
        hidden_size,
    )
    scores = network.add_matrix_multiply(
        proj_queries, trt.MatrixOperation.NONE, proj_text, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scale = _scalar(network, 1.0 / math.sqrt(float(hidden_size)), 2)
    scores = network.add_elementwise(scores, scale, trt.ElementWiseOperation.PROD).get_output(0)
    scores = _clamp(network, scores, -12.0, 12.0)
    pred_logits = network.add_shuffle(scores)
    pred_logits.reshape_dims = (1, num_queries)
    pred_logits_t = pred_logits.get_output(0)
    pred_logits_t.name = "pred_logits"
    network.mark_output(pred_logits_t)

    prompt_norm = _layer_norm(
        network,
        encoder_hidden,
        weights,
        "mask_decoder.prompt_cross_attn_norm",
        hidden_size,
        layer_norm_eps,
    )
    prompt_attn = _attention(
        network,
        prompt_norm,
        prompt_features,
        prompt_features,
        weights,
        "mask_decoder.prompt_cross_attn",
        hidden_size=hidden_size,
        num_heads=mask_num_heads,
        q_seq=seq_len,
        kv_seq=text_seq_len,
        mask=text_mask,
    )
    encoder_for_masks = network.add_elementwise(
        encoder_hidden, prompt_attn, trt.ElementWiseOperation.SUM
    ).get_output(0)

    pixel = _rows_to_nchw(network, encoder_for_masks, hidden_size, enc_h, enc_w)
    for pixel_idx, level in enumerate((1, 0)):
        target_h, target_w = fpn_shapes[level]
        pixel = _nearest_resize_2d(network, pixel, target_h, target_w)
        pixel = network.add_elementwise(
            pixel, fpn_hidden[level], trt.ElementWiseOperation.SUM
        ).get_output(0)
        pixel = graph_ops.add_conv2d(
            network,
            pixel,
            weights[f"mask_decoder.pixel_decoder.conv_layers.{pixel_idx}.weight"],
            weights[f"mask_decoder.pixel_decoder.conv_layers.{pixel_idx}.bias"],
            hidden_size,
            (3, 3),
            padding=(1, 1),
        )
        pixel = _group_norm_4d(
            network,
            pixel,
            weights,
            f"mask_decoder.pixel_decoder.norms.{pixel_idx}",
            channels=hidden_size,
            groups=8,
            eps=layer_norm_eps,
        )
        pixel = network.add_activation(pixel, trt.ActivationType.RELU).get_output(0)

    instance = graph_ops.add_conv2d(
        network,
        pixel,
        weights["mask_decoder.instance_projection.weight"],
        weights["mask_decoder.instance_projection.bias"],
        hidden_size,
        (1, 1),
    )
    mask_embeddings = decoder_queries
    for idx in range(3):
        mask_embeddings = _linear(
            network,
            mask_embeddings,
            weights,
            f"mask_decoder.mask_embedder.layers.{idx}",
            hidden_size,
            hidden_size,
        )
        if idx < 2:
            mask_embeddings = network.add_activation(
                mask_embeddings, trt.ActivationType.RELU
            ).get_output(0)

    mask_h, mask_w = fpn_shapes[0]
    instance_flat = network.add_shuffle(instance)
    instance_flat.reshape_dims = (hidden_size, mask_h * mask_w)
    masks = network.add_matrix_multiply(
        mask_embeddings,
        trt.MatrixOperation.NONE,
        instance_flat.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    masks_out = network.add_shuffle(masks)
    masks_out.reshape_dims = (1, num_queries, mask_h, mask_w)
    masks_t = masks_out.get_output(0)
    masks_t.name = "pred_masks"
    network.mark_output(masks_t)

    if last_presence_logits is not None:
        presence_out = network.add_shuffle(last_presence_logits)
        presence_out.reshape_dims = (1, 1)
        presence_t = presence_out.get_output(0)
        presence_t.name = "presence_logits"
        network.mark_output(presence_t)

    if verbose:
        print(
            "[sam3-core-builder] Building TRT engine "
            f"(text={text_seq_len}, hidden={hidden_size}, queries={num_queries}, "
            f"vision={enc_h}x{enc_w}, mask={mask_h}x{mask_w}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for SAM3 core")
    return bytes(plan)

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import sys
from typing import TYPE_CHECKING
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


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


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
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        scale = network.add_cast(scale, trt.float32).get_output(0)
        shift = network.add_cast(shift, trt.float32).get_output(0)
    # Standard LayerNorm without affine
    eps_t = add_constant(network, (1, 1), np.array([eps], dtype=np.float32), dtype=np.float32)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom = network.add_unary(
        network.add_elementwise(var.get_output(0), eps_t, trt.ElementWiseOperation.SUM).get_output(
            0
        ),
        trt.UnaryOperation.SQRT,
    )
    recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Adaptive modulation: norm(x) * (1 + scale) + shift
    one = add_constant(network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32)
    scale_plus_one = network.add_elementwise(one, scale, trt.ElementWiseOperation.SUM)
    scaled = network.add_elementwise(
        normalized.get_output(0), scale_plus_one.get_output(0), trt.ElementWiseOperation.PROD
    )
    result = network.add_elementwise(
        scaled.get_output(0), shift, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


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
            np.log(n_clamped / max_exact) / np.log(max_dist / max_exact) * (num_bkts - max_exact)
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


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


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


# Standard Dit Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict


def build_standard_dit_engine(
    weights: WeightDict,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    context_dim: int,
    num_patches: int,
    text_seq_len: int = 512,
    eps: float = 1e-06,
    verbose: bool = False,
) -> bytes:
    """Build DiT denoiser TRT engine plan.

    Args:
        weights: Weight dict with DiT weights. Expected keys per layer:
            - blocks.{i}.attn1.to_q/to_k/to_v.weight/bias (self-attn)
            - blocks.{i}.attn1.to_out.0.weight/bias
            - blocks.{i}.attn1.norm_q/norm_k.weight (QK norm)
            - blocks.{i}.norm1 (no weight — elementwise_affine=False)
            - blocks.{i}.attn2.to_q/to_k/to_v.weight/bias (cross-attn)
            - blocks.{i}.attn2.to_out.0.weight/bias
            - blocks.{i}.attn2.norm_q/norm_k.weight
            - blocks.{i}.attn2.add_k_proj/add_v_proj.weight/bias (if context needs projection)
            - blocks.{i}.norm2.weight/bias (cross-attn norm, if enabled)
            - blocks.{i}.ffn.net.0.proj.weight/bias (GELU)
            - blocks.{i}.ffn.net.2.weight/bias (output proj)
            - blocks.{i}.norm3 (no weight — elementwise_affine=False)
            - blocks.{i}.scale_shift_table [1, 6, dim]
            Global:
            - norm_out (no weight — elementwise_affine=False)
            - proj_out.weight/bias
            - scale_shift_table [1, 2, dim]
        dim: Hidden dimension of the DiT.
        num_heads: Number of attention heads.
        num_layers: Number of DiT blocks.
        ffn_dim: Feed-forward inner dimension.
        context_dim: Text encoder output dimension (before projection).
        num_patches: Total number of patches (T/pt * H/ph * W/pw).
        text_seq_len: Maximum text sequence length.
        qk_norm: Apply RMSNorm to Q and K.
        cross_attn_norm: Apply LayerNorm before cross-attention.
        ffn_activation: Activation for FFN.
        use_rope: Apply RoPE to self-attention Q/K. When False, the engine
            omits rotary_cos/rotary_sin inputs (suitable for models that use
            fixed position embeddings, e.g. PixArt).
        eps: LayerNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    head_dim = dim // num_heads
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    hidden_inp = network.add_input("hidden_states", trt.float32, (num_patches, dim))
    temb_inp = network.add_input("timestep_embedding", trt.float32, (1, 6 * dim))
    time_embed_inp = network.add_input("time_embed", trt.float32, (1, dim))
    encoder_hidden = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, context_dim)
    )
    cross_attn_mask = None
    cross_attn_mask = network.add_input("encoder_attention_mask", trt.float32, (1, 1, text_seq_len))
    add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    hidden = hidden_inp
    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        sst = weights[f"{prefix}.scale_shift_table"]
        sst_const = add_constant(network, (1, 6 * dim), sst.reshape(1, 6 * dim))
        modulation = network.add_elementwise(sst_const, temb_inp, trt.ElementWiseOperation.SUM)
        chunks = []
        for i in range(6):
            s = network.add_slice(
                modulation.get_output(0), start=(0, i * dim), shape=(1, dim), stride=(1, 1)
            )
            chunks.append(s.get_output(0))
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = chunks
        normed = add_adaptive_layernorm(network, hidden, scale_sa, shift_sa, dim, eps)
        q = add_matmul_rhs_constant(
            network, normed, dim, dim, weights[f"{prefix}.attn1.to_q.weight"]
        )
        k = add_matmul_rhs_constant(
            network, normed, dim, dim, weights[f"{prefix}.attn1.to_k.weight"]
        )
        v = add_matmul_rhs_constant(
            network, normed, dim, dim, weights[f"{prefix}.attn1.to_v.weight"]
        )
        q_bias = weights.get(f"{prefix}.attn1.to_q.bias")
        if q_bias is not None:
            q = add_bias_sum(network, q, dim, q_bias)
        k_bias = weights.get(f"{prefix}.attn1.to_k.bias")
        if k_bias is not None:
            k = add_bias_sum(network, k, dim, k_bias)
        v_bias = weights.get(f"{prefix}.attn1.to_v.bias")
        if v_bias is not None:
            v = add_bias_sum(network, v, dim, v_bias)
        context_flat = add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=num_patches,
            kv_seq=num_patches,
            tag=f"{prefix}.attn1",
        )
        attn_out = add_matmul_rhs_constant(
            network, context_flat, dim, dim, weights[f"{prefix}.attn1.to_out.0.weight"]
        )
        o_bias = weights.get(f"{prefix}.attn1.to_out.0.bias")
        if o_bias is not None:
            attn_out = add_bias_sum(network, attn_out, dim, o_bias)
        gated = network.add_elementwise(attn_out, gate_sa, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)
        cross_normed = hidden
        cross_q = add_matmul_rhs_constant(
            network, cross_normed, dim, dim, weights[f"{prefix}.attn2.to_q.weight"]
        )
        cq_bias = weights.get(f"{prefix}.attn2.to_q.bias")
        if cq_bias is not None:
            cross_q = add_bias_sum(network, cross_q, dim, cq_bias)
        add_k_proj_w = weights.get(f"{prefix}.attn2.add_k_proj.weight")
        if add_k_proj_w is not None:
            cross_k = add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim, add_k_proj_w
            )
            add_k_bias = weights.get(f"{prefix}.attn2.add_k_proj.bias")
            if add_k_bias is not None:
                cross_k = add_bias_sum(network, cross_k, dim, add_k_bias)
            cross_v = add_matmul_rhs_constant(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights[f"{prefix}.attn2.add_v_proj.weight"],
            )
            add_v_bias = weights.get(f"{prefix}.attn2.add_v_proj.bias")
            if add_v_bias is not None:
                cross_v = add_bias_sum(network, cross_v, dim, add_v_bias)
        else:
            cross_k = add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim, weights[f"{prefix}.attn2.to_k.weight"]
            )
            ck_bias = weights.get(f"{prefix}.attn2.to_k.bias")
            if ck_bias is not None:
                cross_k = add_bias_sum(network, cross_k, dim, ck_bias)
            cross_v = add_matmul_rhs_constant(
                network, encoder_hidden, context_dim, dim, weights[f"{prefix}.attn2.to_v.weight"]
            )
            cv_bias = weights.get(f"{prefix}.attn2.to_v.bias")
            if cv_bias is not None:
                cross_v = add_bias_sum(network, cross_v, dim, cv_bias)
        cross_mask_4d = None
        if cross_attn_mask is not None:
            cross_mask = network.add_shuffle(cross_attn_mask)
            cross_mask.reshape_dims = (1, 1, 1, text_seq_len)
            cross_mask_4d = cross_mask.get_output(0)
        c_context_flat = add_attention_from_rows(
            network,
            cross_q,
            cross_k,
            cross_v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=num_patches,
            kv_seq=text_seq_len,
            mask=cross_mask_4d,
            tag=f"{prefix}.attn2",
        )
        cross_out = add_matmul_rhs_constant(
            network, c_context_flat, dim, dim, weights[f"{prefix}.attn2.to_out.0.weight"]
        )
        co_bias = weights.get(f"{prefix}.attn2.to_out.0.bias")
        if co_bias is not None:
            cross_out = add_bias_sum(network, cross_out, dim, co_bias)
        hidden = network.add_elementwise(
            hidden, cross_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        ffn_normed = add_adaptive_layernorm(network, hidden, scale_ff, shift_ff, dim, eps)
        ffn_fc1 = add_matmul_rhs_constant(
            network, ffn_normed, dim, ffn_dim, weights[f"{prefix}.ffn.net.0.proj.weight"]
        )
        fc1_bias = weights.get(f"{prefix}.ffn.net.0.proj.bias")
        if fc1_bias is not None:
            ffn_fc1 = add_bias_sum(network, ffn_fc1, ffn_dim, fc1_bias)
        ffn_act = add_gelu_new(network, ffn_fc1)
        ffn_fc2 = add_matmul_rhs_constant(
            network, ffn_act, ffn_dim, dim, weights[f"{prefix}.ffn.net.2.weight"]
        )
        fc2_bias = weights.get(f"{prefix}.ffn.net.2.bias")
        if fc2_bias is not None:
            ffn_fc2 = add_bias_sum(network, ffn_fc2, dim, fc2_bias)
        gated_ff = network.add_elementwise(ffn_fc2, gate_ff, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated_ff.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)
    final_sst = weights["scale_shift_table"]
    final_sst_const = add_constant(network, (1, 2 * dim), final_sst.reshape(1, 2 * dim))
    time_embed_tiled = network.add_concatenation([time_embed_inp, time_embed_inp])
    time_embed_tiled.axis = 1
    final_modulation = network.add_elementwise(
        final_sst_const, time_embed_tiled.get_output(0), trt.ElementWiseOperation.SUM
    )
    final_shift = network.add_slice(
        final_modulation.get_output(0), start=(0, 0), shape=(1, dim), stride=(1, 1)
    )
    final_scale = network.add_slice(
        final_modulation.get_output(0), start=(0, dim), shape=(1, dim), stride=(1, 1)
    )
    hidden = add_adaptive_layernorm(
        network, hidden, final_scale.get_output(0), final_shift.get_output(0), dim, eps
    )
    proj_out_w = weights["proj_out.weight"]
    out_dim = proj_out_w.shape[1]
    output = add_matmul_rhs_constant(network, hidden, dim, out_dim, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = add_bias_sum(network, output, out_dim, proj_out_b)
    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)
    print(
        f"[dit-builder] Building TRT engine (dim={dim}, layers={num_layers}, patches={num_patches}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for DiT")
    return bytes(plan)

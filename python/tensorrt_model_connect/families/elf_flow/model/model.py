"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import sys
from typing import TYPE_CHECKING
from ..config import make_elf_rope_cache, resolve_elf_config
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


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


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
        ts_reshaped.get_output(0), freqs_const, trt.ElementWiseOperation.PROD
    )

    # cos and sin
    cos_part = network.add_unary(args.get_output(0), trt.UnaryOperation.COS)
    sin_part = network.add_unary(args.get_output(0), trt.UnaryOperation.SIN)

    # Concatenate [cos, sin] -> [1, freq_dim]
    embed = network.add_concatenation([cos_part.get_output(0), sin_part.get_output(0)])
    embed.axis = 1

    return embed.get_output(0)


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


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


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


# Builder


trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from .components.config import ModelConfig


def _storage_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _dense(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    y = add_matmul_rhs_constant(network, x, lhs_width, rhs_width, weight, dtype=dtype)
    if bias is not None:
        y = add_bias_sum(network, y, rhs_width, bias, dtype=dtype)
    return y


def _scalar_2d(network: trt.INetworkDefinition, scalar: trt.ITensor) -> trt.ITensor:
    reshaped = network.add_shuffle(scalar)
    reshaped.reshape_dims = (1, 1)
    return reshaped.get_output(0)


def _timestep_mlp(
    network: trt.INetworkDefinition,
    scalar: trt.ITensor,
    hidden_size: int,
    weights: "WeightDict",
    prefix: str,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    emb = add_timestep_embedding(
        network, scalar, hidden_size, freq_dim=256, max_period=10000.0, dtype=dtype
    )
    emb = _dense(
        network,
        emb,
        256,
        hidden_size,
        weights[f"{prefix}.mlp_0.w"],
        weights[f"{prefix}.mlp_0.b"],
        dtype=dtype,
    )
    emb = add_silu(network, emb)
    return _dense(
        network,
        emb,
        hidden_size,
        hidden_size,
        weights[f"{prefix}.mlp_2.w"],
        weights[f"{prefix}.mlp_2.b"],
        dtype=dtype,
    )


def _prefix_tokens(
    network: trt.INetworkDefinition,
    emb: trt.ITensor,
    token_weights: np.ndarray,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    _, n_tokens, hidden_size = token_weights.shape
    tokens = add_constant(
        network, (n_tokens, hidden_size), token_weights.reshape(n_tokens, hidden_size), dtype=dtype
    )
    return network.add_elementwise(tokens, emb, trt.ElementWiseOperation.SUM).get_output(0)


def _add_transformer_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: "WeightDict",
    layer_idx: int,
    cfg: dict,
    eps_tensor: trt.ITensor,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    *,
    dtype: np.dtype,
) -> trt.ITensor:
    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    total_seq = cfg["total_seq"]
    p = f"layer.{layer_idx}"

    normed = add_rms_norm(
        network, hidden, hidden_size, weights[f"{p}.norm1"], eps_tensor, dtype=dtype
    )

    qkv = _dense(
        network,
        normed,
        hidden_size,
        3 * hidden_size,
        weights[f"{p}.attn.qkv.w"],
        weights[f"{p}.attn.qkv.b"],
        dtype=dtype,
    )
    q = network.add_slice(qkv, (0, 0), (total_seq, hidden_size), (1, 1)).get_output(0)
    k = network.add_slice(qkv, (0, hidden_size), (total_seq, hidden_size), (1, 1)).get_output(0)
    v = network.add_slice(qkv, (0, 2 * hidden_size), (total_seq, hidden_size), (1, 1)).get_output(0)

    q = add_rms_norm_per_head(
        network,
        q,
        num_heads,
        head_dim,
        weights[f"{p}.attn.q_norm"],
        eps_tensor,
        dtype=dtype,
        sequence_length=total_seq,
    )
    k = add_rms_norm_per_head(
        network,
        k,
        num_heads,
        head_dim,
        weights[f"{p}.attn.k_norm"],
        eps_tensor,
        dtype=dtype,
        sequence_length=total_seq,
    )

    q = add_apply_rope_native_sequence(
        network,
        q,
        num_heads,
        head_dim,
        cos_cache,
        sin_cache,
        rotary_embedding_dim=head_dim,
        interleaved=True,
        sequence_length=total_seq,
    )
    k = add_apply_rope_native_sequence(
        network,
        k,
        num_heads,
        head_dim,
        cos_cache,
        sin_cache,
        rotary_embedding_dim=head_dim,
        interleaved=True,
        sequence_length=total_seq,
    )

    attn = add_attention_from_rows(
        network,
        q,
        k,
        v,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=total_seq,
        kv_seq=total_seq,
        causal=False,
    )
    attn = _dense(
        network,
        attn,
        hidden_size,
        hidden_size,
        weights[f"{p}.attn.proj.w"],
        weights[f"{p}.attn.proj.b"],
        dtype=dtype,
    )
    hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)

    normed = add_rms_norm(
        network, hidden, hidden_size, weights[f"{p}.norm2"], eps_tensor, dtype=dtype
    )
    w12 = weights[f"{p}.mlp.w12.w"]
    actual_ffn = int(w12.shape[1] // 2)
    fused = _dense(
        network,
        normed,
        hidden_size,
        2 * actual_ffn,
        weights[f"{p}.mlp.w12.w"],
        weights[f"{p}.mlp.w12.b"],
        dtype=dtype,
    )
    x1 = network.add_slice(fused, (0, 0), (total_seq, actual_ffn), (1, 1)).get_output(0)
    x2 = network.add_slice(fused, (0, actual_ffn), (total_seq, actual_ffn), (1, 1)).get_output(0)
    gate = add_silu(network, x1)
    gated = network.add_elementwise(gate, x2, trt.ElementWiseOperation.PROD).get_output(0)
    mlp_out = _dense(
        network,
        gated,
        actual_ffn,
        hidden_size,
        weights[f"{p}.mlp.w3.w"],
        weights[f"{p}.mlp.w3.b"],
        dtype=dtype,
    )
    return network.add_elementwise(hidden, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)


def build_elf_flow_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_seq_length: int | None = None,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build an ELF denoiser/decoder engine from GitHub ELF weights."""
    cfg = resolve_elf_config(config, max_seq_length)
    hidden_size = cfg["hidden_size"]
    text_dim = cfg["text_encoder_dim"]
    input_dim = cfg["input_dim"]
    max_length = cfg["max_length"]
    config.raw["_elf_engine_max_length"] = max_length
    dtype = _storage_dtype(precision)

    if cfg["vocab_size"] <= 0:
        raise ValueError("ELF config must set vocab_size for the decoder head")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    # ELF sampling accumulates small denoising differences across many steps.
    # Keep the fp32 build in full fp32 rather than TensorRT's default TF32 path
    # so replay parity against the GitHub JAX implementation stays tight.
    builder_config.clear_flag(trt.BuilderFlag.TF32)

    latent = network.add_input("latent", trt.float32, (max_length, input_dim))
    timestep = network.add_input("timestep", trt.float32, (1,))
    decoder_mode = network.add_input("decoder_mode", trt.float32, (1,))
    self_cond_cfg = None
    if cfg["num_self_cond_cfg_tokens"] > 0:
        self_cond_cfg = network.add_input("self_cond_cfg_scale", trt.float32, (1,))

    eps_tensor = add_constant(network, (1, 1), np.array([cfg["rms_norm_eps"]], dtype=np.float32))

    x = latent
    if input_dim == 2 * text_dim:
        x = _dense(
            network,
            x,
            input_dim,
            text_dim,
            weights["self_cond_proj.w"],
            weights["self_cond_proj.b"],
            dtype=dtype,
        )
    elif input_dim != text_dim:
        raise ValueError(
            f"ELF input_dim={input_dim} must equal text_encoder_dim={text_dim} or 2x that dimension"
        )

    x = _dense(
        network, x, text_dim, cfg["bottleneck_dim"], weights["text_proj.proj1.w"], None, dtype=dtype
    )
    x = _dense(
        network,
        x,
        cfg["bottleneck_dim"],
        hidden_size,
        weights["text_proj.proj2.w"],
        weights["text_proj.proj2.b"],
        dtype=dtype,
    )

    mode_tokens = 0
    if cfg["num_model_mode_tokens"] > 0:
        mode_tokens = cfg["num_model_mode_tokens"]
        mode = add_constant(
            network,
            (mode_tokens, hidden_size),
            weights["mode_tokens"].reshape(mode_tokens, hidden_size),
            dtype=dtype,
        )
        gated = network.add_elementwise(
            mode, _scalar_2d(network, decoder_mode), trt.ElementWiseOperation.PROD
        ).get_output(0)
        cat = network.add_concatenation([gated, x])
        cat.axis = 0
        x = cat.get_output(0)

    prefix_parts: list[trt.ITensor] = []
    time_emb = _timestep_mlp(network, timestep, hidden_size, weights, "t_embedder", dtype=dtype)
    prefix_parts.append(_prefix_tokens(network, time_emb, weights["t_emb_tokens"], dtype=dtype))
    if cfg["num_self_cond_cfg_tokens"] > 0:
        assert self_cond_cfg is not None
        sc_emb = _timestep_mlp(
            network, self_cond_cfg, hidden_size, weights, "self_cond_cfg_embedder", dtype=dtype
        )
        prefix_parts.append(
            _prefix_tokens(network, sc_emb, weights["self_cond_cfg_tokens"], dtype=dtype)
        )

    prefix_len = cfg["num_time_tokens"] + cfg["num_self_cond_cfg_tokens"]
    prefix_cat = network.add_concatenation(prefix_parts)
    prefix_cat.axis = 0
    cat = network.add_concatenation([prefix_cat.get_output(0), x])
    cat.axis = 0
    hidden = cat.get_output(0)

    total_seq = prefix_len + mode_tokens + max_length
    cfg = dict(cfg)
    cfg["total_seq"] = total_seq

    cos_np, sin_np = make_elf_rope_cache(
        max_length=max_length,
        head_dim=cfg["head_dim"],
        prefix_tokens=prefix_len + mode_tokens,
        theta=cfg["rope_theta"],
    )
    cos_cache = add_constant(network, cos_np.shape, cos_np, dtype=dtype)
    sin_cache = add_constant(network, sin_np.shape, sin_np, dtype=dtype)

    for layer_idx in range(cfg["depth"]):
        hidden = _add_transformer_block(
            network, hidden, weights, layer_idx, cfg, eps_tensor, cos_cache, sin_cache, dtype=dtype
        )

    body = network.add_slice(
        hidden, (prefix_len + mode_tokens, 0), (max_length, hidden_size), (1, 1)
    ).get_output(0)

    proj = _dense(
        network,
        body,
        hidden_size,
        text_dim,
        weights["decoder.proj.w"],
        weights["decoder.proj.b"],
        dtype=dtype,
    )
    proj = add_gelu_new(network, proj, dtype=dtype)
    logits = _dense(
        network,
        proj,
        text_dim,
        cfg["vocab_size"],
        weights["decoder.unembed.w"],
        weights["decoder.unembed.b"],
        dtype=dtype,
    )
    logits = network.add_elementwise(
        logits, _scalar_2d(network, decoder_mode), trt.ElementWiseOperation.PROD
    ).get_output(0)

    denoised = add_rms_norm(
        network, body, hidden_size, weights["final.norm"], eps_tensor, dtype=dtype
    )
    denoised = _dense(
        network,
        denoised,
        hidden_size,
        text_dim,
        weights["final.linear.w"],
        weights["final.linear.b"],
        dtype=dtype,
    )

    denoised = network.add_cast(denoised, trt.float32).get_output(0)
    denoised.name = "denoised"
    network.mark_output(denoised)
    logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "decoder_logits"
    network.mark_output(logits)

    print(
        "[elf-builder] Building ELF TensorRT engine "
        f"(variant={cfg['variant']}, hidden={hidden_size}, layers={cfg['depth']}, "
        f"seq={max_length}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for ELF")
    return bytes(plan)

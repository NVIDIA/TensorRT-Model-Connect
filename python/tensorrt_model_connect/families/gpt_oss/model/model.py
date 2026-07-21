# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS TensorRT graph operations."""

from __future__ import annotations
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
    truncate: bool = True,
    attention_factor: float | None = None,
) -> np.ndarray:
    """Build a YaRN RoPE table for TRT native IRotaryEmbeddingLayer.

    Matches transformers ``_compute_yarn_parameters``: ``truncate=False``
    keeps float correction-range bounds, and cos/sin values are scaled by
    ``attention_factor`` (defaulting to the paper's ``0.1 * ln(factor) + 1``
    mscale when unset).

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

    low = _yarn_correction_dim(
        beta_fast, head_dim, rope_theta, original_max_position_embeddings)
    high = _yarn_correction_dim(
        beta_slow, head_dim, rope_theta, original_max_position_embeddings)
    if truncate:
        low = np.floor(low)
        high = np.ceil(high)
    low = max(float(low), 0.0)
    high = min(float(high), float(head_dim - 1))
    if low == high:
        high += 0.001
    ramp = np.clip(
        (np.arange(half, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    inv_freq = freq_inter * ramp + freq_extra * (1 - ramp)

    if attention_factor is None:
        attention_factor = (
            0.1 * float(np.log(scaling_factor)) + 1.0
            if scaling_factor > 1.0 else 1.0)

    positions = np.arange(max_cache_length, dtype=np.float64)[:, None]
    angles = positions * inv_freq[None, :]
    table = (np.cos(angles) if cosine else np.sin(angles)) * attention_factor
    return table.astype(np.float32)


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
) -> np.ndarray:
    """Build GPT-OSS's half-dimension RoPE cache."""
    head_dim = validate_native_rope_dim(head_dim)
    half = head_dim // 2
    default = 1.0 if cosine else 0.0
    if max_cache_length <= 0 or rope_theta <= 0.0:
        return np.full((max(max_cache_length, 1), max(half, 1)), default, dtype=np.float32)
    inv_freq = rope_theta ** (
        -np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.arange(max_cache_length, dtype=np.float64)[:, None] * inv_freq
    return (np.cos(angles) if cosine else np.sin(angles)).astype(np.float32)


def resolve_rope_parameters(config) -> dict:
    """Return GPT-OSS RoPE metadata across Transformers config spellings."""
    raw = getattr(config, "raw", None) or {}
    for key in ("rope_parameters", "rope_scaling"):
        params = raw.get(key)
        if isinstance(params, dict):
            return params
    return {}


def make_rope_half_tables(
    config,
    attention_window: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build GPT-OSS cosine and sine caches, including HF-exact YaRN."""
    params = resolve_rope_parameters(config)
    if params.get("rope_type", params.get("type", "default")) == "yarn":
        attention_factor = params.get("attention_factor")
        yarn = dict(
            scaling_factor=float(params.get("factor", 1.0)),
            original_max_position_embeddings=int(
                params.get("original_max_position_embeddings", 4096)),
            beta_fast=float(params.get("beta_fast", 32.0)),
            beta_slow=float(params.get("beta_slow", 1.0)),
            truncate=bool(params.get("truncate", True)),
            attention_factor=(
                None if attention_factor is None else float(attention_factor)),
        )
        return (
            make_yarn_rope_table_half_dim(
                attention_window, head_dim, config.rope_theta, True, **yarn),
            make_yarn_rope_table_half_dim(
                attention_window, head_dim, config.rope_theta, False, **yarn),
        )
    return (
        make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True),
        make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False),
    )


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
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


def add_sliding_window_mask(
    network: trt.INetworkDefinition,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    attention_window: int,
    sliding_window: int,
) -> trt.ITensor:
    """Add GPT-OSS's local-attention window to the runtime causal mask."""
    iota = add_constant(
        network, (1, attention_window),
        np.arange(attention_window, dtype=np.float32).reshape(1, -1))
    pos_float = network.add_cast(position_id, trt.float32).get_output(0)
    pos_2d = network.add_shuffle(pos_float)
    pos_2d.reshape_dims = (1, 1)
    span = add_constant(
        network, (1, 1),
        np.array([[float(sliding_window - 1)]], dtype=np.float32))
    threshold = network.add_elementwise(
        pos_2d.get_output(0), span, trt.ElementWiseOperation.SUB).get_output(0)
    out_of_window = network.add_elementwise(
        iota, threshold, trt.ElementWiseOperation.LESS).get_output(0)
    penalty = network.add_select(
        out_of_window,
        add_constant(network, (1, 1), np.array([[-1e9]], dtype=np.float32)),
        add_constant(network, (1, 1), np.array([[0.0]], dtype=np.float32)),
    ).get_output(0)
    if penalty.dtype != attention_mask.dtype:
        penalty = network.add_cast(penalty, attention_mask.dtype).get_output(0)
    return network.add_elementwise(
        attention_mask, penalty, trt.ElementWiseOperation.SUM).get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_2d: trt.ITensor,
    sin_cache_2d: trt.ITensor,
    position_id: trt.ITensor,
    rotary_embedding_dim: int,
) -> trt.ITensor:
    """Apply non-interleaved RoPE to one GPT-OSS decoder token."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim

    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, 1)

    pos_2d = network.add_shuffle(position_id)
    pos_2d.reshape_dims = (1, 1)

    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_2d,
        sin_cache_2d,
        False,
        rotary_embedding_dim,
    )
    rope.set_input(3, pos_2d.get_output(0))

    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, 1)


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
) -> trt.ITensor:
    """Run GPT-OSS's masked, scaled native attention fallback."""
    output_dtype = q_4d.dtype
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
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


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
) -> trt.ITensor:
    """Apply GPT-OSS native attention to row-major GQA tensors."""
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    ctx_4d = add_attention_core(
        network, q_4d, k_4d, v_4d, mask=mask, scale=scale)
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
    )


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


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Apply GPT-OSS RMSNorm."""
    return add_rms_norm(
        network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)

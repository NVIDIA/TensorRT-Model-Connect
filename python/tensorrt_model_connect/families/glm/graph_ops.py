# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph operations required by GLM's native-KV decoder."""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
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
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    return network.add_constant(shape, weights).get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    output = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, output, lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_tensor = add_constant(
        network,
        bias_shape,
        np.asarray(bias).reshape(bias_shape),
        dtype=dtype,
    )
    bias_tensor = _cast_back_to_trt_dtype(network, bias_tensor, inp.dtype)
    output = network.add_elementwise(
        inp,
        bias_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, output, inp.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    output_dtype = inp.dtype
    if dtype != np.float32:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)

    squared = network.add_elementwise(
        inp,
        inp,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mean = network.add_reduce(
        squared,
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    denominator = network.add_elementwise(
        mean,
        eps_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    denominator = network.add_unary(
        denominator,
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    reciprocal = network.add_unary(
        denominator,
        trt.UnaryOperation.RECIP,
    ).get_output(0)
    normalized = network.add_elementwise(
        inp,
        reciprocal,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    gamma_tensor = add_constant(
        network,
        (1, hidden_size),
        gamma,
        dtype=np.float32,
    )
    result = network.add_elementwise(
        normalized,
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, result, output_dtype)


def validate_native_rope_dim(
    rotary_embedding_dim: int,
    *,
    field_name: str = "rotary_embedding_dim",
) -> int:
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
    """Return HF-exact inverse frequencies without an O(context) table."""

    rotary_dim = validate_native_rope_dim(int(head_dim * partial_rotary_factor))
    rope_theta = float(rope_theta)
    if not np.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError(
            f"TRT native RoPE requires rope_theta to be finite and positive; got {rope_theta}"
        )

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GLM native active-position RoPE requires PyTorch at build time "
            "to reproduce Hugging Face FP32 inverse frequencies exactly"
        ) from exc

    with torch.no_grad():
        dimensions = torch.arange(
            0,
            rotary_dim,
            2,
            dtype=torch.int64,
            device="cpu",
        ).to(dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / rotary_dim))
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
    """Build cos/sin rows only for positions active in this execution."""

    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    if inv_freq.ndim != 1 or inv_freq.size == 0:
        raise ValueError("active RoPE inverse frequencies must be a non-empty rank-1 array")

    positions = network.add_cast(position_id, trt.float32).get_output(0)
    position_column = network.add_shuffle(positions)
    position_column.reshape_dims = (-1, 1)
    frequency_row = add_constant(
        network,
        (1, int(inv_freq.size)),
        inv_freq.reshape(1, -1),
        dtype=np.float32,
    )
    angles = network.add_elementwise(
        position_column.get_output(0),
        frequency_row,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)

    cos_rows = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin_rows = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_cache = network.add_shuffle(cos_rows)
    cos_cache.reshape_dims = (1, -1, int(inv_freq.size))
    sin_cache = network.add_shuffle(sin_rows)
    sin_cache.reshape_dims = (1, -1, int(inv_freq.size))

    cos_output = cos_cache.get_output(0)
    sin_output = sin_cache.get_output(0)
    if cos_output.dtype != output_dtype:
        cos_output = network.add_cast(cos_output, output_dtype).get_output(0)
        sin_output = network.add_cast(sin_output, output_dtype).get_output(0)
    return cos_output, sin_output


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    num_heads: int,
    head_dim: int,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape runtime-dynamic [S,H*D] rows to [1,H,S,D]."""

    rows = network.add_shuffle(tensor)
    if tag:
        rows.name = tag + "_s_h_d"
    rows.reshape_dims = (-1, num_heads, head_dim)
    rows.second_transpose = trt.Permutation([1, 0, 2])

    output = network.add_shuffle(rows.get_output(0))
    if tag:
        output.name = tag + "_1_h_s_d"
    output.reshape_dims = (1, num_heads, -1, head_dim)
    return output.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    attention_size: int,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape runtime-dynamic [1,H,S,D] to [S,H*D]."""

    output = network.add_shuffle(tensor)
    if tag:
        output.name = tag + "_s_h_d"
    output.first_transpose = trt.Permutation([0, 2, 1, 3])
    output.reshape_dims = (-1, attention_size)
    return output.get_output(0)


def add_apply_active_rope(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    rotary_embedding_dim: int,
    *,
    interleaved: bool,
) -> trt.ITensor:
    """Apply native RoPE to already-selected runtime position rows."""

    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    inp_4d = reshape_rows_to_heads_4d(
        network,
        inp,
        num_heads,
        head_dim,
    )
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache,
        sin_cache,
        interleaved,
        rotary_embedding_dim,
    )
    return reshape_heads_4d_to_rows(
        network,
        rope.get_output(0),
        num_heads * head_dim,
    )


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
    scale: float,
    tag: str,
) -> dict[str, trt.ITensor]:
    """Update user-owned full-capacity KV storage and attend its valid prefix."""

    if not hasattr(network, "add_kv_cache_update") or not hasattr(network, "add_attention_v2"):
        raise RuntimeError(
            "GLM native KV cache requires TensorRT add_kv_cache_update and add_attention_v2 support"
        )

    k_update_4d = reshape_rows_to_heads_4d(
        network,
        k_update,
        num_kv_heads,
        head_dim,
        tag=tag + ".k_update",
    )
    v_update_4d = reshape_rows_to_heads_4d(
        network,
        v_update,
        num_kv_heads,
        head_dim,
        tag=tag + ".v_update",
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
        raise RuntimeError("TensorRT failed to create GLM KV-cache update layers")
    update_k.name = tag + ".cache_k_update"
    update_v.name = tag + ".cache_v_update"
    updated_k = update_k.get_output(0)
    updated_v = update_v.get_output(0)

    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        tag=tag + ".q",
    )
    if q_4d.dtype != trt.bfloat16:
        raise ValueError("GLM native KV attention requires BF16 queries")
    q_fp32 = network.add_cast(q_4d, trt.float32).get_output(0)
    scale_tensor = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=np.float32),
        dtype=np.float32,
    )
    q_scaled = network.add_elementwise(
        q_fp32,
        scale_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    q_scaled = network.add_cast(q_scaled, trt.bfloat16).get_output(0)

    attention = network.add_attention_v2(
        q_scaled,
        updated_k,
        updated_v,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.LOWER_RIGHT,
    )
    if attention is None:
        raise RuntimeError("TensorRT failed to create GLM native attention")
    attention.decomposable = False
    attention.key_value_lengths = key_value_lengths
    attention.name = tag

    context = reshape_heads_4d_to_rows(
        network,
        attention.get_output(0),
        num_heads * head_dim,
        tag=tag + ".ctx",
    )
    return {
        "context": context,
        "present_k": updated_k,
        "present_v": updated_v,
    }

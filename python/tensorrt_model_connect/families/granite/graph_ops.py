# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT operations used by Granite's native-KV decoder builder."""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


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
    rhs = _cast(network, rhs, lhs.dtype)
    matmul = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast(network, matmul.get_output(0), lhs.dtype)


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
        inp = _cast(network, inp, trt.float32)
        eps_tensor = _cast(network, eps_tensor, trt.float32)
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
    root = network.add_unary(
        denominator,
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    reciprocal = network.add_unary(
        root,
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
    return _cast(network, result, output_dtype)


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
    output_dtype = inp.dtype
    seq_dim = -1 if sequence_length is None else sequence_length
    reshape_in = network.add_shuffle(inp)
    reshape_in.reshape_dims = (seq_dim, num_heads, head_dim)
    reshaped = reshape_in.get_output(0)
    if dtype != np.float32:
        reshaped = _cast(network, reshaped, trt.float32)
        eps_tensor = _cast(network, eps_tensor, trt.float32)
    eps_3d = network.add_shuffle(eps_tensor)
    eps_3d.reshape_dims = (1, 1, 1)
    squared = network.add_elementwise(
        reshaped,
        reshaped,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mean = network.add_reduce(
        squared,
        trt.ReduceOperation.AVG,
        1 << 2,
        keep_dims=True,
    ).get_output(0)
    denominator = network.add_elementwise(
        mean,
        eps_3d.get_output(0),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    root = network.add_unary(
        denominator,
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    reciprocal = network.add_unary(
        root,
        trt.UnaryOperation.RECIP,
    ).get_output(0)
    normalized = network.add_elementwise(
        reshaped,
        reciprocal,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)

    gamma_array = np.asarray(gamma, dtype=np.float32)
    if gamma_array.size == head_dim:
        gamma_shape = (1, 1, head_dim)
    else:
        gamma_shape = (1, num_heads, head_dim)
    gamma_tensor = add_constant(
        network,
        gamma_shape,
        gamma_array.reshape(gamma_shape),
        dtype=np.float32,
    )
    result = network.add_elementwise(
        normalized,
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    result = _cast(network, result, output_dtype)
    reshape_out = network.add_shuffle(result)
    reshape_out.reshape_dims = (seq_dim, num_heads * head_dim)
    return reshape_out.get_output(0)


def validate_native_rope_dim(rotary_embedding_dim: int) -> int:
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2 != 0:
        raise ValueError(
            f"TensorRT native RoPE requires an even dimension >= 2, got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def make_native_active_rope_inv_freq(
    head_dim: int,
    rope_theta: float,
) -> np.ndarray:
    rotary_dim = validate_native_rope_dim(head_dim)
    rope_theta = float(rope_theta)
    if not np.isfinite(rope_theta) or rope_theta <= 0.0:
        raise ValueError("Granite rope_theta must be finite and positive")
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Granite native-KV build requires PyTorch to generate "
            "Hugging Face-exact RoPE frequencies"
        ) from exc

    with torch.no_grad():
        exponents = (
            torch.arange(
                0,
                rotary_dim,
                2,
                dtype=torch.int64,
                device="cpu",
            ).to(dtype=torch.float32)
            / rotary_dim
        )
        inv_freq = 1.0 / (rope_theta**exponents)
    return np.asarray(
        inv_freq.detach().cpu().contiguous().numpy(),
        dtype=np.float32,
    ).copy()


def add_active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    inv_freq: np.ndarray,
    output_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    inv_freq = np.asarray(inv_freq, dtype=np.float32)
    if inv_freq.ndim != 1 or inv_freq.size == 0:
        raise ValueError("active RoPE frequencies must be a non-empty vector")

    positions = network.add_cast(position_id, trt.float32).get_output(0)
    position_column = network.add_shuffle(positions)
    position_column.reshape_dims = (-1, 1)
    frequency_row = add_constant(
        network,
        (1, int(inv_freq.size)),
        inv_freq.reshape(1, -1),
    )
    angles = network.add_elementwise(
        position_column.get_output(0),
        frequency_row,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cosine_3d = network.add_shuffle(cosine)
    cosine_3d.reshape_dims = (1, -1, int(inv_freq.size))
    sine_3d = network.add_shuffle(sine)
    sine_3d.reshape_dims = (1, -1, int(inv_freq.size))
    return (
        _cast(network, cosine_3d.get_output(0), output_dtype),
        _cast(network, sine_3d.get_output(0), output_dtype),
    )


def _rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None,
    tag: str | None = None,
) -> trt.ITensor:
    seq_dim = -1 if sequence_length is None else sequence_length
    heads = network.add_shuffle(tensor)
    if tag:
        heads.name = tag + ".heads"
    heads.reshape_dims = (seq_dim, num_heads, head_dim)
    heads.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(heads.get_output(0))
    batched.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return batched.get_output(0)


def _heads_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    width: int,
    sequence_length: int | None,
    tag: str | None = None,
) -> trt.ITensor:
    seq_dim = -1 if sequence_length is None else sequence_length
    rows = network.add_shuffle(tensor)
    if tag:
        rows.name = tag + ".rows"
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (seq_dim, width)
    return rows.get_output(0)


def add_apply_rope_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    position_id: trt.ITensor | None,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = 1,
) -> trt.ITensor:
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    inp_4d = _rows_to_heads(
        network,
        inp,
        num_heads,
        head_dim,
        sequence_length,
    )
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache,
        sin_cache,
        interleaved,
        rotary_embedding_dim,
    )
    if position_id is not None:
        seq_dim = -1 if sequence_length is None else sequence_length
        positions = network.add_shuffle(position_id)
        positions.reshape_dims = (1, seq_dim)
        rope.set_input(3, positions.get_output(0))
    return _heads_to_rows(
        network,
        rope.get_output(0),
        num_heads * head_dim,
        sequence_length,
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
    q_seq: int | None,
    scale: float | None = None,
    tag: str | None = None,
) -> dict[str, trt.ITensor]:
    if not hasattr(network, "add_kv_cache_update") or not hasattr(network, "add_attention_v2"):
        raise RuntimeError(
            "Granite native KV requires TensorRT KV-cache update and attention-v2 APIs"
        )

    k_update_4d = _rows_to_heads(
        network,
        k_update,
        num_kv_heads,
        head_dim,
        q_seq,
        None if tag is None else tag + ".k",
    )
    v_update_4d = _rows_to_heads(
        network,
        v_update,
        num_kv_heads,
        head_dim,
        q_seq,
        None if tag is None else tag + ".v",
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
        raise RuntimeError("TensorRT failed to create Granite KV-cache update layers")
    if tag:
        update_k.name = tag + ".cache_k_update"
        update_v.name = tag + ".cache_v_update"
    updated_k = update_k.get_output(0)
    updated_v = update_v.get_output(0)

    q_4d = _rows_to_heads(
        network,
        q,
        num_heads,
        head_dim,
        q_seq,
        None if tag is None else tag + ".q",
    )
    if q_4d.dtype != trt.float16:
        raise ValueError("Granite native KV attention requires FP16 queries")
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim))
    q_fp32 = network.add_cast(q_4d, trt.float32).get_output(0)
    scale_tensor = add_constant(
        network,
        (1, 1, 1, 1),
        np.array([[[[scale]]]], dtype=np.float32),
    )
    q_scaled = network.add_elementwise(
        q_fp32,
        scale_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    q_scaled = network.add_cast(q_scaled, trt.float16).get_output(0)

    attention = network.add_attention_v2(
        q_scaled,
        updated_k,
        updated_v,
        trt.AttentionNormalizationOp.SOFTMAX,
        trt.CausalMaskKind.LOWER_RIGHT,
    )
    if attention is None:
        raise RuntimeError("TensorRT failed to create Granite native attention")
    attention.decomposable = False
    attention.key_value_lengths = key_value_lengths
    if tag:
        attention.name = tag

    return {
        "context": _heads_to_rows(
            network,
            attention.get_output(0),
            num_heads * head_dim,
            q_seq,
            None if tag is None else tag + ".context",
        ),
        "present_k": updated_k,
        "present_v": updated_v,
    }

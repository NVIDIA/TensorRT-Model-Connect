# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT model graph and utility implementation."""

from __future__ import annotations


import numpy as np
from tensorrt_model_connect import trt_compat
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from ..weights import WeightDict, _has_tensor, _load_tensor, _open_safetensors
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


def _add_matrix_multiply_with_fp32_accumulation(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
) -> trt.ITensor:
    """Request TensorRT's fused FP16 GEMM with FP32 accumulation."""
    output_dtype = lhs.dtype
    if lhs.dtype == trt.float16 and rhs.dtype == trt.float16:
        lhs = network.add_cast(lhs, trt.float32).get_output(0)
        rhs = network.add_cast(rhs, trt.float32).get_output(0)
    output = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, output, output_dtype)


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
    return _add_matrix_multiply_with_fp32_accumulation(network, lhs, rhs)


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


# Ltx Dit Builder


if TYPE_CHECKING:
    from collections.abc import Mapping

graph_ops: Any = None


def _ensure_trt() -> Any:
    return trt


def _ensure_graph_ops() -> Any:
    global graph_ops
    if graph_ops is None:
        from . import model as graph_ops_module

        graph_ops = graph_ops_module
    return graph_ops


def _target_np_dtype(precision: str) -> np.dtype:
    return np.float16 if precision == "fp16" else np.float32


def _trt_dtype(precision: str) -> trt.DataType:
    trt_module = _ensure_trt()
    return trt_module.float16 if precision == "fp16" else trt_module.float32


def _cast_back(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _transpose(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr.T, dtype=dtype)


def _array(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=dtype)


def load_ltx_dit_weights(
    model_dir: str | Path,
    *,
    num_layers: int = 28,
    precision: str = "fp16",
) -> WeightDict:
    """Load LTX denoiser weights from a diffusers transformer directory."""
    readers = _open_safetensors(Path(model_dir))
    dtype = _target_np_dtype(precision)
    weights = WeightDict()

    def t(name: str) -> np.ndarray:
        return _transpose(_load_tensor(readers, name), dtype)

    def f(name: str, *, norm: bool = False) -> np.ndarray:
        return _array(_load_tensor(readers, name), np.float32 if norm else dtype)

    def maybe(name: str, *, norm: bool = False) -> np.ndarray | None:
        if not _has_tensor(readers, name):
            return None
        return f(name, norm=norm)

    weights["scale_shift_table"] = f("scale_shift_table")
    weights["proj_in.weight"] = t("proj_in.weight")
    weights["proj_in.bias"] = f("proj_in.bias")

    for layer in ("linear_1", "linear_2"):
        p = f"time_embed.emb.timestep_embedder.{layer}"
        weights[f"{p}.weight"] = t(f"{p}.weight")
        weights[f"{p}.bias"] = f(f"{p}.bias")
    weights["time_embed.linear.weight"] = t("time_embed.linear.weight")
    weights["time_embed.linear.bias"] = f("time_embed.linear.bias")

    for layer in ("linear_1", "linear_2"):
        p = f"caption_projection.{layer}"
        weights[f"{p}.weight"] = t(f"{p}.weight")
        weights[f"{p}.bias"] = f(f"{p}.bias")

    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        weights[f"{p}.scale_shift_table"] = f(f"{p}.scale_shift_table")

        for attn in ("attn1", "attn2"):
            ap = f"{p}.{attn}"
            weights[f"{ap}.norm_q.weight"] = f(f"{ap}.norm_q.weight", norm=True)
            weights[f"{ap}.norm_k.weight"] = f(f"{ap}.norm_k.weight", norm=True)
            for proj in ("to_q", "to_k", "to_v"):
                weights[f"{ap}.{proj}.weight"] = t(f"{ap}.{proj}.weight")
                bias = maybe(f"{ap}.{proj}.bias")
                if bias is not None:
                    weights[f"{ap}.{proj}.bias"] = bias
            weights[f"{ap}.to_out.0.weight"] = t(f"{ap}.to_out.0.weight")
            bias = maybe(f"{ap}.to_out.0.bias")
            if bias is not None:
                weights[f"{ap}.to_out.0.bias"] = bias

        weights[f"{p}.ff.net.0.proj.weight"] = t(f"{p}.ff.net.0.proj.weight")
        weights[f"{p}.ff.net.0.proj.bias"] = f(f"{p}.ff.net.0.proj.bias")
        weights[f"{p}.ff.net.2.weight"] = t(f"{p}.ff.net.2.weight")
        weights[f"{p}.ff.net.2.bias"] = f(f"{p}.ff.net.2.bias")

    weights["proj_out.weight"] = t("proj_out.weight")
    weights["proj_out.bias"] = f("proj_out.bias")
    return weights


def build_ltx_dit_engine(
    weights: "Mapping[str, np.ndarray]",
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    text_seq_len: int = 128,
    text_dim: int = 4096,
    in_channels: int = 128,
    dim: int = 2048,
    num_heads: int = 32,
    num_layers: int = 28,
    frame_rate: int = 25,
    temporal_compression_ratio: int = 8,
    spatial_compression_ratio: int = 32,
    eps: float = 1e-6,
    precision: str = "fp16",
    verbose: bool = False,
) -> bytes:
    """Build the LTX transformer denoiser as a TensorRT plan."""
    if precision not in ("fp16", "fp32"):
        raise ValueError("LTX DiT raw builder currently supports fp16 or fp32")

    _ensure_trt()
    _ensure_graph_ops()
    seq_len = latent_frames * latent_height * latent_width
    head_dim = dim // num_heads
    trt_dtype = _trt_dtype(precision)
    np_dtype = _target_np_dtype(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    hidden_in = network.add_input("hidden_states", trt_dtype, (1, seq_len, in_channels))
    encoder_in = network.add_input("encoder_hidden_states", trt_dtype, (1, text_seq_len, text_dim))
    timestep_in = network.add_input("timestep", trt.float32, (1,))
    encoder_mask = network.add_input("encoder_attention_mask", trt.float32, (1, text_seq_len))

    block_eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    qk_eps_t = graph_ops.add_constant(network, (1, 1), np.array([1e-5], dtype=np.float32))

    hidden = _drop_batch(network, hidden_in, (seq_len, in_channels))
    encoder_hidden = _drop_batch(network, encoder_in, (text_seq_len, text_dim))

    hidden = _linear(network, hidden, in_channels, dim, weights, "proj_in", np_dtype)

    time_proj = graph_ops.add_timestep_embedding(network, timestep_in, dim=256, dtype=np.float32)
    embedded_timestep = _linear(
        network,
        time_proj,
        256,
        dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_1",
        np_dtype,
    )
    embedded_timestep = graph_ops.add_silu(network, embedded_timestep)
    embedded_timestep = _linear(
        network,
        embedded_timestep,
        dim,
        dim,
        weights,
        "time_embed.emb.timestep_embedder.linear_2",
        np_dtype,
    )
    temb = graph_ops.add_silu(network, embedded_timestep)
    temb = _linear(network, temb, dim, 6 * dim, weights, "time_embed.linear", np_dtype)

    context = _linear(
        network,
        encoder_hidden,
        text_dim,
        dim,
        weights,
        "caption_projection.linear_1",
        np_dtype,
    )
    context = graph_ops.add_gelu_new(network, context, dtype=np_dtype)
    context = _linear(
        network,
        context,
        dim,
        dim,
        weights,
        "caption_projection.linear_2",
        np_dtype,
    )

    rotary_cos, rotary_sin = make_ltx_rope_tables(
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        dim=dim,
        frame_rate=frame_rate,
        temporal_compression_ratio=temporal_compression_ratio,
        spatial_compression_ratio=spatial_compression_ratio,
    )
    rotary_cos_t = graph_ops.add_constant(network, (seq_len, dim), rotary_cos, dtype=np.float32)
    rotary_sin_t = graph_ops.add_constant(network, (seq_len, dim), rotary_sin, dtype=np.float32)
    rot_half = graph_ops.add_constant(
        network,
        (dim, dim),
        _make_ltx_rotate_half_matrix(dim, num_heads, interleaved=True),
        dtype=np.float32,
    )

    cross_mask = _make_cross_attention_mask(network, encoder_mask, text_seq_len=text_seq_len)

    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _ltx_block_modulation(
            network, temb, weights[f"{p}.scale_shift_table"], dim
        )

        norm_hidden = graph_ops.add_rms_norm(
            network,
            hidden,
            dim,
            np.ones(dim, dtype=np.float32),
            block_eps_t,
            dtype=np_dtype,
        )
        norm_hidden = _modulate(network, norm_hidden, scale_msa, shift_msa)

        attn_hidden = _ltx_attention(
            network,
            norm_hidden,
            None,
            None,
            weights,
            f"{p}.attn1",
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq_len=seq_len,
            kv_seq_len=seq_len,
            eps_t=qk_eps_t,
            dtype=np_dtype,
            rotary_cos=rotary_cos_t,
            rotary_sin=rotary_sin_t,
            rot_half=rot_half,
        )
        hidden = _residual_gated(network, hidden, attn_hidden, gate_msa)

        cross_hidden = _ltx_attention(
            network,
            hidden,
            context,
            cross_mask,
            weights,
            f"{p}.attn2",
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq_len=seq_len,
            kv_seq_len=text_seq_len,
            eps_t=qk_eps_t,
            dtype=np_dtype,
        )
        hidden = network.add_elementwise(
            hidden, cross_hidden, trt.ElementWiseOperation.SUM
        ).get_output(0)

        ff_norm = graph_ops.add_rms_norm(
            network,
            hidden,
            dim,
            np.ones(dim, dtype=np.float32),
            block_eps_t,
            dtype=np_dtype,
        )
        ff_norm = _modulate(network, ff_norm, scale_mlp, shift_mlp)
        ff_out = _ffn(network, ff_norm, weights, p, dim, np_dtype)
        hidden = _residual_gated(network, hidden, ff_out, gate_mlp)

    shift, scale = _final_modulation(network, embedded_timestep, weights["scale_shift_table"], dim)
    out = graph_ops.add_layer_norm(
        network,
        hidden,
        dim,
        np.ones(dim, dtype=np.float32),
        np.zeros(dim, dtype=np.float32),
        block_eps_t,
        dtype=np_dtype,
    )
    out = _modulate(network, out, scale, shift)
    out = _linear(network, out, dim, in_channels, weights, "proj_out", np_dtype)

    out_batched = network.add_shuffle(out)
    out_batched.reshape_dims = (1, seq_len, in_channels)
    out_fp32 = network.add_cast(out_batched.get_output(0), trt.float32).get_output(0)
    out_fp32.name = "sample"
    network.mark_output(out_fp32)

    print(
        "[ltx-dit] Building TRT engine "
        f"(precision={precision}, tokens={seq_len}, layers={num_layers}, "
        f"dim={dim}, heads={num_heads}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for LTX DiT")
    return bytes(plan)


def make_ltx_rope_tables(
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    dim: int,
    frame_rate: int,
    temporal_compression_ratio: int = 8,
    spatial_compression_ratio: int = 32,
    base_num_frames: int = 20,
    base_height: int = 2048,
    base_width: int = 2048,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute LTX's 3-axis RoPE tables for the fixed latent grid."""
    grid_f, grid_h, grid_w = np.meshgrid(
        np.arange(latent_frames, dtype=np.float32),
        np.arange(latent_height, dtype=np.float32),
        np.arange(latent_width, dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([grid_f, grid_h, grid_w], axis=-1).reshape(-1, 3)
    coords[:, 0] *= (temporal_compression_ratio / float(frame_rate)) / base_num_frames
    coords[:, 1] *= spatial_compression_ratio / base_height
    coords[:, 2] *= spatial_compression_ratio / base_width

    freq_count = dim // 6
    freqs = theta ** np.linspace(
        math.log(1.0, theta),
        math.log(theta, theta),
        freq_count,
        dtype=np.float32,
    )
    freqs = freqs * (math.pi / 2.0)
    # Diffusers flattens the RoPE features as frequency triplets:
    # [t_freq0, h_freq0, w_freq0, t_freq1, h_freq1, w_freq1, ...].
    # Keeping the [token, freq, axis] order here is required before the
    # final repeat-interleave over real/imaginary pairs.
    angles = freqs[None, :, None] * (coords[:, None, :] * 2.0 - 1.0)
    angles = angles.reshape(coords.shape[0], -1)
    cos = np.repeat(np.cos(angles), 2, axis=-1).astype(np.float32)
    sin = np.repeat(np.sin(angles), 2, axis=-1).astype(np.float32)
    pad = dim % 6
    if pad:
        cos = np.concatenate([np.ones((coords.shape[0], pad), dtype=np.float32), cos], axis=-1)
        sin = np.concatenate([np.zeros((coords.shape[0], pad), dtype=np.float32), sin], axis=-1)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _make_ltx_rotate_half_matrix(
    dim: int,
    num_heads: int,
    *,
    interleaved: bool,
) -> np.ndarray:
    """Return a row-vector matrix for RoPE rotate-half within each attention head."""
    if num_heads <= 0 or dim % num_heads != 0:
        raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

    head_dim = dim // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE rotate-half requires even head_dim, got {head_dim}")

    matrix = np.zeros((dim, dim), dtype=np.float32)
    for head in range(num_heads):
        start = head * head_dim
        if interleaved:
            for offset in range(0, head_dim, 2):
                a = start + offset
                b = a + 1
                matrix[b, a] = -1.0
                matrix[a, b] = 1.0
        else:
            half = head_dim // 2
            for offset in range(half):
                a = start + offset
                b = a + half
                matrix[b, a] = -1.0
                matrix[a, b] = 1.0
    return matrix


def _drop_batch(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    shape: tuple[int, ...],
) -> trt.ITensor:
    s = network.add_shuffle(tensor)
    s.reshape_dims = shape
    return s.get_output(0)


def _linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    in_dim: int,
    out_dim: int,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    dtype: np.dtype,
) -> trt.ITensor:
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_dim, out_dim, weights[f"{prefix}.weight"], dtype=dtype
    )
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias, dtype=dtype)
    return out


def _modulate(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    scale: trt.ITensor,
    shift: trt.ITensor,
) -> trt.ITensor:
    one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    one = _cast_back(network, one, x.dtype)
    scale = _cast_back(network, scale, x.dtype)
    shift = _cast_back(network, shift, x.dtype)
    scale_plus_one = network.add_elementwise(one, scale, trt.ElementWiseOperation.SUM).get_output(0)
    scaled = network.add_elementwise(x, scale_plus_one, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(scaled, shift, trt.ElementWiseOperation.SUM).get_output(0)


def _ltx_block_modulation(
    network: trt.INetworkDefinition,
    temb: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> list[trt.ITensor]:
    chunks: list[trt.ITensor] = []
    for i in range(6):
        t = network.add_slice(temb, (0, i * dim), (1, dim), (1, 1)).get_output(0)
        c = graph_ops.add_constant(network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype)
        c = _cast_back(network, c, t.dtype)
        chunks.append(network.add_elementwise(t, c, trt.ElementWiseOperation.SUM).get_output(0))
    return chunks


def _final_modulation(
    network: trt.INetworkDefinition,
    embedded_timestep: trt.ITensor,
    table: np.ndarray,
    dim: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    out = []
    for i in range(2):
        c = graph_ops.add_constant(network, (1, dim), table[i].reshape(1, dim), dtype=table.dtype)
        c = _cast_back(network, c, embedded_timestep.dtype)
        out.append(
            network.add_elementwise(embedded_timestep, c, trt.ElementWiseOperation.SUM).get_output(
                0
            )
        )
    return out[0], out[1]


def _make_cross_attention_mask(
    network: trt.INetworkDefinition,
    mask: trt.ITensor,
    *,
    text_seq_len: int,
) -> trt.ITensor:
    one = graph_ops.add_constant(
        network, (1, text_seq_len), np.ones((1, text_seq_len), dtype=np.float32)
    )
    inv = network.add_elementwise(one, mask, trt.ElementWiseOperation.SUB).get_output(0)
    neg = graph_ops.add_constant(network, (1, 1), np.array([-10000.0], dtype=np.float32))
    additive = network.add_elementwise(inv, neg, trt.ElementWiseOperation.PROD)
    mask4 = network.add_shuffle(additive.get_output(0))
    mask4.reshape_dims = (1, 1, 1, text_seq_len)
    return mask4.get_output(0)


def _to_attention_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    x3 = network.add_shuffle(x)
    x3.reshape_dims = (seq_len, num_heads, head_dim)
    x3.second_transpose = trt.Permutation([1, 0, 2])
    x4 = network.add_shuffle(x3.get_output(0))
    x4.reshape_dims = (1, num_heads, seq_len, head_dim)
    return x4.get_output(0)


def _from_attention_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    x3 = network.add_shuffle(x)
    x3.reshape_dims = (num_heads, seq_len, head_dim)
    flat = network.add_shuffle(x3.get_output(0))
    flat.first_transpose = trt.Permutation([1, 0, 2])
    flat.reshape_dims = (seq_len, num_heads * head_dim)
    return flat.get_output(0)


def _apply_ltx_rope(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    cos: trt.ITensor,
    sin: trt.ITensor,
    rot_half: trt.ITensor,
) -> trt.ITensor:
    out_dtype = x.dtype
    x_fp32 = x if x.dtype == trt.float32 else network.add_cast(x, trt.float32).get_output(0)
    rotated = network.add_matrix_multiply(
        x_fp32, trt.MatrixOperation.NONE, rot_half, trt.MatrixOperation.NONE
    ).get_output(0)
    x_cos = network.add_elementwise(x_fp32, cos, trt.ElementWiseOperation.PROD).get_output(0)
    rot_sin = network.add_elementwise(rotated, sin, trt.ElementWiseOperation.PROD).get_output(0)
    out = network.add_elementwise(x_cos, rot_sin, trt.ElementWiseOperation.SUM).get_output(0)
    return _cast_back(network, out, out_dtype)


def _ltx_attention(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    context: trt.ITensor | None,
    mask: trt.ITensor | None,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    q_seq_len: int,
    kv_seq_len: int,
    eps_t: trt.ITensor,
    dtype: np.dtype,
    rotary_cos: trt.ITensor | None = None,
    rotary_sin: trt.ITensor | None = None,
    rot_half: trt.ITensor | None = None,
) -> trt.ITensor:
    kv_source = hidden if context is None else context
    q = _linear(network, hidden, dim, dim, weights, f"{prefix}.to_q", dtype)
    k = _linear(network, kv_source, dim, dim, weights, f"{prefix}.to_k", dtype)
    v = _linear(network, kv_source, dim, dim, weights, f"{prefix}.to_v", dtype)

    q = graph_ops.add_rms_norm(
        network, q, dim, weights[f"{prefix}.norm_q.weight"], eps_t, dtype=dtype
    )
    k = graph_ops.add_rms_norm(
        network, k, dim, weights[f"{prefix}.norm_k.weight"], eps_t, dtype=dtype
    )

    if rotary_cos is not None and rotary_sin is not None and rot_half is not None:
        q = _apply_ltx_rope(network, q, rotary_cos, rotary_sin, rot_half)
        k = _apply_ltx_rope(network, k, rotary_cos, rotary_sin, rot_half)

    q4 = _to_attention_4d(network, q, seq_len=q_seq_len, num_heads=num_heads, head_dim=head_dim)
    k4 = _to_attention_4d(network, k, seq_len=kv_seq_len, num_heads=num_heads, head_dim=head_dim)
    v4 = _to_attention_4d(network, v, seq_len=kv_seq_len, num_heads=num_heads, head_dim=head_dim)
    if mask is not None:
        mask = _cast_back(network, mask, q4.dtype)
    ctx4 = graph_ops._add_attention_core(  # noqa: SLF001 - shared TRT primitive
        network, q4, k4, v4, causal=False, mask=mask
    )
    ctx = _from_attention_4d(
        network, ctx4, seq_len=q_seq_len, num_heads=num_heads, head_dim=head_dim
    )
    return _linear(network, ctx, dim, dim, weights, f"{prefix}.to_out.0", dtype)


def _residual_gated(
    network: trt.INetworkDefinition,
    residual: trt.ITensor,
    branch: trt.ITensor,
    gate: trt.ITensor,
) -> trt.ITensor:
    gate = _cast_back(network, gate, branch.dtype)
    gated = network.add_elementwise(branch, gate, trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        residual, gated.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)


def _ffn(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    weights: "Mapping[str, np.ndarray]",
    prefix: str,
    dim: int,
    dtype: np.dtype,
) -> trt.ITensor:
    fc1_w = weights[f"{prefix}.ff.net.0.proj.weight"]
    ffn_dim = fc1_w.shape[1]
    x = graph_ops.add_matmul_rhs_constant(network, hidden, dim, ffn_dim, fc1_w, dtype=dtype)
    x = graph_ops.add_bias_sum(
        network, x, ffn_dim, weights[f"{prefix}.ff.net.0.proj.bias"], dtype=dtype
    )
    x = graph_ops.add_gelu_new(network, x, dtype=dtype)
    x = graph_ops.add_matmul_rhs_constant(
        network, x, ffn_dim, dim, weights[f"{prefix}.ff.net.2.weight"], dtype=dtype
    )
    return graph_ops.add_bias_sum(network, x, dim, weights[f"{prefix}.ff.net.2.bias"], dtype=dtype)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched multi-token (prefill) layers for the Qwen3.8 hybrid decoder.

The decode engine handles one token per call. The prefill engine built from
these layers handles ``chunk_len`` tokens per call: every projection, norm and
MLP runs as one GEMM over ``[chunk_len, hidden]`` rows, so each weight matrix is
read once per chunk instead of once per token.

Only the two token-mixing parts need sequence-aware rewrites:

  Gated DeltaNet
    - conv1d: causal depthwise conv over ``[conv_state | new tokens]``.
    - delta rule: chunked (WY) form. Inside one block of ``C`` tokens the
      sequential recurrence
          S <- exp(g_t) * S;  S <- S + k_t (x) beta_t * (v_t - S^T k_t)
      is rewritten with batched matmuls over ``[heads, C, C]`` tiles plus one
      unit-lower-triangular inverse, computed as a product of ``(I + N^(2^i))``
      factors (``N`` is strictly lower triangular, hence nilpotent).
  Full attention
    - one ``[chunk_len, cache + chunk_len]`` masked attention instead of
      ``chunk_len`` single-row ones.

Tokens past ``valid_len`` (padding of a partial last chunk) are made inert:
``g = 0`` and ``beta = 0`` leave the recurrent state untouched, and the conv
state is gathered at ``valid_len`` instead of at the end of the chunk. Rows
past ``valid_len`` in the K/V outputs and hidden states are garbage the
runtime must ignore.

The math is in fp32, like the decode engine's recurrent step.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph_blocks
from . import graph_ops

# Added before exp() to the entries a decay matrix must not use, so that
# exp() yields exactly 0 instead of overflowing for future positions.
_MASKED_EXPONENT = -1.0e9


def _owned(quant_ctx, name: str) -> bool:
    return quant_ctx is not None and quant_ctx.profile.should_quantize(name)


def _f32(network, tensor: trt.ITensor) -> trt.ITensor:
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def _const32(network, values) -> trt.ITensor:
    arr = np.asarray(values, dtype=np.float32)
    return graph_ops.add_constant(network, arr.shape, arr, dtype=np.float32)


def _shuffle(network, tensor, *, reshape=None, first=None, second=None) -> trt.ITensor:
    layer = network.add_shuffle(tensor)
    if first is not None:
        layer.first_transpose = trt.Permutation(list(first))
    if reshape is not None:
        layer.reshape_dims = tuple(reshape)
    if second is not None:
        layer.second_transpose = trt.Permutation(list(second))
    return layer.get_output(0)


def _slice(network, tensor, start, shape) -> trt.ITensor:
    return network.add_slice(
        tensor, start=tuple(start), shape=tuple(shape), stride=(1,) * len(shape)
    ).get_output(0)


def _ew(network, a, b, op) -> trt.ITensor:
    return network.add_elementwise(a, b, op).get_output(0)


def _mm(network, a, b, *, ta=False, tb=False) -> trt.ITensor:
    return network.add_matrix_multiply(
        a,
        trt.MatrixOperation.TRANSPOSE if ta else trt.MatrixOperation.NONE,
        b,
        trt.MatrixOperation.TRANSPOSE if tb else trt.MatrixOperation.NONE,
    ).get_output(0)


def add_valid_row_mask(network, valid_len: trt.ITensor, chunk_len: int) -> trt.ITensor:
    """fp32 ``[chunk_len, 1]`` tensor: 1.0 for rows < valid_len, else 0.0.

    ``valid_len`` is an int32 ``[1]`` network input.
    """
    iota = graph_ops.add_constant(
        network, (chunk_len, 1), np.arange(chunk_len, dtype=np.int32).reshape(chunk_len, 1),
        dtype=np.int32)
    nv = _shuffle(network, valid_len, reshape=(1, 1))
    less = _ew(network, iota, nv, trt.ElementWiseOperation.LESS)
    return network.add_cast(less, trt.float32).get_output(0)


def add_conv_state_gather_index(network, valid_len: trt.ITensor, d_conv: int) -> trt.ITensor:
    """int32 ``[d_conv]`` column indices ``valid_len + [0, d_conv)``.

    Applied to ``[old conv state | chunk_len new columns]`` it selects the
    ``d_conv`` most recent inputs as of the last valid token.
    """
    ramp = graph_ops.add_constant(
        network, (d_conv,), np.arange(d_conv, dtype=np.int32), dtype=np.int32)
    return _ew(network, valid_len, ramp, trt.ElementWiseOperation.SUM)


def _cumsum_last(network, x: trt.ITensor, length: int) -> trt.ITensor:
    """Inclusive cumulative sum of an fp32 ``[H, length]`` tensor along axis 1."""
    axis = network.add_constant((), trt.Weights(np.array([1], dtype=np.int32))).get_output(0)
    layer = network.add_cumulative(x, axis, trt.CumulativeOperation.SUM, False, False)
    return layer.get_output(0)


_INVERSE_BLOCK = 16


def _add_unit_lower_inverse(network, lower: trt.ITensor, *, num_heads: int, size: int,
                            block: int = _INVERSE_BLOCK) -> trt.ITensor:
    """``(I + lower)^-1`` for a strictly lower triangular fp32 ``[H, size, size]``.

    The textbook doubling product ``(I+N)(I+N^2)(I+N^4)...`` over the whole
    matrix is unusable in fp32: its partial sums are truncated Neumann series
    whose entries grow combinatorially (hundreds to millions for the highly
    correlated keys of real checkpoints) and cancel only in exact arithmetic.
    Here doubling is applied only inside ``block x block`` diagonal blocks,
    where the growth is bounded by C(block-1, p), and the off-diagonal blocks
    come from block forward substitution, ``X[i, :i] = -D_i^-1 L[i, :i] X[:i, :i]``,
    whose intermediates stay the size of the true inverse.
    """
    H, C = num_heads, size
    b = min(block, C)
    nb = C // b
    eye = _const32(network, np.eye(C).reshape(1, C, C))
    in_block = np.kron(np.eye(nb), np.ones((b, b))).reshape(1, C, C)

    neg = _ew(network, _ew(network, lower, _const32(network, in_block),
                           trt.ElementWiseOperation.PROD),
              _const32(network, [[[-1.0]]]), trt.ElementWiseOperation.PROD)
    inv_diag = _ew(network, eye, neg, trt.ElementWiseOperation.SUM)
    power = neg
    for _ in range(b.bit_length() - 2):  # (I+N)(I+N^2)...(I+N^(b/2)); N^b = 0 per block
        power = _mm(network, power, power)
        inv_diag = _mm(network, inv_diag, _ew(network, eye, power, trt.ElementWiseOperation.SUM))
    if nb == 1:
        return inv_diag

    rows = [_slice(network, inv_diag, (0, 0, 0), (H, b, C))]
    for i in range(1, nb):
        diag_row = _slice(network, inv_diag, (0, i * b, 0), (H, b, C))  # zero left of block i
        d_inv = _slice(network, diag_row, (0, 0, i * b), (H, b, b))
        slab = _slice(network, lower, (0, i * b, 0), (H, b, i * b))
        if len(rows) == 1:
            done = rows[0]
        else:
            done = network.add_concatenation(rows)
            done.axis = 1
            done = done.get_output(0)
        x_done = _slice(network, done, (0, 0, 0), (H, i * b, i * b))
        left = _mm(network, d_inv, _mm(network, slab, x_done))
        left = _ew(network, left, _const32(network, [[[-1.0]]]), trt.ElementWiseOperation.PROD)
        parts = [left]
        parts.append(_slice(network, diag_row, (0, 0, i * b), (H, b, C - i * b)))
        row = network.add_concatenation(parts)
        row.axis = 2
        rows.append(row.get_output(0))
    full = network.add_concatenation(rows)
    full.axis = 1
    return full.get_output(0)


def add_delta_rule_chunk(
    network,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    g: trt.ITensor,
    beta: trt.ITensor,
    state: trt.ITensor,
    *,
    block_len: int,
    num_heads: int,
    head_dim: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Gated delta rule over one block of ``block_len`` tokens, all fp32.

    Args:
        q, k: ``[H, C, D]``. ``q`` is already scaled by 1/sqrt(D); both are
            L2-normalised and expanded to ``H`` value heads.
        v: ``[H, C, D]``.
        g: ``[H, C]`` per-token log decay (<= 0).
        beta: ``[H, C]`` write strength.
        state: ``[H, D, D]`` recurrent state, K on axis 1 and V on axis 2.

    Returns:
        (out ``[H, C, D]``, new_state ``[H, D, D]``) -- identical to running
        the token-by-token recurrence ``C`` times.
    """
    H, C = num_heads, block_len
    if C < 2 or C & (C - 1):
        raise ValueError(f"delta-rule block must be a power of two >= 2, got {C}")

    gc = _cumsum_last(network, g, C)  # [H, C] inclusive cumulative log decay
    gc_col = _shuffle(network, gc, reshape=(H, C, 1))
    gc_row = _shuffle(network, gc, reshape=(H, 1, C))
    diff = _ew(network, gc_col, gc_row, trt.ElementWiseOperation.SUB)  # [t, j] = gc_t - gc_j

    strict = np.where(np.tri(C, k=-1, dtype=bool), 0.0, _MASKED_EXPONENT).reshape(1, C, C)
    incl = np.where(np.tri(C, k=0, dtype=bool), 0.0, _MASKED_EXPONENT).reshape(1, C, C)
    decay_strict = network.add_unary(
        _ew(network, diff, _const32(network, strict), trt.ElementWiseOperation.SUM),
        trt.UnaryOperation.EXP).get_output(0)
    decay_incl = network.add_unary(
        _ew(network, diff, _const32(network, incl), trt.ElementWiseOperation.SUM),
        trt.UnaryOperation.EXP).get_output(0)

    beta_col = _shuffle(network, beta, reshape=(H, C, 1))
    k_beta = _ew(network, k, beta_col, trt.ElementWiseOperation.PROD)
    v_beta = _ew(network, v, beta_col, trt.ElementWiseOperation.PROD)
    exp_gc = network.add_unary(
        _shuffle(network, gc, reshape=(H, C, 1)), trt.UnaryOperation.EXP).get_output(0)
    k_beta_g = _ew(network, k_beta, exp_gc, trt.ElementWiseOperation.PROD)

    # (I + L)^-1 with L[t, j] = beta_t * exp(gc_t - gc_j) * (k_t . k_j), j < t.
    lower = _ew(network, _mm(network, k_beta, k, tb=True), decay_strict,
                trt.ElementWiseOperation.PROD)
    inv = _add_unit_lower_inverse(network, lower, num_heads=H, size=C)
    u = _mm(network, inv, v_beta)  # [H, C, D]
    w = _mm(network, inv, k_beta_g)  # [H, C, D]
    delta = _ew(network, u, _mm(network, w, state), trt.ElementWiseOperation.SUB)

    q_g = _ew(network, q, exp_gc, trt.ElementWiseOperation.PROD)
    attn = _ew(network, _mm(network, q, k, tb=True), decay_incl, trt.ElementWiseOperation.PROD)
    out = _ew(network, _mm(network, q_g, state), _mm(network, attn, delta),
              trt.ElementWiseOperation.SUM)

    gc_last = _slice(network, gc, (0, C - 1), (H, 1))  # [H, 1]
    k_decay = network.add_unary(
        _ew(network, gc_last, gc, trt.ElementWiseOperation.SUB),
        trt.UnaryOperation.EXP).get_output(0)
    k_decayed = _ew(network, k, _shuffle(network, k_decay, reshape=(H, C, 1)),
                    trt.ElementWiseOperation.PROD)
    state_decay = network.add_unary(
        _shuffle(network, gc_last, reshape=(H, 1, 1)), trt.UnaryOperation.EXP).get_output(0)
    new_state = _ew(
        network,
        _ew(network, state, state_decay, trt.ElementWiseOperation.PROD),
        _mm(network, k_decayed, delta, ta=True),
        trt.ElementWiseOperation.SUM)
    return out, new_state


def add_deltanet_layer_prefill(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    valid_rows: trt.ITensor,
    conv_gather_index: trt.ITensor,
    weights,
    prefix: str,
    chunk_len: int,
    block_len: int,
    hidden_size: int,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx=None,
) -> dict[str, trt.ITensor]:
    """One Gated DeltaNet layer over ``chunk_len`` tokens.

    ``hidden`` is ``[chunk_len, hidden_size]``. Returns the same
    ``{hidden, present_conv, present_ssm}`` as the single-token layer, with
    ``hidden`` being ``[chunk_len, hidden_size]``.
    """
    T = chunk_len
    qk_dim = num_kv_heads * head_dim
    matmul = graph_blocks.make_matmul_fn(network, dtype, quant_ctx)

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype)
    qkv = matmul(normed, hidden_size, conv_dim, weights.get(f"{prefix}.deltanet_in_proj_qkv"),
                 f"{prefix}.deltanet_in_proj_qkv")  # [T, conv_dim]
    z = matmul(normed, hidden_size, d_inner, weights.get(f"{prefix}.deltanet_z_proj"),
               f"{prefix}.deltanet_z_proj")  # [T, d_inner]
    a_raw = matmul(normed, hidden_size, num_heads, weights.get(f"{prefix}.deltanet_a_proj"),
                   f"{prefix}.deltanet_a_proj")  # [T, H]
    b_raw = matmul(normed, hidden_size, num_heads, weights.get(f"{prefix}.deltanet_b_proj"),
                   f"{prefix}.deltanet_b_proj")  # [T, H]

    # ===== causal depthwise conv1d over [old state | new tokens] =====
    qkv_cols = _shuffle(network, qkv, first=(1, 0))  # [conv_dim, T]
    ext = network.add_concatenation([conv_state_in, qkv_cols])
    ext.axis = 1
    ext_t = ext.get_output(0)  # [conv_dim, d_conv + T]
    # The state after the last valid token is the d_conv newest columns.
    present_conv = network.add_gather(ext_t, conv_gather_index, 1).get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype)
    if conv_w.dtype != ext_t.dtype:
        conv_w = network.add_cast(conv_w, ext_t.dtype).get_output(0)
    conv_acc = None
    for j in range(d_conv):
        window = _slice(network, ext_t, (0, 1 + j), (conv_dim, T))
        tap = _slice(network, conv_w, (0, j), (conv_dim, 1))
        term = _ew(network, window, tap, trt.ElementWiseOperation.PROD)
        conv_acc = term if conv_acc is None else _ew(
            network, conv_acc, term, trt.ElementWiseOperation.SUM)
    conv_bias = graph_ops.add_constant(
        network, (conv_dim, 1),
        np.asarray(weights[f"{prefix}.conv1d_bias"]).reshape(conv_dim, 1), dtype=dtype)
    if conv_bias.dtype != conv_acc.dtype:
        conv_bias = network.add_cast(conv_bias, conv_acc.dtype).get_output(0)
    conv_acc = _ew(network, conv_acc, conv_bias, trt.ElementWiseOperation.SUM)
    qkv_act = graph_ops.add_activation(network, conv_acc, "silu", dtype=dtype)  # [conv_dim, T]

    # ===== split Q, K, V and bring them to [heads, T, dim] =====
    def heads_first(row_start: int, rows: int, heads: int) -> trt.ITensor:
        part = _slice(network, qkv_act, (row_start, 0), (rows, T))
        return _shuffle(network, part, reshape=(heads, head_dim, T), second=(0, 2, 1))

    q_h = heads_first(0, qk_dim, num_kv_heads)  # [Hk, T, D]
    k_h = heads_first(qk_dim, qk_dim, num_kv_heads)
    v_h = heads_first(2 * qk_dim, d_inner, num_heads)  # [H, T, D]

    q_h = graph_ops.add_l2_norm(network, q_h, 2, eps=1e-6, dtype=dtype)
    k_h = graph_ops.add_l2_norm(network, k_h, 2, eps=1e-6, dtype=dtype)

    group = num_heads // num_kv_heads
    if group > 1:
        ones = _const32(network, np.ones((1, group, 1, 1)))

        def expand(x: trt.ITensor) -> trt.ITensor:
            x4 = _shuffle(network, _f32(network, x), reshape=(num_kv_heads, 1, T, head_dim))
            tiled = _ew(network, x4, ones, trt.ElementWiseOperation.PROD)
            return _shuffle(network, tiled, reshape=(num_heads, T, head_dim))

        q_h, k_h = expand(q_h), expand(k_h)
    else:
        q_h, k_h = _f32(network, q_h), _f32(network, k_h)
    v_h = _f32(network, v_h)
    q_h = _ew(network, q_h, _const32(network, [[[1.0 / np.sqrt(head_dim)]]]),
              trt.ElementWiseOperation.PROD)

    # ===== per-token decay and write strength, inert on padding rows =====
    a_biased = _ew(network, _f32(network, a_raw),
                   _const32(network, np.asarray(weights[f"{prefix}.dt_bias"]).reshape(1, -1)),
                   trt.ElementWiseOperation.SUM)
    # softplus(x) = max(x, 0) + log(1 + exp(-|x|)). The textbook log(1 + exp(x))
    # overflows to +inf for large x; an inf log-decay turns every cumulative
    # difference after it into NaN, which the one-token recurrence never sees.
    x_pos = _ew(network, a_biased, _const32(network, [[0.0]]), trt.ElementWiseOperation.MAX)
    x_neg = _ew(network, _ew(network, a_biased, _const32(network, [[-1.0]]),
                             trt.ElementWiseOperation.PROD),
                a_biased, trt.ElementWiseOperation.MAX)  # |x|, via max(-x, x)
    neg_abs = _ew(network, x_neg, _const32(network, [[-1.0]]), trt.ElementWiseOperation.PROD)
    softplus = _ew(
        network,
        x_pos,
        network.add_unary(
            _ew(network, network.add_unary(neg_abs, trt.UnaryOperation.EXP).get_output(0),
                _const32(network, [[1.0]]), trt.ElementWiseOperation.SUM),
            trt.UnaryOperation.LOG).get_output(0),
        trt.ElementWiseOperation.SUM)
    decay = _ew(network, softplus,
                _const32(network, np.asarray(weights[f"{prefix}.A"]).reshape(1, -1)),
                trt.ElementWiseOperation.PROD)  # [T, H]
    decay = _ew(network, decay, valid_rows, trt.ElementWiseOperation.PROD)
    g_h = _shuffle(network, decay, first=(1, 0))  # [H, T]

    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID).get_output(0)
    beta = _ew(network, _f32(network, beta), valid_rows, trt.ElementWiseOperation.PROD)
    beta_h = _shuffle(network, beta, first=(1, 0))  # [H, T]

    # ===== delta rule, block by block =====
    state = _f32(network, ssm_state_in)
    outs = []
    for c in range(T // block_len):
        s0 = c * block_len
        out_c, state = add_delta_rule_chunk(
            network,
            _slice(network, q_h, (0, s0, 0), (num_heads, block_len, head_dim)),
            _slice(network, k_h, (0, s0, 0), (num_heads, block_len, head_dim)),
            _slice(network, v_h, (0, s0, 0), (num_heads, block_len, head_dim)),
            _slice(network, g_h, (0, s0), (num_heads, block_len)),
            _slice(network, beta_h, (0, s0), (num_heads, block_len)),
            state, block_len=block_len, num_heads=num_heads, head_dim=head_dim)
        outs.append(out_c)
    present_ssm = state
    if len(outs) == 1:
        recurrent = outs[0]
    else:
        cat = network.add_concatenation(outs)
        cat.axis = 1
        recurrent = cat.get_output(0)  # [H, T, D]

    # ===== gated per-head RMSNorm: weight * norm(out) * silu(z) =====
    recurrent = _shuffle(network, recurrent, first=(1, 0, 2))  # [T, H, D]
    if recurrent.dtype != hidden.dtype:
        recurrent = network.add_cast(recurrent, hidden.dtype).get_output(0)
    norm_in = recurrent
    norm_out_dtype = norm_in.dtype
    if dtype != np.float32:
        norm_in = network.add_cast(norm_in, trt.float32).get_output(0)
    sq = _ew(network, norm_in, norm_in, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq, trt.ReduceOperation.AVG, 1 << 2, keep_dims=True).get_output(0)
    denom = network.add_unary(
        _ew(network, mean, _const32(network, [[[1e-6]]]), trt.ElementWiseOperation.SUM),
        trt.UnaryOperation.SQRT).get_output(0)
    normalized = _ew(network, norm_in, network.add_unary(
        denom, trt.UnaryOperation.RECIP).get_output(0), trt.ElementWiseOperation.PROD)
    flat = _shuffle(network, normalized, reshape=(T, d_inner))
    gamma = graph_ops.add_constant(
        network, (1, d_inner), weights[f"{prefix}.deltanet_norm"], dtype=np.float32)
    normed_out = _ew(network, flat, gamma, trt.ElementWiseOperation.PROD)
    if normed_out.dtype != norm_out_dtype:
        normed_out = network.add_cast(normed_out, norm_out_dtype).get_output(0)
    gated = _ew(network, normed_out, graph_ops.add_activation(network, z, "silu", dtype=dtype),
                trt.ElementWiseOperation.PROD)

    out = matmul(gated, d_inner, hidden_size, weights.get(f"{prefix}.deltanet_out_proj"),
                 f"{prefix}.deltanet_out_proj")
    hidden_after_attn = _ew(network, hidden, out, trt.ElementWiseOperation.SUM)

    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size, weights[f"{prefix}.post_attn_norm"],
        eps_tensor, dtype=dtype)
    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed, weights=weights, prefix=prefix, hidden_size=hidden_size,
        mlp_size=mlp_size, dtype=dtype, quant_ctx=quant_ctx)
    return {
        "hidden": _ew(network, hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def add_full_attention_layer_prefill(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_ids: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights,
    prefix: str,
    chunk_len: int,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx=None,
) -> dict[str, trt.ITensor]:
    """One full self-attention layer over ``chunk_len`` tokens.

    ``attention_mask`` is the additive ``[chunk_len, max_cache_length + chunk_len]``
    mask: the first ``max_cache_length`` columns select the valid cache prefix,
    the last ``chunk_len`` columns are causal over the new tokens.
    Returns ``{hidden, present_k, present_v}`` with ``present_*`` shaped
    ``[chunk_len, kv_attention_size]``.
    """
    T = chunk_len
    normed = graph_blocks.apply_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"), eps_tensor, "rmsnorm", dtype=dtype)

    matmul = graph_blocks.make_matmul_fn(network, dtype, quant_ctx)
    q = matmul(normed, hidden_size, attn_size, weights.get(f"{prefix}.w_q"), f"{prefix}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size, weights.get(f"{prefix}.w_k"),
               f"{prefix}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size, weights.get(f"{prefix}.w_v"),
               f"{prefix}.w_v")

    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype, sequence_length=T)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype,
            sequence_length=T)

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor, position_ids,
        rotary_embedding_dim, sequence_length=T)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor, position_ids,
        rotary_embedding_dim, sequence_length=T)

    all_k = network.add_concatenation([cache_k, k])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=T, kv_seq=max_cache_length + T, mask=mask_4d)

    gate_name = f"{prefix}.w_gate_attn"
    if weights.get(gate_name) is not None or _owned(quant_ctx, gate_name):
        gate = matmul(normed, hidden_size, attn_size, weights.get(gate_name), gate_name)
        context = _ew(network, context,
                      network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
                      trt.ElementWiseOperation.PROD)

    attn_out = matmul(context, attn_size, hidden_size, weights.get(f"{prefix}.w_o"),
                      f"{prefix}.w_o")
    hidden_after_attn = _ew(network, hidden, attn_out, trt.ElementWiseOperation.SUM)

    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size, weights[f"{prefix}.post_attn_norm"],
        eps_tensor, dtype=dtype)
    mlp_out = graph_blocks.add_swiglu_mlp(
        network, post_normed, weights=weights, prefix=prefix, hidden_size=hidden_size,
        mlp_size=mlp_size, dtype=dtype, quant_ctx=quant_ctx)
    return {
        "hidden": _ew(network, hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM),
        "present_k": k,
        "present_v": v,
    }

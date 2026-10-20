# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Static-shape Gated DeltaNet prefill chunk (T tokens, batch 1) as cuTile kernels.

Adapted from NVIDIA/TileGym (MIT): the recurrent gated delta rule kernel
(`tilegym/ops/cutile/recurrent_gated_delta_rule.py`) and the gate and gated-norm kernels
(`tilegym/transformers/qwen3_5/kernels`). See also gdn_tilegym.py for the single-token decode set.

A prefill call processes one chunk of T tokens. `valid_len` (int32 in device memory, 1 <= L <= T)
is the number of real tokens; the rest are padding (last chunk of a prompt). Padding tokens do not
touch the state: padding rows have g = 0 and beta = 0, so the recurrence leaves the state unchanged, the new conv state is
the last four real inputs, and output rows at or after `valid_len` are zero.

Stages, in launch order:
  1. conv_kernel       : y[T, CD] = silu(causal conv1d over [old state tail, x]) + bias
  2. conv_state_kernel : new conv state = last 4 real inputs
  3. gate_kernel       : g[T, H] = A * softplus(a + dt_bias), beta[T, H] = sigmoid(b); both 0 for padding rows
  4. intra_kernel      : per chunk of CS tokens: decay, triangular solve (WY form), corrected values
  5. inter_kernel      : state passing over the chunks and the chunk outputs (TF32 tensor cores)
  6. norm_kernel       : out[T, H, V] = gamma * rmsnorm(o) * silu(z), zero for padding rows
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


def _mm(a, b):
    """fp32 matmul on TF32 tensor cores (fp32 accumulate)."""
    a = ct.astype(a, ct.tfloat32)
    b = ct.astype(b, ct.tfloat32)
    return ct.astype(ct.matmul(a, b), ct.float32)


def _solve_unit_lower(A, CS):
    """(I - A)^-1 style solve used by the WY form: rows of strict-lower `A` folded in order.

    The serial order is kept on purpose: a Neumann/doubling product is unstable in fp32 on the
    nearly collinear keys of real models (see prefill_graph.py).
    """
    offs = ct.arange(CS, dtype=ct.int32)
    for i in range(1, CS):
        is_row = ct.expand_dims(offs == i, axis=1)
        row = ct.sum(ct.where(is_row, A, 0.0), axis=0)
        correction = ct.sum(ct.expand_dims(row, axis=1) * A, axis=0)
        A = A + ct.where(is_row, ct.expand_dims(correction, axis=0), 0.0)
    eye = ct.where(ct.expand_dims(offs, axis=1) == ct.expand_dims(offs, axis=0), 1.0, 0.0)
    return A + eye


@ct.kernel
def conv_kernel(
    x,  # (T, D)
    state_in,  # (D, 4)
    weight,  # (D, 4)
    bias,  # (D,)
    output,  # (T, D)
    BLOCK_D: ConstInt,
):
    """Grid: (cdiv(D, BLOCK_D), T). The input sequence is xs = [state[:, 1:4], x[0..T-1]]."""
    bid_d = ct.bid(0)
    t = ct.bid(1)
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = bid_d * BLOCK_D + offs

    acc = ct.astype(ct.gather(bias, (d_idx,), check_bounds=True), ct.float32)
    for k in range(4):
        j = t + k
        src_state = min(j + 1, 3)
        src_x = max(j - 3, 0)
        from_x = ct.astype(min(max(j - 2, 0), 1), ct.float32)
        sv = ct.astype(ct.gather(state_in, (d_idx, src_state), check_bounds=True), ct.float32)
        xv = ct.astype(ct.gather(x, (src_x, d_idx), check_bounds=True), ct.float32)
        wk = ct.astype(ct.gather(weight, (d_idx, k), check_bounds=True), ct.float32)
        acc = acc + wk * (from_x * xv + (1.0 - from_x) * sv)

    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(acc))))
    ct.scatter(output, (t, d_idx), ct.astype(acc * sig, output.dtype), check_bounds=True)


@ct.kernel
def conv_state_kernel(
    x,  # (T, D)
    state_in,  # (D, 4)
    valid_len,  # (1,) int32
    state_out,  # (D, 4)
    BLOCK_D: ConstInt,
):
    """Grid: (cdiv(D, BLOCK_D),). New state column i is xs[valid_len - 1 + i]."""
    bid_d = ct.bid(0)
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = bid_d * BLOCK_D + offs
    length = ct.gather(valid_len, (0,), check_bounds=False)

    for i in range(4):
        j = length - 1 + i
        src_state = min(max(j + 1, 0), 3)
        src_x = max(j - 3, 0)
        from_x = ct.astype(min(max(j - 2, 0), 1), ct.float32)
        sv = ct.astype(ct.gather(state_in, (d_idx, src_state), check_bounds=True), ct.float32)
        xv = ct.astype(ct.gather(x, (src_x, d_idx), check_bounds=True), ct.float32)
        val = from_x * xv + (1.0 - from_x) * sv
        ct.scatter(state_out, (d_idx, i), ct.astype(val, state_out.dtype), check_bounds=True)


@ct.kernel
def gate_kernel(
    b_in,  # (T, H)
    a_in,  # (T, H)
    A,  # (H,)  = -exp(A_log)
    dt_bias,  # (H,)
    valid_len,  # (1,) int32
    beta_out,  # (T, H)
    g_out,  # (T, H) fp32
    TILE_H: ConstInt,
):
    """Grid: (T,)."""
    t = ct.bid(0)
    offs = ct.arange(TILE_H, dtype=ct.int32)

    b = ct.astype(ct.gather(b_in, (t, offs), check_bounds=True), ct.float32)
    a = ct.astype(ct.gather(a_in, (t, offs), check_bounds=True), ct.float32)
    a_coef = ct.astype(ct.gather(A, (offs,), check_bounds=True), ct.float32)
    dt_b = ct.astype(ct.gather(dt_bias, (offs,), check_bounds=True), ct.float32)

    length = ct.gather(valid_len, (0,), check_bounds=False)
    keep = ct.astype(min(max(length - t, 0), 1), ct.float32)  # padding rows: g = 0, beta = 0
    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b)))) * keep
    g = a_coef * ct.log(ct.add(1.0, ct.exp(a + dt_b))) * keep

    ct.scatter(beta_out, (t, offs), ct.astype(beta, beta_out.dtype), check_bounds=True)
    ct.scatter(g_out, (t, offs), ct.astype(g, g_out.dtype), check_bounds=True)


@ct.kernel(occupancy=2)
def intra_kernel(
    Query,  # (1, T, H_K, K) strided view of y
    Key,  # (1, T, H_K, K)
    Value,  # (1, T, H, V)
    Gate,  # (1, T, H) fp32
    Beta,  # (1, T, H)
    Q_out,  # (H, NC, CS, K) fp32: normalised, scaled q
    K_out,  # (H, NC, CS, K) fp32: normalised k
    V_corr,  # (H, NC, CS, V) fp32
    K_cumdecay,  # (H, NC, CS, K) fp32
    G_cum,  # (H, NC, CS) fp32
    scale: ConstFloat,
    CS: ConstInt,
    BLOCK_K: ConstInt,
):
    """Grid: (H, NC). One chunk of CS tokens of one head: cumulative decay, the triangular solve
    of the WY form, and the corrected values (inter-chunk work is in inter_kernel)."""
    h = ct.bid(0)
    c = ct.bid(1)
    hk = h // (Value.shape[2] // Query.shape[2])
    zero = ct.PaddingMode.ZERO
    k_t = ct.astype(ct.load(Key, index=(0, c, hk, 0), shape=(1, CS, 1, BLOCK_K),
                            padding_mode=zero).reshape((CS, BLOCK_K)), ct.float32)
    k_t = k_t * ct.rsqrt(ct.sum(k_t * k_t, axis=1, keepdims=True) + 1e-6)
    q_t = ct.astype(ct.load(Query, index=(0, c, hk, 0), shape=(1, CS, 1, BLOCK_K),
                            padding_mode=zero).reshape((CS, BLOCK_K)), ct.float32)
    q_t = q_t * ct.rsqrt(ct.sum(q_t * q_t, axis=1, keepdims=True) + 1e-6) * scale
    beta = ct.astype(ct.load(Beta, index=(0, c, h), shape=(1, CS, 1),
                             padding_mode=zero).reshape((CS,)), ct.float32)
    g = ct.astype(ct.load(Gate, index=(0, c, h), shape=(1, CS, 1),
                          padding_mode=zero).reshape((CS,)), ct.float32)
    g_cum = ct.cumsum(g, axis=0)

    r = ct.arange(CS, dtype=ct.int32)
    rr = ct.expand_dims(r, axis=1)
    cc = ct.expand_dims(r, axis=0)
    decay = ct.where(rr >= cc, ct.exp(ct.expand_dims(g_cum, axis=1) - ct.expand_dims(g_cum, axis=0)), 0.0)
    kb = k_t * ct.expand_dims(beta, axis=1)
    attn = ct.where(rr > cc, -(_mm(kb, ct.transpose(k_t)) * decay), 0.0)
    attn = _solve_unit_lower(attn, CS)

    v_t = ct.astype(ct.load(Value, index=(0, c, h, 0), shape=(1, CS, 1, BLOCK_K),
                            padding_mode=zero).reshape((CS, BLOCK_K)), ct.float32)
    v_corr = _mm(attn, v_t * ct.expand_dims(beta, axis=1))
    k_cum = _mm(attn, kb * ct.expand_dims(ct.exp(g_cum), axis=1))

    ct.store(G_cum, index=(h, c, 0), tile=ct.reshape(g_cum, (1, 1, CS)))
    ct.store(V_corr, index=(h, c, 0, 0), tile=ct.reshape(v_corr, (1, 1, CS, BLOCK_K)))
    ct.store(K_cumdecay, index=(h, c, 0, 0), tile=ct.reshape(k_cum, (1, 1, CS, BLOCK_K)))
    ct.store(Q_out, index=(h, c, 0, 0), tile=ct.reshape(q_t, (1, 1, CS, BLOCK_K)))
    ct.store(K_out, index=(h, c, 0, 0), tile=ct.reshape(k_t, (1, 1, CS, BLOCK_K)))


@ct.kernel
def inter_kernel(
    Q_ch,  # (H, NC, CS, K) fp32
    K_ch,  # (H, NC, CS, K) fp32
    V_corr,  # (H, NC, CS, V) fp32
    K_cumdecay,  # (H, NC, CS, K) fp32
    G_cum,  # (H, NC, CS) fp32
    Output,  # (1, T, H, V) activation dtype
    StateIn,  # (1, H, K, V) fp32
    StateOut,  # (1, H, K, V) fp32
    CS: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
):
    """Grid: (H, cdiv(V, BLOCK_V)). Sequential over the chunks (state passing); the number of
    chunks is the runtime extent of axis 1 of Q_ch."""
    h = ct.bid(0)
    pv = ct.bid(1)
    zero = ct.PaddingMode.ZERO
    S = ct.astype(ct.load(StateIn, index=(0, h, 0, pv), shape=(1, 1, BLOCK_K, BLOCK_V),
                          padding_mode=zero).reshape((BLOCK_K, BLOCK_V)), ct.float32)
    r = ct.arange(CS, dtype=ct.int32)
    causal = ct.expand_dims(r, axis=1) >= ct.expand_dims(r, axis=0)
    for ci in range(Q_ch.shape[1]):
        k_cum = ct.load(K_cumdecay, index=(h, ci, 0, 0), shape=(1, 1, CS, BLOCK_K),
                        padding_mode=zero).reshape((CS, BLOCK_K))
        v_corr = ct.load(V_corr, index=(h, ci, 0, pv), shape=(1, 1, CS, BLOCK_V),
                         padding_mode=zero).reshape((CS, BLOCK_V))
        v_new = v_corr - _mm(k_cum, S)
        q = ct.load(Q_ch, index=(h, ci, 0, 0), shape=(1, 1, CS, BLOCK_K),
                    padding_mode=zero).reshape((CS, BLOCK_K))
        k = ct.load(K_ch, index=(h, ci, 0, 0), shape=(1, 1, CS, BLOCK_K),
                    padding_mode=zero).reshape((CS, BLOCK_K))
        gc = ct.load(G_cum, index=(h, ci, 0), shape=(1, 1, CS), padding_mode=zero).reshape((CS,))
        inter = _mm(q * ct.expand_dims(ct.exp(gc), axis=1), S)
        qk = _mm(q, ct.transpose(k))
        decay = ct.where(causal, ct.exp(ct.expand_dims(gc, axis=1) - ct.expand_dims(gc, axis=0)), 0.0)
        out = inter + _mm(ct.where(causal, qk * decay, 0.0), v_new)
        ct.store(Output, index=(0, ci, h, pv),
                 tile=ct.astype(ct.reshape(out, (1, CS, 1, BLOCK_V)), Output.dtype))
        g_last = ct.astype(ct.load(G_cum, index=(h, ci, CS - 1), shape=(), padding_mode=zero), ct.float32)
        k_w = k * ct.expand_dims(ct.exp(g_last - gc), axis=1)
        S = S * ct.exp(g_last) + _mm(ct.transpose(k_w), v_new)
    ct.store(StateOut, index=(0, h, 0, pv), tile=ct.reshape(S, (1, 1, BLOCK_K, BLOCK_V)))


@ct.kernel
def norm_kernel(
    hidden_states,  # (T*H, D)
    gate,  # (T*H, D)
    weight,  # (H, D) fp32
    valid_len,  # (1,) int32
    output,  # (T*H, D)
    eps: ConstFloat,
    D: ConstInt,
    TILE_D: ConstInt,
    H: ConstInt,
):
    """Grid: (T*H,). Row r is token r // H, head r % H."""
    bid = ct.bid(0)
    t = bid // H
    h = bid % H
    length = ct.gather(valid_len, (0,), check_bounds=False)
    keep = ct.astype(min(max(length - t, 0), 1), ct.float32)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    x = ct.astype(ct.gather(hidden_states, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    g = ct.astype(ct.gather(gate, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (h, offs), padding_value=0.0, check_bounds=True), ct.float32)

    variance = ct.sum(x * x) * ct.truediv(1.0, D)
    normed = x * ct.rsqrt(variance + eps)
    silu_g = g * ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    result = ct.astype(keep * (w * normed * silu_g), output.dtype)

    ct.scatter(output, (bid, offs), result, check_bounds=True)

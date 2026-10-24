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

import os

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# Operand type of the chunk matmuls. fp16 has the same 10-bit mantissa as TF32 and twice the tensor
# core rate, so the chunk kernels are 10-15% faster (fp32 accumulate either way).
_MM_DTYPE = ct.float16 if os.environ.get("TRTMC_GDN_PREFILL_MM", "fp16") == "fp16" else ct.tfloat32


def _mm(a, b):
    """fp32 matmul on tensor cores (fp32 accumulate)."""
    a = ct.astype(a, _MM_DTYPE)
    b = ct.astype(b, _MM_DTYPE)
    return ct.astype(ct.matmul(a, b), ct.float32)


_SOLVE_BLOCK = 8
# Doubling steps that merge the diagonal blocks: log2(chunk / block).
_SOLVE_MERGES = max(1, (int(os.environ.get("TRTMC_GDN_PREFILL_CHUNK", "32")) // _SOLVE_BLOCK).bit_length() - 1)


def _solve_unit_lower(A, CS):
    """T = (I - A)^-1 for strict-lower `A`, the triangular solve of the WY form.

    The diagonal blocks of _SOLVE_BLOCK rows are solved row by row (all blocks at once, one
    matmul per row index), then the blocks are merged with the nilpotent series of the
    off-diagonal part, (I + N)^-1 = (I - N)(I + N^2)(I + N^4)... With 8-row blocks the series
    stays accurate on the nearly collinear keys of real models, where a plain Neumann product
    over the whole chunk is not.
    """
    offs = ct.arange(CS, dtype=ct.int32)
    rows = ct.expand_dims(offs, axis=1)
    cols = ct.expand_dims(offs, axis=0)
    in_block = (rows // _SOLVE_BLOCK) == (cols // _SOLVE_BLOCK)
    diag = ct.where(in_block, A, 0.0)
    off_diag = ct.where(in_block, 0.0, A)
    for i in range(1, _SOLVE_BLOCK):
        is_row = ct.expand_dims((offs % _SOLVE_BLOCK) == i, axis=1)
        correction = _mm(ct.where(is_row, diag, 0.0), diag)
        diag = diag + ct.where(is_row, correction, 0.0)
    inv_diag = diag + ct.where(rows == cols, 1.0, 0.0)
    N = -_mm(inv_diag, off_diag)
    X = inv_diag - _mm(N, inv_diag)
    for _ in range(_SOLVE_MERGES - 1):
        N = _mm(N, N)
        X = X + _mm(N, X)
    return X


# Tile of the conv stage: one program walks CONV_ROWS tokens of CONV_BLOCK channels, so the
# per-program setup is shared and every load is a whole row.
CONV_BLOCK = 128
CONV_ROWS = 8


@ct.kernel
def conv_kernel(
    x,  # (T, D)
    state_in,  # (D, 4)
    weight,  # (D, 4)
    bias,  # (D,)
    output,  # (T, D)
    BLOCK_D: ConstInt,
    TT: ConstInt,
):
    """Grid: (cdiv(D, BLOCK_D), cdiv(T, TT)). One program walks TT consecutive tokens of BLOCK_D
    channels. The input sequence is xs = [state[:, 1:4], x[0..T-1]]; rows past T are not stored."""
    bd = ct.bid(0)
    bt = ct.bid(1)
    zero = ct.PaddingMode.ZERO
    d_idx = bd * BLOCK_D + ct.arange(BLOCK_D, dtype=ct.int32)
    bias_v = ct.astype(ct.gather(bias, (d_idx,), check_bounds=True), ct.float32)
    w0 = ct.astype(ct.gather(weight, (d_idx, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (d_idx, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (d_idx, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (d_idx, 3), check_bounds=True), ct.float32)
    s1 = ct.astype(ct.gather(state_in, (d_idx, 1), check_bounds=True), ct.float32)
    s2 = ct.astype(ct.gather(state_in, (d_idx, 2), check_bounds=True), ct.float32)
    s3 = ct.astype(ct.gather(state_in, (d_idx, 3), check_bounds=True), ct.float32)
    for i in range(TT):
        t = bt * TT + i
        acc = bias_v
        for k in range(4):
            j = t + k
            row = j - 3
            xr = ct.astype(ct.load(x, index=(max(row, 0), bd), shape=(1, BLOCK_D),
                                   padding_mode=zero).reshape((BLOCK_D,)), ct.float32)
            sv = ct.where(j == 0, s1, ct.where(j == 1, s2, s3))
            val = ct.where(row >= 0, xr, sv)
            wk = w0 if k == 0 else (w1 if k == 1 else (w2 if k == 2 else w3))
            acc = acc + wk * val
        sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(acc))))
        ct.store(output, index=(t, bd), tile=ct.reshape(ct.astype(acc * sig, output.dtype), (1, BLOCK_D)))


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


def _intra_chunk(Query, Key, Value, Gate, Beta, h, c, scale, CS, BLOCK_K):
    """Everything about chunk c of value head h that does not depend on the running state:
    returns (g_cum, v_corr, k_cumdecay, q * exp(g_cum), k * exp(g_last - g_cum), q k^T mask)."""
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
    exp_g = ct.expand_dims(ct.exp(g_cum), axis=1)
    k_cum = _mm(attn, kb * exp_g)
    g_last = ct.sum(ct.where(r == CS - 1, g_cum, 0.0), axis=0)
    q_g = q_t * exp_g
    k_w = k_t * ct.expand_dims(ct.exp(g_last - g_cum), axis=1)
    qk_mask = ct.where(rr >= cc, _mm(q_t, ct.transpose(k_t)) * decay, 0.0)
    return g_cum, v_corr, k_cum, q_g, k_w, qk_mask


@ct.kernel(occupancy=2)
def intra_kernel(
    Query,  # (1, T, H_K, K) strided view of y
    Key,  # (1, T, H_K, K)
    Value,  # (1, T, H, V)
    Gate,  # (1, T, H) fp32
    Beta,  # (1, T, H)
    KQ,  # (H, NC, 2 * CS, K) fp32: rows [0, CS) k_cumdecay, rows [CS, 2 * CS) q * exp(g_cum)
    K_w,  # (H, NC, CS, K) fp32: k * exp(g_last - g_cum)
    QK_mask,  # (H, NC, CS, CS) fp32: causal, decayed q k^T
    V_corr,  # (H, NC, CS, V) fp32
    G_cum,  # (H, NC, CS) fp32
    scale: ConstFloat,
    CS: ConstInt,
    BLOCK_K: ConstInt,
):
    """Grid: (H, NC). One chunk of CS tokens of one head: cumulative decay, the triangular solve
    of the WY form, the corrected values, and every product that does not depend on the state
    (inter_kernel only keeps the sequential state chain)."""
    h = ct.bid(0)
    c = ct.bid(1)
    g_cum, v_corr, k_cum, q_g, k_w, qk_mask = _intra_chunk(
        Query, Key, Value, Gate, Beta, h, c, scale, CS, BLOCK_K)
    ct.store(G_cum, index=(h, c, 0), tile=ct.reshape(g_cum, (1, 1, CS)))
    ct.store(V_corr, index=(h, c, 0, 0), tile=ct.reshape(v_corr, (1, 1, CS, BLOCK_K)))
    ct.store(KQ, index=(h, c, 0, 0), tile=ct.reshape(k_cum, (1, 1, CS, BLOCK_K)))
    ct.store(KQ, index=(h, c, 1, 0), tile=ct.reshape(q_g, (1, 1, CS, BLOCK_K)))
    ct.store(K_w, index=(h, c, 0, 0), tile=ct.reshape(k_w, (1, 1, CS, BLOCK_K)))
    ct.store(QK_mask, index=(h, c, 0, 0), tile=ct.reshape(qk_mask, (1, 1, CS, CS)))


@ct.kernel(occupancy=2)
def intra_cuda_kernel(
    Query,  # (1, T, H_K, K) strided view of y
    Key,  # (1, T, H_K, K)
    Value,  # (1, T, H, V)
    Gate,  # (1, T, H) fp32
    Beta,  # (1, T, H)
    KQ,  # (H, NC, 2 * CS, K) fp16: rows [0, CS) k_cumdecay, rows [CS, 2 * CS) q * exp(g_cum)
    K_wT,  # (H, NC, K, CS) fp16: (k * exp(g_last - g_cum))^T
    QK_mask,  # (H, NC, CS, CS) fp16: causal, decayed q k^T
    V_corr,  # (H, NC, CS, V) fp32
    G_cum,  # (H, NC, CS) fp32
    scale: ConstFloat,
    CS: ConstInt,
    BLOCK_K: ConstInt,
):
    """intra_kernel for the CUDA state-chain kernel (native_plugins/gdn_chain_inter.cu): the
    operands that only feed matmuls are stored as fp16 (the matmuls round them to fp16 anyway),
    K_w transposed."""
    h = ct.bid(0)
    c = ct.bid(1)
    g_cum, v_corr, k_cum, q_g, k_w, qk_mask = _intra_chunk(
        Query, Key, Value, Gate, Beta, h, c, scale, CS, BLOCK_K)
    ct.store(G_cum, index=(h, c, 0), tile=ct.reshape(g_cum, (1, 1, CS)))
    ct.store(V_corr, index=(h, c, 0, 0), tile=ct.reshape(v_corr, (1, 1, CS, BLOCK_K)))
    ct.store(KQ, index=(h, c, 0, 0), tile=ct.reshape(ct.astype(k_cum, KQ.dtype), (1, 1, CS, BLOCK_K)))
    ct.store(KQ, index=(h, c, 1, 0), tile=ct.reshape(ct.astype(q_g, KQ.dtype), (1, 1, CS, BLOCK_K)))
    ct.store(K_wT, index=(h, c, 0, 0),
             tile=ct.reshape(ct.astype(ct.transpose(k_w), K_wT.dtype), (1, 1, BLOCK_K, CS)))
    ct.store(QK_mask, index=(h, c, 0, 0), tile=ct.reshape(ct.astype(qk_mask, QK_mask.dtype), (1, 1, CS, CS)))


@ct.kernel
def inter_kernel(
    KQ,  # (H, NC, 2 * CS, K) fp32
    K_w,  # (H, NC, CS, K) fp32
    QK_mask,  # (H, NC, CS, CS) fp32
    V_corr,  # (H, NC, CS, V) fp32
    G_cum,  # (H, NC, CS) fp32
    Output,  # (1, T, H, V) activation dtype
    StateIn,  # (1, H, K, V) fp32
    StateOut,  # (1, H, K, V) fp32
    CS: ConstInt,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
):
    """Grid: (H, cdiv(V, BLOCK_V)). Sequential over the chunks (state passing); the number of
    chunks is the runtime extent of axis 1 of KQ. k_cumdecay @ S and (q * exp(g_cum)) @ S are one
    matmul on the stacked operand."""
    h = ct.bid(0)
    pv = ct.bid(1)
    zero = ct.PaddingMode.ZERO
    S = ct.astype(ct.load(StateIn, index=(0, h, 0, pv), shape=(1, 1, BLOCK_K, BLOCK_V),
                          padding_mode=zero).reshape((BLOCK_K, BLOCK_V)), ct.float32)
    for ci in range(KQ.shape[1]):
        kq = ct.load(KQ, index=(h, ci, 0, 0), shape=(1, 1, 2 * CS, BLOCK_K),
                     padding_mode=zero).reshape((2 * CS, BLOCK_K))
        v_corr = ct.load(V_corr, index=(h, ci, 0, pv), shape=(1, 1, CS, BLOCK_V),
                         padding_mode=zero).reshape((CS, BLOCK_V))
        qk_mask = ct.load(QK_mask, index=(h, ci, 0, 0), shape=(1, 1, CS, CS),
                          padding_mode=zero).reshape((CS, CS))
        k_w = ct.load(K_w, index=(h, ci, 0, 0), shape=(1, 1, CS, BLOCK_K),
                      padding_mode=zero).reshape((CS, BLOCK_K))
        from_state = _mm(kq, S)
        v_new = v_corr - ct.extract(from_state, (0, 0), shape=(CS, BLOCK_V))
        out = ct.extract(from_state, (1, 0), shape=(CS, BLOCK_V)) + _mm(qk_mask, v_new)
        ct.store(Output, index=(0, ci, h, pv),
                 tile=ct.astype(ct.reshape(out, (1, CS, 1, BLOCK_V)), Output.dtype))
        g_last = ct.astype(ct.load(G_cum, index=(h, ci, CS - 1), shape=(), padding_mode=zero), ct.float32)
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

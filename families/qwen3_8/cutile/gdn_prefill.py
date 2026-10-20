# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Static-shape Gated DeltaNet prefill chunk (T tokens, batch 1) as cuTile kernels.

Adapted from NVIDIA/TileGym (MIT): the recurrent gated delta rule kernel
(`tilegym/ops/cutile/recurrent_gated_delta_rule.py`) and the gate and gated-norm kernels
(`tilegym/transformers/qwen3_5/kernels`). See also gdn_tilegym.py for the single-token decode set.

A prefill call processes one chunk of T tokens. `valid_len` (int32 in device memory, 1 <= L <= T)
is the number of real tokens; the rest are padding (last chunk of a prompt). Padding tokens do not
touch the state: the recurrence runs only for the first `valid_len` tokens, the new conv state is
the last four real inputs, and output rows at or after `valid_len` are zero.

Stages, in launch order:
  1. conv_kernel       : y[T, CD] = silu(causal conv1d over [old state tail, x]) + bias
  2. conv_state_kernel : new conv state = last 4 real inputs
  3. gate_kernel       : g[T, H] = A * softplus(a + dt_bias), beta[T, H] = sigmoid(b)
  4. delta_kernel      : delta rule over the real tokens; q, k, v are strided views of y
  5. norm_kernel       : out[T, H, V] = gamma * rmsnorm(o) * silu(z), zero for padding rows
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


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

    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b))))
    g = a_coef * ct.log(ct.add(1.0, ct.exp(a + dt_b)))

    ct.scatter(beta_out, (t, offs), ct.astype(beta, beta_out.dtype), check_bounds=True)
    ct.scatter(g_out, (t, offs), ct.astype(g, g_out.dtype), check_bounds=True)


@ct.kernel
def delta_kernel(
    Query,  # (1, T, H_K, K)  strided view of y
    Key,  # (1, T, H_K, K)
    Value,  # (1, T, H, V)
    Gate,  # (1, T, H) fp32
    Beta,  # (1, T, H)
    Output,  # (1, T, H, V)
    StateIn,  # (1, H, K, V) fp32
    StateOut,  # (1, H, K, V) fp32
    valid_len,  # (1,) int32
    scale: ConstFloat,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
):
    """Grid: (H, cdiv(V, BLOCK_V), 1). Sequential over the first `valid_len` tokens."""
    idx_hv = ct.bid(0)
    idx_v = ct.bid(1)

    H = Query.shape[2]
    HV = Value.shape[2]
    idx_h = idx_hv // (HV // H)
    length = ct.gather(valid_len, (0,), check_bounds=False)

    State = ct.load(
        StateIn,
        index=(0, idx_hv, 0, idx_v),
        shape=(1, 1, BLOCK_K, BLOCK_V),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((BLOCK_K, BLOCK_V))
    State = ct.astype(State, ct.float32)

    for idx_t in range(length):
        QueryT = ct.load(
            Query, index=(0, idx_t, idx_h, 0), shape=(1, 1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_K,))
        QueryT = ct.astype(QueryT, ct.float32)
        KeyT = ct.load(
            Key, index=(0, idx_t, idx_h, 0), shape=(1, 1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_K,))
        KeyT = ct.astype(KeyT, ct.float32)

        QueryT = QueryT * ct.rsqrt(ct.sum(QueryT * QueryT, axis=0) + 1e-6)
        KeyT = KeyT * ct.rsqrt(ct.sum(KeyT * KeyT, axis=0) + 1e-6)
        QueryT = QueryT * scale

        ValueT = ct.load(
            Value, index=(0, idx_t, idx_hv, idx_v), shape=(1, 1, 1, BLOCK_V),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_V,))
        ValueT = ct.astype(ValueT, ct.float32)

        gate_t = ct.astype(ct.gather(Gate, (0, idx_t, idx_hv), check_bounds=False), ct.float32)
        beta_t = ct.astype(ct.gather(Beta, (0, idx_t, idx_hv), check_bounds=False), ct.float32)

        State = State * ct.exp(gate_t)
        KeyCol = ct.expand_dims(KeyT, axis=1)
        KvMemT = ct.sum(State * KeyCol, axis=0)
        DeltaT = (ValueT - KvMemT) * beta_t
        State = State + KeyCol * ct.expand_dims(DeltaT, axis=0)
        OutT = ct.sum(State * ct.expand_dims(QueryT, axis=1), axis=0)

        ct.store(
            Output,
            index=(0, idx_t, idx_hv, idx_v),
            tile=ct.astype(ct.reshape(OutT, (1, 1, 1, BLOCK_V)), Output.dtype),
        )

    ct.store(StateOut, index=(0, idx_hv, 0, idx_v),
             tile=ct.reshape(State, (1, 1, BLOCK_K, BLOCK_V)))


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

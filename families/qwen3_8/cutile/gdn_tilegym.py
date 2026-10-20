# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Single-token Gated DeltaNet decode as four cuTile kernels, adapted from NVIDIA/TileGym.

Sources (https://github.com/NVIDIA/TileGym, MIT):
  - tilegym/transformers/qwen3_5/kernels/causal_conv1d_update_silu.py
  - tilegym/transformers/qwen3_5/kernels/gdr_preprocess.py
  - tilegym/ops/cutile/recurrent_gated_delta_rule.py  (decode stream-V kernel)
  - tilegym/transformers/qwen3_5/kernels/rms_norm_gated.py

Changes for the TensorRT plugin (see native_plugins/gdn_tilegym_plugin.cpp):
  - Every non-array argument is a compile-time constant (the cubin is built for fixed shapes).
  - The conv kernel reads the old state and writes a new state buffer (TensorRT tensors are not
    updated in place) and adds the conv bias.
  - The gate kernel takes `A` (= -exp(A_log)) directly.
  - The norm kernel reads a per-element weight [H, D] instead of a shared [D] weight.
  - Batch is 1 and T is 1.

Stages, in launch order:
  1. conv_silu : y[CD] = silu(conv1d([state[:, 1:], x]) + bias), new conv state
  2. gate      : g[H] = A * softplus(a + dt_bias), beta[H] = sigmoid(b)
                 (ab_gate instead when the plugin also computes the a and b projections)
  3. gdn       : delta rule on S [H, K, V] with q, k, v taken from y; o[H, V]
  4. norm      : out[H, V] = gamma * rmsnorm(o) * silu(z)
"""

import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
ConstBool = ct.Constant[bool]


@ct.kernel
def conv_silu_kernel(
    x,  # (D,)         new qkv row
    state_in,  # (D, 4)
    weight,  # (D, 4)
    bias,  # (D,)
    output,  # (D,)         silu(conv)
    state_out,  # (D, 4)
    BLOCK_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = bid * BLOCK_D + offs

    s1 = ct.astype(ct.gather(state_in, (d_idx, 1), check_bounds=True), ct.float32)
    s2 = ct.astype(ct.gather(state_in, (d_idx, 2), check_bounds=True), ct.float32)
    s3 = ct.astype(ct.gather(state_in, (d_idx, 3), check_bounds=True), ct.float32)
    xv = ct.astype(ct.gather(x, (d_idx,), check_bounds=True), ct.float32)

    w0 = ct.astype(ct.gather(weight, (d_idx, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (d_idx, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (d_idx, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (d_idx, 3), check_bounds=True), ct.float32)
    bv = ct.astype(ct.gather(bias, (d_idx,), check_bounds=True), ct.float32)

    dot = s1 * w0 + s2 * w1 + s3 * w2 + xv * w3 + bv
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(dot))))
    result = dot * sig

    ct.scatter(output, (d_idx,), ct.astype(result, output.dtype), check_bounds=True)
    ct.scatter(state_out, (d_idx, 0), ct.astype(s1, state_out.dtype), check_bounds=True)
    ct.scatter(state_out, (d_idx, 1), ct.astype(s2, state_out.dtype), check_bounds=True)
    ct.scatter(state_out, (d_idx, 2), ct.astype(s3, state_out.dtype), check_bounds=True)
    ct.scatter(state_out, (d_idx, 3), ct.astype(xv, state_out.dtype), check_bounds=True)


@ct.kernel
def gate_kernel(
    b_in,  # (H,)
    a_in,  # (H,)
    A,  # (H,)  = -exp(A_log)
    dt_bias,  # (H,)
    beta_out,  # (H,)
    g_out,  # (H,) fp32
    TILE_H: ConstInt,
):
    offs = ct.arange(TILE_H, dtype=ct.int32)

    b = ct.astype(ct.gather(b_in, (offs,), check_bounds=True), ct.float32)
    a = ct.astype(ct.gather(a_in, (offs,), check_bounds=True), ct.float32)
    a_coef = ct.astype(ct.gather(A, (offs,), check_bounds=True), ct.float32)
    dt_b = ct.astype(ct.gather(dt_bias, (offs,), check_bounds=True), ct.float32)

    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b))))
    sp = ct.log(ct.add(1.0, ct.exp(a + dt_b)))
    g = a_coef * sp

    ct.scatter(beta_out, (offs,), ct.astype(beta, beta_out.dtype), check_bounds=True)
    ct.scatter(g_out, (offs,), ct.astype(g, g_out.dtype), check_bounds=True)


@ct.kernel
def ab_gate_kernel(
    x,  # (K,)  normalized hidden row, the input of the a and b projections
    w,  # (2H, K) fp32, rows 0..H-1 = a projection, rows H..2H-1 = b projection
    A,  # (H,)  = -exp(A_log)
    dt_bias,  # (H,)
    beta_out,  # (H,)
    g_out,  # (H,) fp32
    H: ConstInt,
    K: ConstInt,
    TILE_K: ConstInt,
):
    """Grid: (H,). The a and b projections of one head as dot products, then the gate.

    a and b are rounded to the activation dtype, as the output of a GEMM would be.
    """
    h = ct.bid(0)
    offs = ct.arange(TILE_K, dtype=ct.int32)
    acc_a = ct.zeros((TILE_K,), ct.float32)
    acc_b = ct.zeros((TILE_K,), ct.float32)
    for i in range(ct.cdiv(K, TILE_K)):
        idx = i * TILE_K + offs
        xv = ct.astype(ct.gather(x, (idx,), padding_value=0.0, check_bounds=True), ct.float32)
        wa = ct.gather(w, (h, idx), padding_value=0.0, check_bounds=True)
        wb = ct.gather(w, (h + H, idx), padding_value=0.0, check_bounds=True)
        acc_a = acc_a + xv * wa
        acc_b = acc_b + xv * wb
    a = ct.astype(ct.astype(ct.sum(acc_a), beta_out.dtype), ct.float32)
    b = ct.astype(ct.astype(ct.sum(acc_b), beta_out.dtype), ct.float32)

    a_coef = ct.astype(ct.gather(A, (h,), check_bounds=False), ct.float32)
    dt_b = ct.astype(ct.gather(dt_bias, (h,), check_bounds=False), ct.float32)
    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b))))
    g = a_coef * ct.log(ct.add(1.0, ct.exp(a + dt_b)))

    ct.scatter(beta_out, (h,), ct.astype(beta, beta_out.dtype), check_bounds=False)
    ct.scatter(g_out, (h,), ct.astype(g, g_out.dtype), check_bounds=False)


@ct.kernel
def gdn_kernel(
    Query,  # (1, 1, H_K, K)
    Key,  # (1, 1, H_K, K)
    Value,  # (1, 1, H, V)
    Gate,  # (1, 1, H) fp32
    Beta,  # (1, 1, H)
    Output,  # (1, 1, H, V)
    StateIn,  # (1, H, K, V) fp32
    StateOut,  # (1, H, K, V) fp32
    scale: ConstFloat,
    BLOCK_K: ConstInt,
    BLOCK_V: ConstInt,
    STREAM_V_TILE: ConstInt,
):
    """Grid: (H, ceil(V / BLOCK_V), 1)."""
    idx_hv = ct.bid(0)
    idx_v_block = ct.bid(1)

    H = Query.shape[2]
    HV = Value.shape[2]
    V = Value.shape[3]
    idx_h = idx_hv // (HV // H)

    remaining_v = V - idx_v_block * BLOCK_V
    valid_v = min(BLOCK_V, remaining_v)
    num_stream_v_tiles = ct.cdiv(valid_v, STREAM_V_TILE)

    QueryT = ct.load(
        Query, index=(0, 0, idx_h, 0), shape=(1, 1, 1, BLOCK_K), padding_mode=ct.PaddingMode.ZERO
    ).reshape((BLOCK_K,))
    KeyT = ct.load(
        Key, index=(0, 0, idx_h, 0), shape=(1, 1, 1, BLOCK_K), padding_mode=ct.PaddingMode.ZERO
    ).reshape((BLOCK_K,))
    QueryT = ct.astype(QueryT, ct.float32)
    KeyT = ct.astype(KeyT, ct.float32)

    QueryT = QueryT * ct.rsqrt(ct.sum(QueryT * QueryT, axis=0) + 1e-6)
    KeyT = KeyT * ct.rsqrt(ct.sum(KeyT * KeyT, axis=0) + 1e-6)
    QueryT = QueryT * scale

    gate_t = ct.astype(ct.gather(Gate, (0, 0, idx_hv), check_bounds=False), ct.float32)
    beta_t = ct.astype(ct.gather(Beta, (0, 0, idx_hv), check_bounds=False), ct.float32)
    gamma_t = ct.exp(gate_t)

    KeyCol = ct.expand_dims(KeyT, axis=1)
    QueryCol = ct.expand_dims(QueryT, axis=1)
    key_query_dot = ct.sum(KeyT * QueryT, axis=0)

    for idx_stream_v in range(num_stream_v_tiles):
        idx_sv = idx_v_block * (BLOCK_V // STREAM_V_TILE) + idx_stream_v
        ValueT = ct.load(
            Value,
            index=(0, 0, idx_hv, idx_sv),
            shape=(1, 1, 1, STREAM_V_TILE),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((STREAM_V_TILE,))
        ValueT = ct.astype(ValueT, ct.float32)

        State = ct.load(
            StateIn,
            index=(0, idx_hv, 0, idx_sv),
            shape=(1, 1, BLOCK_K, STREAM_V_TILE),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_K, STREAM_V_TILE))
        State = ct.astype(State, ct.float32)

        State = State * gamma_t
        KvMemT = ct.sum(State * KeyCol, axis=0)
        OutBaseT = ct.sum(State * QueryCol, axis=0)
        DeltaT = (ValueT - KvMemT) * beta_t
        OutT = OutBaseT + DeltaT * key_query_dot
        State = State + KeyCol * ct.expand_dims(DeltaT, axis=0)

        ct.store(
            Output,
            index=(0, 0, idx_hv, idx_sv),
            tile=ct.astype(ct.reshape(OutT, (1, 1, 1, STREAM_V_TILE)), Output.dtype),
        )
        ct.store(
            StateOut,
            index=(0, idx_hv, 0, idx_sv),
            tile=ct.reshape(State, (1, 1, BLOCK_K, STREAM_V_TILE)),
        )


@ct.kernel
def norm_gated_kernel(
    hidden_states,  # (N, D)
    gate,  # (N, D)
    weight,  # (N, D) fp32
    output,  # (N, D)
    eps: ConstFloat,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    h = ct.astype(ct.gather(hidden_states, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    g = ct.astype(ct.gather(gate, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)

    variance = ct.sum(h * h) * ct.truediv(1.0, D)
    normed = h * ct.rsqrt(variance + eps)
    sig_g = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    silu_g = g * sig_g
    result = ct.astype(w * normed * silu_g, output.dtype)

    ct.scatter(output, (bid, offs), result, check_bounds=True)

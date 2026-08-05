# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Nemotron-H hybrid builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_blocks, graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import (
    _add_selected_latent_experts,
    _add_stable_softplus,
    _mark_debug_output,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _slice_middle_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=1)[rank])


def _precision_dtypes(precision: str) -> tuple[np.dtype, trt.DataType]:
    """Return constant-storage and runtime dtypes for a TP build."""
    if precision == "fp16":
        return np.float16, trt.float16
    if precision == "bf16":
        return np.float16, trt.bfloat16
    if precision == "fp32":
        return np.float32, trt.float32
    raise ValueError(
        f"Unsupported Nemotron-H precision {precision!r}; "
        "expected fp32, fp16, or bf16"
    )


def _add_typed_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    storage_dtype: np.dtype,
    runtime_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant and cast its runtime tensor to the requested dtype.

    BF16 weights use NumPy FP16 storage because NumPy has no portable BF16
    dtype. The explicit TensorRT cast prevents BF16 builds from accidentally
    executing those constants as FP16.
    """
    const = graph_ops.add_constant(network, shape, values, dtype=storage_dtype)
    if const.dtype != runtime_dtype:
        const = network.add_cast(const, runtime_dtype).get_output(0)
    return const


def _add_constant_like(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    like: trt.ITensor,
    *,
    storage_dtype: np.dtype,
) -> trt.ITensor:
    return _add_typed_constant(
        network,
        shape,
        values,
        storage_dtype=storage_dtype,
        runtime_dtype=like.dtype,
    )


def _kv_rank_layout(
    config: "ModelConfig", parallel: "ParallelConfig"
) -> tuple[int, int]:
    """Return the first global KV head and local KV-head count for a rank."""
    num_kv_heads = int(config.num_key_value_heads)
    tp_size = int(parallel.tp_size)
    if num_kv_heads <= 0:
        raise ValueError("Nemotron-H requires num_key_value_heads > 0")

    if num_kv_heads % tp_size == 0:
        local_kv_heads = num_kv_heads // tp_size
        return int(parallel.rank) * local_kv_heads, local_kv_heads

    if tp_size % num_kv_heads == 0:
        ranks_per_kv_head = tp_size // num_kv_heads
        kv_head = int(parallel.rank) // ranks_per_kv_head
        return kv_head, 1

    raise ValueError(
        "Nemotron-H tensor parallel requires num_key_value_heads and tp_size "
        "to divide evenly for aligned GQA sharding "
        f"({num_kv_heads} vs {tp_size})"
    )


def _take_last_dim_segments(arr: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[..., start:end] for start, end in segments], axis=-1)
    )


def _take_first_dim_segments(arr: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[start:end, ...] for start, end in segments], axis=0)
    )


def _mamba2_rank_dims(weights: "WeightDict", parallel: "ParallelConfig") -> dict[str, int]:
    rank = parallel.rank
    tp = parallel.tp_size
    d_inner = int(weights["_d_inner"])
    d_state = int(weights["_d_state"])
    mamba_heads = int(weights["_mamba_num_heads"])
    head_dim = int(weights["_mamba_head_dim"])
    n_groups = int(weights["_n_groups"])
    groups_state = n_groups * d_state
    local_heads = mamba_heads // tp
    local_groups = n_groups // tp
    local_d_inner = local_heads * head_dim
    local_groups_state = local_groups * d_state
    local_conv_dim = local_d_inner + 2 * local_groups_state
    return {
        "rank": rank,
        "tp": tp,
        "d_inner": d_inner,
        "d_state": d_state,
        "mamba_heads": mamba_heads,
        "head_dim": head_dim,
        "n_groups": n_groups,
        "groups_state": groups_state,
        "local_heads": local_heads,
        "local_groups": local_groups,
        "local_d_inner": local_d_inner,
        "local_groups_state": local_groups_state,
        "local_conv_dim": local_conv_dim,
        "inner_start": rank * local_d_inner,
        "group_state_start": rank * local_groups_state,
        "head_start": rank * local_heads,
    }


def _slice_mamba_in_proj(weight: np.ndarray, dims: dict[str, int]) -> np.ndarray:
    d_inner = dims["d_inner"]
    groups_state = dims["groups_state"]
    conv_dim = int(d_inner + 2 * groups_state)
    inner_start = dims["inner_start"]
    local_d_inner = dims["local_d_inner"]
    group_state_start = dims["group_state_start"]
    local_groups_state = dims["local_groups_state"]
    head_start = dims["head_start"]
    local_heads = dims["local_heads"]
    segments = [
        (inner_start, inner_start + local_d_inner),
        (d_inner + inner_start, d_inner + inner_start + local_d_inner),
        (
            d_inner + d_inner + group_state_start,
            d_inner + d_inner + group_state_start + local_groups_state,
        ),
        (
            d_inner + d_inner + groups_state + group_state_start,
            d_inner + d_inner + groups_state + group_state_start + local_groups_state,
        ),
        (
            d_inner + conv_dim + head_start,
            d_inner + conv_dim + head_start + local_heads,
        ),
    ]
    return _take_last_dim_segments(weight, segments)


def _slice_conv_dim(value: np.ndarray, dims: dict[str, int]) -> np.ndarray:
    d_inner = dims["d_inner"]
    groups_state = dims["groups_state"]
    inner_start = dims["inner_start"]
    local_d_inner = dims["local_d_inner"]
    group_state_start = dims["group_state_start"]
    local_groups_state = dims["local_groups_state"]
    segments = [
        (inner_start, inner_start + local_d_inner),
        (d_inner + group_state_start, d_inner + group_state_start + local_groups_state),
        (
            d_inner + groups_state + group_state_start,
            d_inner + groups_state + group_state_start + local_groups_state,
        ),
    ]
    return _take_first_dim_segments(value, segments)


def _validate_nemotron_h_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Nemotron-H tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "Nemotron-H tensor parallel requires num_attention_heads divisible by tp_size "
            f"({config.num_attention_heads} vs {tp})"
        )
    # K/V heads either shard evenly over ranks or replicate one aligned KV
    # head across a contiguous rank group. Full-bank replication would map
    # local Q heads to the wrong KV head in TensorRT's local GQA expansion.
    _kv_rank_layout(config, parallel)

    for key in ("_d_inner", "_mamba_num_heads", "_n_groups", "_mlp_size"):
        if int(weights[key]) % tp != 0:
            raise ValueError(
                f"Nemotron-H tensor parallel requires {key} divisible by tp_size "
                f"({weights[key]} vs {tp})"
            )

    if int(weights.get("_num_moe_layers", 0)) > 0:
        for key in ("_moe_intermediate_size", "_shared_expert_intermediate_size"):
            value = int(weights.get(key, 0))
            if value and value % tp != 0:
                raise ValueError(
                    f"Nemotron-H tensor parallel requires {key} divisible by tp_size "
                    f"({value} vs {tp})"
                )


def shard_nemotron_h_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local Nemotron-H weights for the TP builder."""
    _validate_nemotron_h_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    dims = _mamba2_rank_dims(weights, parallel)
    kv_head_start, local_kv_heads = _kv_rank_layout(config, parallel)
    head_dim = int(config.head_dim)
    kv_start = kv_head_start * head_dim
    kv_end = kv_start + local_kv_heads * head_dim
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith(".mamba_in_proj"):
            out[key] = _slice_mamba_in_proj(value, dims)
        elif key.endswith((".conv1d_weight", ".conv1d_bias")):
            out[key] = _slice_conv_dim(value, dims)
        elif key.endswith((".A", ".D", ".dt_bias")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".mamba_norm"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".mamba_out_proj"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".experts.w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".experts.w_down"):
            out[key] = _slice_middle_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".shared_expert.w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".shared_expert.w_down"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".router", ".router_bias", ".moe_fc1", ".moe_fc2")):
            out[key] = value
        elif key.endswith(".w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_down"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_q"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_k", ".w_v")):
            out[key] = np.ascontiguousarray(value[..., kv_start:kv_end])
        elif key.endswith(".w_o"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_d_inner"] = dims["local_d_inner"]
    out["_conv_dim"] = dims["local_conv_dim"]
    out["_mamba_num_heads"] = dims["local_heads"]
    out["_n_groups"] = dims["local_groups"]
    out["_attention_size"] = int(weights["_attention_size"]) // parallel.tp_size
    out["_mlp_size"] = int(weights["_mlp_size"]) // parallel.tp_size

    out["_num_key_value_heads"] = local_kv_heads
    out["_kv_attention_size"] = local_kv_heads * head_dim

    if int(weights.get("_num_moe_layers", 0)) > 0:
        out["_moe_intermediate_size"] = (
            int(weights["_moe_intermediate_size"]) // parallel.tp_size
        )
        shared_intermediate = int(weights.get("_shared_expert_intermediate_size", 0))
        if shared_intermediate:
            out["_shared_expert_intermediate_size"] = (
                shared_intermediate // parallel.tp_size
            )

    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_mamba2_tp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_state: int,
    d_conv: int,
    conv_dim: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    n_groups: int,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Mamba-2 SSD layer (single-step decode).

    Mamba-2 in_proj splits: [gate(d_inner), hidden_B_C(conv_dim), dt(nheads)]
    Conv1d operates on hidden_B_C (d_inner + 2*n_groups*d_state channels).
    After conv+SiLU, split: hidden[d_inner], B[n_groups*d_state], C[n_groups*d_state].
    SSM state shape: [nheads, headdim, d_state] for full headdim-aware state.

    Returns: {hidden, present_conv, present_ssm}
    """
    groups_state_size = n_groups * d_state

    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projection =====
    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim, weights[f"{prefix}.mamba_in_proj"], dtype=dtype
    )  # [1, proj_dim]

    # Split: gate [d_inner], hidden_B_C [conv_dim], dt [nheads]
    offset = 0
    gate_slice = network.add_slice(projected, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    gate = gate_slice.get_output(0)
    offset += d_inner

    hbc_slice = network.add_slice(projected, start=(0, offset), shape=(1, conv_dim), stride=(1, 1))
    hidden_B_C = hbc_slice.get_output(0)
    offset += conv_dim

    dt_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, mamba_num_heads), stride=(1, 1)
    )
    dt_raw = dt_slice.get_output(0)

    # ===== 3. Conv1d step on hidden_B_C =====
    # conv_state_in: [conv_dim, d_conv]
    # hidden_B_C: [1, conv_dim] -> [conv_dim, 1]
    hbc_col = network.add_shuffle(hidden_B_C)
    hbc_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(conv_dim, d_conv - 1), stride=(1, 1)
        )
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), hbc_col.get_output(0)]
        )
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = hbc_col.get_output(0)

    conv_w = _add_constant_like(
        network,
        (conv_dim, d_conv),
        weights[f"{prefix}.conv1d_weight"],
        present_conv,
        storage_dtype=dtype,
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)

    # ===== 4. Split hidden, B, C from activated output =====
    hidden_x_slice = network.add_slice(
        hbc_activated, start=(0, 0), shape=(1, d_inner), stride=(1, 1)
    )
    hidden_x = hidden_x_slice.get_output(0)

    B_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner), shape=(1, groups_state_size), stride=(1, 1)
    )
    B_raw = B_raw_slice.get_output(0)

    C_raw_slice = network.add_slice(
        hbc_activated,
        start=(0, d_inner + groups_state_size),
        shape=(1, groups_state_size),
        stride=(1, 1),
    )
    C_raw = C_raw_slice.get_output(0)

    # ===== 5. dt: add bias + softplus =====
    dt_bias_const = _add_constant_like(
        network,
        (1, mamba_num_heads),
        weights[f"{prefix}.dt_bias"],
        dt_raw,
        storage_dtype=dtype,
    )
    dt_biased = network.add_elementwise(dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    # The checkpoint contains dt_bias values as large as 33.5. A naive FP16
    # exp overflows above ~11, while the original Mamba kernel evaluates this
    # softplus stably. Keep this scalar recurrence boundary in FP32.
    dt_for_state = dt_biased.get_output(0)
    if dt_for_state.dtype != trt.float32:
        dt_for_state = network.add_cast(dt_for_state, trt.float32).get_output(0)
    dt = _add_stable_softplus(network, dt_for_state)  # [1, mamba_num_heads]

    # ===== 6. Multi-head SSM step =====
    # A: [nheads] -> [nheads, 1, 1] for broadcast
    A_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1, 1),
        weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1),
        dtype=np.float32,
    )

    # dt: [1, nheads] -> [nheads, 1, 1]
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)

    # dA = exp(dt * A): broadcast to [nheads, headdim, d_state]
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    # B: [1, n_groups*d_state] -> [n_groups, d_state] -> expand to [nheads, d_state]
    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups

    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = _add_constant_like(
            network,
            (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=dtype),
            B_3d.get_output(0),
            storage_dtype=dtype,
        )
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        B_heads_s = network.add_shuffle(B_tiled.get_output(0))
        B_heads_s.reshape_dims = (mamba_num_heads, d_state)
        B_heads = B_heads_s.get_output(0)
    else:
        B_heads = B_grouped.get_output(0)

    # C: same group expansion
    C_grouped = network.add_shuffle(C_raw)
    C_grouped.reshape_dims = (n_groups, d_state)

    if heads_per_group > 1:
        C_3d = network.add_shuffle(C_grouped.get_output(0))
        C_3d.reshape_dims = (n_groups, 1, d_state)
        C_tiled = network.add_elementwise(
            C_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        C_heads_s = network.add_shuffle(C_tiled.get_output(0))
        C_heads_s.reshape_dims = (mamba_num_heads, d_state)
        C_heads = C_heads_s.get_output(0)
    else:
        C_heads = C_grouped.get_output(0)

    # x: [1, d_inner] -> [nheads, headdim]
    x_heads = network.add_shuffle(hidden_x)
    x_heads.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # dBx[h,d,s] = dt[h] * B[h,s] * x[h,d]
    # dt_B: [nheads, 1, 1] * [nheads, 1, d_state] -> [nheads, 1, d_state]
    B_3d_expand = network.add_shuffle(B_heads)
    B_3d_expand.reshape_dims = (mamba_num_heads, 1, d_state)
    B_for_state = B_3d_expand.get_output(0)
    if B_for_state.dtype != trt.float32:
        B_for_state = network.add_cast(B_for_state, trt.float32).get_output(0)
    dt_B = network.add_elementwise(dt_col.get_output(0), B_for_state, trt.ElementWiseOperation.PROD)

    # x: [nheads, headdim] -> [nheads, headdim, 1]
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)
    x_for_state = x_3d.get_output(0)
    if x_for_state.dtype != trt.float32:
        x_for_state = network.add_cast(x_for_state, trt.float32).get_output(0)

    # dBx: [nheads, headdim, 1] * [nheads, 1, d_state] -> [nheads, headdim, d_state]
    dBx = network.add_elementwise(x_for_state, dt_B.get_output(0), trt.ElementWiseOperation.PROD)

    # SSM update: new_ssm = dA * ssm_state + dBx
    # ssm_state_in: [nheads, headdim, d_state]
    decay = network.add_elementwise(dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)  # [nheads, headdim, d_state]

    # y[h,d] = sum_s(ssm_state[h,d,s] * C[h,s])
    # C: [nheads, d_state] -> [nheads, d_state, 1]
    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    C_for_state = C_col.get_output(0)
    if C_for_state.dtype != trt.float32:
        C_for_state = network.add_cast(C_for_state, trt.float32).get_output(0)
    # batch matmul: [nheads, headdim, d_state] @ [nheads, d_state, 1] -> [nheads, headdim, 1]
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_for_state, trt.MatrixOperation.NONE
    )
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    # D skip: D[h] * x[h,d]
    D_const = graph_ops.add_constant(
        network,
        (mamba_num_heads, 1),
        weights[f"{prefix}.D"].reshape(mamba_num_heads, 1),
        dtype=np.float32,
    )
    x_for_skip = x_heads.get_output(0)
    if x_for_skip.dtype != trt.float32:
        x_for_skip = network.add_cast(x_for_skip, trt.float32).get_output(0)
    Dx = network.add_elementwise(D_const, x_for_skip, trt.ElementWiseOperation.PROD)

    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )
    # [nheads, headdim] -> [1, d_inner]
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)
    y_for_gate = y_flat.get_output(0)
    if y_for_gate.dtype != gate.dtype:
        y_for_gate = network.add_cast(y_for_gate, gate.dtype).get_output(0)

    # ===== 7. Gated Group RMSNorm (norm_before_gate=False) =====
    # HF: output = weight * group_rms_norm(y * silu(gate))
    # Gate is applied BEFORE normalization. RMSNorm is per-group,
    # with group_size = d_inner // n_groups.
    mamba_norm_w = weights[f"{prefix}.mamba_norm"]
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32), dtype=np.float32
    )

    # Step 1: Apply silu(gate) to y BEFORE norm
    gate_activated = graph_ops.add_activation(network, gate, "silu", dtype=dtype)
    y_gated = network.add_elementwise(y_for_gate, gate_activated, trt.ElementWiseOperation.PROD)

    # Step 2: Group RMSNorm — reshape to [n_groups, group_size], norm per group
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)
    norm_input = y_grouped.get_output(0)
    norm_output_dtype = norm_input.dtype
    if dtype != np.float32:
        norm_input = network.add_cast(norm_input, trt.float32).get_output(0)

    sq = network.add_elementwise(norm_input, norm_input, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        norm_input, recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Reshape back to [1, d_inner] and apply weight
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), mamba_norm_w, dtype=np.float32)
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    gated_tensor = gated.get_output(0)
    if gated_tensor.dtype != norm_output_dtype:
        gated_tensor = network.add_cast(gated_tensor, norm_output_dtype).get_output(0)

    # ===== 8. Output projection + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network,
        gated_tensor,
        d_inner,
        hidden_size,
        weights[f"{prefix}.mamba_out_proj"],
        dtype=dtype,
    )

    out = add_all_reduce_sum(network, out, tp_size)
    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_mlp_tp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    normed = graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        eps_tensor,
        dtype=dtype,
    )
    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size, weights[f"{prefix}.w_up"], dtype=dtype
    )
    activated = graph_ops.add_activation(network, up, "relu2", dtype=dtype)
    down = graph_ops.add_matmul_rhs_constant(
        network,
        activated,
        mlp_size,
        hidden_size,
        weights[f"{prefix}.w_down"],
        dtype=dtype,
    )
    down = add_all_reduce_sum(network, down, tp_size)
    residual = network.add_elementwise(hidden, down, trt.ElementWiseOperation.SUM)
    return {"hidden": residual.get_output(0)}


def _add_moe_tp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    moe_latent: int,
    shared_expert_intermediate: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add a TP factorized-latent routed and shared-expert block."""
    if not 0 < top_k <= num_experts:
        raise ValueError(
            f"MoE top_k must be in [1, {num_experts}], got {top_k}"
        )

    normed = graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        eps_tensor,
        dtype=dtype,
    )
    # The reference router and its learned score correction run in FP32.
    # Keep sigmoid, selection, and combine weights in FP32 as well so expert
    # choices do not change when the rest of the layer runs in FP16/BF16.
    router_input = normed
    if router_input.dtype != trt.float32:
        router_input = network.add_cast(router_input, trt.float32).get_output(0)
    router_logits = graph_ops.add_matmul_rhs_constant(
        network,
        router_input,
        hidden_size,
        num_experts,
        weights[f"{prefix}.router"],
        dtype=np.float32,
    )
    scores = network.add_activation(
        router_logits, trt.ActivationType.SIGMOID
    ).get_output(0)

    router_bias = weights.get(f"{prefix}.router_bias")
    if router_bias is not None:
        bias = graph_ops.add_constant(
            network,
            (1, num_experts),
            router_bias.reshape(1, num_experts),
            dtype=np.float32,
        )
        sel_scores = network.add_elementwise(
            scores, bias, trt.ElementWiseOperation.SUM
        ).get_output(0)
    else:
        sel_scores = scores

    topk = network.add_topk(sel_scores, trt.TopKOperation.MAX, top_k, 1 << 1)
    top_indices = topk.get_output(1)
    idx_1d = network.add_shuffle(top_indices)
    idx_1d.reshape_dims = (top_k,)
    combine_w = network.add_gather(scores, idx_1d.get_output(0), 1).get_output(0)

    if norm_topk_prob:
        sum_w = network.add_reduce(
            combine_w, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
        )
        tiny = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([1e-20], dtype=np.float32),
            dtype=np.float32,
        )
        denominator = network.add_elementwise(
            sum_w.get_output(0), tiny, trt.ElementWiseOperation.SUM
        )
        combine_w = network.add_elementwise(
            combine_w, denominator.get_output(0), trt.ElementWiseOperation.DIV
        ).get_output(0)

    if routed_scaling_factor != 1.0:
        scale = graph_ops.add_constant(
            network,
            (1, 1),
            np.array([routed_scaling_factor], dtype=np.float32),
            dtype=np.float32,
        )
        combine_w = network.add_elementwise(
            combine_w, scale, trt.ElementWiseOperation.PROD
        ).get_output(0)

    latent_in = graph_ops.add_matmul_rhs_constant(
        network,
        normed,
        hidden_size,
        moe_latent,
        weights[f"{prefix}.moe_fc1"],
        dtype=dtype,
    )
    selected_experts = _add_selected_latent_experts(
        network,
        latent_in,
        top_indices,
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"],
        top_k=top_k,
        dtype=dtype,
    )
    if selected_experts.dtype != trt.float32:
        selected_experts = network.add_cast(
            selected_experts, trt.float32
        ).get_output(0)

    combine_w_3d = network.add_shuffle(combine_w)
    combine_w_3d.reshape_dims = (top_k, 1, 1)
    weighted_experts = network.add_elementwise(
        selected_experts,
        combine_w_3d.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    routed_latent = network.add_reduce(
        weighted_experts,
        trt.ReduceOperation.SUM,
        1 << 0,
        keep_dims=False,
    ).get_output(0)

    # Expert down projections are row-sharded, so each rank has a partial
    # latent result. Sum those partials in FP32 before the replicated fc2;
    # reducing after fc2 changes rounding and repeats avoidable projection work.
    routed_latent = add_all_reduce_sum(network, routed_latent, tp_size)
    if routed_latent.dtype != latent_in.dtype:
        routed_latent = network.add_cast(
            routed_latent, latent_in.dtype
        ).get_output(0)
    routed_hidden = graph_ops.add_matmul_rhs_constant(
        network,
        routed_latent,
        moe_latent,
        hidden_size,
        weights[f"{prefix}.moe_fc2"],
        dtype=dtype,
    )

    shared_up = weights.get(f"{prefix}.shared_expert.w_up")
    if shared_up is not None and shared_expert_intermediate > 0:
        up = graph_ops.add_matmul_rhs_constant(
            network,
            normed,
            hidden_size,
            shared_expert_intermediate,
            shared_up,
            dtype=dtype,
        )
        activated = graph_ops.add_activation(network, up, "relu2", dtype=dtype)
        shared_hidden = graph_ops.add_matmul_rhs_constant(
            network,
            activated,
            shared_expert_intermediate,
            hidden_size,
            weights[f"{prefix}.shared_expert.w_down"],
            dtype=dtype,
        )
        shared_hidden = add_all_reduce_sum(network, shared_hidden, tp_size)
        moe_out = network.add_elementwise(
            routed_hidden, shared_hidden, trt.ElementWiseOperation.SUM
        ).get_output(0)
    else:
        moe_out = routed_hidden

    residual = network.add_elementwise(hidden, moe_out, trt.ElementWiseOperation.SUM)
    return {"hidden": residual.get_output(0)}


def build_nemotron_h_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_nemotron_h_tp_engine requires tensor_parallel mode with tp_size > 1"
        )
    if quant_ctx is not None:
        raise ValueError(
            "Nemotron-H tensor-parallel builds do not support quantization"
        )

    work_np_dtype, work_trt_dtype = _precision_dtypes(precision)
    rank_weights = shard_nemotron_h_weights(config, weights, parallel=parallel)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    layer_types: list[str] = rank_weights["_layer_types"]
    requested_fp32_layers = frozenset(
        int(layer) for layer in config.raw.get("_fp32_layers", ())
    )
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer < 0 or layer > num_layers
    )
    if invalid_fp32_layers:
        raise ValueError(
            f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}"
        )
    low_precision = precision in ("fp16", "bf16")
    use_fp32_io = low_precision and num_layers in requested_fp32_layers
    io_np_dtype = np.float32 if use_fp32_io else work_np_dtype
    io_trt_dtype = trt.float32 if use_fp32_io else work_trt_dtype

    d_inner = int(rank_weights["_d_inner"])
    d_state = int(rank_weights["_d_state"])
    d_conv = int(rank_weights["_d_conv"])
    conv_dim = int(rank_weights["_conv_dim"])
    mamba_num_heads = int(rank_weights["_mamba_num_heads"])
    mamba_head_dim = int(rank_weights["_mamba_head_dim"])
    n_groups = int(rank_weights["_n_groups"])
    num_mamba = int(rank_weights["_num_mamba_layers"])
    num_attn = int(rank_weights["_num_attention_layers"])
    num_moe = int(rank_weights.get("_num_moe_layers", 0))
    attention_size = int(rank_weights["_attention_size"])
    mlp_size = int(rank_weights["_mlp_size"])
    num_experts = int(rank_weights.get("_num_experts", 0))
    top_k = int(rank_weights.get("_num_experts_per_tok", 0))
    moe_intermediate = int(rank_weights.get("_moe_intermediate_size", 0))
    moe_latent = int(rank_weights.get("_moe_latent_size", 0))
    shared_expert_intermediate = int(
        rank_weights.get("_shared_expert_intermediate_size", 0)
    )
    routed_scaling_factor = float(rank_weights.get("_routed_scaling_factor", 1.0))
    norm_topk_prob = bool(rank_weights.get("_norm_topk_prob", True))

    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(rank_weights["_num_key_value_heads"])
    head_dim = int(config.head_dim)
    kv_attention_size = int(rank_weights["_kv_attention_size"])
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window)
    )

    conv_state_inputs = []
    ssm_state_inputs = []
    for mi in range(num_mamba):
        conv_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi),
                trt.float32,
                (conv_dim, d_conv),
            )
        )
        ssm_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("ssm_state", mi),
                trt.float32,
                (mamba_num_heads, mamba_head_dim, d_state),
            )
        )

    cache_k_inputs = []
    cache_v_inputs = []
    for ai in range(num_attn):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", ai),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                work_trt_dtype,
                (max_cache_length, kv_attention_size),
            )
        )

    embedding_table = _add_typed_constant(
        network,
        (vocab, hidden),
        rank_weights["embedding"],
        storage_dtype=io_np_dtype,
        runtime_dtype=io_trt_dtype,
    )
    eps_tensor = _add_typed_constant(
        network,
        (1, 1),
        np.array([config.rms_norm_eps], dtype=work_np_dtype),
        storage_dtype=work_np_dtype,
        runtime_dtype=work_trt_dtype,
    )
    io_eps_tensor = (
        _add_typed_constant(
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=np.float32),
            storage_dtype=np.float32,
            runtime_dtype=trt.float32,
        )
        if use_fp32_io
        else eps_tensor
    )

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_conv_outputs = []
    present_ssm_outputs = []
    present_k_outputs = []
    present_v_outputs = []
    mamba_counter = 0
    attn_counter = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_type = layer_types[layer_idx]
        use_fp32_layer = low_precision and layer_idx in requested_fp32_layers
        layer_np_dtype = np.float32 if use_fp32_layer else work_np_dtype
        layer_trt_dtype = trt.float32 if use_fp32_layer else work_trt_dtype
        layer_hidden = hidden_state
        layer_eps = eps_tensor
        if layer_hidden.dtype != layer_trt_dtype:
            layer_hidden = network.add_cast(
                layer_hidden, layer_trt_dtype
            ).get_output(0)
        if layer_eps.dtype != layer_trt_dtype:
            layer_eps = network.add_cast(layer_eps, layer_trt_dtype).get_output(0)

        if layer_type == "mamba2":
            conv_state = conv_state_inputs[mamba_counter]
            if conv_state.dtype != layer_trt_dtype:
                conv_state = network.add_cast(
                    conv_state, layer_trt_dtype
                ).get_output(0)
            result = _add_mamba2_tp_layer(
                network=network,
                hidden=layer_hidden,
                conv_state_in=conv_state,
                ssm_state_in=ssm_state_inputs[mamba_counter],
                eps_tensor=layer_eps,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                d_inner=d_inner,
                d_state=d_state,
                d_conv=d_conv,
                conv_dim=conv_dim,
                mamba_num_heads=mamba_num_heads,
                mamba_head_dim=mamba_head_dim,
                n_groups=n_groups,
                tp_size=parallel.tp_size,
                dtype=layer_np_dtype,
            )
            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])
            mamba_counter += 1
        elif layer_type == "mlp":
            result = _add_mlp_tp_layer(
                network=network,
                hidden=layer_hidden,
                eps_tensor=layer_eps,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                mlp_size=mlp_size,
                tp_size=parallel.tp_size,
                dtype=layer_np_dtype,
            )
            hidden_state = result["hidden"]
        elif layer_type == "attention":
            cache_k = cache_k_inputs[attn_counter]
            cache_v = cache_v_inputs[attn_counter]
            layer_mask = attention_mask
            if cache_k.dtype != layer_trt_dtype:
                cache_k = network.add_cast(cache_k, layer_trt_dtype).get_output(0)
            if cache_v.dtype != layer_trt_dtype:
                cache_v = network.add_cast(cache_v, layer_trt_dtype).get_output(0)
            if layer_mask.dtype != layer_trt_dtype:
                layer_mask = network.add_cast(
                    layer_mask, layer_trt_dtype
                ).get_output(0)
            result = graph_blocks.add_attention_block(
                network,
                layer_hidden,
                cache_k,
                cache_v,
                layer_mask,
                position_id,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                attention_size=attention_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_cache_length=max_cache_length,
                eps_tensor=layer_eps,
                dtype=layer_np_dtype,
            )
            attn_out = add_all_reduce_sum(
                network, result["attn_out"], parallel.tp_size
            )
            hidden_state = network.add_elementwise(
                layer_hidden, attn_out, trt.ElementWiseOperation.SUM
            ).get_output(0)
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1
        elif layer_type == "moe":
            result = _add_moe_tp_layer(
                network=network,
                hidden=layer_hidden,
                eps_tensor=layer_eps,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                num_experts=num_experts,
                top_k=top_k,
                moe_latent=moe_latent,
                shared_expert_intermediate=shared_expert_intermediate,
                routed_scaling_factor=routed_scaling_factor,
                norm_topk_prob=norm_topk_prob,
                tp_size=parallel.tp_size,
                dtype=layer_np_dtype,
            )
            hidden_state = result["hidden"]
        else:
            raise ValueError(
                f"Unsupported Nemotron-H layer type {layer_type!r} at layer {layer_idx}"
            )

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    if hidden_state.dtype != io_trt_dtype:
        hidden_state = network.add_cast(hidden_state, io_trt_dtype).get_output(0)
    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            io_eps_tensor,
            dtype=io_np_dtype,
        )

    logits = graph_ops.add_matmul_rhs_constant(
        network,
        hidden_state,
        hidden,
        vocab,
        rank_weights["w_lm_head"],
        dtype=io_np_dtype,
    )
    logits = graph_ops.add_bias_sum(
        network,
        logits,
        vocab,
        np.zeros(vocab, dtype=io_np_dtype),
        dtype=io_np_dtype,
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for mi in range(num_mamba):
        present_conv = present_conv_outputs[mi]
        present_ssm = present_ssm_outputs[mi]
        if present_conv.dtype != trt.float32:
            present_conv = network.add_cast(present_conv, trt.float32).get_output(0)
        if present_ssm.dtype != trt.float32:
            present_ssm = network.add_cast(present_ssm, trt.float32).get_output(0)
        present_conv.name = graph_ops.layer_tensor_name("present_conv", mi)
        present_ssm.name = graph_ops.layer_tensor_name("present_ssm", mi)
        network.mark_output(present_conv)
        network.mark_output(present_ssm)

    for ai in range(num_attn):
        present_k = present_k_outputs[ai]
        present_v = present_v_outputs[ai]
        if present_k.dtype != work_trt_dtype:
            present_k = network.add_cast(present_k, work_trt_dtype).get_output(0)
        if present_v.dtype != work_trt_dtype:
            present_v = network.add_cast(present_v, work_trt_dtype).get_output(0)
        present_k.name = graph_ops.layer_tensor_name("present_k", ai)
        present_v.name = graph_ops.layer_tensor_name("present_v", ai)
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            "[trtmc build] Nemotron-H TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_mamba_heads={mamba_num_heads}, local_attn_heads={num_heads}, "
            f"local_mlp={mlp_size}, num_moe={num_moe}, experts={num_experts}, "
            f"top_k={top_k}, local_moe_intermediate={moe_intermediate}, "
            f"precision={precision})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Nemotron-H TP engine build failed")
    return bytes(plan)

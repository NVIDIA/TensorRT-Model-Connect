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
from .plugin import _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


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
    if int(config.num_key_value_heads) % tp != 0:
        raise ValueError(
            "Nemotron-H tensor parallel requires num_key_value_heads divisible by tp_size "
            f"({config.num_key_value_heads} vs {tp})"
        )

    for key in ("_d_inner", "_mamba_num_heads", "_n_groups", "_mlp_size"):
        if int(weights[key]) % tp != 0:
            raise ValueError(
                f"Nemotron-H tensor parallel requires {key} divisible by tp_size "
                f"({weights[key]} vs {tp})"
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
        elif key.endswith(".w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_down"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_q", ".w_k", ".w_v")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
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
) -> dict[str, trt.ITensor]:
    groups_state_size = n_groups * d_state

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor
    )

    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim, weights[f"{prefix}.mamba_in_proj"]
    )

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

    conv_w = graph_ops.add_constant(network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"]
    )
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu")

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

    dt_bias_const = graph_ops.add_constant(
        network, (1, mamba_num_heads), weights[f"{prefix}.dt_bias"]
    )
    dt_biased = network.add_elementwise(dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    dt_exp = network.add_unary(dt_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt = dt_softplus.get_output(0)

    A_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1, 1), weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1)
    )
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)
    dtA = network.add_elementwise(dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups
    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1), np.ones((1, heads_per_group, 1), dtype=np.float32)
        )
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        B_heads_s = network.add_shuffle(B_tiled.get_output(0))
        B_heads_s.reshape_dims = (mamba_num_heads, d_state)
        B_heads = B_heads_s.get_output(0)
    else:
        B_heads = B_grouped.get_output(0)

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

    x_heads = network.add_shuffle(hidden_x)
    x_heads.reshape_dims = (mamba_num_heads, mamba_head_dim)
    B_3d_expand = network.add_shuffle(B_heads)
    B_3d_expand.reshape_dims = (mamba_num_heads, 1, d_state)
    dt_B = network.add_elementwise(
        dt_col.get_output(0), B_3d_expand.get_output(0), trt.ElementWiseOperation.PROD
    )
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)
    dBx = network.add_elementwise(
        x_3d.get_output(0), dt_B.get_output(0), trt.ElementWiseOperation.PROD
    )

    decay = network.add_elementwise(dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_ssm.get_output(0)

    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE, C_col.get_output(0), trt.MatrixOperation.NONE
    )
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    D_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1), weights[f"{prefix}.D"].reshape(mamba_num_heads, 1)
    )
    Dx = network.add_elementwise(D_const, x_heads.get_output(0), trt.ElementWiseOperation.PROD)
    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM
    )
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)

    gate_activated = graph_ops.add_activation(network, gate, "silu")
    y_gated = network.add_elementwise(
        y_flat.get_output(0), gate_activated, trt.ElementWiseOperation.PROD
    )
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)

    sq = network.add_elementwise(
        y_grouped.get_output(0), y_grouped.get_output(0), trt.ElementWiseOperation.PROD
    )
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_small = graph_ops.add_constant(network, (1, 1), np.array([1e-5], dtype=np.float32))
    denom_in = network.add_elementwise(mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        y_grouped.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), weights[f"{prefix}.mamba_norm"])
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )

    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size, weights[f"{prefix}.mamba_out_proj"]
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
) -> dict[str, trt.ITensor]:
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor
    )
    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size, weights[f"{prefix}.w_up"]
    )
    activated = graph_ops.add_activation(network, up, "relu2")
    down = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size, weights[f"{prefix}.w_down"]
    )
    down = add_all_reduce_sum(network, down, tp_size)
    residual = network.add_elementwise(hidden, down, trt.ElementWiseOperation.SUM)
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
    del precision, quant_ctx
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_nemotron_h_tp_engine requires tensor_parallel mode with tp_size > 1"
        )

    rank_weights = shard_nemotron_h_weights(config, weights, parallel=parallel)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    layer_types: list[str] = rank_weights["_layer_types"]

    d_inner = int(rank_weights["_d_inner"])
    d_state = int(rank_weights["_d_state"])
    d_conv = int(rank_weights["_d_conv"])
    conv_dim = int(rank_weights["_conv_dim"])
    mamba_num_heads = int(rank_weights["_mamba_num_heads"])
    mamba_head_dim = int(rank_weights["_mamba_head_dim"])
    n_groups = int(rank_weights["_n_groups"])
    num_mamba = int(rank_weights["_num_mamba_layers"])
    num_attn = int(rank_weights["_num_attention_layers"])
    attention_size = int(rank_weights["_attention_size"])
    mlp_size = int(rank_weights["_mlp_size"])

    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(config.num_key_value_heads) // parallel.tp_size
    head_dim = int(config.head_dim)
    kv_attention_size = num_kv_heads * head_dim
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

    conv_state_inputs = []
    ssm_state_inputs = []
    for mi in range(num_mamba):
        conv_state_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("conv_state", mi), trt.float32, (conv_dim, d_conv)
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
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", ai),
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
        )

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), rank_weights["embedding"])
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
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
        lt = layer_types[layer_idx]
        if lt == "mamba2":
            result = _add_mamba2_tp_layer(
                network=network,
                hidden=hidden_state,
                conv_state_in=conv_state_inputs[mamba_counter],
                ssm_state_in=ssm_state_inputs[mamba_counter],
                eps_tensor=eps_tensor,
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
            )
            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])
            mamba_counter += 1
        elif lt == "mlp":
            result = _add_mlp_tp_layer(
                network=network,
                hidden=hidden_state,
                eps_tensor=eps_tensor,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                mlp_size=mlp_size,
                tp_size=parallel.tp_size,
            )
            hidden_state = result["hidden"]
        elif lt == "attention":
            result = graph_blocks.add_attention_block(
                network,
                hidden_state,
                cache_k_inputs[attn_counter],
                cache_v_inputs[attn_counter],
                attention_mask,
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
                eps_tensor=eps_tensor,
            )
            attn_out = add_all_reduce_sum(network, result["attn_out"], parallel.tp_size)
            residual = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = residual.get_output(0)
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(network, hidden_state, hidden, final_norm, eps_tensor)

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_lm_head"]
    )
    logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for mi in range(num_mamba):
        present_conv_outputs[mi].name = graph_ops.layer_tensor_name("present_conv", mi)
        present_ssm_outputs[mi].name = graph_ops.layer_tensor_name("present_ssm", mi)
        network.mark_output(present_conv_outputs[mi])
        network.mark_output(present_ssm_outputs[mi])

    for ai in range(num_attn):
        present_k_outputs[ai].name = graph_ops.layer_tensor_name("present_k", ai)
        present_v_outputs[ai].name = graph_ops.layer_tensor_name("present_v", ai)
        network.mark_output(present_k_outputs[ai])
        network.mark_output(present_v_outputs[ai])

    if verbose:
        print(
            "[trtmc build] Nemotron-H TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_mamba_heads={mamba_num_heads}, local_attn_heads={num_heads}, "
            f"local_mlp={mlp_size})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Nemotron-H TP engine build failed")
    return bytes(plan)

"""Tensor-parallel Nemotron-H hybrid builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_blocks, graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _slice_middle_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=1)[rank])


def _take_last_dim_segments(arr: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[..., start:end] for start, end in segments], axis=-1))


def _take_first_dim_segments(arr: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[start:end, ...] for start, end in segments], axis=0))


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
            f"({config.num_attention_heads} vs {tp})")
    # When num_key_value_heads is divisible by tp_size, we shard K/V across
    # ranks; otherwise we replicate K/V (every rank carries the full KV head
    # bank) -- mirrors the policy used by the Qwen-MoE TP builder. This is
    # required for highly-GQA models like Nemotron-3-Super (num_kv_heads=2)
    # running at tp_size=8.

    for key in ("_d_inner", "_mamba_num_heads", "_n_groups", "_mlp_size"):
        if int(weights[key]) % tp != 0:
            raise ValueError(
                f"Nemotron-H tensor parallel requires {key} divisible by tp_size "
                f"({weights[key]} vs {tp})")

    # MoE checks only apply if the layer stack actually contains MoE-FFN
    # ("E") layers. We shard the routed-expert intermediate dimension across
    # ranks (column-shard up, row-shard down -> allreduce), and shard the
    # shared-expert intermediate too. The latent dimension is replicated.
    if int(weights.get("_num_moe_layers", 0)) > 0:
        moe_inter = int(weights.get("_moe_intermediate_size", 0))
        if moe_inter % tp != 0:
            raise ValueError(
                "Nemotron-H tensor parallel requires moe_intermediate_size "
                f"divisible by tp_size ({moe_inter} vs {tp})")
        shared_inter = int(weights.get("_shared_expert_intermediate_size", 0))
        if shared_inter and shared_inter % tp != 0:
            raise ValueError(
                "Nemotron-H tensor parallel requires "
                "moe_shared_expert_intermediate_size divisible by tp_size "
                f"({shared_inter} vs {tp})")


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
    shard_kv = int(config.num_key_value_heads) % parallel.tp_size == 0
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
        # ---- MoE 'E' layer: per-expert packed banks. Shard the routed
        # expert intermediate dim (axis -1 on up: [E, latent, inter],
        # axis 1 on down: [E, inter, latent]). The latent dim stays
        # replicated, and fc1/fc2 latent projections + router stay
        # replicated. Must match BEFORE the generic .w_up/.w_down rules.
        elif key.endswith(".experts.w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".experts.w_down"):
            out[key] = _slice_middle_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".shared_expert.w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".shared_expert.w_down"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".router", ".router_bias",
                            ".moe_fc1", ".moe_fc2")):
            # Router scoring and latent projections are replicated.
            out[key] = value
        elif key.endswith(".w_up"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_down"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".w_q"):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_k", ".w_v")):
            if shard_kv:
                out[key] = _slice_last_dim(
                    value, parallel.rank, parallel.tp_size)
            else:
                out[key] = value
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
    if int(weights.get("_num_moe_layers", 0)) > 0:
        out["_moe_intermediate_size"] = (
            int(weights["_moe_intermediate_size"]) // parallel.tp_size)
        shared_inter = int(
            weights.get("_shared_expert_intermediate_size", 0))
        if shared_inter:
            out["_shared_expert_intermediate_size"] = (
                shared_inter // parallel.tp_size)
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
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor)

    proj_dim = d_inner + conv_dim + mamba_num_heads
    projected = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, proj_dim, weights[f"{prefix}.mamba_in_proj"])

    offset = 0
    gate_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    gate = gate_slice.get_output(0)
    offset += d_inner

    hbc_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, conv_dim), stride=(1, 1))
    hidden_B_C = hbc_slice.get_output(0)
    offset += conv_dim

    dt_slice = network.add_slice(
        projected, start=(0, offset), shape=(1, mamba_num_heads), stride=(1, 1))
    dt_raw = dt_slice.get_output(0)

    hbc_col = network.add_shuffle(hidden_B_C)
    hbc_col.reshape_dims = (conv_dim, 1)
    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(conv_dim, d_conv - 1),
            stride=(1, 1))
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), hbc_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = hbc_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"])
    hbc_activated = graph_ops.add_activation(network, conv_out, "silu")

    hidden_x_slice = network.add_slice(
        hbc_activated, start=(0, 0), shape=(1, d_inner), stride=(1, 1))
    hidden_x = hidden_x_slice.get_output(0)
    B_raw_slice = network.add_slice(
        hbc_activated, start=(0, d_inner), shape=(1, groups_state_size), stride=(1, 1))
    B_raw = B_raw_slice.get_output(0)
    C_raw_slice = network.add_slice(
        hbc_activated,
        start=(0, d_inner + groups_state_size),
        shape=(1, groups_state_size),
        stride=(1, 1))
    C_raw = C_raw_slice.get_output(0)

    dt_bias_const = graph_ops.add_constant(
        network, (1, mamba_num_heads), weights[f"{prefix}.dt_bias"])
    dt_biased = network.add_elementwise(
        dt_raw, dt_bias_const, trt.ElementWiseOperation.SUM)
    dt_exp = network.add_unary(dt_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    dt_exp_p1 = network.add_elementwise(
        dt_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    dt_softplus = network.add_unary(dt_exp_p1.get_output(0), trt.UnaryOperation.LOG)
    dt = dt_softplus.get_output(0)

    A_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1, 1),
        weights[f"{prefix}.A"].reshape(mamba_num_heads, 1, 1))
    dt_col = network.add_shuffle(dt)
    dt_col.reshape_dims = (mamba_num_heads, 1, 1)
    dtA = network.add_elementwise(
        dt_col.get_output(0), A_const, trt.ElementWiseOperation.PROD)
    dA = network.add_unary(dtA.get_output(0), trt.UnaryOperation.EXP)

    B_grouped = network.add_shuffle(B_raw)
    B_grouped.reshape_dims = (n_groups, d_state)
    heads_per_group = mamba_num_heads // n_groups
    if heads_per_group > 1:
        B_3d = network.add_shuffle(B_grouped.get_output(0))
        B_3d.reshape_dims = (n_groups, 1, d_state)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=np.float32))
        B_tiled = network.add_elementwise(
            B_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
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
            C_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
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
        dt_col.get_output(0), B_3d_expand.get_output(0), trt.ElementWiseOperation.PROD)
    x_3d = network.add_shuffle(x_heads.get_output(0))
    x_3d.reshape_dims = (mamba_num_heads, mamba_head_dim, 1)
    dBx = network.add_elementwise(
        x_3d.get_output(0), dt_B.get_output(0), trt.ElementWiseOperation.PROD)

    decay = network.add_elementwise(
        dA.get_output(0), ssm_state_in, trt.ElementWiseOperation.PROD)
    new_ssm = network.add_elementwise(
        decay.get_output(0), dBx.get_output(0), trt.ElementWiseOperation.SUM)
    present_ssm = new_ssm.get_output(0)

    C_col = network.add_shuffle(C_heads)
    C_col.reshape_dims = (mamba_num_heads, d_state, 1)
    y_matmul = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.NONE,
        C_col.get_output(0), trt.MatrixOperation.NONE)
    y_squeeze = network.add_shuffle(y_matmul.get_output(0))
    y_squeeze.reshape_dims = (mamba_num_heads, mamba_head_dim)

    D_const = graph_ops.add_constant(
        network, (mamba_num_heads, 1),
        weights[f"{prefix}.D"].reshape(mamba_num_heads, 1))
    Dx = network.add_elementwise(
        D_const, x_heads.get_output(0), trt.ElementWiseOperation.PROD)
    y_plus_D = network.add_elementwise(
        y_squeeze.get_output(0), Dx.get_output(0), trt.ElementWiseOperation.SUM)
    y_flat = network.add_shuffle(y_plus_D.get_output(0))
    y_flat.reshape_dims = (1, d_inner)

    gate_activated = graph_ops.add_activation(network, gate, "silu")
    y_gated = network.add_elementwise(
        y_flat.get_output(0), gate_activated, trt.ElementWiseOperation.PROD)
    group_size = d_inner // n_groups
    y_grouped = network.add_shuffle(y_gated.get_output(0))
    y_grouped.reshape_dims = (n_groups, group_size)

    sq = network.add_elementwise(
        y_grouped.get_output(0), y_grouped.get_output(0), trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=np.float32))
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        y_grouped.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD)
    y_flat_normed = network.add_shuffle(normalized.get_output(0))
    y_flat_normed.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(
        network, (1, d_inner), weights[f"{prefix}.mamba_norm"])
    gated = network.add_elementwise(
        y_flat_normed.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size,
        weights[f"{prefix}.mamba_out_proj"])
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
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor)
    up = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, mlp_size, weights[f"{prefix}.w_up"])
    activated = graph_ops.add_activation(network, up, "relu2")
    down = graph_ops.add_matmul_rhs_constant(
        network, activated, mlp_size, hidden_size, weights[f"{prefix}.w_down"])
    down = add_all_reduce_sum(network, down, tp_size)
    residual = network.add_elementwise(hidden, down, trt.ElementWiseOperation.SUM)
    return {"hidden": residual.get_output(0)}


def _add_packed_latent_experts_tp(
    network: trt.INetworkDefinition,
    latent_in: trt.ITensor,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    """Run all routed experts on the rank-local sharded intermediate dim.

    Same shape/dataflow as the non-TP variant in plugin.py, except w_up/w_down
    contain only this rank's slice of the moe_intermediate dim:
      w_up:   [num_experts, latent, moe_intermediate // tp]
      w_down: [num_experts, moe_intermediate // tp, latent]
    The down-proj output is in latent dim and must be summed across ranks
    (we defer that allreduce until after the routed+shared combine).
    """
    num_experts, latent_size, _ = w_up.shape

    inp_3d = network.add_shuffle(latent_in)
    inp_3d.reshape_dims = (1, 1, latent_size)
    expert_scale = graph_ops.add_constant(
        network, (num_experts, 1, 1),
        np.ones((num_experts, 1, 1), dtype=np.float32))
    batched = network.add_elementwise(
        inp_3d.get_output(0), expert_scale, trt.ElementWiseOperation.PROD)

    up_w = graph_ops.add_constant(network, w_up.shape, w_up)
    down_w = graph_ops.add_constant(network, w_down.shape, w_down)

    up = network.add_matrix_multiply(
        batched.get_output(0), trt.MatrixOperation.NONE,
        up_w, trt.MatrixOperation.NONE)
    relu = network.add_activation(up.get_output(0), trt.ActivationType.RELU)
    relu2 = network.add_elementwise(
        relu.get_output(0), relu.get_output(0),
        trt.ElementWiseOperation.PROD)
    down = network.add_matrix_multiply(
        relu2.get_output(0), trt.MatrixOperation.NONE,
        down_w, trt.MatrixOperation.NONE)
    return down.get_output(0)


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
    moe_intermediate: int,
    moe_latent: int,
    shared_expert_intermediate: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    """Tensor-parallel Nemotron-3-Super latent MoE block.

    Sharding policy (mirrors the routed-+-shared-expert flow in plugin.py):
      - router gate, router_bias, fc1_latent_proj, fc2_latent_proj REPLICATED
      - per-expert up: column-shard (moe_intermediate axis)
      - per-expert down: row-shard (contracting moe_intermediate axis); each
        rank's expert output is a partial sum in latent space
      - shared expert: column-shard up + row-shard down on the hidden->shared
        intermediate->hidden path; each rank's shared output is partial in
        hidden space
      - The routed latent partial is projected back to hidden by the replicated
        fc2; both partials are summed locally and then all-reduced once.
    """
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # Replicated latent down-projection (every rank computes full latent).
    latent_in = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, moe_latent,
        weights[f"{prefix}.moe_fc1"])

    # Per-rank sharded expert bank in latent space.
    expert_outs = _add_packed_latent_experts_tp(
        network, latent_in,
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"])

    # Replicated router (same gating decision on every rank).
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_experts,
        weights[f"{prefix}.router"])
    scores_l = network.add_activation(
        router_logits, trt.ActivationType.SIGMOID)
    scores = scores_l.get_output(0)

    router_bias = weights.get(f"{prefix}.router_bias")
    if router_bias is not None:
        bias_const = graph_ops.add_constant(
            network, (1, num_experts),
            router_bias.reshape(1, num_experts))
        sel_l = network.add_elementwise(
            scores, bias_const, trt.ElementWiseOperation.SUM)
        sel_scores = sel_l.get_output(0)
    else:
        sel_scores = scores

    topk = network.add_topk(
        sel_scores, trt.TopKOperation.MAX, top_k, 1 << 1)
    top_indices = topk.get_output(1)

    idx_1d = network.add_shuffle(top_indices)
    idx_1d.reshape_dims = (top_k,)
    gathered = network.add_gather(scores, idx_1d.get_output(0), 1)
    raw_weights = gathered.get_output(0)

    if norm_topk_prob:
        sum_w = network.add_reduce(
            raw_weights, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
        norm_w = network.add_elementwise(
            raw_weights, sum_w.get_output(0),
            trt.ElementWiseOperation.DIV)
        combine_w = norm_w.get_output(0)
    else:
        combine_w = raw_weights

    if routed_scaling_factor != 1.0:
        scale_const = graph_ops.add_constant(
            network, (1, 1),
            np.array([routed_scaling_factor], dtype=np.float32))
        scaled = network.add_elementwise(
            combine_w, scale_const, trt.ElementWiseOperation.PROD)
        combine_w = scaled.get_output(0)

    routed_latent = None
    for k in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        w_slice = network.add_slice(
            combine_w, start=(0, k), shape=(1, 1), stride=(1, 1))
        w_reshape = network.add_shuffle(w_slice.get_output(0))
        w_reshape.reshape_dims = (1, 1, 1)

        pick = network.add_gather(expert_outs, idx_flat.get_output(0), 0)
        scaled_pick = network.add_elementwise(
            pick.get_output(0), w_reshape.get_output(0),
            trt.ElementWiseOperation.PROD)
        flat_pick = network.add_shuffle(scaled_pick.get_output(0))
        flat_pick.reshape_dims = (1, moe_latent)

        if routed_latent is None:
            routed_latent = flat_pick.get_output(0)
        else:
            sum_l = network.add_elementwise(
                routed_latent, flat_pick.get_output(0),
                trt.ElementWiseOperation.SUM)
            routed_latent = sum_l.get_output(0)

    # Replicated latent->hidden projection. After this the routed_hidden is
    # still a partial-sum across ranks (because the routed_latent only sums
    # this rank's slice of moe_intermediate).
    routed_hidden = graph_ops.add_matmul_rhs_constant(
        network, routed_latent, moe_latent, hidden_size,
        weights[f"{prefix}.moe_fc2"])

    shared_w_up = weights.get(f"{prefix}.shared_expert.w_up")
    if shared_w_up is not None and shared_expert_intermediate > 0:
        s_up = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, shared_expert_intermediate,
            shared_w_up)
        s_relu = network.add_activation(s_up, trt.ActivationType.RELU)
        s_relu2 = network.add_elementwise(
            s_relu.get_output(0), s_relu.get_output(0),
            trt.ElementWiseOperation.PROD)
        shared_hidden = graph_ops.add_matmul_rhs_constant(
            network, s_relu2.get_output(0), shared_expert_intermediate,
            hidden_size,
            weights[f"{prefix}.shared_expert.w_down"])
        combined = network.add_elementwise(
            routed_hidden, shared_hidden, trt.ElementWiseOperation.SUM)
        mlp_partial = combined.get_output(0)
    else:
        mlp_partial = routed_hidden

    # Single all-reduce on the combined hidden partial.
    mlp_out = add_all_reduce_sum(network, mlp_partial, tp_size)
    residual = network.add_elementwise(
        hidden, mlp_out, trt.ElementWiseOperation.SUM)
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
            "build_nemotron_h_tp_engine requires tensor_parallel mode with tp_size > 1")

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
    num_moe = int(rank_weights.get("_num_moe_layers", 0))
    attention_size = int(rank_weights["_attention_size"])
    mlp_size = int(rank_weights["_mlp_size"])
    # MoE metadata (only meaningful when num_moe > 0). Note that
    # _moe_intermediate_size has already been divided by tp_size by the
    # sharding step, while _moe_latent_size stays full (replicated).
    num_experts = int(rank_weights.get("_num_experts", 0))
    num_experts_per_tok = int(rank_weights.get("_num_experts_per_tok", 0))
    moe_intermediate = int(rank_weights.get("_moe_intermediate_size", 0))
    moe_latent = int(rank_weights.get("_moe_latent_size", 0))
    shared_expert_intermediate = int(
        rank_weights.get("_shared_expert_intermediate_size", 0))
    routed_scaling_factor = float(
        rank_weights.get("_routed_scaling_factor", 1.0))
    norm_topk_prob = bool(rank_weights.get("_norm_topk_prob", True))

    num_heads = int(config.num_attention_heads) // parallel.tp_size
    # K/V sharding policy mirrors qwen_moe: shard when divisible, replicate
    # otherwise (so highly-GQA models like Nemotron-3-Super with KV=2 still
    # work at large tp_size).
    shard_kv = int(config.num_key_value_heads) % parallel.tp_size == 0
    num_kv_heads = (
        int(config.num_key_value_heads) // parallel.tp_size
        if shard_kv else int(config.num_key_value_heads))
    head_dim = int(config.head_dim)
    kv_attention_size = num_kv_heads * head_dim
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    conv_state_inputs = []
    ssm_state_inputs = []
    for mi in range(num_mamba):
        conv_state_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("conv_state", mi),
            trt.float32, (conv_dim, d_conv)))
        ssm_state_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("ssm_state", mi),
            trt.float32, (mamba_num_heads, mamba_head_dim, d_state)))

    cache_k_inputs = []
    cache_v_inputs = []
    for ai in range(num_attn):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", ai),
            trt.float32, (max_cache_length, kv_attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", ai),
            trt.float32, (max_cache_length, kv_attention_size)))

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["embedding"])
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

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
                network, hidden_state,
                cache_k_inputs[attn_counter],
                cache_v_inputs[attn_counter],
                attention_mask, position_id,
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
                position_type="none",
            )
            attn_out = add_all_reduce_sum(network, result["attn_out"], parallel.tp_size)
            residual = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            hidden_state = residual.get_output(0)
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1
        elif lt == "moe":
            result = _add_moe_tp_layer(
                network=network,
                hidden=hidden_state,
                eps_tensor=eps_tensor,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                moe_intermediate=moe_intermediate,
                moe_latent=moe_latent,
                shared_expert_intermediate=shared_expert_intermediate,
                routed_scaling_factor=routed_scaling_factor,
                norm_topk_prob=norm_topk_prob,
                tp_size=parallel.tp_size,
            )
            hidden_state = result["hidden"]

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_ops.add_rms_norm(
            network, hidden_state, hidden, final_norm, eps_tensor)

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_lm_head"])
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, np.zeros(vocab, dtype=np.float32))
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
            f"local_mlp={mlp_size}, num_moe={num_moe}, "
            f"experts={num_experts}, top_k={num_experts_per_tok}, "
            f"local_moe_inter={moe_intermediate}, moe_latent={moe_latent})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Nemotron-H TP engine build failed")
    return bytes(plan)

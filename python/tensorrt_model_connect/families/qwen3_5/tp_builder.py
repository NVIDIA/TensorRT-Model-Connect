"""Tensor-parallel Qwen3.5 hybrid (Gated DeltaNet + GQA self-attention) builder.

Qwen3.5 differs from Nemotron-H even though both are "hybrid Mamba+attention"
family models:

  * Linear-attention layers use **Gated DeltaNet** (delta-rule linear
    attention), not Mamba-2 SSD. The SSM state is shaped
    ``[num_heads, head_dim, head_dim]`` -- there is no separate ``d_state``.
  * The DeltaNet input projection is split into FOUR matrices:
    ``in_proj_qkv`` (Q+K+V), ``in_proj_z`` (gate), ``in_proj_a`` (decay),
    ``in_proj_b`` (beta). For TP we shard each of them on the head axis.
  * Full-attention layers use partial RoPE + per-head q_norm / k_norm with
    ``(1+w)`` centering and a **post-attention output gate** fused into the
    q_proj weight matrix (split into ``w_q`` + ``w_gate_attn`` at load time).
  * SwiGLU MLP (``w_gate`` + ``w_up`` + ``w_down``).

Sharding policy mirrors ``nemotron_h/tp_builder.py``:

  * Embeddings, norms (input/post/final), q_norm/k_norm, deltanet_norm,
    A/dt_bias (per head) -- all sharded along the head axis where applicable;
    scalars and lm_head stay replicated.
  * Column-parallel weights (``.w_q``, ``.w_gate_attn``, ``.w_k``, ``.w_v``,
    ``.w_gate``, ``.w_up``, ``.deltanet_z_proj``, ``.deltanet_a_proj``,
    ``.deltanet_b_proj``): last-dim shard.
  * Row-parallel weights (``.w_o``, ``.w_down``, ``.deltanet_out_proj``):
    first-dim shard, followed by ``add_all_reduce_sum``.
  * ``.deltanet_in_proj_qkv`` is a structured column shard with three
    segments (Q[qk_dim] + K[qk_dim] + V[d_inner]); each segment is sharded
    independently on the head axis.
  * ``.conv1d_weight``/``.conv1d_bias`` are first-dim sharded with matching
    segments.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .plugin import _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig
    from ...parallel_config import ParallelConfig


# ---------------------------------------------------------------------------
# Generic slicing helpers
# ---------------------------------------------------------------------------

def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _take_last_dim_segments(
    arr: np.ndarray, segments: list[tuple[int, int]],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[..., start:end] for start, end in segments], axis=-1))


def _take_first_dim_segments(
    arr: np.ndarray, segments: list[tuple[int, int]],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[start:end, ...] for start, end in segments], axis=0))


# ---------------------------------------------------------------------------
# Per-rank DeltaNet dimensions
# ---------------------------------------------------------------------------

def _deltanet_rank_dims(weights: "WeightDict", parallel: "ParallelConfig") -> dict[str, int]:
    rank = parallel.rank
    tp = parallel.tp_size
    d_inner = int(weights["_d_inner"])
    dn_heads = int(weights["_deltanet_num_heads"])
    dn_kv_heads = int(weights["_deltanet_num_kv_heads"])
    head_dim = int(weights["_deltanet_head_dim"])
    qk_dim = dn_kv_heads * head_dim

    local_dn_heads = dn_heads // tp
    local_dn_kv_heads = dn_kv_heads // tp
    local_d_inner = local_dn_heads * head_dim
    local_qk_dim = local_dn_kv_heads * head_dim
    local_conv_dim = local_qk_dim + local_qk_dim + local_d_inner

    return {
        "rank": rank,
        "tp": tp,
        "d_inner": d_inner,
        "qk_dim": qk_dim,
        "dn_heads": dn_heads,
        "dn_kv_heads": dn_kv_heads,
        "head_dim": head_dim,
        "local_dn_heads": local_dn_heads,
        "local_dn_kv_heads": local_dn_kv_heads,
        "local_d_inner": local_d_inner,
        "local_qk_dim": local_qk_dim,
        "local_conv_dim": local_conv_dim,
        # Global offsets for the three in_proj_qkv segments [Q | K | V]
        "q_offset": rank * local_qk_dim,
        "k_offset": qk_dim + rank * local_qk_dim,
        "v_offset": 2 * qk_dim + rank * local_d_inner,
        # Per-rank head offset (a_proj / b_proj / A / dt_bias use this)
        "head_start": rank * local_dn_heads,
    }


def _slice_deltanet_in_proj_qkv(
    weight: np.ndarray, dims: dict[str, int],
) -> np.ndarray:
    """Shard the [hidden, conv_dim] in_proj_qkv into [hidden, local_conv_dim]."""
    qk_dim = dims["qk_dim"]
    d_inner = dims["d_inner"]
    q_offset = dims["q_offset"]
    k_offset = dims["k_offset"]
    v_offset = dims["v_offset"]
    local_qk_dim = dims["local_qk_dim"]
    local_d_inner = dims["local_d_inner"]
    segments = [
        (q_offset, q_offset + local_qk_dim),
        (k_offset, k_offset + local_qk_dim),
        (v_offset, v_offset + local_d_inner),
    ]
    # weight is stored transposed: [hidden, conv_dim]; the conv_dim axis is -1
    return _take_last_dim_segments(weight, segments)


def _slice_conv1d_along_first(
    value: np.ndarray, dims: dict[str, int],
) -> np.ndarray:
    """Shard conv1d_weight [conv_dim, d_conv] or conv1d_bias [conv_dim].

    Matches the in_proj_qkv segmentation in the global conv_dim axis (axis 0).
    """
    qk_dim = dims["qk_dim"]
    d_inner = dims["d_inner"]
    q_offset = dims["q_offset"]
    k_offset = dims["k_offset"]
    v_offset = dims["v_offset"]
    local_qk_dim = dims["local_qk_dim"]
    local_d_inner = dims["local_d_inner"]
    # Note: q_offset/k_offset/v_offset here are taken from the *global*
    # conv_dim coordinate space because conv1d operates over the same
    # concatenated [Q|K|V] layout that in_proj_qkv produces.
    segments = [
        (q_offset, q_offset + local_qk_dim),
        (k_offset, k_offset + local_qk_dim),
        (v_offset, v_offset + local_d_inner),
    ]
    return _take_first_dim_segments(value, segments)


# ---------------------------------------------------------------------------
# Validation + weight sharding entry point
# ---------------------------------------------------------------------------

def _validate_qwen35_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Qwen3.5 tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "Qwen3.5 tensor parallel requires num_attention_heads divisible by tp_size "
            f"({config.num_attention_heads} vs {tp})")
    if int(config.num_key_value_heads) % tp != 0:
        raise ValueError(
            "Qwen3.5 tensor parallel requires num_key_value_heads divisible by tp_size "
            f"({config.num_key_value_heads} vs {tp})")

    for key in ("_deltanet_num_heads", "_deltanet_num_kv_heads", "_mlp_size"):
        if int(weights[key]) % tp != 0:
            raise ValueError(
                f"Qwen3.5 tensor parallel requires {key} divisible by tp_size "
                f"({weights[key]} vs {tp})")


def shard_qwen35_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local Qwen3.5 weights for the TP builder."""
    _validate_qwen35_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    dims = _deltanet_rank_dims(weights, parallel)
    tp = parallel.tp_size
    rank = parallel.rank
    head_dim_attn = int(config.head_dim)

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        # ---- DeltaNet (linear attention) ----
        if key.endswith(".deltanet_in_proj_qkv"):
            out[key] = _slice_deltanet_in_proj_qkv(value, dims)
        elif key.endswith(".deltanet_z_proj"):
            # [hidden, d_inner] -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".deltanet_a_proj", ".deltanet_b_proj")):
            # [hidden, num_heads] -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".A", ".dt_bias")):
            # Per-head [num_heads] -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".conv1d_weight"):
            out[key] = _slice_conv1d_along_first(value, dims)
        elif key.endswith(".conv1d_bias"):
            out[key] = _slice_conv1d_along_first(value, dims)
        elif key.endswith(".deltanet_norm"):
            # Per-channel weight over d_inner -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".deltanet_out_proj"):
            # [d_inner, hidden] -- first-dim shard (row-parallel)
            out[key] = _slice_first_dim(value, rank, tp)

        # ---- Full attention ----
        elif key.endswith((".w_q", ".w_gate_attn")):
            # [hidden, attn_size] -- last-dim shard (column-parallel)
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_k", ".w_v")):
            # [hidden, kv_size] -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".w_o"):
            # [attn_size, hidden] -- first-dim shard (row-parallel)
            out[key] = _slice_first_dim(value, rank, tp)
        elif key.endswith(".q_norm"):
            # plugin tiles to [num_heads * head_dim]; reshape to (heads, head_dim),
            # shard heads, then flatten.
            heads = value.size // head_dim_attn
            reshaped = value.reshape(heads, head_dim_attn)
            out[key] = _slice_first_dim(reshaped, rank, tp).reshape(-1)
        elif key.endswith(".k_norm"):
            heads = value.size // head_dim_attn
            reshaped = value.reshape(heads, head_dim_attn)
            out[key] = _slice_first_dim(reshaped, rank, tp).reshape(-1)

        # ---- SwiGLU MLP ----
        elif key.endswith((".w_gate", ".w_up")):
            # [hidden, mlp_size] -- last-dim shard
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".w_down"):
            # [mlp_size, hidden] -- first-dim shard
            out[key] = _slice_first_dim(value, rank, tp)

        else:
            out[key] = value

    out["_d_inner"] = dims["local_d_inner"]
    out["_conv_dim"] = dims["local_conv_dim"]
    out["_deltanet_num_heads"] = dims["local_dn_heads"]
    out["_deltanet_num_kv_heads"] = dims["local_dn_kv_heads"]
    # attn_size = num_heads * head_dim, sharded by num_heads/tp
    out["_attn_size"] = int(weights["_attn_size"]) // tp
    out["_mlp_size"] = int(weights["_mlp_size"]) // tp
    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


# ---------------------------------------------------------------------------
# TP-aware DeltaNet layer
# ---------------------------------------------------------------------------

def _add_deltanet_tp_layer(
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
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    """Single-step Gated DeltaNet decode for TP rank.

    All per-head dimensions are LOCAL to this rank. The output projection is
    followed by all-reduce(SUM) to combine partial sums across ranks.
    """
    qk_dim = num_kv_heads * head_dim

    # ===== 1. RMSNorm (replicated) =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # ===== 2. Input projections (column-parallel) =====
    qkv = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, conv_dim,
        weights[f"{prefix}.deltanet_in_proj_qkv"])
    z = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, d_inner,
        weights[f"{prefix}.deltanet_z_proj"])
    a_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_a_proj"])
    b_raw = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, num_heads,
        weights[f"{prefix}.deltanet_b_proj"])

    # ===== 3. Conv1d step on QKV =====
    qkv_col = network.add_shuffle(qkv)
    qkv_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in,
            start=(0, 1),
            shape=(conv_dim, d_conv - 1),
            stride=(1, 1))
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), qkv_col.get_output(0)])
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = qkv_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"])
    conv_prod = network.add_elementwise(
        present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM,
        1 << 1, keep_dims=True)
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim,
        weights[f"{prefix}.conv1d_bias"])
    qkv_activated = graph_ops.add_activation(network, conv_out, "silu")

    # ===== 4. Split local Q / K / V =====
    offset = 0
    q_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    q_raw_t = q_slice.get_output(0)
    offset += qk_dim

    k_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    k_raw_t = k_slice.get_output(0)
    offset += qk_dim

    v_slice = network.add_slice(
        qkv_activated, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    v_raw = v_slice.get_output(0)

    # ===== 5. L2-normalize Q and K (per local head) =====
    q_heads_in = network.add_shuffle(q_raw_t)
    q_heads_in.reshape_dims = (num_kv_heads, head_dim)
    q_normed = graph_ops.add_l2_norm(network, q_heads_in.get_output(0), 1, eps=1e-6)

    k_heads_in = network.add_shuffle(k_raw_t)
    k_heads_in.reshape_dims = (num_kv_heads, head_dim)
    k_normed = graph_ops.add_l2_norm(network, k_heads_in.get_output(0), 1, eps=1e-6)

    # ===== 6. Expand local Q,K from num_kv_heads -> num_heads =====
    heads_per_group = num_heads // num_kv_heads
    if heads_per_group > 1:
        q_3d = network.add_shuffle(q_normed)
        q_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        tile_ones = graph_ops.add_constant(
            network, (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=np.float32))
        q_tiled = network.add_elementwise(
            q_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        q_expanded_s = network.add_shuffle(q_tiled.get_output(0))
        q_expanded_s.reshape_dims = (num_heads, head_dim)
        q_expanded = q_expanded_s.get_output(0)

        k_3d = network.add_shuffle(k_normed)
        k_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        k_tiled = network.add_elementwise(
            k_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD)
        k_t_s = network.add_shuffle(k_tiled.get_output(0))
        k_t_s.reshape_dims = (num_heads, head_dim)
        k_t = k_t_s.get_output(0)
    else:
        q_expanded = q_normed
        k_t = k_normed

    v_heads = network.add_shuffle(v_raw)
    v_heads.reshape_dims = (num_heads, head_dim)
    v_t = v_heads.get_output(0)

    # ===== 7. Decay: -exp(A_log) * softplus(a + dt_bias), all per local head =====
    A_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.A"])
    dt_bias_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"])
    a_biased = network.add_elementwise(
        a_raw, dt_bias_const, trt.ElementWiseOperation.SUM)

    a_exp = network.add_unary(a_biased.get_output(0), trt.UnaryOperation.EXP)
    one = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    a_exp_p1 = network.add_elementwise(
        a_exp.get_output(0), one, trt.ElementWiseOperation.SUM)
    a_softplus = network.add_unary(
        a_exp_p1.get_output(0), trt.UnaryOperation.LOG)

    decay_flat = network.add_elementwise(
        A_const, a_softplus.get_output(0), trt.ElementWiseOperation.PROD)
    decay_reshaped = network.add_shuffle(decay_flat.get_output(0))
    decay_reshaped.reshape_dims = (num_heads, 1, 1)
    decay_exp = network.add_unary(
        decay_reshaped.get_output(0), trt.UnaryOperation.EXP)

    # ===== 8. Beta = sigmoid(b) per local head =====
    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID)
    beta_reshaped = network.add_shuffle(beta.get_output(0))
    beta_reshaped.reshape_dims = (num_heads, 1)

    # ===== 9. Delta-rule state update (per local head) =====
    decayed_state = network.add_elementwise(
        decay_exp.get_output(0), ssm_state_in,
        trt.ElementWiseOperation.PROD)

    k_col = network.add_shuffle(k_t)
    k_col.reshape_dims = (num_heads, head_dim, 1)
    kv_old_3d = network.add_matrix_multiply(
        decayed_state.get_output(0), trt.MatrixOperation.TRANSPOSE,
        k_col.get_output(0), trt.MatrixOperation.NONE)
    kv_old = network.add_shuffle(kv_old_3d.get_output(0))
    kv_old.reshape_dims = (num_heads, head_dim)

    v_minus_old = network.add_elementwise(
        v_t, kv_old.get_output(0), trt.ElementWiseOperation.SUB)
    v_delta = network.add_elementwise(
        v_minus_old.get_output(0), beta_reshaped.get_output(0),
        trt.ElementWiseOperation.PROD)

    k_col2 = network.add_shuffle(k_t)
    k_col2.reshape_dims = (num_heads, head_dim, 1)
    v_delta_row = network.add_shuffle(v_delta.get_output(0))
    v_delta_row.reshape_dims = (num_heads, 1, head_dim)
    outer_prod = network.add_matrix_multiply(
        k_col2.get_output(0), trt.MatrixOperation.NONE,
        v_delta_row.get_output(0), trt.MatrixOperation.NONE)

    new_state = network.add_elementwise(
        decayed_state.get_output(0), outer_prod.get_output(0),
        trt.ElementWiseOperation.SUM)
    present_ssm = new_state.get_output(0)

    # ===== 9e. output = state_new^T @ (q * scale) =====
    q_scale = graph_ops.add_constant(
        network, (1, 1),
        np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
    q_scaled = network.add_elementwise(
        q_expanded, q_scale, trt.ElementWiseOperation.PROD)
    q_col = network.add_shuffle(q_scaled.get_output(0))
    q_col.reshape_dims = (num_heads, head_dim, 1)
    output_3d = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.TRANSPOSE,
        q_col.get_output(0), trt.MatrixOperation.NONE)
    output_flat = network.add_shuffle(output_3d.get_output(0))
    output_flat.reshape_dims = (1, d_inner)

    # ===== 10. Per-head Gated RMSNorm + silu(z) =====
    deltanet_norm_w = weights[f"{prefix}.deltanet_norm"]
    eps_small = graph_ops.add_constant(
        network, (1, 1), np.array([1e-6], dtype=np.float32))

    output_heads = network.add_shuffle(output_flat.get_output(0))
    output_heads.reshape_dims = (num_heads, head_dim)

    sq = network.add_elementwise(
        output_heads.get_output(0), output_heads.get_output(0),
        trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_small, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        output_heads.get_output(0), recip.get_output(0),
        trt.ElementWiseOperation.PROD)

    norm_flat = network.add_shuffle(normalized.get_output(0))
    norm_flat.reshape_dims = (1, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), deltanet_norm_w)
    normed_output = network.add_elementwise(
        norm_flat.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    z_activated = graph_ops.add_activation(network, z, "silu")
    gated = network.add_elementwise(
        normed_output.get_output(0), z_activated,
        trt.ElementWiseOperation.PROD)

    # ===== 11. Row-parallel out_proj + all-reduce + residual =====
    out = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), d_inner, hidden_size,
        weights[f"{prefix}.deltanet_out_proj"])
    out = add_all_reduce_sum(network, out, tp_size)

    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # ===== 12. Post-attention norm + SwiGLU MLP + residual =====
    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor)
    mlp_out = _add_swiglu_mlp_tp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=int(weights["_mlp_size"]),
        tp_size=tp_size,
    )
    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


# ---------------------------------------------------------------------------
# TP-aware full attention layer (partial RoPE + output gating)
# ---------------------------------------------------------------------------

def _add_full_attention_tp_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    """Qwen3.5 full attention with QK-norm, partial RoPE, and output gate."""
    attention_window = max_cache_length + 1

    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], eps_tensor)

    # Column-parallel QKV
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attn_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"])

    # Per-head QK norm (gamma already pre-sharded to local heads)
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor)

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, rotary_embedding_dim)

    present_k = k
    present_v = v

    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=1, kv_seq=attention_window,
        mask=mask_4d)

    # Output gating BEFORE o_proj (HF order).
    gate_attn_w = weights.get(f"{prefix}.w_gate_attn")
    attn_out = context_flat
    if gate_attn_w is not None:
        gate = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attn_size, gate_attn_w)
        gate_sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        gated = network.add_elementwise(
            attn_out, gate_sigmoid.get_output(0),
            trt.ElementWiseOperation.PROD)
        attn_out = gated.get_output(0)

    # Row-parallel o_proj + all-reduce
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_out, attn_size, hidden_size,
        weights[f"{prefix}.w_o"])
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)

    residual = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    post_normed = graph_ops.add_rms_norm(
        network, hidden_after_attn, hidden_size,
        weights[f"{prefix}.post_attn_norm"], eps_tensor)
    mlp_out = _add_swiglu_mlp_tp(
        network, post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        tp_size=tp_size,
    )
    mlp_residual = network.add_elementwise(
        hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


# ---------------------------------------------------------------------------
# TP-aware SwiGLU MLP
# ---------------------------------------------------------------------------

def _add_swiglu_mlp_tp(
    network: trt.INetworkDefinition,
    hidden_in: trt.ITensor,
    *,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    tp_size: int,
) -> trt.ITensor:
    """SwiGLU MLP without the residual: silu(gate(x)) * up(x) -> down -> AR."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, hidden_in, hidden_size, mlp_size,
        weights[f"{prefix}.w_gate"])
    up = graph_ops.add_matmul_rhs_constant(
        network, hidden_in, hidden_size, mlp_size,
        weights[f"{prefix}.w_up"])
    gate_activated = graph_ops.add_activation(network, gate, "silu")
    gated = network.add_elementwise(
        gate_activated, up, trt.ElementWiseOperation.PROD)
    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), mlp_size, hidden_size,
        weights[f"{prefix}.w_down"])
    return add_all_reduce_sum(network, down, tp_size)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_qwen35_tp_engine(
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
    """Build a per-rank Qwen3.5 hybrid TRT engine for tensor parallel."""
    del precision, quant_ctx
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_qwen35_tp_engine requires tensor_parallel mode with tp_size > 1")

    rank_weights = shard_qwen35_weights(config, weights, parallel=parallel)

    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    layer_types: list[str] = rank_weights["_layer_types"]

    d_inner = int(rank_weights["_d_inner"])
    d_conv = int(rank_weights["_d_conv"])
    conv_dim = int(rank_weights["_conv_dim"])
    dn_heads = int(rank_weights["_deltanet_num_heads"])
    dn_kv_heads = int(rank_weights["_deltanet_num_kv_heads"])
    dn_head_dim = int(rank_weights["_deltanet_head_dim"])
    num_mamba = int(rank_weights["_num_mamba_layers"])
    num_attn = int(rank_weights["_num_attention_layers"])
    attn_size = int(rank_weights["_attn_size"])
    mlp_size = int(rank_weights["_mlp_size"])
    partial_rotary_factor: float = float(rank_weights["_partial_rotary_factor"])
    rope_theta: float = float(rank_weights["_rope_theta"])

    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(config.num_key_value_heads) // parallel.tp_size
    head_dim = int(config.head_dim)
    kv_attention_size = num_kv_heads * head_dim
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
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
            trt.float32, (dn_heads, dn_head_dim, dn_head_dim)))

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

    rotary_embedding_dim = int(head_dim * partial_rotary_factor)

    cos_half = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, rope_theta,
        cosine=True, partial_rotary_factor=partial_rotary_factor)
    sin_half = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, rope_theta,
        cosine=False, partial_rotary_factor=partial_rotary_factor)
    cos_half_tensor = graph_ops.add_constant(
        network, cos_half.shape, cos_half)
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half.shape, sin_half)

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
        if lt == "deltanet":
            result = _add_deltanet_tp_layer(
                network=network,
                hidden=hidden_state,
                conv_state_in=conv_state_inputs[mamba_counter],
                ssm_state_in=ssm_state_inputs[mamba_counter],
                eps_tensor=eps_tensor,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                d_inner=d_inner,
                d_conv=d_conv,
                conv_dim=conv_dim,
                num_heads=dn_heads,
                num_kv_heads=dn_kv_heads,
                head_dim=dn_head_dim,
                tp_size=parallel.tp_size,
            )
            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])
            mamba_counter += 1
        elif lt == "attention":
            result = _add_full_attention_tp_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[attn_counter],
                cache_v=cache_v_inputs[attn_counter],
                attention_mask=attention_mask,
                position_id=position_id,
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                eps_tensor=eps_tensor,
                weights=rank_weights,
                prefix=prefix,
                hidden_size=hidden,
                attn_size=attn_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_embedding_dim=rotary_embedding_dim,
                max_cache_length=max_cache_length,
                mlp_size=mlp_size,
                tp_size=parallel.tp_size,
            )
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1

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
            "[trtmc build] Qwen3.5 TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_dn_heads={dn_heads}, local_attn_heads={num_heads}, "
            f"local_mlp={mlp_size})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3.5 TP engine build failed")
    return bytes(plan)

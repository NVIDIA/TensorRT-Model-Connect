# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek-V2 builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from . import moe_routing
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .standard_decoder_builder import _apply_norm, _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _is_moe_layer(weights: "WeightDict", layer_idx: int) -> bool:
    first = int(weights["_first_k_dense_replace"])
    freq = int(weights["_moe_layer_freq"])
    return layer_idx >= first and (layer_idx - first) % freq == 0


def _validate_deepseek_v2_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError(
            "DeepSeek-V2 tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "DeepSeek-V2 tensor parallel requires num_attention_heads "
            f"divisible by tp_size ({config.num_attention_heads} vs {tp})")

    checks = {
        "moe_intermediate_size": int(weights["_moe_intermediate_size"]),
        "shared_intermediate_size": int(weights["_shared_intermediate_size"]),
        "dense_intermediate_size": int(config.intermediate_size),
    }
    for name, value in checks.items():
        if value % tp != 0:
            raise ValueError(
                f"DeepSeek-V2 tensor parallel requires {name} divisible by "
                f"tp_size ({value} vs {tp})")


def shard_deepseek_v2_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local DeepSeek-V2 weights for the TP builder."""
    _validate_deepseek_v2_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_q_b", ".w_kv_b", ".w_gate", ".w_up")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = int(weights["_attention_size"]) // parallel.tp_size
    out["_moe_intermediate_size"] = (
        int(weights["_moe_intermediate_size"]) // parallel.tp_size)
    out["_shared_intermediate_size"] = (
        int(weights["_shared_intermediate_size"]) // parallel.tp_size)
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_swiglu_local(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down)


def _add_swiglu_tp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    tp_size: int,
) -> trt.ITensor:
    local = _add_swiglu_local(
        network, inp, hidden_size, intermediate_size, w_gate, w_up, w_down)
    return add_all_reduce_sum(network, local, tp_size)


def _add_deepseek_tp_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    moe_intermediate: int,
    num_experts_per_tok: int,
    shared_intermediate: int,
    tp_size: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "softmax",
    topk_method: str = "greedy",
    n_group: int = 1,
    topk_group: int = 1,
) -> trt.ITensor:
    top_indices, scaled_weights = moe_routing.add_router(
        network,
        inp,
        weights[f"{prefix}.router"],
        hidden_size=hidden_size,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        scoring_func=scoring_func,
        topk_method=topk_method,
        correction_bias=weights.get(f"{prefix}.router_score_bias"),
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )

    expert_outputs = []
    for expert_idx in range(n_routed_experts):
        expert_outputs.append(_add_swiglu_local(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{expert_idx}.w_gate"],
            weights[f"{prefix}.expert.{expert_idx}.w_up"],
            weights[f"{prefix}.expert.{expert_idx}.w_down"],
        ))

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)

    routed_local = None
    for top_idx in range(num_experts_per_tok):
        idx_slice = network.add_slice(
            top_indices, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)
        w_slice = network.add_slice(
            scaled_weights, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        expert_out = network.add_gather(stacked_out, idx_flat.get_output(0), 0)
        selected_weight = w_slice.get_output(0)
        if selected_weight.dtype != expert_out.get_output(0).dtype:
            selected_weight = network.add_cast(
                selected_weight,
                expert_out.get_output(0).dtype,
            ).get_output(0)
        scaled = network.add_elementwise(
            expert_out.get_output(0), selected_weight,
            trt.ElementWiseOperation.PROD)
        if routed_local is None:
            routed_local = scaled.get_output(0)
        else:
            summed = network.add_elementwise(
                routed_local, scaled.get_output(0), trt.ElementWiseOperation.SUM)
            routed_local = summed.get_output(0)

    shared_local = _add_swiglu_local(
        network, inp, hidden_size, shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
    )
    local_total = network.add_elementwise(
        routed_local, shared_local, trt.ElementWiseOperation.SUM)
    return add_all_reduce_sum(network, local_total.get_output(0), tp_size)


def _add_mla_attention_tp(
    *,
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    attention_window = max_cache_length + 1
    k_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_total = num_heads * k_head_dim
    rope_total = num_heads * qk_rope_head_dim

    if q_lora_rank is not None and q_lora_rank > 0:
        q_compressed = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_lora_rank,
            weights[f"{prefix}.w_q_a"])
        q_compressed = graph_ops.add_rms_norm(
            network, q_compressed, q_lora_rank,
            weights[f"{prefix}.q_a_norm"], eps_tensor)
        q = graph_ops.add_matmul_rhs_constant(
            network, q_compressed, q_lora_rank, q_total,
            weights[f"{prefix}.w_q_b"])
    else:
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_total,
            weights[f"{prefix}.w_q"])

    q_reshaped = network.add_shuffle(q)
    q_reshaped.reshape_dims = (num_heads, k_head_dim)
    q_nope = network.add_slice(
        q_reshaped.get_output(0), start=(0, 0),
        shape=(num_heads, qk_nope_head_dim), stride=(1, 1)).get_output(0)
    q_rope_slice = network.add_slice(
        q_reshaped.get_output(0), start=(0, qk_nope_head_dim),
        shape=(num_heads, qk_rope_head_dim), stride=(1, 1))
    q_rope_flat = network.add_shuffle(q_rope_slice.get_output(0))
    q_rope_flat.reshape_dims = (1, rope_total)
    q_rope_roped = graph_ops.add_apply_rope_native(
        network, q_rope_flat.get_output(0), num_heads, qk_rope_head_dim,
        cos_half_tensor, sin_half_tensor, position_id,
        qk_rope_head_dim, interleaved=True)
    q_rope_heads = network.add_shuffle(q_rope_roped)
    q_rope_heads.reshape_dims = (num_heads, qk_rope_head_dim)
    q_full_cat = network.add_concatenation([q_nope, q_rope_heads.get_output(0)])
    q_full_cat.axis = 1

    kv_a_dim = kv_lora_rank + qk_rope_head_dim
    c_kv = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_a_dim,
        weights[f"{prefix}.w_kv_a"])
    c_kv_latent = network.add_slice(
        c_kv, start=(0, 0), shape=(1, kv_lora_rank), stride=(1, 1)
    ).get_output(0)
    k_rope_pass = network.add_slice(
        c_kv, start=(0, kv_lora_rank), shape=(1, qk_rope_head_dim),
        stride=(1, 1)).get_output(0)
    c_kv_normed = graph_ops.add_rms_norm(
        network, c_kv_latent, kv_lora_rank,
        weights[f"{prefix}.kv_a_norm"], eps_tensor)

    kv_b_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
    kv_expanded = graph_ops.add_matmul_rhs_constant(
        network, c_kv_normed, kv_lora_rank, kv_b_out_dim,
        weights[f"{prefix}.w_kv_b"])
    kv_per_head = network.add_shuffle(kv_expanded)
    kv_per_head.reshape_dims = (num_heads, qk_nope_head_dim + v_head_dim)
    k_nope = network.add_slice(
        kv_per_head.get_output(0), start=(0, 0),
        shape=(num_heads, qk_nope_head_dim), stride=(1, 1)).get_output(0)
    v_heads = network.add_slice(
        kv_per_head.get_output(0), start=(0, qk_nope_head_dim),
        shape=(num_heads, v_head_dim), stride=(1, 1)).get_output(0)

    k_rope_roped = graph_ops.add_apply_rope_native(
        network, k_rope_pass, 1, qk_rope_head_dim,
        cos_half_tensor, sin_half_tensor, position_id,
        qk_rope_head_dim, interleaved=True)
    k_rope_copies = [k_rope_roped for _ in range(num_heads)]
    k_rope_broadcast = network.add_concatenation(k_rope_copies)
    k_rope_broadcast.axis = 0
    k_full_cat = network.add_concatenation([k_nope, k_rope_broadcast.get_output(0)])
    k_full_cat.axis = 1

    pad_size = k_head_dim - v_head_dim
    if pad_size > 0:
        zero_pad = graph_ops.add_constant(
            network, (num_heads, pad_size),
            np.zeros((num_heads, pad_size), dtype=np.float32))
        v_padded_cat = network.add_concatenation([v_heads, zero_pad])
        v_padded_cat.axis = 1
        v_padded = v_padded_cat.get_output(0)
    else:
        v_padded = v_heads

    k_flat = network.add_shuffle(k_full_cat.get_output(0))
    k_flat.reshape_dims = (1, attention_size)
    v_flat = network.add_shuffle(v_padded)
    v_flat.reshape_dims = (1, attention_size)
    present_k = k_flat.get_output(0)
    present_v = v_flat.get_output(0)

    all_k = network.add_concatenation([cache_k, present_k])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, present_v])
    all_v.axis = 0
    q_flat = network.add_shuffle(q_full_cat.get_output(0))
    q_flat.reshape_dims = (1, attention_size)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    attn_context = graph_ops.add_attention_from_rows(
        network, q_flat.get_output(0), all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=k_head_dim, q_seq=1,
        kv_seq=attention_window, mask=mask_4d, scale=attn_scale)

    if pad_size > 0:
        context_heads = network.add_shuffle(attn_context)
        context_heads.reshape_dims = (num_heads, k_head_dim)
        context_sliced = network.add_slice(
            context_heads.get_output(0), start=(0, 0),
            shape=(num_heads, v_head_dim), stride=(1, 1))
        context_flat = network.add_shuffle(context_sliced.get_output(0))
        context_flat.reshape_dims = (1, num_heads * v_head_dim)
        attn_context = context_flat.get_output(0)

    v_total = num_heads * v_head_dim
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, attn_context, v_total, hidden_size,
        weights[f"{prefix}.w_o"])
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)

    return {
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def _add_decoder_layer_tp(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale: float,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    q_lora_rank,
    attention_size: int,
    max_cache_length: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    tp_size: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "softmax",
    topk_method: str = "greedy",
    n_group: int = 1,
    topk_group: int = 1,
) -> dict[str, trt.ITensor]:
    norm1 = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], None, eps_tensor, "rmsnorm")
    attn = _add_mla_attention_tp(
        network=network,
        normed=norm1,
        cache_k=cache_k,
        cache_v=cache_v,
        attention_mask=attention_mask,
        position_id=position_id,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        attn_scale=attn_scale,
        eps_tensor=eps_tensor,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attention_size=attention_size,
        max_cache_length=max_cache_length,
        tp_size=tp_size,
    )
    residual1 = network.add_elementwise(
        hidden, attn["attn_out"], trt.ElementWiseOperation.SUM)
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"], None, eps_tensor, "rmsnorm")

    if is_moe_layer:
        mlp_out = _add_deepseek_tp_moe_block(
            network, norm2, weights, prefix,
            hidden_size, n_routed_experts, moe_intermediate,
            num_experts_per_tok, shared_intermediate, tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func=scoring_func,
            topk_method=topk_method,
            n_group=n_group,
            topk_group=topk_group)
    else:
        mlp_out = _add_swiglu_tp(
            network, norm2, hidden_size, dense_intermediate,
            weights[f"{prefix}.w_gate"],
            weights[f"{prefix}.w_up"],
            weights[f"{prefix}.w_down"],
            tp_size,
        )

    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


def _make_rope_tables(
    config: "ModelConfig",
    attention_window: int,
    qk_rope_head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    rope_scaling = config.raw.get("rope_scaling")
    if rope_scaling and rope_scaling.get("type") == "yarn":
        yarn_kwargs = dict(
            scaling_factor=rope_scaling["factor"],
            original_max_position_embeddings=rope_scaling[
                "original_max_position_embeddings"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        return (
            graph_ops.make_yarn_rope_table_half_dim(
                attention_window, qk_rope_head_dim, config.rope_theta,
                True, **yarn_kwargs, interleaved=True),
            graph_ops.make_yarn_rope_table_half_dim(
                attention_window, qk_rope_head_dim, config.rope_theta,
                False, **yarn_kwargs, interleaved=True),
        )
    return (
        graph_ops.make_rope_table_half_dim(
            attention_window, qk_rope_head_dim,
            config.rope_theta, True, interleaved=True),
        graph_ops.make_rope_table_half_dim(
            attention_window, qk_rope_head_dim,
            config.rope_theta, False, interleaved=True),
    )


def build_deepseek_v2_tp_engine(
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
            "build_deepseek_v2_tp_engine requires tensor_parallel mode with "
            "tp_size > 1")

    rank_weights = shard_deepseek_v2_weights(config, weights, parallel=parallel)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads) // parallel.tp_size
    qk_nope_head_dim = int(rank_weights["_qk_nope_head_dim"])
    qk_rope_head_dim = int(rank_weights["_qk_rope_head_dim"])
    v_head_dim = int(rank_weights["_v_head_dim"])
    kv_lora_rank = int(rank_weights["_kv_lora_rank"])
    q_lora_rank = rank_weights["_q_lora_rank"]
    n_routed_experts = int(rank_weights["_n_routed_experts"])
    num_experts_per_tok = int(rank_weights["_num_experts_per_tok"])
    moe_intermediate = int(rank_weights["_moe_intermediate_size"])
    shared_intermediate = int(rank_weights["_shared_intermediate_size"])
    dense_intermediate = int(config.intermediate_size) // parallel.tp_size
    norm_topk_prob = bool(rank_weights["_norm_topk_prob"])
    routed_scaling_factor = float(rank_weights["_routed_scaling_factor"])
    scoring_func = str(rank_weights["_scoring_func"])
    topk_method = str(rank_weights["_topk_method"])
    n_group = int(rank_weights["_n_group"])
    topk_group = int(rank_weights["_topk_group"])
    moe_routing.validate_router_contract(
        scoring_func=scoring_func,
        topk_method=topk_method,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
    )

    k_head_dim = qk_nope_head_dim + qk_rope_head_dim
    attention_size = num_heads * k_head_dim
    attention_window = max_cache_length + 1
    attn_scale = 1.0 / np.sqrt(max(k_head_dim, 1))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))
    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            trt.float32, (max_cache_length, attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            trt.float32, (max_cache_length, attention_size)))

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["embedding"])
    cos_half_np, sin_half_np = _make_rope_tables(
        config, attention_window, qk_rope_head_dim)
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_decoder_layer_tp(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            attn_scale=attn_scale,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            attention_size=attention_size,
            max_cache_length=max_cache_length,
            is_moe_layer=_is_moe_layer(rank_weights, layer_idx),
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate=moe_intermediate,
            shared_intermediate=shared_intermediate,
            dense_intermediate=dense_intermediate,
            tp_size=parallel.tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            scoring_func=scoring_func,
            topk_method=topk_method,
            n_group=n_group,
            topk_group=topk_group,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(
                network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm, None,
            eps_tensor, "rmsnorm")

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_out"])
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, np.zeros(vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        present_k_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_k", layer_idx)
        present_v_outputs[layer_idx].name = graph_ops.layer_tensor_name(
            "present_v", layer_idx)
        network.mark_output(present_k_outputs[layer_idx])
        network.mark_output(present_v_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] DeepSeek-V2 TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_heads={num_heads}, local_attn={attention_size}, "
            f"local_moe={moe_intermediate})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT DeepSeek-V2 TP engine build failed")
    return bytes(plan)

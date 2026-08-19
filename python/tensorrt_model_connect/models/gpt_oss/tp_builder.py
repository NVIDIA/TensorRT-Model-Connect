# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel GPT-OSS MoE builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_blocks, graph_ops
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .default_decoder import _apply_norm, _mark_debug_output
from .utils import make_rope_half_tables

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_gpt_oss_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("GPT-OSS tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "GPT-OSS tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})")
    if int(config.num_key_value_heads) % tp != 0:
        raise ValueError(
            "GPT-OSS tensor parallel requires num_key_value_heads divisible by "
            f"tp_size ({config.num_key_value_heads} vs {tp})")

    attention_size = int(weights.get("_attention_size", config.attention_size))
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights,
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=attention_size // int(config.num_attention_heads),
    )
    checks = {
        "attention_size": attention_size,
        "kv_attention_size": kv_attention_size,
        "moe_intermediate_size": int(weights["_moe_intermediate_size"]),
    }
    for name, value in checks.items():
        if value % tp != 0:
            raise ValueError(
                f"GPT-OSS tensor parallel requires {name} divisible by tp_size "
                f"({value} vs {tp})")


def shard_gpt_oss_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local GPT-OSS weights for the TP builder."""
    _validate_gpt_oss_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".w_gate", ".w_up")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".q_bias", ".k_bias", ".v_bias", ".gate_bias", ".up_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith(".sinks"):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = int(weights["_attention_size"]) // parallel.tp_size
    out["_moe_intermediate_size"] = (
        int(weights["_moe_intermediate_size"]) // parallel.tp_size)
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_gpt_oss_expert_local(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    gate_bias: np.ndarray | None = None,
    up_bias: np.ndarray | None = None,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    if gate_bias is not None:
        gate = graph_ops.add_bias_sum(network, gate, intermediate_size, gate_bias)

    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)
    if up_bias is not None:
        up = graph_ops.add_bias_sum(network, up, intermediate_size, up_bias)

    limit_const = graph_ops.add_constant(
        network, (1, 1), np.array([limit], dtype=np.float32))
    neg_limit_const = graph_ops.add_constant(
        network, (1, 1), np.array([-limit], dtype=np.float32))
    gate = network.add_elementwise(
        gate, limit_const, trt.ElementWiseOperation.MIN).get_output(0)
    up = network.add_elementwise(
        up, limit_const, trt.ElementWiseOperation.MIN).get_output(0)
    up = network.add_elementwise(
        up, neg_limit_const, trt.ElementWiseOperation.MAX).get_output(0)

    alpha_const = graph_ops.add_constant(
        network, (1, 1), np.array([alpha], dtype=np.float32))
    gate_scaled = network.add_elementwise(
        gate, alpha_const, trt.ElementWiseOperation.PROD).get_output(0)
    sigmoid = network.add_activation(
        gate_scaled, trt.ActivationType.SIGMOID).get_output(0)
    glu = network.add_elementwise(
        gate, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)

    one_const = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    up_plus_one = network.add_elementwise(
        up, one_const, trt.ElementWiseOperation.SUM).get_output(0)
    gated = network.add_elementwise(
        up_plus_one, glu, trt.ElementWiseOperation.PROD).get_output(0)

    return graph_ops.add_matmul_rhs_constant(
        network, gated, intermediate_size, hidden_size, w_down)


def _add_weighted_down_bias(
    network: trt.INetworkDefinition,
    weights: "WeightDict",
    prefix: str,
    top_indices: trt.ITensor,
    routing_weights: trt.ITensor,
    *,
    num_experts: int,
    hidden_size: int,
    top_k: int,
) -> trt.ITensor | None:
    bias_arrays = [
        weights.get(f"{prefix}.expert.{expert_idx}.down_bias")
        for expert_idx in range(num_experts)
    ]
    if not all(isinstance(bias, np.ndarray) for bias in bias_arrays):
        return None

    bias_tensors = [
        graph_ops.add_constant(
            network, (1, hidden_size), bias.reshape(1, hidden_size))
        for bias in bias_arrays
    ]
    stacked = network.add_concatenation(bias_tensors)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)

    result = None
    for top_idx in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)
        w_slice = network.add_slice(
            routing_weights, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        bias = network.add_gather(stacked_out, idx_flat.get_output(0), 0)
        scaled = network.add_elementwise(
            bias.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)
        if result is None:
            result = scaled.get_output(0)
        else:
            summed = network.add_elementwise(
                result, scaled.get_output(0), trt.ElementWiseOperation.SUM)
            result = summed.get_output(0)
    return result


def _add_gpt_oss_tp_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    tp_size: int,
    top_k: int = 4,
) -> trt.ITensor:
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts, weights[f"{prefix}.router"])
    router_bias = weights.get(f"{prefix}.router_bias")
    if router_bias is not None:
        router_logits = graph_ops.add_bias_sum(
            network, router_logits, num_experts, router_bias)

    topk = network.add_topk(
        router_logits, trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)
    sm = network.add_softmax(top_values)
    sm.axes = 1 << 1
    routing_weights = sm.get_output(0)

    expert_outputs = []
    for expert_idx in range(num_experts):
        expert_outputs.append(_add_gpt_oss_expert_local(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{expert_idx}.w_gate"],
            weights[f"{prefix}.expert.{expert_idx}.w_up"],
            weights[f"{prefix}.expert.{expert_idx}.w_down"],
            weights.get(f"{prefix}.expert.{expert_idx}.gate_bias"),
            weights.get(f"{prefix}.expert.{expert_idx}.up_bias"),
        ))

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)

    local_result = None
    for top_idx in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)
        w_slice = network.add_slice(
            routing_weights, start=(0, top_idx), shape=(1, 1), stride=(1, 1))
        expert_out = network.add_gather(stacked_out, idx_flat.get_output(0), 0)
        scaled = network.add_elementwise(
            expert_out.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)
        if local_result is None:
            local_result = scaled.get_output(0)
        else:
            summed = network.add_elementwise(
                local_result, scaled.get_output(0), trt.ElementWiseOperation.SUM)
            local_result = summed.get_output(0)

    reduced = add_all_reduce_sum(network, local_result, tp_size)
    bias = _add_weighted_down_bias(
        network, weights, prefix, top_indices, routing_weights,
        num_experts=num_experts, hidden_size=hidden_size, top_k=top_k)
    if bias is None:
        return reduced
    with_bias = network.add_elementwise(
        reduced, bias, trt.ElementWiseOperation.SUM)
    return with_bias.get_output(0)


def _add_gpt_oss_tp_attention(
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    *,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale_tensor: trt.ITensor,
    tp_size: int,
) -> dict[str, trt.ITensor]:
    attention_window = max_cache_length + 1
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"])

    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(network, q, attention_size, q_bias)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(network, k, kv_attention_size, k_bias)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(network, v, kv_attention_size, v_bias)

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)

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

    sinks = weights.get(f"{prefix}.sinks")
    if sinks is None:
        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
        context_tensor = graph_ops.add_attention_from_rows(
            network, q, all_k.get_output(0), all_v.get_output(0),
            num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
            q_seq=1, kv_seq=attention_window, mask=mask_4d)
    else:
        q_heads = network.add_shuffle(q)
        q_heads.reshape_dims = (num_heads, 1, head_dim)
        k_heads = network.add_shuffle(all_k.get_output(0))
        k_heads.reshape_dims = (attention_window, num_kv_heads, head_dim)
        v_heads = network.add_shuffle(all_v.get_output(0))
        v_heads.reshape_dims = (attention_window, num_kv_heads, head_dim)
        k_heads.second_transpose = trt.Permutation([1, 0, 2])
        v_heads.second_transpose = trt.Permutation([1, 0, 2])
        k_heads_t = k_heads.get_output(0)
        v_heads_t = v_heads.get_output(0)
        if num_kv_heads != num_heads:
            group_size = num_heads // num_kv_heads
            k_slices = []
            v_slices = []
            for kvh in range(num_kv_heads):
                ks = network.add_slice(
                    k_heads_t, start=(kvh, 0, 0),
                    shape=(1, attention_window, head_dim), stride=(1, 1, 1))
                vs = network.add_slice(
                    v_heads_t, start=(kvh, 0, 0),
                    shape=(1, attention_window, head_dim), stride=(1, 1, 1))
                k_slices.extend([ks.get_output(0)] * group_size)
                v_slices.extend([vs.get_output(0)] * group_size)
            k_expand = network.add_concatenation(k_slices)
            k_expand.axis = 0
            v_expand = network.add_concatenation(v_slices)
            v_expand.axis = 0
            k_heads_t = k_expand.get_output(0)
            v_heads_t = v_expand.get_output(0)

        score = network.add_matrix_multiply(
            q_heads.get_output(0), trt.MatrixOperation.NONE,
            k_heads_t, trt.MatrixOperation.TRANSPOSE)
        scaled = network.add_elementwise(
            score.get_output(0), attn_scale_tensor,
            trt.ElementWiseOperation.PROD)
        mask3d = network.add_shuffle(attention_mask)
        mask3d.reshape_dims = (1, 1, attention_window)
        masked = network.add_elementwise(
            scaled.get_output(0), mask3d.get_output(0),
            trt.ElementWiseOperation.SUM)

        sinks_const = graph_ops.add_constant(
            network, (num_heads, 1, 1), sinks.reshape(num_heads, 1, 1))
        combined = network.add_concatenation([masked.get_output(0), sinks_const])
        combined.axis = 2
        max_val = network.add_reduce(
            combined.get_output(0), trt.ReduceOperation.MAX, 1 << 2,
            keep_dims=True)
        stable = network.add_elementwise(
            combined.get_output(0), max_val.get_output(0),
            trt.ElementWiseOperation.SUB)
        softmax = network.add_softmax(stable.get_output(0))
        softmax.axes = 1 << 2
        scores = network.add_slice(
            softmax.get_output(0),
            start=(0, 0, 0), shape=(num_heads, 1, attention_window),
            stride=(1, 1, 1))
        context_heads = network.add_matrix_multiply(
            scores.get_output(0), trt.MatrixOperation.NONE,
            v_heads_t, trt.MatrixOperation.NONE)
        context_flat = network.add_shuffle(context_heads.get_output(0))
        context_flat.reshape_dims = (1, attention_size)
        context_tensor = context_flat.get_output(0)

    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_tensor, attention_size, hidden_size,
        weights[f"{prefix}.w_o"])
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, o_bias)

    return {
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def _add_gpt_oss_tp_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    num_experts: int,
    moe_intermediate: int,
    tp_size: int,
    top_k: int = 4,
) -> dict[str, trt.ITensor]:
    normed = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], None, eps_tensor, "rmsnorm")

    attn = _add_gpt_oss_tp_attention(
        network, normed, cache_k, cache_v, attention_mask, position_id,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        max_cache_length=max_cache_length,
        cos_half_tensor=cos_half_tensor, sin_half_tensor=sin_half_tensor,
        attn_scale_tensor=attn_scale_tensor, tp_size=tp_size)

    residual1 = network.add_elementwise(
        hidden, attn["attn_out"], trt.ElementWiseOperation.SUM)
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"], None, eps_tensor, "rmsnorm")
    moe_out = _add_gpt_oss_tp_moe_block(
        network, norm2, weights, prefix,
        hidden_size, num_experts, moe_intermediate, tp_size, top_k)
    residual2 = network.add_elementwise(
        residual1.get_output(0), moe_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


def _make_rope_tables(
    config: "ModelConfig",
    attention_window: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    return make_rope_half_tables(config, attention_window, head_dim)


def build_gpt_oss_tp_engine(
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
            "build_gpt_oss_tp_engine requires tensor_parallel mode with "
            "tp_size > 1")

    rank_weights = shard_gpt_oss_weights(config, weights, parallel=parallel)
    attention_size = int(rank_weights["_attention_size"])
    num_experts = int(rank_weights["_num_experts"])
    moe_intermediate = int(rank_weights["_moe_intermediate_size"])
    top_k = int(rank_weights["_num_experts_per_tok"])
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(config.num_key_value_heads) // parallel.tp_size
    head_dim = attention_size // num_heads
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
    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            trt.float32, (max_cache_length, kv_attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            trt.float32, (max_cache_length, kv_attention_size)))

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["embedding"])
    cos_half_np, sin_half_np = _make_rope_tables(
        config, attention_window, head_dim)
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))
    attn_scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([1.0 / np.sqrt(max(head_dim, 1))], dtype=np.float32))

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # Sliding-attention layers attend only within the configured window;
    # feed them a mask that additionally hides out-of-window cache columns.
    layer_types = list(config.raw.get("layer_types") or [])
    sliding_window = int(config.raw.get("sliding_window") or 0)
    sliding_attention_mask = None
    if sliding_window > 0 and "sliding_attention" in layer_types:
        sliding_attention_mask = graph_ops.add_sliding_window_mask(
            network, attention_mask, position_id,
            attention_window, sliding_window)

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_type = (
            layer_types[layer_idx] if layer_idx < len(layer_types)
            else "full_attention")
        layer_mask = (
            sliding_attention_mask
            if (sliding_attention_mask is not None
                and layer_type == "sliding_attention")
            else attention_mask)
        result = _add_gpt_oss_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=layer_mask,
            position_id=position_id,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            attn_scale_tensor=attn_scale_tensor,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            num_experts=num_experts,
            moe_intermediate=moe_intermediate,
            tp_size=parallel.tp_size,
            top_k=top_k,
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
    lm_bias = rank_weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, vocab, lm_bias)
    else:
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
            "[trtmc build] GPT-OSS TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_attn={attention_size}, local_moe={moe_intermediate}, "
            f"top_k={top_k})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT GPT-OSS TP engine build failed")
    return bytes(plan)

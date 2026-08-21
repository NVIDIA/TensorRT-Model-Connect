# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel OLMo-2 decoder builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from . import graph_blocks
from . import graph_ops
from .checkpoint_mapper import WeightDict
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

if TYPE_CHECKING:
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def shard_olmo2_weights(
    config: "ModelConfig",
    weights: WeightDict,
    *,
    parallel: "ParallelConfig",
) -> WeightDict:
    """Return rank-local OLMo-2 decoder weights for TP builds."""
    if "_kv_attention_size" in weights:
        return shard_standard_decoder_weights(config, weights, parallel)

    with_metadata = WeightDict(weights)
    first_k = with_metadata.get("layer.0.w_k")
    if not isinstance(first_k, np.ndarray):
        raise ValueError("OLMo2 tensor parallel requires layer.0.w_k")
    with_metadata["_kv_attention_size"] = int(first_k.shape[1])
    return shard_standard_decoder_weights(config, with_metadata, parallel)


def _add_distributed_rms_norm(
    network,
    inp,
    *,
    local_hidden_size: int,
    global_hidden_size: int,
    gamma: np.ndarray,
    eps_tensor,
    tp_size: int,
):
    """RMSNorm over a tensor-parallel-sharded hidden dimension."""
    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    local_sum = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    global_sum = add_all_reduce_sum(
        network, local_sum.get_output(0), tp_size)
    inv_hidden = graph_ops.add_constant(
        network, (1, 1),
        np.array([1.0 / float(global_hidden_size)], dtype=np.float32))
    mean = network.add_elementwise(
        global_sum, inv_hidden, trt.ElementWiseOperation.PROD)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = graph_ops.add_constant(
        network, (1, local_hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)
    return scaled.get_output(0)


def build_olmo2_tp_engine(
    config: "ModelConfig",
    weights: WeightDict,
    max_cache_length: int,
    *,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local tensor-parallel OLMo-2 decoder engine.

    This mirrors the single-device OLMo-2 builder. The only graph changes are
    rank-local Q/K/V and MLP inner projections plus all-reduce joins after the
    row-parallel attention output and MLP down projections.
    """
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_olmo2_tp_engine requires tensor_parallel mode with "
            "tp_size > 1")
    weights = shard_olmo2_weights(config, weights, parallel=parallel)

    from tensorrt_model_connect import trt_compat
    trt = trt_compat.get_trt()

    attention_size: int = weights.get(
        "_attention_size", config.attention_size // parallel.tp_size)
    mlp_size: int = weights.get(
        "_mlp_size", config.intermediate_size // parallel.tp_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    global_attention_size = attention_size * parallel.tp_size
    global_kv_attention_size = kv_attention_size * parallel.tp_size
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # Inputs
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            trt.float32, (max_cache_length, kv_attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            trt.float32, (max_cache_length, kv_attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    # Constants
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"])

    cos_table_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True)
    sin_table_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False)

    cos_tensor = graph_ops.add_constant(
        network, cos_table_np.shape, cos_table_np)
    sin_tensor = graph_ops.add_constant(
        network, sin_table_np.shape, sin_table_np)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

    # Embedding lookup
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)

    # Decoder layers
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # ---- Attention (no pre-norm, QK norm inside) ----
        q = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, attention_size,
            weights[f"{prefix}.w_q"])
        k = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, kv_attention_size,
            weights[f"{prefix}.w_k"])
        v = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, kv_attention_size,
            weights[f"{prefix}.w_v"])

        # QK RMSNorm is sharded to the local Q/K dimensions.
        q_norm_w = weights.get(f"{prefix}.q_norm")
        if q_norm_w is not None:
            q = _add_distributed_rms_norm(
                network, q,
                local_hidden_size=attention_size,
                global_hidden_size=global_attention_size,
                gamma=q_norm_w,
                eps_tensor=eps_tensor,
                tp_size=parallel.tp_size)
        k_norm_w = weights.get(f"{prefix}.k_norm")
        if k_norm_w is not None:
            k = _add_distributed_rms_norm(
                network, k,
                local_hidden_size=kv_attention_size,
                global_hidden_size=global_kv_attention_size,
                gamma=k_norm_w,
                eps_tensor=eps_tensor,
                tp_size=parallel.tp_size)

        # RoPE
        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_tensor, sin_tensor, position_id, head_dim)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_tensor, sin_tensor, position_id, head_dim)

        # Save present K/V
        present_k = k
        present_v = v

        # Cache concat
        k_reshape = network.add_shuffle(k)
        k_reshape.reshape_dims = (1, kv_attention_size)
        v_reshape = network.add_shuffle(v)
        v_reshape.reshape_dims = (1, kv_attention_size)

        all_k = network.add_concatenation(
            [cache_k_inputs[layer_idx], k_reshape.get_output(0)])
        all_k.axis = 0
        all_v = network.add_concatenation(
            [cache_v_inputs[layer_idx], v_reshape.get_output(0)])
        all_v.axis = 0

        mask_reshape = network.add_shuffle(attention_mask)
        mask_reshape.reshape_dims = (1, 1, 1, attention_window)

        context_flat = graph_ops.add_attention_from_rows(
            network, q, all_k.get_output(0), all_v.get_output(0),
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=1, kv_seq=attention_window,
            mask=mask_reshape.get_output(0))

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat,
            attention_size, hidden, weights[f"{prefix}.w_o"])
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)

        # ---- Post-attention norm ----
        normed_attn = graph_ops.add_rms_norm(
            network, attn_out, hidden,
            weights[f"{prefix}.post_attn_norm"], eps_tensor)
        residual1 = network.add_elementwise(
            hidden_state, normed_attn,
            trt.ElementWiseOperation.SUM)
        post_attn_state = residual1.get_output(0)

        # ---- MLP (SwiGLU, no pre-norm) ----
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, post_attn_state, weights=weights, prefix=prefix,
            hidden_size=hidden, mlp_size=mlp_size)
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)

        # ---- Post-feedforward norm ----
        normed_mlp = graph_ops.add_rms_norm(
            network, mlp_out, hidden,
            weights[f"{prefix}.post_ff_norm"], eps_tensor)
        residual2 = network.add_elementwise(
            post_attn_state, normed_mlp,
            trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

        present_k_outputs.append(present_k)
        present_v_outputs.append(present_v)

    # Final norm
    hidden_state = graph_ops.add_rms_norm(
        network, hidden_state, hidden,
        weights["final_norm"], eps_tensor)

    # LM head
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, out_vocab, weights["w_out"])
    b_out = np.zeros(out_vocab, dtype=np.float32)
    logits = graph_ops.add_bias_sum(network, logits, out_vocab, b_out)

    logits.name = "logits"
    network.mark_output(logits)

    # Present K/V outputs
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # Build engine
    if verbose:
        print(f"[trtmc build] Building OLMo2 TP engine "
              f"(rank={parallel.rank}/{parallel.tp_size}, "
              f"{num_layers} layers, hidden={hidden}, "
              f"attn={attention_size}, mlp={mlp_size}, "
              f"cache={max_cache_length}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("OLMo2 tensor-parallel engine build failed")

    return bytes(plan)

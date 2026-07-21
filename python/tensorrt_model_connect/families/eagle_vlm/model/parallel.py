# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Eagle VLM text-backbone builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from ..weights import WeightDict
from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)
from ..plugin import _make_llama3_rope_table_half_dim

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ....parallel_config import ParallelConfig


def shard_eagle_vlm_weights(
    config: "ModelConfig",
    weights: WeightDict,
    *,
    parallel: "ParallelConfig",
) -> WeightDict:
    """Return rank-local Eagle VLM text weights for TP builds."""
    return shard_standard_decoder_weights(config, weights, parallel)


def build_eagle_vlm_tp_engine(
    config: "ModelConfig",
    weights: WeightDict,
    max_cache_length: int,
    *,
    is_reranker: bool = False,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local tensor-parallel Eagle VLM text engine.

    This mirrors the single-device Eagle text builder. The only graph changes
    are rank-local Q/K/V and MLP inner projections plus all-reduce joins after
    row-parallel attention and MLP output projections.
    """
    del precision, quant_ctx

    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("build_eagle_vlm_tp_engine requires tensor_parallel mode with tp_size > 1")
    rank_weights = shard_eagle_vlm_weights(config, weights, parallel=parallel)

    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    from . import model as graph_ops
    from . import model as graph_blocks

    attention_size = int(
        rank_weights.get("_attention_size", config.attention_size // parallel.tp_size)
    )
    mlp_size = int(rank_weights.get("_mlp_size", config.intermediate_size // parallel.tp_size))
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        rank_weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    seq_length = max_cache_length

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    input_ids = network.add_input("input_ids", trt.int32, (seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (seq_length,))
    input_embed = network.add_input("input_embed", trt.float32, (seq_length, hidden))
    use_input_embed = network.add_input("use_input_embed", trt.float32, (seq_length,))

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), rank_weights["embedding"])
    gather = network.add_gather(embedding_table, input_ids, 0)
    token_embed = gather.get_output(0)

    use_reshape = network.add_shuffle(use_input_embed)
    use_reshape.reshape_dims = (seq_length, 1)
    ones_bcast = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    inv_use = network.add_elementwise(
        ones_bcast, use_reshape.get_output(0), trt.ElementWiseOperation.SUB
    )
    embed_part = network.add_elementwise(
        input_embed, use_reshape.get_output(0), trt.ElementWiseOperation.PROD
    )
    token_part = network.add_elementwise(
        token_embed, inv_use.get_output(0), trt.ElementWiseOperation.PROD
    )
    merged = network.add_elementwise(
        embed_part.get_output(0), token_part.get_output(0), trt.ElementWiseOperation.SUM
    )
    hidden_state = merged.get_output(0)

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    llm_config = config.raw.get("llm_config", config.raw)
    rope_params = llm_config.get("rope_scaling") or llm_config.get("rope_parameters", {})
    rope_type = rope_params.get("rope_type", "")
    if rope_type == "llama3":
        cos_half_np = _make_llama3_rope_table_half_dim(
            seq_length,
            head_dim,
            config.rope_theta,
            True,
            factor=rope_params.get("factor", 1.0),
            low_freq_factor=rope_params.get("low_freq_factor", 1.0),
            high_freq_factor=rope_params.get("high_freq_factor", 1.0),
            original_max_position_embeddings=rope_params.get(
                "original_max_position_embeddings", 8192
            ),
        )
        sin_half_np = _make_llama3_rope_table_half_dim(
            seq_length,
            head_dim,
            config.rope_theta,
            False,
            factor=rope_params.get("factor", 1.0),
            low_freq_factor=rope_params.get("low_freq_factor", 1.0),
            high_freq_factor=rope_params.get("high_freq_factor", 1.0),
            original_max_position_embeddings=rope_params.get(
                "original_max_position_embeddings", 8192
            ),
        )
    else:
        cos_half_np = graph_ops.make_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, True
        )
        sin_half_np = graph_ops.make_rope_table_half_dim(
            seq_length, head_dim, config.rope_theta, False
        )
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np)
    rope_position_ids = graph_ops.add_constant(
        network, (seq_length,), np.arange(seq_length, dtype=np.int32), dtype=np.int32
    )
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_const = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(network, (1,), np.array([-1e10], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_const, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_row = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_row.reshape_dims = (1, seq_length)
    query_zeros = graph_ops.add_constant(
        network, (seq_length, 1), np.zeros((seq_length, 1), dtype=np.float32)
    )
    pad_mask_2d = network.add_elementwise(
        query_zeros, pad_mask_row.get_output(0), trt.ElementWiseOperation.SUM
    )
    pad_mask_4d = graph_ops.add_2d_mask_to_4d(network, pad_mask_2d.get_output(0))

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        norm1 = graph_blocks.add_rms_norm(
            network,
            hidden_state,
            hidden,
            rank_weights[f"{prefix}.input_norm"],
            eps_tensor,
        )

        q = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, attention_size, rank_weights[f"{prefix}.w_q"]
        )
        k = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, rank_weights[f"{prefix}.w_k"]
        )
        v = graph_ops.add_matmul_rhs_constant(
            network, norm1, hidden, kv_attention_size, rank_weights[f"{prefix}.w_v"]
        )

        q_rope = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_half_tensor,
            sin_half_tensor,
            rope_position_ids,
            head_dim,
            sequence_length=seq_length,
        )
        k_rope = graph_ops.add_apply_rope_native(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_half_tensor,
            sin_half_tensor,
            rope_position_ids,
            head_dim,
            sequence_length=seq_length,
        )

        attn_concat = graph_ops.add_attention_from_rows(
            network,
            q_rope,
            k_rope,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=seq_length,
            kv_seq=seq_length,
            mask=pad_mask_4d,
            scale=attn_scale,
        )

        proj_out = graph_ops.add_matmul_rhs_constant(
            network, attn_concat, attention_size, hidden, rank_weights[f"{prefix}.w_o"]
        )
        proj_out = add_all_reduce_sum(network, proj_out, parallel.tp_size)

        residual1 = network.add_elementwise(hidden_state, proj_out, trt.ElementWiseOperation.SUM)

        norm2 = graph_blocks.add_rms_norm(
            network,
            residual1.get_output(0),
            hidden,
            rank_weights[f"{prefix}.post_attn_norm"],
            eps_tensor,
        )

        mlp_out = graph_blocks.add_swiglu_mlp(
            network,
            norm2,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
        )
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)

        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
        )
        hidden_state = residual2.get_output(0)

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = graph_blocks.add_rms_norm(
            network, hidden_state, hidden, final_norm, eps_tensor
        )

    if is_reranker and "score_weight" in rank_weights:
        score = graph_ops.add_matmul_rhs_constant(
            network,
            hidden_state,
            hidden,
            rank_weights["score_weight"].shape[1],
            rank_weights["score_weight"],
        )
        if "score_bias" in rank_weights:
            score = graph_ops.add_bias_sum(
                network, score, rank_weights["score_weight"].shape[1], rank_weights["score_bias"]
            )
        score.name = "score"
        network.mark_output(score)
    else:
        hidden_state.name = "hidden_states"
        network.mark_output(hidden_state)

    mode_str = "reranking" if is_reranker else "embedding"
    if verbose:
        print(
            f"[trtmc build] Building Eagle {mode_str} TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, "
            f"{num_layers} layers, hidden={hidden}, "
            f"attn={attention_size}, mlp={mlp_size}, "
            f"seq_len={seq_length}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Eagle VLM tensor-parallel engine build failed")

    return bytes(plan)

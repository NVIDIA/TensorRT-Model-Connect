# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeBERTa-owned tensor-parallel encoder builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys

import numpy as np
import tensorrt as trt

from .graph import model as graph_ops


@dataclass(frozen=True)
class ParallelConfig:
    tp_size: int = 1
    rank: int = -1

    @property
    def enabled(self) -> bool:
        return self.tp_size > 1

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("DeBERTa tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("DeBERTa tensor-parallel rank is outside the requested world")


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    config = value or ParallelConfig()
    config.validate()
    return config


def add_all_reduce_sum(network, tensor, tp_size: int):
    if int(tp_size) <= 1:
        return tensor
    layer = network.add_dist_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create DeBERTa ALL_REDUCE")
    layer.num_ranks = int(tp_size)
    return layer.get_output(0)


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_deberta_tp(config, weights, parallel) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("DeBERTa tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            "DeBERTa tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})"
        )
    if config.intermediate_size % tp != 0:
        raise ValueError(
            "DeBERTa tensor parallel requires intermediate_size divisible by "
            f"tp_size ({config.intermediate_size} vs {tp})"
        )

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layer.{layer_idx}"
        for key in (
            f"{prefix}.w_q",
            f"{prefix}.w_k",
            f"{prefix}.w_v",
            f"{prefix}.pos_proj",
            f"{prefix}.pos_q_proj",
        ):
            if key in weights and weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim must be divisible by tp_size")
        for key in (f"{prefix}.q_bias", f"{prefix}.v_bias", f"{prefix}.pos_q_proj_bias"):
            if key in weights and weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} dim must be divisible by tp_size")
        if weights[f"{prefix}.w_o"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_o input dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc1"].shape[-1] % tp != 0:
            raise ValueError(f"{prefix}.w_fc1 output dim must be divisible by tp_size")
        if weights[f"{prefix}.w_fc2"].shape[0] % tp != 0:
            raise ValueError(f"{prefix}.w_fc2 input dim must be divisible by tp_size")


def shard_deberta_weights(config, weights, *, parallel):
    """Return rank-local DeBERTa weights for the TP builder."""
    _validate_deberta_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".pos_proj", ".pos_q_proj", ".w_fc1")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".q_bias", ".v_bias", ".pos_q_proj_bias", ".fc1_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = config.attention_size // parallel.tp_size
    out["_intermediate_size"] = config.intermediate_size // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_seq_layer_norm(network, inp, hidden_size, gamma, beta, eps):
    return graph_ops.add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps)


def build_tp_deberta_encoder_engine(
    config,
    weights,
    max_seq_length,
    *,
    verbose=False,
    parallel_config=None,
):
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_tp_deberta_encoder_engine requires tensor_parallel mode and tp_size > 1"
        )
    weights = shard_deberta_weights(config, weights, parallel=parallel)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    full_num_heads = config.num_attention_heads
    num_heads = config.num_attention_heads // parallel.tp_size
    head_dim = hidden // full_num_heads
    intermediate = config.intermediate_size // parallel.tp_size
    eps = config.rms_norm_eps

    deberta_cfg = weights.get("_deberta_config", {})
    position_biased_input = deberta_cfg.get("position_biased_input", True)
    type_vocab_size = deberta_cfg.get("type_vocab_size", 0)
    max_relative_positions = deberta_cfg.get("max_relative_positions", 512)
    pos_att_type = deberta_cfg.get("pos_att_type", ["c2p", "p2c"])

    hidden_act = config.hidden_act or config.raw.get("hidden_act", "gelu")

    scale_factor = 1 + len(pos_att_type)
    attn_scale = 1.0 / np.sqrt(head_dim * scale_factor).item()

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.builder_optimization_level = 1
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    token_type_ids = network.add_input("token_type_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # Attention mask: [seq] -> [1, 1, seq] additive
    mask_float = network.add_cast(attention_mask_input, trt.float32)
    ones_c = graph_ops.add_constant(network, (1,), np.array([1.0], dtype=np.float32))
    neg_large = graph_ops.add_constant(network, (1,), np.array([-1e9], dtype=np.float32))
    inv_mask = network.add_elementwise(
        ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Embedding
    embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    embed_out = word_embed.get_output(0)

    if position_biased_input and "position_embedding" in weights:
        pos_embed_table = graph_ops.add_constant(
            network, weights["position_embedding"].shape, weights["position_embedding"]
        )
        pos_indices = graph_ops.add_constant(
            network, (max_seq_length,), np.arange(max_seq_length, dtype=np.int32).astype(np.float32)
        )
        pos_int = network.add_cast(pos_indices, trt.int32)
        pos_embed = network.add_gather(pos_embed_table, pos_int.get_output(0), 0)
        embed_out = network.add_elementwise(
            embed_out, pos_embed.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    if type_vocab_size > 0 and "token_type_embedding" in weights:
        tt_table = graph_ops.add_constant(
            network, (type_vocab_size, hidden), weights["token_type_embedding"]
        )
        tt_embed = network.add_gather(tt_table, token_type_ids, 0)
        embed_out = network.add_elementwise(
            embed_out, tt_embed.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden_state = _add_seq_layer_norm(
        network, embed_out, hidden, weights["embed_norm"], weights["embed_norm_beta"], eps
    )

    # Relative position data
    att_span = min(max_seq_length, max_relative_positions)
    full_rel_emb = weights["rel_embeddings"]
    rel_slice_start = max_relative_positions - att_span
    rel_slice_end = max_relative_positions + att_span
    rel_emb_sliced = full_rel_emb[rel_slice_start:rel_slice_end, :]

    rel_emb_tensor = graph_ops.add_constant(network, (2 * att_span, hidden), rel_emb_sliced)

    q_ids = np.arange(max_seq_length, dtype=np.int64)
    k_ids = np.arange(max_seq_length, dtype=np.int64)
    rel_pos = q_ids[:, None] - k_ids[None, :]

    c2p_pos_np = np.clip(rel_pos + att_span, 0, 2 * att_span - 1).astype(np.int32)
    c2p_pos_expanded = np.broadcast_to(
        c2p_pos_np[np.newaxis, :, :], (num_heads, max_seq_length, max_seq_length)
    ).copy()
    c2p_weights = trt.Weights(np.ascontiguousarray(c2p_pos_expanded, dtype=np.int32))
    c2p_pos_tensor = network.add_constant(
        (num_heads, max_seq_length, max_seq_length), c2p_weights
    ).get_output(0)

    p2c_pos_np = np.clip(-rel_pos + att_span, 0, 2 * att_span - 1).astype(np.int32)
    p2c_pos_expanded = np.broadcast_to(
        p2c_pos_np[np.newaxis, :, :], (num_heads, max_seq_length, max_seq_length)
    ).copy()
    p2c_weights = trt.Weights(np.ascontiguousarray(p2c_pos_expanded, dtype=np.int32))
    p2c_pos_tensor = network.add_constant(
        (num_heads, max_seq_length, max_seq_length), p2c_weights
    ).get_output(0)

    # Encoder layers
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hidden_state = _add_deberta_layer(
            network=network,
            hidden=hidden_state,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_length=max_seq_length,
            attn_scale=attn_scale,
            scale_factor=scale_factor,
            attn_mask=pad_mask_reshape.get_output(0),
            rel_emb_tensor=rel_emb_tensor,
            c2p_pos_tensor=c2p_pos_tensor,
            p2c_pos_tensor=p2c_pos_tensor,
            pos_att_type=pos_att_type,
            att_span=att_span,
            hidden_act=hidden_act,
            eps=eps,
            tp_size=parallel.tp_size,
        )

    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(
            f"[trtmc build] Building DeBERTa encoder "
            f"({num_layers} layers, hidden={hidden}, tp={parallel.tp_size}, "
            f"seq={max_seq_length})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def _add_deberta_layer(
    *,
    network,
    hidden,
    weights,
    prefix,
    hidden_size,
    intermediate_size,
    num_heads,
    head_dim,
    seq_length,
    attn_scale,
    scale_factor,
    attn_mask,
    rel_emb_tensor,
    c2p_pos_tensor,
    p2c_pos_tensor,
    pos_att_type,
    att_span,
    hidden_act,
    eps,
    tp_size,
):
    attention_size = num_heads * head_dim

    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"]
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"]
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"]
    )

    q = graph_ops.add_bias_sum(network, q, attention_size, weights[f"{prefix}.q_bias"])
    v = graph_ops.add_bias_sum(network, v, attention_size, weights[f"{prefix}.v_bias"])

    q_heads = network.add_shuffle(q)
    q_heads.reshape_dims = (seq_length, num_heads, head_dim)
    q_heads.second_transpose = trt.Permutation([1, 0, 2])

    k_heads = network.add_shuffle(k)
    k_heads.reshape_dims = (seq_length, num_heads, head_dim)
    k_heads.second_transpose = trt.Permutation([1, 0, 2])

    v_heads = network.add_shuffle(v)
    v_heads.reshape_dims = (seq_length, num_heads, head_dim)
    v_heads.second_transpose = trt.Permutation([1, 0, 2])

    scale_tensor = graph_ops.add_constant(
        network, (1, 1, 1), np.array([attn_scale], dtype=np.float32)
    )
    q_scaled = network.add_elementwise(
        q_heads.get_output(0), scale_tensor, trt.ElementWiseOperation.PROD
    )

    c2c_score = network.add_matrix_multiply(
        q_scaled.get_output(0),
        trt.MatrixOperation.NONE,
        k_heads.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    attention_scores = c2c_score.get_output(0)

    if "c2p" in pos_att_type:
        pos_key = graph_ops.add_matmul_rhs_constant(
            network, rel_emb_tensor, hidden_size, attention_size, weights[f"{prefix}.pos_proj"]
        )
        pos_key_heads = network.add_shuffle(pos_key)
        pos_key_heads.reshape_dims = (2 * att_span, num_heads, head_dim)
        pos_key_heads.second_transpose = trt.Permutation([1, 0, 2])

        c2p_att = network.add_matrix_multiply(
            q_scaled.get_output(0),
            trt.MatrixOperation.NONE,
            pos_key_heads.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )
        c2p_gather_layer = network.add_gather_v2(
            c2p_att.get_output(0), c2p_pos_tensor, trt.GatherMode.ELEMENT
        )
        c2p_gather_layer.axis = 2
        c2p_gathered = c2p_gather_layer
        attention_scores = network.add_elementwise(
            attention_scores, c2p_gathered.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    if "p2c" in pos_att_type:
        pos_query = graph_ops.add_matmul_rhs_constant(
            network, rel_emb_tensor, hidden_size, attention_size, weights[f"{prefix}.pos_q_proj"]
        )
        pos_query = graph_ops.add_bias_sum(
            network, pos_query, attention_size, weights[f"{prefix}.pos_q_proj_bias"]
        )

        pos_scale = graph_ops.add_constant(
            network, (1, 1, 1), np.array([1.0 / np.sqrt(head_dim * scale_factor)], dtype=np.float32)
        )
        pos_q_heads = network.add_shuffle(pos_query)
        pos_q_heads.reshape_dims = (2 * att_span, num_heads, head_dim)
        pos_q_heads.second_transpose = trt.Permutation([1, 0, 2])
        pos_q_scaled = network.add_elementwise(
            pos_q_heads.get_output(0), pos_scale, trt.ElementWiseOperation.PROD
        )

        p2c_att = network.add_matrix_multiply(
            k_heads.get_output(0),
            trt.MatrixOperation.NONE,
            pos_q_scaled.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )
        p2c_gather_layer = network.add_gather_v2(
            p2c_att.get_output(0), p2c_pos_tensor, trt.GatherMode.ELEMENT
        )
        p2c_gather_layer.axis = 2
        p2c_gathered = p2c_gather_layer
        p2c_transposed = network.add_shuffle(p2c_gathered.get_output(0))
        p2c_transposed.first_transpose = trt.Permutation([0, 2, 1])
        attention_scores = network.add_elementwise(
            attention_scores, p2c_transposed.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    # DeBERTa disentangled attention injects content-to-position and
    # position-to-content logits before softmax. Those terms are
    # query/content-dependent, so native IAttention's mask input is
    # insufficient here.
    masked = network.add_elementwise(attention_scores, attn_mask, trt.ElementWiseOperation.SUM)
    softmax = network.add_softmax(masked.get_output(0))
    softmax.axes = 1 << 2

    context_heads = network.add_matrix_multiply(
        softmax.get_output(0),
        trt.MatrixOperation.NONE,
        v_heads.get_output(0),
        trt.MatrixOperation.NONE,
    )
    context_flat = network.add_shuffle(context_heads.get_output(0))
    context_flat.first_transpose = trt.Permutation([1, 0, 2])
    context_flat.reshape_dims = (seq_length, attention_size)

    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat.get_output(0), attention_size, hidden_size, weights[f"{prefix}.w_o"]
    )
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, weights[f"{prefix}.o_bias"])

    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
    )

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size, weights[f"{prefix}.w_fc1"]
    )
    fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"])
    activated = graph_ops.add_activation(network, fc1, hidden_act)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size, weights[f"{prefix}.w_fc2"]
    )
    fc2 = add_all_reduce_sum(network, fc2, tp_size)
    fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"])

    residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network,
        residual2.get_output(0),
        hidden_size,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"],
        eps,
    )
    return normed2

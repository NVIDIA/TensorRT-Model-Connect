# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeBERTa family model - encoder-only with disentangled attention.

DeBERTa uses:
  - Disentangled self-attention with content-to-position (c2p) and
    position-to-content (p2c) attention components
  - Relative position embeddings shared across all layers
  - Fused QKV via in_proj.weight [3*hidden, hidden] with separate q_bias, v_bias
  - pos_proj (c2p) and pos_q_proj (p2c) per layer for relative position attention
  - position_biased_input=False: NO position embeddings added to word embeddings
  - POST-norm (residual then LayerNorm)
  - Bidirectional attention (no causal mask)
  - type_vocab_size=0: no token type embeddings
  - Scale factor = sqrt(head_dim * scale_factor) where scale_factor = 1 + num_pos_att_types

Trace IDs: ARCH-DEBERTA, UD-DEBERTA-PLUGIN
"""

from __future__ import annotations

import json
import re
import tempfile
import time

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()


graph_ops = sys.modules[__name__]


def _load_ln(readers, prefix):
    w = _load_tensor(readers, f"{prefix}.weight")
    b = _load_tensor(readers, f"{prefix}.bias")
    return w.astype(np.float32), b.astype(np.float32)


name = "deberta"
runtime_strategy = "deberta_encoder_only"


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() == "deberta"


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads

    raw = config.raw
    position_biased_input = raw.get("position_biased_input", True)
    max_pos = config.max_position_embeddings
    type_vocab_size = raw.get("type_vocab_size", 0)
    max_relative_positions = raw.get("max_relative_positions", -1)
    if max_relative_positions < 1:
        max_relative_positions = max_pos
    pos_att_type = raw.get("pos_att_type", "")
    if isinstance(pos_att_type, str):
        pos_att_type = [x.strip() for x in pos_att_type.split("|") if x.strip()]

    weights = WeightDict()

    embedding = _load_tensor(readers, "deberta.embeddings.word_embeddings.weight")
    assert embedding.shape == (vocab, hidden)
    weights["embedding"] = embedding.astype(np.float32)

    if position_biased_input and _has_tensor(
        readers, "deberta.embeddings.position_embeddings.weight"
    ):
        pos_embed = _load_tensor(readers, "deberta.embeddings.position_embeddings.weight")
        weights["position_embedding"] = pos_embed.astype(np.float32)

    if type_vocab_size > 0 and _has_tensor(
        readers, "deberta.embeddings.token_type_embeddings.weight"
    ):
        tt_embed = _load_tensor(readers, "deberta.embeddings.token_type_embeddings.weight")
        weights["token_type_embedding"] = tt_embed.astype(np.float32)

    embed_ln_w, embed_ln_b = _load_ln(readers, "deberta.embeddings.LayerNorm")
    weights["embed_norm"] = embed_ln_w
    weights["embed_norm_beta"] = embed_ln_b

    rel_emb = _load_tensor(readers, "deberta.encoder.rel_embeddings.weight")
    weights["rel_embeddings"] = rel_emb.astype(np.float32)

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"deberta.encoder.layer.{layer_idx}"

        in_proj_w = _load_tensor(readers, f"{hf_prefix}.attention.in_proj.weight")
        in_proj_np = np.array(in_proj_w, dtype=np.float32)

        # DeBERTa interleaves QKV per head in in_proj
        reshaped = in_proj_np.reshape(num_heads, 3 * head_dim, hidden)
        q_w = reshaped[:, :head_dim, :].reshape(hidden, hidden)
        k_w = reshaped[:, head_dim : 2 * head_dim, :].reshape(hidden, hidden)
        v_w = reshaped[:, 2 * head_dim :, :].reshape(hidden, hidden)

        weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T)
        weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T)
        weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T)

        q_bias = _load_tensor(readers, f"{hf_prefix}.attention.q_bias")
        v_bias = _load_tensor(readers, f"{hf_prefix}.attention.v_bias")
        weights[f"{prefix}.q_bias"] = np.array(q_bias, dtype=np.float32).flatten()
        weights[f"{prefix}.v_bias"] = np.array(v_bias, dtype=np.float32).flatten()

        o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
        weights[f"{prefix}.w_o"] = np.ascontiguousarray(np.array(o_w, dtype=np.float32).T)
        weights[f"{prefix}.o_bias"] = np.array(
            _load_tensor(readers, f"{hf_prefix}.attention.output.dense.bias"), dtype=np.float32
        ).flatten()

        attn_ln_w, attn_ln_b = _load_ln(readers, f"{hf_prefix}.attention.output.LayerNorm")
        weights[f"{prefix}.post_attn_norm"] = attn_ln_w
        weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

        if "c2p" in pos_att_type:
            pos_proj_w = _load_tensor(readers, f"{hf_prefix}.attention.pos_proj.weight")
            weights[f"{prefix}.pos_proj"] = np.ascontiguousarray(
                np.array(pos_proj_w, dtype=np.float32).T
            )

        if "p2c" in pos_att_type:
            pos_q_w = _load_tensor(readers, f"{hf_prefix}.attention.pos_q_proj.weight")
            pos_q_b = _load_tensor(readers, f"{hf_prefix}.attention.pos_q_proj.bias")
            weights[f"{prefix}.pos_q_proj"] = np.ascontiguousarray(
                np.array(pos_q_w, dtype=np.float32).T
            )
            weights[f"{prefix}.pos_q_proj_bias"] = np.array(pos_q_b, dtype=np.float32).flatten()

        fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
        fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
        fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
        fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")

        weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(np.array(fc1_w, dtype=np.float32).T)
        weights[f"{prefix}.fc1_bias"] = np.array(fc1_b, dtype=np.float32).flatten()
        weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(np.array(fc2_w, dtype=np.float32).T)
        weights[f"{prefix}.fc2_bias"] = np.array(fc2_b, dtype=np.float32).flatten()

        out_ln_w, out_ln_b = _load_ln(readers, f"{hf_prefix}.output.LayerNorm")
        weights[f"{prefix}.output_norm"] = out_ln_w
        weights[f"{prefix}.output_norm_beta"] = out_ln_b

    weights["_deberta_config"] = {
        "position_biased_input": position_biased_input,
        "type_vocab_size": type_vocab_size,
        "max_relative_positions": max_relative_positions,
        "pos_att_type": pos_att_type,
    }

    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="DeBERTa tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("DeBERTa tensor-parallel builds do not support quantization")

        return build_tp_deberta_encoder_engine(
            config,
            weights,
            max_seq_length=max_cache_length,
            verbose=verbose,
            parallel_config=parallel,
        )

    return _build_deberta_encoder_engine(
        config, weights, max_seq_length=max_cache_length, precision=precision, verbose=verbose
    )


def _add_seq_layer_norm(
    network,
    inp,
    hidden_size,
    gamma,
    beta,
    eps,
    *,
    dtype=np.float32,
):
    return graph_ops.add_layer_norm_native(network, inp, hidden_size, gamma, beta, eps, dtype=dtype)


def _build_deberta_encoder_engine(
    config,
    weights,
    max_seq_length,
    *,
    precision="fp32",
    verbose=False,
):
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    intermediate = config.intermediate_size
    eps = config.rms_norm_eps
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported DeBERTa precision: {precision}")

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
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    input_ids = network.add_input("input_ids", trt.int32, (max_seq_length,))
    token_type_ids = network.add_input("token_type_ids", trt.int32, (max_seq_length,))
    attention_mask_input = network.add_input("attention_mask", trt.int32, (max_seq_length,))

    # Attention mask: [seq] -> [1, 1, seq] additive
    mask_float = network.add_cast(attention_mask_input, work_trt_dtype)
    ones_c = graph_ops.add_constant(
        network, (1,), np.array([1.0], dtype=work_np_dtype), dtype=work_np_dtype
    )
    mask_penalty = -1e4 if precision == "fp16" else -1e9
    neg_large = graph_ops.add_constant(
        network, (1,), np.array([mask_penalty], dtype=work_np_dtype), dtype=work_np_dtype
    )
    inv_mask = network.add_elementwise(
        ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB
    )
    pad_penalty = network.add_elementwise(
        inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD
    )
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Embedding
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    embed_out = word_embed.get_output(0)

    if position_biased_input and "position_embedding" in weights:
        pos_embed_table = graph_ops.add_constant(
            network,
            weights["position_embedding"].shape,
            weights["position_embedding"],
            dtype=work_np_dtype,
        )
        pos_indices = graph_ops.add_constant(
            network,
            (max_seq_length,),
            np.arange(max_seq_length, dtype=np.int32).astype(work_np_dtype),
            dtype=work_np_dtype,
        )
        pos_int = network.add_cast(pos_indices, trt.int32)
        pos_embed = network.add_gather(pos_embed_table, pos_int.get_output(0), 0)
        embed_out = network.add_elementwise(
            embed_out, pos_embed.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    if type_vocab_size > 0 and "token_type_embedding" in weights:
        tt_table = graph_ops.add_constant(
            network, (type_vocab_size, hidden), weights["token_type_embedding"], dtype=work_np_dtype
        )
        tt_embed = network.add_gather(tt_table, token_type_ids, 0)
        embed_out = network.add_elementwise(
            embed_out, tt_embed.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden_state = _add_seq_layer_norm(
        network,
        embed_out,
        hidden,
        weights["embed_norm"],
        weights["embed_norm_beta"],
        eps,
        dtype=work_np_dtype,
    )

    # Relative position data
    att_span = min(max_seq_length, max_relative_positions)
    full_rel_emb = weights["rel_embeddings"]
    rel_slice_start = max_relative_positions - att_span
    rel_slice_end = max_relative_positions + att_span
    rel_emb_sliced = full_rel_emb[rel_slice_start:rel_slice_end, :]

    rel_emb_tensor = graph_ops.add_constant(
        network, (2 * att_span, hidden), rel_emb_sliced, dtype=work_np_dtype
    )

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
            dtype=work_np_dtype,
        )

    public_output = hidden_state
    if public_output.dtype != trt.float32:
        public_output = network.add_cast(public_output, trt.float32).get_output(0)
    public_output.name = "hidden_states"
    network.mark_output(public_output)

    if verbose:
        print(
            f"[trtmc build] Building DeBERTa encoder ({num_layers} layers, "
            f"hidden={hidden}, seq={max_seq_length}, precision={precision})",
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
    dtype=np.float32,
):
    attention_size = num_heads * head_dim

    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"], dtype=dtype
    )

    q = graph_ops.add_bias_sum(network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=dtype)
    v = graph_ops.add_bias_sum(network, v, attention_size, weights[f"{prefix}.v_bias"], dtype=dtype)

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
        network, (1, 1, 1), np.array([attn_scale], dtype=dtype), dtype=dtype
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
            network,
            rel_emb_tensor,
            hidden_size,
            attention_size,
            weights[f"{prefix}.pos_proj"],
            dtype=dtype,
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
            network,
            rel_emb_tensor,
            hidden_size,
            attention_size,
            weights[f"{prefix}.pos_q_proj"],
            dtype=dtype,
        )
        pos_query = graph_ops.add_bias_sum(
            network, pos_query, attention_size, weights[f"{prefix}.pos_q_proj_bias"], dtype=dtype
        )

        pos_scale = graph_ops.add_constant(
            network,
            (1, 1, 1),
            np.array([1.0 / np.sqrt(head_dim * scale_factor)], dtype=dtype),
            dtype=dtype,
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
        network,
        context_flat.get_output(0),
        attention_size,
        hidden_size,
        weights[f"{prefix}.w_o"],
        dtype=dtype,
    )
    attn_out = graph_ops.add_bias_sum(
        network, attn_out, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype
    )

    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype=dtype,
    )

    fc1 = graph_ops.add_matmul_rhs_constant(
        network, normed1, hidden_size, intermediate_size, weights[f"{prefix}.w_fc1"], dtype=dtype
    )
    fc1 = graph_ops.add_bias_sum(
        network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"], dtype=dtype
    )
    activated = graph_ops.add_activation(network, fc1, hidden_act)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, activated, intermediate_size, hidden_size, weights[f"{prefix}.w_fc2"], dtype=dtype
    )
    fc2 = graph_ops.add_bias_sum(
        network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], dtype=dtype
    )

    residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(
        network,
        residual2.get_output(0),
        hidden_size,
        weights[f"{prefix}.output_norm"],
        weights[f"{prefix}.output_norm_beta"],
        eps,
        dtype=dtype,
    )
    return normed2


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
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
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
        hidden_state = _add_deberta_tp_layer(
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


def _add_deberta_tp_layer(
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


requires_tokenizer = True


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _ensure_tokenizer_json(model_dir: Path) -> None:
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        with tempfile.TemporaryDirectory(prefix="trtmc-tokenizer-") as temporary:
            generated = Path(temporary) / "tokenizer.json"
            backend = getattr(tokenizer, "backend_tokenizer", None)
            if backend is None:
                backend = getattr(tokenizer, "_tokenizer", None)
            if backend is not None and hasattr(backend, "save"):
                backend.save(str(generated))
            if not generated.is_file():
                tokenizer.save_pretrained(temporary)
            if not generated.is_file():
                raise RuntimeError("tokenizer conversion did not create tokenizer.json")
            with tempfile.NamedTemporaryFile(
                dir=model_dir, prefix=".trtmc-tokenizer-", suffix=".json", delete=False
            ) as output:
                temporary_path = Path(output.name)
                output.write(generated.read_bytes())
            temporary_path.replace(tokenizer_path)
    except Exception as exc:
        print(
            "[trtmc build] Warning: could not generate tokenizer.json "
            f"(C++ runtime may fail to create tokenizer): {exc}",
            file=sys.stderr,
        )


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete deberta bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("deberta does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("deberta does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context

        family_graph_ops = sys.modules[__name__]

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="deberta tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("deberta tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    _ensure_tokenizer_json(model_path)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

    Constants are cast to ``inp.dtype`` so the elementwise ops are valid in
    a STRONGLY_TYPED network when ``inp`` is bf16 (storage np_dtype is
    fp16, runtime trt_dtype is bfloat16) or any other non-matching combo.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(name, value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


def add_activation(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    activation_type: str,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Dispatch activation by name: 'silu', 'gelu_new', 'gelu', 'relu', 'relu2'/'squared_relu'."""
    if activation_type in ("gelu_new", "gelu"):
        return add_gelu_new(network, inp, dtype=dtype)
    elif activation_type == "relu":
        act = network.add_activation(inp, trt.ActivationType.RELU)
        return act.get_output(0)
    elif activation_type in ("relu2", "squared_relu"):
        relu = network.add_activation(inp, trt.ActivationType.RELU)
        sq = network.add_elementwise(
            relu.get_output(0), relu.get_output(0), trt.ElementWiseOperation.PROD
        )
        return sq.get_output(0)
    elif activation_type == "silu":
        sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
        swish = network.add_elementwise(inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
        return swish.get_output(0)
    else:
        raise ValueError(f"Unsupported activation: {activation_type}")


def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)

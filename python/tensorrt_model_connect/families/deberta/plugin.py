"""DeBERTa family plugin - encoder-only with disentangled attention.

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

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
)
from ... import graph_ops
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()

def _load_ln(readers, prefix):
    w = _load_tensor(readers, f"{prefix}.weight")
    b = _load_tensor(readers, f"{prefix}.bias")
    return w.astype(np.float32), b.astype(np.float32)


class DebertaPlugin:
    name = "deberta"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "deberta"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
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

        if position_biased_input and _has_tensor(readers, "deberta.embeddings.position_embeddings.weight"):
            pos_embed = _load_tensor(readers, "deberta.embeddings.position_embeddings.weight")
            weights["position_embedding"] = pos_embed.astype(np.float32)

        if type_vocab_size > 0 and _has_tensor(readers, "deberta.embeddings.token_type_embeddings.weight"):
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

            in_proj_w = _load_tensor(readers, f"{hf_prefix}.attention.self.in_proj.weight")
            in_proj_np = np.array(in_proj_w, dtype=np.float32)

            # DeBERTa interleaves QKV per head in in_proj
            reshaped = in_proj_np.reshape(num_heads, 3 * head_dim, hidden)
            q_w = reshaped[:, :head_dim, :].reshape(hidden, hidden)
            k_w = reshaped[:, head_dim:2*head_dim, :].reshape(hidden, hidden)
            v_w = reshaped[:, 2*head_dim:, :].reshape(hidden, hidden)

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T)
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T)
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T)

            q_bias = _load_tensor(readers, f"{hf_prefix}.attention.self.q_bias")
            v_bias = _load_tensor(readers, f"{hf_prefix}.attention.self.v_bias")
            weights[f"{prefix}.q_bias"] = np.array(q_bias, dtype=np.float32).flatten()
            weights[f"{prefix}.v_bias"] = np.array(v_bias, dtype=np.float32).flatten()

            o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(np.array(o_w, dtype=np.float32).T)
            weights[f"{prefix}.o_bias"] = np.array(
                _load_tensor(readers, f"{hf_prefix}.attention.output.dense.bias"),
                dtype=np.float32).flatten()

            attn_ln_w, attn_ln_b = _load_ln(readers, f"{hf_prefix}.attention.output.LayerNorm")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b

            if "c2p" in pos_att_type:
                pos_proj_w = _load_tensor(readers, f"{hf_prefix}.attention.self.pos_proj.weight")
                weights[f"{prefix}.pos_proj"] = np.ascontiguousarray(
                    np.array(pos_proj_w, dtype=np.float32).T)

            if "p2c" in pos_att_type:
                pos_q_w = _load_tensor(readers, f"{hf_prefix}.attention.self.pos_q_proj.weight")
                pos_q_b = _load_tensor(readers, f"{hf_prefix}.attention.self.pos_q_proj.bias")
                weights[f"{prefix}.pos_q_proj"] = np.ascontiguousarray(
                    np.array(pos_q_w, dtype=np.float32).T)
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
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="DeBERTa tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("DeBERTa tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_deberta_encoder_engine
            return build_tp_deberta_encoder_engine(
                config, weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        return _build_deberta_encoder_engine(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = DebertaPlugin()


def _add_seq_layer_norm(network, inp, hidden_size, gamma, beta, eps):
    return graph_ops.add_layer_norm_native(
        network, inp, hidden_size, gamma, beta, eps)


def _build_deberta_encoder_engine(config, weights, max_seq_length, *, verbose=False):
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads
    intermediate = config.intermediate_size
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
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
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
    inv_mask = network.add_elementwise(ones_c, mask_float.get_output(0), trt.ElementWiseOperation.SUB)
    pad_penalty = network.add_elementwise(inv_mask.get_output(0), neg_large, trt.ElementWiseOperation.PROD)
    pad_mask_reshape = network.add_shuffle(pad_penalty.get_output(0))
    pad_mask_reshape.reshape_dims = (1, 1, max_seq_length)

    # Embedding
    embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"])
    word_embed = network.add_gather(embedding_table, input_ids, 0)
    embed_out = word_embed.get_output(0)

    if position_biased_input and "position_embedding" in weights:
        pos_embed_table = graph_ops.add_constant(network, weights["position_embedding"].shape, weights["position_embedding"])
        pos_indices = graph_ops.add_constant(network, (max_seq_length,), np.arange(max_seq_length, dtype=np.int32).astype(np.float32))
        pos_int = network.add_cast(pos_indices, trt.int32)
        pos_embed = network.add_gather(pos_embed_table, pos_int.get_output(0), 0)
        embed_out = network.add_elementwise(embed_out, pos_embed.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    if type_vocab_size > 0 and "token_type_embedding" in weights:
        tt_table = graph_ops.add_constant(network, (type_vocab_size, hidden), weights["token_type_embedding"])
        tt_embed = network.add_gather(tt_table, token_type_ids, 0)
        embed_out = network.add_elementwise(embed_out, tt_embed.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    hidden_state = _add_seq_layer_norm(network, embed_out, hidden, weights["embed_norm"], weights["embed_norm_beta"], eps)

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
    c2p_pos_expanded = np.broadcast_to(c2p_pos_np[np.newaxis, :, :], (num_heads, max_seq_length, max_seq_length)).copy()
    c2p_weights = trt.Weights(np.ascontiguousarray(c2p_pos_expanded, dtype=np.int32))
    c2p_pos_tensor = network.add_constant((num_heads, max_seq_length, max_seq_length), c2p_weights).get_output(0)

    p2c_pos_np = np.clip(-rel_pos + att_span, 0, 2 * att_span - 1).astype(np.int32)
    p2c_pos_expanded = np.broadcast_to(p2c_pos_np[np.newaxis, :, :], (num_heads, max_seq_length, max_seq_length)).copy()
    p2c_weights = trt.Weights(np.ascontiguousarray(p2c_pos_expanded, dtype=np.int32))
    p2c_pos_tensor = network.add_constant((num_heads, max_seq_length, max_seq_length), p2c_weights).get_output(0)

    # Encoder layers
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hidden_state = _add_deberta_layer(
            network=network, hidden=hidden_state, weights=weights, prefix=prefix,
            hidden_size=hidden, intermediate_size=intermediate, num_heads=num_heads,
            head_dim=head_dim, seq_length=max_seq_length, attn_scale=attn_scale,
            scale_factor=scale_factor, attn_mask=pad_mask_reshape.get_output(0),
            rel_emb_tensor=rel_emb_tensor, c2p_pos_tensor=c2p_pos_tensor,
            p2c_pos_tensor=p2c_pos_tensor, pos_att_type=pos_att_type,
            att_span=att_span, hidden_act=hidden_act, eps=eps)

    hidden_state.name = "hidden_states"
    network.mark_output(hidden_state)

    if verbose:
        print(f"[trtmc build] Building DeBERTa encoder ({num_layers} layers, hidden={hidden}, seq={max_seq_length})", file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)


def _add_deberta_layer(*, network, hidden, weights, prefix, hidden_size, intermediate_size,
                       num_heads, head_dim, seq_length, attn_scale, scale_factor, attn_mask,
                       rel_emb_tensor, c2p_pos_tensor, p2c_pos_tensor, pos_att_type,
                       att_span, hidden_act, eps):
    attention_size = num_heads * head_dim

    q = graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"])

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

    scale_tensor = graph_ops.add_constant(network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    q_scaled = network.add_elementwise(q_heads.get_output(0), scale_tensor, trt.ElementWiseOperation.PROD)

    c2c_score = network.add_matrix_multiply(q_scaled.get_output(0), trt.MatrixOperation.NONE, k_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)
    attention_scores = c2c_score.get_output(0)

    if "c2p" in pos_att_type:
        pos_key = graph_ops.add_matmul_rhs_constant(network, rel_emb_tensor, hidden_size, attention_size, weights[f"{prefix}.pos_proj"])
        pos_key_heads = network.add_shuffle(pos_key)
        pos_key_heads.reshape_dims = (2 * att_span, num_heads, head_dim)
        pos_key_heads.second_transpose = trt.Permutation([1, 0, 2])

        c2p_att = network.add_matrix_multiply(q_scaled.get_output(0), trt.MatrixOperation.NONE, pos_key_heads.get_output(0), trt.MatrixOperation.TRANSPOSE)
        c2p_gather_layer = network.add_gather_v2(c2p_att.get_output(0), c2p_pos_tensor, trt.GatherMode.ELEMENT)
        c2p_gather_layer.axis = 2
        c2p_gathered = c2p_gather_layer
        attention_scores = network.add_elementwise(attention_scores, c2p_gathered.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    if "p2c" in pos_att_type:
        pos_query = graph_ops.add_matmul_rhs_constant(network, rel_emb_tensor, hidden_size, attention_size, weights[f"{prefix}.pos_q_proj"])
        pos_query = graph_ops.add_bias_sum(network, pos_query, attention_size, weights[f"{prefix}.pos_q_proj_bias"])

        pos_scale = graph_ops.add_constant(network, (1, 1, 1), np.array([1.0 / np.sqrt(head_dim * scale_factor)], dtype=np.float32))
        pos_q_heads = network.add_shuffle(pos_query)
        pos_q_heads.reshape_dims = (2 * att_span, num_heads, head_dim)
        pos_q_heads.second_transpose = trt.Permutation([1, 0, 2])
        pos_q_scaled = network.add_elementwise(pos_q_heads.get_output(0), pos_scale, trt.ElementWiseOperation.PROD)

        p2c_att = network.add_matrix_multiply(k_heads.get_output(0), trt.MatrixOperation.NONE, pos_q_scaled.get_output(0), trt.MatrixOperation.TRANSPOSE)
        p2c_gather_layer = network.add_gather_v2(p2c_att.get_output(0), p2c_pos_tensor, trt.GatherMode.ELEMENT)
        p2c_gather_layer.axis = 2
        p2c_gathered = p2c_gather_layer
        p2c_transposed = network.add_shuffle(p2c_gathered.get_output(0))
        p2c_transposed.first_transpose = trt.Permutation([0, 2, 1])
        attention_scores = network.add_elementwise(attention_scores, p2c_transposed.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    # DeBERTa disentangled attention injects content-to-position and
    # position-to-content logits before softmax. Those terms are
    # query/content-dependent, so native IAttention's mask input is
    # insufficient here.
    masked = network.add_elementwise(attention_scores, attn_mask, trt.ElementWiseOperation.SUM)
    softmax = network.add_softmax(masked.get_output(0))
    softmax.axes = 1 << 2

    context_heads = network.add_matrix_multiply(softmax.get_output(0), trt.MatrixOperation.NONE, v_heads.get_output(0), trt.MatrixOperation.NONE)
    context_flat = network.add_shuffle(context_heads.get_output(0))
    context_flat.first_transpose = trt.Permutation([1, 0, 2])
    context_flat.reshape_dims = (seq_length, attention_size)

    attn_out = graph_ops.add_matmul_rhs_constant(network, context_flat.get_output(0), attention_size, hidden_size, weights[f"{prefix}.w_o"])
    attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, weights[f"{prefix}.o_bias"])

    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    normed1 = _add_seq_layer_norm(network, residual1.get_output(0), hidden_size, weights[f"{prefix}.post_attn_norm"], weights[f"{prefix}.post_attn_norm_beta"], eps)

    fc1 = graph_ops.add_matmul_rhs_constant(network, normed1, hidden_size, intermediate_size, weights[f"{prefix}.w_fc1"])
    fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size, weights[f"{prefix}.fc1_bias"])
    activated = graph_ops.add_activation(network, fc1, hidden_act)
    fc2 = graph_ops.add_matmul_rhs_constant(network, activated, intermediate_size, hidden_size, weights[f"{prefix}.w_fc2"])
    fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"])

    residual2 = network.add_elementwise(normed1, fc2, trt.ElementWiseOperation.SUM)
    normed2 = _add_seq_layer_norm(network, residual2.get_output(0), hidden_size, weights[f"{prefix}.output_norm"], weights[f"{prefix}.output_norm_beta"], eps)
    return normed2

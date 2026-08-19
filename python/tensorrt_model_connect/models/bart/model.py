# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BART family model -- encoder-decoder seq2seq model.

BART is an encoder-decoder transformer for text generation (summarization,
translation, etc.):
  - Encoder: token embeddings + learned positional embeddings + LayerNorm
             -> N self-attention layers -> encoder output [seq_len, d_model]
  - Decoder: autoregressive text generation with causal self-attention (KV cache)
             + cross-attention to encoder output + GELU MLP
  - Uses LayerNorm, GELU activation, learned positional embeddings
  - model_type: "bart", architectures: ["BartModel", "BartForConditionalGeneration"]
  - Shared embedding between encoder and decoder
  - Position embeddings have offset=2 (first 2 positions are reserved)
  - Post-norm (normalize_before=False): norm AFTER residual connection

Cross-attention design:
  Same as Whisper -- cross_k/cross_v inputs to the decoder engine are the RAW
  encoder output. Per-layer K/V projections are baked into the decoder TRT graph.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time

from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel


trt = trt_compat.get_trt()

_max_cache_length = 256
name = "bart"
runtime_strategy = "bart_seq2seq_encoder_decoder"


def matches(config) -> bool:
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() in ("bart", "mbart")


def load_weights(model_dir: str, config: ModelConfig) -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)
    raw = config.raw
    enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
    dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
    enc_heads = raw.get("encoder_attention_heads", config.num_attention_heads)
    dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
    enc_ffn = raw.get("encoder_ffn_dim", config.intermediate_size)
    dec_ffn = raw.get("decoder_ffn_dim", config.intermediate_size)
    max_position_embeddings = raw.get("max_position_embeddings", 1024)
    normalize_embedding = raw.get("normalize_embedding", True)

    weights = WeightDict()
    weights["_enc_layers"] = enc_layers
    weights["_dec_layers"] = dec_layers
    weights["_enc_heads"] = enc_heads
    weights["_dec_heads"] = dec_heads
    weights["_enc_ffn"] = enc_ffn
    weights["_dec_ffn"] = dec_ffn
    weights["_max_position_embeddings"] = max_position_embeddings
    weights["_normalize_embedding"] = normalize_embedding

    # Shared embedding (used by both encoder and decoder)
    if _has_tensor(readers, "shared.weight"):
        shared_embed = _load_tensor(readers, "shared.weight").astype(np.float32)
    elif _has_tensor(readers, "model.shared.weight"):
        shared_embed = _load_tensor(readers, "model.shared.weight").astype(np.float32)
    else:
        raise RuntimeError("BART: cannot find shared embedding weight")
    weights["shared_embedding"] = shared_embed

    # Encoder position embeddings (shape [max_pos+2, hidden] due to offset=2)
    for key in ("encoder.embed_positions.weight", "model.encoder.embed_positions.weight"):
        if _has_tensor(readers, key):
            weights["enc_pos_embedding"] = _load_tensor(readers, key).astype(np.float32)
            break
    if "enc_pos_embedding" not in weights:
        raise RuntimeError("BART: cannot find encoder position embeddings")

    # Encoder layernorm_embedding
    if normalize_embedding:
        for prefix in ("encoder", "model.encoder"):
            if _has_tensor(readers, f"{prefix}.layernorm_embedding.weight"):
                weights["enc_embed_norm"] = _load_tensor(
                    readers, f"{prefix}.layernorm_embedding.weight"
                ).astype(np.float32)
                weights["enc_embed_norm_beta"] = _load_tensor(
                    readers, f"{prefix}.layernorm_embedding.bias"
                ).astype(np.float32)
                break

    # Encoder layers
    for i in range(enc_layers):
        hf = f"encoder.layers.{i}"
        if not _has_tensor(readers, f"{hf}.self_attn.q_proj.weight"):
            hf = f"model.encoder.layers.{i}"
        pfx = f"enc_layer.{i}"
        for proj in ("q", "k", "v"):
            weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"enc_{proj}"
            )
            weights[f"{pfx}.b_{proj}"] = _load_tensor(
                readers, f"{hf}.self_attn.{proj}_proj.bias"
            ).astype(np.float32)
        weights[f"{pfx}.w_o"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "enc_o"
        )
        weights[f"{pfx}.b_o"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(
            np.float32
        )
        weights[f"{pfx}.attn_norm"] = _load_tensor(
            readers, f"{hf}.self_attn_layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.attn_norm_beta"] = _load_tensor(
            readers, f"{hf}.self_attn_layer_norm.bias"
        ).astype(np.float32)
        weights[f"{pfx}.w_fc1"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.fc1.weight"), "enc_fc1"
        )
        weights[f"{pfx}.b_fc1"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
        weights[f"{pfx}.w_fc2"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.fc2.weight"), "enc_fc2"
        )
        weights[f"{pfx}.b_fc2"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
        weights[f"{pfx}.ffn_norm"] = _load_tensor(readers, f"{hf}.final_layer_norm.weight").astype(
            np.float32
        )
        weights[f"{pfx}.ffn_norm_beta"] = _load_tensor(
            readers, f"{hf}.final_layer_norm.bias"
        ).astype(np.float32)

    # Decoder position embeddings
    for key in ("decoder.embed_positions.weight", "model.decoder.embed_positions.weight"):
        if _has_tensor(readers, key):
            weights["dec_pos_embedding"] = _load_tensor(readers, key).astype(np.float32)
            break
    if "dec_pos_embedding" not in weights:
        raise RuntimeError("BART: cannot find decoder position embeddings")

    # Decoder layernorm_embedding
    if normalize_embedding:
        for prefix in ("decoder", "model.decoder"):
            if _has_tensor(readers, f"{prefix}.layernorm_embedding.weight"):
                weights["dec_embed_norm"] = _load_tensor(
                    readers, f"{prefix}.layernorm_embedding.weight"
                ).astype(np.float32)
                weights["dec_embed_norm_beta"] = _load_tensor(
                    readers, f"{prefix}.layernorm_embedding.bias"
                ).astype(np.float32)
                break

    # Decoder layers
    for i in range(dec_layers):
        hf = f"decoder.layers.{i}"
        if not _has_tensor(readers, f"{hf}.self_attn.q_proj.weight"):
            hf = f"model.decoder.layers.{i}"
        pfx = f"layer.{i}"
        for proj in ("q", "k", "v"):
            weights[f"{pfx}.w_{proj}"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.self_attn.{proj}_proj.weight"), f"dec_{proj}"
            )
            weights[f"{pfx}.{proj}_bias"] = _load_tensor(
                readers, f"{hf}.self_attn.{proj}_proj.bias"
            ).astype(np.float32)
        weights[f"{pfx}.w_o"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.self_attn.out_proj.weight"), "dec_o"
        )
        weights[f"{pfx}.o_bias"] = _load_tensor(readers, f"{hf}.self_attn.out_proj.bias").astype(
            np.float32
        )
        weights[f"{pfx}.input_norm"] = _load_tensor(
            readers, f"{hf}.self_attn_layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.input_norm_beta"] = _load_tensor(
            readers, f"{hf}.self_attn_layer_norm.bias"
        ).astype(np.float32)
        for proj in ("q", "k", "v"):
            weights[f"{pfx}.cross_w_{proj}"] = _transpose_2d(
                _load_tensor(readers, f"{hf}.encoder_attn.{proj}_proj.weight"), f"xattn_{proj}"
            )
            weights[f"{pfx}.cross_b_{proj}"] = _load_tensor(
                readers, f"{hf}.encoder_attn.{proj}_proj.bias"
            ).astype(np.float32)
        weights[f"{pfx}.cross_w_o"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.encoder_attn.out_proj.weight"), "xattn_o"
        )
        weights[f"{pfx}.cross_b_o"] = _load_tensor(
            readers, f"{hf}.encoder_attn.out_proj.bias"
        ).astype(np.float32)
        weights[f"{pfx}.cross_attn_norm"] = _load_tensor(
            readers, f"{hf}.encoder_attn_layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.cross_attn_norm_beta"] = _load_tensor(
            readers, f"{hf}.encoder_attn_layer_norm.bias"
        ).astype(np.float32)
        weights[f"{pfx}.w_fc1"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.fc1.weight"), "dec_fc1"
        )
        weights[f"{pfx}.fc1_bias"] = _load_tensor(readers, f"{hf}.fc1.bias").astype(np.float32)
        weights[f"{pfx}.w_fc2"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.fc2.weight"), "dec_fc2"
        )
        weights[f"{pfx}.fc2_bias"] = _load_tensor(readers, f"{hf}.fc2.bias").astype(np.float32)
        weights[f"{pfx}.post_attn_norm"] = _load_tensor(
            readers, f"{hf}.final_layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.post_attn_norm_beta"] = _load_tensor(
            readers, f"{hf}.final_layer_norm.bias"
        ).astype(np.float32)

    # LM head
    if _has_tensor(readers, "lm_head.weight"):
        weights["w_out"] = _transpose_2d(_load_tensor(readers, "lm_head.weight"), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(shared_embed.copy(), "embedding_tied")

    return weights


def build_engine(
    config,
    weights,
    max_cache_length,
    *,
    verbose=False,
    debug_layer_outputs=False,
    parallel_config=None,
    precision="fp32",
):
    global _max_cache_length
    _max_cache_length = max_cache_length
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        if precision != "fp32":
            raise NotImplementedError("BART tensor-parallel decoder builds currently require fp32")
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="BART tensor-parallel decoder builds"
        )
        from .decoder_tp_builder import build_bart_tp_decoder_engine

        return build_bart_tp_decoder_engine(
            config,
            weights,
            max_cache_length,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel,
        )

    dec_layers = weights["_dec_layers"]
    dec_heads = weights["_dec_heads"]
    dec_ffn = weights["_dec_ffn"]
    normalize_embedding = weights["_normalize_embedding"]
    hidden = config.hidden_size
    vocab = config.vocab_size
    head_dim = hidden // dec_heads
    attention_window = max_cache_length + 1
    max_enc_seq = max_cache_length
    activation_function = config.hidden_act or "gelu"
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported BART precision: {precision}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (attention_window,))
    cross_attention_mask = network.add_input("cross_attention_mask", trt.float32, (max_enc_seq,))

    cache_k_inputs, cache_v_inputs = [], []
    for i in range(dec_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", i), trt.float32, (max_cache_length, hidden)
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", i), trt.float32, (max_cache_length, hidden)
            )
        )

    cross_k_inputs, cross_v_inputs = [], []
    for i in range(dec_layers):
        cross_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_k", i), trt.float32, (max_enc_seq, hidden)
            )
        )
        cross_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cross_v", i), trt.float32, (max_enc_seq, hidden)
            )
        )

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["shared_embedding"], dtype=work_np_dtype
    )
    pos_embed_np = weights["dec_pos_embedding"]
    pos_embedding_table = graph_ops.add_constant(
        network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype
    )

    tok_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
    # Position offset=2 for BART
    offset_weights = trt.Weights(np.array([2], dtype=np.int32))
    offset_layer = network.add_constant((1,), offset_weights)
    offset_const = offset_layer.get_output(0)
    offset_pos = network.add_elementwise(
        position_id, offset_const, trt.ElementWiseOperation.SUM
    ).get_output(0)
    pos_embed = network.add_gather(pos_embedding_table, offset_pos, 0).get_output(0)
    hidden_state = network.add_elementwise(
        tok_embed, pos_embed, trt.ElementWiseOperation.SUM
    ).get_output(0)

    if normalize_embedding:
        hidden_state = graph_ops.add_layer_norm_native(
            network,
            hidden_state,
            hidden,
            weights["dec_embed_norm"],
            weights["dec_embed_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )

    cache_k_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in cache_k_inputs
    ]
    cache_v_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in cache_v_inputs
    ]
    cross_k_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in cross_k_inputs
    ]
    cross_v_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in cross_v_inputs
    ]
    attention_mask_work = (
        network.add_cast(attention_mask, work_trt_dtype).get_output(0)
        if attention_mask.dtype != work_trt_dtype
        else attention_mask
    )
    cross_attention_mask_work = (
        network.add_cast(cross_attention_mask, work_trt_dtype).get_output(0)
        if cross_attention_mask.dtype != work_trt_dtype
        else cross_attention_mask
    )

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs, present_v_outputs = [], []
    for layer_idx in range(dec_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_bart_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_work[layer_idx],
            cache_v=cache_v_work[layer_idx],
            cross_k=cross_k_work[layer_idx],
            cross_v=cross_v_work[layer_idx],
            attention_mask=attention_mask_work,
            cross_attention_mask=cross_attention_mask_work,
            eps=config.rms_norm_eps,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            num_heads=dec_heads,
            head_dim=head_dim,
            ffn_dim=dec_ffn,
            max_cache_length=max_cache_length,
            max_enc_seq=max_enc_seq,
            activation_function=activation_function,
            dtype=work_np_dtype,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_out"], dtype=work_np_dtype
    )
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, np.zeros(vocab, dtype=work_np_dtype), dtype=work_np_dtype
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(dec_layers):
        present_k = present_k_outputs[i]
        present_v = present_v_outputs[i]
        if present_k.dtype != trt.float32:
            present_k = network.add_cast(present_k, trt.float32).get_output(0)
            present_v = network.add_cast(present_v, trt.float32).get_output(0)
        present_k.name = graph_ops.layer_tensor_name("present_k", i)
        present_v.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            f"[trtmc build] Building BART decoder ({dec_layers}L, "
            f"h={hidden}, heads={dec_heads}, ffn={dec_ffn}, "
            f"cache={max_cache_length}, precision={precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT decoder engine build failed")
    return bytes(plan)


def build_vision_engine(model_dir, config, weights, *, verbose=False, precision="fp32"):
    mcl = _max_cache_length
    return _build_bart_encoder(
        config, weights, max_cache_length=mcl, verbose=verbose, precision=precision
    )


def get_vl_config(config):
    raw = config.raw
    return {
        "encoder_layers": raw.get("encoder_layers", config.num_hidden_layers),
        "decoder_layers": raw.get("decoder_layers", config.num_hidden_layers),
        "encoder_attention_heads": raw.get("encoder_attention_heads", config.num_attention_heads),
        "decoder_attention_heads": raw.get("decoder_attention_heads", config.num_attention_heads),
        "encoder_ffn_dim": raw.get("encoder_ffn_dim", config.intermediate_size),
        "decoder_ffn_dim": raw.get("decoder_ffn_dim", config.intermediate_size),
        "max_position_embeddings": raw.get("max_position_embeddings", 1024),
        "has_vision_engine": True,
        "is_encoder_decoder": True,
        "decoder_start_token_id": raw.get("decoder_start_token_id", 2),
        "forced_bos_token_id": raw.get("forced_bos_token_id", 0),
        "position_embedding_offset": 2,
    }


def _build_bart_encoder(
    config,
    weights,
    *,
    max_cache_length=256,
    verbose=False,
    precision="fp32",
):
    enc_layers = weights["_enc_layers"]
    enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]
    weights["_max_position_embeddings"]
    normalize_embedding = weights["_normalize_embedding"]
    hidden = config.hidden_size
    vocab = config.vocab_size
    max_enc_seq = max_cache_length
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported BART precision: {precision}")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)

    input_ids = network.add_input("input_ids", trt.int32, (max_enc_seq,))
    attention_mask = network.add_input("attention_mask", trt.float32, (max_enc_seq,))

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["shared_embedding"], dtype=work_np_dtype
    )
    enc_pos_np = weights["enc_pos_embedding"]
    pos_embedding_table = graph_ops.add_constant(
        network, enc_pos_np.shape, enc_pos_np, dtype=work_np_dtype
    )

    tok_embed = network.add_gather(embedding_table, input_ids, 0).get_output(0)
    # Position indices [2, 3, ..., max_enc_seq+1] for offset=2
    pos_indices = np.arange(2, max_enc_seq + 2, dtype=np.int32)
    pos_idx_layer = network.add_constant((max_enc_seq,), trt.Weights(pos_indices))
    pos_indices_const = pos_idx_layer.get_output(0)
    pos_embed = network.add_gather(pos_embedding_table, pos_indices_const, 0).get_output(0)

    hs = network.add_elementwise(tok_embed, pos_embed, trt.ElementWiseOperation.SUM).get_output(0)

    if normalize_embedding:
        hs = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights["enc_embed_norm"],
            weights["enc_embed_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )

    # Reshape attention mask [max_enc_seq] -> [1, 1, 1, max_enc_seq]
    # for native IAttention broadcast across heads and query positions.
    enc_mask_4d = network.add_shuffle(attention_mask)
    enc_mask_4d.reshape_dims = (1, 1, 1, max_enc_seq)
    enc_mask = enc_mask_4d.get_output(0)
    if enc_mask.dtype != work_trt_dtype:
        enc_mask = network.add_cast(enc_mask, work_trt_dtype).get_output(0)
    head_dim = hidden // enc_heads
    activation_function = config.hidden_act or "gelu"

    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"
        # Post-norm BART encoder: self-attention with padding mask
        q = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, hs, hidden, hidden, weights[f"{pfx}.w_q"], dtype=work_np_dtype
            ),
            hidden,
            weights[f"{pfx}.b_q"],
            dtype=work_np_dtype,
        )
        k = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, hs, hidden, hidden, weights[f"{pfx}.w_k"], dtype=work_np_dtype
            ),
            hidden,
            weights[f"{pfx}.b_k"],
            dtype=work_np_dtype,
        )
        v = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, hs, hidden, hidden, weights[f"{pfx}.w_v"], dtype=work_np_dtype
            ),
            hidden,
            weights[f"{pfx}.b_v"],
            dtype=work_np_dtype,
        )
        ctx_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=enc_heads,
            head_dim=head_dim,
            q_seq=max_enc_seq,
            kv_seq=max_enc_seq,
            mask=enc_mask,
        )
        attn = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, ctx_flat, hidden, hidden, weights[f"{pfx}.w_o"], dtype=work_np_dtype
            ),
            hidden,
            weights[f"{pfx}.b_o"],
            dtype=work_np_dtype,
        )
        hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.attn_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )

        fc1 = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, hs, hidden, enc_ffn, weights[f"{pfx}.w_fc1"], dtype=work_np_dtype
            ),
            enc_ffn,
            weights[f"{pfx}.b_fc1"],
            dtype=work_np_dtype,
        )
        act = graph_ops.add_activation(network, fc1, activation_function, dtype=work_np_dtype)
        fc2 = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, act, enc_ffn, hidden, weights[f"{pfx}.w_fc2"], dtype=work_np_dtype
            ),
            hidden,
            weights[f"{pfx}.b_fc2"],
            dtype=work_np_dtype,
        )
        hs = network.add_elementwise(hs, fc2, trt.ElementWiseOperation.SUM).get_output(0)
        hs = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.ffn_norm"],
            weights[f"{pfx}.ffn_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )

    if hs.dtype != trt.float32:
        hs = network.add_cast(hs, trt.float32).get_output(0)
    hs.name = "encoder_output"
    network.mark_output(hs)
    if verbose:
        print(
            f"[trtmc build] Building BART encoder ({enc_layers}L, "
            f"h={hidden}, heads={enc_heads}, seq={max_enc_seq}, "
            f"precision={precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT encoder engine build failed")
    return bytes(plan)


def _add_bart_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    cross_attention_mask,
    eps,
    weights,
    prefix,
    hidden_size,
    num_heads,
    head_dim,
    ffn_dim,
    max_cache_length,
    max_enc_seq,
    activation_function="gelu",
    dtype=np.float32,
):
    attention_size = hidden_size
    attention_window = max_cache_length + 1

    # Self-attention (no pre-norm for post-LN BART)
    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.q_bias"],
        dtype=dtype,
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_k"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.k_bias"],
        dtype=dtype,
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, hidden, hidden_size, attention_size, weights[f"{prefix}.w_v"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.v_bias"],
        dtype=dtype,
    )
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0

    m4 = network.add_shuffle(attention_mask)
    m4.reshape_dims = (1, 1, 1, attention_window)
    cf = graph_ops.add_attention_from_rows(
        network,
        q,
        ak.get_output(0),
        av.get_output(0),
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=attention_window,
        mask=m4.get_output(0),
    )
    sa = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cf, attention_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
        ),
        hidden_size,
        weights[f"{prefix}.o_bias"],
        dtype=dtype,
    )
    # Residual + post-norm
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    psa = graph_ops.add_layer_norm_native(
        network,
        psa,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps,
        dtype=dtype,
    )

    # Cross-attention (no pre-norm)
    cq = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, psa, hidden_size, attention_size, weights[f"{prefix}.cross_w_q"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.cross_b_q"],
        dtype=dtype,
    )
    ck_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network,
            cross_k,
            hidden_size,
            attention_size,
            weights[f"{prefix}.cross_w_k"],
            dtype=dtype,
        ),
        attention_size,
        weights[f"{prefix}.cross_b_k"],
        dtype=dtype,
    )
    cv_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network,
            cross_v,
            hidden_size,
            attention_size,
            weights[f"{prefix}.cross_w_v"],
            dtype=dtype,
        ),
        attention_size,
        weights[f"{prefix}.cross_b_v"],
        dtype=dtype,
    )

    cross_mask_4d = network.add_shuffle(cross_attention_mask)
    cross_mask_4d.reshape_dims = (1, 1, 1, max_enc_seq)
    ccf = graph_ops.add_attention_from_rows(
        network,
        cq,
        ck_proj,
        cv_proj,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=max_enc_seq,
        mask=cross_mask_4d.get_output(0),
    )
    ca = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, ccf, attention_size, hidden_size, weights[f"{prefix}.cross_w_o"], dtype=dtype
        ),
        hidden_size,
        weights[f"{prefix}.cross_b_o"],
        dtype=dtype,
    )
    # Residual + post-norm
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    pca = graph_ops.add_layer_norm_native(
        network,
        pca,
        hidden_size,
        weights[f"{prefix}.cross_attn_norm"],
        weights[f"{prefix}.cross_attn_norm_beta"],
        eps,
        dtype=dtype,
    )

    # MLP (no pre-norm, GELU)
    fc1 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, pca, hidden_size, ffn_dim, weights[f"{prefix}.w_fc1"], dtype=dtype
        ),
        ffn_dim,
        weights[f"{prefix}.fc1_bias"],
        dtype=dtype,
    )
    act = graph_ops.add_activation(network, fc1, activation_function, dtype=dtype)
    fc2 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, act, ffn_dim, hidden_size, weights[f"{prefix}.w_fc2"], dtype=dtype
        ),
        hidden_size,
        weights[f"{prefix}.fc2_bias"],
        dtype=dtype,
    )
    # Residual + post-norm
    out = network.add_elementwise(pca, fc2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm_native(
        network,
        out,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype=dtype,
    )

    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def _mark_debug_output(network, tensor, name):
    out = tensor
    if out.dtype != trt.float32:
        out = network.add_cast(out, trt.float32).get_output(0)
    out.name = name
    network.mark_output(out)


requires_tokenizer = True
embed_input = False


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


def _build_local_engine(config, weights, max_cache_length, precision, verbose, parallel, options):
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
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete bart bundle inside its owning family module."""
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
        raise NotImplementedError("bart does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("bart does not use a decoder KV-cache runtime")

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
    quant_plan = None
    if quantize:
        raise ValueError("bart does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="bart tensor-parallel builds")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
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
            config, weights, max_cache_length, precision, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    vision_started = time.monotonic()
    vision_plan = build_vision_engine(
        str(model_path), config, weights, precision=precision, verbose=verbose
    )
    vision_elapsed = time.monotonic() - vision_started
    add_build_timing(timing, "trt_compile_s", vision_elapsed)
    add_build_timing(timing, "trt_compile_vision_engine_s", vision_elapsed)
    write_build_timing(timing)
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

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
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    vl_config = get_vl_config(config)
    if vl_config is not None:
        runtime_config.update(vl_config)

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

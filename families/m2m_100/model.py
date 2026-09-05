# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2M-100/NLLB family plugin -- encoder-decoder multilingual translation model.

M2M-100/NLLB is an encoder-decoder transformer for multilingual translation:
  - Encoder: token embeddings + sinusoidal positional encoding -> N self-attention
             layers (ReLU MLP) -> encoder output [seq_len, d_model]
  - Decoder: autoregressive text generation with causal self-attention (KV cache)
             + cross-attention to encoder output + ReLU MLP
  - Uses LayerNorm (not RMSNorm), ReLU activation, sinusoidal positional embeddings
  - scale_embedding: True (embeddings multiplied by sqrt(d_model))
  - model_type: "m2m_100", architectures: ["M2M100ForConditionalGeneration"]
  - Shared embedding: encoder, decoder, and lm_head all share the same weight

Cross-attention design:
  Same as Whisper -- cross_k/cross_v inputs to the decoder engine are the RAW
  encoder output (same tensor copied to all layers). The per-layer K/V projections
  are baked into the decoder TRT graph.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_torch_checkpoint,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks


_PROCESS_LOGGERS: dict[bool, trt.Logger] = {}


def _get_process_logger(*, verbose: bool) -> trt.Logger:
    """Return the logger that must outlive every TensorRT builder in this process."""
    logger = _PROCESS_LOGGERS.get(verbose)
    if logger is None:
        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        _PROCESS_LOGGERS[verbose] = logger
    return logger


def _make_sinusoidal_pos_embed(
    num_positions: int, embedding_dim: int, padding_idx: int = 1
) -> np.ndarray:
    """Compute sinusoidal positional embeddings (matches M2M100SinusoidalPositionalEmbedding)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    emb = np.arange(num_positions, dtype=np.float32)[:, None] * emb[None, :]
    result = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        result = np.concatenate([result, np.zeros((num_positions, 1), dtype=np.float32)], axis=-1)
    if padding_idx is not None:
        result[padding_idx] = 0.0
    return result


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _M2M100Model:
    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_torch_checkpoint(model_dir_path)
        raw = config.raw
        hidden = config.hidden_size
        enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
        dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
        enc_heads = raw.get("encoder_attention_heads", config.num_attention_heads)
        dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
        enc_ffn = raw.get("encoder_ffn_dim", config.intermediate_size)
        dec_ffn = raw.get("decoder_ffn_dim", config.intermediate_size)
        max_position_embeddings = raw.get("max_position_embeddings", 1024)
        padding_idx = raw.get("pad_token_id", 1)
        scale_embedding = raw.get("scale_embedding", True)

        weights = WeightDict()
        weights["_enc_layers"] = enc_layers
        weights["_dec_layers"] = dec_layers
        weights["_enc_heads"] = enc_heads
        weights["_dec_heads"] = dec_heads
        weights["_enc_ffn"] = enc_ffn
        weights["_dec_ffn"] = dec_ffn
        weights["_max_position_embeddings"] = max_position_embeddings
        weights["_padding_idx"] = padding_idx
        weights["_scale_embedding"] = scale_embedding

        # Shared embedding table -- used by encoder, decoder, and lm_head.
        # In safetensors, it may be stored as lm_head.weight.
        if _has_tensor(readers, "model.shared.weight"):
            shared_embed = _load_tensor(readers, "model.shared.weight").astype(np.float32)
        elif _has_tensor(readers, "lm_head.weight"):
            shared_embed = _load_tensor(readers, "lm_head.weight").astype(np.float32)
        elif _has_tensor(readers, "model.decoder.embed_tokens.weight"):
            shared_embed = _load_tensor(readers, "model.decoder.embed_tokens.weight").astype(
                np.float32
            )
        else:
            raise RuntimeError("Cannot find shared embedding table")
        weights["shared_embedding"] = shared_embed

        # Sinusoidal positional embeddings (computed, not loaded from weights).
        # offset=2 in HF: num_positions = max_position_embeddings + offset
        offset = 2
        num_positions = max_position_embeddings + offset
        pos_embed = _make_sinusoidal_pos_embed(num_positions, hidden, padding_idx)
        weights["sinusoidal_pos_embed"] = pos_embed

        # Encoder layers
        for i in range(enc_layers):
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
            weights[f"{pfx}.ffn_norm"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.ffn_norm_beta"] = _load_tensor(
                readers, f"{hf}.final_layer_norm.bias"
            ).astype(np.float32)

        weights["enc_final_norm"] = _load_tensor(readers, "model.encoder.layer_norm.weight").astype(
            np.float32
        )
        weights["enc_final_norm_beta"] = _load_tensor(
            readers, "model.encoder.layer_norm.bias"
        ).astype(np.float32)

        # Decoder layers
        for i in range(dec_layers):
            hf = f"model.decoder.layers.{i}"
            pfx = f"layer.{i}"
            # Self-attention
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
            weights[f"{pfx}.o_bias"] = _load_tensor(
                readers, f"{hf}.self_attn.out_proj.bias"
            ).astype(np.float32)
            weights[f"{pfx}.input_norm"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.weight"
            ).astype(np.float32)
            weights[f"{pfx}.input_norm_beta"] = _load_tensor(
                readers, f"{hf}.self_attn_layer_norm.bias"
            ).astype(np.float32)
            # Cross-attention
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
            # MLP
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

        weights["final_norm"] = _load_tensor(readers, "model.decoder.layer_norm.weight").astype(
            np.float32
        )
        weights["final_norm_beta"] = _load_tensor(readers, "model.decoder.layer_norm.bias").astype(
            np.float32
        )

        # LM head (tied to shared embedding)
        if _has_tensor(readers, "lm_head.weight"):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, "lm_head.weight"), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(shared_embed.copy(), "lm_head_tied")

        return weights

    @graph_ops.retain_constant_buffers
    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        precision: str = "fp32",
    ) -> bytes:
        """Build the DECODER TRT engine (with cross-attention to encoder output)."""
        dec_layers = weights["_dec_layers"]
        dec_heads = weights["_dec_heads"]
        dec_ffn = weights["_dec_ffn"]
        hidden = config.hidden_size
        vocab = config.vocab_size
        head_dim = hidden // dec_heads
        attention_window = max_cache_length + 1
        scale_embedding = weights["_scale_embedding"]
        embed_scale = math.sqrt(hidden) if scale_embedding else 1.0
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported M2M-100 precision {precision!r}; expected fp32 or fp16")

        # Use a fixed max_source_length for cross-attention.
        # This determines the encoder output dimension the decoder cross-attends to.
        max_source_length = 128

        logger = _get_process_logger(verbose=verbose)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
        cross_attention_mask = network.add_input(
            "cross_attention_mask", trt.float32, (max_source_length,)
        )

        cache_k_inputs, cache_v_inputs = [], []
        for i in range(dec_layers):
            cache_k_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cache_k", i),
                    trt.float32,
                    (max_cache_length, hidden),
                )
            )
            cache_v_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cache_v", i),
                    trt.float32,
                    (max_cache_length, hidden),
                )
            )

        # Cross-attention inputs: raw encoder output
        cross_k_inputs, cross_v_inputs = [], []
        for i in range(dec_layers):
            cross_k_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cross_k", i),
                    trt.float32,
                    (max_source_length, hidden),
                )
            )
            cross_v_inputs.append(
                network.add_input(
                    graph_ops.layer_tensor_name("cross_v", i),
                    trt.float32,
                    (max_source_length, hidden),
                )
            )

        if work_trt_dtype != trt.float32:
            attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
            cross_attention_mask = network.add_cast(
                cross_attention_mask, work_trt_dtype
            ).get_output(0)
            cache_k_inputs = [
                network.add_cast(t, work_trt_dtype).get_output(0) for t in cache_k_inputs
            ]
            cache_v_inputs = [
                network.add_cast(t, work_trt_dtype).get_output(0) for t in cache_v_inputs
            ]
            cross_k_inputs = [
                network.add_cast(t, work_trt_dtype).get_output(0) for t in cross_k_inputs
            ]
            cross_v_inputs = [
                network.add_cast(t, work_trt_dtype).get_output(0) for t in cross_v_inputs
            ]

        # Decoder embedding + positional encoding
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["shared_embedding"], dtype=work_np_dtype
        )

        # Sinusoidal positional embeddings — shift table so 0-based position_id
        # from KvCache maps to the correct M2M-100 position (offset by padding_idx+1=2).
        padding_idx = weights["_padding_idx"]
        pos_embed_np = weights["sinusoidal_pos_embed"]
        dec_pos_table = pos_embed_np[padding_idx + 1 :]
        pos_embedding_table = graph_ops.add_constant(
            network, dec_pos_table.shape, dec_pos_table, dtype=work_np_dtype
        )

        # Embed token + scale + add positional encoding
        token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
        if embed_scale != 1.0:
            scale_const = graph_ops.add_constant(
                network, (1, 1), np.array([embed_scale], dtype=work_np_dtype), dtype=work_np_dtype
            )
            token_embed = network.add_elementwise(
                token_embed, scale_const, trt.ElementWiseOperation.PROD
            ).get_output(0)
        pos_embed = network.add_gather(pos_embedding_table, position_id, 0).get_output(0)
        hidden_state = network.add_elementwise(
            token_embed, pos_embed, trt.ElementWiseOperation.SUM
        ).get_output(0)

        present_k_outputs, present_v_outputs = [], []
        for layer_idx in range(dec_layers):
            prefix = f"layer.{layer_idx}"
            result = _add_m2m100_decoder_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx],
                cache_v=cache_v_inputs[layer_idx],
                cross_k=cross_k_inputs[layer_idx],
                cross_v=cross_v_inputs[layer_idx],
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                eps=config.rms_norm_eps,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                num_heads=dec_heads,
                head_dim=head_dim,
                ffn_dim=dec_ffn,
                max_cache_length=max_cache_length,
                max_source_length=max_source_length,
                dtype=work_np_dtype,
            )
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])

        # Final norm + LM head
        hidden_state = graph_ops.add_layer_norm_native(
            network,
            hidden_state,
            hidden,
            weights["final_norm"],
            weights["final_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )
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
                f"[trtmc build] Building M2M-100 decoder ({dec_layers}L, h={hidden}, "
                f"heads={dec_heads}, ffn={dec_ffn}, cache={max_cache_length})",
                file=sys.stderr,
            )
        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT decoder engine build failed")
        return bytes(plan)

    def build_vision_engine(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> bytes | None:
        """Build the text ENCODER TRT engine (stored as vision_engine_plan in the bundle)."""
        return _build_m2m100_encoder(config, weights, precision=precision, verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        """Inject encoder-decoder config into bundle config.json."""
        raw = config.raw
        enc_layers = raw.get("encoder_layers", config.num_hidden_layers)
        dec_layers = raw.get("decoder_layers", config.num_hidden_layers)
        decoder_start_token_id = raw.get("decoder_start_token_id", 2)
        return {
            "encoder_layers": enc_layers,
            "decoder_layers": dec_layers,
            "max_source_length": 128,
            "decoder_start_token_id": decoder_start_token_id,
            "scale_embedding": raw.get("scale_embedding", True),
            "has_vision_engine": True,
            "is_encoder_decoder": True,
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        """Override top-level config fields for the C++ runtime.

        M2M-100 uses decoder_attention_heads / encoder_attention_heads instead of
        num_attention_heads. The C++ BaseConfig parser needs num_attention_heads.
        """
        raw = config.raw
        dec_heads = raw.get("decoder_attention_heads", config.num_attention_heads)
        return {
            "num_attention_heads": dec_heads,
            "num_key_value_heads": dec_heads,
        }


@graph_ops.retain_constant_buffers
def _build_m2m100_encoder(
    config,
    weights,
    *,
    precision="fp32",
    verbose=False,
):
    """Build M2M-100 text encoder TRT engine."""
    enc_layers = weights["_enc_layers"]
    enc_heads = weights["_enc_heads"]
    enc_ffn = weights["_enc_ffn"]
    hidden = config.hidden_size
    vocab = config.vocab_size
    max_source_length = 128
    scale_embedding = weights["_scale_embedding"]
    embed_scale = math.sqrt(hidden) if scale_embedding else 1.0
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported M2M-100 precision {precision!r}; expected fp32 or fp16")

    logger = _get_process_logger(verbose=verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.clear_flag(trt.BuilderFlag.TF32)

    # Input: token IDs [max_source_length]
    input_ids = network.add_input("input_ids", trt.int32, (max_source_length,))
    attention_mask = network.add_input("attention_mask", trt.float32, (max_source_length,))

    # Embedding lookup + scale
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["shared_embedding"], dtype=work_np_dtype
    )
    hs = network.add_gather(embedding_table, input_ids, 0).get_output(0)
    # hs shape: [max_source_length, hidden]

    if embed_scale != 1.0:
        scale_const = graph_ops.add_constant(
            network, (1, 1), np.array([embed_scale], dtype=work_np_dtype), dtype=work_np_dtype
        )
        hs = network.add_elementwise(hs, scale_const, trt.ElementWiseOperation.PROD).get_output(0)

    # Add sinusoidal positional encoding.
    # Position IDs for encoder: offset positions starting from padding_idx+1=2.
    # For a sequence of length max_source_length, positions are [2, 3, ..., max_source_length+1].
    padding_idx = weights["_padding_idx"]
    pos_embed_np = weights["sinusoidal_pos_embed"]
    # Extract positions [2..max_source_length+1] from the full table
    enc_pos = pos_embed_np[padding_idx + 1 : padding_idx + 1 + max_source_length].copy()
    enc_pos_const = graph_ops.add_constant(
        network, (max_source_length, hidden), enc_pos, dtype=work_np_dtype
    )
    hs = network.add_elementwise(hs, enc_pos_const, trt.ElementWiseOperation.SUM).get_output(0)

    enc_mask_4d = network.add_shuffle(attention_mask)
    enc_mask_4d.reshape_dims = (1, 1, 1, max_source_length)
    enc_mask = enc_mask_4d.get_output(0)
    if enc_mask.dtype != work_trt_dtype:
        enc_mask = network.add_cast(enc_mask, work_trt_dtype).get_output(0)

    # Encoder layers
    for li in range(enc_layers):
        pfx = f"enc_layer.{li}"
        normed = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.attn_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )
        attn = graph_ops.add_self_attention_block(
            network,
            normed,
            w_q=weights[f"{pfx}.w_q"],
            w_k=weights[f"{pfx}.w_k"],
            w_v=weights[f"{pfx}.w_v"],
            w_o=weights[f"{pfx}.w_o"],
            hidden_size=hidden,
            num_heads=enc_heads,
            seq_length=max_source_length,
            q_bias=weights[f"{pfx}.b_q"],
            k_bias=weights[f"{pfx}.b_k"],
            v_bias=weights[f"{pfx}.b_v"],
            o_bias=weights[f"{pfx}.b_o"],
            mask=enc_mask,
            dtype=work_np_dtype,
        )
        hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
        n2 = graph_ops.add_layer_norm_native(
            network,
            hs,
            hidden,
            weights[f"{pfx}.ffn_norm"],
            weights[f"{pfx}.ffn_norm_beta"],
            config.rms_norm_eps,
            dtype=work_np_dtype,
        )
        fc1 = graph_ops.add_bias_sum(
            network,
            graph_ops.add_matmul_rhs_constant(
                network, n2, hidden, enc_ffn, weights[f"{pfx}.w_fc1"], dtype=work_np_dtype
            ),
            enc_ffn,
            weights[f"{pfx}.b_fc1"],
            dtype=work_np_dtype,
        )
        act = graph_ops.add_activation(network, fc1, "relu", dtype=work_np_dtype)
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
        weights["enc_final_norm"],
        weights["enc_final_norm_beta"],
        config.rms_norm_eps,
        dtype=work_np_dtype,
    )
    if hs.dtype != trt.float32:
        hs = network.add_cast(hs, trt.float32).get_output(0)
    hs.name = "encoder_output"
    network.mark_output(hs)

    if verbose:
        print(
            f"[trtmc build] Building M2M-100 encoder ({enc_layers}L, h={hidden}, "
            f"heads={enc_heads}, src_len={max_source_length})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TensorRT encoder engine build failed")
    return bytes(plan)


def _add_m2m100_decoder_layer(
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
    max_source_length,
    dtype=np.float32,
):
    """Single M2M-100 decoder layer: self-attn + cross-attn + relu MLP."""
    attention_size = hidden_size
    attention_window = max_cache_length + 1

    # --- Self-attention ---
    normed = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_beta"],
        eps,
        dtype=dtype,
    )
    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.q_bias"],
        dtype=dtype,
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attention_size, weights[f"{prefix}.w_k"], dtype=dtype
        ),
        attention_size,
        weights[f"{prefix}.k_bias"],
        dtype=dtype,
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, attention_size, weights[f"{prefix}.w_v"], dtype=dtype
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
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    # --- Cross-attention ---
    cn = graph_ops.add_layer_norm_native(
        network,
        psa,
        hidden_size,
        weights[f"{prefix}.cross_attn_norm"],
        weights[f"{prefix}.cross_attn_norm_beta"],
        eps,
        dtype=dtype,
    )
    cq = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, cn, hidden_size, attention_size, weights[f"{prefix}.cross_w_q"], dtype=dtype
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
    cross_mask_4d.reshape_dims = (1, 1, 1, max_source_length)

    ccf = graph_ops.add_attention_from_rows(
        network,
        cq,
        ck_proj,
        cv_proj,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=max_source_length,
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
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)

    # --- ReLU MLP ---
    fn = graph_ops.add_layer_norm_native(
        network,
        pca,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights[f"{prefix}.post_attn_norm_beta"],
        eps,
        dtype=dtype,
    )
    mlp = graph_blocks.add_gelu_fc_mlp(
        network,
        fn,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=ffn_dim,
        dtype=dtype,
    )
    out = network.add_elementwise(pca, mlp, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": present_k, "present_v": present_v}


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "source.spm",
    "spiece.model",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, model: _M2M100Model, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    runtime.update(model.get_bundle_config_overrides(config) or {})
    runtime.update(model.get_vl_config(config) or {})
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one M2M-100 encoder-decoder bundle through family-owned code."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("m2m_100 does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("m2m_100 does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("m2m_100 does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("m2m_100 does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("m2m_100 does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("m2m_100 supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"m2m_100", "nllb"}:
        raise ValueError(f"M2M-100 does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError("M2M-100 precision must be fp32 or fp16")
    max_sequence_length = _positive_int(
        request.max_sequence_length or min(config.max_position_embeddings, 256),
        "max_sequence_length",
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("M2M-100 max_sequence_length exceeds checkpoint capacity")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("M2M-100 does not expose a tensor-parallel builder")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("M2M-100 does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("M2M-100 does not support mixed-precision layers")

    model = _M2M100Model()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"M2M-100 checkpoint has no tokenizer.json: {tokenizer_path}")
    weights = model.load_weights(str(model_dir), config)
    decoder_plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
    )
    encoder_plan = model.build_vision_engine(
        str(model_dir),
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )
    if encoder_plan is None:
        raise RuntimeError("M2M-100 encoder build returned no engine")

    writer.set_header(family="m2m_100", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", decoder_plan)
    writer.add_bytes("encoder.plan", encoder_plan)
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout="single",
        ),
    )
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""XGLM family plugin — sinusoidal positions + GELU FC MLP.

XGLM (facebook/xglm-564M) uses:
  - Sinusoidal position embeddings (computed, not learned, with offset=2)
  - LayerNorm (with beta)
  - 2-projection MLP (fc1/fc2) with GELU activation
  - Separate Q/K/V/O projections with biases
  - Separate lm_head (not tied despite what config says)
  - Config uses d_model, ffn_dim, attention_heads, num_layers
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


def _make_sinusoidal_position_embedding(
    num_positions: int, embedding_dim: int, padding_idx: int = 1,
) -> np.ndarray:
    """Create sinusoidal position embedding table matching HF XGLMSinusoidal."""
    half_dim = embedding_dim // 2
    emb = np.log(10000.0) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim, dtype=np.float32) * -emb)
    positions = np.arange(num_positions, dtype=np.float32)
    emb = positions[:, None] * emb[None, :]
    table = np.zeros((num_positions, embedding_dim), dtype=np.float32)
    table[:, :half_dim] = np.sin(emb)
    table[:, half_dim:] = np.cos(emb)
    table[padding_idx] = 0.0
    return table


class XGLMPlugin:
    name = "xglm"
    runtime_strategy = "xglm_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "xglm"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        # XGLM uses d_model, ffn_dim, attention_heads, num_layers
        hidden = config.hidden_size  # from d_model
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers  # from num_layers
        num_heads = config.num_attention_heads  # from attention_heads

        weights = WeightDict()

        # Token embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        if embedding.shape[0] != vocab:
            raise ValueError(f"Embedding vocabulary size {embedding.shape[0]} != {vocab}")
        weights["embedding"] = embedding.astype(np.float32)

        # XGLM uses scale_embedding: embed * sqrt(hidden_size)
        scale = config.raw.get("scale_embedding", False)
        if scale:
            weights["embedding"] = weights["embedding"] * np.sqrt(
                hidden).astype(np.float32)

        # Sinusoidal position embedding (computed, not stored in checkpoint).
        # XGLM uses padding_idx=1 and offset=2 (positions 0,1 unused).
        max_pos = config.max_position_embeddings
        pos_table = _make_sinusoidal_position_embedding(
            max_pos + 2, hidden, padding_idx=1)
        # Offset: XGLM adds 2 to position indices, so position 0 maps to row 2
        weights["position_embedding"] = pos_table[2:].astype(np.float32)

        attention_size = num_heads * (hidden // num_heads)
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm 1 (pre-attention)
            ln1_w = _load_tensor(
                readers, f"{hf_prefix}.self_attn_layer_norm.weight")
            ln1_b = _load_tensor(
                readers, f"{hf_prefix}.self_attn_layer_norm.bias")
            weights[f"{prefix}.input_norm"] = ln1_w.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_b.astype(np.float32)

            # LayerNorm 2 (pre-MLP)
            ln2_w = _load_tensor(
                readers, f"{hf_prefix}.final_layer_norm.weight")
            ln2_b = _load_tensor(
                readers, f"{hf_prefix}.final_layer_norm.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_b.astype(np.float32)

            # Q/K/V projections
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.out_proj.weight")

            weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

            # QKV biases
            q_bias = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.bias")
            k_bias = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.bias")
            v_bias = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.bias")
            weights[f"{prefix}.q_bias"] = q_bias.astype(np.float32)
            weights[f"{prefix}.k_bias"] = k_bias.astype(np.float32)
            weights[f"{prefix}.v_bias"] = v_bias.astype(np.float32)

            # Output projection bias
            o_bias_key = f"{hf_prefix}.self_attn.out_proj.bias"
            if _has_tensor(readers, o_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(
                    readers, o_bias_key).astype(np.float32)

            # MLP: fc1 and fc2
            fc1_raw = _load_tensor(readers, f"{hf_prefix}.fc1.weight")
            fc2_raw = _load_tensor(readers, f"{hf_prefix}.fc2.weight")
            if mlp_size == 0:
                mlp_size = fc1_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_raw, "fc1")
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_raw, "fc2")

            # MLP biases
            fc1_bias = _load_tensor(readers, f"{hf_prefix}.fc1.bias")
            fc2_bias = _load_tensor(readers, f"{hf_prefix}.fc2.bias")
            weights[f"{prefix}.fc1_bias"] = fc1_bias.astype(np.float32)
            weights[f"{prefix}.fc2_bias"] = fc2_bias.astype(np.float32)

        # Final LayerNorm
        final_ln_w_key = "model.layer_norm.weight"
        final_ln_b_key = "model.layer_norm.bias"
        if _has_tensor(readers, final_ln_w_key):
            weights["final_norm"] = _load_tensor(
                readers, final_ln_w_key).astype(np.float32)
            if _has_tensor(readers, final_ln_b_key):
                weights["final_norm_beta"] = _load_tensor(
                    readers, final_ln_b_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="XGLM tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError(
                    "XGLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "XGLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision, quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="learned",
                activation="gelu",
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="learned",
            activation="gelu",
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = XGLMPlugin()

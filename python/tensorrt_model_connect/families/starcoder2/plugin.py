# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2 family plugin — LayerNorm + GELU FC + RoPE.

StarCoder2 uses standard LLaMA-like RoPE attention but with:
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (fc1/fc2) with GELU activation instead of SwiGLU
  - QKV biases and output projection bias
  - sliding_window attention (handled by limiting max_cache_length)
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


class StarCoder2Plugin:
    name = "starcoder2"
    runtime_strategy = "starcoder2_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "starcoder2"

    def tokenizer_json_bundle_override(self, model_dir: str | Path) -> bytes:
        from .tokenizer_json import tokenizer_json_bundle_override

        return tokenizer_json_bundle_override(model_dir)

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        kv_dim = num_kv_heads * head_dim

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        if embedding.shape != (vocab, hidden):
            raise ValueError(f'Embedding shape {embedding.shape} != ({vocab}, {hidden})')
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm weights + biases
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            input_norm_beta = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.bias")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            post_norm_beta = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.bias")

            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = input_norm_beta.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = post_norm_beta.astype(np.float32)

            # Q/K/V projections (separate)
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")

            # Compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # QKV biases
            q_bias = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.bias")
            k_bias_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.bias")
            v_bias_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.bias")

            weights[f"{prefix}.q_bias"] = q_bias.astype(np.float32)

            weights[f"{prefix}.k_bias"] = k_bias_raw.astype(np.float32)
            weights[f"{prefix}.v_bias"] = v_bias_raw.astype(np.float32)

            # Output projection bias
            o_bias_key = f"{hf_prefix}.self_attn.o_proj.bias"
            if _has_tensor(readers, o_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(
                    readers, o_bias_key).astype(np.float32)

            # MLP: fc1 (c_fc) and fc2 (c_proj)
            fc1_raw = _load_tensor(readers, f"{hf_prefix}.mlp.c_fc.weight")
            fc2_raw = _load_tensor(readers, f"{hf_prefix}.mlp.c_proj.weight")
            if mlp_size == 0:
                mlp_size = fc1_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_raw, "fc1")
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_raw, "fc2")

            # MLP biases
            fc1_bias_key = f"{hf_prefix}.mlp.c_fc.bias"
            fc2_bias_key = f"{hf_prefix}.mlp.c_proj.bias"
            if _has_tensor(readers, fc1_bias_key):
                weights[f"{prefix}.fc1_bias"] = _load_tensor(
                    readers, fc1_bias_key).astype(np.float32)
            if _has_tensor(readers, fc2_bias_key):
                weights[f"{prefix}.fc2_bias"] = _load_tensor(
                    readers, fc2_bias_key).astype(np.float32)

        # Final LayerNorm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        final_norm_beta_key = "model.norm.bias"
        if _has_tensor(readers, final_norm_beta_key):
            weights["final_norm_beta"] = _load_tensor(
                readers, final_norm_beta_key).astype(np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_dim  # type: ignore[assignment]
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
                parallel, feature="StarCoder2 tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError(
                    "StarCoder2 tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "StarCoder2 tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision, quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="rope",
                activation="gelu_new",
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="rope",
            activation="gelu_new",
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = StarCoder2Plugin()

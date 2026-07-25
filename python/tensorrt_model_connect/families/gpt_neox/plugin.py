# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-NeoX family plugin (Pythia, RedPajama) — parallel residual + partial RoPE.

GPT-NeoX / Pythia uses:
  - LayerNorm (with beta)
  - Parallel residual connections (attention and MLP in parallel)
  - Fused QKV projection (query_key_value)
  - Partial rotary embeddings (rotary_pct, e.g. 0.25)
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _transpose_2d,
)
from ...parallel_config import normalize_parallel_config
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


class GPTNeoXPlugin:
    name = "gpt_neox"
    runtime_strategy = "gpt_neox_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("gpt_neox", "gptneox")

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        return not bool(config.raw.get("_fp32_layers"))

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = hidden // num_heads

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "gpt_neox.embed_in.weight")
        assert embedding.shape == (vocab, hidden)
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"gpt_neox.layers.{layer_idx}"

            # Input LayerNorm (pre-attention)
            ln1_w = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            ln1_b = _load_tensor(readers, f"{hf_prefix}.input_layernorm.bias")
            weights[f"{prefix}.input_norm"] = ln1_w.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_b.astype(np.float32)

            # Post-attention LayerNorm (pre-MLP, used in parallel residual)
            ln2_w = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            ln2_b = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_b.astype(np.float32)

            # Fused QKV: [3*hidden, hidden] — standard Linear layout
            qkv_w = _load_tensor(readers, f"{hf_prefix}.attention.query_key_value.weight")
            qkv_b = _load_tensor(readers, f"{hf_prefix}.attention.query_key_value.bias")

            # GPT-NeoX interleaves Q/K/V per head in the output dimension:
            # For each head h, rows [h*3*hd : h*3*hd+hd] are Q,
            # [h*3*hd+hd : h*3*hd+2*hd] are K, [h*3*hd+2*hd : h*3*hd+3*hd] are V.
            q_parts, k_parts, v_parts = [], [], []
            qb_parts, kb_parts, vb_parts = [], [], []
            for h in range(num_heads):
                base = h * 3 * head_dim
                q_parts.append(qkv_w[base : base + head_dim])
                k_parts.append(qkv_w[base + head_dim : base + 2 * head_dim])
                v_parts.append(qkv_w[base + 2 * head_dim : base + 3 * head_dim])
                qb_parts.append(qkv_b[base : base + head_dim])
                kb_parts.append(qkv_b[base + head_dim : base + 2 * head_dim])
                vb_parts.append(qkv_b[base + 2 * head_dim : base + 3 * head_dim])

            q_w = np.concatenate(q_parts, axis=0)  # [hidden, hidden]
            k_w = np.concatenate(k_parts, axis=0)
            v_w = np.concatenate(v_parts, axis=0)

            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            weights[f"{prefix}.q_bias"] = np.concatenate(qb_parts).astype(np.float32)
            weights[f"{prefix}.k_bias"] = np.concatenate(kb_parts).astype(np.float32)
            weights[f"{prefix}.v_bias"] = np.concatenate(vb_parts).astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attention.dense.weight")
            o_b = _load_tensor(readers, f"{hf_prefix}.attention.dense.bias")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")
            weights[f"{prefix}.o_bias"] = o_b.astype(np.float32)

            # MLP: dense_h_to_4h (fc1) and dense_4h_to_h (fc2)
            fc1_w = _load_tensor(readers, f"{hf_prefix}.mlp.dense_h_to_4h.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.mlp.dense_h_to_4h.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.mlp.dense_4h_to_h.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.mlp.dense_4h_to_h.bias")

            if mlp_size == 0:
                mlp_size = fc1_w.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_w, "fc1")
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_w, "fc2")
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

        # Final LayerNorm
        fn_w = _load_tensor(readers, "gpt_neox.final_layer_norm.weight")
        fn_b = _load_tensor(readers, "gpt_neox.final_layer_norm.bias")
        weights["final_norm"] = fn_w.astype(np.float32)
        weights["final_norm_beta"] = fn_b.astype(np.float32)

        # LM head (embed_out)
        lm_head = _load_tensor(readers, "embed_out.weight")
        weights["w_out"] = _transpose_2d(lm_head, "lm_head")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        # GPT-NeoX uses partial rotary: rotary_pct (default 0.25)
        rotary_pct = config.raw.get("rotary_pct", 0.25)
        use_parallel = config.raw.get("use_parallel_residual", True)

        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                partial_rotary_factor=rotary_pct,
                parallel_residual=use_parallel,
                verbose=verbose,
                parallel_config=parallel,
            )
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            partial_rotary_factor=rotary_pct,
            parallel_residual=use_parallel,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


plugin = GPTNeoXPlugin()

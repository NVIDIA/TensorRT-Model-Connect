# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-Neo family plugin — learned positions + separate Q/K/V Linear + Conv1D MLP.

GPT-Neo (EleutherAI) uses:
  - Learned absolute position embeddings (wpe)
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (c_fc/c_proj) with GELU activation (Conv1D layout)
  - Separate Q/K/V Linear projections (NOT fused, NOT Conv1D)
  - Output projection with bias
  - Tied word embeddings (wte == lm_head)
  - Local/global attention alternation (ignored — our causal mask handles it)
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
from ...parallel_config import normalize_parallel_config
from .standard_decoder_builder import build_standard_decoder_engine
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine


class GPTNeoPlugin:
    name = "gpt_neo"
    runtime_strategy = "gpt_neo_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "gpt_neo"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        _head_dim = hidden // num_heads

        weights = WeightDict()

        # Token embedding (wte)
        embedding = _load_tensor(readers, "transformer.wte.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (wpe) — learned absolute positions
        pos_embed = _load_tensor(readers, "transformer.wpe.weight")
        weights["position_embedding"] = pos_embed.astype(np.float32)

        attention_size = hidden
        mlp_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"transformer.h.{layer_idx}"

            # LayerNorm 1 (pre-attention)
            ln1_weight = _load_tensor(readers, f"{hf_prefix}.ln_1.weight")
            ln1_bias = _load_tensor(readers, f"{hf_prefix}.ln_1.bias")
            weights[f"{prefix}.input_norm"] = ln1_weight.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = ln1_bias.astype(np.float32)

            # LayerNorm 2 (pre-MLP)
            ln2_weight = _load_tensor(readers, f"{hf_prefix}.ln_2.weight")
            ln2_bias = _load_tensor(readers, f"{hf_prefix}.ln_2.bias")
            weights[f"{prefix}.post_attn_norm"] = ln2_weight.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = ln2_bias.astype(np.float32)

            # Separate Q/K/V projections — standard Linear [out, in] layout
            q_w = _load_tensor(
                readers, f"{hf_prefix}.attn.attention.q_proj.weight")
            k_w = _load_tensor(
                readers, f"{hf_prefix}.attn.attention.k_proj.weight")
            v_w = _load_tensor(
                readers, f"{hf_prefix}.attn.attention.v_proj.weight")

            # Transpose [out, in] -> [in, out]
            weights[f"{prefix}.w_q"] = _transpose_2d(q_w, "q_proj")
            weights[f"{prefix}.w_k"] = _transpose_2d(k_w, "k_proj")
            weights[f"{prefix}.w_v"] = _transpose_2d(v_w, "v_proj")

            # Output projection (Linear with bias)
            o_w = _load_tensor(
                readers, f"{hf_prefix}.attn.attention.out_proj.weight")
            o_b = _load_tensor(
                readers, f"{hf_prefix}.attn.attention.out_proj.bias")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_w, "o_proj")
            weights[f"{prefix}.o_bias"] = o_b.astype(np.float32)

            # MLP: c_fc and c_proj — nn.Linear [out, in] layout
            mlp_fc_weight = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_fc.weight")
            mlp_fc_bias = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_fc.bias")
            mlp_proj_weight = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_proj.weight")
            mlp_proj_bias = _load_tensor(
                readers, f"{hf_prefix}.mlp.c_proj.bias")

            if mlp_size == 0:
                mlp_size = mlp_fc_weight.shape[0]

            # Linear: [out, in] -> transpose to [in, out]
            weights[f"{prefix}.w_fc1"] = _transpose_2d(
                mlp_fc_weight, "c_fc")
            weights[f"{prefix}.fc1_bias"] = mlp_fc_bias.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(
                mlp_proj_weight, "c_proj")
            weights[f"{prefix}.fc2_bias"] = mlp_proj_bias.astype(np.float32)

        # Final LayerNorm
        ln_f_weight = _load_tensor(readers, "transformer.ln_f.weight")
        ln_f_bias = _load_tensor(readers, "transformer.ln_f.bias")
        weights["final_norm"] = ln_f_weight.astype(np.float32)
        weights["final_norm_beta"] = ln_f_bias.astype(np.float32)

        # LM head — GPT-Neo ties wte and lm_head
        if _has_tensor(readers, "lm_head.weight"):
            lm_head = _load_tensor(readers, "lm_head.weight")
            weights["w_out"] = _transpose_2d(lm_head, "lm_head")
        else:
            # Tied: reuse embedding [vocab, hidden] -> transpose to [hidden, vocab]
            weights["w_out"] = np.ascontiguousarray(
                embedding.T.astype(np.float32))

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
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision, quant_ctx=quant_ctx,
                norm_type="layernorm",
                mlp_type="gelu_fc",
                position_type="learned",
                activation="gelu_new",
                scale_attn_weights=False,
                verbose=verbose,
                parallel_config=parallel)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="learned",
            activation="gelu_new",
            scale_attn_weights=False,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = GPTNeoPlugin()

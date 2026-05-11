"""Falcon family plugin — LayerNorm + GELU FC + RoPE + GQA.

Falcon-3 (TII) uses:
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection MLP (dense_h_to_4h / dense_4h_to_h) with GELU activation
  - RoPE for positional encoding
  - GQA (grouped query attention)
  - Separate Q/K/V projections (no fused QKV)
  - No QKV biases, no output projection bias
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ModelConfig
from ..checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ..standard_decoder_builder import build_standard_decoder_engine


class FalconPlugin:
    name = "falcon"

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return (mt == "falcon" or mt.startswith("falcon")
                or mt in ("refinedweb", "refinedwebmodel"))

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        weights = WeightDict()

        # Detect RW-style naming (falcon-rw-1b uses transformer.* prefix)
        rw_style = _has_tensor(readers, "transformer.word_embeddings.weight")

        # Embedding
        embed_key = ("transformer.word_embeddings.weight" if rw_style
                     else "model.embed_tokens.weight")
        embedding = _load_tensor(readers, embed_key)
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            if rw_style:
                hf_prefix = f"transformer.h.{layer_idx}"
            else:
                hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm weights + biases
            # RW models use ln_attn/ln_mlp; Falcon-3 uses input_layernorm/post_attention_layernorm
            if rw_style:
                input_norm_key = f"{hf_prefix}.ln_attn.weight"
                input_norm_beta_key = f"{hf_prefix}.ln_attn.bias"
                post_norm_key = f"{hf_prefix}.ln_mlp.weight"
                post_norm_beta_key = f"{hf_prefix}.ln_mlp.bias"
                # RW may use input_layernorm instead of ln_attn
                if not _has_tensor(readers, input_norm_key):
                    input_norm_key = f"{hf_prefix}.input_layernorm.weight"
                    input_norm_beta_key = f"{hf_prefix}.input_layernorm.bias"
                    post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
                    post_norm_beta_key = f"{hf_prefix}.post_attention_layernorm.bias"
            else:
                input_norm_key = f"{hf_prefix}.input_layernorm.weight"
                input_norm_beta_key = f"{hf_prefix}.input_layernorm.bias"
                post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"
                post_norm_beta_key = f"{hf_prefix}.post_attention_layernorm.bias"

            input_norm = _load_tensor(readers, input_norm_key)
            input_norm_beta = _load_tensor(readers, input_norm_beta_key)
            post_norm = _load_tensor(readers, post_norm_key)
            post_norm_beta = _load_tensor(readers, post_norm_beta_key)

            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.input_norm_beta"] = input_norm_beta.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = post_norm_beta.astype(np.float32)

            # Q/K/V projections (separate)
            # RW models use self_attention.query_key_value (fused) or
            # self_attention.{query,key,value}; Falcon-3 uses self_attn.{q,k,v}_proj
            if rw_style:
                attn_prefix = f"{hf_prefix}.self_attention"
                # Check for fused QKV
                fused_qkv_key = f"{attn_prefix}.query_key_value.weight"
                if _has_tensor(readers, fused_qkv_key):
                    fused_qkv = _load_tensor(readers, fused_qkv_key)
                    # Falcon-RW uses HEAD-INTERLEAVED fused QKV layout:
                    # [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]
                    # Shape: [num_heads * 3 * head_dim, hidden]
                    # Reshape to [num_heads, 3, head_dim, hidden] then extract
                    fused_qkv = fused_qkv.reshape(
                        num_heads, 3, head_dim, hidden)
                    q_raw = fused_qkv[:, 0, :, :].reshape(q_dim, hidden)
                    k_raw = fused_qkv[:, 1, :, :].reshape(kv_dim, hidden)
                    v_raw = fused_qkv[:, 2, :, :].reshape(kv_dim, hidden)
                else:
                    q_raw = _load_tensor(
                        readers, f"{attn_prefix}.q_proj.weight")
                    k_raw = _load_tensor(
                        readers, f"{attn_prefix}.k_proj.weight")
                    v_raw = _load_tensor(
                        readers, f"{attn_prefix}.v_proj.weight")
                o_raw = _load_tensor(
                    readers, f"{attn_prefix}.dense.weight")
            else:
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

            # Keep compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # QKV biases (fused or separate)
            if rw_style:
                fused_qkv_bias_key = f"{attn_prefix}.query_key_value.bias"
                if _has_tensor(readers, fused_qkv_bias_key):
                    fused_bias = _load_tensor(
                        readers, fused_qkv_bias_key).astype(np.float32)
                    # Same head-interleaved layout as weight
                    fused_bias = fused_bias.reshape(num_heads, 3, head_dim)
                    weights[f"{prefix}.q_bias"] = fused_bias[:, 0, :].reshape(-1)
                    weights[f"{prefix}.k_bias"] = fused_bias[:, 1, :].reshape(-1)
                    weights[f"{prefix}.v_bias"] = fused_bias[:, 2, :].reshape(-1)
                dense_bias_key = f"{attn_prefix}.dense.bias"
                if _has_tensor(readers, dense_bias_key):
                    weights[f"{prefix}.o_bias"] = _load_tensor(
                        readers, dense_bias_key).astype(np.float32)

            # MLP: Falcon uses dense_h_to_4h / dense_4h_to_h
            if rw_style:
                mlp_prefix = f"{hf_prefix}.mlp"
            else:
                mlp_prefix = f"{hf_prefix}.mlp"
            fc1_raw = _load_tensor(
                readers, f"{mlp_prefix}.dense_h_to_4h.weight")
            fc2_raw = _load_tensor(
                readers, f"{mlp_prefix}.dense_4h_to_h.weight")
            if mlp_size == 0:
                mlp_size = fc1_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(fc1_raw, "fc1")
            weights[f"{prefix}.w_fc2"] = _transpose_2d(fc2_raw, "fc2")

            # MLP biases (if present)
            fc1_bias_key = f"{mlp_prefix}.dense_h_to_4h.bias"
            fc2_bias_key = f"{mlp_prefix}.dense_4h_to_h.bias"
            if _has_tensor(readers, fc1_bias_key):
                weights[f"{prefix}.fc1_bias"] = _load_tensor(
                    readers, fc1_bias_key).astype(np.float32)
            if _has_tensor(readers, fc2_bias_key):
                weights[f"{prefix}.fc2_bias"] = _load_tensor(
                    readers, fc2_bias_key).astype(np.float32)

        # Final LayerNorm
        if rw_style:
            final_norm_key = "transformer.ln_f.weight"
            final_norm_beta_key = "transformer.ln_f.bias"
        else:
            final_norm_key = "model.norm.weight"
            final_norm_beta_key = "model.norm.bias"

        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

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
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        # Falcon-RW models use ALiBi; Falcon-3 uses RoPE
        use_alibi = config.raw.get("alibi", False)
        position_type = "alibi" if use_alibi else "rope"

        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type=position_type,
            activation="gelu_new",
            scale_alibi_bias=use_alibi,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = FalconPlugin()

"""DistilBERT family plugin — encoder-only bidirectional transformer.

DistilBERT is a distilled version of BERT with:
  - 6 layers (vs BERT's 12), 768 hidden, 12 heads (from config)
  - Learned absolute position embeddings
  - NO token type embeddings (no segment A/B)
  - NO pooler layer
  - LayerNorm (with beta) instead of RMSNorm
  - 2-projection FFN (lin1/lin2) with GELU activation
  - POST-norm (residual then LayerNorm), not pre-norm
  - Bidirectional attention (no causal mask)

Weight naming:
  - Embeddings: distilbert.embeddings.word_embeddings, position_embeddings, LayerNorm
  - Attention: distilbert.transformer.layer.N.attention.{q_lin,k_lin,v_lin,out_lin}
  - FFN: distilbert.transformer.layer.N.ffn.{lin1,lin2}
  - Norms: distilbert.transformer.layer.N.{sa_layer_norm,output_layer_norm}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
)
from .encoder_builder import build_encoder_engine


class DistilBertPlugin:
    name = "distilbert"
    runtime_strategy = "encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "distilbert"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        max_pos = config.max_position_embeddings

        weights = WeightDict()

        # Word embedding
        embedding = _load_tensor(readers, "distilbert.embeddings.word_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding (learned absolute)
        pos_embed = _load_tensor(readers, "distilbert.embeddings.position_embeddings.weight")
        assert pos_embed.shape == (max_pos, hidden), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {hidden})")
        weights["position_embedding"] = pos_embed.astype(np.float32)

        # DistilBERT has no token_type_embeddings. The encoder builder expects
        # one, so provide a zero table that acts as an identity under addition.
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        weights["token_type_embedding"] = np.zeros(
            (type_vocab_size, hidden), dtype=np.float32)

        # Embedding LayerNorm
        embed_ln_w = _load_tensor(readers, "distilbert.embeddings.LayerNorm.weight")
        embed_ln_b = _load_tensor(readers, "distilbert.embeddings.LayerNorm.bias")
        weights["embed_norm"] = embed_ln_w.astype(np.float32)
        weights["embed_norm_beta"] = embed_ln_b.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"distilbert.transformer.layer.{layer_idx}"

            # Q, K, V projections — HF stores [out, in], transpose to [in, out]
            q_w = _load_tensor(readers, f"{hf_prefix}.attention.q_lin.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attention.k_lin.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attention.v_lin.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # QKV biases
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.q_lin.bias").astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.k_lin.bias").astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.v_lin.bias").astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attention.out_lin.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.out_lin.bias").astype(np.float32)

            # Post-attention LayerNorm (sa_layer_norm)
            sa_ln_w = _load_tensor(readers, f"{hf_prefix}.sa_layer_norm.weight")
            sa_ln_b = _load_tensor(readers, f"{hf_prefix}.sa_layer_norm.bias")
            weights[f"{prefix}.post_attn_norm"] = sa_ln_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = sa_ln_b.astype(np.float32)

            # FFN: lin1 -> GELU -> lin2
            fc1_w = _load_tensor(readers, f"{hf_prefix}.ffn.lin1.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.ffn.lin1.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.ffn.lin2.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.ffn.lin2.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm (output_layer_norm)
            out_ln_w = _load_tensor(readers, f"{hf_prefix}.output_layer_norm.weight")
            out_ln_b = _load_tensor(readers, f"{hf_prefix}.output_layer_norm.bias")
            weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
            weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

        # DistilBERT has no pooler layer.

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
    ) -> bytes:
        public_module = sys.modules.get(__package__)
        builder = getattr(public_module, "build_encoder_engine", build_encoder_engine)
        return builder(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = DistilBertPlugin()

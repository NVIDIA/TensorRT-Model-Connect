"""ConvBERT family plugin — encoder-only with mixed attention and span-based dynamic convolution.

ConvBERT uses:
  - Learned absolute position embeddings
  - Token type embeddings (segment A/B)
  - LayerNorm (with beta) instead of RMSNorm
  - HYBRID attention: standard multi-head self-attention on HALF the heads,
    span-based dynamic convolution on the other half
  - SeparableConv1D (depthwise + pointwise) for key_conv_attn
  - Dynamic convolution kernels generated per-position via softmax
  - Unfold/im2col for sliding window feature extraction
  - POST-norm (residual then LayerNorm), not pre-norm
  - 2-projection MLP (fc1/fc2) with GELU activation
  - Output projection maps concatenated [attn_context, conv_context] back to hidden_size

Architecture details (conv-bert-base):
  - hidden_size=768, num_attention_heads=12, head_ratio=2
  - Effective attention heads = num_attention_heads // head_ratio = 6
  - attention_head_size = (hidden_size // effective_heads) // 2 = 64
  - all_head_size = effective_heads * attention_head_size = 384
  - Q,K,V project to all_head_size=384 (not full hidden_size)
  - conv_kernel_size=9 for dynamic convolution
  - Output: concat(attn[seq,384], conv[seq,384]) = [seq,768] -> dense -> hidden
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
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


class ConvBertPlugin:
    name = "convbert"
    runtime_strategy = "convbert_encoder_only"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "convbert"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        max_pos = config.max_position_embeddings
        type_vocab_size = config.raw.get("type_vocab_size", 2)
        embedding_size = config.raw.get("embedding_size", hidden)

        # ConvBERT specific config
        head_ratio = config.raw.get("head_ratio", 2)
        num_attention_heads = config.num_attention_heads
        conv_kernel_size = config.raw.get("conv_kernel_size", 9)

        new_num_attention_heads = num_attention_heads // head_ratio
        if new_num_attention_heads < 1:
            new_num_attention_heads = 1
        attention_head_size = (hidden // new_num_attention_heads) // 2
        all_head_size = new_num_attention_heads * attention_head_size

        weights = WeightDict()

        # Store ConvBERT-specific config in weights for the builder
        weights["_convbert_new_num_heads"] = np.array([new_num_attention_heads], dtype=np.int32)
        weights["_convbert_head_size"] = np.array([attention_head_size], dtype=np.int32)
        weights["_convbert_all_head_size"] = np.array([all_head_size], dtype=np.int32)
        weights["_convbert_conv_kernel_size"] = np.array([conv_kernel_size], dtype=np.int32)

        # Detect prefix
        if _has_tensor(readers, "convbert.embeddings.word_embeddings.weight"):
            root = "convbert"
        elif _has_tensor(readers, "embeddings.word_embeddings.weight"):
            root = ""
        else:
            root = "convbert"

        def _pfx(key):
            return f"{root}.{key}" if root else key

        # Word embedding
        embedding = _load_tensor(readers, _pfx("embeddings.word_embeddings.weight"))
        assert embedding.shape == (vocab, embedding_size), (
            f"Embedding shape {embedding.shape} != ({vocab}, {embedding_size})")
        weights["embedding"] = embedding.astype(np.float32)

        # Position embedding
        pos_embed = _load_tensor(readers, _pfx("embeddings.position_embeddings.weight"))
        assert pos_embed.shape == (max_pos, embedding_size), (
            f"Position embedding shape {pos_embed.shape} != ({max_pos}, {embedding_size})")
        weights["position_embedding"] = pos_embed.astype(np.float32)

        # Token type embedding
        tt_embed = _load_tensor(readers, _pfx("embeddings.token_type_embeddings.weight"))
        assert tt_embed.shape == (type_vocab_size, embedding_size), (
            f"Token type embedding shape {tt_embed.shape} != ({type_vocab_size}, {embedding_size})")
        weights["token_type_embedding"] = tt_embed.astype(np.float32)

        # Embedding LayerNorm
        embed_ln_w = _load_tensor(readers, _pfx("embeddings.LayerNorm.weight"))
        embed_ln_b = _load_tensor(readers, _pfx("embeddings.LayerNorm.bias"))
        weights["embed_norm"] = embed_ln_w.astype(np.float32)
        weights["embed_norm_beta"] = embed_ln_b.astype(np.float32)

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = _pfx(f"encoder.layer.{layer_idx}")

            # Q, K, V projections
            q_w = _load_tensor(readers, f"{hf_prefix}.attention.self.query.weight")
            k_w = _load_tensor(readers, f"{hf_prefix}.attention.self.key.weight")
            v_w = _load_tensor(readers, f"{hf_prefix}.attention.self.value.weight")

            weights[f"{prefix}.w_q"] = np.ascontiguousarray(q_w.T.astype(np.float32))
            weights[f"{prefix}.w_k"] = np.ascontiguousarray(k_w.T.astype(np.float32))
            weights[f"{prefix}.w_v"] = np.ascontiguousarray(v_w.T.astype(np.float32))

            # QKV biases
            weights[f"{prefix}.q_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.query.bias").astype(np.float32)
            weights[f"{prefix}.k_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.key.bias").astype(np.float32)
            weights[f"{prefix}.v_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.self.value.bias").astype(np.float32)

            # ConvBERT-specific: SeparableConv1D weights
            sep_dw = _load_tensor(readers, f"{hf_prefix}.attention.self.key_conv_attn_layer.depthwise.weight")
            sep_pw = _load_tensor(readers, f"{hf_prefix}.attention.self.key_conv_attn_layer.pointwise.weight")
            sep_bias = _load_tensor(readers, f"{hf_prefix}.attention.self.key_conv_attn_layer.bias")

            weights[f"{prefix}.sep_conv_dw"] = sep_dw.astype(np.float32)
            weights[f"{prefix}.sep_conv_pw"] = sep_pw.astype(np.float32)
            weights[f"{prefix}.sep_conv_bias"] = sep_bias.squeeze(-1).astype(np.float32)

            # conv_kernel_layer: linear [all_head_size -> num_heads * kernel_size]
            ck_w = _load_tensor(readers, f"{hf_prefix}.attention.self.conv_kernel_layer.weight")
            ck_b = _load_tensor(readers, f"{hf_prefix}.attention.self.conv_kernel_layer.bias")
            weights[f"{prefix}.conv_kernel_w"] = np.ascontiguousarray(ck_w.T.astype(np.float32))
            weights[f"{prefix}.conv_kernel_bias"] = ck_b.astype(np.float32)

            # conv_out_layer: linear [hidden -> all_head_size]
            co_w = _load_tensor(readers, f"{hf_prefix}.attention.self.conv_out_layer.weight")
            co_b = _load_tensor(readers, f"{hf_prefix}.attention.self.conv_out_layer.bias")
            weights[f"{prefix}.conv_out_w"] = np.ascontiguousarray(co_w.T.astype(np.float32))
            weights[f"{prefix}.conv_out_bias"] = co_b.astype(np.float32)

            # Output projection
            o_w = _load_tensor(readers, f"{hf_prefix}.attention.output.dense.weight")
            weights[f"{prefix}.w_o"] = np.ascontiguousarray(o_w.T.astype(np.float32))
            weights[f"{prefix}.o_bias"] = _load_tensor(
                readers, f"{hf_prefix}.attention.output.dense.bias").astype(np.float32)

            # Post-attention LayerNorm
            attn_ln_w = _load_tensor(readers, f"{hf_prefix}.attention.output.LayerNorm.weight")
            attn_ln_b = _load_tensor(readers, f"{hf_prefix}.attention.output.LayerNorm.bias")
            weights[f"{prefix}.post_attn_norm"] = attn_ln_w.astype(np.float32)
            weights[f"{prefix}.post_attn_norm_beta"] = attn_ln_b.astype(np.float32)

            # FFN
            fc1_w = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.weight")
            fc1_b = _load_tensor(readers, f"{hf_prefix}.intermediate.dense.bias")
            fc2_w = _load_tensor(readers, f"{hf_prefix}.output.dense.weight")
            fc2_b = _load_tensor(readers, f"{hf_prefix}.output.dense.bias")

            weights[f"{prefix}.w_fc1"] = np.ascontiguousarray(fc1_w.T.astype(np.float32))
            weights[f"{prefix}.fc1_bias"] = fc1_b.astype(np.float32)
            weights[f"{prefix}.w_fc2"] = np.ascontiguousarray(fc2_w.T.astype(np.float32))
            weights[f"{prefix}.fc2_bias"] = fc2_b.astype(np.float32)

            # Output LayerNorm
            out_ln_w = _load_tensor(readers, f"{hf_prefix}.output.LayerNorm.weight")
            out_ln_b = _load_tensor(readers, f"{hf_prefix}.output.LayerNorm.bias")
            weights[f"{prefix}.output_norm"] = out_ln_w.astype(np.float32)
            weights[f"{prefix}.output_norm_beta"] = out_ln_b.astype(np.float32)

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
                parallel, feature="ConvBERT tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("ConvBERT tensor-parallel builds do not support quantization")
            from .tp_builder import build_tp_convbert_encoder_engine
            return build_tp_convbert_encoder_engine(
                config, weights,
                max_seq_length=max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        from .builder import build_convbert_encoder_engine
        return build_convbert_encoder_engine(
            config, weights,
            max_seq_length=max_cache_length,
            verbose=verbose)


plugin = ConvBertPlugin()

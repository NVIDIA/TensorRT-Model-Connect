"""Nemotron-4 family plugin — LayerNorm1P + squared ReLU MLP + GQA + partial RoPE.

Nemotron-4 (NVIDIA) uses:
  - NemotronLayerNorm1P: LayerNorm with bias, gamma offset (+1), matching HF's
    ``self.weight + 1`` behavior in NemotronLayerNorm1P.forward()
  - 2-projection MLP (up_proj → relu² → down_proj), no gate projection
  - GQA (grouped query attention)
  - Partial RoPE (partial_rotary_factor, typically 0.5)
  - No attention or MLP biases by default

Weight key mapping:
  HF: model.layers.N.mlp.up_proj.weight   → layer.N.w_fc1
  HF: model.layers.N.mlp.down_proj.weight → layer.N.w_fc2
  HF: model.layers.N.input_layernorm.{weight,bias}
  HF: model.layers.N.post_attention_layernorm.{weight,bias}
  (standard Q/K/V/O projections, same as LLaMA)

Models: nvidia/Nemotron-Mini-4B-Instruct, nvidia/Nemotron-4-Mini-Hindi-4B-Base
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
    _target_np_dtype,
)
from .standard_decoder_builder import build_standard_decoder_engine


class NemotronPlugin:
    name = "nemotron"
    runtime_strategy = "nemotron_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "nemotron"

    def load_weights(
        self, model_dir: str, config: ModelConfig, *, precision: str = "fp32",
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
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(target_dtype)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # LayerNorm1P: gamma offset (+1) is applied here so the engine
            # can use standard LayerNorm. HF stores the raw weight; the +1 is
            # applied in NemotronLayerNorm1P.forward().
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32) + 1.0
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32) + 1.0

            # LayerNorm biases
            input_norm_bias_key = f"{hf_prefix}.input_layernorm.bias"
            post_norm_bias_key = f"{hf_prefix}.post_attention_layernorm.bias"
            if _has_tensor(readers, input_norm_bias_key):
                weights[f"{prefix}.input_norm_beta"] = _load_tensor(
                    readers, input_norm_bias_key).astype(np.float32)
            if _has_tensor(readers, post_norm_bias_key):
                weights[f"{prefix}.post_attn_norm_beta"] = _load_tensor(
                    readers, post_norm_bias_key).astype(np.float32)

            # Q/K/V/O projections (separate, standard Linear [out, in])
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

            q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision=precision)
            o_t = _transpose_2d(o_raw, "o_proj", precision=precision)

            # Keep compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Optional attention biases (attention_bias=True in config)
            for proj, dim in [("q_proj", q_dim), ("k_proj", kv_dim),
                              ("v_proj", kv_dim)]:
                bias_key = f"{hf_prefix}.self_attn.{proj}.bias"
                short = proj[0]  # q, k, v
                if _has_tensor(readers, bias_key):
                    weights[f"{prefix}.{short}_bias"] = _load_tensor(
                        readers, bias_key).astype(target_dtype)

            o_bias_key = f"{hf_prefix}.self_attn.o_proj.bias"
            if _has_tensor(readers, o_bias_key):
                weights[f"{prefix}.o_bias"] = _load_tensor(
                    readers, o_bias_key).astype(target_dtype)

            # 2-projection MLP: up_proj → relu² → down_proj
            # Maps to gelu_fc MLP type: up_proj → w_fc1, down_proj → w_fc2
            up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = up_raw.shape[0]

            weights[f"{prefix}.w_fc1"] = _transpose_2d(
                up_raw, "up_proj", precision=precision)
            weights[f"{prefix}.w_fc2"] = _transpose_2d(
                down_raw, "down_proj", precision=precision)

            # Optional MLP biases (mlp_bias=True in config)
            up_bias_key = f"{hf_prefix}.mlp.up_proj.bias"
            down_bias_key = f"{hf_prefix}.mlp.down_proj.bias"
            if _has_tensor(readers, up_bias_key):
                weights[f"{prefix}.fc1_bias"] = _load_tensor(
                    readers, up_bias_key).astype(target_dtype)
            if _has_tensor(readers, down_bias_key):
                weights[f"{prefix}.fc2_bias"] = _load_tensor(
                    readers, down_bias_key).astype(target_dtype)

        # Final LayerNorm1P (+1 gamma offset)
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32) + 1.0
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        final_norm_bias_key = "model.norm.bias"
        if _has_tensor(readers, final_norm_bias_key):
            weights["final_norm_beta"] = _load_tensor(
                readers, final_norm_bias_key).astype(np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head",
                precision=precision)
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied", precision=precision)

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        partial_rotary = config.raw.get("partial_rotary_factor", 0.5)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            mlp_type="gelu_fc",
            position_type="rope",
            activation="relu2",
            partial_rotary_factor=partial_rotary,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = NemotronPlugin()

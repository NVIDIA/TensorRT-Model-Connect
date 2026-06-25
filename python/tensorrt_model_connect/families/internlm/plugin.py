"""InternLM2 family plugin — handles fused wqkv and non-standard key names.

InternLM2 uses the standard decoder pattern (pre-RMSNorm + RoPE + SwiGLU + GQA)
but with different weight key names and a fused QKV projection:

  Embedding:   model.tok_embeddings.weight    (not model.embed_tokens.weight)
  LM head:     output.weight                  (not lm_head.weight)
  Fused QKV:   attention.wqkv.weight           [q_dim + 2*kv_dim, hidden]
  Output proj: attention.wo.weight             (not self_attn.o_proj.weight)
  MLP gate:    feed_forward.w1.weight          (not mlp.gate_proj.weight)
  MLP up:      feed_forward.w3.weight          (not mlp.up_proj.weight)
  MLP down:    feed_forward.w2.weight          (not mlp.down_proj.weight)
  Input norm:  attention_norm.weight           (not input_layernorm.weight)
  Post norm:   ffn_norm.weight                 (not post_attention_layernorm.weight)
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


class InternLMPlugin:
    name = "internlm"
    runtime_strategy = "internlm_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("internlm")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load InternLM2 weights, splitting fused wqkv and mapping key names."""
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

        # Embedding — InternLM2 uses "model.tok_embeddings.weight"
        embedding = _load_tensor(readers, "model.tok_embeddings.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Norms (1D, no transpose)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.attention_norm.weight")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.ffn_norm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection (group-interleaved) ----
            # InternLM2 interleaves QKV by group:
            #   For each KV group g: [Q_heads_in_group, K_head, V_head]
            # Layout: [Q0,Q1,K0,V0, Q2,Q3,K1,V1, ...] when group_size=2
            wqkv_raw = _load_tensor(
                readers, f"{hf_prefix}.attention.wqkv.weight")
            total_qkv = wqkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            assert total_qkv == expected_qkv, (
                f"Layer {layer_idx} wqkv rows {total_qkv} != "
                f"expected {expected_qkv} (q={q_dim}, kv={kv_dim})")

            group_size = num_heads // num_kv_heads
            rows_per_group = group_size * head_dim + 2 * head_dim
            q_parts, k_parts, v_parts = [], [], []
            for g in range(num_kv_heads):
                start = g * rows_per_group
                q_end = start + group_size * head_dim
                k_end = q_end + head_dim
                v_end = k_end + head_dim
                q_parts.append(wqkv_raw[start:q_end, :])
                k_parts.append(wqkv_raw[q_end:k_end, :])
                v_parts.append(wqkv_raw[k_end:v_end, :])
            q_raw = np.concatenate(q_parts, axis=0)
            k_raw = np.concatenate(k_parts, axis=0)
            v_raw = np.concatenate(v_parts, axis=0)
            del wqkv_raw, q_parts, k_parts, v_parts

            if attention_size == 0:
                attention_size = q_dim

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            del q_raw, k_raw, v_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t

            # Output projection — "attention.wo.weight"
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.attention.wo.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- MLP projections ----
            # w1 = gate, w3 = up, w2 = down
            gate_raw = _load_tensor(
                readers, f"{hf_prefix}.feed_forward.w1.weight")
            up_raw = _load_tensor(
                readers, f"{hf_prefix}.feed_forward.w3.weight")
            down_raw = _load_tensor(
                readers, f"{hf_prefix}.feed_forward.w2.weight")

            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del gate_raw, up_raw, down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head — InternLM2 uses "output.weight"
        lm_head_key = "output.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            # Tied embeddings
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
                parallel, feature="InternLM tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError(
                    "InternLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "InternLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length, precision=precision,
                quant_ctx=quant_ctx, verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = InternLMPlugin()

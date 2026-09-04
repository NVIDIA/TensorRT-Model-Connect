# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3 family plugin — handles fused qkv_proj and gate_up_proj splitting."""

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


class PhiPlugin:
    name = "phi"
    runtime_strategy = "phi_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return (mt.startswith("phi")
                and mt != "phimoe"
                and mt not in ("phi4mm", "phi4_multimodal"))

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Phi-3 weights, splitting fused qkv_proj and gate_up_proj."""
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

            # Norms (1D, no transpose)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection ----
            # Shape: [q_dim + 2*kv_dim, hidden]
            qkv_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.qkv_proj.weight")
            total_qkv = qkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            if total_qkv != expected_qkv:
                raise ValueError(f'Layer {layer_idx} qkv_proj rows {total_qkv} != expected {expected_qkv} (q={q_dim}, kv={kv_dim})')

            q_raw = qkv_raw[:q_dim, :]              # [q_dim, hidden]
            k_raw = qkv_raw[q_dim:q_dim+kv_dim, :]  # [kv_dim, hidden]
            v_raw = qkv_raw[q_dim+kv_dim:, :]       # [kv_dim, hidden]
            del qkv_raw

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

            # Output projection (separate, not fused)
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- Fused gate_up projection ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.gate_up_proj.weight")
            intermediate = gate_up_raw.shape[0] // 2
            if mlp_size == 0:
                mlp_size = intermediate

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            del gate_raw, up_raw

            # Down projection (separate, not fused)
            down_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.down_proj.weight")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
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
                parallel, feature="Phi tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError(
                    "Phi tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "Phi tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length, precision=precision,
                quant_ctx=quant_ctx, verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)


plugin = PhiPlugin()

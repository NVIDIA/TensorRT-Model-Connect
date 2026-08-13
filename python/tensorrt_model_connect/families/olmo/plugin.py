# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo family plugin — non-parametric LayerNorm + tied embeddings.

OLMo v1 (allenai/OLMo-1B-hf) uses:
  - Non-parametric LayerNorm (no learnable gamma/beta)
  - Standard separate Q/K/V/O projections (no GQA in 1B)
  - SwiGLU MLP (gate_proj / up_proj / down_proj)
  - RoPE position embeddings
  - Tied word embeddings (no lm_head weight)
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
    _target_np_dtype,
    _transpose_2d,
)
from ...parallel_config import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
)
from .native_decoder_builder import build_native_decoder_engine
from .native_kv_contract import validate_native_kv_weights
from .standard_decoder_builder import build_standard_decoder_engine


class OlmoPlugin:
    name = "olmo"
    runtime_strategy = "olmo_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "olmo"

    def default_build_precision(self, config: ModelConfig) -> str:
        return "fp16" if native_kv_architecture_capability(config).eligible else "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        capability = native_kv_architecture_capability(config)
        return int(config.max_position_embeddings) if capability.eligible else 256

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        return not bool(config.raw.get("_quantized_build_requested"))

    def load_weights(
        self, model_dir: str, config: ModelConfig, *, precision: str = "fp32",
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        storage_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(storage_dtype)

        mlp_size = 0
        attention_size = 0
        kv_attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # OLMo v1 uses non-parametric LayerNorm (no learnable gamma/beta).
            # Provide gamma=ones, beta=zeros for our LayerNorm implementation.
            input_norm_key = f"{hf_prefix}.input_layernorm.weight"
            post_norm_key = f"{hf_prefix}.post_attention_layernorm.weight"

            if _has_tensor(readers, input_norm_key):
                weights[f"{prefix}.input_norm"] = _load_tensor(
                    readers, input_norm_key).astype(storage_dtype)
                weights[f"{prefix}.input_norm_beta"] = np.zeros(
                    hidden, dtype=storage_dtype)
            else:
                weights[f"{prefix}.input_norm"] = np.ones(
                    hidden, dtype=storage_dtype)
                weights[f"{prefix}.input_norm_beta"] = np.zeros(
                    hidden, dtype=storage_dtype)

            if _has_tensor(readers, post_norm_key):
                weights[f"{prefix}.post_attn_norm"] = _load_tensor(
                    readers, post_norm_key).astype(storage_dtype)
                weights[f"{prefix}.post_attn_norm_beta"] = np.zeros(
                    hidden, dtype=storage_dtype)
            else:
                weights[f"{prefix}.post_attn_norm"] = np.ones(
                    hidden, dtype=storage_dtype)
                weights[f"{prefix}.post_attn_norm_beta"] = np.zeros(
                    hidden, dtype=storage_dtype)

            # Q/K/V/O projections
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            q_hidden = q_raw.shape[0]
            if attention_size == 0:
                attention_size = q_hidden

            q_t = _transpose_2d(q_raw, "q_proj", precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision)
            o_t = _transpose_2d(o_raw, "o_proj", precision)

            # Keep compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t
            if kv_attention_size == 0:
                kv_attention_size = k_t.shape[1]

            # MLP
            gate_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.gate_proj.weight")
            up_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(
                readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]

            weights[f"{prefix}.w_gate"] = _transpose_2d(
                gate_raw, "gate_proj", precision)
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj", precision)
            weights[f"{prefix}.w_down"] = _transpose_2d(
                down_raw, "down_proj", precision)

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(storage_dtype)
            weights["final_norm_beta"] = np.zeros(hidden, dtype=storage_dtype)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=storage_dtype)
            weights["final_norm_beta"] = np.zeros(hidden, dtype=storage_dtype)

        # LM head — OLMo ties embeddings
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head", precision)
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied", precision)

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
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
        capability = native_kv_build_capability(
            config,
            precision=precision,
            max_cache_length=max_cache_length,
            parallel_enabled=parallel.enabled,
            quantized=quant_ctx is not None,
            debug_layer_outputs=debug_layer_outputs,
        )
        if capability.eligible:
            validate_native_kv_weights(config, weights)
            config.raw["_decoder_engine_layout_supported"] = True
            config.raw["_native_kv_cache_metadata"] = {
                "native_kv_contract_version": 1,
                "native_kv_cache": True,
            }
            role = str(config.raw.get("_decoder_engine_role", ""))
            if role not in ("prefill", "decode"):
                raise ValueError(
                    "native OLMo requires explicit split engine role "
                    f"'prefill' or 'decode', got {role!r}")
            return build_native_decoder_engine(
                config, weights, max_cache_length, precision="fp16",
                profile_mode=role, verbose=verbose)

        config.raw.pop("_native_kv_cache_metadata", None)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="layernorm",
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision, quant_ctx=quant_ctx,
            norm_type="layernorm",
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


plugin = OlmoPlugin()

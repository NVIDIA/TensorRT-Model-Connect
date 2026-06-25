"""Gemma family plugin — applies +1.0 to RMSNorm gamma and sqrt(hidden) embed scale."""

from __future__ import annotations

import math
from pathlib import Path

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    load_standard_weights,
)
from ...parallel_config import normalize_parallel_config
from .dual_profile_decoder_tp_builder import build_dual_profile_tp_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


class GemmaPlugin:
    name = "gemma"
    runtime_strategy = "gemma_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("gemma")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        weights = load_standard_weights(model_dir, config, precision=precision)
        readers = _open_safetensors(Path(model_dir))

        # Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"
            weights[f"{prefix}.input_norm"] = weights[f"{prefix}.input_norm"] + 1.0
            weights[f"{prefix}.post_attn_norm"] = weights[f"{prefix}.post_attn_norm"] + 1.0
            pre_ffn_key = f"{hf_prefix}.pre_feedforward_layernorm.weight"
            if _has_tensor(readers, pre_ffn_key):
                weights[f"{prefix}.pre_ffn_norm"] = (
                    _load_tensor(readers, pre_ffn_key).astype("float32") + 1.0)
            post_ffn_key = f"{hf_prefix}.post_feedforward_layernorm.weight"
            if _has_tensor(readers, post_ffn_key):
                weights[f"{prefix}.post_ffn_norm"] = (
                    _load_tensor(readers, post_ffn_key).astype("float32") + 1.0)
        weights["final_norm"] = weights["final_norm"] + 1.0

        # Fix 2: Gemma scales embedding by sqrt(hidden_size).
        scale = math.sqrt(config.hidden_size)
        weights["embedding"] = weights["embedding"] * scale

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel)

        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose)


plugin = GemmaPlugin()

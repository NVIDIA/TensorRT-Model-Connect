# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA family plugin."""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .standard_decoder_builder import build_standard_decoder_engine


def _is_minitron_4b_native_kv_prototype(config: ModelConfig) -> bool:
    """Match the exact Llama-3.1-Minitron-4B model qualified here."""
    rope_scaling = config.raw.get("rope_scaling")
    return (
        config.model_type.lower() == "llama"
        and config.hidden_size == 4096
        and config.intermediate_size == 14336
        and config.num_hidden_layers == 16
        and config.num_attention_heads == 32
        and config.num_key_value_heads == 8
        and config.head_dim == 128
        and config.max_position_embeddings == 131072
        and isinstance(rope_scaling, dict)
        and str(
            rope_scaling.get("rope_type", rope_scaling.get("type", ""))
        ).lower() == "llama3"
        and float(rope_scaling.get("factor", 0.0)) == 8.0
        and int(rope_scaling.get("original_max_position_embeddings", 0))
        == 8192
    )


class LlamaPlugin:
    name = "llama"
    runtime_strategy = "llama_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("llama")

    def default_build_precision(self, config: ModelConfig) -> str:
        return (
            "bf16"
            if _is_minitron_4b_native_kv_prototype(config)
            else "fp32"
        )

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use full capacity only for the native fused-attention path."""
        native_eligible = (
            _is_minitron_4b_native_kv_prototype(config)
            and not bool(config.raw.get("_parallel_build_enabled", False))
            and not bool(config.raw.get("_runtime_dynamic_kv_requested", False))
            and not bool(config.raw.get("_fp32_layers"))
            and str(config.raw.get("_resolved_build_precision", "bf16"))
            in ("fp16", "bf16")
        )
        return int(config.max_position_embeddings) if native_eligible else 256

    def supports_split_decoder_roles(self, config: ModelConfig) -> bool:
        return not bool(config.raw.get("_fp32_layers"))

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(
            model_dir,
            config,
            precision=precision,
            fp32_layers=tuple(config.raw.get("_fp32_layers", ())),
        )

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, debug_layer_outputs: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            native_kv_cache=(
                _is_minitron_4b_native_kv_prototype(config)
                and not bool(config.raw.get("dynamic_kv_cache", False))
                and not debug_layer_outputs
                and not bool(config.raw.get("_fp32_layers"))
                and str(precision).lower() in ("fp16", "bf16")
            ))


plugin = LlamaPlugin()

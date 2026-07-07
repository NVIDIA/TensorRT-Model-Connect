# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma family plugin — applies +1.0 to RMSNorm gamma and sqrt(hidden) embed scale."""

from __future__ import annotations

import math

from tensorrt_model_connect.parallel_config import normalize_parallel_config

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig
from .standard_decoder_builder import build_standard_decoder_engine


def build_dual_profile_tp_decoder_engine(*args, **kwargs) -> bytes:
    from .dual_profile_decoder_tp_builder import (
        build_dual_profile_tp_decoder_engine as _build_dual_profile_tp_decoder_engine,
    )

    return _build_dual_profile_tp_decoder_engine(*args, **kwargs)


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
        load_kwargs: dict[str, str] = {}
        if config.model_type == "gemma3" and isinstance(
            config.raw.get("text_config"), dict
        ):
            load_kwargs = {
                "model_prefix": "language_model.model",
                "lm_head_key": "language_model.lm_head.weight",
            }
        weights = load_standard_weights(
            model_dir,
            config,
            precision=precision,
            **load_kwargs,
        )

        # Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            weights[f"{prefix}.input_norm"] = weights[f"{prefix}.input_norm"] + 1.0
            weights[f"{prefix}.post_attn_norm"] = weights[f"{prefix}.post_attn_norm"] + 1.0
            if f"{prefix}.q_norm" in weights:
                weights[f"{prefix}.q_norm"] = weights[f"{prefix}.q_norm"] + 1.0
            if f"{prefix}.k_norm" in weights:
                weights[f"{prefix}.k_norm"] = weights[f"{prefix}.k_norm"] + 1.0
            if f"{prefix}.pre_ff_norm" in weights:
                weights[f"{prefix}.pre_ff_norm"] = weights[f"{prefix}.pre_ff_norm"] + 1.0
            if f"{prefix}.post_ff_norm" in weights:
                weights[f"{prefix}.post_ff_norm"] = weights[f"{prefix}.post_ff_norm"] + 1.0
        weights["final_norm"] = weights["final_norm"] + 1.0

        # Gemma scales gathered embeddings in the model dtype at runtime.
        weights["_embedding_scale"] = math.sqrt(config.hidden_size)

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

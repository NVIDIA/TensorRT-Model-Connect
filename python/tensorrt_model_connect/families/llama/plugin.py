# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA family plugin."""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .standard_decoder_builder import build_standard_decoder_engine


class LlamaPlugin:
    name = "llama"
    runtime_strategy = "llama_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("llama")

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
            debug_layer_outputs=debug_layer_outputs)


plugin = LlamaPlugin()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo family plugin."""

from __future__ import annotations

from .config import ModelConfig
from .weights import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from .model.parallel import build_dual_profile_tp_decoder_engine
from .model.model import build_standard_decoder_engine


class OlmoPlugin:
    name = "olmo"
    runtime_strategy = "olmo_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "olmo"

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        norm_type = "layernorm"
        if parallel.enabled:
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type=norm_type,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            norm_type=norm_type,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


plugin = OlmoPlugin()

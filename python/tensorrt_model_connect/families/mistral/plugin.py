"""Mistral family plugin."""

from __future__ import annotations

from ...config import ModelConfig
from ...checkpoint_mapper import WeightDict, load_standard_weights
from ...parallel_config import normalize_parallel_config
from .standard_decoder_builder import build_standard_decoder_engine


class MistralPlugin:
    name = "mistral"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("mistral")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        return load_standard_weights(model_dir, config, precision=precision)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            from .dual_profile_decoder_tp_builder import (
                build_dual_profile_tp_decoder_engine,
            )
            return build_dual_profile_tp_decoder_engine(
                config, weights, max_cache_length, precision=precision,
                quant_ctx=quant_ctx, verbose=verbose, parallel_config=parallel)
        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose)


plugin = MistralPlugin()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2-MoE family plugin."""

from __future__ import annotations

from .checkpoint_mapper import WeightDict, load_lfm2_moe_weights
from .config import validate_lfm2_moe_config
from .model import build_lfm2_moe_engine


class Lfm2MoePlugin:
    name = "lfm2_moe"
    runtime_strategy = "lfm2_moe_hybrid_conv_attention"

    def matches(self, model_type: str) -> bool:
        return str(model_type).lower().replace("-", "_") == "lfm2_moe"

    def default_build_precision(self, config: object) -> str:
        validate_lfm2_moe_config(config)
        return "bf16"

    def default_max_cache_length(self, config: object) -> int:
        return validate_lfm2_moe_config(config).default_cache_length

    def load_weights(
        self,
        model_dir: str,
        config: object,
        *,
        precision: str = "bf16",
    ) -> WeightDict:
        return load_lfm2_moe_weights(
            model_dir,
            config,
            precision=precision,
        )

    def build_engine(
        self,
        config: object,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "bf16",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        if quant_ctx is not None:
            raise NotImplementedError("LFM2-MoE v1 does not support quantized builds")
        resolved = validate_lfm2_moe_config(config)
        config.raw["_lfm2_moe_bundle_overrides"] = resolved.bundle_overrides()
        return build_lfm2_moe_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def get_bundle_config_overrides(self, config: object) -> dict[str, object]:
        resolved = validate_lfm2_moe_config(config)
        overrides = resolved.bundle_overrides()
        stored = config.raw.get("_lfm2_moe_bundle_overrides")
        if isinstance(stored, dict):
            overrides.update(stored)
        return overrides


plugin = Lfm2MoePlugin()

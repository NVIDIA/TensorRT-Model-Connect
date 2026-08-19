# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense LFM2 family plugin."""

from __future__ import annotations

from .checkpoint_mapper import WeightDict, load_lfm2_weights
from .config import validate_dense_lfm2_config
from .model import build_lfm2_engine


class Lfm2Plugin:
    name = "lfm2"
    runtime_strategy = "lfm2_hybrid_conv_attention"

    def matches(self, model_type: str) -> bool:
        return str(model_type).lower().replace("-", "_") == "lfm2"

    def default_build_precision(self, config: object) -> str:
        validate_dense_lfm2_config(config)
        return "bf16"

    def default_max_cache_length(self, config: object) -> int:
        return validate_dense_lfm2_config(config).default_cache_length

    def load_weights(
        self,
        model_dir: str,
        config: object,
        *,
        precision: str = "bf16",
    ) -> WeightDict:
        return load_lfm2_weights(
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
            raise NotImplementedError("dense LFM2 v1 does not support quantized builds")
        resolved = validate_dense_lfm2_config(config)
        config.raw["_lfm2_bundle_overrides"] = resolved.bundle_overrides()
        return build_lfm2_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def get_bundle_config_overrides(self, config: object) -> dict[str, object]:
        resolved = validate_dense_lfm2_config(config)
        overrides = resolved.bundle_overrides()
        stored = config.raw.get("_lfm2_bundle_overrides")
        if isinstance(stored, dict):
            overrides.update(stored)
        return overrides


plugin = Lfm2Plugin()

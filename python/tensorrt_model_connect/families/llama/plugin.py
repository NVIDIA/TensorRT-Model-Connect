# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA family plugin."""

from __future__ import annotations

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, load_standard_weights
from .build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
)
from .native_kv_contract import validate_native_kv_weights
from .dual_profile_decoder_builder import build_dual_profile_decoder_engine
from .standard_decoder_builder import build_standard_decoder_engine


class LlamaPlugin:
    name = "llama"
    runtime_strategy = "llama_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("llama")

    def default_build_precision(self, config: ModelConfig) -> str:
        capability = native_kv_architecture_capability(config)
        if capability.eligible:
            return "bf16"
        if capability.applicable:
            raise ValueError(
                "Unsupported dense Llama native-KV model-only build: "
                + capability.reason
            )
        return "fp32"

    def default_max_cache_length(self, config: ModelConfig) -> int:
        """Use the model's complete context for native Llama."""
        capability = native_kv_architecture_capability(config)
        if capability.eligible:
            return int(config.max_position_embeddings)
        if capability.applicable:
            raise ValueError(
                "Unsupported dense Llama native-KV model-only build: "
                + capability.reason
            )
        return 256

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
        capability = native_kv_build_capability(
            config,
            precision=precision,
            max_cache_length=max_cache_length,
            quantized=quant_ctx is not None,
            debug_layer_outputs=debug_layer_outputs,
        )
        if capability.applicable and not capability.eligible:
            config.raw.pop("_native_kv_cache_metadata", None)
            raise ValueError(
                "Unsupported dense Llama native-KV build: "
                + capability.reason
            )
        if capability.eligible:
            validate_native_kv_weights(config, weights)
            config.raw["_decoder_engine_layout_supported"] = True
            config.raw["_native_kv_cache_metadata"] = {
                "native_kv_contract_version": 1,
                "native_kv_cache": True,
            }
            role = str(
                config.raw.get("_decoder_engine_role", "")
            )
            if role not in ("prefill", "decode"):
                raise ValueError(
                    "native Llama requires explicit split engine role "
                    f"'prefill' or 'decode', got {role!r}"
                )
            return build_dual_profile_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision="bf16",
                quant_ctx=None,
                verbose=verbose,
                profile_mode=role,
                native_kv_cache=True,
            )

        config.raw.pop("_native_kv_cache_metadata", None)
        return build_standard_decoder_engine(
            config, weights, max_cache_length, precision=precision,
            quant_ctx=quant_ctx, verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)

    def get_bundle_config_overrides(
        self, config: ModelConfig,
    ) -> dict | None:
        """Mark bundles that use the native KV runtime contract."""
        metadata = config.raw.get("_native_kv_cache_metadata")
        return dict(metadata) if isinstance(metadata, dict) else None


plugin = LlamaPlugin()
